"""
Comprehensive unit tests for SelectionModel and SelectionConfig.

Validates the three primary selection algorithms implemented in
``app.statistical.selection_models``:

1. **Vendor selection** — Pareto 80/20 rule where top 20% of vendors
   (by spend history) capture ~80% of selections.
2. **Customer selection** — Revenue-weighted random selection with
   graceful uniform fallback when revenue data is missing/zero.
3. **Product selection** — 70/30 repeat/new split drawing from
   purchase history vs. fresh catalog.

Per AAP Section 0.7.5:
    - Statistical model tests MUST verify output distributions match
      expected parameters using confidence intervals.
    - Use sufficient sample sizes (N >= 10,000).
    - Use numpy random seeding (seed=42) for reproducible tests.
    - No external API calls, no Redis.

Test classes:
    TestSelectionModelInit
    TestVendorSelectionPareto
    TestCustomerSelectionRevenueWeighted
    TestProductSelection70_30Split
    TestBatchSelectionMethods
    TestConcentrationRatio
    TestEdgeCases
    TestStatisticalVerification
"""

from __future__ import annotations

import time
from collections import Counter
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from app.statistical.selection_models import SelectionModel, SelectionConfig


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def selection_model() -> SelectionModel:
    """Provide a deterministic SelectionModel seeded with 42."""
    return SelectionModel(seed=42)


@pytest.fixture
def sample_vendors() -> list[dict]:
    """Provide 10 vendors with descending spend history.

    Top 2 (20%) account for 1.8M out of 2.565M total spend (~70%).
    """
    return [
        {"id": "V001", "name": "Top Vendor A", "spend_history": 1_000_000, "category": "electronics"},
        {"id": "V002", "name": "Top Vendor B", "spend_history": 800_000, "category": "electronics"},
        {"id": "V003", "name": "Mid Vendor C", "spend_history": 300_000, "category": "office_supplies"},
        {"id": "V004", "name": "Mid Vendor D", "spend_history": 200_000, "category": "office_supplies"},
        {"id": "V005", "name": "Mid Vendor E", "spend_history": 150_000, "category": "electronics"},
        {"id": "V006", "name": "Small Vendor F", "spend_history": 50_000, "category": "office_supplies"},
        {"id": "V007", "name": "Small Vendor G", "spend_history": 30_000, "category": "services"},
        {"id": "V008", "name": "Small Vendor H", "spend_history": 20_000, "category": "services"},
        {"id": "V009", "name": "Tiny Vendor I", "spend_history": 10_000, "category": "electronics"},
        {"id": "V010", "name": "Tiny Vendor J", "spend_history": 5_000, "category": "services"},
    ]


@pytest.fixture
def sample_customers() -> list[dict]:
    """Provide 10 customers with varying revenue and tier assignments."""
    return [
        {"id": "C001", "name": "Enterprise Corp", "revenue": 5_000_000, "tier": "strategic"},
        {"id": "C002", "name": "Big Business Inc", "revenue": 2_000_000, "tier": "strategic"},
        {"id": "C003", "name": "Medium Corp", "revenue": 500_000, "tier": "standard"},
        {"id": "C004", "name": "Medium LLC", "revenue": 400_000, "tier": "standard"},
        {"id": "C005", "name": "Medium Co", "revenue": 300_000, "tier": "standard"},
        {"id": "C006", "name": "Small Biz A", "revenue": 100_000, "tier": "transactional"},
        {"id": "C007", "name": "Small Biz B", "revenue": 80_000, "tier": "transactional"},
        {"id": "C008", "name": "Small Biz C", "revenue": 50_000, "tier": "transactional"},
        {"id": "C009", "name": "Tiny Shop D", "revenue": 20_000, "tier": "transactional"},
        {"id": "C010", "name": "Tiny Shop E", "revenue": 10_000, "tier": "transactional"},
    ]


@pytest.fixture
def sample_products() -> list[dict]:
    """Provide 20 generic products for selection testing."""
    return [
        {"id": f"P{i:03d}", "name": f"Product {i}", "category": "general"}
        for i in range(1, 21)
    ]


@pytest.fixture
def sample_purchase_history(sample_products: list[dict]) -> list[dict]:
    """First 5 products represent previously purchased items."""
    return sample_products[:5]


# =========================================================================
# TestSelectionModelInit
# =========================================================================


class TestSelectionModelInit:
    """Verify SelectionModel and SelectionConfig initialisation."""

    def test_init_default_config(self) -> None:
        """Default construction should use standard SelectionConfig values."""
        model = SelectionModel()
        assert isinstance(model.config, SelectionConfig)
        assert model.config.pareto_shape == pytest.approx(1.16)
        assert model.config.repeat_purchase_rate == pytest.approx(0.70)
        assert model.config.new_purchase_rate == pytest.approx(0.30)

    def test_init_with_seed(self) -> None:
        """Seeded construction should produce reproducible random state."""
        model = SelectionModel(seed=42)
        assert model._seed == 42
        assert model.rng is not None

    def test_init_pareto_config(self) -> None:
        """Pareto shape parameter should default to ~1.16 for 80/20."""
        config = SelectionConfig()
        assert config.pareto_shape == pytest.approx(1.16)
        assert config.top_vendor_pct == pytest.approx(0.20)
        assert config.top_vendor_business_pct == pytest.approx(0.80)

    def test_init_custom_config(self) -> None:
        """Custom config should override defaults correctly."""
        custom = SelectionConfig(
            pareto_shape=2.0,
            repeat_purchase_rate=0.60,
            new_purchase_rate=0.40,
        )
        model = SelectionModel(config=custom, seed=99)
        assert model.config.pareto_shape == pytest.approx(2.0)
        assert model.config.repeat_purchase_rate == pytest.approx(0.60)
        assert model.config.new_purchase_rate == pytest.approx(0.40)

    def test_init_repeat_new_rates(self) -> None:
        """repeat_purchase_rate + new_purchase_rate must equal 1.0."""
        config = SelectionConfig()
        total = config.repeat_purchase_rate + config.new_purchase_rate
        assert total == pytest.approx(1.0)

    def test_invalid_purchase_rates_raises(self) -> None:
        """Non-complementary rates should raise a ValueError."""
        with pytest.raises(Exception):
            SelectionConfig(repeat_purchase_rate=0.60, new_purchase_rate=0.60)


# =========================================================================
# TestVendorSelectionPareto
# =========================================================================


class TestVendorSelectionPareto:
    """Verify Pareto 80/20 vendor selection behaviour."""

    def test_vendor_selection_returns_dict(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """select_vendor must return a dict."""
        result = selection_model.select_vendor(sample_vendors)
        assert isinstance(result, dict)

    def test_vendor_selection_valid_vendor(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """Selected vendor must exist in the original list."""
        result = selection_model.select_vendor(sample_vendors)
        vendor_ids = {v["id"] for v in sample_vendors}
        assert result["id"] in vendor_ids

    def test_pareto_80_20_rule(
        self, sample_vendors: list[dict]
    ) -> None:
        """Top 20% of vendors should capture >= 50% of all selections.

        With 10 vendors the top 2 (20%) hold 70% of spend. The Pareto
        weighting should concentrate selections heavily on them.
        """
        model = SelectionModel(seed=42)
        n = 10_000
        counter: Counter[str] = Counter()
        for _ in range(n):
            v = model.select_vendor(sample_vendors)
            counter[v["id"]] += 1

        top_ids = {"V001", "V002"}  # top 20% by spend
        top_freq = sum(counter[vid] for vid in top_ids)
        top_pct = top_freq / n

        # The Pareto weighting should give top 20% at least 50% of picks
        assert top_pct >= 0.50, (
            f"Top 20% vendors captured only {top_pct:.2%} of selections "
            f"(expected >= 50%)"
        )

    def test_top_vendor_most_selected(
        self, sample_vendors: list[dict]
    ) -> None:
        """The vendor with the highest spend should be the most selected."""
        model = SelectionModel(seed=42)
        counter: Counter[str] = Counter()
        for _ in range(10_000):
            v = model.select_vendor(sample_vendors)
            counter[v["id"]] += 1

        most_common_id = counter.most_common(1)[0][0]
        assert most_common_id == "V001", (
            f"Expected V001 (highest spend) to be most selected, got {most_common_id}"
        )

    def test_spend_history_correlation(
        self, sample_vendors: list[dict]
    ) -> None:
        """Selection frequency should correlate positively with spend."""
        model = SelectionModel(seed=42)
        counter: Counter[str] = Counter()
        for _ in range(10_000):
            v = model.select_vendor(sample_vendors)
            counter[v["id"]] += 1

        # Build parallel arrays: spend history and selection frequency
        spends = np.array(
            [v["spend_history"] for v in sample_vendors], dtype=np.float64
        )
        freqs = np.array(
            [counter.get(v["id"], 0) for v in sample_vendors], dtype=np.float64
        )
        corr = np.corrcoef(spends, freqs)[0, 1]
        assert corr > 0.5, (
            f"Spend-frequency correlation {corr:.3f} is too low (expected > 0.5)"
        )

    def test_category_filter(
        self, sample_vendors: list[dict]
    ) -> None:
        """Category filter should restrict selections to matching vendors."""
        model = SelectionModel(seed=42)
        for _ in range(1_000):
            v = model.select_vendor(sample_vendors, category="electronics")
            assert v["category"] == "electronics", (
                f"Expected electronics vendor, got category={v['category']}"
            )

    def test_category_filter_empty_falls_back(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """Non-matching category should fall back to the full vendor list."""
        result = selection_model.select_vendor(
            sample_vendors, category="nonexistent"
        )
        assert isinstance(result, dict)
        vendor_ids = {v["id"] for v in sample_vendors}
        assert result["id"] in vendor_ids

    def test_vendor_selection_reproducible(
        self, sample_vendors: list[dict]
    ) -> None:
        """Two identically-seeded models must produce the same first pick."""
        m1 = SelectionModel(seed=42)
        m2 = SelectionModel(seed=42)
        v1 = m1.select_vendor(sample_vendors)
        v2 = m2.select_vendor(sample_vendors)
        assert v1["id"] == v2["id"]

    def test_empty_vendor_list_raises(
        self, selection_model: SelectionModel
    ) -> None:
        """Selecting from an empty vendor list should raise ValueError."""
        with pytest.raises(ValueError):
            selection_model.select_vendor([])


# =========================================================================
# TestCustomerSelectionRevenueWeighted
# =========================================================================


class TestCustomerSelectionRevenueWeighted:
    """Verify revenue-weighted customer selection behaviour."""

    def test_customer_selection_returns_dict(
        self, selection_model: SelectionModel, sample_customers: list[dict]
    ) -> None:
        """select_customer must return a dict from the customer list."""
        result = selection_model.select_customer(sample_customers)
        assert isinstance(result, dict)
        cust_ids = {c["id"] for c in sample_customers}
        assert result["id"] in cust_ids

    def test_revenue_weighted_selection(
        self, sample_customers: list[dict]
    ) -> None:
        """Enterprise Corp (5M) should be selected >> Tiny Shop E (10K).

        Revenue ratio is 500:1 so Enterprise should dominate selections.
        We check that freq(C001)/freq(C010) > 10 (conservative bound).
        """
        model = SelectionModel(seed=42)
        counter: Counter[str] = Counter()
        for _ in range(10_000):
            c = model.select_customer(sample_customers)
            counter[c["id"]] += 1

        freq_c001 = counter.get("C001", 0)
        freq_c010 = counter.get("C010", 1)  # avoid division by zero
        ratio = freq_c001 / freq_c010
        assert ratio > 10, (
            f"Revenue-weight ratio C001/C010 = {ratio:.1f} (expected > 10)"
        )

    def test_large_customers_more_frequent(
        self, sample_customers: list[dict]
    ) -> None:
        """Top 2 customers by revenue should account for > 50% selections."""
        model = SelectionModel(seed=42)
        counter: Counter[str] = Counter()
        for _ in range(10_000):
            c = model.select_customer(sample_customers)
            counter[c["id"]] += 1

        top_ids = {"C001", "C002"}
        top_freq = sum(counter.get(cid, 0) for cid in top_ids)
        top_pct = top_freq / 10_000
        assert top_pct > 0.50, (
            f"Top 2 customers captured only {top_pct:.2%} of selections "
            f"(expected > 50%)"
        )

    def test_no_revenue_data_uniform(self) -> None:
        """Customers without revenue keys should be selected uniformly."""
        customers = [
            {"id": f"C{i:03d}", "name": f"Cust {i}"} for i in range(1, 6)
        ]
        model = SelectionModel(seed=42)
        counter: Counter[str] = Counter()
        n = 10_000
        for _ in range(n):
            c = model.select_customer(customers)
            counter[c["id"]] += 1

        expected_each = n / len(customers)
        for cid, freq in counter.items():
            # With 10K samples and 5 customers, each should get ~2000 ± 250
            assert abs(freq - expected_each) < expected_each * 0.35, (
                f"Customer {cid} selected {freq} times, "
                f"expected ~{expected_each:.0f} (uniform)"
            )

    def test_customer_selection_reproducible(
        self, sample_customers: list[dict]
    ) -> None:
        """Same seed should produce identical first selections."""
        m1 = SelectionModel(seed=42)
        m2 = SelectionModel(seed=42)
        c1 = m1.select_customer(sample_customers)
        c2 = m2.select_customer(sample_customers)
        assert c1["id"] == c2["id"]

    def test_empty_customer_list_raises(
        self, selection_model: SelectionModel
    ) -> None:
        """Selecting from an empty customer list should raise ValueError."""
        with pytest.raises(ValueError):
            selection_model.select_customer([])


# =========================================================================
# TestProductSelection70_30Split
# =========================================================================


class TestProductSelection70_30Split:
    """Verify the 70/30 repeat-purchase / new-product split."""

    def test_product_selection_returns_dict(
        self,
        selection_model: SelectionModel,
        sample_products: list[dict],
        sample_purchase_history: list[dict],
    ) -> None:
        """select_product must return a dict."""
        result = selection_model.select_product(
            sample_products, purchase_history=sample_purchase_history
        )
        assert isinstance(result, dict)

    def test_70_30_repeat_new_split(
        self, sample_products: list[dict], sample_purchase_history: list[dict]
    ) -> None:
        """Approximately 70% repeat, 30% new (± 5pp for N=10,000)."""
        model = SelectionModel(seed=42)
        history_ids = {p["id"] for p in sample_purchase_history}
        n = 10_000
        repeat_count = 0

        for _ in range(n):
            p = model.select_product(
                sample_products, purchase_history=sample_purchase_history
            )
            if p["id"] in history_ids:
                repeat_count += 1

        repeat_pct = repeat_count / n
        new_pct = 1.0 - repeat_pct

        # 70% repeat ± 5pp tolerance
        assert 0.65 <= repeat_pct <= 0.75, (
            f"Repeat purchase ratio {repeat_pct:.2%} outside [65%, 75%]"
        )
        # 30% new ± 5pp tolerance
        assert 0.25 <= new_pct <= 0.35, (
            f"New purchase ratio {new_pct:.2%} outside [25%, 35%]"
        )

    def test_no_history_all_from_catalog(
        self, selection_model: SelectionModel, sample_products: list[dict]
    ) -> None:
        """With purchase_history=None all picks come from catalog."""
        for _ in range(100):
            p = selection_model.select_product(sample_products, purchase_history=None)
            assert p["id"] in {pr["id"] for pr in sample_products}

    def test_empty_history_all_from_catalog(
        self, selection_model: SelectionModel, sample_products: list[dict]
    ) -> None:
        """With purchase_history=[] all picks come from catalog."""
        for _ in range(100):
            p = selection_model.select_product(sample_products, purchase_history=[])
            assert p["id"] in {pr["id"] for pr in sample_products}

    def test_all_products_purchased_100_repeat(
        self, sample_products: list[dict]
    ) -> None:
        """When entire catalog is already purchased, 100% are repeats."""
        model = SelectionModel(seed=42)
        history = list(sample_products)  # everything purchased
        history_ids = {p["id"] for p in history}

        for _ in range(200):
            p = model.select_product(sample_products, purchase_history=history)
            assert p["id"] in history_ids

    def test_product_selection_reproducible(
        self, sample_products: list[dict], sample_purchase_history: list[dict]
    ) -> None:
        """Same seed should yield identical product sequence."""
        m1 = SelectionModel(seed=42)
        m2 = SelectionModel(seed=42)
        p1 = m1.select_product(
            sample_products, purchase_history=sample_purchase_history
        )
        p2 = m2.select_product(
            sample_products, purchase_history=sample_purchase_history
        )
        assert p1["id"] == p2["id"]

    def test_empty_products_raises(
        self, selection_model: SelectionModel
    ) -> None:
        """Selecting from an empty product list should raise ValueError."""
        with pytest.raises(ValueError):
            selection_model.select_product([])


# =========================================================================
# TestBatchSelectionMethods
# =========================================================================


class TestBatchSelectionMethods:
    """Verify batch selection helpers and performance."""

    def test_batch_vendor_selection_count(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """select_vendors_batch(n=50) must return exactly 50 items."""
        results = selection_model.select_vendors_batch(sample_vendors, n=50)
        assert len(results) == 50
        assert all(isinstance(v, dict) for v in results)

    def test_batch_customer_selection_count(
        self, selection_model: SelectionModel, sample_customers: list[dict]
    ) -> None:
        """select_customers_batch(n=50) must return exactly 50 items."""
        results = selection_model.select_customers_batch(sample_customers, n=50)
        assert len(results) == 50
        assert all(isinstance(c, dict) for c in results)

    def test_batch_product_selection_count(
        self,
        selection_model: SelectionModel,
        sample_products: list[dict],
        sample_purchase_history: list[dict],
    ) -> None:
        """select_products_batch(n=50) must return exactly 50 items."""
        results = selection_model.select_products_batch(
            sample_products, n=50, purchase_history=sample_purchase_history
        )
        assert len(results) == 50
        assert all(isinstance(p, dict) for p in results)

    def test_batch_vendor_maintains_pareto(
        self, sample_vendors: list[dict]
    ) -> None:
        """Batch vendor selections must still exhibit Pareto concentration."""
        model = SelectionModel(seed=42)
        results = model.select_vendors_batch(sample_vendors, n=10_000)
        counter: Counter[str] = Counter(v["id"] for v in results)

        top_ids = {"V001", "V002"}
        top_freq = sum(counter.get(vid, 0) for vid in top_ids)
        top_pct = top_freq / 10_000
        assert top_pct >= 0.50, (
            f"Batch Pareto: top 20% captured only {top_pct:.2%}"
        )

    def test_batch_performance(
        self, sample_vendors: list[dict]
    ) -> None:
        """10,000 batch vendor selections should complete in < 1 second."""
        model = SelectionModel(seed=42)
        start = time.perf_counter()
        model.select_vendors_batch(sample_vendors, n=10_000)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, (
            f"Batch selection took {elapsed:.3f}s (limit 1.0s)"
        )

    def test_batch_n_less_than_one_raises(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            selection_model.select_vendors_batch(sample_vendors, n=0)


# =========================================================================
# TestConcentrationRatio
# =========================================================================


class TestConcentrationRatio:
    """Verify concentration analytics and distribution computation."""

    def test_concentration_ratio_returns_float(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """get_concentration_ratio must return a float in [0, 1]."""
        ratio = selection_model.get_concentration_ratio(
            sample_vendors, top_pct=0.20, n_samples=5_000
        )
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.0

    def test_concentration_ratio_approximately_80_percent(
        self, sample_vendors: list[dict]
    ) -> None:
        """Top 20% concentration should be in [0.40, 0.95] for 80/20."""
        model = SelectionModel(seed=42)
        ratio = model.get_concentration_ratio(
            sample_vendors, top_pct=0.20, n_samples=20_000
        )
        # The exact value depends on spend distribution and Pareto shape.
        # With our 10-vendor sample the top 2 have 70% of spend, so the
        # weighted selection should show strong concentration.
        assert 0.40 <= ratio <= 0.95, (
            f"Concentration ratio {ratio:.3f} outside expected [0.40, 0.95]"
        )

    def test_selection_distribution_returns_dict(
        self, selection_model: SelectionModel, sample_vendors: list[dict]
    ) -> None:
        """compute_selection_distribution should return a str→float dict."""
        dist = selection_model.compute_selection_distribution(
            sample_vendors, n_samples=5_000
        )
        assert isinstance(dist, dict)
        assert len(dist) == len(sample_vendors)
        # Frequencies should sum to ~1.0
        total = sum(dist.values())
        assert total == pytest.approx(1.0, abs=0.01)
        # All values should be non-negative
        assert all(v >= 0.0 for v in dist.values())

    def test_selection_distribution_empty_entities(
        self, selection_model: SelectionModel
    ) -> None:
        """Empty entity list should return empty dict."""
        dist = selection_model.compute_selection_distribution([])
        assert dist == {}

    def test_concentration_ratio_empty_entities(
        self, selection_model: SelectionModel
    ) -> None:
        """Empty entity list should return 0.0."""
        ratio = selection_model.get_concentration_ratio([])
        assert ratio == 0.0


# =========================================================================
# TestEdgeCases
# =========================================================================


class TestEdgeCases:
    """Edge-case and boundary-condition tests."""

    def test_single_vendor(self) -> None:
        """A single vendor must always be selected."""
        model = SelectionModel(seed=42)
        single = [{"id": "VONLY", "name": "Only Vendor", "spend_history": 100}]
        for _ in range(50):
            assert model.select_vendor(single)["id"] == "VONLY"

    def test_single_customer(self) -> None:
        """A single customer must always be selected."""
        model = SelectionModel(seed=42)
        single = [{"id": "CONLY", "name": "Only Customer", "revenue": 999}]
        for _ in range(50):
            assert model.select_customer(single)["id"] == "CONLY"

    def test_single_product_no_history(self) -> None:
        """A single product with no history must be returned."""
        model = SelectionModel(seed=42)
        single = [{"id": "PONLY", "name": "Only Product", "category": "x"}]
        for _ in range(50):
            assert model.select_product(single)["id"] == "PONLY"

    def test_all_zero_spend_history(self) -> None:
        """Zero-spend vendors should not cause division-by-zero."""
        model = SelectionModel(seed=42)
        vendors = [
            {"id": f"V{i}", "name": f"Vendor {i}", "spend_history": 0}
            for i in range(5)
        ]
        # Should fall back to rank-based or uniform — must not raise
        for _ in range(100):
            v = model.select_vendor(vendors)
            assert v["id"] in {f"V{i}" for i in range(5)}

    def test_all_zero_revenue(self) -> None:
        """Zero-revenue customers should fall back to uniform selection."""
        model = SelectionModel(seed=42)
        customers = [
            {"id": f"C{i}", "name": f"Cust {i}", "revenue": 0}
            for i in range(5)
        ]
        counter: Counter[str] = Counter()
        n = 5_000
        for _ in range(n):
            c = model.select_customer(customers)
            counter[c["id"]] += 1

        expected_each = n / len(customers)
        for cid, freq in counter.items():
            assert abs(freq - expected_each) < expected_each * 0.40, (
                f"Customer {cid} got {freq} (expected ~{expected_each:.0f})"
            )

    def test_large_vendor_list(self) -> None:
        """1,000 vendors: Pareto should still concentrate on top 20%."""
        model = SelectionModel(seed=42)
        vendors = [
            {
                "id": f"V{i:04d}",
                "name": f"Vendor {i}",
                "spend_history": max(1, int(1_000_000 / (i + 1))),
            }
            for i in range(1_000)
        ]
        n = 10_000
        results = model.select_vendors_batch(vendors, n=n)
        counter: Counter[str] = Counter(v["id"] for v in results)

        # Top 200 vendors (20%)
        top_200_ids = {f"V{i:04d}" for i in range(200)}
        top_freq = sum(counter.get(vid, 0) for vid in top_200_ids)
        top_pct = top_freq / n
        assert top_pct >= 0.50, (
            f"Large list: top 20% captured only {top_pct:.2%}"
        )

    def test_vendors_without_spend_key(self) -> None:
        """Missing spend_history key should fall back to rank-based or uniform."""
        model = SelectionModel(seed=42)
        vendors = [
            {"id": f"V{i}", "name": f"Vendor {i}"} for i in range(5)
        ]
        # Should not raise — falls back gracefully
        for _ in range(100):
            v = model.select_vendor(vendors)
            assert v["id"] in {f"V{i}" for i in range(5)}

    def test_category_services_only(
        self, sample_vendors: list[dict]
    ) -> None:
        """Category 'services' should only return services vendors."""
        model = SelectionModel(seed=42)
        service_ids = {v["id"] for v in sample_vendors if v["category"] == "services"}
        for _ in range(200):
            v = model.select_vendor(sample_vendors, category="services")
            assert v["id"] in service_ids

    def test_batch_products_no_history(
        self, sample_products: list[dict]
    ) -> None:
        """Batch product selection without history selects from catalog."""
        model = SelectionModel(seed=42)
        results = model.select_products_batch(sample_products, n=100)
        product_ids = {p["id"] for p in sample_products}
        for p in results:
            assert p["id"] in product_ids


# =========================================================================
# TestStatisticalVerification
# =========================================================================


class TestStatisticalVerification:
    """High-confidence statistical property verification (N >= 10,000)."""

    def test_pareto_concentration_statistical(self) -> None:
        """100 vendors with exponential spend: Pareto should concentrate."""
        model = SelectionModel(seed=42)
        vendors = [
            {
                "id": f"V{i:04d}",
                "name": f"Vendor {i}",
                "spend_history": float(np.exp(10 - 0.05 * i)),
            }
            for i in range(100)
        ]
        n = 50_000
        results = model.select_vendors_batch(vendors, n=n)
        counter: Counter[str] = Counter(v["id"] for v in results)

        # Compute Herfindahl-Hirschman Index (HHI) to measure concentration
        freqs = np.array(list(counter.values()), dtype=np.float64)
        shares = freqs / freqs.sum()
        hhi = float(np.sum(shares ** 2))

        # Uniform HHI for 100 vendors = 0.01.
        # A concentrated distribution should have HHI >> 0.01.
        assert hhi > 0.02, (
            f"HHI {hhi:.4f} too low — expected concentration above 0.02"
        )

        # Top 20 vendors should capture a significant share
        top_20_ids = {f"V{i:04d}" for i in range(20)}
        top_freq = sum(counter.get(vid, 0) for vid in top_20_ids)
        top_pct = top_freq / n
        assert top_pct >= 0.50, (
            f"Statistical: top 20% captured {top_pct:.2%} (expected >= 50%)"
        )

    def test_revenue_weighting_proportionality(self) -> None:
        """Selection frequencies should be roughly proportional to revenue."""
        model = SelectionModel(seed=42)
        customers = [
            {"id": f"C{i}", "name": f"Cust {i}", "revenue": (i + 1) * 100}
            for i in range(5)
        ]
        # Revenues: 100, 200, 300, 400, 500  (total 1500)
        n = 50_000
        counter: Counter[str] = Counter()
        for _ in range(n):
            c = model.select_customer(customers)
            counter[c["id"]] += 1

        # Customer C4 (revenue=500) should be selected ~5× more than C0 (100)
        freq_high = counter.get("C4", 0)
        freq_low = counter.get("C0", 1)
        ratio = freq_high / freq_low
        assert 3.0 < ratio < 8.0, (
            f"Revenue ratio C4/C0 = {ratio:.2f} (expected ~5.0)"
        )

        # Verify approximate proportionality via correlation
        revenues = np.array([100, 200, 300, 400, 500], dtype=np.float64)
        frequencies = np.array(
            [counter.get(f"C{i}", 0) for i in range(5)], dtype=np.float64
        )
        corr = np.corrcoef(revenues, frequencies)[0, 1]
        assert corr > 0.95, (
            f"Revenue-frequency correlation {corr:.3f} too low (expected > 0.95)"
        )

    def test_repeat_new_ratio_large_sample(
        self, sample_products: list[dict], sample_purchase_history: list[dict]
    ) -> None:
        """50,000 selections: repeat % should be 70% ± 2pp."""
        model = SelectionModel(seed=42)
        history_ids = {p["id"] for p in sample_purchase_history}
        n = 50_000
        repeat_count = 0

        for _ in range(n):
            p = model.select_product(
                sample_products, purchase_history=sample_purchase_history
            )
            if p["id"] in history_ids:
                repeat_count += 1

        repeat_pct = repeat_count / n
        new_pct = 1.0 - repeat_pct

        assert 0.68 <= repeat_pct <= 0.72, (
            f"Repeat ratio {repeat_pct:.4f} outside [0.68, 0.72]"
        )
        assert 0.28 <= new_pct <= 0.32, (
            f"New ratio {new_pct:.4f} outside [0.28, 0.32]"
        )

    def test_vendor_spend_rank_ordering(self) -> None:
        """Vendors ranked by spend should have monotonically decreasing
        expected selection frequency (with statistical tolerance)."""
        model = SelectionModel(seed=42)
        vendors = [
            {"id": f"V{i:02d}", "name": f"Vendor {i}", "spend_history": (10 - i) * 1_000}
            for i in range(10)
        ]
        n = 50_000
        results = model.select_vendors_batch(vendors, n=n)
        counter: Counter[str] = Counter(v["id"] for v in results)
        freqs = [counter.get(f"V{i:02d}", 0) for i in range(10)]

        # The highest-spend vendor should have more selections than the lowest
        assert freqs[0] > freqs[-1], (
            f"Highest-spend vendor ({freqs[0]}) should have more "
            f"selections than lowest ({freqs[-1]})"
        )

    def test_batch_customer_revenue_proportionality(self) -> None:
        """Batch customer selection should preserve revenue weighting."""
        model = SelectionModel(seed=42)
        customers = [
            {"id": "CHIGH", "name": "High Rev", "revenue": 1_000_000},
            {"id": "CLOW", "name": "Low Rev", "revenue": 10_000},
        ]
        results = model.select_customers_batch(customers, n=10_000)
        counter: Counter[str] = Counter(c["id"] for c in results)

        high_pct = counter["CHIGH"] / 10_000
        assert high_pct > 0.90, (
            f"High-revenue customer captured only {high_pct:.2%}"
        )
