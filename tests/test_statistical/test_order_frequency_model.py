"""
Comprehensive unit tests for the OrderFrequencyModel class (F-006).

Tests verify that the Poisson process with day-of-week multipliers produces
realistic order generation frequencies matching the specification:
    - Monday:    0.85×
    - Tuesday:   1.00×
    - Wednesday: 1.00×
    - Thursday:  1.10×
    - Friday:    1.20×
    - Saturday:  0.00× (non-business day)
    - Sunday:    0.00× (non-business day)

Testing conventions per AAP Section 0.7.5:
    - Statistical model tests verify output distributions match expected
      parameters using confidence intervals with sufficient sample sizes.
    - N ≥ 10,000 for distribution verification tests.
    - numpy seed=42 for reproducibility.
    - No external API calls or Redis dependencies.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy import stats

from app.statistical.order_frequency_model import (
    OrderFrequencyConfig,
    OrderFrequencyModel,
)

# ---------------------------------------------------------------------------
# Spec-aligned constants for assertions
# ---------------------------------------------------------------------------
DEFAULT_BASE_RATE: float = 5.0
DOW_MULTIPLIERS = {
    0: 0.85,   # Monday
    1: 1.00,   # Tuesday
    2: 1.00,   # Wednesday
    3: 1.10,   # Thursday
    4: 1.20,   # Friday
    5: 0.00,   # Saturday
    6: 0.00,   # Sunday
}
SAMPLE_SIZE: int = 10_000
WEEKEND_SAMPLE_SIZE: int = 1_000


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def order_model() -> OrderFrequencyModel:
    """Return an OrderFrequencyModel with seed=42 for reproducible tests."""
    return OrderFrequencyModel(seed=42)


@pytest.fixture
def monday_date() -> date:
    """Return a known Monday: 2025-01-06."""
    d = date(2025, 1, 6)
    assert d.weekday() == 0, "Expected Monday"
    return d


@pytest.fixture
def tuesday_date() -> date:
    """Return a known Tuesday: 2025-01-07."""
    d = date(2025, 1, 7)
    assert d.weekday() == 1, "Expected Tuesday"
    return d


@pytest.fixture
def wednesday_date() -> date:
    """Return a known Wednesday: 2025-01-08."""
    d = date(2025, 1, 8)
    assert d.weekday() == 2, "Expected Wednesday"
    return d


@pytest.fixture
def thursday_date() -> date:
    """Return a known Thursday: 2025-01-09."""
    d = date(2025, 1, 9)
    assert d.weekday() == 3, "Expected Thursday"
    return d


@pytest.fixture
def friday_date() -> date:
    """Return a known Friday: 2025-01-10."""
    d = date(2025, 1, 10)
    assert d.weekday() == 4, "Expected Friday"
    return d


@pytest.fixture
def saturday_date() -> date:
    """Return a known Saturday: 2025-01-11."""
    d = date(2025, 1, 11)
    assert d.weekday() == 5, "Expected Saturday"
    return d


@pytest.fixture
def sunday_date() -> date:
    """Return a known Sunday: 2025-01-12."""
    d = date(2025, 1, 12)
    assert d.weekday() == 6, "Expected Sunday"
    return d


@pytest.fixture
def week_dates() -> list[date]:
    """Return 7 consecutive dates Mon-Sun starting 2025-01-06."""
    base = date(2025, 1, 6)  # Monday
    return [base + timedelta(days=i) for i in range(7)]


# =========================================================================
# TestOrderFrequencyModelInit
# =========================================================================


class TestOrderFrequencyModelInit:
    """Verify constructor behaviour, default configuration, and seeding."""

    def test_init_default_config(self) -> None:
        """Creating with no args should succeed and use defaults."""
        model = OrderFrequencyModel()
        assert model is not None
        assert model._config.base_rate == pytest.approx(DEFAULT_BASE_RATE)

    def test_init_with_seed(self, tuesday_date: date) -> None:
        """Two models with the same seed produce identical first samples."""
        m1 = OrderFrequencyModel(seed=42)
        m2 = OrderFrequencyModel(seed=42)
        assert m1.sample_order_count(tuesday_date) == m2.sample_order_count(tuesday_date)

    def test_init_day_multipliers_configured(self) -> None:
        """All 7 weekday keys (0-6) must be present in the config."""
        model = OrderFrequencyModel()
        for day_idx in range(7):
            assert day_idx in model._config.day_of_week_multipliers

    def test_init_default_base_rate(self) -> None:
        """Default base_rate is 5.0 per specification."""
        model = OrderFrequencyModel()
        assert model._config.base_rate == pytest.approx(DEFAULT_BASE_RATE)

    def test_init_custom_config(self) -> None:
        """Supplying a custom OrderFrequencyConfig overrides defaults."""
        cfg = OrderFrequencyConfig(base_rate=10.0, max_orders=100)
        model = OrderFrequencyModel(config=cfg, seed=42)
        assert model._config.base_rate == pytest.approx(10.0)
        assert model._config.max_orders == 100

    def test_init_with_random_seed_fixture(self, random_seed: int) -> None:
        """Verify the conftest random_seed fixture integrates correctly."""
        assert random_seed == 42
        model = OrderFrequencyModel(seed=random_seed)
        assert model is not None


# =========================================================================
# TestDayOfWeekMultipliers
# =========================================================================


@pytest.mark.parametrize(
    "base_rate",
    [1.0, 5.0, 10.0],
    ids=["low-rate", "default-rate", "high-rate"],
)
class TestDayOfWeekMultipliersParametrized:
    """Verify day-of-week multiplier ordering across multiple base rates."""

    def test_weekday_ordering_holds(self, base_rate: float) -> None:
        """Mean ordering Mon<Tue<Thu<Fri holds regardless of base_rate."""
        cfg = OrderFrequencyConfig(base_rate=base_rate)
        model = OrderFrequencyModel(config=cfg, seed=42)
        base_monday = date(2025, 1, 6)
        means: dict[int, float] = {}
        for offset in range(5):  # Mon-Fri
            d = base_monday + timedelta(days=offset)
            samples = [model.sample_order_count(d) for _ in range(SAMPLE_SIZE)]
            means[offset] = float(np.mean(samples))
        assert means[0] < means[1]  # Mon 0.85 < Tue 1.0
        assert means[1] < means[3]  # Tue 1.0 < Thu 1.1
        assert means[3] < means[4]  # Thu 1.1 < Fri 1.2


class TestDayOfWeekMultipliers:
    """Verify each day-of-week multiplier via large-sample mean convergence."""

    def _sample_mean_for_date(self, target_date: date, n: int = SAMPLE_SIZE) -> float:
        """Helper: sample *n* order counts and return the mean."""
        model = OrderFrequencyModel(seed=42)
        counts = [model.sample_order_count(target_date) for _ in range(n)]
        return float(np.mean(counts))

    def test_monday_multiplier_is_0_85(self, monday_date: date) -> None:
        expected_lambda = DEFAULT_BASE_RATE * 0.85  # 4.25
        mean = self._sample_mean_for_date(monday_date)
        assert mean == pytest.approx(expected_lambda, abs=0.3)

    def test_tuesday_multiplier_is_1_0(self, tuesday_date: date) -> None:
        expected_lambda = DEFAULT_BASE_RATE * 1.0  # 5.0
        mean = self._sample_mean_for_date(tuesday_date)
        assert mean == pytest.approx(expected_lambda, abs=0.3)

    def test_wednesday_multiplier_is_1_0(self, wednesday_date: date) -> None:
        expected_lambda = DEFAULT_BASE_RATE * 1.0
        mean = self._sample_mean_for_date(wednesday_date)
        assert mean == pytest.approx(expected_lambda, abs=0.3)

    def test_thursday_multiplier_is_1_1(self, thursday_date: date) -> None:
        expected_lambda = DEFAULT_BASE_RATE * 1.1  # 5.5
        mean = self._sample_mean_for_date(thursday_date)
        assert mean == pytest.approx(expected_lambda, abs=0.3)

    def test_friday_multiplier_is_1_2(self, friday_date: date) -> None:
        expected_lambda = DEFAULT_BASE_RATE * 1.2  # 6.0
        mean = self._sample_mean_for_date(friday_date)
        assert mean == pytest.approx(expected_lambda, abs=0.3)

    def test_saturday_returns_zero(self, saturday_date: date) -> None:
        model = OrderFrequencyModel(seed=42)
        counts = [model.sample_order_count(saturday_date) for _ in range(WEEKEND_SAMPLE_SIZE)]
        assert all(c == 0 for c in counts), "Saturday should always yield 0 orders"

    def test_sunday_returns_zero(self, sunday_date: date) -> None:
        model = OrderFrequencyModel(seed=42)
        counts = [model.sample_order_count(sunday_date) for _ in range(WEEKEND_SAMPLE_SIZE)]
        assert all(c == 0 for c in counts), "Sunday should always yield 0 orders"

    def test_relative_ordering_of_days(self, week_dates: list[date]) -> None:
        """Mean ordering: Mon < Tue ≈ Wed < Thu < Fri, Sat = Sun = 0."""
        model = OrderFrequencyModel(seed=42)
        means: dict[int, float] = {}
        for d in week_dates:
            samples = [model.sample_order_count(d) for _ in range(SAMPLE_SIZE)]
            means[d.weekday()] = float(np.mean(samples))

        # Weekend = 0
        assert means[5] == 0.0
        assert means[6] == 0.0

        # Strict weekday ordering by multiplier
        assert means[0] < means[1]  # Mon 0.85 < Tue 1.0
        assert means[1] < means[3]  # Tue 1.0 < Thu 1.1
        assert means[3] < means[4]  # Thu 1.1 < Fri 1.2


# =========================================================================
# TestPoissonDistributionProperties
# =========================================================================


class TestPoissonDistributionProperties:
    """Verify Poisson properties: non-negative ints, mean ≈ variance ≈ λ."""

    def test_sample_returns_non_negative_integer(
        self, order_model: OrderFrequencyModel, tuesday_date: date
    ) -> None:
        samples = [order_model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)]
        arr = np.array(samples)
        assert np.all(arr >= 0), "All counts must be ≥ 0"
        assert all(isinstance(s, int) for s in samples), "All counts must be ints"

    def test_poisson_mean_matches_lambda(self, tuesday_date: date) -> None:
        """For Tuesday (multiplier=1.0), effective λ = 5.0."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)])
        expected_lambda = DEFAULT_BASE_RATE * 1.0
        assert np.mean(samples) == pytest.approx(expected_lambda, abs=0.2)

    def test_poisson_variance_matches_lambda(self, tuesday_date: date) -> None:
        """Poisson variance = λ. For Tuesday λ = 5.0."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)])
        expected_lambda = DEFAULT_BASE_RATE * 1.0
        assert np.var(samples) == pytest.approx(expected_lambda, abs=0.5)

    def test_friday_effective_lambda(self, friday_date: date) -> None:
        """Friday multiplier=1.2 → λ = 6.0."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(friday_date) for _ in range(SAMPLE_SIZE)])
        assert np.mean(samples) == pytest.approx(6.0, abs=0.2)

    def test_monday_effective_lambda(self, monday_date: date) -> None:
        """Monday multiplier=0.85 → λ = 4.25."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(monday_date) for _ in range(SAMPLE_SIZE)])
        assert np.mean(samples) == pytest.approx(4.25, abs=0.2)

    def test_friday_variance_matches_lambda(self, friday_date: date) -> None:
        """For Friday λ = 6.0, variance should be ≈ 6.0."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(friday_date) for _ in range(SAMPLE_SIZE)])
        assert np.var(samples) == pytest.approx(6.0, abs=0.6)

    def test_poisson_theoretical_comparison(self, tuesday_date: date) -> None:
        """Compare empirical distribution against scipy Poisson reference."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)])
        # Theoretical Poisson mean and variance for λ = 5.0
        theoretical_mean = stats.poisson.mean(mu=5.0)
        theoretical_var = stats.poisson.var(mu=5.0)
        assert np.mean(samples) == pytest.approx(float(theoretical_mean), abs=0.2)
        assert np.var(samples) == pytest.approx(float(theoretical_var), abs=0.5)


# =========================================================================
# TestEffectiveRateCalculation
# =========================================================================


class TestEffectiveRateCalculation:
    """Verify get_effective_rate() returns correct analytical λ values."""

    def test_effective_rate_monday(
        self, order_model: OrderFrequencyModel, monday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(monday_date)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 0.85)

    def test_effective_rate_tuesday(
        self, order_model: OrderFrequencyModel, tuesday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(tuesday_date)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 1.0)

    def test_effective_rate_wednesday(
        self, order_model: OrderFrequencyModel, wednesday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(wednesday_date)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 1.0)

    def test_effective_rate_thursday(
        self, order_model: OrderFrequencyModel, thursday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(thursday_date)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 1.1)

    def test_effective_rate_friday(
        self, order_model: OrderFrequencyModel, friday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(friday_date)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 1.2)

    def test_effective_rate_saturday(
        self, order_model: OrderFrequencyModel, saturday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(saturday_date)
        assert rate == pytest.approx(0.0)

    def test_effective_rate_sunday(
        self, order_model: OrderFrequencyModel, sunday_date: date
    ) -> None:
        rate = order_model.get_effective_rate(sunday_date)
        assert rate == pytest.approx(0.0)

    def test_effective_rate_with_seasonal_multiplier(self, monday_date: date) -> None:
        """January Monday with seasonal 1.5× → base_rate × 0.85 × 1.5."""
        cfg = OrderFrequencyConfig(seasonal_multipliers={1: 1.5, 7: 0.8})
        model = OrderFrequencyModel(config=cfg, seed=42)
        # monday_date is 2025-01-06 (January)
        rate = model.get_effective_rate(monday_date)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 0.85 * 1.5)

    def test_effective_rate_july_friday_seasonal(self) -> None:
        """July Friday with seasonal 0.8× → base_rate × 1.2 × 0.8."""
        cfg = OrderFrequencyConfig(seasonal_multipliers={1: 1.5, 7: 0.8})
        model = OrderFrequencyModel(config=cfg, seed=42)
        july_friday = date(2025, 7, 4)  # 2025-07-04 is a Friday
        assert july_friday.weekday() == 4
        rate = model.get_effective_rate(july_friday)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 1.2 * 0.8)

    def test_effective_rate_month_not_in_seasonal(self) -> None:
        """Month without a seasonal key defaults to 1.0 multiplier."""
        cfg = OrderFrequencyConfig(seasonal_multipliers={1: 1.5})
        model = OrderFrequencyModel(config=cfg, seed=42)
        # March Monday — month 3 not in seasonal dict
        march_monday = date(2025, 3, 3)
        assert march_monday.weekday() == 0
        rate = model.get_effective_rate(march_monday)
        assert rate == pytest.approx(DEFAULT_BASE_RATE * 0.85 * 1.0)


# =========================================================================
# TestWeeklyDistribution
# =========================================================================


class TestWeeklyDistribution:
    """Verify get_weekly_distribution() returns correct per-day rates."""

    def test_weekly_distribution_returns_all_days(
        self, order_model: OrderFrequencyModel
    ) -> None:
        dist = order_model.get_weekly_distribution()
        expected_keys = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        assert set(dist.keys()) == expected_keys

    def test_weekly_distribution_values(
        self, order_model: OrderFrequencyModel
    ) -> None:
        dist = order_model.get_weekly_distribution()
        assert dist["Monday"] == pytest.approx(DEFAULT_BASE_RATE * 0.85)
        assert dist["Tuesday"] == pytest.approx(DEFAULT_BASE_RATE * 1.0)
        assert dist["Wednesday"] == pytest.approx(DEFAULT_BASE_RATE * 1.0)
        assert dist["Thursday"] == pytest.approx(DEFAULT_BASE_RATE * 1.1)
        assert dist["Friday"] == pytest.approx(DEFAULT_BASE_RATE * 1.2)
        assert dist["Saturday"] == pytest.approx(0.0)
        assert dist["Sunday"] == pytest.approx(0.0)

    def test_weekly_total(self, order_model: OrderFrequencyModel) -> None:
        """Sum = base_rate × (0.85 + 1.0 + 1.0 + 1.1 + 1.2 + 0 + 0) = base_rate × 5.15."""
        dist = order_model.get_weekly_distribution()
        total = sum(dist.values())
        expected = DEFAULT_BASE_RATE * (0.85 + 1.0 + 1.0 + 1.1 + 1.2 + 0.0 + 0.0)
        assert total == pytest.approx(expected)

    def test_weekly_distribution_custom_base_rate(
        self, order_model: OrderFrequencyModel
    ) -> None:
        """Override base_rate via the optional parameter."""
        dist = order_model.get_weekly_distribution(base_rate=10.0)
        assert dist["Monday"] == pytest.approx(10.0 * 0.85)
        assert dist["Friday"] == pytest.approx(10.0 * 1.2)
        assert dist["Saturday"] == pytest.approx(0.0)


# =========================================================================
# TestBatchSampling
# =========================================================================


class TestBatchSampling:
    """Verify sample_order_counts_batch() for correctness and reproducibility."""

    def test_batch_returns_correct_count(
        self, order_model: OrderFrequencyModel, tuesday_date: date
    ) -> None:
        dates = [tuesday_date] * 100
        counts = order_model.sample_order_counts_batch(dates)
        assert len(counts) == 100

    def test_batch_returns_list_of_ints(
        self, order_model: OrderFrequencyModel, tuesday_date: date
    ) -> None:
        dates = [tuesday_date] * 50
        counts = order_model.sample_order_counts_batch(dates)
        assert isinstance(counts, list)
        assert all(isinstance(c, (int, np.integer)) for c in counts)

    def test_batch_weekend_dates_all_zero(
        self, order_model: OrderFrequencyModel, saturday_date: date, sunday_date: date
    ) -> None:
        dates = [saturday_date] * 50 + [sunday_date] * 50
        counts = order_model.sample_order_counts_batch(dates)
        assert all(c == 0 for c in counts), "All weekend batch entries must be 0"

    def test_batch_reproducible(self, tuesday_date: date) -> None:
        m1 = OrderFrequencyModel(seed=42)
        m2 = OrderFrequencyModel(seed=42)
        dates = [tuesday_date] * 200
        assert m1.sample_order_counts_batch(dates) == m2.sample_order_counts_batch(dates)

    def test_batch_mixed_weekdays_weekends(self, week_dates: list[date]) -> None:
        model = OrderFrequencyModel(seed=42)
        counts = model.sample_order_counts_batch(week_dates)
        assert len(counts) == 7
        # Indices 5 and 6 are Saturday and Sunday → must be 0
        assert counts[5] == 0
        assert counts[6] == 0
        # Weekday entries should be non-negative
        for idx in range(5):
            assert counts[idx] >= 0

    def test_batch_empty_dates(self, order_model: OrderFrequencyModel) -> None:
        counts = order_model.sample_order_counts_batch([])
        assert counts == []

    def test_batch_single_date(
        self, order_model: OrderFrequencyModel, friday_date: date
    ) -> None:
        counts = order_model.sample_order_counts_batch([friday_date])
        assert len(counts) == 1
        assert isinstance(counts[0], (int, np.integer))
        assert counts[0] >= 0


# =========================================================================
# TestMinMaxClipping
# =========================================================================


class TestMinMaxClipping:
    """Verify min_orders / max_orders clamping behaviour."""

    def test_orders_clipped_to_min(self, tuesday_date: date) -> None:
        """With min_orders=0 (default), no negative values should appear."""
        model = OrderFrequencyModel(seed=42)
        samples = [model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)]
        assert all(s >= 0 for s in samples)

    def test_orders_clipped_to_max(self, tuesday_date: date) -> None:
        """A very high base_rate should still be clipped to max_orders."""
        cfg = OrderFrequencyConfig(base_rate=100.0, max_orders=20)
        model = OrderFrequencyModel(config=cfg, seed=42)
        samples = [model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)]
        assert all(s <= 20 for s in samples), "All counts must be ≤ max_orders=20"

    def test_clipping_applies_in_batch(self, tuesday_date: date) -> None:
        """Batch sampling must also respect max_orders."""
        cfg = OrderFrequencyConfig(base_rate=100.0, max_orders=15)
        model = OrderFrequencyModel(config=cfg, seed=42)
        dates = [tuesday_date] * 500
        counts = model.sample_order_counts_batch(dates)
        assert all(c <= 15 for c in counts)

    def test_custom_min_orders(self, tuesday_date: date) -> None:
        """Setting min_orders > 0 ensures a floor on weekday counts."""
        cfg = OrderFrequencyConfig(base_rate=0.5, min_orders=1)
        model = OrderFrequencyModel(config=cfg, seed=42)
        samples = [model.sample_order_count(tuesday_date) for _ in range(500)]
        # Poisson(0.5) would produce many zeros, but min_orders=1 clamps them up
        assert all(s >= 1 for s in samples)

    def test_weekend_not_affected_by_min_orders(self, saturday_date: date) -> None:
        """Weekends should return 0 regardless of min_orders because λ=0."""
        cfg = OrderFrequencyConfig(min_orders=5)
        model = OrderFrequencyModel(config=cfg, seed=42)
        # Saturday multiplier = 0.0 → short-circuits to return 0 before clipping
        count = model.sample_order_count(saturday_date)
        assert count == 0


# =========================================================================
# TestCustomerTypeFrequency
# =========================================================================


class TestCustomerTypeFrequency:
    """Verify sample_customer_order_frequency() tier multipliers."""

    def test_strategic_customer_higher_rate(self) -> None:
        """Strategic tier (2×) should produce roughly double the base rate."""
        model = OrderFrequencyModel(seed=42)
        samples = [model.sample_customer_order_frequency("strategic") for _ in range(SAMPLE_SIZE)]
        mean = float(np.mean(samples))
        # 2 × base_rate = 10.0
        assert mean == pytest.approx(DEFAULT_BASE_RATE * 2.0, abs=0.5)

    def test_transactional_customer_lower_rate(self) -> None:
        """Transactional tier (0.5×) → half base rate."""
        model = OrderFrequencyModel(seed=42)
        samples = [model.sample_customer_order_frequency("transactional") for _ in range(SAMPLE_SIZE)]
        mean = float(np.mean(samples))
        assert mean == pytest.approx(DEFAULT_BASE_RATE * 0.5, abs=0.3)

    def test_standard_customer_base_rate(self) -> None:
        """Standard tier (1×) → exactly base rate."""
        model = OrderFrequencyModel(seed=42)
        samples = [model.sample_customer_order_frequency("standard") for _ in range(SAMPLE_SIZE)]
        mean = float(np.mean(samples))
        assert mean == pytest.approx(DEFAULT_BASE_RATE * 1.0, abs=0.3)

    def test_unknown_type_falls_back_to_standard(self) -> None:
        """Unrecognised customer type should fall back to 'standard' (1×)."""
        model = OrderFrequencyModel(seed=42)
        samples = [model.sample_customer_order_frequency("unknown_tier") for _ in range(SAMPLE_SIZE)]
        mean = float(np.mean(samples))
        assert mean == pytest.approx(DEFAULT_BASE_RATE * 1.0, abs=0.3)

    def test_customer_frequency_returns_int(self) -> None:
        model = OrderFrequencyModel(seed=42)
        result = model.sample_customer_order_frequency("standard")
        assert isinstance(result, int)

    def test_customer_frequency_non_negative(self) -> None:
        model = OrderFrequencyModel(seed=42)
        samples = [model.sample_customer_order_frequency("transactional") for _ in range(1_000)]
        assert all(s >= 0 for s in samples)

    def test_customer_frequency_clipped_to_max(self) -> None:
        """Max-orders clipping applies to customer-type samples too."""
        cfg = OrderFrequencyConfig(base_rate=100.0, max_orders=25)
        model = OrderFrequencyModel(config=cfg, seed=42)
        samples = [model.sample_customer_order_frequency("strategic") for _ in range(SAMPLE_SIZE)]
        assert all(s <= 25 for s in samples)


# =========================================================================
# TestOrderTimingWithinDay
# =========================================================================


class TestOrderTimingWithinDay:
    """Verify sample_order_times() generates sorted, bounded arrival hours."""

    def test_order_times_within_business_hours(self, order_model: OrderFrequencyModel) -> None:
        times = order_model.sample_order_times(count=100)
        for t in times:
            assert 8.0 <= t < 17.0, f"Time {t} outside business hours"

    def test_order_times_sorted(self, order_model: OrderFrequencyModel) -> None:
        times = order_model.sample_order_times(count=200)
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1], "Times must be sorted"

    def test_order_times_count_matches(self, order_model: OrderFrequencyModel) -> None:
        for requested in (0, 1, 5, 50, 100):
            times = order_model.sample_order_times(count=requested)
            assert len(times) == max(requested, 0)

    def test_order_times_zero_count(self, order_model: OrderFrequencyModel) -> None:
        times = order_model.sample_order_times(count=0)
        assert times == []

    def test_order_times_negative_count(self, order_model: OrderFrequencyModel) -> None:
        times = order_model.sample_order_times(count=-5)
        assert times == []

    def test_order_times_returns_floats(self, order_model: OrderFrequencyModel) -> None:
        times = order_model.sample_order_times(count=10)
        assert all(isinstance(t, float) for t in times)

    def test_order_times_custom_hours(self) -> None:
        """Custom business hours should be respected."""
        model = OrderFrequencyModel(seed=42)
        times = model.sample_order_times(count=50, business_start_hour=9, business_end_hour=15)
        for t in times:
            assert 9.0 <= t < 15.0

    def test_order_times_invalid_hours_returns_empty(self) -> None:
        """If start >= end, an empty list is returned."""
        model = OrderFrequencyModel(seed=42)
        times = model.sample_order_times(count=10, business_start_hour=17, business_end_hour=8)
        assert times == []


# =========================================================================
# TestEdgeCases
# =========================================================================


class TestEdgeCases:
    """Edge cases: extreme rates, single dates, reproducibility guarantees."""

    def test_very_high_base_rate(self, tuesday_date: date) -> None:
        """base_rate=100 still produces valid integers."""
        cfg = OrderFrequencyConfig(base_rate=100.0, max_orders=500)
        model = OrderFrequencyModel(config=cfg, seed=42)
        samples = [model.sample_order_count(tuesday_date) for _ in range(1_000)]
        assert all(isinstance(s, int) for s in samples)
        assert all(0 <= s <= 500 for s in samples)

    def test_single_date_batch(
        self, order_model: OrderFrequencyModel, tuesday_date: date
    ) -> None:
        counts = order_model.sample_order_counts_batch([tuesday_date])
        assert isinstance(counts, list)
        assert len(counts) == 1

    def test_reproducibility_across_calls(self, tuesday_date: date) -> None:
        """Sequential calls to two identically-seeded models yield same sequence."""
        m1 = OrderFrequencyModel(seed=42)
        m2 = OrderFrequencyModel(seed=42)
        seq1 = [m1.sample_order_count(tuesday_date) for _ in range(50)]
        seq2 = [m2.sample_order_count(tuesday_date) for _ in range(50)]
        assert seq1 == seq2

    def test_different_seeds_different_results(self, tuesday_date: date) -> None:
        m1 = OrderFrequencyModel(seed=42)
        m2 = OrderFrequencyModel(seed=99)
        seq1 = [m1.sample_order_count(tuesday_date) for _ in range(100)]
        seq2 = [m2.sample_order_count(tuesday_date) for _ in range(100)]
        # Extremely unlikely to be identical with different seeds
        assert seq1 != seq2

    def test_large_batch_does_not_error(self, tuesday_date: date) -> None:
        """Batch of 10,000 dates completes without error."""
        model = OrderFrequencyModel(seed=42)
        dates = [tuesday_date] * 10_000
        counts = model.sample_order_counts_batch(dates)
        assert len(counts) == 10_000

    def test_config_min_orders_floor(self) -> None:
        """OrderFrequencyConfig enforces min_orders ≥ 0."""
        cfg = OrderFrequencyConfig(min_orders=0)
        assert cfg.min_orders == 0

    def test_model_with_none_seasonal_multipliers(self, tuesday_date: date) -> None:
        """None seasonal_multipliers should not cause errors."""
        cfg = OrderFrequencyConfig(seasonal_multipliers=None)
        model = OrderFrequencyModel(config=cfg, seed=42)
        count = model.sample_order_count(tuesday_date)
        assert isinstance(count, int)
        assert count >= 0


# =========================================================================
# TestStatisticalDistributionVerification
# =========================================================================


class TestStatisticalDistributionVerification:
    """Deep statistical verification: mean ≈ var ≈ λ for Poisson."""

    def test_mean_variance_equality_tuesday(self, tuesday_date: date) -> None:
        """Poisson fundamental property: E[X] = Var(X) = λ."""
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)])
        lam = DEFAULT_BASE_RATE * 1.0
        assert np.mean(samples) == pytest.approx(lam, abs=0.2)
        assert np.var(samples) == pytest.approx(lam, abs=0.5)

    def test_mean_variance_equality_friday(self, friday_date: date) -> None:
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(friday_date) for _ in range(SAMPLE_SIZE)])
        lam = DEFAULT_BASE_RATE * 1.2
        assert np.mean(samples) == pytest.approx(lam, abs=0.2)
        assert np.var(samples) == pytest.approx(lam, abs=0.6)

    def test_mean_variance_equality_monday(self, monday_date: date) -> None:
        model = OrderFrequencyModel(seed=42)
        samples = np.array([model.sample_order_count(monday_date) for _ in range(SAMPLE_SIZE)])
        lam = DEFAULT_BASE_RATE * 0.85
        assert np.mean(samples) == pytest.approx(lam, abs=0.2)
        assert np.var(samples) == pytest.approx(lam, abs=0.5)

    def test_seasonal_multiplier_affects_distribution(self, monday_date: date) -> None:
        """With January seasonal=1.5 the effective lambda shifts proportionally."""
        cfg = OrderFrequencyConfig(seasonal_multipliers={1: 1.5})
        model = OrderFrequencyModel(config=cfg, seed=42)
        samples = np.array([model.sample_order_count(monday_date) for _ in range(SAMPLE_SIZE)])
        expected_lam = DEFAULT_BASE_RATE * 0.85 * 1.5
        assert np.mean(samples) == pytest.approx(expected_lam, abs=0.3)

    def test_batch_matches_individual_statistics(self, tuesday_date: date) -> None:
        """Batch-sampled counts should have same distribution as individual calls."""
        model_batch = OrderFrequencyModel(seed=42)
        dates = [tuesday_date] * SAMPLE_SIZE
        batch_counts = np.array(model_batch.sample_order_counts_batch(dates))

        model_indiv = OrderFrequencyModel(seed=42)
        indiv_counts = np.array([model_indiv.sample_order_count(tuesday_date) for _ in range(SAMPLE_SIZE)])

        # Both should converge to the same lambda
        lam = DEFAULT_BASE_RATE * 1.0
        assert np.mean(batch_counts) == pytest.approx(lam, abs=0.2)
        assert np.mean(indiv_counts) == pytest.approx(lam, abs=0.2)
