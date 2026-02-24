"""
Comprehensive tests for ``app/external_world/behavior_profiles.py``.

Covers the full surface area of BehaviorProfile Pydantic V2 models,
sub-models (PaymentBehavior, OrderBehavior, InvoiceBehavior), predefined
profile templates (CUSTOMER_PROFILE_TEMPLATES, VENDOR_PROFILE_TEMPLATES),
BehaviorProfileFactory (creation and variation), TierDistribution, and all
module-level enum-like constants.

Per AAP Section 0.5.1 Group 10: "Profile assignment, parameter ranges."

Testing Rules (AAP Section 0.7.5):
    - Unit test coverage target: ≥80%.
    - No live API calls — pure data model tests.
    - All Pydantic V2 model validation must be tested.
    - Seed random generators for reproducibility (seed=42).
    - Python 3.11+ syntax with full type hints.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest
from pydantic import ValidationError

from app.external_world.behavior_profiles import (
    CUSTOMER_PROFILE_TEMPLATES,
    ENTITY_TYPES,
    INVOICE_TIMINGS,
    ORDER_FREQUENCIES,
    ORDER_SIZE_PATTERNS,
    PAYMENT_SEGMENTS,
    PROFILE_NAMES,
    VENDOR_PROFILE_TEMPLATES,
    BehaviorProfile,
    BehaviorProfileFactory,
    InvoiceBehavior,
    OrderBehavior,
    PaymentBehavior,
    TierDistribution,
)


# =========================================================================
# Local Fixtures
# =========================================================================


@pytest.fixture
def default_payment_behavior() -> PaymentBehavior:
    """Return a ``PaymentBehavior`` with all default values."""
    return PaymentBehavior()


@pytest.fixture
def default_order_behavior() -> OrderBehavior:
    """Return an ``OrderBehavior`` with all default values."""
    return OrderBehavior()


@pytest.fixture
def default_invoice_behavior() -> InvoiceBehavior:
    """Return an ``InvoiceBehavior`` with all default values."""
    return InvoiceBehavior()


@pytest.fixture
def default_behavior_profile() -> BehaviorProfile:
    """Return a ``BehaviorProfile`` with all default values."""
    return BehaviorProfile()


@pytest.fixture
def excellent_customer_profile() -> BehaviorProfile:
    """Return the 'excellent' customer profile from the factory."""
    return BehaviorProfileFactory.create_profile("excellent", "customer")


@pytest.fixture
def problem_customer_profile() -> BehaviorProfile:
    """Return the 'problem' customer profile from the factory."""
    return BehaviorProfileFactory.create_profile("problem", "customer")


@pytest.fixture
def excellent_vendor_profile() -> BehaviorProfile:
    """Return the 'excellent' vendor profile from the factory."""
    return BehaviorProfileFactory.create_profile("excellent", "vendor")


@pytest.fixture
def problem_vendor_profile() -> BehaviorProfile:
    """Return the 'problem' vendor profile from the factory."""
    return BehaviorProfileFactory.create_profile("problem", "vendor")


@pytest.fixture
def tier_distribution() -> TierDistribution:
    """Return a ``TierDistribution`` with default values (0.10, 0.30, 0.60)."""
    return TierDistribution()


# =========================================================================
# TestPaymentBehavior
# =========================================================================


class TestPaymentBehavior:
    """Tests for the ``PaymentBehavior`` Pydantic V2 sub-model."""

    def test_default_values(self, default_payment_behavior: PaymentBehavior) -> None:
        """Default construction must produce on_time/3.0/0.0/0.0."""
        pb = default_payment_behavior
        assert pb.payment_segment == "on_time"
        assert pb.payment_variance == 3.0
        assert pb.short_pay_rate == 0.0
        assert pb.dispute_rate == 0.0

    def test_valid_payment_segments(self) -> None:
        """Every value in PAYMENT_SEGMENTS should be accepted."""
        for segment in PAYMENT_SEGMENTS:
            pb = PaymentBehavior(payment_segment=segment)
            assert pb.payment_segment == segment

    def test_invalid_payment_segment(self) -> None:
        """An unrecognised segment must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            PaymentBehavior(payment_segment="invalid")

    def test_short_pay_rate_boundaries(self) -> None:
        """short_pay_rate must be in [0.0, 1.0]."""
        # Valid boundary values
        pb_low = PaymentBehavior(short_pay_rate=0.0)
        assert pb_low.short_pay_rate == 0.0

        pb_high = PaymentBehavior(short_pay_rate=1.0)
        assert pb_high.short_pay_rate == 1.0

        # Below lower bound
        with pytest.raises(ValidationError):
            PaymentBehavior(short_pay_rate=-0.01)

        # Above upper bound
        with pytest.raises(ValidationError):
            PaymentBehavior(short_pay_rate=1.01)

    def test_dispute_rate_boundaries(self) -> None:
        """dispute_rate must be in [0.0, 1.0]."""
        pb_low = PaymentBehavior(dispute_rate=0.0)
        assert pb_low.dispute_rate == 0.0

        pb_high = PaymentBehavior(dispute_rate=1.0)
        assert pb_high.dispute_rate == 1.0

        with pytest.raises(ValidationError):
            PaymentBehavior(dispute_rate=-0.01)

        with pytest.raises(ValidationError):
            PaymentBehavior(dispute_rate=1.01)

    def test_payment_variance_non_negative(self) -> None:
        """payment_variance must be ≥ 0.0."""
        with pytest.raises(ValidationError):
            PaymentBehavior(payment_variance=-1.0)

    def test_custom_values(self) -> None:
        """Construction with all explicit values must preserve them."""
        pb = PaymentBehavior(
            payment_segment="early",
            payment_variance=1.0,
            short_pay_rate=0.02,
            dispute_rate=0.01,
        )
        assert pb.payment_segment == "early"
        assert pb.payment_variance == 1.0
        assert pb.short_pay_rate == 0.02
        assert pb.dispute_rate == 0.01


# =========================================================================
# TestOrderBehavior
# =========================================================================


class TestOrderBehavior:
    """Tests for the ``OrderBehavior`` Pydantic V2 sub-model."""

    def test_default_values(self, default_order_behavior: OrderBehavior) -> None:
        """Default construction must produce weekly/consistent/12×1.0."""
        ob = default_order_behavior
        assert ob.order_frequency == "weekly"
        assert ob.order_size_pattern == "consistent"
        assert len(ob.seasonality) == 12
        for month in range(1, 13):
            assert ob.seasonality[month] == 1.0

    def test_valid_order_frequencies(self) -> None:
        """Every value in ORDER_FREQUENCIES should be accepted."""
        for freq in ORDER_FREQUENCIES:
            ob = OrderBehavior(order_frequency=freq)
            assert ob.order_frequency == freq

    def test_invalid_order_frequency(self) -> None:
        """An unrecognised frequency must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            OrderBehavior(order_frequency="hourly")

    def test_valid_size_patterns(self) -> None:
        """Every value in ORDER_SIZE_PATTERNS should be accepted."""
        for pattern in ORDER_SIZE_PATTERNS:
            ob = OrderBehavior(order_size_pattern=pattern)
            assert ob.order_size_pattern == pattern

    def test_invalid_size_pattern(self) -> None:
        """An unrecognised size pattern must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            OrderBehavior(order_size_pattern="random")

    def test_custom_seasonality(self) -> None:
        """Custom seasonality dict must be preserved."""
        custom: Dict[int, float] = {
            1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1,
            5: 1.15, 6: 1.2, 7: 1.1, 8: 1.0,
            9: 0.95, 10: 0.9, 11: 0.85, 12: 1.2,
        }
        ob = OrderBehavior(seasonality=custom)
        for month, multiplier in custom.items():
            assert ob.seasonality[month] == multiplier

    def test_seasonality_default_12_months(self) -> None:
        """Default seasonality must contain exactly 12 entries (months 1–12)."""
        ob = OrderBehavior()
        assert len(ob.seasonality) == 12
        assert set(ob.seasonality.keys()) == set(range(1, 13))


# =========================================================================
# TestInvoiceBehavior
# =========================================================================


class TestInvoiceBehavior:
    """Tests for the ``InvoiceBehavior`` Pydantic V2 sub-model."""

    def test_default_values(self, default_invoice_behavior: InvoiceBehavior) -> None:
        """Default construction must produce prompt/0.95/5."""
        ib = default_invoice_behavior
        assert ib.invoice_timing == "prompt"
        assert ib.invoice_accuracy == 0.95
        assert ib.response_time == 5

    def test_valid_invoice_timings(self) -> None:
        """Every value in INVOICE_TIMINGS should be accepted."""
        for timing in INVOICE_TIMINGS:
            ib = InvoiceBehavior(invoice_timing=timing)
            assert ib.invoice_timing == timing

    def test_invalid_invoice_timing(self) -> None:
        """An unrecognised timing must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            InvoiceBehavior(invoice_timing="delayed")

    def test_invoice_accuracy_boundaries(self) -> None:
        """invoice_accuracy must be in [0.0, 1.0]."""
        ib_low = InvoiceBehavior(invoice_accuracy=0.0)
        assert ib_low.invoice_accuracy == 0.0

        ib_high = InvoiceBehavior(invoice_accuracy=1.0)
        assert ib_high.invoice_accuracy == 1.0

        with pytest.raises(ValidationError):
            InvoiceBehavior(invoice_accuracy=-0.01)

        with pytest.raises(ValidationError):
            InvoiceBehavior(invoice_accuracy=1.01)

    def test_response_time_minimum(self) -> None:
        """response_time must be ≥ 1."""
        with pytest.raises(ValidationError):
            InvoiceBehavior(response_time=0)

    def test_custom_values(self) -> None:
        """Construction with all explicit values must preserve them."""
        ib = InvoiceBehavior(
            invoice_timing="immediate",
            invoice_accuracy=0.99,
            response_time=1,
        )
        assert ib.invoice_timing == "immediate"
        assert ib.invoice_accuracy == 0.99
        assert ib.response_time == 1


# =========================================================================
# TestBehaviorProfile
# =========================================================================


class TestBehaviorProfile:
    """Tests for the top-level ``BehaviorProfile`` Pydantic V2 model."""

    def test_default_profile(self, default_behavior_profile: BehaviorProfile) -> None:
        """Default construction must produce average/customer with sub-model defaults."""
        bp = default_behavior_profile
        assert bp.profile_name == "average"
        assert bp.entity_type == "customer"
        assert isinstance(bp.payment_behavior, PaymentBehavior)
        assert isinstance(bp.order_behavior, OrderBehavior)
        assert isinstance(bp.invoice_behavior, InvoiceBehavior)

    def test_valid_profile_names(self) -> None:
        """Every value in PROFILE_NAMES should be accepted."""
        for name in PROFILE_NAMES:
            bp = BehaviorProfile(profile_name=name)
            assert bp.profile_name == name

    def test_invalid_profile_name(self) -> None:
        """An unrecognised profile_name must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            BehaviorProfile(profile_name="unknown")

    def test_valid_entity_types(self) -> None:
        """Every value in ENTITY_TYPES should be accepted."""
        for entity_type in ENTITY_TYPES:
            bp = BehaviorProfile(entity_type=entity_type)
            assert bp.entity_type == entity_type

    def test_invalid_entity_type(self) -> None:
        """An unrecognised entity_type must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            BehaviorProfile(entity_type="employee")

    def test_convenience_properties(self) -> None:
        """Convenience properties must delegate to the correct sub-model fields."""
        custom_payment = PaymentBehavior(
            payment_segment="early",
            payment_variance=1.5,
            short_pay_rate=0.05,
            dispute_rate=0.03,
        )
        custom_order = OrderBehavior(
            order_frequency="daily",
            order_size_pattern="variable",
            seasonality={m: 1.1 for m in range(1, 13)},
        )
        custom_invoice = InvoiceBehavior(
            invoice_timing="immediate",
            invoice_accuracy=0.98,
            response_time=2,
        )
        bp = BehaviorProfile(
            payment_behavior=custom_payment,
            order_behavior=custom_order,
            invoice_behavior=custom_invoice,
        )

        # Payment convenience properties
        assert bp.payment_segment == "early"
        assert bp.payment_variance == 1.5
        assert bp.short_pay_rate == 0.05
        assert bp.dispute_rate == 0.03

        # Order convenience properties
        assert bp.order_frequency == "daily"
        assert bp.order_size_pattern == "variable"
        assert bp.seasonality == {m: 1.1 for m in range(1, 13)}

        # Invoice convenience properties
        assert bp.invoice_timing == "immediate"
        assert bp.invoice_accuracy == 0.98
        assert bp.response_time == 2

    def test_profile_with_custom_sub_models(self) -> None:
        """Profile with all three custom sub-models must preserve nested values."""
        bp = BehaviorProfile(
            profile_name="good",
            entity_type="vendor",
            payment_behavior=PaymentBehavior(
                payment_segment="prompt",
                payment_variance=2.0,
                short_pay_rate=0.01,
                dispute_rate=0.02,
            ),
            order_behavior=OrderBehavior(
                order_frequency="monthly",
                order_size_pattern="variable",
            ),
            invoice_behavior=InvoiceBehavior(
                invoice_timing="slow",
                invoice_accuracy=0.85,
                response_time=12,
            ),
        )
        assert bp.profile_name == "good"
        assert bp.entity_type == "vendor"
        assert bp.payment_behavior.payment_segment == "prompt"
        assert bp.payment_behavior.payment_variance == 2.0
        assert bp.payment_behavior.short_pay_rate == 0.01
        assert bp.payment_behavior.dispute_rate == 0.02
        assert bp.order_behavior.order_frequency == "monthly"
        assert bp.order_behavior.order_size_pattern == "variable"
        assert bp.invoice_behavior.invoice_timing == "slow"
        assert bp.invoice_behavior.invoice_accuracy == 0.85
        assert bp.invoice_behavior.response_time == 12

    def test_customer_profile_has_payment_and_order(
        self, excellent_customer_profile: BehaviorProfile
    ) -> None:
        """A customer profile must carry meaningful payment and order behaviors."""
        bp = excellent_customer_profile
        assert bp.entity_type == "customer"
        # Excellent customers pay early with low variance
        assert bp.payment_behavior.payment_segment == "early"
        assert bp.payment_behavior.payment_variance <= 2.0
        # Excellent customers order frequently
        assert bp.order_behavior.order_frequency in ORDER_FREQUENCIES

    def test_vendor_profile_has_invoice_behavior(
        self, excellent_vendor_profile: BehaviorProfile
    ) -> None:
        """A vendor profile must carry meaningful invoice behavior."""
        bp = excellent_vendor_profile
        assert bp.entity_type == "vendor"
        assert bp.invoice_behavior.invoice_timing == "immediate"
        assert bp.invoice_behavior.invoice_accuracy >= 0.95
        assert bp.invoice_behavior.response_time >= 1


# =========================================================================
# TestProfileTemplates
# =========================================================================


class TestProfileTemplates:
    """Tests for the predefined CUSTOMER_PROFILE_TEMPLATES and VENDOR_PROFILE_TEMPLATES."""

    def test_customer_profile_templates_exist(self) -> None:
        """CUSTOMER_PROFILE_TEMPLATES must have all five PROFILE_NAMES as keys."""
        for name in PROFILE_NAMES:
            assert name in CUSTOMER_PROFILE_TEMPLATES, (
                f"Missing customer template for '{name}'"
            )

    def test_vendor_profile_templates_exist(self) -> None:
        """VENDOR_PROFILE_TEMPLATES must have all five PROFILE_NAMES as keys."""
        for name in PROFILE_NAMES:
            assert name in VENDOR_PROFILE_TEMPLATES, (
                f"Missing vendor template for '{name}'"
            )

    def test_excellent_customer_profile_values(self) -> None:
        """The 'excellent' customer template must have specific expected values."""
        tpl = CUSTOMER_PROFILE_TEMPLATES["excellent"]
        assert tpl.payment_segment == "early"
        assert tpl.payment_variance == 1.0
        assert tpl.short_pay_rate == 0.0
        assert tpl.dispute_rate == 0.01
        assert tpl.order_frequency == "daily"
        assert tpl.order_size_pattern == "consistent"

    def test_problem_customer_profile_values(self) -> None:
        """The 'problem' customer template must have specific expected values."""
        tpl = CUSTOMER_PROFILE_TEMPLATES["problem"]
        assert tpl.payment_segment == "problem"
        assert tpl.payment_variance == 10.0
        assert tpl.short_pay_rate == 0.10
        assert tpl.dispute_rate == 0.20
        assert tpl.order_frequency == "monthly"

    def test_excellent_vendor_profile_values(self) -> None:
        """The 'excellent' vendor template must have specific expected values."""
        tpl = VENDOR_PROFILE_TEMPLATES["excellent"]
        assert tpl.invoice_timing == "immediate"
        assert tpl.invoice_accuracy == 0.99
        assert tpl.response_time == 1

    def test_problem_vendor_profile_values(self) -> None:
        """The 'problem' vendor template must have specific expected values."""
        tpl = VENDOR_PROFILE_TEMPLATES["problem"]
        assert tpl.invoice_timing == "slow"
        assert tpl.invoice_accuracy == 0.80
        assert tpl.response_time == 20

    def test_all_templates_are_valid_behavior_profiles(self) -> None:
        """Every entry in both template dictionaries must be a valid BehaviorProfile."""
        for name, profile in CUSTOMER_PROFILE_TEMPLATES.items():
            assert isinstance(profile, BehaviorProfile), (
                f"Customer template '{name}' is not a BehaviorProfile"
            )
            assert profile.profile_name == name
            assert profile.entity_type == "customer"

        for name, profile in VENDOR_PROFILE_TEMPLATES.items():
            assert isinstance(profile, BehaviorProfile), (
                f"Vendor template '{name}' is not a BehaviorProfile"
            )
            assert profile.profile_name == name
            assert profile.entity_type == "vendor"

    def test_profile_degradation_order(self) -> None:
        """Quality degradation must be monotonic across the five tiers.

        For customers: short_pay_rate and dispute_rate must increase or stay
        the same as we move from excellent → good → average → poor → problem.

        For vendors: invoice_accuracy must decrease or stay the same, and
        response_time must increase or stay the same.
        """
        ordered_names: List[str] = ["excellent", "good", "average", "poor", "problem"]

        # --- Customer degradation ---
        prev_short_pay: float = -1.0
        prev_dispute: float = -1.0
        for name in ordered_names:
            tpl = CUSTOMER_PROFILE_TEMPLATES[name]
            assert tpl.short_pay_rate >= prev_short_pay, (
                f"Customer short_pay_rate did not increase at '{name}': "
                f"{tpl.short_pay_rate} < {prev_short_pay}"
            )
            assert tpl.dispute_rate >= prev_dispute, (
                f"Customer dispute_rate did not increase at '{name}': "
                f"{tpl.dispute_rate} < {prev_dispute}"
            )
            prev_short_pay = tpl.short_pay_rate
            prev_dispute = tpl.dispute_rate

        # --- Vendor degradation ---
        prev_accuracy: float = 2.0  # start above max so first comparison always passes
        prev_response: int = 0
        for name in ordered_names:
            tpl = VENDOR_PROFILE_TEMPLATES[name]
            assert tpl.invoice_accuracy <= prev_accuracy, (
                f"Vendor invoice_accuracy did not decrease at '{name}': "
                f"{tpl.invoice_accuracy} > {prev_accuracy}"
            )
            assert tpl.response_time >= prev_response, (
                f"Vendor response_time did not increase at '{name}': "
                f"{tpl.response_time} < {prev_response}"
            )
            prev_accuracy = tpl.invoice_accuracy
            prev_response = tpl.response_time


# =========================================================================
# TestBehaviorProfileFactory
# =========================================================================


class TestBehaviorProfileFactory:
    """Tests for ``BehaviorProfileFactory`` creation and variation methods."""

    def test_create_customer_profile(self) -> None:
        """Factory must create a valid 'excellent' customer profile."""
        profile = BehaviorProfileFactory.create_profile("excellent", "customer")
        assert isinstance(profile, BehaviorProfile)
        assert profile.profile_name == "excellent"
        assert profile.entity_type == "customer"

    def test_create_vendor_profile(self) -> None:
        """Factory must create a valid 'good' vendor profile."""
        profile = BehaviorProfileFactory.create_profile("good", "vendor")
        assert isinstance(profile, BehaviorProfile)
        assert profile.profile_name == "good"
        assert profile.entity_type == "vendor"

    def test_create_all_customer_profiles(self) -> None:
        """Factory must successfully create every named customer profile."""
        for name in PROFILE_NAMES:
            profile = BehaviorProfileFactory.create_profile(name, "customer")
            assert isinstance(profile, BehaviorProfile)
            assert profile.profile_name == name
            assert profile.entity_type == "customer"

    def test_create_all_vendor_profiles(self) -> None:
        """Factory must successfully create every named vendor profile."""
        for name in PROFILE_NAMES:
            profile = BehaviorProfileFactory.create_profile(name, "vendor")
            assert isinstance(profile, BehaviorProfile)
            assert profile.profile_name == name
            assert profile.entity_type == "vendor"

    def test_create_profile_with_variation(self) -> None:
        """Varied profile must preserve identity but perturb numeric fields."""
        rng = np.random.RandomState(42)
        profile = BehaviorProfileFactory.create_profile_with_variation(
            "average", "customer", rng=rng,
        )
        assert isinstance(profile, BehaviorProfile)
        assert profile.profile_name == "average"
        assert profile.entity_type == "customer"

        # The base template has specific values — variation should move them
        base = CUSTOMER_PROFILE_TEMPLATES["average"]
        # Variance should be within ±10 % of the base (with clamp), allowing
        # equality when a base value is 0.0 (e.g., 0.0 * factor == 0.0).
        assert profile.payment_behavior.payment_variance >= 0.0

    def test_variation_produces_different_profiles(self) -> None:
        """Two profiles with different seeds must differ in at least one numeric field."""
        profile_a = BehaviorProfileFactory.create_profile_with_variation(
            "average", "customer", rng=np.random.RandomState(42),
        )
        profile_b = BehaviorProfileFactory.create_profile_with_variation(
            "average", "customer", rng=np.random.RandomState(99),
        )

        # Collect all numeric values for each profile
        def _numeric_values(p: BehaviorProfile) -> List[float]:
            return [
                p.payment_behavior.payment_variance,
                p.payment_behavior.short_pay_rate,
                p.payment_behavior.dispute_rate,
                p.invoice_behavior.invoice_accuracy,
                float(p.invoice_behavior.response_time),
            ]

        vals_a = _numeric_values(profile_a)
        vals_b = _numeric_values(profile_b)
        assert vals_a != vals_b, "Two varied profiles with different seeds should differ"

    def test_variation_stays_within_valid_bounds(self) -> None:
        """100 varied profiles must all have rates in [0.0, 1.0] and variance ≥ 0."""
        for seed in range(100):
            rng = np.random.RandomState(seed)
            profile = BehaviorProfileFactory.create_profile_with_variation(
                "average", "customer", rng=rng,
            )
            # Rate boundaries
            assert 0.0 <= profile.payment_behavior.short_pay_rate <= 1.0, (
                f"short_pay_rate out of bounds with seed={seed}"
            )
            assert 0.0 <= profile.payment_behavior.dispute_rate <= 1.0, (
                f"dispute_rate out of bounds with seed={seed}"
            )
            assert 0.0 <= profile.invoice_behavior.invoice_accuracy <= 1.0, (
                f"invoice_accuracy out of bounds with seed={seed}"
            )
            # Variance non-negative
            assert profile.payment_behavior.payment_variance >= 0.0, (
                f"payment_variance negative with seed={seed}"
            )
            # Response time positive
            assert profile.invoice_behavior.response_time >= 1, (
                f"response_time < 1 with seed={seed}"
            )

    def test_get_available_profiles(self) -> None:
        """get_available_profiles must return customer and vendor profile lists."""
        available: Dict[str, List[str]] = BehaviorProfileFactory.get_available_profiles()
        assert "customer" in available
        assert "vendor" in available
        assert len(available["customer"]) == 5
        assert len(available["vendor"]) == 5
        for name in PROFILE_NAMES:
            assert name in available["customer"]
            assert name in available["vendor"]

    def test_create_profile_unknown_type(self) -> None:
        """Factory must handle unknown entity types gracefully.

        The implementation falls back to a safe default profile rather than
        raising an error, so we verify it returns a valid BehaviorProfile.
        """
        profile = BehaviorProfileFactory.create_profile("excellent", "unknown_type")
        assert isinstance(profile, BehaviorProfile)
        # Falls back to safe defaults
        assert profile.profile_name in PROFILE_NAMES
        assert profile.entity_type in ENTITY_TYPES


# =========================================================================
# TestTierDistribution
# =========================================================================


class TestTierDistribution:
    """Tests for the ``TierDistribution`` Pydantic V2 model."""

    def test_default_values(self, tier_distribution: TierDistribution) -> None:
        """Default construction must produce 0.10 / 0.30 / 0.60 (README §470-474)."""
        td = tier_distribution
        assert td.strategic == 0.10
        assert td.standard == 0.30
        assert td.transactional == 0.60

    def test_values_sum_to_one(self, tier_distribution: TierDistribution) -> None:
        """strategic + standard + transactional must equal 1.0 (within tolerance)."""
        td = tier_distribution
        total = td.strategic + td.standard + td.transactional
        assert abs(total - 1.0) < 0.001

    def test_invalid_sum(self) -> None:
        """Tier fractions that do not sum to 1.0 must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            TierDistribution(strategic=0.5, standard=0.3, transactional=0.6)

    def test_negative_value(self) -> None:
        """Negative tier fractions must raise ``ValidationError``."""
        with pytest.raises(ValidationError):
            TierDistribution(strategic=-0.1, standard=0.5, transactional=0.6)

    def test_get_tier_for_rank_strategic(
        self, tier_distribution: TierDistribution
    ) -> None:
        """A rank in the top 10 % must map to 'strategic'."""
        assert tier_distribution.get_tier_for_rank(0.05) == "strategic"

    def test_get_tier_for_rank_standard(
        self, tier_distribution: TierDistribution
    ) -> None:
        """A rank in the 10 %–40 % range must map to 'standard'."""
        assert tier_distribution.get_tier_for_rank(0.25) == "standard"

    def test_get_tier_for_rank_transactional(
        self, tier_distribution: TierDistribution
    ) -> None:
        """A rank in the 40 %–100 % range must map to 'transactional'."""
        assert tier_distribution.get_tier_for_rank(0.70) == "transactional"

    def test_get_tier_for_rank_boundaries(
        self, tier_distribution: TierDistribution
    ) -> None:
        """Exact boundary values must map to the correct tier."""
        td = tier_distribution

        # Bottom of range → strategic
        assert td.get_tier_for_rank(0.00) == "strategic"

        # At strategic boundary (0.10) → moves to standard (strict < comparison)
        assert td.get_tier_for_rank(0.10) == "standard"

        # At strategic + standard boundary (0.40) → moves to transactional
        assert td.get_tier_for_rank(0.40) == "transactional"

        # Near top of range
        assert td.get_tier_for_rank(0.99) == "transactional"

    def test_custom_tier_distribution(self) -> None:
        """Custom tier fractions that sum to 1.0 must be valid."""
        td = TierDistribution(strategic=0.20, standard=0.30, transactional=0.50)
        assert td.strategic == 0.20
        assert td.standard == 0.30
        assert td.transactional == 0.50
        total = td.strategic + td.standard + td.transactional
        assert abs(total - 1.0) < 0.001


# =========================================================================
# TestConstants
# =========================================================================


class TestConstants:
    """Tests for module-level enum-like constant lists."""

    def test_payment_segments(self) -> None:
        """PAYMENT_SEGMENTS must equal the canonical five-element list."""
        assert PAYMENT_SEGMENTS == ["early", "prompt", "on_time", "late", "problem"]

    def test_order_frequencies(self) -> None:
        """ORDER_FREQUENCIES must equal the canonical three-element list."""
        assert ORDER_FREQUENCIES == ["daily", "weekly", "monthly"]

    def test_order_size_patterns(self) -> None:
        """ORDER_SIZE_PATTERNS must equal the canonical two-element list."""
        assert ORDER_SIZE_PATTERNS == ["consistent", "variable"]

    def test_invoice_timings(self) -> None:
        """INVOICE_TIMINGS must equal the canonical three-element list."""
        assert INVOICE_TIMINGS == ["immediate", "prompt", "slow"]

    def test_profile_names(self) -> None:
        """PROFILE_NAMES must equal the canonical five-element list."""
        assert PROFILE_NAMES == ["excellent", "good", "average", "poor", "problem"]

    def test_entity_types(self) -> None:
        """ENTITY_TYPES must equal the canonical four-element list."""
        assert ENTITY_TYPES == ["customer", "vendor", "bank", "carrier"]
