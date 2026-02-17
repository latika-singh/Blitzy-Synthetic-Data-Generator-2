"""
Comprehensive unit tests for AmountDistribution class (F-006).

Tests the statistical models for realistic ERP transaction amounts implemented
in app/statistical/amount_distributions.py. Covers log-normal PO sampling,
vendor invoice matching, customer order tier sizing, business-appropriate
rounding, early-payment discount application, statistical distribution
verification with confidence intervals (N>=10,000), batch sampling, and
edge cases.

Per AAP Section 0.5.1 Group 10: "Log-normal sampling with s=1.2/scale=5000,
rounding rules, discount application, min $100/max $500K bounds."

Per AAP Section 0.7.5: "Statistical model tests must verify output
distributions match expected parameters using confidence intervals with
sufficient sample sizes."

References:
    - README.md lines 512-538: Amount Distribution Parameters specification
    - AAP Section 0.7.5: Testing and Quality Standards
"""

import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from scipy import stats

from app.statistical.amount_distributions import AmountDistribution, AmountConfig


# ---------------------------------------------------------------------------
# Constants — Specification parameters for assertions
# ---------------------------------------------------------------------------

# Log-normal PO distribution parameters (README.md line 520)
PO_SHAPE = 1.2  # s parameter
PO_SCALE = 5_000.0  # scale parameter

# Amount bounds (README.md lines 516-517)
MIN_AMOUNT = 100.0
MAX_AMOUNT = 500_000.0

# Vendor invoice match rate (README.md line 522)
INVOICE_MATCH_RATE = 0.95
INVOICE_VARIANCE_PCT = 0.05  # ±5%

# Customer tier ranges (README.md lines 526-528)
TIER_RANGES = {
    "small": (500.0, 5_000.0),
    "medium": (2_000.0, 50_000.0),
    "large": (10_000.0, 500_000.0),
}

# Reproducible seed value used throughout tests
SEED = 42

# Large sample size for statistical confidence (AAP Section 0.7.5)
LARGE_N = 10_000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def amount_dist() -> AmountDistribution:
    """Provide a seeded AmountDistribution for reproducible tests.

    Uses seed=42 as specified by AAP for deterministic numpy.random.RandomState
    seeding, ensuring identical results across test runs.

    Returns:
        An AmountDistribution instance seeded with 42.
    """
    return AmountDistribution(seed=SEED)


@pytest.fixture
def amount_config() -> AmountConfig:
    """Provide a default AmountConfig with specification-compliant values.

    Returns an AmountConfig with all defaults matching the README.md
    specification: min=$100, max=$500K, po_shape=1.2, po_scale=5000,
    invoice_match_rate=0.95, invoice_variance_pct=0.05.

    Returns:
        A default AmountConfig instance.
    """
    return AmountConfig()


# ===========================================================================
# Test Class 1: TestAmountDistributionInit
# ===========================================================================


class TestAmountDistributionInit:
    """Verify AmountDistribution initialization and configuration binding.

    Ensures the constructor properly sets up distribution parameters,
    random state, and configuration defaults matching the specification.
    """

    def test_init_default_config(self) -> None:
        """AmountDistribution initializes with default config when no args given."""
        dist = AmountDistribution()
        assert dist.config is not None
        assert isinstance(dist.config, AmountConfig)
        assert dist.rng is not None

    def test_init_with_seed(self) -> None:
        """AmountDistribution with seed=42 produces reproducible first sample."""
        dist1 = AmountDistribution(seed=SEED)
        dist2 = AmountDistribution(seed=SEED)
        sample1 = dist1.sample_po_amount()
        sample2 = dist2.sample_po_amount()
        assert sample1 == sample2, (
            f"Seeded distributions should produce identical results: {sample1} != {sample2}"
        )

    def test_init_min_max_bounds(self, amount_config: AmountConfig) -> None:
        """Default config has min_amount=$100 and max_amount=$500,000."""
        assert amount_config.min_amount == MIN_AMOUNT
        assert amount_config.max_amount == MAX_AMOUNT

    def test_init_po_distribution_params(self, amount_config: AmountConfig) -> None:
        """Default config has log-normal s=1.2 and scale=5000."""
        assert amount_config.po_shape == PO_SHAPE
        assert amount_config.po_scale == PO_SCALE

    def test_init_invoice_match_rate(self, amount_config: AmountConfig) -> None:
        """Default config has invoice_match_rate=0.95."""
        assert amount_config.invoice_match_rate == INVOICE_MATCH_RATE

    def test_init_invoice_variance_pct(self, amount_config: AmountConfig) -> None:
        """Default config has invoice_variance_pct=0.05."""
        assert amount_config.invoice_variance_pct == INVOICE_VARIANCE_PCT

    def test_init_customer_tiers(self, amount_config: AmountConfig) -> None:
        """Default config has small, medium, and large customer tiers."""
        tier_names = set(amount_config.customer_tiers.keys())
        assert tier_names == {"small", "medium", "large"}

    def test_init_with_custom_config(self) -> None:
        """AmountDistribution accepts and uses a custom AmountConfig."""
        custom_cfg = AmountConfig(
            min_amount=200.0,
            max_amount=100_000.0,
            po_shape=1.0,
            po_scale=3_000.0,
        )
        dist = AmountDistribution(config=custom_cfg, seed=SEED)
        assert dist.config.min_amount == 200.0
        assert dist.config.max_amount == 100_000.0
        assert dist.config.po_shape == 1.0
        assert dist.config.po_scale == 3_000.0


# ===========================================================================
# Test Class 2: TestPurchaseOrderAmountSampling
# ===========================================================================


class TestPurchaseOrderAmountSampling:
    """Verify purchase order amount sampling from the log-normal distribution.

    PO amounts use scipy.stats.lognorm(s=1.2, scale=5000), clipped to
    [$100, $500,000], and rounded to business-appropriate precision.
    """

    def test_sample_po_amount_returns_float(self, amount_dist: AmountDistribution) -> None:
        """Single PO sample (n=1) returns a Python float, not an array."""
        result = amount_dist.sample_po_amount(n=1)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_sample_po_amount_within_bounds(self, amount_dist: AmountDistribution) -> None:
        """All 10,000 PO samples fall within [$100, $500,000]."""
        samples = amount_dist.sample_po_amount(n=LARGE_N)
        assert np.all(samples >= MIN_AMOUNT), (
            f"Found sample below min: {np.min(samples)}"
        )
        assert np.all(samples <= MAX_AMOUNT), (
            f"Found sample above max: {np.max(samples)}"
        )

    def test_sample_po_amount_min_bound(self, amount_dist: AmountDistribution) -> None:
        """No PO sample falls below the configured minimum of $100."""
        samples = amount_dist.sample_po_amount(n=LARGE_N)
        min_val = np.min(samples)
        assert min_val >= MIN_AMOUNT, f"Minimum {min_val} is below {MIN_AMOUNT}"

    def test_sample_po_amount_max_bound(self, amount_dist: AmountDistribution) -> None:
        """No PO sample exceeds the configured maximum of $500,000."""
        samples = amount_dist.sample_po_amount(n=LARGE_N)
        max_val = np.max(samples)
        assert max_val <= MAX_AMOUNT, f"Maximum {max_val} exceeds {MAX_AMOUNT}"

    def test_sample_po_amount_distribution_shape(self) -> None:
        """PO samples exhibit right-skewed log-normal shape (mean > median).

        For lognorm(s=1.2, scale=5000) the theoretical median is the scale
        parameter (5000) and the mean is ~10,272, so mean > median always.
        """
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        sample_mean = float(np.mean(samples))
        sample_median = float(np.median(samples))

        # Right-skewed: mean > median
        assert sample_mean > sample_median, (
            f"Distribution should be right-skewed: mean={sample_mean} <= median={sample_median}"
        )

        # Median should be in a reasonable range for the parameterization
        assert 1_000.0 <= sample_median <= 20_000.0, (
            f"Median {sample_median} outside expected range [1000, 20000]"
        )

    def test_sample_po_amount_batch(self, amount_dist: AmountDistribution) -> None:
        """Batch sampling with n=100 returns exactly 100 values as ndarray."""
        result = amount_dist.sample_po_amount(n=100)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert len(result) == 100, f"Expected 100 samples, got {len(result)}"

    def test_sample_po_amount_reproducible(self) -> None:
        """Two independently seeded instances produce identical first 10 samples."""
        dist1 = AmountDistribution(seed=SEED)
        dist2 = AmountDistribution(seed=SEED)
        samples1 = [dist1.sample_po_amount() for _ in range(10)]
        samples2 = [dist2.sample_po_amount() for _ in range(10)]
        assert samples1 == samples2, "Identically seeded instances should match"

    def test_sample_po_amount_lognormal_params(self) -> None:
        """Log of PO samples approximates Normal(ln(5000), 1.2).

        For a lognorm(s=1.2, scale=5000) distribution, taking the log of
        samples should yield approximately Normal with mean=ln(5000)≈8.52
        and std≈1.2. Clipping and rounding shift the values slightly.
        """
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        log_samples = np.log(samples)
        log_mean = float(np.mean(log_samples))
        log_std = float(np.std(log_samples))
        expected_log_mean = float(np.log(PO_SCALE))  # ln(5000) ≈ 8.52

        # Allow tolerance for clipping and rounding effects
        assert abs(log_mean - expected_log_mean) < 0.3, (
            f"Log-mean {log_mean:.3f} too far from ln({PO_SCALE})={expected_log_mean:.3f}"
        )
        assert abs(log_std - PO_SHAPE) < 0.3, (
            f"Log-std {log_std:.3f} too far from {PO_SHAPE}"
        )

    def test_sample_po_amount_raises_on_invalid_n(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Requesting n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            amount_dist.sample_po_amount(n=0)


# ===========================================================================
# Test Class 3: TestVendorInvoiceAmount
# ===========================================================================


class TestVendorInvoiceAmount:
    """Verify vendor invoice amount generation relative to PO amount.

    Per specification: 95% of invoices match PO exactly; the remaining
    5% have variances within ±5% of PO amount.
    """

    def test_vendor_invoice_match_rate(self) -> None:
        """Approximately 95% of invoices match the PO amount exactly.

        With N=10,000 samples at p=0.95, we expect ~9,500 matches.
        Tolerance is [93%, 97%] to cover sampling variance at this N.
        """
        dist = AmountDistribution(seed=SEED)
        po_amount = 5_000.0
        invoices = [dist.sample_vendor_invoice_amount(po_amount) for _ in range(LARGE_N)]
        match_count = sum(1 for inv in invoices if inv == po_amount)
        match_rate = match_count / LARGE_N

        assert 0.93 <= match_rate <= 0.97, (
            f"Match rate {match_rate:.4f} outside [0.93, 0.97] tolerance "
            f"(expected ~0.95, matches={match_count}/{LARGE_N})"
        )

    def test_vendor_invoice_variance_within_5_pct(self) -> None:
        """Non-matching invoices are all within ±5% of the PO amount.

        For PO=$10,000, all non-matching invoices should be in
        [$9,500, $10,500].
        """
        dist = AmountDistribution(seed=SEED)
        po_amount = 10_000.0
        lower_bound = po_amount * (1.0 - INVOICE_VARIANCE_PCT)
        upper_bound = po_amount * (1.0 + INVOICE_VARIANCE_PCT)

        non_matching = []
        for _ in range(LARGE_N):
            inv = dist.sample_vendor_invoice_amount(po_amount)
            if inv != po_amount:
                non_matching.append(inv)

        # We should have some non-matching invoices (~5% of 10,000 = ~500)
        assert len(non_matching) > 0, "Expected some non-matching invoices"

        for inv_amt in non_matching:
            assert lower_bound <= inv_amt <= upper_bound, (
                f"Non-matching invoice {inv_amt} outside "
                f"[{lower_bound}, {upper_bound}] for PO={po_amount}"
            )

    def test_vendor_invoice_returns_positive(self) -> None:
        """All vendor invoice amounts are strictly positive."""
        dist = AmountDistribution(seed=SEED)
        po_amount = 5_000.0
        for _ in range(1_000):
            inv = dist.sample_vendor_invoice_amount(po_amount)
            assert inv > 0, f"Invoice amount must be positive, got {inv}"

    def test_vendor_invoice_within_bounds(self) -> None:
        """All vendor invoice amounts respect the global minimum of $100."""
        dist = AmountDistribution(seed=SEED)
        po_amount = 5_000.0
        for _ in range(1_000):
            inv = dist.sample_vendor_invoice_amount(po_amount)
            assert inv >= MIN_AMOUNT, (
                f"Invoice {inv} below min_amount {MIN_AMOUNT}"
            )

    def test_vendor_invoice_small_po(self) -> None:
        """Vendor invoice for a small PO ($150) is reasonable and positive."""
        dist = AmountDistribution(seed=SEED)
        po_amount = 150.0
        for _ in range(100):
            inv = dist.sample_vendor_invoice_amount(po_amount)
            assert inv > 0, f"Invoice must be positive, got {inv}"
            assert inv >= MIN_AMOUNT, (
                f"Invoice {inv} below min_amount {MIN_AMOUNT}"
            )

    def test_vendor_invoice_raises_on_negative_po(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Requesting an invoice for a negative PO amount raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            amount_dist.sample_vendor_invoice_amount(-500.0)


# ===========================================================================
# Test Class 4: TestCustomerOrderAmount
# ===========================================================================


class TestCustomerOrderAmount:
    """Verify customer order amount sampling by tier (small/medium/large).

    Each tier has its own log-normal distribution and clip bounds:
    - Small:  $500–$5,000
    - Medium: $2,000–$50,000
    - Large:  $10,000–$500,000
    """

    def test_small_tier_range(self) -> None:
        """All 1,000 small-tier samples fall within [$500, $5,000]."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_customer_order_amount(tier="small", n=1_000)
        tier_min, tier_max = TIER_RANGES["small"]
        assert np.all(samples >= tier_min), (
            f"Small tier minimum violated: {np.min(samples)} < {tier_min}"
        )
        assert np.all(samples <= tier_max), (
            f"Small tier maximum violated: {np.max(samples)} > {tier_max}"
        )

    def test_medium_tier_range(self) -> None:
        """All 1,000 medium-tier samples fall within [$2,000, $50,000]."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_customer_order_amount(tier="medium", n=1_000)
        tier_min, tier_max = TIER_RANGES["medium"]
        assert np.all(samples >= tier_min), (
            f"Medium tier minimum violated: {np.min(samples)} < {tier_min}"
        )
        assert np.all(samples <= tier_max), (
            f"Medium tier maximum violated: {np.max(samples)} > {tier_max}"
        )

    def test_large_tier_range(self) -> None:
        """All 1,000 large-tier samples fall within [$10,000, $500,000]."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_customer_order_amount(tier="large", n=1_000)
        tier_min, tier_max = TIER_RANGES["large"]
        assert np.all(samples >= tier_min), (
            f"Large tier minimum violated: {np.min(samples)} < {tier_min}"
        )
        assert np.all(samples <= tier_max), (
            f"Large tier maximum violated: {np.max(samples)} > {tier_max}"
        )

    def test_unknown_tier_defaults_to_medium(self) -> None:
        """An unrecognised tier falls back to 'medium' range [$2,000, $50,000]."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_customer_order_amount(tier="unknown", n=1_000)
        tier_min, tier_max = TIER_RANGES["medium"]
        assert np.all(samples >= tier_min), (
            f"Unknown-tier fallback minimum violated: {np.min(samples)} < {tier_min}"
        )
        assert np.all(samples <= tier_max), (
            f"Unknown-tier fallback maximum violated: {np.max(samples)} > {tier_max}"
        )

    def test_tier_means_are_ordered(self) -> None:
        """Mean of small-tier < medium-tier < large-tier samples.

        With 5,000 samples per tier, the law of large numbers ensures the
        sample means reflect the underlying distribution ordering.
        """
        dist = AmountDistribution(seed=SEED)
        mean_small = float(np.mean(dist.sample_customer_order_amount(tier="small", n=5_000)))
        mean_medium = float(np.mean(dist.sample_customer_order_amount(tier="medium", n=5_000)))
        mean_large = float(np.mean(dist.sample_customer_order_amount(tier="large", n=5_000)))

        assert mean_small < mean_medium < mean_large, (
            f"Tier means not ordered: small={mean_small:.2f}, "
            f"medium={mean_medium:.2f}, large={mean_large:.2f}"
        )

    def test_customer_order_reproducible(self) -> None:
        """Same seed produces identical customer order amounts."""
        dist1 = AmountDistribution(seed=SEED)
        dist2 = AmountDistribution(seed=SEED)
        samples1 = [dist1.sample_customer_order_amount(tier="medium") for _ in range(10)]
        samples2 = [dist2.sample_customer_order_amount(tier="medium") for _ in range(10)]
        assert samples1 == samples2, "Identically seeded instances should match"

    def test_customer_order_single_returns_float(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Single customer order sample (n=1) returns a Python float."""
        result = amount_dist.sample_customer_order_amount(tier="medium", n=1)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_customer_order_raises_on_invalid_n(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Requesting n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            amount_dist.sample_customer_order_amount(tier="small", n=0)


# ===========================================================================
# Test Class 5: TestRoundingBehavior
# ===========================================================================


class TestRoundingBehavior:
    """Verify business-appropriate rounding logic.

    Auto-precision tiers:
    - amount < $1,000:   round to nearest $10
    - amount < $10,000:  round to nearest $100
    - amount < $100,000: round to nearest $1,000
    - amount >= $100,000: round to nearest $5,000
    """

    def test_round_4847_to_4800(self, amount_dist: AmountDistribution) -> None:
        """$4,847.32 rounds to $4,800 (auto-precision $100 for < $10K)."""
        result = amount_dist.round_to_nearest(4847.32)
        assert result == 4800.0, f"Expected 4800.0, got {result}"

    def test_round_124389_to_125000(self, amount_dist: AmountDistribution) -> None:
        """$124,389 rounds to $125,000 (auto-precision $5,000 for >= $100K)."""
        result = amount_dist.round_to_nearest(124389.0)
        assert result == 125000.0, f"Expected 125000.0, got {result}"

    def test_round_with_explicit_precision(self, amount_dist: AmountDistribution) -> None:
        """Explicit precision=100 rounds $4,847.32 to $4,800."""
        result = amount_dist.round_to_nearest(4847.32, precision=100)
        assert result == 4800.0, f"Expected 4800.0, got {result}"

    def test_round_small_amount(self, amount_dist: AmountDistribution) -> None:
        """$247.50 rounds to $250 (auto-precision $10 for < $1K)."""
        result = amount_dist.round_to_nearest(247.50)
        assert result == 250.0, f"Expected 250.0, got {result}"

    def test_round_medium_amount(self, amount_dist: AmountDistribution) -> None:
        """$15,678 rounds to $16,000 with explicit precision=$1,000."""
        result = amount_dist.round_to_nearest(15678.0, precision=1000)
        assert result == 16000.0, f"Expected 16000.0, got {result}"

    def test_round_preserves_minimum(self, amount_dist: AmountDistribution) -> None:
        """Rounded result is guaranteed to be >= min_amount ($100)."""
        result = amount_dist.round_to_nearest(50.0)
        assert result >= MIN_AMOUNT, (
            f"Rounded result {result} below min_amount {MIN_AMOUNT}"
        )

    def test_round_zero_returns_minimum(self, amount_dist: AmountDistribution) -> None:
        """Rounding a very small amount returns at least min_amount ($100)."""
        result = amount_dist.round_to_nearest(0.0)
        assert result >= MIN_AMOUNT, (
            f"Rounding 0 should yield at least {MIN_AMOUNT}, got {result}"
        )

    def test_auto_precision_thresholds(self, amount_dist: AmountDistribution) -> None:
        """Auto-precision selects the correct rounding unit per threshold.

        Verifies that each amount magnitude bracket maps to its expected
        precision: <$1K→$10, <$10K→$100, <$100K→$1,000, >=$100K→$5,000.
        """
        # < $1,000 → precision $10 (505 / 10 = 50.5 → 50 * 10 = 500)
        assert amount_dist.round_to_nearest(505.0) == 500.0

        # $1,000–$10,000 → precision $100 (4847.32 / 100 = 48.4732 → 48 * 100 = 4800)
        assert amount_dist.round_to_nearest(4847.32) == 4800.0

        # $10,000–$100,000 → precision $1,000 (15678 / 1000 = 15.678 → 16 * 1000 = 16000)
        assert amount_dist.round_to_nearest(15678.0) == 16000.0

        # >= $100,000 → precision $5,000 (124389 / 5000 = 24.8778 → 25 * 5000 = 125000)
        assert amount_dist.round_to_nearest(124389.0) == 125000.0

    def test_round_negative_returns_minimum(self, amount_dist: AmountDistribution) -> None:
        """Rounding a negative value returns at least min_amount ($100)."""
        result = amount_dist.round_to_nearest(-10.0)
        assert result >= MIN_AMOUNT, (
            f"Rounding negative should yield at least {MIN_AMOUNT}, got {result}"
        )


# ===========================================================================
# Test Class 6: TestDiscountApplication
# ===========================================================================


class TestDiscountApplication:
    """Verify early-payment discount calculations.

    Standard terms: 2/10 Net 30 — 2% discount if paid within 10 days.
    apply_discount(amount, discount_rate, days_to_pay, discount_days=10).
    """

    def test_2_10_net_30_within_discount(self, amount_dist: AmountDistribution) -> None:
        """Payment on day 8 receives 2% discount: $5,000 → $4,900."""
        result = amount_dist.apply_discount(5000.0, 0.02, days_to_pay=8, discount_days=10)
        assert result == 4900.0, f"Expected 4900.0, got {result}"

    def test_2_10_net_30_outside_discount(self, amount_dist: AmountDistribution) -> None:
        """Payment on day 15 receives no discount: $5,000 → $5,000."""
        result = amount_dist.apply_discount(5000.0, 0.02, days_to_pay=15, discount_days=10)
        assert result == 5000.0, f"Expected 5000.0, got {result}"

    def test_discount_on_exact_day(self, amount_dist: AmountDistribution) -> None:
        """Payment exactly on day 10 still qualifies for discount: $5,000 → $4,900."""
        result = amount_dist.apply_discount(5000.0, 0.02, days_to_pay=10, discount_days=10)
        assert result == 4900.0, f"Expected 4900.0, got {result}"

    def test_discount_zero_rate(self, amount_dist: AmountDistribution) -> None:
        """Zero discount rate returns full amount: $5,000 → $5,000."""
        result = amount_dist.apply_discount(5000.0, 0.0, days_to_pay=5, discount_days=10)
        assert result == 5000.0, f"Expected 5000.0, got {result}"

    def test_discount_large_amount(self, amount_dist: AmountDistribution) -> None:
        """2% discount on $100,000: $100,000 * 0.98 = $98,000."""
        result = amount_dist.apply_discount(100_000.0, 0.02, days_to_pay=5)
        assert result == 98000.0, f"Expected 98000.0, got {result}"

    def test_discount_result_rounded(self, amount_dist: AmountDistribution) -> None:
        """Discount results undergo business-appropriate rounding.

        For $4,567 at 2% discount: raw = 4475.66, auto-precision $100
        → round(4475.66 / 100) * 100 = round(44.7566) * 100 = 45 * 100 = 4500.
        """
        result = amount_dist.apply_discount(4567.0, 0.02, days_to_pay=5, discount_days=10)
        # 4567 * 0.98 = 4475.66, rounded with precision 100 => 4500
        assert result == 4500.0, f"Expected 4500.0, got {result}"

    def test_discount_day_zero_qualifies(self, amount_dist: AmountDistribution) -> None:
        """Payment on day 0 (immediate) qualifies for discount."""
        result = amount_dist.apply_discount(5000.0, 0.02, days_to_pay=0, discount_days=10)
        assert result == 4900.0, f"Expected 4900.0, got {result}"

    def test_discount_raises_on_negative_amount(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Negative amount raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            amount_dist.apply_discount(-500.0, 0.02, days_to_pay=5)

    def test_discount_raises_on_invalid_rate(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Discount rate outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            amount_dist.apply_discount(5000.0, 1.5, days_to_pay=5)

    def test_discount_raises_on_negative_days(
        self, amount_dist: AmountDistribution
    ) -> None:
        """Negative days_to_pay raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            amount_dist.apply_discount(5000.0, 0.02, days_to_pay=-1)


# ===========================================================================
# Test Class 7: TestStatisticalDistributionVerification
# ===========================================================================


class TestStatisticalDistributionVerification:
    """Critical statistical tests verifying output distributions match spec.

    Per AAP Section 0.7.5: "Statistical model tests must verify output
    distributions match expected parameters using confidence intervals
    with sufficient sample sizes (N>=10,000)."
    """

    def test_po_amount_lognormal_verification(self) -> None:
        """Sample mean approximates theoretical lognorm mean within CI.

        For lognorm(s=1.2, scale=5000):
            theoretical_mean = scale * exp(s²/2)
                             = 5000 * exp(0.72)
                             ≈ 10,272

        With N=10,000, clipping, and rounding, the sample mean should
        be within ±30% of the theoretical value.
        """
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        sample_mean = float(np.mean(samples))

        theoretical_mean = PO_SCALE * np.exp(PO_SHAPE ** 2 / 2.0)
        # ~10,272

        # Allow generous tolerance for clipping and rounding effects
        lower = theoretical_mean * 0.70
        upper = theoretical_mean * 1.30
        assert lower <= sample_mean <= upper, (
            f"Sample mean {sample_mean:.2f} outside [{lower:.2f}, {upper:.2f}] "
            f"(theoretical mean={theoretical_mean:.2f})"
        )

    def test_po_amount_skewness(self) -> None:
        """PO amount distribution exhibits positive skewness (right-skewed).

        Log-normal distributions are always right-skewed: mean > median.
        Additionally, more than 50% of samples should fall below the mean.
        """
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        sample_mean = float(np.mean(samples))
        sample_median = float(np.median(samples))

        # Characteristic 1: mean > median (right-skewed)
        assert sample_mean > sample_median, (
            f"Expected right-skew (mean > median): mean={sample_mean:.2f}, "
            f"median={sample_median:.2f}"
        )

        # Characteristic 2: majority of values below the mean
        fraction_below_mean = float(np.mean(samples < sample_mean))
        assert fraction_below_mean > 0.5, (
            f"Expected >50% of values below mean, got {fraction_below_mean:.4f}"
        )

    def test_invoice_variance_distribution(self) -> None:
        """Non-matching invoice variances are distributed within [-5%, +5%].

        For 10,000 vendor invoices against PO=$10,000, the non-matching
        invoices (~500) should have variances spanning the full [-5%, +5%]
        range with both positive and negative deviations present.
        """
        dist = AmountDistribution(seed=SEED)
        po_amount = 10_000.0

        non_matching = []
        for _ in range(LARGE_N):
            inv = dist.sample_vendor_invoice_amount(po_amount)
            if inv != po_amount:
                non_matching.append(inv)

        assert len(non_matching) > 100, (
            f"Expected significant non-matching set, got {len(non_matching)}"
        )

        # Compute relative variances
        variances = [(inv - po_amount) / po_amount for inv in non_matching]
        variances_arr = np.array(variances)

        # All variances within [-5%, +5%]
        assert np.all(variances_arr >= -INVOICE_VARIANCE_PCT - 1e-9), (
            f"Variance below -5%: {np.min(variances_arr):.6f}"
        )
        assert np.all(variances_arr <= INVOICE_VARIANCE_PCT + 1e-9), (
            f"Variance above +5%: {np.max(variances_arr):.6f}"
        )

        # Both positive and negative variances should be present
        has_positive = bool(np.any(variances_arr > 0))
        has_negative = bool(np.any(variances_arr < 0))
        assert has_positive, "Expected some positive variances in non-matching invoices"
        assert has_negative, "Expected some negative variances in non-matching invoices"

    def test_po_amount_theoretical_std(self) -> None:
        """Sample std approximates the theoretical lognorm std.

        For lognorm(s=1.2, scale=5000):
            theoretical_std = scale * sqrt(exp(s²) - 1) * exp(s²/2)
                            ≈ 18,435

        Clipping and rounding will compress the spread somewhat.
        """
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        sample_std = float(np.std(samples))
        theoretical_std = float(stats.lognorm.std(s=PO_SHAPE, scale=PO_SCALE))

        # Allow wide tolerance for clipping/rounding distortion
        assert sample_std > theoretical_std * 0.50, (
            f"Sample std {sample_std:.2f} far below theoretical {theoretical_std:.2f}"
        )
        assert sample_std < theoretical_std * 1.50, (
            f"Sample std {sample_std:.2f} far above theoretical {theoretical_std:.2f}"
        )

    def test_po_scipy_theoretical_mean_value(self) -> None:
        """Validate that scipy computes the expected theoretical mean.

        Theoretical mean of lognorm(s=1.2, scale=5000) should be
        approximately 10,272 (= 5000 * exp(0.72)).
        """
        theoretical_mean = float(stats.lognorm.mean(s=PO_SHAPE, scale=PO_SCALE))
        expected = PO_SCALE * np.exp(PO_SHAPE ** 2 / 2.0)
        assert theoretical_mean == pytest.approx(expected, rel=1e-6), (
            f"scipy theoretical mean {theoretical_mean} != formula {expected}"
        )


# ===========================================================================
# Test Class 8: TestBatchSampling
# ===========================================================================


class TestBatchSampling:
    """Verify the unified batch sampling interface.

    sample_batch(distribution_type, n, **kwargs) always returns np.ndarray.
    """

    def test_batch_po_returns_array(self, amount_dist: AmountDistribution) -> None:
        """sample_batch('purchase_order', 100) returns ndarray of length 100."""
        result = amount_dist.sample_batch("purchase_order", 100)
        assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
        assert len(result) == 100, f"Expected 100 elements, got {len(result)}"

    def test_batch_purchase_order_type(self, amount_dist: AmountDistribution) -> None:
        """Purchase order batch samples are all within [min, max] bounds."""
        batch = amount_dist.sample_batch("purchase_order", 500)
        assert np.all(batch >= MIN_AMOUNT)
        assert np.all(batch <= MAX_AMOUNT)

    def test_batch_vendor_invoice_type(self, amount_dist: AmountDistribution) -> None:
        """Vendor invoice batch with po_amount returns correct length."""
        batch = amount_dist.sample_batch(
            "vendor_invoice", 100, po_amount=5_000.0
        )
        assert isinstance(batch, np.ndarray)
        assert len(batch) == 100
        assert np.all(batch > 0)

    def test_batch_customer_order_type(self, amount_dist: AmountDistribution) -> None:
        """Customer order batch for small tier returns correct length."""
        batch = amount_dist.sample_batch("customer_order", 100, tier="small")
        assert isinstance(batch, np.ndarray)
        assert len(batch) == 100
        tier_min, tier_max = TIER_RANGES["small"]
        assert np.all(batch >= tier_min)
        assert np.all(batch <= tier_max)

    def test_batch_all_types(self, amount_dist: AmountDistribution) -> None:
        """All three distribution types produce valid ndarray batches."""
        for dtype in ("purchase_order", "vendor_invoice", "customer_order"):
            kwargs = {}
            if dtype == "vendor_invoice":
                kwargs["po_amount"] = 5_000.0
            elif dtype == "customer_order":
                kwargs["tier"] = "medium"

            batch = amount_dist.sample_batch(dtype, 50, **kwargs)
            assert isinstance(batch, np.ndarray), f"{dtype}: expected ndarray"
            assert len(batch) == 50, f"{dtype}: expected 50 elements, got {len(batch)}"
            assert np.all(batch > 0), f"{dtype}: all values should be positive"

    def test_batch_performance(self) -> None:
        """Generating 10,000 PO samples completes in under 1 second.

        Performance target: >=10,000 samples/second per AAP specification.
        """
        dist = AmountDistribution(seed=SEED)
        start = time.time()
        batch = dist.sample_batch("purchase_order", LARGE_N)
        elapsed = time.time() - start

        assert len(batch) == LARGE_N
        assert elapsed < 1.0, (
            f"Batch sampling took {elapsed:.3f}s, expected <1.0s for {LARGE_N} samples"
        )

    def test_batch_unknown_type_raises(self, amount_dist: AmountDistribution) -> None:
        """Unknown distribution_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution_type"):
            amount_dist.sample_batch("invalid_type", 10)

    def test_batch_invalid_n_raises(self, amount_dist: AmountDistribution) -> None:
        """Requesting n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 1"):
            amount_dist.sample_batch("purchase_order", 0)


# ===========================================================================
# Test Class 9: TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Verify edge-case behaviour and robustness guarantees.

    Covers single-sample semantics, large batches, and invariants like
    no negative values and no NaN values in any output path.
    """

    def test_single_sample_returns_scalar(self, amount_dist: AmountDistribution) -> None:
        """sample_po_amount(n=1) returns a float scalar, not an array."""
        result = amount_dist.sample_po_amount(n=1)
        assert isinstance(result, float), f"Expected float, got {type(result)}"
        assert not isinstance(result, np.ndarray)

    def test_large_batch(self) -> None:
        """100,000 samples complete without error and return correct length."""
        dist = AmountDistribution(seed=SEED)
        batch = dist.sample_po_amount(n=100_000)
        assert isinstance(batch, np.ndarray)
        assert len(batch) == 100_000

    def test_negative_amount_never_returned(self) -> None:
        """No PO sample is ever negative across 10,000 draws."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        assert np.all(samples >= 0), f"Found negative: {np.min(samples)}"

    def test_nan_never_returned(self) -> None:
        """No PO sample is ever NaN across 10,000 draws."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        assert not np.any(np.isnan(samples)), "Found NaN in PO samples"

    def test_customer_order_nan_never_returned(self) -> None:
        """No customer-order sample is ever NaN for any tier."""
        dist = AmountDistribution(seed=SEED)
        for tier in ("small", "medium", "large"):
            samples = dist.sample_customer_order_amount(tier=tier, n=1_000)
            assert not np.any(np.isnan(np.array([samples]) if isinstance(samples, float) else samples)), (
                f"Found NaN in {tier} tier samples"
            )

    def test_vendor_invoice_nan_never_returned(self) -> None:
        """No vendor invoice amount is ever NaN."""
        dist = AmountDistribution(seed=SEED)
        invoices = np.array([dist.sample_vendor_invoice_amount(5_000.0) for _ in range(1_000)])
        assert not np.any(np.isnan(invoices)), "Found NaN in vendor invoices"

    def test_all_po_amounts_are_finite(self) -> None:
        """All PO amounts are finite (no inf)."""
        dist = AmountDistribution(seed=SEED)
        samples = dist.sample_po_amount(n=LARGE_N)
        assert np.all(np.isfinite(samples)), "Found infinite value in PO samples"

    def test_batch_single_element(self, amount_dist: AmountDistribution) -> None:
        """sample_batch with n=1 returns a single-element ndarray."""
        result = amount_dist.sample_batch("purchase_order", 1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert result[0] >= MIN_AMOUNT


# ===========================================================================
# Test Class 10: TestAmountConfigValidation
# ===========================================================================


class TestAmountConfigValidation:
    """Verify Pydantic V2 validation on AmountConfig fields.

    AmountConfig enforces constraints: min_amount >= 0, max_amount > 0,
    po_shape > 0, po_scale > 0, invoice_match_rate in [0, 1],
    invoice_variance_pct in [0, 1].
    """

    def test_default_config_is_valid(self) -> None:
        """Default AmountConfig passes all Pydantic validation."""
        config = AmountConfig()
        assert config.min_amount == MIN_AMOUNT
        assert config.max_amount == MAX_AMOUNT

    def test_config_custom_values(self) -> None:
        """Custom AmountConfig values are accepted when within constraints."""
        config = AmountConfig(
            min_amount=50.0,
            max_amount=200_000.0,
            po_shape=0.8,
            po_scale=3_000.0,
            invoice_match_rate=0.90,
            invoice_variance_pct=0.10,
        )
        assert config.min_amount == 50.0
        assert config.max_amount == 200_000.0
        assert config.po_shape == 0.8
        assert config.po_scale == 3_000.0
        assert config.invoice_match_rate == 0.90
        assert config.invoice_variance_pct == 0.10

    def test_config_serialization_round_trip(self) -> None:
        """AmountConfig round-trips through model_dump/model_validate."""
        original = AmountConfig()
        data = original.model_dump()
        restored = AmountConfig.model_validate(data)
        assert restored.min_amount == original.min_amount
        assert restored.max_amount == original.max_amount
        assert restored.po_shape == original.po_shape
        assert restored.po_scale == original.po_scale

    def test_config_customer_tiers_have_required_keys(self) -> None:
        """Each customer tier has min, max, shape, and scale keys."""
        config = AmountConfig()
        for tier_name, tier_params in config.customer_tiers.items():
            assert "min" in tier_params, f"Tier '{tier_name}' missing 'min'"
            assert "max" in tier_params, f"Tier '{tier_name}' missing 'max'"
            assert "shape" in tier_params, f"Tier '{tier_name}' missing 'shape'"
            assert "scale" in tier_params, f"Tier '{tier_name}' missing 'scale'"


# ===========================================================================
# Test Class 11: TestIntegrationScenarios
# ===========================================================================


class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple AmountDistribution methods.

    Simulates realistic usage patterns where PO amounts feed into invoice
    generation and discounts are applied in sequence.
    """

    def test_po_to_invoice_pipeline(self) -> None:
        """Generate PO, then invoice, then apply discount — end-to-end."""
        dist = AmountDistribution(seed=SEED)

        po_amount = dist.sample_po_amount()
        assert isinstance(po_amount, float)
        assert MIN_AMOUNT <= po_amount <= MAX_AMOUNT

        invoice_amount = dist.sample_vendor_invoice_amount(po_amount)
        assert isinstance(invoice_amount, float)
        assert invoice_amount > 0

        discounted = dist.apply_discount(invoice_amount, 0.02, days_to_pay=7)
        assert isinstance(discounted, float)
        assert discounted <= invoice_amount  # Discount should not increase amount

    def test_multi_tier_order_generation(self) -> None:
        """Generate orders across all three tiers in one workflow."""
        dist = AmountDistribution(seed=SEED)

        for tier_name, (expected_min, expected_max) in TIER_RANGES.items():
            amount = dist.sample_customer_order_amount(tier=tier_name)
            assert isinstance(amount, float)
            assert expected_min <= amount <= expected_max, (
                f"Tier '{tier_name}': {amount} not in [{expected_min}, {expected_max}]"
            )

    def test_batch_invoice_from_po_batch(self) -> None:
        """Generate PO batch, then create matching invoice batch."""
        dist = AmountDistribution(seed=SEED)

        po_batch = dist.sample_batch("purchase_order", 50)
        assert len(po_batch) == 50

        invoice_batch = dist.sample_batch(
            "vendor_invoice", 50, po_amounts=po_batch
        )
        assert len(invoice_batch) == 50
        assert np.all(invoice_batch > 0)

    def test_rounding_applied_to_po_samples(self) -> None:
        """PO amounts produced by sampling are already rounded.

        Verify that round_to_nearest applied to a PO sample yields the
        same value (idempotent), confirming that sample_po_amount already
        rounds internally.
        """
        dist = AmountDistribution(seed=SEED)
        for _ in range(100):
            po = dist.sample_po_amount()
            re_rounded = dist.round_to_nearest(po)
            assert po == re_rounded, (
                f"PO amount {po} changed after re-rounding to {re_rounded}"
            )

    def test_full_simulation_day_amounts(self) -> None:
        """Simulate a full day's worth of transactions (mixed types).

        Generates 100 POs, 100 invoices, and 100 customer orders to verify
        no errors, NaN, or out-of-bounds values in an aggregate workload.
        """
        dist = AmountDistribution(seed=SEED)

        po_amounts = dist.sample_batch("purchase_order", 100)
        assert len(po_amounts) == 100
        assert np.all(po_amounts >= MIN_AMOUNT)
        assert np.all(po_amounts <= MAX_AMOUNT)
        assert not np.any(np.isnan(po_amounts))

        invoice_amounts = dist.sample_batch(
            "vendor_invoice", 100, po_amounts=po_amounts
        )
        assert len(invoice_amounts) == 100
        assert np.all(invoice_amounts > 0)
        assert not np.any(np.isnan(invoice_amounts))

        customer_amounts = dist.sample_batch("customer_order", 100, tier="medium")
        assert len(customer_amounts) == 100
        tier_min, tier_max = TIER_RANGES["medium"]
        assert np.all(customer_amounts >= tier_min)
        assert np.all(customer_amounts <= tier_max)
        assert not np.any(np.isnan(customer_amounts))
