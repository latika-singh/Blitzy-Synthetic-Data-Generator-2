"""
Comprehensive unit tests for the PaymentTimingModel (F-006).

Tests the 5-segment mixture model for realistic customer payment timing,
covering:
    - Model initialisation and configuration validation
    - Segment distribution parameters (Normal and LogNormal)
    - Due-date calculation from payment terms
    - Profile-weighted segment selection (5 profiles)
    - Weekend adjustment (Saturday/Sunday → Monday)
    - End-to-end ``get_payment_date()`` integration
    - Statistical distribution verification (N ≥ 10,000 with confidence
      intervals, per AAP § 0.7.5)
    - Batch sampling
    - Edge cases

Testing conventions:
    - ``numpy`` seed = 42 for reproducible results
    - ``pytest.approx()`` for floating-point tolerance
    - No external API calls, no Redis — pure statistical module
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import stats

from app.statistical.payment_timing_model import (
    PaymentSegment,
    PaymentTimingConfig,
    PaymentTimingModel,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def payment_timing_model() -> PaymentTimingModel:
    """Return a deterministic PaymentTimingModel seeded at 42."""
    return PaymentTimingModel(seed=42)


@pytest.fixture
def sample_invoice_date() -> date:
    """Return 2025-01-15 (a Wednesday) for predictable date arithmetic."""
    return date(2025, 1, 15)


@pytest.fixture
def saturday_invoice_date() -> date:
    """Return a date whose Net 30 due date falls on a Saturday.

    date(2025, 1, 4) + 30 days = date(2025, 2, 3) — but we need a
    Saturday result.  date(2025, 1, 3) + 30 = 2025-02-02 (Sunday).
    date(2025, 1, 2) + 30 = 2025-02-01 (Saturday). ✓
    """
    return date(2025, 1, 2)


# =========================================================================
# TestPaymentTimingModelInit
# =========================================================================


class TestPaymentTimingModelInit:
    """Verify PaymentTimingModel construction and default configuration."""

    def test_init_default_config(self) -> None:
        """Model initialises successfully with no arguments."""
        model = PaymentTimingModel()
        assert model is not None
        assert model.config is not None
        assert isinstance(model.config, PaymentTimingConfig)

    def test_init_with_seed(self) -> None:
        """Two instances with the same seed produce identical first results."""
        model_a = PaymentTimingModel(seed=42)
        model_b = PaymentTimingModel(seed=42)
        inv = date(2025, 3, 1)
        result_a = model_a.get_payment_date(inv, "Net 30", "average")
        result_b = model_b.get_payment_date(inv, "Net 30", "average")
        assert result_a == result_b

    def test_init_segments_defined(self) -> None:
        """Model has exactly 5 segments with the canonical names."""
        model = PaymentTimingModel()
        segment_names = [s.name for s in model.config.segments]
        assert segment_names == [
            "early_discount",
            "prompt",
            "on_time",
            "late",
            "problem",
        ]

    def test_segment_weights_sum_to_one(self) -> None:
        """Segment prior weights sum to 1.0 (within floating-point tolerance)."""
        model = PaymentTimingModel()
        total = sum(s.weight for s in model.config.segments)
        assert total == pytest.approx(1.0, abs=1e-10)


# =========================================================================
# TestSegmentDistributionParameters
# =========================================================================


class TestSegmentDistributionParameters:
    """Verify each segment's distribution parameters match the specification."""

    @staticmethod
    def _segment_by_name(
        model: PaymentTimingModel, name: str
    ) -> PaymentSegment:
        """Helper: retrieve a PaymentSegment by name."""
        for seg in model.config.segments:
            if seg.name == name:
                return seg
        raise ValueError(f"Segment '{name}' not found")

    def test_early_discount_segment(
        self, payment_timing_model: PaymentTimingModel
    ) -> None:
        """early_discount: Normal(loc=-8, scale=2), weight=0.20."""
        seg = self._segment_by_name(payment_timing_model, "early_discount")
        assert seg.weight == pytest.approx(0.20)
        assert seg.distribution_type == "normal"
        assert seg.loc == pytest.approx(-8.0)
        assert seg.scale == pytest.approx(2.0)

    def test_prompt_segment(
        self, payment_timing_model: PaymentTimingModel
    ) -> None:
        """prompt: Normal(loc=-2, scale=3), weight=0.30."""
        seg = self._segment_by_name(payment_timing_model, "prompt")
        assert seg.weight == pytest.approx(0.30)
        assert seg.distribution_type == "normal"
        assert seg.loc == pytest.approx(-2.0)
        assert seg.scale == pytest.approx(3.0)

    def test_on_time_segment(
        self, payment_timing_model: PaymentTimingModel
    ) -> None:
        """on_time: Normal(loc=2, scale=4), weight=0.25."""
        seg = self._segment_by_name(payment_timing_model, "on_time")
        assert seg.weight == pytest.approx(0.25)
        assert seg.distribution_type == "normal"
        assert seg.loc == pytest.approx(2.0)
        assert seg.scale == pytest.approx(4.0)

    def test_late_segment(
        self, payment_timing_model: PaymentTimingModel
    ) -> None:
        """late: LogNormal(s=1.0, loc=10, scale=5), weight=0.15."""
        seg = self._segment_by_name(payment_timing_model, "late")
        assert seg.weight == pytest.approx(0.15)
        assert seg.distribution_type == "lognormal"
        assert seg.loc == pytest.approx(10.0)
        assert seg.scale == pytest.approx(5.0)
        assert seg.s == pytest.approx(1.0)

    def test_problem_segment(
        self, payment_timing_model: PaymentTimingModel
    ) -> None:
        """problem: LogNormal(s=1.2, loc=30, scale=15), weight=0.10."""
        seg = self._segment_by_name(payment_timing_model, "problem")
        assert seg.weight == pytest.approx(0.10)
        assert seg.distribution_type == "lognormal"
        assert seg.loc == pytest.approx(30.0)
        assert seg.scale == pytest.approx(15.0)
        assert seg.s == pytest.approx(1.2)


# =========================================================================
# TestDueDateCalculation
# =========================================================================


class TestDueDateCalculation:
    """Verify due-date computation from various payment-terms strings."""

    def test_net_30_terms(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """'Net 30' → invoice_date + 30 days."""
        due = payment_timing_model._calculate_due_date(
            sample_invoice_date, "Net 30"
        )
        assert due == date(2025, 2, 14)

    def test_net_60_terms(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """'Net 60' → invoice_date + 60 days."""
        due = payment_timing_model._calculate_due_date(
            sample_invoice_date, "Net 60"
        )
        assert due == date(2025, 3, 16)

    def test_due_on_receipt(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """'Due on Receipt' → same as invoice_date."""
        due = payment_timing_model._calculate_due_date(
            sample_invoice_date, "Due on Receipt"
        )
        assert due == date(2025, 1, 15)

    def test_2_10_net_30(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """'2/10 Net 30' → due in 30 days (discount period handled elsewhere)."""
        due = payment_timing_model._calculate_due_date(
            sample_invoice_date, "2/10 Net 30"
        )
        assert due == date(2025, 2, 14)

    def test_unknown_terms_default(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """Unrecognised terms fall back to 30 days."""
        due = payment_timing_model._calculate_due_date(
            sample_invoice_date, "Some Unknown Terms"
        )
        assert due == sample_invoice_date + timedelta(days=30)


# =========================================================================
# TestProfileWeightedSelection
# =========================================================================


class TestProfileWeightedSelection:
    """Verify that profile weights control segment selection probabilities."""

    @staticmethod
    def _sample_segments(
        model: PaymentTimingModel,
        profile: str,
        n: int = 10_000,
    ) -> Counter:
        """Sample *n* segments for *profile* and return a Counter."""
        counts: Counter = Counter()
        for _ in range(n):
            seg = model._select_segment(profile)
            counts[seg] += 1
        return counts

    def test_excellent_profile_favors_early(self) -> None:
        """>50 % of excellent-profile selections are 'early_discount' (spec 60 %)."""
        model = PaymentTimingModel(seed=42)
        counts = self._sample_segments(model, "excellent", 10_000)
        early_pct = counts["early_discount"] / 10_000
        assert early_pct > 0.50

    def test_problem_profile_favors_late(self) -> None:
        """>70 % of problem-profile selections are 'late' or 'problem'."""
        model = PaymentTimingModel(seed=42)
        counts = self._sample_segments(model, "problem", 10_000)
        late_pct = (counts["late"] + counts["problem"]) / 10_000
        assert late_pct > 0.70

    def test_average_profile_distribution(self) -> None:
        """Average profile roughly matches [10 %, 30 %, 35 %, 20 %, 5 %]."""
        model = PaymentTimingModel(seed=42)
        counts = self._sample_segments(model, "average", 10_000)
        n = 10_000
        # Allow ±5 percentage-point tolerance for statistical noise
        assert counts["early_discount"] / n == pytest.approx(0.10, abs=0.05)
        assert counts["prompt"] / n == pytest.approx(0.30, abs=0.05)
        assert counts["on_time"] / n == pytest.approx(0.35, abs=0.05)
        assert counts["late"] / n == pytest.approx(0.20, abs=0.05)
        assert counts["problem"] / n == pytest.approx(0.05, abs=0.05)

    def test_unknown_profile_defaults_to_average(self) -> None:
        """An unrecognised profile falls back to 'average' distribution."""
        model_a = PaymentTimingModel(seed=42)
        model_b = PaymentTimingModel(seed=42)
        # Both should produce identical results because unknown → average
        results_unknown = [
            model_a._select_segment("nonexistent_profile") for _ in range(100)
        ]
        results_average = [
            model_b._select_segment("average") for _ in range(100)
        ]
        assert results_unknown == results_average

    def test_all_profiles_weights_sum_to_one(self) -> None:
        """Every profile's weight vector sums to 1.0."""
        model = PaymentTimingModel()
        for profile, weights in model.config.profile_weights.items():
            assert sum(weights) == pytest.approx(1.0, abs=1e-10), (
                f"Profile '{profile}' weights do not sum to 1.0"
            )


# =========================================================================
# TestWeekendAdjustment
# =========================================================================


class TestWeekendAdjustment:
    """Verify Saturday/Sunday → next-Monday adjustment logic."""

    def test_saturday_adjusted_to_monday(self) -> None:
        """Saturday 2025-02-01 (weekday 5) → Monday 2025-02-03."""
        saturday = date(2025, 2, 1)
        assert saturday.weekday() == 5  # sanity
        result = PaymentTimingModel._adjust_for_weekend(saturday)
        assert result == date(2025, 2, 3)
        assert result.weekday() == 0  # Monday

    def test_sunday_adjusted_to_monday(self) -> None:
        """Sunday 2025-02-02 (weekday 6) → Monday 2025-02-03."""
        sunday = date(2025, 2, 2)
        assert sunday.weekday() == 6
        result = PaymentTimingModel._adjust_for_weekend(sunday)
        assert result == date(2025, 2, 3)
        assert result.weekday() == 0

    def test_weekday_unchanged(self) -> None:
        """Wednesday 2025-01-15 remains unchanged."""
        wednesday = date(2025, 1, 15)
        assert wednesday.weekday() == 2
        result = PaymentTimingModel._adjust_for_weekend(wednesday)
        assert result == wednesday

    def test_friday_unchanged(self) -> None:
        """Friday 2025-01-17 is NOT moved forward."""
        friday = date(2025, 1, 17)
        assert friday.weekday() == 4
        result = PaymentTimingModel._adjust_for_weekend(friday)
        assert result == friday

    def test_payment_dates_never_on_weekend(self) -> None:
        """1,000 payment dates from get_payment_date never land on a weekend."""
        model = PaymentTimingModel(seed=42)
        inv = date(2025, 1, 15)
        for _ in range(1_000):
            pd = model.get_payment_date(inv, "Net 30", "average")
            assert pd.weekday() < 5, f"Weekend date detected: {pd}"


# =========================================================================
# TestGetPaymentDate — end-to-end integration
# =========================================================================


class TestGetPaymentDate:
    """Integration tests combining due-date, segment, offset, and weekend."""

    def test_get_payment_date_returns_date(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """Return type is ``datetime.date``."""
        result = payment_timing_model.get_payment_date(
            sample_invoice_date, "Net 30", "average"
        )
        assert isinstance(result, date)

    def test_get_payment_date_net_30_excellent(self) -> None:
        """Excellent customers tend to pay before the due date (mean offset < 0)."""
        model = PaymentTimingModel(seed=42)
        inv = date(2025, 1, 15)
        due = inv + timedelta(days=30)

        offsets = []
        for _ in range(1_000):
            pd = model.get_payment_date(inv, "Net 30", "excellent")
            offsets.append((pd - due).days)
        mean_offset = np.mean(offsets)
        # Excellent profile heavily favours early_discount (60 %) and prompt (30 %)
        assert mean_offset < 0, (
            f"Expected negative mean offset for excellent, got {mean_offset}"
        )

    def test_get_payment_date_net_30_problem(self) -> None:
        """Problem customers tend to pay well after the due date (mean offset > 0)."""
        model = PaymentTimingModel(seed=42)
        inv = date(2025, 1, 15)
        due = inv + timedelta(days=30)

        offsets = []
        for _ in range(1_000):
            pd = model.get_payment_date(inv, "Net 30", "problem")
            offsets.append((pd - due).days)
        mean_offset = np.mean(offsets)
        assert mean_offset > 0, (
            f"Expected positive mean offset for problem, got {mean_offset}"
        )

    def test_get_payment_date_reproducible(self) -> None:
        """Two models with seed=42 and identical inputs → identical output."""
        inv = date(2025, 6, 1)
        terms = "Net 30"
        profile = "good"

        model_a = PaymentTimingModel(seed=42)
        model_b = PaymentTimingModel(seed=42)
        assert model_a.get_payment_date(inv, terms, profile) == \
               model_b.get_payment_date(inv, terms, profile)

    def test_get_payment_date_different_seeds_different_results(self) -> None:
        """Different seeds produce different payment dates (with high probability)."""
        inv = date(2025, 6, 1)
        terms = "Net 30"
        profile = "average"

        model_a = PaymentTimingModel(seed=1)
        model_b = PaymentTimingModel(seed=99)

        # Sample several to virtually guarantee at least one differs
        results_a = [
            model_a.get_payment_date(inv, terms, profile) for _ in range(20)
        ]
        results_b = [
            model_b.get_payment_date(inv, terms, profile) for _ in range(20)
        ]
        assert results_a != results_b

    def test_payment_date_reasonable_range(
        self,
        payment_timing_model: PaymentTimingModel,
        sample_invoice_date: date,
    ) -> None:
        """Payment date falls within -30 … +180 days from the due date."""
        due = sample_invoice_date + timedelta(days=30)
        for _ in range(500):
            pd = payment_timing_model.get_payment_date(
                sample_invoice_date, "Net 30", "average"
            )
            offset_days = (pd - due).days
            assert -30 <= offset_days <= 180, (
                f"Out-of-range offset {offset_days} days (payment={pd}, due={due})"
            )


# =========================================================================
# TestStatisticalDistributionVerification (AAP § 0.7.5)
# =========================================================================


class TestStatisticalDistributionVerification:
    """Verify output distributions match specification parameters.

    Uses N ≥ 10,000 samples with confidence-interval assertions rather
    than exact equality, as required by AAP Section 0.7.5.
    """

    @staticmethod
    def _sample_offsets_for_segment(
        segment_name: str, n: int = 10_000, seed: int = 42
    ) -> np.ndarray:
        """Sample *n* raw offsets from a specific segment's distribution.

        Creates a dedicated model and calls ``_sample_offset`` repeatedly
        to collect distribution samples *without* segment selection
        (directly forces the target segment).
        """
        model = PaymentTimingModel(seed=seed)
        offsets = np.array(
            [model._sample_offset(segment_name) for _ in range(n)]
        )
        return offsets

    # -- Normal segments ---------------------------------------------------

    def test_early_discount_distribution_stats(self) -> None:
        """early_discount ≈ Normal(loc=-8, scale=2): mean ≈ -8, std ≈ 2."""
        offsets = self._sample_offsets_for_segment("early_discount", 10_000)
        assert np.mean(offsets) == pytest.approx(-8.0, abs=0.5)
        assert np.std(offsets) == pytest.approx(2.0, abs=0.5)

    def test_prompt_distribution_stats(self) -> None:
        """prompt ≈ Normal(loc=-2, scale=3): mean ≈ -2, std ≈ 3."""
        offsets = self._sample_offsets_for_segment("prompt", 10_000)
        assert np.mean(offsets) == pytest.approx(-2.0, abs=0.5)
        assert np.std(offsets) == pytest.approx(3.0, abs=0.5)

    def test_on_time_distribution_stats(self) -> None:
        """on_time ≈ Normal(loc=2, scale=4): mean ≈ 2, std ≈ 4."""
        offsets = self._sample_offsets_for_segment("on_time", 10_000)
        assert np.mean(offsets) == pytest.approx(2.0, abs=0.5)
        assert np.std(offsets) == pytest.approx(4.0, abs=0.5)

    # -- Mixture-model integration -----------------------------------------

    def test_mixture_model_overall_payment_behavior(self) -> None:
        """10,000 payment dates (average / Net 30) show multi-modal character.

        * Both early (< due) and late (> due) payments exist.
        * Roughly ≥ 30 % of payments fall within ±10 days of the due date.
        """
        model = PaymentTimingModel(seed=42)
        inv = date(2025, 1, 15)
        due = inv + timedelta(days=30)

        offsets = []
        for _ in range(10_000):
            pd = model.get_payment_date(inv, "Net 30", "average")
            offsets.append((pd - due).days)

        offsets_arr = np.array(offsets)
        # Both early and late payments should exist
        assert np.any(offsets_arr < 0), "No early payments found"
        assert np.any(offsets_arr > 0), "No late payments found"

        # A significant portion within ±10 days of due date
        within_10 = np.sum(np.abs(offsets_arr) <= 10) / len(offsets_arr)
        assert within_10 > 0.30, (
            f"Only {within_10:.1%} within ±10 days — expected ≥ 30 %"
        )

    def test_profile_affects_overall_distribution(self) -> None:
        """Excellent-profile mean offset is significantly lower than problem."""
        inv = date(2025, 1, 15)
        due = inv + timedelta(days=30)

        # Excellent profile
        model_exc = PaymentTimingModel(seed=42)
        offsets_exc = []
        for _ in range(5_000):
            pd = model_exc.get_payment_date(inv, "Net 30", "excellent")
            offsets_exc.append((pd - due).days)

        # Problem profile
        model_prob = PaymentTimingModel(seed=42)
        offsets_prob = []
        for _ in range(5_000):
            pd = model_prob.get_payment_date(inv, "Net 30", "problem")
            offsets_prob.append((pd - due).days)

        mean_exc = np.mean(offsets_exc)
        mean_prob = np.mean(offsets_prob)
        assert mean_exc < mean_prob, (
            f"Excellent mean ({mean_exc:.1f}) should be < problem mean "
            f"({mean_prob:.1f})"
        )


# =========================================================================
# TestBatchSampling
# =========================================================================


class TestBatchSampling:
    """Verify the batch-sampling API ``sample_payment_dates_batch``."""

    def test_batch_sampling_returns_correct_count(self) -> None:
        """Requesting 100 payment dates returns exactly 100."""
        model = PaymentTimingModel(seed=42)
        n = 100
        inv_dates = [date(2025, 1, 15)] * n
        terms = ["Net 30"] * n
        profiles = ["average"] * n

        results = model.sample_payment_dates_batch(inv_dates, terms, profiles)
        assert len(results) == n
        assert all(isinstance(d, date) for d in results)

    def test_batch_sampling_reproducible(self) -> None:
        """Same seed → identical batch output."""
        n = 50
        inv_dates = [date(2025, 3, 10)] * n
        terms = ["Net 30"] * n
        profiles = ["good"] * n

        model_a = PaymentTimingModel(seed=42)
        model_b = PaymentTimingModel(seed=42)

        batch_a = model_a.sample_payment_dates_batch(inv_dates, terms, profiles)
        batch_b = model_b.sample_payment_dates_batch(inv_dates, terms, profiles)
        assert batch_a == batch_b


# =========================================================================
# TestEdgeCases
# =========================================================================


class TestEdgeCases:
    """Boundary conditions and unusual inputs."""

    def test_empty_terms_string(self) -> None:
        """Empty payment terms default to 30 days."""
        model = PaymentTimingModel(seed=42)
        inv = date(2025, 1, 15)
        due = model._calculate_due_date(inv, "")
        assert due == inv + timedelta(days=30)

    def test_very_late_payment_still_valid(self) -> None:
        """Even with extreme segment samples the return value is a valid date."""
        model = PaymentTimingModel(seed=42)
        inv = date(2025, 1, 15)
        # Problem profile can produce very large offsets
        for _ in range(200):
            pd = model.get_payment_date(inv, "Net 30", "problem")
            assert isinstance(pd, date)
            # Must be after invoice date minus a reasonable window
            assert pd >= inv - timedelta(days=60)

    def test_leap_year_handling(self) -> None:
        """Invoice near Feb 29 on a leap year produces a valid date."""
        model = PaymentTimingModel(seed=42)
        # 2024 is a leap year
        inv = date(2024, 2, 29)
        pd = model.get_payment_date(inv, "Net 30", "average")
        assert isinstance(pd, date)
        # Due date would be 2024-03-30
        due = inv + timedelta(days=30)
        assert due == date(2024, 3, 30)
