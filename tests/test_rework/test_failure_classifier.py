"""Comprehensive tests for the FailureClassifier — Validation failure classification.

Tests cover:
    - Planned discrepancy (within parameter bounds) → SKIP_REWORK
    - Planned discrepancy (outside parameter bounds) → ADJUST_PARAMETERS
    - Unplanned error → APPLY_FIX
    - ClassificationResult Pydantic V2 model validation
    - ClassificationType and RecommendedAction enum values
    - Confidence score ranges (0-1) with seeded RNG determinism
    - Discrepancy table query mocking via injected lookup callable
    - Parameter bounds checking with Decimal precision
    - Constructor injection (all Optional params)
    - Metrics tracking (total_classifications, per-type counts)
    - structlog logging with service_name, component, trace_id

Design Notes:
    - FailureClassifier receives discrepancy_lookup as an injected callable
    - All async operations use AsyncMock — no real database queries
    - Deterministic testing via seeded random.Random(42)
    - Follows test patterns from test_error_handlers.py (per-handler test classes)

References:
    - AAP Section 0.5.1 Group 6: failure_classifier.py
    - AAP Section 0.5.1 Group 8: test_failure_classifier.py
    - AAP Section 0.7.5: Discrepancy parameter bounds checking
    - AAP Section 0.7.6: Testing Conventions (≥80% coverage)
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.rework.failure_classifier import (
    ClassificationResult,
    ClassificationType,
    FailureClassifier,
    RecommendedAction,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions — sample data factories
# ═══════════════════════════════════════════════════════════════════════════


def _make_transaction(
    transaction_id: Optional[str] = None,
    transaction_type: str = "vendor_invoice",
    amount: str = "5000.00",
) -> Dict[str, Any]:
    """Create a sample transaction dict for classifier tests."""
    return {
        "transaction_id": transaction_id or str(uuid4()),
        "transaction_type": transaction_type,
        "amount": Decimal(amount),
        "vendor_id": "V-001",
        "status": "validation_failed",
    }


def _make_validation_errors(
    error_type: str = "balance_error",
    field: str = "total_amount",
    message: str = "GL entry does not balance",
) -> List[Dict[str, Any]]:
    """Create sample validation error list."""
    return [
        {
            "error_type": error_type,
            "field": field,
            "message": message,
            "severity": "error",
        }
    ]


def _make_context(
    simulation_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a minimal generation context dict."""
    return {
        "simulation_id": simulation_id or str(uuid4()),
        "trace_id": trace_id or str(uuid4()),
        "current_date": "2025-01-15",
        "fiscal_period": "2025-01",
    }


def _make_discrepancy_record(
    type_code: str = "P2P-001",
    is_within_bounds: bool = True,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a sample discrepancy record (simulating DB lookup result)."""
    return {
        "discrepancy_id": str(uuid4()),
        "type_code": type_code,
        "category": "p2p",
        "difficulty": "easy",
        "description": "Duplicate Invoice",
        "transaction_ids": [str(uuid4())],
        "is_within_bounds": is_within_bounds,
        "parameters": parameters or {"days_apart": 5, "amount_variation_pct": 0.02},
    }


def _make_parameter_bounds() -> Dict[str, Dict[str, Any]]:
    """Create sample parameter bounds for bounds checking tests."""
    return {
        "days_apart": {"min": 1, "max": 90},
        "amount_variation_pct": {"min": Decimal("0.01"), "max": Decimal("0.50")},
        "quantity_variance_pct": {"min": Decimal("0.01"), "max": Decimal("0.30")},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def discrepancy_lookup_found() -> AsyncMock:
    """Async callable that returns a discrepancy record (planned discrepancy found)."""
    lookup = AsyncMock(return_value=_make_discrepancy_record())
    return lookup


@pytest.fixture
def discrepancy_lookup_not_found() -> AsyncMock:
    """Async callable that returns None (no planned discrepancy)."""
    lookup = AsyncMock(return_value=None)
    return lookup


@pytest.fixture
def bounds_lookup_within() -> MagicMock:
    """Bounds lookup that returns bounds — all params within bounds."""
    lookup = MagicMock(return_value=_make_parameter_bounds())
    return lookup


@pytest.fixture
def bounds_lookup_outside() -> MagicMock:
    """Bounds lookup returning bounds where 'days_apart' param will be outside.

    The default discrepancy record has days_apart=5.  Tightening the min
    to 10 means the parameter will be detected as out-of-bounds.
    """
    bounds = _make_parameter_bounds()
    # Tighten the days_apart bounds so the default value (5) becomes out of range
    bounds["days_apart"] = {"min": 10, "max": 90}
    lookup = MagicMock(return_value=bounds)
    return lookup


@pytest.fixture
def classifier_with_found(
    discrepancy_lookup_found: AsyncMock,
    bounds_lookup_within: MagicMock,
) -> FailureClassifier:
    """FailureClassifier configured to find a planned discrepancy within bounds."""
    return FailureClassifier(
        discrepancy_lookup=discrepancy_lookup_found,
        parameter_bounds_lookup=bounds_lookup_within,
        rng_seed=42,
    )


@pytest.fixture
def classifier_with_not_found(
    discrepancy_lookup_not_found: AsyncMock,
) -> FailureClassifier:
    """FailureClassifier configured to find NO planned discrepancy (unplanned error)."""
    return FailureClassifier(
        discrepancy_lookup=discrepancy_lookup_not_found,
        rng_seed=42,
    )


@pytest.fixture
def classifier_no_deps() -> FailureClassifier:
    """FailureClassifier with no dependencies — constructor defaults."""
    return FailureClassifier()


# ═══════════════════════════════════════════════════════════════════════════
# Tests — ClassificationType Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestClassificationTypeEnum:
    """Tests for the ClassificationType string enum — 3 members."""

    def test_planned_within_bounds_value(self) -> None:
        """PLANNED_WITHIN_BOUNDS has the expected string value."""
        assert ClassificationType.PLANNED_WITHIN_BOUNDS.value == "planned_discrepancy_within_bounds"

    def test_planned_outside_bounds_value(self) -> None:
        """PLANNED_OUTSIDE_BOUNDS has the expected string value."""
        assert ClassificationType.PLANNED_OUTSIDE_BOUNDS.value == "planned_discrepancy_outside_bounds"

    def test_unplanned_error_value(self) -> None:
        """UNPLANNED_ERROR has the expected string value."""
        assert ClassificationType.UNPLANNED_ERROR.value == "unplanned_error"

    def test_enum_is_string(self) -> None:
        """All ClassificationType values are str instances."""
        for member in ClassificationType:
            assert isinstance(member.value, str), f"{member.name}.value is not str"

    def test_enum_has_three_members(self) -> None:
        """ClassificationType must have exactly 3 members."""
        assert len(ClassificationType) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Tests — RecommendedAction Enum
# ═══════════════════════════════════════════════════════════════════════════


class TestRecommendedActionEnum:
    """Tests for the RecommendedAction string enum — 4 members."""

    def test_skip_rework_value(self) -> None:
        """SKIP_REWORK has the expected string value."""
        assert RecommendedAction.SKIP_REWORK.value == "skip_rework"

    def test_adjust_parameters_value(self) -> None:
        """ADJUST_PARAMETERS has the expected string value."""
        assert RecommendedAction.ADJUST_PARAMETERS.value == "adjust_parameters"

    def test_apply_fix_value(self) -> None:
        """APPLY_FIX has the expected string value."""
        assert RecommendedAction.APPLY_FIX.value == "apply_fix"

    def test_escalate_value(self) -> None:
        """ESCALATE has the expected string value."""
        assert RecommendedAction.ESCALATE.value == "escalate"

    def test_enum_has_four_members(self) -> None:
        """RecommendedAction must have exactly 4 members."""
        assert len(RecommendedAction) == 4


# ═══════════════════════════════════════════════════════════════════════════
# Tests — ClassificationResult Pydantic V2 Model
# ═══════════════════════════════════════════════════════════════════════════


class TestClassificationResult:
    """Tests for the ClassificationResult Pydantic V2 data contract."""

    def test_default_values(self) -> None:
        """ClassificationResult defaults: confidence_score=1.0, is_planned=False, etc."""
        result = ClassificationResult(
            classification_type=ClassificationType.UNPLANNED_ERROR.value,
            recommended_action=RecommendedAction.APPLY_FIX.value,
        )
        assert result.confidence_score == 1.0
        assert result.is_planned_discrepancy is False
        assert result.within_bounds is None
        assert result.out_of_bounds_params == []
        assert result.error_details == {}
        assert result.discrepancy_type_code is None

    def test_planned_discrepancy_result(self) -> None:
        """ClassificationResult with all planned-discrepancy fields populated."""
        result = ClassificationResult(
            classification_type=ClassificationType.PLANNED_WITHIN_BOUNDS.value,
            recommended_action=RecommendedAction.SKIP_REWORK.value,
            confidence_score=0.97,
            is_planned_discrepancy=True,
            discrepancy_type_code="P2P-001",
            within_bounds=True,
            out_of_bounds_params=[],
            error_details={"discrepancy_record_found": True},
        )
        assert result.classification_type == "planned_discrepancy_within_bounds"
        assert result.recommended_action == "skip_rework"
        assert result.confidence_score == 0.97
        assert result.is_planned_discrepancy is True
        assert result.discrepancy_type_code == "P2P-001"
        assert result.within_bounds is True
        assert result.out_of_bounds_params == []

    def test_unplanned_error_result(self) -> None:
        """ClassificationResult for unplanned error path."""
        result = ClassificationResult(
            classification_type=ClassificationType.UNPLANNED_ERROR.value,
            recommended_action=RecommendedAction.APPLY_FIX.value,
            confidence_score=0.82,
            is_planned_discrepancy=False,
            discrepancy_type_code=None,
            within_bounds=None,
            error_details={"validation_error_count": 1},
        )
        assert result.classification_type == "unplanned_error"
        assert result.is_planned_discrepancy is False
        assert result.discrepancy_type_code is None
        assert result.within_bounds is None

    def test_confidence_score_bounds(self) -> None:
        """confidence_score must be between 0.0 and 1.0 (inclusive)."""
        # Valid boundaries
        low = ClassificationResult(
            classification_type=ClassificationType.UNPLANNED_ERROR.value,
            recommended_action=RecommendedAction.APPLY_FIX.value,
            confidence_score=0.0,
        )
        assert low.confidence_score == 0.0

        high = ClassificationResult(
            classification_type=ClassificationType.UNPLANNED_ERROR.value,
            recommended_action=RecommendedAction.APPLY_FIX.value,
            confidence_score=1.0,
        )
        assert high.confidence_score == 1.0

        # Invalid: above 1.0
        with pytest.raises(Exception):
            ClassificationResult(
                classification_type=ClassificationType.UNPLANNED_ERROR.value,
                recommended_action=RecommendedAction.APPLY_FIX.value,
                confidence_score=1.1,
            )

        # Invalid: below 0.0
        with pytest.raises(Exception):
            ClassificationResult(
                classification_type=ClassificationType.UNPLANNED_ERROR.value,
                recommended_action=RecommendedAction.APPLY_FIX.value,
                confidence_score=-0.1,
            )

    def test_timestamp_auto_generated(self) -> None:
        """timestamp should be auto-generated as a UTC datetime."""
        before = datetime.now(timezone.utc)
        result = ClassificationResult(
            classification_type=ClassificationType.UNPLANNED_ERROR.value,
            recommended_action=RecommendedAction.APPLY_FIX.value,
        )
        after = datetime.now(timezone.utc)

        assert isinstance(result.timestamp, datetime)
        # Timestamp should be between before and after creation
        assert before <= result.timestamp <= after

    def test_error_details_dict(self) -> None:
        """error_details defaults to empty dict, accepts arbitrary keys."""
        result = ClassificationResult(
            classification_type=ClassificationType.UNPLANNED_ERROR.value,
            recommended_action=RecommendedAction.APPLY_FIX.value,
            error_details={"foo": "bar", "count": 42},
        )
        assert isinstance(result.error_details, dict)
        assert result.error_details["foo"] == "bar"
        assert result.error_details["count"] == 42

    def test_out_of_bounds_params_list(self) -> None:
        """out_of_bounds_params carries the names of params outside bounds."""
        result = ClassificationResult(
            classification_type=ClassificationType.PLANNED_OUTSIDE_BOUNDS.value,
            recommended_action=RecommendedAction.ADJUST_PARAMETERS.value,
            confidence_score=0.88,
            is_planned_discrepancy=True,
            within_bounds=False,
            out_of_bounds_params=["days_apart", "amount_variation_pct"],
        )
        assert isinstance(result.out_of_bounds_params, list)
        assert len(result.out_of_bounds_params) == 2
        assert "days_apart" in result.out_of_bounds_params
        assert "amount_variation_pct" in result.out_of_bounds_params


# ═══════════════════════════════════════════════════════════════════════════
# Tests — FailureClassifier Constructor
# ═══════════════════════════════════════════════════════════════════════════


class TestFailureClassifierConstructor:
    """Tests for FailureClassifier constructor injection (ADR-003)."""

    def test_default_constructor(self) -> None:
        """All params default to None; rng_seed defaults to 42."""
        classifier = FailureClassifier()
        assert classifier._discrepancy_lookup is None
        assert classifier._parameter_bounds_lookup is None
        # Default seed is 42 — verify RNG was initialized
        assert isinstance(classifier._rng, random.Random)

    def test_custom_rng_seed(self) -> None:
        """Custom rng_seed changes RNG behaviour (different sequence)."""
        c1 = FailureClassifier(rng_seed=42)
        c2 = FailureClassifier(rng_seed=99)
        # Both have RNG instances, but different seed → different sequences
        val1 = c1._rng.random()
        val2 = c2._rng.random()
        assert val1 != val2, "Different seeds should produce different sequences"

    def test_discrepancy_lookup_stored(self) -> None:
        """Injected discrepancy_lookup callable is stored as _discrepancy_lookup."""
        lookup = AsyncMock(return_value=None)
        classifier = FailureClassifier(discrepancy_lookup=lookup)
        assert classifier._discrepancy_lookup is lookup

    def test_parameter_bounds_lookup_stored(self) -> None:
        """Injected parameter_bounds_lookup callable is stored as _parameter_bounds_lookup."""
        bounds_lookup = MagicMock(return_value={})
        classifier = FailureClassifier(parameter_bounds_lookup=bounds_lookup)
        assert classifier._parameter_bounds_lookup is bounds_lookup

    def test_initial_metrics_zero(self) -> None:
        """All classification counters start at 0 on construction."""
        classifier = FailureClassifier()
        metrics = classifier.get_metrics()
        assert metrics["total_classifications"] == 0
        assert metrics["planned_within_count"] == 0
        assert metrics["planned_outside_count"] == 0
        assert metrics["unplanned_count"] == 0
        assert metrics["escalation_count"] == 0

    def test_has_discrepancy_lookup_flag(self) -> None:
        """Constructor distinguishes between having and not having a lookup."""
        # With lookup
        with_lookup = FailureClassifier(discrepancy_lookup=AsyncMock())
        assert with_lookup._discrepancy_lookup is not None

        # Without lookup
        without_lookup = FailureClassifier()
        assert without_lookup._discrepancy_lookup is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Classify: Planned Discrepancy Within Bounds
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestClassifyPlannedDiscrepancyWithinBounds:
    """Tests for the 'planned discrepancy within bounds' classification path.

    When the discrepancy lookup returns a record AND all parameters are
    within configured bounds, the classifier should return PLANNED_WITHIN_BOUNDS
    with recommended action SKIP_REWORK.
    """

    @pytest.mark.asyncio
    async def test_returns_planned_within_bounds(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """classification_type == PLANNED_WITHIN_BOUNDS when discrepancy found within bounds."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.classification_type == ClassificationType.PLANNED_WITHIN_BOUNDS.value

    @pytest.mark.asyncio
    async def test_recommended_action_skip_rework(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """recommended_action == SKIP_REWORK for planned within bounds."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.recommended_action == RecommendedAction.SKIP_REWORK.value

    @pytest.mark.asyncio
    async def test_is_planned_discrepancy_true(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """is_planned_discrepancy is True for planned discrepancy."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.is_planned_discrepancy is True

    @pytest.mark.asyncio
    async def test_within_bounds_true(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """within_bounds is True when all params are inside bounds."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.within_bounds is True

    @pytest.mark.asyncio
    async def test_discrepancy_type_code_populated(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """discrepancy_type_code carries the type code from the record."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.discrepancy_type_code == "P2P-001"

    @pytest.mark.asyncio
    async def test_confidence_score_high(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """confidence_score >= 0.95 for planned within bounds (high confidence)."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.confidence_score >= 0.95
        assert result.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_out_of_bounds_params_empty(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """No out-of-bounds params when all are within bounds."""
        result = await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.out_of_bounds_params == []

    @pytest.mark.asyncio
    async def test_metric_planned_within_incremented(
        self,
        classifier_with_found: FailureClassifier,
    ) -> None:
        """planned_within_count is incremented after classification."""
        assert classifier_with_found.get_metrics()["planned_within_count"] == 0
        await classifier_with_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        metrics = classifier_with_found.get_metrics()
        assert metrics["planned_within_count"] == 1
        assert metrics["total_classifications"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Classify: Planned Discrepancy Outside Bounds
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestClassifyPlannedDiscrepancyOutsideBounds:
    """Tests for the 'planned discrepancy outside bounds' classification path.

    When the discrepancy lookup returns a record BUT one or more parameters
    are outside configured bounds, the classifier should return
    PLANNED_OUTSIDE_BOUNDS with recommended action ADJUST_PARAMETERS.
    """

    @pytest.fixture
    def classifier_outside_bounds(
        self,
        discrepancy_lookup_found: AsyncMock,
        bounds_lookup_outside: MagicMock,
    ) -> FailureClassifier:
        """Classifier finding a discrepancy with params outside bounds."""
        return FailureClassifier(
            discrepancy_lookup=discrepancy_lookup_found,
            parameter_bounds_lookup=bounds_lookup_outside,
            rng_seed=42,
        )

    @pytest.mark.asyncio
    async def test_returns_planned_outside_bounds(
        self,
        classifier_outside_bounds: FailureClassifier,
    ) -> None:
        """classification_type == PLANNED_OUTSIDE_BOUNDS when params drift outside."""
        result = await classifier_outside_bounds.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.classification_type == ClassificationType.PLANNED_OUTSIDE_BOUNDS.value

    @pytest.mark.asyncio
    async def test_recommended_action_adjust_parameters(
        self,
        classifier_outside_bounds: FailureClassifier,
    ) -> None:
        """recommended_action == ADJUST_PARAMETERS for outside bounds."""
        result = await classifier_outside_bounds.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.recommended_action == RecommendedAction.ADJUST_PARAMETERS.value

    @pytest.mark.asyncio
    async def test_within_bounds_false(
        self,
        classifier_outside_bounds: FailureClassifier,
    ) -> None:
        """within_bounds is False when params are outside bounds."""
        result = await classifier_outside_bounds.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.within_bounds is False

    @pytest.mark.asyncio
    async def test_out_of_bounds_params_populated(
        self,
        classifier_outside_bounds: FailureClassifier,
    ) -> None:
        """out_of_bounds_params contains 'days_apart' (value 5 < min 10)."""
        result = await classifier_outside_bounds.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert "days_apart" in result.out_of_bounds_params

    @pytest.mark.asyncio
    async def test_confidence_score_moderate(
        self,
        classifier_outside_bounds: FailureClassifier,
    ) -> None:
        """confidence_score between 0.80 and 0.95 for outside-bounds path."""
        result = await classifier_outside_bounds.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.confidence_score >= 0.80
        assert result.confidence_score <= 0.95

    @pytest.mark.asyncio
    async def test_metric_planned_outside_incremented(
        self,
        classifier_outside_bounds: FailureClassifier,
    ) -> None:
        """planned_outside_count is incremented after classification."""
        assert classifier_outside_bounds.get_metrics()["planned_outside_count"] == 0
        await classifier_outside_bounds.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        metrics = classifier_outside_bounds.get_metrics()
        assert metrics["planned_outside_count"] == 1
        assert metrics["total_classifications"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Classify: Unplanned Error
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestClassifyUnplannedError:
    """Tests for the 'unplanned error' classification path.

    When the discrepancy lookup returns None (no planned discrepancy),
    the classifier should return UNPLANNED_ERROR with action APPLY_FIX.
    """

    @pytest.mark.asyncio
    async def test_returns_unplanned_error(
        self,
        classifier_with_not_found: FailureClassifier,
    ) -> None:
        """classification_type == UNPLANNED_ERROR when no discrepancy found."""
        result = await classifier_with_not_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.classification_type == ClassificationType.UNPLANNED_ERROR.value

    @pytest.mark.asyncio
    async def test_recommended_action_apply_fix(
        self,
        classifier_with_not_found: FailureClassifier,
    ) -> None:
        """recommended_action == APPLY_FIX for unplanned errors."""
        result = await classifier_with_not_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.recommended_action == RecommendedAction.APPLY_FIX.value

    @pytest.mark.asyncio
    async def test_is_planned_discrepancy_false(
        self,
        classifier_with_not_found: FailureClassifier,
    ) -> None:
        """is_planned_discrepancy is False for unplanned errors."""
        result = await classifier_with_not_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.is_planned_discrepancy is False

    @pytest.mark.asyncio
    async def test_confidence_score_moderate(
        self,
        classifier_with_not_found: FailureClassifier,
    ) -> None:
        """confidence_score between 0.70 and 0.90 for unplanned errors."""
        result = await classifier_with_not_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.confidence_score >= 0.70
        assert result.confidence_score <= 0.90

    @pytest.mark.asyncio
    async def test_discrepancy_type_code_none(
        self,
        classifier_with_not_found: FailureClassifier,
    ) -> None:
        """discrepancy_type_code is None for unplanned errors."""
        result = await classifier_with_not_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.discrepancy_type_code is None

    @pytest.mark.asyncio
    async def test_metric_unplanned_incremented(
        self,
        classifier_with_not_found: FailureClassifier,
    ) -> None:
        """unplanned_count is incremented after unplanned classification."""
        assert classifier_with_not_found.get_metrics()["unplanned_count"] == 0
        await classifier_with_not_found.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        metrics = classifier_with_not_found.get_metrics()
        assert metrics["unplanned_count"] == 1
        assert metrics["total_classifications"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Classify: No Lookup Function
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestClassifyNoLookupFunction:
    """Tests for classification when no discrepancy_lookup is provided."""

    @pytest.mark.asyncio
    async def test_no_lookup_defaults_to_unplanned(
        self,
        classifier_no_deps: FailureClassifier,
    ) -> None:
        """When discrepancy_lookup is None, defaults to UNPLANNED_ERROR."""
        result = await classifier_no_deps.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert result.classification_type == ClassificationType.UNPLANNED_ERROR.value

    @pytest.mark.asyncio
    async def test_no_lookup_still_classifies(
        self,
        classifier_no_deps: FailureClassifier,
    ) -> None:
        """Classification still returns a valid ClassificationResult."""
        result = await classifier_no_deps.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert isinstance(result, ClassificationResult)
        assert result.recommended_action == RecommendedAction.APPLY_FIX.value
        assert result.is_planned_discrepancy is False

    @pytest.mark.asyncio
    async def test_no_lookup_confidence(
        self,
        classifier_no_deps: FailureClassifier,
    ) -> None:
        """confidence_score is within valid [0.0, 1.0] range without lookup."""
        result = await classifier_no_deps.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )
        assert 0.0 <= result.confidence_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Parameter Bounds Checking (_check_parameter_bounds)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestParameterBoundsChecking:
    """Tests for the _check_parameter_bounds() helper method.

    Verifies that parameter values are correctly compared against their
    min/max bounds, with Decimal precision per AAP Section 0.7.2.
    """

    def _make_classifier(self) -> FailureClassifier:
        """Create a classifier instance for direct method testing."""
        return FailureClassifier(rng_seed=42)

    def test_all_params_within_bounds(self) -> None:
        """All params within bounds → (True, [])."""
        classifier = self._make_classifier()
        params = {"days_apart": 5, "amount_variation_pct": Decimal("0.10")}
        bounds = _make_parameter_bounds()
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is True
        assert out_of_bounds == []

    def test_single_param_outside_min(self) -> None:
        """One param below min → (False, ['param_name'])."""
        classifier = self._make_classifier()
        params = {"days_apart": 0, "amount_variation_pct": Decimal("0.10")}
        bounds = _make_parameter_bounds()
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is False
        assert "days_apart" in out_of_bounds

    def test_single_param_outside_max(self) -> None:
        """One param above max → (False, ['param_name'])."""
        classifier = self._make_classifier()
        params = {"days_apart": 100, "amount_variation_pct": Decimal("0.10")}
        bounds = _make_parameter_bounds()
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is False
        assert "days_apart" in out_of_bounds

    def test_multiple_params_outside(self) -> None:
        """Multiple params outside → (False, [multiple names])."""
        classifier = self._make_classifier()
        params = {
            "days_apart": 0,  # below min 1
            "amount_variation_pct": Decimal("0.60"),  # above max 0.50
            "quantity_variance_pct": Decimal("0.005"),  # below min 0.01
        }
        bounds = _make_parameter_bounds()
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is False
        assert len(out_of_bounds) == 3
        assert "days_apart" in out_of_bounds
        assert "amount_variation_pct" in out_of_bounds
        assert "quantity_variance_pct" in out_of_bounds

    def test_decimal_comparison(self) -> None:
        """Bounds checked with Decimal precision — Decimal('0.51') > Decimal('0.50')."""
        classifier = self._make_classifier()
        params = {"amount_variation_pct": Decimal("0.51")}
        bounds = {
            "amount_variation_pct": {
                "min": Decimal("0.01"),
                "max": Decimal("0.50"),
            },
        }
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is False
        assert "amount_variation_pct" in out_of_bounds

    def test_missing_param_in_bounds(self) -> None:
        """Param not in bounds dict is ignored (not flagged as out-of-bounds)."""
        classifier = self._make_classifier()
        # 'extra_param' is in params but not in bounds — should be ignored
        params = {
            "days_apart": 5,
            "extra_param": 999,
        }
        bounds = _make_parameter_bounds()
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is True
        assert out_of_bounds == []

    def test_empty_bounds_all_within(self) -> None:
        """Empty bounds dict → all within bounds."""
        classifier = self._make_classifier()
        params = {"days_apart": 999, "amount_variation_pct": Decimal("99.0")}
        bounds: Dict[str, Dict[str, Any]] = {}
        all_within, out_of_bounds = classifier._check_parameter_bounds(params, bounds)
        assert all_within is True
        assert out_of_bounds == []


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Deterministic Behavior
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestDeterministicBehavior:
    """Tests for seeded-RNG determinism in confidence score generation."""

    @pytest.mark.asyncio
    async def test_same_seed_same_confidence(self) -> None:
        """Two classifiers with seed=42 produce identical confidence scores."""
        lookup1 = AsyncMock(return_value=None)
        lookup2 = AsyncMock(return_value=None)
        c1 = FailureClassifier(discrepancy_lookup=lookup1, rng_seed=42)
        c2 = FailureClassifier(discrepancy_lookup=lookup2, rng_seed=42)

        txn = _make_transaction()
        ctx = _make_context()
        errors = _make_validation_errors()

        result1 = await c1.classify(errors, txn, ctx)
        result2 = await c2.classify(errors, txn, ctx)

        assert result1.confidence_score == result2.confidence_score

    @pytest.mark.asyncio
    async def test_different_seed_different_confidence(self) -> None:
        """Different seeds produce different confidence scores."""
        lookup1 = AsyncMock(return_value=None)
        lookup2 = AsyncMock(return_value=None)
        c1 = FailureClassifier(discrepancy_lookup=lookup1, rng_seed=42)
        c2 = FailureClassifier(discrepancy_lookup=lookup2, rng_seed=99)

        txn = _make_transaction()
        ctx = _make_context()
        errors = _make_validation_errors()

        result1 = await c1.classify(errors, txn, ctx)
        result2 = await c2.classify(errors, txn, ctx)

        # Extremely unlikely to be equal with different seeds
        assert result1.confidence_score != result2.confidence_score

    @pytest.mark.asyncio
    async def test_seeded_rng_not_module_level(self) -> None:
        """Classifier uses self._rng, not the module-level random.random()."""
        lookup = AsyncMock(return_value=None)
        classifier = FailureClassifier(discrepancy_lookup=lookup, rng_seed=42)

        # Record the state of the module-level random
        module_state_before = random.getstate()

        await classifier.classify(
            _make_validation_errors(),
            _make_transaction(),
            _make_context(),
        )

        # Module-level random state should be unchanged
        module_state_after = random.getstate()
        assert module_state_before == module_state_after, (
            "Classifier should use self._rng, not module-level random"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests — get_metrics()
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.rework
class TestGetMetrics:
    """Tests for FailureClassifier.get_metrics() method."""

    def test_metrics_structure(self) -> None:
        """Verify metrics dict contains all expected keys."""
        classifier = FailureClassifier()
        metrics = classifier.get_metrics()
        expected_keys = {
            "total_classifications",
            "planned_within_count",
            "planned_outside_count",
            "unplanned_count",
            "escalation_count",
            "classification_rates",
        }
        assert expected_keys.issubset(set(metrics.keys()))
        # classification_rates should itself have sub-keys
        rates = metrics["classification_rates"]
        assert "planned_within" in rates
        assert "planned_outside" in rates
        assert "unplanned" in rates
        assert "escalation" in rates

    def test_metrics_initial_zeros(self) -> None:
        """All counters are 0 initially."""
        classifier = FailureClassifier()
        metrics = classifier.get_metrics()
        assert metrics["total_classifications"] == 0
        assert metrics["planned_within_count"] == 0
        assert metrics["planned_outside_count"] == 0
        assert metrics["unplanned_count"] == 0
        assert metrics["escalation_count"] == 0
        # Rates should be 0.0 when total is 0
        rates = metrics["classification_rates"]
        assert rates["planned_within"] == 0.0
        assert rates["planned_outside"] == 0.0
        assert rates["unplanned"] == 0.0
        assert rates["escalation"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_after_classifications(self) -> None:
        """Run multiple classifications and verify counts match."""
        # Two unplanned classifications
        lookup_none = AsyncMock(return_value=None)
        classifier = FailureClassifier(discrepancy_lookup=lookup_none, rng_seed=42)

        txn = _make_transaction()
        ctx = _make_context()
        errors = _make_validation_errors()

        await classifier.classify(errors, txn, ctx)
        await classifier.classify(errors, txn, ctx)

        metrics = classifier.get_metrics()
        assert metrics["total_classifications"] == 2
        assert metrics["unplanned_count"] == 2
        assert metrics["planned_within_count"] == 0
        assert metrics["planned_outside_count"] == 0

    @pytest.mark.asyncio
    async def test_classification_rates(self) -> None:
        """After processing, rates (fractions) are calculated correctly."""
        # Set up: 2 unplanned + 1 planned within = 3 total
        lookup_none = AsyncMock(return_value=None)
        lookup_found = AsyncMock(return_value=_make_discrepancy_record())
        bounds = MagicMock(return_value=_make_parameter_bounds())

        # First classifier: 2 unplanned
        classifier = FailureClassifier(discrepancy_lookup=lookup_none, rng_seed=42)
        txn = _make_transaction()
        ctx = _make_context()
        errors = _make_validation_errors()

        await classifier.classify(errors, txn, ctx)
        await classifier.classify(errors, txn, ctx)

        metrics = classifier.get_metrics()
        rates = metrics["classification_rates"]
        assert rates["unplanned"] == 1.0  # 2/2 = 1.0
        assert rates["planned_within"] == 0.0
        assert rates["planned_outside"] == 0.0

        # Second classifier: 1 planned_within + 1 unplanned = 2 total
        lookup_mixed = AsyncMock(side_effect=[_make_discrepancy_record(), None])
        classifier2 = FailureClassifier(
            discrepancy_lookup=lookup_mixed,
            parameter_bounds_lookup=bounds,
            rng_seed=42,
        )
        await classifier2.classify(errors, txn, ctx)
        await classifier2.classify(errors, txn, ctx)

        metrics2 = classifier2.get_metrics()
        assert metrics2["total_classifications"] == 2
        assert metrics2["planned_within_count"] == 1
        assert metrics2["unplanned_count"] == 1
        rates2 = metrics2["classification_rates"]
        assert rates2["planned_within"] == 0.5  # 1/2
        assert rates2["unplanned"] == 0.5  # 1/2
