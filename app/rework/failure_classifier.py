"""Failure Classifier — Classifies validation failures for the rework loop.

The ``FailureClassifier`` determines whether a validation failure is:

1. **Planned Discrepancy (Within Bounds)** — The failure was caused by an
   intentionally injected discrepancy and the injected parameters are within
   the configured bounds. Action: the rework loop should NOT attempt to fix
   this (the discrepancy is intentional).

2. **Planned Discrepancy (Outside Bounds)** — The failure was caused by an
   intentionally injected discrepancy, but the parameters drifted outside
   configured bounds. Action: the rework loop should adjust parameters back
   to within bounds.

3. **Unplanned Error** — The failure was NOT caused by an intentionally
   injected discrepancy. This is a genuine processing error. Action: the
   rework loop should attempt to fix it.

Classification Method:
    The classifier queries the discrepancy records (via injected lookup function)
    to check if a discrepancy was intentionally injected for the failed transaction.
    If yes, it checks whether the discrepancy parameters are within configured
    bounds. If no discrepancy record exists, the failure is classified as
    an unplanned error.

Design Decisions:
    - Constructor injection (ADR-003): discrepancy lookup via injected callable
    - Pydantic V2: ClassificationResult model for typed output
    - structlog: JSON to stdout with service_name, component, trace_id
    - Deterministic: Seeded random.Random for confidence score jitter

References:
    - AAP Section 0.5.1 Group 6: failure_classifier.py
    - AAP Section 0.7.5: Discrepancy parameter bounds checking
    - AAP Section 0.7.1: Constructor injection, Pydantic V2
"""

from __future__ import annotations

import asyncio
import inspect
import random
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.exceptions import ReworkLoopError

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "FailureClassifier",
    "ClassificationResult",
    "ClassificationType",
    "RecommendedAction",
]

# ═══════════════════════════════════════════════════════════════════════════
# Module-level structured logger (AAP §0.7.7 — structlog to stdout only)
# ═══════════════════════════════════════════════════════════════════════════

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Enums — Classification types and recommended actions
# ═══════════════════════════════════════════════════════════════════════════


class ClassificationType(str, Enum):
    """Classification types for validation failures.

    Three mutually-exclusive categories that the ``FailureClassifier`` assigns
    to each validation failure based on whether it was caused by a planned
    discrepancy injection and whether the injected parameters are within
    configured bounds.

    Members:
        PLANNED_WITHIN_BOUNDS: Planned discrepancy with parameters inside
            configured bounds — rework should skip (discrepancy is intentional).
        PLANNED_OUTSIDE_BOUNDS: Planned discrepancy with parameters that
            drifted outside configured bounds — rework should adjust them.
        UNPLANNED_ERROR: No planned discrepancy found — genuine processing
            error that needs a fix scenario applied.
    """

    PLANNED_WITHIN_BOUNDS = "planned_discrepancy_within_bounds"
    PLANNED_OUTSIDE_BOUNDS = "planned_discrepancy_outside_bounds"
    UNPLANNED_ERROR = "unplanned_error"


class RecommendedAction(str, Enum):
    """Recommended actions based on failure classification.

    Maps each classification type to a concrete action the rework loop
    engine should take in response.

    Members:
        SKIP_REWORK: Planned discrepancy within bounds — no fix needed,
            preserve the intentional discrepancy.
        ADJUST_PARAMETERS: Planned discrepancy outside bounds — clamp
            parameters back to within configured bounds.
        APPLY_FIX: Unplanned error — select and apply a fix scenario
            from the ``FixScenarioCatalog``.
        ESCALATE: Cannot classify or all fix attempts exhausted —
            escalate to admin for human review.
    """

    SKIP_REWORK = "skip_rework"
    ADJUST_PARAMETERS = "adjust_parameters"
    APPLY_FIX = "apply_fix"
    ESCALATE = "escalate"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════


class ClassificationResult(BaseModel):
    """Result of classifying a validation failure.

    This Pydantic V2 model is the cross-subsystem data contract returned by
    ``FailureClassifier.classify()``. It carries the classification type,
    recommended action, confidence score, and detailed context about the
    classification decision.

    Attributes:
        classification_type: One of :class:`ClassificationType` values.
        recommended_action: One of :class:`RecommendedAction` values.
        confidence_score: Classification confidence between 0.0 and 1.0.
            Higher values indicate greater certainty in the classification.
        is_planned_discrepancy: ``True`` if the failure was caused by a
            planned discrepancy injection, ``False`` otherwise.
        discrepancy_type_code: Discrepancy type code (e.g. ``'P2P-001'``)
            if this is a planned discrepancy; ``None`` otherwise.
        within_bounds: ``True`` if all discrepancy parameters are within
            configured bounds; ``False`` if any parameter is outside bounds;
            ``None`` if not a planned discrepancy.
        out_of_bounds_params: List of parameter names that are outside their
            configured bounds (empty if all within bounds or not planned).
        error_details: Additional classification context, including validation
            error summaries and lookup metadata.
        timestamp: UTC timestamp when the classification was performed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification_type: str = Field(
        ...,
        description="One of ClassificationType values",
    )
    recommended_action: str = Field(
        ...,
        description="One of RecommendedAction values",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Classification confidence (0-1)",
    )
    is_planned_discrepancy: bool = Field(
        default=False,
        description="Whether this is a planned discrepancy",
    )
    discrepancy_type_code: Optional[str] = Field(
        default=None,
        description="Discrepancy type code if planned (e.g., 'P2P-001')",
    )
    within_bounds: Optional[bool] = Field(
        default=None,
        description="Whether discrepancy params are within bounds",
    )
    out_of_bounds_params: List[str] = Field(
        default_factory=list,
        description="Param names that are outside bounds",
    )
    error_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional classification details",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ═══════════════════════════════════════════════════════════════════════════
# FailureClassifier — Core classification engine
# ═══════════════════════════════════════════════════════════════════════════


class FailureClassifier:
    """Classifies validation failures as planned discrepancy or unplanned error.

    Uses injected discrepancy lookup function to check if a failure was
    caused by an intentionally injected discrepancy.  If a discrepancy record
    is found, the classifier checks whether the injected parameters are within
    configured bounds and produces one of three classification types.

    All constructor parameters are keyword-only and Optional with ``None``
    default to enable testing flexibility and incremental integration
    (ADR-003).

    Args:
        discrepancy_lookup: Async or sync callable that takes a transaction_id
            (str or UUID) and returns a discrepancy record dict, or ``None``
            if no discrepancy was injected for that transaction.
        parameter_bounds_lookup: Callable that takes a discrepancy type_code
            (str) and returns a dict mapping parameter names to
            ``{"min": value, "max": value}`` bounds.
        rng_seed: Seed for the deterministic ``random.Random`` instance used
            for confidence score jitter.  MUST use seeded RNG per AAP §0.7.1.

    Example::

        classifier = FailureClassifier(
            discrepancy_lookup=my_lookup_fn,
            parameter_bounds_lookup=my_bounds_fn,
            rng_seed=42,
        )
        result = await classifier.classify(errors, txn, ctx)
    """

    def __init__(
        self,
        *,
        discrepancy_lookup: Optional[Callable] = None,
        parameter_bounds_lookup: Optional[Callable] = None,
        rng_seed: int = 42,
    ) -> None:
        # Injected dependencies
        self._discrepancy_lookup = discrepancy_lookup
        self._parameter_bounds_lookup = parameter_bounds_lookup

        # Seeded RNG for deterministic confidence jitter (NEVER module-level)
        self._rng = random.Random(rng_seed)

        # Classification metrics counters
        self._total_classifications: int = 0
        self._planned_within_count: int = 0
        self._planned_outside_count: int = 0
        self._unplanned_count: int = 0
        self._escalation_count: int = 0

        logger.info(
            "failure_classifier_initialized",
            service_name="transactions",
            component="FailureClassifier",
            has_discrepancy_lookup=discrepancy_lookup is not None,
            has_bounds_lookup=parameter_bounds_lookup is not None,
            rng_seed=rng_seed,
        )

    # ------------------------------------------------------------------
    # Core classification method
    # ------------------------------------------------------------------

    async def classify(
        self,
        validation_errors: List[Dict[str, Any]],
        transaction: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ClassificationResult:
        """Classify a validation failure.

        Implements a 3-path classification algorithm:

        1. If a planned discrepancy record exists for the transaction AND
           all injected parameters are within configured bounds →
           **PLANNED_WITHIN_BOUNDS** (action: SKIP_REWORK).

        2. If a planned discrepancy record exists but one or more parameters
           are outside configured bounds → **PLANNED_OUTSIDE_BOUNDS**
           (action: ADJUST_PARAMETERS).

        3. If no discrepancy record exists → **UNPLANNED_ERROR**
           (action: APPLY_FIX).

        If the classification process itself fails fatally, the method
        falls back to ESCALATE and raises ``ReworkLoopError``.

        Args:
            validation_errors: List of validation error detail dicts, each
                containing at minimum ``type``, ``field``, and ``message``.
            transaction: The failed transaction data dictionary.  Must
                contain a ``transaction_id`` key for discrepancy lookup.
            context: Generation context carrying ``simulation_id``,
                ``trace_id``, ``current_date``, and other simulation state.

        Returns:
            :class:`ClassificationResult` with classification type,
            recommended action, confidence score, and detailed context.

        Raises:
            ReworkLoopError: If the classification process encounters a
                fatal error that prevents classification (e.g. lookup
                function raises an unrecoverable exception).
        """
        self._total_classifications += 1
        transaction_id = transaction.get("transaction_id")

        # Extract error summary for logging / details
        error_summary = self._summarize_errors(validation_errors)

        try:
            # ---- Step 1: Attempt discrepancy lookup ----
            discrepancy_record: Optional[Dict[str, Any]] = None

            if self._discrepancy_lookup is not None and transaction_id is not None:
                discrepancy_record = await self._invoke_lookup(transaction_id)

            # ---- Step 2: Branch on planned vs. unplanned ----
            if discrepancy_record is not None:
                # This is a planned discrepancy — check parameter bounds
                type_code = discrepancy_record.get("type_code", "")
                discrepancy_params = discrepancy_record.get("parameters", {})

                all_within_bounds = True
                out_of_bounds_params: List[str] = []

                if self._parameter_bounds_lookup is not None and type_code:
                    bounds = self._parameter_bounds_lookup(type_code)
                    if bounds:
                        all_within_bounds, out_of_bounds_params = (
                            self._check_parameter_bounds(discrepancy_params, bounds)
                        )

                if all_within_bounds:
                    # Path 1: Planned discrepancy, within bounds
                    classification = ClassificationType.PLANNED_WITHIN_BOUNDS
                    action = RecommendedAction.SKIP_REWORK
                    confidence = 0.95 + self._rng.uniform(0, 0.05)
                    self._planned_within_count += 1
                else:
                    # Path 2: Planned discrepancy, outside bounds
                    classification = ClassificationType.PLANNED_OUTSIDE_BOUNDS
                    action = RecommendedAction.ADJUST_PARAMETERS
                    confidence = 0.80 + self._rng.uniform(0, 0.15)
                    self._planned_outside_count += 1

                result = ClassificationResult(
                    classification_type=classification.value,
                    recommended_action=action.value,
                    confidence_score=round(confidence, 6),
                    is_planned_discrepancy=True,
                    discrepancy_type_code=type_code or None,
                    within_bounds=all_within_bounds,
                    out_of_bounds_params=out_of_bounds_params,
                    error_details={
                        "validation_error_count": len(validation_errors),
                        "error_summary": error_summary,
                        "discrepancy_record_found": True,
                        "discrepancy_type_code": type_code,
                        "parameter_count": len(discrepancy_params),
                        "out_of_bounds_count": len(out_of_bounds_params),
                    },
                )
            else:
                # Path 3: Unplanned error — no discrepancy record found
                classification = ClassificationType.UNPLANNED_ERROR
                action = RecommendedAction.APPLY_FIX
                confidence = 0.70 + self._rng.uniform(0, 0.20)
                self._unplanned_count += 1

                result = ClassificationResult(
                    classification_type=classification.value,
                    recommended_action=action.value,
                    confidence_score=round(confidence, 6),
                    is_planned_discrepancy=False,
                    discrepancy_type_code=None,
                    within_bounds=None,
                    out_of_bounds_params=[],
                    error_details={
                        "validation_error_count": len(validation_errors),
                        "error_summary": error_summary,
                        "discrepancy_record_found": False,
                        "lookup_available": self._discrepancy_lookup is not None,
                        "transaction_id_present": transaction_id is not None,
                    },
                )

        except ReworkLoopError:
            # Re-raise domain-specific errors
            raise
        except Exception as exc:
            # Fatal classification error → escalate
            self._escalation_count += 1
            logger.error(
                "failure_classification_error",
                service_name="transactions",
                component="FailureClassifier",
                transaction_id=str(transaction_id) if transaction_id else None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                simulation_id=context.get("simulation_id"),
                trace_id=context.get("trace_id"),
            )
            raise ReworkLoopError(
                message=f"Classification failed: {exc}",
                details={
                    "transaction_id": str(transaction_id) if transaction_id else None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "validation_error_count": len(validation_errors),
                },
            ) from exc

        # ---- Step 3: Log classification result ----
        logger.info(
            "failure_classified",
            service_name="transactions",
            component="FailureClassifier",
            transaction_id=str(transaction_id) if transaction_id else None,
            classification=result.classification_type,
            recommended_action=result.recommended_action,
            confidence=round(result.confidence_score, 3),
            is_planned=result.is_planned_discrepancy,
            discrepancy_type=result.discrepancy_type_code,
            within_bounds=result.within_bounds,
            simulation_id=context.get("simulation_id"),
            trace_id=context.get("trace_id"),
        )

        return result

    # ------------------------------------------------------------------
    # Parameter bounds checking
    # ------------------------------------------------------------------

    def _check_parameter_bounds(
        self,
        discrepancy_params: Dict[str, Any],
        bounds: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """Check if discrepancy parameters are within configured bounds.

        Iterates over each parameter defined in *bounds* and compares the
        corresponding value in *discrepancy_params* against its ``min`` and
        ``max`` bounds.  Numeric comparisons use ``Decimal`` for precision
        (AAP §0.7.2 — never use ``float`` for financial amounts).

        Parameters that are missing from *discrepancy_params* are silently
        skipped — only parameters that exist AND fall outside their bounds
        are counted as violations.

        Args:
            discrepancy_params: Actual parameter values from the discrepancy
                record, keyed by parameter name.
            bounds: Parameter name → ``{"min": value, "max": value}`` bounds
                dictionary.  Values can be numeric (int, float, str) and
                will be converted to ``Decimal`` for comparison.

        Returns:
            A tuple of ``(all_within_bounds, out_of_bounds_param_names)``
            where *all_within_bounds* is ``True`` if every checked parameter
            falls within its bounds, and *out_of_bounds_param_names* lists
            the parameter names that are outside bounds.
        """
        out_of_bounds: List[str] = []

        for param_name, bound_spec in bounds.items():
            if param_name not in discrepancy_params:
                # Parameter not present in discrepancy record — skip
                continue

            actual_value = discrepancy_params[param_name]
            if actual_value is None:
                continue

            try:
                actual_decimal = Decimal(str(actual_value))
            except (InvalidOperation, ValueError, TypeError):
                # Non-numeric parameter — cannot compare against bounds,
                # treat as out-of-bounds to be safe
                out_of_bounds.append(param_name)
                logger.debug(
                    "parameter_bounds_non_numeric",
                    service_name="transactions",
                    component="FailureClassifier",
                    param_name=param_name,
                    actual_value=str(actual_value),
                )
                continue

            # Check minimum bound
            min_bound = bound_spec.get("min")
            if min_bound is not None:
                try:
                    min_decimal = Decimal(str(min_bound))
                    if actual_decimal < min_decimal:
                        out_of_bounds.append(param_name)
                        continue
                except (InvalidOperation, ValueError, TypeError):
                    pass  # Skip invalid bound spec gracefully

            # Check maximum bound
            max_bound = bound_spec.get("max")
            if max_bound is not None:
                try:
                    max_decimal = Decimal(str(max_bound))
                    if actual_decimal > max_decimal:
                        out_of_bounds.append(param_name)
                        continue
                except (InvalidOperation, ValueError, TypeError):
                    pass  # Skip invalid bound spec gracefully

        all_within = len(out_of_bounds) == 0

        if not all_within:
            logger.debug(
                "parameter_bounds_check_failed",
                service_name="transactions",
                component="FailureClassifier",
                total_checked=len(bounds),
                out_of_bounds_count=len(out_of_bounds),
                out_of_bounds_params=out_of_bounds,
            )

        return all_within, out_of_bounds

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return classification metrics.

        Provides a snapshot of classification activity and distribution
        across the three classification types plus escalation count.

        Returns:
            Dictionary with total and per-type counts, plus classification
            rates expressed as fractions of total classifications.
        """
        total = self._total_classifications

        return {
            "total_classifications": total,
            "planned_within_count": self._planned_within_count,
            "planned_outside_count": self._planned_outside_count,
            "unplanned_count": self._unplanned_count,
            "escalation_count": self._escalation_count,
            "classification_rates": {
                "planned_within": (
                    self._planned_within_count / total if total > 0 else 0.0
                ),
                "planned_outside": (
                    self._planned_outside_count / total if total > 0 else 0.0
                ),
                "unplanned": (
                    self._unplanned_count / total if total > 0 else 0.0
                ),
                "escalation": (
                    self._escalation_count / total if total > 0 else 0.0
                ),
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _invoke_lookup(
        self,
        transaction_id: Any,
    ) -> Optional[Dict[str, Any]]:
        """Invoke the discrepancy lookup function, handling both sync and async.

        The discrepancy lookup callable may be either synchronous or
        asynchronous.  This helper inspects the callable and awaits it
        if necessary.

        Args:
            transaction_id: The transaction identifier to look up.

        Returns:
            Discrepancy record dict if found, ``None`` otherwise.

        Raises:
            ReworkLoopError: If the lookup function raises an exception.
        """
        if self._discrepancy_lookup is None:
            return None

        try:
            result = self._discrepancy_lookup(transaction_id)
            # Handle coroutines from async callables
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            logger.warning(
                "discrepancy_lookup_failed",
                service_name="transactions",
                component="FailureClassifier",
                transaction_id=str(transaction_id),
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise ReworkLoopError(
                message=f"Discrepancy lookup failed for transaction {transaction_id}: {exc}",
                details={
                    "transaction_id": str(transaction_id),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc

    @staticmethod
    def _summarize_errors(
        validation_errors: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Create a compact summary of validation errors for logging.

        Extracts the ``type``, ``field``, and ``message`` keys from each
        error dictionary to produce a concise summary suitable for inclusion
        in structured log entries and ``ClassificationResult.error_details``.

        Args:
            validation_errors: List of validation error detail dictionaries.

        Returns:
            List of compact summary dictionaries with ``type``, ``field``,
            and ``message`` keys.
        """
        summaries: List[Dict[str, str]] = []
        for error in validation_errors:
            summaries.append({
                "type": str(error.get("type", "unknown")),
                "field": str(error.get("field", "")),
                "message": str(error.get("message", "")),
            })
        return summaries
