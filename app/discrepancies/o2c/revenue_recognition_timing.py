"""O2C-006: Revenue Recognition Timing Error discrepancy type.

Simulates premature revenue recognition — recording revenue in an earlier
fiscal period than when goods were shipped or services were delivered.
This is a common financial statement manipulation technique.

In the normal O2C flow, revenue is recognized when:
1. The performance obligation is satisfied (goods shipped or services rendered)
2. The transaction is posted to the correct fiscal period

This discrepancy shifts the revenue recognition date earlier, potentially
crossing period boundaries and inflating current-period revenue.

Detection Method: ``period_analysis`` — comparing invoice dates, shipment dates,
and GL posting dates against fiscal period boundaries.

Difficulty: ``medium`` — requires period boundary analysis and date comparison.

Parameter Bounds:
    days_premature: int, range [1, 30] — number of days revenue is recognized
        before the actual delivery/shipment date.

References:
    - AAP Section 0.5.1 Group 4: GLPostingEngine period validation
    - AAP Section 0.5.1 Group 5: O2C-006 Revenue Recognition Timing
    - AAP Section 0.7.2: Financial Integrity Rules
    - AAP Section 0.7.5: Discrepancy Injection Rules
"""

from __future__ import annotations

import copy
import random
from datetime import timedelta
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Tuple

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["RevenueRecognitionTiming"]


class RevenueRecognitionTiming(BaseDiscrepancy):
    """O2C-006: Revenue Recognition Timing Error discrepancy.

    Simulates premature revenue recognition by shifting the revenue
    recognition date earlier than the actual shipment/delivery date.
    The entire invoice/revenue amount is treated as the financial impact
    because the revenue is recognized in the wrong fiscal period.

    This class extends :class:`BaseDiscrepancy` and implements the
    :meth:`inject` method to:

    1. Deep-copy the transaction to preserve the original.
    2. Validate the ``days_premature`` parameter against configured bounds.
    3. Shift the ``revenue_recognition_date`` (or ``gl_posting_date``)
       earlier by ``days_premature`` days using :class:`datetime.timedelta`.
    4. Set ``period_override`` to ``True`` to indicate manual period
       manipulation.
    5. Derive fiscal period identifiers for both original and manipulated
       dates to support period-analysis detection.
    6. Return the modified transaction and ground truth data.

    Class-Level Attributes:
        type_code: ``"O2C-006"``
        category: ``"o2c"``
        difficulty: ``"medium"``
        name: ``"Revenue Recognition Timing Error"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"period_analysis"``
        PARAMETER_BOUNDS: ``{"days_premature": {"min": 1, "max": 30, "type": "int"}}``

    Example::

        injector = RevenueRecognitionTiming()
        rng = random.Random(42)
        modified, ground_truth = injector.inject(
            transaction={"invoice_id": "INV-2024-0001", ...},
            params={"days_premature": 15},
            rng=rng,
        )
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-006"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Revenue Recognition Timing Error"
    description: ClassVar[str] = (
        "Revenue recognized in an earlier fiscal period than when goods were "
        "shipped or services delivered, inflating current-period revenue."
    )
    detection_method: ClassVar[str] = "period_analysis"

    # ------------------------------------------------------------------
    # Parameter bounds — days_premature: [1, 30] (integer)
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "days_premature": {"min": 1, "max": 30, "type": "int"},
    }

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a revenue recognition timing discrepancy.

        Shifts the revenue recognition date earlier by ``days_premature`` days,
        potentially moving revenue into an earlier fiscal period.  The entire
        invoice/revenue amount is treated as the financial impact because
        the full revenue amount is recognized in the wrong period.

        Processing Steps:
            1. Deep-copy the transaction via :meth:`_copy_transaction`.
            2. Validate ``params`` against :attr:`PARAMETER_BOUNDS` via
               :meth:`_validate_params`.
            3. Extract ``days_premature`` — if not provided, generate a random
               value in ``[1, 30]`` using the seeded ``rng``.
            4. Locate the original recognition date from the transaction,
               checking ``revenue_recognition_date``, ``gl_posting_date``,
               ``invoice_date``, and ``posting_date`` in priority order.
            5. Compute ``new_date = original_date - timedelta(days=days_premature)``.
            6. Update the transaction with the shifted date and set
               ``period_override`` to ``True``.
            7. Determine the financial impact as the full invoice/revenue
               amount (``Decimal``).
            8. Build ground truth data via :meth:`_create_ground_truth_data`.
            9. Log the injection via :meth:`_log_injection`.

        Args:
            transaction: The original O2C transaction data dictionary.
                Expected keys include:

                - ``"revenue_recognition_date"`` or ``"gl_posting_date"`` or
                  ``"invoice_date"`` or ``"posting_date"`` — the date to shift
                - ``"invoice_amount"`` or ``"revenue_amount"`` or
                  ``"total_amount"`` — the monetary value
                - ``"transaction_id"`` or ``"invoice_id"`` — identifier
                - ``"shipment_date"`` — preserved for audit trail
                - ``"fiscal_period"`` — optional current period identifier

            params: Injection parameters.  Recognised keys:

                - ``"days_premature"`` (int): Number of days to shift the
                  recognition date earlier.  Validated against
                  ``PARAMETER_BOUNDS`` (1–30).  If absent, a random
                  value in ``[1, 30]`` is generated via ``rng``.

            rng: A seeded :class:`random.Random` instance for deterministic
                behaviour.  CRITICAL: MUST use this RNG exclusively — NEVER
                use ``random.random()`` or ``random.choice()`` on the
                module-level RNG.

        Returns:
            A tuple of:

            - **modified_transaction** — Transaction with the revenue
              recognition date shifted earlier and ``period_override``
              set to ``True``.
            - **ground_truth_data** — Dictionary conforming to the
              ``GroundTruthGenerator`` schema, including affected fields,
              original/modified values, financial impact, and metadata.

        Raises:
            DiscrepancyInjectionError: If the transaction lacks a usable
                date field or a monetary amount field, or if any other
                injection logic error occurs.
        """
        try:
            # ----- Step 1: Deep copy transaction -----
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ----- Step 2: Validate parameters -----
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # ----- Step 3: Extract days_premature -----
            days_premature: int = int(
                validated_params.get("days_premature", rng.randint(1, 30))
            )

            # ----- Step 4: Locate original recognition date -----
            original_date = _extract_date_field(transaction)
            if original_date is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required date field for revenue "
                    "recognition timing injection — expected one of: "
                    "revenue_recognition_date, gl_posting_date, "
                    "invoice_date, posting_date",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": transaction.get(
                            "transaction_id",
                            transaction.get("invoice_id", "unknown"),
                        ),
                        "available_keys": list(transaction.keys()),
                    },
                )

            # Identify which key held the date for consistent updates
            date_key: str = _resolve_date_key(transaction)

            # ----- Step 5: Shift recognition date earlier -----
            new_date = original_date - timedelta(days=days_premature)

            # ----- Step 6: Modify transaction -----
            # Set the shifted recognition date using the same key found
            modified["revenue_recognition_date"] = new_date
            # Also update gl_posting_date if it was the source key
            if date_key in ("gl_posting_date", "posting_date"):
                modified[date_key] = new_date
            # Mark as manual period override
            modified["period_override"] = True
            # Preserve original shipment_date for audit trail
            # (do NOT modify shipment_date — it is evidence)
            if "shipment_date" not in modified:
                modified["shipment_date"] = original_date
            # Store the original recognition date for traceability
            modified["original_recognition_date"] = original_date

            # Derive fiscal period identifiers (YYYY-MM format)
            original_period: str = _derive_period(original_date)
            manipulated_period: str = _derive_period(new_date)

            # Update the period assignment on the transaction
            if "fiscal_period" in modified:
                modified["original_fiscal_period"] = modified["fiscal_period"]
            modified["fiscal_period"] = manipulated_period

            # ----- Step 7: Determine financial impact -----
            financial_impact: Decimal = _extract_amount(transaction)
            if financial_impact is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required monetary amount field for "
                    "revenue recognition timing injection — expected one of: "
                    "invoice_amount, revenue_amount, total_amount, amount",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": transaction.get(
                            "transaction_id",
                            transaction.get("invoice_id", "unknown"),
                        ),
                        "available_keys": list(transaction.keys()),
                    },
                )

            # ----- Step 8: Build ground truth data -----
            transaction_id: str = str(
                modified.get(
                    "transaction_id",
                    modified.get("invoice_id", "unknown"),
                )
            )

            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=["revenue_recognition_date", "period_override"],
                original_values={
                    "revenue_recognition_date": (
                        original_date.isoformat()
                        if hasattr(original_date, "isoformat")
                        else str(original_date)
                    ),
                    "period_override": False,
                    "fiscal_period": original_period,
                },
                modified_values={
                    "revenue_recognition_date": (
                        new_date.isoformat()
                        if hasattr(new_date, "isoformat")
                        else str(new_date)
                    ),
                    "period_override": True,
                    "fiscal_period": manipulated_period,
                },
                financial_impact=financial_impact,
                description=(
                    f"Revenue recognized {days_premature} days before delivery "
                    f"\u2014 period manipulation"
                ),
                extra_metadata={
                    "days_premature": days_premature,
                    "original_period": original_period,
                    "manipulated_period": manipulated_period,
                    "date_key_used": date_key,
                    "period_boundary_crossed": (
                        original_period != manipulated_period
                    ),
                },
            )

            # ----- Step 9: Log injection -----
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "simulation_id": transaction.get("simulation_id"),
                    "trace_id": transaction.get("trace_id"),
                },
            )

            logger.debug(
                "revenue_recognition_timing_injected",
                service_name="transactions",
                component="RevenueRecognitionTiming",
                type_code=self.type_code,
                transaction_id=transaction_id,
                days_premature=days_premature,
                original_date=str(original_date),
                new_date=str(new_date),
                original_period=original_period,
                manipulated_period=manipulated_period,
                period_boundary_crossed=(
                    original_period != manipulated_period
                ),
                financial_impact=str(financial_impact),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise DiscrepancyInjectionError without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Revenue recognition timing injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": transaction.get(
                        "transaction_id",
                        transaction.get("invoice_id", "unknown"),
                    ),
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
            ) from exc


# ---------------------------------------------------------------------------
# Module-level helper functions (private)
# ---------------------------------------------------------------------------

# Priority order for date field lookup
_DATE_FIELD_PRIORITY: Tuple[str, ...] = (
    "revenue_recognition_date",
    "gl_posting_date",
    "invoice_date",
    "posting_date",
)

# Priority order for amount field lookup
_AMOUNT_FIELD_PRIORITY: Tuple[str, ...] = (
    "invoice_amount",
    "revenue_amount",
    "total_amount",
    "amount",
)


def _extract_date_field(transaction: Dict[str, Any]) -> Any:
    """Extract the best available date field from a transaction dictionary.

    Searches ``_DATE_FIELD_PRIORITY`` in order and returns the first non-None
    value found.  Returns ``None`` if no usable date field exists.

    Args:
        transaction: The transaction data dictionary.

    Returns:
        The date value from the highest-priority date field, or ``None``.
    """
    for key in _DATE_FIELD_PRIORITY:
        value = transaction.get(key)
        if value is not None:
            return value
    return None


def _resolve_date_key(transaction: Dict[str, Any]) -> str:
    """Determine which date key is present in the transaction.

    Returns the key name (string) of the first non-None date field found
    in ``_DATE_FIELD_PRIORITY``.  Falls back to ``"revenue_recognition_date"``
    if no field is found (defensive default).

    Args:
        transaction: The transaction data dictionary.

    Returns:
        The string key name of the date field used.
    """
    for key in _DATE_FIELD_PRIORITY:
        if transaction.get(key) is not None:
            return key
    return "revenue_recognition_date"


def _derive_period(dt: Any) -> str:
    """Derive a fiscal period identifier from a date-like object.

    Attempts to call ``strftime`` for proper date objects.  Falls back to
    string conversion if the value is not a standard ``date``/``datetime``.

    The period format is ``"YYYY-MM"`` (e.g., ``"2024-03"``).

    Args:
        dt: A date, datetime, or string representation of a date.

    Returns:
        A period identifier string in ``"YYYY-MM"`` format, or a best-effort
        string if ``strftime`` is not available.
    """
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m")
    # Fall back to string parsing for non-date types
    dt_str = str(dt)
    if len(dt_str) >= 7 and dt_str[4] == "-":
        return dt_str[:7]
    return dt_str


def _extract_amount(transaction: Dict[str, Any]) -> Decimal | None:
    """Extract the monetary amount from a transaction dictionary.

    Searches ``_AMOUNT_FIELD_PRIORITY`` in order and returns the first
    non-None value found, coerced to :class:`Decimal`.

    All monetary values MUST use ``Decimal`` — NEVER ``float`` — per
    AAP §0.7.2 Financial Integrity Rules.

    Args:
        transaction: The transaction data dictionary.

    Returns:
        The amount as a :class:`Decimal`, or ``None`` if no amount field
        is found.
    """
    for key in _AMOUNT_FIELD_PRIORITY:
        value = transaction.get(key)
        if value is not None:
            if isinstance(value, Decimal):
                return value
            try:
                return Decimal(str(value))
            except Exception:
                continue
    return None
