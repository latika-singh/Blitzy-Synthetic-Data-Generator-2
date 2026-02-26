"""CTL-004: Backdated Transaction — simulates temporal manipulation of transaction dates.

Implements the Backdated Transaction discrepancy type that modifies the transaction
date to be significantly earlier than the posting/entry date. This simulates a
common fraud/error pattern where transactions are deliberately dated in the past,
potentially to:
    - Record transactions in a prior fiscal period
    - Manipulate period-end financial results
    - Circumvent period close controls
    - Backdate invoices to meet payment terms

The ``days_back`` parameter controls how far into the past the transaction is
backdated (1–90 days). This is a 'medium' difficulty discrepancy because
detection requires comparing the transaction date against the posting date or
system entry date, which requires multi-field analysis.

Catalog Entry:
    Type Code: CTL-004
    Category: control
    Difficulty: medium
    Detection Method: date_check
    Parameters: days_back (1-90) — number of days to backdate the transaction

References:
    - AAP Section 0.5.1 Group 5: CTL-004 Backdated Transaction
    - AAP Section 0.7.5: Discrepancy Injection Rules (parameter bounds)
    - app/orchestration/time_controller.py: TimeController date management
    - app/orchestration/fiscal_calendar.py: FiscalCalendar period boundaries
"""

from __future__ import annotations

import copy
import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Optional, Tuple

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
__all__: list[str] = ["BackdatedTransaction"]

# ---------------------------------------------------------------------------
# Parameter bounds from discrepancy catalog (AAP §0.7.5)
# ---------------------------------------------------------------------------
PARAMETER_BOUNDS: Dict[str, Dict[str, Any]] = {
    "days_back": {"min": 1, "max": 90, "type": "int"},
}

# ---------------------------------------------------------------------------
# Date field definitions — priority-ordered lists of known field names
# ---------------------------------------------------------------------------

# Date fields that can be backdated, in priority order.
# The injector will prefer earlier entries when selecting a target field.
BACKDATABLE_DATE_FIELDS: list[str] = [
    "transaction_date",
    "document_date",
    "posting_date",
    "invoice_date",
    "order_date",
    "receipt_date",
    "payment_date",
    "entry_date",
]

# Reference date fields represent the "current" or "system" date that the
# backdated date should be compared against during detection.  The injector
# uses these to establish the temporal baseline for the backdating offset.
REFERENCE_DATE_FIELDS: list[str] = [
    "posting_date",
    "entry_date",
    "system_date",
    "created_at",
    "created_date",
]


class BackdatedTransaction(BaseDiscrepancy):
    """CTL-004: Backdated Transaction.

    Injects a backdating violation by modifying the transaction date to be
    significantly earlier than the posting/entry date.

    This is a 'medium' difficulty discrepancy because detection requires
    comparing the transaction date against the posting or system entry date
    — a multi-field temporal analysis rather than a simple single-field check.

    Class Attributes:
        type_code: ``"CTL-004"`` — unique catalog identifier.
        category: ``"control"`` — control discrepancy category.
        difficulty: ``"medium"`` — requires multi-field temporal analysis.
        name: ``"Backdated Transaction"`` — human-readable name.
        description: Detailed description of the backdating scenario.
        detection_method: ``"date_check"`` — detected via date comparison.
        PARAMETER_BOUNDS: Bounds for ``days_back`` parameter (1–90).
    """

    # ------------------------------------------------------------------
    # Class-level attributes — override BaseDiscrepancy defaults
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "CTL-004"
    category: ClassVar[str] = "control"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Backdated Transaction"
    description: ClassVar[str] = (
        "Transaction date set significantly in the past relative to "
        "the posting/entry date"
    )
    detection_method: ClassVar[str] = "date_check"

    # Parameter bounds for this discrepancy type (consumed by _validate_params)
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = PARAMETER_BOUNDS

    # ------------------------------------------------------------------
    # inject() — core contract implementation
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a backdated transaction discrepancy.

        Modifies the transaction date to be ``days_back`` days earlier than
        the reference date (posting_date, entry_date, or system_date).  If
        no explicit reference date field exists, the target date field itself
        serves as the baseline.

        Args:
            transaction: Transaction data dictionary containing date fields.
                At least one date field from :data:`BACKDATABLE_DATE_FIELDS`
                or :data:`REFERENCE_DATE_FIELDS` must be present.
            params: Injection parameters.  Recognised keys:

                - ``"days_back"`` (int): Number of days to backdate the
                  transaction (1–90).  If absent, a random value within
                  bounds is generated using *rng*.

            rng: A seeded :class:`random.Random` instance for deterministic
                behaviour.  MUST use this — NEVER module-level ``random``.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)`` where:

            - **modified_transaction** contains the backdated date field.
            - **ground_truth_data** contains type_code, affected_fields,
              original/modified values, financial_impact, description, and
              metadata including ``days_back``, ``reference_date``,
              ``reference_field``, ``backdated_field``, and
              ``may_cross_period_boundary``.

        Raises:
            DiscrepancyInjectionError: If no date fields are found in the
                transaction dictionary.
        """
        # Step 1: Deep-copy the transaction to preserve the original
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # Step 2: Validate and normalise injection parameters
        validated_params: Dict[str, Any] = self._validate_params(params, PARAMETER_BOUNDS)

        # Step 3: Extract days_back (or generate a random value within bounds)
        days_back: int = int(
            validated_params.get("days_back", rng.randint(1, 90))
        )

        # Step 4: Locate the reference date — the temporal baseline
        reference_field: Optional[str] = None
        reference_date: Optional[date] = None

        for field_name in REFERENCE_DATE_FIELDS:
            if field_name in modified:
                parsed = self._parse_date(modified[field_name])
                if parsed is not None:
                    reference_field = field_name
                    reference_date = parsed
                    break

        # Step 5: Locate the target date field to backdate
        target_field: Optional[str] = None
        original_date: Optional[date] = None
        original_raw_value: Any = None

        # Collect all candidate backdatable fields that are NOT the reference
        candidates: List[str] = []
        for field_name in BACKDATABLE_DATE_FIELDS:
            if field_name in modified and field_name != reference_field:
                parsed = self._parse_date(modified[field_name])
                if parsed is not None:
                    candidates.append(field_name)

        if candidates:
            # Pick a target from the candidates — first is highest priority,
            # but if multiple exist use rng.choice() for deterministic variety
            if len(candidates) == 1:
                target_field = candidates[0]
            else:
                target_field = rng.choice(candidates)
            original_raw_value = modified[target_field]
            original_date = self._parse_date(original_raw_value)
        elif reference_field is not None and reference_date is not None:
            # No separate backdatable field found — use the reference field
            # itself as both reference and target
            target_field = reference_field
            original_raw_value = modified[target_field]
            original_date = reference_date
        else:
            # Last resort: scan ALL fields for anything that looks like a date
            all_date_fields: List[str] = list(
                set(BACKDATABLE_DATE_FIELDS) | set(REFERENCE_DATE_FIELDS)
            )
            for field_name in all_date_fields:
                if field_name in modified:
                    parsed = self._parse_date(modified[field_name])
                    if parsed is not None:
                        target_field = field_name
                        original_raw_value = modified[field_name]
                        original_date = parsed
                        # Also set as reference if we had none
                        if reference_date is None:
                            reference_field = field_name
                            reference_date = parsed
                        break

        # If we still have no target, scan for ANY key containing "date"
        if target_field is None or original_date is None:
            for key, value in modified.items():
                parsed = self._parse_date(value)
                if parsed is not None:
                    target_field = key
                    original_raw_value = value
                    original_date = parsed
                    if reference_date is None:
                        reference_field = key
                        reference_date = parsed
                    break

        # If absolutely no date fields exist, raise an error
        if target_field is None or original_date is None:
            raise DiscrepancyInjectionError(
                "No date fields found in transaction for backdating",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_keys": list(modified.keys()),
                },
            )

        # Ensure reference date has a fallback
        if reference_date is None:
            reference_field = target_field
            reference_date = original_date

        # Step 7: Calculate the backdated date
        backdated_date: date = original_date - timedelta(days=days_back)

        # Step 8-9: Set the modified value, preserving the original format
        modified[target_field] = self._format_date(backdated_date, original_raw_value)

        # Step 10: Financial impact is zero — backdating is a process/control risk
        financial_impact: Decimal = Decimal("0")

        # Determine whether the backdating crosses a fiscal month boundary
        may_cross_period_boundary: bool = (
            original_date.month != backdated_date.month
            or original_date.year != backdated_date.year
        )

        # Step 11: Create structured ground truth data
        ground_truth: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=[target_field],
            original_values={target_field: str(original_date.isoformat())},
            modified_values={target_field: str(backdated_date.isoformat())},
            financial_impact=financial_impact,
            description=(
                f"Transaction backdated by {days_back} days "
                f"from {original_date.isoformat()} to {backdated_date.isoformat()}"
            ),
            extra_metadata={
                "days_back": days_back,
                "reference_date": str(reference_date.isoformat()),
                "reference_field": reference_field,
                "backdated_field": target_field,
                "may_cross_period_boundary": may_cross_period_boundary,
            },
        )

        # Step 12: Log the injection event with structured context
        transaction_id: str = str(
            modified.get("transaction_id", modified.get("id", "unknown"))
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "days_back": days_back,
                "original_date": str(original_date.isoformat()),
                "backdated_date": str(backdated_date.isoformat()),
                "target_field": target_field,
                "reference_field": reference_field,
                "may_cross_period_boundary": may_cross_period_boundary,
            },
        )

        # Step 13: Return modified transaction and ground truth
        return modified, ground_truth

    # ------------------------------------------------------------------
    # Helper: _parse_date()
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """Parse a date value from various Python and string representations.

        Handles the following input types:

        - :class:`datetime.date` objects — returned as-is.
        - :class:`datetime.datetime` objects — the ``.date()`` component
          is extracted.
        - ISO-8601 date strings (``"YYYY-MM-DD"``) — parsed via
          :meth:`date.fromisoformat`.
        - ISO-8601 datetime strings (``"YYYY-MM-DDTHH:MM:SS"``) — the
          date component is extracted.
        - Any other type or unparseable string — returns ``None``.

        Args:
            value: The value to attempt parsing as a date.

        Returns:
            A :class:`datetime.date` object, or ``None`` if parsing fails.
        """
        if value is None:
            return None

        # Already a date object (but not datetime, which is a subclass)
        if isinstance(value, date) and not isinstance(value, datetime):
            return value

        # datetime object — extract the date component
        if isinstance(value, datetime):
            return value.date()

        # String representations
        if isinstance(value, str):
            # Try ISO-8601 date first ("YYYY-MM-DD")
            try:
                return date.fromisoformat(value[:10])
            except (ValueError, IndexError):
                pass

            # Try full ISO-8601 datetime ("YYYY-MM-DDTHH:MM:SS...")
            try:
                return datetime.fromisoformat(value).date()
            except (ValueError, TypeError):
                pass

        return None

    # ------------------------------------------------------------------
    # Helper: _format_date()
    # ------------------------------------------------------------------
    @staticmethod
    def _format_date(d: date, original_format: Any) -> Any:
        """Format a date to match the original field's Python type.

        Preserves the type convention of the original value so that
        downstream consumers receive consistent data types:

        - If the original was a :class:`str`, the result is an ISO-8601
          date string (``"YYYY-MM-DD"``).
        - If the original was a :class:`datetime.datetime`, the result is
          a :class:`datetime.datetime` at midnight UTC.
        - If the original was a :class:`datetime.date`, the result is a
          :class:`datetime.date` object.
        - For any other type, the ISO-8601 string representation is used
          as a safe fallback.

        Args:
            d: The date to format.
            original_format: The original field value, used solely to
                determine the target Python type.

        Returns:
            The formatted date in the same Python type as *original_format*.
        """
        if isinstance(original_format, str):
            # If the original string contained a time component, include it
            if "T" in original_format:
                return datetime(
                    d.year, d.month, d.day, 0, 0, 0
                ).isoformat()
            return d.isoformat()

        if isinstance(original_format, datetime):
            return datetime(d.year, d.month, d.day, 0, 0, 0)

        if isinstance(original_format, date):
            return d

        # Fallback: ISO string for unknown types
        return d.isoformat()
