"""GL-003: Period-End Adjusting Entry — simulates suspicious period-end timing.

Implements the Suspicious Adjusting Entry discrepancy type that modifies a journal
entry's posting date to fall within the last few days before period close. This is
a common audit red flag because:

    - Late adjustments may be used to manipulate period-end financial results
    - Period-close pressure can lead to hastily approved entries
    - Quarter-end and year-end adjustments receive heightened audit scrutiny
    - Pattern of late adjustments may indicate earnings management

The ``days_before_close`` parameter controls how close to the period end the
entry is positioned (1–5 days). Entries closer to period close (1 day) are more
suspicious than those 5 days before.

Fiscal Period Context (from ``app/orchestration/fiscal_calendar.py``):
    - Period lifecycle: OPEN → CLOSING → CLOSED
    - Monthly periods: 12 per fiscal year
    - Period end dates: Last calendar day of each month
    - Quarter-end and year-end flags trigger heightened scrutiny

Catalog Entry:
    Type Code: GL-003
    Category: gl
    Difficulty: medium
    Detection Method: period_analysis
    Parameters: days_before_close (1–5)

References:
    - AAP Section 0.5.1 Group 5: GL-003 Suspicious Adjusting Entry
    - AAP Section 0.7.5: Discrepancy Injection Rules (parameter bounds)
    - app/orchestration/fiscal_calendar.py: FiscalCalendar, FiscalPeriod, PeriodStatus
    - app/orchestration/time_controller.py: PeriodClosing/PeriodClosed events
"""

from __future__ import annotations

import calendar
import copy
import random
from datetime import date, timedelta
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
__all__ = ["SuspiciousAdjusting"]

# ---------------------------------------------------------------------------
# Parameter bounds from discrepancy catalog
# ---------------------------------------------------------------------------
PARAMETER_BOUNDS: Dict[str, Dict[str, Any]] = {
    "days_before_close": {"min": 1, "max": 5, "type": "int"},
}

# ---------------------------------------------------------------------------
# Date fields that can be moved to period end.
# Iterated in priority order — the first match is used.
# ---------------------------------------------------------------------------
POSTABLE_DATE_FIELDS: list[str] = [
    "posting_date",
    "entry_date",
    "transaction_date",
    "document_date",
    "effective_date",
    "journal_date",
]

# ---------------------------------------------------------------------------
# Adjusting entry type indicators — keywords that signal an adjusting entry
# ---------------------------------------------------------------------------
ADJUSTING_ENTRY_TYPES: list[str] = [
    "adjusting",
    "adjustment",
    "period_end",
    "close",
    "reclassification",
    "accrual",
    "reclass",
]

# ---------------------------------------------------------------------------
# Fields that may hold the entry type/classification
# ---------------------------------------------------------------------------
_ENTRY_TYPE_FIELDS: list[str] = [
    "entry_type",
    "je_type",
    "type",
    "journal_type",
    "entry_category",
]

# ---------------------------------------------------------------------------
# Quarter-end months for calendar-year fiscal alignment (Mar, Jun, Sep, Dec)
# ---------------------------------------------------------------------------
_QUARTER_END_MONTHS: frozenset[int] = frozenset({3, 6, 9, 12})


class SuspiciousAdjusting(BaseDiscrepancy):
    """GL-003: Period-End Adjusting Entry (suspicious timing).

    Injects a suspicious adjusting entry by moving the posting date to
    within the last few days of the fiscal period. Also marks the entry
    type as 'adjusting' if not already so marked.

    This is a 'medium' difficulty discrepancy because detection requires:

    1. Knowing the fiscal period boundaries (from FiscalCalendar)
    2. Checking the posting date proximity to period end
    3. Optionally checking the entry type for adjusting indicators
    4. Assessing whether the timing is genuinely suspicious in context

    Attributes:
        type_code: ``"GL-003"``
        category: ``"gl"``
        difficulty: ``"medium"``
        name: Human-readable name for this discrepancy type.
        description: Detailed description.
        detection_method: ``"period_analysis"``
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "GL-003"
    category: ClassVar[str] = "gl"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Suspicious Period-End Adjusting Entry"
    description: ClassVar[str] = (
        "Journal entry posted in the last few days before period close, "
        "a common pattern for suspicious period-end manipulations"
    )
    detection_method: ClassVar[str] = "period_analysis"

    # Parameter bounds for _validate_params() default lookup
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = PARAMETER_BOUNDS

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection contract
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a suspicious period-end adjusting entry.

        Moves the posting date to within *days_before_close* days of the
        period end date and marks the entry type as an adjusting entry.

        Args:
            transaction: Journal entry transaction data dictionary.
            params: Injection parameters.  Recognised key:
                ``"days_before_close"`` (int, 1–5) — how many days before
                the period end to place the posting date.  If not provided,
                a value is chosen uniformly at random from 1–5 using *rng*.
            rng: Seeded :class:`random.Random` instance for deterministic
                reproducibility.  CRITICAL: only this instance is used for
                random choices — NEVER the module-level RNG.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``
            where *modified_transaction* has the adjusted posting date and
            entry type, and *ground_truth_data* follows the schema expected
            by :class:`~app.discrepancies.ground_truth_generator.GroundTruthGenerator`.

        Raises:
            DiscrepancyInjectionError: If no postable date fields are
                found in the transaction dictionary.
        """
        # Step 1: Deep copy the transaction to preserve the original
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # Step 2: Validate params against configured bounds (auto-adjust=True)
        validated_params = self._validate_params(params, PARAMETER_BOUNDS)

        # Step 3: Extract or generate days_before_close
        days_before_close: int = validated_params.get("days_before_close")  # type: ignore[assignment]
        if days_before_close is None:
            days_before_close = rng.randint(1, 5)

        # Ensure integral type after potential Decimal-based clamping
        days_before_close = int(days_before_close)

        # Step 4: Find the first postable date field present in the transaction
        date_field_name: Optional[str] = None
        original_date_raw: Any = None
        for field_name in POSTABLE_DATE_FIELDS:
            if field_name in modified:
                date_field_name = field_name
                original_date_raw = modified[field_name]
                break

        # Step 5: Raise if no date field found
        if date_field_name is None:
            raise DiscrepancyInjectionError(
                "No postable date fields found in the transaction",
                details={
                    "discrepancy_type": self.type_code,
                    "searched_fields": POSTABLE_DATE_FIELDS,
                    "transaction_keys": list(modified.keys()),
                },
            )

        # Step 6: Parse the original date — handle date objects and ISO strings
        original_date: date = self._parse_date(original_date_raw)

        # Step 7: Calculate the period end date for the month of the original date
        period_end: date = self._get_period_end(original_date)

        # Step 8: Calculate the new posting date
        new_date: date = period_end - timedelta(days=days_before_close)

        # Step 9: Ensure the new date is still within the same month
        month_start = date(original_date.year, original_date.month, 1)
        if new_date < month_start:
            new_date = month_start

        # Step 10: Set the modified transaction's date field
        modified[date_field_name] = self._format_date(new_date, original_date_raw)

        # Step 11: Modify or add the entry type
        affected_fields: List[str] = [date_field_name]
        original_values: Dict[str, Any] = {
            date_field_name: str(original_date),
        }
        modified_values: Dict[str, Any] = {
            date_field_name: str(new_date),
        }

        # Find an existing entry type field and modify it, or add one
        entry_type_field: Optional[str] = None
        original_entry_type: Optional[str] = None
        new_entry_type: str = rng.choice(ADJUSTING_ENTRY_TYPES)

        for etype_field in _ENTRY_TYPE_FIELDS:
            if etype_field in modified:
                entry_type_field = etype_field
                original_entry_type = str(modified[etype_field])
                modified[etype_field] = new_entry_type
                break

        if entry_type_field is None:
            # No existing entry type field — add "entry_type" to the transaction
            entry_type_field = "entry_type"
            original_entry_type = "NOT_PRESENT"
            modified[entry_type_field] = new_entry_type

        affected_fields.append(entry_type_field)
        original_values[entry_type_field] = original_entry_type
        modified_values[entry_type_field] = new_entry_type

        # Step 12: Check quarter-end and year-end flags
        is_quarter_end: bool = self._is_quarter_end(new_date)
        is_year_end: bool = self._is_year_end(new_date)

        # Step 13: Determine financial impact from transaction amount
        financial_impact: Decimal = self._extract_financial_impact(modified)

        # Step 14: Build ground truth description
        gt_description = (
            f"Adjusting entry posted {days_before_close} day(s) before "
            f"period close ({period_end})"
        )
        if is_year_end:
            gt_description += " — YEAR-END (heightened audit scrutiny)"
        elif is_quarter_end:
            gt_description += " — QUARTER-END (heightened audit scrutiny)"

        # Step 15: Create ground truth data via inherited helper
        ground_truth_data: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=affected_fields,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=financial_impact,
            description=gt_description,
            extra_metadata={
                "days_before_close": days_before_close,
                "period_end_date": str(period_end),
                "original_date": str(original_date),
                "new_date": str(new_date),
                "is_quarter_end": is_quarter_end,
                "is_year_end": is_year_end,
                "entry_type": new_entry_type,
                "date_field": date_field_name,
            },
        )

        # Step 16: Log the injection event
        transaction_id: str = str(
            modified.get("transaction_id")
            or modified.get("journal_entry_id")
            or modified.get("je_id")
            or modified.get("id")
            or "unknown"
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "days_before_close": days_before_close,
                "period_end_date": str(period_end),
                "is_quarter_end": is_quarter_end,
                "is_year_end": is_year_end,
            },
        )

        return modified, ground_truth_data

    # ------------------------------------------------------------------
    # Static Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_period_end(d: date) -> date:
        """Calculate the last day of the month for a given date.

        Uses :func:`calendar.monthrange` to handle variable month lengths
        (28–31 days), including leap-year February handling.

        Args:
            d: Any :class:`date` within the target month.

        Returns:
            A :class:`date` representing the last calendar day of
            that month.
        """
        _, last_day = calendar.monthrange(d.year, d.month)
        return date(d.year, d.month, last_day)

    @staticmethod
    def _is_quarter_end(d: date) -> bool:
        """Check if the date falls in a quarter-end month.

        Quarter-end months for a calendar-year fiscal alignment are
        March (3), June (6), September (9), and December (12).

        Args:
            d: The date to check.

        Returns:
            ``True`` if *d*'s month is a quarter-end month.
        """
        return d.month in _QUARTER_END_MONTHS

    @staticmethod
    def _is_year_end(d: date) -> bool:
        """Check if the date falls in the year-end month (December).

        For a calendar-year fiscal alignment, December is the final
        month of the fiscal year and receives the highest level of
        audit scrutiny.

        Args:
            d: The date to check.

        Returns:
            ``True`` if *d*'s month is December (12).
        """
        return d.month == 12

    @staticmethod
    def _parse_date(value: Any) -> date:
        """Parse a date from various representations.

        Handles:
        - :class:`datetime.date` objects (returned as-is)
        - ISO-8601 date strings (``"YYYY-MM-DD"``)
        - ISO-8601 datetime strings (date portion extracted)

        Args:
            value: The raw date value from the transaction dictionary.

        Returns:
            A :class:`date` object.

        Raises:
            DiscrepancyInjectionError: If the value cannot be parsed
                as a date.
        """
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                # Handle both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS..." formats
                return date.fromisoformat(value[:10])
            except (ValueError, TypeError) as exc:
                raise DiscrepancyInjectionError(
                    f"Cannot parse date from string: {value!r}",
                    details={"raw_value": str(value), "error": str(exc)},
                ) from exc
        raise DiscrepancyInjectionError(
            f"Unsupported date type: {type(value).__name__}",
            details={"raw_value": str(value), "type": type(value).__name__},
        )

    @staticmethod
    def _format_date(new_date: date, original_raw: Any) -> Any:
        """Format the new date to match the original field's type.

        If the original was a :class:`date` object, returns a :class:`date`.
        If the original was a string, returns an ISO-8601 date string.

        Args:
            new_date: The computed new date value.
            original_raw: The original raw value from the transaction,
                used to determine the output format.

        Returns:
            The formatted date in the same type as *original_raw*.
        """
        if isinstance(original_raw, date):
            return new_date
        # Default to ISO string
        return new_date.isoformat()

    @staticmethod
    def _extract_financial_impact(transaction: Dict[str, Any]) -> Decimal:
        """Extract the financial impact amount from a transaction.

        Searches for common amount fields in priority order.  If the
        transaction contains journal entry lines, sums the debit amounts.
        Falls back to ``Decimal("0")`` if no amount can be determined.

        All returned values are :class:`Decimal` — NEVER ``float``.

        Args:
            transaction: The transaction dictionary to inspect.

        Returns:
            The financial impact as a :class:`Decimal`.
        """
        # Direct amount fields (priority order)
        for amount_field in ("amount", "total_amount", "entry_amount", "je_amount"):
            if amount_field in transaction:
                raw = transaction[amount_field]
                try:
                    return abs(Decimal(str(raw)))
                except Exception:
                    continue

        # Sum of journal entry line debits
        for lines_field in ("je_lines", "journal_lines", "lines", "entry_lines"):
            if lines_field in transaction and isinstance(transaction[lines_field], list):
                total = Decimal("0")
                for line in transaction[lines_field]:
                    if isinstance(line, dict):
                        for debit_field in ("debit_amount", "debit"):
                            if debit_field in line:
                                try:
                                    total += abs(Decimal(str(line[debit_field])))
                                except Exception:
                                    pass
                if total > Decimal("0"):
                    return total

        return Decimal("0")
