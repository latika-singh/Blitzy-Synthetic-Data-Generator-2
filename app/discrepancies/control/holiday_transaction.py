"""CTL-005: Transaction on Holiday/Weekend — simulates non-business-day processing.

Implements the Holiday Transaction discrepancy type that modifies the transaction
processing date to fall on a weekend (Saturday or Sunday) or a US Federal holiday.
This is an unusual activity indicator because ERP transactions are normally
processed only during business hours on business days.

The ``BusinessCalendar`` (``app/orchestration/business_calendar.py``) defines
business days as Monday–Friday, 08:00–17:00, excluding US Federal holidays
(2024–2026). This discrepancy violates that norm.

US Federal Holidays (from BusinessCalendar):
    - New Year's Day (Jan 1)
    - Martin Luther King Jr. Day (3rd Monday Jan)
    - Presidents' Day (3rd Monday Feb)
    - Memorial Day (Last Monday May)
    - Juneteenth (Jun 19)
    - Independence Day (Jul 4)
    - Labor Day (1st Monday Sep)
    - Columbus Day (2nd Monday Oct)
    - Veterans Day (Nov 11)
    - Thanksgiving (4th Thursday Nov)
    - Christmas Day (Dec 25)

Weekend/holiday observed rule: Sat → Fri, Sun → Mon.

Catalog Entry:
    Type Code: CTL-005
    Category: control
    Difficulty: easy
    Detection Method: date_check
    Parameters: None (no configurable parameters)

References:
    - AAP Section 0.5.1 Group 5: CTL-005 Transaction on Holiday/Weekend
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - app/orchestration/business_calendar.py: BusinessCalendar, is_business_day()
    - app/orchestration/time_controller.py: TimeController integration
"""

from __future__ import annotations

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
__all__ = ["HolidayTransaction"]

# ---------------------------------------------------------------------------
# US Federal Holidays by Year
# ---------------------------------------------------------------------------
# Matching BusinessCalendar's definitions (2024-2026).
# These are the ACTUAL holiday dates — the discrepancy uses them to move
# a transaction onto a non-business day.  Note: The observed-date rule
# (Sat → Fri, Sun → Mon) is applied by BusinessCalendar for defining which
# weekdays are off; here we store the canonical holiday dates that fall on
# weekdays since we also have weekends as injection targets.
#
# For the discrepancy's purposes we store holidays that fall on weekdays
# because moving a transaction to a weekend is handled by _find_nearest_weekend.
# Holidays on weekdays are uniquely interesting — they look like regular dates
# but are actually non-business days.

US_FEDERAL_HOLIDAYS: Dict[int, List[Tuple[date, str]]] = {
    2024: [
        (date(2024, 1, 1), "New Year's Day"),
        (date(2024, 1, 15), "Martin Luther King Jr. Day"),
        (date(2024, 2, 19), "Presidents' Day"),
        (date(2024, 5, 27), "Memorial Day"),
        (date(2024, 6, 19), "Juneteenth"),
        (date(2024, 7, 4), "Independence Day"),
        (date(2024, 9, 2), "Labor Day"),
        (date(2024, 10, 14), "Columbus Day"),
        (date(2024, 11, 11), "Veterans Day"),
        (date(2024, 11, 28), "Thanksgiving Day"),
        (date(2024, 12, 25), "Christmas Day"),
    ],
    2025: [
        (date(2025, 1, 1), "New Year's Day"),
        (date(2025, 1, 20), "Martin Luther King Jr. Day"),
        (date(2025, 2, 17), "Presidents' Day"),
        (date(2025, 5, 26), "Memorial Day"),
        (date(2025, 6, 19), "Juneteenth"),
        (date(2025, 7, 4), "Independence Day"),
        (date(2025, 9, 1), "Labor Day"),
        (date(2025, 10, 13), "Columbus Day"),
        (date(2025, 11, 11), "Veterans Day"),
        (date(2025, 11, 27), "Thanksgiving Day"),
        (date(2025, 12, 25), "Christmas Day"),
    ],
    2026: [
        (date(2026, 1, 1), "New Year's Day"),
        (date(2026, 1, 19), "Martin Luther King Jr. Day"),
        (date(2026, 2, 16), "Presidents' Day"),
        (date(2026, 5, 25), "Memorial Day"),
        (date(2026, 6, 19), "Juneteenth"),
        # Jul 4 is Saturday in 2026 — observed on Fri Jul 3; we also store
        # the actual date (Jul 4) which IS a Saturday (already non-business day).
        (date(2026, 7, 3), "Independence Day (observed)"),
        (date(2026, 9, 7), "Labor Day"),
        (date(2026, 10, 12), "Columbus Day"),
        (date(2026, 11, 11), "Veterans Day"),
        (date(2026, 11, 26), "Thanksgiving Day"),
        (date(2026, 12, 25), "Christmas Day"),
    ],
}

# ---------------------------------------------------------------------------
# Processable Date Fields
# ---------------------------------------------------------------------------
# Date fields within a transaction dictionary that can be moved to a
# non-business day.  Ordered by priority — the first match is used.
PROCESSABLE_DATE_FIELDS: List[str] = [
    "processing_date",
    "transaction_date",
    "entry_date",
    "posting_date",
    "document_date",
    "invoice_date",
    "payment_date",
    "receipt_date",
    "order_date",
    "created_date",
]


# ---------------------------------------------------------------------------
# HolidayTransaction — CTL-005
# ---------------------------------------------------------------------------
class HolidayTransaction(BaseDiscrepancy):
    """CTL-005: Transaction on Holiday/Weekend.

    Injects a holiday/weekend processing violation by moving the transaction
    date to fall on a Saturday, Sunday, or US Federal holiday.

    This is an 'easy' difficulty discrepancy because the detection method
    is straightforward: check if the processing date falls on a non-business
    day using the BusinessCalendar's ``is_business_day()`` method.

    Injection Strategy:
        1. Locate the first processable date field in the transaction.
        2. Randomly choose between weekend or holiday target (50/50).
        3. For **weekend**: find the nearest Saturday or Sunday.
        4. For **holiday**: pick a random US Federal holiday in the same year;
           fall back to weekend if no holiday data is available.
        5. Replace the date field value (preserving original format).
        6. Record ground truth with ``financial_impact = Decimal("0")``
           since this is a control/process issue, not a direct financial one.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — override BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "CTL-005"
    category: ClassVar[str] = "control"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Holiday/Weekend Transaction"
    description: ClassVar[str] = (
        "Transaction processed on a weekend or US Federal holiday "
        "when normal business operations are closed"
    )
    detection_method: ClassVar[str] = "date_check"

    # CTL-005 has no configurable parameters
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Core injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a holiday/weekend transaction discrepancy.

        Moves the first available date field in the transaction to a
        non-business day (weekend or US Federal holiday).

        Args:
            transaction: Transaction data dictionary containing at least
                one date field from :data:`PROCESSABLE_DATE_FIELDS`.
            params: Injection parameters.  CTL-005 has **no** configurable
                parameters — this argument is accepted for contract
                compliance but is effectively unused.
            rng: Seeded :class:`random.Random` instance for deterministic
                reproducibility.  **CRITICAL**: MUST use this RNG, NEVER
                ``random.random()`` or the module-level RNG.

        Returns:
            A 2-tuple of ``(modified_transaction, ground_truth_data)``:

            - **modified_transaction**: Deep copy of the original with the
              selected date field moved to a non-business day.
            - **ground_truth_data**: Dictionary suitable for
              ``GroundTruthGenerator`` consumption.

        Raises:
            DiscrepancyInjectionError: If no processable date field is found
                in the transaction dictionary.
        """
        # 1. Deep-copy transaction to preserve the original
        modified = self._copy_transaction(transaction)

        # 1b. Validate parameters (no-op for CTL-005 — no numeric params,
        # but called for pattern consistency per review feedback)
        self._validate_params(params, self.PARAMETER_BOUNDS)

        # 2. Find the first available date field
        date_field: Optional[str] = None
        raw_value: Any = None
        for field_name in PROCESSABLE_DATE_FIELDS:
            if field_name in modified and modified[field_name] is not None:
                date_field = field_name
                raw_value = modified[field_name]
                break

        if date_field is None:
            raise DiscrepancyInjectionError(
                "No date fields found in transaction for holiday injection",
                details={
                    "discrepancy_type": self.type_code,
                    "searched_fields": PROCESSABLE_DATE_FIELDS,
                    "transaction_keys": list(transaction.keys()),
                },
            )

        # 3. Parse the original date value
        original_date = self._parse_date(raw_value)
        if original_date is None:
            raise DiscrepancyInjectionError(
                f"Unable to parse date from field '{date_field}': {raw_value!r}",
                details={
                    "discrepancy_type": self.type_code,
                    "field": date_field,
                    "raw_value": str(raw_value),
                },
            )

        # 4. Decide injection type: weekend (50%) or holiday (50%)
        #    Fall back to weekend if no holiday data available.
        injection_type = rng.choice(["weekend", "holiday"])

        non_business_date: date
        day_name: str
        non_business_day_type: str

        if injection_type == "holiday":
            holiday_result = self._find_nearest_holiday(original_date, rng)
            if holiday_result is not None:
                non_business_date, day_name = holiday_result
                non_business_day_type = "holiday"
            else:
                # Fallback to weekend when no holiday data for this year
                non_business_date, day_name = self._find_nearest_weekend(
                    original_date, rng,
                )
                non_business_day_type = "weekend"
        else:
            non_business_date, day_name = self._find_nearest_weekend(
                original_date, rng,
            )
            non_business_day_type = "weekend"

        # 5. Set the modified date (preserve original format)
        modified[date_field] = self._format_date(non_business_date, raw_value)

        # 6. Financial impact is zero — control/process issue only
        financial_impact = Decimal("0")

        # 7. Build human-readable description
        description = (
            f"Transaction processed on {non_business_date.isoformat()} "
            f"({day_name}) — a non-business day"
        )

        # 8. Create ground truth data
        ground_truth = self._create_ground_truth_data(
            affected_fields=[date_field],
            original_values={date_field: str(original_date)},
            modified_values={date_field: str(non_business_date)},
            financial_impact=financial_impact,
            description=description,
            extra_metadata={
                "non_business_day_type": non_business_day_type,
                "day_name": day_name,
                "original_date": str(original_date),
                "modified_date": str(non_business_date),
            },
        )

        # 9. Log the injection event (use modified copy for consistency)
        transaction_id = str(
            modified.get("transaction_id")
            or modified.get("id")
            or "unknown"
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "date_field": date_field,
                "non_business_day_type": non_business_day_type,
                "day_name": day_name,
                "original_date": str(original_date),
                "modified_date": str(non_business_date),
            },
        )

        return modified, ground_truth

    # ------------------------------------------------------------------
    # Helper: _find_nearest_weekend
    # ------------------------------------------------------------------
    @staticmethod
    def _find_nearest_weekend(
        d: date,
        rng: random.Random,
    ) -> Tuple[date, str]:
        """Find the nearest Saturday or Sunday to a given date.

        If the reference date *d* already falls on a weekend, it is returned
        as-is.  Otherwise the method computes the distances to the preceding
        and upcoming Saturday and Sunday, then uses ``rng.choice()`` to
        pick one deterministically.

        Args:
            d: The reference date.
            rng: Seeded :class:`random.Random` for choosing Saturday vs
                Sunday when both are equidistant.

        Returns:
            A tuple of ``(weekend_date, day_name)`` where *day_name* is
            ``"Saturday"`` or ``"Sunday"``.
        """
        weekday = d.weekday()  # 0=Mon … 6=Sun

        # Already on a weekend — return it directly
        if weekday == 5:  # Saturday
            return d, "Saturday"
        if weekday == 6:  # Sunday
            return d, "Sunday"

        # Choose target: Saturday (weekday=5) or Sunday (weekday=6)
        target_day_choice = rng.choice(["Saturday", "Sunday"])
        target_weekday = 5 if target_day_choice == "Saturday" else 6

        # Calculate offset to the *next* occurrence of target_weekday
        days_forward = (target_weekday - weekday) % 7
        if days_forward == 0:
            days_forward = 7  # Should not happen since we excluded weekends above

        # Calculate offset to the *previous* occurrence
        days_backward = (weekday - target_weekday) % 7
        if days_backward == 0:
            days_backward = 7

        # Pick the nearer one; tie-break randomly
        if days_forward < days_backward:
            weekend_date = d + timedelta(days=days_forward)
        elif days_backward < days_forward:
            weekend_date = d - timedelta(days=days_backward)
        else:
            # Equidistant — pick randomly
            if rng.choice([True, False]):
                weekend_date = d + timedelta(days=days_forward)
            else:
                weekend_date = d - timedelta(days=days_backward)

        return weekend_date, target_day_choice

    # ------------------------------------------------------------------
    # Helper: _find_nearest_holiday
    # ------------------------------------------------------------------
    @staticmethod
    def _find_nearest_holiday(
        d: date,
        rng: random.Random,
    ) -> Optional[Tuple[date, str]]:
        """Find a US Federal holiday for the same year as *d*.

        Looks up :data:`US_FEDERAL_HOLIDAYS` for ``d.year``.  If holiday
        data is available, picks one at random using ``rng.choice()``.

        Args:
            d: The reference date whose year determines the holiday pool.
            rng: Seeded :class:`random.Random` for selecting among
                candidate holidays.

        Returns:
            A tuple of ``(holiday_date, holiday_name)`` chosen from the
            pool, or ``None`` if no holiday data exists for the year.
        """
        year_holidays = US_FEDERAL_HOLIDAYS.get(d.year)
        if not year_holidays:
            return None

        # Pick a random holiday from the year's pool
        chosen_date, chosen_name = rng.choice(year_holidays)
        return chosen_date, chosen_name

    # ------------------------------------------------------------------
    # Helper: _parse_date
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_date(value: Any) -> Optional[date]:
        """Parse a date from various formats.

        Handles:
        - ``date`` objects — returned directly.
        - ISO-8601 strings (``"YYYY-MM-DD"``) — parsed via
          :meth:`date.fromisoformat`.
        - ``datetime`` objects — the ``.date()`` component is extracted.
        - Other string representations — best-effort ISO parse.

        Args:
            value: The raw value from the transaction dictionary.

        Returns:
            A :class:`date` object, or ``None`` if parsing fails.
        """
        if isinstance(value, date):
            return value

        # Handle datetime objects (date is a superclass of datetime in
        # Python's type hierarchy, but datetime instances also match the
        # date isinstance check above; this handles any edge cases)
        if hasattr(value, "date") and callable(value.date):
            try:
                return value.date()
            except Exception:
                pass

        if isinstance(value, str):
            # Try ISO-8601 format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
            try:
                # Handle datetime strings by taking only the date portion
                date_str = value[:10] if len(value) >= 10 else value
                return date.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass

        return None

    # ------------------------------------------------------------------
    # Helper: _format_date
    # ------------------------------------------------------------------
    @staticmethod
    def _format_date(d: date, original_format: Any) -> Any:
        """Format a date to match the original field's format.

        Preserves the original value's type:
        - If the original was a ``str``, returns ISO-8601 string
          (``"YYYY-MM-DD"``).
        - If the original was a ``date`` object, returns a ``date`` object.
        - For datetime-like originals, returns ISO-8601 string since we
          only modify the date component.

        Args:
            d: The new date to format.
            original_format: The original value from the transaction,
                used solely to determine the output type.

        Returns:
            The formatted date in a type matching *original_format*.
        """
        if isinstance(original_format, str):
            return d.isoformat()
        if isinstance(original_format, date):
            return d
        # Default: ISO string for any other type
        return d.isoformat()
