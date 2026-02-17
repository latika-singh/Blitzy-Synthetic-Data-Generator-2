"""Business calendar with US Federal holidays (2024-2026), company holidays, half days,
working hours (8:00-17:00), and lunch break (12:00-13:00).

Provides business day queries and date arithmetic for simulation time management.
This is a foundational module with NO internal dependencies — it relies only on
the Python standard library, the ``holidays`` package, ``structlog``, and ``pydantic``.

Key performance contract:
    * ``is_business_day`` uses O(1) set lookups — no linear scans.
    * All calendar operations target < 10 ms latency.

Working hours model (per README.md lines 426-428):
    * Standard start : 08:00
    * Standard end   : 17:00
    * Lunch break    : 12:00 – 13:00
    * Effective hours : 8.0 h per full business day (9 h − 1 h lunch)
"""

from __future__ import annotations

import calendar as _calendar_mod
from datetime import date, time, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Holidays library — imported with graceful fallback
# ---------------------------------------------------------------------------
try:
    import holidays as holidays_lib

    HAS_HOLIDAYS_LIB = True
except ImportError:  # pragma: no cover
    HAS_HOLIDAYS_LIB = False

# ---------------------------------------------------------------------------
# Module-level logger (structlog to stdout, per AAP §0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Working-hours constants (per README.md lines 426-428)
# ---------------------------------------------------------------------------
STANDARD_START: time = time(8, 0)   # 8:00 AM
STANDARD_END: time = time(17, 0)    # 5:00 PM
LUNCH_START: time = time(12, 0)     # 12:00 PM
LUNCH_END: time = time(13, 0)       # 1:00 PM
EFFECTIVE_HOURS_PER_DAY: float = 8.0  # 9 hours − 1 hour lunch = 8 effective hours


# ---------------------------------------------------------------------------
# Helper: compute the *n*-th weekday of a given month/year
# ---------------------------------------------------------------------------
def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* (0=Mon … 6=Sun) in *month*/*year*.

    ``n`` is 1-based (1 = first, 2 = second, …).  A negative ``n`` counts from
    the end of the month (−1 = last).
    """
    if n > 0:
        # First day of the month
        first_day = date(year, month, 1)
        # Days until the target weekday
        days_ahead = (weekday - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=days_ahead)
        return first_occurrence + timedelta(weeks=n - 1)
    else:
        # Last day of the month
        _, last_day_num = _calendar_mod.monthrange(year, month)
        last_day = date(year, month, last_day_num)
        days_back = (last_day.weekday() - weekday) % 7
        last_occurrence = last_day - timedelta(days=days_back)
        return last_occurrence + timedelta(weeks=n + 1)  # n is negative; +1 gives "last"


def _observed_date(holiday_date: date) -> date:
    """Apply the US *observed* rule: Sat → Fri, Sun → Mon."""
    wd = holiday_date.weekday()
    if wd == 5:  # Saturday
        return holiday_date - timedelta(days=1)
    if wd == 6:  # Sunday
        return holiday_date + timedelta(days=1)
    return holiday_date


# ---------------------------------------------------------------------------
# BusinessCalendar
# ---------------------------------------------------------------------------
class BusinessCalendar:
    """Business calendar managing US Federal holidays, company holidays, half days,
    working hours, and business-day arithmetic.

    Attributes:
        federal_holidays:  ``Set[date]`` of US Federal holidays loaded from the
            ``holidays`` library (or computed via fallback for 2024-2026).
        company_holidays:  ``Set[date]`` of custom company-defined holidays.
        half_days:         ``Set[date]`` of half-working-day dates.
        standard_start:    Start of working hours (default 08:00).
        standard_end:      End of working hours (default 17:00).
        lunch_break:       Tuple of (start, end) times for lunch (12:00-13:00).
        country:           Country code used for holiday generation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        country: str = "US",
        start_year: int = 2024,
        end_year: int = 2026,
        standard_start: Optional[time] = None,
        standard_end: Optional[time] = None,
    ) -> None:
        self.country: str = country
        self.standard_start: time = standard_start or STANDARD_START
        self.standard_end: time = standard_end or STANDARD_END
        self.lunch_break: Tuple[time, time] = (LUNCH_START, LUNCH_END)

        # Holiday and half-day sets (O(1) lookup)
        self.federal_holidays: Set[date] = set()
        self.company_holidays: Set[date] = set()
        self.half_days: Set[date] = set()

        # Human-readable names keyed by date
        self._holiday_names: Dict[date, str] = {}

        # Half-day end-times (default: noon)
        self._half_day_end_times: Dict[date, time] = {}

        # Populate federal holidays
        self.load_holidays(country, start_year, end_year)

        logger.info(
            "business_calendar_initialized",
            country=self.country,
            holiday_count=len(self.federal_holidays),
            start_year=start_year,
            end_year=end_year,
        )

    # ------------------------------------------------------------------
    # Holiday loading
    # ------------------------------------------------------------------
    def load_holidays(
        self,
        country: str = "US",
        start_year: int = 2024,
        end_year: int = 2026,
    ) -> None:
        """Load federal holidays for *country* spanning [start_year, end_year].

        Uses the ``holidays`` library when available; otherwise falls back to a
        manual computation covering the 8 core US Federal holidays listed in
        the specification (New Year's Day, MLK Day, Presidents' Day, Memorial Day,
        Independence Day, Labor Day, Thanksgiving, Christmas) plus observed-date
        handling.
        """
        if HAS_HOLIDAYS_LIB:
            us_holidays = holidays_lib.US(years=range(start_year, end_year + 1))
            for hol_date, hol_name in us_holidays.items():
                self.federal_holidays.add(hol_date)
                self._holiday_names[hol_date] = hol_name
        else:
            self._load_fallback_holidays(start_year, end_year)

        logger.info(
            "holidays_loaded",
            count=len(self.federal_holidays),
            source="holidays_lib" if HAS_HOLIDAYS_LIB else "fallback",
        )

    def _load_fallback_holidays(self, start_year: int, end_year: int) -> None:
        """Manually compute the 8 US Federal holidays for each year in range.

        Handles the *observed* rule (Sat → Fri, Sun → Mon).
        """
        for year in range(start_year, end_year + 1):
            holidays_for_year: List[Tuple[date, str]] = [
                (date(year, 1, 1), "New Year's Day"),
                (_nth_weekday_of_month(year, 1, 0, 3), "Martin Luther King Jr. Day"),
                (_nth_weekday_of_month(year, 2, 0, 3), "Presidents' Day"),
                (_nth_weekday_of_month(year, 5, 0, -1), "Memorial Day"),
                (date(year, 7, 4), "Independence Day"),
                (_nth_weekday_of_month(year, 9, 0, 1), "Labor Day"),
                (_nth_weekday_of_month(year, 11, 3, 4), "Thanksgiving Day"),
                (date(year, 12, 25), "Christmas Day"),
            ]

            for hol_date, hol_name in holidays_for_year:
                observed = _observed_date(hol_date)
                self.federal_holidays.add(hol_date)
                self._holiday_names[hol_date] = hol_name
                if observed != hol_date:
                    self.federal_holidays.add(observed)
                    self._holiday_names[observed] = f"{hol_name} (observed)"

    # ------------------------------------------------------------------
    # Business-day queries  (README.md lines 387-391)
    # ------------------------------------------------------------------
    def is_business_day(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is a working business day.

        A date is **not** a business day if it falls on a weekend (Sat/Sun),
        is a federal holiday, or is a company-defined holiday.

        Performance: O(1) via set membership tests.
        """
        if check_date.weekday() >= 5:
            return False
        if check_date in self.federal_holidays:
            return False
        if check_date in self.company_holidays:
            return False
        return True

    def get_next_business_day(self, from_date: date) -> date:
        """Return the first business day strictly **after** *from_date*."""
        next_date = from_date + timedelta(days=1)
        while not self.is_business_day(next_date):
            next_date += timedelta(days=1)
        return next_date

    def get_previous_business_day(self, from_date: date) -> date:
        """Return the first business day strictly **before** *from_date*."""
        prev_date = from_date - timedelta(days=1)
        while not self.is_business_day(prev_date):
            prev_date -= timedelta(days=1)
        return prev_date

    def count_business_days(self, start: date, end: date) -> int:
        """Count business days in [start, end) — inclusive of *start*, exclusive of *end*.

        If *end* <= *start* the result is 0.
        """
        if end <= start:
            return 0
        count = 0
        current = start
        while current < end:
            if self.is_business_day(current):
                count += 1
            current += timedelta(days=1)
        return count

    def add_business_days(self, from_date: date, days: int) -> date:
        """Add (or subtract) *days* business days to *from_date*.

        * ``days > 0`` → advance forward, skipping non-business days.
        * ``days < 0`` → go backward, skipping non-business days.
        * ``days == 0`` → return *from_date* itself (even if it is not a
          business day; callers can normalise separately).
        """
        if days == 0:
            return from_date

        step = 1 if days > 0 else -1
        remaining = abs(days)
        current = from_date

        while remaining > 0:
            current += timedelta(days=step)
            if self.is_business_day(current):
                remaining -= 1
        return current

    # ------------------------------------------------------------------
    # Holiday management (README.md lines 401-402)
    # ------------------------------------------------------------------
    def add_company_holiday(self, holiday_date: date, name: str = "") -> None:
        """Register a custom company holiday on *holiday_date*.

        Company holidays are treated identically to federal holidays when
        evaluating ``is_business_day``.
        """
        self.company_holidays.add(holiday_date)
        if name:
            self._holiday_names[holiday_date] = name
        logger.debug(
            "company_holiday_added",
            date=holiday_date.isoformat(),
            name=name,
        )

    def remove_company_holiday(self, holiday_date: date) -> bool:
        """Remove a previously added company holiday.

        Returns ``True`` if the holiday was present and removed, ``False``
        otherwise.
        """
        if holiday_date in self.company_holidays:
            self.company_holidays.discard(holiday_date)
            # Remove the name only if it is not also a federal holiday
            if holiday_date not in self.federal_holidays:
                self._holiday_names.pop(holiday_date, None)
            return True
        return False

    def add_half_day(self, half_day_date: date, end_time: time = time(12, 0)) -> None:
        """Mark *half_day_date* as a half working day ending at *end_time*.

        Half days are still considered business days but report reduced working
        hours via ``get_working_hours``.
        """
        self.half_days.add(half_day_date)
        self._half_day_end_times[half_day_date] = end_time
        logger.debug(
            "half_day_added",
            date=half_day_date.isoformat(),
            end_time=end_time.isoformat(),
        )

    def is_holiday(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is any kind of holiday (federal or company)."""
        return check_date in self.federal_holidays or check_date in self.company_holidays

    def is_half_day(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* has been marked as a half day."""
        return check_date in self.half_days

    def get_holiday_name(self, check_date: date) -> Optional[str]:
        """Return the human-readable name for the holiday on *check_date*, or ``None``."""
        return self._holiday_names.get(check_date)

    # ------------------------------------------------------------------
    # Working-hours queries (per README.md lines 425-428)
    # ------------------------------------------------------------------
    def is_within_working_hours(self, check_time: time) -> bool:
        """Return ``True`` if *check_time* is within standard working hours
        **and** outside the lunch break window.

        Working hours: ``standard_start`` (08:00) – ``standard_end`` (17:00).
        Lunch break:   ``LUNCH_START`` (12:00) – ``LUNCH_END`` (13:00).
        """
        if check_time < self.standard_start or check_time >= self.standard_end:
            return False
        lunch_start, lunch_end = self.lunch_break
        if lunch_start <= check_time < lunch_end:
            return False
        return True

    def get_working_hours(self, check_date: date) -> float:
        """Return effective working hours for *check_date*.

        * Non-business day → ``0.0``
        * Half day → hours from ``standard_start`` to the half-day end time,
          minus any lunch overlap.
        * Full business day → ``EFFECTIVE_HOURS_PER_DAY`` (8.0).
        """
        if not self.is_business_day(check_date):
            return 0.0

        if check_date in self.half_days:
            half_end = self._half_day_end_times.get(check_date, time(12, 0))
            return self._calculate_effective_hours(self.standard_start, half_end)

        return EFFECTIVE_HOURS_PER_DAY

    def get_working_hours_in_range(self, start: date, end: date) -> float:
        """Return total effective working hours for all business days in [start, end).

        Inclusive of *start*, exclusive of *end*.
        """
        if end <= start:
            return 0.0
        total = 0.0
        current = start
        while current < end:
            total += self.get_working_hours(current)
            current += timedelta(days=1)
        return total

    # ------------------------------------------------------------------
    # Metrics / state introspection
    # ------------------------------------------------------------------
    def get_calendar_state(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of calendar configuration."""
        return {
            "country": self.country,
            "federal_holiday_count": len(self.federal_holidays),
            "company_holiday_count": len(self.company_holidays),
            "half_day_count": len(self.half_days),
            "standard_start": self.standard_start.isoformat(),
            "standard_end": self.standard_end.isoformat(),
            "lunch_break": (
                self.lunch_break[0].isoformat(),
                self.lunch_break[1].isoformat(),
            ),
        }

    def get_holidays_for_year(self, year: int) -> List[Dict[str, Any]]:
        """Return a list of holiday entries for *year*.

        Each entry is a dict with keys ``date``, ``name``, and ``type``
        (``"federal"`` or ``"company"``).
        """
        result: List[Dict[str, Any]] = []

        for hol_date in sorted(self.federal_holidays):
            if hol_date.year == year:
                result.append(
                    {
                        "date": hol_date.isoformat(),
                        "name": self._holiday_names.get(hol_date, "Federal Holiday"),
                        "type": "federal",
                    }
                )

        for hol_date in sorted(self.company_holidays):
            if hol_date.year == year:
                result.append(
                    {
                        "date": hol_date.isoformat(),
                        "name": self._holiday_names.get(hol_date, "Company Holiday"),
                        "type": "company",
                    }
                )

        return result

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate metrics about loaded holidays."""
        return {
            "total_holidays": len(self.federal_holidays) + len(self.company_holidays),
            "federal_count": len(self.federal_holidays),
            "company_count": len(self.company_holidays),
            "half_day_count": len(self.half_days),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _time_to_hours(t: time) -> float:
        """Convert a ``time`` object to fractional hours since midnight."""
        return t.hour + t.minute / 60.0 + t.second / 3600.0

    def _calculate_effective_hours(self, start_t: time, end_t: time) -> float:
        """Compute effective working hours between two times, subtracting any
        lunch-break overlap.
        """
        start_h = self._time_to_hours(start_t)
        end_h = self._time_to_hours(end_t)

        if end_h <= start_h:
            return 0.0

        total = end_h - start_h

        # Subtract lunch overlap
        lunch_start_h = self._time_to_hours(self.lunch_break[0])
        lunch_end_h = self._time_to_hours(self.lunch_break[1])

        overlap_start = max(start_h, lunch_start_h)
        overlap_end = min(end_h, lunch_end_h)
        if overlap_end > overlap_start:
            total -= overlap_end - overlap_start

        return max(total, 0.0)
