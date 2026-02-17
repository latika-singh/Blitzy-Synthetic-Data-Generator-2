"""Comprehensive tests for TimeController, BusinessCalendar, and FiscalCalendar.

Coverage targets:
    - BusinessCalendar: US Federal holidays 2024–2026, working hours 8AM–5PM,
      lunch break 12:00–13:00, company holidays, half days, business-day arithmetic.
    - FiscalCalendar: Configurable fiscal year start (Jan, Jul), monthly/quarterly
      periods, period lifecycle open→closing→closed, quarter-end and year-end flags.
    - TimeController: Day advancement skipping weekends/holidays, advance_by_hours,
      fiscal period queries, business day queries, event publishing.

Performance SLA validation:
    - Calendar operations < 10 ms (README.md line 46).
    - Business day advancement < 30 seconds (README.md line 45).

Rules (AAP §0.7.5):
    - All external dependencies mocked where applicable.
    - Async tests use pytest-asyncio (asyncio_mode=auto in pytest.ini).
    - No live API calls, no external Redis.
    - Unit test coverage target ≥ 80%.
"""

from __future__ import annotations

import asyncio
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.orchestration.business_calendar import BusinessCalendar
from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod, PeriodStatus
from app.orchestration.time_controller import TimeController


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def business_calendar() -> BusinessCalendar:
    """Default BusinessCalendar with US Federal holidays for 2024–2026."""
    return BusinessCalendar(country="US", start_year=2024, end_year=2026)


@pytest.fixture
def fiscal_calendar() -> FiscalCalendar:
    """FiscalCalendar with standard January fiscal year start, monthly periods."""
    return FiscalCalendar(fiscal_year_start_month=1, period_type="monthly", start_year=2024, num_years=3)


@pytest.fixture
def fiscal_calendar_july() -> FiscalCalendar:
    """FiscalCalendar with July fiscal year start (common for government).

    With July start and start_year=2024:
        - FY2025: Jul 2024 → Jun 2025
        - FY2026: Jul 2025 → Jun 2026
        - FY2027: Jul 2026 → Jun 2027
    """
    return FiscalCalendar(fiscal_year_start_month=7, period_type="monthly", start_year=2024, num_years=3)


@pytest.fixture
def fiscal_calendar_quarterly() -> FiscalCalendar:
    """FiscalCalendar with January start and quarterly periods (4 per year)."""
    return FiscalCalendar(fiscal_year_start_month=1, period_type="quarterly", start_year=2024, num_years=3)


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Mock EventBus with async publish method for verifying event publication."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def time_controller(
    business_calendar: BusinessCalendar,
    fiscal_calendar: FiscalCalendar,
    mock_event_bus: AsyncMock,
) -> TimeController:
    """TimeController starting on the first business day of 2024 (Jan 2, since Jan 1 is a holiday)."""
    return TimeController(
        start_date=date(2024, 1, 2),
        business_calendar=business_calendar,
        fiscal_calendar=fiscal_calendar,
        event_bus=mock_event_bus,
    )


# ============================================================================
# TestBusinessCalendarHolidays — US Federal holiday verification 2024-2026
# ============================================================================


class TestBusinessCalendarHolidays:
    """Verify that the BusinessCalendar correctly identifies US Federal holidays.

    The holidays library provides accurate dates for the 8 core US Federal
    holidays specified in README.md lines 415-428.
    """

    # --- 2024 US Federal Holidays ---

    def test_new_years_day_2024(self, business_calendar: BusinessCalendar) -> None:
        """New Year's Day: January 1, 2024."""
        assert business_calendar.is_holiday(date(2024, 1, 1)) is True

    def test_mlk_day_2024(self, business_calendar: BusinessCalendar) -> None:
        """Martin Luther King Jr. Day: 3rd Monday of January 2024 = January 15."""
        assert business_calendar.is_holiday(date(2024, 1, 15)) is True

    def test_presidents_day_2024(self, business_calendar: BusinessCalendar) -> None:
        """Presidents' Day: 3rd Monday of February 2024 = February 19."""
        assert business_calendar.is_holiday(date(2024, 2, 19)) is True

    def test_memorial_day_2024(self, business_calendar: BusinessCalendar) -> None:
        """Memorial Day: Last Monday of May 2024 = May 27."""
        assert business_calendar.is_holiday(date(2024, 5, 27)) is True

    def test_independence_day_2024(self, business_calendar: BusinessCalendar) -> None:
        """Independence Day: July 4, 2024 (Thursday)."""
        assert business_calendar.is_holiday(date(2024, 7, 4)) is True

    def test_labor_day_2024(self, business_calendar: BusinessCalendar) -> None:
        """Labor Day: 1st Monday of September 2024 = September 2."""
        assert business_calendar.is_holiday(date(2024, 9, 2)) is True

    def test_thanksgiving_2024(self, business_calendar: BusinessCalendar) -> None:
        """Thanksgiving: 4th Thursday of November 2024 = November 28."""
        assert business_calendar.is_holiday(date(2024, 11, 28)) is True

    def test_christmas_2024(self, business_calendar: BusinessCalendar) -> None:
        """Christmas Day: December 25, 2024 (Wednesday)."""
        assert business_calendar.is_holiday(date(2024, 12, 25)) is True

    # --- 2025 Spot-checks ---

    def test_new_years_day_2025(self, business_calendar: BusinessCalendar) -> None:
        """New Year's Day: January 1, 2025 (Wednesday)."""
        assert business_calendar.is_holiday(date(2025, 1, 1)) is True

    def test_independence_day_2025(self, business_calendar: BusinessCalendar) -> None:
        """Independence Day: July 4, 2025 (Friday)."""
        assert business_calendar.is_holiday(date(2025, 7, 4)) is True

    def test_christmas_2025(self, business_calendar: BusinessCalendar) -> None:
        """Christmas Day: December 25, 2025 (Thursday)."""
        assert business_calendar.is_holiday(date(2025, 12, 25)) is True

    # --- 2026 Spot-check ---

    def test_new_years_day_2026(self, business_calendar: BusinessCalendar) -> None:
        """New Year's Day: January 1, 2026 (Thursday)."""
        assert business_calendar.is_holiday(date(2026, 1, 1)) is True

    # --- Non-holidays ---

    def test_regular_weekday_not_holiday(self, business_calendar: BusinessCalendar) -> None:
        """A regular Friday (March 15, 2024) is NOT a federal holiday."""
        assert business_calendar.is_holiday(date(2024, 3, 15)) is False

    def test_weekend_not_federal_holiday(self, business_calendar: BusinessCalendar) -> None:
        """Saturday March 16, 2024 is a weekend but NOT a federal holiday."""
        # is_holiday checks the holiday SET, not weekend status
        assert business_calendar.is_holiday(date(2024, 3, 16)) is False


# ============================================================================
# TestBusinessDayLogic — business day identification and arithmetic
# ============================================================================


class TestBusinessDayLogic:
    """Verify business day identification, next/previous lookups, counting,
    and date arithmetic."""

    # --- is_business_day ---

    def test_weekday_monday_is_business_day(self, business_calendar: BusinessCalendar) -> None:
        """Monday March 11, 2024 is a business day."""
        assert business_calendar.is_business_day(date(2024, 3, 11)) is True

    def test_weekday_friday_is_business_day(self, business_calendar: BusinessCalendar) -> None:
        """Friday March 15, 2024 is a business day."""
        assert business_calendar.is_business_day(date(2024, 3, 15)) is True

    def test_saturday_not_business_day(self, business_calendar: BusinessCalendar) -> None:
        """Saturday March 16, 2024 is NOT a business day."""
        assert business_calendar.is_business_day(date(2024, 3, 16)) is False

    def test_sunday_not_business_day(self, business_calendar: BusinessCalendar) -> None:
        """Sunday March 17, 2024 is NOT a business day."""
        assert business_calendar.is_business_day(date(2024, 3, 17)) is False

    def test_federal_holiday_not_business_day(self, business_calendar: BusinessCalendar) -> None:
        """Independence Day (July 4, 2024) is NOT a business day."""
        assert business_calendar.is_business_day(date(2024, 7, 4)) is False

    # --- get_next_business_day ---

    def test_get_next_business_day_from_friday(self, business_calendar: BusinessCalendar) -> None:
        """Next business day after Friday March 15 is Monday March 18."""
        assert business_calendar.get_next_business_day(date(2024, 3, 15)) == date(2024, 3, 18)

    def test_get_next_business_day_from_holiday(self, business_calendar: BusinessCalendar) -> None:
        """Next business day after July 4, 2024 (Thursday, holiday) is July 5 (Friday)."""
        assert business_calendar.get_next_business_day(date(2024, 7, 4)) == date(2024, 7, 5)

    def test_get_next_business_day_from_regular_day(self, business_calendar: BusinessCalendar) -> None:
        """Next business day after Monday March 11 is Tuesday March 12."""
        assert business_calendar.get_next_business_day(date(2024, 3, 11)) == date(2024, 3, 12)

    # --- get_previous_business_day ---

    def test_get_previous_business_day_from_monday(self, business_calendar: BusinessCalendar) -> None:
        """Previous business day before Monday March 18 is Friday March 15."""
        assert business_calendar.get_previous_business_day(date(2024, 3, 18)) == date(2024, 3, 15)

    def test_get_previous_business_day_from_day_after_holiday(self, business_calendar: BusinessCalendar) -> None:
        """Previous business day before July 5, 2024 is July 3 (Wed, skipping Jul 4 holiday)."""
        assert business_calendar.get_previous_business_day(date(2024, 7, 5)) == date(2024, 7, 3)

    # --- count_business_days (start inclusive, end exclusive per implementation) ---

    def test_count_business_days_full_week(self, business_calendar: BusinessCalendar) -> None:
        """Count business days Mon-Fri: [Mar 11, Mar 16) = 5 business days."""
        # Mar 11 Mon, Mar 12 Tue, Mar 13 Wed, Mar 14 Thu, Mar 15 Fri = 5 biz days
        count = business_calendar.count_business_days(date(2024, 3, 11), date(2024, 3, 16))
        assert count == 5

    def test_count_business_days_with_holiday(self, business_calendar: BusinessCalendar) -> None:
        """Count business days July 1-5 (exclusive end Jul 6): Jul 1,2,3 = biz, Jul 4 holiday, Jul 5 biz = 4."""
        count = business_calendar.count_business_days(date(2024, 7, 1), date(2024, 7, 6))
        assert count == 4

    def test_count_business_days_same_date_returns_zero(self, business_calendar: BusinessCalendar) -> None:
        """Counting from a date to itself returns 0."""
        count = business_calendar.count_business_days(date(2024, 3, 11), date(2024, 3, 11))
        assert count == 0

    # --- add_business_days ---

    def test_add_business_days_simple(self, business_calendar: BusinessCalendar) -> None:
        """Adding 5 business days from Mon Mar 11 → Mon Mar 18 (skips weekend)."""
        result = business_calendar.add_business_days(date(2024, 3, 11), 5)
        assert result == date(2024, 3, 18)

    def test_add_business_days_skips_weekend_and_holiday(self, business_calendar: BusinessCalendar) -> None:
        """Adding 5 biz days from Jul 1: Jul 2(+1), Jul 3(+2), Jul 5(+3, skip Jul 4), Jul 8(+4), Jul 9(+5)."""
        result = business_calendar.add_business_days(date(2024, 7, 1), 5)
        assert result == date(2024, 7, 9)

    def test_add_business_days_zero(self, business_calendar: BusinessCalendar) -> None:
        """Adding 0 business days returns the same date."""
        result = business_calendar.add_business_days(date(2024, 3, 11), 0)
        assert result == date(2024, 3, 11)


# ============================================================================
# TestWorkingHours — working hours boundaries and lunch break
# ============================================================================


class TestWorkingHours:
    """Verify working hours per README.md lines 426-428:
    standard_start=08:00, standard_end=17:00, lunch_break=(12:00, 13:00)."""

    def test_standard_start_is_8am(self, business_calendar: BusinessCalendar) -> None:
        """Standard working day start is 8:00 AM."""
        assert business_calendar.standard_start == time(8, 0)

    def test_standard_end_is_5pm(self, business_calendar: BusinessCalendar) -> None:
        """Standard working day end is 5:00 PM."""
        assert business_calendar.standard_end == time(17, 0)

    def test_lunch_break_is_noon_to_1pm(self, business_calendar: BusinessCalendar) -> None:
        """Lunch break is from 12:00 PM to 1:00 PM."""
        assert business_calendar.lunch_break == (time(12, 0), time(13, 0))

    def test_is_working_hour_at_9am(self, business_calendar: BusinessCalendar) -> None:
        """9:00 AM is within working hours."""
        assert business_calendar.is_within_working_hours(time(9, 0)) is True

    def test_is_working_hour_at_7am(self, business_calendar: BusinessCalendar) -> None:
        """7:00 AM is before working hours."""
        assert business_calendar.is_within_working_hours(time(7, 0)) is False

    def test_is_working_hour_at_6pm(self, business_calendar: BusinessCalendar) -> None:
        """6:00 PM is after working hours."""
        assert business_calendar.is_within_working_hours(time(18, 0)) is False

    def test_is_working_hour_during_lunch(self, business_calendar: BusinessCalendar) -> None:
        """12:30 PM is during lunch break — NOT within working hours."""
        assert business_calendar.is_within_working_hours(time(12, 30)) is False

    def test_is_working_hour_at_noon_is_lunch(self, business_calendar: BusinessCalendar) -> None:
        """12:00 PM (noon) is the start of lunch — NOT within working hours."""
        assert business_calendar.is_within_working_hours(time(12, 0)) is False

    def test_is_working_hour_at_boundary_8am(self, business_calendar: BusinessCalendar) -> None:
        """8:00 AM (exactly at start) IS within working hours (inclusive start)."""
        assert business_calendar.is_within_working_hours(time(8, 0)) is True

    def test_is_working_hour_at_boundary_5pm(self, business_calendar: BusinessCalendar) -> None:
        """5:00 PM (exactly at end) is NOT within working hours (exclusive end)."""
        assert business_calendar.is_within_working_hours(time(17, 0)) is False

    def test_is_working_hour_at_1pm_after_lunch(self, business_calendar: BusinessCalendar) -> None:
        """1:00 PM (end of lunch) IS within working hours (inclusive)."""
        assert business_calendar.is_within_working_hours(time(13, 0)) is True

    def test_is_working_hour_at_11_59am(self, business_calendar: BusinessCalendar) -> None:
        """11:59 AM is just before lunch — within working hours."""
        assert business_calendar.is_within_working_hours(time(11, 59)) is True


# ============================================================================
# TestCustomHolidaysAndHalfDays — company-specific calendar management
# ============================================================================


class TestCustomHolidaysAndHalfDays:
    """Verify adding custom company holidays and half days to the calendar."""

    def test_add_company_holiday(self, business_calendar: BusinessCalendar) -> None:
        """Adding a company holiday makes the date a non-business day."""
        business_calendar.add_company_holiday(date(2024, 12, 31), "New Year's Eve")
        assert business_calendar.is_business_day(date(2024, 12, 31)) is False

    def test_add_half_day(self, business_calendar: BusinessCalendar) -> None:
        """Adding a half day registers the date in the half_days set."""
        business_calendar.add_half_day(date(2024, 12, 24))
        assert date(2024, 12, 24) in business_calendar.half_days

    def test_half_day_is_still_business_day(self, business_calendar: BusinessCalendar) -> None:
        """Half days are still considered business days."""
        business_calendar.add_half_day(date(2024, 12, 24))
        assert business_calendar.is_business_day(date(2024, 12, 24)) is True

    def test_company_holidays_tracked_separately(self, business_calendar: BusinessCalendar) -> None:
        """Company holidays are stored in a separate set from federal holidays."""
        business_calendar.add_company_holiday(date(2024, 8, 1), "Company Summer Day")
        assert date(2024, 8, 1) in business_calendar.company_holidays
        assert date(2024, 8, 1) not in business_calendar.federal_holidays

    def test_company_holiday_makes_date_holiday(self, business_calendar: BusinessCalendar) -> None:
        """is_holiday returns True for company holidays."""
        business_calendar.add_company_holiday(date(2024, 8, 1), "Company Day")
        assert business_calendar.is_holiday(date(2024, 8, 1)) is True


# ============================================================================
# TestFiscalCalendarPeriods — period structure for Jan/Jul/Quarterly calendars
# ============================================================================


class TestFiscalCalendarPeriods:
    """Verify FiscalCalendar generates correct period structures.

    Covers January start (standard), July start (government), and quarterly
    configurations per README.md lines 433-454.
    """

    # --- January start (monthly, standard) ---

    def test_january_start_has_12_monthly_periods_per_year(self, fiscal_calendar: FiscalCalendar) -> None:
        """January-start monthly calendar has 12 periods per fiscal year (36 total for 3 years)."""
        assert len(fiscal_calendar.periods) == 36

    def test_january_start_first_period_starts_jan_1(self, fiscal_calendar: FiscalCalendar) -> None:
        """First period of FY2024 starts January 1, 2024."""
        p1 = fiscal_calendar.periods[0]
        assert p1.start_date == date(2024, 1, 1)
        assert p1.period_number == 1

    def test_january_start_last_period_ends_dec_31(self, fiscal_calendar: FiscalCalendar) -> None:
        """Period 12 of FY2024 ends December 31, 2024."""
        p12 = fiscal_calendar.periods[11]
        assert p12.end_date == date(2024, 12, 31)
        assert p12.period_number == 12

    def test_january_start_quarter_end_flags(self, fiscal_calendar: FiscalCalendar) -> None:
        """Periods 3,6,9,12 have is_quarter_end=True for Jan start."""
        # FY2024 periods (index 0-11)
        assert fiscal_calendar.periods[2].is_quarter_end is True   # Period 3 (March)
        assert fiscal_calendar.periods[5].is_quarter_end is True   # Period 6 (June)
        assert fiscal_calendar.periods[8].is_quarter_end is True   # Period 9 (September)
        assert fiscal_calendar.periods[11].is_quarter_end is True  # Period 12 (December)

    def test_january_start_non_quarter_end(self, fiscal_calendar: FiscalCalendar) -> None:
        """Periods 1,2,4,5,7,8,10,11 do NOT have quarter-end flag."""
        assert fiscal_calendar.periods[0].is_quarter_end is False  # Period 1 (January)
        assert fiscal_calendar.periods[1].is_quarter_end is False  # Period 2 (February)
        assert fiscal_calendar.periods[3].is_quarter_end is False  # Period 4 (April)

    def test_january_start_year_end_flag(self, fiscal_calendar: FiscalCalendar) -> None:
        """Only period 12 (December) of each FY has is_year_end=True."""
        # FY2024 periods (index 0-11)
        for i in range(11):
            assert fiscal_calendar.periods[i].is_year_end is False, f"Period {i + 1} should not be year-end"
        assert fiscal_calendar.periods[11].is_year_end is True

    # --- July start (monthly, government fiscal year) ---

    def test_july_start_has_36_periods(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """July-start monthly calendar has 36 periods (3 years × 12)."""
        assert len(fiscal_calendar_july.periods) == 36

    def test_july_start_first_period_starts_july(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """First period starts in July."""
        p1 = fiscal_calendar_july.periods[0]
        assert p1.start_date.month == 7
        assert p1.start_date.year == 2024

    def test_july_start_last_period_of_first_fy_ends_june(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """Period 12 of the first FY ends in June of the following year."""
        p12 = fiscal_calendar_july.periods[11]
        assert p12.end_date.month == 6
        assert p12.end_date.year == 2025

    def test_july_start_quarter_end_months(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """Quarter ends for July-start: Sep(p3), Dec(p6), Mar(p9), Jun(p12)."""
        # First FY (index 0-11)
        assert fiscal_calendar_july.periods[2].is_quarter_end is True   # P3 (Sep)
        assert fiscal_calendar_july.periods[5].is_quarter_end is True   # P6 (Dec)
        assert fiscal_calendar_july.periods[8].is_quarter_end is True   # P9 (Mar)
        assert fiscal_calendar_july.periods[11].is_quarter_end is True  # P12 (Jun)

    # --- Quarterly periods ---

    def test_quarterly_periods_has_4_per_year(self, fiscal_calendar_quarterly: FiscalCalendar) -> None:
        """Quarterly calendar has 4 periods per year (12 total for 3 years)."""
        assert len(fiscal_calendar_quarterly.periods) == 12

    def test_quarterly_period_1_spans_jan_to_mar(self, fiscal_calendar_quarterly: FiscalCalendar) -> None:
        """First quarterly period spans January through March."""
        p1 = fiscal_calendar_quarterly.periods[0]
        assert p1.start_date.month == 1
        assert p1.end_date.month == 3
        assert p1.end_date.day == 31

    def test_quarterly_period_2_spans_apr_to_jun(self, fiscal_calendar_quarterly: FiscalCalendar) -> None:
        """Second quarterly period spans April through June."""
        p2 = fiscal_calendar_quarterly.periods[1]
        assert p2.start_date.month == 4
        assert p2.end_date.month == 6
        assert p2.end_date.day == 30


# ============================================================================
# TestFiscalPeriodLifecycle — open → closing → closed transitions
# ============================================================================


class TestFiscalPeriodLifecycle:
    """Verify fiscal period lifecycle transitions per README.md line 448:
    OPEN → CLOSING → CLOSED."""

    def test_initial_period_status_is_open(self, fiscal_calendar: FiscalCalendar) -> None:
        """All periods start with status 'open'."""
        p = fiscal_calendar.get_current_period(as_of_date=date(2024, 6, 15))
        assert p is not None
        assert p.status == PeriodStatus.OPEN.value

    def test_start_closing_transitions_to_closing(self, fiscal_calendar: FiscalCalendar) -> None:
        """start_closing_period transitions status from OPEN to CLOSING."""
        result = fiscal_calendar.start_closing_period(1)
        assert result is True
        p = fiscal_calendar.periods[0]
        assert p.status == PeriodStatus.CLOSING.value

    def test_close_period_transitions_to_closed(self, fiscal_calendar: FiscalCalendar) -> None:
        """close_period transitions status to CLOSED."""
        fiscal_calendar.start_closing_period(1)
        result = fiscal_calendar.close_period(1)
        assert result is True
        p = fiscal_calendar.periods[0]
        assert p.status == PeriodStatus.CLOSED.value

    def test_open_period_restores_open_status(self, fiscal_calendar: FiscalCalendar) -> None:
        """open_period transitions back to OPEN from any state."""
        fiscal_calendar.close_period(1)
        result = fiscal_calendar.open_period(1)
        assert result is True
        p = fiscal_calendar.periods[0]
        assert p.status == PeriodStatus.OPEN.value

    def test_full_lifecycle_open_closing_closed(self, fiscal_calendar: FiscalCalendar) -> None:
        """Full lifecycle: OPEN → CLOSING → CLOSED."""
        p = fiscal_calendar.periods[0]
        # Step 1: Initially OPEN
        assert p.status == PeriodStatus.OPEN.value

        # Step 2: Transition to CLOSING
        fiscal_calendar.start_closing_period(1)
        assert p.status == PeriodStatus.CLOSING.value

        # Step 3: Transition to CLOSED
        fiscal_calendar.close_period(1)
        assert p.status == PeriodStatus.CLOSED.value

    def test_get_current_period_returns_fiscal_period(self, fiscal_calendar: FiscalCalendar) -> None:
        """get_current_period returns a FiscalPeriod instance."""
        result = fiscal_calendar.get_current_period(as_of_date=date(2024, 3, 15))
        assert result is not None
        assert isinstance(result, FiscalPeriod)

    def test_close_nonexistent_period_returns_false(self, fiscal_calendar: FiscalCalendar) -> None:
        """Closing a non-existent period returns False."""
        result = fiscal_calendar.close_period(99)
        assert result is False


# ============================================================================
# TestFiscalCalendarQueries — date-based fiscal period lookups
# ============================================================================


class TestFiscalCalendarQueries:
    """Verify FiscalCalendar query methods for period lookup, period-end,
    quarter-end, year-end, and fiscal year determination."""

    def test_get_period_for_date_march(self, fiscal_calendar: FiscalCalendar) -> None:
        """March 15, 2024 falls in period 3."""
        p = fiscal_calendar.get_period_for_date(date(2024, 3, 15))
        assert p is not None
        assert p.period_number == 3

    def test_get_period_for_date_jan_1(self, fiscal_calendar: FiscalCalendar) -> None:
        """January 1, 2024 falls in period 1."""
        p = fiscal_calendar.get_period_for_date(date(2024, 1, 1))
        assert p is not None
        assert p.period_number == 1

    def test_get_period_for_date_dec_31(self, fiscal_calendar: FiscalCalendar) -> None:
        """December 31, 2024 falls in period 12."""
        p = fiscal_calendar.get_period_for_date(date(2024, 12, 31))
        assert p is not None
        assert p.period_number == 12

    def test_is_period_end_last_day_of_month(self, fiscal_calendar: FiscalCalendar) -> None:
        """January 31, 2024 is the end of period 1."""
        assert fiscal_calendar.is_period_end(date(2024, 1, 31)) is True

    def test_is_period_end_mid_month(self, fiscal_calendar: FiscalCalendar) -> None:
        """January 15, 2024 is NOT a period end."""
        assert fiscal_calendar.is_period_end(date(2024, 1, 15)) is False

    def test_is_quarter_end_march_31(self, fiscal_calendar: FiscalCalendar) -> None:
        """March 31, 2024 is a quarter end (end of Q1)."""
        assert fiscal_calendar.is_quarter_end(date(2024, 3, 31)) is True

    def test_is_quarter_end_june_30(self, fiscal_calendar: FiscalCalendar) -> None:
        """June 30, 2024 is a quarter end (end of Q2)."""
        assert fiscal_calendar.is_quarter_end(date(2024, 6, 30)) is True

    def test_is_quarter_end_mid_quarter(self, fiscal_calendar: FiscalCalendar) -> None:
        """February 29, 2024 is NOT a quarter end (mid-Q1)."""
        assert fiscal_calendar.is_quarter_end(date(2024, 2, 29)) is False

    def test_is_year_end_december_31(self, fiscal_calendar: FiscalCalendar) -> None:
        """December 31, 2024 IS the year-end for Jan-start fiscal year."""
        assert fiscal_calendar.is_year_end(date(2024, 12, 31)) is True

    def test_is_year_end_not_december(self, fiscal_calendar: FiscalCalendar) -> None:
        """June 30, 2024 is NOT the year-end for Jan-start fiscal year."""
        assert fiscal_calendar.is_year_end(date(2024, 6, 30)) is False

    def test_is_year_end_july_fiscal_year(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """June 30, 2025 IS the year-end for July-start fiscal year (FY2025: Jul 2024 - Jun 2025)."""
        assert fiscal_calendar_july.is_year_end(date(2025, 6, 30)) is True

    def test_get_fiscal_year_jan_start(self, fiscal_calendar: FiscalCalendar) -> None:
        """For Jan start, fiscal year equals calendar year."""
        assert fiscal_calendar.get_fiscal_year(date(2024, 6, 15)) == 2024

    def test_get_fiscal_year_july_start_after_july(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """For July start, Aug 2024 belongs to FY2025 (FY ending Jun 2025)."""
        assert fiscal_calendar_july.get_fiscal_year(date(2024, 8, 15)) == 2025

    def test_get_fiscal_year_july_start_before_july(self, fiscal_calendar_july: FiscalCalendar) -> None:
        """For July start, May 2025 belongs to FY2025 (FY ending Jun 2025)."""
        assert fiscal_calendar_july.get_fiscal_year(date(2025, 5, 15)) == 2025

    def test_get_period_for_date_returns_none_outside_range(self, fiscal_calendar: FiscalCalendar) -> None:
        """A date far outside the generated range returns None."""
        result = fiscal_calendar.get_period_for_date(date(2000, 1, 1))
        assert result is None


# ============================================================================
# TestFiscalPeriodModel — FiscalPeriod Pydantic model construction
# ============================================================================


class TestFiscalPeriodModel:
    """Verify direct construction and field access of the FiscalPeriod model."""

    def test_fiscal_period_creation(self) -> None:
        """FiscalPeriod can be constructed with required fields."""
        fp = FiscalPeriod(
            period_number=1,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            is_quarter_end=False,
            is_year_end=False,
            status=PeriodStatus.OPEN,
        )
        assert fp.period_number == 1
        assert fp.start_date == date(2024, 1, 1)
        assert fp.end_date == date(2024, 1, 31)
        assert fp.status == PeriodStatus.OPEN.value

    def test_fiscal_period_quarter_end_flags(self) -> None:
        """FiscalPeriod correctly stores quarter-end and year-end flags."""
        fp = FiscalPeriod(
            period_number=3,
            start_date=date(2024, 3, 1),
            end_date=date(2024, 3, 31),
            is_quarter_end=True,
            is_year_end=False,
            status=PeriodStatus.OPEN,
        )
        assert fp.is_quarter_end is True
        assert fp.is_year_end is False

    def test_fiscal_period_year_end_flags(self) -> None:
        """FiscalPeriod with year-end flag set."""
        fp = FiscalPeriod(
            period_number=12,
            start_date=date(2024, 12, 1),
            end_date=date(2024, 12, 31),
            is_quarter_end=True,
            is_year_end=True,
            status=PeriodStatus.OPEN,
        )
        assert fp.is_quarter_end is True
        assert fp.is_year_end is True

    def test_fiscal_period_status_string_coercion(self) -> None:
        """PeriodStatus enum values are coerced to strings via use_enum_values."""
        fp = FiscalPeriod(
            period_number=1,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            status=PeriodStatus.CLOSING,
        )
        # With use_enum_values=True, status is stored as string "closing"
        assert fp.status == "closing"


# ============================================================================
# TestTimeControllerAdvancement — async day and hour advancement
# ============================================================================


class TestTimeControllerAdvancement:
    """Verify TimeController day advancement (skipping weekends, holidays),
    hour advancement, and event publication on day advancement.

    Per README.md lines 378-384.
    """

    async def test_advance_to_next_business_day_simple(self, time_controller: TimeController) -> None:
        """Advance from Tuesday Jan 2 → Wednesday Jan 3, 2024."""
        time_controller.current_date = date(2024, 1, 2)
        result = await time_controller.advance_to_next_business_day()
        assert time_controller.current_date == date(2024, 1, 3)
        assert result == date(2024, 1, 3)

    async def test_advance_to_next_business_day_skips_weekend(self, time_controller: TimeController) -> None:
        """Advance from Friday Mar 15 → Monday Mar 18 (skip Sat/Sun)."""
        time_controller.current_date = date(2024, 3, 15)
        await time_controller.advance_to_next_business_day()
        assert time_controller.current_date == date(2024, 3, 18)

    async def test_advance_to_next_business_day_skips_holiday(self, time_controller: TimeController) -> None:
        """Advance from Wednesday Jul 3 → Friday Jul 5 (skip Jul 4 holiday)."""
        time_controller.current_date = date(2024, 7, 3)
        await time_controller.advance_to_next_business_day()
        assert time_controller.current_date == date(2024, 7, 5)

    async def test_advance_skips_holiday_and_weekend(self, time_controller: TimeController) -> None:
        """Advance from Thursday with Friday=holiday → Monday (skip holiday+weekend)."""
        # Add a custom holiday on Friday March 22, 2024
        time_controller.business_calendar.add_company_holiday(
            date(2024, 3, 22), "Custom Friday Holiday"
        )
        time_controller.current_date = date(2024, 3, 21)  # Thursday
        await time_controller.advance_to_next_business_day()
        assert time_controller.current_date == date(2024, 3, 25)  # Monday

    async def test_advance_triggers_event_bus_publish(
        self, time_controller: TimeController, mock_event_bus: AsyncMock
    ) -> None:
        """Day advancement invokes event_bus.publish (for DayStart / PeriodClosing events).

        Per README.md line 381: 'Trigger day-start events'.
        Note: event_bus.publish is only called when a period boundary is crossed.
        We test a period boundary crossing (Jan → Feb) to verify event publication.
        """
        # Set current date to last business day of January 2024
        time_controller.current_date = date(2024, 1, 31)  # Wednesday Jan 31
        await time_controller.advance_to_next_business_day()
        # The advance crosses from period 1 (Jan) to period 2 (Feb)
        # This should trigger PeriodClosing and PeriodClosed events
        assert mock_event_bus.publish.called

    async def test_advance_resets_time_to_work_start(self, time_controller: TimeController) -> None:
        """After advancing, current_time is reset to 8:00 AM of the new date."""
        time_controller.current_date = date(2024, 3, 11)
        await time_controller.advance_to_next_business_day()
        expected_time = datetime.combine(date(2024, 3, 12), time(8, 0), tzinfo=timezone.utc)
        assert time_controller.current_time == expected_time

    async def test_advance_by_hours_within_morning(self, time_controller: TimeController) -> None:
        """Advance by 2 hours from 8:00 AM → 10:00 AM."""
        time_controller.current_date = date(2024, 3, 11)
        time_controller.current_time = datetime(2024, 3, 11, 8, 0, tzinfo=timezone.utc)
        await time_controller.advance_by_hours(2)
        assert time_controller.current_time.hour == 10

    async def test_advance_by_hours_across_lunch(self, time_controller: TimeController) -> None:
        """Advance by 5 hours from 8:00 AM crosses the lunch break.

        8:00 → 12:00 (4h consumed) → skip lunch → 13:00 → 14:00 (1h consumed) = 5h total.
        Result: 2:00 PM.
        """
        time_controller.current_date = date(2024, 3, 11)
        time_controller.current_time = datetime(2024, 3, 11, 8, 0, tzinfo=timezone.utc)
        await time_controller.advance_by_hours(5)
        assert time_controller.current_time.hour == 14  # 2:00 PM

    async def test_advance_by_zero_hours(self, time_controller: TimeController) -> None:
        """Advance by 0 hours returns the unchanged current_time."""
        original_time = time_controller.current_time
        result = await time_controller.advance_by_hours(0)
        assert result == original_time


# ============================================================================
# TestTimeControllerCalendarQueries — delegation to BusinessCalendar
# ============================================================================


class TestTimeControllerCalendarQueries:
    """Verify TimeController correctly delegates business calendar queries."""

    def test_is_business_day_delegates(self, time_controller: TimeController) -> None:
        """TimeController.is_business_day delegates to BusinessCalendar."""
        assert time_controller.is_business_day(date(2024, 3, 11)) is True
        assert time_controller.is_business_day(date(2024, 3, 16)) is False  # Saturday

    def test_get_next_business_day_delegates(self, time_controller: TimeController) -> None:
        """TimeController.get_next_business_day delegates to BusinessCalendar."""
        assert time_controller.get_next_business_day(date(2024, 3, 15)) == date(2024, 3, 18)

    def test_get_previous_business_day_delegates(self, time_controller: TimeController) -> None:
        """TimeController.get_previous_business_day delegates to BusinessCalendar."""
        assert time_controller.get_previous_business_day(date(2024, 3, 18)) == date(2024, 3, 15)

    def test_count_business_days_delegates(self, time_controller: TimeController) -> None:
        """TimeController.count_business_days delegates to BusinessCalendar."""
        count = time_controller.count_business_days(date(2024, 3, 11), date(2024, 3, 16))
        assert count == 5

    def test_add_business_days_delegates(self, time_controller: TimeController) -> None:
        """TimeController.add_business_days delegates to BusinessCalendar."""
        result = time_controller.add_business_days(date(2024, 3, 11), 5)
        assert isinstance(result, date)
        assert result == date(2024, 3, 18)


# ============================================================================
# TestTimeControllerFiscalQueries — delegation to FiscalCalendar
# ============================================================================


class TestTimeControllerFiscalQueries:
    """Verify TimeController correctly delegates fiscal calendar queries."""

    def test_get_fiscal_period_delegates(self, time_controller: TimeController) -> None:
        """TimeController.get_fiscal_period delegates to FiscalCalendar.get_period_for_date."""
        p = time_controller.get_fiscal_period(date(2024, 6, 15))
        assert p is not None
        assert p.period_number == 6

    def test_is_period_end_delegates(self, time_controller: TimeController) -> None:
        """TimeController.is_period_end delegates to FiscalCalendar."""
        assert time_controller.is_period_end(date(2024, 1, 31)) is True
        assert time_controller.is_period_end(date(2024, 1, 15)) is False

    def test_is_quarter_end_delegates(self, time_controller: TimeController) -> None:
        """TimeController.is_quarter_end delegates to FiscalCalendar."""
        assert time_controller.is_quarter_end(date(2024, 3, 31)) is True
        assert time_controller.is_quarter_end(date(2024, 2, 29)) is False

    def test_is_year_end_delegates(self, time_controller: TimeController) -> None:
        """TimeController.is_year_end delegates to FiscalCalendar."""
        assert time_controller.is_year_end(date(2024, 12, 31)) is True
        assert time_controller.is_year_end(date(2024, 6, 30)) is False

    def test_get_fiscal_year_delegates(self, time_controller: TimeController) -> None:
        """TimeController.get_fiscal_year delegates to FiscalCalendar."""
        assert time_controller.get_fiscal_year(date(2024, 6, 15)) == 2024


# ============================================================================
# TestHolidayLoading — TimeController holiday management
# ============================================================================


class TestHolidayLoading:
    """Verify TimeController holiday loading and custom holiday management."""

    def test_load_holiday_calendar_us(self, time_controller: TimeController) -> None:
        """load_holiday_calendar reloads US holidays (July 4 should remain a holiday)."""
        time_controller.load_holiday_calendar(country="US")
        assert time_controller.business_calendar.is_holiday(date(2024, 7, 4)) is True

    def test_add_custom_holiday_via_time_controller(self, time_controller: TimeController) -> None:
        """add_custom_holiday registers a company holiday via BusinessCalendar."""
        time_controller.add_custom_holiday(date(2024, 8, 5), "Company Appreciation Day")
        assert time_controller.is_business_day(date(2024, 8, 5)) is False

    def test_add_custom_holiday_appears_in_company_holidays(self, time_controller: TimeController) -> None:
        """Custom holidays added via TimeController show up in company_holidays."""
        time_controller.add_custom_holiday(date(2024, 9, 13), "Founders Day")
        assert date(2024, 9, 13) in time_controller.business_calendar.company_holidays


# ============================================================================
# TestPerformanceSLAs — calendar operation latency verification
# ============================================================================


class TestPerformanceSLAs:
    """Verify calendar operations complete within the 10 ms SLA
    per README.md line 46: 'Calendar Operations: Date calculations and
    business day checks < 10ms'."""

    def test_is_business_day_performance(self, business_calendar: BusinessCalendar) -> None:
        """is_business_day averages < 10 ms over 100 iterations."""
        start = time_module.perf_counter()
        for _ in range(100):
            business_calendar.is_business_day(date(2024, 6, 15))
        duration = time_module.perf_counter() - start
        avg_ms = (duration / 100) * 1000
        assert avg_ms < 10, f"is_business_day avg {avg_ms:.3f}ms exceeds 10ms SLA"

    def test_get_next_business_day_performance(self, business_calendar: BusinessCalendar) -> None:
        """get_next_business_day averages < 10 ms over 100 iterations."""
        start = time_module.perf_counter()
        for _ in range(100):
            business_calendar.get_next_business_day(date(2024, 6, 15))
        duration = time_module.perf_counter() - start
        avg_ms = (duration / 100) * 1000
        assert avg_ms < 10, f"get_next_business_day avg {avg_ms:.3f}ms exceeds 10ms SLA"

    def test_count_business_days_performance(self, business_calendar: BusinessCalendar) -> None:
        """count_business_days for a full year averages < 10 ms over 100 iterations."""
        start = time_module.perf_counter()
        for _ in range(100):
            business_calendar.count_business_days(date(2024, 1, 1), date(2024, 12, 31))
        duration = time_module.perf_counter() - start
        avg_ms = (duration / 100) * 1000
        assert avg_ms < 10, f"count_business_days avg {avg_ms:.3f}ms exceeds 10ms SLA"

    def test_is_period_end_performance(self, fiscal_calendar: FiscalCalendar) -> None:
        """is_period_end averages < 10 ms over 100 iterations."""
        start = time_module.perf_counter()
        for _ in range(100):
            fiscal_calendar.is_period_end(date(2024, 6, 30))
        duration = time_module.perf_counter() - start
        avg_ms = (duration / 100) * 1000
        assert avg_ms < 10, f"is_period_end avg {avg_ms:.3f}ms exceeds 10ms SLA"


# ============================================================================
# TestTimeControllerState — initial state verification
# ============================================================================


class TestTimeControllerState:
    """Verify TimeController initial state after construction."""

    def test_initial_date_set(self, time_controller: TimeController) -> None:
        """current_date is set to the provided start_date (Jan 2, 2024)."""
        assert time_controller.current_date == date(2024, 1, 2)

    def test_initial_time_set(self, time_controller: TimeController) -> None:
        """current_time is set to 8:00 AM UTC on the start date."""
        expected = datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)
        assert time_controller.current_time == expected

    def test_has_business_calendar(self, time_controller: TimeController) -> None:
        """TimeController has a non-None business_calendar."""
        assert time_controller.business_calendar is not None
        assert isinstance(time_controller.business_calendar, BusinessCalendar)

    def test_has_fiscal_calendar(self, time_controller: TimeController) -> None:
        """TimeController has a non-None fiscal_calendar."""
        assert time_controller.fiscal_calendar is not None
        assert isinstance(time_controller.fiscal_calendar, FiscalCalendar)

    def test_default_calendars_when_none_provided(self) -> None:
        """TimeController creates default calendars when none are injected."""
        tc = TimeController(start_date=date(2024, 1, 2))
        assert tc.business_calendar is not None
        assert tc.fiscal_calendar is not None

    def test_get_current_state_returns_dict(self, time_controller: TimeController) -> None:
        """get_current_state returns a serialisable dictionary."""
        state = time_controller.get_current_state()
        assert isinstance(state, dict)
        assert "current_date" in state
        assert "current_time" in state
        assert "is_business_day" in state
        assert "days_advanced" in state

    def test_get_metrics_returns_dict(self, time_controller: TimeController) -> None:
        """get_metrics returns operational metrics dictionary."""
        metrics = time_controller.get_metrics()
        assert isinstance(metrics, dict)
        assert "days_advanced" in metrics
        assert "current_date" in metrics
