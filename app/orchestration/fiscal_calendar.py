"""Fiscal calendar management with configurable fiscal year start, monthly/quarterly
periods, and period lifecycle (open/closing/closed). Supports quarter-end and year-end
flags for period-close processing triggers.

This module is a foundational component of the orchestration layer (F-004 Time
Controller). It generates fiscal periods for multiple fiscal years, tracks their
lifecycle state, and provides efficient date-based lookup for period queries.

Key capabilities:
  - Configurable fiscal year start month (1-12, where 1=January)
  - Monthly (12 periods/year) or quarterly (4 periods/year) period types
  - Period lifecycle: OPEN -> CLOSING -> CLOSED
  - Quarter-end and year-end flags for triggering period-close procedures
  - Date-to-period lookup, fiscal year / quarter calculation
  - Calendar state and metrics reporting

Performance target: all calendar operations < 10 ms (pure computation, no I/O).
"""
from __future__ import annotations

import calendar
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class PeriodStatus(str, Enum):
    """Fiscal period lifecycle states.

    The lifecycle flows: OPEN -> CLOSING -> CLOSED.
      - OPEN:    Period is active and accepting transactions.
      - CLOSING: Period is in the process of being closed (period-end procedures).
      - CLOSED:  Period is fully closed; no new transactions allowed.
    """

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class FiscalPeriod(BaseModel):
    """Pydantic V2 model representing a single fiscal period.

    Attributes:
        period_number: 1-based period number within the fiscal year
                       (1-12 for monthly, 1-4 for quarterly).
        start_date:    First calendar day of the period.
        end_date:      Last calendar day of the period.
        is_quarter_end: True if this period is the last period of a fiscal quarter.
        is_year_end:    True if this period is the last period of the fiscal year.
        status:         Current lifecycle status (open / closing / closed).
        fiscal_year:    Fiscal year label this period belongs to.
        quarter:        Fiscal quarter (1-4) this period falls within.
    """

    period_number: int = Field(
        ..., ge=1, description="1-based period number within the fiscal year"
    )
    start_date: date = Field(..., description="First calendar day of the period")
    end_date: date = Field(..., description="Last calendar day of the period")
    is_quarter_end: bool = Field(
        default=False,
        description="True if this is the last period of a fiscal quarter",
    )
    is_year_end: bool = Field(
        default=False,
        description="True if this is the last period of the fiscal year",
    )
    status: PeriodStatus = Field(
        default=PeriodStatus.OPEN, description="Period lifecycle status"
    )
    fiscal_year: int = Field(
        default=0, description="Fiscal year this period belongs to"
    )
    quarter: int = Field(
        default=0, ge=0, le=4, description="Fiscal quarter (1-4)"
    )

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# FiscalCalendar
# ---------------------------------------------------------------------------


class FiscalCalendar:
    """Manages configurable fiscal year periods with lifecycle tracking.

    The calendar pre-generates fiscal periods for a configurable number of
    years so that all date-based lookups are simple list scans (< 10 ms for
    typical 3-year / 36-period datasets).

    Args:
        fiscal_year_start_month: Month (1-12) when the fiscal year begins.
            1 = January (calendar-year fiscal year), 7 = July (common US
            government fiscal year), etc.
        period_type: ``"monthly"`` (12 periods/year) or ``"quarterly"``
            (4 periods/year).
        start_year: First calendar year covered by the generated periods.
        num_years:  Number of fiscal years to generate.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        fiscal_year_start_month: int = 1,
        period_type: str = "monthly",
        start_year: int = 2024,
        num_years: int = 3,
    ) -> None:
        # Validate fiscal_year_start_month
        if not 1 <= fiscal_year_start_month <= 12:
            raise ValueError(
                f"fiscal_year_start_month must be 1-12, got {fiscal_year_start_month}"
            )

        # Validate period_type
        valid_period_types = ("monthly", "quarterly")
        if period_type not in valid_period_types:
            raise ValueError(
                f"period_type must be one of {valid_period_types}, got '{period_type}'"
            )

        self.fiscal_year_start_month: int = fiscal_year_start_month
        self.period_type: str = period_type
        self.periods: List[FiscalPeriod] = []

        self._generate_periods(start_year, num_years)

        logger.info(
            "fiscal_calendar_initialized",
            fiscal_year_start_month=self.fiscal_year_start_month,
            period_type=self.period_type,
            periods_count=len(self.periods),
            start_year=start_year,
            num_years=num_years,
        )

    # ------------------------------------------------------------------
    # Period generation (private)
    # ------------------------------------------------------------------

    def _generate_periods(self, start_year: int, num_years: int) -> None:
        """Generate all fiscal periods for *num_years* fiscal years.

        For each fiscal year the method creates either 12 (monthly) or 4
        (quarterly) ``FiscalPeriod`` objects, computing correct start/end
        dates, quarter assignments, and quarter-end / year-end flags.

        The *fiscal year label* follows common accounting conventions:
          - If ``fiscal_year_start_month == 1`` the label equals the
            calendar year (FY 2024 runs Jan 2024 – Dec 2024).
          - If ``fiscal_year_start_month > 1`` the label equals the
            calendar year in which the fiscal year **ends** (e.g. FY 2025
            runs Jul 2024 – Jun 2025 when start month is 7).
        """
        self.periods = []

        periods_per_year = 12 if self.period_type == "monthly" else 4
        months_per_period = 1 if self.period_type == "monthly" else 3

        for fy_offset in range(num_years):
            # Calendar year in which this fiscal year's first month falls
            cal_start_year = start_year + fy_offset

            # Fiscal year label
            if self.fiscal_year_start_month == 1:
                fy_label = cal_start_year
            else:
                # Fiscal year label = calendar year the FY *ends* in
                fy_label = cal_start_year + 1

            for period_idx in range(periods_per_year):
                period_number = period_idx + 1  # 1-based

                # First month offset (0-based) from fiscal year start
                first_month_offset = period_idx * months_per_period

                # Calculate the calendar (year, month) for the first month
                # of this period, handling year roll-over.
                start_month_abs = (
                    self.fiscal_year_start_month + first_month_offset - 1
                )  # 0-based
                period_start_year = cal_start_year + start_month_abs // 12
                period_start_month = start_month_abs % 12 + 1  # back to 1-based

                # Calculate the calendar (year, month) for the last month of
                # this period.
                last_month_offset = first_month_offset + months_per_period - 1
                end_month_abs = (
                    self.fiscal_year_start_month + last_month_offset - 1
                )  # 0-based
                period_end_year = cal_start_year + end_month_abs // 12
                period_end_month = end_month_abs % 12 + 1

                start_date, _ = self._get_period_dates(
                    period_start_year, period_start_month
                )
                _, end_date = self._get_period_dates(period_end_year, period_end_month)

                # Quarter: 1-4 (group periods into sets of 3 months)
                if self.period_type == "monthly":
                    quarter = (period_idx // 3) + 1
                else:
                    quarter = period_number  # each quarterly period IS a quarter

                # Quarter-end flag
                if self.period_type == "monthly":
                    is_quarter_end = period_number % 3 == 0
                else:
                    is_quarter_end = True  # every quarterly period is a quarter end

                # Year-end flag
                is_year_end = period_number == periods_per_year

                period = FiscalPeriod(
                    period_number=period_number,
                    start_date=start_date,
                    end_date=end_date,
                    is_quarter_end=is_quarter_end,
                    is_year_end=is_year_end,
                    status=PeriodStatus.OPEN,
                    fiscal_year=fy_label,
                    quarter=quarter,
                )
                self.periods.append(period)

    def _get_period_dates(self, year: int, month: int) -> tuple[date, date]:
        """Return ``(first_day, last_day)`` for the given calendar *year* and *month*.

        Uses :func:`calendar.monthrange` to determine the last day of the
        month correctly, including leap-year handling for February.
        """
        first_day = date(year, month, 1)
        _, last_day_num = calendar.monthrange(year, month)
        last_day = date(year, month, last_day_num)
        return first_day, last_day

    # ------------------------------------------------------------------
    # Period lifecycle operations
    # ------------------------------------------------------------------

    def _find_period(
        self, period_number: int, fiscal_year: Optional[int] = None
    ) -> Optional[FiscalPeriod]:
        """Locate a period by its number and optional fiscal year.

        When *fiscal_year* is ``None`` the **first** period matching
        *period_number* is returned (useful when only one fiscal year is
        loaded or the caller doesn't care about the year).
        """
        for period in self.periods:
            if period.period_number == period_number:
                if fiscal_year is None or period.fiscal_year == fiscal_year:
                    return period
        return None

    def open_period(
        self, period_number: int, fiscal_year: Optional[int] = None
    ) -> bool:
        """Transition a period back to **OPEN** status.

        Args:
            period_number: 1-based period number.
            fiscal_year:   Optional fiscal year label to disambiguate.

        Returns:
            ``True`` if the period was found and opened, ``False`` otherwise.
        """
        period = self._find_period(period_number, fiscal_year)
        if period is None:
            logger.warning(
                "period_not_found",
                action="open_period",
                period_number=period_number,
                fiscal_year=fiscal_year,
            )
            return False

        previous_status = period.status
        period.status = PeriodStatus.OPEN
        logger.info(
            "period_opened",
            period_number=period_number,
            fiscal_year=period.fiscal_year,
            previous_status=previous_status,
        )
        return True

    def start_closing_period(
        self, period_number: int, fiscal_year: Optional[int] = None
    ) -> bool:
        """Transition a period to **CLOSING** status.

        This is the intermediate state before a period is fully closed.
        Typically triggers period-end review procedures.

        Args:
            period_number: 1-based period number.
            fiscal_year:   Optional fiscal year label to disambiguate.

        Returns:
            ``True`` if the period was found and transitioned, ``False`` otherwise.
        """
        period = self._find_period(period_number, fiscal_year)
        if period is None:
            logger.warning(
                "period_not_found",
                action="start_closing_period",
                period_number=period_number,
                fiscal_year=fiscal_year,
            )
            return False

        previous_status = period.status
        period.status = PeriodStatus.CLOSING
        logger.info(
            "period_closing_started",
            period_number=period_number,
            fiscal_year=period.fiscal_year,
            previous_status=previous_status,
        )
        return True

    def close_period(
        self, period_number: int, fiscal_year: Optional[int] = None
    ) -> bool:
        """Transition a period to **CLOSED** status.

        Once closed no new transactions should be posted to this period.

        Args:
            period_number: 1-based period number.
            fiscal_year:   Optional fiscal year label to disambiguate.

        Returns:
            ``True`` if the period was found and closed, ``False`` otherwise.
        """
        period = self._find_period(period_number, fiscal_year)
        if period is None:
            logger.warning(
                "period_not_found",
                action="close_period",
                period_number=period_number,
                fiscal_year=fiscal_year,
            )
            return False

        previous_status = period.status
        period.status = PeriodStatus.CLOSED
        logger.info(
            "period_closed",
            period_number=period_number,
            fiscal_year=period.fiscal_year,
            previous_status=previous_status,
        )
        return True

    # ------------------------------------------------------------------
    # Period queries
    # ------------------------------------------------------------------

    def get_current_period(
        self, as_of_date: Optional[date] = None
    ) -> Optional[FiscalPeriod]:
        """Return the fiscal period containing *as_of_date*.

        If *as_of_date* is ``None`` the current system date is used.  In a
        simulation context callers should always pass the simulation's
        current date explicitly.

        Returns:
            The matching :class:`FiscalPeriod` or ``None`` if no period
            covers the requested date.
        """
        check_date = as_of_date if as_of_date is not None else date.today()
        return self.get_period_for_date(check_date)

    def get_period_for_date(self, check_date: date) -> Optional[FiscalPeriod]:
        """Find and return the :class:`FiscalPeriod` that contains *check_date*.

        Performs a linear scan — efficient for typical workloads (≤ 36
        periods for 3 years of monthly periods).
        """
        for period in self.periods:
            if period.start_date <= check_date <= period.end_date:
                return period
        return None

    def is_period_end(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is the last day of **any** fiscal period."""
        return any(period.end_date == check_date for period in self.periods)

    def is_quarter_end(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is the last day of a quarter-end period."""
        return any(
            period.end_date == check_date and period.is_quarter_end
            for period in self.periods
        )

    def is_year_end(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is the last day of a year-end period."""
        return any(
            period.end_date == check_date and period.is_year_end
            for period in self.periods
        )

    def get_fiscal_year(self, check_date: date) -> int:
        """Determine which fiscal year *check_date* falls in.

        The fiscal year label follows standard accounting convention:

        * ``fiscal_year_start_month == 1``:  FY label = calendar year.
        * ``fiscal_year_start_month > 1``:   FY label = calendar year in
          which the fiscal year **ends**.  For example with a July start,
          August 2024 belongs to FY 2025.

        If the date falls within a pre-generated period we use the stored
        ``fiscal_year`` attribute directly.  Otherwise we compute it from
        the formula.
        """
        period = self.get_period_for_date(check_date)
        if period is not None:
            return period.fiscal_year

        # Fallback computation for dates outside generated range
        if self.fiscal_year_start_month == 1:
            return check_date.year

        if check_date.month >= self.fiscal_year_start_month:
            return check_date.year + 1
        return check_date.year

    def get_quarter(self, check_date: date) -> int:
        """Determine the fiscal quarter (1-4) for *check_date*.

        If the date falls within a pre-generated period the stored quarter
        is returned.  Otherwise the quarter is computed from the month
        offset relative to the fiscal year start month.
        """
        period = self.get_period_for_date(check_date)
        if period is not None:
            return period.quarter

        # Fallback computation
        months_offset = (check_date.month - self.fiscal_year_start_month) % 12
        return (months_offset // 3) + 1

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    def get_periods_by_status(self, status: PeriodStatus) -> List[FiscalPeriod]:
        """Return all periods with the given *status*."""
        target = status.value if isinstance(status, PeriodStatus) else status
        return [p for p in self.periods if p.status == target]

    def get_periods_for_fiscal_year(self, fiscal_year: int) -> List[FiscalPeriod]:
        """Return all periods belonging to the given *fiscal_year*."""
        return [p for p in self.periods if p.fiscal_year == fiscal_year]

    def get_open_periods(self) -> List[FiscalPeriod]:
        """Convenience method: return all periods with status **OPEN**."""
        return self.get_periods_by_status(PeriodStatus.OPEN)

    # ------------------------------------------------------------------
    # State & metrics
    # ------------------------------------------------------------------

    def get_calendar_state(self) -> Dict[str, Any]:
        """Return a summary dictionary describing the current calendar configuration
        and period status counts.
        """
        open_count = sum(
            1 for p in self.periods if p.status == PeriodStatus.OPEN.value
        )
        closing_count = sum(
            1 for p in self.periods if p.status == PeriodStatus.CLOSING.value
        )
        closed_count = sum(
            1 for p in self.periods if p.status == PeriodStatus.CLOSED.value
        )

        return {
            "fiscal_year_start_month": self.fiscal_year_start_month,
            "period_type": self.period_type,
            "total_periods": len(self.periods),
            "open_count": open_count,
            "closing_count": closing_count,
            "closed_count": closed_count,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the fiscal calendar."""
        fiscal_years_covered: List[int] = sorted(
            {p.fiscal_year for p in self.periods}
        )

        open_count = sum(
            1 for p in self.periods if p.status == PeriodStatus.OPEN.value
        )
        closing_count = sum(
            1 for p in self.periods if p.status == PeriodStatus.CLOSING.value
        )
        closed_count = sum(
            1 for p in self.periods if p.status == PeriodStatus.CLOSED.value
        )

        return {
            "total_periods": len(self.periods),
            "by_status": {
                "open": open_count,
                "closing": closing_count,
                "closed": closed_count,
            },
            "fiscal_years_covered": fiscal_years_covered,
            "fiscal_year_start_month": self.fiscal_year_start_month,
            "period_type": self.period_type,
        }
