"""Time controller for simulation time progression.

Manages date/time state, business day advancement (skipping weekends and
holidays), intra-day hour progression, and fiscal calendar queries.  Publishes
DayStart and PeriodClosing / PeriodClosed events through the EventBus.

Key responsibilities:
    - Maintain authoritative ``current_date`` and ``current_time`` for the
      running simulation.
    - Advance to the next business day, automatically skipping weekends and
      holidays via the injected :class:`BusinessCalendar`.
    - Advance intra-day time in hour increments, respecting the
      8:00–17:00 working-hours window and the 12:00–13:00 lunch break.
    - Delegate fiscal-period queries (period lookup, quarter-end, year-end)
      to the injected :class:`FiscalCalendar`.
    - Detect fiscal-period boundary crossings during day advancement and
      publish ``PeriodClosing`` / ``PeriodClosed`` events via the
      :class:`EventBus`.

Performance contracts (AAP §0.1.2):
    - Business day advancement: < 30 seconds.
    - Calendar operations: < 10 ms (pure computation, O(1) set lookups).

References:
    - README.md lines 369–403  : TimeController API surface.
    - README.md lines 425–428  : Working hours and lunch break.
    - AAP §0.5.1 Group 6       : Orchestration layer (F-003, F-004).
    - AAP §0.7.1               : Constructor injection; EventBus sole async
      notification mechanism.
    - AAP §0.7.6               : structlog to stdout only.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

from app.orchestration.business_calendar import BusinessCalendar
from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod

if TYPE_CHECKING:
    from app.events.event_bus import EventBus

# ---------------------------------------------------------------------------
# Module-level logger (structlog to stdout, per AAP §0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Working-hours constants (per README.md lines 425-428)
# ---------------------------------------------------------------------------
WORK_START: time = time(8, 0)     # 08:00 start of business day
WORK_END: time = time(17, 0)      # 17:00 end of business day
LUNCH_START: time = time(12, 0)   # 12:00 lunch begins
LUNCH_END: time = time(13, 0)     # 13:00 lunch ends
EFFECTIVE_HOURS_PER_DAY: float = 8.0  # 9h gross − 1h lunch = 8h effective


# ---------------------------------------------------------------------------
# Pydantic V2 model for cross-module state serialisation
# ---------------------------------------------------------------------------


class TimeControllerState(BaseModel):
    """Pydantic V2 model representing a serialisable snapshot of the
    :class:`TimeController`'s current state.

    Used at subsystem boundaries per AAP §0.7.1.
    """

    current_date: str = Field(
        ..., description="Current simulation date in ISO-8601 format."
    )
    current_time: str = Field(
        ..., description="Current simulation datetime in ISO-8601 format."
    )
    is_business_day: bool = Field(
        ..., description="Whether the current date is a business day."
    )
    fiscal_period: Optional[Dict[str, Any]] = Field(
        default=None, description="Current fiscal period information."
    )
    days_advanced: int = Field(
        default=0, description="Total business days advanced so far."
    )
    simulation_start_date: str = Field(
        ..., description="The date the simulation started in ISO-8601 format."
    )


# ---------------------------------------------------------------------------
# TimeController
# ---------------------------------------------------------------------------


class TimeController:
    """Manages simulation time progression.

    Integrates :class:`BusinessCalendar` and :class:`FiscalCalendar` for
    date/time state, business-day advancement, intra-day hour advancement,
    and fiscal-period queries.  Publishes ``PeriodClosing`` / ``PeriodClosed``
    events via the injected :class:`EventBus` when a fiscal-period boundary
    is crossed during day advancement.

    All dependencies are supplied via constructor injection (AAP §0.7.1).

    Args:
        start_date: Simulation start date.
        business_calendar: Optional :class:`BusinessCalendar`; a default
            instance is created when ``None``.
        fiscal_calendar: Optional :class:`FiscalCalendar`; a default
            instance is created when ``None``.
        event_bus: Optional :class:`EventBus` for publishing cross-subsystem
            async notifications.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        start_date: date,
        business_calendar: Optional[BusinessCalendar] = None,
        fiscal_calendar: Optional[FiscalCalendar] = None,
        event_bus: Optional["EventBus"] = None,
    ) -> None:
        # --- Time state ---
        self.current_date: date = start_date
        self.current_time: datetime = datetime.combine(
            start_date, WORK_START, tzinfo=timezone.utc
        )

        # --- Injected calendars (default to fresh instances) ---
        self.business_calendar: BusinessCalendar = (
            business_calendar if business_calendar is not None else BusinessCalendar()
        )
        self.fiscal_calendar: FiscalCalendar = (
            fiscal_calendar if fiscal_calendar is not None else FiscalCalendar()
        )

        # --- EventBus reference (optional; used for async notifications) ---
        self._event_bus: Optional["EventBus"] = event_bus

        # --- Metrics ---
        self._days_advanced: int = 0
        self._start_simulation_date: date = start_date

        logger.info(
            "time_controller_initialized",
            start_date=start_date.isoformat(),
            has_event_bus=event_bus is not None,
        )

    # ------------------------------------------------------------------
    # Time advancement — async (may publish events)
    # ------------------------------------------------------------------

    async def advance_to_next_business_day(self) -> date:
        """Advance simulation time to the next business day.

        Skips weekends and holidays using the injected
        :class:`BusinessCalendar`.  After advancing, checks for
        fiscal-period boundary crossings and publishes
        ``PeriodClosing`` / ``PeriodClosed`` events as needed.

        Performance contract: < 30 seconds (AAP §0.1.2).

        Returns:
            The new ``current_date`` after advancement.
        """
        previous_date = self.current_date

        # Move forward at least one calendar day
        next_date = self.current_date + timedelta(days=1)

        # Skip non-business days (weekends, holidays)
        while not self.business_calendar.is_business_day(next_date):
            next_date += timedelta(days=1)

        # Update time state — reset to start of business day
        self.current_date = next_date
        self.current_time = datetime.combine(
            next_date, WORK_START, tzinfo=timezone.utc
        )
        self._days_advanced += 1

        # Detect and publish fiscal-period transitions
        await self._check_and_publish_period_events(previous_date, next_date)

        # Determine current fiscal period for logging
        fiscal_period = self.fiscal_calendar.get_period_for_date(next_date)
        period_info: Optional[str] = None
        if fiscal_period is not None:
            period_info = (
                f"FY{fiscal_period.fiscal_year}-P{fiscal_period.period_number}"
            )

        logger.info(
            "day_advanced",
            new_date=next_date.isoformat(),
            previous_date=previous_date.isoformat(),
            days_advanced=self._days_advanced,
            fiscal_period=period_info,
        )

        return self.current_date

    async def advance_by_hours(self, hours: int) -> datetime:
        """Advance simulation time by *hours* working hours.

        Respects working-hours boundaries (08:00–17:00) and the lunch
        break (12:00–13:00).  When the resulting time exceeds the end of
        the current business day, the simulation automatically rolls over
        to the next business day and continues consuming remaining hours.

        Args:
            hours: Number of effective working hours to advance (may be 0
                   or positive).

        Returns:
            The updated ``current_time`` after advancement.
        """
        if hours <= 0:
            return self.current_time

        remaining_hours = float(hours)

        while remaining_hours > 0.0:
            current_t = self.current_time.time()  # naive time for comparison

            # If we're before working hours, snap to start of day
            if current_t < WORK_START:
                self.current_time = datetime.combine(
                    self.current_date, WORK_START, tzinfo=timezone.utc
                )
                current_t = WORK_START

            # If we're during lunch, skip to end of lunch
            if LUNCH_START <= current_t < LUNCH_END:
                self.current_time = datetime.combine(
                    self.current_date, LUNCH_END, tzinfo=timezone.utc
                )
                current_t = LUNCH_END

            # If we're at or past end of day, advance to next business day
            if current_t >= WORK_END:
                await self.advance_to_next_business_day()
                continue

            # Calculate available working hours until next boundary
            available = self._hours_until_next_boundary()

            if remaining_hours <= available:
                # Advance within the current segment
                self.current_time += timedelta(hours=remaining_hours)

                # If we landed in the lunch break, skip past it
                new_t = self.current_time.time()
                if LUNCH_START <= new_t < LUNCH_END:
                    self.current_time = datetime.combine(
                        self.current_date, LUNCH_END, tzinfo=timezone.utc
                    )

                remaining_hours = 0.0
            else:
                # Consume all available hours in this segment then continue
                remaining_hours -= available
                self.current_time += timedelta(hours=available)

                # After consuming this segment, skip lunch or end of day
                new_t = self.current_time.time()
                if LUNCH_START <= new_t < LUNCH_END:
                    self.current_time = datetime.combine(
                        self.current_date, LUNCH_END, tzinfo=timezone.utc
                    )
                elif new_t >= WORK_END:
                    if remaining_hours > 0.0:
                        await self.advance_to_next_business_day()

        return self.current_time

    async def advance_to_time(self, target_time: time) -> datetime:
        """Advance simulation time to a specific clock time on the current
        or next business day.

        If *target_time* is earlier than the current time-of-day, the
        simulation advances to the next business day first, then sets the
        time to *target_time*.

        Args:
            target_time: Target wall-clock time to set.

        Returns:
            The updated ``current_time``.
        """
        current_t = self.current_time.time()  # naive time for comparison

        if target_time <= current_t:
            # Already past this time today — advance to next business day
            await self.advance_to_next_business_day()

        self.current_time = datetime.combine(
            self.current_date, target_time, tzinfo=timezone.utc
        )
        return self.current_time

    # ------------------------------------------------------------------
    # Business calendar delegation (synchronous pure computation)
    # ------------------------------------------------------------------

    def is_business_day(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is a working business day.

        Delegates to :meth:`BusinessCalendar.is_business_day`.
        """
        return self.business_calendar.is_business_day(check_date)

    def get_next_business_day(self, from_date: date) -> date:
        """Return the first business day strictly after *from_date*.

        Delegates to :meth:`BusinessCalendar.get_next_business_day`.
        """
        return self.business_calendar.get_next_business_day(from_date)

    def get_previous_business_day(self, from_date: date) -> date:
        """Return the first business day strictly before *from_date*.

        Delegates to :meth:`BusinessCalendar.get_previous_business_day`.
        """
        return self.business_calendar.get_previous_business_day(from_date)

    def count_business_days(self, start: date, end: date) -> int:
        """Count business days in ``[start, end)`` (inclusive start,
        exclusive end).

        Delegates to :meth:`BusinessCalendar.count_business_days`.
        """
        return self.business_calendar.count_business_days(start, end)

    def add_business_days(self, from_date: date, days: int) -> date:
        """Add (or subtract) *days* business days to *from_date*.

        Delegates to :meth:`BusinessCalendar.add_business_days`.
        """
        return self.business_calendar.add_business_days(from_date, days)

    # ------------------------------------------------------------------
    # Fiscal calendar delegation (synchronous pure computation)
    # ------------------------------------------------------------------

    def get_fiscal_period(self, check_date: date) -> Optional[FiscalPeriod]:
        """Return the fiscal period containing *check_date*, or ``None``.

        Delegates to :meth:`FiscalCalendar.get_period_for_date`.
        """
        return self.fiscal_calendar.get_period_for_date(check_date)

    def is_period_end(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is the last day of any fiscal
        period.

        Delegates to :meth:`FiscalCalendar.is_period_end`.
        """
        return self.fiscal_calendar.is_period_end(check_date)

    def is_quarter_end(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is the last day of a
        quarter-end period.

        Delegates to :meth:`FiscalCalendar.is_quarter_end`.
        """
        return self.fiscal_calendar.is_quarter_end(check_date)

    def is_year_end(self, check_date: date) -> bool:
        """Return ``True`` if *check_date* is the last day of a
        year-end period.

        Delegates to :meth:`FiscalCalendar.is_year_end`.
        """
        return self.fiscal_calendar.is_year_end(check_date)

    def get_fiscal_year(self, check_date: date) -> int:
        """Return the fiscal year that *check_date* falls in.

        Delegates to :meth:`FiscalCalendar.get_fiscal_year`.
        """
        return self.fiscal_calendar.get_fiscal_year(check_date)

    # ------------------------------------------------------------------
    # Holiday management
    # ------------------------------------------------------------------

    def load_holiday_calendar(self, country: str = "US") -> None:
        """(Re-)load the holiday calendar for the specified country.

        Delegates to :meth:`BusinessCalendar.load_holidays` and logs the
        operation.
        """
        self.business_calendar.load_holidays(country)
        logger.info("holiday_calendar_loaded", country=country)

    def add_custom_holiday(self, holiday_date: date, name: str) -> None:
        """Register a custom company holiday.

        Delegates to :meth:`BusinessCalendar.add_company_holiday` and logs
        the operation.
        """
        self.business_calendar.add_company_holiday(holiday_date, name)
        logger.info(
            "custom_holiday_added",
            date=holiday_date.isoformat(),
            name=name,
        )

    # ------------------------------------------------------------------
    # State and metrics
    # ------------------------------------------------------------------

    def get_current_state(self) -> Dict[str, Any]:
        """Return a serialisable dictionary describing the controller's
        current time state.

        Includes current date/time, business-day status, fiscal period
        information, day advancement count, and simulation start date.
        """
        fiscal_period = self.fiscal_calendar.get_period_for_date(self.current_date)
        period_info: Optional[Dict[str, Any]] = None
        if fiscal_period is not None:
            period_info = {
                "period_number": fiscal_period.period_number,
                "fiscal_year": fiscal_period.fiscal_year,
                "quarter": fiscal_period.quarter,
                "start_date": fiscal_period.start_date.isoformat(),
                "end_date": fiscal_period.end_date.isoformat(),
                "status": fiscal_period.status,
                "is_quarter_end": fiscal_period.is_quarter_end,
                "is_year_end": fiscal_period.is_year_end,
            }

        return {
            "current_date": self.current_date.isoformat(),
            "current_time": self.current_time.isoformat(),
            "is_business_day": self.business_calendar.is_business_day(
                self.current_date
            ),
            "fiscal_period": period_info,
            "days_advanced": self._days_advanced,
            "simulation_start_date": self._start_simulation_date.isoformat(),
        }

    def get_working_hours_remaining(self) -> float:
        """Calculate effective working hours remaining in the current
        business day.

        Working hours: 08:00–17:00 minus lunch 12:00–13:00 = 8 effective
        hours per full day.

        Returns:
            Remaining effective hours (0.0 if outside working hours or on
            a non-business day).
        """
        if not self.business_calendar.is_business_day(self.current_date):
            return 0.0

        current_t = self.current_time.time()  # naive time for comparison

        # Before working hours — full day remaining
        if current_t < WORK_START:
            return EFFECTIVE_HOURS_PER_DAY

        # After working hours — nothing remaining
        if current_t >= WORK_END:
            return 0.0

        # Compute raw hours from now until end of day
        now_seconds = (
            current_t.hour * 3600 + current_t.minute * 60 + current_t.second
        )
        end_seconds = WORK_END.hour * 3600 + WORK_END.minute * 60
        remaining_seconds = float(end_seconds - now_seconds)

        # Subtract lunch if it hasn't passed yet
        lunch_start_seconds = LUNCH_START.hour * 3600 + LUNCH_START.minute * 60
        lunch_end_seconds = LUNCH_END.hour * 3600 + LUNCH_END.minute * 60

        if now_seconds < lunch_start_seconds:
            # Full lunch break still ahead
            remaining_seconds -= float(lunch_end_seconds - lunch_start_seconds)
        elif now_seconds < lunch_end_seconds:
            # Currently in lunch — subtract remaining lunch time
            remaining_seconds -= float(lunch_end_seconds - now_seconds)
        # else: lunch already passed — no deduction

        return max(remaining_seconds / 3600.0, 0.0)

    def is_within_working_hours(self) -> bool:
        """Return ``True`` if the current simulation time is within
        working hours (08:00–17:00) and **not** during the lunch break
        (12:00–13:00).
        """
        if not self.business_calendar.is_business_day(self.current_date):
            return False

        current_t = self.current_time.time()  # naive time for comparison

        # Must be within [WORK_START, WORK_END)
        if current_t < WORK_START or current_t >= WORK_END:
            return False

        # Must not be during lunch [LUNCH_START, LUNCH_END)
        if LUNCH_START <= current_t < LUNCH_END:
            return False

        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the time controller.

        Provides a summary of simulation progress and current state
        suitable for aggregation by :class:`SimulationMetrics`.
        """
        fiscal_period = self.fiscal_calendar.get_period_for_date(self.current_date)
        period_info: Optional[str] = None
        if fiscal_period is not None:
            period_info = (
                f"FY{fiscal_period.fiscal_year}-P{fiscal_period.period_number}"
            )

        return {
            "days_advanced": self._days_advanced,
            "simulation_start_date": self._start_simulation_date.isoformat(),
            "current_date": self.current_date.isoformat(),
            "current_time": self.current_time.isoformat(),
            "fiscal_period": period_info,
            "is_business_day": self.business_calendar.is_business_day(
                self.current_date
            ),
            "working_hours_remaining": self.get_working_hours_remaining(),
            "is_within_working_hours": self.is_within_working_hours(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _check_and_publish_period_events(
        self, previous_date: date, new_date: date
    ) -> None:
        """Detect fiscal-period boundary crossings between *previous_date*
        and *new_date*, and publish ``PeriodClosing`` / ``PeriodClosed``
        events via the :class:`EventBus`.

        The method also drives fiscal-calendar lifecycle transitions:
        ``start_closing_period`` → ``close_period`` → ``open_period``.

        Args:
            previous_date: The simulation date **before** advancement.
            new_date: The simulation date **after** advancement.
        """
        if self._event_bus is None:
            return

        old_period = self.fiscal_calendar.get_period_for_date(previous_date)
        new_period = self.fiscal_calendar.get_period_for_date(new_date)

        if old_period is None or new_period is None:
            return

        # No boundary crossed — same period
        if (
            old_period.period_number == new_period.period_number
            and old_period.fiscal_year == new_period.fiscal_year
        ):
            return

        # --- Import event classes at runtime (avoids circular imports) ---
        from app.events.event_types import PeriodClosed, PeriodClosing

        # Determine period type label for the old period
        period_type = "monthly"
        if old_period.is_year_end:
            period_type = "yearly"
        elif old_period.is_quarter_end:
            period_type = "quarterly"

        # --- Publish PeriodClosing ---
        closing_event = PeriodClosing(
            payload={
                "period_type": period_type,
                "period_number": old_period.period_number,
                "period_start": old_period.start_date.isoformat(),
                "period_end": old_period.end_date.isoformat(),
                "fiscal_year": old_period.fiscal_year,
            }
        )
        await self._event_bus.publish(closing_event)

        # --- Drive fiscal-calendar lifecycle ---
        self.fiscal_calendar.start_closing_period(
            old_period.period_number, old_period.fiscal_year
        )
        self.fiscal_calendar.close_period(
            old_period.period_number, old_period.fiscal_year
        )

        # --- Publish PeriodClosed ---
        closed_event = PeriodClosed(
            payload={
                "period_type": period_type,
                "period_number": old_period.period_number,
                "period_start": old_period.start_date.isoformat(),
                "period_end": old_period.end_date.isoformat(),
                "fiscal_year": old_period.fiscal_year,
                "closing_summary": {
                    "previous_date": previous_date.isoformat(),
                    "new_date": new_date.isoformat(),
                },
            }
        )
        await self._event_bus.publish(closed_event)

        # --- Open the new period if it is not already open ---
        from app.orchestration.fiscal_calendar import PeriodStatus

        if new_period.status != PeriodStatus.OPEN.value:
            self.fiscal_calendar.open_period(
                new_period.period_number, new_period.fiscal_year
            )

        logger.info(
            "period_transition",
            old_period=f"FY{old_period.fiscal_year}-P{old_period.period_number}",
            new_period=f"FY{new_period.fiscal_year}-P{new_period.period_number}",
            period_type=period_type,
        )

    def _hours_until_next_boundary(self) -> float:
        """Return effective working hours from ``current_time`` until the
        next scheduling boundary (lunch start or end of day).

        This helper enables :meth:`advance_by_hours` to consume time in
        contiguous segments separated by the lunch break.
        """
        current_t = self.current_time.time()  # naive time for comparison
        now_seconds = (
            current_t.hour * 3600 + current_t.minute * 60 + current_t.second
        )
        lunch_start_seconds = LUNCH_START.hour * 3600 + LUNCH_START.minute * 60
        end_seconds = WORK_END.hour * 3600 + WORK_END.minute * 60

        if now_seconds < lunch_start_seconds:
            # Next boundary is lunch start
            return float(lunch_start_seconds - now_seconds) / 3600.0
        else:
            # Next boundary is end of day (we've already skipped past lunch)
            return float(end_seconds - now_seconds) / 3600.0
