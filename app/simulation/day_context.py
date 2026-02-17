"""Daily simulation context model for the Agent & Orchestration Engine.

This module defines the ``DayContext`` Pydantic V2 data model that carries the
current simulation date, fiscal period information, active-agent counts, daily
transaction counts, LLM-usage metrics, event counts, and performance data
through each daily processing cycle.

``DayContext`` is the **most foundational file** in the simulation package.  It
is consumed by ``SimulationEngine`` (which creates a fresh instance for every
simulated business day) and ``SimulationMetrics`` (which extracts snapshot data
from it), but it depends on **no other simulation-internal files** — only on
Pydantic and the Python standard library.

Key design decisions
--------------------
* **Pydantic V2 BaseModel** with ``validate_assignment=True`` so that field
  values are validated whenever they are mutated during the daily cycle (AAP
  §0.7.1).
* ``simulation_date`` is a **required** field (no default) so that every
  ``DayContext`` is unambiguously tied to a business day.
* All numeric counters carry ``ge=0`` constraints; ``average_agent_utilization``
  is further bounded to ``[0.0, 1.0]``.
* A ``to_daily_summary_dict()`` method produces the exact dictionary expected by
  the daily-summary structured-log entry (README.md lines 1748-1757).
* Convenience helpers (``is_period_closing``, ``is_quarter_end``,
  ``is_year_end``, ``transaction_completion_rate``, ``total_agent_count``) and
  batch-update methods (``update_from_agent_metrics``,
  ``update_from_llm_metrics``) reduce boilerplate in consuming code.
* A ``create_for_date`` classmethod factory enables clean instantiation.

Typical usage::

    from datetime import date
    from app.simulation.day_context import DayContext

    ctx = DayContext.create_for_date(date(2024, 1, 2), day_number=1)
    ctx.transactions_generated = 45
    summary = ctx.to_daily_summary_dict()
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Embedded sub-model: FiscalPeriodInfo
# ---------------------------------------------------------------------------


class FiscalPeriodInfo(BaseModel):
    """Fiscal-period context embedded within each :class:`DayContext`.

    Carries the fiscal year, month, quarter, period lifecycle status, and
    flags for quarter-end and year-end periods.  The ``period_status`` field
    follows the lifecycle defined in the fiscal calendar specification:
    ``"open"`` → ``"closing"`` → ``"closed"``.
    """

    fiscal_year: int = Field(
        default=0,
        description="Fiscal year, e.g. 2024.",
    )
    fiscal_month: int = Field(
        default=0,
        description="Month within the fiscal year (1-12).",
    )
    fiscal_quarter: int = Field(
        default=0,
        description="Quarter within the fiscal year (1-4).",
    )
    period_status: str = Field(
        default="open",
        description=(
            "Lifecycle status of the current fiscal period.  "
            'One of "open", "closing", or "closed".'
        ),
    )
    is_quarter_end: bool = Field(
        default=False,
        description="True when the current period falls at a quarter boundary.",
    )
    is_year_end: bool = Field(
        default=False,
        description="True when the current period falls at the fiscal year-end.",
    )
    period_name: str = Field(
        default="",
        description='Human-readable period name, e.g. "January 2024".',
    )


# ---------------------------------------------------------------------------
# Main model: DayContext
# ---------------------------------------------------------------------------


class DayContext(BaseModel):
    """Carries daily simulation state through the processing pipeline.

    Created fresh for each simulated business day by
    :class:`~app.simulation.simulation_engine.SimulationEngine`.  Passed to
    every subsystem so they know the simulation date and can record metrics.

    The model is **mutable** (``validate_assignment=True``) — subsystems
    increment counters and attach metadata during the processing cycle, and
    each mutation is validated by Pydantic.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "simulation_id": "abc-123",
                "day_number": 1,
                "simulation_date": "2024-01-02",
                "transactions_generated": 45,
                "agents_active": 12,
                "average_agent_utilization": 0.75,
                "llm_requests": 30,
                "llm_cost_usd": 1.50,
                "day_duration_seconds": 22.5,
            }
        },
    )

    # -- Core identification ------------------------------------------------

    simulation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the simulation run.",
    )
    day_number: int = Field(
        default=0,
        ge=0,
        description="Sequential day counter (1-based within a simulation run).",
    )
    simulation_date: date = Field(
        ...,
        description="The simulated business day (required — no default).",
    )

    # -- Fiscal period info -------------------------------------------------

    fiscal_period: FiscalPeriodInfo = Field(
        default_factory=FiscalPeriodInfo,
        description="Fiscal-period context for this simulated day.",
    )

    # -- Agent state --------------------------------------------------------

    agents_active: int = Field(
        default=0,
        ge=0,
        description="Count of agents that processed work items this day.",
    )
    agents_idle: int = Field(
        default=0,
        ge=0,
        description="Count of idle agents.",
    )
    agents_error: int = Field(
        default=0,
        ge=0,
        description="Count of agents in error state.",
    )
    average_agent_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Average proportion of time agents were busy (0.0-1.0).",
    )
    agent_decisions_made: int = Field(
        default=0,
        ge=0,
        description="Total decisions made by all agents this day.",
    )

    # -- Transaction counts -------------------------------------------------

    transactions_generated: int = Field(
        default=0,
        ge=0,
        description="Total transactions generated this day.",
    )
    transactions_routed: int = Field(
        default=0,
        ge=0,
        description="Transactions routed to agent work queues.",
    )
    transactions_completed: int = Field(
        default=0,
        ge=0,
        description="Transactions that completed successfully.",
    )
    transactions_failed: int = Field(
        default=0,
        ge=0,
        description="Transactions that failed during processing.",
    )
    customer_orders_generated: int = Field(
        default=0,
        ge=0,
        description="Customer orders produced by ExternalWorldManager.",
    )
    vendor_invoices_generated: int = Field(
        default=0,
        ge=0,
        description="Vendor invoices produced by ExternalWorldManager.",
    )

    # -- LLM usage ----------------------------------------------------------

    llm_requests: int = Field(
        default=0,
        ge=0,
        description="LLM API calls made this day.",
    )
    llm_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total LLM cost for this day in USD.",
    )
    llm_average_latency_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Average LLM response latency in seconds.",
    )

    # -- Event metrics ------------------------------------------------------

    events_published: int = Field(
        default=0,
        ge=0,
        description="Events published through EventBus this day.",
    )
    events_processed: int = Field(
        default=0,
        ge=0,
        description="Events successfully processed by handlers.",
    )

    # -- Performance --------------------------------------------------------

    day_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock seconds to process this simulated day.",
    )
    errors: int = Field(
        default=0,
        ge=0,
        description="Total errors encountered during this day.",
    )
    warnings: int = Field(
        default=0,
        ge=0,
        description="Total warnings encountered during this day.",
    )

    # -- Metadata -----------------------------------------------------------

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extensible metadata dictionary for subsystems to attach "
            "additional context."
        ),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this DayContext was created.",
    )

    # -----------------------------------------------------------------------
    # Convenience methods
    # -----------------------------------------------------------------------

    def to_daily_summary_dict(self) -> Dict[str, Any]:
        """Return a dictionary matching the structured daily-summary log format.

        The keys produced here correspond exactly to the keyword arguments
        expected by the ``daily_summary`` structlog event documented in
        README.md lines 1748-1757::

            logger.info(
                "daily_summary",
                simulation_date=str(current_date),
                transactions_generated=transaction_count,
                agents_active=active_agent_count,
                average_agent_utilization=avg_utilization,
                llm_requests=llm_request_count,
                llm_cost_usd=total_llm_cost,
                duration_seconds=day_duration,
            )
        """
        return {
            "simulation_date": str(self.simulation_date),
            "transactions_generated": self.transactions_generated,
            "agents_active": self.agents_active,
            "average_agent_utilization": round(self.average_agent_utilization, 4),
            "llm_requests": self.llm_requests,
            "llm_cost_usd": round(self.llm_cost_usd, 4),
            "duration_seconds": round(self.day_duration_seconds, 2),
        }

    def is_period_closing(self) -> bool:
        """Return ``True`` if the fiscal period is in the ``"closing"`` state.

        Useful for triggering period-end processing workflows such as
        accrual calculations, reconciliation runs, and close-period
        journal entries.
        """
        return self.fiscal_period.period_status == "closing"

    def is_quarter_end(self) -> bool:
        """Return ``True`` if the current fiscal period is a quarter-end."""
        return self.fiscal_period.is_quarter_end

    def is_year_end(self) -> bool:
        """Return ``True`` if the current fiscal period is a fiscal year-end."""
        return self.fiscal_period.is_year_end

    # -----------------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------------

    @property
    def transaction_completion_rate(self) -> float:
        """Fraction of generated transactions that completed successfully.

        Returns a value in ``[0.0, 1.0]``.  When no transactions have been
        generated the rate is ``0.0`` (avoids division by zero).  The target
        threshold is ≥ 99.5 % (Criterion #13).
        """
        return self.transactions_completed / max(self.transactions_generated, 1)

    @property
    def total_agent_count(self) -> int:
        """Total number of agents across all states (active + idle + error)."""
        return self.agents_active + self.agents_idle + self.agents_error

    # -----------------------------------------------------------------------
    # Batch-update helpers
    # -----------------------------------------------------------------------

    def update_from_agent_metrics(
        self,
        active: int,
        idle: int,
        error: int,
        utilization: float,
        decisions: int,
    ) -> None:
        """Set all agent-related fields in one call.

        Parameters
        ----------
        active:
            Number of agents that processed work items.
        idle:
            Number of idle agents.
        error:
            Number of agents in error state.
        utilization:
            Average agent utilization ratio (0.0–1.0).
        decisions:
            Total decisions made by agents.
        """
        self.agents_active = active
        self.agents_idle = idle
        self.agents_error = error
        self.average_agent_utilization = utilization
        self.agent_decisions_made = decisions

    def update_from_llm_metrics(
        self,
        requests: int,
        cost_usd: float,
        avg_latency: float,
    ) -> None:
        """Set all LLM-related fields in one call.

        Parameters
        ----------
        requests:
            Number of LLM API calls made.
        cost_usd:
            Total LLM cost in USD.
        avg_latency:
            Average LLM response latency in seconds.
        """
        self.llm_requests = requests
        self.llm_cost_usd = cost_usd
        self.llm_average_latency_seconds = avg_latency

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def create_for_date(
        cls,
        simulation_date: date,
        simulation_id: str = "",
        day_number: int = 0,
    ) -> "DayContext":
        """Create a fresh :class:`DayContext` for a given date.

        This is the preferred factory method for
        :class:`~app.simulation.simulation_engine.SimulationEngine` to use
        when initialising the context for a new simulated business day.

        Parameters
        ----------
        simulation_date:
            The business day to simulate.
        simulation_id:
            An optional simulation-run identifier.  If empty, a new UUID is
            generated automatically.
        day_number:
            Sequential counter for the day within the simulation run.

        Returns
        -------
        DayContext
            A new instance with all counters at their default (zero) values.
        """
        return cls(
            simulation_date=simulation_date,
            simulation_id=simulation_id or str(uuid4()),
            day_number=day_number,
        )
