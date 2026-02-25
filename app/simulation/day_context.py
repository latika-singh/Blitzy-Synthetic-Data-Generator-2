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
  ``update_from_llm_metrics``, ``update_from_p3_metrics``) reduce boilerplate
  in consuming code.
* A ``create_for_date`` classmethod factory enables clean instantiation.

Extends the base daily state with Project 3 fields for transaction workflow
state (open POs, pending invoices, GL entries, rework items) and transaction
workflow metrics (P2P/O2C cycle counts, discrepancy injection, rework loop,
GL posting, period close).

Typical usage::

    from datetime import date
    from app.simulation.day_context import DayContext

    ctx = DayContext.create_for_date(date(2024, 1, 2), day_number=1)
    ctx.transactions_generated = 45
    summary = ctx.to_daily_summary_dict()
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List
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
                "p2p_cycles_completed": 5,
                "o2c_cycles_completed": 8,
                "gl_entries_posted": 26,
                "discrepancies_injected": 1,
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

    # -- P3: Transaction Workflow State ----------------------------------------

    open_purchase_orders: List[Any] = Field(
        default_factory=list,
        description=(
            "Purchase orders awaiting goods receipt. Enables "
            "GoodsReceiptGenerator to find open POs for receipt processing."
        ),
    )
    pending_vendor_invoices: List[Any] = Field(
        default_factory=list,
        description=(
            "Vendor invoices awaiting three-way match. Enables "
            "VendorInvoiceProcessor to process pending invoices."
        ),
    )
    open_sales_orders: List[Any] = Field(
        default_factory=list,
        description=(
            "Sales orders awaiting shipment. Enables "
            "ShipmentGenerator to find open SOs for shipment processing."
        ),
    )
    pending_customer_invoices: List[Any] = Field(
        default_factory=list,
        description=(
            "Customer invoices awaiting payment. Enables "
            "CustomerPaymentProcessor to process pending invoices."
        ),
    )
    unposted_gl_entries: List[Any] = Field(
        default_factory=list,
        description=(
            "GL entries pending batch posting. Enables "
            "GLPostingEngine to post pending entries in batch."
        ),
    )
    active_rework_items: List[Any] = Field(
        default_factory=list,
        description=(
            "Transactions currently in the rework loop awaiting "
            "fix scenario application or escalation."
        ),
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

    # -- P3: Transaction Workflow Metrics ---------------------------------------

    p2p_cycles_completed: int = Field(
        default=0,
        ge=0,
        description="Number of complete P2P cycles (PO→Receipt→Invoice→Payment) this day.",
    )
    o2c_cycles_completed: int = Field(
        default=0,
        ge=0,
        description="Number of complete O2C cycles (Order→Ship→Invoice→Payment) this day.",
    )
    gl_entries_posted: int = Field(
        default=0,
        ge=0,
        description="Number of GL journal entries posted this day.",
    )
    discrepancies_injected: int = Field(
        default=0,
        ge=0,
        description="Number of discrepancies injected into transactions this day.",
    )
    ground_truths_created: int = Field(
        default=0,
        ge=0,
        description="Number of ground truth records created for injected discrepancies this day.",
    )
    rework_attempts: int = Field(
        default=0,
        ge=0,
        description="Number of rework loop attempts (fix attempts on failed transactions) this day.",
    )
    rework_successes: int = Field(
        default=0,
        ge=0,
        description="Number of successful rework fixes this day.",
    )
    rework_escalations: int = Field(
        default=0,
        ge=0,
        description="Number of rework escalations (exceeded max attempts or unresolvable) this day.",
    )
    period_closes_completed: int = Field(
        default=0,
        ge=0,
        description="Number of period close operations completed this day (0 or 1 typically).",
    )
    trial_balance_checks_passed: int = Field(
        default=0,
        ge=0,
        description="Number of trial balance validations that passed this day.",
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
            # P3: Transaction Workflow Metrics
            "p2p_cycles_completed": self.p2p_cycles_completed,
            "o2c_cycles_completed": self.o2c_cycles_completed,
            "gl_entries_posted": self.gl_entries_posted,
            "discrepancies_injected": self.discrepancies_injected,
            "ground_truths_created": self.ground_truths_created,
            "rework_attempts": self.rework_attempts,
            "rework_successes": self.rework_successes,
            "rework_escalations": self.rework_escalations,
            "period_closes_completed": self.period_closes_completed,
            "trial_balance_checks_passed": self.trial_balance_checks_passed,
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

    @property
    def rework_success_rate(self) -> float:
        """Fraction of rework attempts that succeeded.

        Returns a value in ``[0.0, 1.0]``.  When no rework attempts have
        been made the rate is ``0.0``.
        """
        return self.rework_successes / max(self.rework_attempts, 1)

    @property
    def ground_truth_coverage(self) -> float:
        """Fraction of injected discrepancies with corresponding ground truth records.

        Returns a value in ``[0.0, 1.0]``.  Target is 100% (1.0).
        When no discrepancies have been injected the rate is ``0.0``.
        """
        return self.ground_truths_created / max(self.discrepancies_injected, 1)

    @property
    def total_p3_transactions(self) -> int:
        """Total P3 transaction cycles (P2P + O2C) completed this day."""
        return self.p2p_cycles_completed + self.o2c_cycles_completed

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

    def update_from_p3_metrics(
        self,
        p2p_cycles: int = 0,
        o2c_cycles: int = 0,
        gl_entries: int = 0,
        discrepancies: int = 0,
        ground_truths: int = 0,
        rework_attempts: int = 0,
        rework_successes: int = 0,
        rework_escalations: int = 0,
        period_closes: int = 0,
        trial_balance_passed: int = 0,
    ) -> None:
        """Set all P3 transaction workflow metric fields in one call.

        Parameters
        ----------
        p2p_cycles:
            Complete P2P cycles (PO→Receipt→Invoice→Payment).
        o2c_cycles:
            Complete O2C cycles (Order→Ship→Invoice→Payment).
        gl_entries:
            GL journal entries posted.
        discrepancies:
            Discrepancies injected into transactions.
        ground_truths:
            Ground truth records created.
        rework_attempts:
            Rework loop fix attempts.
        rework_successes:
            Successful rework fixes.
        rework_escalations:
            Rework escalations (unresolvable errors).
        period_closes:
            Period close operations completed.
        trial_balance_passed:
            Trial balance validations passed.
        """
        self.p2p_cycles_completed = p2p_cycles
        self.o2c_cycles_completed = o2c_cycles
        self.gl_entries_posted = gl_entries
        self.discrepancies_injected = discrepancies
        self.ground_truths_created = ground_truths
        self.rework_attempts = rework_attempts
        self.rework_successes = rework_successes
        self.rework_escalations = rework_escalations
        self.period_closes_completed = period_closes
        self.trial_balance_checks_passed = trial_balance_passed

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
