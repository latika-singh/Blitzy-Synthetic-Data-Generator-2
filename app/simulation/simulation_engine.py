"""Main Simulation Engine — composition root and daily cycle coordinator.

The :class:`SimulationEngine` is the **top-level coordinator** for the Agent &
Orchestration Engine (Project 2) and the Transaction Workflows & Discrepancies
system (Project 3).  It serves two fundamental roles:

1. **Composition Root** — it receives every major subsystem via constructor
   injection (per AAP §0.4.3 and §0.7.1) and wires them together:
   :class:`TimeController`, :class:`ExternalWorldManager`,
   :class:`WorkflowOrchestrator`, :class:`EventBus`, :class:`AgentRegistry`,
   :class:`SimulationMetrics`, and the Project 3 subsystems
   (:class:`GLPostingEngine`, :class:`DiscrepancyInjector`,
   :class:`ReworkLoopEngine`, :class:`PeriodCloseManager`, plus P2P/O2C
   generator dictionaries).

2. **Daily Cycle Orchestrator** — its :meth:`SimulationEngine.run` method
   executes the main ``async`` loop that processes one simulated business day
   at a time through an extended pipeline (AAP §0.5.1 Group 8):

   1. **Advance Day** — publish a ``DayStart`` event via the EventBus.
   2. **Generate Interactions** — call ``ExternalWorldManager`` to produce
      customer orders, vendor invoices, and bank statements.
   2b. **P3 Transaction Generation** — invoke P2P/O2C generators to create
       complete transaction cycles (PO→Receipt→Invoice→Payment, etc.).
   2c. **GL Posting** — post pending GL journal entries via ``GLPostingEngine``.
   3a. **Discrepancy Injection** — inject discrepancies into generated
       transactions via ``DiscrepancyInjector`` before routing to agents.
   3b. **Route Transactions** — call ``WorkflowOrchestrator`` to assign
       generated transactions to the correct agent roles.
   4. **Process Agent Work** — wait (with timeout) for agents to drain
      their work queues.
   4.5. **Rework Loop** — validate completed transactions via
        ``ReworkLoopEngine`` and apply fix scenarios.
   4.6. **Period Close** — execute fiscal period close via
        ``PeriodCloseManager`` when the period is closing.
   5. **Persist Events & Finalize** — publish a ``DayComplete`` event and
      log the mandatory ``daily_summary`` structured entry (README.md
      lines 1748-1757).

Configuration is held in :class:`SimulationConfig`, a Pydantic V2 ``BaseModel``
that validates all tuneable parameters at subsystem boundaries.

Performance contracts (AAP §0.1.2):
    - Advance 1 business day: < 30 seconds (Criterion #16).
    - Agent concurrency:      ≥ 20 simultaneous agents (Criterion #4).
    - Workflow concurrency:    ≥ 100 concurrent workflows (Criterion #14).
    - Agent absolute timeout:  120 seconds (AAP timeout table).

References:
    - README.md lines 1748-1757 : daily_summary log format.
    - AAP §0.4.3               : Dependency injection wiring.
    - AAP §0.5.1 Group 8       : SimulationEngine specification.
    - AAP §0.7.1               : Constructor injection; Pydantic V2 models.
    - AAP §0.7.6               : structlog to stdout only.
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from app.simulation.day_context import DayContext
from app.simulation.simulation_metrics import SimulationMetrics

if TYPE_CHECKING:
    from app.orchestration.time_controller import TimeController
    from app.orchestration.workflow_orchestrator import WorkflowOrchestrator
    from app.external_world.external_entity_manager import ExternalWorldManager
    from app.events.event_bus import EventBus
    from app.agents.agent_registry import AgentRegistry
    # Project 3 subsystem imports (TYPE_CHECKING only)
    from app.transactions.gl.gl_posting_engine import GLPostingEngine
    from app.discrepancies.discrepancy_injector import DiscrepancyInjector
    from app.rework.rework_loop_engine import ReworkLoopEngine
    from app.transactions.gl.period_close_manager import PeriodCloseManager

# ---------------------------------------------------------------------------
# Module-level logger (structlog to stdout, per AAP §0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SimulationConfig — Pydantic V2 configuration model
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """Validated configuration for a simulation run.

    All tuneable parameters for the :class:`SimulationEngine` are declared here
    as Pydantic V2 fields with sensible defaults drawn from the specification.
    The model enforces boundary constraints (``ge``, ``le``) at assignment time.

    Attributes:
        simulation_id:
            Unique simulation run identifier.  A UUID-4 string is generated
            by default so that every run is traceable.
        start_date:
            First simulated business day.  Defaults to 2024-01-02 (the first
            weekday of fiscal year 2024).
        end_date:
            Last simulated business day (inclusive).
        max_transactions_per_day:
            Hard cap on transactions generated per simulated day.
        agent_processing_timeout_seconds:
            Absolute maximum time allowed for all agents to finish their
            work queues in a single simulated day (120 s per AAP timeout
            table).
        day_processing_timeout_seconds:
            Target wall-clock budget for processing one business day
            (Criterion #16: < 30 seconds).
        max_concurrent_agents:
            Upper bound on simultaneously active agents (Criterion #4: ≥ 20).
        max_concurrent_workflows:
            Upper bound on active workflow instances (Criterion #14: ≥ 100).
        enable_event_persistence:
            When *True*, events published through the EventBus are persisted
            to the EventStore.
        seed:
            Optional integer seed for reproducible RNG across subsystems.
        log_daily_summary:
            When *True*, emit the mandatory ``daily_summary`` structured log
            entry after every simulated day.
    """

    simulation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique simulation run identifier.",
    )
    start_date: date = Field(
        default_factory=lambda: date(2024, 1, 2),
        description="First simulated business day (default: 2024-01-02).",
    )
    end_date: date = Field(
        default_factory=lambda: date(2024, 12, 31),
        description="Last simulated business day (inclusive).",
    )
    max_transactions_per_day: int = Field(
        default=100,
        ge=1,
        description="Hard cap on daily transaction generation.",
    )
    agent_processing_timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description=(
            "Absolute max wait for agents to complete work queues "
            "(120 s per AAP timeout table)."
        ),
    )
    day_processing_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "Target wall-clock budget per business day "
            "(Criterion #16: < 30 seconds)."
        ),
    )
    max_concurrent_agents: int = Field(
        default=20,
        ge=1,
        description="Maximum simultaneously active agents (Criterion #4).",
    )
    max_concurrent_workflows: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent workflow instances (Criterion #14).",
    )
    enable_event_persistence: bool = Field(
        default=True,
        description="Persist events to EventStore when True.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional RNG seed for reproducible simulations.",
    )
    log_daily_summary: bool = Field(
        default=True,
        description="Emit mandatory daily_summary structured log entry.",
    )

    # -- Project 3: Transaction Workflow toggles ----------------------------

    enable_p3_transactions: bool = Field(
        default=True,
        description=(
            "Master toggle for Project 3 transaction generation (P2P/O2C "
            "cycles). When False, P3 generators are silently skipped even "
            "if injected."
        ),
    )
    transaction_batch_size: int = Field(
        default=100,
        ge=1,
        description=(
            "Batch size for P3 transaction generation per simulated day. "
            "Maps to the TRANSACTION_BATCH_SIZE environment variable."
        ),
    )
    enable_discrepancy_injection: bool = Field(
        default=True,
        description=(
            "Toggle for the discrepancy injection pipeline. When False, "
            "transactions proceed without discrepancy injection even if the "
            "DiscrepancyInjector is injected."
        ),
    )
    enable_rework_loop: bool = Field(
        default=True,
        description=(
            "Toggle for the rework loop validation. When False, completed "
            "transactions skip rework validation even if the "
            "ReworkLoopEngine is injected."
        ),
    )


# ---------------------------------------------------------------------------
# SimulationEngine — main loop and composition root
# ---------------------------------------------------------------------------


class SimulationEngine:
    """Main simulation engine — the composition root and daily cycle coordinator.

    Drives the daily cycle::

        advance day → generate interactions → generate P3 transactions
        → GL posting → discrepancy injection → route transactions
        → process agent work → rework loop → period close → persist events

    Acts as the dependency injection composition root, wiring all Project 2
    subsystems plus Project 3 transaction workflow subsystems together (AAP
    §0.4.3).  Every subsystem is received via constructor injection; ``None``
    values are tolerated for optional subsystems so the engine can be
    instantiated in lightweight test scenarios without the full dependency graph.

    Project 3 subsystems are guarded by config toggles (``enable_p3_transactions``,
    ``enable_discrepancy_injection``, ``enable_rework_loop``) AND by ``None``
    checks — both must be satisfied for P3 processing to execute.  This enables
    partial composition for testing and incremental rollout.

    Parameters
    ----------
    config:
        Validated :class:`SimulationConfig`.  Defaults are used when *None*.
    time_controller:
        Injected :class:`TimeController` for business-day advancement.
    external_world_manager:
        Injected :class:`ExternalWorldManager` for generating daily
        customer/vendor/bank interactions.
    workflow_orchestrator:
        Injected :class:`WorkflowOrchestrator` for routing transactions to
        agent roles.
    event_bus:
        Injected :class:`EventBus` for publishing DayStart/DayComplete events.
    agent_registry:
        Injected :class:`AgentRegistry` for monitoring agent work-queue drain.
    metrics:
        Injected :class:`SimulationMetrics` for accumulating daily results.
    gl_posting_engine:
        Injected :class:`GLPostingEngine` for posting GL journal entries.
        When *None*, GL posting is silently skipped.
    discrepancy_injector:
        Injected :class:`DiscrepancyInjector` for injecting discrepancies
        into generated transactions.  When *None*, discrepancy injection
        is silently skipped.
    rework_loop_engine:
        Injected :class:`ReworkLoopEngine` for validating completed
        transactions and applying fix scenarios.  When *None*, rework
        validation is silently skipped.
    period_close_manager:
        Injected :class:`PeriodCloseManager` for executing the 10-step
        fiscal period close process.  When *None*, period close processing
        is silently skipped.
    p2p_generators:
        Dictionary of P2P generator instances keyed by name (e.g.
        ``{"purchase_order": PurchaseOrderGenerator(...)}``) for generating
        complete P2P transaction cycles.
    o2c_generators:
        Dictionary of O2C generator instances keyed by name for generating
        complete O2C transaction cycles.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        time_controller: Optional[TimeController] = None,
        external_world_manager: Optional[ExternalWorldManager] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
        event_bus: Optional[EventBus] = None,
        agent_registry: Optional[AgentRegistry] = None,
        metrics: Optional[SimulationMetrics] = None,
        # ── Project 3 subsystems (ADR-003: all Optional, None default) ──
        gl_posting_engine: Optional["GLPostingEngine"] = None,
        discrepancy_injector: Optional["DiscrepancyInjector"] = None,
        rework_loop_engine: Optional["ReworkLoopEngine"] = None,
        period_close_manager: Optional["PeriodCloseManager"] = None,
        p2p_generators: Optional[Dict[str, Any]] = None,
        o2c_generators: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config: SimulationConfig = config or SimulationConfig()

        # Injected subsystems (all Optional to support lightweight testing)
        self._time_controller = time_controller
        self._external_world_manager = external_world_manager
        self._workflow_orchestrator = workflow_orchestrator
        self._event_bus = event_bus
        self._agent_registry = agent_registry

        # ── P3 subsystems (all Optional per ADR-003) ──
        self._gl_posting_engine = gl_posting_engine
        self._discrepancy_injector = discrepancy_injector
        self._rework_loop_engine = rework_loop_engine
        self._period_close_manager = period_close_manager
        self._p2p_generators: Dict[str, Any] = p2p_generators or {}
        self._o2c_generators: Dict[str, Any] = o2c_generators or {}

        # Metrics — default to a fresh instance if not injected
        self._metrics: SimulationMetrics = metrics or SimulationMetrics(
            simulation_id=self.config.simulation_id,
        )

        # Runtime state
        self._running: bool = False
        self._paused: bool = False
        self._current_day_context: Optional[DayContext] = None
        self._simulation_id: str = self.config.simulation_id
        self._days_processed: int = 0

        logger.info(
            "simulation_engine_initialized",
            simulation_id=self._simulation_id,
            start_date=str(self.config.start_date),
            end_date=str(self.config.end_date),
            max_transactions_per_day=self.config.max_transactions_per_day,
            max_concurrent_agents=self.config.max_concurrent_agents,
            max_concurrent_workflows=self.config.max_concurrent_workflows,
            gl_posting_engine_available=gl_posting_engine is not None,
            discrepancy_injector_available=discrepancy_injector is not None,
            rework_loop_engine_available=rework_loop_engine is not None,
            period_close_manager_available=period_close_manager is not None,
            p2p_generators_count=len(self._p2p_generators),
            o2c_generators_count=len(self._o2c_generators),
        )

    # ------------------------------------------------------------------
    # Main Run Loop (async)
    # ------------------------------------------------------------------

    async def run(self) -> Dict[str, Any]:
        """Execute the simulation from ``start_date`` to ``end_date``.

        This is the **main entry point** for the simulation.  It loops over
        business days, executing the 5-step daily pipeline for each day, and
        returns a summary dictionary upon completion or early stop.

        The loop respects the :attr:`_running` flag which can be cleared by
        :meth:`stop` or :meth:`pause` from a concurrent coroutine.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary containing ``simulation_id``,
            ``days_processed``, ``total_duration_seconds``, and ``metrics``
            (output of :meth:`SimulationMetrics.get_summary`).
        """
        self._running = True
        self._paused = False
        simulation_start_time: float = time.monotonic()
        current_date: date = self.config.start_date
        days_processed: int = 0

        logger.info(
            "simulation_started",
            simulation_id=self._simulation_id,
            start_date=str(self.config.start_date),
            end_date=str(self.config.end_date),
        )

        while self._running and current_date <= self.config.end_date:
            # Honour pause requests — spin-wait until resumed or stopped
            while self._paused and self._running:
                await asyncio.sleep(0.1)
            if not self._running:
                break

            day_start_time: float = time.monotonic()

            # Create a fresh DayContext for this business day
            day_context = DayContext(
                simulation_date=current_date,
                simulation_id=self._simulation_id,
                day_number=days_processed + 1,
            )
            self._current_day_context = day_context

            try:
                await self._process_day(day_context)

                # Record timing
                day_duration: float = time.monotonic() - day_start_time
                day_context.day_duration_seconds = day_duration

                # Record metrics
                self._metrics.record_day_completed(day_context)

                # Emit daily summary log
                if self.config.log_daily_summary:
                    self._log_daily_summary(day_context)

                days_processed += 1

            except Exception as exc:
                day_duration = time.monotonic() - day_start_time
                logger.error(
                    "simulation_day_error",
                    simulation_date=str(current_date),
                    error=str(exc),
                    day_duration_seconds=round(day_duration, 2),
                    simulation_id=self._simulation_id,
                )
                self._metrics.record_day_error(current_date, str(exc))

            # Advance to the next business day
            if self._time_controller is not None:
                current_date = await self._time_controller.advance_to_next_business_day()
            else:
                # Simple fallback for testing without a TimeController:
                # skip weekends naïvely
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)

            # Explicit yield point so concurrent coroutines (stop, pause)
            # can execute between days.
            await asyncio.sleep(0)

        # Simulation complete
        self._running = False
        self._days_processed = days_processed
        total_duration: float = time.monotonic() - simulation_start_time

        logger.info(
            "simulation_completed",
            simulation_id=self._simulation_id,
            days_processed=days_processed,
            total_duration_seconds=round(total_duration, 2),
        )

        return {
            "simulation_id": self._simulation_id,
            "days_processed": days_processed,
            "total_duration_seconds": round(total_duration, 4),
            "metrics": self._metrics.get_summary(),
        }

    # ------------------------------------------------------------------
    # 5-Step Daily Processing Pipeline
    # ------------------------------------------------------------------

    async def _process_day(self, day_context: DayContext) -> None:
        """Execute the 5-step daily processing pipeline for a single business day.

        Steps:
            1. Publish **DayStart** event.
            2. Generate external interactions (customer orders, vendor invoices).
            3. Route generated transactions to agent roles.
            4. Wait for agents to complete their work queues (with timeout).
            5. Publish **DayComplete** event and finalise.

        Parameters
        ----------
        day_context:
            The mutable :class:`DayContext` to populate during the pipeline.
        """
        sim_date = day_context.simulation_date

        # ── Step 1: Publish DayStart Event ───────────────────────────
        if self._event_bus is not None:
            day_start_event = self._create_event(
                event_type="DayStart",
                payload={
                    "simulation_date": str(sim_date),
                    "day_number": day_context.day_number,
                },
            )
            await self._event_bus.publish(day_start_event)
        logger.debug(
            "day_step_advance",
            simulation_date=str(sim_date),
            day_number=day_context.day_number,
        )

        # ── Step 2: Generate External Interactions ───────────────────
        interactions: Dict[str, Any]
        if self._external_world_manager is not None:
            interactions = await self._external_world_manager.generate_daily_interactions(
                simulation_date=sim_date,
            )
        else:
            interactions = {
                "customer_orders": [],
                "vendor_invoices": [],
                "customer_payments": [],
            }

        customer_orders: List[Any] = interactions.get("customer_orders", [])
        vendor_invoices: List[Any] = interactions.get("vendor_invoices", [])

        day_context.customer_orders_generated = len(customer_orders)
        day_context.vendor_invoices_generated = len(vendor_invoices)
        day_context.transactions_generated = (
            day_context.customer_orders_generated
            + day_context.vendor_invoices_generated
        )

        logger.debug(
            "day_step_generate_interactions",
            simulation_date=str(sim_date),
            customer_orders=len(customer_orders),
            vendor_invoices=len(vendor_invoices),
            total_transactions=day_context.transactions_generated,
        )

        # ── Step 2b: P3 Transaction Generation (P2P + O2C) ──────────
        # Guarded by enable_p3_transactions config toggle and injected generators.
        # Each generator receives the DayContext and returns a result dict with
        # cycles_completed count. Failures per-generator are logged and skipped.
        total_p2p_cycles: int = 0
        total_o2c_cycles: int = 0

        if self.config.enable_p3_transactions and self._p2p_generators:
            for gen_name, gen in self._p2p_generators.items():
                try:
                    gen_result = await gen.generate(day_context)
                    cycles = (
                        gen_result.get("cycles_completed", 0)
                        if isinstance(gen_result, dict)
                        else 0
                    )
                    total_p2p_cycles += cycles
                except Exception as p2p_err:
                    logger.warning(
                        "p2p_generator_error",
                        generator=gen_name,
                        error=str(p2p_err),
                        simulation_date=str(sim_date),
                    )

        if self.config.enable_p3_transactions and self._o2c_generators:
            for gen_name, gen in self._o2c_generators.items():
                try:
                    gen_result = await gen.generate(day_context)
                    cycles = (
                        gen_result.get("cycles_completed", 0)
                        if isinstance(gen_result, dict)
                        else 0
                    )
                    total_o2c_cycles += cycles
                except Exception as o2c_err:
                    logger.warning(
                        "o2c_generator_error",
                        generator=gen_name,
                        error=str(o2c_err),
                        simulation_date=str(sim_date),
                    )

        day_context.p2p_cycles_completed = total_p2p_cycles
        day_context.o2c_cycles_completed = total_o2c_cycles

        logger.debug(
            "day_step_p3_transaction_generation",
            simulation_date=str(sim_date),
            p2p_cycles=total_p2p_cycles,
            o2c_cycles=total_o2c_cycles,
        )

        # ── Step 2c: P3 GL Posting ───────────────────────────────────
        # Post pending GL journal entries immediately after transaction
        # generation, before routing to agents.  Guarded by injected engine.
        gl_entries_posted: int = 0
        if self._gl_posting_engine is not None:
            try:
                gl_result = await self._gl_posting_engine.post_pending_entries(
                    day_context=day_context,
                )
                if isinstance(gl_result, dict):
                    gl_entries_posted = gl_result.get("entries_posted", 0)
                day_context.gl_entries_posted = gl_entries_posted
            except Exception as gl_err:
                logger.warning(
                    "gl_posting_error",
                    error=str(gl_err),
                    simulation_date=str(sim_date),
                )

        # ── Step 3a: P3 Discrepancy Injection ────────────────────────
        # Pass generated transactions through the discrepancy injection
        # pipeline BEFORE routing to agents.  Guarded by config toggle
        # and injected injector.
        discrepancies_injected: int = 0
        ground_truths_created: int = 0
        if (
            self.config.enable_discrepancy_injection
            and self._discrepancy_injector is not None
        ):
            try:
                disc_result = await self._discrepancy_injector.process_batch(
                    day_context=day_context,
                )
                if isinstance(disc_result, dict):
                    discrepancies_injected = disc_result.get(
                        "discrepancies_injected", 0,
                    )
                    ground_truths_created = disc_result.get(
                        "ground_truths_created", 0,
                    )
                day_context.discrepancies_injected = discrepancies_injected
                day_context.ground_truths_created = ground_truths_created
            except Exception as disc_err:
                logger.warning(
                    "discrepancy_injection_error",
                    error=str(disc_err),
                    simulation_date=str(sim_date),
                )

        # ── Step 3b: Route Transactions to Agents ────────────────────
        transactions_routed: int = 0
        if self._workflow_orchestrator is not None:
            for order in customer_orders:
                try:
                    await self._workflow_orchestrator.route_transaction(
                        transaction=order,
                        transaction_type="sales_order",
                    )
                    transactions_routed += 1
                except Exception as route_err:
                    logger.warning(
                        "transaction_routing_error",
                        transaction_type="sales_order",
                        error=str(route_err),
                        simulation_date=str(sim_date),
                    )

            for invoice in vendor_invoices:
                try:
                    await self._workflow_orchestrator.route_transaction(
                        transaction=invoice,
                        transaction_type="vendor_invoice",
                    )
                    transactions_routed += 1
                except Exception as route_err:
                    logger.warning(
                        "transaction_routing_error",
                        transaction_type="vendor_invoice",
                        error=str(route_err),
                        simulation_date=str(sim_date),
                    )

        day_context.transactions_routed = transactions_routed
        logger.debug(
            "day_step_route_transactions",
            simulation_date=str(sim_date),
            transactions_routed=transactions_routed,
        )

        # ── Step 4: Process Agent Work (with timeout) ────────────────
        if self._agent_registry is not None:
            await self._wait_for_agents_to_complete(
                timeout=self.config.agent_processing_timeout_seconds,
            )
            # Collect agent metrics from the registry
            capacity = self._agent_registry.get_capacity_metrics()
            idle_agents = self._agent_registry.get_idle_agents()
            all_agents = self._agent_registry.get_all_agents()
            total_agents = self._agent_registry.get_agent_count()
            busy_count: int = capacity.get("busy_count", 0)
            idle_count: int = capacity.get("idle_count", 0)
            error_count: int = capacity.get("error_count", 0)
            utilization_pct: float = capacity.get("utilization_pct", 0.0)

            day_context.agents_active = busy_count + idle_count
            day_context.agents_idle = idle_count
            day_context.agents_error = error_count
            # Convert percentage (0-100) to ratio (0.0-1.0)
            day_context.average_agent_utilization = min(
                utilization_pct / 100.0, 1.0,
            )

        logger.debug(
            "day_step_process_agents",
            simulation_date=str(sim_date),
            agents_active=day_context.agents_active,
            average_utilization=day_context.average_agent_utilization,
        )

        # ── Step 4.5: P3 Rework Loop Validation ─────────────────────
        # Validate completed transactions and apply fix scenarios.
        # Guarded by config toggle and injected engine.  Max 3 attempts
        # per transaction (per AAP §0.1.2); escalates unresolvable errors.
        rework_attempts: int = 0
        rework_successes: int = 0
        rework_escalations: int = 0
        if self.config.enable_rework_loop and self._rework_loop_engine is not None:
            try:
                rework_result = await self._rework_loop_engine.process_day(
                    day_context=day_context,
                )
                if isinstance(rework_result, dict):
                    rework_attempts = rework_result.get("attempts", 0)
                    rework_successes = rework_result.get("successes", 0)
                    rework_escalations = rework_result.get("escalations", 0)
                day_context.rework_attempts = rework_attempts
                day_context.rework_successes = rework_successes
                day_context.rework_escalations = rework_escalations
            except Exception as rework_err:
                logger.warning(
                    "rework_loop_error",
                    error=str(rework_err),
                    simulation_date=str(sim_date),
                )

        logger.debug(
            "day_step_rework_loop",
            simulation_date=str(sim_date),
            rework_attempts=rework_attempts,
            rework_successes=rework_successes,
            rework_escalations=rework_escalations,
        )

        # ── Step 4.6: P3 Period Close Check ──────────────────────────
        # Execute the 10-step fiscal period close process when the
        # day_context indicates a period is closing.  Produces
        # period_closes_completed and trial_balance_checks_passed metrics.
        if self._period_close_manager is not None and day_context.is_period_closing():
            try:
                pc_result = await self._period_close_manager.execute_period_close(
                    day_context=day_context,
                )
                if isinstance(pc_result, dict):
                    day_context.period_closes_completed = 1
                    day_context.trial_balance_checks_passed = pc_result.get(
                        "trial_balance_passed", 0,
                    )
            except Exception as pc_err:
                logger.error(
                    "period_close_error",
                    error=str(pc_err),
                    simulation_date=str(sim_date),
                )

        # ── Step 5: Persist Events & Finalize ────────────────────────
        if self._event_bus is not None:
            day_complete_event = self._create_event(
                event_type="DayComplete",
                payload={
                    # Existing P2 fields
                    "simulation_date": str(sim_date),
                    "transactions_generated": day_context.transactions_generated,
                    "transactions_routed": day_context.transactions_routed,
                    "agents_active": day_context.agents_active,
                    # P3: Transaction Workflow metrics
                    "p2p_cycles_completed": day_context.p2p_cycles_completed,
                    "o2c_cycles_completed": day_context.o2c_cycles_completed,
                    "gl_entries_posted": day_context.gl_entries_posted,
                    "discrepancies_injected": day_context.discrepancies_injected,
                    "rework_attempts": day_context.rework_attempts,
                    "rework_successes": day_context.rework_successes,
                    "rework_escalations": day_context.rework_escalations,
                },
            )
            await self._event_bus.publish(day_complete_event)

        logger.debug(
            "day_step_persist_events",
            simulation_date=str(sim_date),
            transactions_generated=day_context.transactions_generated,
            p2p_cycles=day_context.p2p_cycles_completed,
            o2c_cycles=day_context.o2c_cycles_completed,
            gl_entries_posted=day_context.gl_entries_posted,
            discrepancies_injected=day_context.discrepancies_injected,
            rework_attempts=day_context.rework_attempts,
            period_closes=day_context.period_closes_completed,
        )

    # ------------------------------------------------------------------
    # Agent processing helper
    # ------------------------------------------------------------------

    async def _wait_for_agents_to_complete(self, timeout: float = 30.0) -> None:
        """Wait for all agents to finish processing their current work queues.

        Polls the :class:`AgentRegistry` every 100 ms until all agent work
        queues are empty and all agents are idle, or the *timeout* expires.
        On timeout, a warning is logged but the pipeline continues — agents
        will catch up on subsequent days.

        Parameters
        ----------
        timeout:
            Maximum wall-clock seconds to wait.
        """
        if self._agent_registry is None:
            return

        deadline: float = time.monotonic() + timeout
        while time.monotonic() < deadline:
            capacity = self._agent_registry.get_capacity_metrics()
            total_queue: int = capacity.get("total_queue_depth", 0)
            busy: int = capacity.get("busy_count", 0)

            if total_queue == 0 and busy == 0:
                # All work done
                return

            await asyncio.sleep(0.1)

        # Timeout reached — log warning and move on
        remaining_metrics = self._agent_registry.get_capacity_metrics()
        logger.warning(
            "agent_processing_timeout",
            timeout_seconds=timeout,
            remaining_queue_depth=remaining_metrics.get("total_queue_depth", 0),
            remaining_busy_agents=remaining_metrics.get("busy_count", 0),
        )

    # ------------------------------------------------------------------
    # Event factory helper
    # ------------------------------------------------------------------

    def _create_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Any:
        """Create an :class:`Event` instance for publishing via the EventBus.

        Imports :class:`Event` from ``app.events.event_types`` lazily to
        prevent circular imports at module level (the EventBus is a
        TYPE_CHECKING-only import).

        Parameters
        ----------
        event_type:
            String discriminator for the event (e.g. ``"DayStart"``).
        payload:
            JSONB-compatible dictionary carrying event-specific data.

        Returns
        -------
        Event
            A new event instance ready for publishing.
        """
        from app.events.event_types import Event

        return Event(
            event_type=event_type,
            payload=payload,
            simulation_id=UUID(self._simulation_id) if self._simulation_id else uuid4(),
        )

    # ------------------------------------------------------------------
    # Daily Summary Logging (README.md lines 1748-1757)
    # ------------------------------------------------------------------

    def _log_daily_summary(self, day_context: DayContext) -> None:
        """Emit the structured ``daily_summary`` log entry.

        The keyword arguments **exactly** match the format documented in
        README.md lines 1748-1757 and required by AAP §0.7.6::

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
        logger.info(
            "daily_summary",
            simulation_date=str(day_context.simulation_date),
            transactions_generated=day_context.transactions_generated,
            agents_active=day_context.agents_active,
            average_agent_utilization=day_context.average_agent_utilization,
            llm_requests=day_context.llm_requests,
            llm_cost_usd=day_context.llm_cost_usd,
            duration_seconds=day_context.day_duration_seconds,
            # P3 daily metrics
            p2p_cycles_completed=day_context.p2p_cycles_completed,
            o2c_cycles_completed=day_context.o2c_cycles_completed,
            gl_entries_posted=day_context.gl_entries_posted,
            discrepancies_injected=day_context.discrepancies_injected,
            rework_attempts=day_context.rework_attempts,
            rework_successes=day_context.rework_successes,
        )

    # ------------------------------------------------------------------
    # Control Methods
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Signal the simulation to stop after the current day completes.

        Sets the internal running flag to *False*, causing the main
        :meth:`run` loop to exit gracefully after it finishes processing
        the current simulated business day.
        """
        self._running = False
        self._paused = False
        logger.info(
            "simulation_stopping",
            simulation_id=self._simulation_id,
        )

    async def pause(self) -> None:
        """Pause the simulation loop.

        The engine enters a spin-wait between days until :meth:`resume` or
        :meth:`stop` is called.
        """
        self._paused = True
        logger.info(
            "simulation_paused",
            simulation_id=self._simulation_id,
        )

    async def resume(self) -> None:
        """Resume a paused simulation.

        Clears the pause flag so the main :meth:`run` loop continues
        processing the next business day.
        """
        self._paused = False
        logger.info(
            "simulation_resumed",
            simulation_id=self._simulation_id,
        )

    def is_running(self) -> bool:
        """Return *True* if the simulation loop is currently active."""
        return self._running

    def get_current_day_context(self) -> Optional[DayContext]:
        """Return the :class:`DayContext` for the day currently being processed.

        Returns *None* if the simulation has not yet started or has already
        completed.
        """
        return self._current_day_context

    # ------------------------------------------------------------------
    # Metrics Access
    # ------------------------------------------------------------------

    def get_metrics(self) -> SimulationMetrics:
        """Return the :class:`SimulationMetrics` instance for this run."""
        return self._metrics

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Return a comprehensive summary of the simulation run.

        Includes the simulation configuration, metrics summary, running
        state, and days-processed count.  Suitable for structured logging
        and external monitoring consumption.
        """
        return {
            "simulation_id": self._simulation_id,
            "config": {
                "start_date": str(self.config.start_date),
                "end_date": str(self.config.end_date),
                "max_transactions_per_day": self.config.max_transactions_per_day,
                "max_concurrent_agents": self.config.max_concurrent_agents,
                "max_concurrent_workflows": self.config.max_concurrent_workflows,
                "enable_event_persistence": self.config.enable_event_persistence,
                "seed": self.config.seed,
            },
            "p3_config": {
                "gl_posting_engine_available": self._gl_posting_engine is not None,
                "discrepancy_injector_available": self._discrepancy_injector is not None,
                "rework_loop_engine_available": self._rework_loop_engine is not None,
                "period_close_manager_available": self._period_close_manager is not None,
                "p2p_generators_count": len(self._p2p_generators),
                "o2c_generators_count": len(self._o2c_generators),
                "enable_p3_transactions": self.config.enable_p3_transactions,
                "enable_discrepancy_injection": self.config.enable_discrepancy_injection,
                "enable_rework_loop": self.config.enable_rework_loop,
                "transaction_batch_size": self.config.transaction_batch_size,
            },
            "is_running": self._running,
            "is_paused": self._paused,
            "days_processed": self._days_processed,
            "metrics": self._metrics.get_summary(),
        }

    # ------------------------------------------------------------------
    # Subsystem Health Check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Check availability and status of each injected subsystem.

        Returns a dictionary mapping subsystem names to availability status
        dictionaries.  For the ``agent_registry``, the agent count is also
        included.  This method never raises — all subsystem checks are
        wrapped in exception handlers.

        Returns
        -------
        Dict[str, Any]
            Health status for every subsystem.
        """
        health: Dict[str, Any] = {
            "simulation_id": self._simulation_id,
            "is_running": self._running,
            "is_paused": self._paused,
            "subsystems": {},
        }

        # Time controller
        try:
            health["subsystems"]["time_controller"] = {
                "available": self._time_controller is not None,
                "status": "ok" if self._time_controller is not None else "not_injected",
            }
        except Exception as exc:
            health["subsystems"]["time_controller"] = {
                "available": False,
                "status": f"error: {exc}",
            }

        # External world manager
        try:
            health["subsystems"]["external_world_manager"] = {
                "available": self._external_world_manager is not None,
                "status": "ok" if self._external_world_manager is not None else "not_injected",
            }
        except Exception as exc:
            health["subsystems"]["external_world_manager"] = {
                "available": False,
                "status": f"error: {exc}",
            }

        # Workflow orchestrator
        try:
            health["subsystems"]["workflow_orchestrator"] = {
                "available": self._workflow_orchestrator is not None,
                "status": "ok" if self._workflow_orchestrator is not None else "not_injected",
            }
        except Exception as exc:
            health["subsystems"]["workflow_orchestrator"] = {
                "available": False,
                "status": f"error: {exc}",
            }

        # Event bus
        try:
            health["subsystems"]["event_bus"] = {
                "available": self._event_bus is not None,
                "status": "ok" if self._event_bus is not None else "not_injected",
            }
        except Exception as exc:
            health["subsystems"]["event_bus"] = {
                "available": False,
                "status": f"error: {exc}",
            }

        # Agent registry
        try:
            if self._agent_registry is not None:
                agent_count = self._agent_registry.get_agent_count()
                capacity = self._agent_registry.get_capacity_metrics()
                health["subsystems"]["agent_registry"] = {
                    "available": True,
                    "status": "ok",
                    "agent_count": agent_count,
                    "capacity_metrics": capacity,
                }
            else:
                health["subsystems"]["agent_registry"] = {
                    "available": False,
                    "status": "not_injected",
                }
        except Exception as exc:
            health["subsystems"]["agent_registry"] = {
                "available": False,
                "status": f"error: {exc}",
            }

        # ── P3 subsystem health checks ───────────────────────────────
        for p3_name, p3_ref in (
            ("gl_posting_engine", self._gl_posting_engine),
            ("discrepancy_injector", self._discrepancy_injector),
            ("rework_loop_engine", self._rework_loop_engine),
            ("period_close_manager", self._period_close_manager),
        ):
            try:
                health["subsystems"][p3_name] = {
                    "available": p3_ref is not None,
                    "status": "ok" if p3_ref is not None else "not_injected",
                }
            except Exception as exc:
                health["subsystems"][p3_name] = {
                    "available": False,
                    "status": f"error: {exc}",
                }

        return health
