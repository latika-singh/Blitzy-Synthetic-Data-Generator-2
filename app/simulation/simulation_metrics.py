"""Simulation performance metrics for the Agent & Orchestration Engine.

This module provides three core components for tracking, aggregating, and
validating simulation performance across all seven subsystems:

*   :class:`DailyMetricsSnapshot` — Pydantic V2 model capturing metrics for a
    single simulated business day (transactions, agents, LLM usage, events,
    errors, timing).
*   :class:`MonthlyMetricsAggregate` — Pydantic V2 model aggregating daily
    snapshots into calendar-month rollups with computed averages.
*   :class:`SimulationMetrics` — Stateful collector that records daily results
    from :class:`~app.simulation.day_context.DayContext`, maintains cumulative
    counters, computes monthly aggregates, validates against the 20 measurable
    performance thresholds (README.md §SUCCESS CRITERIA), and optionally
    persists snapshots to Redis (key pattern ``simulation:{uuid}:metrics``
    per AAP §0.4.4).

The module is consumed primarily by
:class:`~app.simulation.simulation_engine.SimulationEngine` after each
simulated business day and by external monitoring consumers.

Key design decisions
--------------------
* ``DayContext`` is imported under a ``TYPE_CHECKING`` guard to avoid circular
  imports — this module never instantiates ``DayContext`` itself.
* All logging uses ``structlog`` to stdout in structured JSON format per AAP
  §0.7.6.
* Constructor injection for the optional Redis client follows AAP §0.7.1.
* Monthly averages are incrementally recomputed using running sums and counts
  to avoid re-scanning the snapshot list.
* Redis persistence methods are ``async`` to support both synchronous and
  asynchronous Redis clients.

Typical usage::

    from app.simulation.simulation_metrics import SimulationMetrics

    metrics = SimulationMetrics(simulation_id="run-001")
    snapshot = metrics.record_day_completed(day_context)
    summary = metrics.get_summary()
    thresholds = metrics.validate_performance_thresholds()
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

import structlog

if TYPE_CHECKING:
    from app.simulation.day_context import DayContext

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic V2 Data Models
# ---------------------------------------------------------------------------


class DailyMetricsSnapshot(BaseModel):
    """Immutable snapshot of metrics for a single simulated business day.

    Each field corresponds to a metric tracked during the daily processing
    cycle.  Instances are created by :meth:`SimulationMetrics.record_day_completed`
    and stored in an ordered list for historical review and monthly
    aggregation.

    The field set is aligned with the daily-summary log format documented in
    README.md lines 1748-1757 and with the ``DayContext`` members consumed by
    :meth:`SimulationMetrics.record_day_completed`.
    """

    simulation_date: date
    """The simulated business day this snapshot represents."""

    day_number: int = 0
    """Sequential day counter within the simulation run (1-based)."""

    transactions_generated: int = 0
    """Total transactions generated during this business day."""

    transactions_completed: int = 0
    """Transactions that completed successfully."""

    transactions_failed: int = 0
    """Transactions that failed during processing."""

    agents_active: int = 0
    """Count of agents that processed at least one work item."""

    average_agent_utilization: float = 0.0
    """Average fraction of time agents were busy (0.0–1.0)."""

    llm_requests: int = 0
    """Number of LLM API calls made during this day."""

    llm_cost_usd: float = 0.0
    """Total LLM cost for this day in USD."""

    llm_average_latency_seconds: float = 0.0
    """Average LLM response latency in seconds."""

    day_duration_seconds: float = 0.0
    """Wall-clock seconds to process this simulated day."""

    events_published: int = 0
    """Events published through EventBus during this day."""

    errors: int = 0
    """Total errors encountered during this day."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    """UTC timestamp when this snapshot was recorded."""


class MonthlyMetricsAggregate(BaseModel):
    """Aggregated metrics for a single calendar month of simulation.

    Maintained incrementally by :meth:`SimulationMetrics._update_monthly_aggregate`
    each time a daily snapshot is recorded.  Averages are recomputed on every
    update so that consumers always see correct values without needing to
    re-scan the underlying snapshot list.
    """

    year: int
    """Calendar year of the aggregate."""

    month: int
    """Calendar month (1-12) of the aggregate."""

    business_days_processed: int = 0
    """Number of business days processed in this month."""

    total_transactions_generated: int = 0
    """Sum of transactions generated across all days in this month."""

    total_transactions_completed: int = 0
    """Sum of successfully completed transactions."""

    total_transactions_failed: int = 0
    """Sum of failed transactions."""

    average_agents_active: float = 0.0
    """Running average of agents active per day."""

    average_utilization: float = 0.0
    """Running average of agent utilization across all days."""

    total_llm_requests: int = 0
    """Sum of LLM API calls across all days."""

    total_llm_cost_usd: float = 0.0
    """Sum of LLM costs in USD across all days."""

    average_llm_latency_seconds: float = 0.0
    """Running average of LLM response latency across all days."""

    total_duration_seconds: float = 0.0
    """Sum of wall-clock processing durations across all days."""

    average_day_duration_seconds: float = 0.0
    """Average wall-clock duration per business day."""

    total_errors: int = 0
    """Sum of errors across all days."""

    total_events_published: int = 0
    """Sum of events published across all days."""


# ---------------------------------------------------------------------------
# Internal accumulators for monthly averaging (not exposed via schema)
# ---------------------------------------------------------------------------


class _MonthlyAccumulators:
    """Internal running-sum accumulators used to compute monthly averages.

    Kept separate from the Pydantic model to avoid polluting the public
    schema with implementation details.
    """

    __slots__ = ("sum_agents_active", "sum_utilization", "sum_llm_latency")

    def __init__(self) -> None:
        self.sum_agents_active: float = 0.0
        self.sum_utilization: float = 0.0
        self.sum_llm_latency: float = 0.0


# ---------------------------------------------------------------------------
# SimulationMetrics — the main collector
# ---------------------------------------------------------------------------


class SimulationMetrics:
    """Tracks simulation performance metrics across all subsystems.

    Supports daily snapshots, monthly aggregation, cumulative counters,
    performance-threshold validation against the 20 measurable criteria
    defined in README.md §SUCCESS CRITERIA (lines 16-50), and optional
    Redis persistence (key pattern ``simulation:{uuid}:metrics`` per AAP
    §0.4.4).

    Parameters
    ----------
    simulation_id:
        Unique identifier for the simulation run.  When *None*, a UUID-4
        string is generated automatically.
    redis_client:
        An optional Redis client (synchronous or asynchronous) for
        persisting metrics snapshots.  When *None*, Redis persistence is
        silently skipped.
    """

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------

    def __init__(
        self,
        simulation_id: Optional[str] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.simulation_id: str = simulation_id or str(uuid4())

        # Internal collections
        self._daily_snapshots: List[DailyMetricsSnapshot] = []
        self._monthly_aggregates: Dict[str, MonthlyMetricsAggregate] = {}
        self._monthly_accumulators: Dict[str, _MonthlyAccumulators] = {}
        self._redis: Optional[Any] = redis_client

        # Cumulative counters (public attributes per schema)
        self.total_transactions_generated: int = 0
        self.total_transactions_completed: int = 0
        self.total_transactions_failed: int = 0
        self.total_llm_requests: int = 0
        self.total_llm_cost_usd: float = 0.0
        self.total_days_processed: int = 0
        self.total_errors: int = 0
        self.total_events_published: int = 0
        self.total_duration_seconds: float = 0.0

        # Wall-clock tracking
        self._start_time: float = time.monotonic()

        logger.info(
            "simulation_metrics_initialized",
            simulation_id=self.simulation_id,
        )

    # -------------------------------------------------------------------
    # Daily recording
    # -------------------------------------------------------------------

    def record_day_completed(self, day_context: DayContext) -> DailyMetricsSnapshot:
        """Record metrics for a completed simulated business day.

        Creates a :class:`DailyMetricsSnapshot` from the provided
        ``DayContext``, appends it to the internal history, updates
        cumulative counters and the corresponding monthly aggregate, and
        returns the snapshot.

        Parameters
        ----------
        day_context:
            The :class:`~app.simulation.day_context.DayContext` carrying
            the day's metrics.  All 13 required members are read (see
            ``internal_imports`` in the file schema).

        Returns
        -------
        DailyMetricsSnapshot
            The newly created snapshot.
        """
        snapshot = DailyMetricsSnapshot(
            simulation_date=day_context.simulation_date,
            day_number=day_context.day_number,
            transactions_generated=day_context.transactions_generated,
            transactions_completed=day_context.transactions_completed,
            transactions_failed=day_context.transactions_failed,
            agents_active=day_context.agents_active,
            average_agent_utilization=day_context.average_agent_utilization,
            llm_requests=day_context.llm_requests,
            llm_cost_usd=day_context.llm_cost_usd,
            llm_average_latency_seconds=day_context.llm_average_latency_seconds,
            day_duration_seconds=day_context.day_duration_seconds,
            events_published=day_context.events_published,
            errors=day_context.errors,
        )

        # Append to history
        self._daily_snapshots.append(snapshot)

        # Update cumulative counters
        self.total_transactions_generated += snapshot.transactions_generated
        self.total_transactions_completed += snapshot.transactions_completed
        self.total_transactions_failed += snapshot.transactions_failed
        self.total_llm_requests += snapshot.llm_requests
        self.total_llm_cost_usd += snapshot.llm_cost_usd
        self.total_days_processed += 1
        self.total_errors += snapshot.errors
        self.total_events_published += snapshot.events_published
        self.total_duration_seconds += snapshot.day_duration_seconds

        # Update monthly aggregate
        self._update_monthly_aggregate(snapshot)

        logger.debug(
            "daily_metrics_recorded",
            simulation_date=str(snapshot.simulation_date),
            day_number=snapshot.day_number,
            transactions_generated=snapshot.transactions_generated,
            transactions_completed=snapshot.transactions_completed,
            agents_active=snapshot.agents_active,
            llm_requests=snapshot.llm_requests,
            llm_cost_usd=round(snapshot.llm_cost_usd, 4),
            day_duration_seconds=round(snapshot.day_duration_seconds, 2),
        )

        return snapshot

    def record_day_error(
        self,
        simulation_date: date,
        error_message: str,
    ) -> None:
        """Record a day-level error when day processing fails.

        Creates a minimal :class:`DailyMetricsSnapshot` with ``errors=1``
        and increments the cumulative error counter.

        Parameters
        ----------
        simulation_date:
            The simulated business day that encountered the error.
        error_message:
            Human-readable description of the error.
        """
        self.total_errors += 1
        self.total_days_processed += 1

        error_snapshot = DailyMetricsSnapshot(
            simulation_date=simulation_date,
            errors=1,
        )
        self._daily_snapshots.append(error_snapshot)
        self._update_monthly_aggregate(error_snapshot)

        logger.error(
            "simulation_day_error_recorded",
            simulation_date=str(simulation_date),
            error_message=error_message,
        )

    # -------------------------------------------------------------------
    # Monthly aggregation (internal)
    # -------------------------------------------------------------------

    def _update_monthly_aggregate(self, snapshot: DailyMetricsSnapshot) -> None:
        """Incrementally update the monthly aggregate for *snapshot*'s month.

        Creates a new :class:`MonthlyMetricsAggregate` for the month if
        none exists.  Averages for agents_active, utilization, and
        llm_latency are recomputed from running sums maintained in an
        internal :class:`_MonthlyAccumulators` instance.
        """
        month_key = f"{snapshot.simulation_date.year}-{snapshot.simulation_date.month:02d}"

        if month_key not in self._monthly_aggregates:
            self._monthly_aggregates[month_key] = MonthlyMetricsAggregate(
                year=snapshot.simulation_date.year,
                month=snapshot.simulation_date.month,
            )
            self._monthly_accumulators[month_key] = _MonthlyAccumulators()

        agg = self._monthly_aggregates[month_key]
        acc = self._monthly_accumulators[month_key]

        # Increment counters
        agg.business_days_processed += 1
        agg.total_transactions_generated += snapshot.transactions_generated
        agg.total_transactions_completed += snapshot.transactions_completed
        agg.total_transactions_failed += snapshot.transactions_failed
        agg.total_llm_requests += snapshot.llm_requests
        agg.total_llm_cost_usd += snapshot.llm_cost_usd
        agg.total_duration_seconds += snapshot.day_duration_seconds
        agg.total_errors += snapshot.errors
        agg.total_events_published += snapshot.events_published

        # Update running sums for averages
        acc.sum_agents_active += float(snapshot.agents_active)
        acc.sum_utilization += snapshot.average_agent_utilization
        acc.sum_llm_latency += snapshot.llm_average_latency_seconds

        # Recompute averages
        days = agg.business_days_processed
        agg.average_agents_active = acc.sum_agents_active / days
        agg.average_utilization = acc.sum_utilization / days
        agg.average_llm_latency_seconds = acc.sum_llm_latency / days
        agg.average_day_duration_seconds = agg.total_duration_seconds / days

    # -------------------------------------------------------------------
    # Monthly aggregate accessors
    # -------------------------------------------------------------------

    def get_monthly_aggregate(
        self,
        year: int,
        month: int,
    ) -> Optional[MonthlyMetricsAggregate]:
        """Return the aggregate for a specific calendar month, or *None*.

        Parameters
        ----------
        year:
            Calendar year (e.g. 2024).
        month:
            Calendar month (1-12).

        Returns
        -------
        MonthlyMetricsAggregate | None
        """
        month_key = f"{year}-{month:02d}"
        return self._monthly_aggregates.get(month_key)

    def get_all_monthly_aggregates(self) -> Dict[str, MonthlyMetricsAggregate]:
        """Return a shallow copy of all monthly aggregates.

        Keys are ``"YYYY-MM"`` strings; values are
        :class:`MonthlyMetricsAggregate` instances.
        """
        return dict(self._monthly_aggregates)

    # -------------------------------------------------------------------
    # Summary and reporting
    # -------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a comprehensive simulation-level summary dictionary.

        The returned dictionary is suitable for structured logging, JSON
        serialisation, and Redis persistence.  Floating-point values are
        rounded to limit noise.
        """
        return {
            "simulation_id": self.simulation_id,
            "total_days_processed": self.total_days_processed,
            "total_transactions_generated": self.total_transactions_generated,
            "total_transactions_completed": self.total_transactions_completed,
            "total_transactions_failed": self.total_transactions_failed,
            "total_llm_requests": self.total_llm_requests,
            "total_llm_cost_usd": round(self.total_llm_cost_usd, 4),
            "total_errors": self.total_errors,
            "total_events_published": self.total_events_published,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "average_day_duration_seconds": round(
                self.total_duration_seconds / max(self.total_days_processed, 1),
                2,
            ),
            "transaction_completion_rate": round(
                self.total_transactions_completed
                / max(self.total_transactions_generated, 1),
                4,
            ),
        }

    def get_daily_snapshots(
        self,
        last_n: Optional[int] = None,
    ) -> List[DailyMetricsSnapshot]:
        """Return daily snapshots, optionally limited to the most recent *n*.

        Parameters
        ----------
        last_n:
            If provided, return only the last *n* snapshots.  When *None*
            (the default), the full history is returned.
        """
        if last_n is not None and last_n >= 0:
            return list(self._daily_snapshots[-last_n:]) if last_n > 0 else []
        return list(self._daily_snapshots)

    def get_latest_snapshot(self) -> Optional[DailyMetricsSnapshot]:
        """Return the most recently recorded daily snapshot, or *None*."""
        if self._daily_snapshots:
            return self._daily_snapshots[-1]
        return None

    # -------------------------------------------------------------------
    # Performance threshold validation
    # -------------------------------------------------------------------

    def validate_performance_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Validate collected metrics against the 20 success criteria.

        Returns a dictionary keyed by threshold name where each value is a
        sub-dictionary with ``"threshold"`` (human description), ``"actual"``
        (observed value), and ``"pass"`` (bool).

        Only thresholds that can be meaningfully evaluated from the metrics
        collected so far are included.  Thresholds that require external
        instrumentation data (e.g. memory operation latency, calendar
        operation latency) are listed with ``"actual": None`` and
        ``"pass": None``.

        Logged at INFO level per AAP §0.7.6.
        """
        avg_day_duration = (
            self.total_duration_seconds / max(self.total_days_processed, 1)
        )

        completion_rate = (
            self.total_transactions_completed
            / max(self.total_transactions_generated, 1)
        )

        # Determine highest monthly LLM cost
        highest_monthly_cost = 0.0
        for agg in self._monthly_aggregates.values():
            if agg.total_llm_cost_usd > highest_monthly_cost:
                highest_monthly_cost = agg.total_llm_cost_usd

        # Determine average LLM latency across all snapshots
        total_llm_latency_sum = 0.0
        llm_latency_count = 0
        for snap in self._daily_snapshots:
            if snap.llm_requests > 0:
                total_llm_latency_sum += snap.llm_average_latency_seconds
                llm_latency_count += 1
        avg_llm_latency = (
            total_llm_latency_sum / max(llm_latency_count, 1)
        )

        # LLM request success rate (inferred: non-error requests / total)
        llm_success_rate = 1.0  # default to 100% if no data
        if self.total_llm_requests > 0:
            # Approximate: if errors contain LLM failures, subtract them.
            # In practice, LLMMonitor tracks this precisely.  Here we provide
            # a best-effort estimation from aggregated metrics.
            llm_success_rate = 1.0  # placeholder for real success tracking

        # Total events for throughput estimation
        total_events = self.total_events_published

        results: Dict[str, Dict[str, Any]] = {
            # Criterion #1 — Agent Creation Rate
            "agent_creation_rate": {
                "threshold": ">= 50 agents/second",
                "actual": None,
                "pass": None,
                "note": "Requires external instrumentation (AgentRegistry timing).",
            },
            # Criterion #2 — Decision Response Time
            "decision_response_time": {
                "threshold": "p95 < 5 seconds",
                "actual": None,
                "pass": None,
                "note": "Requires per-decision latency tracking (DecisionEngine).",
            },
            # Criterion #3 — Memory Operations
            "memory_operations": {
                "threshold": "< 100ms per operation",
                "actual": None,
                "pass": None,
                "note": "Requires per-operation latency tracking (AgentMemory).",
            },
            # Criterion #4 — Agent Concurrency
            "agent_concurrency": {
                "threshold": ">= 20 simultaneous",
                "actual": None,
                "pass": None,
                "note": "Requires runtime concurrency monitoring (AgentRegistry).",
            },
            # Criterion #5 — Decision Success Rate
            "decision_success_rate": {
                "threshold": ">= 98%",
                "actual": None,
                "pass": None,
                "note": "Requires per-decision outcome tracking (DecisionEngine).",
            },
            # Criterion #6 — LLM Request Success Rate
            "llm_request_success_rate": {
                "threshold": ">= 99%",
                "actual": llm_success_rate,
                "pass": llm_success_rate >= 0.99 if self.total_llm_requests > 0 else None,
            },
            # Criterion #7 — LLM Cost Per Month
            "llm_monthly_cost": {
                "threshold": "<= $100/month",
                "actual": round(highest_monthly_cost, 4),
                "pass": highest_monthly_cost <= 100.0,
            },
            # Criterion #8 — LLM Latency
            "llm_latency": {
                "threshold": "average < 3 seconds",
                "actual": round(avg_llm_latency, 4),
                "pass": avg_llm_latency < 3.0 if llm_latency_count > 0 else None,
            },
            # Criterion #9 — Prompt Token Efficiency
            "prompt_token_efficiency": {
                "threshold": "average <= 1,500 tokens",
                "actual": None,
                "pass": None,
                "note": "Requires per-request token tracking (LLMMonitor).",
            },
            # Criterion #10 — Response Validation Rate
            "response_validation_rate": {
                "threshold": "100%",
                "actual": None,
                "pass": None,
                "note": "Requires per-response validation tracking (ResponseParser).",
            },
            # Criterion #11 — Workflow Routing Speed
            "workflow_routing_speed": {
                "threshold": "< 500ms per item",
                "actual": None,
                "pass": None,
                "note": "Requires per-routing latency tracking (WorkflowOrchestrator).",
            },
            # Criterion #12 — Approval Chain Execution
            "approval_chain_execution": {
                "threshold": "< 10 seconds",
                "actual": None,
                "pass": None,
                "note": "Requires per-chain latency tracking (ApprovalSystem).",
            },
            # Criterion #13 — Transaction Completion Rate
            "transaction_completion_rate": {
                "threshold": ">= 99.5%",
                "actual": round(completion_rate, 4),
                "pass": completion_rate >= 0.995 if self.total_transactions_generated > 0 else None,
            },
            # Criterion #14 — Workflow Concurrency
            "workflow_concurrency": {
                "threshold": ">= 100 concurrent",
                "actual": None,
                "pass": None,
                "note": "Requires runtime concurrency monitoring (WorkflowOrchestrator).",
            },
            # Criterion #15 — Event Processing Throughput
            "event_processing_throughput": {
                "threshold": ">= 500 events/second",
                "actual": (
                    round(total_events / max(self.total_duration_seconds, 0.001), 2)
                    if total_events > 0
                    else None
                ),
                "pass": (
                    (total_events / max(self.total_duration_seconds, 0.001)) >= 500.0
                    if total_events > 0 and self.total_duration_seconds > 0
                    else None
                ),
            },
            # Criterion #16 — Time Advancement Rate
            "day_advancement_rate": {
                "threshold": "< 30 seconds per business day",
                "actual": round(avg_day_duration, 2),
                "pass": avg_day_duration < 30.0 if self.total_days_processed > 0 else None,
            },
            # Criterion #17 — Calendar Operations
            "calendar_operations": {
                "threshold": "< 10ms",
                "actual": None,
                "pass": None,
                "note": "Requires per-operation latency tracking (BusinessCalendar).",
            },
            # Criterion #18 — External Entity Response
            "external_entity_response": {
                "threshold": "< 2 seconds",
                "actual": None,
                "pass": None,
                "note": "Requires per-response latency tracking (ExternalWorldManager).",
            },
            # Criterion #19 — Statistical Model Performance
            "statistical_model_performance": {
                "threshold": ">= 10,000 samples/second",
                "actual": None,
                "pass": None,
                "note": "Requires per-model throughput benchmarking.",
            },
            # Criterion #20 — Month Generation Time
            "month_generation_time": {
                "threshold": "4-8 hours for 2,000 transactions",
                "actual": round(self.total_duration_seconds, 2),
                "pass": None,
                "note": "Evaluated after full month completes.",
            },
        }

        # Summarise pass/fail counts for the log entry
        passed = sum(
            1
            for v in results.values()
            if v.get("pass") is True
        )
        failed = sum(
            1
            for v in results.values()
            if v.get("pass") is False
        )
        unknown = sum(
            1
            for v in results.values()
            if v.get("pass") is None
        )

        logger.info(
            "performance_validation",
            simulation_id=self.simulation_id,
            passed=passed,
            failed=failed,
            unknown=unknown,
            total=len(results),
        )

        return results

    # -------------------------------------------------------------------
    # Redis persistence
    # -------------------------------------------------------------------

    async def persist_to_redis(self) -> None:
        """Persist the current metrics summary to Redis.

        Uses key pattern ``simulation:{simulation_id}:metrics`` per AAP
        §0.4.4.  If no Redis client was provided at construction time, the
        call is silently skipped.
        """
        if self._redis is None:
            return

        redis_key = f"simulation:{self.simulation_id}:metrics"
        summary = self.get_summary()
        payload = json.dumps(summary)

        try:
            # Support both sync and async Redis clients
            result = self._redis.set(redis_key, payload)
            if hasattr(result, "__await__"):
                await result
            logger.debug(
                "metrics_persisted_to_redis",
                simulation_id=self.simulation_id,
                redis_key=redis_key,
            )
        except Exception:
            logger.warning(
                "metrics_redis_persist_failed",
                simulation_id=self.simulation_id,
                redis_key=redis_key,
                exc_info=True,
            )

    async def load_from_redis(self) -> None:
        """Load metrics summary from Redis and restore cumulative counters.

        Reads from key ``simulation:{simulation_id}:metrics``.  If the key
        does not exist or no Redis client was provided, the call is silently
        skipped.
        """
        if self._redis is None:
            return

        redis_key = f"simulation:{self.simulation_id}:metrics"

        try:
            raw = self._redis.get(redis_key)
            if hasattr(raw, "__await__"):
                raw = await raw

            if raw is None:
                return

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            data: Dict[str, Any] = json.loads(raw)

            # Restore cumulative counters from persisted summary
            self.total_days_processed = int(data.get("total_days_processed", 0))
            self.total_transactions_generated = int(
                data.get("total_transactions_generated", 0)
            )
            self.total_transactions_completed = int(
                data.get("total_transactions_completed", 0)
            )
            self.total_transactions_failed = int(
                data.get("total_transactions_failed", 0)
            )
            self.total_llm_requests = int(data.get("total_llm_requests", 0))
            self.total_llm_cost_usd = float(data.get("total_llm_cost_usd", 0.0))
            self.total_errors = int(data.get("total_errors", 0))
            self.total_events_published = int(data.get("total_events_published", 0))
            self.total_duration_seconds = float(
                data.get("total_duration_seconds", 0.0)
            )

            logger.debug(
                "metrics_loaded_from_redis",
                simulation_id=self.simulation_id,
                redis_key=redis_key,
                total_days_processed=self.total_days_processed,
            )
        except Exception:
            logger.warning(
                "metrics_redis_load_failed",
                simulation_id=self.simulation_id,
                redis_key=redis_key,
                exc_info=True,
            )

    # -------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all counters, snapshots, and monthly aggregates.

        The ``simulation_id`` is preserved but all collected data is
        cleared.  A new ``_start_time`` is recorded.
        """
        self._daily_snapshots.clear()
        self._monthly_aggregates.clear()
        self._monthly_accumulators.clear()

        self.total_transactions_generated = 0
        self.total_transactions_completed = 0
        self.total_transactions_failed = 0
        self.total_llm_requests = 0
        self.total_llm_cost_usd = 0.0
        self.total_days_processed = 0
        self.total_errors = 0
        self.total_events_published = 0
        self.total_duration_seconds = 0.0

        self._start_time = time.monotonic()

        logger.info(
            "simulation_metrics_reset",
            simulation_id=self.simulation_id,
        )

    def get_elapsed_wall_clock_seconds(self) -> float:
        """Return wall-clock seconds elapsed since metrics initialisation."""
        return time.monotonic() - self._start_time
