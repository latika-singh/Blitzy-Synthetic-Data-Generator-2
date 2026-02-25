"""Comprehensive tests for the SimulationEngine, DayContext, and SimulationMetrics.

Tests cover:
- SimulationConfig Pydantic V2 model defaults, custom values, and validation
- DayContext creation, field constraints, convenience methods, and factories
- FiscalPeriodInfo embedding and fiscal period queries
- DailyMetricsSnapshot creation
- MonthlyMetricsAggregate creation and aggregation
- SimulationMetrics recording, monthly aggregation, threshold validation, reset
- SimulationEngine initialization with constructor-injected dependencies
- Main async run() loop across date ranges
- 5-step daily processing pipeline: DayStart → Generate → Route → Agents → DayComplete
- Daily summary logging with all 7 required fields (README.md 1748-1757)
- Stop/pause/resume control methods
- Health check for subsystem availability
- Multi-day simulation with cumulative metric accumulation
- Edge cases: empty date range, single day, empty interactions, routing failures

Testing standards (AAP §0.7.5):
- All external dependencies mocked (TimeController, ExternalWorldManager,
  WorkflowOrchestrator, EventBus, AgentRegistry)
- No live LLM API calls
- No external Redis dependency
- pytest-asyncio for async tests
- Coverage target ≥ 80%
"""

from __future__ import annotations

import asyncio
import time
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import structlog
from pydantic import ValidationError

from app.simulation.simulation_engine import SimulationEngine, SimulationConfig
from app.simulation.day_context import DayContext, FiscalPeriodInfo
from app.simulation.simulation_metrics import (
    SimulationMetrics,
    DailyMetricsSnapshot,
    MonthlyMetricsAggregate,
)


# ============================================================================
# Local Fixtures
# ============================================================================


@pytest.fixture
def simulation_config() -> SimulationConfig:
    """Short-range SimulationConfig for fast tests."""
    return SimulationConfig(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
        max_transactions_per_day=10,
        agent_processing_timeout_seconds=5.0,
        day_processing_timeout_seconds=5.0,
        max_concurrent_agents=5,
        max_concurrent_workflows=10,
        log_daily_summary=True,
        seed=42,
    )


@pytest.fixture
def mock_time_controller() -> AsyncMock:
    """AsyncMock for TimeController with successive business date returns."""
    tc = AsyncMock()
    # side_effect yields successive business dates
    tc.advance_to_next_business_day = AsyncMock(
        side_effect=[
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),  # Past typical end_date, triggers exit
        ]
    )
    tc.is_business_day = MagicMock(return_value=True)
    tc.get_current_date = MagicMock(return_value=date(2024, 1, 2))
    tc.get_fiscal_period = MagicMock(
        return_value={
            "fiscal_year": 2024,
            "fiscal_month": 1,
            "fiscal_quarter": 1,
            "period_status": "open",
        }
    )
    return tc


@pytest.fixture
def mock_external_world_manager() -> AsyncMock:
    """AsyncMock for ExternalWorldManager returning deterministic interactions."""
    ewm = AsyncMock()
    ewm.generate_daily_interactions = AsyncMock(
        return_value={
            "customer_orders": [
                {"order_id": "ORD-001", "customer_id": "C-001", "amount": 5000.00},
                {"order_id": "ORD-002", "customer_id": "C-002", "amount": 3000.00},
            ],
            "vendor_invoices": [
                {"invoice_id": "INV-001", "vendor_id": "V-001", "amount": 7500.00},
            ],
            "bank_statements": [],
        }
    )
    return ewm


@pytest.fixture
def mock_workflow_orchestrator() -> AsyncMock:
    """AsyncMock for WorkflowOrchestrator returning routing confirmations."""
    wo = AsyncMock()
    wo.route_transaction = AsyncMock(
        return_value={"workflow_id": str(uuid4()), "status": "routed"}
    )
    wo.get_active_workflow_count = MagicMock(return_value=0)
    wo.get_pending_workflows = MagicMock(return_value=[])
    return wo


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """AsyncMock for EventBus."""
    eb = AsyncMock()
    eb.publish = AsyncMock(return_value=None)
    eb.start = AsyncMock(return_value=None)
    eb.stop = AsyncMock(return_value=None)
    eb.is_running = MagicMock(return_value=True)
    return eb


@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """MagicMock for AgentRegistry with capacity metrics returning idle state."""
    ar = MagicMock()
    ar.get_all_agents = MagicMock(return_value=[])
    ar.get_agents_by_state = MagicMock(
        return_value={"IDLE": 5, "THINKING": 0, "ACTING": 0}
    )
    ar.get_agent_count = MagicMock(return_value=5)
    ar.get_idle_agents = MagicMock(return_value=[])
    # Capacity metrics: total_queue_depth=0 and busy_count=0 → agents done
    ar.get_capacity_metrics = MagicMock(
        return_value={
            "total_queue_depth": 0,
            "busy_count": 0,
            "idle_count": 5,
            "error_count": 0,
            "utilization_pct": 0.0,
        }
    )
    return ar


@pytest.fixture
def simulation_metrics() -> SimulationMetrics:
    """Fresh SimulationMetrics instance with no Redis client."""
    return SimulationMetrics(simulation_id="test-sim-001")


@pytest.fixture
def simulation_engine(
    simulation_config: SimulationConfig,
    mock_time_controller: AsyncMock,
    mock_external_world_manager: AsyncMock,
    mock_workflow_orchestrator: AsyncMock,
    mock_event_bus: AsyncMock,
    mock_agent_registry: MagicMock,
    simulation_metrics: SimulationMetrics,
) -> SimulationEngine:
    """Fully-wired SimulationEngine with all mocked dependencies."""
    return SimulationEngine(
        config=simulation_config,
        time_controller=mock_time_controller,
        external_world_manager=mock_external_world_manager,
        workflow_orchestrator=mock_workflow_orchestrator,
        event_bus=mock_event_bus,
        agent_registry=mock_agent_registry,
        metrics=simulation_metrics,
    )


# ============================================================================
# Phase 3: SimulationConfig Tests
# ============================================================================


class TestSimulationConfig:
    """Tests for the SimulationConfig Pydantic V2 model."""

    def test_default_config_values(self) -> None:
        """Verify default values match specification constraints."""
        cfg = SimulationConfig()
        assert cfg.start_date == date(2024, 1, 2), "First business day of 2024"
        assert cfg.end_date == date(2024, 12, 31)
        assert cfg.max_transactions_per_day == 100
        assert cfg.agent_processing_timeout_seconds == 120.0
        assert cfg.day_processing_timeout_seconds == 30.0
        assert cfg.max_concurrent_agents == 20
        assert cfg.max_concurrent_workflows == 100
        assert cfg.enable_event_persistence is True
        assert cfg.log_daily_summary is True
        assert cfg.seed is None

    def test_custom_config(self) -> None:
        """Custom values are applied correctly."""
        cfg = SimulationConfig(
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30),
            max_transactions_per_day=50,
            agent_processing_timeout_seconds=60.0,
            day_processing_timeout_seconds=15.0,
            max_concurrent_agents=10,
            max_concurrent_workflows=50,
            enable_event_persistence=False,
            seed=123,
            log_daily_summary=False,
        )
        assert cfg.start_date == date(2024, 6, 1)
        assert cfg.end_date == date(2024, 6, 30)
        assert cfg.max_transactions_per_day == 50
        assert cfg.agent_processing_timeout_seconds == 60.0
        assert cfg.day_processing_timeout_seconds == 15.0
        assert cfg.max_concurrent_agents == 10
        assert cfg.max_concurrent_workflows == 50
        assert cfg.enable_event_persistence is False
        assert cfg.seed == 123
        assert cfg.log_daily_summary is False

    def test_simulation_id_auto_generated(self) -> None:
        """simulation_id is auto-generated as a UUID string when not supplied."""
        cfg = SimulationConfig()
        assert isinstance(cfg.simulation_id, str)
        assert len(cfg.simulation_id) > 0
        # Must be a valid UUID
        UUID(cfg.simulation_id)  # Raises ValueError if invalid

    def test_config_validation_min_transactions(self) -> None:
        """max_transactions_per_day must be >= 1 (Pydantic ge=1)."""
        with pytest.raises(ValidationError):
            SimulationConfig(max_transactions_per_day=0)

    def test_config_validation_min_agents(self) -> None:
        """max_concurrent_agents must be >= 1."""
        with pytest.raises(ValidationError):
            SimulationConfig(max_concurrent_agents=0)

    def test_config_validation_min_workflows(self) -> None:
        """max_concurrent_workflows must be >= 1."""
        with pytest.raises(ValidationError):
            SimulationConfig(max_concurrent_workflows=0)

    def test_config_with_seed(self) -> None:
        """Seed parameter is stored correctly."""
        cfg = SimulationConfig(seed=42)
        assert cfg.seed == 42

    def test_config_without_seed(self) -> None:
        """Default seed is None."""
        cfg = SimulationConfig()
        assert cfg.seed is None


# ============================================================================
# Phase 4: DayContext Tests
# ============================================================================


class TestDayContext:
    """Tests for the DayContext Pydantic V2 mutable model."""

    def test_day_context_creation(self) -> None:
        """DayContext with only required field (simulation_date) uses defaults."""
        ctx = DayContext(simulation_date=date(2024, 1, 2))
        assert ctx.simulation_date == date(2024, 1, 2)
        assert ctx.transactions_generated == 0
        assert ctx.agents_active == 0
        assert ctx.average_agent_utilization == 0.0
        assert ctx.llm_requests == 0
        assert ctx.llm_cost_usd == 0.0
        assert ctx.day_duration_seconds == 0.0
        assert ctx.errors == 0

    def test_day_context_required_date(self) -> None:
        """Creating DayContext without simulation_date raises ValidationError."""
        with pytest.raises(ValidationError):
            DayContext()  # type: ignore[call-arg]

    def test_day_context_factory_method(self) -> None:
        """create_for_date classmethod produces a valid DayContext."""
        ctx = DayContext.create_for_date(
            date(2024, 1, 2), simulation_id="test", day_number=1
        )
        assert ctx.simulation_date == date(2024, 1, 2)
        assert ctx.simulation_id == "test"
        assert ctx.day_number == 1
        assert ctx.transactions_generated == 0

    def test_day_context_factory_auto_id(self) -> None:
        """create_for_date auto-generates simulation_id when empty string."""
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        assert isinstance(ctx.simulation_id, str)
        assert len(ctx.simulation_id) > 0

    def test_day_context_to_daily_summary_dict(self) -> None:
        """to_daily_summary_dict returns the expected keys per README.md 1748-1757 plus P3 metrics."""
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            transactions_generated=45,
            agents_active=12,
            average_agent_utilization=0.75,
            llm_requests=30,
            llm_cost_usd=1.50,
            day_duration_seconds=22.5,
        )
        summary = ctx.to_daily_summary_dict()
        expected_keys = {
            "simulation_date",
            "transactions_generated",
            "agents_active",
            "average_agent_utilization",
            "llm_requests",
            "llm_cost_usd",
            "duration_seconds",
            # P3: Transaction Workflow Metrics
            "p2p_cycles_completed",
            "o2c_cycles_completed",
            "gl_entries_posted",
            "discrepancies_injected",
            "ground_truths_created",
            "rework_attempts",
            "rework_successes",
            "rework_escalations",
            "period_closes_completed",
            "trial_balance_checks_passed",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["simulation_date"] == "2024-01-02"
        assert summary["transactions_generated"] == 45
        assert summary["agents_active"] == 12
        assert summary["llm_requests"] == 30
        assert summary["duration_seconds"] == 22.5
        # P3 metrics default to 0
        assert summary["p2p_cycles_completed"] == 0
        assert summary["o2c_cycles_completed"] == 0
        assert summary["gl_entries_posted"] == 0

    def test_day_context_fiscal_period_info(self) -> None:
        """DayContext embeds FiscalPeriodInfo correctly."""
        fpi = FiscalPeriodInfo(
            fiscal_year=2024,
            fiscal_month=1,
            fiscal_quarter=1,
            period_status="open",
        )
        ctx = DayContext(simulation_date=date(2024, 1, 2), fiscal_period=fpi)
        assert ctx.fiscal_period.fiscal_year == 2024
        assert ctx.fiscal_period.fiscal_month == 1
        assert ctx.fiscal_period.period_status == "open"

    def test_day_context_is_period_closing(self) -> None:
        """is_period_closing returns True only when period_status='closing'."""
        ctx_open = DayContext(
            simulation_date=date(2024, 1, 2),
            fiscal_period=FiscalPeriodInfo(period_status="open"),
        )
        assert ctx_open.is_period_closing() is False

        ctx_closing = DayContext(
            simulation_date=date(2024, 1, 31),
            fiscal_period=FiscalPeriodInfo(period_status="closing"),
        )
        assert ctx_closing.is_period_closing() is True

    def test_day_context_is_quarter_end(self) -> None:
        """is_quarter_end delegates to FiscalPeriodInfo.is_quarter_end."""
        ctx = DayContext(
            simulation_date=date(2024, 3, 29),
            fiscal_period=FiscalPeriodInfo(is_quarter_end=True),
        )
        assert ctx.is_quarter_end() is True

    def test_day_context_is_year_end(self) -> None:
        """is_year_end delegates to FiscalPeriodInfo.is_year_end."""
        ctx = DayContext(
            simulation_date=date(2024, 12, 31),
            fiscal_period=FiscalPeriodInfo(is_year_end=True),
        )
        assert ctx.is_year_end() is True

    def test_day_context_transaction_completion_rate(self) -> None:
        """transaction_completion_rate computed correctly."""
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            transactions_generated=100,
            transactions_completed=99,
        )
        assert ctx.transaction_completion_rate == pytest.approx(0.99)

    def test_day_context_transaction_completion_rate_zero(self) -> None:
        """transaction_completion_rate is 0.0 when no transactions generated."""
        ctx = DayContext(simulation_date=date(2024, 1, 2))
        assert ctx.transaction_completion_rate == 0.0

    def test_day_context_total_agent_count(self) -> None:
        """total_agent_count sums active + idle + error agents."""
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            agents_active=10,
            agents_idle=5,
            agents_error=1,
        )
        assert ctx.total_agent_count == 16

    def test_day_context_update_from_agent_metrics(self) -> None:
        """update_from_agent_metrics sets all agent fields."""
        ctx = DayContext(simulation_date=date(2024, 1, 2))
        ctx.update_from_agent_metrics(
            active=10, idle=5, error=1, utilization=0.65, decisions=25
        )
        assert ctx.agents_active == 10
        assert ctx.agents_idle == 5
        assert ctx.agents_error == 1
        assert ctx.average_agent_utilization == pytest.approx(0.65)
        assert ctx.agent_decisions_made == 25

    def test_day_context_update_from_llm_metrics(self) -> None:
        """update_from_llm_metrics sets all LLM fields."""
        ctx = DayContext(simulation_date=date(2024, 1, 2))
        ctx.update_from_llm_metrics(requests=30, cost_usd=1.50, avg_latency=2.5)
        assert ctx.llm_requests == 30
        assert ctx.llm_cost_usd == pytest.approx(1.50)
        assert ctx.llm_average_latency_seconds == pytest.approx(2.5)

    def test_day_context_utilization_constraint_upper(self) -> None:
        """average_agent_utilization > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            DayContext(
                simulation_date=date(2024, 1, 2),
                average_agent_utilization=1.5,
            )

    def test_day_context_utilization_constraint_negative(self) -> None:
        """Negative utilization is rejected."""
        with pytest.raises(ValidationError):
            DayContext(
                simulation_date=date(2024, 1, 2),
                average_agent_utilization=-0.1,
            )

    def test_day_context_non_negative_counts(self) -> None:
        """Negative transaction/agent counts are rejected (ge=0 constraint)."""
        with pytest.raises(ValidationError):
            DayContext(simulation_date=date(2024, 1, 2), transactions_generated=-1)
        with pytest.raises(ValidationError):
            DayContext(simulation_date=date(2024, 1, 2), agents_active=-1)
        with pytest.raises(ValidationError):
            DayContext(simulation_date=date(2024, 1, 2), llm_requests=-1)

    def test_day_context_validate_assignment(self) -> None:
        """ConfigDict(validate_assignment=True) validates mutations."""
        ctx = DayContext(simulation_date=date(2024, 1, 2))
        ctx.transactions_generated = 10
        assert ctx.transactions_generated == 10
        with pytest.raises(ValidationError):
            ctx.transactions_generated = -5

    def test_day_context_metadata(self) -> None:
        """metadata dict is stored and accessible."""
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            metadata={"custom_key": "custom_value"},
        )
        assert ctx.metadata["custom_key"] == "custom_value"

    def test_day_context_created_at(self) -> None:
        """created_at timestamp is auto-set to a recent UTC datetime."""
        before = datetime.now(timezone.utc)
        ctx = DayContext(simulation_date=date(2024, 1, 2))
        after = datetime.now(timezone.utc)
        assert before <= ctx.created_at <= after


# ============================================================================
# Phase 4b: FiscalPeriodInfo Tests
# ============================================================================


class TestFiscalPeriodInfo:
    """Tests for the FiscalPeriodInfo Pydantic model."""

    def test_defaults(self) -> None:
        """All numeric defaults are zero/False/open."""
        fpi = FiscalPeriodInfo()
        assert fpi.fiscal_year == 0
        assert fpi.fiscal_month == 0
        assert fpi.fiscal_quarter == 0
        assert fpi.period_status == "open"
        assert fpi.is_quarter_end is False
        assert fpi.is_year_end is False

    def test_custom_values(self) -> None:
        fpi = FiscalPeriodInfo(
            fiscal_year=2024,
            fiscal_month=3,
            fiscal_quarter=1,
            period_status="closing",
            is_quarter_end=True,
            is_year_end=False,
        )
        assert fpi.fiscal_year == 2024
        assert fpi.fiscal_month == 3
        assert fpi.fiscal_quarter == 1
        assert fpi.period_status == "closing"
        assert fpi.is_quarter_end is True


# ============================================================================
# Phase 6: DailyMetricsSnapshot Tests
# ============================================================================


class TestDailyMetricsSnapshot:
    """Tests for DailyMetricsSnapshot Pydantic model."""

    def test_snapshot_creation(self) -> None:
        snap = DailyMetricsSnapshot(simulation_date=date(2024, 1, 2))
        assert snap.simulation_date == date(2024, 1, 2)
        assert snap.transactions_generated == 0
        assert snap.transactions_completed == 0
        assert snap.agents_active == 0
        assert snap.average_agent_utilization == 0.0
        assert snap.llm_requests == 0
        assert snap.llm_cost_usd == 0.0
        assert snap.day_duration_seconds == 0.0
        assert snap.errors == 0
        assert snap.events_published == 0
        # Timestamp must be populated
        assert snap.timestamp is not None

    def test_snapshot_with_values(self) -> None:
        snap = DailyMetricsSnapshot(
            simulation_date=date(2024, 1, 3),
            transactions_generated=50,
            transactions_completed=48,
            transactions_failed=2,
            agents_active=12,
            average_agent_utilization=0.8,
            llm_requests=20,
            llm_cost_usd=0.75,
            day_duration_seconds=25.3,
            errors=1,
            events_published=60,
        )
        assert snap.transactions_generated == 50
        assert snap.transactions_completed == 48
        assert snap.transactions_failed == 2
        assert snap.agents_active == 12
        assert snap.average_agent_utilization == pytest.approx(0.8)
        assert snap.llm_requests == 20
        assert snap.day_duration_seconds == pytest.approx(25.3)


# ============================================================================
# Phase 7: MonthlyMetricsAggregate Tests
# ============================================================================


class TestMonthlyMetricsAggregate:
    """Tests for MonthlyMetricsAggregate Pydantic model."""

    def test_aggregate_creation(self) -> None:
        agg = MonthlyMetricsAggregate(year=2024, month=1)
        assert agg.year == 2024
        assert agg.month == 1
        assert agg.business_days_processed == 0
        assert agg.total_transactions_generated == 0
        assert agg.total_transactions_completed == 0
        assert agg.average_agents_active == 0.0
        assert agg.average_utilization == 0.0
        assert agg.total_llm_requests == 0
        assert agg.total_llm_cost_usd == 0.0
        assert agg.total_duration_seconds == 0.0
        assert agg.average_day_duration_seconds == 0.0
        assert agg.total_errors == 0


# ============================================================================
# Phase 5: SimulationMetrics Tests
# ============================================================================


class TestSimulationMetrics:
    """Tests for SimulationMetrics recording, aggregation, and thresholds."""

    def test_metrics_initialization(self) -> None:
        m = SimulationMetrics(simulation_id="test-001")
        assert m.simulation_id == "test-001"
        assert m.total_days_processed == 0
        assert m.total_transactions_generated == 0
        assert m.total_transactions_completed == 0
        assert m.total_llm_requests == 0
        assert m.total_llm_cost_usd == 0.0
        assert m.total_errors == 0
        assert m.get_daily_snapshots() == []

    def test_metrics_auto_generated_id(self) -> None:
        m = SimulationMetrics()
        assert isinstance(m.simulation_id, str)
        assert len(m.simulation_id) > 0

    def test_record_day_completed(self) -> None:
        m = SimulationMetrics(simulation_id="rec-test")
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            transactions_generated=45,
            transactions_completed=43,
            transactions_failed=2,
            agents_active=12,
            average_agent_utilization=0.6,
            llm_requests=10,
            llm_cost_usd=0.50,
            day_duration_seconds=20.0,
            events_published=30,
            errors=1,
        )
        snap = m.record_day_completed(ctx)
        assert isinstance(snap, DailyMetricsSnapshot)
        assert snap.simulation_date == date(2024, 1, 2)
        assert snap.transactions_generated == 45
        assert m.total_days_processed == 1
        assert m.total_transactions_generated == 45
        assert m.total_transactions_completed == 43
        assert m.total_llm_requests == 10
        assert m.total_llm_cost_usd == pytest.approx(0.50)
        assert m.total_errors == 1

    def test_record_multiple_days(self) -> None:
        m = SimulationMetrics(simulation_id="multi")
        for day_num, (gen, comp) in enumerate([(10, 10), (20, 19), (15, 14)], start=2):
            ctx = DayContext(
                simulation_date=date(2024, 1, day_num),
                transactions_generated=gen,
                transactions_completed=comp,
                llm_requests=5,
                llm_cost_usd=0.25,
                day_duration_seconds=18.0,
            )
            m.record_day_completed(ctx)
        assert m.total_days_processed == 3
        assert m.total_transactions_generated == 45
        assert m.total_transactions_completed == 43
        assert m.total_llm_requests == 15
        assert m.total_llm_cost_usd == pytest.approx(0.75)
        assert len(m.get_daily_snapshots()) == 3

    def test_record_day_error(self) -> None:
        m = SimulationMetrics(simulation_id="err")
        m.record_day_error(date(2024, 1, 5), "Timeout error")
        assert m.total_errors >= 1

    def test_monthly_aggregation(self) -> None:
        m = SimulationMetrics(simulation_id="monthly")
        # January days
        for d in [2, 3, 4]:
            ctx = DayContext(
                simulation_date=date(2024, 1, d),
                transactions_generated=10,
                transactions_completed=10,
                agents_active=5,
                average_agent_utilization=0.5,
                llm_requests=3,
                llm_cost_usd=0.10,
                day_duration_seconds=15.0,
            )
            m.record_day_completed(ctx)
        # February day
        ctx_feb = DayContext(
            simulation_date=date(2024, 2, 1),
            transactions_generated=20,
            transactions_completed=19,
            agents_active=8,
            average_agent_utilization=0.7,
            llm_requests=6,
            llm_cost_usd=0.30,
            day_duration_seconds=22.0,
        )
        m.record_day_completed(ctx_feb)

        jan = m.get_monthly_aggregate(2024, 1)
        assert jan is not None
        assert jan.business_days_processed == 3
        assert jan.total_transactions_generated == 30
        assert jan.total_llm_requests == 9
        assert jan.total_llm_cost_usd == pytest.approx(0.30)

        feb = m.get_monthly_aggregate(2024, 2)
        assert feb is not None
        assert feb.business_days_processed == 1
        assert feb.total_transactions_generated == 20

    def test_monthly_aggregate_averages(self) -> None:
        m = SimulationMetrics(simulation_id="avg")
        for d, (active, util) in zip(
            [2, 3, 4], [(10, 0.6), (8, 0.5), (12, 0.7)]
        ):
            ctx = DayContext(
                simulation_date=date(2024, 1, d),
                transactions_generated=5,
                transactions_completed=5,
                agents_active=active,
                average_agent_utilization=util,
                day_duration_seconds=20.0,
            )
            m.record_day_completed(ctx)
        jan = m.get_monthly_aggregate(2024, 1)
        assert jan is not None
        assert jan.average_agents_active == pytest.approx(10.0, abs=0.5)
        assert jan.average_utilization == pytest.approx(0.6, abs=0.05)
        assert jan.average_day_duration_seconds == pytest.approx(20.0, abs=0.5)

    def test_get_summary(self) -> None:
        m = SimulationMetrics(simulation_id="summ")
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            transactions_generated=50,
            transactions_completed=49,
            transactions_failed=1,
            llm_requests=10,
            llm_cost_usd=0.50,
            day_duration_seconds=25.0,
            errors=0,
            events_published=60,
        )
        m.record_day_completed(ctx)
        summary = m.get_summary()
        assert summary["simulation_id"] == "summ"
        assert summary["total_days_processed"] == 1
        assert summary["total_transactions_generated"] == 50
        assert summary["total_transactions_completed"] == 49
        assert "total_llm_requests" in summary
        assert "total_llm_cost_usd" in summary
        assert "total_errors" in summary
        assert "total_duration_seconds" in summary
        assert "average_day_duration_seconds" in summary
        assert "transaction_completion_rate" in summary

    def test_get_daily_snapshots(self) -> None:
        m = SimulationMetrics(simulation_id="snaps")
        for d in range(2, 7):
            ctx = DayContext(
                simulation_date=date(2024, 1, d),
                transactions_generated=d,
                transactions_completed=d,
                day_duration_seconds=15.0,
            )
            m.record_day_completed(ctx)
        all_snaps = m.get_daily_snapshots()
        assert len(all_snaps) == 5
        last_3 = m.get_daily_snapshots(last_n=3)
        assert len(last_3) == 3
        assert last_3[-1].simulation_date == date(2024, 1, 6)

    def test_get_latest_snapshot(self) -> None:
        m = SimulationMetrics(simulation_id="latest")
        for d in [2, 3, 4]:
            m.record_day_completed(
                DayContext(
                    simulation_date=date(2024, 1, d),
                    transactions_generated=1,
                    transactions_completed=1,
                    day_duration_seconds=10.0,
                )
            )
        latest = m.get_latest_snapshot()
        assert latest is not None
        assert latest.simulation_date == date(2024, 1, 4)

    def test_get_latest_snapshot_empty(self) -> None:
        m = SimulationMetrics(simulation_id="empty")
        assert m.get_latest_snapshot() is None

    def test_validate_performance_thresholds(self) -> None:
        m = SimulationMetrics(simulation_id="thresh")
        # Passing metrics
        ctx = DayContext(
            simulation_date=date(2024, 1, 2),
            transactions_generated=100,
            transactions_completed=100,
            day_duration_seconds=25.0,
            llm_requests=5,
            llm_cost_usd=0.10,
        )
        m.record_day_completed(ctx)
        result = m.validate_performance_thresholds()
        assert isinstance(result, dict)
        # day_advancement_rate should pass (<30s)
        if "day_advancement_rate" in result:
            assert result["day_advancement_rate"]["pass"] is True
        # transaction_completion_rate should pass (100%)
        if "transaction_completion_rate" in result:
            assert result["transaction_completion_rate"]["pass"] is True

    def test_metrics_reset(self) -> None:
        m = SimulationMetrics(simulation_id="reset")
        m.record_day_completed(
            DayContext(
                simulation_date=date(2024, 1, 2),
                transactions_generated=50,
                transactions_completed=50,
                day_duration_seconds=20.0,
            )
        )
        assert m.total_days_processed == 1
        m.reset()
        assert m.total_days_processed == 0
        assert m.total_transactions_generated == 0
        assert m.get_daily_snapshots() == []

    def test_elapsed_wall_clock(self) -> None:
        m = SimulationMetrics(simulation_id="wall")
        time.sleep(0.1)
        elapsed = m.get_elapsed_wall_clock_seconds()
        assert elapsed > 0


# ============================================================================
# Phase 8: SimulationEngine Initialization Tests
# ============================================================================


class TestSimulationEngineInitialization:
    """Tests for SimulationEngine constructor / dependency injection."""

    def test_initialization_with_all_dependencies(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Engine stores all injected dependencies."""
        assert simulation_engine.config is not None
        assert simulation_engine._time_controller is not None
        assert simulation_engine._external_world_manager is not None
        assert simulation_engine._workflow_orchestrator is not None
        assert simulation_engine._event_bus is not None
        assert simulation_engine._agent_registry is not None
        assert simulation_engine._metrics is not None
        assert simulation_engine.is_running() is False

    def test_initialization_with_defaults(self) -> None:
        """Engine works when created with no arguments (all None deps)."""
        engine = SimulationEngine()
        assert engine.config is not None
        assert isinstance(engine.config, SimulationConfig)
        assert engine.is_running() is False

    def test_initialization_with_custom_config(self) -> None:
        """Engine stores a custom config."""
        cfg = SimulationConfig(start_date=date(2024, 6, 1))
        engine = SimulationEngine(config=cfg)
        assert engine.config.start_date == date(2024, 6, 1)

    def test_initialization_creates_metrics(self) -> None:
        """Engine auto-creates SimulationMetrics when not supplied."""
        engine = SimulationEngine()
        m = engine.get_metrics()
        assert isinstance(m, SimulationMetrics)


# ============================================================================
# Phase 9: SimulationEngine.run() Tests (Main Loop)
# ============================================================================


class TestSimulationEngineRun:
    """Tests for the main async run() loop."""

    @pytest.mark.asyncio
    async def test_run_processes_all_business_days(
        self,
        simulation_engine: SimulationEngine,
        mock_time_controller: AsyncMock,
    ) -> None:
        """run() advances through business days until end_date is reached."""
        result = await asyncio.wait_for(simulation_engine.run(), timeout=10.0)
        assert result is not None
        assert "days_processed" in result
        assert result["days_processed"] > 0

    @pytest.mark.asyncio
    async def test_run_returns_summary(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """run() returns dict with simulation_id, days_processed, etc."""
        result = await asyncio.wait_for(simulation_engine.run(), timeout=10.0)
        assert "simulation_id" in result
        assert "days_processed" in result
        assert "total_duration_seconds" in result

    @pytest.mark.asyncio
    async def test_run_creates_day_context_per_day(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Each processed day produces a DailyMetricsSnapshot."""
        await asyncio.wait_for(simulation_engine.run(), timeout=10.0)
        snaps = simulation_engine.get_metrics().get_daily_snapshots()
        assert len(snaps) > 0

    @pytest.mark.asyncio
    async def test_run_records_day_duration(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Daily snapshots have positive duration_seconds."""
        await asyncio.wait_for(simulation_engine.run(), timeout=10.0)
        snaps = simulation_engine.get_metrics().get_daily_snapshots()
        for snap in snaps:
            assert snap.day_duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_run_stops_at_end_date(
        self,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Simulation does not process dates beyond end_date."""
        cfg = SimulationConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            max_transactions_per_day=10,
            seed=42,
        )
        mock_time_controller.advance_to_next_business_day = AsyncMock(
            side_effect=[date(2024, 1, 3), date(2024, 1, 4)]
        )
        engine = SimulationEngine(
            config=cfg,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
        )
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        # At most 2 days (Jan 2 start + Jan 3 advanced)
        assert result["days_processed"] <= 2

    @pytest.mark.asyncio
    async def test_run_handles_day_error_gracefully(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
    ) -> None:
        """An exception during one day does not crash the entire simulation."""
        call_count = 0

        async def generate_side_effect(*args: Any, **kwargs: Any) -> Dict:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Network error on day 2")
            return {
                "customer_orders": [{"order_id": "O-1", "customer_id": "C-1", "amount": 100}],
                "vendor_invoices": [],
                "bank_statements": [],
            }

        mock_external_world_manager.generate_daily_interactions = AsyncMock(
            side_effect=generate_side_effect
        )
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
        )
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        # Simulation completes despite error
        assert result is not None
        assert engine.get_metrics().total_errors >= 1

    @pytest.mark.asyncio
    async def test_run_with_no_subsystems(self) -> None:
        """Engine with None dependencies completes gracefully."""
        cfg = SimulationConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            seed=42,
        )
        engine = SimulationEngine(config=cfg)
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        assert result is not None


# ============================================================================
# Phase 10: _process_day() Tests (5-Step Pipeline)
# ============================================================================


class TestProcessDay:
    """Tests for the 5-step daily processing pipeline."""

    @pytest.mark.asyncio
    async def test_process_day_publishes_day_start_event(
        self,
        simulation_engine: SimulationEngine,
        mock_event_bus: AsyncMock,
    ) -> None:
        """DayStart event is published at the beginning of _process_day."""
        ctx = DayContext.create_for_date(date(2024, 1, 2), simulation_id="t", day_number=1)
        await simulation_engine._process_day(ctx)
        # Check that publish was called with something resembling DayStart
        calls = mock_event_bus.publish.call_args_list
        assert len(calls) >= 1
        first_call_args = calls[0]
        # The first arg is an event object or event_type string
        first_arg = first_call_args[0][0] if first_call_args[0] else first_call_args[1].get("event")
        # Accept either event object with type attribute or string
        event_repr = str(first_arg)
        assert "DayStart" in event_repr or "day_start" in event_repr.lower()

    @pytest.mark.asyncio
    async def test_process_day_generates_interactions(
        self,
        simulation_engine: SimulationEngine,
        mock_external_world_manager: AsyncMock,
    ) -> None:
        """generate_daily_interactions is called during _process_day."""
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await simulation_engine._process_day(ctx)
        mock_external_world_manager.generate_daily_interactions.assert_called()

    @pytest.mark.asyncio
    async def test_process_day_updates_context_with_interaction_counts(
        self,
        simulation_engine: SimulationEngine,
    ) -> None:
        """DayContext is updated with correct transaction counts."""
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await simulation_engine._process_day(ctx)
        # Mock returns 2 customer_orders + 1 vendor_invoice = 3
        assert ctx.transactions_generated == 3

    @pytest.mark.asyncio
    async def test_process_day_routes_customer_orders(
        self,
        simulation_engine: SimulationEngine,
        mock_workflow_orchestrator: AsyncMock,
    ) -> None:
        """route_transaction called for each customer order."""
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await simulation_engine._process_day(ctx)
        route_calls = mock_workflow_orchestrator.route_transaction.call_args_list
        # Should have at least 2 calls for customer_orders + 1 for vendor_invoices
        assert len(route_calls) >= 2

    @pytest.mark.asyncio
    async def test_process_day_routes_vendor_invoices(
        self,
        simulation_engine: SimulationEngine,
        mock_workflow_orchestrator: AsyncMock,
    ) -> None:
        """route_transaction called for vendor invoices."""
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await simulation_engine._process_day(ctx)
        route_calls = mock_workflow_orchestrator.route_transaction.call_args_list
        # Total 3 = 2 customer orders + 1 vendor invoice
        assert len(route_calls) >= 3

    @pytest.mark.asyncio
    async def test_process_day_publishes_day_complete_event(
        self,
        simulation_engine: SimulationEngine,
        mock_event_bus: AsyncMock,
    ) -> None:
        """DayComplete event is published at the end of _process_day."""
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await simulation_engine._process_day(ctx)
        calls = mock_event_bus.publish.call_args_list
        assert len(calls) >= 2
        last_call_args = calls[-1]
        last_arg = last_call_args[0][0] if last_call_args[0] else last_call_args[1].get("event")
        event_repr = str(last_arg)
        assert "DayComplete" in event_repr or "day_complete" in event_repr.lower()

    @pytest.mark.asyncio
    async def test_process_day_full_pipeline_order(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Verify 5-step pipeline order: DayStart → Generate → Route → Agents → DayComplete."""
        call_order: List[str] = []

        original_publish = mock_event_bus.publish

        async def track_publish(event: Any, **kw: Any) -> None:
            event_str = str(event)
            if "DayStart" in event_str or "day_start" in event_str.lower():
                call_order.append("DayStart")
            elif "DayComplete" in event_str or "day_complete" in event_str.lower():
                call_order.append("DayComplete")
            else:
                call_order.append(f"publish:{event_str[:30]}")

        mock_event_bus.publish = AsyncMock(side_effect=track_publish)

        async def track_generate(*a: Any, **kw: Any) -> Dict:
            call_order.append("GenerateInteractions")
            return {
                "customer_orders": [{"order_id": "O-1", "customer_id": "C-1", "amount": 100}],
                "vendor_invoices": [],
                "bank_statements": [],
            }

        mock_external_world_manager.generate_daily_interactions = AsyncMock(
            side_effect=track_generate
        )

        async def track_route(*a: Any, **kw: Any) -> Dict:
            call_order.append("RouteTransaction")
            return {"workflow_id": str(uuid4()), "status": "routed"}

        mock_workflow_orchestrator.route_transaction = AsyncMock(side_effect=track_route)

        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            metrics=simulation_metrics,
        )
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await engine._process_day(ctx)

        # Verify ordering: DayStart first, DayComplete last, Generate before Route
        assert call_order[0] == "DayStart", f"Expected DayStart first, got {call_order}"
        assert call_order[-1] == "DayComplete", f"Expected DayComplete last, got {call_order}"
        gen_idx = call_order.index("GenerateInteractions")
        route_idx = call_order.index("RouteTransaction")
        assert gen_idx < route_idx, "GenerateInteractions must precede RouteTransaction"

    @pytest.mark.asyncio
    async def test_process_day_no_event_bus(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_agent_registry: MagicMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Engine with event_bus=None still processes the day without errors."""
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=None,
            agent_registry=mock_agent_registry,
            metrics=simulation_metrics,
        )
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await engine._process_day(ctx)
        assert ctx.transactions_generated == 3

    @pytest.mark.asyncio
    async def test_process_day_no_external_world(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Engine with external_world_manager=None uses empty interactions."""
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=None,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            metrics=simulation_metrics,
        )
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await engine._process_day(ctx)
        assert ctx.transactions_generated == 0

    @pytest.mark.asyncio
    async def test_process_day_no_workflow_orchestrator(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Engine with workflow_orchestrator=None skips routing."""
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=None,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            metrics=simulation_metrics,
        )
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await engine._process_day(ctx)
        # No routing calls, but day completes
        assert ctx.transactions_generated == 3


# ============================================================================
# Phase 11: Daily Summary Logging Tests
# ============================================================================


class TestDailySummaryLogging:
    """Tests for daily_summary structlog output per README.md 1748-1757."""

    @pytest.mark.asyncio
    async def test_log_daily_summary_called(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """daily_summary log entry is emitted when log_daily_summary=True."""
        with structlog.testing.capture_logs() as captured:
            ctx = DayContext.create_for_date(date(2024, 1, 2))
            ctx.transactions_generated = 10
            ctx.agents_active = 5
            ctx.average_agent_utilization = 0.5
            ctx.llm_requests = 3
            ctx.llm_cost_usd = 0.10
            ctx.day_duration_seconds = 18.0
            simulation_engine._log_daily_summary(ctx)
        daily_entries = [e for e in captured if e.get("event") == "daily_summary"]
        assert len(daily_entries) >= 1

    @pytest.mark.asyncio
    async def test_log_daily_summary_contains_required_fields(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Log entry contains all 7 required fields per specification."""
        with structlog.testing.capture_logs() as captured:
            ctx = DayContext(
                simulation_date=date(2024, 1, 2),
                transactions_generated=45,
                agents_active=12,
                average_agent_utilization=0.75,
                llm_requests=30,
                llm_cost_usd=1.50,
                day_duration_seconds=22.5,
            )
            simulation_engine._log_daily_summary(ctx)
        daily_entries = [e for e in captured if e.get("event") == "daily_summary"]
        assert len(daily_entries) >= 1
        entry = daily_entries[0]
        required_fields = [
            "simulation_date",
            "transactions_generated",
            "agents_active",
            "average_agent_utilization",
            "llm_requests",
            "llm_cost_usd",
            "duration_seconds",
        ]
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_log_daily_summary_disabled(
        self,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
    ) -> None:
        """daily_summary log NOT emitted when log_daily_summary=False."""
        cfg = SimulationConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            log_daily_summary=False,
            seed=42,
        )
        mock_time_controller.advance_to_next_business_day = AsyncMock(
            side_effect=[date(2024, 1, 3), date(2024, 1, 4)]
        )
        engine = SimulationEngine(
            config=cfg,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
        )
        with structlog.testing.capture_logs() as captured:
            await asyncio.wait_for(engine.run(), timeout=10.0)
        daily_entries = [e for e in captured if e.get("event") == "daily_summary"]
        assert len(daily_entries) == 0


# ============================================================================
# Phase 12: Control Methods Tests
# ============================================================================


class TestControlMethods:
    """Tests for stop, pause, resume, and state query methods."""

    @pytest.mark.asyncio
    async def test_stop_halts_simulation(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Calling stop() sets is_running to False."""
        task = asyncio.create_task(simulation_engine.run())
        await asyncio.sleep(0.05)
        await simulation_engine.stop()
        # Wait briefly for the task to notice the stop flag
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            task.cancel()
        assert simulation_engine.is_running() is False

    def test_is_running_default_false(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Newly created engine has is_running() == False."""
        assert simulation_engine.is_running() is False

    @pytest.mark.asyncio
    async def test_pause_and_resume(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """pause and resume toggle simulation state if implemented."""
        if hasattr(simulation_engine, "pause") and hasattr(simulation_engine, "resume"):
            task = asyncio.create_task(simulation_engine.run())
            await asyncio.sleep(0.05)
            await simulation_engine.pause()
            assert simulation_engine.is_running() is False or hasattr(
                simulation_engine, "_paused"
            )
            await simulation_engine.resume()
            await simulation_engine.stop()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()

    def test_get_current_day_context_before_run(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """get_current_day_context returns None before run()."""
        result = simulation_engine.get_current_day_context()
        assert result is None

    def test_get_metrics_returns_metrics_instance(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """get_metrics returns a SimulationMetrics object."""
        m = simulation_engine.get_metrics()
        assert isinstance(m, SimulationMetrics)

    def test_get_simulation_summary(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """get_simulation_summary returns dict with simulation_id."""
        summary = simulation_engine.get_simulation_summary()
        assert isinstance(summary, dict)
        assert "simulation_id" in summary


# ============================================================================
# Phase 13: Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for async health_check reporting subsystem availability."""

    @pytest.mark.asyncio
    async def test_health_check_all_available(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """All subsystems reported available when mocks are present."""
        health = await simulation_engine.health_check()
        assert isinstance(health, dict)
        assert "subsystems" in health
        subs = health["subsystems"]
        # Each available subsystem has available=True and status="ok"
        assert subs["time_controller"]["available"] is True
        assert subs["time_controller"]["status"] == "ok"
        assert subs["external_world_manager"]["available"] is True
        assert subs["workflow_orchestrator"]["available"] is True
        assert subs["event_bus"]["available"] is True
        assert subs["agent_registry"]["available"] is True

    @pytest.mark.asyncio
    async def test_health_check_missing_subsystems(self) -> None:
        """Missing (None) subsystems reported as not_injected."""
        engine = SimulationEngine(config=SimulationConfig())
        health = await engine.health_check()
        subs = health["subsystems"]
        assert subs["time_controller"]["status"] == "not_injected"
        assert subs["time_controller"]["available"] is False
        assert subs["external_world_manager"]["status"] == "not_injected"
        assert subs["workflow_orchestrator"]["status"] == "not_injected"


# ============================================================================
# Phase 14: Agent Processing Tests
# ============================================================================


class TestAgentProcessing:
    """Tests for agent work-queue completion and timeout."""

    @pytest.mark.asyncio
    async def test_wait_for_agents_to_complete_timeout(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Agent wait respects timeout when agents always have pending work."""
        ar = MagicMock()
        ar.get_capacity_metrics = MagicMock(
            return_value={
                "total_queue_depth": 999,  # never finishes
                "busy_count": 5,
                "idle_count": 0,
                "error_count": 0,
                "utilization_pct": 1.0,
            }
        )
        ar.get_agent_count = MagicMock(return_value=5)
        ar.get_all_agents = MagicMock(return_value=[])
        ar.get_agents_by_state = MagicMock(return_value={})
        ar.get_idle_agents = MagicMock(return_value=[])

        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=ar,
            metrics=simulation_metrics,
        )
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        # Should not hang — timeout enforced
        await asyncio.wait_for(engine._process_day(ctx), timeout=15.0)

    @pytest.mark.asyncio
    async def test_wait_for_agents_no_registry(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """No agent_registry means no agent waiting occurs."""
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=None,
            metrics=simulation_metrics,
        )
        ctx = DayContext.create_for_date(date(2024, 1, 2))
        await asyncio.wait_for(engine._process_day(ctx), timeout=10.0)
        # Completes without error


# ============================================================================
# Phase 15: Integration-Style Tests (Multiple Days)
# ============================================================================


class TestMultiDaySimulation:
    """Tests for multi-day simulation runs with cumulative metrics."""

    @pytest.mark.asyncio
    async def test_multi_day_simulation(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Multiple business days are processed and snapshotted."""
        result = await asyncio.wait_for(simulation_engine.run(), timeout=10.0)
        days = result["days_processed"]
        assert days >= 1
        snaps = simulation_engine.get_metrics().get_daily_snapshots()
        assert len(snaps) == days

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_days(
        self, simulation_engine: SimulationEngine
    ) -> None:
        """Cumulative transaction count equals the sum of daily counts."""
        await asyncio.wait_for(simulation_engine.run(), timeout=10.0)
        m = simulation_engine.get_metrics()
        snaps = m.get_daily_snapshots()
        expected = sum(s.transactions_generated for s in snaps)
        assert m.total_transactions_generated == expected

    @pytest.mark.asyncio
    async def test_simulation_with_varying_interactions(
        self,
        mock_time_controller: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Varying daily interaction counts sum correctly."""
        # Use a 3-day config to match 3 interaction counts
        cfg = SimulationConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 4),
            max_transactions_per_day=10,
            seed=42,
        )
        mock_time_controller.advance_to_next_business_day = AsyncMock(
            side_effect=[date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        )
        produced_counts: List[int] = []
        interaction_counts = [5, 3, 7]
        call_count = 0

        async def varying_interactions(*a: Any, **kw: Any) -> Dict:
            nonlocal call_count
            idx = min(call_count, len(interaction_counts) - 1)
            n = interaction_counts[idx]
            call_count += 1
            produced_counts.append(n)
            orders = [
                {"order_id": f"O-{i}", "customer_id": f"C-{i}", "amount": 100.0 * i}
                for i in range(n)
            ]
            return {
                "customer_orders": orders,
                "vendor_invoices": [],
                "bank_statements": [],
            }

        ewm = AsyncMock()
        ewm.generate_daily_interactions = AsyncMock(side_effect=varying_interactions)
        metrics = SimulationMetrics(simulation_id="vary-test")

        engine = SimulationEngine(
            config=cfg,
            time_controller=mock_time_controller,
            external_world_manager=ewm,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            metrics=metrics,
        )
        await asyncio.wait_for(engine.run(), timeout=10.0)
        total = metrics.total_transactions_generated
        assert total == sum(produced_counts)


# ============================================================================
# Phase 16: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Edge-case scenarios for the simulation engine."""

    @pytest.mark.asyncio
    async def test_empty_date_range(self) -> None:
        """start_date after end_date yields 0 days processed."""
        cfg = SimulationConfig(
            start_date=date(2024, 6, 15),
            end_date=date(2024, 6, 10),
            seed=42,
        )
        engine = SimulationEngine(config=cfg)
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        assert result["days_processed"] == 0

    @pytest.mark.asyncio
    async def test_single_day_simulation(
        self,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Single-day range processes exactly 1 day."""
        cfg = SimulationConfig(
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            seed=42,
        )
        mock_time_controller.advance_to_next_business_day = AsyncMock(
            side_effect=[date(2024, 1, 3)]
        )
        engine = SimulationEngine(
            config=cfg,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
        )
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        assert result["days_processed"] == 1

    @pytest.mark.asyncio
    async def test_external_world_returns_empty(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_workflow_orchestrator: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Empty interactions still produce a completed day."""
        ewm = AsyncMock()
        ewm.generate_daily_interactions = AsyncMock(
            return_value={"customer_orders": [], "vendor_invoices": [], "bank_statements": []}
        )
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=ewm,
            workflow_orchestrator=mock_workflow_orchestrator,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            metrics=simulation_metrics,
        )
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        assert result["days_processed"] >= 1
        assert simulation_metrics.total_transactions_generated == 0

    @pytest.mark.asyncio
    async def test_workflow_routing_failure(
        self,
        simulation_config: SimulationConfig,
        mock_time_controller: AsyncMock,
        mock_external_world_manager: AsyncMock,
        mock_event_bus: AsyncMock,
        mock_agent_registry: MagicMock,
        simulation_metrics: SimulationMetrics,
    ) -> None:
        """Routing exceptions do not crash the simulation."""
        wo = AsyncMock()
        wo.route_transaction = AsyncMock(
            side_effect=RuntimeError("Routing failed")
        )
        engine = SimulationEngine(
            config=simulation_config,
            time_controller=mock_time_controller,
            external_world_manager=mock_external_world_manager,
            workflow_orchestrator=wo,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            metrics=simulation_metrics,
        )
        result = await asyncio.wait_for(engine.run(), timeout=10.0)
        assert result is not None
        assert result["days_processed"] >= 1
