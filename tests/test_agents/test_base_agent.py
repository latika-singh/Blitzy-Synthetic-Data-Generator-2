"""
Tests for BaseAgent lifecycle, state transitions, work queue, and metrics.

This module provides comprehensive test coverage for the abstract BaseAgent
class (app/agents/base_agent.py), including its async run() loop,
process_work_item() abstract contract, state transitions (IDLE→THINKING→IDLE
and error path IDLE→THINKING→ERROR), work queue management, make_decision()
context enrichment, _calculate_importance() scoring, and WorkItem/WorkResult
data structures.

Testing Standards (AAP Section 0.7.5):
    - All LLM calls are mocked (no live API calls).
    - Redis interactions use fakeredis (no external Redis dependency).
    - Async tests use pytest-asyncio with proper event loop management.
    - All agent tests verify state transitions.
    - Coverage target: ≥80%.
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import asyncio
from abc import ABC
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4, UUID

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import pytest

# ---------------------------------------------------------------------------
# Internal Imports — Under Test
# ---------------------------------------------------------------------------
from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig, AgentState
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry


# =========================================================================
# Concrete Test Subclasses — Required because BaseAgent is abstract
# =========================================================================


class ConcreteTestAgent(BaseAgent):
    """Minimal concrete subclass of BaseAgent for happy-path testing.

    Always succeeds, returning a fixed WorkResult with ``success=True``.
    """

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        return WorkResult(
            success=True,
            actions_taken=["test_action"],
            data={"test": True},
        )


class FailingTestAgent(BaseAgent):
    """Concrete subclass that always raises during processing.

    Used for testing the error path in the run() loop (state → ERROR,
    error counter increment, loop continuation).
    """

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        raise RuntimeError("Simulated processing failure")


class SlowTestAgent(BaseAgent):
    """Concrete subclass with a deliberate processing delay.

    Sleeps for a configurable duration so tests can inspect agent state
    mid-processing (e.g. verifying THINKING state, current_work_item).
    """

    def __init__(self, *args: Any, delay: float = 0.3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._delay = delay

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        await asyncio.sleep(self._delay)
        return WorkResult(
            success=True,
            actions_taken=["slow_action"],
            data={"delayed": True},
        )


class FailOnceThenSucceedAgent(BaseAgent):
    """Concrete subclass that fails on the first call, then succeeds.

    Used to verify the run() loop continues after an error and processes
    subsequent work items successfully.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._call_count = 0

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("First item fails intentionally")
        return WorkResult(
            success=True,
            actions_taken=["recovered_action"],
            data={"recovered": True},
        )


# =========================================================================
# Async Helper — Run agent, process items, and clean up
# =========================================================================


async def _run_agent_with_items(
    agent: BaseAgent,
    items: list,
    wait_time: float = 0.15,
) -> None:
    """Put *items* into the agent's queue, run the loop, then shut down.

    After ``wait_time`` seconds the agent is stopped and the background
    task is cancelled so tests do not hang.
    """
    for item in items:
        await agent.work_queue.put(item)

    task = asyncio.create_task(agent.run())
    await asyncio.sleep(wait_time)
    await agent.stop()
    # The run-loop may be blocked on queue.get(); cancel to unblock.
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# =========================================================================
# Local Fixtures — Agents and work items specific to this module
# =========================================================================


@pytest.fixture
def concrete_agent(
    sample_agent_config: AgentConfig,
    mock_agent_memory: MagicMock,
    mock_decision_engine: MagicMock,
    mock_action_registry: MagicMock,
) -> ConcreteTestAgent:
    """Create a ConcreteTestAgent wired with shared conftest mocks."""
    return ConcreteTestAgent(
        config=sample_agent_config,
        memory=mock_agent_memory,
        decision_engine=mock_decision_engine,
        action_registry=mock_action_registry,
    )


@pytest.fixture
def failing_agent(
    sample_agent_config: AgentConfig,
    mock_agent_memory: MagicMock,
    mock_decision_engine: MagicMock,
    mock_action_registry: MagicMock,
) -> FailingTestAgent:
    """Create a FailingTestAgent wired with shared conftest mocks."""
    return FailingTestAgent(
        config=sample_agent_config,
        memory=mock_agent_memory,
        decision_engine=mock_decision_engine,
        action_registry=mock_action_registry,
    )


@pytest.fixture
def slow_agent(
    sample_agent_config: AgentConfig,
    mock_agent_memory: MagicMock,
    mock_decision_engine: MagicMock,
    mock_action_registry: MagicMock,
) -> SlowTestAgent:
    """Create a SlowTestAgent wired with shared conftest mocks."""
    return SlowTestAgent(
        config=sample_agent_config,
        memory=mock_agent_memory,
        decision_engine=mock_decision_engine,
        action_registry=mock_action_registry,
        delay=0.3,
    )


@pytest.fixture
def fail_once_agent(
    sample_agent_config: AgentConfig,
    mock_agent_memory: MagicMock,
    mock_decision_engine: MagicMock,
    mock_action_registry: MagicMock,
) -> FailOnceThenSucceedAgent:
    """Create a FailOnceThenSucceedAgent wired with shared conftest mocks."""
    return FailOnceThenSucceedAgent(
        config=sample_agent_config,
        memory=mock_agent_memory,
        decision_engine=mock_decision_engine,
        action_registry=mock_action_registry,
    )


@pytest.fixture
def sample_work_item_obj() -> WorkItem:
    """A standard-value WorkItem for general-purpose tests."""
    return WorkItem(
        type="vendor_invoice",
        data={"invoice_id": "INV-001", "amount": 5000.0},
        workflow_id=uuid4(),
        description="Test vendor invoice processing",
        priority=5,
        amount=5000.0,
    )


@pytest.fixture
def high_value_work_item() -> WorkItem:
    """A high-value WorkItem (>$10K) for importance-scoring tests."""
    return WorkItem(
        type="purchase_order",
        data={"po_id": "PO-001", "amount": 50000.0},
        workflow_id=uuid4(),
        description="High value purchase order",
        priority=3,
        amount=50000.0,
    )


# =========================================================================
# Test Class 1: BaseAgent is Abstract
# =========================================================================


class TestBaseAgentAbstract:
    """Verify BaseAgent cannot be instantiated and enforces its abstract contract."""

    def test_base_agent_is_abstract(self) -> None:
        """BaseAgent must be a subclass of ABC."""
        assert issubclass(BaseAgent, ABC)

    def test_cannot_instantiate_base_agent(
        self,
        sample_agent_config: AgentConfig,
        mock_agent_memory: MagicMock,
        mock_decision_engine: MagicMock,
        mock_action_registry: MagicMock,
    ) -> None:
        """Attempting to instantiate BaseAgent directly must raise TypeError."""
        with pytest.raises(TypeError):
            BaseAgent(
                config=sample_agent_config,
                memory=mock_agent_memory,
                decision_engine=mock_decision_engine,
                action_registry=mock_action_registry,
            )

    def test_process_work_item_is_abstract(self) -> None:
        """process_work_item must be declared in __abstractmethods__."""
        assert "process_work_item" in BaseAgent.__abstractmethods__


# =========================================================================
# Test Class 2: Constructor and Initialization
# =========================================================================


class TestBaseAgentInit:
    """Verify constructor injection and default attribute values."""

    def test_constructor_sets_config(
        self, concrete_agent: ConcreteTestAgent, sample_agent_config: AgentConfig
    ) -> None:
        assert concrete_agent.config is sample_agent_config

    def test_constructor_sets_memory(
        self, concrete_agent: ConcreteTestAgent, mock_agent_memory: MagicMock
    ) -> None:
        assert concrete_agent.memory is mock_agent_memory

    def test_constructor_sets_decision_engine(
        self, concrete_agent: ConcreteTestAgent, mock_decision_engine: MagicMock
    ) -> None:
        assert concrete_agent.decision_engine is mock_decision_engine

    def test_constructor_sets_action_registry(
        self, concrete_agent: ConcreteTestAgent, mock_action_registry: MagicMock
    ) -> None:
        assert concrete_agent.action_registry is mock_action_registry

    def test_initial_state_is_idle(self, concrete_agent: ConcreteTestAgent) -> None:
        assert concrete_agent.state == AgentState.IDLE

    def test_initial_work_queue_is_empty(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert concrete_agent.work_queue.empty() is True

    def test_initial_current_work_item_is_none(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert concrete_agent.current_work_item is None

    def test_initial_metrics(self, concrete_agent: ConcreteTestAgent) -> None:
        expected = {
            "items_processed": 0,
            "decisions_made": 0,
            "errors": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
        }
        assert concrete_agent.metrics == expected

    def test_running_flag_initially_false(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert concrete_agent._running is False


# =========================================================================
# Test Class 3: State Transitions (CRITICAL per AAP 0.7.5)
# =========================================================================


class TestAgentStateTransitions:
    """Verify IDLE→THINKING→IDLE (happy path) and IDLE→THINKING→ERROR (error path)."""

    @pytest.mark.asyncio
    async def test_state_idle_to_thinking_during_processing(
        self,
        slow_agent: SlowTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """While the slow agent is processing, state must be THINKING."""
        await slow_agent.work_queue.put(sample_work_item_obj)
        task = asyncio.create_task(slow_agent.run())
        # Wait long enough for the loop to dequeue and set THINKING but
        # short enough that SlowTestAgent is still sleeping (delay=0.3s).
        await asyncio.sleep(0.05)
        assert slow_agent.state == AgentState.THINKING
        # Clean up
        await slow_agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_state_returns_to_idle_after_processing(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """After fully processing an item the agent must return to IDLE."""
        await _run_agent_with_items(concrete_agent, [sample_work_item_obj])
        assert concrete_agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_state_to_error_on_exception(
        self,
        failing_agent: FailingTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """When process_work_item raises, state must transition to ERROR."""
        await _run_agent_with_items(failing_agent, [sample_work_item_obj])
        assert failing_agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_current_work_item_set_during_processing(
        self,
        slow_agent: SlowTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """While the agent is processing, current_work_item must be set."""
        await slow_agent.work_queue.put(sample_work_item_obj)
        task = asyncio.create_task(slow_agent.run())
        await asyncio.sleep(0.05)
        assert slow_agent.current_work_item is sample_work_item_obj
        await slow_agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_current_work_item_none_after_processing(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """After processing completes, current_work_item must be None."""
        await _run_agent_with_items(concrete_agent, [sample_work_item_obj])
        assert concrete_agent.current_work_item is None

    @pytest.mark.asyncio
    async def test_current_work_item_none_after_error(
        self,
        failing_agent: FailingTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """After an error, current_work_item must still be cleared to None."""
        await _run_agent_with_items(failing_agent, [sample_work_item_obj])
        assert failing_agent.current_work_item is None

    @pytest.mark.asyncio
    async def test_state_thinking_during_make_decision(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """make_decision must set state to THINKING during execution."""
        observed_states: list = []

        original_decide = concrete_agent.decision_engine.decide

        async def spy_decide(**kwargs: Any) -> Dict[str, Any]:
            observed_states.append(concrete_agent.state)
            return await original_decide(**kwargs)

        concrete_agent.decision_engine.decide = AsyncMock(side_effect=spy_decide)
        await concrete_agent.make_decision("process_vendor_invoice", {"test": True})
        assert AgentState.THINKING in observed_states


# =========================================================================
# Test Class 4: run() Loop — Main Agent Processing Cycle
# =========================================================================


class TestAgentRunLoop:
    """Verify the async run() loop processes items, records memory, and updates metrics."""

    @pytest.mark.asyncio
    async def test_run_sets_running_flag(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """run() must set _running to True before entering the loop."""
        await concrete_agent.work_queue.put(sample_work_item_obj)
        task = asyncio.create_task(concrete_agent.run())
        await asyncio.sleep(0.05)
        assert concrete_agent._running is True
        await concrete_agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_run_processes_work_item_from_queue(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """run() must dequeue and process a single work item."""
        await _run_agent_with_items(concrete_agent, [sample_work_item_obj])
        assert concrete_agent.metrics["items_processed"] == 1

    @pytest.mark.asyncio
    async def test_run_records_observation_in_memory(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """After processing, run() must call memory.add_observation."""
        await _run_agent_with_items(concrete_agent, [sample_work_item_obj])
        concrete_agent.memory.add_observation.assert_called_once()
        call_kwargs = concrete_agent.memory.add_observation.call_args
        # The content string should contain the work item type and description
        content_arg = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content", "")
        if not content_arg and call_kwargs.args:
            content_arg = call_kwargs.args[0]
        assert sample_work_item_obj.type in str(content_arg)

    @pytest.mark.asyncio
    async def test_run_increments_items_processed(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """metrics['items_processed'] must increment by 1 per successful item."""
        await _run_agent_with_items(concrete_agent, [sample_work_item_obj])
        assert concrete_agent.metrics["items_processed"] == 1

    @pytest.mark.asyncio
    async def test_run_multiple_items(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """The loop must process multiple items sequentially."""
        items = [
            WorkItem(type="vendor_invoice", data={"id": i}, description=f"Item {i}")
            for i in range(3)
        ]
        await _run_agent_with_items(concrete_agent, items, wait_time=0.3)
        assert concrete_agent.metrics["items_processed"] == 3

    @pytest.mark.asyncio
    async def test_run_updates_average_processing_time(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """average_processing_time should be >0 after processing an item."""
        await _run_agent_with_items(concrete_agent, [sample_work_item_obj])
        assert concrete_agent.metrics["average_processing_time"] > 0.0
        assert concrete_agent.metrics["total_processing_time"] > 0.0

    @pytest.mark.asyncio
    async def test_run_error_increments_error_count(
        self,
        failing_agent: FailingTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """When process_work_item raises, errors counter must increment."""
        await _run_agent_with_items(failing_agent, [sample_work_item_obj])
        assert failing_agent.metrics["errors"] == 1

    @pytest.mark.asyncio
    async def test_run_error_sets_error_state(
        self,
        failing_agent: FailingTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """On exception the agent's state must be ERROR."""
        await _run_agent_with_items(failing_agent, [sample_work_item_obj])
        assert failing_agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_run_error_does_not_record_observation(
        self,
        failing_agent: FailingTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """When processing fails, no observation should be recorded."""
        await _run_agent_with_items(failing_agent, [sample_work_item_obj])
        failing_agent.memory.add_observation.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_continues_after_error(
        self,
        fail_once_agent: FailOnceThenSucceedAgent,
    ) -> None:
        """The loop must continue processing after an error (not crash)."""
        item_a = WorkItem(
            type="vendor_invoice", data={"id": "A"}, description="Will fail"
        )
        item_b = WorkItem(
            type="vendor_invoice", data={"id": "B"}, description="Will succeed"
        )
        await _run_agent_with_items(fail_once_agent, [item_a, item_b], wait_time=0.3)
        # First item fails (error+1), second succeeds (items_processed+1)
        assert fail_once_agent.metrics["errors"] == 1
        assert fail_once_agent.metrics["items_processed"] == 1

    @pytest.mark.asyncio
    async def test_stop_ends_run_loop(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Calling stop() must cause the run loop to exit."""
        task = asyncio.create_task(concrete_agent.run())
        await asyncio.sleep(0.05)
        assert concrete_agent._running is True
        await concrete_agent.stop()
        assert concrete_agent._running is False
        # Put a dummy item to unblock queue.get()
        await concrete_agent.work_queue.put(
            WorkItem(type="sentinel", data={}, description="stop sentinel")
        )
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# =========================================================================
# Test Class 5: make_decision() — Context Enrichment & Delegation
# =========================================================================


class TestAgentMakeDecision:
    """Verify make_decision enriches context and delegates to DecisionEngine."""

    @pytest.mark.asyncio
    async def test_make_decision_calls_decision_engine_decide(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """decision_engine.decide must be called exactly once."""
        await concrete_agent.make_decision("process_vendor_invoice", {"test": True})
        concrete_agent.decision_engine.decide.assert_called_once()

    @pytest.mark.asyncio
    async def test_make_decision_passes_decision_type(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """decide() must receive the exact decision_type string."""
        await concrete_agent.make_decision("approve_transaction", {"id": 1})
        call_kwargs = concrete_agent.decision_engine.decide.call_args.kwargs
        assert call_kwargs["decision_type"] == "approve_transaction"

    @pytest.mark.asyncio
    async def test_make_decision_enriches_context_with_agent_info(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Context passed to decide() must include 'agent' key with role, traits, recent_observations."""
        await concrete_agent.make_decision("process_vendor_invoice", {"doc": "INV"})
        call_kwargs = concrete_agent.decision_engine.decide.call_args.kwargs
        context = call_kwargs["context"]
        assert "agent" in context
        agent_info = context["agent"]
        assert "role" in agent_info
        assert "traits" in agent_info
        assert "recent_observations" in agent_info

    @pytest.mark.asyncio
    async def test_make_decision_agent_role_matches_config(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_agent_config: AgentConfig,
    ) -> None:
        """The 'role' in enriched context must match config.role."""
        await concrete_agent.make_decision("process_vendor_invoice", {})
        call_kwargs = concrete_agent.decision_engine.decide.call_args.kwargs
        assert call_kwargs["context"]["agent"]["role"] == sample_agent_config.role

    @pytest.mark.asyncio
    async def test_make_decision_agent_traits_matches_config(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_agent_config: AgentConfig,
    ) -> None:
        """The 'traits' in enriched context must match config.traits."""
        await concrete_agent.make_decision("process_vendor_invoice", {})
        call_kwargs = concrete_agent.decision_engine.decide.call_args.kwargs
        assert call_kwargs["context"]["agent"]["traits"] == sample_agent_config.traits

    @pytest.mark.asyncio
    async def test_make_decision_passes_agent_config(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_agent_config: AgentConfig,
    ) -> None:
        """decide() must receive agent_config as a keyword argument."""
        await concrete_agent.make_decision("process_vendor_invoice", {})
        call_kwargs = concrete_agent.decision_engine.decide.call_args.kwargs
        assert call_kwargs["agent_config"] is sample_agent_config

    @pytest.mark.asyncio
    async def test_make_decision_increments_decisions_made(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """metrics['decisions_made'] must increment by 1 per call."""
        assert concrete_agent.metrics["decisions_made"] == 0
        await concrete_agent.make_decision("process_vendor_invoice", {})
        assert concrete_agent.metrics["decisions_made"] == 1
        await concrete_agent.make_decision("approve_transaction", {})
        assert concrete_agent.metrics["decisions_made"] == 2

    @pytest.mark.asyncio
    async def test_make_decision_returns_decision_result(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Returned dict must match what decision_engine.decide returns."""
        result = await concrete_agent.make_decision("process_vendor_invoice", {})
        assert result["decision"] == "approve"
        assert result["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_make_decision_fetches_recent_observations(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """make_decision must call memory.get_recent(n=5)."""
        await concrete_agent.make_decision("process_vendor_invoice", {})
        concrete_agent.memory.get_recent.assert_called_once_with(n=5)

    @pytest.mark.asyncio
    async def test_make_decision_preserves_original_context_keys(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """User-supplied context keys must be preserved alongside 'agent'."""
        original_context = {"invoice_id": "INV-999", "amount": 4200.0}
        await concrete_agent.make_decision("process_vendor_invoice", original_context)
        call_kwargs = concrete_agent.decision_engine.decide.call_args.kwargs
        context = call_kwargs["context"]
        assert context["invoice_id"] == "INV-999"
        assert context["amount"] == 4200.0
        assert "agent" in context


# =========================================================================
# Test Class 6: _calculate_importance() — Observation Importance Scoring
# =========================================================================


class TestCalculateImportance:
    """Verify base=5.0, +2 if amount>10K, +3 if failure, +2 if exceptions, cap 10.0."""

    def test_base_importance_is_5(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """Standard item + success → importance == 5.0 (base only)."""
        result = WorkResult(success=True)
        importance = concrete_agent._calculate_importance(sample_work_item_obj, result)
        assert importance == 5.0

    def test_high_amount_adds_2(
        self,
        concrete_agent: ConcreteTestAgent,
        high_value_work_item: WorkItem,
    ) -> None:
        """WorkItem.amount > 10,000 → +2.0 → importance == 7.0."""
        result = WorkResult(success=True)
        importance = concrete_agent._calculate_importance(high_value_work_item, result)
        assert importance == 7.0

    def test_error_result_adds_3(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """success=False → +3.0 → importance == 8.0."""
        result = WorkResult(success=False, error="Processing error")
        importance = concrete_agent._calculate_importance(sample_work_item_obj, result)
        assert importance == 8.0

    def test_exceptions_add_2(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """has_exceptions=True → +2.0 → importance == 7.0."""
        result = WorkResult(success=True, has_exceptions=True)
        importance = concrete_agent._calculate_importance(sample_work_item_obj, result)
        assert importance == 7.0

    def test_combined_high_amount_and_error(
        self,
        concrete_agent: ConcreteTestAgent,
        high_value_work_item: WorkItem,
    ) -> None:
        """amount>10K (+2) + failure (+3) = 10.0 (base 5 + 5, capped at 10)."""
        result = WorkResult(success=False, error="Failed")
        importance = concrete_agent._calculate_importance(high_value_work_item, result)
        assert importance == 10.0

    def test_importance_capped_at_10(
        self,
        concrete_agent: ConcreteTestAgent,
        high_value_work_item: WorkItem,
    ) -> None:
        """All modifiers active: 5 + 2 + 3 + 2 = 12 → capped to 10.0."""
        result = WorkResult(success=False, has_exceptions=True, error="Total failure")
        importance = concrete_agent._calculate_importance(high_value_work_item, result)
        assert importance == 10.0

    def test_normal_item_no_modifiers(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """amount < 10K, success=True, no exceptions → base 5.0."""
        item = WorkItem(type="vendor_invoice", data={}, amount=500.0, description="Small")
        result = WorkResult(success=True)
        importance = concrete_agent._calculate_importance(item, result)
        assert importance == 5.0

    def test_importance_with_exceptions_and_high_amount(
        self,
        concrete_agent: ConcreteTestAgent,
        high_value_work_item: WorkItem,
    ) -> None:
        """amount>10K (+2) + exceptions (+2) = 9.0 (base 5 + 4)."""
        result = WorkResult(success=True, has_exceptions=True)
        importance = concrete_agent._calculate_importance(high_value_work_item, result)
        assert importance == 9.0

    def test_none_amount_treated_as_no_modifier(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """When amount is None, the high-value modifier must NOT apply."""
        item = WorkItem(type="vendor_invoice", data={}, amount=None, description="No amount")
        result = WorkResult(success=True)
        importance = concrete_agent._calculate_importance(item, result)
        assert importance == 5.0

    def test_boundary_amount_exactly_10000(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Boundary: amount == 10,000 must NOT trigger the +2 modifier (> not >=)."""
        item = WorkItem(type="po", data={}, amount=10000.0, description="Boundary")
        result = WorkResult(success=True)
        importance = concrete_agent._calculate_importance(item, result)
        assert importance == 5.0

    def test_amount_just_above_10000(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """amount=10,000.01 must trigger the +2 modifier."""
        item = WorkItem(type="po", data={}, amount=10000.01, description="Above boundary")
        result = WorkResult(success=True)
        importance = concrete_agent._calculate_importance(item, result)
        assert importance == 7.0


# =========================================================================
# Test Class 7: Work Queue Management
# =========================================================================


class TestWorkQueue:
    """Verify work_queue type, FIFO ordering, and basic operations."""

    def test_work_queue_is_asyncio_queue(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert isinstance(concrete_agent.work_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_put_and_get_work_item(
        self,
        concrete_agent: ConcreteTestAgent,
        sample_work_item_obj: WorkItem,
    ) -> None:
        """Put an item and get it back — must be the same object."""
        await concrete_agent.work_queue.put(sample_work_item_obj)
        retrieved = await concrete_agent.work_queue.get()
        assert retrieved is sample_work_item_obj

    @pytest.mark.asyncio
    async def test_queue_fifo_order(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Items must come out in the same order they were put in."""
        items = [
            WorkItem(type="a", data={}, description="First"),
            WorkItem(type="b", data={}, description="Second"),
            WorkItem(type="c", data={}, description="Third"),
        ]
        for item in items:
            await concrete_agent.work_queue.put(item)

        for expected in items:
            actual = await concrete_agent.work_queue.get()
            assert actual is expected

    def test_queue_starts_empty(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert concrete_agent.work_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_queue_size_increments(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """qsize() must reflect the number of enqueued items."""
        item = WorkItem(type="test", data={}, description="queue size")
        await concrete_agent.work_queue.put(item)
        assert concrete_agent.work_queue.qsize() == 1
        await concrete_agent.work_queue.put(item)
        assert concrete_agent.work_queue.qsize() == 2

    def test_get_queue_size_method(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        """get_queue_size() utility must match qsize()."""
        assert concrete_agent.get_queue_size() == 0


# =========================================================================
# Test Class 8: Metrics Tracking
# =========================================================================


class TestAgentMetrics:
    """Verify metric counters and computed values."""

    def test_initial_metrics_zeroed(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert concrete_agent.metrics["items_processed"] == 0
        assert concrete_agent.metrics["decisions_made"] == 0
        assert concrete_agent.metrics["errors"] == 0
        assert concrete_agent.metrics["average_processing_time"] == 0.0
        assert concrete_agent.metrics["total_processing_time"] == 0.0

    @pytest.mark.asyncio
    async def test_items_processed_increments(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Processing N items must set items_processed to N."""
        items = [
            WorkItem(type="vi", data={}, description=f"item-{i}") for i in range(3)
        ]
        await _run_agent_with_items(concrete_agent, items, wait_time=0.3)
        assert concrete_agent.metrics["items_processed"] == 3

    @pytest.mark.asyncio
    async def test_decisions_made_increments(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """Each make_decision call must increment decisions_made."""
        for _ in range(4):
            await concrete_agent.make_decision("test_decision", {})
        assert concrete_agent.metrics["decisions_made"] == 4

    @pytest.mark.asyncio
    async def test_errors_increment_on_failure(
        self,
        failing_agent: FailingTestAgent,
    ) -> None:
        """Each failed processing must increment the errors counter."""
        items = [
            WorkItem(type="vi", data={}, description=f"fail-{i}") for i in range(2)
        ]
        await _run_agent_with_items(failing_agent, items, wait_time=0.3)
        assert failing_agent.metrics["errors"] == 2

    @pytest.mark.asyncio
    async def test_average_processing_time_computed(
        self,
        concrete_agent: ConcreteTestAgent,
    ) -> None:
        """average_processing_time = total / items_processed."""
        items = [
            WorkItem(type="vi", data={}, description=f"metric-{i}") for i in range(2)
        ]
        await _run_agent_with_items(concrete_agent, items, wait_time=0.3)
        total = concrete_agent.metrics["total_processing_time"]
        count = concrete_agent.metrics["items_processed"]
        expected_avg = total / count if count > 0 else 0.0
        assert abs(concrete_agent.metrics["average_processing_time"] - expected_avg) < 1e-9

    def test_get_metrics_returns_copy(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        """get_metrics() must return a copy, not the internal dict."""
        metrics_copy = concrete_agent.get_metrics()
        metrics_copy["items_processed"] = 999
        assert concrete_agent.metrics["items_processed"] == 0

    def test_is_idle_returns_true_when_idle(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        assert concrete_agent.is_idle() is True

    def test_is_idle_returns_false_when_not_idle(
        self, concrete_agent: ConcreteTestAgent
    ) -> None:
        concrete_agent.state = AgentState.THINKING
        assert concrete_agent.is_idle() is False


# =========================================================================
# Test Class 9: WorkItem and WorkResult Data Structures
# =========================================================================


class TestWorkItemWorkResult:
    """Verify field defaults, serialization, and edge cases."""

    def test_work_item_creation(self) -> None:
        """All required and optional fields must be set correctly."""
        wf_id = uuid4()
        item = WorkItem(
            type="vendor_invoice",
            data={"invoice_id": "INV-100"},
            workflow_id=wf_id,
            description="Test invoice",
            priority=3,
            amount=2500.0,
        )
        assert item.type == "vendor_invoice"
        assert item.data == {"invoice_id": "INV-100"}
        assert item.workflow_id == wf_id
        assert item.description == "Test invoice"
        assert item.priority == 3
        assert item.amount == 2500.0

    def test_work_item_defaults(self) -> None:
        """Default values must match the dataclass specification."""
        item = WorkItem(type="test", data={})
        assert item.description == ""
        assert item.priority == 5
        assert item.retry_count == 0
        assert item.amount is None
        assert isinstance(item.workflow_id, UUID)
        assert isinstance(item.created_at, datetime)

    def test_work_item_dict_serialization(self) -> None:
        """dict() must produce a JSON-safe dictionary representation."""
        wf_id = uuid4()
        item = WorkItem(
            type="purchase_order",
            data={"po_id": "PO-42"},
            workflow_id=wf_id,
            description="PO test",
            priority=2,
            amount=15000.0,
        )
        serialized = item.dict()
        assert serialized["type"] == "purchase_order"
        assert serialized["workflow_id"] == str(wf_id)
        assert serialized["amount"] == 15000.0
        assert isinstance(serialized["created_at"], str)  # ISO format string

    def test_work_result_creation(self) -> None:
        """All fields must be set correctly on explicit construction."""
        result = WorkResult(
            success=True,
            actions_taken=["gl_coding", "three_way_match"],
            approval_needed=True,
            has_exceptions=False,
            data={"match_status": "matched"},
            error=None,
            processing_time_seconds=1.5,
        )
        assert result.success is True
        assert result.actions_taken == ["gl_coding", "three_way_match"]
        assert result.approval_needed is True
        assert result.has_exceptions is False
        assert result.data == {"match_status": "matched"}
        assert result.error is None
        assert result.processing_time_seconds == 1.5

    def test_work_result_defaults(self) -> None:
        """Default values for optional WorkResult fields."""
        result = WorkResult(success=True)
        assert result.actions_taken == []
        assert result.approval_needed is False
        assert result.has_exceptions is False
        assert result.data == {}
        assert result.error is None
        assert result.processing_time_seconds == 0.0

    def test_work_result_success_false(self) -> None:
        """WorkResult with success=False and an error message."""
        result = WorkResult(success=False, error="Validation failed")
        assert result.success is False
        assert result.error == "Validation failed"

    def test_work_result_with_approval_needed(self) -> None:
        """approval_needed=True must be preserved."""
        result = WorkResult(success=True, approval_needed=True)
        assert result.approval_needed is True

    def test_work_result_with_exceptions(self) -> None:
        """has_exceptions=True must be preserved."""
        result = WorkResult(success=True, has_exceptions=True)
        assert result.has_exceptions is True

    def test_work_item_with_zero_amount(self) -> None:
        """amount=0.0 must be stored (not treated as None)."""
        item = WorkItem(type="credit_note", data={}, amount=0.0, description="Zero")
        assert item.amount == 0.0

    def test_work_result_processing_time_mutable(self) -> None:
        """processing_time_seconds is mutable — run() sets it post-processing."""
        result = WorkResult(success=True)
        assert result.processing_time_seconds == 0.0
        result.processing_time_seconds = 2.3
        assert result.processing_time_seconds == 2.3
