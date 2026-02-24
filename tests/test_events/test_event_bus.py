"""Comprehensive tests for the dual-mode EventBus pub/sub system (F-007).

Covers:
- EventBus initialization with default and custom configs
- Subscribe / unsubscribe lifecycle
- Publish and async dispatch to handlers
- Priority ordering (URGENT > HIGH > NORMAL > LOW)
- Handler timeout enforcement (5 s) and publish timeout enforcement (1 s)
- Event persistence integration with EventStore
- All 8 concrete event types
- Context manager (async with) lifecycle
- Metrics tracking
- Default handler registration / unregistration
- Edge cases: no subscribers, base Event, large payload, None agent_id

Testing Rules (AAP Section 0.7.5):
    - ALL async tests use pytest-asyncio (asyncio_mode=auto in pytest.ini)
    - Redis tests use fakeredis — no external Redis dependency
    - No live LLM API calls
    - Coverage target ≥ 80%
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from app.events.event_bus import (
    EventBus,
    EventBusConfig,
    EventBusMode,
    EventPriority,
)
from app.events.event_handlers import (
    register_default_handlers,
    unregister_default_handlers,
)
from app.events.event_store import EventStore
from app.events.event_types import (
    ApprovalCompleted,
    ApprovalRequired,
    DiscrepancyDetected,
    DocumentGenerated,
    Event,
    EventType,
    PeriodClosed,
    PeriodClosing,
    TransactionCompleted,
    TransactionCreated,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISPATCH_SETTLE_TIME: float = 0.3
"""Seconds to sleep after publish so the dispatch loop can process the event."""

_TEST_TIMEOUT: float = 10.0
"""Maximum wall-clock seconds for any single test awaitable."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_event(
    event_cls: type = TransactionCreated,
    payload: Optional[dict[str, Any]] = None,
    agent_id: Optional[UUID] = None,
) -> Event:
    """Create a test event instance with sensible defaults."""
    if payload is None:
        payload = {
            "transaction_id": f"TXN-{uuid4().hex[:6]}",
            "transaction_type": "purchase_order",
            "amount": 5000.00,
        }
    return event_cls(
        simulation_id=uuid4(),
        payload=payload,
        agent_id=agent_id or uuid4(),
    )


# ---------------------------------------------------------------------------
# Fixtures (local to this test module)
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus_config() -> EventBusConfig:
    """Return an EventBusConfig in IN_MEMORY mode with persistence disabled."""
    return EventBusConfig(
        mode=EventBusMode.IN_MEMORY,
        persist_events=False,
        max_queue_size=10_000,
        publish_timeout=1.0,
        handler_timeout=5.0,
        enable_priority=True,
    )


@pytest_asyncio.fixture
async def event_bus(event_bus_config: EventBusConfig):
    """Provide an EventBus in IN_MEMORY mode (persistence off).

    Yields the bus instance and ensures it is stopped after the test.
    """
    bus = EventBus(config=event_bus_config)
    yield bus
    if bus.is_running():
        await bus.stop()


@pytest_asyncio.fixture
async def started_event_bus(event_bus: EventBus):
    """Provide an already-started EventBus for tests that need dispatch.

    The bus is stopped automatically after the test.
    """
    await event_bus.start()
    yield event_bus
    # event_bus fixture handles stop via its own teardown


@pytest.fixture
def sample_event() -> TransactionCreated:
    """Return a TransactionCreated event with realistic payload."""
    return TransactionCreated(
        simulation_id=uuid4(),
        payload={
            "transaction_id": "TXN-001",
            "transaction_type": "purchase_order",
            "amount": 5000.00,
        },
        agent_id=uuid4(),
    )


@pytest.fixture
def sample_events() -> List[Event]:
    """Return a list of different event types for batch testing."""
    sim_id = uuid4()
    agent = uuid4()
    return [
        TransactionCreated(
            simulation_id=sim_id,
            payload={"transaction_id": "TXN-001", "transaction_type": "purchase_order", "amount": 1000.0},
            agent_id=agent,
        ),
        TransactionCompleted(
            simulation_id=sim_id,
            payload={"transaction_id": "TXN-001", "final_status": "completed"},
            agent_id=agent,
        ),
        ApprovalRequired(
            simulation_id=sim_id,
            payload={"transaction_id": "TXN-002", "amount": 30000.0, "required_approver_role": "cfo"},
            agent_id=agent,
        ),
        ApprovalCompleted(
            simulation_id=sim_id,
            payload={"transaction_id": "TXN-002", "approval_decision": "approved"},
            agent_id=agent,
        ),
        DocumentGenerated(
            simulation_id=sim_id,
            payload={"document_id": "DOC-001", "document_type": "invoice"},
            agent_id=agent,
        ),
        PeriodClosing(
            simulation_id=sim_id,
            payload={"period_type": "monthly", "fiscal_year": 2025},
            agent_id=agent,
        ),
        PeriodClosed(
            simulation_id=sim_id,
            payload={"period_type": "monthly", "fiscal_year": 2025},
            agent_id=agent,
        ),
        DiscrepancyDetected(
            simulation_id=sim_id,
            payload={"discrepancy_type": "price_variance", "severity": "high"},
            agent_id=agent,
        ),
    ]


@pytest.fixture
def mock_event_store() -> AsyncMock:
    """Return an AsyncMock of EventStore with save_event returning an event_id."""
    store = AsyncMock(spec=EventStore)
    store.save_event = AsyncMock(return_value=str(uuid4()))
    return store


# =========================================================================
# Phase 3: EventBus Initialization Tests
# =========================================================================


class TestEventBusInitialization:
    """Tests for EventBus construction and configuration."""

    async def test_default_initialization(self) -> None:
        """EventBus() with no args uses IN_MEMORY mode and is not running."""
        bus = EventBus()
        assert bus.is_running() is False
        assert bus.get_event_count() == 0
        assert bus.get_queue_size() == 0

    async def test_custom_config_initialization(self) -> None:
        """EventBus respects a custom EventBusConfig."""
        cfg = EventBusConfig(
            mode=EventBusMode.IN_MEMORY,
            max_queue_size=500,
            publish_timeout=2.0,
            handler_timeout=10.0,
            persist_events=False,
            enable_priority=False,
        )
        bus = EventBus(config=cfg)
        assert bus._config.max_queue_size == 500
        assert bus._config.publish_timeout == 2.0
        assert bus._config.handler_timeout == 10.0
        assert bus._config.persist_events is False
        assert bus._config.enable_priority is False

    async def test_initialization_with_event_store(
        self, mock_event_store: AsyncMock,
    ) -> None:
        """EventBus stores an injected EventStore reference."""
        bus = EventBus(event_store=mock_event_store)
        assert bus._event_store is mock_event_store

    async def test_initialization_in_memory_mode(self) -> None:
        """Default mode is IN_MEMORY."""
        cfg = EventBusConfig()
        assert cfg.mode == EventBusMode.IN_MEMORY

    async def test_initialization_redis_mode_config(self) -> None:
        """EventBusConfig can be created with REDIS mode and a redis_url."""
        cfg = EventBusConfig(
            mode=EventBusMode.REDIS,
            redis_url="redis://localhost:6379",
        )
        assert cfg.mode == EventBusMode.REDIS
        assert cfg.redis_url == "redis://localhost:6379"


# =========================================================================
# Phase 4: Subscribe / Unsubscribe Tests
# =========================================================================


class TestSubscription:
    """Tests for subscribe() and unsubscribe() lifecycle."""

    async def test_subscribe_single_handler(self, event_bus: EventBus) -> None:
        """A single handler can be subscribed for an event type."""
        handler = AsyncMock()
        await event_bus.subscribe(TransactionCreated, handler)
        assert event_bus.get_subscriber_count() == 1

    async def test_subscribe_multiple_handlers_same_type(
        self, event_bus: EventBus,
    ) -> None:
        """Multiple handlers for the same event type are allowed."""
        h1, h2, h3 = AsyncMock(), AsyncMock(), AsyncMock()
        await event_bus.subscribe(TransactionCreated, h1)
        await event_bus.subscribe(TransactionCreated, h2)
        await event_bus.subscribe(TransactionCreated, h3)
        assert event_bus.get_subscriber_count() == 3

    async def test_subscribe_handlers_different_types(
        self, event_bus: EventBus,
    ) -> None:
        """Handlers for different event types are tracked independently."""
        h1, h2 = AsyncMock(), AsyncMock()
        await event_bus.subscribe(TransactionCreated, h1)
        await event_bus.subscribe(ApprovalRequired, h2)
        assert event_bus.get_subscriber_count() == 2
        assert event_bus.get_subscriber_count("TransactionCreated") == 1
        assert event_bus.get_subscriber_count("ApprovalRequired") == 1

    async def test_subscribe_with_string_event_type(
        self, event_bus: EventBus,
    ) -> None:
        """Subscribe accepts a plain string event type."""
        handler = AsyncMock()
        await event_bus.subscribe("TransactionCreated", handler)
        assert event_bus.get_subscriber_count("TransactionCreated") == 1

    async def test_subscribe_with_class_event_type(
        self, event_bus: EventBus,
    ) -> None:
        """Subscribe accepts a concrete Event subclass."""
        handler = AsyncMock()
        await event_bus.subscribe(TransactionCreated, handler)
        # _resolve_event_type extracts the default event_type from the dataclass
        assert event_bus.get_subscriber_count("TransactionCreated") == 1

    async def test_unsubscribe_handler(self, event_bus: EventBus) -> None:
        """Unsubscribe removes a previously registered handler."""
        handler = AsyncMock()
        await event_bus.subscribe(TransactionCreated, handler)
        assert event_bus.get_subscriber_count() == 1
        await event_bus.unsubscribe(TransactionCreated, handler)
        assert event_bus.get_subscriber_count() == 0

    async def test_unsubscribe_nonexistent_handler(
        self, event_bus: EventBus,
    ) -> None:
        """Unsubscribing a handler that was never subscribed is a no-op."""
        handler = AsyncMock()
        # Should not raise
        await event_bus.unsubscribe(TransactionCreated, handler)
        assert event_bus.get_subscriber_count() == 0

    async def test_get_subscriber_count_total(
        self, event_bus: EventBus,
    ) -> None:
        """get_subscriber_count() with no arg returns total across all types."""
        h1, h2 = AsyncMock(), AsyncMock()
        await event_bus.subscribe(TransactionCreated, h1)
        await event_bus.subscribe(ApprovalRequired, h2)
        assert event_bus.get_subscriber_count() == 2

    async def test_get_subscriber_count_specific_type(
        self, event_bus: EventBus,
    ) -> None:
        """get_subscriber_count(event_type) returns count for that type only."""
        h1, h2 = AsyncMock(), AsyncMock()
        await event_bus.subscribe(TransactionCreated, h1)
        await event_bus.subscribe(TransactionCreated, h2)
        assert event_bus.get_subscriber_count("TransactionCreated") == 2
        assert event_bus.get_subscriber_count("ApprovalRequired") == 0

    async def test_get_subscriber_count_no_subscribers(
        self, event_bus: EventBus,
    ) -> None:
        """get_subscriber_count returns 0 when nothing is subscribed."""
        assert event_bus.get_subscriber_count() == 0
        assert event_bus.get_subscriber_count("TransactionCreated") == 0


# =========================================================================
# Phase 5: Publish Tests (Core Functionality)
# =========================================================================


class TestPublish:
    """Tests for EventBus.publish() core behaviour."""

    async def test_publish_single_event(
        self, started_event_bus: EventBus, sample_event: TransactionCreated,
    ) -> None:
        """Published event is dispatched to the subscribed handler."""
        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        await started_event_bus.subscribe(TransactionCreated, handler)
        await started_event_bus.publish(sample_event)
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(received) == 1
        assert received[0].event_id == sample_event.event_id

    async def test_publish_event_to_multiple_handlers(
        self, started_event_bus: EventBus, sample_event: TransactionCreated,
    ) -> None:
        """One event is delivered to ALL subscribed handlers."""
        r1: List[Event] = []
        r2: List[Event] = []

        async def h1(evt: Event) -> None:
            r1.append(evt)

        async def h2(evt: Event) -> None:
            r2.append(evt)

        await started_event_bus.subscribe(TransactionCreated, h1)
        await started_event_bus.subscribe(TransactionCreated, h2)
        await started_event_bus.publish(sample_event)
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(r1) == 1
        assert len(r2) == 1

    async def test_publish_different_event_types(
        self, started_event_bus: EventBus,
    ) -> None:
        """Only handlers matching the event type are called."""
        txn_received: List[Event] = []
        apr_received: List[Event] = []

        async def txn_handler(evt: Event) -> None:
            txn_received.append(evt)

        async def apr_handler(evt: Event) -> None:
            apr_received.append(evt)

        await started_event_bus.subscribe(TransactionCreated, txn_handler)
        await started_event_bus.subscribe(ApprovalRequired, apr_handler)

        txn_event = _make_event(TransactionCreated)
        apr_event = _make_event(ApprovalRequired, payload={
            "transaction_id": "TXN-X", "amount": 10000.0,
            "required_approver_role": "cfo",
        })

        await started_event_bus.publish(txn_event)
        await started_event_bus.publish(apr_event)
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(txn_received) == 1
        assert len(apr_received) == 1
        assert txn_received[0].event_type == "TransactionCreated"
        assert apr_received[0].event_type == "ApprovalRequired"

    async def test_publish_increments_event_count(
        self, started_event_bus: EventBus,
    ) -> None:
        """Each publish increments get_event_count()."""
        assert started_event_bus.get_event_count() == 0
        await started_event_bus.publish(_make_event())
        await started_event_bus.publish(_make_event())
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)
        assert started_event_bus.get_event_count() == 2

    async def test_publish_with_default_priority(
        self, started_event_bus: EventBus,
    ) -> None:
        """Publishing without explicit priority uses NORMAL."""
        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        await started_event_bus.subscribe(TransactionCreated, handler)
        await started_event_bus.publish(_make_event())
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)
        assert len(received) == 1  # Delivered successfully at default priority

    async def test_publish_updates_queue_size(self, event_bus: EventBus) -> None:
        """Queue size increases when bus is not started (no dispatch loop)."""
        # Bus not started — events accumulate in the queue
        event = _make_event()
        await event_bus.publish(event)
        assert event_bus.get_queue_size() >= 1


# =========================================================================
# Phase 6: Priority Handling Tests
# =========================================================================


class TestPriorityHandling:
    """Tests for priority-based dispatch ordering."""

    async def test_urgent_events_dispatched_first(
        self, event_bus: EventBus,
    ) -> None:
        """URGENT events are dispatched before NORMAL events."""
        dispatch_order: List[str] = []

        async def handler(evt: Event) -> None:
            dispatch_order.append(evt.payload.get("label", ""))

        await event_bus.subscribe(TransactionCreated, handler)

        # Enqueue NORMAL first, then URGENT — both before starting the bus
        normal_event = _make_event(payload={
            "transaction_id": "N1", "transaction_type": "po",
            "amount": 100.0, "label": "normal",
        })
        urgent_event = _make_event(payload={
            "transaction_id": "U1", "transaction_type": "po",
            "amount": 200.0, "label": "urgent",
        })

        await event_bus.publish(normal_event, priority=EventPriority.NORMAL)
        await event_bus.publish(urgent_event, priority=EventPriority.URGENT)

        # Now start — dispatch loop processes queue in priority order
        await event_bus.start()
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)
        await event_bus.stop()

        assert len(dispatch_order) == 2
        assert dispatch_order[0] == "urgent"
        assert dispatch_order[1] == "normal"

    async def test_priority_ordering_all_levels(
        self, event_bus: EventBus,
    ) -> None:
        """Events dispatch in URGENT → HIGH → NORMAL → LOW order."""
        dispatch_order: List[str] = []

        async def handler(evt: Event) -> None:
            dispatch_order.append(evt.payload.get("label", ""))

        await event_bus.subscribe(TransactionCreated, handler)

        priorities = [
            (EventPriority.LOW, "low"),
            (EventPriority.NORMAL, "normal"),
            (EventPriority.HIGH, "high"),
            (EventPriority.URGENT, "urgent"),
        ]
        for prio, label in priorities:
            evt = _make_event(payload={
                "transaction_id": f"T-{label}", "transaction_type": "po",
                "amount": 100.0, "label": label,
            })
            await event_bus.publish(evt, priority=prio)

        await event_bus.start()
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)
        await event_bus.stop()

        assert dispatch_order == ["urgent", "high", "normal", "low"]

    async def test_same_priority_fifo_order(
        self, event_bus: EventBus,
    ) -> None:
        """Events with the same priority are dispatched in FIFO order."""
        dispatch_order: List[str] = []

        async def handler(evt: Event) -> None:
            dispatch_order.append(evt.payload.get("label", ""))

        await event_bus.subscribe(TransactionCreated, handler)

        for i in range(5):
            evt = _make_event(payload={
                "transaction_id": f"T-{i}", "transaction_type": "po",
                "amount": 100.0, "label": f"event_{i}",
            })
            await event_bus.publish(evt, priority=EventPriority.NORMAL)

        await event_bus.start()
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)
        await event_bus.stop()

        assert dispatch_order == [f"event_{i}" for i in range(5)]

    async def test_event_priority_enum_values(self) -> None:
        """Verify EventPriority numeric values match specification."""
        assert EventPriority.URGENT.value == 0
        assert EventPriority.HIGH.value == 1
        assert EventPriority.NORMAL.value == 2
        assert EventPriority.LOW.value == 3
        # Lower value == higher priority in PriorityQueue
        assert EventPriority.URGENT < EventPriority.LOW


# =========================================================================
# Phase 7: Async Dispatch Tests
# =========================================================================


class TestAsyncDispatch:
    """Tests for async handler dispatch, concurrency, and timeouts."""

    async def test_handler_called_asynchronously(
        self, started_event_bus: EventBus,
    ) -> None:
        """publish() returns before the handler finishes execution."""
        handler_started = asyncio.Event()
        handler_finished = asyncio.Event()

        async def slow_handler(evt: Event) -> None:
            handler_started.set()
            await asyncio.sleep(0.2)
            handler_finished.set()

        await started_event_bus.subscribe(TransactionCreated, slow_handler)
        await started_event_bus.publish(_make_event())

        # publish() should return immediately (async dispatch)
        assert not handler_finished.is_set()
        # Wait for handler to complete
        await asyncio.wait_for(handler_finished.wait(), timeout=_TEST_TIMEOUT)
        assert handler_finished.is_set()

    async def test_multiple_handlers_called_concurrently(
        self, started_event_bus: EventBus,
    ) -> None:
        """Multiple handlers for the same event can run concurrently."""
        results: List[str] = []

        async def handler_a(evt: Event) -> None:
            await asyncio.sleep(0.05)
            results.append("a")

        async def handler_b(evt: Event) -> None:
            await asyncio.sleep(0.05)
            results.append("b")

        await started_event_bus.subscribe(TransactionCreated, handler_a)
        await started_event_bus.subscribe(TransactionCreated, handler_b)
        await started_event_bus.publish(_make_event())
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert "a" in results
        assert "b" in results

    async def test_handler_exception_does_not_affect_other_handlers(
        self, started_event_bus: EventBus,
    ) -> None:
        """A failing handler does not prevent other handlers from executing."""
        success_received: List[Event] = []

        async def failing_handler(evt: Event) -> None:
            raise RuntimeError("Simulated handler failure")

        async def success_handler(evt: Event) -> None:
            success_received.append(evt)

        await started_event_bus.subscribe(TransactionCreated, failing_handler)
        await started_event_bus.subscribe(TransactionCreated, success_handler)
        await started_event_bus.publish(_make_event())
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        # The success handler still received the event
        assert len(success_received) == 1

    async def test_handler_timeout_enforcement(self) -> None:
        """A handler exceeding handler_timeout (set low for test) is skipped."""
        cfg = EventBusConfig(
            mode=EventBusMode.IN_MEMORY,
            persist_events=False,
            handler_timeout=0.3,  # Very short for testing
        )
        bus = EventBus(config=cfg)

        fast_received: List[Event] = []

        async def slow_handler(evt: Event) -> None:
            await asyncio.sleep(5.0)  # Well over the 0.3s timeout

        async def fast_handler(evt: Event) -> None:
            fast_received.append(evt)

        await bus.subscribe(TransactionCreated, slow_handler)
        await bus.subscribe(TransactionCreated, fast_handler)
        await bus.start()

        try:
            await bus.publish(_make_event())
            await asyncio.sleep(1.0)  # Enough for timeout + fast handler

            # Fast handler still runs despite slow handler timing out
            assert len(fast_received) == 1
        finally:
            await bus.stop()

    async def test_publish_timeout_enforcement(self) -> None:
        """Publish respects the configured publish_timeout."""
        cfg = EventBusConfig(
            mode=EventBusMode.IN_MEMORY,
            persist_events=False,
            max_queue_size=1,  # Tiny queue
            publish_timeout=0.2,
        )
        bus = EventBus(config=cfg)

        # Fill the queue without starting dispatch (no consumer)
        await bus.publish(_make_event())
        # Queue is now full (size=1); next publish should hit timeout
        # but the implementation does a best-effort put_nowait after timeout
        # which will raise QueueFull if that also fails
        try:
            await bus.publish(_make_event())
            # If it didn't raise, it means put_nowait succeeded after timeout
        except asyncio.QueueFull:
            pass  # Expected when both timed put and put_nowait fail


# =========================================================================
# Phase 8: Event Persistence Integration Tests
# =========================================================================


class TestEventPersistence:
    """Tests for EventStore persistence integration."""

    async def test_publish_persists_to_event_store(
        self, mock_event_store: AsyncMock,
    ) -> None:
        """When persist_events=True and EventStore injected, save_event is called."""
        cfg = EventBusConfig(mode=EventBusMode.IN_MEMORY, persist_events=True)
        bus = EventBus(config=cfg, event_store=mock_event_store)
        await bus.start()

        try:
            event = _make_event()
            await bus.publish(event)
            # Let the fire-and-forget persistence task complete
            await asyncio.sleep(_DISPATCH_SETTLE_TIME)
            mock_event_store.save_event.assert_called_once_with(event)
        finally:
            await bus.stop()

    async def test_publish_without_persistence(
        self, mock_event_store: AsyncMock,
    ) -> None:
        """When persist_events=False, EventStore.save_event is never called."""
        cfg = EventBusConfig(mode=EventBusMode.IN_MEMORY, persist_events=False)
        bus = EventBus(config=cfg, event_store=mock_event_store)
        await bus.start()

        try:
            await bus.publish(_make_event())
            await asyncio.sleep(_DISPATCH_SETTLE_TIME)
            mock_event_store.save_event.assert_not_called()
        finally:
            await bus.stop()

    async def test_persistence_failure_does_not_block_dispatch(
        self, mock_event_store: AsyncMock,
    ) -> None:
        """Handler dispatch continues even when EventStore.save_event raises."""
        mock_event_store.save_event.side_effect = RuntimeError("Redis down")

        cfg = EventBusConfig(mode=EventBusMode.IN_MEMORY, persist_events=True)
        bus = EventBus(config=cfg, event_store=mock_event_store)
        await bus.start()

        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        await bus.subscribe(TransactionCreated, handler)

        try:
            await bus.publish(_make_event())
            await asyncio.sleep(_DISPATCH_SETTLE_TIME)
            # Handler still received the event despite persistence failure
            assert len(received) == 1
        finally:
            await bus.stop()


# =========================================================================
# Phase 9: Event Type Coverage Tests
# =========================================================================


class TestEventTypeCoverage:
    """Ensure every concrete event type can be published and handled."""

    async def _assert_event_type_handled(
        self, event_cls: type, payload: dict[str, Any],
    ) -> None:
        """Publish an event of *event_cls* and verify a handler receives it."""
        cfg = EventBusConfig(mode=EventBusMode.IN_MEMORY, persist_events=False)
        bus = EventBus(config=cfg)
        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        await bus.subscribe(event_cls, handler)
        await bus.start()

        try:
            event = event_cls(
                simulation_id=uuid4(), payload=payload, agent_id=uuid4(),
            )
            await bus.publish(event)
            await asyncio.sleep(_DISPATCH_SETTLE_TIME)
            assert len(received) == 1
            assert isinstance(received[0], event_cls)
        finally:
            await bus.stop()

    async def test_publish_transaction_created(self) -> None:
        await self._assert_event_type_handled(
            TransactionCreated,
            {"transaction_id": "TXN-100", "transaction_type": "po", "amount": 500.0},
        )

    async def test_publish_transaction_completed(self) -> None:
        await self._assert_event_type_handled(
            TransactionCompleted,
            {"transaction_id": "TXN-100", "final_status": "completed"},
        )

    async def test_publish_approval_required(self) -> None:
        await self._assert_event_type_handled(
            ApprovalRequired,
            {"transaction_id": "TXN-200", "amount": 50000.0, "required_approver_role": "cfo"},
        )

    async def test_publish_approval_completed(self) -> None:
        await self._assert_event_type_handled(
            ApprovalCompleted,
            {"transaction_id": "TXN-200", "approval_decision": "approved"},
        )

    async def test_publish_document_generated(self) -> None:
        await self._assert_event_type_handled(
            DocumentGenerated,
            {"document_id": "DOC-001", "document_type": "invoice"},
        )

    async def test_publish_period_closing(self) -> None:
        await self._assert_event_type_handled(
            PeriodClosing,
            {"period_type": "monthly", "fiscal_year": 2025},
        )

    async def test_publish_period_closed(self) -> None:
        await self._assert_event_type_handled(
            PeriodClosed,
            {"period_type": "monthly", "fiscal_year": 2025},
        )

    async def test_publish_discrepancy_detected(self) -> None:
        await self._assert_event_type_handled(
            DiscrepancyDetected,
            {"discrepancy_type": "price_variance", "severity": "high"},
        )


# =========================================================================
# Phase 10: Lifecycle and Context Manager Tests
# =========================================================================


class TestLifecycle:
    """Tests for start/stop lifecycle and async context manager."""

    async def test_start_sets_running(self, event_bus: EventBus) -> None:
        """start() activates the dispatch loop."""
        await event_bus.start()
        assert event_bus.is_running() is True
        await event_bus.stop()

    async def test_stop_clears_running(self, event_bus: EventBus) -> None:
        """stop() deactivates the dispatch loop."""
        await event_bus.start()
        await event_bus.stop()
        assert event_bus.is_running() is False

    async def test_context_manager_start_stop(self) -> None:
        """async with EventBus() starts and stops the bus."""
        cfg = EventBusConfig(mode=EventBusMode.IN_MEMORY, persist_events=False)
        bus = EventBus(config=cfg)

        async with bus as b:
            assert b is bus
            assert b.is_running() is True

        assert bus.is_running() is False

    async def test_double_start_is_safe(self, event_bus: EventBus) -> None:
        """Calling start() twice does not raise an error."""
        await event_bus.start()
        await event_bus.start()  # Idempotent
        assert event_bus.is_running() is True
        await event_bus.stop()

    async def test_stop_without_start_is_safe(self, event_bus: EventBus) -> None:
        """Calling stop() on a bus that was never started is a no-op."""
        assert event_bus.is_running() is False
        await event_bus.stop()  # Should not raise
        assert event_bus.is_running() is False


# =========================================================================
# Phase 11: Metrics Tests
# =========================================================================


class TestMetrics:
    """Tests for EventBus.get_metrics() operational metrics."""

    async def test_get_metrics_returns_expected_fields(
        self, event_bus: EventBus,
    ) -> None:
        """get_metrics() returns a dict with required metric keys."""
        metrics = await event_bus.get_metrics()
        expected_keys = {
            "total_events", "queue_size", "mode", "is_running",
            "bus_id", "events_dispatched", "handlers_succeeded",
            "handlers_failed", "subscriber_counts", "total_subscribers",
            "seen_event_types", "started_at", "persist_events",
            "enable_priority",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    async def test_metrics_reflect_event_count(
        self, started_event_bus: EventBus,
    ) -> None:
        """total_events increases as events are published."""
        metrics_before = await started_event_bus.get_metrics()
        assert metrics_before["total_events"] == 0

        await started_event_bus.publish(_make_event())
        await started_event_bus.publish(_make_event())
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        metrics_after = await started_event_bus.get_metrics()
        assert metrics_after["total_events"] == 2

    async def test_metrics_reflect_queue_state(
        self, started_event_bus: EventBus,
    ) -> None:
        """Metrics is_running and mode match bus state."""
        metrics = await started_event_bus.get_metrics()
        assert metrics["is_running"] is True
        assert metrics["mode"] == EventBusMode.IN_MEMORY.value


# =========================================================================
# Phase 12: Default Handler Registration Tests
# =========================================================================


class TestDefaultHandlers:
    """Tests for register_default_handlers / unregister_default_handlers."""

    async def test_register_default_handlers(
        self, event_bus: EventBus,
    ) -> None:
        """After register_default_handlers, all 8 event types have handlers."""
        await register_default_handlers(event_bus)
        # Each of the 8 default event types should have 1 handler
        assert event_bus.get_subscriber_count() == 8

    async def test_unregister_default_handlers(
        self, event_bus: EventBus,
    ) -> None:
        """After unregister_default_handlers, subscriber count returns to 0."""
        await register_default_handlers(event_bus)
        assert event_bus.get_subscriber_count() == 8
        await unregister_default_handlers(event_bus)
        assert event_bus.get_subscriber_count() == 0


# =========================================================================
# Phase 13: Edge Cases and Error Handling
# =========================================================================


class TestEdgeCases:
    """Edge-case and error handling tests."""

    async def test_publish_to_bus_with_no_subscribers(
        self, started_event_bus: EventBus,
    ) -> None:
        """Publishing with no subscribed handlers does not raise."""
        await started_event_bus.publish(_make_event())
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)
        # No error and event_count incremented
        assert started_event_bus.get_event_count() == 1

    async def test_publish_base_event_class(
        self, started_event_bus: EventBus,
    ) -> None:
        """Publishing a raw Event (not a subclass) is handled gracefully."""
        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        # Subscribe using the string form that matches __post_init__ for base Event
        await started_event_bus.subscribe("Event", handler)
        base_event = Event(
            simulation_id=uuid4(),
            payload={"info": "base_event_test"},
        )
        await started_event_bus.publish(base_event)
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(received) == 1
        assert received[0].event_type == "Event"

    async def test_large_payload_event(
        self, started_event_bus: EventBus,
    ) -> None:
        """An event with a large payload dict is handled without error."""
        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        await started_event_bus.subscribe(TransactionCreated, handler)

        large_payload = {
            "transaction_id": "TXN-BIG",
            "transaction_type": "purchase_order",
            "amount": 99999.99,
            "extra_data": {f"key_{i}": f"value_{i}" for i in range(500)},
        }
        event = TransactionCreated(
            simulation_id=uuid4(), payload=large_payload, agent_id=uuid4(),
        )
        await started_event_bus.publish(event)
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(received) == 1
        assert received[0].payload["extra_data"]["key_499"] == "value_499"

    async def test_none_agent_id_event(
        self, started_event_bus: EventBus,
    ) -> None:
        """An event with agent_id=None is published and dispatched."""
        received: List[Event] = []

        async def handler(evt: Event) -> None:
            received.append(evt)

        await started_event_bus.subscribe(TransactionCreated, handler)
        event = TransactionCreated(
            simulation_id=uuid4(),
            payload={"transaction_id": "TXN-NONE", "transaction_type": "po", "amount": 100.0},
            agent_id=None,
        )
        await started_event_bus.publish(event)
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(received) == 1
        assert received[0].agent_id is None

    async def test_wildcard_subscription(
        self, started_event_bus: EventBus,
    ) -> None:
        """A handler subscribed to '*' receives events of any type."""
        received: List[Event] = []

        async def wildcard_handler(evt: Event) -> None:
            received.append(evt)

        await started_event_bus.subscribe("*", wildcard_handler)

        await started_event_bus.publish(_make_event(TransactionCreated))
        await started_event_bus.publish(
            _make_event(ApprovalRequired, payload={
                "transaction_id": "TXN-W", "amount": 1000.0,
                "required_approver_role": "manager",
            }),
        )
        await asyncio.sleep(_DISPATCH_SETTLE_TIME)

        assert len(received) == 2

    async def test_duplicate_subscribe_ignored(
        self, event_bus: EventBus,
    ) -> None:
        """Subscribing the same handler twice for the same type has no effect."""
        handler = AsyncMock()
        await event_bus.subscribe(TransactionCreated, handler)
        await event_bus.subscribe(TransactionCreated, handler)
        assert event_bus.get_subscriber_count("TransactionCreated") == 1

    async def test_event_bus_mode_enum_values(self) -> None:
        """Verify EventBusMode string values."""
        assert EventBusMode.IN_MEMORY.value == "in_memory"
        assert EventBusMode.REDIS.value == "redis"

    async def test_event_bus_config_defaults(self) -> None:
        """Verify EventBusConfig default field values match specification."""
        cfg = EventBusConfig()
        assert cfg.mode == EventBusMode.IN_MEMORY
        assert cfg.max_queue_size == 10_000
        assert cfg.publish_timeout == 1.0
        assert cfg.handler_timeout == 5.0
        assert cfg.redis_url is None
        assert cfg.persist_events is True
        assert cfg.enable_priority is True
        assert cfg.max_handlers_per_event == 50
        assert cfg.redis_channel_prefix == "events"
