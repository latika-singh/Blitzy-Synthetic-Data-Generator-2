"""Comprehensive tests for EventStore JSONB persistent event storage.

Tests cover the full EventStore API: save_event, get_event, get_events (with
query filters for simulation_id, event_type, start_date, end_date, limit),
replay_events (event sourcing), delete_event, get_event_count, get_metrics,
connect/disconnect lifecycle, and async context manager support.

CRITICAL Testing Rules (AAP Section 0.7.5):
    - ALL async tests use pytest-asyncio with proper event loop management.
    - Redis tests use fakeredis in-memory mock — NO external Redis dependency.
    - Unit test coverage target: ≥ 80% for the EventStore module.

Redis Key Patterns and TTLs (AAP Section 0.4.1 / 0.4.4):
    - Primary key: ``event:{event_id}`` (``event:{uuid}``)
    - Event TTL: 604,800 seconds (7 days)
    - Auto-cleanup interval: 3,600 seconds
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import fakeredis
import fakeredis.aioredis
import pytest
import pytest_asyncio

from app.events.event_store import EventStore, EventStoreConfig
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
# Fixtures — local to this test module
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fake_redis() -> fakeredis.aioredis.FakeRedis:
    """Provide an async fakeredis client for EventStore tests.

    Yields a fresh FakeRedis instance with ``decode_responses=True``
    (matching EventStore's connect behaviour) and cleans up after the
    test by flushing all data and closing the connection.
    """
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield redis_client
    await redis_client.flushall()
    await redis_client.aclose()


@pytest.fixture
def store_config() -> EventStoreConfig:
    """Provide a default EventStoreConfig for tests.

    Uses all specification-mandated defaults:
        - key_prefix = "event"
        - event_ttl_seconds = 604800 (7 days)
        - max_events_per_query = 1000
        - cleanup_interval_seconds = 3600
        - enable_indexing = True
    """
    return EventStoreConfig()


@pytest_asyncio.fixture
async def event_store(
    fake_redis: fakeredis.aioredis.FakeRedis,
    store_config: EventStoreConfig,
) -> EventStore:
    """Provide a connected EventStore backed by fakeredis.

    The store is ready to use immediately (connection established via the
    injected fake_redis client).  Cleanup calls ``disconnect()`` after the
    test completes.
    """
    store = EventStore(config=store_config, redis_client=fake_redis)
    yield store
    try:
        await store.disconnect()
    except Exception:
        pass


@pytest.fixture
def sample_simulation_id() -> UUID:
    """Return a fixed UUID for deterministic testing."""
    return UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@pytest.fixture
def sample_event(sample_simulation_id: UUID) -> TransactionCreated:
    """Return a single TransactionCreated event with known data."""
    return TransactionCreated(
        simulation_id=sample_simulation_id,
        payload={
            "transaction_id": "TXN-001",
            "transaction_type": "purchase_order",
            "amount": 5000.00,
            "currency": "USD",
        },
        agent_id=uuid4(),
    )


@pytest.fixture
def sample_events_batch(sample_simulation_id: UUID) -> List[Event]:
    """Return a batch of 5 events of different types, same simulation.

    Each event has an explicit timestamp with increasing offsets so that
    chronological ordering can be reliably verified by replay and query tests.
    """
    base_time = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    agent = uuid4()

    return [
        TransactionCreated(
            simulation_id=sample_simulation_id,
            payload={
                "transaction_id": "TXN-BATCH-001",
                "transaction_type": "purchase_order",
                "amount": 3000.00,
            },
            timestamp=base_time,
            agent_id=agent,
        ),
        ApprovalRequired(
            simulation_id=sample_simulation_id,
            payload={
                "transaction_id": "TXN-BATCH-001",
                "required_approver_role": "purchasing_manager",
                "amount": 3000.00,
            },
            timestamp=base_time + timedelta(minutes=1),
            agent_id=agent,
        ),
        ApprovalCompleted(
            simulation_id=sample_simulation_id,
            payload={
                "transaction_id": "TXN-BATCH-001",
                "approval_decision": "approved",
                "approver_agent_id": str(uuid4()),
            },
            timestamp=base_time + timedelta(minutes=5),
            agent_id=agent,
        ),
        DocumentGenerated(
            simulation_id=sample_simulation_id,
            payload={
                "document_id": "DOC-001",
                "document_type": "purchase_order_pdf",
                "transaction_id": "TXN-BATCH-001",
            },
            timestamp=base_time + timedelta(minutes=10),
            agent_id=agent,
        ),
        TransactionCompleted(
            simulation_id=sample_simulation_id,
            payload={
                "transaction_id": "TXN-BATCH-001",
                "final_status": "completed",
                "duration_seconds": 900.0,
            },
            timestamp=base_time + timedelta(minutes=15),
            agent_id=agent,
        ),
    ]


@pytest.fixture
def events_different_simulations() -> List[Event]:
    """Return events belonging to two different simulation IDs.

    Simulation A gets 3 events, Simulation B gets 2 events.  Useful for
    verifying that query filters correctly isolate events by simulation.
    """
    sim_a = UUID("11111111-1111-1111-1111-111111111111")
    sim_b = UUID("22222222-2222-2222-2222-222222222222")
    base_time = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    return [
        # Simulation A — 3 events
        TransactionCreated(
            simulation_id=sim_a,
            payload={"transaction_id": "A-TXN-001"},
            timestamp=base_time,
        ),
        ApprovalRequired(
            simulation_id=sim_a,
            payload={
                "transaction_id": "A-TXN-001",
                "required_approver_role": "manager",
            },
            timestamp=base_time + timedelta(minutes=2),
        ),
        TransactionCompleted(
            simulation_id=sim_a,
            payload={
                "transaction_id": "A-TXN-001",
                "final_status": "completed",
            },
            timestamp=base_time + timedelta(minutes=10),
        ),
        # Simulation B — 2 events
        TransactionCreated(
            simulation_id=sim_b,
            payload={"transaction_id": "B-TXN-001"},
            timestamp=base_time + timedelta(minutes=1),
        ),
        ApprovalCompleted(
            simulation_id=sim_b,
            payload={
                "transaction_id": "B-TXN-001",
                "approval_decision": "approved",
            },
            timestamp=base_time + timedelta(minutes=5),
        ),
    ]


# ---------------------------------------------------------------------------
# Phase 3: EventStoreConfig Tests
# ---------------------------------------------------------------------------


class TestEventStoreConfig:
    """Validate EventStoreConfig Pydantic V2 model defaults and overrides."""

    def test_default_config(self) -> None:
        """All specification-mandated defaults are applied when no args given."""
        config = EventStoreConfig()
        assert config.key_prefix == "event"
        assert config.event_ttl_seconds == 604800
        assert config.max_events_per_query == 1000
        assert config.cleanup_interval_seconds == 3600
        assert config.enable_indexing is True
        assert config.index_prefix == "event_index"
        assert config.redis_url == "redis://localhost:6379"

    def test_custom_config(self) -> None:
        """Custom values override defaults correctly."""
        config = EventStoreConfig(
            redis_url="redis://custom:6380",
            key_prefix="custom_event",
            event_ttl_seconds=86400,
            index_prefix="custom_index",
            max_events_per_query=500,
            cleanup_interval_seconds=1800,
            enable_indexing=False,
        )
        assert config.redis_url == "redis://custom:6380"
        assert config.key_prefix == "custom_event"
        assert config.event_ttl_seconds == 86400
        assert config.index_prefix == "custom_index"
        assert config.max_events_per_query == 500
        assert config.cleanup_interval_seconds == 1800
        assert config.enable_indexing is False

    def test_ttl_matches_specification(self) -> None:
        """TTL must be exactly 604,800 seconds (7 days) per README.md line 1955."""
        config = EventStoreConfig()
        assert config.event_ttl_seconds == 604800, (
            "Default TTL must be 604800s (7 days) per specification"
        )
        assert config.event_ttl_seconds == 7 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# Phase 4: EventStore Initialization Tests
# ---------------------------------------------------------------------------


class TestEventStoreInitialization:
    """Verify EventStore construction and initial state."""

    async def test_initialization_with_redis_client(
        self, fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        """Store with an injected redis_client is immediately connected."""
        store = EventStore(redis_client=fake_redis)
        metrics = await store.get_metrics()
        assert metrics["connected"] is True
        assert metrics["total_events"] == 0

    async def test_initialization_with_config_only(self) -> None:
        """Store with only a config stores the config but is not connected."""
        config = EventStoreConfig(redis_url="redis://nowhere:6379")
        store = EventStore(config=config)
        metrics = await store.get_metrics()
        assert metrics["connected"] is False
        assert metrics["event_ttl_seconds"] == 604800

    async def test_initialization_with_defaults(self) -> None:
        """Store created with zero args applies default config."""
        store = EventStore()
        metrics = await store.get_metrics()
        assert metrics["connected"] is False
        assert metrics["event_ttl_seconds"] == 604800
        assert metrics["enable_indexing"] is True


# ---------------------------------------------------------------------------
# Phase 5: save_event Tests (Primary Persistence)
# ---------------------------------------------------------------------------


class TestSaveEvent:
    """Verify save_event stores events in Redis with correct keys and TTLs."""

    async def test_save_event_returns_event_id(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """save_event returns the event's UUID as a string."""
        result = await event_store.save_event(sample_event)
        assert isinstance(result, str)
        assert result == str(sample_event.event_id)

    async def test_save_event_stores_in_redis(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_event: TransactionCreated,
    ) -> None:
        """Saved event is retrievable directly from Redis under event:{id}."""
        event_id_str = await event_store.save_event(sample_event)

        primary_key = f"event:{event_id_str}"
        raw: Optional[str] = await fake_redis.get(primary_key)
        assert raw is not None, f"Key {primary_key} not found in Redis"

        data: Dict[str, Any] = json.loads(raw)
        assert "event_id" in data
        assert "simulation_id" in data
        assert "event_type" in data
        assert "payload" in data
        assert "timestamp" in data
        assert "agent_id" in data

    async def test_save_event_sets_ttl(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_event: TransactionCreated,
    ) -> None:
        """Stored event key has 7-day TTL (604800s) per specification."""
        event_id_str = await event_store.save_event(sample_event)
        primary_key = f"event:{event_id_str}"

        ttl: int = await fake_redis.ttl(primary_key)
        # Allow a small tolerance for execution time
        assert 604790 <= ttl <= 604800, (
            f"TTL {ttl} not within expected range [604790, 604800]"
        )

    async def test_save_event_serializes_uuid_fields(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_event: TransactionCreated,
    ) -> None:
        """UUID fields (event_id, simulation_id) are serialized as strings."""
        event_id_str = await event_store.save_event(sample_event)
        raw = await fake_redis.get(f"event:{event_id_str}")
        data = json.loads(raw)

        # Must be string representations, not UUID objects
        assert isinstance(data["event_id"], str)
        assert isinstance(data["simulation_id"], str)
        # Verify round-trip by reconstructing UUIDs
        assert UUID(data["event_id"]) == sample_event.event_id
        assert UUID(data["simulation_id"]) == sample_event.simulation_id

    async def test_save_event_serializes_datetime(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_event: TransactionCreated,
    ) -> None:
        """Timestamp is serialized as an ISO-8601 string."""
        event_id_str = await event_store.save_event(sample_event)
        raw = await fake_redis.get(f"event:{event_id_str}")
        data = json.loads(raw)

        ts_str = data["timestamp"]
        assert isinstance(ts_str, str)
        parsed = datetime.fromisoformat(ts_str)
        # Verify the round-trip preserves the value
        assert abs((parsed - sample_event.timestamp).total_seconds()) < 1.0

    async def test_save_event_serializes_payload_as_jsonb(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        """Complex nested payloads are correctly serialized as JSONB."""
        nested_payload = {
            "transaction_id": "TXN-NESTED",
            "details": {
                "line_items": [
                    {"sku": "ITEM-001", "qty": 10, "price": 99.99},
                    {"sku": "ITEM-002", "qty": 5, "price": 49.50},
                ],
                "metadata": {"source": "vendor_portal", "batch": True},
            },
        }
        sim_id = uuid4()
        event = TransactionCreated(
            simulation_id=sim_id,
            payload=nested_payload,
        )
        event_id_str = await event_store.save_event(event)
        raw = await fake_redis.get(f"event:{event_id_str}")
        data = json.loads(raw)

        assert data["payload"] == nested_payload
        assert data["payload"]["details"]["line_items"][0]["sku"] == "ITEM-001"

    async def test_save_event_with_none_agent_id(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_simulation_id: UUID,
    ) -> None:
        """Event with agent_id=None serializes agent_id as null/None."""
        event = TransactionCreated(
            simulation_id=sample_simulation_id,
            payload={"transaction_id": "TXN-NO-AGENT"},
            agent_id=None,
        )
        event_id_str = await event_store.save_event(event)
        raw = await fake_redis.get(f"event:{event_id_str}")
        data = json.loads(raw)

        assert data["agent_id"] is None

    async def test_save_multiple_events(
        self,
        event_store: EventStore,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_events_batch: List[Event],
    ) -> None:
        """Multiple events each get their own unique Redis key."""
        saved_ids: List[str] = []
        for event in sample_events_batch:
            eid = await event_store.save_event(event)
            saved_ids.append(eid)

        # All IDs are unique
        assert len(set(saved_ids)) == len(saved_ids)

        # Each event individually retrievable from Redis
        for eid in saved_ids:
            raw = await fake_redis.get(f"event:{eid}")
            assert raw is not None, f"Event {eid} not found in Redis"


# ---------------------------------------------------------------------------
# Phase 6: get_event (Single Event Retrieval) Tests
# ---------------------------------------------------------------------------


class TestGetEvent:
    """Verify get_event retrieves and deserializes events correctly."""

    async def test_get_event_by_id(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """Retrieve a saved event by its ID and verify all fields."""
        event_id_str = await event_store.save_event(sample_event)

        retrieved = await event_store.get_event(event_id_str)
        assert retrieved is not None
        assert str(retrieved.event_id) == event_id_str
        assert retrieved.simulation_id == sample_event.simulation_id
        assert retrieved.event_type == "TransactionCreated"
        assert retrieved.payload == sample_event.payload
        # agent_id round-trip
        if sample_event.agent_id is not None:
            assert retrieved.agent_id == sample_event.agent_id

    async def test_get_event_not_found(
        self, event_store: EventStore,
    ) -> None:
        """Requesting a non-existent event returns None."""
        result = await event_store.get_event(uuid4())
        assert result is None

    async def test_get_event_deserializes_correctly(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """Deserialized event is an Event instance with matching fields."""
        event_id_str = await event_store.save_event(sample_event)
        retrieved = await event_store.get_event(event_id_str)

        assert isinstance(retrieved, Event)
        assert retrieved.event_type == EventType.TRANSACTION_CREATED.value
        assert retrieved.payload["transaction_id"] == "TXN-001"
        assert retrieved.payload["amount"] == 5000.00

    async def test_get_event_accepts_string_id(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """get_event works when event_id is a plain string."""
        event_id_str = await event_store.save_event(sample_event)
        retrieved = await event_store.get_event(event_id_str)
        assert retrieved is not None
        assert str(retrieved.event_id) == event_id_str

    async def test_get_event_accepts_uuid_id(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """get_event works when event_id is a UUID object."""
        await event_store.save_event(sample_event)
        retrieved = await event_store.get_event(sample_event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == sample_event.event_id


# ---------------------------------------------------------------------------
# Phase 7: get_events (Query with Filters) Tests
# ---------------------------------------------------------------------------


class TestGetEvents:
    """Verify get_events query filtering, ordering, and limit behaviour."""

    async def test_get_events_by_simulation_id(
        self,
        event_store: EventStore,
        events_different_simulations: List[Event],
    ) -> None:
        """Only events for the requested simulation_id are returned."""
        for evt in events_different_simulations:
            await event_store.save_event(evt)

        sim_a = UUID("11111111-1111-1111-1111-111111111111")
        sim_b = UUID("22222222-2222-2222-2222-222222222222")

        results_a = await event_store.get_events(simulation_id=sim_a)
        results_b = await event_store.get_events(simulation_id=sim_b)

        assert len(results_a) == 3
        assert len(results_b) == 2

        # All results belong to the correct simulation
        for evt in results_a:
            assert evt.simulation_id == sim_a
        for evt in results_b:
            assert evt.simulation_id == sim_b

    async def test_get_events_filter_by_event_type(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
        sample_simulation_id: UUID,
    ) -> None:
        """Filtering by event_type returns only matching events."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
            event_type="TransactionCreated",
        )
        assert len(results) == 1
        assert results[0].event_type == "TransactionCreated"

    async def test_get_events_filter_by_start_date(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Events before start_date are excluded from results."""
        yesterday = datetime(2025, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
        today = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        tomorrow = datetime(2025, 6, 2, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"day": "yesterday"},
                timestamp=yesterday,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"day": "today"},
                timestamp=today,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"day": "tomorrow"},
                timestamp=tomorrow,
            ),
        ]
        for evt in events:
            await event_store.save_event(evt)

        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
            start_date=today,
        )
        assert len(results) == 2
        assert all(e.payload["day"] in ("today", "tomorrow") for e in results)

    async def test_get_events_filter_by_end_date(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Events after end_date are excluded from results."""
        yesterday = datetime(2025, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
        today = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        tomorrow = datetime(2025, 6, 2, 12, 0, 0, tzinfo=timezone.utc)

        events = [
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"day": "yesterday"},
                timestamp=yesterday,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"day": "today"},
                timestamp=today,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"day": "tomorrow"},
                timestamp=tomorrow,
            ),
        ]
        for evt in events:
            await event_store.save_event(evt)

        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
            end_date=today,
        )
        assert len(results) == 2
        assert all(e.payload["day"] in ("yesterday", "today") for e in results)

    async def test_get_events_filter_by_date_range(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Combined start_date and end_date restricts to the date window."""
        t1 = datetime(2025, 6, 1, 8, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        t3 = datetime(2025, 6, 1, 16, 0, 0, tzinfo=timezone.utc)
        t4 = datetime(2025, 6, 1, 20, 0, 0, tzinfo=timezone.utc)

        events = [
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"slot": "morning"},
                timestamp=t1,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"slot": "noon"},
                timestamp=t2,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"slot": "afternoon"},
                timestamp=t3,
            ),
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"slot": "evening"},
                timestamp=t4,
            ),
        ]
        for evt in events:
            await event_store.save_event(evt)

        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
            start_date=t2,
            end_date=t3,
        )
        assert len(results) == 2
        slots = {e.payload["slot"] for e in results}
        assert slots == {"noon", "afternoon"}

    async def test_get_events_combined_filters(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
        sample_simulation_id: UUID,
    ) -> None:
        """simulation_id + event_type + start_date applied together."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        # ApprovalRequired is at T+1min (base_time + 1 min)
        # Filter: after T+0.5min to include ApprovalRequired only
        base_time = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
            event_type="ApprovalRequired",
            start_date=base_time + timedelta(seconds=30),
        )
        assert len(results) == 1
        assert results[0].event_type == "ApprovalRequired"

    async def test_get_events_returns_empty_for_nonexistent_simulation(
        self, event_store: EventStore,
    ) -> None:
        """Querying a simulation with no events returns an empty list."""
        result = await event_store.get_events(
            simulation_id=UUID("99999999-9999-9999-9999-999999999999"),
        )
        assert result == []

    async def test_get_events_sorted_by_timestamp(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Returned events are sorted chronologically (ascending)."""
        t3 = datetime(2025, 6, 1, 15, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Save deliberately out of order
        for ts, label in [(t3, "third"), (t1, "first"), (t2, "second")]:
            evt = TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"order": label},
                timestamp=ts,
            )
            await event_store.save_event(evt)

        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
        )
        assert len(results) == 3
        assert results[0].payload["order"] == "first"
        assert results[1].payload["order"] == "second"
        assert results[2].payload["order"] == "third"

    async def test_get_events_respects_limit(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Specifying limit caps the number of returned events."""
        base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            evt = TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"index": i},
                timestamp=base + timedelta(minutes=i),
            )
            await event_store.save_event(evt)

        results = await event_store.get_events(
            simulation_id=sample_simulation_id,
            limit=5,
        )
        assert len(results) == 5

    async def test_get_events_default_limit(
        self, store_config: EventStoreConfig,
    ) -> None:
        """Default limit is max_events_per_query (1000)."""
        assert store_config.max_events_per_query == 1000


# ---------------------------------------------------------------------------
# Phase 8: replay_events (Event Sourcing) Tests
# ---------------------------------------------------------------------------


class TestReplayEvents:
    """Verify replay_events returns chronological events for state reconstruction."""

    async def test_replay_events_returns_chronological_order(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
        sample_simulation_id: UUID,
    ) -> None:
        """Replayed events are ordered by timestamp ascending."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        replayed = await event_store.replay_events(
            simulation_id=sample_simulation_id,
        )
        assert len(replayed) == 5

        # Verify chronological order
        for i in range(len(replayed) - 1):
            assert replayed[i].timestamp <= replayed[i + 1].timestamp

    async def test_replay_events_with_target_date(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
        sample_simulation_id: UUID,
    ) -> None:
        """Events after target_date are excluded from replay."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        # target_date at T+6min — includes TransactionCreated (T+0),
        # ApprovalRequired (T+1min), and ApprovalCompleted (T+5min)
        base_time = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
        cutoff = base_time + timedelta(minutes=6)

        replayed = await event_store.replay_events(
            simulation_id=sample_simulation_id,
            target_date=cutoff,
        )
        assert len(replayed) == 3
        # All replayed events are at or before the cutoff
        for evt in replayed:
            assert evt.timestamp <= cutoff

    async def test_replay_events_empty_simulation(
        self, event_store: EventStore,
    ) -> None:
        """Replaying a non-existent simulation returns an empty list."""
        result = await event_store.replay_events(
            simulation_id=UUID("99999999-9999-9999-9999-999999999999"),
        )
        assert result == []

    async def test_replay_events_preserves_event_types(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
        sample_simulation_id: UUID,
    ) -> None:
        """Each replayed event retains its original event_type."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        replayed = await event_store.replay_events(
            simulation_id=sample_simulation_id,
        )
        expected_types = [
            "TransactionCreated",
            "ApprovalRequired",
            "ApprovalCompleted",
            "DocumentGenerated",
            "TransactionCompleted",
        ]
        actual_types = [e.event_type for e in replayed]
        assert actual_types == expected_types

    async def test_replay_events_preserves_payloads(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
        sample_simulation_id: UUID,
    ) -> None:
        """Replayed event payloads are intact and match originals."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        replayed = await event_store.replay_events(
            simulation_id=sample_simulation_id,
        )

        for original, restored in zip(sample_events_batch, replayed):
            assert restored.payload == original.payload


# ---------------------------------------------------------------------------
# Phase 9: delete_event Tests
# ---------------------------------------------------------------------------


class TestDeleteEvent:
    """Verify delete_event removes events from Redis and indexes."""

    async def test_delete_event_removes_from_redis(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """Deleted event is no longer retrievable via get_event."""
        event_id_str = await event_store.save_event(sample_event)
        assert await event_store.get_event(event_id_str) is not None

        await event_store.delete_event(event_id_str)
        assert await event_store.get_event(event_id_str) is None

    async def test_delete_event_returns_true_on_success(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
    ) -> None:
        """Deleting an existing event returns True."""
        event_id_str = await event_store.save_event(sample_event)
        result = await event_store.delete_event(event_id_str)
        assert result is True

    async def test_delete_nonexistent_event_returns_false(
        self, event_store: EventStore,
    ) -> None:
        """Deleting a non-existent event returns False."""
        result = await event_store.delete_event(uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# Phase 10: Event Count and Metrics Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Verify event counting and operational metrics."""

    async def test_get_event_count_total(
        self,
        event_store: EventStore,
        sample_events_batch: List[Event],
    ) -> None:
        """Total event count matches the number of saved events."""
        for evt in sample_events_batch:
            await event_store.save_event(evt)

        count = await event_store.get_event_count()
        assert count == len(sample_events_batch)

    async def test_get_event_count_by_simulation_id(
        self,
        event_store: EventStore,
        events_different_simulations: List[Event],
    ) -> None:
        """Event count filtered by simulation_id returns correct totals."""
        for evt in events_different_simulations:
            await event_store.save_event(evt)

        sim_a = UUID("11111111-1111-1111-1111-111111111111")
        sim_b = UUID("22222222-2222-2222-2222-222222222222")

        count_a = await event_store.get_event_count(simulation_id=sim_a)
        count_b = await event_store.get_event_count(simulation_id=sim_b)
        assert count_a == 3
        assert count_b == 2

    async def test_get_metrics_returns_expected_fields(
        self, event_store: EventStore,
    ) -> None:
        """Metrics dictionary contains all mandatory fields."""
        metrics = await event_store.get_metrics()

        assert "total_events" in metrics
        assert "connected" in metrics
        assert "event_ttl_seconds" in metrics
        assert "store_id" in metrics
        assert "events_saved" in metrics
        assert "events_deleted" in metrics
        assert "events_queried" in metrics
        assert "enable_indexing" in metrics

        assert metrics["connected"] is True
        assert metrics["event_ttl_seconds"] == 604800


# ---------------------------------------------------------------------------
# Phase 11: Serialization Round-Trip Tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Verify save → retrieve round-trip fidelity for all event types."""

    async def test_event_round_trip_all_types(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """All 8 event types survive a save/retrieve round-trip intact."""
        agent = uuid4()
        type_payload_pairs = [
            (
                TransactionCreated,
                {
                    "transaction_id": "RT-TXN-001",
                    "transaction_type": "purchase_order",
                    "amount": 1500.00,
                    "currency": "USD",
                },
            ),
            (
                TransactionCompleted,
                {
                    "transaction_id": "RT-TXN-001",
                    "final_status": "completed",
                    "duration_seconds": 120.5,
                },
            ),
            (
                ApprovalRequired,
                {
                    "transaction_id": "RT-TXN-002",
                    "required_approver_role": "purchasing_manager",
                    "amount": 25000.00,
                    "threshold_level": "level_2",
                },
            ),
            (
                ApprovalCompleted,
                {
                    "transaction_id": "RT-TXN-002",
                    "approval_decision": "approved",
                    "reasoning": "Within budget allocation",
                },
            ),
            (
                DocumentGenerated,
                {
                    "document_id": "DOC-RT-001",
                    "document_type": "purchase_order_pdf",
                    "transaction_id": "RT-TXN-001",
                },
            ),
            (
                PeriodClosing,
                {
                    "period_type": "monthly",
                    "period_start": "2025-06-01",
                    "period_end": "2025-06-30",
                    "fiscal_year": 2025,
                },
            ),
            (
                PeriodClosed,
                {
                    "period_type": "monthly",
                    "period_start": "2025-06-01",
                    "period_end": "2025-06-30",
                    "fiscal_year": 2025,
                    "closing_summary": {"total_je": 42},
                },
            ),
            (
                DiscrepancyDetected,
                {
                    "discrepancy_type": "price_variance",
                    "transaction_id": "RT-TXN-003",
                    "severity": "medium",
                    "details": "PO price $100 vs Invoice $105",
                },
            ),
        ]

        for event_cls, payload in type_payload_pairs:
            original = event_cls(
                simulation_id=sample_simulation_id,
                payload=payload,
                agent_id=agent,
            )
            eid = await event_store.save_event(original)
            retrieved = await event_store.get_event(eid)

            assert retrieved is not None, (
                f"Round-trip failed for {event_cls.__name__}: event not found"
            )
            assert retrieved.event_type == original.event_type
            assert retrieved.simulation_id == original.simulation_id
            assert retrieved.payload == original.payload
            assert retrieved.agent_id == original.agent_id

    async def test_event_with_complex_nested_payload(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Deeply nested payload structures survive the round-trip."""
        nested = {
            "level_1": {
                "level_2": {
                    "level_3": {
                        "values": [1, 2.5, "three", None, True],
                        "nested_dict": {"key": "value"},
                    }
                }
            },
            "array_of_dicts": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
            ],
        }
        event = TransactionCreated(
            simulation_id=sample_simulation_id,
            payload=nested,
        )
        eid = await event_store.save_event(event)
        retrieved = await event_store.get_event(eid)

        assert retrieved is not None
        assert retrieved.payload == nested
        assert retrieved.payload["level_1"]["level_2"]["level_3"]["values"][2] == "three"

    async def test_event_with_empty_payload(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """An event with payload={} round-trips to an empty dict."""
        event = TransactionCreated(
            simulation_id=sample_simulation_id,
            payload={},
        )
        eid = await event_store.save_event(event)
        retrieved = await event_store.get_event(eid)

        assert retrieved is not None
        assert retrieved.payload == {}


# ---------------------------------------------------------------------------
# Phase 12: Context Manager Tests
# ---------------------------------------------------------------------------


class TestContextManager:
    """Verify async context manager lifecycle."""

    async def test_async_context_manager(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
        store_config: EventStoreConfig,
        sample_simulation_id: UUID,
    ) -> None:
        """EventStore is functional inside 'async with' and cleans up on exit."""
        async with EventStore(
            config=store_config, redis_client=fake_redis,
        ) as store:
            # Should be able to perform operations inside the context
            event = TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"context_manager": True},
            )
            eid = await store.save_event(event)
            retrieved = await store.get_event(eid)
            assert retrieved is not None
            assert retrieved.payload["context_manager"] is True

        # After context exit, the store should be disconnected
        metrics = await store.get_metrics()
        assert metrics["connected"] is False


# ---------------------------------------------------------------------------
# Phase 13: Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify edge-case behaviour and error resilience."""

    async def test_save_event_with_very_large_payload(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """A ~10 KB payload is saved and retrieved without corruption."""
        large_data = {
            f"key_{i}": f"value_{'x' * 100}_{i}" for i in range(80)
        }
        event = TransactionCreated(
            simulation_id=sample_simulation_id,
            payload=large_data,
        )
        eid = await event_store.save_event(event)
        retrieved = await event_store.get_event(eid)

        assert retrieved is not None
        assert retrieved.payload == large_data

    async def test_concurrent_save_events(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Multiple concurrent save operations all succeed."""
        events = [
            TransactionCreated(
                simulation_id=sample_simulation_id,
                payload={"concurrent_index": i},
                timestamp=datetime(2025, 6, 1, 10, i, 0, tzinfo=timezone.utc),
            )
            for i in range(20)
        ]

        event_ids = await asyncio.gather(
            *(event_store.save_event(e) for e in events)
        )

        # All 20 events saved with unique IDs
        assert len(event_ids) == 20
        assert len(set(event_ids)) == 20

        # All individually retrievable
        for eid in event_ids:
            retrieved = await event_store.get_event(eid)
            assert retrieved is not None

    async def test_get_events_with_no_indexes(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
        sample_simulation_id: UUID,
    ) -> None:
        """With enable_indexing=False, get_events returns empty (no index)."""
        config = EventStoreConfig(enable_indexing=False)
        store = EventStore(config=config, redis_client=fake_redis)

        event = TransactionCreated(
            simulation_id=sample_simulation_id,
            payload={"no_index": True},
        )
        eid = await store.save_event(event)

        # The event itself should be retrievable by direct key
        retrieved = await store.get_event(eid)
        assert retrieved is not None

        # But get_events uses indexes — without them, results are empty
        results = await store.get_events(
            simulation_id=sample_simulation_id,
        )
        assert results == []

    async def test_connection_error_without_connect(self) -> None:
        """Operations on a disconnected store raise ConnectionError."""
        store = EventStore()
        event = TransactionCreated(
            simulation_id=uuid4(),
            payload={"test": True},
        )
        with pytest.raises(ConnectionError):
            await store.save_event(event)

    async def test_delete_removes_from_indexes(
        self,
        event_store: EventStore,
        sample_event: TransactionCreated,
        sample_simulation_id: UUID,
    ) -> None:
        """After deletion, the event no longer appears in query results."""
        eid = await event_store.save_event(sample_event)
        events_before = await event_store.get_events(
            simulation_id=sample_simulation_id,
        )
        assert len(events_before) == 1

        await event_store.delete_event(eid)
        events_after = await event_store.get_events(
            simulation_id=sample_simulation_id,
        )
        assert len(events_after) == 0

    async def test_metrics_track_operations(
        self,
        event_store: EventStore,
        sample_simulation_id: UUID,
    ) -> None:
        """Metrics counters update after save, query, and delete."""
        event = TransactionCreated(
            simulation_id=sample_simulation_id,
            payload={"track": True},
        )
        eid = await event_store.save_event(event)
        await event_store.get_events(simulation_id=sample_simulation_id)
        await event_store.delete_event(eid)

        metrics = await event_store.get_metrics()
        assert metrics["events_saved"] >= 1
        assert metrics["events_queried"] >= 1
        assert metrics["events_deleted"] >= 1
