"""Persistent event storage with JSONB payloads, UUID indexing, query filters, and replay capability.

Redis-backed with 7-day TTL for event data. Provides save_event(), get_events() with query
filters (simulation_id, event_type, start_date/end_date), and replay_events() for state
reconstruction.

Key Design Decisions:
    - Primary storage uses Redis key-value pairs with ``event:{uuid}`` key pattern.
    - Secondary indexes use Redis sorted sets with UTC timestamps as scores, enabling
      efficient ZRANGEBYSCORE range queries.
    - Three index tiers: per-simulation, per-event-type, and combined simulation+type.
    - All event payloads are serialized as JSONB (JSON strings) in Redis.
    - 7-day TTL (604,800 seconds) is enforced on every stored event.
    - Cleanup runs on demand to remove orphaned index entries whose keys have expired.

References:
    - README.md lines 628–649: EventStore specification
    - AAP Section 0.4.1: Redis key pattern ``event:{event_id}``
    - AAP Section 0.4.4: Event TTL = 7 days (604,800 seconds)
    - AAP Section 0.7.2: All Redis interactions must set appropriate TTLs
    - AAP Section 0.7.4: Redis connection URLs with credentials must not appear in logs
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from app.events.event_types import (
    ApprovalCompleted,
    ApprovalRequired,
    DiscrepancyDetected,
    DocumentGenerated,
    Event,
    PeriodClosed,
    PeriodClosing,
    TransactionCompleted,
    TransactionCreated,
)

# Try importing redis.asyncio; degrade gracefully if not available so that
# the module can still be loaded for type-checking and testing without Redis.
try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Event type registry — maps event_type discriminator strings to concrete
# Event subclasses for deserialization.  Mirrors the EVENT_TYPE_REGISTRY in
# event_types.py but kept as a private constant here so that event_store
# never imports the public registry symbol (avoids tight coupling).
# ---------------------------------------------------------------------------

_EVENT_TYPE_REGISTRY: Dict[str, Type[Event]] = {
    "TransactionCreated": TransactionCreated,
    "TransactionCompleted": TransactionCompleted,
    "ApprovalRequired": ApprovalRequired,
    "ApprovalCompleted": ApprovalCompleted,
    "DocumentGenerated": DocumentGenerated,
    "PeriodClosing": PeriodClosing,
    "PeriodClosed": PeriodClosed,
    "DiscrepancyDetected": DiscrepancyDetected,
}

# Pattern used to strip credentials from Redis URLs before logging.
_REDIS_CRED_RE = re.compile(r"://[^@]+@")


def _sanitize_redis_url(url: str) -> str:
    """Remove embedded credentials from a Redis URL for safe logging.

    Replaces ``redis://user:password@host`` with ``redis://***@host`` so
    that connection URLs containing secrets never appear in structured logs.
    """
    return _REDIS_CRED_RE.sub("://***@", url)


# ---------------------------------------------------------------------------
# Configuration — Pydantic V2 model at the subsystem boundary
# ---------------------------------------------------------------------------


class EventStoreConfig(BaseModel):
    """Configuration for :class:`EventStore`.

    All fields carry sensible defaults derived from the specification so that
    an ``EventStore`` can be instantiated with zero arguments for development
    and testing.

    Attributes:
        redis_url: Redis connection URL.  Credentials in the URL are *never*
            written to logs.
        key_prefix: Prefix for primary event keys (``event:{uuid}``).
        event_ttl_seconds: Time-to-live for stored events (default 7 days).
        index_prefix: Prefix for sorted-set secondary indexes.
        max_events_per_query: Hard cap on events returned by a single query.
        cleanup_interval_seconds: Recommended interval between cleanup runs.
        enable_indexing: Whether to maintain sorted-set secondary indexes.
    """

    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for event persistence.",
    )
    key_prefix: str = Field(
        default="event",
        description="Prefix for primary event keys in Redis.",
    )
    event_ttl_seconds: int = Field(
        default=604800,
        ge=60,
        description="TTL for stored events in seconds (default 7 days = 604,800s).",
    )
    index_prefix: str = Field(
        default="event_index",
        description="Prefix for sorted-set secondary indexes.",
    )
    max_events_per_query: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of events returned by a single query.",
    )
    cleanup_interval_seconds: int = Field(
        default=3600,
        ge=60,
        description="Recommended cleanup interval in seconds.",
    )
    enable_indexing: bool = Field(
        default=True,
        description="Whether to maintain sorted-set secondary indexes.",
    )


# ---------------------------------------------------------------------------
# EventStore — persistent event storage
# ---------------------------------------------------------------------------


class EventStore:
    """Persistent event storage backed by Redis with JSONB payloads.

    The store provides full CRUD operations on events together with
    sorted-set secondary indexes for efficient range queries by simulation,
    event type, and timestamp.

    Usage as an async context manager::

        async with EventStore(config) as store:
            event_id = await store.save_event(my_event)
            events = await store.get_events(simulation_id)

    Constructor injection is used for all dependencies (AAP §0.7.1).
    A pre-built ``redis_client`` can be injected for testing with *fakeredis*.

    Args:
        config: Optional :class:`EventStoreConfig`.  Defaults are used when
            ``None`` is provided.
        redis_client: Optional pre-configured async Redis client.  When
            supplied, :meth:`connect` will skip client creation and use it
            directly.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        config: Optional[EventStoreConfig] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self._config: EventStoreConfig = config or EventStoreConfig()
        self._redis: Optional[Any] = redis_client
        self._connected: bool = redis_client is not None
        self._store_id: UUID = uuid4()  # Unique identifier for this store instance
        self._event_count: int = 0
        self._events_saved: int = 0
        self._events_deleted: int = 0
        self._events_queried: int = 0
        self._last_cleanup: Optional[datetime] = None
        self._event_ttl: timedelta = timedelta(seconds=self._config.event_ttl_seconds)

        logger.info(
            "event_store_initialized",
            key_prefix=self._config.key_prefix,
            event_ttl_seconds=self._config.event_ttl_seconds,
            enable_indexing=self._config.enable_indexing,
        )

    # -------------------------------------------------------- connect / close

    async def connect(self) -> None:
        """Establish a Redis connection if not already connected.

        Uses ``redis.asyncio.from_url`` to create an async client and
        verifies connectivity with a PING command.

        Raises:
            ConnectionError: When Redis is unreachable or PING fails.
            RuntimeError: When the ``redis`` package is not installed.
        """
        if self._connected and self._redis is not None:
            return

        if aioredis is None:
            raise RuntimeError(
                "The 'redis' package is required for EventStore but is not "
                "installed.  Install it with: pip install redis>=7.0.0"
            )

        try:
            self._redis = aioredis.from_url(
                self._config.redis_url,
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True
            logger.info(
                "event_store_connected",
                redis_url=_sanitize_redis_url(self._config.redis_url),
            )
        except Exception as exc:
            self._connected = False
            logger.error(
                "event_store_connection_failed",
                redis_url=_sanitize_redis_url(self._config.redis_url),
                error=str(exc),
            )
            raise

    async def disconnect(self) -> None:
        """Close the Redis connection and release resources."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                # Best-effort close — tolerate errors during teardown.
                pass
        self._redis = None
        self._connected = False
        logger.info("event_store_disconnected")

    # --------------------------------------------------- context manager

    async def __aenter__(self) -> "EventStore":
        """Enter the async context manager — connect to Redis."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the async context manager — disconnect from Redis."""
        await self.disconnect()

    # ---------------------------------------------------------------- helpers

    def _ensure_connected(self) -> None:
        """Raise if the store is not connected to Redis."""
        if not self._connected or self._redis is None:
            raise ConnectionError(
                "EventStore is not connected to Redis.  Call connect() or "
                "use the async context manager first."
            )

    @staticmethod
    def _to_str(value: Union[str, UUID]) -> str:
        """Coerce a UUID or string to a plain string."""
        return str(value)

    def _primary_key(self, event_id: Union[str, UUID]) -> str:
        """Build the primary Redis key for an event: ``event:{uuid}``."""
        return f"{self._config.key_prefix}:{self._to_str(event_id)}"

    def _sim_index_key(self, simulation_id: Union[str, UUID]) -> str:
        """Build the simulation-scoped sorted-set index key."""
        return (
            f"{self._config.index_prefix}:simulation:"
            f"{self._to_str(simulation_id)}"
        )

    def _type_index_key(self, event_type: str) -> str:
        """Build the event-type-scoped sorted-set index key."""
        return f"{self._config.index_prefix}:type:{event_type}"

    def _combined_index_key(
        self,
        simulation_id: Union[str, UUID],
        event_type: str,
    ) -> str:
        """Build the combined simulation+type sorted-set index key."""
        return (
            f"{self._config.index_prefix}:simulation:"
            f"{self._to_str(simulation_id)}:type:{event_type}"
        )

    @staticmethod
    def _ts_score(ts: datetime) -> float:
        """Convert a datetime to a float score for sorted-set indexing.

        Uses the POSIX timestamp (seconds since epoch) as the score value.
        """
        return ts.timestamp()

    # -------------------------------------------------------- serialization

    def _serialize_event(self, event: Event) -> str:
        """Serialize an :class:`Event` to a JSON string for Redis storage.

        UUID fields are converted to strings, datetimes to ISO-8601, and
        the resulting dictionary is compact-encoded via ``json.dumps``.

        Args:
            event: The event instance to serialize.

        Returns:
            A compact JSON string suitable for ``SET`` in Redis.
        """
        data = event.to_dict()
        return json.dumps(data, separators=(",", ":"), default=str)

    def _deserialize_event(self, data: str) -> Event:
        """Deserialize a JSON string back into the correct Event subclass.

        The ``event_type`` discriminator in the parsed dictionary is used to
        look up the concrete class in :data:`_EVENT_TYPE_REGISTRY`.  If the
        type is unrecognised, a base :class:`Event` is returned.

        Args:
            data: A JSON string previously produced by :meth:`_serialize_event`.

        Returns:
            An :class:`Event` (or subclass) instance.

        Raises:
            json.JSONDecodeError: If *data* is not valid JSON.
            ValueError: If required fields are missing or malformed.
        """
        raw: Dict[str, Any] = json.loads(data)
        return Event.from_dict(raw)

    # --------------------------------------------------- persistence (CRUD)

    async def save_event(self, event: Event) -> str:
        """Persist an event to Redis with JSONB payload and 7-day TTL.

        The event is stored at ``event:{event_id}`` and indexed in up to
        three sorted sets (simulation, type, combined) when indexing is
        enabled.

        Args:
            event: The event to persist.

        Returns:
            The string representation of the event's UUID.

        Raises:
            ConnectionError: If the store is not connected.
        """
        self._ensure_connected()

        event_id_str = self._to_str(event.event_id)
        primary_key = self._primary_key(event.event_id)
        serialized = self._serialize_event(event)
        score = self._ts_score(event.timestamp)
        ttl = self._config.event_ttl_seconds

        # Use a pipeline for atomicity and reduced round-trips.
        pipe = self._redis.pipeline(transaction=False)

        # Primary storage with TTL
        pipe.set(primary_key, serialized, ex=ttl)

        # Secondary indexes (sorted sets keyed by timestamp score)
        if self._config.enable_indexing:
            sim_id_str = self._to_str(event.simulation_id)

            sim_key = self._sim_index_key(event.simulation_id)
            pipe.zadd(sim_key, {event_id_str: score})
            pipe.expire(sim_key, ttl)

            type_key = self._type_index_key(event.event_type)
            pipe.zadd(type_key, {event_id_str: score})
            pipe.expire(type_key, ttl)

            combined_key = self._combined_index_key(
                event.simulation_id, event.event_type
            )
            pipe.zadd(combined_key, {event_id_str: score})
            pipe.expire(combined_key, ttl)

        await pipe.execute()

        self._event_count += 1
        self._events_saved += 1

        logger.debug(
            "event_saved",
            event_id=event_id_str,
            event_type=event.event_type,
            simulation_id=self._to_str(event.simulation_id),
        )

        return event_id_str

    async def get_event(self, event_id: Union[str, UUID]) -> Optional[Event]:
        """Retrieve a single event by its UUID.

        Args:
            event_id: The event's unique identifier (string or UUID).

        Returns:
            The deserialized :class:`Event`, or ``None`` if the key does not
            exist (or has expired).
        """
        self._ensure_connected()

        primary_key = self._primary_key(event_id)
        raw: Optional[str] = await self._redis.get(primary_key)

        if raw is None:
            logger.debug(
                "event_not_found",
                event_id=self._to_str(event_id),
            )
            return None

        event = self._deserialize_event(raw)
        logger.debug(
            "event_retrieved",
            event_id=self._to_str(event_id),
            event_type=event.event_type,
        )
        return event

    async def get_events(
        self,
        simulation_id: Union[str, UUID],
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Event]:
        """Query events with filters on simulation, type, and time range.

        Uses sorted-set secondary indexes for efficient range retrieval via
        ``ZRANGEBYSCORE``.

        Args:
            simulation_id: Restrict to events belonging to this simulation.
            event_type: Optional event-type discriminator filter.
            start_date: Inclusive lower bound on event timestamp.
            end_date: Inclusive upper bound on event timestamp.
            limit: Maximum number of events to return.  Capped by
                :attr:`EventStoreConfig.max_events_per_query`.

        Returns:
            A list of :class:`Event` instances sorted by timestamp ascending.
        """
        self._ensure_connected()

        # Determine the appropriate index key
        if event_type is not None:
            index_key = self._combined_index_key(simulation_id, event_type)
        else:
            index_key = self._sim_index_key(simulation_id)

        # Score bounds
        min_score: Union[float, str] = "-inf"
        max_score: Union[float, str] = "+inf"
        if start_date is not None:
            min_score = self._ts_score(start_date)
        if end_date is not None:
            max_score = self._ts_score(end_date)

        # Effective limit
        effective_limit = self._config.max_events_per_query
        if limit is not None:
            effective_limit = min(limit, self._config.max_events_per_query)

        # Retrieve event IDs from the sorted-set index
        event_ids: List[str] = await self._redis.zrangebyscore(
            index_key,
            min=min_score,
            max=max_score,
            start=0,
            num=effective_limit,
        )

        if not event_ids:
            logger.debug(
                "events_queried",
                simulation_id=self._to_str(simulation_id),
                event_type=event_type,
                count=0,
            )
            return []

        # Batch-fetch full event payloads using a pipeline
        primary_keys = [self._primary_key(eid) for eid in event_ids]
        pipe = self._redis.pipeline(transaction=False)
        for pk in primary_keys:
            pipe.get(pk)
        raw_values: List[Optional[str]] = await pipe.execute()

        # Deserialize, skipping expired/missing entries
        events: List[Event] = []
        for raw in raw_values:
            if raw is not None:
                try:
                    events.append(self._deserialize_event(raw))
                except (json.JSONDecodeError, ValueError, KeyError) as exc:
                    logger.warning(
                        "event_deserialization_failed",
                        error=str(exc),
                    )

        # Ensure chronological order (index order should already be correct,
        # but an explicit sort guarantees correctness after filtering).
        events.sort(key=lambda e: e.timestamp)

        self._events_queried += 1

        logger.debug(
            "events_queried",
            simulation_id=self._to_str(simulation_id),
            event_type=event_type,
            count=len(events),
        )

        return events

    # --------------------------------------------------------- event replay

    async def replay_events(
        self,
        simulation_id: Union[str, UUID],
        target_date: Optional[datetime] = None,
    ) -> List[Event]:
        """Replay events for state reconstruction (event sourcing).

        Returns all events for *simulation_id* up to *target_date*, sorted
        chronologically.  The caller is responsible for applying the events
        to rebuild the desired state.

        Args:
            simulation_id: The simulation whose events should be replayed.
            target_date: Optional upper bound on event timestamps.  When
                ``None``, all events for the simulation are returned.

        Returns:
            A chronologically sorted list of :class:`Event` instances.
        """
        events = await self.get_events(
            simulation_id=simulation_id,
            end_date=target_date,
        )

        logger.info(
            "events_replayed",
            simulation_id=self._to_str(simulation_id),
            target_date=target_date.isoformat() if target_date else None,
            event_count=len(events),
        )

        return events

    # ----------------------------------------------------------- delete

    async def delete_event(self, event_id: Union[str, UUID]) -> bool:
        """Delete an event and remove it from all secondary indexes.

        Args:
            event_id: The event to delete.

        Returns:
            ``True`` if the event existed and was deleted, ``False``
            otherwise.
        """
        self._ensure_connected()

        primary_key = self._primary_key(event_id)
        event_id_str = self._to_str(event_id)

        # Fetch event data first so we can clean up indexes.
        raw: Optional[str] = await self._redis.get(primary_key)
        if raw is None:
            return False

        # Parse just enough to determine index keys.
        try:
            data: Dict[str, Any] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Corrupt payload — just delete the primary key.
            await self._redis.delete(primary_key)
            self._events_deleted += 1
            return True

        pipe = self._redis.pipeline(transaction=False)
        pipe.delete(primary_key)

        if self._config.enable_indexing:
            sim_id = data.get("simulation_id", "")
            evt_type = data.get("event_type", "")

            if sim_id:
                pipe.zrem(self._sim_index_key(sim_id), event_id_str)
            if evt_type:
                pipe.zrem(self._type_index_key(evt_type), event_id_str)
            if sim_id and evt_type:
                pipe.zrem(
                    self._combined_index_key(sim_id, evt_type),
                    event_id_str,
                )

        await pipe.execute()
        self._events_deleted += 1
        self._event_count = max(0, self._event_count - 1)

        logger.debug(
            "event_deleted",
            event_id=event_id_str,
        )

        return True

    # ----------------------------------------------------------- cleanup

    async def cleanup_expired_events(self) -> int:
        """Remove orphaned index entries whose primary keys have expired.

        Scans each known sorted-set index key, checks for members whose
        primary event key no longer exists in Redis, and removes the stale
        entries.  This is the recommended periodic maintenance task (see
        ``cleanup_interval_seconds`` in :class:`EventStoreConfig`).

        Returns:
            The number of orphaned index entries removed.
        """
        self._ensure_connected()
        cleaned = 0

        # Scan for all index keys matching our prefix pattern.
        index_pattern = f"{self._config.index_prefix}:*"
        cursor: Union[int, str] = 0
        index_keys: List[str] = []

        while True:
            cursor, keys = await self._redis.scan(
                cursor=cursor,
                match=index_pattern,
                count=200,
            )
            index_keys.extend(keys)
            if cursor == 0:
                break

        for idx_key in index_keys:
            # Fetch all members of this sorted set
            members: List[str] = await self._redis.zrange(idx_key, 0, -1)
            if not members:
                continue

            # Check which primary keys still exist
            primary_keys = [self._primary_key(m) for m in members]
            pipe = self._redis.pipeline(transaction=False)
            for pk in primary_keys:
                pipe.exists(pk)
            existence: List[int] = await pipe.execute()

            # Remove members whose primary keys have expired
            stale_members = [
                m for m, exists in zip(members, existence) if not exists
            ]
            if stale_members:
                await self._redis.zrem(idx_key, *stale_members)
                cleaned += len(stale_members)

        self._last_cleanup = datetime.now(timezone.utc)

        logger.info(
            "event_store_cleanup",
            cleaned_count=cleaned,
            indexes_scanned=len(index_keys),
        )

        return cleaned

    # --------------------------------------------------------- count / metrics

    async def get_event_count(
        self,
        simulation_id: Optional[Union[str, UUID]] = None,
    ) -> int:
        """Return the number of stored events.

        Args:
            simulation_id: If provided, count only events in the given
                simulation's index.  Otherwise return the running total.

        Returns:
            An integer event count.
        """
        if simulation_id is not None:
            self._ensure_connected()
            sim_key = self._sim_index_key(simulation_id)
            count: int = await self._redis.zcard(sim_key)
            return count

        return self._event_count

    async def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for monitoring and observability.

        The ``redis_url`` value is sanitized to strip any embedded
        credentials before inclusion in the metrics dictionary.

        Returns:
            A dictionary with event store health and performance data.
        """
        return {
            "store_id": str(self._store_id),
            "total_events": self._event_count,
            "events_saved": self._events_saved,
            "events_deleted": self._events_deleted,
            "events_queried": self._events_queried,
            "connected": self._connected,
            "redis_url": _sanitize_redis_url(self._config.redis_url),
            "event_ttl_seconds": self._config.event_ttl_seconds,
            "event_ttl_human": str(self._event_ttl),
            "enable_indexing": self._config.enable_indexing,
            "last_cleanup": (
                self._last_cleanup.isoformat() if self._last_cleanup else None
            ),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EventStore",
    "EventStoreConfig",
]
