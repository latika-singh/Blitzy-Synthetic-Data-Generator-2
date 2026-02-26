"""Dual-mode Event Bus with asynchronous pub/sub for cross-subsystem coordination.

The ``EventBus`` is the **sole mechanism** for cross-subsystem asynchronous
notifications within the Agent & Orchestration Engine (F-007).  Two operating
modes are supported:

- **In-Memory** (default): Uses :class:`asyncio.PriorityQueue` for workloads
  under 500 events/second.  Zero external dependencies beyond the Python
  standard library.
- **Redis Pub/Sub** (optional): Scales to >5,000 events/second by distributing
  events across processes via Redis channels with the naming convention
  ``events:{event_type}``.

Key design properties:

- **Priority dispatch**: Events carry an :class:`EventPriority` (URGENT, HIGH,
  NORMAL, LOW).  URGENT events are dequeued first thanks to
  :class:`asyncio.PriorityQueue` ordering on ``(priority_value, seq, event)``.
- **Publish timeout**: 1 second hard limit (README.md §1776).
- **Handler timeout**: 5 seconds per handler invocation (README.md §1777).
  A timed-out handler is skipped — other handlers still execute.
- **Persistence**: When an :class:`EventStore` is injected via constructor,
  every published event is persisted as a fire-and-forget background task.
- **Wildcard subscribers**: Handlers subscribed to ``"*"`` receive every event.
- **Async context manager**: ``async with EventBus() as bus: …``

Project 3 Compatibility:
    No code modifications are required for P3 event support.  The EventBus
    is type-agnostic — ``publish()`` accepts any :class:`Event` instance and
    ``subscribe()`` accepts string event type discriminators, :class:`EventType`
    enum values, or :class:`Event` subclasses.  The 4 new P3 event types
    (``GLEntryPosted``, ``ReworkStarted``, ``ReworkCompleted``,
    ``BalanceUpdated``) defined in ``app.events.event_types`` are handled
    natively by the existing dispatch, subscription, and persistence logic.
    P3 transaction generators, the GL posting engine, and the rework loop
    engine consume the EventBus via constructor injection (ADR-003) and
    publish P3 events through the standard ``publish()`` interface.

References:
    - README.md lines 592–624: EventBus specification
    - AAP §0.4.1: Redis Pub/Sub channel pattern ``events:{event_type}``
    - AAP §0.4.3: EventStore injected via constructor
    - AAP §0.7.1: Constructor injection pattern; EventBus is the sole async
      notification mechanism
    - AAP §0.7.6: All logging via structlog to stdout
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from app.events.event_types import Event, EventType

if TYPE_CHECKING:
    from app.events.event_store import EventStore

# Optional Redis import — degrade gracefully when the package is absent so the
# module can still be loaded for in-memory mode, type-checking, and testing.
try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventPriority(IntEnum):
    """Priority levels for event dispatch ordering.

    Lower numeric values are dequeued first by :class:`asyncio.PriorityQueue`,
    so ``URGENT`` (0) is dispatched before ``LOW`` (3).
    """

    URGENT = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class EventBusMode(str, Enum):
    """Operating mode for the :class:`EventBus`.

    - ``IN_MEMORY``: Pure asyncio.PriorityQueue for <500 events/sec.
    - ``REDIS``: Redis Pub/Sub for >5,000 events/sec.
    """

    IN_MEMORY = "in_memory"
    REDIS = "redis"


# ---------------------------------------------------------------------------
# Configuration — Pydantic V2 model at the subsystem boundary
# ---------------------------------------------------------------------------


class EventBusConfig(BaseModel):
    """Validated configuration for :class:`EventBus`.

    All fields carry sensible defaults so that an ``EventBus`` can be
    instantiated with zero arguments for development and unit testing.

    Attributes:
        mode: Operating mode — in-memory (default) or Redis Pub/Sub.
        max_queue_size: Maximum depth of the internal priority queue.
        publish_timeout: Hard timeout in seconds for :meth:`EventBus.publish`
            (per README.md §1776 — 1 second).
        handler_timeout: Hard timeout in seconds for each handler invocation
            (per README.md §1777 — 5 seconds).
        redis_url: Redis connection URL; only required when *mode* is ``REDIS``.
        redis_channel_prefix: Prefix for Redis Pub/Sub channel names.
            Channels are named ``{prefix}:{event_type}``.
        persist_events: Whether to persist published events via an injected
            :class:`EventStore` instance.
        max_handlers_per_event: Maximum number of handler callbacks allowed
            per event type.  Guards against unbounded handler accumulation.
        enable_priority: Whether priority ordering is active.  When ``False``,
            all events are treated as ``NORMAL``.
    """

    mode: EventBusMode = Field(
        default=EventBusMode.IN_MEMORY,
        description="Operating mode: in_memory or redis.",
    )
    max_queue_size: int = Field(
        default=10_000,
        ge=1,
        description="Maximum internal priority queue depth.",
    )
    publish_timeout: float = Field(
        default=1.0,
        gt=0,
        description="Publish timeout in seconds (spec: 1s).",
    )
    handler_timeout: float = Field(
        default=5.0,
        gt=0,
        description="Per-handler execution timeout in seconds (spec: 5s).",
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for Redis Pub/Sub mode.",
    )
    redis_channel_prefix: str = Field(
        default="events",
        description="Channel prefix for Redis Pub/Sub (events:{event_type}).",
    )
    persist_events: bool = Field(
        default=True,
        description="Persist events via EventStore when available.",
    )
    max_handlers_per_event: int = Field(
        default=50,
        ge=1,
        description="Max handler callbacks per event type.",
    )
    enable_priority: bool = Field(
        default=True,
        description="Enable priority-based dispatch ordering.",
    )


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Dual-mode asynchronous event bus for cross-subsystem coordination.

    Provides publish/subscribe semantics with priority dispatch, optional
    Redis Pub/Sub scaling, and EventStore persistence.

    Usage::

        async with EventBus() as bus:
            await bus.subscribe("TransactionCreated", my_handler)
            await bus.publish(TransactionCreated(payload={...}))

    Constructor injection is used for all external dependencies (AAP §0.7.1).

    Args:
        config: Optional :class:`EventBusConfig`.  Defaults apply when ``None``.
        event_store: Optional :class:`EventStore` for fire-and-forget event
            persistence.
    """

    # Handler callable type alias — supports both sync and async signatures
    HandlerType = Callable[..., Union[None, Awaitable[None]]]

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        config: Optional[EventBusConfig] = None,
        event_store: Optional[EventStore] = None,
    ) -> None:
        self._bus_id: UUID = uuid4()
        self._config: EventBusConfig = config or EventBusConfig()
        self._event_store: Optional[EventStore] = event_store

        # Subscriber registry — maps event_type string → list of callables
        self._subscribers: Dict[str, List[Callable[..., Any]]] = {}

        # Track unique event types that have been published
        self._seen_event_types: Set[str] = set()

        # Internal priority queue for ordered dispatch
        self._queue: asyncio.PriorityQueue[tuple[int, int, Event]] = (
            asyncio.PriorityQueue(maxsize=self._config.max_queue_size)
        )

        # Lifecycle and metrics
        self._running: bool = False
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._redis_listener_task: Optional[asyncio.Task[None]] = None
        self._event_count: int = 0
        self._events_dispatched: int = 0
        self._handlers_succeeded: int = 0
        self._handlers_failed: int = 0
        self._started_at: Optional[datetime] = None

        # Redis client (initialised lazily during start() for REDIS mode)
        self._redis_client: Optional[Any] = None
        self._redis_pubsub: Optional[Any] = None

        logger.info(
            "event_bus_initialized",
            bus_id=str(self._bus_id),
            mode=self._config.mode.value,
            max_queue_size=self._config.max_queue_size,
            publish_timeout=self._config.publish_timeout,
            handler_timeout=self._config.handler_timeout,
            persist_events=self._config.persist_events,
            enable_priority=self._config.enable_priority,
        )

    # --------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Start the background dispatch loop and optional Redis listener.

        Idempotent — calling ``start()`` on a running bus is a no-op.
        """
        if self._running:
            logger.debug("event_bus_already_running")
            return

        self._running = True
        self._started_at = datetime.now(timezone.utc)

        # Launch the in-memory dispatch loop
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="eventbus-dispatch"
        )

        # Redis mode — initialise connection and start remote listener
        if self._config.mode == EventBusMode.REDIS:
            await self._init_redis()

        logger.info("event_bus_started", mode=self._config.mode.value)

    async def stop(self) -> None:
        """Stop the dispatch loop and release resources.

        Idempotent — calling ``stop()`` on an already-stopped bus is a no-op.
        """
        if not self._running:
            return

        self._running = False

        # Gather all active tasks for concurrent cancellation
        tasks_to_cancel: List[asyncio.Task[None]] = []
        if self._dispatch_task is not None and not self._dispatch_task.done():
            self._dispatch_task.cancel()
            tasks_to_cancel.append(self._dispatch_task)
        if (
            self._redis_listener_task is not None
            and not self._redis_listener_task.done()
        ):
            self._redis_listener_task.cancel()
            tasks_to_cancel.append(self._redis_listener_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._dispatch_task = None
        self._redis_listener_task = None

        # Close Redis connections
        if self._redis_pubsub is not None:
            try:
                await self._redis_pubsub.unsubscribe()
                await self._redis_pubsub.aclose()
            except Exception:
                pass
            self._redis_pubsub = None

        if self._redis_client is not None:
            try:
                await self._redis_client.aclose()
            except Exception:
                pass
            self._redis_client = None

        logger.info(
            "event_bus_stopped",
            total_events_processed=self._event_count,
            events_dispatched=self._events_dispatched,
            handlers_succeeded=self._handlers_succeeded,
            handlers_failed=self._handlers_failed,
        )

    # ------------------------------------------------------------- pub / sub

    async def publish(
        self,
        event: Event,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Publish an event for asynchronous dispatch to subscribers.

        The event is placed onto the internal priority queue, optionally
        persisted to the :class:`EventStore`, and optionally forwarded to
        Redis Pub/Sub.

        Raises:
            asyncio.TimeoutError: If the publish operation exceeds the
                configured ``publish_timeout`` (default 1 s).
            asyncio.QueueFull: If the internal queue is at capacity and the
                timeout expires.
        """
        effective_priority = (
            priority if self._config.enable_priority else EventPriority.NORMAL
        )

        try:
            await asyncio.wait_for(
                self._enqueue(event, effective_priority),
                timeout=self._config.publish_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "event_publish_timeout",
                event_id=str(event.event_id),
                event_type=event.event_type,
                priority=effective_priority.name,
                timeout=self._config.publish_timeout,
            )
            # Best-effort: try non-blocking put so we don't lose the event
            try:
                self._queue.put_nowait(
                    (effective_priority.value, self._event_count, event)
                )
                self._event_count += 1
            except asyncio.QueueFull:
                logger.error(
                    "event_publish_queue_full",
                    event_id=str(event.event_id),
                    event_type=event.event_type,
                    queue_size=self._queue.qsize(),
                )
                raise
            return

        # Fire-and-forget persistence
        if (
            self._config.persist_events
            and self._event_store is not None
        ):
            asyncio.create_task(
                self._persist_event(event), name="eventbus-persist"
            )

        # Redis Pub/Sub forwarding
        if (
            self._config.mode == EventBusMode.REDIS
            and self._redis_client is not None
        ):
            asyncio.create_task(
                self._redis_publish(event), name="eventbus-redis-pub"
            )

        logger.debug(
            "event_published",
            event_id=str(event.event_id),
            event_type=event.event_type,
            priority=effective_priority.name,
            queue_size=self._queue.qsize(),
        )

    async def subscribe(
        self,
        event_type: Union[str, Type[Event]],
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler callback for a specific event type.

        Args:
            event_type: Either a string event-type name (e.g.
                ``"TransactionCreated"``) or a concrete :class:`Event`
                subclass.  Use ``"*"`` for wildcard subscriptions.
            handler: An async callable ``(Event) -> None``.

        Raises:
            ValueError: If the handler limit for this event type is reached.
        """
        event_type_str = self._resolve_event_type(event_type)

        if event_type_str not in self._subscribers:
            self._subscribers[event_type_str] = []

        handlers = self._subscribers[event_type_str]

        if len(handlers) >= self._config.max_handlers_per_event:
            raise ValueError(
                f"Maximum handlers ({self._config.max_handlers_per_event}) "
                f"reached for event type '{event_type_str}'."
            )

        if handler not in handlers:
            handlers.append(handler)

        logger.debug(
            "handler_subscribed",
            event_type=event_type_str,
            handler_name=getattr(handler, "__name__", repr(handler)),
            total_handlers=len(handlers),
        )

    async def unsubscribe(
        self,
        event_type: Union[str, Type[Event]],
        handler: Callable[..., Any],
    ) -> None:
        """Remove a previously registered handler for an event type.

        Silently succeeds if the handler was not registered.

        Args:
            event_type: The event type string or :class:`Event` subclass.
            handler: The handler callback to remove.
        """
        event_type_str = self._resolve_event_type(event_type)
        handlers = self._subscribers.get(event_type_str, [])

        if handler in handlers:
            handlers.remove(handler)
            logger.debug(
                "handler_unsubscribed",
                event_type=event_type_str,
                handler_name=getattr(handler, "__name__", repr(handler)),
                remaining_handlers=len(handlers),
            )

    # --------------------------------------------------------- dispatch loop

    async def _dispatch_loop(self) -> None:
        """Background loop that dequeues events and dispatches to handlers.

        Runs continuously while ``_running`` is ``True``.  Uses a short
        timeout on ``queue.get`` to allow the loop to check the running flag
        periodically rather than blocking indefinitely.
        """
        logger.debug("dispatch_loop_started")
        try:
            while self._running:
                try:
                    priority_val, seq, event = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # No event available — loop back to check _running flag
                    continue

                try:
                    await self._dispatch_to_handlers(event)
                    self._events_dispatched += 1
                except Exception as exc:
                    logger.error(
                        "dispatch_error",
                        event_id=str(event.event_id),
                        event_type=event.event_type,
                        error=str(exc),
                    )
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            logger.debug("dispatch_loop_cancelled")
            raise
        except Exception as exc:
            logger.error("dispatch_loop_fatal", error=str(exc))

    async def _dispatch_to_handlers(self, event: Event) -> None:
        """Invoke all registered handlers for a single event.

        Handlers subscribed to the event's specific type **and** wildcard
        (``"*"``) handlers are both called.  Each handler is wrapped in a
        ``wait_for`` with the configured ``handler_timeout`` (default 5 s).
        A failing or timed-out handler does **not** prevent other handlers
        from executing.
        """
        event_type_str: str = event.event_type

        # Collect specific + wildcard handlers
        specific_handlers = list(self._subscribers.get(event_type_str, []))
        wildcard_handlers = list(self._subscribers.get("*", []))
        all_handlers = specific_handlers + wildcard_handlers

        if not all_handlers:
            return

        handlers_called = len(all_handlers)
        succeeded = 0
        failed = 0

        for handler in all_handlers:
            handler_name = getattr(handler, "__name__", repr(handler))
            try:
                result = handler(event)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await asyncio.wait_for(
                        result, timeout=self._config.handler_timeout
                    )
                succeeded += 1
            except asyncio.TimeoutError:
                failed += 1
                logger.error(
                    "handler_timeout",
                    event_id=str(event.event_id),
                    event_type=event_type_str,
                    handler_name=handler_name,
                    timeout=self._config.handler_timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failed += 1
                logger.error(
                    "handler_error",
                    event_id=str(event.event_id),
                    event_type=event_type_str,
                    handler_name=handler_name,
                    error=str(exc),
                )

        self._handlers_succeeded += succeeded
        self._handlers_failed += failed

        logger.debug(
            "event_dispatched",
            event_id=str(event.event_id),
            event_type=event_type_str,
            handlers_called=handlers_called,
            handlers_succeeded=succeeded,
            handlers_failed=failed,
        )

    # -------------------------------------------------------- internal helpers

    async def _enqueue(self, event: Event, priority: EventPriority) -> None:
        """Put an event onto the priority queue with a sequence number."""
        seq = self._event_count
        await self._queue.put((priority.value, seq, event))
        self._event_count += 1
        self._seen_event_types.add(event.event_type)

    async def _persist_event(self, event: Event) -> None:
        """Fire-and-forget persistence to the injected :class:`EventStore`."""
        if self._event_store is None:
            return
        try:
            await self._event_store.save_event(event)
        except Exception as exc:
            logger.error(
                "event_persist_error",
                event_id=str(event.event_id),
                event_type=event.event_type,
                error=str(exc),
            )

    @staticmethod
    def _resolve_event_type(event_type: Union[str, Type[Event]]) -> str:
        """Resolve an event-type argument to its canonical string form.

        Accepts either a plain string (returned as-is) or a concrete
        :class:`Event` subclass (resolved to its class name or
        ``event_type`` default field value).  Also supports :class:`EventType`
        enum members which are resolved via their ``.value`` attribute.
        """
        if isinstance(event_type, str):
            # Handle EventType enum members (which are str subclasses)
            if isinstance(event_type, EventType):
                return event_type.value
            return event_type

        # It's a class — inspect for a default event_type field value
        if hasattr(event_type, "__dataclass_fields__"):
            fields = event_type.__dataclass_fields__
            et_field = fields.get("event_type")
            if et_field is not None and et_field.default and et_field.default != "":
                return str(et_field.default)

        return event_type.__name__

    # ------------------------------------------------ Redis Pub/Sub (optional)

    async def _init_redis(self) -> None:
        """Initialise the async Redis client and start the listener task.

        Only called when ``config.mode`` is :attr:`EventBusMode.REDIS`.

        Raises:
            RuntimeError: If the ``redis`` package is not installed.
            ConnectionError: If the Redis server is unreachable.
        """
        if aioredis is None:
            raise RuntimeError(
                "The 'redis' package is required for Redis Pub/Sub mode. "
                "Install it with: pip install redis>=7.0.0"
            )

        redis_url = self._config.redis_url or "redis://localhost:6379"
        try:
            self._redis_client = aioredis.from_url(
                redis_url, decode_responses=True
            )
            await self._redis_client.ping()
            logger.info("event_bus_redis_connected")
        except Exception as exc:
            logger.error(
                "event_bus_redis_connection_failed", error=str(exc)
            )
            raise ConnectionError(
                f"Failed to connect to Redis for EventBus: {exc}"
            ) from exc

        # Create Pub/Sub subscriber and start background listener
        self._redis_pubsub = self._redis_client.pubsub()
        # Subscribe to a pattern channel so we receive all event types
        pattern = f"{self._config.redis_channel_prefix}:*"
        await self._redis_pubsub.psubscribe(pattern)
        self._redis_listener_task = asyncio.create_task(
            self._redis_listener(), name="eventbus-redis-listener"
        )
        logger.info(
            "event_bus_redis_subscribed",
            channel_pattern=pattern,
        )

    async def _redis_publish(self, event: Event) -> None:
        """Publish an event to the Redis Pub/Sub channel.

        Channel name follows the AAP §0.4.1 pattern:
        ``{redis_channel_prefix}:{event_type}``.
        """
        if self._redis_client is None:
            return

        channel = (
            f"{self._config.redis_channel_prefix}:{event.event_type}"
        )
        try:
            serialized = json.dumps(event.to_dict(), default=str)
            await self._redis_client.publish(channel, serialized)
            logger.debug(
                "event_redis_published",
                event_id=str(event.event_id),
                channel=channel,
            )
        except Exception as exc:
            logger.error(
                "event_redis_publish_error",
                event_id=str(event.event_id),
                channel=channel,
                error=str(exc),
            )

    async def _redis_listener(self) -> None:
        """Background task that listens for events from Redis Pub/Sub.

        Received messages are deserialised and dispatched through the local
        handler mechanism, enabling distributed event handling.
        """
        if self._redis_pubsub is None:
            return

        logger.debug("redis_listener_started")
        try:
            while self._running:
                try:
                    message = await asyncio.wait_for(
                        self._redis_pubsub.get_message(
                            ignore_subscribe_messages=True, timeout=0.1
                        ),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                if message.get("type") in ("pmessage", "message"):
                    raw_data = message.get("data")
                    if isinstance(raw_data, (str, bytes)):
                        try:
                            data_str = (
                                raw_data
                                if isinstance(raw_data, str)
                                else raw_data.decode("utf-8")
                            )
                            event_dict = json.loads(data_str)
                            event = Event.from_dict(event_dict)
                            await self._dispatch_to_handlers(event)
                        except Exception as exc:
                            logger.error(
                                "redis_listener_deserialize_error",
                                error=str(exc),
                            )
        except asyncio.CancelledError:
            logger.debug("redis_listener_cancelled")
            raise
        except Exception as exc:
            logger.error("redis_listener_fatal", error=str(exc))

    # --------------------------------------------------------- query / metrics

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Return the number of registered handler callbacks.

        Args:
            event_type: If provided, return the count for that specific event
                type.  If ``None``, return the total across all event types.
        """
        if event_type is not None:
            return len(self._subscribers.get(event_type, []))
        return sum(len(h) for h in self._subscribers.values())

    def get_event_count(self) -> int:
        """Return the total number of events published since start."""
        return self._event_count

    def get_queue_size(self) -> int:
        """Return the current number of events waiting in the priority queue."""
        return self._queue.qsize()

    def is_running(self) -> bool:
        """Return ``True`` if the event bus dispatch loop is active."""
        return self._running

    async def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of operational metrics.

        Returns:
            Dictionary containing total_events, queue_size, subscriber_counts,
            mode, is_running, events_dispatched, handlers_succeeded/failed,
            bus_id, and seen_event_types.
        """
        subscriber_counts: Dict[str, int] = {
            et: len(handlers)
            for et, handlers in self._subscribers.items()
        }
        return {
            "bus_id": str(self._bus_id),
            "total_events": self._event_count,
            "queue_size": self._queue.qsize(),
            "subscriber_counts": subscriber_counts,
            "total_subscribers": self.get_subscriber_count(),
            "seen_event_types": sorted(self._seen_event_types),
            "mode": self._config.mode.value,
            "is_running": self._running,
            "events_dispatched": self._events_dispatched,
            "handlers_succeeded": self._handlers_succeeded,
            "handlers_failed": self._handlers_failed,
            "started_at": (
                self._started_at.isoformat() if self._started_at else None
            ),
            "persist_events": self._config.persist_events,
            "enable_priority": self._config.enable_priority,
        }

    # ---------------------------------------------------- context manager

    async def __aenter__(self) -> EventBus:
        """Enter the async context manager — start the dispatch loop."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the async context manager — stop the dispatch loop."""
        await self.stop()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EventBus",
    "EventBusConfig",
    "EventBusMode",
    "EventPriority",
]
