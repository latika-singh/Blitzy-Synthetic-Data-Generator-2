"""Event System package (F-007) — Infrastructure layer for event-driven coordination.

Provides the EventBus (dual-mode async pub/sub), EventStore (persistent event
storage), 8 event type definitions, and default handler registrations.  This
package is consumed by nearly every other subsystem in the Agent &
Orchestration Engine.

Primary mode: In-memory asyncio.Queue for <500 events/sec workloads.
Optional mode: Redis Pub/Sub for >5,000 events/sec workloads.

Core components:
    - EventBus: Publish/subscribe event distribution with priority handling
    - EventStore: JSONB persistence with UUID indexing and event replay
    - Event types: 8 defined event dataclasses for cross-subsystem coordination
    - Event handlers: Default handler registrations for standard event processing

Usage example::

    from app.events import EventBus, EventBusConfig, EventPriority
    from app.events import TransactionCreated

    bus = EventBus(EventBusConfig())
    async with bus:
        await bus.publish(
            TransactionCreated(payload={"transaction_id": "TX-001"}),
            priority=EventPriority.NORMAL,
        )

References:
    - AAP §0.5.1 Group 2: EventBus, EventStore, event types, event handlers
    - AAP §0.7.1: EventBus is the sole cross-subsystem async notification
      mechanism — direct method calls between subsystems are only permitted
      for synchronous queries
    - AAP §0.7.3: Package init must be lightweight — no object instantiation
      or I/O at import time
    - README.md lines 592–649: Event system specification
"""

# ---------------------------------------------------------------------------
# Event Types — base class and 8 concrete event dataclasses
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Event Bus — dual-mode async pub/sub with priority dispatch
# ---------------------------------------------------------------------------

from app.events.event_bus import (
    EventBus,
    EventBusConfig,
    EventBusMode,
    EventPriority,
)

# ---------------------------------------------------------------------------
# Event Store — JSONB persistence with UUID indexing and replay
# ---------------------------------------------------------------------------

from app.events.event_store import (
    EventStore,
    EventStoreConfig,
)

# ---------------------------------------------------------------------------
# Event Handlers — default handler registrations and registry
# ---------------------------------------------------------------------------

from app.events.event_handlers import (
    EventHandlerRegistry,
    register_default_handlers,
    unregister_default_handlers,
)

# ---------------------------------------------------------------------------
# Public API — all symbols re-exported for ``from app.events import ...``
# ---------------------------------------------------------------------------

__all__ = [
    "ApprovalCompleted",
    "ApprovalRequired",
    "DiscrepancyDetected",
    "DocumentGenerated",
    "Event",
    "EventBus",
    "EventBusConfig",
    "EventBusMode",
    "EventHandlerRegistry",
    "EventPriority",
    "EventStore",
    "EventStoreConfig",
    "PeriodClosed",
    "PeriodClosing",
    "TransactionCompleted",
    "TransactionCreated",
    "register_default_handlers",
    "unregister_default_handlers",
]
