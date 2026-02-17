"""Default event handler registrations for cross-subsystem coordination.

Provides factory functions to register standard handlers that enable loose
coupling between the Agent, Orchestration, Simulation, and External World
subsystems.  Each of the 8 event types defined in :mod:`app.events.event_types`
has a corresponding async handler that logs structured information and can be
extended by downstream subsystems through the :class:`EventHandlerRegistry`.

Key design properties:

- **Async-first**: Every handler is ``async def`` to satisfy the
  :class:`~app.events.event_bus.EventBus` dispatch contract.
- **Structured logging**: All handlers emit structured log entries via
  *structlog* at INFO level (WARNING for discrepancies), complying with
  AAP §0.7.6 logging standards.
- **5-second handler timeout**: Handlers are designed to complete well
  within the 5-second per-handler execution budget (README.md §1777).
- **Loose coupling**: This module imports only from ``app.events.event_types``
  (runtime) and ``app.events.event_bus`` (TYPE_CHECKING only), ensuring no
  direct coupling to Agent, Orchestration, or External World subsystems.
- **Constructor injection**: Dependencies (the :class:`EventBus` instance) are
  passed via :func:`register_default_handlers`, never via global state.

References:
    - AAP §0.5.1 Group 2: Event handler registrations
    - AAP §0.7.1: EventBus is the sole async notification mechanism
    - AAP §0.7.6: Structured logging for every handler invocation
    - README.md §1776–1777: Publish (1 s) and handler (5 s) timeouts
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
)
from uuid import UUID

import structlog

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

if TYPE_CHECKING:
    from app.events.event_bus import EventBus

# ---------------------------------------------------------------------------
# Module logger — structured JSON to stdout per AAP §0.7.6
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Handler Type Alias
# ---------------------------------------------------------------------------

EventHandlerFunc = Callable[[Event], Any]
"""Type alias for event handler callables.

All concrete handlers are ``async def(event: Event) -> None`` but the alias
uses :data:`Any` as the return annotation for maximum compatibility with both
sync and async handler signatures accepted by :class:`EventBus`.
"""

# ---------------------------------------------------------------------------
# Default Handler Implementations (8 handlers, one per event type)
# ---------------------------------------------------------------------------


async def handle_transaction_created(event: Event) -> None:
    """Handle a :class:`TransactionCreated` event.

    Logs the creation of a new transaction with its key identifiers and
    extracted transaction type.  Serves as the entry point for downstream
    workflow-routing coordination.

    Args:
        event: The published :class:`TransactionCreated` event instance.
    """
    transaction_type: Optional[str] = event.payload.get("transaction_type")
    transaction_id: Optional[str] = event.payload.get("transaction_id")
    amount: Optional[Any] = event.payload.get("amount")

    logger.info(
        "transaction_created_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        transaction_type=transaction_type,
        transaction_id=transaction_id,
        amount=amount,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_transaction_completed(event: Event) -> None:
    """Handle a :class:`TransactionCompleted` event.

    Logs the final state of a completed transaction including its terminal
    status and overall duration.

    Args:
        event: The published :class:`TransactionCompleted` event instance.
    """
    final_status: Optional[str] = event.payload.get("final_status")
    transaction_id: Optional[str] = event.payload.get("transaction_id")
    completion_reason: Optional[str] = event.payload.get("completion_reason")
    duration_seconds: Optional[float] = event.payload.get("duration_seconds")

    logger.info(
        "transaction_completed_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        transaction_id=transaction_id,
        final_status=final_status,
        completion_reason=completion_reason,
        duration_seconds=duration_seconds,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_approval_required(event: Event) -> None:
    """Handle an :class:`ApprovalRequired` event.

    Logs the approval request with the monetary amount and required approver
    role.  Coordinates between the workflow orchestrator and the approval
    system by signalling that a higher-authority decision is pending.

    Args:
        event: The published :class:`ApprovalRequired` event instance.
    """
    transaction_id: Optional[str] = event.payload.get("transaction_id")
    required_approver_role: Optional[str] = event.payload.get(
        "required_approver_role"
    )
    threshold_level: Optional[str] = event.payload.get("threshold_level")
    amount: Optional[Any] = event.payload.get("amount")

    logger.info(
        "approval_required_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        transaction_id=transaction_id,
        required_approver_role=required_approver_role,
        threshold_level=threshold_level,
        amount=amount,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_approval_completed(event: Event) -> None:
    """Handle an :class:`ApprovalCompleted` event.

    Logs the approval decision (approved / rejected) along with the
    approver's reasoning.

    Args:
        event: The published :class:`ApprovalCompleted` event instance.
    """
    approval_decision: Optional[str] = event.payload.get("approval_decision")
    transaction_id: Optional[str] = event.payload.get("transaction_id")
    approver_agent_id: Optional[str] = event.payload.get("approver_agent_id")
    reasoning: Optional[str] = event.payload.get("reasoning")

    logger.info(
        "approval_completed_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        transaction_id=transaction_id,
        approval_decision=approval_decision,
        approver_agent_id=approver_agent_id,
        reasoning=reasoning,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_document_generated(event: Event) -> None:
    """Handle a :class:`DocumentGenerated` event.

    Logs the generation of a document artifact including its type and the
    associated transaction identifier.

    Args:
        event: The published :class:`DocumentGenerated` event instance.
    """
    document_type: Optional[str] = event.payload.get("document_type")
    document_id: Optional[str] = event.payload.get("document_id")
    transaction_id: Optional[str] = event.payload.get("transaction_id")

    logger.info(
        "document_generated_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        document_type=document_type,
        document_id=document_id,
        transaction_id=transaction_id,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_period_closing(event: Event) -> None:
    """Handle a :class:`PeriodClosing` event.

    Logs the start of a fiscal period closing process.  Critical for fiscal
    calendar coordination — downstream subscribers may trigger accrual
    calculations, reconciliation tasks, or financial statement preparation.

    Args:
        event: The published :class:`PeriodClosing` event instance.
    """
    period_type: Optional[str] = event.payload.get("period_type")
    period_start: Optional[str] = event.payload.get("period_start")
    period_end: Optional[str] = event.payload.get("period_end")
    fiscal_year: Optional[Any] = event.payload.get("fiscal_year")

    logger.info(
        "period_closing_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        fiscal_year=fiscal_year,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_period_closed(event: Event) -> None:
    """Handle a :class:`PeriodClosed` event.

    Logs the completion of a fiscal period close, marking the period as
    immutable.  No further postings or adjustments are permitted after this
    event is emitted.

    Args:
        event: The published :class:`PeriodClosed` event instance.
    """
    period_type: Optional[str] = event.payload.get("period_type")
    period_start: Optional[str] = event.payload.get("period_start")
    period_end: Optional[str] = event.payload.get("period_end")
    fiscal_year: Optional[Any] = event.payload.get("fiscal_year")
    has_closing_summary: bool = "closing_summary" in event.payload

    logger.info(
        "period_closed_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        fiscal_year=fiscal_year,
        has_closing_summary=has_closing_summary,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


async def handle_discrepancy_detected(event: Event) -> None:
    """Handle a :class:`DiscrepancyDetected` event.

    Logs the detection of a discrepancy at WARNING level since discrepancies
    represent potential data-integrity issues that may require escalation to
    senior agents.  This handler is the entry point for exception-handling
    workflows.

    Args:
        event: The published :class:`DiscrepancyDetected` event instance.
    """
    discrepancy_type: Optional[str] = event.payload.get("discrepancy_type")
    transaction_id: Optional[str] = event.payload.get("transaction_id")
    severity: Optional[str] = event.payload.get("severity")
    details: Optional[str] = event.payload.get("details")
    detected_by_agent_id: Optional[str] = event.payload.get(
        "detected_by_agent_id"
    )

    logger.warning(
        "discrepancy_detected_handled",
        event_id=str(event.event_id),
        simulation_id=str(event.simulation_id),
        event_type=event.event_type,
        discrepancy_type=discrepancy_type,
        transaction_id=transaction_id,
        severity=severity,
        details=details,
        detected_by_agent_id=detected_by_agent_id,
        agent_id=str(event.agent_id) if event.agent_id is not None else None,
        timestamp=event.timestamp.isoformat() if event.timestamp else None,
    )


# ---------------------------------------------------------------------------
# Default Handler Mapping — maps concrete event types to handler functions
# ---------------------------------------------------------------------------

_DEFAULT_HANDLER_PAIRS: List[tuple[type, EventHandlerFunc]] = [
    (TransactionCreated, handle_transaction_created),
    (TransactionCompleted, handle_transaction_completed),
    (ApprovalRequired, handle_approval_required),
    (ApprovalCompleted, handle_approval_completed),
    (DocumentGenerated, handle_document_generated),
    (PeriodClosing, handle_period_closing),
    (PeriodClosed, handle_period_closed),
    (DiscrepancyDetected, handle_discrepancy_detected),
]
"""Internal mapping of event-type classes to their default handler functions.

Used by :func:`register_default_handlers`, :func:`unregister_default_handlers`,
and :func:`get_default_handler_mapping` to keep the wiring DRY.
"""

_DEFAULT_HANDLER_COUNT: int = len(_DEFAULT_HANDLER_PAIRS)
"""Expected number of default handlers — used for logging and assertions."""


# ---------------------------------------------------------------------------
# Registration Factory Functions
# ---------------------------------------------------------------------------


async def register_default_handlers(event_bus: "EventBus") -> None:
    """Register all 8 default handlers with the given :class:`EventBus`.

    Iterates over the canonical handler mapping and calls
    ``event_bus.subscribe(EventTypeClass, handler)`` for each pair, wiring
    the cross-subsystem coordination layer.

    Args:
        event_bus: The :class:`EventBus` instance to subscribe handlers to.
            Injected by the caller (typically :class:`SimulationEngine`).

    Raises:
        ValueError: Propagated from :meth:`EventBus.subscribe` if the
            handler limit for any event type is already reached.
    """
    for event_cls, handler in _DEFAULT_HANDLER_PAIRS:
        await event_bus.subscribe(event_cls, handler)

    logger.info(
        "default_handlers_registered",
        handler_count=_DEFAULT_HANDLER_COUNT,
    )


async def unregister_default_handlers(event_bus: "EventBus") -> None:
    """Unregister all 8 default handlers from the given :class:`EventBus`.

    Safely removes each handler via ``event_bus.unsubscribe()``.  If a
    handler was not previously registered, the call silently succeeds
    (per :meth:`EventBus.unsubscribe` contract).

    Args:
        event_bus: The :class:`EventBus` instance to unsubscribe from.
    """
    for event_cls, handler in _DEFAULT_HANDLER_PAIRS:
        await event_bus.unsubscribe(event_cls, handler)

    logger.info(
        "default_handlers_unregistered",
        handler_count=_DEFAULT_HANDLER_COUNT,
    )


def get_default_handler_mapping() -> Dict[str, EventHandlerFunc]:
    """Return a mapping of event-type strings to their default handler functions.

    Useful for introspection, testing, and custom registration workflows
    where callers want to inspect or selectively register handlers.

    Returns:
        A dictionary keyed by event-type string (e.g. ``"TransactionCreated"``)
        with handler function values.

    Example::

        mapping = get_default_handler_mapping()
        assert "TransactionCreated" in mapping
        assert mapping["TransactionCreated"] is handle_transaction_created
    """
    return {
        event_cls.__name__: handler
        for event_cls, handler in _DEFAULT_HANDLER_PAIRS
    }


# ---------------------------------------------------------------------------
# EventHandlerRegistry — utility class for tracking handler registrations
# ---------------------------------------------------------------------------


class EventHandlerRegistry:
    """Centralized registry tracking which handlers are registered for which
    event types.

    This is an **in-process bookkeeping utility** for testing, introspection,
    and custom registration workflows.  It does **not** replace the
    :class:`EventBus` dispatch mechanism — the :class:`EventBus` owns actual
    event delivery.

    The registry maintains a ``Dict[str, List[EventHandlerFunc]]`` structure
    where each key is an event-type string and each value is an ordered list
    of handler functions registered for that type.

    Example::

        registry = EventHandlerRegistry()
        registry.register("TransactionCreated", handle_transaction_created)
        handlers = registry.get_handlers("TransactionCreated")
        assert handle_transaction_created in handlers
    """

    def __init__(self) -> None:
        """Initialise an empty handler registry."""
        self._registry: Dict[str, List[EventHandlerFunc]] = {}

        logger.debug("event_handler_registry_initialized")

    # ----------------------------------------------------------------- mutate

    def register(self, event_type: str, handler: EventHandlerFunc) -> None:
        """Register a handler for the given event type.

        If the handler is already registered for this event type the call is
        a no-op, preventing duplicate registrations.

        Args:
            event_type: Event-type string (e.g. ``"TransactionCreated"``).
            handler: The handler callable to register.
        """
        if event_type not in self._registry:
            self._registry[event_type] = []

        handlers = self._registry[event_type]

        if handler not in handlers:
            handlers.append(handler)
            logger.debug(
                "handler_registered_in_registry",
                event_type=event_type,
                handler_name=getattr(handler, "__name__", repr(handler)),
                total_handlers=len(handlers),
            )

    def unregister(self, event_type: str, handler: EventHandlerFunc) -> None:
        """Remove a handler registration for the given event type.

        Silently succeeds if the handler was not registered.

        Args:
            event_type: Event-type string.
            handler: The handler callable to remove.
        """
        handlers = self._registry.get(event_type, [])

        if handler in handlers:
            handlers.remove(handler)
            logger.debug(
                "handler_unregistered_from_registry",
                event_type=event_type,
                handler_name=getattr(handler, "__name__", repr(handler)),
                remaining_handlers=len(handlers),
            )

        # Clean up empty lists to keep the registry tidy
        if event_type in self._registry and not self._registry[event_type]:
            del self._registry[event_type]

    # ------------------------------------------------------------------ query

    def get_handlers(self, event_type: str) -> List[EventHandlerFunc]:
        """Return the list of handlers registered for *event_type*.

        Returns:
            A (possibly empty) list of handler functions.  The returned list
            is a **copy** to prevent callers from mutating internal state.
        """
        return list(self._registry.get(event_type, []))

    def get_all_registrations(self) -> Dict[str, List[EventHandlerFunc]]:
        """Return the complete registry state.

        Returns:
            A dictionary mapping every registered event-type string to its
            list of handlers.  Both the outer dict and inner lists are
            **copies** to prevent external mutation.
        """
        return {
            event_type: list(handlers)
            for event_type, handlers in self._registry.items()
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EventHandlerFunc",
    "EventHandlerRegistry",
    "get_default_handler_mapping",
    "handle_approval_completed",
    "handle_approval_required",
    "handle_discrepancy_detected",
    "handle_document_generated",
    "handle_period_closed",
    "handle_period_closing",
    "handle_transaction_completed",
    "handle_transaction_created",
    "register_default_handlers",
    "unregister_default_handlers",
]
