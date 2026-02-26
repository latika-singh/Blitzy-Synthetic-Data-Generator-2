"""Event type definitions for the Event System (F-007).

Defines 12 event dataclasses used for cross-subsystem coordination through the
EventBus. Each event carries a UUID identifier, simulation context, typed payload,
and optional agent attribution.

Event Schema (per README.md lines 642-648):
    - event_id: UUID — unique event identifier
    - simulation_id: UUID — which simulation this event belongs to
    - event_type: str — string discriminator for event type
    - payload: JSONB (Dict[str, Any]) — structured event data
    - timestamp: datetime — when the event occurred (UTC)
    - agent_id: Optional[UUID] — which agent generated this event

P2 Event Types (per README.md lines 599-607):
    1. TransactionCreated — new transaction initiated
    2. TransactionCompleted — transaction reached final state
    3. ApprovalRequired — transaction requires higher-authority approval
    4. ApprovalCompleted — approval decision made
    5. DocumentGenerated — document artifact produced
    6. PeriodClosing — fiscal period closing process started
    7. PeriodClosed — fiscal period fully closed and immutable
    8. DiscrepancyDetected — discrepancy found during processing

P3 Event Types (Project 3 — Transaction Workflows & Discrepancies):
    9. GLEntryPosted — journal entry posted to General Ledger
    10. ReworkStarted — rework loop began processing a failed transaction
    11. ReworkCompleted — rework attempt finished (fixed/failed/escalated)
    12. BalanceUpdated — GL account balance updated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Type
from uuid import UUID, uuid4


# ---------------------------------------------------------------------------
# EventType Enum — type-safe string constants for all 8 event types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Type-safe string constants for all event types.

    Each enum value matches the corresponding event dataclass name, enabling
    reliable dispatch and registry lookup throughout the event system.

    The original 8 P2 event types are retained; the additional P3 event types
    (GL_ENTRY_POSTED, REWORK_STARTED, REWORK_COMPLETED, BALANCE_UPDATED)
    extend the enum for Project 3 transaction workflows, discrepancy injection,
    and rework loop operations.
    """

    # ---- P2 Event Types (original 8) ----
    TRANSACTION_CREATED = "TransactionCreated"
    TRANSACTION_COMPLETED = "TransactionCompleted"
    APPROVAL_REQUIRED = "ApprovalRequired"
    APPROVAL_COMPLETED = "ApprovalCompleted"
    DOCUMENT_GENERATED = "DocumentGenerated"
    PERIOD_CLOSING = "PeriodClosing"
    PERIOD_CLOSED = "PeriodClosed"
    DISCREPANCY_DETECTED = "DiscrepancyDetected"

    # ---- P3 Event Types (Project 3 — Transaction Workflows & Discrepancies) ----
    GL_ENTRY_POSTED = "GLEntryPosted"
    REWORK_STARTED = "ReworkStarted"
    REWORK_COMPLETED = "ReworkCompleted"
    BALANCE_UPDATED = "BalanceUpdated"


# ---------------------------------------------------------------------------
# Base Event Class
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Base event class for all event types in the Event System.

    All events carry a UUID identifier, simulation context, typed payload, UTC
    timestamp, and optional agent attribution.  Subclasses set their own
    ``event_type`` discriminator via ``field(default=..., init=False)`` and may
    expose convenience properties for common payload keys.

    Attributes:
        event_id: Unique identifier for this event instance.
        simulation_id: The simulation run this event belongs to.
        event_type: String discriminator identifying the concrete event type.
        payload: JSONB-compatible dictionary carrying event-specific data.
        timestamp: UTC datetime indicating when the event occurred.
        agent_id: Optional identifier of the agent that generated this event.
    """

    event_id: UUID = field(default_factory=uuid4)
    simulation_id: UUID = field(default_factory=uuid4)
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: Optional[UUID] = None

    def __post_init__(self) -> None:
        """Set event_type to the class name when not already set by a subclass."""
        if not self.event_type:
            self.event_type = self.__class__.__name__

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this event to a JSON-compatible dictionary.

        UUID fields are converted to strings and the timestamp to ISO-8601
        format so the resulting dictionary can be stored as JSONB in Redis or
        serialized to JSON without additional transformation.

        Returns:
            A dictionary suitable for ``json.dumps`` or JSONB persistence.
        """
        result: Dict[str, Any] = {
            "event_id": str(self.event_id),
            "simulation_id": str(self.simulation_id),
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": str(self.agent_id) if self.agent_id is not None else None,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Event:
        """Deserialize an event from a dictionary.

        The ``event_type`` key in *data* is used to look up the correct
        concrete subclass in :data:`EVENT_TYPE_REGISTRY`.  If the type is not
        recognised, a base :class:`Event` instance is returned with the
        ``event_type`` field set verbatim from the dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict` or loaded
                from a JSONB store.

        Returns:
            An instance of the appropriate :class:`Event` subclass, or of
            :class:`Event` itself for unrecognised types.

        Raises:
            KeyError: If required fields (``event_id``, ``simulation_id``,
                ``timestamp``) are missing from *data*.
            ValueError: If UUID or datetime parsing fails on malformed data.
        """
        # --- Parse common fields -------------------------------------------
        raw_event_id = data.get("event_id")
        event_id: UUID = (
            UUID(raw_event_id) if isinstance(raw_event_id, str) else (raw_event_id if isinstance(raw_event_id, UUID) else uuid4())
        )

        raw_sim_id = data.get("simulation_id")
        simulation_id: UUID = (
            UUID(raw_sim_id) if isinstance(raw_sim_id, str) else (raw_sim_id if isinstance(raw_sim_id, UUID) else uuid4())
        )

        raw_agent_id = data.get("agent_id")
        agent_id: Optional[UUID] = None
        if raw_agent_id is not None:
            agent_id = UUID(raw_agent_id) if isinstance(raw_agent_id, str) else raw_agent_id

        raw_ts = data.get("timestamp")
        if isinstance(raw_ts, str):
            timestamp = datetime.fromisoformat(raw_ts)
        elif isinstance(raw_ts, datetime):
            timestamp = raw_ts
        else:
            timestamp = datetime.now(timezone.utc)

        # Ensure timestamp is UTC-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        payload: Dict[str, Any] = data.get("payload", {})
        event_type_str: str = data.get("event_type", "")

        # --- Resolve concrete subclass -------------------------------------
        event_cls: Type[Event] = EVENT_TYPE_REGISTRY.get(event_type_str, Event)

        if event_cls is Event:
            # Base Event — pass event_type to constructor
            return Event(
                event_id=event_id,
                simulation_id=simulation_id,
                event_type=event_type_str,
                payload=payload,
                timestamp=timestamp,
                agent_id=agent_id,
            )

        # Concrete subclass — event_type is init=False, set via field default
        instance = event_cls(
            event_id=event_id,
            simulation_id=simulation_id,
            payload=payload,
            timestamp=timestamp,
            agent_id=agent_id,
        )
        return instance


# ---------------------------------------------------------------------------
# Concrete Event Dataclasses (8 types)
# ---------------------------------------------------------------------------


@dataclass
class TransactionCreated(Event):
    """Published when a new transaction is initiated in the simulation.

    Expected payload keys:
        transaction_id (str): Unique transaction identifier.
        transaction_type (str): e.g. "purchase_order", "vendor_invoice".
        amount (float): Monetary amount of the transaction.
        currency (str): ISO 4217 currency code (default "USD").
        initiator_agent_id (str): Agent that created the transaction.
    """

    event_type: str = field(default=EventType.TRANSACTION_CREATED.value, init=False)

    @property
    def transaction_id(self) -> Optional[str]:
        """Convenience accessor for ``payload["transaction_id"]``."""
        return self.payload.get("transaction_id")

    @property
    def transaction_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["transaction_type"]``."""
        return self.payload.get("transaction_type")


@dataclass
class TransactionCompleted(Event):
    """Published when a transaction reaches its final state (completed or failed).

    Expected payload keys:
        transaction_id (str): Unique transaction identifier.
        transaction_type (str): e.g. "purchase_order", "vendor_invoice".
        final_status (str): Terminal status — "completed", "failed", "cancelled".
        completion_reason (str): Human-readable explanation.
        duration_seconds (float): Elapsed wall-clock seconds.
    """

    event_type: str = field(default=EventType.TRANSACTION_COMPLETED.value, init=False)

    @property
    def transaction_id(self) -> Optional[str]:
        """Convenience accessor for ``payload["transaction_id"]``."""
        return self.payload.get("transaction_id")

    @property
    def final_status(self) -> Optional[str]:
        """Convenience accessor for ``payload["final_status"]``."""
        return self.payload.get("final_status")


@dataclass
class ApprovalRequired(Event):
    """Published when a transaction requires approval from a higher-authority agent.

    Expected payload keys:
        transaction_id (str): Transaction requiring approval.
        transaction_type (str): Type of transaction.
        amount (float): Monetary amount triggering the threshold.
        required_approver_role (str): Role needed, e.g. "purchasing_manager".
        threshold_level (str): Approval tier, e.g. "level_1", "level_2".
        requestor_agent_id (str): Agent requesting approval.
    """

    event_type: str = field(default=EventType.APPROVAL_REQUIRED.value, init=False)

    @property
    def required_approver_role(self) -> Optional[str]:
        """Convenience accessor for ``payload["required_approver_role"]``."""
        return self.payload.get("required_approver_role")


@dataclass
class ApprovalCompleted(Event):
    """Published when an approval decision has been made (approved or rejected).

    Expected payload keys:
        transaction_id (str): Transaction that was reviewed.
        approval_decision (str): "approved" or "rejected".
        approver_agent_id (str): Agent that made the decision.
        reasoning (str): Explanation for the decision.
        approval_level (str): Approval tier that was satisfied.
    """

    event_type: str = field(default=EventType.APPROVAL_COMPLETED.value, init=False)

    @property
    def approval_decision(self) -> Optional[str]:
        """Convenience accessor for ``payload["approval_decision"]``."""
        return self.payload.get("approval_decision")


@dataclass
class DocumentGenerated(Event):
    """Published when a document artifact is generated as part of a transaction.

    Expected payload keys:
        document_id (str): Unique document identifier.
        document_type (str): e.g. "purchase_order_pdf", "invoice".
        transaction_id (str): Associated transaction.
        generated_by_agent_id (str): Agent that produced the document.
    """

    event_type: str = field(default=EventType.DOCUMENT_GENERATED.value, init=False)

    @property
    def document_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["document_type"]``."""
        return self.payload.get("document_type")


@dataclass
class PeriodClosing(Event):
    """Published when a fiscal period begins its closing process.

    Triggers period-end procedures such as accrual calculations, reconciliation
    tasks, and financial statement preparation.

    Expected payload keys:
        period_type (str): "monthly", "quarterly", or "yearly".
        period_start (str): ISO-8601 date of period start.
        period_end (str): ISO-8601 date of period end.
        fiscal_year (int): Fiscal year number.
    """

    event_type: str = field(default=EventType.PERIOD_CLOSING.value, init=False)

    @property
    def period_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["period_type"]``."""
        return self.payload.get("period_type")


@dataclass
class PeriodClosed(Event):
    """Published when a fiscal period has been fully closed.

    Signals that the period is now immutable — no further postings or
    adjustments are permitted.

    Expected payload keys:
        period_type (str): "monthly", "quarterly", or "yearly".
        period_start (str): ISO-8601 date of period start.
        period_end (str): ISO-8601 date of period end.
        fiscal_year (int): Fiscal year number.
        closing_summary (dict): Summary statistics of the closed period.
    """

    event_type: str = field(default=EventType.PERIOD_CLOSED.value, init=False)

    @property
    def period_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["period_type"]``."""
        return self.payload.get("period_type")


@dataclass
class DiscrepancyDetected(Event):
    """Published when a discrepancy is detected during transaction processing.

    Triggers exception handling workflows and may escalate to senior agents
    depending on severity.

    Expected payload keys:
        discrepancy_type (str): e.g. "price_variance", "quantity_mismatch".
        transaction_id (str): Transaction where discrepancy was found.
        severity (str): "low", "medium", or "high".
        details (str): Human-readable description of the discrepancy.
        detected_by_agent_id (str): Agent that discovered the discrepancy.
    """

    event_type: str = field(default=EventType.DISCREPANCY_DETECTED.value, init=False)

    @property
    def discrepancy_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["discrepancy_type"]``."""
        return self.payload.get("discrepancy_type")

    @property
    def severity(self) -> Optional[str]:
        """Convenience accessor for ``payload["severity"]``."""
        return self.payload.get("severity")


# ---------------------------------------------------------------------------
# P3 Event Dataclasses — Project 3 Transaction Workflows & Discrepancies
# ---------------------------------------------------------------------------


@dataclass
class GLEntryPosted(Event):
    """Published when a journal entry is successfully posted to the General Ledger.

    Enables downstream subscribers (e.g., metrics, audit trail, balance monitors)
    to react to GL posting events in real-time via the EventBus.

    Expected payload keys:
        journal_entry_id (str): Unique identifier for the journal entry.
        transaction_id (str): Originating transaction identifier.
        entry_type (str): e.g. "standard", "accrual", "reversing", "recurring".
        total_debits (str): Total debit amount (Decimal as string).
        total_credits (str): Total credit amount (Decimal as string).
        line_count (int): Number of journal entry lines.
        period_id (str): Fiscal period the entry was posted to.
        posted_by (str): Agent or system component that posted the entry.
    """

    event_type: str = field(default=EventType.GL_ENTRY_POSTED.value, init=False)

    @property
    def journal_entry_id(self) -> Optional[str]:
        """Convenience accessor for ``payload["journal_entry_id"]``."""
        return self.payload.get("journal_entry_id")

    @property
    def entry_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["entry_type"]``."""
        return self.payload.get("entry_type")


@dataclass
class ReworkStarted(Event):
    """Published when the rework loop begins processing a failed transaction.

    Signals that a transaction has entered the classify → fix → re-validate
    cycle managed by the ReworkLoopEngine.

    Expected payload keys:
        transaction_id (str): Transaction entering rework.
        failure_type (str): Classification: "planned_within_bounds",
            "planned_outside_bounds", or "unplanned_error".
        attempt_number (int): Current rework attempt (1-based, max 3).
        failure_reason (str): Description of the validation failure.
        selected_fix_scenario (str): Fix scenario name selected for this attempt.
    """

    event_type: str = field(default=EventType.REWORK_STARTED.value, init=False)

    @property
    def failure_type(self) -> Optional[str]:
        """Convenience accessor for ``payload["failure_type"]``."""
        return self.payload.get("failure_type")

    @property
    def attempt_number(self) -> Optional[int]:
        """Convenience accessor for ``payload["attempt_number"]``."""
        return self.payload.get("attempt_number")


@dataclass
class ReworkCompleted(Event):
    """Published when a rework attempt finishes (success, failure, or escalation).

    Enables metrics tracking of rework success rates and escalation patterns.

    Expected payload keys:
        transaction_id (str): Transaction that was reworked.
        outcome (str): "fixed", "failed", or "escalated".
        attempt_number (int): Which attempt completed.
        total_attempts (int): Total attempts made for this transaction.
        fix_scenario_used (str): Name of the fix scenario that was applied.
        duration_ms (float): Time spent on this rework attempt.
        escalation_reason (str): Reason for escalation (if outcome is "escalated").
    """

    event_type: str = field(default=EventType.REWORK_COMPLETED.value, init=False)

    @property
    def outcome(self) -> Optional[str]:
        """Convenience accessor for ``payload["outcome"]``."""
        return self.payload.get("outcome")

    @property
    def fix_scenario_used(self) -> Optional[str]:
        """Convenience accessor for ``payload["fix_scenario_used"]``."""
        return self.payload.get("fix_scenario_used")


@dataclass
class BalanceUpdated(Event):
    """Published when an account balance is updated in the GL.

    Supports real-time balance monitoring and sub-ledger reconciliation triggers.

    Expected payload keys:
        account_code (str): GL account code that was updated.
        previous_balance (str): Balance before update (Decimal as string).
        new_balance (str): Balance after update (Decimal as string).
        debit_amount (str): Debit applied (Decimal as string).
        credit_amount (str): Credit applied (Decimal as string).
        period_id (str): Fiscal period of the update.
        journal_entry_id (str): Source journal entry identifier.
    """

    event_type: str = field(default=EventType.BALANCE_UPDATED.value, init=False)

    @property
    def account_code(self) -> Optional[str]:
        """Convenience accessor for ``payload["account_code"]``."""
        return self.payload.get("account_code")

    @property
    def new_balance(self) -> Optional[str]:
        """Convenience accessor for ``payload["new_balance"]``."""
        return self.payload.get("new_balance")


# ---------------------------------------------------------------------------
# Event Type Registry — maps event_type strings to concrete classes
# ---------------------------------------------------------------------------

EVENT_TYPE_REGISTRY: Dict[str, Type[Event]] = {
    # P2 event types (original 8)
    EventType.TRANSACTION_CREATED.value: TransactionCreated,
    EventType.TRANSACTION_COMPLETED.value: TransactionCompleted,
    EventType.APPROVAL_REQUIRED.value: ApprovalRequired,
    EventType.APPROVAL_COMPLETED.value: ApprovalCompleted,
    EventType.DOCUMENT_GENERATED.value: DocumentGenerated,
    EventType.PERIOD_CLOSING.value: PeriodClosing,
    EventType.PERIOD_CLOSED.value: PeriodClosed,
    EventType.DISCREPANCY_DETECTED.value: DiscrepancyDetected,
    # P3 event types (Project 3 additions)
    EventType.GL_ENTRY_POSTED.value: GLEntryPosted,
    EventType.REWORK_STARTED.value: ReworkStarted,
    EventType.REWORK_COMPLETED.value: ReworkCompleted,
    EventType.BALANCE_UPDATED.value: BalanceUpdated,
}


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def create_event(event_type: str, **kwargs: Any) -> Event:
    """Create an event instance of the correct subclass from a type string.

    Looks up *event_type* in :data:`EVENT_TYPE_REGISTRY` and instantiates the
    matching concrete class.  If the type is not recognised, a base
    :class:`Event` is returned with ``event_type`` set to the provided string.

    Args:
        event_type: One of the :class:`EventType` enum values (or any string
            for forward-compatible extensibility).
        **kwargs: Keyword arguments forwarded to the event constructor.  Common
            keys include ``simulation_id``, ``payload``, and ``agent_id``.

    Returns:
        An instance of the appropriate :class:`Event` subclass.

    Examples:
        >>> evt = create_event("TransactionCreated", payload={"transaction_id": "TX-001"})
        >>> evt.event_type
        'TransactionCreated'
        >>> isinstance(evt, TransactionCreated)
        True
    """
    event_cls: Type[Event] = EVENT_TYPE_REGISTRY.get(event_type, Event)

    if event_cls is Event:
        # Unrecognised type — fall back to base Event with explicit event_type
        return Event(event_type=event_type, **kwargs)

    # Concrete subclass — event_type is set automatically (init=False)
    return event_cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Event",
    "EventType",
    # P2 event types
    "TransactionCreated",
    "TransactionCompleted",
    "ApprovalRequired",
    "ApprovalCompleted",
    "DocumentGenerated",
    "PeriodClosing",
    "PeriodClosed",
    "DiscrepancyDetected",
    # P3 event types (Project 3 — Transaction Workflows & Discrepancies)
    "GLEntryPosted",
    "ReworkStarted",
    "ReworkCompleted",
    "BalanceUpdated",
    "EVENT_TYPE_REGISTRY",
    "create_event",
]
