"""Approval system with threshold-based approval chains for purchase orders,
vendor invoices, and journal entries.

Enforces monetary thresholds (PO: $5K/$25K/$100K, Invoice: $10K/$50K/$100K,
JE: $50K), manages approval request lifecycle, and handles rejection
escalation.

Threshold Design (README.md lines 314-331):
    - purchase_order: <$5K no approval, $5K-$25K purchasing_manager,
      $25K-$100K controller, >$100K cfo
    - vendor_invoice: <$10K no approval, $10K-$50K ap_manager,
      $50K-$100K controller, >$100K cfo
    - journal_entry: <$50K senior_accountant, >$50K controller

Approval Workflow Methods (README.md lines 334-337):
    - get_required_approval(txn_type, amount) -> Optional[str]
    - create_approval_request(transaction, approver_role)
    - process_approval(request, decision, notes)
    - escalate_if_rejected(request)

Integration:
    - Publishes ApprovalRequired events via EventBus when a new approval
      request is created (AAP Section 0.7.1).
    - Publishes ApprovalCompleted events via EventBus when an approval
      decision is processed.
    - Uses constructor injection for AgentRegistry and EventBus dependencies.
    - All cross-module data contracts use Pydantic V2 models.
    - Structured JSON logging via structlog to stdout only (AAP Section 0.7.6).

References:
    - README.md lines 310-338 (ApprovalSystem specification)
    - AAP Section 0.5.1 Group 6 (Orchestration Layer)
    - AAP Section 0.7.1 (constructor injection, EventBus notifications)
    - AAP Section 0.7.3 (performance constraints)
    - config/workflows/approval_thresholds.yaml (runtime overrides)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus

# Module-level structured logger (AAP Section 0.7.6 — stdout only)
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# ApprovalDecision Enum
# ---------------------------------------------------------------------------


class ApprovalDecision(str, Enum):
    """Serialization-friendly decision status for approval requests.

    Uses ``str`` mixin so that enum values serialize naturally to JSON
    strings in Pydantic V2 models and structured log output.

    Members:
        APPROVED: The transaction has been approved by the assigned approver.
        REJECTED: The transaction has been rejected; may be escalated.
        ESCALATED: The request has been forwarded to a higher-authority role.
        PENDING: The request is awaiting a decision.
    """

    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    PENDING = "pending"


# ---------------------------------------------------------------------------
# ApprovalRequest — Pydantic V2 cross-module data contract
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    """Immutable record of a single approval request in the approval chain.

    This Pydantic V2 model is the cross-module data contract for approval
    lifecycle tracking.  It captures the requesting agent, the assigned
    approver role, monetary amount, threshold level description, decision
    outcome, and any escalation lineage.

    Attributes:
        request_id: Unique identifier for this approval request.
        transaction_id: Optional UUID linking to the originating transaction.
        transaction_type: Transaction category (e.g. ``"purchase_order"``).
        amount: Monetary amount that triggered the approval threshold.
        requester_agent_id: UUID of the agent that initiated the request.
        approver_role: Role required to approve (e.g. ``"purchasing_manager"``).
        approver_agent_id: UUID of the agent that made the decision (set on
            decision).
        threshold_level: Human-readable threshold description (e.g.
            ``"$5K-$25K"``).
        decision: Current decision status.
        decision_notes: Free-text reasoning provided by the approver.
        decision_timestamp: UTC timestamp when the decision was recorded.
        created_at: UTC timestamp when the request was created.
        escalated_from: If this request was created via escalation, the UUID
            of the original (rejected) request.
        metadata: Arbitrary key-value pairs for extensibility.
    """

    request_id: UUID = Field(default_factory=uuid4, description="Unique approval request ID")
    transaction_id: Optional[UUID] = Field(default=None, description="Originating transaction UUID")
    transaction_type: str = Field(description="Transaction category, e.g. 'purchase_order'")
    amount: float = Field(description="Monetary amount triggering the threshold")
    requester_agent_id: Optional[UUID] = Field(default=None, description="Agent that initiated the request")
    approver_role: str = Field(description="Role required to approve, e.g. 'purchasing_manager'")
    approver_agent_id: Optional[UUID] = Field(default=None, description="Agent that made the decision")
    threshold_level: str = Field(default="", description="Human-readable threshold, e.g. '$5K-$25K'")
    decision: ApprovalDecision = Field(default=ApprovalDecision.PENDING, description="Current decision status")
    decision_notes: Optional[str] = Field(default=None, description="Reasoning from the approver")
    decision_timestamp: Optional[datetime] = Field(default=None, description="UTC time of decision")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation timestamp",
    )
    escalated_from: Optional[UUID] = Field(
        default=None,
        description="UUID of the rejected request this was escalated from",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensibility key-value pairs")

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Approval Thresholds — EXACT match to README.md lines 314-331
# ---------------------------------------------------------------------------

# Each entry is a tuple of (min_amount, max_amount, approver_role).
# - min_amount is inclusive (amount >= min_amount)
# - max_amount is exclusive (amount < max_amount), except when None (unbounded)
# - approver_role is None when no approval is required for that range
#
# CRITICAL: These threshold values are core business rules and MUST match the
# specification exactly.  Any deviation breaks the approval chain contract.

APPROVAL_THRESHOLDS: Dict[str, List[Tuple[float, Optional[float], Optional[str]]]] = {
    "purchase_order": [
        (0, 5000, None),                        # < $5K: no approval needed
        (5000, 25000, "purchasing_manager"),     # $5K-$25K
        (25000, 100000, "controller"),           # $25K-$100K
        (100000, None, "cfo"),                   # > $100K: CFO required
    ],
    "vendor_invoice": [
        (0, 10000, None),                       # < $10K: no approval needed
        (10000, 50000, "ap_manager"),            # $10K-$50K
        (50000, 100000, "controller"),           # $50K-$100K
        (100000, None, "cfo"),                   # > $100K: CFO required
    ],
    "journal_entry": [
        (0, 50000, "senior_accountant"),         # < $50K
        (50000, None, "controller"),             # > $50K
    ],
}


# ---------------------------------------------------------------------------
# ApprovalSystem — Threshold-based approval chain management
# ---------------------------------------------------------------------------


class ApprovalSystem:
    """Manages approval chains and thresholds for ERP transactions.

    Enforces monetary threshold-based approval chains for purchase orders,
    vendor invoices, and journal entries.  Publishes ``ApprovalRequired`` and
    ``ApprovalCompleted`` events through the injected :class:`EventBus` for
    cross-subsystem coordination.

    Constructor injection is used for all dependencies (AAP Section 0.7.1):
        - ``agent_registry``: For looking up available approvers by role.
        - ``event_bus``: For publishing approval lifecycle events.
        - ``thresholds``: Optional override of default threshold definitions.

    Multi-level approvals are designed to complete within < 10 seconds
    (AAP Success Criterion #12).

    Attributes:
        metrics: Real-time dictionary of approval system performance counters.

    Usage::

        system = ApprovalSystem(event_bus=bus)
        role = system.get_required_approval("purchase_order", 30_000)
        # role == "controller"
        req = await system.create_approval_request(
            {"transaction_type": "purchase_order", "amount": 30_000},
            approver_role="controller",
        )
        req = await system.process_approval(
            req.request_id, ApprovalDecision.APPROVED, notes="Looks good"
        )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        event_bus: Optional[EventBus] = None,
        thresholds: Optional[Dict[str, List[Tuple[float, Optional[float], Optional[str]]]]] = None,
    ) -> None:
        """Initialise the ApprovalSystem with injected dependencies.

        Args:
            agent_registry: Optional registry for role-based agent lookup.
            event_bus: Optional event bus for publishing approval events.
            thresholds: Optional threshold overrides.  Defaults to
                :data:`APPROVAL_THRESHOLDS` when ``None``.
        """
        self._agent_registry = agent_registry
        self._event_bus = event_bus
        self._thresholds: Dict[str, List[Tuple[float, Optional[float], Optional[str]]]] = (
            thresholds if thresholds is not None else APPROVAL_THRESHOLDS
        )

        # Request stores
        self._pending_requests: Dict[UUID, ApprovalRequest] = {}
        self._completed_requests: Dict[UUID, ApprovalRequest] = {}

        # Running counters — exposed via ``self.metrics`` and ``get_metrics()``
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "total_escalated": 0,
            "average_approval_time": 0.0,
        }

        # Internal accumulator for computing running average approval time
        self._total_approval_seconds: float = 0.0
        self._completed_count: int = 0

        logger.info("approval_system_initialized", threshold_types=list(self._thresholds.keys()))

    # ------------------------------------------------------------------
    # Core Methods — README.md lines 334-337
    # ------------------------------------------------------------------

    def get_required_approval(self, txn_type: str, amount: float) -> Optional[str]:
        """Determine the approver role required for a transaction amount.

        Iterates through the threshold ranges defined for *txn_type* and
        returns the role whose range contains *amount*.  Returns ``None``
        when no approval is required (either because the amount falls in
        an unapproved band, or because *txn_type* is not configured).

        Args:
            txn_type: Transaction category key (e.g. ``"purchase_order"``).
            amount: Monetary value to evaluate against thresholds.

        Returns:
            The approver role string (e.g. ``"purchasing_manager"``), or
            ``None`` if no approval is required.
        """
        ranges = self._thresholds.get(txn_type)
        if ranges is None:
            logger.debug(
                "approval_check",
                txn_type=txn_type,
                amount=amount,
                required_role=None,
                reason="unknown_transaction_type",
            )
            return None

        required_role: Optional[str] = None
        for min_amount, max_amount, approver_role in ranges:
            if max_amount is None:
                # Unbounded upper range — matches amount >= min_amount
                if amount >= min_amount:
                    required_role = approver_role
                    break
            else:
                # Bounded range — matches min_amount <= amount < max_amount
                if min_amount <= amount < max_amount:
                    required_role = approver_role
                    break

        logger.debug(
            "approval_check",
            txn_type=txn_type,
            amount=amount,
            required_role=required_role,
        )
        return required_role

    async def create_approval_request(
        self,
        transaction: Dict[str, Any],
        approver_role: str,
        requester_agent_id: Optional[UUID] = None,
    ) -> ApprovalRequest:
        """Create and register a new approval request.

        Builds an :class:`ApprovalRequest` from the transaction dict, stores
        it in the pending queue, increments metrics, and publishes an
        ``ApprovalRequired`` event through the injected :class:`EventBus`.

        Args:
            transaction: Dictionary containing at least ``transaction_type``
                and ``amount``.  May also contain ``transaction_id``.
            approver_role: The role required to approve this transaction.
            requester_agent_id: UUID of the requesting agent (optional).

        Returns:
            The newly created :class:`ApprovalRequest`.
        """
        # Extract transaction details with safe defaults
        txn_type: str = transaction.get("transaction_type", "unknown")
        amount: float = float(transaction.get("amount", 0.0))
        raw_txn_id = transaction.get("transaction_id")
        transaction_id: Optional[UUID] = None
        if raw_txn_id is not None:
            transaction_id = raw_txn_id if isinstance(raw_txn_id, UUID) else UUID(str(raw_txn_id))

        # Determine human-readable threshold level description
        threshold_level = self._compute_threshold_level(txn_type, amount)

        request = ApprovalRequest(
            transaction_id=transaction_id,
            transaction_type=txn_type,
            amount=amount,
            requester_agent_id=requester_agent_id,
            approver_role=approver_role,
            threshold_level=threshold_level,
            metadata={"original_transaction": {k: str(v) for k, v in transaction.items()}},
        )

        # Store in pending queue
        self._pending_requests[request.request_id] = request
        self.metrics["total_requests"] += 1

        # Publish ApprovalRequired event via EventBus (AAP Section 0.7.1)
        if self._event_bus is not None:
            try:
                from app.events.event_types import ApprovalRequired

                event = ApprovalRequired(
                    payload={
                        "transaction_id": str(transaction_id) if transaction_id else None,
                        "transaction_type": txn_type,
                        "amount": amount,
                        "required_approver_role": approver_role,
                        "threshold_level": threshold_level,
                        "requestor_agent_id": str(requester_agent_id) if requester_agent_id else None,
                    }
                )
                await self._event_bus.publish(event)
            except Exception as exc:
                # Log but do not block the request creation on event failure
                logger.warning(
                    "approval_event_publish_failed",
                    event_type="ApprovalRequired",
                    request_id=str(request.request_id),
                    error=str(exc),
                )

        logger.info(
            "approval_request_created",
            request_id=str(request.request_id),
            transaction_type=txn_type,
            amount=amount,
            approver_role=approver_role,
            threshold_level=threshold_level,
        )
        return request

    async def process_approval(
        self,
        request_id: UUID,
        decision: ApprovalDecision,
        notes: Optional[str] = None,
        approver_agent_id: Optional[UUID] = None,
    ) -> ApprovalRequest:
        """Record an approval decision on a pending request.

        Updates the request with the decision, moves it from pending to
        completed, adjusts metrics, and publishes an ``ApprovalCompleted``
        event through the injected :class:`EventBus`.

        Args:
            request_id: UUID of the pending request.
            decision: The approval decision (approved, rejected, escalated).
            notes: Optional reasoning text from the approver.
            approver_agent_id: UUID of the agent making the decision.

        Returns:
            The updated :class:`ApprovalRequest`.

        Raises:
            ValueError: If *request_id* is not found in the pending queue.
        """
        request = self._pending_requests.get(request_id)
        if request is None:
            raise ValueError(
                f"Approval request {request_id} not found in pending requests. "
                f"It may have already been processed or does not exist."
            )

        # Update the request with decision details
        request.decision = decision if isinstance(decision, str) else decision.value
        request.decision_notes = notes
        request.decision_timestamp = datetime.now(timezone.utc)
        request.approver_agent_id = approver_agent_id

        # Move from pending to completed
        del self._pending_requests[request_id]
        self._completed_requests[request_id] = request

        # Update metrics
        decision_value = decision if isinstance(decision, str) else decision.value
        if decision_value == ApprovalDecision.APPROVED.value:
            self.metrics["total_approved"] += 1
        elif decision_value == ApprovalDecision.REJECTED.value:
            self.metrics["total_rejected"] += 1
        elif decision_value == ApprovalDecision.ESCALATED.value:
            self.metrics["total_escalated"] += 1

        # Update average approval time
        if request.created_at is not None:
            elapsed = (request.decision_timestamp - request.created_at).total_seconds()
            self._total_approval_seconds += max(elapsed, 0.0)
            self._completed_count += 1
            if self._completed_count > 0:
                self.metrics["average_approval_time"] = (
                    self._total_approval_seconds / self._completed_count
                )

        # Publish ApprovalCompleted event via EventBus (AAP Section 0.7.1)
        if self._event_bus is not None:
            try:
                from app.events.event_types import ApprovalCompleted

                event = ApprovalCompleted(
                    payload={
                        "transaction_id": str(request.transaction_id) if request.transaction_id else None,
                        "approval_decision": decision_value,
                        "approver_agent_id": str(approver_agent_id) if approver_agent_id else None,
                        "reasoning": notes or "",
                        "approval_level": request.threshold_level,
                    }
                )
                await self._event_bus.publish(event)
            except Exception as exc:
                logger.warning(
                    "approval_event_publish_failed",
                    event_type="ApprovalCompleted",
                    request_id=str(request_id),
                    error=str(exc),
                )

        logger.info(
            "approval_processed",
            request_id=str(request_id),
            decision=decision_value,
            approver_role=request.approver_role,
            transaction_type=request.transaction_type,
            amount=request.amount,
        )
        return request

    async def escalate_if_rejected(self, request_id: UUID) -> Optional[ApprovalRequest]:
        """Escalate a rejected request to the next higher approval authority.

        Looks up the completed request, verifies it was rejected, determines
        the next approver role in the threshold chain, and creates a new
        pending request linked to the original via ``escalated_from``.

        Args:
            request_id: UUID of the completed (rejected) request to escalate.

        Returns:
            A new :class:`ApprovalRequest` assigned to the next-level
            approver, or ``None`` if the request was not rejected or no
            higher approver exists.
        """
        # Look up in completed requests
        original = self._completed_requests.get(request_id)
        if original is None:
            # Also check pending — cannot escalate a pending request
            if request_id in self._pending_requests:
                logger.debug(
                    "escalation_skipped",
                    request_id=str(request_id),
                    reason="request_still_pending",
                )
            else:
                logger.debug(
                    "escalation_skipped",
                    request_id=str(request_id),
                    reason="request_not_found",
                )
            return None

        # Only escalate rejected requests
        original_decision = original.decision
        if isinstance(original_decision, ApprovalDecision):
            original_decision = original_decision.value
        if original_decision != ApprovalDecision.REJECTED.value:
            logger.debug(
                "escalation_skipped",
                request_id=str(request_id),
                reason="not_rejected",
                current_decision=original_decision,
            )
            return None

        # Find the next higher approver in the threshold chain
        next_role = self._find_next_approver(original.transaction_type, original.approver_role)
        if next_role is None:
            logger.info(
                "escalation_impossible",
                request_id=str(request_id),
                reason="no_higher_approver",
                current_role=original.approver_role,
                transaction_type=original.transaction_type,
            )
            return None

        # Determine threshold level for the escalated request
        threshold_level = self._compute_threshold_level(original.transaction_type, original.amount)

        # Create escalated approval request
        escalated = ApprovalRequest(
            transaction_id=original.transaction_id,
            transaction_type=original.transaction_type,
            amount=original.amount,
            requester_agent_id=original.requester_agent_id,
            approver_role=next_role,
            threshold_level=threshold_level,
            escalated_from=request_id,
            metadata={
                "escalation_reason": f"Rejected by {original.approver_role}",
                "original_request_id": str(request_id),
                "original_decision_notes": original.decision_notes or "",
            },
        )

        # Store in pending queue
        self._pending_requests[escalated.request_id] = escalated
        self.metrics["total_requests"] += 1
        self.metrics["total_escalated"] += 1

        # Publish ApprovalRequired event for the escalated request
        if self._event_bus is not None:
            try:
                from app.events.event_types import ApprovalRequired

                event = ApprovalRequired(
                    payload={
                        "transaction_id": str(original.transaction_id) if original.transaction_id else None,
                        "transaction_type": original.transaction_type,
                        "amount": original.amount,
                        "required_approver_role": next_role,
                        "threshold_level": threshold_level,
                        "requestor_agent_id": (
                            str(original.requester_agent_id) if original.requester_agent_id else None
                        ),
                    }
                )
                await self._event_bus.publish(event)
            except Exception as exc:
                logger.warning(
                    "approval_event_publish_failed",
                    event_type="ApprovalRequired",
                    request_id=str(escalated.request_id),
                    error=str(exc),
                )

        logger.info(
            "approval_escalated",
            original_request_id=str(request_id),
            new_request_id=str(escalated.request_id),
            new_approver_role=next_role,
            transaction_type=original.transaction_type,
            amount=original.amount,
        )
        return escalated

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def get_pending_requests(self, approver_role: Optional[str] = None) -> List[ApprovalRequest]:
        """Return all pending approval requests, optionally filtered by role.

        Args:
            approver_role: When provided, only return requests assigned to
                this role.

        Returns:
            List of pending :class:`ApprovalRequest` instances.
        """
        if approver_role is None:
            return list(self._pending_requests.values())
        return [
            req
            for req in self._pending_requests.values()
            if req.approver_role == approver_role
        ]

    def get_request(self, request_id: UUID) -> Optional[ApprovalRequest]:
        """Look up a single approval request by ID.

        Searches both pending and completed stores.

        Args:
            request_id: UUID of the request to retrieve.

        Returns:
            The :class:`ApprovalRequest` if found, otherwise ``None``.
        """
        request = self._pending_requests.get(request_id)
        if request is not None:
            return request
        return self._completed_requests.get(request_id)

    def get_approval_chain(self, transaction_id: UUID) -> List[ApprovalRequest]:
        """Return all approval requests related to a transaction, sorted by time.

        Searches both pending and completed request stores for entries whose
        ``transaction_id`` matches.

        Args:
            transaction_id: UUID of the originating transaction.

        Returns:
            List of :class:`ApprovalRequest` instances sorted by
            ``created_at`` ascending (oldest first).
        """
        chain: List[ApprovalRequest] = []
        for req in self._pending_requests.values():
            if req.transaction_id == transaction_id:
                chain.append(req)
        for req in self._completed_requests.values():
            if req.transaction_id == transaction_id:
                chain.append(req)
        chain.sort(key=lambda r: r.created_at)
        return chain

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of approval system metrics.

        Includes running counters plus current pending and completed counts.

        Returns:
            Dictionary with metric keys:
            - total_requests, total_approved, total_rejected, total_escalated
            - average_approval_time (seconds)
            - pending_count, completed_count
        """
        snapshot = dict(self.metrics)
        snapshot["pending_count"] = len(self._pending_requests)
        snapshot["completed_count"] = len(self._completed_requests)
        return snapshot

    def get_threshold_info(self, txn_type: str) -> List[Dict[str, Any]]:
        """Return human-readable threshold information for a transaction type.

        Useful for agents making approval decisions — each entry describes
        a threshold band with its amount range and required approver role.

        Args:
            txn_type: Transaction category key (e.g. ``"purchase_order"``).

        Returns:
            List of dictionaries, each containing:
            - ``min_amount``: Lower bound (inclusive).
            - ``max_amount``: Upper bound (exclusive), or ``None``.
            - ``approver_role``: Required role, or ``None`` for no approval.
            - ``description``: Human-readable summary string.
        """
        ranges = self._thresholds.get(txn_type)
        if ranges is None:
            return []

        info: List[Dict[str, Any]] = []
        for min_amount, max_amount, approver_role in ranges:
            if max_amount is None:
                desc = f">= ${min_amount:,.0f}: requires {approver_role or 'no approval'}"
            elif min_amount == 0:
                desc = f"< ${max_amount:,.0f}: requires {approver_role or 'no approval'}"
            else:
                desc = f"${min_amount:,.0f} - ${max_amount:,.0f}: requires {approver_role or 'no approval'}"
            info.append(
                {
                    "min_amount": min_amount,
                    "max_amount": max_amount,
                    "approver_role": approver_role,
                    "description": desc,
                }
            )
        return info

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _compute_threshold_level(self, txn_type: str, amount: float) -> str:
        """Compute a human-readable threshold level description for an amount.

        Args:
            txn_type: Transaction category key.
            amount: Monetary value.

        Returns:
            A string like ``"$5K-$25K"`` or ``">$100K"`` indicating which
            threshold band the amount falls into, or ``""`` if unknown.
        """
        ranges = self._thresholds.get(txn_type)
        if ranges is None:
            return ""

        for min_amount, max_amount, _role in ranges:
            matched = False
            if max_amount is None:
                if amount >= min_amount:
                    matched = True
            else:
                if min_amount <= amount < max_amount:
                    matched = True

            if matched:
                return self._format_threshold_range(min_amount, max_amount)
        return ""

    @staticmethod
    def _format_threshold_range(min_amount: float, max_amount: Optional[float]) -> str:
        """Format a threshold range as a human-readable string.

        Args:
            min_amount: Lower bound of the range.
            max_amount: Upper bound (``None`` for unbounded).

        Returns:
            Formatted string, e.g. ``"$5K-$25K"`` or ``">$100K"``.
        """
        def _fmt(val: float) -> str:
            """Format a dollar amount into a compact representation."""
            if val >= 1_000_000:
                return f"${val / 1_000_000:.0f}M"
            if val >= 1_000:
                return f"${val / 1_000:.0f}K"
            return f"${val:.0f}"

        if min_amount == 0 and max_amount is not None:
            return f"<{_fmt(max_amount)}"
        if max_amount is None:
            return f">{_fmt(min_amount)}"
        return f"{_fmt(min_amount)}-{_fmt(max_amount)}"

    def _find_next_approver(self, txn_type: str, current_role: str) -> Optional[str]:
        """Find the next higher-authority approver role in the threshold chain.

        Iterates the threshold list for *txn_type*, locates the entry whose
        approver role matches *current_role*, and returns the role from the
        next entry in the list (if any).

        Args:
            txn_type: Transaction category key.
            current_role: The role of the approver that rejected.

        Returns:
            The next approver role string, or ``None`` if no higher
            authority exists in the chain.
        """
        ranges = self._thresholds.get(txn_type)
        if ranges is None:
            return None

        # Collect the ordered list of (index, role) for this transaction type
        roles_in_order: List[Tuple[int, Optional[str]]] = [
            (idx, role) for idx, (_min, _max, role) in enumerate(ranges)
        ]

        # Find the index of the current role
        current_idx: Optional[int] = None
        for idx, role in roles_in_order:
            if role == current_role:
                current_idx = idx
                break

        if current_idx is None:
            return None

        # Look for the next role with a non-None approver
        for idx, role in roles_in_order:
            if idx > current_idx and role is not None:
                return role

        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "APPROVAL_THRESHOLDS",
    "ApprovalSystem",
]
