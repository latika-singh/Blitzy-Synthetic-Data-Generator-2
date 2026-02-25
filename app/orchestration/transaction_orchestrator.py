"""Transaction orchestrator for lifecycle management, artifact tracking, and completeness validation.

Tracks required artifacts per transaction type (purchase orders, vendor invoices,
vendor payments, sales orders, customer invoices/payments, journal entries, goods
receipts, accruals, period close, depreciation, recurring journals) and manages
the full lifecycle:
    register → add_artifact → check_completeness → mark_complete

Extended for Project 3 (Transaction Workflows & Discrepancies) with additional
artifact types for three-way matching results, GL posting on goods receipts,
accrual entries, and period close summaries.

Supports parent/child transaction chaining for linked business processes and
publishes TransactionCompleted events via EventBus for cross-subsystem
coordination (per AAP Section 0.7.1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.events.event_bus import EventBus

from app.events.event_types import TransactionCompleted

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# TransactionStatus Enum
# ---------------------------------------------------------------------------


class TransactionStatus(str, Enum):
    """Transaction lifecycle states.

    Lifecycle flow:
        REGISTERED → IN_PROGRESS / AWAITING_ARTIFACTS → COMPLETE
        Any state → FAILED / CANCELLED (terminal)
    """

    REGISTERED = "registered"
    IN_PROGRESS = "in_progress"
    AWAITING_ARTIFACTS = "awaiting_artifacts"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# TransactionArtifact Model (Pydantic V2)
# ---------------------------------------------------------------------------


class TransactionArtifact(BaseModel):
    """A single artifact produced during transaction processing.

    Artifacts represent discrete outputs of agent work — for example a
    ``po_header``, ``invoice_lines``, ``three_way_match`` result, or
    ``gl_entries`` posting set.

    Attributes:
        artifact_type: Identifier matching one of the expected artifact names
            in :data:`REQUIRED_ARTIFACTS`.
        artifact_data: Free-form dictionary carrying the artifact content.
        created_at: UTC timestamp of artifact creation.
        created_by_agent_id: UUID of the agent that produced this artifact, or
            ``None`` for system-generated artifacts.
    """

    artifact_type: str
    artifact_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_by_agent_id: Optional[UUID] = None

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# TransactionState Model (Pydantic V2)
# ---------------------------------------------------------------------------


class TransactionState(BaseModel):
    """Complete state representation for a tracked transaction.

    Holds the transaction payload, current lifecycle status, collected
    artifacts, links to parent/child transactions, and timing metadata.

    Attributes:
        transaction_id: Unique identifier (auto-generated UUID4).
        transaction_type: Business transaction kind — must be a key in
            :data:`REQUIRED_ARTIFACTS` for completeness checking.
        transaction_data: Original transaction payload as submitted.
        status: Current lifecycle status (see :class:`TransactionStatus`).
        artifacts: Mapping of artifact_type → :class:`TransactionArtifact`.
        required_artifacts: List of artifact types needed for completeness.
        parent_transaction_id: Link to a parent transaction for chaining.
        child_transaction_ids: Links to child transactions.
        created_at: UTC timestamp of registration.
        updated_at: UTC timestamp of last modification.
        completed_at: UTC timestamp of completion (``None`` until complete).
        metadata: Extensible metadata dictionary.
    """

    transaction_id: UUID = Field(default_factory=uuid4)
    transaction_type: str
    transaction_data: Dict[str, Any] = Field(default_factory=dict)
    status: TransactionStatus = TransactionStatus.REGISTERED
    artifacts: Dict[str, TransactionArtifact] = Field(default_factory=dict)
    required_artifacts: List[str] = Field(default_factory=list)
    parent_transaction_id: Optional[UUID] = None
    child_transaction_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Required Artifacts Mapping (per README.md lines 349-355, extended for P3)
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACTS: Dict[str, List[str]] = {
    # ── P2 Core Transaction Types (README specification) ──────────────
    "purchase_order": ["po_header", "po_lines", "approval"],
    "vendor_invoice": [
        "invoice_header",
        "invoice_lines",
        "three_way_match",
        "three_way_match_result",  # P3: ThreeWayMatcher structured result
        "gl_entries",
    ],
    "vendor_payment": [
        "payment_record",
        "payment_allocation",
        "gl_entries",
    ],
    "sales_order": ["order_header", "order_lines"],
    "customer_invoice": ["invoice_header", "invoice_lines", "gl_entries"],
    "customer_payment": ["payment_record", "payment_allocation", "gl_entries"],
    "journal_entry": ["je_header", "je_lines", "approval"],
    "goods_receipt": [
        "receipt_header",
        "receipt_lines",
        "gl_entries",  # P3: GoodsReceiptGenerator posts DR Inventory, CR AP Accrual
    ],
    # ── P3 Period Close & GL Transaction Types ────────────────────────
    "accrual": ["accrual_entry", "je_header", "je_lines", "gl_entries"],
    "period_close": ["period_close_summary", "trial_balance", "gl_entries"],
    "depreciation": ["je_header", "je_lines", "gl_entries"],
    "recurring_journal": ["je_header", "je_lines", "gl_entries"],
}


# ---------------------------------------------------------------------------
# TransactionOrchestrator
# ---------------------------------------------------------------------------


class TransactionOrchestrator:
    """Tracks transaction lifecycles, manages artifacts, validates completeness.

    The orchestrator provides the following lifecycle contract:

    1. **register_transaction** — register a new transaction, auto-resolving
       its required artifacts from the :data:`REQUIRED_ARTIFACTS` mapping.
    2. **add_artifact** — record an artifact produced by agent work.
    3. **check_completeness** — verify whether all required artifacts are
       present.
    4. **mark_complete** — finalise a complete transaction and publish a
       :class:`TransactionCompleted` event.
    5. **get_transaction_chain** — retrieve a linked parent → child chain.

    Constructor injection is used for all dependencies (per AAP Section 0.7.1).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        required_artifacts: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Initialise the TransactionOrchestrator.

        Args:
            event_bus: Optional :class:`EventBus` instance for publishing
                :class:`TransactionCompleted` events.  When ``None``, event
                publishing is silently skipped.
            required_artifacts: Custom artifact-requirement mapping.  When
                ``None`` the module-level :data:`REQUIRED_ARTIFACTS` is used.
        """
        self._event_bus = event_bus

        # Copy so callers cannot mutate the canonical mapping
        if required_artifacts is not None:
            self.required_artifacts: Dict[str, List[str]] = {
                k: list(v) for k, v in required_artifacts.items()
            }
        else:
            self.required_artifacts = {
                k: list(v) for k, v in REQUIRED_ARTIFACTS.items()
            }

        self.transactions: Dict[UUID, TransactionState] = {}

        self.metrics: Dict[str, Any] = {
            "total_registered": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
            "artifacts_added": 0,
        }

        logger.info(
            "transaction_orchestrator_initialized",
            known_transaction_types=list(self.required_artifacts.keys()),
        )

    # ------------------------------------------------------------------
    # Core Lifecycle — register_transaction
    # ------------------------------------------------------------------

    def register_transaction(
        self,
        transaction: Dict[str, Any],
        transaction_type: str,
        parent_transaction_id: Optional[UUID] = None,
    ) -> UUID:
        """Register a new transaction for artifact tracking.

        Creates a :class:`TransactionState`, resolves its required artifacts
        from the configuration mapping, and optionally links it to a parent
        transaction.

        Args:
            transaction: Arbitrary transaction payload dictionary.
            transaction_type: Business type key (e.g. ``"purchase_order"``).
            parent_transaction_id: Optional UUID linking to a parent
                transaction for chain queries.

        Returns:
            The newly generated transaction UUID.

        Raises:
            ValueError: If *parent_transaction_id* is provided but does not
                reference a known transaction.
        """
        required = list(self.required_artifacts.get(transaction_type, []))

        state = TransactionState(
            transaction_id=uuid4(),
            transaction_type=transaction_type,
            transaction_data=transaction,
            status=TransactionStatus.REGISTERED,
            required_artifacts=required,
            parent_transaction_id=parent_transaction_id,
        )

        self.transactions[state.transaction_id] = state

        # Link to parent if specified
        if parent_transaction_id is not None:
            parent = self.transactions.get(parent_transaction_id)
            if parent is None:
                raise ValueError(
                    f"Parent transaction {parent_transaction_id} not found"
                )
            parent.child_transaction_ids.append(state.transaction_id)
            parent.updated_at = datetime.now(timezone.utc)

        self.metrics["total_registered"] += 1

        logger.info(
            "transaction_registered",
            transaction_id=str(state.transaction_id),
            transaction_type=transaction_type,
            required_artifacts_count=len(required),
            parent_transaction_id=(
                str(parent_transaction_id)
                if parent_transaction_id is not None
                else None
            ),
        )

        return state.transaction_id

    # ------------------------------------------------------------------
    # Core Lifecycle — add_artifact
    # ------------------------------------------------------------------

    def add_artifact(
        self,
        transaction_id: UUID,
        artifact_type: str,
        artifact_data: Dict[str, Any],
        agent_id: Optional[UUID] = None,
    ) -> bool:
        """Record an artifact produced during transaction processing.

        Creates a :class:`TransactionArtifact` and stores it against the
        transaction.  The transaction status is moved to ``IN_PROGRESS`` if
        it was previously ``REGISTERED`` or ``AWAITING_ARTIFACTS``.

        Args:
            transaction_id: UUID of the target transaction.
            artifact_type: Artifact name (e.g. ``"po_header"``).
            artifact_data: Free-form content dictionary.
            agent_id: Optional UUID of the agent that produced the artifact.

        Returns:
            ``True`` on success.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        artifact = TransactionArtifact(
            artifact_type=artifact_type,
            artifact_data=artifact_data,
            created_by_agent_id=agent_id,
        )
        state.artifacts[artifact_type] = artifact

        # Transition status to IN_PROGRESS when work begins
        if state.status in (
            TransactionStatus.REGISTERED,
            TransactionStatus.AWAITING_ARTIFACTS,
        ):
            state.status = TransactionStatus.IN_PROGRESS

        state.updated_at = datetime.now(timezone.utc)
        self.metrics["artifacts_added"] += 1

        logger.debug(
            "artifact_added",
            transaction_id=str(transaction_id),
            artifact_type=artifact_type,
            agent_id=str(agent_id) if agent_id is not None else None,
        )

        return True

    # ------------------------------------------------------------------
    # Core Lifecycle — check_completeness
    # ------------------------------------------------------------------

    def check_completeness(self, transaction_id: UUID) -> bool:
        """Check whether all required artifacts have been collected.

        Args:
            transaction_id: UUID of the target transaction.

        Returns:
            ``True`` if every artifact in the transaction's
            ``required_artifacts`` list has a corresponding entry in
            ``artifacts``.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        required: Set[str] = set(state.required_artifacts)
        present: Set[str] = set(state.artifacts.keys())
        missing: List[str] = sorted(required - present)
        is_complete = required.issubset(present)

        logger.debug(
            "completeness_checked",
            transaction_id=str(transaction_id),
            complete=is_complete,
            missing=missing,
            required_count=len(required),
            present_count=len(present),
        )

        return is_complete

    # ------------------------------------------------------------------
    # Core Lifecycle — mark_complete (async for event publishing)
    # ------------------------------------------------------------------

    async def mark_complete(self, transaction_id: UUID) -> bool:
        """Mark a transaction as complete and publish a completion event.

        Validates completeness first; if not all required artifacts are
        present the method logs a warning and returns ``False``.  On
        success a :class:`TransactionCompleted` event is published via the
        injected :class:`EventBus` (if available).

        Args:
            transaction_id: UUID of the target transaction.

        Returns:
            ``True`` if the transaction was successfully marked complete,
            ``False`` if completeness validation failed.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        # Pre-check completeness
        if not self.check_completeness(transaction_id):
            logger.warning(
                "mark_complete_rejected_incomplete",
                transaction_id=str(transaction_id),
                transaction_type=state.transaction_type,
                missing=self.get_missing_artifacts(transaction_id),
            )
            return False

        now = datetime.now(timezone.utc)
        state.status = TransactionStatus.COMPLETE
        state.completed_at = now
        state.updated_at = now
        self.metrics["total_completed"] += 1

        # Compute duration in seconds
        duration_seconds: float = 0.0
        if state.created_at is not None:
            delta = now - state.created_at
            duration_seconds = delta.total_seconds()

        # Publish TransactionCompleted event via EventBus
        if self._event_bus is not None:
            event = TransactionCompleted(
                payload={
                    "transaction_id": str(state.transaction_id),
                    "transaction_type": state.transaction_type,
                    "final_status": "completed",
                    "completion_reason": "All required artifacts present",
                    "duration_seconds": duration_seconds,
                },
            )
            await self._event_bus.publish(event)

        logger.info(
            "transaction_completed",
            transaction_id=str(transaction_id),
            transaction_type=state.transaction_type,
            duration_seconds=round(duration_seconds, 3),
            artifact_count=len(state.artifacts),
        )

        return True

    # ------------------------------------------------------------------
    # Core Lifecycle — get_transaction_chain
    # ------------------------------------------------------------------

    def get_transaction_chain(
        self, transaction_id: UUID
    ) -> List[TransactionState]:
        """Retrieve the full parent → children chain for a transaction.

        Walks upward via ``parent_transaction_id`` to find the root, then
        collects all descendants depth-first via ``child_transaction_ids``.

        Args:
            transaction_id: Any UUID in the chain.

        Returns:
            Ordered list with the root transaction first, followed by
            children in depth-first order.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        # Walk up to root
        root = state
        visited_up: Set[UUID] = {root.transaction_id}
        while root.parent_transaction_id is not None:
            parent = self.transactions.get(root.parent_transaction_id)
            if parent is None or parent.transaction_id in visited_up:
                break
            visited_up.add(parent.transaction_id)
            root = parent

        # Collect descendants depth-first from root
        chain: List[TransactionState] = []
        stack: List[TransactionState] = [root]
        visited: Set[UUID] = set()

        while stack:
            current = stack.pop()
            if current.transaction_id in visited:
                continue
            visited.add(current.transaction_id)
            chain.append(current)

            # Push children in reverse so left-most child is processed first
            for child_id in reversed(current.child_transaction_ids):
                child = self.transactions.get(child_id)
                if child is not None and child.transaction_id not in visited:
                    stack.append(child)

        logger.debug(
            "chain_retrieved",
            root_id=str(root.transaction_id),
            chain_length=len(chain),
            requested_id=str(transaction_id),
        )

        return chain

    # ------------------------------------------------------------------
    # Additional Lifecycle — mark_failed
    # ------------------------------------------------------------------

    def mark_failed(self, transaction_id: UUID, reason: str) -> None:
        """Mark a transaction as failed with a reason.

        Args:
            transaction_id: UUID of the target transaction.
            reason: Human-readable failure reason.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        now = datetime.now(timezone.utc)
        state.status = TransactionStatus.FAILED
        state.updated_at = now
        state.metadata["failure_reason"] = reason
        state.metadata["failure_timestamp"] = now.isoformat()
        self.metrics["total_failed"] += 1

        logger.error(
            "transaction_failed",
            transaction_id=str(transaction_id),
            transaction_type=state.transaction_type,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Additional Lifecycle — cancel_transaction
    # ------------------------------------------------------------------

    def cancel_transaction(self, transaction_id: UUID, reason: str) -> None:
        """Cancel a transaction.

        Args:
            transaction_id: UUID of the target transaction.
            reason: Human-readable cancellation reason.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        now = datetime.now(timezone.utc)
        state.status = TransactionStatus.CANCELLED
        state.updated_at = now
        state.metadata["cancellation_reason"] = reason
        state.metadata["cancellation_timestamp"] = now.isoformat()
        self.metrics["total_cancelled"] += 1

        logger.info(
            "transaction_cancelled",
            transaction_id=str(transaction_id),
            transaction_type=state.transaction_type,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Additional Lifecycle — get_transaction
    # ------------------------------------------------------------------

    def get_transaction(
        self, transaction_id: UUID
    ) -> Optional[TransactionState]:
        """Retrieve a single transaction by ID.

        Args:
            transaction_id: UUID to look up.

        Returns:
            The :class:`TransactionState` or ``None`` if not found.
        """
        return self.transactions.get(transaction_id)

    # ------------------------------------------------------------------
    # Additional Lifecycle — get_incomplete_transactions
    # ------------------------------------------------------------------

    def get_incomplete_transactions(self) -> List[TransactionState]:
        """Return all transactions that have not reached a terminal state.

        Terminal states are ``COMPLETE``, ``FAILED``, and ``CANCELLED``.

        Returns:
            List of non-terminal :class:`TransactionState` instances.
        """
        terminal_statuses: Set[str] = {
            TransactionStatus.COMPLETE.value,
            TransactionStatus.FAILED.value,
            TransactionStatus.CANCELLED.value,
        }
        return [
            txn
            for txn in self.transactions.values()
            if txn.status not in terminal_statuses
        ]

    # ------------------------------------------------------------------
    # Additional Lifecycle — get_missing_artifacts
    # ------------------------------------------------------------------

    def get_missing_artifacts(self, transaction_id: UUID) -> List[str]:
        """Return artifact types that are required but not yet present.

        Args:
            transaction_id: UUID of the target transaction.

        Returns:
            Sorted list of missing artifact type names.

        Raises:
            ValueError: If *transaction_id* does not reference a known
                transaction.
        """
        state = self.transactions.get(transaction_id)
        if state is None:
            raise ValueError(
                f"Transaction {transaction_id} not found"
            )

        required: Set[str] = set(state.required_artifacts)
        present: Set[str] = set(state.artifacts.keys())
        return sorted(required - present)

    # ------------------------------------------------------------------
    # Query — get_transactions_by_type
    # ------------------------------------------------------------------

    def get_transactions_by_type(
        self, transaction_type: str
    ) -> List[TransactionState]:
        """Filter transactions by business type.

        Args:
            transaction_type: e.g. ``"purchase_order"``.

        Returns:
            List of matching :class:`TransactionState` instances.
        """
        return [
            txn
            for txn in self.transactions.values()
            if txn.transaction_type == transaction_type
        ]

    # ------------------------------------------------------------------
    # Query — get_transactions_by_status
    # ------------------------------------------------------------------

    def get_transactions_by_status(
        self, status: TransactionStatus
    ) -> List[TransactionState]:
        """Filter transactions by lifecycle status.

        Args:
            status: Target :class:`TransactionStatus`.

        Returns:
            List of matching :class:`TransactionState` instances.
        """
        # Normalise to the enum's .value for comparison since
        # model_config uses use_enum_values=True.
        status_value = status.value if isinstance(status, TransactionStatus) else status
        return [
            txn
            for txn in self.transactions.values()
            if txn.status == status_value
        ]

    # ------------------------------------------------------------------
    # Query — get_metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return orchestrator metrics and aggregate statistics.

        Returns:
            Dictionary with counters (``total_registered``,
            ``total_completed``, ``total_failed``, ``artifacts_added``),
            plus dynamic aggregates: ``active_transactions``, ``by_type``,
            ``by_status``, and ``average_artifacts_per_transaction``.
        """
        # Start with a copy of base metrics
        result: Dict[str, Any] = dict(self.metrics)

        # Active transactions (non-terminal)
        terminal_values: Set[str] = {
            TransactionStatus.COMPLETE.value,
            TransactionStatus.FAILED.value,
            TransactionStatus.CANCELLED.value,
        }
        active_count = sum(
            1
            for txn in self.transactions.values()
            if txn.status not in terminal_values
        )
        result["active_transactions"] = active_count

        # Breakdown by type
        by_type: Dict[str, int] = {}
        for txn in self.transactions.values():
            by_type[txn.transaction_type] = by_type.get(txn.transaction_type, 0) + 1
        result["by_type"] = by_type

        # Breakdown by status
        by_status: Dict[str, int] = {}
        for txn in self.transactions.values():
            status_key = txn.status if isinstance(txn.status, str) else txn.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1
        result["by_status"] = by_status

        # Average artifacts per transaction
        total_txn = len(self.transactions)
        if total_txn > 0:
            total_artifacts = sum(
                len(txn.artifacts) for txn in self.transactions.values()
            )
            result["average_artifacts_per_transaction"] = round(
                total_artifacts / total_txn, 2
            )
        else:
            result["average_artifacts_per_transaction"] = 0.0

        return result
