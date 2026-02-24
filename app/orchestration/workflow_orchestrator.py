"""Workflow orchestration engine for transaction routing, agent assignment,
and workflow lifecycle management.

Routes transactions to agents via role mapping with idle-agent selection
(smallest queue).  Supports approval chain integration and event-driven
coordination through the injected :class:`EventBus`.

Exports:
    WorkflowStatus       — Str enum for workflow lifecycle states.
    WorkflowConfig       — Pydantic V2 model for orchestrator configuration.
    WorkflowInstance     — Pydantic V2 model tracking a single workflow.
    ROLE_MAPPING         — Module-level constant mapping transaction types to
                           eligible agent roles.
    WorkflowOrchestrator — Central routing engine: transaction routing, agent
                           selection, queue management, workflow lifecycle.

Design Decisions:
    - Pydantic V2 ``BaseModel`` for ``WorkflowConfig`` and ``WorkflowInstance``
      per AAP Section 0.7.1 (cross-module data contracts).
    - ``TYPE_CHECKING`` guard for ``AgentRegistry``, ``BaseAgent``,
      ``ApprovalSystem``, and ``EventBus`` to avoid circular imports.
    - ``AgentState`` and event types are *runtime* imports because they are
      used as comparison values and constructor arguments respectively.
    - Constructor injection for all dependencies (AAP Section 0.7.1).
    - ``EventBus`` is the sole mechanism for cross-subsystem asynchronous
      notifications (AAP Section 0.7.1).
    - Structured JSON logging via ``structlog`` to stdout only
      (AAP Section 0.7.6).

Performance Targets (AAP Section 0.7.3):
    - Workflow routing SLA: < 500 ms (O(n) over idle agents is acceptable
      per README.md lines 1357-1376).
    - Concurrent workflows: >= 100.

Retry Policy (README.md lines 1636-1641):
    - Max 3 retry attempts per workflow, then mark FAILED.

References:
    - README.md lines 277-306 (API sketch)
    - README.md lines 1270-1377 (full implementation example)
    - README.md lines 1346-1355 (ROLE_MAPPING)
    - README.md lines 1636-1641 (retry policy)
    - AAP Section 0.5.1 Group 6 (Orchestration Layer)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

# Runtime imports — values are used directly at execution time, not merely
# for type-checking.
from app.agents.agent_config import AgentState
from app.events.event_types import (
    ApprovalRequired,
    TransactionCompleted,
    TransactionCreated,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.agents.base_agent import BaseAgent
    from app.events.event_bus import EventBus
    from app.orchestration.approval_system import ApprovalSystem

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# WorkflowStatus Enum
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Serialization-friendly workflow lifecycle states.

    Uses ``str`` mixin so enum values serialize naturally to JSON strings
    in Pydantic V2 models and structured log output.

    Lifecycle paths:
        Happy path:  PENDING -> ASSIGNED -> IN_PROGRESS -> COMPLETED
        Approval:    ASSIGNED -> AWAITING_APPROVAL -> ASSIGNED -> COMPLETED
        Retry:       FAILED -> RETRYING -> ASSIGNED -> COMPLETED
        Terminal:    COMPLETED | FAILED | CANCELLED
    """

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


# Terminal statuses — workflows in these states cannot be re-routed.
_TERMINAL_STATUSES = frozenset(
    {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}
)


# ---------------------------------------------------------------------------
# WorkflowConfig — Pydantic V2 configuration model
# ---------------------------------------------------------------------------


class WorkflowConfig(BaseModel):
    """Configuration for the :class:`WorkflowOrchestrator`.

    All defaults are derived from the specification:
        - ``max_concurrent_workflows``: >= 100 (AAP Section 0.7.3)
        - ``routing_timeout_ms``: < 500 ms SLA (AAP Section 0.7.3)
        - ``retry_max_attempts``: 3 (README.md line 1636)

    Attributes:
        max_concurrent_workflows: Upper limit on simultaneously active
            (non-terminal) workflows.  Default 100.
        routing_timeout_ms: Hard time budget for the ``route_transaction``
            method in milliseconds.  Default 500.
        max_queue_depth_per_agent: Maximum work items a single agent queue
            may hold before the agent is considered unavailable.  Default 50.
        retry_max_attempts: Maximum number of re-route attempts for a
            failed workflow before it is marked permanently FAILED.
        retry_delay_seconds: Base delay in seconds before a retry is
            attempted.  Default 5.0.
        enable_approval_routing: Whether to check approval thresholds.
            Default ``True``.
        pending_queue_max_size: Maximum items in the global pending queue
            for transactions awaiting agent availability.  Default 1000.
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    max_concurrent_workflows: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent non-terminal workflows.",
    )
    routing_timeout_ms: int = Field(
        default=500,
        ge=1,
        description="Routing SLA in milliseconds.",
    )
    max_queue_depth_per_agent: int = Field(
        default=50,
        ge=1,
        description="Max work items per agent queue.",
    )
    retry_max_attempts: int = Field(
        default=3,
        ge=0,
        description="Max retry attempts before permanent failure.",
    )
    retry_delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        description="Base delay before retry.",
    )
    enable_approval_routing: bool = Field(
        default=True,
        description="Whether to check approval thresholds.",
    )
    pending_queue_max_size: int = Field(
        default=1000,
        ge=1,
        description="Max items in the global pending queue.",
    )


# ---------------------------------------------------------------------------
# WorkflowInstance — Pydantic V2 model for individual workflow tracking
# ---------------------------------------------------------------------------


class WorkflowInstance(BaseModel):
    """Tracks the lifecycle of a single routed transaction.

    Created by ``WorkflowOrchestrator.route_transaction()`` and updated as
    the workflow progresses through assignment, processing, approval, and
    completion or failure.

    Attributes:
        workflow_id: Unique workflow identifier (auto-generated UUID4).
        transaction_type: Type discriminator (e.g. ``"vendor_invoice"``).
        transaction_data: Raw transaction payload.
        status: Current lifecycle state.
        assigned_agent_id: UUID of the agent processing this workflow.
        created_at: UTC timestamp of workflow creation.
        updated_at: UTC timestamp of last status change.
        completed_at: UTC timestamp when the workflow reached a terminal state.
        error_message: Human-readable error description (set on failure).
        error_timestamp: UTC timestamp when the error occurred.
        retry_count: Number of retry attempts so far.
        approval_required: Whether this workflow requires approval routing.
        approval_role: Role required for approval (e.g. ``"controller"``).
        metadata: Arbitrary key-value metadata for extensions.
    """

    model_config = ConfigDict(use_enum_values=True)

    workflow_id: UUID = Field(default_factory=uuid4)
    transaction_type: str
    transaction_data: Dict[str, Any] = Field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    assigned_agent_id: Optional[UUID] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_timestamp: Optional[datetime] = None
    retry_count: int = 0
    approval_required: bool = False
    approval_role: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ROLE_MAPPING — Transaction type to agent roles (README.md lines 1346-1355)
# ---------------------------------------------------------------------------

ROLE_MAPPING: Dict[str, List[str]] = {
    "vendor_invoice": ["ap_clerk", "ap_manager"],
    "purchase_order": ["purchasing_agent", "purchasing_manager"],
    "sales_order": ["ar_clerk"],
    "customer_payment": ["ar_clerk"],
    "journal_entry": ["accountant", "senior_accountant"],
    "vendor_payment": ["ap_clerk"],
    "goods_receipt": ["warehouse_clerk"],
    "customer_invoice": ["ar_clerk"],
}
"""Maps each transaction type to the list of eligible agent roles.

The roles are ordered by primary handler first (e.g. ``ap_clerk`` before
``ap_manager`` for ``vendor_invoice``).  The WorkflowOrchestrator selects
the idle agent with the smallest queue from these candidate roles.
"""


# ---------------------------------------------------------------------------
# WorkflowOrchestrator — Central routing engine
# ---------------------------------------------------------------------------


class WorkflowOrchestrator:
    """Routes transactions to agents and manages workflow lifecycles.

    The orchestrator is the central routing engine of the orchestration
    layer (F-003).  It maps transaction types to eligible agent roles
    via :data:`ROLE_MAPPING`, selects the idle agent with the smallest
    work queue, enqueues a work item, and tracks the workflow through
    assignment, approval, completion, and failure states.

    Constructor Injection (AAP Section 0.7.1):
        agent_registry:  Provides role-based agent lookup.
        approval_system: Determines whether approval routing is required.
        config:          Orchestrator tuning parameters.
        event_bus:       (Optional) Publishes TransactionCreated,
                         ApprovalRequired, and TransactionCompleted events.

    Performance:
        - Routing SLA: < 500 ms (agent lookup is O(n) over idle candidates)
        - Concurrent workflows: >= 100

    Attributes:
        workflows:   Dict mapping workflow UUIDs to WorkflowInstance objects.
        work_queues: Dict mapping agent UUIDs to asyncio.Queue objects.
        metrics:     Real-time operational metrics dictionary.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        agent_registry: AgentRegistry,
        approval_system: ApprovalSystem,
        config: Optional[WorkflowConfig] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialise the orchestrator with injected dependencies.

        No I/O is performed during initialisation to keep construction
        lightweight and support rapid startup.

        Args:
            agent_registry:  Central agent lookup / lifecycle manager.
            approval_system: Threshold-based approval chain engine.
            config:          Orchestrator configuration.  Defaults to
                             ``WorkflowConfig()`` if not supplied.
            event_bus:       Optional event bus for cross-subsystem
                             notifications.
        """
        self.agent_registry: AgentRegistry = agent_registry
        self.approval_system: ApprovalSystem = approval_system
        self.config: WorkflowConfig = config if config is not None else WorkflowConfig()
        self.event_bus: Optional[EventBus] = event_bus

        # Workflow tracking — Dict[UUID, WorkflowInstance]
        self.workflows: Dict[UUID, WorkflowInstance] = {}

        # Per-agent work queues — Dict[UUID, asyncio.Queue]
        self.work_queues: Dict[UUID, asyncio.Queue] = {}

        # Global pending queue for transactions awaiting agent availability.
        self._pending_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.pending_queue_max_size,
        )

        # Operational metrics
        self.metrics: Dict[str, Any] = {
            "total_routed": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_retried": 0,
            "average_queue_depth": 0.0,
            "average_processing_time": 0.0,
        }

        # Accumulator for computing running average processing time
        self._total_processing_time: float = 0.0

        logger.info(
            "workflow_orchestrator_initialized",
            max_concurrent_workflows=self.config.max_concurrent_workflows,
            routing_timeout_ms=self.config.routing_timeout_ms,
            max_queue_depth_per_agent=self.config.max_queue_depth_per_agent,
            pending_queue_max_size=self.config.pending_queue_max_size,
        )

    # ------------------------------------------------------------------
    # Primary Routing — README.md lines 1289-1344
    # ------------------------------------------------------------------

    async def route_transaction(
        self,
        transaction: Dict[str, Any],
        transaction_type: str,
    ) -> UUID:
        """Route a transaction to an appropriate agent.

        Performs the following steps (must complete within 500 ms SLA):
            1. Create a :class:`WorkflowInstance`.
            2. Enforce concurrent workflow limit.
            3. Look up eligible roles via :data:`ROLE_MAPPING`.
            4. Check approval thresholds via
               ``ApprovalSystem.get_required_approval``.
            5. Find the idle agent with the smallest queue.
            6. Enqueue a work item on the selected agent.
            7. Publish a ``TransactionCreated`` event (if event bus present).
            8. Log the routing decision.

        If no idle agent is available the workflow is placed on the
        global pending queue for later processing.

        Args:
            transaction:      Raw transaction data dictionary.
            transaction_type: Transaction category (e.g. ``"vendor_invoice"``).

        Returns:
            The UUID of the newly created :class:`WorkflowInstance`.

        Raises:
            RuntimeError: If the concurrent workflow limit is exceeded.
        """
        # 1. Create workflow instance
        workflow = WorkflowInstance(
            transaction_type=transaction_type,
            transaction_data=transaction,
            status=WorkflowStatus.PENDING,
        )
        self.workflows[workflow.workflow_id] = workflow

        # 2. Enforce concurrent workflow limit
        active_count = sum(
            1
            for wf in self.workflows.values()
            if wf.status not in _TERMINAL_STATUSES
        )
        if active_count > self.config.max_concurrent_workflows:
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = (
                f"Concurrent workflow limit exceeded "
                f"({self.config.max_concurrent_workflows})"
            )
            workflow.error_timestamp = datetime.now(timezone.utc)
            workflow.updated_at = datetime.now(timezone.utc)
            self.metrics["total_failed"] += 1
            logger.warning(
                "concurrent_workflow_limit_exceeded",
                workflow_id=str(workflow.workflow_id),
                active_count=active_count,
                limit=self.config.max_concurrent_workflows,
            )
            return workflow.workflow_id

        # 3. Determine eligible agent roles
        agent_roles = self._get_required_roles(transaction_type)
        if not agent_roles:
            await self._queue_for_later(workflow)
            return workflow.workflow_id

        # 4. Check if approval is required
        if self.config.enable_approval_routing:
            amount = transaction.get("amount", 0)
            if isinstance(amount, (int, float)):
                approval_role = self.approval_system.get_required_approval(
                    transaction_type, float(amount)
                )
                if approval_role is not None:
                    workflow.approval_required = True
                    workflow.approval_role = approval_role

        # 5. Find the idle agent with the smallest work queue
        agent = await self._find_available_agent(agent_roles)
        if agent is None:
            await self._queue_for_later(workflow)
            return workflow.workflow_id

        # 6. Create work item and enqueue on the agent
        work_item: Dict[str, Any] = {
            "type": transaction_type,
            "data": transaction,
            "workflow_id": str(workflow.workflow_id),
        }
        await agent.work_queue.put(work_item)

        # Ensure we track the agent's queue in our mirror dict
        agent_id: UUID = agent.config.agent_id
        if agent_id not in self.work_queues:
            self.work_queues[agent_id] = agent.work_queue

        # 7. Update workflow state
        workflow.assigned_agent_id = agent_id
        workflow.status = WorkflowStatus.ASSIGNED
        workflow.updated_at = datetime.now(timezone.utc)

        # 8. Increment routing metric
        self.metrics["total_routed"] += 1

        # 9. Publish TransactionCreated event
        if self.event_bus is not None:
            event = TransactionCreated(
                payload={
                    "transaction_type": transaction_type,
                    "workflow_id": str(workflow.workflow_id),
                    "amount": transaction.get("amount"),
                    "assigned_agent_id": str(agent_id),
                },
            )
            try:
                await self.event_bus.publish(event)
            except Exception:  # pragma: no cover — best-effort event publish
                logger.warning(
                    "event_publish_failed",
                    event_type="TransactionCreated",
                    workflow_id=str(workflow.workflow_id),
                )

        # 10. Structured log entry (AAP Section 0.7.6)
        queue_depth = agent.work_queue.qsize()
        logger.info(
            "transaction_routed",
            workflow_id=str(workflow.workflow_id),
            transaction_type=transaction_type,
            agent_id=str(agent_id),
            agent_role=agent.config.role,
            queue_depth=queue_depth,
        )

        # Update average queue depth metric
        self._update_average_queue_depth()

        return workflow.workflow_id

    # ------------------------------------------------------------------
    # Role Lookup — README.md lines 1346-1355
    # ------------------------------------------------------------------

    def _get_required_roles(self, transaction_type: str) -> List[str]:
        """Return eligible agent roles for a transaction type.

        Looks up the transaction type in :data:`ROLE_MAPPING`.  Logs a
        warning and returns an empty list when no mapping exists.

        Args:
            transaction_type: Transaction category key.

        Returns:
            List of eligible role identifiers (may be empty).
        """
        roles = ROLE_MAPPING.get(transaction_type, [])
        if not roles:
            logger.warning(
                "no_roles_for_transaction_type",
                transaction_type=transaction_type,
            )
        return list(roles)  # Return a copy to prevent mutation

    # ------------------------------------------------------------------
    # Agent Selection — README.md lines 1357-1376
    # ------------------------------------------------------------------

    async def _find_available_agent(
        self, roles: List[str]
    ) -> Optional[BaseAgent]:
        """Select the idle agent with the smallest work queue.

        Retrieves all agents matching the given roles from the
        :class:`AgentRegistry`, filters to ``AgentState.IDLE``, then
        selects the agent whose ``work_queue.qsize()`` is minimal.
        Additionally filters out agents whose queue depth exceeds
        ``config.max_queue_depth_per_agent``.

        Args:
            roles: List of eligible role identifiers.

        Returns:
            The selected :class:`BaseAgent`, or ``None`` if no idle
            agent with capacity is available.
        """
        candidates = self.agent_registry.get_agents_by_roles(roles)
        if not candidates:
            return None

        # Filter to idle agents whose queue is below the depth threshold
        idle_agents = [
            agent
            for agent in candidates
            if (
                agent.state == AgentState.IDLE
                and agent.work_queue.qsize() < self.config.max_queue_depth_per_agent
            )
        ]

        if not idle_agents:
            return None

        # Select the agent with the smallest queue (O(n) — well within SLA)
        selected = min(idle_agents, key=lambda a: a.work_queue.qsize())
        return selected

    # ------------------------------------------------------------------
    # Pending Queue Management
    # ------------------------------------------------------------------

    async def _queue_for_later(self, workflow: WorkflowInstance) -> None:
        """Place a workflow on the pending queue for later routing.

        Called when no idle agent is available or when no roles match
        the transaction type.  The workflow remains in ``PENDING`` status
        until :meth:`process_pending_queue` picks it up.

        Args:
            workflow: The workflow instance to defer.
        """
        workflow.status = WorkflowStatus.PENDING
        workflow.updated_at = datetime.now(timezone.utc)

        try:
            self._pending_queue.put_nowait(workflow.workflow_id)
        except asyncio.QueueFull:
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = "Pending queue is full"
            workflow.error_timestamp = datetime.now(timezone.utc)
            workflow.updated_at = datetime.now(timezone.utc)
            self.metrics["total_failed"] += 1
            logger.error(
                "pending_queue_full",
                workflow_id=str(workflow.workflow_id),
                transaction_type=workflow.transaction_type,
            )
            return

        logger.warning(
            "no_agent_available",
            workflow_id=str(workflow.workflow_id),
            transaction_type=workflow.transaction_type,
        )

    # ------------------------------------------------------------------
    # Approval Routing
    # ------------------------------------------------------------------

    async def route_for_approval(
        self, workflow_id: UUID, approver_role: str
    ) -> bool:
        """Route a workflow to an approver agent.

        Finds an idle agent matching ``approver_role``, enqueues an
        approval work item, and publishes an ``ApprovalRequired`` event.

        Args:
            workflow_id:   UUID of the workflow requiring approval.
            approver_role: Required approver role (e.g. ``"controller"``).

        Returns:
            ``True`` if the approval work item was successfully enqueued,
            ``False`` otherwise (e.g. no matching idle agent).
        """
        workflow = self.workflows.get(workflow_id)
        if workflow is None:
            logger.warning(
                "approval_route_workflow_not_found",
                workflow_id=str(workflow_id),
            )
            return False

        workflow.status = WorkflowStatus.AWAITING_APPROVAL
        workflow.updated_at = datetime.now(timezone.utc)

        # Find an idle approver agent
        agent = await self._find_available_agent([approver_role])
        if agent is None:
            logger.warning(
                "no_approver_available",
                workflow_id=str(workflow_id),
                approver_role=approver_role,
            )
            return False

        # Enqueue approval work item
        approval_work_item: Dict[str, Any] = {
            "type": "approval",
            "data": {
                "workflow_id": str(workflow_id),
                "transaction_type": workflow.transaction_type,
                "transaction_data": workflow.transaction_data,
                "approver_role": approver_role,
            },
            "workflow_id": str(workflow_id),
        }
        await agent.work_queue.put(approval_work_item)

        agent_id: UUID = agent.config.agent_id
        if agent_id not in self.work_queues:
            self.work_queues[agent_id] = agent.work_queue

        # Publish ApprovalRequired event
        if self.event_bus is not None:
            event = ApprovalRequired(
                payload={
                    "workflow_id": str(workflow_id),
                    "transaction_type": workflow.transaction_type,
                    "required_approver_role": approver_role,
                    "amount": workflow.transaction_data.get("amount"),
                },
            )
            try:
                await self.event_bus.publish(event)
            except Exception:  # pragma: no cover
                logger.warning(
                    "event_publish_failed",
                    event_type="ApprovalRequired",
                    workflow_id=str(workflow_id),
                )

        logger.info(
            "approval_routed",
            workflow_id=str(workflow_id),
            approver_role=approver_role,
            agent_id=str(agent_id),
            queue_depth=agent.work_queue.qsize(),
        )
        return True

    # ------------------------------------------------------------------
    # Workflow Lifecycle — Completion
    # ------------------------------------------------------------------

    async def complete_workflow(
        self,
        workflow_id: UUID,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a workflow as completed.

        Sets the terminal status, records the completion timestamp, updates
        metrics, and publishes a ``TransactionCompleted`` event.

        Args:
            workflow_id: UUID of the workflow to complete.
            result:      Optional result payload to store in metadata.
        """
        workflow = self.workflows.get(workflow_id)
        if workflow is None:
            logger.warning(
                "complete_workflow_not_found",
                workflow_id=str(workflow_id),
            )
            return

        now = datetime.now(timezone.utc)
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = now
        workflow.updated_at = now
        if result is not None:
            workflow.metadata["result"] = result

        # Update metrics
        self.metrics["total_completed"] += 1
        duration = (now - workflow.created_at).total_seconds()
        self._total_processing_time += duration
        completed_count = self.metrics["total_completed"]
        if completed_count > 0:
            self.metrics["average_processing_time"] = (
                self._total_processing_time / completed_count
            )

        # Publish TransactionCompleted event
        if self.event_bus is not None:
            event = TransactionCompleted(
                payload={
                    "workflow_id": str(workflow_id),
                    "transaction_type": workflow.transaction_type,
                    "final_status": "completed",
                    "duration_seconds": duration,
                },
            )
            try:
                await self.event_bus.publish(event)
            except Exception:  # pragma: no cover
                logger.warning(
                    "event_publish_failed",
                    event_type="TransactionCompleted",
                    workflow_id=str(workflow_id),
                )

        logger.info(
            "workflow_completed",
            workflow_id=str(workflow_id),
            transaction_type=workflow.transaction_type,
            duration_seconds=round(duration, 3),
        )

    # ------------------------------------------------------------------
    # Workflow Lifecycle — Failure
    # ------------------------------------------------------------------

    async def fail_workflow(self, workflow_id: UUID, error: str) -> None:
        """Mark a workflow as failed.

        Records the error message and timestamp, increments the failure
        metric, and logs the failure at ERROR level.

        Args:
            workflow_id: UUID of the workflow to fail.
            error:       Human-readable error description.
        """
        workflow = self.workflows.get(workflow_id)
        if workflow is None:
            logger.warning(
                "fail_workflow_not_found",
                workflow_id=str(workflow_id),
            )
            return

        now = datetime.now(timezone.utc)
        workflow.status = WorkflowStatus.FAILED
        workflow.error_message = error
        workflow.error_timestamp = now
        workflow.updated_at = now

        self.metrics["total_failed"] += 1

        logger.error(
            "workflow_failed",
            workflow_id=str(workflow_id),
            transaction_type=workflow.transaction_type,
            error=error,
        )

    # ------------------------------------------------------------------
    # Workflow Lifecycle — Retry (README.md lines 1636-1641)
    # ------------------------------------------------------------------

    async def retry_workflow(self, workflow_id: UUID) -> bool:
        """Attempt to retry a failed workflow.

        Increments the retry counter, sets status to ``RETRYING``, and
        re-routes the transaction.  If the maximum number of retries has
        been reached the workflow is permanently marked ``FAILED``.

        Args:
            workflow_id: UUID of the workflow to retry.

        Returns:
            ``True`` if the retry was successfully queued, ``False`` if
            the workflow was not found or the retry limit was exceeded.
        """
        workflow = self.workflows.get(workflow_id)
        if workflow is None:
            logger.warning(
                "retry_workflow_not_found",
                workflow_id=str(workflow_id),
            )
            return False

        if workflow.retry_count >= self.config.retry_max_attempts:
            workflow.status = WorkflowStatus.FAILED
            workflow.error_message = (
                f"Max retries exceeded ({self.config.retry_max_attempts})"
            )
            workflow.error_timestamp = datetime.now(timezone.utc)
            workflow.updated_at = datetime.now(timezone.utc)
            self.metrics["total_failed"] += 1
            logger.error(
                "workflow_max_retries_exceeded",
                workflow_id=str(workflow_id),
                retry_count=workflow.retry_count,
                max_attempts=self.config.retry_max_attempts,
            )
            return False

        workflow.retry_count += 1
        workflow.status = WorkflowStatus.RETRYING
        workflow.error_message = None
        workflow.error_timestamp = None
        workflow.updated_at = datetime.now(timezone.utc)

        self.metrics["total_retried"] += 1

        # Attempt to re-route the transaction
        agent_roles = self._get_required_roles(workflow.transaction_type)
        if not agent_roles:
            await self._queue_for_later(workflow)
            return True

        agent = await self._find_available_agent(agent_roles)
        if agent is None:
            await self._queue_for_later(workflow)
            return True

        work_item: Dict[str, Any] = {
            "type": workflow.transaction_type,
            "data": workflow.transaction_data,
            "workflow_id": str(workflow.workflow_id),
        }
        await agent.work_queue.put(work_item)

        agent_id: UUID = agent.config.agent_id
        if agent_id not in self.work_queues:
            self.work_queues[agent_id] = agent.work_queue

        workflow.assigned_agent_id = agent_id
        workflow.status = WorkflowStatus.ASSIGNED
        workflow.updated_at = datetime.now(timezone.utc)

        logger.info(
            "workflow_retried",
            workflow_id=str(workflow_id),
            transaction_type=workflow.transaction_type,
            retry_count=workflow.retry_count,
            agent_id=str(agent_id),
        )
        return True

    # ------------------------------------------------------------------
    # Workflow Queries
    # ------------------------------------------------------------------

    def get_workflow(self, workflow_id: UUID) -> Optional[WorkflowInstance]:
        """Retrieve a workflow instance by its UUID.

        Args:
            workflow_id: Unique workflow identifier.

        Returns:
            The :class:`WorkflowInstance`, or ``None`` if not found.
        """
        return self.workflows.get(workflow_id)

    def get_active_workflows(self) -> List[WorkflowInstance]:
        """Return all workflows that are **not** in a terminal state.

        Terminal states are: ``COMPLETED``, ``FAILED``, ``CANCELLED``.

        Returns:
            List of non-terminal :class:`WorkflowInstance` objects.
        """
        return [
            wf
            for wf in self.workflows.values()
            if wf.status not in _TERMINAL_STATUSES
        ]

    # ------------------------------------------------------------------
    # Pending Queue Processing
    # ------------------------------------------------------------------

    async def process_pending_queue(self) -> int:
        """Attempt to route all pending workflows.

        Drains the pending queue and tries to route each deferred
        transaction.  Workflows that still cannot be routed are placed
        back on the pending queue.

        Returns:
            The number of workflows successfully routed in this pass.
        """
        routed_count = 0
        requeue: List[UUID] = []

        while not self._pending_queue.empty():
            try:
                wf_id: UUID = self._pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            workflow = self.workflows.get(wf_id)
            if workflow is None:
                continue

            # Skip workflows that have already been completed or cancelled
            if workflow.status in _TERMINAL_STATUSES:
                continue

            agent_roles = self._get_required_roles(workflow.transaction_type)
            if not agent_roles:
                requeue.append(wf_id)
                continue

            agent = await self._find_available_agent(agent_roles)
            if agent is None:
                requeue.append(wf_id)
                continue

            # Route successfully
            work_item: Dict[str, Any] = {
                "type": workflow.transaction_type,
                "data": workflow.transaction_data,
                "workflow_id": str(workflow.workflow_id),
            }
            await agent.work_queue.put(work_item)

            agent_id: UUID = agent.config.agent_id
            if agent_id not in self.work_queues:
                self.work_queues[agent_id] = agent.work_queue

            workflow.assigned_agent_id = agent_id
            workflow.status = WorkflowStatus.ASSIGNED
            workflow.updated_at = datetime.now(timezone.utc)
            self.metrics["total_routed"] += 1
            routed_count += 1

            logger.info(
                "pending_workflow_routed",
                workflow_id=str(workflow.workflow_id),
                transaction_type=workflow.transaction_type,
                agent_id=str(agent_id),
            )

        # Re-queue workflows that could not be routed yet
        for wf_id in requeue:
            try:
                self._pending_queue.put_nowait(wf_id)
            except asyncio.QueueFull:  # pragma: no cover
                logger.error(
                    "pending_requeue_failed",
                    workflow_id=str(wf_id),
                )

        return routed_count

    # ------------------------------------------------------------------
    # Queue Depth Queries
    # ------------------------------------------------------------------

    def get_agent_queue_depth(self, agent_id: UUID) -> int:
        """Return the current work queue depth for a specific agent.

        Args:
            agent_id: UUID of the agent.

        Returns:
            Current queue size, or ``0`` if the agent is not tracked.
        """
        queue = self.work_queues.get(agent_id)
        if queue is not None:
            return queue.qsize()
        return 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of operational metrics.

        Includes the base counter metrics plus derived values:

        - ``active_workflows``: Number of non-terminal workflows.
        - ``pending_count``:    Current size of the pending queue.
        - ``workflows_by_status``: Breakdown of workflow counts per status.

        Returns:
            Dictionary of metric key/value pairs.
        """
        # Build status breakdown
        status_counts: Dict[str, int] = {}
        for wf in self.workflows.values():
            status_key = wf.status if isinstance(wf.status, str) else wf.status.value
            status_counts[status_key] = status_counts.get(status_key, 0) + 1

        result = dict(self.metrics)
        result["active_workflows"] = len(self.get_active_workflows())
        result["pending_count"] = self._pending_queue.qsize()
        result["workflows_by_status"] = status_counts
        return result

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _update_average_queue_depth(self) -> None:
        """Recompute the average work queue depth across tracked agents."""
        if not self.work_queues:
            self.metrics["average_queue_depth"] = 0.0
            return

        total_depth = sum(q.qsize() for q in self.work_queues.values())
        self.metrics["average_queue_depth"] = total_depth / len(self.work_queues)
