"""
BaseAgent Abstract Base Class — Foundation for all 12 specialized agent types.

This module defines the core abstract base class for the Agent System (F-001)
and the data contracts (WorkItem, WorkResult) used across the entire agent
and orchestration subsystems.

Exports:
    BaseAgent  — Abstract base class providing constructor injection
                 (config, memory, decision_engine, action_registry), async run()
                 loop, abstract process_work_item(), make_decision(), importance
                 calculation, and metrics tracking.
    WorkItem   — Dataclass representing a unit of work to be processed by an
                 agent (transaction type, data payload, workflow tracking ID,
                 priority, retry count, amount, timestamps).
    WorkResult — Dataclass representing the outcome of processing a WorkItem
                 (success flag, actions taken, approval needs, exception flags,
                 result data, error messages, processing duration).

Design Decisions:
    - Constructor injection per AAP Section 0.7.1: all dependencies are injected
      via __init__ (config, memory, decision_engine, action_registry). No module
      should instantiate its own dependencies.
    - asyncio.Queue for the agent work queue per README.md line 796 — FIFO task
      buffer that integrates naturally with the async run() loop.
    - Lightweight __init__ with no I/O to support ≥50 agents/second creation rate
      (AAP Section 0.7.3).
    - structlog for all logging per AAP Section 0.7.6 — structured JSON to stdout.
    - State machine: IDLE → THINKING → (ACTING) → IDLE, with ERROR path on
      exceptions. States are defined by AgentState enum from agent_config module.

Timeout Constraints (AAP Section 0.1.2):
    - Simple decision:       10 seconds
    - LLM decision:          30 seconds
    - Complex workflow:       60 seconds
    - Absolute max:          120 seconds
    - Memory add observation:  1 second
    - Memory retrieve relevant: 2 seconds

Performance Targets (AAP Section 0.7.3):
    - Agent creation:     ≥ 50 agents/second
    - Decision latency:   p95 < 5 seconds
    - Agent concurrency:  ≥ 20 simultaneous agents

References:
    - README.md lines 772–890 (BaseAgent specification and code examples)
    - README.md lines 1324–1328 (WorkItem usage in WorkflowOrchestrator)
    - README.md lines 937–949 (WorkResult usage in APClerkAgent example)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import structlog

from app.agents.agent_config import AgentConfig, AgentState
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# WorkItem Dataclass — Unit of work for agent processing
# ---------------------------------------------------------------------------
@dataclass
class WorkItem:
    """A unit of work to be processed by an agent.

    WorkItems are created by the WorkflowOrchestrator when routing transactions
    to agents (README.md lines 1324–1328) and placed into the agent's asyncio
    work queue. Each WorkItem carries the transaction type, data payload,
    workflow tracking identifier, priority, and optional monetary amount.

    Priority follows a 1–10 scale where lower values indicate higher priority
    (1 = highest priority, 10 = lowest priority). The default priority of 5
    represents normal processing.

    Attributes:
        type:        Transaction type identifier (e.g. "vendor_invoice",
                     "purchase_order", "journal_entry").
        data:        Transaction data payload as a flexible dictionary.
        workflow_id: Unique UUID for workflow tracking, auto-generated.
        description: Human-readable description of the work item.
        priority:    Priority level 1–10 (lower = higher priority, default 5).
        retry_count: Number of processing retry attempts (default 0).
        amount:      Transaction monetary amount, if applicable (default None).
        created_at:  UTC timestamp of when the work item was created.
    """

    type: str
    data: Dict[str, Any]
    workflow_id: UUID = field(default_factory=uuid4)
    description: str = ""
    priority: int = 5
    retry_count: int = 0
    amount: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def dict(self) -> Dict[str, Any]:
        """Serialize the work item to a dictionary for logging and persistence.

        Converts UUID and datetime fields to string representations for
        JSON-safe serialization. This method is used in structured logging
        throughout the agent framework.

        Returns:
            Dictionary containing all work item fields with JSON-safe types.
        """
        return {
            "type": self.type,
            "data": self.data,
            "workflow_id": str(self.workflow_id),
            "description": self.description,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "amount": self.amount,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# WorkResult Dataclass — Outcome of processing a WorkItem
# ---------------------------------------------------------------------------
@dataclass
class WorkResult:
    """Result of processing a WorkItem by an agent.

    Produced by specialized agent subclasses from their ``process_work_item()``
    implementation (README.md lines 937–949). Contains the success/failure
    status, list of actions executed, approval flags, exception indicators,
    result data, and processing duration.

    Attributes:
        success:                Whether processing completed successfully.
        actions_taken:          List of action identifiers that were executed
                                (e.g. ["gl_coding_assigned", "three_way_match_performed"]).
        approval_needed:        Whether further approval is required before the
                                transaction can proceed.
        has_exceptions:         Whether exceptions or discrepancies were detected
                                during processing.
        data:                   Result data payload as a flexible dictionary.
        error:                  Error message string if processing failed (None on success).
        processing_time_seconds: Wall-clock processing duration in seconds,
                                 set by the BaseAgent.run() loop after processing.
    """

    success: bool
    actions_taken: List[str] = field(default_factory=list)
    approval_needed: bool = False
    has_exceptions: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# BaseAgent Abstract Base Class
# ---------------------------------------------------------------------------
class BaseAgent(ABC):
    """Abstract base class for all agent types in the Agent System (F-001).

    Provides core agent capabilities through constructor-injected dependencies:
        - **Memory management** via AgentMemory (dual-stream: observations + reflections)
        - **Decision-making** via DecisionEngine (4-layer hybrid pipeline)
        - **Action execution** via ActionRegistry (14 ERP action types)
        - **Work queue processing** via asyncio.Queue (FIFO async task buffer)

    The agent lifecycle follows a state machine:
        IDLE → THINKING → (ACTING) → IDLE   (happy path)
        IDLE → THINKING → ERROR              (error path)

    Every agent decision produces a structured log entry including agent_id,
    agent_role, decision_type, context summary, decision result, and
    duration_seconds (AAP Section 0.7.6).

    All 12 specialized agent subclasses must extend this class and implement
    the abstract ``process_work_item()`` method:
        - APClerkAgent, APManagerAgent
        - ARClerkAgent, ARManagerAgent
        - PurchasingAgent, PurchasingManagerAgent
        - WarehouseClerkAgent, WarehouseManagerAgent
        - AccountantAgent, SeniorAccountantAgent
        - ControllerAgent, CFOAgent

    Constructor Injection (AAP Section 0.7.1):
        config:          AgentConfig with identity, traits, work hours, LLM settings
        memory:          AgentMemory with dual-stream observations/reflections
        decision_engine: DecisionEngine with 4-layer hybrid pipeline
        action_registry: ActionRegistry with 14 ERP action types

    Performance Targets:
        - Agent creation: ≥ 50 agents/second (lightweight __init__, no I/O)
        - Decision latency: p95 < 5 seconds
        - Agent concurrency: ≥ 20 simultaneous agents

    Timeout Constraints:
        - Simple decision:        10 seconds
        - LLM decision:           30 seconds
        - Complex workflow:        60 seconds
        - Absolute max:           120 seconds
        - Memory add observation:   1 second
        - Memory retrieve relevant: 2 seconds
    """

    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        decision_engine: DecisionEngine,
        action_registry: ActionRegistry,
    ) -> None:
        """Initialize the agent with constructor-injected dependencies.

        No I/O or heavy computation occurs during initialization to ensure
        the ≥50 agents/second creation rate target is met.

        Args:
            config:          Agent configuration (identity, traits, work hours).
            memory:          Dual-stream memory system (observations + reflections).
            decision_engine: 4-layer hybrid decision pipeline.
            action_registry: Registry of 14 ERP action types.
        """
        # Injected dependencies
        self.config: AgentConfig = config
        self.memory: AgentMemory = memory
        self.decision_engine: DecisionEngine = decision_engine
        self.action_registry: ActionRegistry = action_registry

        # Agent state machine — starts IDLE
        self.state: AgentState = AgentState.IDLE

        # Async work queue — FIFO task buffer (README.md line 796)
        self.work_queue: asyncio.Queue = asyncio.Queue()

        # Currently processing work item (None when idle)
        self.current_work_item: Optional[WorkItem] = None

        # Internal flag to control the run() loop
        self._running: bool = False

        # Performance metrics tracking
        self.metrics: Dict[str, Any] = {
            "items_processed": 0,
            "decisions_made": 0,
            "errors": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0,
        }

        # Log agent creation with structured context (AAP Section 0.7.6)
        logger.info(
            "agent_created",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            traits=self.config.traits,
        )

    # ------------------------------------------------------------------
    # Abstract Method — Must be implemented by all 12 specialized agents
    # ------------------------------------------------------------------
    @abstractmethod
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a work item according to the agent's specialization.

        This is the primary extension point for specialized agent subclasses.
        Each of the 12 agent types implements this method to handle their
        specific transaction types (e.g., APClerkAgent processes vendor
        invoices, PurchasingAgent creates purchase orders).

        The implementation should:
            1. Inspect work_item.type to determine the transaction type
            2. Use self.make_decision() for decision-making
            3. Use self.action_registry for action execution
            4. Return a WorkResult with success/failure and actions taken

        Args:
            work_item: The WorkItem to process.

        Returns:
            WorkResult indicating success/failure, actions taken, and result data.
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Main Run Loop — README.md lines 812–845
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Main agent loop — continuously process work items from the queue.

        Processing cycle per work item:
            1. Wait for a work item on the asyncio.Queue
            2. Set state to THINKING and record the current work item
            3. Delegate to process_work_item() (implemented by subclass)
            4. Measure processing time and update the WorkResult
            5. Record an observation in agent memory with importance scoring
            6. Update performance metrics (items processed, average time)
            7. Reset state to IDLE and clear current work item
            8. Log structured completion event

        On exception:
            - Log the error with full context (agent_id, error, work_item)
            - Set state to ERROR
            - Increment error counter
            - Clear current work item reference
            - Continue processing (do not crash the loop)

        The loop runs until stop() is called, which sets _running to False.
        """
        self._running = True

        logger.info(
            "agent_run_started",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
        )

        while self._running:
            work_item: Optional[WorkItem] = None
            try:
                # Wait for work item from the async queue
                work_item = await self.work_queue.get()

                # Start timing the processing
                start_time: float = time.monotonic()

                # Transition to THINKING state
                self.state = AgentState.THINKING
                self.current_work_item = work_item

                # Delegate to the specialized subclass implementation
                result: WorkResult = await self.process_work_item(work_item)

                # Measure elapsed processing time
                processing_time: float = time.monotonic() - start_time
                result.processing_time_seconds = processing_time

                # Record observation in agent memory with importance scoring
                importance: float = self._calculate_importance(work_item, result)
                await self.memory.add_observation(
                    content=f"Processed {work_item.type}: {work_item.description}",
                    importance=importance,
                )

                # Update performance metrics
                self.metrics["items_processed"] += 1
                self.metrics["total_processing_time"] += processing_time
                total_items: int = self.metrics["items_processed"]
                self.metrics["average_processing_time"] = (
                    self.metrics["total_processing_time"] / total_items
                )

                # Return to IDLE state
                self.state = AgentState.IDLE
                self.current_work_item = None

                # Log structured completion event (AAP Section 0.7.6)
                logger.info(
                    "agent_work_completed",
                    agent_id=str(self.config.agent_id),
                    role=self.config.role,
                    work_item_type=work_item.type,
                    success=result.success,
                    processing_time=round(processing_time, 4),
                    approval_needed=result.approval_needed,
                    has_exceptions=result.has_exceptions,
                    actions_taken=result.actions_taken,
                    items_processed=self.metrics["items_processed"],
                )

            except Exception as exc:
                # Log the error with full context
                logger.error(
                    "agent_error",
                    agent_id=str(self.config.agent_id),
                    role=self.config.role,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    work_item=work_item.dict() if work_item else None,
                )

                # Transition to ERROR state
                self.state = AgentState.ERROR
                self.metrics["errors"] += 1

                # Clear current work item reference
                self.current_work_item = None

    # ------------------------------------------------------------------
    # Stop — Graceful shutdown of the run loop
    # ------------------------------------------------------------------
    async def stop(self) -> None:
        """Gracefully stop the agent's run loop.

        Sets the internal _running flag to False, which causes the run()
        loop to exit after the current work item (if any) finishes processing.
        """
        self._running = False
        logger.info(
            "agent_stop_requested",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            items_processed=self.metrics["items_processed"],
        )

    # ------------------------------------------------------------------
    # Decision Making — README.md lines 847–871
    # ------------------------------------------------------------------
    async def make_decision(
        self, decision_type: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a decision using the 4-layer hybrid Decision Engine.

        Enriches the provided context with agent-specific information
        (role, personality traits, recent observations from memory) before
        delegating to the DecisionEngine's 4-layer pipeline:
            Layer 1 — Statistical (amounts, entities, timing)
            Layer 2 — LLM (conditional: descriptions, notes, edge cases)
            Layer 3 — Validation (schema, range, business rules; max 3 retries)
            Layer 4 — Deterministic (GL postings, balances, business rules)

        Args:
            decision_type: Type of decision to make (e.g. "process_vendor_invoice",
                           "approve_transaction", "determine_amount").
            context:       Context data dictionary for the decision. Keys vary
                           by decision type. An "agent" key will be added with
                           role, traits, and recent observations.

        Returns:
            Decision result dictionary from the DecisionEngine containing
            merged results from all executed pipeline layers.
        """
        # Set state to THINKING during decision-making
        self.state = AgentState.THINKING

        # Enrich context with agent-specific information
        recent_observations: List[Dict[str, Any]] = await self.memory.get_recent(n=5)
        context["agent"] = {
            "role": self.config.role,
            "traits": self.config.traits,
            "recent_observations": recent_observations,
        }

        # Delegate to the 4-layer Decision Engine
        decision: Dict[str, Any] = await self.decision_engine.decide(
            decision_type=decision_type,
            context=context,
            agent_config=self.config,
        )

        # Track decision metrics
        self.metrics["decisions_made"] += 1

        logger.debug(
            "agent_decision_made",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            decision_type=decision_type,
            decisions_made=self.metrics["decisions_made"],
        )

        return decision

    # ------------------------------------------------------------------
    # Importance Calculation — README.md lines 873–889
    # ------------------------------------------------------------------
    def _calculate_importance(
        self, work_item: WorkItem, result: WorkResult
    ) -> float:
        """Calculate importance score (0.0–10.0) for memory observation.

        The importance score determines how strongly an observation is retained
        in the agent's memory during eviction. Higher importance observations
        survive longer when the observation stream reaches its capacity limit
        (max 1,000 entries).

        Scoring rules:
            - Base importance:                5.0
            - High-value transaction (>$10K): +2.0
            - Processing failure:             +3.0
            - Exceptions/discrepancies:       +2.0
            - Maximum cap:                    10.0

        Args:
            work_item: The processed work item.
            result:    The processing result.

        Returns:
            Importance score clamped to [0.0, 10.0].
        """
        importance: float = 5.0  # Base importance for all observations

        # Increase importance for high-value transactions (> $10,000)
        if work_item.amount is not None and work_item.amount > 10000:
            importance += 2.0

        # Increase importance for processing failures
        if not result.success:
            importance += 3.0

        # Increase importance for exceptions/discrepancies
        if result.has_exceptions:
            importance += 2.0

        # Cap at maximum importance of 10.0
        return min(importance, 10.0)

    # ------------------------------------------------------------------
    # Utility Methods — Metrics, State, and Queue Introspection
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Return a copy of the agent's current performance metrics.

        The returned dictionary is a shallow copy, so modifications to it
        will not affect the agent's internal metrics state.

        Returns:
            Dictionary with keys:
                - items_processed: int — total work items completed
                - decisions_made: int — total decisions via DecisionEngine
                - errors: int — total processing errors encountered
                - average_processing_time: float — mean processing seconds
                - total_processing_time: float — cumulative processing seconds
        """
        return dict(self.metrics)

    def is_idle(self) -> bool:
        """Check whether the agent is currently idle and available for work.

        Returns:
            True if the agent's state is IDLE, False otherwise.
        """
        return self.state == AgentState.IDLE

    def get_queue_size(self) -> int:
        """Return the current number of work items waiting in the agent's queue.

        This is used by the WorkflowOrchestrator for load-balanced agent
        selection (smallest queue depth wins).

        Returns:
            Number of pending work items in the asyncio.Queue.
        """
        return self.work_queue.qsize()
