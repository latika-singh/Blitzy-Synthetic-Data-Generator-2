"""
Agent Registry and Timeout Manager — Centralized agent lifecycle management.

This module provides the ``AgentRegistry`` for managing the lifecycle of all
agent instances, and the ``AgentTimeoutManager`` for monitoring and escalating
consecutive timeout events on a per-agent basis.

Exports:
    AGENT_TIMEOUT_ESCALATION — Configuration dict for timeout escalation policy.
    MAX_AGENTS               — Hard cap on simultaneous registered agents (50).
    AgentTimeoutManager      — Per-agent timeout tracking with auto-restart.
    AgentRegistryConfig      — Pydantic V2 config model for registry settings.
    AgentRegistry            — Central registry for agent lookup, availability,
                               capacity monitoring, and Redis state persistence.

Design Decisions:
    - Dict-based role indexing (``_agents_by_role``) provides O(1) role lookup,
      meeting the <500 ms workflow routing SLA (AAP Section 0.7.3).
    - ``register_agent`` enforces the ``max_agents`` cap (default 50) to stay
      within the 35 MB Redis memory budget (50 agents × ~700 KB).
    - All I/O-bound methods (register, deregister, update_agent_state) are async
      to support Redis persistence without blocking the event loop.
    - Lightweight construction — no I/O in ``__init__`` — supports ≥50 agents/sec
      creation rate (AAP Section 0.7.3).
    - ``AgentTimeoutManager`` uses a 3-consecutive-timeout threshold with a 60 s
      cooldown before auto-restart, per README.md lines 1781-1833.
    - Structured JSON logging via ``structlog`` to stdout only (AAP Section 0.7.6).
    - Pydantic V2 ``BaseModel`` for ``AgentRegistryConfig`` at the subsystem
      boundary (AAP Section 0.7.1).
    - Constructor injection for the optional ``redis_client`` (AAP Section 0.7.1).

Redis Integration:
    Key pattern : ``agent:{agent_id}:state``
    TTL         : 24 hours (86 400 seconds)
    Value       : JSON-serialised agent state snapshot

Performance Targets:
    - Agent creation rate: ≥ 50 agents/second
    - Workflow routing (role lookup): < 500 milliseconds
    - Concurrent agents: ≥ 20

References:
    - README.md lines 1781-1833 (timeout escalation)
    - README.md lines 1270-1377 (WorkflowOrchestrator integration)
    - AAP Section 0.4.1 (Redis key patterns and TTLs)
    - AAP Section 0.5.1 Group 5 (AgentRegistry specification)
"""

import asyncio
import json
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional, Dict, List, Any

from pydantic import BaseModel, Field
import structlog

from app.agents.agent_config import AgentConfig, AgentState
from app.agents.base_agent import BaseAgent

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants — Timeout Escalation Policy (README.md lines 1781-1833)
# ---------------------------------------------------------------------------
AGENT_TIMEOUT_ESCALATION: Dict[str, Any] = {
    "consecutive_timeouts_threshold": 3,
    "escalation_action": "notify_supervisor",
    "auto_restart_agent": True,
    "cooldown_period_seconds": 60,
}
"""Timeout escalation configuration.

Keys:
    consecutive_timeouts_threshold (int):
        Number of consecutive timeouts before escalation triggers. Default 3.
    escalation_action (str):
        Action to take on escalation. Currently ``"notify_supervisor"``.
    auto_restart_agent (bool):
        Whether to automatically restart the agent after cooldown.
    cooldown_period_seconds (int):
        Seconds to wait before attempting an agent restart. Default 60.
"""

MAX_AGENTS: int = 50
"""Hard upper limit on simultaneously registered agents.

Derived from AAP Section 0.1.2 constraint: max 50 agents × ~700 KB ≈ 35 MB
total agent memory in Redis. Enforced in ``AgentRegistry.register_agent()``.
"""


# ---------------------------------------------------------------------------
# AgentTimeoutManager — Per-agent timeout tracking and escalation
# ---------------------------------------------------------------------------
class AgentTimeoutManager:
    """Manages consecutive timeout tracking and escalation for a single agent.

    Each agent gets its own ``AgentTimeoutManager`` instance, created
    automatically when the agent is registered with the ``AgentRegistry``.

    Escalation Behaviour:
        1. Each timeout increments ``consecutive_timeouts``.
        2. When ``consecutive_timeouts`` reaches the configured threshold (3),
           ``_escalate()`` is called.
        3. Escalation logs a critical event and, if ``auto_restart_agent`` is
           True, waits ``cooldown_period_seconds`` (60 s) then calls
           ``_restart_agent()``.
        4. On any successful operation, the caller should invoke ``reset()`` to
           clear the consecutive counter.

    Attributes:
        agent_id:             UUID of the managed agent.
        consecutive_timeouts: Running count of timeouts without a success reset.
        last_timeout:         Timestamp of the most recent timeout event.
        total_timeouts:       Lifetime count of all timeout events for this agent.
    """

    __slots__ = ("agent_id", "consecutive_timeouts", "last_timeout", "total_timeouts")

    def __init__(self, agent_id: UUID) -> None:
        """Initialise the timeout manager for a specific agent.

        Args:
            agent_id: Unique identifier of the agent to monitor.
        """
        self.agent_id: UUID = agent_id
        self.consecutive_timeouts: int = 0
        self.last_timeout: Optional[datetime] = None
        self.total_timeouts: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_timeout(self, operation: str, duration: float) -> None:
        """Record a timeout event and escalate if the threshold is reached.

        Increments both ``consecutive_timeouts`` and ``total_timeouts``,
        updates the ``last_timeout`` timestamp, and triggers escalation when
        the consecutive count reaches the configured threshold.

        Args:
            operation: Human-readable name of the operation that timed out
                       (e.g. ``"process_work_item"``, ``"llm_completion"``).
            duration:  Elapsed wall-clock seconds before the timeout occurred.
        """
        self.consecutive_timeouts += 1
        self.total_timeouts += 1
        self.last_timeout = datetime.now()

        logger.error(
            "agent_timeout",
            agent_id=str(self.agent_id),
            operation=operation,
            duration_seconds=round(duration, 4),
            consecutive_timeouts=self.consecutive_timeouts,
            total_timeouts=self.total_timeouts,
        )

        threshold: int = AGENT_TIMEOUT_ESCALATION["consecutive_timeouts_threshold"]
        if self.consecutive_timeouts >= threshold:
            await self._escalate()

    async def _escalate(self) -> None:
        """Escalate after reaching the consecutive timeout threshold.

        Logs a critical-level event and, when ``auto_restart_agent`` is
        enabled, sleeps for the configured cooldown period before invoking
        ``_restart_agent()``.
        """
        logger.critical(
            "agent_timeout_escalation",
            agent_id=str(self.agent_id),
            consecutive_timeouts=self.consecutive_timeouts,
            escalation_action=AGENT_TIMEOUT_ESCALATION["escalation_action"],
            auto_restart=AGENT_TIMEOUT_ESCALATION["auto_restart_agent"],
        )

        if AGENT_TIMEOUT_ESCALATION["auto_restart_agent"]:
            cooldown: int = AGENT_TIMEOUT_ESCALATION["cooldown_period_seconds"]
            logger.info(
                "agent_cooldown_started",
                agent_id=str(self.agent_id),
                cooldown_seconds=cooldown,
            )
            await asyncio.sleep(cooldown)
            await self._restart_agent()

    async def _restart_agent(self) -> None:
        """Signal that the agent should be restarted.

        Resets the consecutive timeout counter and logs the restart event.
        The actual agent restart orchestration (stopping the old run-loop and
        spawning a new one) is handled by the ``AgentRegistry`` that owns
        the agent — this method simply resets local tracking state.
        """
        logger.info(
            "agent_restart",
            agent_id=str(self.agent_id),
            previous_consecutive_timeouts=self.consecutive_timeouts,
        )
        self.consecutive_timeouts = 0

    def reset(self) -> None:
        """Reset the consecutive timeout counter on a successful operation.

        Should be called by the agent framework whenever a work item or
        decision completes successfully, so that isolated timeouts do not
        accumulate across unrelated operations.
        """
        self.consecutive_timeouts = 0


# ---------------------------------------------------------------------------
# AgentRegistryConfig — Pydantic V2 configuration model
# ---------------------------------------------------------------------------
class AgentRegistryConfig(BaseModel):
    """Configuration for the ``AgentRegistry``.

    Pydantic V2 ``BaseModel`` at the subsystem boundary (AAP Section 0.7.1).

    Attributes:
        max_agents:               Maximum number of simultaneously registered
                                  agents. Default 50, valid range [1, 200].
        state_ttl_seconds:        Time-to-live in seconds for Redis agent state
                                  keys. Default 86 400 (24 hours).
        enable_redis_persistence: Whether to persist agent state to Redis.
        redis_key_prefix:         Prefix for Redis keys (``"agent"`` →
                                  ``agent:{id}:state``).
    """

    max_agents: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of simultaneously registered agents.",
    )
    state_ttl_seconds: int = Field(
        default=86400,
        ge=60,
        description="TTL in seconds for Redis agent-state keys (default 24 h).",
    )
    enable_redis_persistence: bool = Field(
        default=True,
        description="Enable/disable Redis persistence of agent state.",
    )
    redis_key_prefix: str = Field(
        default="agent",
        description="Redis key prefix (e.g. 'agent' → 'agent:{id}:state').",
    )


# ---------------------------------------------------------------------------
# AgentRegistry — Central agent lifecycle management
# ---------------------------------------------------------------------------
class AgentRegistry:
    """Centralized registry for agent lifecycle, lookup, and monitoring.

    Responsibilities:
        - **Register / deregister** agents and enforce the ``max_agents`` cap.
        - **Role-based lookup** via O(1) dict indexing (``get_agents_by_roles``).
        - **Availability tracking** — filter agents by ``AgentState.IDLE``.
        - **Capacity monitoring** — queue depths, utilization, error rates.
        - **Redis persistence** of agent state snapshots with configurable TTL.
        - **Timeout management** via per-agent ``AgentTimeoutManager`` instances.

    Used by the ``WorkflowOrchestrator`` to find idle agents for a set of
    roles and select the one with the smallest work queue:

        >>> candidates = registry.get_idle_agents(roles=["ap_clerk", "ap_manager"])
        >>> best = registry.get_agent_with_smallest_queue(roles=["ap_clerk"])

    Constructor Injection (AAP Section 0.7.1):
        ``config``       — ``AgentRegistryConfig`` (optional, defaults apply).
        ``redis_client`` — An async Redis client instance (optional).

    Attributes:
        config: Registry configuration.
    """

    def __init__(
        self,
        config: Optional[AgentRegistryConfig] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        """Initialise the agent registry.

        No I/O is performed during construction to preserve the ≥50 agents/sec
        creation target.

        Args:
            config:       Registry configuration. Falls back to defaults if
                          ``None``.
            redis_client: Optional async Redis client for state persistence.
                          When ``None``, Redis persistence is silently skipped.
        """
        self.config: AgentRegistryConfig = config or AgentRegistryConfig()

        # Primary storage — agent_id → BaseAgent instance
        self._agents: Dict[UUID, BaseAgent] = {}

        # Secondary index — role string → list of agent UUIDs (O(1) role lookup)
        self._agents_by_role: Dict[str, List[UUID]] = {}

        # Per-agent timeout managers
        self._timeout_managers: Dict[UUID, AgentTimeoutManager] = {}

        # Optional Redis client (may be None in test or standalone mode)
        self._redis: Optional[Any] = redis_client

        logger.info(
            "agent_registry_initialized",
            max_agents=self.config.max_agents,
            redis_enabled=self._redis is not None,
            redis_key_prefix=self.config.redis_key_prefix,
        )

    # ------------------------------------------------------------------
    # Registration / Deregistration
    # ------------------------------------------------------------------

    async def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the registry.

        Enforces the ``max_agents`` cap, indexes the agent by ID and role,
        creates a ``AgentTimeoutManager``, and optionally persists the initial
        state to Redis.

        Args:
            agent: The ``BaseAgent`` (or subclass) instance to register.

        Raises:
            ValueError: If the registry has already reached ``max_agents``.
            ValueError: If an agent with the same ``agent_id`` is already
                        registered.
        """
        agent_id: UUID = agent.config.agent_id
        role: str = agent.config.role

        # Enforce capacity limit
        if len(self._agents) >= self.config.max_agents:
            raise ValueError(
                f"Cannot register agent {agent_id}: registry is at capacity "
                f"({self.config.max_agents}/{self.config.max_agents})."
            )

        # Prevent duplicate registration
        if agent_id in self._agents:
            raise ValueError(
                f"Agent {agent_id} is already registered. "
                "Deregister it first if you need to re-register."
            )

        # Store in primary index
        self._agents[agent_id] = agent

        # Store in role index
        if role not in self._agents_by_role:
            self._agents_by_role[role] = []
        self._agents_by_role[role].append(agent_id)

        # Create timeout manager
        self._timeout_managers[agent_id] = AgentTimeoutManager(agent_id)

        # Persist to Redis if enabled
        if self._redis is not None and self.config.enable_redis_persistence:
            await self._persist_agent_state(agent)

        logger.info(
            "agent_registered",
            agent_id=str(agent_id),
            role=role,
            total_agents=len(self._agents),
        )

    async def deregister_agent(self, agent_id: UUID) -> None:
        """Remove an agent from the registry.

        Cleans up all indexes (primary, role, timeout manager) and deletes
        the corresponding Redis state key if persistence is enabled.

        Args:
            agent_id: UUID of the agent to remove.

        Raises:
            KeyError: If the ``agent_id`` is not found in the registry.
        """
        if agent_id not in self._agents:
            raise KeyError(f"Agent {agent_id} is not registered.")

        agent: BaseAgent = self._agents[agent_id]
        role: str = agent.config.role

        # Remove from primary index
        del self._agents[agent_id]

        # Remove from role index
        if role in self._agents_by_role:
            try:
                self._agents_by_role[role].remove(agent_id)
            except ValueError:
                pass  # Already removed — defensive cleanup
            # Clean up empty role lists to prevent memory leaks
            if not self._agents_by_role[role]:
                del self._agents_by_role[role]

        # Remove timeout manager
        self._timeout_managers.pop(agent_id, None)

        # Remove from Redis if enabled
        if self._redis is not None and self.config.enable_redis_persistence:
            redis_key = (
                f"{self.config.redis_key_prefix}:{agent_id}:state"
            )
            try:
                await self._redis.delete(redis_key)
            except Exception as exc:
                logger.warning(
                    "redis_delete_failed",
                    agent_id=str(agent_id),
                    redis_key=redis_key,
                    error=str(exc),
                )

        logger.info(
            "agent_deregistered",
            agent_id=str(agent_id),
            role=role,
            remaining_agents=len(self._agents),
        )

    # ------------------------------------------------------------------
    # Lookup Methods
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: UUID) -> Optional[BaseAgent]:
        """Retrieve a registered agent by its UUID.

        Args:
            agent_id: The unique identifier of the agent.

        Returns:
            The ``BaseAgent`` instance, or ``None`` if not found.
        """
        return self._agents.get(agent_id)

    def get_agents_by_roles(self, roles: List[str]) -> List[BaseAgent]:
        """Retrieve all agents matching any of the given roles.

        Uses the pre-built ``_agents_by_role`` dict for O(1) lookup per role,
        meeting the <500 ms workflow routing SLA (AAP Section 0.7.3).

        Args:
            roles: List of role identifiers (e.g. ``["ap_clerk", "ap_manager"]``).

        Returns:
            List of matching ``BaseAgent`` instances (may be empty).
        """
        result: List[BaseAgent] = []
        seen: set = set()
        for role in roles:
            for aid in self._agents_by_role.get(role, []):
                if aid not in seen:
                    seen.add(aid)
                    agent = self._agents.get(aid)
                    if agent is not None:
                        result.append(agent)
        return result

    def get_idle_agents(
        self, roles: Optional[List[str]] = None
    ) -> List[BaseAgent]:
        """Retrieve agents in the ``IDLE`` state, optionally filtered by role.

        Args:
            roles: If provided, only return idle agents whose role is in
                   this list. If ``None``, return all idle agents.

        Returns:
            List of idle ``BaseAgent`` instances.
        """
        if roles is not None:
            candidates = self.get_agents_by_roles(roles)
        else:
            candidates = list(self._agents.values())

        return [
            agent for agent in candidates
            if agent.state == AgentState.IDLE
        ]

    def get_agent_with_smallest_queue(
        self, roles: List[str]
    ) -> Optional[BaseAgent]:
        """Select the idle agent with the smallest work queue for the given roles.

        This is the primary agent selection method used by the
        ``WorkflowOrchestrator`` (README.md lines 1270-1377). If multiple
        agents have the same queue size the first encountered is returned.

        Args:
            roles: List of acceptable role identifiers.

        Returns:
            The idle ``BaseAgent`` with the smallest queue, or ``None`` if no
            idle agents are available for the requested roles.
        """
        idle_agents: List[BaseAgent] = self.get_idle_agents(roles=roles)
        if not idle_agents:
            return None
        return min(idle_agents, key=lambda a: a.work_queue.qsize())

    def get_all_agents(self) -> List[BaseAgent]:
        """Return a list of all currently registered agents.

        Returns:
            List of all ``BaseAgent`` instances in registration order.
        """
        return list(self._agents.values())

    # ------------------------------------------------------------------
    # Capacity and Metrics
    # ------------------------------------------------------------------

    def get_agent_count(self) -> int:
        """Return the current number of registered agents.

        Returns:
            Integer count of agents in the registry.
        """
        return len(self._agents)

    def get_capacity_metrics(self) -> Dict[str, Any]:
        """Compute a snapshot of registry capacity and utilisation metrics.

        Returns:
            Dictionary with the following keys:

            - ``total_agents``      — number of registered agents.
            - ``max_agents``        — configured agent cap.
            - ``idle_count``        — agents in ``IDLE`` state.
            - ``busy_count``        — agents in ``THINKING`` or ``ACTING`` state.
            - ``waiting_count``     — agents in ``WAITING`` state.
            - ``error_count``       — agents in ``ERROR`` state.
            - ``avg_queue_depth``   — mean work-queue size across all agents.
            - ``total_queue_depth`` — sum of all work-queue sizes.
            - ``role_distribution`` — dict mapping role → agent count.
            - ``utilization_pct``   — percentage of non-idle agents.
        """
        total: int = len(self._agents)
        idle_count: int = 0
        busy_count: int = 0
        waiting_count: int = 0
        error_count: int = 0
        total_queue_depth: int = 0

        for agent in self._agents.values():
            state = agent.state
            if state == AgentState.IDLE:
                idle_count += 1
            elif state in (AgentState.THINKING, AgentState.ACTING):
                busy_count += 1
            elif state == AgentState.WAITING:
                waiting_count += 1
            elif state == AgentState.ERROR:
                error_count += 1
            total_queue_depth += agent.work_queue.qsize()

        avg_queue_depth: float = (
            total_queue_depth / total if total > 0 else 0.0
        )

        role_distribution: Dict[str, int] = {
            role: len(ids) for role, ids in self._agents_by_role.items()
        }

        utilization_pct: float = (
            ((total - idle_count) / total * 100.0) if total > 0 else 0.0
        )

        return {
            "total_agents": total,
            "max_agents": self.config.max_agents,
            "idle_count": idle_count,
            "busy_count": busy_count,
            "waiting_count": waiting_count,
            "error_count": error_count,
            "avg_queue_depth": round(avg_queue_depth, 2),
            "total_queue_depth": total_queue_depth,
            "role_distribution": role_distribution,
            "utilization_pct": round(utilization_pct, 2),
        }

    # ------------------------------------------------------------------
    # Timeout Manager Access
    # ------------------------------------------------------------------

    def get_timeout_manager(
        self, agent_id: UUID
    ) -> Optional[AgentTimeoutManager]:
        """Retrieve the timeout manager for a specific agent.

        Args:
            agent_id: UUID of the agent.

        Returns:
            The ``AgentTimeoutManager`` instance, or ``None`` if the agent
            is not registered.
        """
        return self._timeout_managers.get(agent_id)

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    async def update_agent_state(
        self, agent_id: UUID, new_state: AgentState
    ) -> None:
        """Update an agent's state and optionally persist to Redis.

        Args:
            agent_id:  UUID of the agent whose state should change.
            new_state: The target ``AgentState`` value.

        Raises:
            KeyError: If the ``agent_id`` is not found in the registry.
        """
        agent: Optional[BaseAgent] = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Agent {agent_id} is not registered.")

        old_state: AgentState = agent.state
        agent.state = new_state

        logger.info(
            "agent_state_updated",
            agent_id=str(agent_id),
            role=agent.config.role,
            old_state=old_state.value if isinstance(old_state, AgentState) else str(old_state),
            new_state=new_state.value if isinstance(new_state, AgentState) else str(new_state),
        )

        # Persist updated state to Redis
        if self._redis is not None and self.config.enable_redis_persistence:
            await self._persist_agent_state(agent)

    # ------------------------------------------------------------------
    # Redis Persistence (Internal)
    # ------------------------------------------------------------------

    async def _persist_agent_state(self, agent: BaseAgent) -> None:
        """Serialise and persist the agent's state snapshot to Redis.

        Writes a JSON blob to ``{prefix}:{agent_id}:state`` with the
        configured TTL (default 24 hours / 86 400 seconds).

        Args:
            agent: The agent whose state to persist.
        """
        if self._redis is None or not self.config.enable_redis_persistence:
            return

        agent_id: UUID = agent.config.agent_id
        redis_key: str = f"{self.config.redis_key_prefix}:{agent_id}:state"

        state_snapshot: Dict[str, Any] = {
            "agent_id": str(agent_id),
            "role": agent.config.role,
            "state": agent.state.value if isinstance(agent.state, AgentState) else str(agent.state),
            "queue_size": agent.work_queue.qsize(),
            "metrics": agent.metrics,
            "persisted_at": datetime.now().isoformat(),
        }

        try:
            payload: str = json.dumps(state_snapshot)
            await self._redis.set(
                redis_key,
                payload,
                ex=self.config.state_ttl_seconds,
            )
            logger.debug(
                "agent_state_persisted",
                agent_id=str(agent_id),
                redis_key=redis_key,
                ttl_seconds=self.config.state_ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "redis_persist_failed",
                agent_id=str(agent_id),
                redis_key=redis_key,
                error=str(exc),
            )
