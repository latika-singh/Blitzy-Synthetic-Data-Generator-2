"""Centralized error handlers for the Agent & Orchestration Engine.

Implements three handler classes covering the three major error domains
of the platform:

* **LLMErrorHandler** — LLM API errors (rate limits, timeouts, auth
  failures) with retry / fallback / escalation strategies.
* **AgentErrorHandler** — Agent work-processing errors with automatic
  retry, skip, and escalation based on error type and retry budget.
* **WorkflowErrorHandler** — Workflow lifecycle errors with retry
  queueing and failure notification.
* **TransactionErrorHandler** — Handles Project 3 transaction processing
  errors with operation-specific retry policies (8 operations), circuit
  breaker patterns (3 domains), and structured fallback behaviors per
  AAP Section 0.7.4.

Additionally defines two custom exception types consumed by the
``AgentErrorHandler``:

* **LLMError** — Raised when an LLM completion fails in a way that the
  ``LLMClient`` cannot internally recover from.
* **DatabaseError** — Raised on persistent Redis / data-store failures
  that the caller cannot retry transparently.

Design Decisions
----------------
- ``from __future__ import annotations`` enables postponed evaluation of
  type hints so that ``TYPE_CHECKING``-guarded imports work correctly
  without runtime circular dependency issues.
- ``anthropic`` exception types are checked with a safe ``try/except
  ImportError`` pattern so the module remains importable even when the
  SDK is not installed (e.g. in minimal test environments).
- All logging uses ``structlog`` to stdout in structured JSON format per
  AAP Section 0.7.6.  No file handlers, no Prometheus / Grafana / APM.
- ``request_context`` is scrubbed of ``api_key`` before logging to
  prevent credential leakage (AAP Section 0.7.4).
- Constructor injection is used for ``WorkflowErrorHandler`` dependencies
  per AAP Section 0.7.1.
- ``TransactionErrorHandler`` implements circuit breakers with three states
  (CLOSED, OPEN, HALF_OPEN) tracked per-domain. Failure thresholds and
  recovery timeouts are sourced from ``app.transactions.constants``.
- P3 exception imports use a guarded ``try/except ImportError`` pattern
  (like the Anthropic SDK guard) so the module stays importable when P3
  modules are not installed yet.

References
----------
- README.md lines 1496-1646 (ERROR HANDLING SPECIFICATION)
- AAP Section 0.5.1 Group 9 (Cross-cutting Error Handling)
- AAP Section 0.7.1 (Constructor injection, EventBus as sole async
  notification mechanism)
- AAP Section 0.7.4 (API key security)
- AAP Section 0.7.6 (structlog logging standard)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

import structlog
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Safe import of anthropic exception types.  When the SDK is not installed
# (e.g. in lightweight test environments) the handler falls through to the
# generic ``else`` branch for any LLM errors.
# ---------------------------------------------------------------------------
try:
    import anthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ANTHROPIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Safe import of P3 transaction exception types.  When the transaction
# modules are not yet installed (e.g. during initial P2 testing), the handler
# falls back to catching the generic Exception type.
# ---------------------------------------------------------------------------
try:
    from app.transactions.exceptions import (
        BalanceError,
        ConcurrencyError,
        DiscrepancyInjectionError,
        GLPostingError,
        PaymentAllocationError,
        PeriodClosedError,
        ReworkLoopError,
        ThreeWayMatchError,
        TransactionError,
        TransactionGenerationError,
    )
    _P3_EXCEPTIONS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _P3_EXCEPTIONS_AVAILABLE = False

# ---------------------------------------------------------------------------
# TYPE_CHECKING-only imports — prevents circular runtime dependencies with
# the agent and orchestration modules that may import from this module.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from app.agents.agent_config import AgentState
    from app.agents.base_agent import BaseAgent, WorkItem
    from app.orchestration.workflow_orchestrator import WorkflowInstance

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Custom Exception Types
# ═══════════════════════════════════════════════════════════════════════════


class LLMError(Exception):
    """LLM-specific error raised when an LLM completion fails irrecoverably.

    This exception is raised by the ``LLMClient`` layer when the underlying
    provider returns an error that cannot be transparently retried (e.g. all
    retries exhausted, invalid response structure).  The ``AgentErrorHandler``
    matches on this type to decide between retry and skip actions.
    """


class DatabaseError(Exception):
    """Database / data-store error for persistent Redis or storage failures.

    Raised when a Redis operation or other data persistence step fails in a
    way that the calling code cannot retry transparently.  The
    ``AgentErrorHandler`` treats this as an escalation-worthy event.
    """


# ═══════════════════════════════════════════════════════════════════════════
# Helper — scrub sensitive fields from request context before logging
# ═══════════════════════════════════════════════════════════════════════════


def _scrub_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *context* with sensitive keys redacted.

    Prevents API keys and other credentials from appearing in structured
    log output (AAP Section 0.7.4).

    Args:
        context: The raw request context dictionary.

    Returns:
        A shallow copy of *context* with values for sensitive keys replaced
        by the string ``"***REDACTED***"``.
    """
    _SENSITIVE_KEYS = frozenset({
        "api_key",
        "apikey",
        "api_secret",
        "authorization",
        "auth_token",
        "token",
        "secret",
        "password",
        "credential",
    })
    scrubbed: Dict[str, Any] = {}
    for key, value in context.items():
        if key.lower() in _SENSITIVE_KEYS:
            scrubbed[key] = "***REDACTED***"
        else:
            scrubbed[key] = value
    return scrubbed


# ═══════════════════════════════════════════════════════════════════════════
# LLMErrorHandler — README.md lines 1502-1551
# ═══════════════════════════════════════════════════════════════════════════


class LLMErrorHandler:
    """Handle LLM-specific errors with retry, fallback, and escalation.

    Categorises incoming exceptions from the Anthropic (or OpenAI) SDK and
    returns a recommended action string that the caller can use to decide
    the next step:

    * ``"retry"``    — Retry the same request (possibly with adjusted
      parameters such as reduced ``max_tokens``).
    * ``"fallback"`` — Switch to the fallback / cheaper model and retry.
    * ``"escalate"`` — The error is unrecoverable; propagate to the caller.

    Error categories (checked in order of specificity, since several
    anthropic exception types inherit from ``anthropic.APIError``):

    1. **Rate-limit errors** (``anthropic.RateLimitError``) — sleep for
       the ``retry_after`` interval (default 60 s) then retry.
    2. **Timeout errors** (``anthropic.APITimeoutError``) — retry up to
       2 times with ``max_tokens`` reduced by 0.7× each time, then
       fall back to the cheaper model.
    3. **Authentication errors** (``anthropic.AuthenticationError``) —
       fatal; escalate immediately.
    4. **Generic API errors** (``anthropic.APIError``) — retry once,
       then fall back to the cheaper model.
    5. **Unknown / non-anthropic errors** — log and escalate.
    """

    async def handle_llm_error(
        self,
        error: Exception,
        request_context: Dict[str, Any],
    ) -> Optional[str]:
        """Handle an LLM error and recommend a recovery action.

        Args:
            error: The exception raised during the LLM completion request.
            request_context: Mutable dictionary carrying request metadata
                (``retry_count``, ``max_tokens``, ``model``, etc.).
                **Note**: ``api_key`` and other credential fields are
                scrubbed from log output automatically.

        Returns:
            One of ``"retry"``, ``"fallback"``, or ``"escalate"``.
        """
        safe_ctx = _scrub_context(request_context)

        # ------------------------------------------------------------------
        # 1. Rate-limit error → wait and retry
        # ------------------------------------------------------------------
        if _ANTHROPIC_AVAILABLE and isinstance(error, anthropic.RateLimitError):
            retry_after: float = float(getattr(error, "retry_after", None) or 60)
            logger.warning(
                "llm_rate_limit",
                retry_after_seconds=retry_after,
                context=safe_ctx,
            )
            await asyncio.sleep(retry_after)
            return "retry"

        # ------------------------------------------------------------------
        # 2. Timeout error → reduce max_tokens and retry up to 2 times
        # ------------------------------------------------------------------
        if _ANTHROPIC_AVAILABLE and isinstance(error, anthropic.APITimeoutError):
            current_retries = request_context.get("retry_count", 0)
            if current_retries < 2:
                new_max_tokens = int(
                    request_context.get("max_tokens", 1000) * 0.7
                )
                request_context["max_tokens"] = new_max_tokens
                logger.warning(
                    "llm_timeout_retry",
                    retry_count=current_retries,
                    new_max_tokens=new_max_tokens,
                    context=safe_ctx,
                )
                return "retry"
            else:
                logger.warning(
                    "llm_timeout_fallback",
                    retry_count=current_retries,
                    context=safe_ctx,
                )
                return "fallback"

        # ------------------------------------------------------------------
        # 3. Authentication error → FATAL, escalate immediately
        # ------------------------------------------------------------------
        if _ANTHROPIC_AVAILABLE and isinstance(
            error, anthropic.AuthenticationError
        ):
            logger.error(
                "llm_auth_error",
                error_type=type(error).__name__,
                error_message=str(error),
                context=safe_ctx,
            )
            return "escalate"

        # ------------------------------------------------------------------
        # 4. Generic API error → retry once, then fallback
        # ------------------------------------------------------------------
        if _ANTHROPIC_AVAILABLE and isinstance(error, anthropic.APIError):
            current_retries = request_context.get("retry_count", 0)
            if current_retries < 1:
                logger.warning(
                    "llm_api_error_retry",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    retry_count=current_retries,
                    context=safe_ctx,
                )
                return "retry"
            else:
                logger.warning(
                    "llm_api_error_fallback",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    retry_count=current_retries,
                    context=safe_ctx,
                )
                return "fallback"

        # ------------------------------------------------------------------
        # 5. Unknown / non-anthropic error → escalate
        # ------------------------------------------------------------------
        logger.error(
            "llm_unknown_error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=safe_ctx,
        )
        return "escalate"


# ═══════════════════════════════════════════════════════════════════════════
# AgentErrorHandler — README.md lines 1556-1608
# ═══════════════════════════════════════════════════════════════════════════


class AgentErrorHandler:
    """Handle agent-specific errors during work item processing.

    Determines the recommended recovery action for a failed work item:

    * ``"skip"``     — Discard the work item (e.g. invalid input).
    * ``"retry"``    — Re-attempt processing (retry budget permitting).
    * ``"escalate"`` — Escalate to the monitoring / alerting layer.

    Error categories (checked in order):

    1. ``pydantic.ValidationError`` — invalid input data → skip.
    2. ``LLMError`` — LLM failure → retry up to 3 times, then skip.
    3. ``DatabaseError`` — data-store failure → escalate immediately.
    4. Unknown errors — retry once, then skip.

    Side effects:
        - Sets ``agent.state`` to ``AgentState.ERROR``.
        - Increments ``agent.metrics["errors"]``.
        - May increment ``work_item.retry_count``.
    """

    async def handle_agent_error(
        self,
        agent: BaseAgent,
        work_item: WorkItem,
        error: Exception,
    ) -> str:
        """Handle an error that occurred while an agent processed a work item.

        Args:
            agent: The agent instance that encountered the error.  Provides
                ``config.agent_id``, ``config.role``, ``state``, and
                ``metrics`` for logging and state mutation.
            work_item: The work item being processed when the error occurred.
                Provides ``type`` and ``retry_count``.
            error: The exception that was raised.

        Returns:
            One of ``"retry"``, ``"skip"``, or ``"escalate"``.
        """
        # Lazy runtime import to avoid circular dependency — AgentState is
        # a lightweight enum and is safe to import at call time.
        from app.agents.agent_config import AgentState  # noqa: WPS433

        # Step 1 — Structured error log (AAP Section 0.7.6)
        logger.error(
            "agent_error",
            agent_id=str(agent.config.agent_id),
            agent_role=agent.config.role,
            work_item_type=work_item.type,
            error=str(error),
            exc_info=True,
        )

        # Step 2 — Update agent state to ERROR
        agent.state = AgentState.ERROR
        agent.metrics["errors"] = agent.metrics.get("errors", 0) + 1

        # Step 3 — Determine action based on error type
        # -----------------------------------------------------------------
        # 3a. ValidationError — invalid input → skip
        # -----------------------------------------------------------------
        if isinstance(error, ValidationError):
            logger.warning(
                "agent_validation_error_skip",
                agent_id=str(agent.config.agent_id),
                work_item_type=work_item.type,
            )
            return "skip"

        # -----------------------------------------------------------------
        # 3b. LLMError — retry up to 3 times, then skip
        # -----------------------------------------------------------------
        if isinstance(error, LLMError):
            if work_item.retry_count < 3:
                work_item.retry_count += 1
                logger.info(
                    "agent_llm_error_retry",
                    agent_id=str(agent.config.agent_id),
                    retry_count=work_item.retry_count,
                )
                return "retry"
            else:
                logger.warning(
                    "agent_llm_error_skip",
                    agent_id=str(agent.config.agent_id),
                    retry_count=work_item.retry_count,
                )
                return "skip"

        # -----------------------------------------------------------------
        # 3c. DatabaseError — escalate immediately
        # -----------------------------------------------------------------
        if isinstance(error, DatabaseError):
            logger.error(
                "agent_database_error_escalate",
                agent_id=str(agent.config.agent_id),
                work_item_type=work_item.type,
            )
            return "escalate"

        # -----------------------------------------------------------------
        # 3d. Unknown error — retry once, then skip
        # -----------------------------------------------------------------
        if work_item.retry_count < 1:
            work_item.retry_count += 1
            logger.info(
                "agent_unknown_error_retry",
                agent_id=str(agent.config.agent_id),
                retry_count=work_item.retry_count,
            )
            return "retry"
        else:
            logger.warning(
                "agent_unknown_error_skip",
                agent_id=str(agent.config.agent_id),
                retry_count=work_item.retry_count,
            )
            return "skip"


# ═══════════════════════════════════════════════════════════════════════════
# WorkflowErrorHandler — README.md lines 1613-1645
# ═══════════════════════════════════════════════════════════════════════════


class WorkflowErrorHandler:
    """Handle workflow-level errors with retry queueing and failure alerts.

    Dependencies are injected via the constructor (AAP Section 0.7.1):

    * ``orchestrator`` — Object with an async ``retry_workflow(workflow_id)``
      method for re-queuing failed workflows.  Typically a
      :class:`~app.orchestration.workflow_orchestrator.WorkflowOrchestrator`.
    * ``monitoring`` — Object with an async ``alert_workflow_failed(workflow)``
      method for failure notifications.

    Recovery logic:

    * If ``workflow.retry_count < 3`` → increment retry counter, set
      status to ``"retrying"``, and re-queue via the orchestrator.
    * Otherwise → set status to ``"failed"`` and notify the monitoring
      layer.
    """

    def __init__(
        self,
        orchestrator: Any = None,
        monitoring: Any = None,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            orchestrator: Orchestrator instance with ``retry_workflow()``
                method.  May be ``None`` during testing.
            monitoring: Monitoring / alerting instance with
                ``alert_workflow_failed()`` method.  May be ``None``
                during testing.
        """
        self.orchestrator = orchestrator
        self.monitoring = monitoring

    async def handle_workflow_error(
        self,
        workflow: WorkflowInstance,
        error: Exception,
    ) -> None:
        """Handle an error at the workflow level.

        Mutates the *workflow* instance in place (sets ``status``,
        ``error_message``, ``error_timestamp``, and ``retry_count``),
        then either re-queues the workflow for retry or marks it as
        permanently failed.

        Args:
            workflow: The ``WorkflowInstance`` that encountered the error.
            error: The exception that was raised.
        """
        # Step 1 — Set error state on the workflow
        workflow.status = "error"
        workflow.error_message = str(error)
        workflow.error_timestamp = datetime.utcnow()

        # Step 2 — Structured error log (AAP Section 0.7.6)
        logger.error(
            "workflow_error",
            workflow_id=str(workflow.workflow_id),
            transaction_type=workflow.transaction_type,
            error=str(error),
            retry_count=workflow.retry_count,
            exc_info=True,
        )

        # Step 3 — Recovery: retry or fail
        if workflow.retry_count < 3:
            workflow.retry_count += 1
            workflow.status = "retrying"
            if self.orchestrator is not None:
                try:
                    await self.orchestrator.retry_workflow(
                        workflow.workflow_id,
                    )
                    logger.info(
                        "workflow_retry_queued",
                        workflow_id=str(workflow.workflow_id),
                        retry_count=workflow.retry_count,
                    )
                except Exception as retry_err:
                    logger.error(
                        "workflow_retry_failed",
                        workflow_id=str(workflow.workflow_id),
                        retry_error=str(retry_err),
                    )
                    workflow.status = "failed"
            else:
                logger.warning(
                    "workflow_retry_no_orchestrator",
                    workflow_id=str(workflow.workflow_id),
                    retry_count=workflow.retry_count,
                )
        else:
            workflow.status = "failed"
            if self.monitoring is not None:
                try:
                    await self.monitoring.alert_workflow_failed(workflow)
                    logger.info(
                        "workflow_failed_alert_sent",
                        workflow_id=str(workflow.workflow_id),
                    )
                except Exception as alert_err:
                    logger.error(
                        "workflow_failed_alert_error",
                        workflow_id=str(workflow.workflow_id),
                        alert_error=str(alert_err),
                    )
            else:
                logger.warning(
                    "workflow_failed_no_monitoring",
                    workflow_id=str(workflow.workflow_id),
                    retry_count=workflow.retry_count,
                )


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker — P3 Transaction Error Domains
# ═══════════════════════════════════════════════════════════════════════════


class CircuitBreakerState(Enum):
    """Circuit breaker states for P3 transaction error domains."""

    CLOSED = "closed"       # Normal operation — requests pass through
    OPEN = "open"           # Failures exceeded threshold — requests blocked
    HALF_OPEN = "half_open"  # Recovery timeout elapsed — allowing test request


class CircuitBreaker:
    """Simple circuit breaker implementation for transaction error domains.

    Tracks failure counts and transitions between CLOSED, OPEN, and HALF_OPEN
    states based on configured thresholds and recovery timeouts.

    Three circuit breakers are configured per AAP Section 0.1.2:
    - GL posting: 10 failures → OPEN, 60s recovery
    - Discrepancy injection: 20 failures → OPEN, 30s recovery
    - Rework loop: 50 failures → OPEN, 120s recovery
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        recovery_timeout_seconds: float,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self._state = CircuitBreakerState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitBreakerState:
        """Current circuit breaker state, considering recovery timeout."""
        if self._state == CircuitBreakerState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout_seconds:
                self._state = CircuitBreakerState.HALF_OPEN
        return self._state

    @property
    def is_open(self) -> bool:
        """True if the circuit is OPEN (blocking requests)."""
        return self.state == CircuitBreakerState.OPEN

    def record_success(self) -> None:
        """Record a successful operation — reset failure count and close circuit."""
        self._failure_count = 0
        self._state = CircuitBreakerState.CLOSED
        logger.debug(
            "circuit_breaker_success",
            breaker_name=self.name,
            state=self._state.value,
        )

    def record_failure(self) -> None:
        """Record a failed operation — increment count and potentially open circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                breaker_name=self.name,
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
                recovery_timeout_seconds=self.recovery_timeout_seconds,
            )
        else:
            logger.debug(
                "circuit_breaker_failure_recorded",
                breaker_name=self.name,
                failure_count=self._failure_count,
                threshold=self.failure_threshold,
            )

    def reset(self) -> None:
        """Reset the circuit breaker to initial CLOSED state."""
        self._failure_count = 0
        self._state = CircuitBreakerState.CLOSED
        self._last_failure_time = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TransactionErrorHandler — Project 3 Transaction Workflows
# ═══════════════════════════════════════════════════════════════════════════


class TransactionErrorHandler:
    """Handle Project 3 transaction processing errors with operation-specific
    retry policies, circuit breaker patterns, and structured fallback behaviors.

    Implements all 8 retry policies from AAP Section 0.1.2:

    +--------------------------+--------+------------------+---------+------------------------------+
    | Operation                | Retries| Backoff          | Timeout | Fallback                     |
    +--------------------------+--------+------------------+---------+------------------------------+
    | P2P cycle generation     | 2      | Linear (1s, 2s)  | 60s     | Skip transaction             |
    | O2C cycle generation     | 2      | Linear (1s, 2s)  | 60s     | Skip transaction             |
    | GL posting               | 3      | Exp (1s, 2s, 4s) | 30s     | Rollback transaction         |
    | Three-way matching       | 2      | Linear (500ms,1s) | 10s     | Mark as exception            |
    | Discrepancy injection    | 1      | None             | 5s      | Skip discrepancy             |
    | Rework loop fix          | 3      | Linear (1,2,3s)  | 30s     | Escalate to admin            |
    | Period close             | 1      | None             | 300s    | Halt, require manual interv. |
    | Balance update           | 2      | Linear (500ms,1s) | 10s     | Rollback transaction         |
    +--------------------------+--------+------------------+---------+------------------------------+

    Implements 3 circuit breakers per AAP Section 0.1.2:
    - GL posting: 10 failures → OPEN, 60s recovery
    - Discrepancy injection: 20 failures → OPEN, 30s recovery
    - Rework loop: 50 failures → OPEN, 120s recovery

    Dependencies are injected via the constructor (AAP Section 0.7.1).
    All parameters are Optional with None defaults.
    """

    def __init__(
        self,
        *,
        monitoring: Any = None,
    ) -> None:
        """Initialise with injected dependencies.

        Args:
            monitoring: Optional monitoring/alerting instance with
                ``alert_transaction_failed()`` method.
        """
        self.monitoring = monitoring

        # Initialize circuit breakers per AAP Section 0.1.2
        self._circuit_breakers: Dict[str, CircuitBreaker] = {
            "gl_posting": CircuitBreaker(
                name="gl_posting",
                failure_threshold=10,
                recovery_timeout_seconds=60.0,
            ),
            "discrepancy_injection": CircuitBreaker(
                name="discrepancy_injection",
                failure_threshold=20,
                recovery_timeout_seconds=30.0,
            ),
            "rework_loop": CircuitBreaker(
                name="rework_loop",
                failure_threshold=50,
                recovery_timeout_seconds=120.0,
            ),
        }

        # Retry policies — matches AAP Section 0.1.2 Retry Policy Table EXACTLY
        self._retry_policies: Dict[str, Dict[str, Any]] = {
            "p2p_cycle_generation": {
                "max_attempts": 2,
                "backoff_strategy": "linear",
                "backoff_seconds": [1.0, 2.0],
                "timeout_seconds": 60.0,
                "fallback": "skip_transaction",
            },
            "o2c_cycle_generation": {
                "max_attempts": 2,
                "backoff_strategy": "linear",
                "backoff_seconds": [1.0, 2.0],
                "timeout_seconds": 60.0,
                "fallback": "skip_transaction",
            },
            "gl_posting": {
                "max_attempts": 3,
                "backoff_strategy": "exponential",
                "backoff_seconds": [1.0, 2.0, 4.0],
                "timeout_seconds": 30.0,
                "fallback": "rollback_transaction",
            },
            "three_way_matching": {
                "max_attempts": 2,
                "backoff_strategy": "linear",
                "backoff_seconds": [0.5, 1.0],
                "timeout_seconds": 10.0,
                "fallback": "mark_as_exception",
            },
            "discrepancy_injection": {
                "max_attempts": 1,
                "backoff_strategy": "none",
                "backoff_seconds": [],
                "timeout_seconds": 5.0,
                "fallback": "skip_discrepancy",
            },
            "rework_loop_fix": {
                "max_attempts": 3,
                "backoff_strategy": "linear",
                "backoff_seconds": [1.0, 2.0, 3.0],
                "timeout_seconds": 30.0,
                "fallback": "escalate_to_admin",
            },
            "period_close": {
                "max_attempts": 1,
                "backoff_strategy": "none",
                "backoff_seconds": [],
                "timeout_seconds": 300.0,
                "fallback": "halt_manual_intervention",
            },
            "balance_update": {
                "max_attempts": 2,
                "backoff_strategy": "linear",
                "backoff_seconds": [0.5, 1.0],
                "timeout_seconds": 10.0,
                "fallback": "rollback_transaction",
            },
        }

    def get_circuit_breaker(self, domain: str) -> Optional[CircuitBreaker]:
        """Get the circuit breaker for a given domain.

        Args:
            domain: One of 'gl_posting', 'discrepancy_injection', 'rework_loop'.

        Returns:
            The CircuitBreaker instance, or None if the domain has no circuit
            breaker.
        """
        return self._circuit_breakers.get(domain)

    def get_retry_policy(self, operation: str) -> Dict[str, Any]:
        """Get the retry policy for a given operation.

        Args:
            operation: Operation name from the retry policy table.

        Returns:
            Dict with max_attempts, backoff_strategy, backoff_seconds,
            timeout_seconds, and fallback keys.

        Raises:
            KeyError: If the operation is not in the retry policy table.
        """
        if operation not in self._retry_policies:
            raise KeyError(
                f"Unknown operation '{operation}'. "
                f"Valid operations: {sorted(self._retry_policies.keys())}"
            )
        return self._retry_policies[operation]

    async def handle_transaction_error(
        self,
        operation: str,
        error: Exception,
        *,
        retry_count: int = 0,
        trace_id: Optional[str] = None,
        simulation_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Handle a transaction processing error and return the recommended action.

        Args:
            operation: One of the 8 operation types from the retry policy table:
                'p2p_cycle_generation', 'o2c_cycle_generation', 'gl_posting',
                'three_way_matching', 'discrepancy_injection', 'rework_loop_fix',
                'period_close', 'balance_update'.
            error: The exception that occurred.
            retry_count: Number of retry attempts already made for this
                operation.
            trace_id: Distributed trace identifier for structured logging.
            simulation_id: Simulation run identifier for structured logging.
            transaction_id: Transaction identifier for structured logging.
            context: Optional additional context dictionary (will be scrubbed).

        Returns:
            One of: 'retry', 'skip', 'rollback', 'mark_exception',
            'escalate', 'halt'.
        """
        safe_ctx = _scrub_context(context or {})
        policy = self._retry_policies.get(operation)

        if policy is None:
            logger.error(
                "transaction_error_unknown_operation",
                operation=operation,
                error_type=type(error).__name__,
                error_message=str(error),
                trace_id=trace_id,
                simulation_id=simulation_id,
                service_name="transactions",
                component="TransactionErrorHandler",
            )
            return "escalate"

        # --- Step 1: Check circuit breaker for domains that have one ---
        breaker = self._circuit_breakers.get(operation)
        if breaker is not None and breaker.is_open:
            logger.warning(
                "transaction_error_circuit_open",
                operation=operation,
                breaker_name=breaker.name,
                breaker_state=breaker.state.value,
                fallback=policy["fallback"],
                trace_id=trace_id,
                simulation_id=simulation_id,
                transaction_id=transaction_id,
                service_name="transactions",
                component="TransactionErrorHandler",
            )
            return self._fallback_to_action(policy["fallback"])

        # --- Step 2: Structured error log (AAP Section 0.7.6/0.7.7) ---
        logger.error(
            "transaction_error",
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            retry_count=retry_count,
            max_attempts=policy["max_attempts"],
            timeout_seconds=policy["timeout_seconds"],
            fallback=policy["fallback"],
            trace_id=trace_id,
            simulation_id=simulation_id,
            transaction_id=transaction_id,
            context=safe_ctx,
            service_name="transactions",
            component="TransactionErrorHandler",
            exc_info=True,
        )

        # --- Step 3: Record failure on circuit breaker if applicable ---
        if breaker is not None:
            breaker.record_failure()

        # --- Step 4: Check if retry budget remains ---
        if retry_count < policy["max_attempts"]:
            # Compute backoff delay
            backoff_seconds = policy["backoff_seconds"]
            if backoff_seconds and retry_count < len(backoff_seconds):
                delay = backoff_seconds[retry_count]
            else:
                delay = 0.0

            if delay > 0:
                logger.info(
                    "transaction_error_retry_backoff",
                    operation=operation,
                    retry_count=retry_count,
                    backoff_seconds=delay,
                    trace_id=trace_id,
                    simulation_id=simulation_id,
                    transaction_id=transaction_id,
                    service_name="transactions",
                    component="TransactionErrorHandler",
                )
                await asyncio.sleep(delay)

            return "retry"

        # --- Step 5: Retry budget exhausted — apply fallback ---
        logger.warning(
            "transaction_error_retries_exhausted",
            operation=operation,
            retry_count=retry_count,
            fallback=policy["fallback"],
            trace_id=trace_id,
            simulation_id=simulation_id,
            transaction_id=transaction_id,
            service_name="transactions",
            component="TransactionErrorHandler",
        )

        # Notify monitoring if available
        if self.monitoring is not None:
            try:
                await self.monitoring.alert_transaction_failed(
                    operation=operation,
                    error=error,
                    trace_id=trace_id,
                    simulation_id=simulation_id,
                    transaction_id=transaction_id,
                )
            except Exception as alert_err:
                logger.error(
                    "transaction_error_alert_failed",
                    operation=operation,
                    alert_error=str(alert_err),
                    trace_id=trace_id,
                    simulation_id=simulation_id,
                    service_name="transactions",
                    component="TransactionErrorHandler",
                )

        return self._fallback_to_action(policy["fallback"])

    @staticmethod
    def _fallback_to_action(fallback: str) -> str:
        """Convert a fallback policy name to a caller-actionable return string.

        Mapping:
            skip_transaction       → 'skip'
            rollback_transaction   → 'rollback'
            mark_as_exception      → 'mark_exception'
            skip_discrepancy       → 'skip'
            escalate_to_admin      → 'escalate'
            halt_manual_intervention → 'halt'

        Args:
            fallback: Fallback policy string from the retry policy table.

        Returns:
            Normalized action string.
        """
        _FALLBACK_MAP: Dict[str, str] = {
            "skip_transaction": "skip",
            "rollback_transaction": "rollback",
            "mark_as_exception": "mark_exception",
            "skip_discrepancy": "skip",
            "escalate_to_admin": "escalate",
            "halt_manual_intervention": "halt",
        }
        return _FALLBACK_MAP.get(fallback, "escalate")


# ═══════════════════════════════════════════════════════════════════════════
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "LLMErrorHandler",
    "AgentErrorHandler",
    "WorkflowErrorHandler",
    "TransactionErrorHandler",
    "CircuitBreaker",
    "CircuitBreakerState",
    "LLMError",
    "DatabaseError",
]
