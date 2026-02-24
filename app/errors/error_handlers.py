"""Centralized error handlers for the Agent & Orchestration Engine.

Implements three handler classes covering the three major error domains
of the platform:

* **LLMErrorHandler** — LLM API errors (rate limits, timeouts, auth
  failures) with retry / fallback / escalation strategies.
* **AgentErrorHandler** — Agent work-processing errors with automatic
  retry, skip, and escalation based on error type and retry budget.
* **WorkflowErrorHandler** — Workflow lifecycle errors with retry
  queueing and failure notification.

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
from datetime import datetime
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
# Module Exports
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "LLMErrorHandler",
    "AgentErrorHandler",
    "WorkflowErrorHandler",
    "LLMError",
    "DatabaseError",
]
