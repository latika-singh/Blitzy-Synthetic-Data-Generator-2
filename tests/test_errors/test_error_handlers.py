"""Comprehensive tests for app.errors.error_handlers module.

Covers all three error handler classes (LLMErrorHandler, AgentErrorHandler,
WorkflowErrorHandler), custom exception types (LLMError, DatabaseError),
and the _scrub_context helper.

References:
- AAP Section 0.5.1 Group 9 (Cross-cutting Error Handling)
- README.md lines 1496-1646
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest
from pydantic import ValidationError, BaseModel

from app.errors.error_handlers import (
    AgentErrorHandler,
    DatabaseError,
    LLMError,
    LLMErrorHandler,
    WorkflowErrorHandler,
    _scrub_context,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════════


class _StrictModel(BaseModel):
    """Helper Pydantic model used to produce real ValidationError instances."""

    value: int


def _make_validation_error() -> ValidationError:
    """Create a real Pydantic ValidationError for testing."""
    try:
        _StrictModel(value="not_an_int")  # type: ignore[arg-type]
    except ValidationError as exc:
        return exc
    raise RuntimeError("Expected ValidationError")


def _make_agent(role: str = "ap_clerk") -> SimpleNamespace:
    """Create a lightweight mock agent for AgentErrorHandler tests."""
    return SimpleNamespace(
        config=SimpleNamespace(
            agent_id=uuid.uuid4(),
            role=role,
        ),
        state="idle",
        metrics={},
    )


def _make_work_item(item_type: str = "process_invoice", retry_count: int = 0) -> SimpleNamespace:
    """Create a lightweight mock work item."""
    return SimpleNamespace(
        type=item_type,
        retry_count=retry_count,
    )


def _make_workflow(
    transaction_type: str = "purchase_order",
    retry_count: int = 0,
) -> SimpleNamespace:
    """Create a lightweight mock WorkflowInstance."""
    return SimpleNamespace(
        workflow_id=uuid.uuid4(),
        transaction_type=transaction_type,
        retry_count=retry_count,
        status="active",
        error_message=None,
        error_timestamp=None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tests — _scrub_context helper
# ═══════════════════════════════════════════════════════════════════════════


class TestScrubContext:
    """Tests for the _scrub_context helper function."""

    def test_scrub_api_key(self) -> None:
        ctx = {"api_key": "sk-secret-123", "model": "claude"}
        result = _scrub_context(ctx)
        assert result["api_key"] == "***REDACTED***"
        assert result["model"] == "claude"

    def test_scrub_multiple_sensitive_keys(self) -> None:
        ctx = {
            "api_key": "key1",
            "authorization": "Bearer tok",
            "password": "p@ss",
            "model": "gpt-4",
        }
        result = _scrub_context(ctx)
        assert result["api_key"] == "***REDACTED***"
        assert result["authorization"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["model"] == "gpt-4"

    def test_scrub_case_insensitive(self) -> None:
        ctx = {"API_KEY": "val", "Token": "val2"}
        result = _scrub_context(ctx)
        assert result["API_KEY"] == "***REDACTED***"
        assert result["Token"] == "***REDACTED***"

    def test_scrub_does_not_mutate_original(self) -> None:
        ctx = {"api_key": "original"}
        result = _scrub_context(ctx)
        assert ctx["api_key"] == "original"
        assert result["api_key"] == "***REDACTED***"

    def test_scrub_empty_context(self) -> None:
        assert _scrub_context({}) == {}

    def test_scrub_no_sensitive_keys(self) -> None:
        ctx = {"model": "claude", "temperature": 0.7}
        result = _scrub_context(ctx)
        assert result == ctx

    def test_scrub_all_known_sensitive_keys(self) -> None:
        sensitive = [
            "api_key", "apikey", "api_secret", "authorization",
            "auth_token", "token", "secret", "password", "credential",
        ]
        ctx = {k: f"val_{k}" for k in sensitive}
        result = _scrub_context(ctx)
        for k in sensitive:
            assert result[k] == "***REDACTED***"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Custom Exception Types
# ═══════════════════════════════════════════════════════════════════════════


class TestCustomExceptions:
    """Tests for LLMError and DatabaseError."""

    def test_llm_error_is_exception(self) -> None:
        assert issubclass(LLMError, Exception)

    def test_llm_error_message(self) -> None:
        err = LLMError("completion failed")
        assert str(err) == "completion failed"

    def test_database_error_is_exception(self) -> None:
        assert issubclass(DatabaseError, Exception)

    def test_database_error_message(self) -> None:
        err = DatabaseError("redis timeout")
        assert str(err) == "redis timeout"

    def test_llm_error_raise_catch(self) -> None:
        with pytest.raises(LLMError):
            raise LLMError("test")

    def test_database_error_raise_catch(self) -> None:
        with pytest.raises(DatabaseError):
            raise DatabaseError("test")


# ═══════════════════════════════════════════════════════════════════════════
# Tests — LLMErrorHandler
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMErrorHandler:
    """Tests for the LLMErrorHandler class."""

    @pytest.fixture()
    def handler(self) -> LLMErrorHandler:
        return LLMErrorHandler()

    @pytest.fixture()
    def base_context(self) -> Dict[str, Any]:
        return {"model": "claude-sonnet", "max_tokens": 1000, "retry_count": 0}

    async def test_rate_limit_error_returns_retry(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        # Patch asyncio.sleep to avoid real delay
        with patch("app.errors.error_handlers.asyncio.sleep", new_callable=AsyncMock):
            result = await handler.handle_llm_error(error, base_context)
        assert result == "retry"

    async def test_timeout_error_retry_reduces_max_tokens(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = anthropic.APITimeoutError(request=MagicMock())
        base_context["retry_count"] = 0
        result = await handler.handle_llm_error(error, base_context)
        assert result == "retry"
        assert base_context["max_tokens"] == 700  # 1000 * 0.7

    async def test_timeout_error_fallback_after_retries(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = anthropic.APITimeoutError(request=MagicMock())
        base_context["retry_count"] = 2
        result = await handler.handle_llm_error(error, base_context)
        assert result == "fallback"

    async def test_auth_error_returns_escalate(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = anthropic.AuthenticationError(
            message="invalid key",
            response=MagicMock(status_code=401, headers={}),
            body=None,
        )
        result = await handler.handle_llm_error(error, base_context)
        assert result == "escalate"

    async def test_generic_api_error_retry_first(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = anthropic.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
        base_context["retry_count"] = 0
        result = await handler.handle_llm_error(error, base_context)
        assert result == "retry"

    async def test_generic_api_error_fallback_after_retry(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = anthropic.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
        base_context["retry_count"] = 1
        result = await handler.handle_llm_error(error, base_context)
        assert result == "fallback"

    async def test_unknown_error_returns_escalate(
        self, handler: LLMErrorHandler, base_context: Dict[str, Any]
    ) -> None:
        error = RuntimeError("unexpected failure")
        result = await handler.handle_llm_error(error, base_context)
        assert result == "escalate"

    async def test_context_scrubbed_in_logging(
        self, handler: LLMErrorHandler
    ) -> None:
        ctx = {"api_key": "secret", "model": "claude", "retry_count": 0}
        error = RuntimeError("test")
        result = await handler.handle_llm_error(error, ctx)
        assert result == "escalate"
        # The original context should still have the key (not mutated to redacted)
        assert ctx["api_key"] == "secret"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — AgentErrorHandler
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentErrorHandler:
    """Tests for the AgentErrorHandler class."""

    @pytest.fixture()
    def handler(self) -> AgentErrorHandler:
        return AgentErrorHandler()

    async def test_validation_error_returns_skip(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item()
        error = _make_validation_error()
        result = await handler.handle_agent_error(agent, work_item, error)
        assert result == "skip"

    async def test_validation_error_sets_error_state(
        self, handler: AgentErrorHandler
    ) -> None:
        from app.agents.agent_config import AgentState

        agent = _make_agent()
        work_item = _make_work_item()
        error = _make_validation_error()
        await handler.handle_agent_error(agent, work_item, error)
        assert agent.state == AgentState.ERROR

    async def test_validation_error_increments_error_count(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item()
        error = _make_validation_error()
        await handler.handle_agent_error(agent, work_item, error)
        assert agent.metrics["errors"] == 1

    async def test_llm_error_retry_under_budget(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item(retry_count=0)
        error = LLMError("model unavailable")
        result = await handler.handle_agent_error(agent, work_item, error)
        assert result == "retry"
        assert work_item.retry_count == 1

    async def test_llm_error_skip_after_3_retries(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item(retry_count=3)
        error = LLMError("model unavailable")
        result = await handler.handle_agent_error(agent, work_item, error)
        assert result == "skip"

    async def test_database_error_returns_escalate(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item()
        error = DatabaseError("redis connection lost")
        result = await handler.handle_agent_error(agent, work_item, error)
        assert result == "escalate"

    async def test_unknown_error_retry_first_time(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item(retry_count=0)
        error = RuntimeError("unexpected")
        result = await handler.handle_agent_error(agent, work_item, error)
        assert result == "retry"
        assert work_item.retry_count == 1

    async def test_unknown_error_skip_after_1_retry(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        work_item = _make_work_item(retry_count=1)
        error = RuntimeError("unexpected")
        result = await handler.handle_agent_error(agent, work_item, error)
        assert result == "skip"

    async def test_multiple_errors_increment_count(
        self, handler: AgentErrorHandler
    ) -> None:
        agent = _make_agent()
        for i in range(3):
            work_item = _make_work_item(retry_count=0)
            await handler.handle_agent_error(agent, work_item, LLMError("fail"))
        assert agent.metrics["errors"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# Tests — WorkflowErrorHandler
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkflowErrorHandler:
    """Tests for the WorkflowErrorHandler class."""

    async def test_retry_under_budget_sets_retrying_status(self) -> None:
        orchestrator = AsyncMock()
        handler = WorkflowErrorHandler(orchestrator=orchestrator)
        workflow = _make_workflow(retry_count=0)
        await handler.handle_workflow_error(workflow, RuntimeError("transient"))
        assert workflow.status == "retrying"
        assert workflow.retry_count == 1
        orchestrator.retry_workflow.assert_awaited_once()

    async def test_retry_calls_orchestrator(self) -> None:
        orchestrator = AsyncMock()
        handler = WorkflowErrorHandler(orchestrator=orchestrator)
        workflow = _make_workflow(retry_count=1)
        await handler.handle_workflow_error(workflow, RuntimeError("transient"))
        assert workflow.retry_count == 2
        orchestrator.retry_workflow.assert_awaited_once_with(workflow.workflow_id)

    async def test_retry_orchestrator_failure_marks_failed(self) -> None:
        orchestrator = AsyncMock()
        orchestrator.retry_workflow.side_effect = RuntimeError("queue full")
        handler = WorkflowErrorHandler(orchestrator=orchestrator)
        workflow = _make_workflow(retry_count=0)
        await handler.handle_workflow_error(workflow, RuntimeError("original"))
        assert workflow.status == "failed"

    async def test_fail_after_3_retries(self) -> None:
        monitoring = AsyncMock()
        handler = WorkflowErrorHandler(monitoring=monitoring)
        workflow = _make_workflow(retry_count=3)
        await handler.handle_workflow_error(workflow, RuntimeError("persistent"))
        assert workflow.status == "failed"
        monitoring.alert_workflow_failed.assert_awaited_once_with(workflow)

    async def test_fail_alert_error_handled(self) -> None:
        monitoring = AsyncMock()
        monitoring.alert_workflow_failed.side_effect = RuntimeError("alert fail")
        handler = WorkflowErrorHandler(monitoring=monitoring)
        workflow = _make_workflow(retry_count=3)
        # Should not raise
        await handler.handle_workflow_error(workflow, RuntimeError("persistent"))
        assert workflow.status == "failed"

    async def test_no_orchestrator_still_sets_retrying(self) -> None:
        handler = WorkflowErrorHandler()
        workflow = _make_workflow(retry_count=0)
        await handler.handle_workflow_error(workflow, RuntimeError("test"))
        assert workflow.status == "retrying"
        assert workflow.retry_count == 1

    async def test_no_monitoring_still_marks_failed(self) -> None:
        handler = WorkflowErrorHandler()
        workflow = _make_workflow(retry_count=3)
        await handler.handle_workflow_error(workflow, RuntimeError("test"))
        assert workflow.status == "failed"

    async def test_error_message_stored(self) -> None:
        handler = WorkflowErrorHandler()
        workflow = _make_workflow(retry_count=3)
        await handler.handle_workflow_error(workflow, ValueError("bad value"))
        assert workflow.error_message == "bad value"

    async def test_error_timestamp_set(self) -> None:
        handler = WorkflowErrorHandler()
        workflow = _make_workflow(retry_count=0)
        before = datetime.utcnow()
        await handler.handle_workflow_error(workflow, RuntimeError("t"))
        assert workflow.error_timestamp is not None
        assert workflow.error_timestamp >= before
