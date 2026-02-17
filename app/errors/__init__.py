"""Cross-cutting error handling module for the Agent & Orchestration Engine.

Provides centralized error handler classes for LLM integration errors,
agent processing errors, and workflow lifecycle errors, along with
custom exception types used throughout the platform.

Exports:
    LLMErrorHandler      — Handles LLM API errors (rate limits, timeouts,
                           auth failures) with retry/fallback/escalation.
    AgentErrorHandler    — Handles agent work processing errors with
                           retry/skip/escalation strategies.
    WorkflowErrorHandler — Handles workflow-level errors with retry
                           queuing and failure notification.
    LLMError             — Custom exception for LLM-specific errors.
    DatabaseError        — Custom exception for database-specific errors.
"""

from app.errors.error_handlers import (
    AgentErrorHandler,
    DatabaseError,
    LLMError,
    LLMErrorHandler,
    WorkflowErrorHandler,
)

__all__ = [
    "LLMErrorHandler",
    "AgentErrorHandler",
    "WorkflowErrorHandler",
    "LLMError",
    "DatabaseError",
]
