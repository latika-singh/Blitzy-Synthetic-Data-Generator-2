"""Cross-cutting error handling package for the Agent & Orchestration Engine.

This package provides centralized error handling for the three major error
domains of the platform:

* **LLMErrorHandler** — Handles LLM API errors (rate limits, timeouts,
  authentication failures) with retry, fallback to cheaper models, and
  escalation strategies.
* **AgentErrorHandler** — Handles agent work-item processing errors with
  automatic skip (validation errors), retry (LLM/unknown errors up to budget),
  and escalation (database errors).
* **WorkflowErrorHandler** — Handles workflow lifecycle errors with up to 3
  retries via the orchestrator, then failure notification through monitoring.

All handlers use ``structlog`` for structured JSON logging to stdout — no file
handlers, no Prometheus / Grafana / APM integration (per AAP Section 0.7.6).

This module is part of the cross-cutting error handling layer (Group 9 in
AAP Section 0.5.1).  Handler instances are injected via constructors where
needed (per AAP Section 0.7.1).

Additionally re-exports the two custom exception types defined in
:mod:`app.errors.error_handlers` for convenience:

* **LLMError** — Raised when an LLM completion fails irrecoverably.
* **DatabaseError** — Raised on persistent Redis / data-store failures.

Usage::

    from app.errors import LLMErrorHandler, AgentErrorHandler, WorkflowErrorHandler

References:
    - README.md lines 1496–1646 (Error Handling Specification)
    - AAP Section 0.5.1 Group 9 (Cross-cutting Error Handling)
    - AAP Section 0.7.1 (Constructor injection)
    - AAP Section 0.7.6 (structlog logging standard)
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
]
