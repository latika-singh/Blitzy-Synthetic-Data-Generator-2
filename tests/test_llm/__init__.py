"""
Test sub-package for the LLM Integration module (F-002, app/llm/).

Contains test modules:
- test_llm_client: Provider abstraction, retry logic, fallback, cost recording (mocked APIs)
- test_prompt_manager: YAML template loading, context assembly, token estimation
- test_llm_queue: Redis Streams enqueue/dequeue, priority, batch processing, circuit breaker
- test_llm_monitor: LLMMonitor metrics, LLMBudgetManager budget enforcement
- test_response_parser: JSON extraction, Pydantic schema validation, retry loop

CRITICAL Testing Rules:
- ALL LLM tests use mocked API responses — ZERO live API calls
- ALL Redis tests use fakeredis in-memory mock
- ALL async tests use pytest-asyncio
- Coverage target: >= 80%
"""
