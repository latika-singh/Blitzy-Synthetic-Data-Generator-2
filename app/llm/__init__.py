"""LLM Integration package (F-002) — Unified async LLM client with comprehensive resilience.

Provides LLMClient (unified async interface for Anthropic Claude and OpenAI/Azure),
PromptManager (YAML template loading with context assembly), LLMQueue (Redis Streams
queuing with priority and circuit breaker), LLMMonitor (cost/latency tracking with
budget enforcement), and ResponseParser (structured JSON extraction with validation).

Supports providers:
- Anthropic (Claude Sonnet 4, Claude Haiku 4)
- OpenAI (GPT-4 Turbo, GPT-3.5 Turbo)
- Azure OpenAI

Performance targets:
- Average LLM latency: < 3 seconds
- LLM request success rate: >= 99%
- Monthly budget: $100 USD hard cap
- Prompt token efficiency: <= 1,500 tokens per decision
"""

# ---------------------------------------------------------------------------
# Public re-exports — lightweight init, no I/O, no object instantiation.
# Per AAP Section 0.7.3: "Agent initialization must be lightweight with lazy
# resource loading" — only import symbols, never construct instances.
# Per AAP Section 0.7.1: "Constructor injection is required for all
# dependencies" — no singletons or global instances at package level.
# ---------------------------------------------------------------------------

from app.llm.llm_config import LLMConfig, LLMProviderType
from app.llm.llm_client import LLMClient
from app.llm.llm_queue import LLMQueue, LLMRateLimiter
from app.llm.llm_monitor import LLMBudgetManager, LLMMonitor
from app.llm.prompt_manager import PromptManager
from app.llm.response_parser import ResponseParser

# ---------------------------------------------------------------------------
# Explicit public API — consumed by app.agents.decision_engine,
# app.simulation.simulation_engine, and tests throughout the project.
# Alphabetically sorted per agent_prompt Phase 3 specification.
# ---------------------------------------------------------------------------

__all__ = [
    "LLMBudgetManager",
    "LLMClient",
    "LLMConfig",
    "LLMMonitor",
    "LLMProviderType",
    "LLMQueue",
    "LLMRateLimiter",
    "PromptManager",
    "ResponseParser",
]
