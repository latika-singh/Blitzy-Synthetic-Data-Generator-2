"""
Agent & Orchestration Engine — Root Application Package.

Project 2 of the Synthetic ERP Data Generation Platform.

This package implements an AI-powered Agent & Orchestration Engine that
simulates realistic ERP employee behavior through autonomous, memory-equipped,
LLM-augmented agents. The engine drives a full procure-to-pay and
order-to-cash cycle using hybrid statistical and LLM-based decision-making,
workflow orchestration, time management, and external world simulation.

Features
--------
F-001 — Agent System
    Abstract BaseAgent framework with personality traits, dual-stream memory,
    14-action registry, and 12 specialized agent subclasses across six ERP
    functional areas (AP, AR, Purchasing, Warehouse, Accounting, Financial
    Control).

F-002 — LLM Integration
    Unified LLMClient supporting Anthropic and OpenAI providers, with a
    PromptManager (6 YAML templates), LLMQueue (Redis Streams), LLMMonitor
    (cost tracking), and budget enforcement.

F-003 — Workflow Orchestration
    WorkflowOrchestrator for transaction-to-agent routing,
    TransactionOrchestrator for artifact tracking, and ApprovalSystem for
    monetary-threshold-based approval chains.

F-004 — Time Controller
    TimeController with BusinessCalendar (US Federal holidays 2024–2026,
    working hours) and FiscalCalendar (configurable fiscal year,
    monthly/quarterly periods with open/closing/closed lifecycle).

F-005 — External World Simulation
    ExternalWorldManager with tiered entity pools (Strategic, Standard,
    Transactional) and simulators for customers, vendors, and banks.

F-006 — Statistical Models
    Distribution models for amounts (log-normal), payment timing (5-segment
    mixture), order frequency (Poisson with day-of-week effects), and entity
    selection (Pareto 80/20).

F-007 — Event System
    Dual-mode EventBus (in-memory asyncio.Queue / optional Redis Pub/Sub),
    EventStore with JSONB persistence, 8 event types, and event replay.

Sub-packages
------------
agents
    Core agent framework: BaseAgent, AgentConfig, AgentMemory, AgentRegistry,
    ActionRegistry, DecisionEngine, and 12 specialized agent implementations.
llm
    LLM integration: LLMClient, LLMConfig, LLMQueue, LLMMonitor,
    PromptManager, and ResponseParser.
orchestration
    Workflow routing, approval chains, time control, and calendar management.
external_world
    External entity simulation with tiered behavior profiles and simulators.
statistical
    Statistical distribution models for amounts, timing, frequency, and
    entity selection.
events
    Event-driven coordination: EventBus, EventStore, event types, and
    handlers.
simulation
    Top-level SimulationEngine, DayContext, and SimulationMetrics.
errors
    Cross-cutting error handlers for LLM, agent, and workflow subsystems.
"""

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

__version__: str = "0.1.0"
"""Semantic version aligned with *pyproject.toml* ``[project].version``."""

__all__: list[str] = [
    "agents",
    "llm",
    "orchestration",
    "external_world",
    "statistical",
    "events",
    "simulation",
    "errors",
]
"""Public sub-packages exposed by the ``app`` package.

Sub-packages are intentionally **not** imported eagerly at the package level
so that ``import app`` remains fast and avoids pulling in heavyweight
dependencies (e.g. ``sentence-transformers``, ``scipy``, LLM provider SDKs).
Consumers should import the specific sub-package they need::

    from app.agents.base_agent import BaseAgent
    from app.llm.llm_client import LLMClient
"""
