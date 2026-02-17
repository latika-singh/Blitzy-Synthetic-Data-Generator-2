"""
Agent System package (F-001) — Core framework for autonomous ERP employee simulation.

This package implements the Agent System subsystem of Project 2 (AI-powered Agent
& Orchestration Engine). It provides the foundational framework for creating
autonomous, memory-equipped, LLM-augmented agents that simulate realistic ERP
employee behavior across six functional areas: Accounts Payable, Accounts
Receivable, Purchasing, Warehouse, Accounting, and Financial Control.

Public API Classes:
    BaseAgent            — Abstract base class for all 12 specialized agent types.
                           Provides constructor injection (config, memory,
                           decision_engine, action_registry), async run() loop,
                           abstract process_work_item(), make_decision(), importance
                           calculation, and metrics tracking.
    WorkItem             — Dataclass representing a unit of work (type, data,
                           workflow_id, priority, amount, created_at) to be queued
                           and processed by agents.
    WorkResult           — Dataclass representing processing outcomes (success,
                           actions_taken, approval_needed, has_exceptions, data,
                           error, processing_time_seconds).
    AgentConfig          — Pydantic V2 model for agent configuration including
                           identity (agent_id, role, name), personality traits
                           (thoroughness, risk_tolerance, efficiency, compliance
                           each 0.0–1.0), work schedule, and LLM settings.
    AgentState           — String enum for agent lifecycle states: IDLE, THINKING,
                           ACTING, WAITING, ERROR.
    AgentMemory          — Dual-stream memory system with observation stream
                           (max 1,000 entries, ~500 bytes each) and reflection
                           stream (max 100 entries, ~2 KB each). Semantic retrieval
                           via all-MiniLM-L6-v2 cosine similarity with Redis
                           persistence (30-day TTL).
    AgentRegistry        — Centralized agent lifecycle management with role-based
                           O(1) lookup, availability tracking, capacity monitoring,
                           and Redis state persistence (24-hour TTL). Supports up
                           to 50 simultaneous agents.
    AgentTimeoutManager  — Per-agent consecutive timeout tracking with a 3-timeout
                           escalation threshold and 60-second cooldown auto-restart.
    ActionRegistry       — Registry of 14 ERP action types spanning procurement,
                           AP, AR, warehouse, and accounting functions. Each action
                           includes required_inputs, validation_rules, and execute().
    DecisionEngine       — 4-layer hybrid decision pipeline:
                           Statistical → LLM (conditional) → Validation → Deterministic.
                           Financial calculations (GL postings, balances) are
                           EXCLUSIVELY computed in the Deterministic Layer.

Architecture Notes:
    - Every specialized agent MUST extend BaseAgent and implement process_work_item().
    - The 4-layer Decision Engine pipeline is inviolable — no agent may skip or
      reorder layers.
    - All cross-module data contracts use Pydantic V2 models for validation.
    - Constructor injection is required for all dependencies.
    - No heavy computation occurs at import time — agent creation rate ≥ 50/second.

Performance Targets:
    - Agent creation:     ≥ 50 agents/second
    - Decision latency:   p95 < 5 seconds
    - Agent concurrency:  ≥ 20 simultaneous agents

References:
    - README.md lines 772–890   (BaseAgent specification)
    - README.md lines 937–949   (WorkResult in APClerkAgent)
    - README.md lines 1324–1328 (WorkItem in WorkflowOrchestrator)
    - AAP Section 0.5.1 Group 5 (Agent System file plan)
"""

from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig, AgentState
from app.agents.agent_memory import AgentMemory
from app.agents.agent_registry import AgentRegistry, AgentTimeoutManager
from app.agents.action_registry import ActionRegistry
from app.agents.decision_engine import DecisionEngine

__all__ = [
    "BaseAgent",
    "WorkItem",
    "WorkResult",
    "AgentConfig",
    "AgentState",
    "AgentMemory",
    "AgentRegistry",
    "AgentTimeoutManager",
    "ActionRegistry",
    "DecisionEngine",
]
