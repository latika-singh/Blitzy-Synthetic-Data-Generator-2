"""Intelligent Rework Loop package (Project 3) — Autonomous error correction system.

This package implements the Intelligent Rework Loop for Project 3: Transaction
Workflows & Discrepancies. The rework loop classifies validation failures,
selects fix scenarios from a catalog, applies corrections (max 3 attempts per
transaction), and escalates unresolvable errors for human review.

Core Components:
    ReworkLoopEngine
        Orchestrates the classify → select fix → apply → re-validate → escalate
        cycle. Enforces max 3 attempts per transaction. Tracks fix scenarios
        used and their success rates. Escalates if >5% failure rate (configurable
        via REWORK_ESCALATION_THRESHOLD env var). Implements circuit breaker:
        50 failures → open circuit, 120s recovery timeout.

    FailureClassifier
        Classifies validation failures as planned discrepancy (within/outside
        configured parameter bounds) or unplanned error. Returns classification
        result with confidence score and recommended action.

    FixScenarioCatalog
        Registry of 20+ fix scenarios (ADJUST_AMOUNT_TO_RANGE, FIX_DATE_SEQUENCE,
        CORRECT_ENTITY_REFERENCE, REGENERATE_GL_ENTRY, etc.) with names, step
        definitions, and success rates. Sorted by success rate for selection.
        Excludes previously failed scenarios from re-selection.

    FixScenarioExecutor
        Executes fix scenario steps against a transaction. Handles per-scenario
        timeout (10s). Logs fix attempt details (scenario name, steps executed,
        duration, success/failure).

Architecture:
    - Constructor injection (ADR-003) for all dependencies
    - Pydantic V2 for all data contracts at subsystem boundaries
    - EventBus (ADR-001) for rework event publication
    - structlog for JSON structured logging to stdout
    - Deterministic seeded random.Random for reproducibility
    - tenacity for retry with configurable backoff

Circuit Breaker Configuration (AAP §0.1.2):
    - Failure threshold: 50 consecutive failures → circuit OPEN
    - Recovery timeout: 120 seconds before attempting half-open probe
    - Fallback: Escalate to admin

Retry Policy (AAP §0.1.2):
    - Max attempts: 3
    - Backoff strategy: Linear (1s, 2s, 3s)
    - Timeout: 30 seconds
    - Fallback: Escalate to admin

References:
    - AAP Section 0.5.1 Group 6: Intelligent Rework Loop
    - AAP Section 0.7.4: Error Handling Conventions
    - AAP Section 0.7.1: Constructor injection, Pydantic V2, EventBus
"""

from __future__ import annotations

from app.rework.rework_loop_engine import ReworkLoopEngine
from app.rework.failure_classifier import FailureClassifier
from app.rework.fix_scenario_catalog import FixScenarioCatalog
from app.rework.fix_scenario_executor import FixScenarioExecutor

__all__: list[str] = [
    "ReworkLoopEngine",
    "FailureClassifier",
    "FixScenarioCatalog",
    "FixScenarioExecutor",
]
