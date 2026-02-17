"""
Simulation Engine package — the composition root of Project 2.

This package provides the top-level orchestrator that wires all seven
subsystems together and drives the daily simulation cycle:

    advance day → generate interactions → route transactions
    → process agent work → persist events

Primary exports:
    SimulationEngine   – Main async loop and dependency-injection composition
                         root.  Receives TimeController, ExternalWorldManager,
                         WorkflowOrchestrator, EventBus, and SimulationMetrics
                         via constructor injection.
    DayContext          – Pydantic V2 model carrying daily simulation state
                         (date, fiscal period, agent metrics, transaction
                         counts, LLM usage, events) through the pipeline.
    SimulationMetrics   – Performance tracking with daily snapshots and
                         monthly aggregation.  Validates against the 20
                         measurable thresholds defined in the specification.

Performance targets:
    • Advance 1 business day within < 30 seconds
    • Complete 1 month of simulation (2 000 transactions) in 4–8 hours

All logging uses ``structlog`` to stdout in structured JSON format.
"""

from app.simulation.day_context import DayContext
from app.simulation.simulation_engine import SimulationConfig, SimulationEngine
from app.simulation.simulation_metrics import SimulationMetrics

__all__ = [
    "DayContext",
    "SimulationConfig",
    "SimulationEngine",
    "SimulationMetrics",
]
