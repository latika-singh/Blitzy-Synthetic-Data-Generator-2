"""Simulation Engine package — the composition root of Project 2.

This package is the **top-level orchestrator** of the Agent & Orchestration
Engine.  It wires all seven subsystems together via constructor injection
(AAP §0.4.3, §0.7.1) and drives the daily simulation cycle::

    advance day → generate interactions → route transactions
    → process agent work → persist events

Primary exports
---------------
SimulationEngine
    Main ``async`` loop and dependency-injection composition root.  Receives
    :class:`~app.orchestration.time_controller.TimeController`,
    :class:`~app.external_world.external_entity_manager.ExternalWorldManager`,
    :class:`~app.orchestration.workflow_orchestrator.WorkflowOrchestrator`,
    :class:`~app.events.event_bus.EventBus`, and
    :class:`SimulationMetrics` via constructor injection.

DayContext
    Pydantic V2 model carrying daily simulation state (date, fiscal period,
    agent metrics, transaction counts, LLM usage, events) through the
    processing pipeline.  Created fresh for every simulated business day.

SimulationMetrics
    Performance tracking with daily snapshots and monthly aggregation.
    Validates against the 20 measurable performance thresholds defined in
    the specification (README.md §SUCCESS CRITERIA).

Performance targets (AAP §0.1.2)
---------------------------------
* Advance 1 business day within < 30 seconds.
* Complete 1 month of simulation (~2,000 transactions) in 4–8 hours.

All logging uses ``structlog`` to stdout in structured JSON format
(AAP §0.7.6).
"""

from app.simulation.day_context import DayContext
from app.simulation.simulation_engine import SimulationEngine
from app.simulation.simulation_metrics import SimulationMetrics

__all__ = [
    "DayContext",
    "SimulationEngine",
    "SimulationMetrics",
]
