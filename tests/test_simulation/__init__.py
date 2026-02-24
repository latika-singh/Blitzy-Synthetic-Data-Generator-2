"""
Test sub-package for the Simulation Engine module (app/simulation/).

Contains tests for:
- test_simulation_engine.py: Tests for SimulationEngine — the main composition root
  and daily cycle coordinator. Covers:
  - SimulationEngine initialization with constructor-injected dependencies
  - SimulationConfig Pydantic V2 model with default values matching performance
    criteria (#4: >=20 agents, #14: >=100 workflows, #16: <30s day advancement)
  - Main async run() loop execution across date ranges
  - 5-step daily processing pipeline: advance day -> generate interactions ->
    route transactions -> process agent work -> persist events
  - DayContext creation, population, and field constraints (Pydantic V2 with
    validate_assignment=True, non-negative counts, utilization [0.0-1.0])
  - DayContext convenience methods: to_daily_summary_dict(), is_period_closing(),
    is_quarter_end(), is_year_end(), transaction_completion_rate property
  - SimulationMetrics recording, daily/monthly aggregation, threshold validation
  - Daily summary logging with all 7 required fields: simulation_date,
    transactions_generated, agents_active, average_agent_utilization,
    llm_requests, llm_cost_usd, duration_seconds
  - Stop/pause/resume control methods
  - Health check for subsystem availability
  - Error handling: day-level errors don't crash the simulation
  - Multi-day simulation with cumulative metric accumulation

Testing standards:
- Uses pytest-asyncio for async simulation loop tests
- All external dependencies (TimeController, ExternalWorldManager,
  WorkflowOrchestrator, EventBus, AgentRegistry) are mocked
- No live LLM API calls (mocked)
- No external Redis dependency (mocked or fakeredis)
- Target coverage: >=80%
"""
