"""
Tests for the Agent System module (app/agents/).

Contains test modules for:
- test_base_agent: BaseAgent lifecycle, async run() loop, state transitions, work queue, metrics
- test_agent_memory: Dual-stream memory (observations/reflections), semantic retrieval, cleanup
- test_agent_config: AgentConfig Pydantic validation, trait boundaries, AgentState enum
- test_decision_engine: 4-layer pipeline (Statistical→LLM→Validation→Deterministic)
- test_action_registry: 14 action type registration, input validation, execution
- test_specialized_agents: All 12 specialized agent types (AP, AR, Purchasing, Warehouse, Accounting, Financial Control)

Testing Standards:
- All LLM calls are mocked (no live API calls)
- Redis interactions use fakeredis
- Async tests use pytest-asyncio
- Coverage target: ≥80%
"""
