"""
Test suite for the Synthetic ERP Data Generation Platform — Project 2: Agent & Orchestration Engine.

Contains test modules organized into sub-packages mirroring app/ structure:
- test_agents/ — BaseAgent, AgentMemory, AgentConfig, DecisionEngine, ActionRegistry, specialized agents
- test_llm/ — LLMClient, PromptManager, LLMQueue, LLMMonitor, ResponseParser
- test_orchestration/ — WorkflowOrchestrator, ApprovalSystem, TransactionOrchestrator, TimeController
- test_external_world/ — ExternalEntityManager, BehaviorProfiles, Simulators
- test_statistical/ — PaymentTimingModel, AmountDistributions, OrderFrequencyModel, SelectionModels
- test_events/ — EventBus, EventStore
- test_simulation/ — SimulationEngine

Testing Standards:
- Target coverage: ≥80%
- All LLM tests use mocked API responses (no live API calls)
- Redis tests use fakeredis in-memory mock
- Async tests use pytest-asyncio with asyncio_mode=auto
"""
