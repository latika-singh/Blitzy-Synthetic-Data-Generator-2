"""
Shared Test Fixtures and Mock Factories for the Project 2 Test Suite.

This module provides the foundational test infrastructure used across all
test modules in the Synthetic ERP Data Generation Platform — Agent &
Orchestration Engine (Project 2).

CRITICAL Testing Rules (AAP Section 0.7.5):
    - All LLM integration tests use mocked API responses — NO live API calls.
    - Redis tests use fakeredis in-memory mock — NO external Redis dependency.
    - Async tests use pytest-asyncio with proper event loop management.
    - All agent tests verify state transitions (IDLE → THINKING → ACTING → IDLE).
    - Unit test coverage target: ≥ 80% across all source modules.

Fixture Categories:
    1. Redis Mocks           — mock_redis, mock_async_redis
    2. Agent System           — sample_agent_config, factory, memory/decision/action mocks
    3. LLM Integration        — config, client mock, monitor, budget, prompt_manager, parser
    4. Event System           — event_bus (in-memory), event_store (fakeredis)
    5. Work Items/Results     — sample_work_item, factory, sample_work_result
    6. Statistical Utilities  — random_seed (numpy deterministic seeding)
    7. Context Utilities      — company context, transaction context
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import asyncio
from datetime import datetime, date
from typing import Any, AsyncGenerator, Callable, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import fakeredis
import fakeredis.aioredis
import numpy as np
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Internal Imports — Agent System (app.agents)
# ---------------------------------------------------------------------------
from app.agents.agent_config import (
    AgentConfig,
    AgentState,
    DEFAULT_TRAITS,
    VALID_AGENT_ROLES,
)
from app.agents.agent_memory import AgentMemory, AgentMemoryConfig
from app.agents.action_registry import ActionRegistry, ActionResult
from app.agents.decision_engine import DecisionEngine
from app.agents.agent_registry import AgentRegistry
from app.agents.base_agent import BaseAgent, WorkItem, WorkResult

# ---------------------------------------------------------------------------
# Internal Imports — LLM Integration (app.llm)
# ---------------------------------------------------------------------------
from app.llm.llm_config import LLMConfig, LLMProviderType
from app.llm.llm_client import LLMClient
from app.llm.llm_monitor import LLMBudgetManager, LLMMonitor
from app.llm.llm_queue import LLMQueue
from app.llm.prompt_manager import PromptManager
from app.llm.response_parser import ResponseParser

# ---------------------------------------------------------------------------
# Internal Imports — Event System (app.events)
# ---------------------------------------------------------------------------
from app.events.event_bus import EventBus
from app.events.event_store import EventStore
from app.events.event_types import (
    Event,
    EventType,
    TransactionCreated,
    TransactionCompleted,
    ApprovalRequired,
    ApprovalCompleted,
    PeriodClosing,
)

# ---------------------------------------------------------------------------
# Deterministic UUIDs for Reproducible Tests
# ---------------------------------------------------------------------------
FIXED_AGENT_UUID = UUID("12345678-1234-5678-1234-567812345678")
FIXED_EMPLOYEE_UUID = UUID("22345678-2234-5678-2234-567822345678")
FIXED_COMPANY_UUID = UUID("32345678-3234-5678-3234-567832345678")
FIXED_WORKFLOW_UUID = UUID("42345678-4234-5678-4234-567842345678")
FIXED_SIMULATION_UUID = UUID("52345678-5234-5678-5234-567852345678")


# =========================================================================
# Section 1: Redis Mock Fixtures
# Per AAP Section 0.7.5: "Redis tests must use fakeredis"
# =========================================================================


@pytest.fixture
def mock_redis() -> fakeredis.FakeRedis:
    """Provide a synchronous fakeredis client for unit tests.

    Creates a fresh FakeRedis instance that is automatically flushed
    after each test to prevent cross-test contamination.

    Yields:
        A ``fakeredis.FakeRedis`` instance ready for synchronous Redis ops.
    """
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    yield redis_client
    redis_client.flushall()


@pytest_asyncio.fixture
async def mock_async_redis() -> AsyncGenerator[fakeredis.aioredis.FakeRedis, None]:
    """Provide an asynchronous fakeredis client for async unit tests.

    Creates a fresh async FakeRedis instance that is automatically
    flushed and closed after each test.

    Yields:
        A ``fakeredis.aioredis.FakeRedis`` instance ready for async Redis ops.
    """
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield redis_client
    await redis_client.flushall()
    await redis_client.aclose()


# =========================================================================
# Section 2: Agent System Fixtures
# =========================================================================


@pytest.fixture
def sample_agent_config() -> AgentConfig:
    """Provide a fully-populated AgentConfig with deterministic UUIDs.

    The config uses fixed UUIDs for reproducibility and the default
    role ``ap_clerk`` with standard trait values covering all four
    required personality dimensions.

    Returns:
        A complete ``AgentConfig`` instance suitable for most agent tests.
    """
    return AgentConfig(
        agent_id=FIXED_AGENT_UUID,
        role="ap_clerk",
        name="Test AP Clerk",
        employee_id=FIXED_EMPLOYEE_UUID,
        company_id=FIXED_COMPANY_UUID,
        traits={
            "thoroughness": 0.7,
            "risk_tolerance": 0.5,
            "efficiency": 0.6,
            "compliance": 0.8,
        },
        work_hours_start=8,
        work_hours_end=17,
        temperature=0.7,
        max_tokens=1000,
    )


@pytest.fixture
def sample_agent_config_factory() -> Callable[..., AgentConfig]:
    """Provide a factory function for creating customisable AgentConfig instances.

    The factory generates a unique ``agent_id`` per call (unless overridden)
    and falls back to ``DEFAULT_TRAITS`` when no explicit traits are supplied.

    Returns:
        A callable ``make_config(role, name, traits, **kwargs) -> AgentConfig``.

    Example::

        config = sample_agent_config_factory(role="cfo", name="Test CFO")
    """

    def make_config(
        role: str = "ap_clerk",
        name: str = "Test Agent",
        traits: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> AgentConfig:
        """Build an AgentConfig with sensible defaults and optional overrides.

        Args:
            role: Agent role identifier from VALID_AGENT_ROLES.
            name: Human-readable agent name.
            traits: Personality trait dict; uses DEFAULT_TRAITS if None.
            **kwargs: Additional AgentConfig fields to override (e.g.,
                agent_id, employee_id, company_id, work_hours_start, etc.).

        Returns:
            A new ``AgentConfig`` instance.
        """
        effective_traits = traits if traits is not None else DEFAULT_TRAITS.copy()
        config_fields: Dict[str, Any] = {
            "agent_id": kwargs.pop("agent_id", uuid4()),
            "role": role,
            "name": name,
            "employee_id": kwargs.pop("employee_id", uuid4()),
            "company_id": kwargs.pop("company_id", FIXED_COMPANY_UUID),
            "traits": effective_traits,
            "work_hours_start": kwargs.pop("work_hours_start", 8),
            "work_hours_end": kwargs.pop("work_hours_end", 17),
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", 1000),
        }
        # Merge any remaining overrides
        config_fields.update(kwargs)
        return AgentConfig(**config_fields)

    return make_config


@pytest.fixture
def mock_agent_memory() -> MagicMock:
    """Provide a fully-configured mock of ``AgentMemory``.

    All async methods are configured as ``AsyncMock`` instances with
    sensible default return values.  Synchronous methods use ``MagicMock``.

    Returns:
        A ``MagicMock`` spec'd against ``AgentMemory`` with pre-configured
        return values for all public methods.
    """
    memory = MagicMock(spec=AgentMemory)

    # Async methods — return sensible defaults
    memory.add_observation = AsyncMock(return_value=None)
    memory.retrieve_relevant = AsyncMock(return_value=[])
    memory.get_recent = AsyncMock(return_value=[])
    memory.generate_reflection = AsyncMock(return_value=None)

    # Sync methods — return zero-state stats
    memory.get_observation_count = MagicMock(return_value=0)
    memory.get_reflection_count = MagicMock(return_value=0)
    memory.get_memory_stats = MagicMock(
        return_value={
            "observation_count": 0,
            "reflection_count": 0,
            "embedding_model_loaded": False,
            "redis_enabled": False,
        }
    )

    return memory


@pytest.fixture
def mock_decision_engine() -> MagicMock:
    """Provide a fully-configured mock of ``DecisionEngine``.

    The ``decide()`` async mock returns a standard approval decision dict.
    Registration methods are configured as no-op MagicMocks.

    Returns:
        A ``MagicMock`` spec'd against ``DecisionEngine``.
    """
    engine = MagicMock(spec=DecisionEngine)

    # Primary decision method — returns a standard approval decision
    engine.decide = AsyncMock(
        return_value={
            "decision": "approve",
            "confidence": 0.85,
            "reasoning": "Test decision — approved based on policy compliance.",
        }
    )

    # Metrics and registration — lightweight mocks
    engine.get_metrics = MagicMock(
        return_value={
            "total_decisions": 0,
            "statistical_only": 0,
            "llm_invoked": 0,
            "validation_retries": 0,
            "failures": 0,
        }
    )
    engine.register_statistical_model = MagicMock(return_value=None)
    engine.register_validation_rule = MagicMock(return_value=None)
    engine.register_deterministic_rule = MagicMock(return_value=None)

    return engine


@pytest.fixture
def mock_action_registry() -> MagicMock:
    """Provide a mock ``ActionRegistry`` with pre-configured responses.

    The mock simulates a registry where all 14 ERP actions exist and
    all validation passes by default.

    Returns:
        A ``MagicMock`` spec'd against ``ActionRegistry``.
    """
    registry = MagicMock(spec=ActionRegistry)

    # All known actions "exist"
    registry.action_exists = MagicMock(return_value=True)

    # Validation returns no errors by default
    registry.validate_inputs = MagicMock(return_value=[])

    # Execute returns success
    registry.execute = AsyncMock(
        return_value=ActionResult(
            success=True,
            action_id="process_vendor_invoice",
            outputs={"status": "completed"},
            error=None,
        )
    )

    # Return all 14 action IDs matching the real ActionRegistry
    registry.get_all_action_ids = MagicMock(
        return_value=[
            "create_purchase_order",
            "approve_purchase_order",
            "receive_goods",
            "process_vendor_invoice",
            "match_three_way",
            "approve_invoice",
            "schedule_payment",
            "create_sales_order",
            "ship_order",
            "create_customer_invoice",
            "apply_payment",
            "create_journal_entry",
            "reconcile_account",
            "close_period",
        ]
    )

    return registry


@pytest.fixture
def action_registry() -> ActionRegistry:
    """Provide a REAL ``ActionRegistry`` instance with all 14 default actions.

    The ActionRegistry is self-contained with no external dependencies,
    making it safe for use in unit tests without mocking.

    Returns:
        A fully-initialised ``ActionRegistry`` with all default ERP actions.
    """
    return ActionRegistry()


@pytest_asyncio.fixture
async def agent_registry(
    mock_async_redis: fakeredis.aioredis.FakeRedis,
) -> AgentRegistry:
    """Provide a REAL ``AgentRegistry`` backed by fakeredis.

    This enables integration-level tests for agent lifecycle management
    (register, lookup, availability tracking) without a real Redis server.

    Args:
        mock_async_redis: Injected async fakeredis fixture.

    Returns:
        An ``AgentRegistry`` instance using the fakeredis backend.
    """
    return AgentRegistry(redis_client=mock_async_redis)


# =========================================================================
# Section 3: LLM Integration Fixtures
# Per AAP Section 0.7.5: "All LLM integration tests must use mocked API
# responses — no live API calls in the test suite"
# =========================================================================


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Provide an ``LLMConfig`` with safe test values.

    CRITICAL (AAP §0.7.4): The ``api_key`` is a clearly fake test key
    that must NEVER be a real credential.

    Returns:
        An ``LLMConfig`` configured for Anthropic Claude with test defaults.
    """
    return LLMConfig(
        provider=LLMProviderType.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        fallback_model="claude-haiku-4-20250514",
        api_key="test-api-key-not-real",
        max_tokens=1000,
        temperature=0.7,
        monthly_budget_usd=100.0,
        budget_warning_threshold=0.80,
        budget_block_threshold=1.0,
        request_timeout_seconds=30.0,
        max_retries=3,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Provide a mocked ``LLMClient`` that never makes real API calls.

    The ``complete()`` method returns a test JSON string simulating
    an approval decision.  ``get_metrics()`` returns a zero-state dict.

    CRITICAL: No real LLM API calls are made — AAP Section 0.7.5.

    Returns:
        A ``MagicMock`` spec'd against ``LLMClient``.
    """
    client = MagicMock(spec=LLMClient)

    # complete() returns a JSON response string
    client.complete = AsyncMock(
        return_value='{"decision": "approve", "reasoning": "Test response", "confidence": 0.9}'
    )

    # get_metrics() returns a zero-state metrics dict
    client.get_metrics = MagicMock(
        return_value={
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_latency": 0.0,
            "using_fallback": False,
        }
    )

    # close() is an async no-op
    client.close = AsyncMock(return_value=None)

    return client


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Provide a dict simulating a typical raw LLM API response.

    Includes content, token counts, and model identifier matching the
    Anthropic Claude Sonnet 4 response envelope.

    Returns:
        A dictionary with ``content``, ``input_tokens``, ``output_tokens``,
        and ``model`` keys.
    """
    return {
        "content": '{"decision": "approve", "confidence": 0.85}',
        "input_tokens": 500,
        "output_tokens": 100,
        "model": "claude-sonnet-4-20250514",
    }


@pytest.fixture
def llm_monitor() -> LLMMonitor:
    """Provide a REAL ``LLMMonitor`` instance with an embedded budget manager.

    The monitor and its budget manager have no external dependencies,
    making them safe for direct use in unit tests.

    Returns:
        An ``LLMMonitor`` with a real ``LLMBudgetManager`` ($100 budget).
    """
    budget_mgr = LLMBudgetManager(
        monthly_budget_usd=100.0,
        warning_threshold=0.80,
        block_threshold=1.0,
    )
    return LLMMonitor(budget_manager=budget_mgr)


@pytest.fixture
def llm_budget_manager() -> LLMBudgetManager:
    """Provide a REAL ``LLMBudgetManager`` with the standard $100 budget.

    Returns:
        An ``LLMBudgetManager`` instance ($100/month, 80% warning, 100% block).
    """
    return LLMBudgetManager(
        monthly_budget_usd=100.0,
        warning_threshold=0.80,
        block_threshold=1.0,
    )


@pytest.fixture
def prompt_manager(tmp_path: Any) -> PromptManager:
    """Provide a REAL ``PromptManager`` loaded with minimal test YAML templates.

    Creates a temporary directory with two representative prompt templates
    (process_vendor_invoice and approve_transaction) and returns a
    ``PromptManager`` instance pointing at that directory.

    Args:
        tmp_path: pytest built-in temporary directory fixture.

    Returns:
        A ``PromptManager`` with test templates loaded.
    """
    # Create minimal test prompt templates matching the expected YAML schema
    process_invoice_yaml = """\
system: |
  You are an experienced Accounts Payable clerk at {company_name}.
  Your role is to process vendor invoices accurately and efficiently.
  Thoroughness level: {thoroughness}
  Compliance level: {compliance}

user: |
  Please process the following vendor invoice:
  Invoice ID: {invoice_id}
  Vendor: {vendor_name}
  Amount: ${amount}
  Description: {description}

  Respond with a JSON object containing:
  - match_status: "matched" or "unmatched"
  - gl_coding: GL account code
  - notes: processing notes
  - recommendation: "approve" or "reject" or "escalate"
"""

    approve_transaction_yaml = """\
system: |
  You are a financial approver at {company_name}.
  Risk tolerance: {risk_tolerance}
  Compliance level: {compliance}

user: |
  Please review and make an approval decision for:
  Transaction Type: {transaction_type}
  Amount: ${amount}
  Requester: {requester_name}
  Description: {description}

  Respond with a JSON object containing:
  - decision: "approve" or "reject" or "escalate"
  - reasoning: your detailed reasoning
"""

    # Write templates to temporary directory
    (tmp_path / "process_vendor_invoice.yaml").write_text(process_invoice_yaml)
    (tmp_path / "approve_transaction.yaml").write_text(approve_transaction_yaml)

    return PromptManager(template_dir=str(tmp_path))


@pytest.fixture
def response_parser() -> ResponseParser:
    """Provide a REAL ``ResponseParser`` instance.

    The parser has no external dependencies and is safe for direct use
    in unit tests for JSON extraction and schema validation testing.

    Returns:
        A ``ResponseParser`` with default configuration (max_retries=3).
    """
    return ResponseParser(max_retries=3)


# =========================================================================
# Section 4: Event System Fixtures
# =========================================================================


@pytest.fixture
def event_bus() -> EventBus:
    """Provide a REAL ``EventBus`` in in-memory mode (asyncio.Queue).

    No Redis is required — the bus operates purely with in-process
    asyncio primitives, which is ideal for unit testing per AAP
    Section 0.7.5.

    Returns:
        An ``EventBus`` instance in IN_MEMORY mode.
    """
    return EventBus()


@pytest_asyncio.fixture
async def event_store(
    mock_async_redis: fakeredis.aioredis.FakeRedis,
) -> AsyncGenerator[EventStore, None]:
    """Provide a REAL ``EventStore`` backed by fakeredis.

    The store is created with the injected async fakeredis client,
    enabling integration-level event persistence tests without a real
    Redis server.

    Args:
        mock_async_redis: Injected async fakeredis fixture.

    Yields:
        An ``EventStore`` instance connected to fakeredis.
    """
    store = EventStore(redis_client=mock_async_redis)
    yield store


# =========================================================================
# Section 5: Work Item / Work Result Fixtures
# =========================================================================


@pytest.fixture
def sample_work_item() -> WorkItem:
    """Provide a standard ``WorkItem`` for vendor invoice processing.

    Uses deterministic UUIDs and realistic field values to enable
    reproducible agent processing tests.

    Returns:
        A ``WorkItem`` with type ``"vendor_invoice"``, priority 5, and
        amount $5,000.00.
    """
    return WorkItem(
        type="vendor_invoice",
        data={
            "invoice_id": "INV-001",
            "vendor_id": "V-001",
            "amount": 5000.00,
            "vendor_name": "Acme Supplies",
            "po_number": "PO-2024-001",
        },
        workflow_id=FIXED_WORKFLOW_UUID,
        description="Process vendor invoice INV-001",
        priority=5,
        amount=5000.00,
    )


@pytest.fixture
def sample_work_item_factory() -> Callable[..., WorkItem]:
    """Provide a factory function for creating customisable ``WorkItem`` instances.

    Returns:
        A callable ``make_work_item(type, data, **kwargs) -> WorkItem``.

    Example::

        item = sample_work_item_factory(
            type="purchase_order",
            data={"po_id": "PO-002"},
            amount=25000.0,
        )
    """

    def make_work_item(
        type: str = "vendor_invoice",
        data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> WorkItem:
        """Build a WorkItem with sensible defaults and optional overrides.

        Args:
            type: Transaction type identifier.
            data: Transaction data payload; defaults to a minimal invoice dict.
            **kwargs: Additional WorkItem fields to override (e.g.,
                workflow_id, description, priority, amount).

        Returns:
            A new ``WorkItem`` instance.
        """
        effective_data = data if data is not None else {
            "invoice_id": "INV-001",
            "vendor_id": "V-001",
            "amount": 5000.00,
        }
        return WorkItem(
            type=type,
            data=effective_data,
            workflow_id=kwargs.pop("workflow_id", uuid4()),
            description=kwargs.pop("description", f"Process {type}"),
            priority=kwargs.pop("priority", 5),
            amount=kwargs.pop("amount", None),
            **kwargs,
        )

    return make_work_item


@pytest.fixture
def sample_work_result() -> WorkResult:
    """Provide a standard successful ``WorkResult``.

    Represents the outcome of a successfully processed vendor invoice
    with a single action taken and no exceptions or approvals needed.

    Returns:
        A ``WorkResult`` with success=True and one completed action.
    """
    return WorkResult(
        success=True,
        actions_taken=["process_vendor_invoice"],
        approval_needed=False,
        has_exceptions=False,
        data={"status": "completed", "gl_account": "2100-00"},
    )


# =========================================================================
# Section 6: Statistical Model Fixtures
# =========================================================================


@pytest.fixture
def random_seed() -> int:
    """Seed numpy and Python random generators for reproducible tests.

    Sets ``numpy.random.seed(42)`` so that all statistical model tests
    produce deterministic outputs, regardless of execution order.

    Per AAP: "numpy for random state seeding for reproducible simulations."

    Returns:
        The seed value ``42``.
    """
    np.random.seed(42)
    return 42


# =========================================================================
# Section 7: Utility / Context Fixtures
# =========================================================================


@pytest.fixture
def sample_company_context() -> Dict[str, Any]:
    """Provide a standard company context dictionary for prompt rendering.

    The context contains fields commonly used across prompt templates
    when assembling the ``{company_name}`` and ``{company_id}`` placeholders.

    Returns:
        A dictionary with ``company_name``, ``company_id``, ``fiscal_year``,
        and ``industry`` keys.
    """
    return {
        "company_name": "Test Corp",
        "company_id": str(FIXED_COMPANY_UUID),
        "fiscal_year": "2025",
        "industry": "Manufacturing",
    }


@pytest.fixture
def sample_transaction_context() -> Dict[str, Any]:
    """Provide a standard transaction context dictionary for prompt testing.

    Contains typical fields required by prompt templates: amounts,
    identifiers, descriptions, and vendor/customer information.

    Returns:
        A dictionary representing a complete transaction context.
    """
    return {
        "transaction_type": "vendor_invoice",
        "transaction_id": "TXN-2025-001",
        "invoice_id": "INV-001",
        "vendor_id": "V-001",
        "vendor_name": "Acme Supplies",
        "amount": 5000.00,
        "currency": "USD",
        "description": "Office supplies Q1 2025",
        "po_number": "PO-2024-001",
        "requester_name": "Test AP Clerk",
        "thoroughness": 0.7,
        "risk_tolerance": 0.5,
        "compliance": 0.8,
    }
