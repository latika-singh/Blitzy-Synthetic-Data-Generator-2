"""
Shared Test Fixtures and Mock Factories for the Project 2 and Project 3 Test Suite.

This module provides the foundational test infrastructure used across all
test modules in the Synthetic ERP Data Generation Platform — Agent &
Orchestration Engine (Project 2) and Transaction Workflows & Discrepancies
(Project 3).

CRITICAL Testing Rules (AAP Section 0.7.5):
    - All LLM integration tests use mocked API responses — NO live API calls.
    - Redis tests use fakeredis in-memory mock — NO external Redis dependency.
    - Async tests use pytest-asyncio with proper event loop management.
    - All agent tests verify state transitions (IDLE → THINKING → ACTING → IDLE).
    - Unit test coverage target: ≥ 80% across all source modules.
    - All financial calculations use Python Decimal (prec=28, ROUND_HALF_UP).

Fixture Categories:
    1. Redis Mocks           — mock_redis, mock_async_redis
    2. Agent System           — sample_agent_config, factory, memory/decision/action mocks
    3. LLM Integration        — config, client mock, monitor, budget, prompt_manager, parser
    4. Event System           — event_bus (in-memory), event_store (fakeredis)
    5. Work Items/Results     — sample_work_item, factory, sample_work_result
    6. Statistical Utilities  — random_seed (numpy deterministic seeding)
    7. Context Utilities      — company context, transaction context
    8. Project 3 Transactions — mock DB session, GL engine, discrepancy injector,
       rework engine, transaction data factories, generation context, discrepancy config
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
# Internal Imports — Transaction System (app.transactions) [Project 3]
# ---------------------------------------------------------------------------
from decimal import Decimal, ROUND_HALF_UP
import random

# Note: P3 modules are imported conditionally or via string references
# to avoid import errors when P3 modules are not yet built.
# For mock-based fixtures, we use AsyncMock and MagicMock throughout.

# ---------------------------------------------------------------------------
# Deterministic UUIDs for Reproducible Tests
# ---------------------------------------------------------------------------
FIXED_AGENT_UUID = UUID("12345678-1234-5678-1234-567812345678")
FIXED_EMPLOYEE_UUID = UUID("22345678-2234-5678-2234-567822345678")
FIXED_COMPANY_UUID = UUID("32345678-3234-5678-3234-567832345678")
FIXED_WORKFLOW_UUID = UUID("42345678-4234-5678-4234-567842345678")
FIXED_SIMULATION_UUID = UUID("52345678-5234-5678-5234-567852345678")

# P3 — Project 3 Fixed UUIDs
FIXED_TRANSACTION_UUID = UUID("62345678-6234-5678-6234-567862345678")
FIXED_PO_UUID = UUID("72345678-7234-5678-7234-567872345678")
FIXED_SO_UUID = UUID("82345678-8234-5678-8234-567882345678")
FIXED_INVOICE_UUID = UUID("92345678-9234-5678-9234-567892345678")
FIXED_JOURNAL_ENTRY_UUID = UUID("a2345678-a234-5678-a234-5678a2345678")
FIXED_DISCREPANCY_UUID = UUID("b2345678-b234-5678-b234-5678b2345678")


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


# =========================================================================
# Section 8: Project 3 — Transaction Workflow Fixtures
# Per AAP Section 0.7.6: Test infrastructure for P2P, O2C, GL, Discrepancy, Rework
# =========================================================================


@pytest.fixture
def mock_db_session():
    """Provide an AsyncMock simulating Project 1's get_session() pattern.

    Simulates the async context manager pattern:
        async with get_session() as session:
            await session.execute(...)
            await session.commit()

    The mock supports: execute(), commit(), rollback(), flush(), refresh(),
    add(), delete(), and get() operations.

    Returns:
        An AsyncMock configured as a database session.
    """
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    session.delete = MagicMock()
    session.get = AsyncMock(return_value=None)

    # Make it work as async context manager
    context_manager = AsyncMock()
    context_manager.__aenter__ = AsyncMock(return_value=session)
    context_manager.__aexit__ = AsyncMock(return_value=False)

    return session


@pytest.fixture
def mock_gl_engine():
    """Provide an AsyncMock of GLPostingEngine with pre-configured responses.

    Pre-configured methods:
        - post_journal_entry() → returns mock journal entry dict with balanced DR/CR
        - validate_balance() → returns True (balanced)
        - check_trial_balance() → returns True (trial balance zero)
        - get_account_balance() → returns Decimal("10000.00")

    Returns:
        An AsyncMock configured as a GL Posting Engine.
    """
    engine = AsyncMock()
    engine.post_journal_entry = AsyncMock(return_value={
        "journal_entry_id": str(FIXED_JOURNAL_ENTRY_UUID),
        "status": "posted",
        "total_debits": Decimal("1000.00"),
        "total_credits": Decimal("1000.00"),
        "is_balanced": True,
    })
    engine.validate_balance = AsyncMock(return_value=True)
    engine.check_trial_balance = AsyncMock(return_value=True)
    engine.get_account_balance = AsyncMock(return_value=Decimal("10000.00"))
    engine.get_metrics = MagicMock(return_value={
        "total_entries_posted": 0,
        "total_debits": Decimal("0.00"),
        "total_credits": Decimal("0.00"),
    })
    return engine


@pytest.fixture
def mock_discrepancy_injector():
    """Provide an AsyncMock of DiscrepancyInjector with configurable behavior.

    Pre-configured with:
        - injection_rate: 0.02 (2% default)
        - should_inject() → returns False by default
        - inject() → returns (original_transaction, None) by default
        - get_metrics() → returns injection stats

    Returns:
        An AsyncMock configured as a Discrepancy Injector.
    """
    injector = AsyncMock()
    injector.injection_rate = 0.02
    injector.should_inject = MagicMock(return_value=False)
    injector.inject = AsyncMock(side_effect=lambda txn, ctx: (txn, None))
    injector.get_metrics = MagicMock(return_value={
        "total_checked": 0,
        "total_injected": 0,
        "injection_rate": 0.02,
        "types_injected": {},
    })
    return injector


@pytest.fixture
def mock_rework_engine():
    """Provide an AsyncMock of ReworkLoopEngine with configurable behavior.

    Pre-configured with:
        - max_attempts: 3
        - timeout_seconds: 30.0
        - process_validation_failure() → returns resolved ReworkResult mock

    Returns:
        An AsyncMock configured as a Rework Loop Engine.
    """
    engine = AsyncMock()
    engine.max_attempts = 3
    engine.timeout_seconds = 30.0
    engine.process_validation_failure = AsyncMock(return_value={
        "transaction_id": str(FIXED_TRANSACTION_UUID),
        "resolved": True,
        "escalated": False,
        "attempts": [],
        "final_status": "resolved",
        "total_duration_ms": 150.0,
    })
    engine.get_metrics = MagicMock(return_value={
        "total_processed": 0,
        "total_resolved": 0,
        "total_escalated": 0,
        "total_failed": 0,
        "resolution_rate": 0.0,
    })
    return engine


# =========================================================================
# Section 8.5: Sample Transaction Data Factories
# Factory fixtures for all 8 transaction types — each returns a callable
# that produces sample data dicts with keyword-override support.
# =========================================================================


@pytest.fixture
def sample_purchase_order() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Purchase Order data.

    Returns:
        A callable that creates PO header/lines dicts with vendor, products, amounts.
        Supports keyword overrides for any field.
    """
    def _factory(**overrides: Any) -> Dict[str, Any]:
        po = {
            "transaction_id": str(FIXED_PO_UUID),
            "transaction_type": "purchase_order",
            "po_number": "PO-2025-0001",
            "vendor_id": "V-001",
            "vendor_name": "Acme Supplies",
            "order_date": "2025-01-15",
            "expected_delivery_date": "2025-02-15",
            "status": "approved",
            "currency": "USD",
            "total_amount": Decimal("5000.00"),
            "lines": [
                {
                    "line_number": 1,
                    "product_id": "PROD-001",
                    "description": "Office Supplies",
                    "quantity": Decimal("100"),
                    "unit_price": Decimal("25.00"),
                    "line_total": Decimal("2500.00"),
                },
                {
                    "line_number": 2,
                    "product_id": "PROD-002",
                    "description": "Computer Equipment",
                    "quantity": Decimal("5"),
                    "unit_price": Decimal("500.00"),
                    "line_total": Decimal("2500.00"),
                },
            ],
            "approval_chain": [
                {"approver_role": "purchasing_manager", "approved": True}
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        po.update(overrides)
        return po
    return _factory


@pytest.fixture
def sample_goods_receipt() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Goods Receipt data linked to a PO."""
    def _factory(**overrides: Any) -> Dict[str, Any]:
        receipt = {
            "transaction_id": str(uuid4()),
            "transaction_type": "goods_receipt",
            "receipt_number": "GR-2025-0001",
            "po_number": "PO-2025-0001",
            "po_id": str(FIXED_PO_UUID),
            "vendor_id": "V-001",
            "receipt_date": "2025-02-10",
            "status": "received",
            "lines": [
                {
                    "line_number": 1,
                    "product_id": "PROD-001",
                    "quantity_received": Decimal("100"),
                    "quantity_ordered": Decimal("100"),
                },
                {
                    "line_number": 2,
                    "product_id": "PROD-002",
                    "quantity_received": Decimal("5"),
                    "quantity_ordered": Decimal("5"),
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        receipt.update(overrides)
        return receipt
    return _factory


@pytest.fixture
def sample_vendor_invoice() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Vendor Invoice data linked to a PO with AP coding."""
    def _factory(**overrides: Any) -> Dict[str, Any]:
        invoice = {
            "transaction_id": str(FIXED_INVOICE_UUID),
            "transaction_type": "vendor_invoice",
            "invoice_number": "VINV-2025-0001",
            "po_number": "PO-2025-0001",
            "po_id": str(FIXED_PO_UUID),
            "vendor_id": "V-001",
            "vendor_name": "Acme Supplies",
            "invoice_date": "2025-02-15",
            "due_date": "2025-03-17",
            "payment_terms": "Net 30",
            "status": "pending_match",
            "currency": "USD",
            "total_amount": Decimal("5000.00"),
            "tax_amount": Decimal("0.00"),
            "lines": [
                {
                    "line_number": 1,
                    "product_id": "PROD-001",
                    "description": "Office Supplies",
                    "quantity": Decimal("100"),
                    "unit_price": Decimal("25.00"),
                    "line_total": Decimal("2500.00"),
                    "gl_account": "6100-00",
                },
                {
                    "line_number": 2,
                    "product_id": "PROD-002",
                    "description": "Computer Equipment",
                    "quantity": Decimal("5"),
                    "unit_price": Decimal("500.00"),
                    "line_total": Decimal("2500.00"),
                    "gl_account": "1500-00",
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        invoice.update(overrides)
        return invoice
    return _factory


@pytest.fixture
def sample_sales_order() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Sales Order data with customer, products."""
    def _factory(**overrides: Any) -> Dict[str, Any]:
        so = {
            "transaction_id": str(FIXED_SO_UUID),
            "transaction_type": "sales_order",
            "so_number": "SO-2025-0001",
            "customer_id": "C-001",
            "customer_name": "Beta Corp",
            "order_date": "2025-01-20",
            "expected_ship_date": "2025-02-01",
            "status": "confirmed",
            "currency": "USD",
            "total_amount": Decimal("8000.00"),
            "credit_limit": Decimal("50000.00"),
            "current_ar_balance": Decimal("12000.00"),
            "lines": [
                {
                    "line_number": 1,
                    "product_id": "PROD-010",
                    "description": "Widget A",
                    "quantity": Decimal("200"),
                    "unit_price": Decimal("30.00"),
                    "line_total": Decimal("6000.00"),
                },
                {
                    "line_number": 2,
                    "product_id": "PROD-011",
                    "description": "Widget B",
                    "quantity": Decimal("40"),
                    "unit_price": Decimal("50.00"),
                    "line_total": Decimal("2000.00"),
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        so.update(overrides)
        return so
    return _factory


@pytest.fixture
def sample_shipment() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Shipment data linked to a SO."""
    def _factory(**overrides: Any) -> Dict[str, Any]:
        shipment = {
            "transaction_id": str(uuid4()),
            "transaction_type": "shipment",
            "shipment_number": "SHP-2025-0001",
            "so_number": "SO-2025-0001",
            "so_id": str(FIXED_SO_UUID),
            "customer_id": "C-001",
            "ship_date": "2025-02-01",
            "carrier": "FedEx",
            "tracking_number": "TRACK-123456789",
            "status": "shipped",
            "lines": [
                {
                    "line_number": 1,
                    "product_id": "PROD-010",
                    "quantity_shipped": Decimal("200"),
                    "quantity_ordered": Decimal("200"),
                },
                {
                    "line_number": 2,
                    "product_id": "PROD-011",
                    "quantity_shipped": Decimal("40"),
                    "quantity_ordered": Decimal("40"),
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        shipment.update(overrides)
        return shipment
    return _factory


@pytest.fixture
def sample_customer_invoice() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Customer Invoice linked to a shipment."""
    def _factory(**overrides: Any) -> Dict[str, Any]:
        invoice = {
            "transaction_id": str(uuid4()),
            "transaction_type": "customer_invoice",
            "invoice_number": "INV-2025-0001",
            "so_number": "SO-2025-0001",
            "so_id": str(FIXED_SO_UUID),
            "customer_id": "C-001",
            "customer_name": "Beta Corp",
            "invoice_date": "2025-02-02",
            "due_date": "2025-03-04",
            "payment_terms": "Net 30",
            "status": "outstanding",
            "currency": "USD",
            "total_amount": Decimal("8000.00"),
            "lines": [
                {
                    "line_number": 1,
                    "product_id": "PROD-010",
                    "description": "Widget A",
                    "quantity": Decimal("200"),
                    "unit_price": Decimal("30.00"),
                    "line_total": Decimal("6000.00"),
                    "gl_account": "4100-00",
                },
                {
                    "line_number": 2,
                    "product_id": "PROD-011",
                    "description": "Widget B",
                    "quantity": Decimal("40"),
                    "unit_price": Decimal("50.00"),
                    "line_total": Decimal("2000.00"),
                    "gl_account": "4100-00",
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        invoice.update(overrides)
        return invoice
    return _factory


@pytest.fixture
def sample_journal_entry() -> Callable[..., Dict[str, Any]]:
    """Factory for sample balanced Journal Entry with debit/credit lines.

    Generates a balanced JE where SUM(debits) = SUM(credits) = $1000.00.
    Per AAP Section 0.7.2: GL Balance Invariant within $0.01 tolerance.
    """
    def _factory(**overrides: Any) -> Dict[str, Any]:
        je = {
            "journal_entry_id": str(FIXED_JOURNAL_ENTRY_UUID),
            "transaction_type": "journal_entry",
            "je_number": "JE-2025-0001",
            "posting_date": "2025-01-31",
            "fiscal_period": "2025-01",
            "description": "Monthly accrual entry",
            "status": "posted",
            "currency": "USD",
            "total_debits": Decimal("1000.00"),
            "total_credits": Decimal("1000.00"),
            "lines": [
                {
                    "line_number": 1,
                    "account_code": "6100-00",
                    "account_name": "Office Supplies Expense",
                    "debit": Decimal("1000.00"),
                    "credit": Decimal("0.00"),
                    "description": "Accrued supplies expense",
                },
                {
                    "line_number": 2,
                    "account_code": "2100-00",
                    "account_name": "Accounts Payable",
                    "debit": Decimal("0.00"),
                    "credit": Decimal("1000.00"),
                    "description": "Accrued AP",
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        je.update(overrides)
        return je
    return _factory


@pytest.fixture
def sample_payment() -> Callable[..., Dict[str, Any]]:
    """Factory for sample Payment with allocations."""
    def _factory(**overrides: Any) -> Dict[str, Any]:
        payment = {
            "transaction_id": str(uuid4()),
            "transaction_type": "vendor_payment",
            "payment_number": "PAY-2025-0001",
            "vendor_id": "V-001",
            "vendor_name": "Acme Supplies",
            "payment_date": "2025-03-15",
            "payment_method": "ACH",
            "total_amount": Decimal("5000.00"),
            "discount_amount": Decimal("0.00"),
            "net_amount": Decimal("5000.00"),
            "status": "completed",
            "currency": "USD",
            "allocations": [
                {
                    "invoice_id": str(FIXED_INVOICE_UUID),
                    "invoice_number": "VINV-2025-0001",
                    "allocated_amount": Decimal("5000.00"),
                    "discount_taken": Decimal("0.00"),
                },
            ],
            "simulation_id": str(FIXED_SIMULATION_UUID),
        }
        payment.update(overrides)
        return payment
    return _factory


# =========================================================================
# Section 8.6: Deterministic Seeding Helpers
# =========================================================================


@pytest.fixture
def deterministic_rng():
    """Provide a seeded random.Random(42) instance for reproducible tests.

    Per AAP Section 0.7.1: All random operations MUST use seeded random.Random
    instances. This fixture provides a pre-seeded RNG that produces identical
    sequences across test runs.

    Returns:
        A seeded ``random.Random(42)`` instance.
    """
    return random.Random(42)


# =========================================================================
# Section 8.7: Generation Context Factory
# =========================================================================


@pytest.fixture
def sample_generation_context() -> Callable[..., Dict[str, Any]]:
    """Factory for GenerationContext-compatible dicts.

    Creates context dicts carrying simulation_id, current_date, fiscal_period,
    discrepancy_config, rng_seed — all fields required by the GenerationContext
    Pydantic model in app.transactions.base_generator.

    Returns:
        A callable that creates generation context dicts. Supports keyword overrides.
    """
    def _factory(**overrides: Any) -> Dict[str, Any]:
        ctx = {
            "simulation_id": str(FIXED_SIMULATION_UUID),
            "trace_id": str(uuid4()),
            "current_date": "2025-01-15",
            "fiscal_period": "2025-01",
            "fiscal_year": "2025",
            "company_id": str(FIXED_COMPANY_UUID),
            "rng_seed": 42,
            "discrepancy_config": {
                "enabled": True,
                "injection_rate": 0.02,
                "difficulty_distribution": {
                    "easy": 0.70,
                    "medium": 0.30,
                    "hard": 0.00,
                },
                "auto_adjust_to_bounds": True,
            },
            "rework_config": {
                "enabled": True,
                "max_attempts": 3,
                "timeout_seconds": 30.0,
                "escalation_threshold": 0.05,
            },
        }
        ctx.update(overrides)
        return ctx
    return _factory


# =========================================================================
# Section 8.8: Discrepancy Config Fixtures
# =========================================================================


@pytest.fixture
def sample_discrepancy_config() -> Dict[str, Any]:
    """Provide a standard discrepancy injection configuration.

    Per AAP Section 0.7.5:
        - Rate: 2% (±1% tolerance)
        - Distribution: Easy 70%, Medium 30%, Hard 0%
        - Auto-adjust: True
    """
    return {
        "enabled": True,
        "injection_rate": 0.02,
        "difficulty_distribution": {
            "easy": 0.70,
            "medium": 0.30,
            "hard": 0.00,
        },
        "auto_adjust_to_bounds": True,
        "parameter_bounds": {
            "duplicate_invoice_days_apart": {"min": 1, "max": 90},
            "price_variance_percent": {"min": 0.01, "max": 0.50},
            "quantity_variance_percent": {"min": 0.01, "max": 0.30},
        },
    }


@pytest.fixture
def sample_discrepancy_record() -> Callable[..., Dict[str, Any]]:
    """Factory for sample discrepancy records (ground truth).

    Generates records matching the 16-field ground truth schema from
    app.discrepancies.ground_truth_generator.

    Returns:
        A callable that creates discrepancy record dicts.
    """
    def _factory(**overrides: Any) -> Dict[str, Any]:
        record = {
            "discrepancy_id": str(FIXED_DISCREPANCY_UUID),
            "type_code": "P2P-001",
            "category": "p2p",
            "difficulty": "easy",
            "description": "Duplicate Invoice",
            "transaction_ids": [str(FIXED_INVOICE_UUID)],
            "affected_fields": ["invoice_number", "amount"],
            "original_values": {"invoice_number": "VINV-2025-0001"},
            "modified_values": {"invoice_number": "VINV-2025-0001-DUP"},
            "detection_method": "duplicate_check",
            "financial_impact": str(Decimal("5000.00")),
            "injection_timestamp": "2025-01-15T10:00:00Z",
            "simulation_id": str(FIXED_SIMULATION_UUID),
            "fiscal_period": "2025-01",
            "is_within_bounds": True,
            "parameters": {"days_apart": 5, "amount_variation_pct": 0.0},
        }
        record.update(overrides)
        return record
    return _factory
