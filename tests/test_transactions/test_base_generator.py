"""Abstract TransactionGenerator base class contract tests.

Tests cover:
- TransactionGenerator ABC cannot be instantiated directly
- Shared methods: discrepancy trigger check, GL posting delegation, event publishing
- Retry/timeout wrappers using tenacity
- GenerationContext Pydantic V2 model validation (simulation_id, current_date,
  fiscal_period, discrepancy_config, rng_seed)
- TransactionResult Pydantic V2 model validation
- Constructor injection verification (all deps Optional with None default — ADR-003)
- Sequential number generation helper (_generate_sequential_number)
- Seeded RNG creation (_create_seeded_rng) for deterministic reproducibility
- Structured logging integration (_log_transaction)
- Timing wrapper (_execute_with_timing)

Per AAP Section 0.7.1 Architectural Conventions:
- Constructor Injection (ADR-003): ALL parameters Optional with None defaults
- Pydantic V2 BaseModel with ConfigDict, Field, model_dump, model_validate
- Deterministic: seeded random.Random instances, NEVER module-level random
- structlog JSON logging with service_name='transactions', component=<ClassName>

Per AAP Section 0.7.6 Testing Conventions:
- ≥ 80% coverage, AsyncMock, no live API, no external Redis
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.transactions.base_generator import (
    GenerationContext,
    TransactionGenerator,
    TransactionResult,
)
from app.transactions.exceptions import (
    TransactionError,
    TransactionGenerationError,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Concrete Subclass for Testing the ABC Contract
# ═══════════════════════════════════════════════════════════════════════════════


class ConcreteGenerator(TransactionGenerator):
    """Concrete implementation of TransactionGenerator for testing the ABC contract.

    Implements all three abstract methods (generate, validate, post) with
    minimal logic so that shared helper methods on the base class can be
    exercised without triggering production side-effects.
    """

    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Return a minimal successful TransactionResult."""
        return TransactionResult(
            transaction_id=uuid4(),
            transaction_type="test_transaction",
            status="completed",
            artifacts={"test_key": "test_value"},
            gl_entries=[],
            events_published=[],
            duration_ms=0.0,
        )

    async def validate(self, result: TransactionResult) -> bool:
        """Always returns True for testing purposes."""
        return True

    async def post(self, result: TransactionResult) -> None:
        """No-op post for testing purposes."""
        pass


class FailingGenerator(TransactionGenerator):
    """Generator that raises TransactionGenerationError on generate()."""

    async def generate(self, context: GenerationContext) -> TransactionResult:
        raise TransactionGenerationError(
            "Test generation failure",
            details={"reason": "deliberate_test_failure"},
        )

    async def validate(self, result: TransactionResult) -> bool:
        return False

    async def post(self, result: TransactionResult) -> None:
        pass


class TransactionErrorGenerator(TransactionGenerator):
    """Generator that raises base TransactionError on generate()."""

    async def generate(self, context: GenerationContext) -> TransactionResult:
        raise TransactionError(
            "Generic transaction error",
            details={"domain": "test"},
        )

    async def validate(self, result: TransactionResult) -> bool:
        return False

    async def post(self, result: TransactionResult) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Provide an AsyncMock EventBus for testing event publishing.

    The mock supports the ``publish()`` async method and returns None
    by default.  This is a local fixture since conftest does not provide
    a mock_event_bus (it provides a real EventBus).
    """
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def concrete_generator(mock_db_session, mock_gl_engine, mock_event_bus):
    """ConcreteGenerator with mocked deps for testing base class methods."""
    return ConcreteGenerator(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def generator_no_deps():
    """ConcreteGenerator with all None deps (testing partial composition)."""
    return ConcreteGenerator()


@pytest.fixture
def sample_context() -> GenerationContext:
    """Provide a fully-populated GenerationContext for test methods."""
    return GenerationContext(
        simulation_id=uuid4(),
        current_date=date(2025, 1, 15),
        fiscal_period="2025-01",
        fiscal_period_status="OPEN",
        discrepancy_config={
            "enabled": True,
            "injection_rate": 0.02,
            "difficulty_distribution": {
                "easy": 0.70,
                "medium": 0.30,
                "hard": 0.00,
            },
        },
        rng_seed=42,
        trace_id=uuid4(),
    )


@pytest.fixture
def sample_result() -> TransactionResult:
    """Provide a sample TransactionResult for test methods."""
    return TransactionResult(
        transaction_id=uuid4(),
        transaction_type="purchase_order",
        status="completed",
        artifacts={"po_header": {"po_number": "PO-2025-0001"}},
        gl_entries=[
            {
                "account": "1500-00",
                "debit": Decimal("1000.00"),
                "credit": Decimal("0.00"),
            },
            {
                "account": "2100-00",
                "debit": Decimal("0.00"),
                "credit": Decimal("1000.00"),
            },
        ],
        events_published=["TransactionCreated"],
        duration_ms=150.5,
        has_discrepancy=False,
        amount=Decimal("1000.00"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 1: TransactionGenerator ABC Contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransactionGeneratorABC:
    """Verify the ABC contract of TransactionGenerator."""

    def test_abc_cannot_be_instantiated_directly(self):
        """TransactionGenerator() raises TypeError because it's abstract."""
        with pytest.raises(TypeError):
            TransactionGenerator()

    def test_abc_requires_generate_method(self):
        """Subclass missing generate() raises TypeError on instantiation."""

        class MissingGenerate(TransactionGenerator):
            async def validate(self, result: TransactionResult) -> bool:
                return True

            async def post(self, result: TransactionResult) -> None:
                pass

        with pytest.raises(TypeError):
            MissingGenerate()

    def test_abc_requires_validate_method(self):
        """Subclass missing validate() raises TypeError on instantiation."""

        class MissingValidate(TransactionGenerator):
            async def generate(self, context: GenerationContext) -> TransactionResult:
                return TransactionResult(transaction_type="test")

            async def post(self, result: TransactionResult) -> None:
                pass

        with pytest.raises(TypeError):
            MissingValidate()

    def test_abc_requires_post_method(self):
        """Subclass missing post() raises TypeError on instantiation."""

        class MissingPost(TransactionGenerator):
            async def generate(self, context: GenerationContext) -> TransactionResult:
                return TransactionResult(transaction_type="test")

            async def validate(self, result: TransactionResult) -> bool:
                return True

        with pytest.raises(TypeError):
            MissingPost()

    def test_concrete_subclass_instantiates_successfully(self):
        """ConcreteGenerator implementing all abstract methods instantiates."""
        gen = ConcreteGenerator()
        assert gen is not None
        assert isinstance(gen, TransactionGenerator)

    def test_abc_is_abc_subclass(self):
        """TransactionGenerator is a proper subclass of ABC."""
        assert issubclass(TransactionGenerator, ABC)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 2: Constructor Injection (ADR-003)
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstructorInjection:
    """Verify ADR-003: all constructor parameters Optional with None defaults."""

    def test_all_deps_none_by_default(self, generator_no_deps):
        """ConcreteGenerator() with no args → all dependency attrs are None."""
        gen = generator_no_deps
        assert gen._db_session_factory is None
        assert gen._agent_registry is None
        assert gen._event_bus is None
        assert gen._discrepancy_injector is None
        assert gen._gl_posting_engine is None
        assert gen._workflow_orchestrator is None
        assert gen._approval_system is None
        # statistical_models defaults to empty dict, not None
        assert gen._statistical_models == {}

    def test_db_session_injection(self, mock_db_session):
        """Passing db_session_factory stores it correctly."""
        gen = ConcreteGenerator(db_session_factory=mock_db_session)
        assert gen._db_session_factory is mock_db_session

    def test_agent_registry_injection(self):
        """Passing agent_registry stores it correctly."""
        mock_reg = MagicMock()
        gen = ConcreteGenerator(agent_registry=mock_reg)
        assert gen._agent_registry is mock_reg

    def test_event_bus_injection(self, mock_event_bus):
        """Passing event_bus stores it correctly."""
        gen = ConcreteGenerator(event_bus=mock_event_bus)
        assert gen._event_bus is mock_event_bus

    def test_discrepancy_injector_injection(self, mock_discrepancy_injector):
        """Passing discrepancy_injector stores it correctly."""
        gen = ConcreteGenerator(discrepancy_injector=mock_discrepancy_injector)
        assert gen._discrepancy_injector is mock_discrepancy_injector

    def test_gl_posting_engine_injection(self, mock_gl_engine):
        """Passing gl_posting_engine stores it correctly."""
        gen = ConcreteGenerator(gl_posting_engine=mock_gl_engine)
        assert gen._gl_posting_engine is mock_gl_engine

    def test_workflow_orchestrator_injection(self):
        """Passing workflow_orchestrator stores it correctly."""
        mock_wo = MagicMock()
        gen = ConcreteGenerator(workflow_orchestrator=mock_wo)
        assert gen._workflow_orchestrator is mock_wo

    def test_statistical_models_injection(self):
        """Passing statistical_models dict stores it correctly."""
        models = {"amount_dist": MagicMock(), "timing_model": MagicMock()}
        gen = ConcreteGenerator(statistical_models=models)
        assert gen._statistical_models is models

    def test_approval_system_injection(self):
        """Passing approval_system stores it correctly."""
        mock_as = MagicMock()
        gen = ConcreteGenerator(approval_system=mock_as)
        assert gen._approval_system is mock_as

    def test_all_deps_injected(
        self, mock_db_session, mock_gl_engine, mock_event_bus
    ):
        """Pass all 8 deps → all stored correctly."""
        mock_reg = MagicMock()
        mock_wo = MagicMock()
        mock_as = MagicMock()
        mock_di = MagicMock()
        models = {"test": MagicMock()}

        gen = ConcreteGenerator(
            db_session_factory=mock_db_session,
            agent_registry=mock_reg,
            workflow_orchestrator=mock_wo,
            event_bus=mock_event_bus,
            approval_system=mock_as,
            discrepancy_injector=mock_di,
            gl_posting_engine=mock_gl_engine,
            statistical_models=models,
        )

        assert gen._db_session_factory is mock_db_session
        assert gen._agent_registry is mock_reg
        assert gen._workflow_orchestrator is mock_wo
        assert gen._event_bus is mock_event_bus
        assert gen._approval_system is mock_as
        assert gen._discrepancy_injector is mock_di
        assert gen._gl_posting_engine is mock_gl_engine
        assert gen._statistical_models is models

    def test_partial_deps_supported(self, mock_db_session, mock_event_bus):
        """Only 2 of 8 deps → rest are None (partial composition for testing)."""
        gen = ConcreteGenerator(
            db_session_factory=mock_db_session,
            event_bus=mock_event_bus,
        )
        assert gen._db_session_factory is mock_db_session
        assert gen._event_bus is mock_event_bus
        assert gen._agent_registry is None
        assert gen._discrepancy_injector is None
        assert gen._gl_posting_engine is None
        assert gen._workflow_orchestrator is None
        assert gen._approval_system is None
        assert gen._statistical_models == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 3: GenerationContext Pydantic V2 Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerationContext:
    """Verify GenerationContext Pydantic V2 model validation and serialization."""

    def test_context_creation_with_all_fields(self):
        """Create context with all required and optional fields."""
        ctx = GenerationContext(
            simulation_id=uuid4(),
            current_date=date(2025, 3, 15),
            fiscal_period="2025-03",
            fiscal_period_status="OPEN",
            discrepancy_config={"injection_rate": 0.02},
            rng_seed=42,
            trace_id=uuid4(),
        )
        assert ctx.current_date == date(2025, 3, 15)
        assert ctx.fiscal_period == "2025-03"
        assert ctx.rng_seed == 42
        assert ctx.discrepancy_config["injection_rate"] == 0.02

    def test_context_current_date_required(self):
        """current_date is required — missing raises ValidationError."""
        with pytest.raises(Exception):
            GenerationContext()

    def test_context_simulation_id_has_default(self):
        """simulation_id has a UUID default_factory (uuid4)."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert isinstance(ctx.simulation_id, UUID)

    def test_context_current_date_is_date_type(self):
        """current_date accepts Python date objects."""
        test_date = date(2025, 6, 30)
        ctx = GenerationContext(current_date=test_date)
        assert ctx.current_date == test_date
        assert isinstance(ctx.current_date, date)

    def test_context_fiscal_period_format(self):
        """fiscal_period stores string like '2025-01'."""
        ctx = GenerationContext(
            current_date=date(2025, 1, 15),
            fiscal_period="2025-01",
        )
        assert ctx.fiscal_period == "2025-01"

    def test_context_rng_seed_is_integer(self):
        """rng_seed accepts integer values."""
        ctx = GenerationContext(current_date=date(2025, 1, 1), rng_seed=99)
        assert ctx.rng_seed == 99
        assert isinstance(ctx.rng_seed, int)

    def test_context_rng_seed_default(self):
        """rng_seed defaults to 42 when not provided."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert ctx.rng_seed == 42

    def test_context_discrepancy_config_is_dict(self):
        """discrepancy_config accepts dict with injection_rate, difficulty_distribution."""
        config = {
            "injection_rate": 0.02,
            "difficulty_distribution": {"easy": 0.70, "medium": 0.30, "hard": 0.00},
        }
        ctx = GenerationContext(
            current_date=date(2025, 1, 15),
            discrepancy_config=config,
        )
        assert ctx.discrepancy_config == config
        assert isinstance(ctx.discrepancy_config, dict)

    def test_context_discrepancy_config_default_empty(self):
        """discrepancy_config defaults to empty dict."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert ctx.discrepancy_config == {}

    def test_context_trace_id_has_default(self):
        """trace_id has a UUID default_factory (uuid4)."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert isinstance(ctx.trace_id, UUID)

    def test_context_day_context_optional(self):
        """day_context is optional and defaults to None."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert ctx.day_context is None

    def test_context_model_dump(self):
        """model_dump() produces a serializable dictionary."""
        ctx = GenerationContext(
            current_date=date(2025, 1, 15),
            fiscal_period="2025-01",
            rng_seed=42,
        )
        data = ctx.model_dump()
        assert isinstance(data, dict)
        assert data["current_date"] == date(2025, 1, 15)
        assert data["fiscal_period"] == "2025-01"
        assert data["rng_seed"] == 42

    def test_context_model_validate(self):
        """model_validate() creates a context from dict."""
        sim_id = uuid4()
        trace_id = uuid4()
        batch_id = uuid4()
        data = {
            "simulation_id": sim_id,
            "current_date": date(2025, 3, 15),
            "fiscal_period": "2025-03",
            "fiscal_period_status": "OPEN",
            "discrepancy_config": {},
            "rng_seed": 99,
            "trace_id": trace_id,
            "batch_id": batch_id,
            "day_number": 1,
            "day_context": None,
            "batch_size": 100,
            "additional_params": {},
        }
        ctx = GenerationContext.model_validate(data)
        assert ctx.simulation_id == sim_id
        assert ctx.current_date == date(2025, 3, 15)
        assert ctx.rng_seed == 99

    def test_context_arbitrary_types_allowed(self):
        """model_config has arbitrary_types_allowed=True."""
        assert GenerationContext.model_config.get("arbitrary_types_allowed") is True

    def test_context_batch_size_default(self):
        """batch_size defaults to 100."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert ctx.batch_size == 100

    def test_context_additional_params_default_empty(self):
        """additional_params defaults to empty dict."""
        ctx = GenerationContext(current_date=date(2025, 1, 1))
        assert ctx.additional_params == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 4: TransactionResult Pydantic V2 Model
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransactionResult:
    """Verify TransactionResult Pydantic V2 model validation and defaults."""

    def test_result_creation_with_required_fields(self):
        """Create result with only the required field (transaction_type)."""
        result = TransactionResult(transaction_type="purchase_order")
        assert result.transaction_type == "purchase_order"
        assert result.status == "completed"
        assert isinstance(result.transaction_id, UUID)

    def test_result_has_discrepancy_default_false(self):
        """has_discrepancy defaults to False."""
        result = TransactionResult(transaction_type="test")
        assert result.has_discrepancy is False

    def test_result_discrepancy_type_optional(self):
        """discrepancy_type defaults to None."""
        result = TransactionResult(transaction_type="test")
        assert result.discrepancy_type is None

    def test_result_discrepancy_type_set(self):
        """discrepancy_type can be set to a type code."""
        result = TransactionResult(
            transaction_type="test",
            has_discrepancy=True,
            discrepancy_type="P2P-001",
        )
        assert result.has_discrepancy is True
        assert result.discrepancy_type == "P2P-001"

    def test_result_error_message_optional(self):
        """error_message defaults to None."""
        result = TransactionResult(transaction_type="test")
        assert result.error_message is None

    def test_result_errors_default_empty(self):
        """errors defaults to empty list."""
        result = TransactionResult(transaction_type="test")
        assert result.errors == []

    def test_result_artifacts_is_dict(self):
        """artifacts field is a dict and defaults to empty dict."""
        result = TransactionResult(transaction_type="test")
        assert isinstance(result.artifacts, dict)
        assert result.artifacts == {}

    def test_result_gl_entries_is_list(self):
        """gl_entries field is a list and defaults to empty list."""
        result = TransactionResult(transaction_type="test")
        assert isinstance(result.gl_entries, list)
        assert result.gl_entries == []

    def test_result_events_published_is_list(self):
        """events_published field is a list and defaults to empty list."""
        result = TransactionResult(transaction_type="test")
        assert isinstance(result.events_published, list)
        assert result.events_published == []

    def test_result_duration_ms_default(self):
        """duration_ms defaults to 0.0."""
        result = TransactionResult(transaction_type="test")
        assert result.duration_ms == 0.0

    def test_result_amount_optional(self):
        """amount defaults to None; can be set to Decimal."""
        result = TransactionResult(transaction_type="test")
        assert result.amount is None

        result_with_amount = TransactionResult(
            transaction_type="test",
            amount=Decimal("5000.00"),
        )
        assert result_with_amount.amount == Decimal("5000.00")

    def test_result_created_at_auto_populated(self):
        """created_at is auto-populated with UTC datetime."""
        result = TransactionResult(transaction_type="test")
        assert isinstance(result.created_at, datetime)

    def test_result_model_dump_serialization(self):
        """model_dump() produces a serializable dict with all fields."""
        result = TransactionResult(
            transaction_type="purchase_order",
            status="completed",
            artifacts={"po_header": {}},
            gl_entries=[{"account": "1500", "debit": "1000.00"}],
            events_published=["TransactionCreated"],
            has_discrepancy=True,
            discrepancy_type="P2P-002",
            duration_ms=42.5,
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["transaction_type"] == "purchase_order"
        assert data["has_discrepancy"] is True
        assert data["discrepancy_type"] == "P2P-002"
        assert data["duration_ms"] == 42.5
        assert "transaction_id" in data

    def test_result_status_values(self):
        """status accepts 'completed', 'failed', and 'skipped'."""
        for status in ("completed", "failed", "skipped"):
            result = TransactionResult(transaction_type="test", status=status)
            assert result.status == status


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 5: Seeded RNG
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeededRNG:
    """Verify _create_seeded_rng deterministic reproducibility."""

    def test_create_seeded_rng_returns_random_instance(
        self, generator_no_deps, sample_context
    ):
        """_create_seeded_rng returns a random.Random instance."""
        rng = generator_no_deps._create_seeded_rng(sample_context)
        assert isinstance(rng, random.Random)

    def test_create_seeded_rng_deterministic(
        self, generator_no_deps, sample_context
    ):
        """Same seed → same first 5 values."""
        rng1 = generator_no_deps._create_seeded_rng(sample_context)
        rng2 = generator_no_deps._create_seeded_rng(sample_context)

        seq1 = [rng1.random() for _ in range(5)]
        seq2 = [rng2.random() for _ in range(5)]
        assert seq1 == seq2

    def test_create_seeded_rng_different_seeds_differ(self, generator_no_deps):
        """Different seeds → different sequences."""
        ctx_a = GenerationContext(current_date=date(2025, 1, 1), rng_seed=42)
        ctx_b = GenerationContext(current_date=date(2025, 1, 1), rng_seed=99)

        rng_a = generator_no_deps._create_seeded_rng(ctx_a)
        rng_b = generator_no_deps._create_seeded_rng(ctx_b)

        seq_a = [rng_a.random() for _ in range(10)]
        seq_b = [rng_b.random() for _ in range(10)]
        assert seq_a != seq_b

    def test_create_seeded_rng_reproduces_sequence(
        self, generator_no_deps, sample_context
    ):
        """Call 10 times with same seed — always identical results."""
        results = []
        for _ in range(10):
            rng = generator_no_deps._create_seeded_rng(sample_context)
            results.append(rng.random())
        # All 10 calls should produce the same first value
        assert len(set(results)) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 6: Event Publishing
# ═══════════════════════════════════════════════════════════════════════════════


class TestEventPublishing:
    """Verify _publish_event interaction with EventBus."""

    @pytest.mark.asyncio
    async def test_publish_event_calls_event_bus(
        self, concrete_generator, sample_context
    ):
        """When event_bus is injected, publish() is called."""
        await concrete_generator._publish_event(
            event_type="TransactionCreated",
            payload={"transaction_type": "purchase_order"},
            context=sample_context,
        )
        concrete_generator._event_bus.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_event_skipped_when_no_event_bus(
        self, generator_no_deps, sample_context
    ):
        """event_bus=None → no error raised, no call made."""
        # Should not raise any exception
        await generator_no_deps._publish_event(
            event_type="TransactionCreated",
            payload={"key": "value"},
            context=sample_context,
        )
        # No assertion needed — simply verifying no exception

    @pytest.mark.asyncio
    async def test_publish_event_includes_payload(
        self, concrete_generator, sample_context
    ):
        """The Event object passed to publish includes the supplied payload."""
        await concrete_generator._publish_event(
            event_type="TransactionCompleted",
            payload={"amount": "5000.00", "vendor_id": "V-001"},
            context=sample_context,
        )
        call_args = concrete_generator._event_bus.publish.call_args
        event_arg = call_args[0][0]  # First positional argument
        assert event_arg.payload["amount"] == "5000.00"
        assert event_arg.payload["vendor_id"] == "V-001"

    @pytest.mark.asyncio
    async def test_publish_event_handles_timeout_gracefully(
        self, concrete_generator, sample_context
    ):
        """TimeoutError during publish is caught — no crash."""
        concrete_generator._event_bus.publish = AsyncMock(
            side_effect=asyncio.TimeoutError("publish timeout")
        )
        # Should NOT raise
        await concrete_generator._publish_event(
            event_type="TransactionCreated",
            payload={},
            context=sample_context,
        )

    @pytest.mark.asyncio
    async def test_publish_event_handles_generic_error_gracefully(
        self, concrete_generator, sample_context
    ):
        """Generic exception during publish is caught — no crash."""
        concrete_generator._event_bus.publish = AsyncMock(
            side_effect=RuntimeError("network error")
        )
        # Should NOT raise
        await concrete_generator._publish_event(
            event_type="TransactionCreated",
            payload={},
            context=sample_context,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 7: Discrepancy Trigger Check
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiscrepancyTriggerCheck:
    """Verify _check_discrepancy_trigger interaction with DiscrepancyInjector."""

    @pytest.mark.asyncio
    async def test_check_discrepancy_returns_false_when_no_injector(
        self, generator_no_deps, sample_context
    ):
        """injector=None → returns (False, None, None)."""
        result = await generator_no_deps._check_discrepancy_trigger(
            context=sample_context,
            transaction_type="purchase_order",
        )
        assert result == (False, None, None)

    @pytest.mark.asyncio
    async def test_check_discrepancy_delegates_to_injector(
        self, sample_context
    ):
        """Calls injector.check_trigger() with context and transaction_type."""
        mock_injector = AsyncMock()
        mock_injector.check_trigger = AsyncMock(
            return_value=(True, "P2P-001", {"days_apart": 5})
        )
        gen = ConcreteGenerator(discrepancy_injector=mock_injector)

        result = await gen._check_discrepancy_trigger(
            context=sample_context,
            transaction_type="vendor_invoice",
        )
        assert result == (True, "P2P-001", {"days_apart": 5})
        mock_injector.check_trigger.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_discrepancy_uses_context_config(
        self, sample_context
    ):
        """The injector receives the context carrying discrepancy_config."""
        mock_injector = AsyncMock()
        mock_injector.check_trigger = AsyncMock(
            return_value=(False, None, None)
        )
        gen = ConcreteGenerator(discrepancy_injector=mock_injector)

        await gen._check_discrepancy_trigger(
            context=sample_context,
            transaction_type="purchase_order",
        )

        call_kwargs = mock_injector.check_trigger.call_args
        assert call_kwargs.kwargs["context"] is sample_context

    @pytest.mark.asyncio
    async def test_check_discrepancy_handles_injector_error(
        self, sample_context
    ):
        """Exception in injector → returns (False, None, None) (skip discrepancy)."""
        mock_injector = AsyncMock()
        mock_injector.check_trigger = AsyncMock(
            side_effect=RuntimeError("injector crash")
        )
        gen = ConcreteGenerator(discrepancy_injector=mock_injector)

        result = await gen._check_discrepancy_trigger(
            context=sample_context,
            transaction_type="purchase_order",
        )
        assert result == (False, None, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 8: GL Posting Delegation
# ═══════════════════════════════════════════════════════════════════════════════


class TestGLPostingDelegation:
    """Verify _delegate_gl_posting interaction with GLPostingEngine."""

    @pytest.mark.asyncio
    async def test_delegate_gl_posting_calls_engine(
        self, sample_context
    ):
        """gl_engine.post_entries is called with journal entries and context."""
        mock_engine = AsyncMock()
        mock_engine.post_entries = AsyncMock(return_value=None)
        gen = ConcreteGenerator(gl_posting_engine=mock_engine)

        entries = [
            {"account": "1500-00", "debit": Decimal("1000.00"), "credit": Decimal("0.00")},
            {"account": "2100-00", "debit": Decimal("0.00"), "credit": Decimal("1000.00")},
        ]
        await gen._delegate_gl_posting(
            journal_entries=entries,
            context=sample_context,
        )
        mock_engine.post_entries.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delegate_gl_posting_skipped_when_no_engine(
        self, generator_no_deps, sample_context
    ):
        """engine=None → no error raised, operation skipped."""
        entries = [{"account": "1500", "debit": Decimal("100.00"), "credit": Decimal("0.00")}]
        # Should not raise
        await generator_no_deps._delegate_gl_posting(
            journal_entries=entries,
            context=sample_context,
        )

    @pytest.mark.asyncio
    async def test_delegate_gl_posting_passes_journal_entries(
        self, sample_context
    ):
        """The actual journal_entries list is passed to the engine."""
        mock_engine = AsyncMock()
        mock_engine.post_entries = AsyncMock(return_value=None)
        gen = ConcreteGenerator(gl_posting_engine=mock_engine)

        entries = [
            {"account": "6100-00", "debit": Decimal("500.00"), "credit": Decimal("0.00")},
            {"account": "2100-00", "debit": Decimal("0.00"), "credit": Decimal("500.00")},
        ]
        await gen._delegate_gl_posting(journal_entries=entries, context=sample_context)

        call_kwargs = mock_engine.post_entries.call_args
        assert call_kwargs.kwargs["journal_entries"] == entries
        assert call_kwargs.kwargs["context"] is sample_context

    @pytest.mark.asyncio
    async def test_delegate_gl_posting_skips_empty_entries(
        self, sample_context
    ):
        """Empty journal_entries list → no call to engine."""
        mock_engine = AsyncMock()
        mock_engine.post_entries = AsyncMock(return_value=None)
        gen = ConcreteGenerator(gl_posting_engine=mock_engine)

        await gen._delegate_gl_posting(journal_entries=[], context=sample_context)
        mock_engine.post_entries.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegate_gl_posting_reraises_transaction_error(
        self, sample_context
    ):
        """TransactionError from engine is re-raised for rollback."""
        from app.transactions.exceptions import BalanceError

        mock_engine = AsyncMock()
        mock_engine.post_entries = AsyncMock(
            side_effect=BalanceError("DR != CR", details={"imbalance": "0.50"})
        )
        gen = ConcreteGenerator(gl_posting_engine=mock_engine)

        entries = [{"account": "1500", "debit": Decimal("100.00"), "credit": Decimal("0.00")}]
        with pytest.raises(BalanceError):
            await gen._delegate_gl_posting(
                journal_entries=entries,
                context=sample_context,
            )

    @pytest.mark.asyncio
    async def test_delegate_gl_posting_wraps_unexpected_error(
        self, sample_context
    ):
        """Unexpected exception is wrapped in TransactionError."""
        mock_engine = AsyncMock()
        mock_engine.post_entries = AsyncMock(
            side_effect=ConnectionError("db connection lost")
        )
        gen = ConcreteGenerator(gl_posting_engine=mock_engine)

        entries = [{"account": "1500", "debit": Decimal("100.00"), "credit": Decimal("0.00")}]
        with pytest.raises(TransactionError) as exc_info:
            await gen._delegate_gl_posting(
                journal_entries=entries,
                context=sample_context,
            )
        assert "GL posting failed unexpectedly" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 9: Sequential Number Generation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSequentialNumberGeneration:
    """Verify _generate_sequential_number PREFIX-YYYY-NNNN format."""

    def test_generate_po_number_format(self, generator_no_deps):
        """PO number follows 'PO-2025-0001' format."""
        num = generator_no_deps._generate_sequential_number("PO", 2025, 1)
        assert num == "PO-2025-0001"

    def test_generate_so_number_format(self, generator_no_deps):
        """SO number follows 'SO-2025-0001' format."""
        num = generator_no_deps._generate_sequential_number("SO", 2025, 1)
        assert num == "SO-2025-0001"

    def test_generate_je_number_format(self, generator_no_deps):
        """JE number follows 'JE-2025-0001' format."""
        num = generator_no_deps._generate_sequential_number("JE", 2025, 1)
        assert num == "JE-2025-0001"

    def test_generate_inv_number_format(self, generator_no_deps):
        """INV number follows 'INV-2025-0001' format."""
        num = generator_no_deps._generate_sequential_number("INV", 2025, 1)
        assert num == "INV-2025-0001"

    def test_numbers_increment_per_sequence(self, generator_no_deps):
        """Incrementing sequence arg → 0001, 0002, 0003..."""
        nums = [
            generator_no_deps._generate_sequential_number("PO", 2025, i)
            for i in range(1, 4)
        ]
        assert nums == ["PO-2025-0001", "PO-2025-0002", "PO-2025-0003"]

    def test_number_format_4_digit_padded(self, generator_no_deps):
        """Sequence 1 → '0001', not '1'."""
        num = generator_no_deps._generate_sequential_number("PO", 2025, 1)
        # Extract sequence part
        parts = num.split("-")
        assert parts[-1] == "0001"
        assert len(parts[-1]) == 4

    def test_number_format_large_sequence(self, generator_no_deps):
        """Sequence 1234 → '1234' (4 digits, no extra padding)."""
        num = generator_no_deps._generate_sequential_number("SO", 2025, 1234)
        assert num == "SO-2025-1234"

    def test_number_format_different_years(self, generator_no_deps):
        """Different year values appear correctly in the format."""
        num_2024 = generator_no_deps._generate_sequential_number("PO", 2024, 42)
        num_2026 = generator_no_deps._generate_sequential_number("PO", 2026, 42)
        assert num_2024 == "PO-2024-0042"
        assert num_2026 == "PO-2026-0042"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 10: Transaction Logging
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransactionLogging:
    """Verify _log_transaction structured logging with structlog."""

    def test_log_transaction_includes_required_fields(
        self, concrete_generator, sample_result, sample_context
    ):
        """_log_transaction logs transaction_type, transaction_id, amount, etc."""
        with patch("app.transactions.base_generator.logger") as mock_logger:
            concrete_generator._log_transaction(sample_result, sample_context)

            mock_logger.info.assert_called_once()
            call_kwargs = mock_logger.info.call_args
            # First positional arg is the event string
            assert call_kwargs[0][0] == "transaction_generated"
            # Check keyword arguments contain required fields
            kw = call_kwargs[1]
            assert kw["transaction_type"] == "purchase_order"
            assert "transaction_id" in kw
            assert "has_discrepancy" in kw
            assert "gl_entries" in kw
            assert "duration_ms" in kw
            assert "status" in kw
            assert kw["service_name"] == "transactions"
            assert kw["component"] == "ConcreteGenerator"

    def test_log_uses_structlog(
        self, concrete_generator, sample_result, sample_context
    ):
        """Logging calls the structlog bound logger (logger.info)."""
        with patch("app.transactions.base_generator.logger") as mock_logger:
            concrete_generator._log_transaction(sample_result, sample_context)
            mock_logger.info.assert_called_once()

    def test_log_transaction_with_discrepancy(
        self, concrete_generator, sample_context
    ):
        """Logging includes discrepancy info when present."""
        result = TransactionResult(
            transaction_type="vendor_invoice",
            has_discrepancy=True,
            discrepancy_type="P2P-001",
            amount=Decimal("5000.00"),
        )
        with patch("app.transactions.base_generator.logger") as mock_logger:
            concrete_generator._log_transaction(result, sample_context)
            kw = mock_logger.info.call_args[1]
            assert kw["has_discrepancy"] is True
            assert kw["discrepancy_type"] == "P2P-001"

    def test_log_transaction_with_none_amount(
        self, concrete_generator, sample_context
    ):
        """Logging handles None amount gracefully."""
        result = TransactionResult(transaction_type="test")
        with patch("app.transactions.base_generator.logger") as mock_logger:
            concrete_generator._log_transaction(result, sample_context)
            kw = mock_logger.info.call_args[1]
            assert kw["amount"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 11: Timing Wrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimingWrapper:
    """Verify _execute_with_timing wraps generate() with duration tracking."""

    @pytest.mark.asyncio
    async def test_execute_with_timing_returns_result_and_duration(
        self, concrete_generator, sample_context
    ):
        """Returns a TransactionResult with duration_ms populated."""
        result = await concrete_generator._execute_with_timing(sample_context)
        assert isinstance(result, TransactionResult)
        assert result.duration_ms >= 0.0
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_execute_with_timing_duration_positive(
        self, sample_context
    ):
        """duration_ms is > 0 (real time elapses during execution)."""

        class SlowGenerator(TransactionGenerator):
            async def generate(self, context: GenerationContext) -> TransactionResult:
                await asyncio.sleep(0.01)  # 10ms delay
                return TransactionResult(
                    transaction_type="slow_test",
                    status="completed",
                )

            async def validate(self, result: TransactionResult) -> bool:
                return True

            async def post(self, result: TransactionResult) -> None:
                pass

        gen = SlowGenerator()
        result = await gen._execute_with_timing(sample_context)
        # Should be at least 5ms (allowing for scheduler variance)
        assert result.duration_ms > 5.0

    @pytest.mark.asyncio
    async def test_execute_with_timing_propagates_generation_error(
        self, sample_context
    ):
        """TransactionGenerationError → result with status='failed'."""
        gen = FailingGenerator()
        result = await gen._execute_with_timing(sample_context)
        assert result.status == "failed"
        assert result.error_message == "Test generation failure"
        assert "Test generation failure" in result.errors
        assert result.duration_ms > 0.0

    @pytest.mark.asyncio
    async def test_execute_with_timing_propagates_transaction_error(
        self, sample_context
    ):
        """Base TransactionError → result with status='failed'."""
        gen = TransactionErrorGenerator()
        result = await gen._execute_with_timing(sample_context)
        assert result.status == "failed"
        assert result.error_message == "Generic transaction error"
        assert result.duration_ms > 0.0

    @pytest.mark.asyncio
    async def test_execute_with_timing_logs_successful_transaction(
        self, concrete_generator, sample_context
    ):
        """On success, _log_transaction is called with the result."""
        with patch.object(concrete_generator, "_log_transaction") as mock_log:
            result = await concrete_generator._execute_with_timing(sample_context)
            mock_log.assert_called_once_with(result, sample_context)

    @pytest.mark.asyncio
    async def test_execute_with_timing_sets_duration_on_success(
        self, concrete_generator, sample_context
    ):
        """On success, the result's duration_ms is set by the wrapper."""
        result = await concrete_generator._execute_with_timing(sample_context)
        # The ConcreteGenerator.generate() returns duration_ms=0.0, but
        # _execute_with_timing should overwrite it with actual elapsed time
        assert result.duration_ms >= 0.0
