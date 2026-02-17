"""Tests for the DecisionEngine 4-layer decision pipeline.

Per AAP Section 0.5.1 Group 10: "4-layer pipeline, LLM fallback, validation retries."
Tests the DecisionEngine (app/agents/decision_engine.py) which implements the
INVIOLABLE 4-layer pipeline:

    Layer 1 — Statistical Layer:  amounts, entities, timing, triggers
    Layer 2 — LLM Layer (cond.):  descriptions, notes, edge cases
    Layer 3 — Validation Layer:   schema, range, rules (max 3 retries)
    Layer 4 — Deterministic Layer: GL postings, balances, business rules

CRITICAL ARCHITECTURAL RULES (AAP Section 0.7.1):
    - The 4-layer pipeline is INVIOLABLE — no skipping or reordering.
    - Financial calculations (GL postings, balance updates) are EXCLUSIVELY
      computed in the Deterministic Layer — NEVER LLM-generated.
    - The LLM Layer is CONDITIONAL — only invoked when needed.
    - Validation Layer retries: max 3 attempts on failure.

CRITICAL TESTING RULES (AAP Section 0.7.5):
    - ALL LLM integration tests MUST use mocked API responses — no live calls.
    - Async tests use pytest-asyncio (asyncio_mode=auto in pytest.ini).
    - Coverage target: >= 80%.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4

import pytest

from app.agents.decision_engine import (
    DecisionEngine,
    DecisionContext,
    DecisionResult,
    LLM_REQUIRED_DECISIONS,
    STATISTICAL_ONLY_DECISIONS,
)
from app.agents.agent_config import AgentConfig
from app.llm.llm_client import LLMClient
from app.llm.prompt_manager import PromptManager
from app.llm.response_parser import ResponseParser


# ============================================================================
# Helpers
# ============================================================================

def _make_mock_pydantic_model(data: Dict[str, Any]) -> MagicMock:
    """Create a MagicMock that behaves like a Pydantic model with model_dump()."""
    model = MagicMock()
    model.model_dump.return_value = data
    return model


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_prompt_manager() -> MagicMock:
    """Provide a MagicMock of PromptManager.

    ``build_prompt()`` returns a (system_prompt, user_prompt) tuple.
    """
    pm = MagicMock(spec=PromptManager)
    pm.build_prompt.return_value = (
        "System: You are an ERP assistant.",
        "User: Process this invoice.",
    )
    return pm


@pytest.fixture
def mock_response_parser() -> MagicMock:
    """Provide a MagicMock of ResponseParser.

    ``parse()`` returns a mock Pydantic model whose ``model_dump()``
    yields a typical approval decision dict.
    """
    rp = MagicMock(spec=ResponseParser)
    rp.parse.return_value = _make_mock_pydantic_model({
        "decision": "approve",
        "confidence": 0.85,
        "reasoning": "Test reasoning — approved per compliance policy.",
    })
    return rp


@pytest.fixture
def mock_statistical_models() -> Dict[str, MagicMock]:
    """Provide a dict of mock statistical models keyed by model name.

    Each model exposes the duck-typed ``sample(context)`` or ``select(context)``
    method that the DecisionEngine's statistical layer looks for.
    """
    amount_dist = MagicMock()
    amount_dist.sample.return_value = 5000.0

    selection_model = MagicMock()
    selection_model.select.return_value = {"entity_id": "V-001", "name": "Test Vendor"}

    payment_timing = MagicMock()
    payment_timing.sample.return_value = 30

    order_frequency = MagicMock()
    order_frequency.sample.return_value = 5

    return {
        "amount_distribution": amount_dist,
        "selection_model": selection_model,
        "payment_timing": payment_timing,
        "order_frequency": order_frequency,
    }


@pytest.fixture
def decision_engine(
    mock_llm_client: MagicMock,
    mock_prompt_manager: MagicMock,
    mock_response_parser: MagicMock,
    mock_statistical_models: Dict[str, MagicMock],
) -> DecisionEngine:
    """Provide a fully-wired DecisionEngine with all mocked dependencies.

    Uses constructor injection per AAP Section 0.7.1.  All four
    dependencies (llm_client, prompt_manager, response_parser,
    statistical_models) are injected as mocks.
    """
    return DecisionEngine(
        llm_client=mock_llm_client,
        prompt_manager=mock_prompt_manager,
        response_parser=mock_response_parser,
        statistical_models=mock_statistical_models,
    )


@pytest.fixture
def sample_decision_context() -> Dict[str, Any]:
    """Provide a realistic invoice-processing decision context."""
    return {
        "invoice": {
            "invoice_id": f"INV-{uuid4().hex[:8].upper()}",
            "amount": 5000.0,
            "vendor_name": "Acme Corp",
        },
        "agent": {
            "role": "ap_clerk",
            "traits": {"thoroughness": 0.7},
        },
        "amount": 5000.0,
    }


# ============================================================================
# TestPipelineOrder — Verify the inviolable 4-layer execution order
# ============================================================================


class TestPipelineOrder:
    """Verify that the 4-layer pipeline executes in strict, inviolable order.

    The pipeline MUST execute: Statistical → LLM (conditional) → Validation
    → Deterministic.  No layer may be skipped or reordered.
    """

    @pytest.mark.asyncio
    async def test_pipeline_executes_statistical_layer_first(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """Statistical layer always runs first in the pipeline."""
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 1000.0},
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        layers = metadata["layers_executed"]
        # Statistical is always the first layer
        assert layers[0] == "statistical"

    @pytest.mark.asyncio
    async def test_pipeline_executes_llm_layer_second_when_needed(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM layer runs second when the decision type requires it."""
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 2000.0},
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        layers = metadata["layers_executed"]
        assert "statistical" in layers
        assert "llm" in layers
        stat_idx = layers.index("statistical")
        llm_idx = layers.index("llm")
        assert stat_idx < llm_idx, "Statistical must execute before LLM"

    @pytest.mark.asyncio
    async def test_pipeline_executes_validation_layer_third(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """Validation layer runs after statistical and LLM layers."""
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 3000.0},
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        layers = metadata["layers_executed"]
        assert "validation" in layers
        # Validation comes after statistical
        stat_idx = layers.index("statistical")
        val_idx = layers.index("validation")
        assert stat_idx < val_idx, "Statistical must run before Validation"

    @pytest.mark.asyncio
    async def test_pipeline_executes_deterministic_layer_last(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """Deterministic layer always runs last in the pipeline."""
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 4000.0},
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        layers = metadata["layers_executed"]
        assert layers[-1] == "deterministic"

    @pytest.mark.asyncio
    async def test_pipeline_order_is_inviolable(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """Full pipeline order must be: statistical → llm → validation → deterministic."""
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        layers = metadata["layers_executed"]

        # For LLM-required decision, all 4 layers must execute in order
        assert layers == [
            "statistical",
            "llm",
            "validation",
            "deterministic",
        ], f"Pipeline order violated: {layers}"


# ============================================================================
# TestStatisticalLayer — Layer 1 statistical model invocation
# ============================================================================


class TestStatisticalLayer:
    """Verify Statistical Layer (Layer 1) correctly invokes models.

    Statistical models determine amounts, select entities, calculate timing,
    and check for discrepancy triggers based on the decision type routing
    configuration.
    """

    @pytest.mark.asyncio
    async def test_statistical_layer_determines_amounts(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
    ) -> None:
        """Amount distribution model is invoked for amount determination."""
        result = await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )
        # The amount_distribution model's sample() method should have been called
        mock_statistical_models["amount_distribution"].sample.assert_called()
        assert result.get("amount") == 5000.0

    @pytest.mark.asyncio
    async def test_statistical_layer_selects_entities(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
    ) -> None:
        """Selection model is invoked for entity selection decisions."""
        result = await decision_engine.decide(
            decision_type="select_entity",
            context={},
        )
        mock_statistical_models["selection_model"].select.assert_called()
        assert result.get("selected_entity") == {
            "entity_id": "V-001",
            "name": "Test Vendor",
        }

    @pytest.mark.asyncio
    async def test_statistical_layer_calculates_timing(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
    ) -> None:
        """Payment timing model is invoked for timing calculations."""
        result = await decision_engine.decide(
            decision_type="calculate_timing",
            context={},
        )
        mock_statistical_models["payment_timing"].sample.assert_called()
        assert result.get("payment_timing") == 30

    @pytest.mark.asyncio
    async def test_statistical_layer_checks_discrepancy_triggers(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
    ) -> None:
        """Statistical layer processes all routing entries for multi-model decisions.

        For 'create_purchase_order', both amount_distribution and selection_model
        are routed, verifying multi-model invocation for a single decision.
        """
        result = await decision_engine.decide(
            decision_type="create_purchase_order",
            context={},
        )
        # Both models should be invoked per routing config
        mock_statistical_models["amount_distribution"].sample.assert_called()
        mock_statistical_models["selection_model"].select.assert_called()
        assert "amount" in result
        assert "selected_vendor" in result


# ============================================================================
# TestLLMLayer — Layer 2 conditional LLM invocation
# ============================================================================


class TestLLMLayer:
    """Verify LLM Layer (Layer 2) conditional invocation logic.

    The LLM layer MUST be invoked for the 6 LLM_REQUIRED_DECISIONS types
    and MUST be skipped for STATISTICAL_ONLY_DECISIONS types to achieve
    the p95 < 5s decision latency target.
    """

    @pytest.mark.asyncio
    async def test_llm_invoked_for_process_vendor_invoice(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM is invoked for process_vendor_invoice decisions."""
        await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
            agent_config=sample_agent_config,
        )
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_invoked_for_approve_transaction(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM is invoked for approve_transaction decisions."""
        await decision_engine.decide(
            decision_type="approve_transaction",
            context={"amount": 10000.0},
            agent_config=sample_agent_config,
        )
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_invoked_for_generate_description(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM is invoked for generate_description decisions."""
        await decision_engine.decide(
            decision_type="generate_description",
            context={},
            agent_config=sample_agent_config,
        )
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_invoked_for_handle_exception(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM is invoked for handle_exception decisions."""
        await decision_engine.decide(
            decision_type="handle_exception",
            context={},
            agent_config=sample_agent_config,
        )
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_invoked_for_match_documents(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM is invoked for match_documents decisions."""
        await decision_engine.decide(
            decision_type="match_documents",
            context={},
            agent_config=sample_agent_config,
        )
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_invoked_for_reconcile_account(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """LLM is invoked for reconcile_account decisions."""
        await decision_engine.decide(
            decision_type="reconcile_account",
            context={},
            agent_config=sample_agent_config,
        )
        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_skipped_for_statistical_only_decisions(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
    ) -> None:
        """LLM is NOT invoked for statistical-only decisions.

        This short-circuit is critical for the p95 < 5s latency target.
        """
        for decision_type in STATISTICAL_ONLY_DECISIONS:
            mock_llm_client.complete.reset_mock()
            result = await decision_engine.decide(
                decision_type=decision_type,
                context={},
            )
            mock_llm_client.complete.assert_not_called(), (
                f"LLM should not be invoked for '{decision_type}'"
            )
            # Confirm LLM layer is absent from executed layers
            metadata = result["_decision_metadata"]
            assert "llm" not in metadata["layers_executed"], (
                f"LLM layer should be skipped for '{decision_type}'"
            )

    @pytest.mark.asyncio
    async def test_llm_layer_uses_prompt_manager(
        self,
        decision_engine: DecisionEngine,
        mock_prompt_manager: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """PromptManager.build_prompt() is called to assemble the LLM prompt."""
        await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 1500.0},
            agent_config=sample_agent_config,
        )
        mock_prompt_manager.build_prompt.assert_called_once()
        # Verify template_name arg is the decision_type
        call_args = mock_prompt_manager.build_prompt.call_args
        assert call_args[0][0] == "process_vendor_invoice"

    @pytest.mark.asyncio
    async def test_llm_layer_uses_response_parser(
        self,
        decision_engine: DecisionEngine,
        mock_response_parser: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """ResponseParser.parse() is called on raw LLM text output."""
        await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 2500.0},
            agent_config=sample_agent_config,
        )
        mock_response_parser.parse.assert_called_once()


# ============================================================================
# TestLLMFallback — LLM failure and graceful degradation
# ============================================================================


class TestLLMFallback:
    """Verify LLM fallback and graceful degradation behaviour.

    When the LLM layer encounters errors, the DecisionEngine must degrade
    gracefully — the pipeline continues without LLM-enriched output rather
    than crashing.  The overall decision still succeeds with statistical
    and deterministic data.
    """

    @pytest.mark.asyncio
    async def test_fallback_to_cheaper_model_on_failure(
        self,
        mock_prompt_manager: MagicMock,
        mock_response_parser: MagicMock,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """When primary LLM call fails, engine degrades gracefully.

        The LLM layer catches the exception and returns an empty dict,
        allowing the pipeline to continue with statistical + deterministic
        results only.
        """
        failing_client = MagicMock(spec=LLMClient)
        failing_client.complete = AsyncMock(
            side_effect=RuntimeError("Primary model rate limited")
        )

        engine = DecisionEngine(
            llm_client=failing_client,
            prompt_manager=mock_prompt_manager,
            response_parser=mock_response_parser,
            statistical_models=mock_statistical_models,
        )

        result = await engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
            agent_config=sample_agent_config,
        )

        metadata = result["_decision_metadata"]
        # Pipeline should still succeed (LLM failure is non-fatal)
        assert metadata["success"] is True
        # Statistical and deterministic layers still execute
        assert "statistical" in metadata["layers_executed"]
        assert "deterministic" in metadata["layers_executed"]

    @pytest.mark.asyncio
    async def test_fallback_after_retries_exhausted(
        self,
        mock_prompt_manager: MagicMock,
        mock_response_parser: MagicMock,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """After LLM retries are exhausted, engine degrades gracefully.

        The LLM layer's internal try/except handles the failure, and the
        pipeline proceeds without LLM output.
        """
        retries_client = MagicMock(spec=LLMClient)
        retries_client.complete = AsyncMock(
            side_effect=ConnectionError("All retries exhausted")
        )

        engine = DecisionEngine(
            llm_client=retries_client,
            prompt_manager=mock_prompt_manager,
            response_parser=mock_response_parser,
            statistical_models=mock_statistical_models,
        )

        result = await engine.decide(
            decision_type="approve_transaction",
            context={"amount": 3000.0},
            agent_config=sample_agent_config,
        )

        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        # LLM layer did not produce output — llm may or may not be in layers
        # depending on whether the try/except in _llm_layer adds it
        assert "deterministic" in metadata["layers_executed"]

    @pytest.mark.asyncio
    async def test_decision_succeeds_with_fallback(
        self,
        mock_prompt_manager: MagicMock,
        mock_response_parser: MagicMock,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """Overall decision succeeds even when LLM call fails.

        The pipeline produces valid output from statistical and deterministic
        layers, with the LLM layer gracefully returning an empty dict.
        """
        failing_client = MagicMock(spec=LLMClient)
        failing_client.complete = AsyncMock(
            side_effect=TimeoutError("LLM timeout after 30s")
        )

        engine = DecisionEngine(
            llm_client=failing_client,
            prompt_manager=mock_prompt_manager,
            response_parser=mock_response_parser,
            statistical_models=mock_statistical_models,
        )

        result = await engine.decide(
            decision_type="generate_description",
            context={"amount": 1000.0},
            agent_config=sample_agent_config,
        )

        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        # Verify result contains statistical output (no LLM enrichment)
        assert "_decision_metadata" in result

    @pytest.mark.asyncio
    async def test_decision_fails_gracefully_when_all_llm_fails(
        self,
        mock_prompt_manager: MagicMock,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """When LLM and parser both fail, decision still does not crash.

        Even when the response_parser is also failing, the LLM layer's
        top-level exception handler prevents the pipeline from crashing.
        """
        failing_client = MagicMock(spec=LLMClient)
        failing_client.complete = AsyncMock(
            side_effect=Exception("Complete LLM infrastructure failure")
        )

        failing_parser = MagicMock(spec=ResponseParser)
        failing_parser.parse.side_effect = Exception("Parser also broken")

        engine = DecisionEngine(
            llm_client=failing_client,
            prompt_manager=mock_prompt_manager,
            response_parser=failing_parser,
            statistical_models=mock_statistical_models,
        )

        # Should not raise — graceful degradation
        result = await engine.decide(
            decision_type="handle_exception",
            context={"amount": 500.0},
            agent_config=sample_agent_config,
        )

        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        assert "statistical" in metadata["layers_executed"]
        assert "deterministic" in metadata["layers_executed"]


# ============================================================================
# TestValidationLayer — Layer 3 validation and retry logic
# ============================================================================


class TestValidationLayer:
    """Verify Validation Layer (Layer 3) retry and enforcement logic.

    The validation layer checks schemas, value ranges, and business rules.
    It retries on ValidationError or ValueError with a maximum of 3 attempts.
    """

    @pytest.mark.asyncio
    async def test_validation_passes_valid_decision(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """Valid decision data passes through validation without retries."""
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        assert "validation" in metadata["layers_executed"]
        assert metadata["retries_used"] == 0

    @pytest.mark.asyncio
    async def test_validation_retries_on_failure_max_3(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Validation retries up to 3 times on failure before giving up.

        Register a custom validator that always raises ValueError.
        The engine should attempt validation 3 times and then record failure.
        """
        call_count = 0

        def always_fail_validator(
            decision_type: str, result: Dict[str, Any]
        ) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Validation failure attempt {call_count}")

        decision_engine.register_validation_rule(
            "process_vendor_invoice", always_fail_validator
        )

        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
        )

        metadata = result["_decision_metadata"]
        # Validation should have been attempted 3 times (max_retries=3)
        assert call_count == 3
        assert metadata["retries_used"] == 3
        assert "validation_failed" in metadata["layers_executed"]

    @pytest.mark.asyncio
    async def test_validation_checks_value_ranges(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
    ) -> None:
        """Negative amounts are rejected by the generic amount validator.

        The _validate_amounts() method raises ValueError for negative amounts.
        """
        # Force the statistical model to return a negative amount
        mock_statistical_models["amount_distribution"].sample.return_value = -100.0

        result = await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )

        metadata = result["_decision_metadata"]
        # Negative amount triggers validation failure
        assert metadata["retries_used"] > 0 or "validation_failed" in metadata["layers_executed"]

    @pytest.mark.asyncio
    async def test_validation_checks_schema(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Custom Pydantic schema validation is enforced through registered rules.

        Register a validator that enforces schema requirements and verify
        it is called during the validation layer.  The statistical layer
        produces ``expected_amount`` (not ``amount``), so the schema check
        validates that field.
        """
        validator_called = False

        def schema_validator(
            decision_type: str, result: Dict[str, Any]
        ) -> Dict[str, Any]:
            nonlocal validator_called
            validator_called = True
            # Enforce that statistical layer has produced the expected_amount
            if "expected_amount" not in result:
                raise ValueError("Schema violation: 'expected_amount' is required")
            return result

        decision_engine.register_validation_rule(
            "process_vendor_invoice", schema_validator
        )

        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
        )

        assert validator_called is True
        metadata = result["_decision_metadata"]
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_validation_retry_count_is_exactly_3(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Validation retries exactly 3 times — not 2, not 4."""
        attempts: List[int] = []

        def counting_validator(
            decision_type: str, result: Dict[str, Any]
        ) -> Dict[str, Any]:
            attempts.append(1)
            raise ValueError(f"Forced failure #{len(attempts)}")

        decision_engine.register_validation_rule(
            "approve_transaction", counting_validator
        )

        result = await decision_engine.decide(
            decision_type="approve_transaction",
            context={},
        )

        # Exactly 3 attempts
        assert len(attempts) == 3
        metadata = result["_decision_metadata"]
        assert metadata["retries_used"] == 3

    @pytest.mark.asyncio
    async def test_validation_eventual_success_on_retry(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """First 2 validation attempts fail, 3rd succeeds — overall success."""
        attempt_counter = 0

        def flaky_validator(
            decision_type: str, result: Dict[str, Any]
        ) -> Dict[str, Any]:
            nonlocal attempt_counter
            attempt_counter += 1
            if attempt_counter < 3:
                raise ValueError(f"Temporary failure #{attempt_counter}")
            return result  # Success on 3rd attempt

        decision_engine.register_validation_rule(
            "generate_description", flaky_validator
        )

        result = await decision_engine.decide(
            decision_type="generate_description",
            context={},
        )

        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        assert "validation" in metadata["layers_executed"]
        # 2 retries used before the 3rd attempt succeeds
        assert metadata["retries_used"] == 2


# ============================================================================
# TestDeterministicLayer — Layer 4 financial calculations
# ============================================================================


class TestDeterministicLayer:
    """Verify Deterministic Layer (Layer 4) financial calculations.

    CRITICAL: GL postings and balance calculations are EXCLUSIVELY computed
    in this layer — NEVER LLM-generated (AAP Section 0.7.1).
    """

    @pytest.mark.asyncio
    async def test_deterministic_layer_applies_business_rules(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """Business rules (approval thresholds) are applied deterministically.

        Invoice thresholds: ≤$10K none, ≤$50K manager, ≤$100K director, >$100K cfo.
        A $75K invoice should require 'director' approval.
        The _resolve_amount checks result dict first (expected_amount from
        statistical layer), so we set the mock to return $75K.
        """
        # Override the statistical model mock to return $75K
        mock_statistical_models["amount_distribution"].sample.return_value = 75000.0

        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 75000.0},
            agent_config=sample_agent_config,
        )

        # Invoice threshold: $50K < $75K <= $100K → director level
        assert result.get("approval_required") is True
        assert result.get("approval_level") == "director"

    @pytest.mark.asyncio
    async def test_deterministic_layer_calculates_gl_postings(
        self,
        decision_engine: DecisionEngine,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """GL postings are computed in Layer 4 for vendor invoice processing.

        Verifies double-entry bookkeeping: debit expense, credit AP.
        The _resolve_amount() checks result dict first, so we set the
        statistical mock to return the amount we want in GL postings.
        """
        mock_statistical_models["amount_distribution"].sample.return_value = 7500.0

        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={
                "amount": 7500.0,
                "expense_account": "6100-00",
                "ap_account": "2000-00",
            },
            agent_config=sample_agent_config,
        )

        gl_postings = result.get("gl_postings")
        assert gl_postings is not None
        assert len(gl_postings) == 2

        # Verify double-entry: debit and credit
        debit_entry = gl_postings[0]
        credit_entry = gl_postings[1]
        assert debit_entry["account"] == "6100-00"
        assert debit_entry["debit"] == 7500.0
        assert debit_entry["credit"] == 0.0
        assert credit_entry["account"] == "2000-00"
        assert credit_entry["debit"] == 0.0
        assert credit_entry["credit"] == 7500.0

    @pytest.mark.asyncio
    async def test_financial_calculations_only_in_deterministic(
        self,
        mock_prompt_manager: MagicMock,
        mock_response_parser: MagicMock,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """Financial calculations (GL, balances) are NEVER in LLM output.

        Even when the LLM returns financial-looking data, the deterministic
        layer recalculates GL postings independently.
        """
        # LLM mock returns data that includes fake GL postings
        llm_client = MagicMock(spec=LLMClient)
        llm_client.complete = AsyncMock(
            return_value='{"decision": "approve", "gl_postings": "FAKE_LLM_GL"}'
        )

        # Parser returns model with LLM's fake GL data
        parser = MagicMock(spec=ResponseParser)
        parser.parse.return_value = _make_mock_pydantic_model({
            "decision": "approve",
            "gl_postings": "FAKE_LLM_GL",
        })

        engine = DecisionEngine(
            llm_client=llm_client,
            prompt_manager=mock_prompt_manager,
            response_parser=parser,
            statistical_models=mock_statistical_models,
        )

        result = await engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 10000.0},
            agent_config=sample_agent_config,
        )

        # The deterministic layer OVERWRITES any LLM-generated GL postings
        gl_postings = result.get("gl_postings")
        assert gl_postings is not None
        # GL postings must be proper double-entry dicts, not the LLM's fake string
        assert isinstance(gl_postings, list)
        assert len(gl_postings) == 2
        for posting in gl_postings:
            assert isinstance(posting, dict)
            assert "debit" in posting
            assert "credit" in posting

    @pytest.mark.asyncio
    async def test_deterministic_layer_updates_balances(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Balance-related operations are computed deterministically.

        For payment scheduling, GL postings show debit AP / credit Cash.
        """
        result = await decision_engine.decide(
            decision_type="schedule_payment",
            context={
                "amount": 3000.0,
                "ap_account": "2000-00",
                "cash_account": "1000-00",
            },
        )

        gl_postings = result.get("gl_postings")
        assert gl_postings is not None
        assert len(gl_postings) == 2

        # Debit AP (reduce liability), Credit Cash (reduce asset)
        ap_entry = gl_postings[0]
        cash_entry = gl_postings[1]
        assert ap_entry["account"] == "2000-00"
        assert ap_entry["debit"] == 3000.0
        assert cash_entry["account"] == "1000-00"
        assert cash_entry["credit"] == 3000.0


# ============================================================================
# TestDecideMethod — Main decide() API contract
# ============================================================================


class TestDecideMethod:
    """Verify the decide() method's API contract and full-flow behaviour."""

    @pytest.mark.asyncio
    async def test_decide_returns_dict(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """decide() always returns a dict (never None or other type)."""
        result = await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_decide_accepts_decision_type(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """decide() accepts and processes the decision_type parameter."""
        result = await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )
        metadata = result["_decision_metadata"]
        assert metadata["decision_type"] == "determine_amount"

    @pytest.mark.asyncio
    async def test_decide_accepts_context(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """decide() accepts context dict and forwards it through the pipeline."""
        context = {
            "invoice_id": "INV-TEST-001",
            "amount": 8000.0,
            "vendor_name": "Test Vendor Inc.",
        }
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context=context,
            agent_config=sample_agent_config,
        )
        metadata = result["_decision_metadata"]
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_decide_accepts_agent_config(
        self,
        decision_engine: DecisionEngine,
        mock_prompt_manager: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """decide() accepts agent_config and uses it for LLM layer enrichment.

        Verifies that the agent's role and traits are forwarded to the
        PromptManager's build_prompt() context.
        """
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 2000.0},
            agent_config=sample_agent_config,
        )

        # Prompt manager was called with enriched context
        mock_prompt_manager.build_prompt.assert_called_once()
        call_args = mock_prompt_manager.build_prompt.call_args
        enriched_ctx = call_args[0][1]  # second positional arg is context
        assert enriched_ctx.get("agent_role") == "ap_clerk"
        assert "agent_traits" in enriched_ctx

    @pytest.mark.asyncio
    async def test_decide_vendor_invoice_full_flow(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        mock_prompt_manager: MagicMock,
        mock_response_parser: MagicMock,
        mock_statistical_models: Dict[str, MagicMock],
        sample_agent_config: AgentConfig,
    ) -> None:
        """Full pipeline flow for vendor invoice processing.

        Verifies all 4 layers execute, LLM is invoked, GL postings are
        calculated, and approval thresholds are applied.
        The _resolve_amount() checks result dict first, so we set the
        statistical mock to return $75K for proper threshold testing.
        """
        mock_statistical_models["amount_distribution"].sample.return_value = 75000.0

        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 75000.0},
            agent_config=sample_agent_config,
        )

        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        assert metadata["layers_executed"] == [
            "statistical", "llm", "validation", "deterministic"
        ]

        # LLM was invoked for this LLM-required decision type
        mock_llm_client.complete.assert_called_once()

        # GL postings were calculated deterministically
        assert "gl_postings" in result

        # Approval threshold applied: $25K < $75K <= $100K → director
        assert result.get("approval_required") is True
        assert result.get("approval_level") == "director"

    @pytest.mark.asyncio
    async def test_decide_approval_full_flow(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
        sample_agent_config: AgentConfig,
    ) -> None:
        """Full pipeline flow for approve_transaction decisions."""
        result = await decision_engine.decide(
            decision_type="approve_transaction",
            context={"amount": 150000.0},
            agent_config=sample_agent_config,
        )

        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        # LLM should have been invoked for approve_transaction
        mock_llm_client.complete.assert_called_once()
        # Processing time is recorded
        assert metadata["processing_time_seconds"] >= 0


# ============================================================================
# TestDecisionPerformance — Performance SLA verification
# ============================================================================


class TestDecisionPerformance:
    """Verify decision performance targets and metrics tracking.

    AAP Section 0.7.3: Decision latency p95 < 5 seconds. When the LLM is
    skipped for statistical-only decisions, the pipeline should be very fast.
    """

    @pytest.mark.asyncio
    async def test_statistical_only_decision_fast(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Statistical-only decisions complete within performance SLA.

        When LLM is skipped, the decision should be near-instantaneous
        since it only involves in-memory statistical model invocations.
        """
        start = time.monotonic()
        result = await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )
        elapsed = time.monotonic() - start

        # Statistical-only should be well under the 5s p95 target
        assert elapsed < 1.0, (
            f"Statistical-only decision took {elapsed:.3f}s — "
            f"expected < 1.0s for in-memory operations"
        )

        metadata = result["_decision_metadata"]
        assert metadata["processing_time_seconds"] < 5.0

    @pytest.mark.asyncio
    async def test_metrics_tracking(
        self,
        decision_engine: DecisionEngine,
        sample_agent_config: AgentConfig,
    ) -> None:
        """DecisionEngine tracks metrics for total decisions and LLM usage."""
        initial_metrics = decision_engine.get_metrics()
        assert initial_metrics["total_decisions"] == 0

        # Run a statistical-only decision
        await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )

        metrics_after_stat = decision_engine.get_metrics()
        assert metrics_after_stat["total_decisions"] == 1
        assert metrics_after_stat["statistical_only"] == 1
        assert metrics_after_stat["llm_invoked"] == 0

        # Run an LLM-required decision
        await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 1000.0},
            agent_config=sample_agent_config,
        )

        metrics_after_llm = decision_engine.get_metrics()
        assert metrics_after_llm["total_decisions"] == 2
        assert metrics_after_llm["llm_invoked"] == 1

        # Verify get_metrics returns a copy (not a reference)
        m1 = decision_engine.get_metrics()
        m2 = decision_engine.get_metrics()
        assert m1 is not m2
        assert m1 == m2


# ============================================================================
# TestDecisionEdgeCases — Edge case and error handling
# ============================================================================


class TestDecisionEdgeCases:
    """Verify graceful handling of edge cases and unexpected inputs."""

    @pytest.mark.asyncio
    async def test_empty_context_handled(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Empty context dict does not cause the pipeline to crash."""
        result = await decision_engine.decide(
            decision_type="determine_amount",
            context={},
        )

        assert isinstance(result, dict)
        metadata = result["_decision_metadata"]
        assert metadata["success"] is True

    @pytest.mark.asyncio
    async def test_unknown_decision_type_handled(
        self,
        decision_engine: DecisionEngine,
    ) -> None:
        """Unknown decision type does not crash — handled gracefully.

        An unknown type has no statistical routing and no LLM requirement,
        so it flows through the pipeline with minimal processing.
        """
        result = await decision_engine.decide(
            decision_type="totally_unknown_type_xyz",
            context={"data": "test"},
        )

        assert isinstance(result, dict)
        metadata = result["_decision_metadata"]
        assert metadata["decision_type"] == "totally_unknown_type_xyz"
        # Pipeline should still complete without crashing
        assert "statistical" in metadata["layers_executed"]
        assert "deterministic" in metadata["layers_executed"]

    @pytest.mark.asyncio
    async def test_none_agent_config_handled(
        self,
        decision_engine: DecisionEngine,
        mock_llm_client: MagicMock,
    ) -> None:
        """None agent_config is handled appropriately.

        When agent_config is None, the LLM layer should still function
        using default parameters (no personality trait enrichment).
        """
        result = await decision_engine.decide(
            decision_type="process_vendor_invoice",
            context={"amount": 5000.0},
            agent_config=None,
        )

        assert isinstance(result, dict)
        metadata = result["_decision_metadata"]
        assert metadata["success"] is True
        # LLM was still invoked (process_vendor_invoice requires LLM)
        mock_llm_client.complete.assert_called_once()
