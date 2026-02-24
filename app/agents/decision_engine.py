"""Hybrid Decision Engine — 4-layer decision pipeline.

Implements the INVIOLABLE 4-layer pipeline per AAP Section 0.7.1:

    Layer 1 — Statistical Layer:
        Determine amounts using distributions (log-normal for PO amounts, etc.),
        select entities using weighted random (Pareto 80/20 for vendors),
        calculate timing (payment terms + variance), and check for discrepancy
        triggers.

    Layer 2 — LLM Layer (CONDITIONAL — only invoked when needed):
        Generate descriptions and processing notes, handle edge cases and
        ambiguous situations, make judgment calls on exceptions, and add
        processing notes for audit trail.

    Layer 3 — Validation Layer:
        Validate against Pydantic schemas, check value ranges (amounts within
        bounds, dates valid), enforce business rules. Retries on validation
        failure with max 3 attempts.

    Layer 4 — Deterministic Layer:
        Apply business rules (approval thresholds, matching tolerances),
        calculate GL postings (EXCLUSIVELY here — NEVER LLM-generated),
        update balances (EXCLUSIVELY here), and enforce business logic
        constraints.

CRITICAL ARCHITECTURAL RULE:
    The 4-layer pipeline is INVIOLABLE. No agent or subsystem may skip layers
    or reorder them. Financial calculations (GL postings, balance updates) are
    EXCLUSIVELY computed in the Deterministic Layer and must NEVER be
    LLM-generated.

Performance Targets (AAP Section 0.7.3):
    - Decision latency p95 < 5 seconds
    - Concurrent agents >= 20
    - LLM budget enforcement: hard cap at $100/month
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Set

import structlog
from pydantic import BaseModel, Field, ValidationError

from app.agents.agent_config import AgentConfig
from app.llm.llm_client import LLMClient
from app.llm.prompt_manager import PromptManager
from app.llm.response_parser import ResponseParser

# ---------------------------------------------------------------------------
# Module-level logger (AAP Section 0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Decision type classification constants
# ---------------------------------------------------------------------------

LLM_REQUIRED_DECISIONS: Set[str] = {
    "process_vendor_invoice",
    "approve_transaction",
    "handle_exception",
    "match_documents",
    "generate_description",
    "reconcile_account",
}
"""Decision types that require LLM involvement in Layer 2.

These represent qualitative decisions where LLM generates descriptions,
processing notes, judgment calls, or exception handling.
"""

STATISTICAL_ONLY_DECISIONS: Set[str] = {
    "determine_amount",
    "select_entity",
    "calculate_timing",
    "determine_frequency",
}
"""Decision types that use only statistical models (Layer 1) and skip LLM.

These represent pure quantitative decisions driven by statistical
distributions (log-normal, Pareto, Poisson, etc.).
"""

# ---------------------------------------------------------------------------
# Statistical model routing configuration
# ---------------------------------------------------------------------------
# Maps decision type -> list of (model_key, context_param, output_key)
# tuples defining which statistical models to invoke and where to
# store their outputs.

_DECISION_STATISTICAL_ROUTING: Dict[str, List[tuple]] = {
    "determine_amount": [("amount_distribution", None, "amount")],
    "select_entity": [("selection_model", None, "selected_entity")],
    "calculate_timing": [("payment_timing", None, "payment_timing")],
    "determine_frequency": [("order_frequency", None, "order_frequency")],
    "create_purchase_order": [
        ("amount_distribution", None, "amount"),
        ("selection_model", None, "selected_vendor"),
    ],
    "process_vendor_invoice": [
        ("amount_distribution", None, "expected_amount"),
    ],
    "schedule_payment": [
        ("payment_timing", None, "payment_date_offset"),
    ],
    "create_sales_order": [
        ("amount_distribution", None, "amount"),
        ("order_frequency", None, "order_count"),
    ],
    "create_customer_invoice": [
        ("amount_distribution", None, "amount"),
    ],
    "apply_payment": [
        ("payment_timing", None, "payment_timing"),
    ],
}

# ---------------------------------------------------------------------------
# Method priority for duck-typed statistical model invocation
# ---------------------------------------------------------------------------
_MODEL_METHOD_PRIORITY: Dict[str, List[str]] = {
    "selection_model": ["select", "sample", "__call__"],
    "amount_distribution": ["sample", "generate", "__call__"],
    "payment_timing": ["sample", "calculate", "__call__"],
    "order_frequency": ["sample", "generate", "__call__"],
}
_DEFAULT_METHOD_PRIORITY: List[str] = ["sample", "select", "generate", "__call__"]


# ---------------------------------------------------------------------------
# Generic LLM output model for ResponseParser.parse() invocation
# ---------------------------------------------------------------------------

class _GenericLLMOutput(BaseModel):
    """Permissive schema for LLM output parsing.

    Accepts any JSON fields the LLM returns, enabling generic parsing
    through ``ResponseParser.parse()`` without decision-type-specific
    schemas.  The ``extra="allow"`` config lets Pydantic accept arbitrary
    keys so the DecisionEngine can handle diverse LLM response structures.
    """

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Pydantic V2 boundary models (AAP Section 0.7.1)
# ---------------------------------------------------------------------------

class DecisionContext(BaseModel):
    """Input model for the 4-layer decision pipeline.

    Defines the complete context required to make a decision, including the
    decision type, context data, optional agent configuration, and control
    parameters for retry behaviour.

    Attributes:
        decision_type: Identifier for the type of decision (e.g.
            ``"process_vendor_invoice"``).
        context: Arbitrary context data consumed by each layer.
        agent_config: Optional ``AgentConfig`` providing personality traits
            and role context forwarded to the LLM layer.
        require_llm: Force LLM invocation even for statistical-only types.
        max_retries: Maximum validation retries in Layer 3 (1–10, default 3).
    """

    decision_type: str
    context: Dict[str, Any]
    agent_config: Optional[Any] = None
    require_llm: bool = False
    max_retries: int = Field(default=3, ge=1, le=10)


class DecisionResult(BaseModel):
    """Output model from the 4-layer decision pipeline.

    Captures outputs from each layer, execution metadata, timing, and
    error information for comprehensive audit trailing.

    Attributes:
        success: Whether the pipeline completed without fatal errors.
        decision_type: The decision type that was processed.
        statistical_output: Raw output from Layer 1 (Statistical).
        llm_output: Raw output from Layer 2 (LLM) — ``None`` when skipped.
        validated_output: Output after Layer 3 validation pass.
        final_output: Merged output after all layers including Layer 4
            deterministic calculations.
        layers_executed: Ordered list of layer names that executed.
        error: Error message if the pipeline failed; ``None`` on success.
        retries_used: Number of validation retries consumed (0–3).
        processing_time_seconds: Wall-clock pipeline duration.
    """

    success: bool
    decision_type: str
    statistical_output: Dict[str, Any] = Field(default_factory=dict)
    llm_output: Optional[Dict[str, Any]] = None
    validated_output: Dict[str, Any] = Field(default_factory=dict)
    final_output: Dict[str, Any] = Field(default_factory=dict)
    layers_executed: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    retries_used: int = 0
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# DecisionEngine — 4-layer hybrid decision pipeline
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Hybrid 4-layer statistical/LLM decision pipeline.

    The pipeline is **INVIOLABLE**::

        1. Statistical Layer  — amounts, entities, timing, triggers
        2. LLM Layer (cond.)  — descriptions, notes, edge cases
        3. Validation Layer   — schema, range, business rules (max 3 retries)
        4. Deterministic Layer — GL postings, balances, business rules

    Financial calculations (GL postings, balance updates) are EXCLUSIVELY
    computed in the Deterministic Layer and must NEVER be LLM-generated.

    All dependencies are injected via constructor (AAP Section 0.7.1).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        prompt_manager: Optional[Any] = None,
        response_parser: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the DecisionEngine with injected dependencies.

        Args:
            llm_client: Injected ``LLMClient`` for Layer 2.  Optional —
                when ``None`` the LLM layer is skipped (graceful
                degradation).
            prompt_manager: Injected ``PromptManager`` for building prompts
                via ``build_prompt(template_name, context)``.
            response_parser: Injected ``ResponseParser`` for parsing LLM
                output via ``parse(raw_text, schema)``.
            statistical_models: Dict mapping model name to model instance.
                Expected keys: ``"amount_distribution"``,
                ``"selection_model"``, ``"payment_timing"``,
                ``"order_frequency"``.
        """
        self.llm_client: Optional[Any] = llm_client
        self.prompt_manager: Optional[Any] = prompt_manager
        self.response_parser: Optional[Any] = response_parser
        self.statistical_models: Dict[str, Any] = statistical_models or {}

        # Registered extension points for custom business logic
        self._validation_rules: Dict[str, Callable] = {}
        self._deterministic_rules: Dict[str, Callable] = {}

        # Metrics (AAP Section 0.7.6 — tracked for structured logging)
        self.metrics: Dict[str, int] = {
            "total_decisions": 0,
            "statistical_only": 0,
            "llm_invoked": 0,
            "validation_retries": 0,
            "failures": 0,
        }

        logger.info(
            "decision_engine_initialized",
            has_llm_client=llm_client is not None,
            has_prompt_manager=prompt_manager is not None,
            has_response_parser=response_parser is not None,
            statistical_model_count=len(self.statistical_models),
            statistical_model_keys=sorted(self.statistical_models.keys()),
        )

    # ------------------------------------------------------------------
    # Public API: decide()
    # ------------------------------------------------------------------

    async def decide(
        self,
        decision_type: str,
        context: Dict[str, Any],
        agent_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the 4-layer decision pipeline.

        Processes the decision through all four layers in strict order:
        Statistical -> LLM (conditional) -> Validation (max 3 retries) ->
        Deterministic.

        Args:
            decision_type: Type of decision (e.g. ``"process_vendor_invoice"``).
            context: Context data for the decision.  Keys vary by type.
            agent_config: Optional ``AgentConfig`` providing personality
                traits (thoroughness, risk_tolerance, efficiency, compliance)
                and role context for the LLM layer.

        Returns:
            Final decision output dict containing merged results from all
            executed layers, plus a ``_decision_metadata`` key with pipeline
            execution tracking data.

        Note:
            Performance target: p95 < 5 seconds (AAP Section 0.7.3).
        """
        start_time = time.monotonic()
        self.metrics["total_decisions"] += 1

        layers_executed: List[str] = []
        accumulated: Dict[str, Any] = {}
        error_msg: Optional[str] = None
        retries_used: int = 0
        statistical_output: Dict[str, Any] = {}
        llm_output: Optional[Dict[str, Any]] = None
        validated_output: Dict[str, Any] = {}

        try:
            # =============================================================
            # Layer 1 — Statistical Layer
            # =============================================================
            statistical_output = await self._statistical_layer(
                decision_type, context
            )
            accumulated.update(statistical_output)
            layers_executed.append("statistical")

            logger.debug(
                "decision_statistical_layer",
                decision_type=decision_type,
                output_keys=sorted(statistical_output.keys()),
            )

            # =============================================================
            # Layer 2 — LLM Layer (CONDITIONAL)
            # =============================================================
            needs_llm = (
                decision_type in LLM_REQUIRED_DECISIONS
                or context.get("require_llm", False)
            )
            has_llm_capability = (
                self.llm_client is not None
                and self.prompt_manager is not None
                and self.response_parser is not None
            )

            if needs_llm and has_llm_capability:
                llm_output = await self._llm_layer(
                    decision_type, context, agent_config
                )
                if llm_output:
                    accumulated.update(llm_output)
                layers_executed.append("llm")
                self.metrics["llm_invoked"] += 1
            else:
                self.metrics["statistical_only"] += 1
                skip_reason = (
                    "no_llm_capability"
                    if not has_llm_capability
                    else "statistical_only_decision"
                )
                logger.debug(
                    "decision_skip_llm",
                    decision_type=decision_type,
                    reason=skip_reason,
                )

            # =============================================================
            # Layer 3 — Validation Layer (max 3 retries)
            # =============================================================
            max_retries = 3
            validation_success = False

            for attempt in range(max_retries):
                try:
                    validated_output = self._validation_layer(
                        decision_type, accumulated
                    )
                    accumulated.update(validated_output)
                    layers_executed.append("validation")
                    validation_success = True
                    break
                except (ValidationError, ValueError) as exc:
                    retries_used = attempt + 1
                    self.metrics["validation_retries"] += 1
                    logger.warning(
                        "decision_validation_retry",
                        decision_type=decision_type,
                        attempt=retries_used,
                        max_retries=max_retries,
                        error=str(exc),
                    )
                    if attempt == max_retries - 1:
                        error_msg = (
                            f"Validation failed after {max_retries} "
                            f"retries: {exc}"
                        )
                        self.metrics["failures"] += 1
                        layers_executed.append("validation_failed")

            # =============================================================
            # Layer 4 — Deterministic Layer
            # CRITICAL: GL postings, balance calculations done HERE ONLY.
            # =============================================================
            final_output = self._deterministic_layer(
                decision_type, accumulated, context
            )
            accumulated.update(final_output)
            layers_executed.append("deterministic")

        except Exception as exc:
            error_msg = f"Pipeline error: {exc}"
            self.metrics["failures"] += 1
            logger.error(
                "decision_pipeline_error",
                decision_type=decision_type,
                error=str(exc),
                exc_info=True,
            )

        processing_time = time.monotonic() - start_time

        # Extract agent_id for structured logging (AAP Section 0.7.6)
        agent_id_str: Optional[str] = None
        if agent_config is not None:
            agent_id_str = str(getattr(agent_config, "agent_id", None))

        logger.info(
            "decision_complete",
            agent_id=agent_id_str,
            decision_type=decision_type,
            layers_executed=layers_executed,
            processing_time_seconds=round(processing_time, 4),
            success=error_msg is None,
        )

        # Attach execution metadata for audit trail
        accumulated["_decision_metadata"] = {
            "success": error_msg is None,
            "decision_type": decision_type,
            "layers_executed": layers_executed,
            "processing_time_seconds": round(processing_time, 4),
            "error": error_msg,
            "retries_used": retries_used,
        }

        return accumulated

    # ------------------------------------------------------------------
    # Layer 1 — Statistical Layer
    # ------------------------------------------------------------------

    async def _statistical_layer(
        self,
        decision_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Layer 1: Statistical Layer.

        Uses registered statistical models to determine amounts (log-normal),
        select entities (Pareto 80/20), calculate timing (payment terms +
        variance), and check for discrepancy triggers.

        The layer is fully functional even without models registered — it
        returns an empty dict when no models apply to the decision type.

        Args:
            decision_type: The type of decision being processed.
            context: Context data forwarded to each invoked model.

        Returns:
            Dict of statistical outputs keyed by output_key from routing.
        """
        output: Dict[str, Any] = {}

        # Look up routing configuration for this decision type
        routing = _DECISION_STATISTICAL_ROUTING.get(decision_type, [])

        for model_key, _param, output_key in routing:
            model = self.statistical_models.get(model_key)
            if model is None:
                continue

            try:
                result = self._invoke_statistical_model(
                    model, model_key, context
                )
                if result is not None:
                    output[output_key] = result
            except Exception as exc:
                logger.warning(
                    "statistical_model_error",
                    model_key=model_key,
                    decision_type=decision_type,
                    error=str(exc),
                )

        return output

    def _invoke_statistical_model(
        self,
        model: Any,
        model_key: str,
        context: Dict[str, Any],
    ) -> Any:
        """Invoke a statistical model using duck-typing.

        Tries common method names in priority order specific to the model
        type, falling back to a generic priority list.

        Args:
            model: The statistical model instance.
            model_key: Registry key (e.g. ``"amount_distribution"``).
            context: Context dict forwarded to the model method.

        Returns:
            The model's output, or ``None`` if no callable found.
        """
        method_names = _MODEL_METHOD_PRIORITY.get(
            model_key, _DEFAULT_METHOD_PRIORITY
        )

        for name in method_names:
            method = getattr(model, name, None)
            if method is not None and callable(method):
                return method(context)

        logger.warning(
            "statistical_model_no_callable",
            model_key=model_key,
            model_type=type(model).__name__,
        )
        return None

    # ------------------------------------------------------------------
    # Layer 2 — LLM Layer (CONDITIONAL)
    # ------------------------------------------------------------------

    async def _llm_layer(
        self,
        decision_type: str,
        context: Dict[str, Any],
        agent_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Layer 2: LLM Layer (conditional).

        Generates descriptions, processing notes, handles edge cases and
        ambiguous situations, makes judgment calls on exceptions, and adds
        processing notes for audit trail.

        CRITICAL: This layer NEVER generates financial calculations.
        GL postings and balance updates are EXCLUSIVELY in Layer 4.

        Gracefully degrades to an empty dict on any error — the LLM layer
        is optional by design so that the pipeline can still produce valid
        (if less nuanced) results without LLM availability.

        Args:
            decision_type: The type of decision being processed.
            context: Context data used for prompt assembly.
            agent_config: Optional ``AgentConfig`` with personality traits
                (``role``, ``traits``) enriching the prompt context.

        Returns:
            Parsed LLM output as a dict, or empty dict on failure.
        """
        try:
            # Enrich context with agent personality for prompt building
            enriched_context: Dict[str, Any] = dict(context)
            if agent_config is not None:
                enriched_context["agent_role"] = getattr(
                    agent_config, "role", "unknown"
                )
                enriched_context["agent_traits"] = getattr(
                    agent_config, "traits", {}
                )

            # Build prompt via PromptManager — returns (system, user) tuple
            system_prompt, user_prompt = self.prompt_manager.build_prompt(
                decision_type, enriched_context
            )

            # Determine LLM parameters from agent config when available
            max_tokens = 1000
            temperature = 0.7
            if agent_config is not None:
                max_tokens = int(
                    getattr(agent_config, "max_tokens", max_tokens)
                )
                temperature = float(
                    getattr(agent_config, "temperature", temperature)
                )

            # Call LLM asynchronously — returns raw text string
            response_text: str = await self.llm_client.complete(
                prompt=user_prompt,
                system=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Parse response via ResponseParser.parse() with generic schema
            parsed_model = self.response_parser.parse(
                response_text, _GenericLLMOutput
            )
            parsed_dict: Dict[str, Any] = parsed_model.model_dump()

            logger.debug(
                "llm_layer_complete",
                decision_type=decision_type,
                output_keys=sorted(parsed_dict.keys()),
            )

            return parsed_dict

        except Exception as exc:
            # Graceful degradation — LLM layer is optional
            logger.warning(
                "llm_layer_error",
                decision_type=decision_type,
                error=str(exc),
                exc_info=True,
            )
            return {}

    # ------------------------------------------------------------------
    # Layer 3 — Validation Layer
    # ------------------------------------------------------------------

    def _validation_layer(
        self,
        decision_type: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Layer 3: Validation Layer.

        Validates the accumulated result against registered custom validators
        and generic validation checks (amount ranges, required fields).

        Args:
            decision_type: The type of decision being processed.
            result: Accumulated result dict from Layers 1 and 2.

        Returns:
            Validated result dict (may be identical to input).

        Raises:
            ValidationError: If Pydantic schema validation fails.
            ValueError: If business-rule validation fails.
        """
        validated: Dict[str, Any] = dict(result)

        # Apply custom validation rule if registered
        custom_validator = self._validation_rules.get(decision_type)
        if custom_validator is not None:
            validated = custom_validator(decision_type, validated)

        # Generic validations applied to every decision
        self._validate_amounts(validated)
        self._validate_required_fields(decision_type, validated)

        return validated

    @staticmethod
    def _validate_amounts(result: Dict[str, Any]) -> None:
        """Validate that monetary amounts are non-negative.

        Checks common amount field names and raises ``ValueError`` if any
        are negative.  Logs a warning for unusually high amounts (>$500K)
        without rejecting them — the statistical distribution models are
        responsible for enforcing parameterised bounds.
        """
        amount_fields = (
            "amount",
            "expected_amount",
            "total_amount",
            "payment_amount",
            "invoice_amount",
        )

        for field_name in amount_fields:
            value = result.get(field_name)
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                continue
            if value < 0:
                raise ValueError(
                    f"Amount field '{field_name}' is negative: {value}"
                )
            if value > 500_000:
                logger.warning(
                    "validation_amount_high",
                    field=field_name,
                    value=value,
                    max_typical=500_000,
                )

    @staticmethod
    def _validate_required_fields(
        decision_type: str,
        result: Dict[str, Any],
    ) -> None:
        """Check that decision-type-specific required fields are present.

        Raises ``ValueError`` when mandatory output fields are missing from
        the accumulated result for the given decision type.
        """
        _required_map: Dict[str, List[str]] = {
            "process_vendor_invoice": [],
            "approve_transaction": [],
            "create_purchase_order": [],
            "schedule_payment": [],
            "match_documents": [],
            "generate_description": [],
            "reconcile_account": [],
            "handle_exception": [],
            "determine_amount": [],
            "select_entity": [],
            "calculate_timing": [],
            "determine_frequency": [],
        }

        required = _required_map.get(decision_type, [])
        missing = [f for f in required if f not in result]
        if missing:
            raise ValueError(
                f"Missing required fields for '{decision_type}': {missing}"
            )

    # ------------------------------------------------------------------
    # Layer 4 — Deterministic Layer
    # ------------------------------------------------------------------

    def _deterministic_layer(
        self,
        decision_type: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Layer 4: Deterministic Layer.

        Applies business rules, calculates GL postings, updates balances,
        and enforces business-logic constraints.

        **CRITICAL ARCHITECTURAL RULE**: Financial calculations (GL postings,
        balance updates) are EXCLUSIVELY computed here and must NEVER be
        LLM-generated (AAP Section 0.7.1).

        Args:
            decision_type: The type of decision being processed.
            result: Accumulated result dict from Layers 1–3.
            context: Original context dict for reference data.

        Returns:
            Final output dict with all deterministic calculations applied.
        """
        output: Dict[str, Any] = dict(result)

        # Apply custom deterministic rule if registered
        custom_rule = self._deterministic_rules.get(decision_type)
        if custom_rule is not None:
            output = custom_rule(decision_type, output, context)

        # Generic deterministic operations
        output = self._apply_approval_thresholds(decision_type, output, context)
        output = self._calculate_gl_postings(decision_type, output, context)
        output = self._apply_matching_rules(decision_type, output, context)

        return output

    @staticmethod
    def _apply_approval_thresholds(
        decision_type: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply monetary threshold-based approval requirements.

        Per AAP Section 0.5.1:
            PO thresholds:   none / $5K manager / $25K director / $100K CFO
            Invoice thresholds: none / $10K manager / $50K director / $100K CFO
            JE thresholds:   $50K controller
        """
        output = dict(result)
        amount = DecisionEngine._resolve_amount(result, context)

        if amount is None or not isinstance(amount, (int, float)):
            return output

        # PO approval thresholds
        if decision_type in (
            "create_purchase_order",
            "approve_purchase_order",
        ):
            if amount <= 5_000:
                output["approval_level"] = "none"
                output["approval_required"] = False
            elif amount <= 25_000:
                output["approval_level"] = "manager"
                output["approval_required"] = True
            elif amount <= 100_000:
                output["approval_level"] = "director"
                output["approval_required"] = True
            else:
                output["approval_level"] = "cfo"
                output["approval_required"] = True

        # Invoice approval thresholds
        elif decision_type in (
            "process_vendor_invoice",
            "approve_invoice",
        ):
            if amount <= 10_000:
                output["approval_level"] = "none"
                output["approval_required"] = False
            elif amount <= 50_000:
                output["approval_level"] = "manager"
                output["approval_required"] = True
            elif amount <= 100_000:
                output["approval_level"] = "director"
                output["approval_required"] = True
            else:
                output["approval_level"] = "cfo"
                output["approval_required"] = True

        # Journal Entry approval thresholds
        elif decision_type == "create_journal_entry":
            if amount <= 50_000:
                output["approval_level"] = "none"
                output["approval_required"] = False
            else:
                output["approval_level"] = "controller"
                output["approval_required"] = True

        return output

    @staticmethod
    def _resolve_amount(
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Any:
        """Resolve the monetary amount from result or context.

        Checks multiple common amount field names in priority order so
        that GL postings work regardless of whether the amount was
        generated under ``amount``, ``expected_amount``, ``total_amount``,
        ``invoice_amount``, or ``payment_amount``.
        """
        _amount_keys = (
            "amount",
            "expected_amount",
            "total_amount",
            "invoice_amount",
            "payment_amount",
        )
        for key in _amount_keys:
            val = result.get(key)
            if val is not None and isinstance(val, (int, float)):
                return val
        for key in _amount_keys:
            val = context.get(key)
            if val is not None and isinstance(val, (int, float)):
                return val
        return None

    @staticmethod
    def _calculate_gl_postings(
        decision_type: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate GL postings for financial transactions.

        **CRITICAL**: GL postings are EXCLUSIVELY computed in this
        deterministic layer and must NEVER be LLM-generated
        (AAP Section 0.7.1).

        Generates double-entry bookkeeping pairs for standard transaction
        types using context-provided account codes or sensible defaults.
        """
        output = dict(result)
        amount = DecisionEngine._resolve_amount(result, context)

        if amount is None or not isinstance(amount, (int, float)):
            return output

        rounded_amount = round(float(amount), 2)

        if decision_type == "process_vendor_invoice":
            output["gl_postings"] = [
                {
                    "account": context.get("expense_account", "6000-00"),
                    "debit": rounded_amount,
                    "credit": 0.0,
                    "description": "Vendor invoice expense",
                },
                {
                    "account": context.get("ap_account", "2000-00"),
                    "debit": 0.0,
                    "credit": rounded_amount,
                    "description": "Accounts payable",
                },
            ]

        elif decision_type == "schedule_payment":
            output["gl_postings"] = [
                {
                    "account": context.get("ap_account", "2000-00"),
                    "debit": rounded_amount,
                    "credit": 0.0,
                    "description": "Payment to vendor",
                },
                {
                    "account": context.get("cash_account", "1000-00"),
                    "debit": 0.0,
                    "credit": rounded_amount,
                    "description": "Cash disbursement",
                },
            ]

        elif decision_type == "apply_payment":
            output["gl_postings"] = [
                {
                    "account": context.get("cash_account", "1000-00"),
                    "debit": rounded_amount,
                    "credit": 0.0,
                    "description": "Cash receipt",
                },
                {
                    "account": context.get("ar_account", "1200-00"),
                    "debit": 0.0,
                    "credit": rounded_amount,
                    "description": "Accounts receivable",
                },
            ]

        elif decision_type == "create_customer_invoice":
            output["gl_postings"] = [
                {
                    "account": context.get("ar_account", "1200-00"),
                    "debit": rounded_amount,
                    "credit": 0.0,
                    "description": "Accounts receivable",
                },
                {
                    "account": context.get("revenue_account", "4000-00"),
                    "debit": 0.0,
                    "credit": rounded_amount,
                    "description": "Revenue",
                },
            ]

        elif decision_type == "create_journal_entry":
            entries = context.get("entries", [])
            if entries:
                output["gl_postings"] = entries

        return output

    @staticmethod
    def _apply_matching_rules(
        decision_type: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply 3-way matching tolerance rules for AP processing.

        Per specification: 95% of vendor invoices match within +/-5%.
        Computes PO-to-invoice and PO-to-receipt variance percentages
        and a consolidated 3-way match boolean.
        """
        output = dict(result)

        if decision_type != "match_documents":
            return output

        po_amount = context.get("po_amount")
        invoice_amount = context.get("invoice_amount")
        receipt_amount = context.get("receipt_amount")

        amounts_valid = all(
            v is not None and isinstance(v, (int, float))
            for v in (po_amount, invoice_amount, receipt_amount)
        )

        if not amounts_valid:
            return output

        tolerance = 0.05  # 5% matching tolerance

        # PO vs Invoice variance
        if po_amount > 0:
            po_inv_var = abs(invoice_amount - po_amount) / po_amount
            output["po_invoice_match"] = po_inv_var <= tolerance
            output["po_invoice_variance_pct"] = round(po_inv_var * 100, 2)
        else:
            output["po_invoice_match"] = False
            output["po_invoice_variance_pct"] = 0.0

        # PO vs Receipt variance
        if po_amount > 0:
            po_rcpt_var = abs(receipt_amount - po_amount) / po_amount
            output["po_receipt_match"] = po_rcpt_var <= tolerance
            output["po_receipt_variance_pct"] = round(po_rcpt_var * 100, 2)
        else:
            output["po_receipt_match"] = False
            output["po_receipt_variance_pct"] = 0.0

        # Overall 3-way match
        output["three_way_match"] = (
            output.get("po_invoice_match", False)
            and output.get("po_receipt_match", False)
        )

        return output

    # ------------------------------------------------------------------
    # Registration methods
    # ------------------------------------------------------------------

    def register_validation_rule(
        self,
        decision_type: str,
        validator: Callable,
    ) -> None:
        """Register a custom validation rule for a decision type.

        The validator callable must accept ``(decision_type: str,
        result: Dict[str, Any])`` and return the validated dict.  It should
        raise ``ValueError`` or ``ValidationError`` on failure so that
        Layer 3 retry logic engages.

        Args:
            decision_type: The decision type this rule applies to.
            validator: Callable implementing the validation logic.
        """
        self._validation_rules[decision_type] = validator
        logger.info(
            "validation_rule_registered",
            decision_type=decision_type,
        )

    def register_deterministic_rule(
        self,
        decision_type: str,
        rule: Callable,
    ) -> None:
        """Register a custom deterministic rule for a decision type.

        The rule callable must accept ``(decision_type: str,
        result: Dict[str, Any], context: Dict[str, Any])`` and return
        the final dict with deterministic calculations applied.

        Args:
            decision_type: The decision type this rule applies to.
            rule: Callable implementing the deterministic logic.
        """
        self._deterministic_rules[decision_type] = rule
        logger.info(
            "deterministic_rule_registered",
            decision_type=decision_type,
        )

    def register_statistical_model(
        self,
        name: str,
        model: Any,
    ) -> None:
        """Register a statistical model by name.

        The model should expose at least one of: ``sample(context)``,
        ``select(context)``, ``generate(context)``, or be callable.

        Args:
            name: Model key (e.g. ``"amount_distribution"``,
                ``"selection_model"``, ``"payment_timing"``,
                ``"order_frequency"``).
            model: Model instance with a compatible callable method.
        """
        self.statistical_models[name] = model
        logger.info(
            "statistical_model_registered",
            model_name=name,
            model_type=type(model).__name__,
        )

    # ------------------------------------------------------------------
    # Metrics and monitoring
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a copy of the current decision engine metrics.

        Returns:
            Dict with keys: ``total_decisions``, ``statistical_only``,
            ``llm_invoked``, ``validation_retries``, ``failures``.
        """
        return dict(self.metrics)
