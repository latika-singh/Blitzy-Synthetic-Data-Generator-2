"""Fix Scenario Catalog — Registry of fix scenarios for the rework loop.

The ``FixScenarioCatalog`` maintains a registry of 20+ fix scenarios that the
rework loop engine can apply to correct validation failures. Each scenario
defines a sequence of steps that address a specific category of error.

Scenario Selection Strategy:
    1. Filter scenarios by applicable error type
    2. Exclude scenarios that have been previously attempted and failed
       for this specific transaction (to avoid infinite loops)
    3. Sort remaining scenarios by success rate (highest first)
    4. Return the top-ranked scenario

Fix Scenarios Included (20+):
    ADJUST_AMOUNT_TO_RANGE        — Clamp transaction amount to valid range
    FIX_DATE_SEQUENCE             — Correct temporal ordering violations
    CORRECT_ENTITY_REFERENCE      — Fix vendor/customer ID references
    REGENERATE_GL_ENTRY           — Re-create GL journal entry
    RECALCULATE_BALANCE           — Recalculate account balance
    FIX_APPROVAL_CHAIN            — Correct approval routing
    CORRECT_PERIOD_ASSIGNMENT     — Fix fiscal period assignment
    RELINK_DOCUMENTS              — Reconnect PO/receipt/invoice linkages
    FIX_THREE_WAY_MATCH           — Resolve three-way match failures
    ADJUST_PAYMENT_ALLOCATION     — Fix FIFO payment allocation
    CORRECT_TAX_CALCULATION       — Fix tax amount calculation
    FIX_CURRENCY_ROUNDING         — Resolve rounding discrepancies
    CORRECT_DISCOUNT_APPLICATION  — Fix discount calculation
    FIX_DUPLICATE_DETECTION       — Resolve duplicate reference errors
    REGENERATE_DOCUMENT_NUMBER    — Fix document number sequence
    CORRECT_QUANTITY_VARIANCE     — Resolve qty mismatch (receipt/invoice)
    FIX_CREDIT_CHECK              — Resolve credit limit failures
    CORRECT_POSTING_ACCOUNTS      — Fix GL account codes
    FIX_PAYMENT_TERMS             — Correct payment term dates
    REGENERATE_TRACKING_NUMBER    — Fix shipment tracking
    CORRECT_STATUS_TRANSITION     — Fix invalid status transitions
    FIX_INVENTORY_BALANCE         — Correct inventory quantity

Design Decisions:
    - Registry pattern modeled after app/agents/action_registry.py
    - Constructor injection (ADR-003)
    - Pydantic V2 for FixScenario model
    - structlog for JSON logging

References:
    - AAP Section 0.5.1 Group 6: fix_scenario_catalog.py
    - AAP Section 0.7.4: Error handling conventions
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
)
from uuid import uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

# Module-level structured logger (AAP Section 0.7.7 — stdout only)
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# FixScenarioName Enum — all 22 registered fix scenario names
# ---------------------------------------------------------------------------


class FixScenarioName(str, Enum):
    """Names for all registered fix scenarios.

    Each member maps to a unique fix scenario that addresses a specific
    category of validation error.  The string values match the scenario
    names stored in the ``FixScenarioCatalog`` registry.
    """

    ADJUST_AMOUNT_TO_RANGE = "ADJUST_AMOUNT_TO_RANGE"
    FIX_DATE_SEQUENCE = "FIX_DATE_SEQUENCE"
    CORRECT_ENTITY_REFERENCE = "CORRECT_ENTITY_REFERENCE"
    REGENERATE_GL_ENTRY = "REGENERATE_GL_ENTRY"
    RECALCULATE_BALANCE = "RECALCULATE_BALANCE"
    FIX_APPROVAL_CHAIN = "FIX_APPROVAL_CHAIN"
    CORRECT_PERIOD_ASSIGNMENT = "CORRECT_PERIOD_ASSIGNMENT"
    RELINK_DOCUMENTS = "RELINK_DOCUMENTS"
    FIX_THREE_WAY_MATCH = "FIX_THREE_WAY_MATCH"
    ADJUST_PAYMENT_ALLOCATION = "ADJUST_PAYMENT_ALLOCATION"
    CORRECT_TAX_CALCULATION = "CORRECT_TAX_CALCULATION"
    FIX_CURRENCY_ROUNDING = "FIX_CURRENCY_ROUNDING"
    CORRECT_DISCOUNT_APPLICATION = "CORRECT_DISCOUNT_APPLICATION"
    FIX_DUPLICATE_DETECTION = "FIX_DUPLICATE_DETECTION"
    REGENERATE_DOCUMENT_NUMBER = "REGENERATE_DOCUMENT_NUMBER"
    CORRECT_QUANTITY_VARIANCE = "CORRECT_QUANTITY_VARIANCE"
    FIX_CREDIT_CHECK = "FIX_CREDIT_CHECK"
    CORRECT_POSTING_ACCOUNTS = "CORRECT_POSTING_ACCOUNTS"
    FIX_PAYMENT_TERMS = "FIX_PAYMENT_TERMS"
    REGENERATE_TRACKING_NUMBER = "REGENERATE_TRACKING_NUMBER"
    CORRECT_STATUS_TRANSITION = "CORRECT_STATUS_TRANSITION"
    FIX_INVENTORY_BALANCE = "FIX_INVENTORY_BALANCE"


# ---------------------------------------------------------------------------
# FixStep — Pydantic V2 model for a single step within a fix scenario
# ---------------------------------------------------------------------------


class FixStep(BaseModel):
    """A single step within a fix scenario.

    Each step has a 1-based execution order, a human-readable name and
    description, and a ``handler_key`` that the
    :class:`FixScenarioExecutor` uses to look up the implementation
    function at runtime.

    Attributes:
        step_order: 1-based execution order within the parent scenario.
        step_name: Short identifier (e.g. ``'validate_current_amount'``).
        description: Human-readable description of what this step does.
        handler_key: Key used by the executor's handler registry to locate
            the callable that implements this step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_order: int = Field(
        ...,
        ge=1,
        description="Execution order (1-based)",
    )
    step_name: str = Field(
        ...,
        description="Step identifier (e.g., 'validate_current_amount')",
    )
    description: str = Field(
        ...,
        description="Human-readable description of what this step does",
    )
    handler_key: str = Field(
        ...,
        description="Key to look up in executor's handler registry",
    )


# ---------------------------------------------------------------------------
# FixScenario — Pydantic V2 model for a complete fix scenario definition
# ---------------------------------------------------------------------------


class FixScenario(BaseModel):
    """A fix scenario with its steps, success rate, and applicable error types.

    Each scenario represents a repeatable correction strategy that the
    rework loop can apply to a transaction in order to resolve a specific
    category of validation failure.  The ``success_rate`` is dynamically
    updated by :meth:`FixScenarioCatalog.record_outcome` as outcomes are
    recorded.

    Attributes:
        scenario_id: Unique identifier (auto-generated UUID4 string).
        name: Scenario name from :class:`FixScenarioName` enum.
        description: Human-readable description.
        steps: Ordered list of :class:`FixStep` entries to execute.
        success_rate: Historical success rate in the range ``[0.0, 1.0]``.
        applicable_error_types: List of error-type strings this scenario
            can address (e.g. ``'balance_error'``, ``'date_sequence_error'``).
        total_attempts: Cumulative count of times this scenario has been
            attempted.
        total_successes: Cumulative count of successful applications.
        enabled: Whether this scenario is currently available for selection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique scenario identifier (auto-generated UUID4)",
    )
    name: str = Field(
        ...,
        description="Unique scenario name from FixScenarioName enum",
    )
    description: str = Field(
        ...,
        description="Human-readable description",
    )
    steps: List[FixStep] = Field(
        default_factory=list,
        description="Ordered list of fix steps",
    )
    success_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Historical success rate (0-1)",
    )
    applicable_error_types: List[str] = Field(
        default_factory=list,
        description=(
            "Error types this scenario can fix "
            "(e.g., 'balance_error', 'date_sequence_error')"
        ),
    )
    total_attempts: int = Field(
        default=0,
        ge=0,
        description="Total times this scenario has been attempted",
    )
    total_successes: int = Field(
        default=0,
        ge=0,
        description="Total successful applications",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this scenario is available for selection",
    )


# ---------------------------------------------------------------------------
# FixScenarioCatalog — Registry of fix scenarios for the rework loop
# ---------------------------------------------------------------------------


class FixScenarioCatalog:
    """Registry of fix scenarios for the rework loop.

    Provides scenario registration, selection by error type with exclusion
    of previously failed scenarios, and dynamic success rate tracking.
    Modeled after :class:`app.agents.action_registry.ActionRegistry`.

    Constructor injection (ADR-003): the single ``load_defaults`` parameter
    controls whether the 22 built-in scenarios are registered on init.

    Thread Safety:
        This class is **not** thread-safe.  In an async context each
        ``ReworkLoopEngine`` should own a single ``FixScenarioCatalog``
        instance and share it through constructor injection.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        load_defaults: bool = True,
    ) -> None:
        """Initialise the catalog.

        Args:
            load_defaults: When ``True`` (the default) the 22 built-in fix
                scenarios are registered automatically.  Set to ``False``
                for testing with an empty catalog.
        """
        self._scenarios: Dict[str, FixScenario] = {}

        if load_defaults:
            self._register_default_scenarios()

        logger.info(
            "fix_scenario_catalog_initialized",
            service_name="transactions",
            component="FixScenarioCatalog",
            load_defaults=load_defaults,
            scenario_count=len(self._scenarios),
        )

    # ------------------------------------------------------------------ #
    # Public API — select_scenario
    # ------------------------------------------------------------------ #

    def select_scenario(
        self,
        error_type: str,
        excluded_scenarios: Optional[Set[str]] = None,
    ) -> Optional[FixScenario]:
        """Select the best fix scenario for an error type.

        Selection algorithm:
            1. Filter by ``applicable_error_types`` containing *error_type*
            2. Filter out disabled scenarios (``enabled is False``)
            3. Exclude scenarios in *excluded_scenarios* set (previously
               failed for this transaction)
            4. Sort by ``success_rate`` descending
            5. Return top-ranked scenario, or ``None`` if none available

        Args:
            error_type: The type of validation error
                (e.g. ``"balance_error"``).
            excluded_scenarios: Set of scenario names to exclude from
                selection.

        Returns:
            The highest-ranked applicable :class:`FixScenario`, or ``None``
            if no applicable scenario is available.
        """
        excluded = excluded_scenarios or set()

        # Step 1-3: filter applicable, enabled, non-excluded
        candidates: List[FixScenario] = [
            scenario
            for scenario in self._scenarios.values()
            if (
                error_type in scenario.applicable_error_types
                and scenario.enabled
                and scenario.name not in excluded
            )
        ]

        # Step 4: sort by success_rate descending
        candidates.sort(key=lambda s: s.success_rate, reverse=True)

        # Step 5: pick the top candidate
        selected = candidates[0] if candidates else None

        logger.debug(
            "scenario_selected",
            service_name="transactions",
            component="FixScenarioCatalog",
            error_type=error_type,
            excluded_count=len(excluded),
            candidates_found=len(candidates),
            selected=selected.name if selected else None,
        )

        return selected

    # ------------------------------------------------------------------ #
    # Public API — register_scenario
    # ------------------------------------------------------------------ #

    def register_scenario(self, scenario: FixScenario) -> None:
        """Register a new fix scenario or replace an existing one.

        Args:
            scenario: Fully populated :class:`FixScenario` instance.

        Side Effects:
            Updates the internal scenario store.  Emits a structured log
            entry at DEBUG level.
        """
        self._scenarios[scenario.name] = scenario

        logger.debug(
            "scenario_registered",
            service_name="transactions",
            component="FixScenarioCatalog",
            scenario_name=scenario.name,
            success_rate=scenario.success_rate,
            error_types=scenario.applicable_error_types,
            steps_count=len(scenario.steps),
        )

    # ------------------------------------------------------------------ #
    # Public API — get_scenario
    # ------------------------------------------------------------------ #

    def get_scenario(self, name: str) -> Optional[FixScenario]:
        """Get a specific scenario by name.

        Args:
            name: Scenario name (e.g. ``"ADJUST_AMOUNT_TO_RANGE"``).

        Returns:
            The :class:`FixScenario` if found, otherwise ``None``.
        """
        return self._scenarios.get(name)

    # ------------------------------------------------------------------ #
    # Public API — list_scenarios
    # ------------------------------------------------------------------ #

    def list_scenarios(self) -> List[FixScenario]:
        """List all registered scenarios sorted by success rate (highest first).

        Returns:
            A list of all :class:`FixScenario` instances ordered by
            ``success_rate`` descending.
        """
        return sorted(
            self._scenarios.values(),
            key=lambda s: s.success_rate,
            reverse=True,
        )

    # ------------------------------------------------------------------ #
    # Public API — record_outcome
    # ------------------------------------------------------------------ #

    def record_outcome(self, scenario_name: str, success: bool) -> None:
        """Record the outcome of applying a fix scenario.

        Updates the scenario's ``total_attempts``, ``total_successes``, and
        recalculates the ``success_rate`` based on historical data.

        Args:
            scenario_name: Name of the scenario whose outcome is being
                recorded.
            success: ``True`` if the fix resolved the validation failure.
        """
        scenario = self._scenarios.get(scenario_name)
        if scenario is None:
            logger.warning(
                "record_outcome_scenario_not_found",
                service_name="transactions",
                component="FixScenarioCatalog",
                scenario_name=scenario_name,
            )
            return

        scenario.total_attempts += 1
        if success:
            scenario.total_successes += 1

        # Recalculate success rate from lifetime data
        scenario.success_rate = (
            scenario.total_successes / scenario.total_attempts
            if scenario.total_attempts > 0
            else 0.0
        )

        logger.debug(
            "outcome_recorded",
            service_name="transactions",
            component="FixScenarioCatalog",
            scenario_name=scenario_name,
            success=success,
            total_attempts=scenario.total_attempts,
            total_successes=scenario.total_successes,
            success_rate=round(scenario.success_rate, 4),
        )

    # ------------------------------------------------------------------ #
    # Public API — get_metrics
    # ------------------------------------------------------------------ #

    def get_metrics(self) -> Dict[str, Any]:
        """Return catalog metrics.

        Returns:
            A dictionary containing aggregate statistics:
            ``total_scenarios``, ``enabled_scenarios``, ``total_attempts``,
            ``total_successes``, ``average_success_rate``, and
            ``per_scenario_stats``.
        """
        all_scenarios = list(self._scenarios.values())
        enabled_count = sum(1 for s in all_scenarios if s.enabled)
        total_attempts = sum(s.total_attempts for s in all_scenarios)
        total_successes = sum(s.total_successes for s in all_scenarios)

        if len(all_scenarios) > 0:
            avg_success_rate = sum(
                s.success_rate for s in all_scenarios
            ) / len(all_scenarios)
        else:
            avg_success_rate = 0.0

        per_scenario_stats: List[Dict[str, Any]] = [
            {
                "name": s.name,
                "success_rate": round(s.success_rate, 4),
                "total_attempts": s.total_attempts,
                "total_successes": s.total_successes,
                "enabled": s.enabled,
                "steps_count": len(s.steps),
                "error_types": s.applicable_error_types,
            }
            for s in sorted(
                all_scenarios, key=lambda x: x.success_rate, reverse=True
            )
        ]

        return {
            "total_scenarios": len(all_scenarios),
            "enabled_scenarios": enabled_count,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "average_success_rate": round(avg_success_rate, 4),
            "per_scenario_stats": per_scenario_stats,
        }

    # ------------------------------------------------------------------ #
    # Private — _register_default_scenarios  (22 scenarios)
    # ------------------------------------------------------------------ #

    def _register_default_scenarios(self) -> None:  # noqa: C901
        """Register all 22 default fix scenarios.

        Each scenario is created with a meaningful name from the
        :class:`FixScenarioName` enum, a description, 2-4 ordered steps,
        a realistic initial success rate, and the applicable error types.
        """

        # 1. ADJUST_AMOUNT_TO_RANGE — success rate 0.85
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value,
                description=(
                    "Clamp transaction amount to the valid configured range"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="validate_current_amount",
                        description=(
                            "Read the current transaction amount and check "
                            "against configured min/max bounds"
                        ),
                        handler_key="validate_current_amount",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="calculate_valid_range",
                        description=(
                            "Determine the nearest valid amount within the "
                            "acceptable range for this transaction type"
                        ),
                        handler_key="calculate_valid_range",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="adjust_amount",
                        description=(
                            "Update the transaction amount to the calculated "
                            "valid value using Decimal precision"
                        ),
                        handler_key="adjust_amount",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_adjustment",
                        description=(
                            "Verify the adjusted amount falls within the "
                            "acceptable range and recalculate dependent totals"
                        ),
                        handler_key="verify_adjustment",
                    ),
                ],
                success_rate=0.85,
                applicable_error_types=[
                    "amount_error",
                    "range_error",
                    "tolerance_error",
                ],
            )
        )

        # 2. FIX_DATE_SEQUENCE — success rate 0.90
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_DATE_SEQUENCE.value,
                description=(
                    "Correct temporal ordering violations in transaction dates"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_date_violation",
                        description=(
                            "Analyse all date fields in the transaction to "
                            "identify which dates violate the expected sequence"
                        ),
                        handler_key="identify_date_violation",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="determine_correct_order",
                        description=(
                            "Calculate the correct chronological order based "
                            "on business rules (PO → receipt → invoice → pay)"
                        ),
                        handler_key="determine_correct_order",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_dates",
                        description=(
                            "Update the out-of-order date fields to restore "
                            "the correct temporal sequence"
                        ),
                        handler_key="update_dates",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_sequence",
                        description=(
                            "Verify all date fields now satisfy the required "
                            "chronological ordering constraints"
                        ),
                        handler_key="verify_sequence",
                    ),
                ],
                success_rate=0.90,
                applicable_error_types=[
                    "date_sequence_error",
                    "temporal_error",
                ],
            )
        )

        # 3. CORRECT_ENTITY_REFERENCE — success rate 0.80
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_ENTITY_REFERENCE.value,
                description=(
                    "Fix invalid or missing vendor/customer ID references"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="lookup_valid_entity",
                        description=(
                            "Search the master data for a valid entity that "
                            "matches the context of the transaction"
                        ),
                        handler_key="lookup_valid_entity",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="update_reference",
                        description=(
                            "Replace the invalid entity reference with the "
                            "correct entity ID from the lookup result"
                        ),
                        handler_key="update_reference",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="verify_entity_exists",
                        description=(
                            "Confirm the updated entity reference resolves to "
                            "an active entity in the master data"
                        ),
                        handler_key="verify_entity_exists",
                    ),
                ],
                success_rate=0.80,
                applicable_error_types=[
                    "entity_reference_error",
                    "missing_reference",
                ],
            )
        )

        # 4. REGENERATE_GL_ENTRY — success rate 0.75
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.REGENERATE_GL_ENTRY.value,
                description=(
                    "Delete and re-create an invalid GL journal entry"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="delete_invalid_entry",
                        description=(
                            "Mark the existing invalid GL journal entry for "
                            "deletion and log the reversal"
                        ),
                        handler_key="delete_invalid_entry",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="recalculate_amounts",
                        description=(
                            "Recalculate debit and credit amounts from the "
                            "source transaction using Decimal precision"
                        ),
                        handler_key="recalculate_amounts",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="create_new_entry",
                        description=(
                            "Create a new balanced GL journal entry with the "
                            "recalculated amounts (DR = CR within $0.01)"
                        ),
                        handler_key="create_new_entry",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_balance",
                        description=(
                            "Verify the new entry satisfies the GL balance "
                            "invariant (SUM debits = SUM credits within $0.01)"
                        ),
                        handler_key="verify_balance",
                    ),
                ],
                success_rate=0.75,
                applicable_error_types=[
                    "balance_error",
                    "gl_posting_error",
                    "journal_error",
                ],
            )
        )

        # 5. RECALCULATE_BALANCE — success rate 0.88
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.RECALCULATE_BALANCE.value,
                description=(
                    "Recalculate an account's running balance from its "
                    "journal entries"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="lock_account",
                        description=(
                            "Acquire a lock on the account balance record "
                            "to prevent concurrent modifications"
                        ),
                        handler_key="lock_account",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="recalculate_from_entries",
                        description=(
                            "Sum all journal entry line amounts for the "
                            "account to compute the correct running balance"
                        ),
                        handler_key="recalculate_from_entries",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_balance",
                        description=(
                            "Update the account balance record with the "
                            "recalculated value"
                        ),
                        handler_key="update_balance",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_trial_balance",
                        description=(
                            "Run a trial balance check to confirm the "
                            "overall balance equation still holds"
                        ),
                        handler_key="verify_trial_balance",
                    ),
                ],
                success_rate=0.88,
                applicable_error_types=[
                    "balance_error",
                    "running_balance_error",
                ],
            )
        )

        # 6. FIX_APPROVAL_CHAIN — success rate 0.82
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_APPROVAL_CHAIN.value,
                description=(
                    "Correct an incomplete or incorrect approval routing "
                    "chain for the transaction"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_approval_gap",
                        description=(
                            "Identify the missing or incorrect approval step "
                            "in the current chain based on thresholds"
                        ),
                        handler_key="identify_approval_gap",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="determine_required_approvers",
                        description=(
                            "Determine the correct approver roles based on "
                            "the transaction type and amount thresholds"
                        ),
                        handler_key="determine_required_approvers",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="route_for_approval",
                        description=(
                            "Route the transaction to the correct approver "
                            "via the workflow orchestrator"
                        ),
                        handler_key="route_for_approval",
                    ),
                ],
                success_rate=0.82,
                applicable_error_types=[
                    "approval_error",
                    "authorization_error",
                ],
            )
        )

        # 7. CORRECT_PERIOD_ASSIGNMENT — success rate 0.92
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_PERIOD_ASSIGNMENT.value,
                description=(
                    "Fix an incorrect fiscal period assignment on the "
                    "transaction"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_correct_period",
                        description=(
                            "Determine the correct fiscal period based on "
                            "the transaction date and fiscal calendar"
                        ),
                        handler_key="identify_correct_period",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="validate_period_open",
                        description=(
                            "Confirm the target fiscal period is in OPEN "
                            "state and can accept postings"
                        ),
                        handler_key="validate_period_open",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="reassign_period",
                        description=(
                            "Update the transaction's fiscal period to the "
                            "correct value"
                        ),
                        handler_key="reassign_period",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_assignment",
                        description=(
                            "Verify the transaction's period assignment is "
                            "consistent with its posting date"
                        ),
                        handler_key="verify_assignment",
                    ),
                ],
                success_rate=0.92,
                applicable_error_types=[
                    "period_error",
                    "closed_period_error",
                ],
            )
        )

        # 8. RELINK_DOCUMENTS — success rate 0.78
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.RELINK_DOCUMENTS.value,
                description=(
                    "Reconnect broken PO/receipt/invoice document linkages"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_related_documents",
                        description=(
                            "Search for related documents (PO, receipt, "
                            "invoice) that should be linked to this "
                            "transaction"
                        ),
                        handler_key="identify_related_documents",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="establish_linkages",
                        description=(
                            "Create the foreign-key linkages between the "
                            "transaction and its related documents"
                        ),
                        handler_key="establish_linkages",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="verify_chain_complete",
                        description=(
                            "Verify the complete document chain is "
                            "consistent and all references resolve"
                        ),
                        handler_key="verify_chain_complete",
                    ),
                ],
                success_rate=0.78,
                applicable_error_types=[
                    "document_linkage_error",
                    "orphan_document_error",
                ],
            )
        )

        # 9. FIX_THREE_WAY_MATCH — success rate 0.70
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_THREE_WAY_MATCH.value,
                description=(
                    "Resolve three-way match failures between PO, receipt, "
                    "and invoice"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="retrieve_po_receipt_invoice",
                        description=(
                            "Retrieve the linked PO, goods receipt, and "
                            "vendor invoice for comparison"
                        ),
                        handler_key="retrieve_po_receipt_invoice",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="identify_variance",
                        description=(
                            "Calculate price and quantity variances between "
                            "the three documents"
                        ),
                        handler_key="identify_variance",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="adjust_within_tolerance",
                        description=(
                            "Adjust amounts or quantities to bring variances "
                            "within tolerance (price ±5%, quantity ±2%)"
                        ),
                        handler_key="adjust_within_tolerance",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_match",
                        description=(
                            "Re-run the three-way match and verify all "
                            "variances are within tolerance"
                        ),
                        handler_key="verify_match",
                    ),
                ],
                success_rate=0.70,
                applicable_error_types=[
                    "three_way_match_error",
                    "tolerance_error",
                ],
            )
        )

        # 10. ADJUST_PAYMENT_ALLOCATION — success rate 0.83
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.ADJUST_PAYMENT_ALLOCATION.value,
                description=(
                    "Fix FIFO payment allocation errors across invoices"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="retrieve_open_invoices",
                        description=(
                            "Retrieve all open invoices for the "
                            "vendor/customer sorted by due date"
                        ),
                        handler_key="retrieve_open_invoices",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="sort_by_date_fifo",
                        description=(
                            "Sort invoices by date ascending to enforce "
                            "FIFO (oldest invoice first) allocation order"
                        ),
                        handler_key="sort_by_date_fifo",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="reallocate_payments",
                        description=(
                            "Reallocate the payment amount across invoices "
                            "in FIFO order using Decimal precision"
                        ),
                        handler_key="reallocate_payments",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_allocation",
                        description=(
                            "Verify no invoice has a negative balance and "
                            "excess creates unapplied cash"
                        ),
                        handler_key="verify_allocation",
                    ),
                ],
                success_rate=0.83,
                applicable_error_types=[
                    "payment_allocation_error",
                    "fifo_error",
                ],
            )
        )

        # 11. CORRECT_TAX_CALCULATION — success rate 0.87
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_TAX_CALCULATION.value,
                description=(
                    "Fix incorrect tax amount calculation on the transaction"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="retrieve_tax_rates",
                        description=(
                            "Look up the applicable tax rates from "
                            "configuration for this transaction type"
                        ),
                        handler_key="retrieve_tax_rates",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="recalculate_tax",
                        description=(
                            "Recalculate tax amounts for each line item "
                            "using the correct rates and Decimal precision"
                        ),
                        handler_key="recalculate_tax",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_line_items",
                        description=(
                            "Update line item tax amounts and recalculate "
                            "the transaction total"
                        ),
                        handler_key="update_line_items",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_totals",
                        description=(
                            "Verify the transaction total equals the sum of "
                            "line items including corrected tax"
                        ),
                        handler_key="verify_totals",
                    ),
                ],
                success_rate=0.87,
                applicable_error_types=[
                    "tax_error",
                    "calculation_error",
                ],
            )
        )

        # 12. FIX_CURRENCY_ROUNDING — success rate 0.93
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_CURRENCY_ROUNDING.value,
                description=(
                    "Resolve rounding discrepancies in monetary amounts"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_rounding_issue",
                        description=(
                            "Identify fields where rounding errors exist "
                            "by comparing Decimal-precise values against "
                            "stored amounts"
                        ),
                        handler_key="identify_rounding_issue",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="apply_half_up_rounding",
                        description=(
                            "Apply ROUND_HALF_UP rounding to all monetary "
                            "fields using Decimal with precision 28"
                        ),
                        handler_key="apply_half_up_rounding",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="verify_precision",
                        description=(
                            "Verify all monetary amounts are correctly "
                            "rounded and totals are consistent"
                        ),
                        handler_key="verify_precision",
                    ),
                ],
                success_rate=0.93,
                applicable_error_types=[
                    "rounding_error",
                    "precision_error",
                ],
            )
        )

        # 13. CORRECT_DISCOUNT_APPLICATION — success rate 0.86
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_DISCOUNT_APPLICATION.value,
                description=(
                    "Fix incorrectly applied discount calculation on the "
                    "transaction"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="validate_discount_rules",
                        description=(
                            "Retrieve and validate the discount rules "
                            "applicable to this transaction"
                        ),
                        handler_key="validate_discount_rules",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="recalculate_discount",
                        description=(
                            "Recalculate the discount amount using the "
                            "correct percentage and Decimal precision"
                        ),
                        handler_key="recalculate_discount",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_amounts",
                        description=(
                            "Update the discount amount, net amount, and "
                            "total on the transaction and its line items"
                        ),
                        handler_key="update_amounts",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_totals",
                        description=(
                            "Verify the updated totals are consistent "
                            "(gross - discount = net)"
                        ),
                        handler_key="verify_totals",
                    ),
                ],
                success_rate=0.86,
                applicable_error_types=[
                    "discount_error",
                    "pricing_error",
                ],
            )
        )

        # 14. FIX_DUPLICATE_DETECTION — success rate 0.72
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_DUPLICATE_DETECTION.value,
                description=(
                    "Resolve duplicate reference errors by identifying and "
                    "removing duplicate records"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_duplicate_records",
                        description=(
                            "Search for records matching the same vendor, "
                            "amount, date, and reference number"
                        ),
                        handler_key="identify_duplicate_records",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="determine_primary",
                        description=(
                            "Determine which record is the primary (original) "
                            "based on creation timestamp"
                        ),
                        handler_key="determine_primary",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="remove_duplicate",
                        description=(
                            "Mark the duplicate record as void and reverse "
                            "any associated GL entries"
                        ),
                        handler_key="remove_duplicate",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_uniqueness",
                        description=(
                            "Verify that the uniqueness constraint is now "
                            "satisfied for the primary record"
                        ),
                        handler_key="verify_uniqueness",
                    ),
                ],
                success_rate=0.72,
                applicable_error_types=[
                    "duplicate_error",
                    "uniqueness_error",
                ],
            )
        )

        # 15. REGENERATE_DOCUMENT_NUMBER — success rate 0.95
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.REGENERATE_DOCUMENT_NUMBER.value,
                description=(
                    "Fix document number sequence errors by generating a "
                    "new valid number"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="get_next_sequence",
                        description=(
                            "Fetch the next available document number from "
                            "the sequence generator for this document type"
                        ),
                        handler_key="get_next_sequence",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="assign_new_number",
                        description=(
                            "Assign the new document number to the "
                            "transaction record"
                        ),
                        handler_key="assign_new_number",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="verify_uniqueness",
                        description=(
                            "Verify the new document number is unique across "
                            "the entire document type namespace"
                        ),
                        handler_key="verify_uniqueness",
                    ),
                ],
                success_rate=0.95,
                applicable_error_types=[
                    "document_number_error",
                    "sequence_error",
                ],
            )
        )

        # 16. CORRECT_QUANTITY_VARIANCE — success rate 0.81
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_QUANTITY_VARIANCE.value,
                description=(
                    "Resolve quantity mismatches between PO, receipt, and "
                    "invoice"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="compare_po_receipt_invoice_qty",
                        description=(
                            "Compare quantities on the PO, goods receipt, "
                            "and invoice for each line item"
                        ),
                        handler_key="compare_po_receipt_invoice_qty",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="identify_variance_source",
                        description=(
                            "Determine which document contains the incorrect "
                            "quantity and the direction of the variance"
                        ),
                        handler_key="identify_variance_source",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="adjust_quantity",
                        description=(
                            "Adjust the quantity to bring it within the "
                            "±2% tolerance threshold"
                        ),
                        handler_key="adjust_quantity",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_match",
                        description=(
                            "Verify the adjusted quantities pass the "
                            "three-way match tolerance checks"
                        ),
                        handler_key="verify_match",
                    ),
                ],
                success_rate=0.81,
                applicable_error_types=[
                    "quantity_error",
                    "variance_error",
                ],
            )
        )

        # 17. FIX_CREDIT_CHECK — success rate 0.77
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_CREDIT_CHECK.value,
                description=(
                    "Resolve credit limit failures on customer orders"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="retrieve_credit_limit",
                        description=(
                            "Fetch the customer's credit limit and current "
                            "AR balance from master data"
                        ),
                        handler_key="retrieve_credit_limit",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="calculate_current_exposure",
                        description=(
                            "Calculate total current credit exposure "
                            "including this order"
                        ),
                        handler_key="calculate_current_exposure",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="adjust_order_amount",
                        description=(
                            "Reduce the order amount to fit within the "
                            "remaining credit headroom"
                        ),
                        handler_key="adjust_order_amount",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_within_limit",
                        description=(
                            "Verify the adjusted order amount keeps total "
                            "exposure within the credit limit"
                        ),
                        handler_key="verify_within_limit",
                    ),
                ],
                success_rate=0.77,
                applicable_error_types=[
                    "credit_error",
                    "limit_error",
                ],
            )
        )

        # 18. CORRECT_POSTING_ACCOUNTS — success rate 0.84
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_POSTING_ACCOUNTS.value,
                description=(
                    "Fix incorrect GL account codes on journal entry lines"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="validate_account_codes",
                        description=(
                            "Validate each account code against the Chart "
                            "of Accounts (is_posting=TRUE check)"
                        ),
                        handler_key="validate_account_codes",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="lookup_correct_accounts",
                        description=(
                            "Look up the correct posting accounts based on "
                            "the transaction type and posting rules"
                        ),
                        handler_key="lookup_correct_accounts",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_posting",
                        description=(
                            "Update the journal entry lines with the correct "
                            "account codes"
                        ),
                        handler_key="update_posting",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_accounts",
                        description=(
                            "Verify all account codes are valid posting "
                            "accounts in the Chart of Accounts"
                        ),
                        handler_key="verify_accounts",
                    ),
                ],
                success_rate=0.84,
                applicable_error_types=[
                    "account_error",
                    "chart_of_accounts_error",
                ],
            )
        )

        # 19. FIX_PAYMENT_TERMS — success rate 0.89
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_PAYMENT_TERMS.value,
                description=(
                    "Correct payment term dates and due date calculations"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="retrieve_payment_terms",
                        description=(
                            "Retrieve the payment terms from the "
                            "vendor/customer master data"
                        ),
                        handler_key="retrieve_payment_terms",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="recalculate_due_date",
                        description=(
                            "Recalculate the due date based on the invoice "
                            "date and payment terms"
                        ),
                        handler_key="recalculate_due_date",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_terms",
                        description=(
                            "Update the payment terms code and calculated "
                            "due date on the transaction"
                        ),
                        handler_key="update_terms",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_dates",
                        description=(
                            "Verify the due date is consistent with the "
                            "payment terms and invoice date"
                        ),
                        handler_key="verify_dates",
                    ),
                ],
                success_rate=0.89,
                applicable_error_types=[
                    "payment_terms_error",
                    "due_date_error",
                ],
            )
        )

        # 20. REGENERATE_TRACKING_NUMBER — success rate 0.96
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.REGENERATE_TRACKING_NUMBER.value,
                description=(
                    "Fix invalid shipment tracking numbers by generating "
                    "new ones"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="generate_new_tracking",
                        description=(
                            "Generate a new tracking number matching the "
                            "carrier's format requirements"
                        ),
                        handler_key="generate_new_tracking",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="update_shipment_record",
                        description=(
                            "Update the shipment record with the new "
                            "tracking number"
                        ),
                        handler_key="update_shipment_record",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="verify_tracking",
                        description=(
                            "Verify the new tracking number is unique and "
                            "meets the carrier's format specification"
                        ),
                        handler_key="verify_tracking",
                    ),
                ],
                success_rate=0.96,
                applicable_error_types=[
                    "tracking_error",
                    "shipment_error",
                ],
            )
        )

        # 21. CORRECT_STATUS_TRANSITION — success rate 0.79
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.CORRECT_STATUS_TRANSITION.value,
                description=(
                    "Fix invalid status transitions in the transaction "
                    "lifecycle"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="identify_current_status",
                        description=(
                            "Read the current transaction status and the "
                            "attempted target status"
                        ),
                        handler_key="identify_current_status",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="determine_valid_transitions",
                        description=(
                            "Look up valid status transitions from the "
                            "transaction lifecycle rules"
                        ),
                        handler_key="determine_valid_transitions",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_status",
                        description=(
                            "Update the transaction to the nearest valid "
                            "status on the path to the target"
                        ),
                        handler_key="update_status",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_lifecycle",
                        description=(
                            "Verify the transaction's status history forms "
                            "a valid lifecycle path"
                        ),
                        handler_key="verify_lifecycle",
                    ),
                ],
                success_rate=0.79,
                applicable_error_types=[
                    "status_error",
                    "lifecycle_error",
                ],
            )
        )

        # 22. FIX_INVENTORY_BALANCE — success rate 0.76
        self.register_scenario(
            FixScenario(
                name=FixScenarioName.FIX_INVENTORY_BALANCE.value,
                description=(
                    "Correct inventory quantity balance from stock movements"
                ),
                steps=[
                    FixStep(
                        step_order=1,
                        step_name="lock_inventory_record",
                        description=(
                            "Acquire a lock on the inventory balance record "
                            "to prevent concurrent modifications"
                        ),
                        handler_key="lock_inventory_record",
                    ),
                    FixStep(
                        step_order=2,
                        step_name="recalculate_from_movements",
                        description=(
                            "Sum all stock movement transactions (receipts, "
                            "issues, adjustments) to compute the correct "
                            "on-hand quantity"
                        ),
                        handler_key="recalculate_from_movements",
                    ),
                    FixStep(
                        step_order=3,
                        step_name="update_balance",
                        description=(
                            "Update the inventory balance record with the "
                            "recalculated quantity"
                        ),
                        handler_key="update_balance",
                    ),
                    FixStep(
                        step_order=4,
                        step_name="verify_stock",
                        description=(
                            "Verify the updated stock balance is non-negative "
                            "and consistent with the GL inventory account"
                        ),
                        handler_key="verify_stock",
                    ),
                ],
                success_rate=0.76,
                applicable_error_types=[
                    "inventory_error",
                    "stock_error",
                ],
            )
        )


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "FixScenarioCatalog",
    "FixScenario",
    "FixStep",
    "FixScenarioName",
]
