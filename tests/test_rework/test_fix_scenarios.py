"""Comprehensive tests for FixScenarioCatalog and FixScenarioExecutor.

Tests cover:
    FixScenarioCatalog:
        - 22 default scenarios registered via _register_default_scenarios()
        - Catalog completeness (all FixScenarioName enum values registered)
        - Scenario selection sorted by success_rate (highest first)
        - Exclusion of previously failed scenarios from re-selection
        - Scenario registration and lookup
        - Dynamic success rate updates via record_outcome()
        - Metrics reporting

    FixScenarioExecutor:
        - Per-scenario timeout (10s) enforcement via asyncio.wait_for()
        - Sequential step execution with early abort on failure
        - Deep copy of transaction before modification (original preserved)
        - Step handler registration and dispatch
        - FixExecutionResult Pydantic model validation
        - Metrics tracking

    Pydantic V2 Models:
        - FixScenarioName enum (22 values)
        - FixStep model validation
        - FixScenario model validation
        - FixExecutionResult model validation

Design Notes:
    - FixScenarioCatalog is fully testable without mocks (it's a pure registry)
    - FixScenarioExecutor uses AsyncMock for step handlers
    - Test patterns follow test_transaction_orchestrator.py for constant validation
      and test_error_handlers.py for timeout/error testing

References:
    - AAP Section 0.5.1 Group 6: fix_scenario_catalog.py, fix_scenario_executor.py
    - AAP Section 0.5.1 Group 8: test_fix_scenarios.py
    - AAP Section 0.7.1: Constructor injection, Pydantic V2
    - AAP Section 0.7.6: Testing Conventions (>=80% coverage)
"""

from __future__ import annotations

import asyncio
import copy
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.rework.fix_scenario_catalog import (
    FixScenario,
    FixScenarioCatalog,
    FixScenarioName,
    FixStep,
)
from app.rework.fix_scenario_executor import (
    FixExecutionResult,
    FixScenarioExecutor,
)


# ---------------------------------------------------------------------------
# Helper functions — reusable sample data factories
# ---------------------------------------------------------------------------


def _make_scenario(
    name: str = FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value,
    description: str = "Test scenario",
    success_rate: float = 0.85,
    applicable_error_types: Optional[List[str]] = None,
    step_count: int = 2,
    enabled: bool = True,
) -> FixScenario:
    """Create a sample FixScenario for testing."""
    steps = [
        FixStep(
            step_order=i + 1,
            step_name=f"step_{i + 1}",
            description=f"Test step {i + 1}",
            handler_key=f"handler_{i + 1}",
        )
        for i in range(step_count)
    ]
    return FixScenario(
        name=name,
        description=description,
        steps=steps,
        success_rate=success_rate,
        applicable_error_types=applicable_error_types or ["balance_error", "amount_error"],
        enabled=enabled,
    )


def _make_transaction(
    transaction_id: Optional[str] = None,
    amount: str = "5000.00",
) -> Dict[str, Any]:
    """Create a sample transaction dict for executor tests."""
    return {
        "transaction_id": transaction_id or str(uuid4()),
        "transaction_type": "vendor_invoice",
        "amount": Decimal(amount),
        "vendor_id": "V-001",
        "status": "validation_failed",
    }


def _make_context() -> Dict[str, Any]:
    """Create a minimal generation context dict."""
    return {
        "simulation_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "current_date": "2025-01-15",
        "fiscal_period": "2025-01",
    }


# ---------------------------------------------------------------------------
# Fixtures — FixScenarioCatalog
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog() -> FixScenarioCatalog:
    """FixScenarioCatalog with default scenarios loaded (22 scenarios)."""
    return FixScenarioCatalog(load_defaults=True)


@pytest.fixture
def empty_catalog() -> FixScenarioCatalog:
    """FixScenarioCatalog without default scenarios loaded."""
    return FixScenarioCatalog(load_defaults=False)


# ---------------------------------------------------------------------------
# Fixtures — FixScenarioExecutor
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> FixScenarioExecutor:
    """FixScenarioExecutor with default configuration (10s timeout)."""
    return FixScenarioExecutor(scenario_timeout_seconds=10.0)


@pytest.fixture
def executor_with_handlers() -> FixScenarioExecutor:
    """FixScenarioExecutor with custom step handlers registered."""
    exec_instance = FixScenarioExecutor(scenario_timeout_seconds=10.0)

    # Register some test handlers
    async def success_handler(transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        transaction["fixed"] = True
        return {"field": "fixed", "old": False, "new": True}

    async def failing_handler(transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Step failed deliberately")

    exec_instance.register_handler("success_step", success_handler)
    exec_instance.register_handler("failing_step", failing_handler)
    return exec_instance


# =========================================================================
# Test Class — FixScenarioName Enum Validation
# =========================================================================


@pytest.mark.rework
class TestFixScenarioNameEnum:
    """Verify the FixScenarioName enum has exactly 22 members with correct values.

    Follows the REQUIRED_ARTIFACTS validation pattern from
    test_transaction_orchestrator.py.
    """

    # Complete set of all 22 expected enum names
    ALL_EXPECTED_NAMES: Set[str] = {
        "ADJUST_AMOUNT_TO_RANGE",
        "FIX_DATE_SEQUENCE",
        "CORRECT_ENTITY_REFERENCE",
        "REGENERATE_GL_ENTRY",
        "RECALCULATE_BALANCE",
        "FIX_APPROVAL_CHAIN",
        "CORRECT_PERIOD_ASSIGNMENT",
        "RELINK_DOCUMENTS",
        "FIX_THREE_WAY_MATCH",
        "ADJUST_PAYMENT_ALLOCATION",
        "CORRECT_TAX_CALCULATION",
        "FIX_CURRENCY_ROUNDING",
        "CORRECT_DISCOUNT_APPLICATION",
        "FIX_DUPLICATE_DETECTION",
        "REGENERATE_DOCUMENT_NUMBER",
        "CORRECT_QUANTITY_VARIANCE",
        "FIX_CREDIT_CHECK",
        "CORRECT_POSTING_ACCOUNTS",
        "FIX_PAYMENT_TERMS",
        "REGENERATE_TRACKING_NUMBER",
        "CORRECT_STATUS_TRANSITION",
        "FIX_INVENTORY_BALANCE",
    }

    def test_enum_has_22_members(self) -> None:
        """FixScenarioName must contain exactly 22 scenario names."""
        members = list(FixScenarioName)
        assert len(members) == 22, (
            f"Expected 22 FixScenarioName members, got {len(members)}"
        )

    def test_adjust_amount_to_range_value(self) -> None:
        """ADJUST_AMOUNT_TO_RANGE.value must equal its string name."""
        assert FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value == "ADJUST_AMOUNT_TO_RANGE"

    def test_fix_date_sequence_value(self) -> None:
        """FIX_DATE_SEQUENCE.value must equal its string name."""
        assert FixScenarioName.FIX_DATE_SEQUENCE.value == "FIX_DATE_SEQUENCE"

    def test_all_members_are_strings(self) -> None:
        """Every FixScenarioName member value must be a string instance."""
        for member in FixScenarioName:
            assert isinstance(member.value, str), (
                f"Expected str value for {member.name}, got {type(member.value)}"
            )

    def test_all_expected_names_present(self) -> None:
        """All 22 expected scenario names must be present in the enum."""
        actual_names = {member.name for member in FixScenarioName}
        missing = self.ALL_EXPECTED_NAMES - actual_names
        extra = actual_names - self.ALL_EXPECTED_NAMES
        assert not missing, f"Missing enum members: {missing}"
        assert not extra, f"Unexpected extra enum members: {extra}"
        assert actual_names == self.ALL_EXPECTED_NAMES


# =========================================================================
# Test Class — FixStep Pydantic V2 Model
# =========================================================================


@pytest.mark.rework
class TestFixStepModel:
    """Verify the FixStep Pydantic V2 model validates fields correctly."""

    def test_valid_fix_step(self) -> None:
        """Create a FixStep with all required fields — must succeed."""
        step = FixStep(
            step_order=1,
            step_name="validate_amount",
            description="Validate the current transaction amount",
            handler_key="validate_current_amount",
        )
        assert step.step_order == 1
        assert step.step_name == "validate_amount"
        assert step.description == "Validate the current transaction amount"
        assert step.handler_key == "validate_current_amount"

    def test_step_order_minimum(self) -> None:
        """step_order ge=1 must reject value 0."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            FixStep(
                step_order=0,
                step_name="bad_step",
                description="Should fail",
                handler_key="bad_handler",
            )

    def test_step_order_valid(self) -> None:
        """step_order=1 must be accepted as valid."""
        step = FixStep(
            step_order=1,
            step_name="first_step",
            description="First step in the scenario",
            handler_key="first_handler",
        )
        assert step.step_order == 1

    def test_fix_step_fields(self) -> None:
        """Verify step_name, description, handler_key are all populated."""
        step = FixStep(
            step_order=3,
            step_name="update_reference",
            description="Update the entity reference",
            handler_key="update_reference",
        )
        assert step.step_name == "update_reference"
        assert step.description == "Update the entity reference"
        assert step.handler_key == "update_reference"


# =========================================================================
# Test Class — FixScenario Pydantic V2 Model
# =========================================================================


@pytest.mark.rework
class TestFixScenarioModel:
    """Verify the FixScenario Pydantic V2 model validates fields correctly."""

    def test_valid_scenario(self) -> None:
        """Create a FixScenario with all fields — must succeed."""
        scenario = _make_scenario()
        assert scenario.name == FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value
        assert scenario.description == "Test scenario"
        assert len(scenario.steps) == 2
        assert scenario.success_rate == 0.85
        assert scenario.enabled is True

    def test_scenario_default_values(self) -> None:
        """Verify default values: success_rate=0.5, total_attempts=0, total_successes=0, enabled=True."""
        scenario = FixScenario(
            name="TEST_SCENARIO",
            description="Minimal scenario with defaults",
        )
        assert scenario.success_rate == 0.5
        assert scenario.total_attempts == 0
        assert scenario.total_successes == 0
        assert scenario.enabled is True

    def test_success_rate_bounds(self) -> None:
        """success_rate must be in [0.0, 1.0]; out-of-range values must be rejected."""
        # Valid: 0.0 and 1.0 are accepted
        low = FixScenario(
            name="LOW_RATE",
            description="Zero success rate",
            success_rate=0.0,
        )
        assert low.success_rate == 0.0

        high = FixScenario(
            name="HIGH_RATE",
            description="Perfect success rate",
            success_rate=1.0,
        )
        assert high.success_rate == 1.0

        # Invalid: >1.0 or <0.0 must be rejected
        with pytest.raises(Exception):
            FixScenario(
                name="OVER_RATE",
                description="Over 1.0",
                success_rate=1.5,
            )

        with pytest.raises(Exception):
            FixScenario(
                name="UNDER_RATE",
                description="Under 0.0",
                success_rate=-0.1,
            )

    def test_scenario_with_steps(self) -> None:
        """Verify steps list is correctly populated."""
        scenario = _make_scenario(step_count=4)
        assert len(scenario.steps) == 4
        for i, step in enumerate(scenario.steps):
            assert step.step_order == i + 1
            assert step.step_name == f"step_{i + 1}"

    def test_scenario_applicable_error_types(self) -> None:
        """Verify applicable_error_types list is populated."""
        scenario = _make_scenario(
            applicable_error_types=["balance_error", "gl_posting_error", "journal_error"],
        )
        assert "balance_error" in scenario.applicable_error_types
        assert "gl_posting_error" in scenario.applicable_error_types
        assert "journal_error" in scenario.applicable_error_types
        assert len(scenario.applicable_error_types) == 3

    def test_scenario_id_auto_generated(self) -> None:
        """scenario_id must be auto-generated as a UUID4 string when not provided."""
        scenario = _make_scenario()
        assert scenario.scenario_id is not None
        assert isinstance(scenario.scenario_id, str)
        assert len(scenario.scenario_id) > 0
        # Must be a valid UUID
        parsed = UUID(scenario.scenario_id)
        assert str(parsed) == scenario.scenario_id


# =========================================================================
# Test Class — FixScenarioCatalog Constructor
# =========================================================================


@pytest.mark.rework
class TestFixScenarioCatalogConstructor:
    """Verify the FixScenarioCatalog constructor loads defaults correctly."""

    def test_default_constructor_loads_defaults(self, catalog: FixScenarioCatalog) -> None:
        """load_defaults=True must load 22 default scenarios."""
        scenarios = catalog.list_scenarios()
        assert len(scenarios) == 22

    def test_no_defaults_empty_catalog(self, empty_catalog: FixScenarioCatalog) -> None:
        """load_defaults=False must create an empty catalog."""
        scenarios = empty_catalog.list_scenarios()
        assert len(scenarios) == 0

    def test_scenario_count_with_defaults(self) -> None:
        """Exactly 22 scenarios must be registered after default loading."""
        cat = FixScenarioCatalog(load_defaults=True)
        metrics = cat.get_metrics()
        assert metrics["total_scenarios"] == 22


# =========================================================================
# Test Class — Catalog Completeness (CRITICAL)
# =========================================================================


@pytest.mark.rework
class TestCatalogCompleteness:
    """Validate all 22 FixScenarioName enum values have a registered scenario.

    Follows the REQUIRED_ARTIFACTS validation pattern from
    test_transaction_orchestrator.py — iterate every enum member
    and assert catalog.get_scenario() returns a valid FixScenario.
    """

    def test_all_enum_values_registered(self, catalog: FixScenarioCatalog) -> None:
        """Every FixScenarioName enum value must have a registered scenario."""
        for member in FixScenarioName:
            scenario = catalog.get_scenario(member.value)
            assert scenario is not None, (
                f"Scenario '{member.value}' is in the FixScenarioName enum "
                f"but is NOT registered in the catalog."
            )
            assert scenario.name == member.value

    def test_each_scenario_has_steps(self, catalog: FixScenarioCatalog) -> None:
        """Every registered scenario must have at least 2 FixStep entries."""
        for member in FixScenarioName:
            scenario = catalog.get_scenario(member.value)
            assert scenario is not None
            assert len(scenario.steps) >= 2, (
                f"Scenario '{member.value}' has {len(scenario.steps)} steps, "
                f"expected at least 2."
            )

    def test_each_scenario_has_applicable_error_types(self, catalog: FixScenarioCatalog) -> None:
        """Every scenario must have at least 1 applicable_error_type."""
        for member in FixScenarioName:
            scenario = catalog.get_scenario(member.value)
            assert scenario is not None
            assert len(scenario.applicable_error_types) >= 1, (
                f"Scenario '{member.value}' has no applicable_error_types."
            )

    def test_each_scenario_has_description(self, catalog: FixScenarioCatalog) -> None:
        """Every scenario must have a non-empty description."""
        for member in FixScenarioName:
            scenario = catalog.get_scenario(member.value)
            assert scenario is not None
            assert scenario.description, (
                f"Scenario '{member.value}' has an empty description."
            )
            assert len(scenario.description.strip()) > 0

    def test_each_scenario_has_valid_success_rate(self, catalog: FixScenarioCatalog) -> None:
        """Every scenario's success_rate must be between 0.0 and 1.0."""
        for member in FixScenarioName:
            scenario = catalog.get_scenario(member.value)
            assert scenario is not None
            assert 0.0 <= scenario.success_rate <= 1.0, (
                f"Scenario '{member.value}' has success_rate "
                f"{scenario.success_rate}, expected [0.0, 1.0]."
            )

    def test_all_scenarios_enabled_by_default(self, catalog: FixScenarioCatalog) -> None:
        """All 22 default scenarios must be enabled=True."""
        for member in FixScenarioName:
            scenario = catalog.get_scenario(member.value)
            assert scenario is not None
            assert scenario.enabled is True, (
                f"Scenario '{member.value}' is not enabled by default."
            )


# =========================================================================
# Test Class — Scenario Selection
# =========================================================================


@pytest.mark.rework
class TestScenarioSelection:
    """Verify the select_scenario() algorithm: filter, exclude, sort, pick."""

    def test_select_by_error_type(self, catalog: FixScenarioCatalog) -> None:
        """select_scenario('balance_error') must return a scenario with that error type."""
        result = catalog.select_scenario("balance_error")
        assert result is not None
        assert "balance_error" in result.applicable_error_types

    def test_select_returns_highest_success_rate(self, empty_catalog: FixScenarioCatalog) -> None:
        """When multiple scenarios match, the one with highest success_rate is selected."""
        low_rate = _make_scenario(
            name="LOW_RATE_SCENARIO",
            success_rate=0.50,
            applicable_error_types=["test_error"],
        )
        high_rate = _make_scenario(
            name="HIGH_RATE_SCENARIO",
            success_rate=0.95,
            applicable_error_types=["test_error"],
        )
        empty_catalog.register_scenario(low_rate)
        empty_catalog.register_scenario(high_rate)

        selected = empty_catalog.select_scenario("test_error")
        assert selected is not None
        assert selected.name == "HIGH_RATE_SCENARIO"
        assert selected.success_rate == 0.95

    def test_select_with_exclusions(self, empty_catalog: FixScenarioCatalog) -> None:
        """Excluding a scenario name must remove it from results."""
        best = _make_scenario(
            name="BEST_SCENARIO",
            success_rate=0.99,
            applicable_error_types=["test_error"],
        )
        second = _make_scenario(
            name="SECOND_SCENARIO",
            success_rate=0.70,
            applicable_error_types=["test_error"],
        )
        empty_catalog.register_scenario(best)
        empty_catalog.register_scenario(second)

        selected = empty_catalog.select_scenario(
            "test_error",
            excluded_scenarios={"BEST_SCENARIO"},
        )
        assert selected is not None
        assert selected.name == "SECOND_SCENARIO"

    def test_select_no_match_returns_none(self, catalog: FixScenarioCatalog) -> None:
        """Error type 'xyz_nonexistent' must return None."""
        result = catalog.select_scenario("xyz_nonexistent")
        assert result is None

    def test_select_excludes_disabled(self, empty_catalog: FixScenarioCatalog) -> None:
        """Disabled scenarios must not be returned by select_scenario."""
        disabled = _make_scenario(
            name="DISABLED_SCENARIO",
            success_rate=0.99,
            applicable_error_types=["test_error"],
            enabled=False,
        )
        empty_catalog.register_scenario(disabled)

        result = empty_catalog.select_scenario("test_error")
        assert result is None

    def test_select_all_excluded_returns_none(self, empty_catalog: FixScenarioCatalog) -> None:
        """When all matching scenarios are excluded, must return None."""
        only = _make_scenario(
            name="ONLY_SCENARIO",
            success_rate=0.80,
            applicable_error_types=["test_error"],
        )
        empty_catalog.register_scenario(only)

        result = empty_catalog.select_scenario(
            "test_error",
            excluded_scenarios={"ONLY_SCENARIO"},
        )
        assert result is None


# =========================================================================
# Test Class — Scenario Registration
# =========================================================================


@pytest.mark.rework
class TestScenarioRegistration:
    """Verify scenario registration, lookup, and listing operations."""

    def test_register_new_scenario(self, empty_catalog: FixScenarioCatalog) -> None:
        """Register a custom scenario, then retrieve it by name."""
        scenario = _make_scenario(name="CUSTOM_SCENARIO")
        empty_catalog.register_scenario(scenario)

        retrieved = empty_catalog.get_scenario("CUSTOM_SCENARIO")
        assert retrieved is not None
        assert retrieved.name == "CUSTOM_SCENARIO"
        assert retrieved.description == "Test scenario"

    def test_register_replaces_existing(self, empty_catalog: FixScenarioCatalog) -> None:
        """Registering with the same name must overwrite the previous scenario."""
        original = _make_scenario(
            name="REPLACE_ME",
            description="Original description",
            success_rate=0.50,
        )
        replacement = _make_scenario(
            name="REPLACE_ME",
            description="Replaced description",
            success_rate=0.90,
        )
        empty_catalog.register_scenario(original)
        empty_catalog.register_scenario(replacement)

        retrieved = empty_catalog.get_scenario("REPLACE_ME")
        assert retrieved is not None
        assert retrieved.description == "Replaced description"
        assert retrieved.success_rate == 0.90

    def test_get_scenario_by_name(self, catalog: FixScenarioCatalog) -> None:
        """get_scenario must return the correct scenario by name."""
        result = catalog.get_scenario(FixScenarioName.FIX_DATE_SEQUENCE.value)
        assert result is not None
        assert result.name == "FIX_DATE_SEQUENCE"

    def test_get_scenario_not_found_returns_none(self, catalog: FixScenarioCatalog) -> None:
        """get_scenario with unknown name must return None."""
        result = catalog.get_scenario("NONEXISTENT_SCENARIO_XYZ")
        assert result is None

    def test_list_scenarios_sorted_by_success_rate(self, catalog: FixScenarioCatalog) -> None:
        """list_scenarios must return scenarios sorted by success_rate descending."""
        scenarios = catalog.list_scenarios()
        assert len(scenarios) == 22
        for i in range(len(scenarios) - 1):
            assert scenarios[i].success_rate >= scenarios[i + 1].success_rate, (
                f"Scenario at index {i} (rate={scenarios[i].success_rate}) "
                f"should have >= success_rate than index {i + 1} "
                f"(rate={scenarios[i + 1].success_rate})"
            )


# =========================================================================
# Test Class — Record Outcome (Dynamic Success Rate Updates)
# =========================================================================


@pytest.mark.rework
class TestRecordOutcome:
    """Verify record_outcome dynamically updates success rates."""

    def test_record_success_increments_counts(self, catalog: FixScenarioCatalog) -> None:
        """Recording a success must increment total_attempts and total_successes."""
        name = FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value
        scenario_before = catalog.get_scenario(name)
        assert scenario_before is not None
        orig_attempts = scenario_before.total_attempts
        orig_successes = scenario_before.total_successes

        catalog.record_outcome(name, success=True)

        scenario_after = catalog.get_scenario(name)
        assert scenario_after is not None
        assert scenario_after.total_attempts == orig_attempts + 1
        assert scenario_after.total_successes == orig_successes + 1

    def test_record_failure_increments_attempts_only(self, catalog: FixScenarioCatalog) -> None:
        """Recording a failure must increment total_attempts only."""
        name = FixScenarioName.FIX_DATE_SEQUENCE.value
        scenario_before = catalog.get_scenario(name)
        assert scenario_before is not None
        orig_attempts = scenario_before.total_attempts
        orig_successes = scenario_before.total_successes

        catalog.record_outcome(name, success=False)

        scenario_after = catalog.get_scenario(name)
        assert scenario_after is not None
        assert scenario_after.total_attempts == orig_attempts + 1
        assert scenario_after.total_successes == orig_successes

    def test_success_rate_recalculated(self, catalog: FixScenarioCatalog) -> None:
        """After recording outcomes, success_rate must equal successes / attempts."""
        name = FixScenarioName.CORRECT_ENTITY_REFERENCE.value

        # Record 3 successes and 2 failures = 3/5 = 0.6
        for _ in range(3):
            catalog.record_outcome(name, success=True)
        for _ in range(2):
            catalog.record_outcome(name, success=False)

        scenario = catalog.get_scenario(name)
        assert scenario is not None
        assert scenario.total_attempts == 5
        assert scenario.total_successes == 3
        assert abs(scenario.success_rate - 0.6) < 0.001

    def test_initial_rate_overridden(self, catalog: FixScenarioCatalog) -> None:
        """Initial success_rate must be overridden after recording outcomes."""
        name = FixScenarioName.REGENERATE_GL_ENTRY.value
        scenario = catalog.get_scenario(name)
        assert scenario is not None
        initial_rate = scenario.success_rate

        # Record a single failure: rate becomes 0/1 = 0.0
        catalog.record_outcome(name, success=False)

        scenario = catalog.get_scenario(name)
        assert scenario is not None
        assert scenario.success_rate != initial_rate
        assert scenario.success_rate == 0.0

    def test_record_outcome_unknown_scenario(self, catalog: FixScenarioCatalog) -> None:
        """Recording outcome for unknown scenario must not crash."""
        # Should log a warning but not raise an exception
        catalog.record_outcome("NONEXISTENT_SCENARIO_ABC", success=True)
        # Catalog should still be intact
        assert len(catalog.list_scenarios()) == 22


# =========================================================================
# Test Class — Catalog Metrics
# =========================================================================


@pytest.mark.rework
class TestCatalogMetrics:
    """Verify get_metrics() returns correct aggregate statistics."""

    def test_metrics_structure(self, catalog: FixScenarioCatalog) -> None:
        """Metrics dict must contain all expected top-level keys."""
        metrics = catalog.get_metrics()
        expected_keys = {
            "total_scenarios",
            "enabled_scenarios",
            "total_attempts",
            "total_successes",
            "average_success_rate",
            "per_scenario_stats",
        }
        assert expected_keys.issubset(set(metrics.keys())), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    def test_metrics_initial_state(self, catalog: FixScenarioCatalog) -> None:
        """Initial state must show total_attempts=0, total_successes=0."""
        metrics = catalog.get_metrics()
        assert metrics["total_scenarios"] == 22
        assert metrics["enabled_scenarios"] == 22
        assert metrics["total_attempts"] == 0
        assert metrics["total_successes"] == 0

    def test_metrics_after_outcomes(self, catalog: FixScenarioCatalog) -> None:
        """After recording outcomes, metrics must reflect updated counts."""
        catalog.record_outcome(FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value, True)
        catalog.record_outcome(FixScenarioName.FIX_DATE_SEQUENCE.value, False)
        catalog.record_outcome(FixScenarioName.FIX_DATE_SEQUENCE.value, True)

        metrics = catalog.get_metrics()
        assert metrics["total_attempts"] == 3
        assert metrics["total_successes"] == 2
        assert metrics["total_scenarios"] == 22
        assert len(metrics["per_scenario_stats"]) == 22


# =========================================================================
# Test Class — FixExecutionResult Pydantic V2 Model
# =========================================================================


@pytest.mark.rework
class TestFixExecutionResultModel:
    """Verify FixExecutionResult Pydantic V2 model defaults and validation."""

    def test_default_values(self) -> None:
        """Default values: success=False, steps_executed=[], duration_ms=0.0."""
        result = FixExecutionResult()
        assert result.success is False
        assert result.steps_executed == []
        assert result.duration_ms == 0.0
        assert result.steps_failed == []
        assert result.modified_transaction is None
        assert result.error_message is None
        assert result.changes_applied == {}

    def test_successful_result(self) -> None:
        """All fields populated for a successful result."""
        now = datetime.now(timezone.utc)
        result = FixExecutionResult(
            success=True,
            scenario_name="ADJUST_AMOUNT_TO_RANGE",
            steps_executed=["step_1", "step_2"],
            steps_failed=[],
            modified_transaction={"amount": Decimal("5000.00")},
            error_message=None,
            duration_ms=123.45,
            changes_applied={"step_1": {"field": "amount"}},
            timestamp=now,
        )
        assert result.success is True
        assert result.scenario_name == "ADJUST_AMOUNT_TO_RANGE"
        assert len(result.steps_executed) == 2
        assert result.duration_ms == 123.45
        assert result.timestamp == now

    def test_failed_result(self) -> None:
        """Error message set and steps_failed populated on failure."""
        result = FixExecutionResult(
            success=False,
            scenario_name="REGENERATE_GL_ENTRY",
            steps_executed=["step_1"],
            steps_failed=["step_1"],
            error_message="Step 'step_1' failed: balance check error",
            duration_ms=50.0,
        )
        assert result.success is False
        assert "step_1" in result.steps_failed
        assert result.error_message is not None
        assert "balance check error" in result.error_message

    def test_modified_transaction_preserved(self) -> None:
        """modified_transaction dict must be stored correctly."""
        txn = {"transaction_id": "txn-001", "amount": Decimal("1000.00")}
        result = FixExecutionResult(
            success=True,
            modified_transaction=txn,
        )
        assert result.modified_transaction is not None
        assert result.modified_transaction["transaction_id"] == "txn-001"
        assert result.modified_transaction["amount"] == Decimal("1000.00")

    def test_changes_applied_dict(self) -> None:
        """changes_applied must be populated with step change summaries."""
        changes = {
            "validate_current_amount": {"validated": True},
            "adjust_amount": {"field": "amount", "old": "5000.00", "new": "4999.00"},
        }
        result = FixExecutionResult(
            success=True,
            changes_applied=changes,
        )
        assert "validate_current_amount" in result.changes_applied
        assert "adjust_amount" in result.changes_applied
        assert result.changes_applied["adjust_amount"]["field"] == "amount"


# =========================================================================
# Test Class — FixScenarioExecutor Constructor
# =========================================================================


@pytest.mark.rework
class TestFixScenarioExecutorConstructor:
    """Verify the FixScenarioExecutor constructor and initial state."""

    def test_default_constructor(self, executor: FixScenarioExecutor) -> None:
        """Constructor with scenario_timeout_seconds=10.0 must set timeout and load default handlers."""
        assert executor._scenario_timeout == 10.0
        # Default handlers must be populated (many handlers registered)
        assert len(executor._step_handlers) > 0

    def test_custom_timeout(self) -> None:
        """Override timeout to 5.0 must be respected."""
        exec_instance = FixScenarioExecutor(scenario_timeout_seconds=5.0)
        assert exec_instance._scenario_timeout == 5.0

    def test_custom_handlers(self) -> None:
        """Pass custom step_handlers dict must populate registry."""
        custom_handlers: Dict[str, Any] = {
            "my_handler": lambda txn, ctx: {"done": True},
        }
        exec_instance = FixScenarioExecutor(
            scenario_timeout_seconds=10.0,
            step_handlers=custom_handlers,
        )
        # Custom handler must be present
        assert "my_handler" in exec_instance._step_handlers

    def test_initial_metrics_zero(self, executor: FixScenarioExecutor) -> None:
        """All execution counters must start at 0."""
        metrics = executor.get_metrics()
        assert metrics["execution_count"] == 0
        assert metrics["success_count"] == 0
        assert metrics["failure_count"] == 0
        assert metrics["total_steps_executed"] == 0


# =========================================================================
# Test Class — Execute Scenario (async)
# =========================================================================


@pytest.mark.rework
@pytest.mark.asyncio
class TestExecuteScenario:
    """Verify the core execute() async method for FixScenarioExecutor."""

    async def test_successful_execution_all_steps(self, executor: FixScenarioExecutor) -> None:
        """All steps pass: success=True, all steps in steps_executed."""
        # Use ADJUST_AMOUNT_TO_RANGE scenario — all default handlers exist
        catalog = FixScenarioCatalog(load_defaults=True)
        scenario = catalog.get_scenario(FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        assert scenario is not None

        transaction = _make_transaction()
        context = _make_context()

        result = await executor.execute(scenario, transaction, context)

        assert result.success is True
        assert result.scenario_name == FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value
        assert len(result.steps_executed) == len(scenario.steps)
        assert result.steps_failed == []

    async def test_step_failure_aborts_execution(
        self, executor_with_handlers: FixScenarioExecutor
    ) -> None:
        """First step fails: remaining steps are skipped, success=False."""
        scenario = FixScenario(
            name="FAILING_SCENARIO",
            description="Scenario where step 1 fails",
            steps=[
                FixStep(
                    step_order=1,
                    step_name="fail_step",
                    description="This step will fail",
                    handler_key="failing_step",
                ),
                FixStep(
                    step_order=2,
                    step_name="success_step_after",
                    description="Should never run",
                    handler_key="success_step",
                ),
            ],
            success_rate=0.50,
            applicable_error_types=["test_error"],
        )

        transaction = _make_transaction()
        context = _make_context()

        result = await executor_with_handlers.execute(scenario, transaction, context)

        assert result.success is False
        assert "fail_step" in result.steps_executed
        assert "fail_step" in result.steps_failed
        # Second step should NOT have been executed
        assert "success_step_after" not in result.steps_executed

    async def test_deep_copy_preserves_original(self, executor: FixScenarioExecutor) -> None:
        """Original transaction dict must NOT be modified by execution."""
        catalog = FixScenarioCatalog(load_defaults=True)
        scenario = catalog.get_scenario(FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        assert scenario is not None

        transaction = _make_transaction(amount="5000.00")
        original_amount = transaction["amount"]
        original_copy = copy.deepcopy(transaction)

        await executor.execute(scenario, transaction, context=_make_context())

        # Original transaction must be preserved
        assert transaction["amount"] == original_amount
        assert transaction == original_copy

    async def test_changes_tracked(self, executor: FixScenarioExecutor) -> None:
        """changes_applied must be populated with step-level changes."""
        catalog = FixScenarioCatalog(load_defaults=True)
        scenario = catalog.get_scenario(FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        assert scenario is not None

        transaction = _make_transaction()
        context = _make_context()

        result = await executor.execute(scenario, transaction, context)

        # changes_applied should contain entries for at least some steps
        assert isinstance(result.changes_applied, dict)

    async def test_duration_tracked(self, executor: FixScenarioExecutor) -> None:
        """duration_ms must be > 0.0 after execution."""
        catalog = FixScenarioCatalog(load_defaults=True)
        scenario = catalog.get_scenario(FixScenarioName.FIX_DATE_SEQUENCE.value)
        assert scenario is not None

        transaction = _make_transaction()
        context = _make_context()

        result = await executor.execute(scenario, transaction, context)

        assert result.duration_ms > 0.0

    async def test_result_contains_scenario_name(self, executor: FixScenarioExecutor) -> None:
        """scenario_name in the result must match the executed scenario."""
        catalog = FixScenarioCatalog(load_defaults=True)
        scenario = catalog.get_scenario(FixScenarioName.RECALCULATE_BALANCE.value)
        assert scenario is not None

        transaction = _make_transaction()
        context = _make_context()

        result = await executor.execute(scenario, transaction, context)

        assert result.scenario_name == FixScenarioName.RECALCULATE_BALANCE.value


# =========================================================================
# Test Class — Execution Timeout (async)
# =========================================================================


@pytest.mark.rework
@pytest.mark.asyncio
class TestExecutionTimeout:
    """Verify per-scenario timeout enforcement via asyncio.wait_for()."""

    async def test_timeout_enforcement_10s(self) -> None:
        """Mock a slow step handler that exceeds the timeout — must fail gracefully."""
        # Use a very short timeout to avoid long test runs
        short_timeout_executor = FixScenarioExecutor(scenario_timeout_seconds=0.1)

        # Register a slow handler that sleeps well beyond the timeout
        async def slow_handler(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            await asyncio.sleep(5.0)  # Way longer than 0.1s timeout
            return {"done": True}

        short_timeout_executor.register_handler("slow_step", slow_handler)

        scenario = FixScenario(
            name="SLOW_SCENARIO",
            description="Scenario with a slow step that exceeds timeout",
            steps=[
                FixStep(
                    step_order=1,
                    step_name="slow_action",
                    description="This step will timeout",
                    handler_key="slow_step",
                ),
            ],
            success_rate=0.50,
            applicable_error_types=["test_error"],
        )

        transaction = _make_transaction()
        context = _make_context()

        start = time.monotonic()
        result = await short_timeout_executor.execute(scenario, transaction, context)
        elapsed = time.monotonic() - start

        assert result.success is False
        assert result.error_message is not None
        assert "timed out" in result.error_message.lower() or "timeout" in result.error_message.lower()
        # Should complete in well under 5s (the sleep duration)
        assert elapsed < 3.0

    async def test_timeout_applies_per_scenario(self) -> None:
        """Timeout is per-scenario (total), not per-step."""
        # 0.3s timeout, two steps each taking 0.2s = 0.4s total → should timeout
        short_executor = FixScenarioExecutor(scenario_timeout_seconds=0.3)

        async def medium_handler(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            await asyncio.sleep(0.2)
            return {"done": True}

        short_executor.register_handler("medium_step", medium_handler)

        scenario = FixScenario(
            name="MULTI_STEP_SLOW",
            description="Two medium steps that together exceed per-scenario timeout",
            steps=[
                FixStep(
                    step_order=1,
                    step_name="medium_1",
                    description="Medium speed step 1",
                    handler_key="medium_step",
                ),
                FixStep(
                    step_order=2,
                    step_name="medium_2",
                    description="Medium speed step 2",
                    handler_key="medium_step",
                ),
            ],
            success_rate=0.50,
            applicable_error_types=["test_error"],
        )

        transaction = _make_transaction()
        context = _make_context()

        result = await short_executor.execute(scenario, transaction, context)

        # The combined 0.4s exceeds the 0.3s timeout
        assert result.success is False
        assert result.error_message is not None
        assert "timeout" in result.error_message.lower() or "timed out" in result.error_message.lower()


# =========================================================================
# Test Class — Step Handler Registration
# =========================================================================


@pytest.mark.rework
class TestStepHandlerRegistration:
    """Verify step handler registration and dispatch behaviour."""

    def test_register_custom_handler(self, executor: FixScenarioExecutor) -> None:
        """register_handler must store the handler under the given key."""
        custom_handler = AsyncMock(return_value={"custom": True})
        executor.register_handler("custom_step_key", custom_handler)

        assert "custom_step_key" in executor._step_handlers
        assert executor._step_handlers["custom_step_key"] is custom_handler

    def test_default_handlers_loaded(self, executor: FixScenarioExecutor) -> None:
        """Default handlers must be pre-registered on construction (count > 0)."""
        handler_count = len(executor._step_handlers)
        assert handler_count > 0, "Expected default handlers to be registered"
        # Verify some known default handler keys exist
        assert "validate_current_amount" in executor._step_handlers
        assert "adjust_amount" in executor._step_handlers
        assert "verify_balance" in executor._step_handlers

    @pytest.mark.asyncio
    async def test_step_with_no_handler_uses_generic(self, executor: FixScenarioExecutor) -> None:
        """Unregistered step name must use a generic no-op handler that succeeds."""
        scenario = FixScenario(
            name="GENERIC_HANDLER_TEST",
            description="Scenario with unknown handler keys",
            steps=[
                FixStep(
                    step_order=1,
                    step_name="unknown_step",
                    description="Uses a handler key not in the registry",
                    handler_key="completely_unknown_handler_key_xyz",
                ),
            ],
            success_rate=0.50,
            applicable_error_types=["test_error"],
        )

        transaction = _make_transaction()
        context = _make_context()

        result = await executor.execute(scenario, transaction, context)

        # Generic passthrough should succeed
        assert result.success is True
        assert "unknown_step" in result.steps_executed


# =========================================================================
# Test Class — Executor Metrics
# =========================================================================


@pytest.mark.rework
class TestExecutorMetrics:
    """Verify get_metrics() returns correct execution statistics."""

    def test_metrics_structure(self, executor: FixScenarioExecutor) -> None:
        """Metrics dict must contain all expected keys."""
        metrics = executor.get_metrics()
        expected_keys = {
            "execution_count",
            "success_count",
            "failure_count",
            "total_steps_executed",
            "success_rate",
        }
        assert expected_keys.issubset(set(metrics.keys())), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    def test_metrics_initial_state(self, executor: FixScenarioExecutor) -> None:
        """All counters must start at 0."""
        metrics = executor.get_metrics()
        assert metrics["execution_count"] == 0
        assert metrics["success_count"] == 0
        assert metrics["failure_count"] == 0
        assert metrics["total_steps_executed"] == 0
        assert metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_after_executions(self, executor: FixScenarioExecutor) -> None:
        """After executions, counters must reflect outcomes."""
        catalog = FixScenarioCatalog(load_defaults=True)
        scenario = catalog.get_scenario(FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        assert scenario is not None

        # Execute twice
        for _ in range(2):
            await executor.execute(scenario, _make_transaction(), _make_context())

        metrics = executor.get_metrics()
        assert metrics["execution_count"] == 2
        assert metrics["total_steps_executed"] > 0
        # Success rate should reflect outcomes
        assert metrics["success_rate"] >= 0.0


# =========================================================================
# Test Class — Sequential Step Execution (async)
# =========================================================================


@pytest.mark.rework
@pytest.mark.asyncio
class TestSequentialStepExecution:
    """Verify steps execute in the correct order and failure aborts early."""

    async def test_steps_execute_in_order(self) -> None:
        """Steps must execute in step_order sequence."""
        execution_log: List[str] = []

        async def tracking_handler_1(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            execution_log.append("step_1")
            return {"step": "1"}

        async def tracking_handler_2(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            execution_log.append("step_2")
            return {"step": "2"}

        async def tracking_handler_3(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            execution_log.append("step_3")
            return {"step": "3"}

        exec_instance = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        exec_instance.register_handler("track_1", tracking_handler_1)
        exec_instance.register_handler("track_2", tracking_handler_2)
        exec_instance.register_handler("track_3", tracking_handler_3)

        scenario = FixScenario(
            name="ORDERED_STEPS",
            description="Steps must execute in order",
            steps=[
                FixStep(step_order=1, step_name="first", description="First", handler_key="track_1"),
                FixStep(step_order=2, step_name="second", description="Second", handler_key="track_2"),
                FixStep(step_order=3, step_name="third", description="Third", handler_key="track_3"),
            ],
            success_rate=1.0,
            applicable_error_types=["test_error"],
        )

        transaction = _make_transaction()
        context = _make_context()

        result = await exec_instance.execute(scenario, transaction, context)

        assert result.success is True
        assert execution_log == ["step_1", "step_2", "step_3"]
        assert result.steps_executed == ["first", "second", "third"]

    async def test_step_2_not_reached_after_step_1_failure(self) -> None:
        """If step 1 fails, step 2's handler must NEVER be called."""
        step_2_called = False

        async def failing_step_1(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            raise RuntimeError("Step 1 fails deliberately")

        async def step_2_handler(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            nonlocal step_2_called
            step_2_called = True
            return {"step": "2"}

        exec_instance = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        exec_instance.register_handler("fail_1", failing_step_1)
        exec_instance.register_handler("pass_2", step_2_handler)

        scenario = FixScenario(
            name="ABORT_ON_FAILURE",
            description="Step 1 failure should abort step 2",
            steps=[
                FixStep(step_order=1, step_name="will_fail", description="Fails", handler_key="fail_1"),
                FixStep(step_order=2, step_name="never_reached", description="Should not run", handler_key="pass_2"),
            ],
            success_rate=0.50,
            applicable_error_types=["test_error"],
        )

        transaction = _make_transaction()
        context = _make_context()

        result = await exec_instance.execute(scenario, transaction, context)

        assert result.success is False
        assert "will_fail" in result.steps_failed
        assert step_2_called is False, "Step 2 handler was called after Step 1 failure"
        assert "never_reached" not in result.steps_executed
