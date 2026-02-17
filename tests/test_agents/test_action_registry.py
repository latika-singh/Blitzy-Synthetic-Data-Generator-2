"""
Comprehensive tests for the ActionRegistry (app/agents/action_registry.py).

Validates all 14 ERP action types across five functional areas (Procurement,
Warehouse, AP, AR, Accounting), their registration, property structure, lookup,
input validation, async execution, functional groupings, and edge cases.

Per AAP Section 0.5.1 Group 10 and AAP Section 0.7.5:
  - Coverage target: >=80%
  - Async tests use @pytest.mark.asyncio (via pytest-asyncio)
  - All tests are self-contained (ActionRegistry has no external deps)

Test Classes:
  TestActionRegistration  — 14 action count, IDs, uniqueness, custom registration
  TestActionProperties    — action_id, required_inputs, validation_rules, handler
  TestActionLookup        — action_exists, get_action, get_all_action_ids
  TestInputValidation     — valid data, missing fields, wrong types, custom rules
  TestActionExecution     — async execute with valid/invalid inputs, handler wiring
  TestActionGroupings     — category-based grouping verification
  TestActionRegistryEdgeCases — empty inputs, None inputs, case sensitivity, extras
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import asyncio
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import pytest

# ---------------------------------------------------------------------------
# Internal Imports — Module Under Test
# ---------------------------------------------------------------------------
from app.agents.action_registry import (
    ActionDefinition,
    ActionRegistry,
    ActionResult,
    ValidationRule,
)


# ---------------------------------------------------------------------------
# Constants — Canonical Action IDs (README.md lines 119-134)
# ---------------------------------------------------------------------------
ALL_14_ACTION_IDS = [
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

# Mapping of valid inputs for each of the 14 actions — used by parametrized
# validation and execution tests.  Every required field is present with a
# value of the correct type.
VALID_INPUTS_MAP = {
    "create_purchase_order": {
        "vendor_id": "V-100",
        "items": [{"product_id": "P-1", "quantity": 10}],
        "total_amount": 5000.0,
        "company_id": "COMP-01",
    },
    "approve_purchase_order": {
        "po_id": "PO-100",
        "approver_id": "EMP-200",
        "decision": "approve",
    },
    "receive_goods": {
        "po_id": "PO-100",
        "received_items": [{"product_id": "P-1", "quantity": 10}],
        "warehouse_id": "WH-01",
    },
    "process_vendor_invoice": {
        "invoice_id": "INV-100",
        "vendor_id": "V-100",
        "amount": 4500.0,
        "line_items": [{"description": "Widgets", "amount": 4500.0}],
    },
    "match_three_way": {
        "invoice_id": "INV-100",
        "po_id": "PO-100",
        "receipt_id": "RCV-100",
    },
    "approve_invoice": {
        "invoice_id": "INV-100",
        "approver_id": "EMP-200",
        "decision": "approve",
    },
    "schedule_payment": {
        "invoice_id": "INV-100",
        "payment_date": "2025-06-15",
        "amount": 4500.0,
        "bank_account": "BA-001",
    },
    "create_sales_order": {
        "customer_id": "C-100",
        "items": [{"product_id": "P-2", "quantity": 5}],
        "total_amount": 2500.0,
    },
    "ship_order": {
        "order_id": "SO-100",
        "warehouse_id": "WH-01",
        "carrier_id": "CR-100",
    },
    "create_customer_invoice": {
        "order_id": "SO-100",
        "customer_id": "C-100",
        "amount": 2500.0,
        "line_items": [{"description": "Gadgets", "amount": 2500.0}],
    },
    "apply_payment": {
        "payment_id": "PAY-100",
        "invoice_id": "INV-200",
        "amount": 2500.0,
    },
    "create_journal_entry": {
        "entries": [
            {"account": "5000", "debit": 1000.0, "credit": 0.0},
            {"account": "2000", "debit": 0.0, "credit": 1000.0},
        ],
        "description": "Test journal entry",
        "total_debit": 1000.0,
        "total_credit": 1000.0,
    },
    "reconcile_account": {
        "account_id": "GL-5000",
        "period": "2025-06",
        "transactions": [{"id": "TXN-1", "amount": 100.0}],
    },
    "close_period": {
        "period": "2025-06",
        "fiscal_year": "FY2025",
    },
}


# ---------------------------------------------------------------------------
# Test-Scoped Fixtures
# ---------------------------------------------------------------------------
# NOTE: The conftest.py ``action_registry`` fixture provides a real
# ActionRegistry.  We also define a local ``empty_registry`` and
# ``mock_agent`` fixture for targeted use in this module.


@pytest.fixture
def empty_registry() -> ActionRegistry:
    """Return an ActionRegistry that has been cleared of all default actions.

    Useful for testing the register_action() mechanism in isolation.
    """
    registry = ActionRegistry()
    # Clear internal state manually so we start empty
    registry._actions.clear()
    registry._categories.clear()
    return registry


@pytest.fixture
def mock_agent() -> MagicMock:
    """Return a lightweight MagicMock representing an agent for execute() calls."""
    agent = MagicMock()
    agent.agent_id = "agent-test-001"
    agent.role = "ap_clerk"
    return agent


# =========================================================================
# Section 1: Test Action Registration
# =========================================================================


class TestActionRegistration:
    """Verify the ActionRegistry initialises with exactly 14 canonical actions."""

    def test_registry_has_14_actions(self, action_registry: ActionRegistry) -> None:
        """ActionRegistry must contain exactly 14 registered actions."""
        all_ids = action_registry.get_all_action_ids()
        assert len(all_ids) == 14, f"Expected 14 actions, got {len(all_ids)}"

    def test_all_14_action_ids_present(self, action_registry: ActionRegistry) -> None:
        """Every canonical action ID must be present in the registry."""
        registered_ids = set(action_registry.get_all_action_ids())
        for action_id in ALL_14_ACTION_IDS:
            assert action_id in registered_ids, (
                f"Action '{action_id}' is missing from the registry"
            )

    def test_action_ids_are_strings(self, action_registry: ActionRegistry) -> None:
        """All action IDs must be of type str."""
        for action_id in action_registry.get_all_action_ids():
            assert isinstance(action_id, str), (
                f"Action ID '{action_id}' is not a string"
            )

    def test_no_duplicate_action_ids(self, action_registry: ActionRegistry) -> None:
        """All 14 action IDs must be unique (no duplicates)."""
        all_ids = action_registry.get_all_action_ids()
        assert len(all_ids) == len(set(all_ids)), (
            "Duplicate action IDs detected"
        )

    def test_register_custom_action(self, action_registry: ActionRegistry) -> None:
        """Registering a new custom action increases the total count."""
        initial_count = len(action_registry.get_all_action_ids())

        custom = ActionDefinition(
            action_id="custom_test_action",
            description="A custom action for testing registration",
            required_inputs={"data": str},
            category="general",
        )
        action_registry.register_action(custom)

        new_count = len(action_registry.get_all_action_ids())
        assert new_count == initial_count + 1
        assert action_registry.action_exists("custom_test_action")

    def test_register_duplicate_overwrites(
        self, action_registry: ActionRegistry
    ) -> None:
        """Re-registering an existing action_id overwrites the previous definition."""
        original = action_registry.get_action("close_period")
        assert original is not None

        replacement = ActionDefinition(
            action_id="close_period",
            description="Replaced close_period for test",
            required_inputs={"period": str},
            category="accounting",
        )
        action_registry.register_action(replacement)

        updated = action_registry.get_action("close_period")
        assert updated is not None
        assert updated.description == "Replaced close_period for test"
        # Total count must remain 14 (overwrite, not add)
        assert len(action_registry.get_all_action_ids()) == 14

    def test_register_action_on_empty_registry(
        self, empty_registry: ActionRegistry
    ) -> None:
        """Registering on an empty registry starts from zero."""
        assert len(empty_registry.get_all_action_ids()) == 0

        action = ActionDefinition(
            action_id="test_action",
            description="First action on empty registry",
            required_inputs={"x": str},
            category="general",
        )
        empty_registry.register_action(action)

        assert len(empty_registry.get_all_action_ids()) == 1
        assert empty_registry.action_exists("test_action")


# =========================================================================
# Section 2: Test Action Properties
# =========================================================================


class TestActionProperties:
    """Verify structural properties of every registered action."""

    def test_each_action_has_action_id(
        self, action_registry: ActionRegistry
    ) -> None:
        """Every action returned by get_action() must have a str action_id."""
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            assert isinstance(action.action_id, str)
            assert action.action_id == action_id

    def test_each_action_has_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """Every action must declare required_inputs as a dict."""
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            assert isinstance(action.required_inputs, dict), (
                f"Action '{action_id}' required_inputs is not a dict"
            )
            assert len(action.required_inputs) > 0, (
                f"Action '{action_id}' has no required_inputs"
            )

    def test_each_action_has_validation_rules(
        self, action_registry: ActionRegistry
    ) -> None:
        """Every action must have validation_rules as a list."""
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            assert isinstance(action.validation_rules, list), (
                f"Action '{action_id}' validation_rules is not a list"
            )

    def test_each_action_has_category(
        self, action_registry: ActionRegistry
    ) -> None:
        """Every action must have a non-empty category string."""
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            assert isinstance(action.category, str)
            assert len(action.category) > 0

    def test_each_action_has_description(
        self, action_registry: ActionRegistry
    ) -> None:
        """Every action must have a non-empty description string."""
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            assert isinstance(action.description, str)
            assert len(action.description.strip()) > 0

    def test_validation_rules_are_validation_rule_instances(
        self, action_registry: ActionRegistry
    ) -> None:
        """Each element in validation_rules must be a ValidationRule dataclass."""
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            for rule in action.validation_rules:
                assert isinstance(rule, ValidationRule), (
                    f"Rule in '{action_id}' is {type(rule)}, "
                    f"expected ValidationRule"
                )
                assert isinstance(rule.field_name, str)
                assert isinstance(rule.rule_type, str)

    # --- Specific action input contracts ---------------------------------

    def test_create_purchase_order_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """create_purchase_order requires vendor_id, items, total_amount, company_id."""
        action = action_registry.get_action("create_purchase_order")
        assert action is not None
        expected_fields = {"vendor_id", "items", "total_amount", "company_id"}
        assert set(action.required_inputs.keys()) == expected_fields

    def test_process_vendor_invoice_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """process_vendor_invoice requires invoice_id, vendor_id, amount, line_items."""
        action = action_registry.get_action("process_vendor_invoice")
        assert action is not None
        expected_fields = {"invoice_id", "vendor_id", "amount", "line_items"}
        assert set(action.required_inputs.keys()) == expected_fields

    def test_create_journal_entry_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """create_journal_entry requires entries, description, total_debit, total_credit."""
        action = action_registry.get_action("create_journal_entry")
        assert action is not None
        expected_fields = {"entries", "description", "total_debit", "total_credit"}
        assert set(action.required_inputs.keys()) == expected_fields

    def test_apply_payment_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """apply_payment requires payment_id, invoice_id, amount."""
        action = action_registry.get_action("apply_payment")
        assert action is not None
        expected_fields = {"payment_id", "invoice_id", "amount"}
        assert set(action.required_inputs.keys()) == expected_fields

    def test_close_period_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """close_period requires period, fiscal_year."""
        action = action_registry.get_action("close_period")
        assert action is not None
        expected_fields = {"period", "fiscal_year"}
        assert set(action.required_inputs.keys()) == expected_fields

    def test_match_three_way_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """match_three_way requires invoice_id, po_id, receipt_id."""
        action = action_registry.get_action("match_three_way")
        assert action is not None
        expected_fields = {"invoice_id", "po_id", "receipt_id"}
        assert set(action.required_inputs.keys()) == expected_fields

    def test_ship_order_required_inputs(
        self, action_registry: ActionRegistry
    ) -> None:
        """ship_order requires order_id, warehouse_id, carrier_id."""
        action = action_registry.get_action("ship_order")
        assert action is not None
        expected_fields = {"order_id", "warehouse_id", "carrier_id"}
        assert set(action.required_inputs.keys()) == expected_fields


# =========================================================================
# Section 3: Test Action Lookup
# =========================================================================


class TestActionLookup:
    """Verify the lookup API: action_exists, get_action, get_all_action_ids."""

    def test_action_exists_returns_true_for_registered(
        self, action_registry: ActionRegistry
    ) -> None:
        """action_exists() returns True for every registered action."""
        for action_id in ALL_14_ACTION_IDS:
            assert action_registry.action_exists(action_id) is True, (
                f"action_exists('{action_id}') returned False"
            )

    def test_action_exists_returns_false_for_unknown(
        self, action_registry: ActionRegistry
    ) -> None:
        """action_exists() returns False for unregistered identifiers."""
        assert action_registry.action_exists("unknown_action") is False
        assert action_registry.action_exists("") is False
        assert action_registry.action_exists("delete_purchase_order") is False

    def test_get_action_returns_action_definition(
        self, action_registry: ActionRegistry
    ) -> None:
        """get_action() returns an ActionDefinition with correct properties."""
        action = action_registry.get_action("create_purchase_order")
        assert action is not None
        assert isinstance(action, ActionDefinition)
        assert action.action_id == "create_purchase_order"
        assert action.category == "procurement"
        assert "vendor_id" in action.required_inputs

    def test_get_action_returns_none_for_unknown(
        self, action_registry: ActionRegistry
    ) -> None:
        """get_action() returns None for unregistered action IDs."""
        result = action_registry.get_action("unknown_action")
        assert result is None

    def test_get_all_action_ids_returns_list(
        self, action_registry: ActionRegistry
    ) -> None:
        """get_all_action_ids() returns a list of 14 strings."""
        all_ids = action_registry.get_all_action_ids()
        assert isinstance(all_ids, list)
        assert len(all_ids) == 14
        for item in all_ids:
            assert isinstance(item, str)

    def test_get_all_action_ids_sorted(
        self, action_registry: ActionRegistry
    ) -> None:
        """get_all_action_ids() returns identifiers in sorted (ascending) order."""
        all_ids = action_registry.get_all_action_ids()
        assert all_ids == sorted(all_ids), (
            "get_all_action_ids() does not return sorted list"
        )

    def test_get_actions_by_category_returns_list(
        self, action_registry: ActionRegistry
    ) -> None:
        """get_actions_by_category() returns a list of ActionDefinition objects."""
        ap_actions = action_registry.get_actions_by_category("ap")
        assert isinstance(ap_actions, list)
        for action in ap_actions:
            assert isinstance(action, ActionDefinition)
            assert action.category == "ap"

    def test_get_actions_by_category_unknown_returns_empty(
        self, action_registry: ActionRegistry
    ) -> None:
        """Unknown category returns an empty list (no error)."""
        result = action_registry.get_actions_by_category("nonexistent_category")
        assert isinstance(result, list)
        assert len(result) == 0


# =========================================================================
# Section 4: Test Input Validation
# =========================================================================


class TestInputValidation:
    """Verify validate_inputs() for valid, invalid, and edge-case inputs."""

    def test_validate_inputs_passes_valid_data(
        self, action_registry: ActionRegistry
    ) -> None:
        """Valid inputs for create_purchase_order produce zero validation errors."""
        errors = action_registry.validate_inputs(
            "create_purchase_order",
            VALID_INPUTS_MAP["create_purchase_order"],
        )
        assert errors == [], f"Unexpected errors: {errors}"

    def test_validate_inputs_fails_missing_required(
        self, action_registry: ActionRegistry
    ) -> None:
        """Omitting a required field produces at least one validation error."""
        # Missing vendor_id for create_purchase_order
        inputs = {
            "items": [{"product_id": "P-1", "quantity": 10}],
            "total_amount": 5000.0,
            "company_id": "COMP-01",
        }
        errors = action_registry.validate_inputs("create_purchase_order", inputs)
        assert len(errors) > 0
        assert any("vendor_id" in e for e in errors), (
            "Error message should reference the missing 'vendor_id' field"
        )

    def test_validate_inputs_fails_wrong_type(
        self, action_registry: ActionRegistry
    ) -> None:
        """Providing a value of the wrong type produces a validation error."""
        inputs = {
            "vendor_id": 12345,  # int instead of str
            "items": [{"product_id": "P-1", "quantity": 10}],
            "total_amount": 5000.0,
            "company_id": "COMP-01",
        }
        errors = action_registry.validate_inputs("create_purchase_order", inputs)
        assert len(errors) > 0
        assert any("vendor_id" in e for e in errors)

    @pytest.mark.parametrize("action_id", ALL_14_ACTION_IDS)
    def test_validate_inputs_for_each_action(
        self, action_registry: ActionRegistry, action_id: str
    ) -> None:
        """All 14 actions pass validation when given correct inputs."""
        inputs = VALID_INPUTS_MAP[action_id]
        errors = action_registry.validate_inputs(action_id, inputs)
        assert errors == [], (
            f"Action '{action_id}' failed validation with valid inputs: {errors}"
        )

    def test_validation_rules_positive_number_enforced(
        self, action_registry: ActionRegistry
    ) -> None:
        """Custom rule: total_amount must be > 0 for create_purchase_order."""
        inputs = {
            "vendor_id": "V-100",
            "items": [{"product_id": "P-1", "quantity": 10}],
            "total_amount": -100.0,
            "company_id": "COMP-01",
        }
        errors = action_registry.validate_inputs("create_purchase_order", inputs)
        assert len(errors) > 0
        assert any("total_amount" in e for e in errors)

    def test_validation_rules_nonempty_list_enforced(
        self, action_registry: ActionRegistry
    ) -> None:
        """Custom rule: items must be a non-empty list."""
        inputs = {
            "vendor_id": "V-100",
            "items": [],
            "total_amount": 500.0,
            "company_id": "COMP-01",
        }
        errors = action_registry.validate_inputs("create_purchase_order", inputs)
        assert len(errors) > 0
        assert any("items" in e for e in errors)

    def test_validation_rules_valid_decision_enforced(
        self, action_registry: ActionRegistry
    ) -> None:
        """Custom rule: decision must be 'approve', 'reject', or 'escalate'."""
        inputs = {
            "po_id": "PO-100",
            "approver_id": "EMP-200",
            "decision": "maybe",  # invalid decision
        }
        errors = action_registry.validate_inputs("approve_purchase_order", inputs)
        assert len(errors) > 0
        assert any("decision" in e for e in errors)

    def test_validate_returns_error_messages_as_strings(
        self, action_registry: ActionRegistry
    ) -> None:
        """All validation error messages are human-readable strings."""
        errors = action_registry.validate_inputs("create_purchase_order", {})
        assert len(errors) > 0
        for error in errors:
            assert isinstance(error, str)
            assert len(error) > 0

    def test_validate_inputs_unknown_action(
        self, action_registry: ActionRegistry
    ) -> None:
        """Validating inputs for an unknown action returns an error."""
        errors = action_registry.validate_inputs("nonexistent_action", {"x": 1})
        assert len(errors) > 0
        assert any("Unknown action" in e or "nonexistent_action" in e for e in errors)

    def test_validate_balanced_entry_enforced(
        self, action_registry: ActionRegistry
    ) -> None:
        """create_journal_entry: total_debit must equal total_credit."""
        inputs = {
            "entries": [
                {"account": "5000", "debit": 1000.0, "credit": 0.0},
                {"account": "2000", "debit": 0.0, "credit": 500.0},
            ],
            "description": "Unbalanced test entry",
            "total_debit": 1000.0,
            "total_credit": 500.0,  # Intentionally unbalanced
        }
        errors = action_registry.validate_inputs("create_journal_entry", inputs)
        assert len(errors) > 0
        assert any("unbalanced" in e.lower() or "debit" in e.lower() for e in errors)

    def test_int_accepted_where_float_expected(
        self, action_registry: ActionRegistry
    ) -> None:
        """Numeric promotion: int is accepted where float is declared."""
        inputs = {
            "vendor_id": "V-100",
            "items": [{"product_id": "P-1", "quantity": 10}],
            "total_amount": 5000,  # int instead of float
            "company_id": "COMP-01",
        }
        errors = action_registry.validate_inputs("create_purchase_order", inputs)
        assert errors == [], f"Int should be accepted for float: {errors}"


# =========================================================================
# Section 5: Test Action Execution
# =========================================================================


class TestActionExecution:
    """Verify the async execute() method for success/failure paths."""

    @pytest.mark.asyncio
    async def test_execute_create_purchase_order(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Execute create_purchase_order with valid inputs returns success."""
        result = await action_registry.execute(
            "create_purchase_order",
            mock_agent,
            VALID_INPUTS_MAP["create_purchase_order"],
        )
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.action_id == "create_purchase_order"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_process_vendor_invoice(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Execute process_vendor_invoice with valid inputs returns success."""
        result = await action_registry.execute(
            "process_vendor_invoice",
            mock_agent,
            VALID_INPUTS_MAP["process_vendor_invoice"],
        )
        assert result.success is True
        assert result.action_id == "process_vendor_invoice"

    @pytest.mark.asyncio
    async def test_execute_create_journal_entry(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Execute create_journal_entry with balanced entries returns success."""
        result = await action_registry.execute(
            "create_journal_entry",
            mock_agent,
            VALID_INPUTS_MAP["create_journal_entry"],
        )
        assert result.success is True
        assert result.action_id == "create_journal_entry"

    @pytest.mark.asyncio
    async def test_execute_close_period(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Execute close_period with valid period data returns success."""
        result = await action_registry.execute(
            "close_period",
            mock_agent,
            VALID_INPUTS_MAP["close_period"],
        )
        assert result.success is True
        assert result.action_id == "close_period"

    @pytest.mark.asyncio
    async def test_execute_returns_action_result(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """execute() always returns an ActionResult instance."""
        result = await action_registry.execute(
            "apply_payment",
            mock_agent,
            VALID_INPUTS_MAP["apply_payment"],
        )
        assert isinstance(result, ActionResult)
        assert hasattr(result, "success")
        assert hasattr(result, "action_id")
        assert hasattr(result, "outputs")
        assert hasattr(result, "error")

    @pytest.mark.asyncio
    async def test_execute_with_invalid_inputs_fails(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """execute() with missing required inputs returns success=False."""
        result = await action_registry.execute(
            "create_purchase_order",
            mock_agent,
            {"vendor_id": "V-100"},  # Missing items, total_amount, company_id
        )
        assert result.success is False
        assert result.error is not None
        assert len(result.error) > 0

    @pytest.mark.asyncio
    async def test_execute_validates_before_executing(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Validation errors prevent the handler from running."""
        # Provide wrong types — validation should catch this before execution
        result = await action_registry.execute(
            "create_purchase_order",
            mock_agent,
            {
                "vendor_id": 999,  # wrong type
                "items": "not a list",  # wrong type
                "total_amount": "NaN",  # wrong type
                "company_id": 42,  # wrong type
            },
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_action_fails(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """execute() with an unregistered action_id returns success=False."""
        result = await action_registry.execute(
            "nonexistent_action",
            mock_agent,
            {"data": "test"},
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execute_passthrough_includes_action_id(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Default (no handler) execution echoes inputs plus _action_id and _status."""
        result = await action_registry.execute(
            "close_period",
            mock_agent,
            VALID_INPUTS_MAP["close_period"],
        )
        assert result.success is True
        assert result.outputs.get("_action_id") == "close_period"
        assert result.outputs.get("_status") == "completed"

    @pytest.mark.asyncio
    async def test_execute_with_custom_handler(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """When a handler is registered, it is invoked during execute()."""
        handler_called = False

        def custom_handler(agent: MagicMock, inputs: dict) -> dict:
            nonlocal handler_called
            handler_called = True
            return {"custom_result": "handled"}

        custom = ActionDefinition(
            action_id="custom_handled_action",
            description="Action with custom handler for test",
            required_inputs={"data": str},
            handler=custom_handler,
            category="general",
        )
        action_registry.register_action(custom)

        result = await action_registry.execute(
            "custom_handled_action",
            mock_agent,
            {"data": "test-payload"},
        )
        assert handler_called is True
        assert result.success is True
        assert result.outputs.get("custom_result") == "handled"

    @pytest.mark.asyncio
    async def test_execute_with_async_handler(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Async handlers are properly awaited during execute()."""
        async def async_handler(agent: MagicMock, inputs: dict) -> dict:
            await asyncio.sleep(0)  # yield control
            return {"async_result": "ok"}

        custom = ActionDefinition(
            action_id="async_handled_action",
            description="Action with async handler for test",
            required_inputs={"data": str},
            handler=async_handler,
            category="general",
        )
        action_registry.register_action(custom)

        result = await action_registry.execute(
            "async_handled_action",
            mock_agent,
            {"data": "test"},
        )
        assert result.success is True
        assert result.outputs.get("async_result") == "ok"

    @pytest.mark.asyncio
    async def test_execute_handler_exception_caught(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """If a handler raises an exception, execute returns success=False."""
        def failing_handler(agent: MagicMock, inputs: dict) -> dict:
            raise ValueError("Handler intentionally failed")

        custom = ActionDefinition(
            action_id="failing_action",
            description="Action that always fails",
            required_inputs={"data": str},
            handler=failing_handler,
            category="general",
        )
        action_registry.register_action(custom)

        result = await action_registry.execute(
            "failing_action",
            mock_agent,
            {"data": "trigger-error"},
        )
        assert result.success is False
        assert result.error is not None
        assert "Handler intentionally failed" in result.error

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action_id", ALL_14_ACTION_IDS)
    async def test_execute_all_14_actions_with_valid_inputs(
        self, action_registry: ActionRegistry, mock_agent: MagicMock,
        action_id: str
    ) -> None:
        """Every one of the 14 actions executes successfully with valid inputs."""
        result = await action_registry.execute(
            action_id,
            mock_agent,
            VALID_INPUTS_MAP[action_id],
        )
        assert result.success is True, (
            f"Action '{action_id}' failed: {result.error}"
        )
        assert result.action_id == action_id


# =========================================================================
# Section 6: Test Functional Groupings
# =========================================================================


class TestActionGroupings:
    """Verify actions are correctly grouped into ERP functional categories."""

    def test_procurement_actions(self, action_registry: ActionRegistry) -> None:
        """Procurement category contains create_purchase_order, approve_purchase_order."""
        procurement = action_registry.get_actions_by_category("procurement")
        procurement_ids = {a.action_id for a in procurement}
        assert procurement_ids == {
            "create_purchase_order",
            "approve_purchase_order",
        }

    def test_warehouse_actions(self, action_registry: ActionRegistry) -> None:
        """Warehouse category contains receive_goods."""
        warehouse = action_registry.get_actions_by_category("warehouse")
        warehouse_ids = {a.action_id for a in warehouse}
        # receive_goods is warehouse; ship_order is categorised under ar
        assert "receive_goods" in warehouse_ids

    def test_ap_actions(self, action_registry: ActionRegistry) -> None:
        """AP category contains process_vendor_invoice, match_three_way,
        approve_invoice, schedule_payment."""
        ap_actions = action_registry.get_actions_by_category("ap")
        ap_ids = {a.action_id for a in ap_actions}
        assert ap_ids == {
            "process_vendor_invoice",
            "match_three_way",
            "approve_invoice",
            "schedule_payment",
        }

    def test_ar_actions(self, action_registry: ActionRegistry) -> None:
        """AR category contains create_sales_order, ship_order,
        create_customer_invoice, apply_payment."""
        ar_actions = action_registry.get_actions_by_category("ar")
        ar_ids = {a.action_id for a in ar_actions}
        assert ar_ids == {
            "create_sales_order",
            "ship_order",
            "create_customer_invoice",
            "apply_payment",
        }

    def test_accounting_actions(self, action_registry: ActionRegistry) -> None:
        """Accounting category contains create_journal_entry,
        reconcile_account, close_period."""
        accounting = action_registry.get_actions_by_category("accounting")
        accounting_ids = {a.action_id for a in accounting}
        assert accounting_ids == {
            "create_journal_entry",
            "reconcile_account",
            "close_period",
        }

    def test_all_categories_covered(self, action_registry: ActionRegistry) -> None:
        """Every action belongs to exactly one of the five expected categories."""
        expected_categories = {"procurement", "warehouse", "ap", "ar", "accounting"}
        actual_categories = set()
        for action_id in action_registry.get_all_action_ids():
            action = action_registry.get_action(action_id)
            assert action is not None
            actual_categories.add(action.category)
        assert actual_categories == expected_categories

    def test_category_action_counts(self, action_registry: ActionRegistry) -> None:
        """Procurement=2, Warehouse=1, AP=4, AR=4, Accounting=3 → Total=14."""
        expected_counts = {
            "procurement": 2,
            "warehouse": 1,
            "ap": 4,
            "ar": 4,
            "accounting": 3,
        }
        for category, expected_count in expected_counts.items():
            actions = action_registry.get_actions_by_category(category)
            assert len(actions) == expected_count, (
                f"Category '{category}': expected {expected_count} actions, "
                f"got {len(actions)}"
            )


# =========================================================================
# Section 7: Test Edge Cases
# =========================================================================


class TestActionRegistryEdgeCases:
    """Verify graceful handling of edge cases and malformed input."""

    @pytest.mark.asyncio
    async def test_empty_inputs_dict(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """execute() with an empty dict fails validation for required fields."""
        result = await action_registry.execute(
            "create_purchase_order",
            mock_agent,
            {},
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_none_inputs_handled_gracefully(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """execute() with None-valued required fields fails gracefully."""
        inputs = {
            "vendor_id": None,
            "items": None,
            "total_amount": None,
            "company_id": None,
        }
        result = await action_registry.execute(
            "create_purchase_order",
            mock_agent,
            inputs,
        )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_extra_inputs_ignored(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Extra fields in inputs are ignored (no error from extra keys)."""
        inputs = dict(VALID_INPUTS_MAP["close_period"])
        inputs["extra_field"] = "should be ignored"
        inputs["another_extra"] = 42
        result = await action_registry.execute(
            "close_period",
            mock_agent,
            inputs,
        )
        assert result.success is True

    def test_action_id_case_sensitivity(
        self, action_registry: ActionRegistry
    ) -> None:
        """Action IDs are case-sensitive: uppercase variant does not exist."""
        assert action_registry.action_exists("create_purchase_order") is True
        assert action_registry.action_exists("Create_Purchase_Order") is False
        assert action_registry.action_exists("CREATE_PURCHASE_ORDER") is False

    def test_validate_inputs_empty_string_fields(
        self, action_registry: ActionRegistry
    ) -> None:
        """Empty string values for required string fields trigger errors."""
        inputs = {
            "vendor_id": "",  # empty string
            "items": [{"product_id": "P-1", "quantity": 10}],
            "total_amount": 5000.0,
            "company_id": "COMP-01",
        }
        errors = action_registry.validate_inputs("create_purchase_order", inputs)
        # The 'required' rule type on vendor_id should catch empty strings
        assert len(errors) > 0

    def test_validate_zero_amount(
        self, action_registry: ActionRegistry
    ) -> None:
        """Amount of 0.0 should fail the positive_number custom validator."""
        inputs = {
            "invoice_id": "INV-100",
            "vendor_id": "V-100",
            "amount": 0.0,
            "line_items": [{"desc": "x", "amount": 0.0}],
        }
        errors = action_registry.validate_inputs("process_vendor_invoice", inputs)
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_execute_result_has_timestamp(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """ActionResult always includes a timestamp."""
        result = await action_registry.execute(
            "close_period",
            mock_agent,
            VALID_INPUTS_MAP["close_period"],
        )
        assert hasattr(result, "timestamp")
        from datetime import datetime
        assert isinstance(result.timestamp, datetime)

    def test_action_definition_dataclass_fields(self) -> None:
        """ActionDefinition has the expected fields from the specification."""
        ad = ActionDefinition(
            action_id="test",
            description="test action",
            required_inputs={"a": str},
        )
        assert ad.action_id == "test"
        assert ad.description == "test action"
        assert ad.required_inputs == {"a": str}
        assert ad.optional_inputs == {}
        assert ad.validation_rules == []
        assert ad.handler is None
        assert ad.category == "general"

    def test_action_result_dataclass_fields(self) -> None:
        """ActionResult has the expected fields from the specification."""
        ar = ActionResult(
            success=True,
            action_id="test",
            outputs={"key": "val"},
        )
        assert ar.success is True
        assert ar.action_id == "test"
        assert ar.outputs == {"key": "val"}
        assert ar.error is None

    def test_validation_rule_dataclass_fields(self) -> None:
        """ValidationRule has the expected fields from the specification."""
        vr = ValidationRule(
            field_name="amount",
            rule_type="range_check",
            min_value=0.0,
            max_value=500000.0,
        )
        assert vr.field_name == "amount"
        assert vr.rule_type == "range_check"
        assert vr.min_value == 0.0
        assert vr.max_value == 500000.0
        assert vr.expected_type is None
        assert vr.custom_validator is None

    @pytest.mark.asyncio
    async def test_multiple_sequential_executions(
        self, action_registry: ActionRegistry, mock_agent: MagicMock
    ) -> None:
        """Multiple sequential execute() calls all succeed independently."""
        for action_id in ALL_14_ACTION_IDS:
            result = await action_registry.execute(
                action_id, mock_agent, VALID_INPUTS_MAP[action_id]
            )
            assert result.success is True, (
                f"Sequential execution of '{action_id}' failed: {result.error}"
            )
