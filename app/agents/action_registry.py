"""
Action Registry for the Agent & Orchestration Engine.

Provides a centralized registry of all 14 ERP action types that agents can perform,
spanning procurement, AP (accounts payable), AR (accounts receivable), warehouse,
and accounting functions. Each registered action includes:
  - action_id: unique string identifier
  - description: human-readable explanation
  - required_inputs: mapping of field name to expected Python type
  - optional_inputs: mapping of optional field name to expected Python type
  - validation_rules: list of ValidationRule instances for input validation
  - handler: optional callable for action execution
  - category: functional area (procurement, ap, ar, accounting, warehouse)

The ActionRegistry is designed for lightweight initialization to support the
>=50 agents/second creation target (AAP Section 0.7.3). All logging uses structlog
to stdout in structured JSON format (AAP Section 0.7.6).

Action Types (14 total, from README.md lines 119-134):
  Procurement (2): create_purchase_order, approve_purchase_order
  Warehouse (1): receive_goods
  AP (4): process_vendor_invoice, match_three_way, approve_invoice, schedule_payment
  AR (4): create_sales_order, ship_order, create_customer_invoice, apply_payment
  Accounting (3): create_journal_entry, reconcile_account, close_period
"""

from typing import Optional, Dict, List, Any, Type, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime

import structlog

# Module-level structured logger (AAP Section 0.7.6)
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ValidationRule:
    """Defines a single validation rule for an action's inputs.

    Attributes:
        field_name: The input field this rule applies to.
        rule_type: One of 'required', 'type_check', 'range_check', 'custom',
                   or 'balanced_entry'.
        expected_type: Expected Python type for 'type_check' rules.
        min_value: Minimum numeric value for 'range_check' rules.
        max_value: Maximum numeric value for 'range_check' rules.
        custom_validator: A callable accepting (value) -> bool for 'custom' rules.
    """

    field_name: str
    rule_type: str
    expected_type: Optional[Type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    custom_validator: Optional[Callable] = None


@dataclass
class ActionResult:
    """Result returned after executing an action.

    Attributes:
        success: Whether the action completed successfully.
        action_id: The identifier of the action that was executed.
        outputs: Dictionary of output data produced by the action.
        error: Human-readable error message if the action failed.
        timestamp: UTC timestamp of when the result was produced.
    """

    success: bool
    action_id: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ActionDefinition:
    """Full definition of a registered action.

    Attributes:
        action_id: Unique action identifier (e.g. 'create_purchase_order').
        description: Human-readable description of what the action does.
        required_inputs: Mapping of required field names to their expected types.
        optional_inputs: Mapping of optional field names to their expected types.
        validation_rules: Ordered list of ValidationRule instances applied during
                          input validation.
        handler: Optional callable ``(agent, inputs) -> Dict[str, Any]`` that
                 implements the action logic. When *None*, the action returns
                 a pass-through result (to be wired in Project 3).
        category: Functional area — one of 'procurement', 'ap', 'ar',
                  'accounting', 'warehouse', or 'general'.
    """

    action_id: str
    description: str
    required_inputs: Dict[str, Type]
    optional_inputs: Dict[str, Type] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    handler: Optional[Callable] = None
    category: str = "general"


# ---------------------------------------------------------------------------
# Validation Helper Callables
# ---------------------------------------------------------------------------


def _is_valid_decision(value: Any) -> bool:
    """Validate that a decision value is one of the allowed strings."""
    return isinstance(value, str) and value in ("approve", "reject", "escalate")


def _is_nonempty_list(value: Any) -> bool:
    """Validate that a value is a non-empty list."""
    return isinstance(value, list) and len(value) > 0


def _is_positive_number(value: Any) -> bool:
    """Validate that a value is a positive number (int or float)."""
    return isinstance(value, (int, float)) and value > 0


def _is_nonempty_string(value: Any) -> bool:
    """Validate that a value is a non-empty string."""
    return isinstance(value, str) and len(value.strip()) > 0


# ---------------------------------------------------------------------------
# ActionRegistry Class
# ---------------------------------------------------------------------------


class ActionRegistry:
    """Centralized registry of all ERP actions that agents can perform.

    On construction the registry pre-populates itself with the 14 canonical
    action types defined in the specification (README.md lines 119-134).
    Additional actions can be registered dynamically via ``register_action``.

    Categories:
        procurement — purchase order lifecycle
        warehouse   — goods receipt and shipment
        ap          — accounts payable (invoices, matching, payments)
        ar          — accounts receivable (sales orders, invoicing, payments)
        accounting  — journal entries, reconciliation, period close

    Thread Safety:
        This class is **not** thread-safe.  In an async context each
        ``SimulationEngine`` should own a single ``ActionRegistry`` instance
        and share it through constructor injection (AAP Section 0.7.1).

    Performance:
        Initialization registers all 14 actions in O(14) ~ O(1), ensuring
        the >= 50 agents/second creation target is achievable when one
        registry is shared across agents.
    """

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    def __init__(self) -> None:
        """Initialise the registry and pre-register the 14 default actions."""
        self._actions: Dict[str, ActionDefinition] = {}
        self._categories: Dict[str, Set[str]] = {}
        self._register_default_actions()
        logger.info(
            "action_registry_initialized",
            action_count=len(self._actions),
            categories=sorted(self._categories.keys()),
        )

    # --------------------------------------------------------------------- #
    # Default Action Registration
    # --------------------------------------------------------------------- #

    def _register_default_actions(self) -> None:  # noqa: C901
        """Register all 14 canonical ERP action types.

        Procurement (2): create_purchase_order, approve_purchase_order
        Warehouse  (1): receive_goods
        AP         (4): process_vendor_invoice, match_three_way,
                        approve_invoice, schedule_payment
        AR         (4): create_sales_order, ship_order,
                        create_customer_invoice, apply_payment
        Accounting (3): create_journal_entry, reconcile_account, close_period
        """

        # ----------------------------------------------------------------- #
        # 1. Procurement — create_purchase_order
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="create_purchase_order",
            description=(
                "Create a new purchase order for a vendor including line "
                "items and total amount. Initiates the procure-to-pay cycle."
            ),
            required_inputs={
                "vendor_id": str,
                "items": list,
                "total_amount": float,
                "company_id": str,
            },
            optional_inputs={
                "notes": str,
                "requested_delivery_date": str,
                "department_id": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="vendor_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="vendor_id", rule_type="type_check",
                    expected_type=str,
                ),
                ValidationRule(
                    field_name="items", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
                ValidationRule(
                    field_name="total_amount", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
                ValidationRule(
                    field_name="company_id", rule_type="required",
                ),
            ],
            category="procurement",
        ))

        # ----------------------------------------------------------------- #
        # 2. Procurement — approve_purchase_order
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="approve_purchase_order",
            description=(
                "Approve, reject, or escalate a purchase order. Decision is "
                "based on monetary thresholds and agent authority level."
            ),
            required_inputs={
                "po_id": str,
                "approver_id": str,
                "decision": str,
            },
            optional_inputs={
                "comments": str,
                "escalation_reason": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="po_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="approver_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="decision", rule_type="custom",
                    custom_validator=_is_valid_decision,
                ),
            ],
            category="procurement",
        ))

        # ----------------------------------------------------------------- #
        # 3. Warehouse — receive_goods
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="receive_goods",
            description=(
                "Record receipt of goods against a purchase order at a "
                "warehouse location. Triggers 3-way match evaluation."
            ),
            required_inputs={
                "po_id": str,
                "received_items": list,
                "warehouse_id": str,
            },
            optional_inputs={
                "receiving_notes": str,
                "inspection_required": bool,
            },
            validation_rules=[
                ValidationRule(
                    field_name="po_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="received_items", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
                ValidationRule(
                    field_name="warehouse_id", rule_type="required",
                ),
            ],
            category="warehouse",
        ))

        # ----------------------------------------------------------------- #
        # 4. AP — process_vendor_invoice
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="process_vendor_invoice",
            description=(
                "Process an incoming vendor invoice by validating line items, "
                "coding to GL accounts, and preparing for 3-way match."
            ),
            required_inputs={
                "invoice_id": str,
                "vendor_id": str,
                "amount": float,
                "line_items": list,
            },
            optional_inputs={
                "due_date": str,
                "payment_terms": str,
                "gl_account": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="invoice_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="vendor_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="amount", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
                ValidationRule(
                    field_name="line_items", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
            ],
            category="ap",
        ))

        # ----------------------------------------------------------------- #
        # 5. AP — match_three_way
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="match_three_way",
            description=(
                "Perform 3-way match between purchase order, goods receipt, "
                "and vendor invoice. Identifies variances and discrepancies."
            ),
            required_inputs={
                "invoice_id": str,
                "po_id": str,
                "receipt_id": str,
            },
            optional_inputs={
                "tolerance_percent": float,
                "auto_approve_threshold": float,
            },
            validation_rules=[
                ValidationRule(
                    field_name="invoice_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="po_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="receipt_id", rule_type="required",
                ),
            ],
            category="ap",
        ))

        # ----------------------------------------------------------------- #
        # 6. AP — approve_invoice
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="approve_invoice",
            description=(
                "Approve, reject, or escalate a vendor invoice based on "
                "monetary thresholds and 3-way match results."
            ),
            required_inputs={
                "invoice_id": str,
                "approver_id": str,
                "decision": str,
            },
            optional_inputs={
                "comments": str,
                "exception_code": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="invoice_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="approver_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="decision", rule_type="custom",
                    custom_validator=_is_valid_decision,
                ),
            ],
            category="ap",
        ))

        # ----------------------------------------------------------------- #
        # 7. AP — schedule_payment
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="schedule_payment",
            description=(
                "Schedule a payment for an approved vendor invoice, "
                "specifying the payment date, amount, and bank account."
            ),
            required_inputs={
                "invoice_id": str,
                "payment_date": str,
                "amount": float,
                "bank_account": str,
            },
            optional_inputs={
                "payment_method": str,
                "reference": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="invoice_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="payment_date", rule_type="required",
                ),
                ValidationRule(
                    field_name="amount", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
                ValidationRule(
                    field_name="bank_account", rule_type="required",
                ),
            ],
            category="ap",
        ))

        # ----------------------------------------------------------------- #
        # 8. AR — create_sales_order
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="create_sales_order",
            description=(
                "Create a new sales order for a customer including line "
                "items and total amount. Initiates the order-to-cash cycle."
            ),
            required_inputs={
                "customer_id": str,
                "items": list,
                "total_amount": float,
            },
            optional_inputs={
                "shipping_address": str,
                "requested_date": str,
                "sales_rep_id": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="customer_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="items", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
                ValidationRule(
                    field_name="total_amount", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
            ],
            category="ar",
        ))

        # ----------------------------------------------------------------- #
        # 9. AR — ship_order
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="ship_order",
            description=(
                "Ship a confirmed sales order from a warehouse via a "
                "designated carrier. Triggers customer invoice creation."
            ),
            required_inputs={
                "order_id": str,
                "warehouse_id": str,
                "carrier_id": str,
            },
            optional_inputs={
                "tracking_number": str,
                "ship_date": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="order_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="warehouse_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="carrier_id", rule_type="required",
                ),
            ],
            category="ar",
        ))

        # ----------------------------------------------------------------- #
        # 10. AR — create_customer_invoice
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="create_customer_invoice",
            description=(
                "Generate a customer invoice for shipped goods, linking to "
                "the original sales order and specifying payment terms."
            ),
            required_inputs={
                "order_id": str,
                "customer_id": str,
                "amount": float,
                "line_items": list,
            },
            optional_inputs={
                "payment_terms": str,
                "due_date": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="order_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="customer_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="amount", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
                ValidationRule(
                    field_name="line_items", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
            ],
            category="ar",
        ))

        # ----------------------------------------------------------------- #
        # 11. AR — apply_payment
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="apply_payment",
            description=(
                "Apply an incoming customer payment against one or more "
                "outstanding invoices. Handles partial and over-payments."
            ),
            required_inputs={
                "payment_id": str,
                "invoice_id": str,
                "amount": float,
            },
            optional_inputs={
                "payment_method": str,
                "reference": str,
                "remaining_balance": float,
            },
            validation_rules=[
                ValidationRule(
                    field_name="payment_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="invoice_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="amount", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
            ],
            category="ar",
        ))

        # ----------------------------------------------------------------- #
        # 12. Accounting — create_journal_entry
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="create_journal_entry",
            description=(
                "Create a balanced journal entry with debit and credit "
                "lines. Total debits must equal total credits."
            ),
            required_inputs={
                "entries": list,
                "description": str,
                "total_debit": float,
                "total_credit": float,
            },
            optional_inputs={
                "reference": str,
                "reversal_date": str,
                "source_module": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="entries", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
                ValidationRule(
                    field_name="description", rule_type="custom",
                    custom_validator=_is_nonempty_string,
                ),
                ValidationRule(
                    field_name="total_debit", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
                ValidationRule(
                    field_name="total_credit", rule_type="custom",
                    custom_validator=_is_positive_number,
                ),
                # Special balanced-entry check: debit must equal credit
                ValidationRule(
                    field_name="total_debit", rule_type="balanced_entry",
                ),
            ],
            category="accounting",
        ))

        # ----------------------------------------------------------------- #
        # 13. Accounting — reconcile_account
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="reconcile_account",
            description=(
                "Reconcile a GL account for a specific fiscal period by "
                "matching transactions and identifying discrepancies."
            ),
            required_inputs={
                "account_id": str,
                "period": str,
                "transactions": list,
            },
            optional_inputs={
                "statement_balance": float,
                "reconciliation_date": str,
            },
            validation_rules=[
                ValidationRule(
                    field_name="account_id", rule_type="required",
                ),
                ValidationRule(
                    field_name="period", rule_type="required",
                ),
                ValidationRule(
                    field_name="transactions", rule_type="custom",
                    custom_validator=_is_nonempty_list,
                ),
            ],
            category="accounting",
        ))

        # ----------------------------------------------------------------- #
        # 14. Accounting — close_period
        # ----------------------------------------------------------------- #
        self.register_action(ActionDefinition(
            action_id="close_period",
            description=(
                "Close a fiscal period, preventing further postings and "
                "initiating period-end reporting procedures."
            ),
            required_inputs={
                "period": str,
                "fiscal_year": str,
            },
            optional_inputs={
                "closing_date": str,
                "adjustment_entries": list,
            },
            validation_rules=[
                ValidationRule(
                    field_name="period", rule_type="required",
                ),
                ValidationRule(
                    field_name="fiscal_year", rule_type="required",
                ),
            ],
            category="accounting",
        ))

    # --------------------------------------------------------------------- #
    # Public API — Registration
    # --------------------------------------------------------------------- #

    def register_action(self, action: ActionDefinition) -> None:
        """Register a new action or replace an existing one.

        Args:
            action: Fully populated ``ActionDefinition`` instance.

        Side Effects:
            Updates the internal action store and category index.
            Emits a structured log entry at DEBUG level.
        """
        self._actions[action.action_id] = action

        if action.category not in self._categories:
            self._categories[action.category] = set()
        self._categories[action.category].add(action.action_id)

        logger.debug(
            "action_registered",
            action_id=action.action_id,
            category=action.category,
        )

    # --------------------------------------------------------------------- #
    # Public API — Lookup
    # --------------------------------------------------------------------- #

    def get_action(self, action_id: str) -> Optional[ActionDefinition]:
        """Return the definition for *action_id*, or ``None`` if not found."""
        return self._actions.get(action_id)

    def get_actions_by_category(self, category: str) -> List[ActionDefinition]:
        """Return all actions belonging to *category*.

        Args:
            category: One of 'procurement', 'ap', 'ar', 'accounting',
                      'warehouse', or 'general'.

        Returns:
            A list of ``ActionDefinition`` objects.  Empty list if the
            category has no registered actions.
        """
        action_ids = self._categories.get(category, set())
        return [
            self._actions[aid]
            for aid in sorted(action_ids)
            if aid in self._actions
        ]

    def get_all_action_ids(self) -> List[str]:
        """Return a sorted list of all registered action identifiers."""
        return sorted(self._actions.keys())

    def action_exists(self, action_id: str) -> bool:
        """Return ``True`` if *action_id* is registered."""
        return action_id in self._actions

    # --------------------------------------------------------------------- #
    # Public API — Validation
    # --------------------------------------------------------------------- #

    def validate_inputs(
        self, action_id: str, inputs: Dict[str, Any]
    ) -> List[str]:
        """Validate *inputs* against the rules for *action_id*.

        Performs three layers of checks:

        1. **Presence** — every key in ``required_inputs`` must appear in
           *inputs*.
        2. **Type** — each value must be compatible with the declared type
           (``isinstance`` check, with numeric promotion for int→float).
        3. **Rules** — each ``ValidationRule`` is evaluated in order.

        Args:
            action_id: The action whose rules should be applied.
            inputs: The candidate input dictionary.

        Returns:
            A list of human-readable error strings.  An empty list means
            all validations passed.
        """
        errors: List[str] = []

        action = self._actions.get(action_id)
        if action is None:
            errors.append(f"Unknown action: {action_id}")
            return errors

        # --- 1. Required field presence and type --------------------------
        for field_name, expected_type in action.required_inputs.items():
            if field_name not in inputs:
                errors.append(
                    f"Missing required input '{field_name}' "
                    f"for action '{action_id}'"
                )
            elif not isinstance(inputs[field_name], expected_type):
                # Allow int where float is expected (numeric promotion)
                if expected_type is float and isinstance(
                    inputs[field_name], (int, float)
                ):
                    continue
                errors.append(
                    f"Input '{field_name}' expected type "
                    f"{expected_type.__name__}, "
                    f"got {type(inputs[field_name]).__name__}"
                )

        # --- 2. Validation rules ------------------------------------------
        for rule in action.validation_rules:
            value = inputs.get(rule.field_name)

            if rule.rule_type == "required":
                if value is None or (
                    isinstance(value, str) and not value.strip()
                ):
                    errors.append(
                        f"Field '{rule.field_name}' is required "
                        f"and cannot be empty"
                    )

            elif rule.rule_type == "type_check":
                if value is not None and rule.expected_type is not None:
                    if not isinstance(value, rule.expected_type):
                        errors.append(
                            f"Field '{rule.field_name}' expected type "
                            f"{rule.expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            elif rule.rule_type == "range_check":
                if value is not None and isinstance(value, (int, float)):
                    if (
                        rule.min_value is not None
                        and value < rule.min_value
                    ):
                        errors.append(
                            f"Field '{rule.field_name}' value {value} "
                            f"is below minimum {rule.min_value}"
                        )
                    if (
                        rule.max_value is not None
                        and value > rule.max_value
                    ):
                        errors.append(
                            f"Field '{rule.field_name}' value {value} "
                            f"exceeds maximum {rule.max_value}"
                        )

            elif rule.rule_type == "custom":
                if rule.custom_validator is not None and value is not None:
                    if not rule.custom_validator(value):
                        errors.append(
                            f"Field '{rule.field_name}' failed custom "
                            f"validation for action '{action_id}'"
                        )

            elif rule.rule_type == "balanced_entry":
                # Special rule: total_debit must equal total_credit
                total_debit = inputs.get("total_debit")
                total_credit = inputs.get("total_credit")
                if (
                    total_debit is not None
                    and total_credit is not None
                    and isinstance(total_debit, (int, float))
                    and isinstance(total_credit, (int, float))
                ):
                    if abs(total_debit - total_credit) > 0.01:
                        errors.append(
                            f"Journal entry is unbalanced: "
                            f"total_debit={total_debit} "
                            f"!= total_credit={total_credit}"
                        )

        if errors:
            logger.warning(
                "action_input_validation_failed",
                action_id=action_id,
                error_count=len(errors),
                errors=errors,
            )

        return errors

    # --------------------------------------------------------------------- #
    # Public API — Execution
    # --------------------------------------------------------------------- #

    async def execute(
        self,
        action_id: str,
        agent: Any,
        inputs: Dict[str, Any],
    ) -> ActionResult:
        """Execute the action identified by *action_id*.

        Validates *inputs* first.  On validation failure an
        ``ActionResult(success=False)`` is returned immediately.

        If a ``handler`` callable is registered it is invoked as
        ``handler(agent, inputs)`` and the return value used as outputs.
        Otherwise a pass-through result echoes inputs back as outputs
        (to be wired with full business logic in Project 3).

        Args:
            action_id: Registered action identifier.
            agent: The agent performing the action (forwarded to handler).
            inputs: Dictionary of input data.

        Returns:
            An ``ActionResult`` indicating success/failure and outputs.
        """
        # Validate inputs first
        validation_errors = self.validate_inputs(action_id, inputs)
        if validation_errors:
            result = ActionResult(
                success=False,
                action_id=action_id,
                error="; ".join(validation_errors),
            )
            logger.warning(
                "action_execution_failed",
                action_id=action_id,
                reason="validation_failed",
                error_count=len(validation_errors),
            )
            return result

        action = self._actions.get(action_id)
        if action is None:
            result = ActionResult(
                success=False,
                action_id=action_id,
                error=f"Unknown action: {action_id}",
            )
            logger.error(
                "action_execution_failed",
                action_id=action_id,
                reason="unknown_action",
            )
            return result

        # Execute via handler or return pass-through
        try:
            if action.handler is not None:
                handler_output = action.handler(agent, inputs)
                # Support both sync and async handlers
                if hasattr(handler_output, "__await__"):
                    handler_output = await handler_output
                outputs = (
                    handler_output
                    if isinstance(handler_output, dict)
                    else {"result": handler_output}
                )
            else:
                # Pass-through: echo inputs as outputs for Project 3 wiring
                outputs = dict(inputs)
                outputs["_action_id"] = action_id
                outputs["_status"] = "completed"

            result = ActionResult(
                success=True,
                action_id=action_id,
                outputs=outputs,
            )
            logger.info(
                "action_executed",
                action_id=action_id,
                success=True,
                category=action.category,
            )
            return result

        except Exception as exc:
            result = ActionResult(
                success=False,
                action_id=action_id,
                error=f"Execution error: {exc!s}",
            )
            logger.error(
                "action_execution_error",
                action_id=action_id,
                error=str(exc),
                category=action.category,
            )
            return result
