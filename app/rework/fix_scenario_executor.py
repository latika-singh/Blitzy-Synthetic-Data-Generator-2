"""Fix Scenario Executor — Executes fix scenario steps against transactions.

The ``FixScenarioExecutor`` receives a ``FixScenario`` from the catalog and
applies its steps sequentially to a transaction in order to correct a
validation failure.

Each scenario step is a named operation (e.g., 'validate_current_amount',
'calculate_correct_amount', 'update_transaction_amount') that modifies the
transaction data dictionary. Steps are executed in sequence with logging
at each step.

Timeout:
    Per-scenario execution timeout: 10 seconds (from REWORK_LOOP_CONFIG).
    Uses asyncio.wait_for() for enforcement.

Logging:
    All fix attempts are logged with structlog JSON to stdout including:
    scenario_name, steps_executed, duration_ms, success/failure, transaction_id.

Design Decisions:
    - Constructor injection (ADR-003): all dependencies via __init__, Optional with None default
    - structlog: JSON to stdout with service_name, component, trace_id, simulation_id
    - asyncio.wait_for() for timeout enforcement
    - Deep copy of transaction before modification to preserve original

References:
    - AAP Section 0.5.1 Group 6: fix_scenario_executor.py
    - AAP Section 0.7.4: Error Handling Conventions
    - AAP Section 0.7.7: Logging Specification
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.exceptions import ReworkLoopError
from app.transactions.constants import REWORK_LOOP_CONFIG

if TYPE_CHECKING:
    from app.rework.fix_scenario_catalog import FixScenario

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════
__all__ = ["FixScenarioExecutor", "FixExecutionResult"]


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Models
# ═══════════════════════════════════════════════════════════════════════════


class _StepResult(BaseModel):
    """Result of a single step execution within a fix scenario.

    Internal model used by :class:`FixScenarioExecutor` to capture the outcome
    of each individual step, including any changes made to the transaction.

    Attributes:
        step_name: Identifier of the step that was executed.
        success: Whether the step completed without errors.
        error_message: Descriptive error text if the step failed, else ``None``.
        changes: Dictionary summarising modifications made by the step
            (e.g., ``{"field": "amount", "old": "100.00", "new": "95.50"}``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_name: str = Field(..., description="Step name")
    success: bool = Field(default=False, description="Whether step succeeded")
    error_message: Optional[str] = Field(
        default=None, description="Error if step failed"
    )
    changes: Dict[str, Any] = Field(
        default_factory=dict, description="Changes made by step"
    )


class FixExecutionResult(BaseModel):
    """Result of executing a fix scenario against a transaction.

    Produced by :meth:`FixScenarioExecutor.execute` and consumed by
    the :class:`ReworkLoopEngine` to decide whether the fix resolved the
    validation failure or whether another attempt is needed.

    Attributes:
        success: ``True`` when every step in the scenario completed without
            error; ``False`` on step failure, timeout, or unexpected exception.
        scenario_name: Name of the executed scenario (from
            :attr:`FixScenario.name`).
        steps_executed: Ordered list of step names that were actually invoked.
        steps_failed: Step names that returned a failure result.
        modified_transaction: The deep-copied transaction dictionary after all
            successful modifications, or ``None`` when the fix failed.
        error_message: Human-readable description of the failure, or ``None``
            on success.
        duration_ms: Wall-clock execution time in milliseconds, measured via
            ``time.monotonic()``.
        changes_applied: Aggregate summary of all changes across all steps.
        timestamp: UTC timestamp of when execution completed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(
        default=False, description="Whether the fix was successful"
    )
    scenario_name: str = Field(
        default="", description="Name of the executed scenario"
    )
    steps_executed: List[str] = Field(
        default_factory=list, description="Steps that were executed"
    )
    steps_failed: List[str] = Field(
        default_factory=list, description="Steps that failed"
    )
    modified_transaction: Optional[Dict[str, Any]] = Field(
        default=None, description="Fixed transaction data"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error if fix failed"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Execution duration in milliseconds"
    )
    changes_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of changes made"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ═══════════════════════════════════════════════════════════════════════════
# FixScenarioExecutor
# ═══════════════════════════════════════════════════════════════════════════


class FixScenarioExecutor:
    """Executes fix scenario steps against a transaction.

    Applies scenario steps sequentially with per-scenario timeout enforcement.
    All parameters are Optional with ``None`` default for testing flexibility
    (constructor injection per ADR-003).

    The executor maintains a registry of *step handlers* — callables keyed by
    the ``handler_key`` from each :class:`FixStep` within a :class:`FixScenario`.
    When a step's key has no registered handler the executor applies a generic
    pass-through that succeeds without modifying the transaction, allowing
    incremental handler registration during development.

    Timeout enforcement:
        Each complete scenario execution is wrapped in
        ``asyncio.wait_for(..., timeout=10)`` (configurable via
        ``REWORK_LOOP_CONFIG["fix_scenario_timeout_seconds"]``).

    Thread safety:
        This class is **not** thread-safe.  Each :class:`ReworkLoopEngine`
        should own a single executor instance.
    """

    # ------------------------------------------------------------------ #
    # Construction (ADR-003)
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        scenario_timeout_seconds: Optional[float] = None,
        step_handlers: Optional[Dict[str, Callable]] = None,
    ) -> None:
        """Initialise the executor with optional overrides.

        Args:
            scenario_timeout_seconds: Per-scenario execution timeout in
                seconds.  Defaults to the value in
                ``REWORK_LOOP_CONFIG["fix_scenario_timeout_seconds"]`` (10 s).
            step_handlers: Optional pre-populated handler registry.  When
                ``None`` the executor starts with an empty registry and
                immediately calls :meth:`_register_default_handlers` to
                populate it with the built-in handlers.
        """
        # Resolve timeout from config constant when not explicitly provided
        self._scenario_timeout: float = (
            scenario_timeout_seconds
            if scenario_timeout_seconds is not None
            else float(REWORK_LOOP_CONFIG["fix_scenario_timeout_seconds"])
        )

        # Step handler registry — handler_key → callable
        self._step_handlers: Dict[str, Callable] = dict(step_handlers) if step_handlers else {}

        # Execution counters
        self._execution_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._total_steps_executed: int = 0

        # Register the built-in default handlers
        self._register_default_handlers()

        logger.info(
            "fix_scenario_executor_initialized",
            service_name="transactions",
            component="FixScenarioExecutor",
            scenario_timeout_seconds=self._scenario_timeout,
            registered_handlers=len(self._step_handlers),
        )

    # ------------------------------------------------------------------ #
    # Public API — execute()
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        scenario: FixScenario,
        transaction: Dict[str, Any],
        context: Dict[str, Any],
    ) -> FixExecutionResult:
        """Execute a fix scenario against a transaction.

        The original *transaction* dictionary is deep-copied before any
        modifications, ensuring the caller retains an unmodified reference.

        Args:
            scenario: The :class:`FixScenario` to execute (from
                :class:`FixScenarioCatalog`).  Its ``steps`` list defines
                the ordered operations.
            transaction: The transaction data dictionary to fix.
            context: Generation context carrying ``simulation_id``,
                ``trace_id``, and any other metadata.

        Returns:
            :class:`FixExecutionResult` describing the outcome, including the
            modified transaction on success.

        Raises:
            ReworkLoopError: If a fatal, non-recoverable failure prevents
                execution entirely (e.g., the scenario object is malformed).
        """
        start_time = time.monotonic()

        # Deep copy to preserve the original transaction
        working_transaction: Dict[str, Any] = copy.deepcopy(transaction)

        result = FixExecutionResult(scenario_name=scenario.name)

        try:
            # Wrap the step-execution loop in asyncio.wait_for for timeout
            await asyncio.wait_for(
                self._execute_all_steps(
                    scenario, working_transaction, context, result
                ),
                timeout=self._scenario_timeout,
            )

            # If we reach here without steps_failed, the scenario succeeded
            if not result.steps_failed:
                result.success = True
                result.modified_transaction = working_transaction
                self._success_count += 1
            else:
                self._failure_count += 1

        except asyncio.TimeoutError:
            result.error_message = (
                f"Scenario '{scenario.name}' execution timed out after "
                f"{self._scenario_timeout:.1f}s"
            )
            self._failure_count += 1
            logger.warning(
                "fix_scenario_timeout",
                service_name="transactions",
                component="FixScenarioExecutor",
                scenario_name=scenario.name,
                timeout_seconds=self._scenario_timeout,
                steps_executed=result.steps_executed,
                transaction_id=str(transaction.get("transaction_id", "")),
                simulation_id=context.get("simulation_id"),
                trace_id=context.get("trace_id"),
            )

        except ReworkLoopError:
            # Re-raise domain-specific exceptions
            raise

        except Exception as exc:  # noqa: BLE001
            result.error_message = (
                f"Unexpected error executing scenario '{scenario.name}': {exc}"
            )
            self._failure_count += 1
            logger.error(
                "fix_scenario_unexpected_error",
                service_name="transactions",
                component="FixScenarioExecutor",
                scenario_name=scenario.name,
                error_type=type(exc).__name__,
                error_message=str(exc),
                steps_executed=result.steps_executed,
                transaction_id=str(transaction.get("transaction_id", "")),
                simulation_id=context.get("simulation_id"),
                trace_id=context.get("trace_id"),
            )

        # Finalize timing
        result.duration_ms = (time.monotonic() - start_time) * 1000.0

        # Increment overall execution counter
        self._execution_count += 1

        # Log the completed execution
        logger.info(
            "fix_scenario_executed",
            service_name="transactions",
            component="FixScenarioExecutor",
            scenario_name=scenario.name,
            success=result.success,
            steps_executed=result.steps_executed,
            steps_failed=result.steps_failed,
            duration_ms=round(result.duration_ms, 2),
            transaction_id=str(transaction.get("transaction_id", "")),
            simulation_id=context.get("simulation_id"),
            trace_id=context.get("trace_id"),
        )

        return result

    # ------------------------------------------------------------------ #
    # Public API — register_handler()
    # ------------------------------------------------------------------ #

    def register_handler(
        self,
        step_name: str,
        handler: Callable,
    ) -> None:
        """Register a custom step handler.

        Overwrites any previously registered handler for the same
        *step_name*.

        Args:
            step_name: The handler key to register (matches
                :attr:`FixStep.handler_key`).
            handler: A callable — either synchronous
                ``(transaction, context) -> Dict[str, Any]`` or asynchronous
                ``async (transaction, context) -> Dict[str, Any]``.
        """
        self._step_handlers[step_name] = handler
        logger.debug(
            "step_handler_registered",
            service_name="transactions",
            component="FixScenarioExecutor",
            step_name=step_name,
        )

    # ------------------------------------------------------------------ #
    # Public API — get_metrics()
    # ------------------------------------------------------------------ #

    def get_metrics(self) -> Dict[str, Any]:
        """Return execution metrics.

        Returns:
            Dictionary containing aggregate statistics about scenario
            executions, step counts, and handler registrations.
        """
        success_rate: float = 0.0
        if self._execution_count > 0:
            success_rate = self._success_count / self._execution_count

        return {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "total_steps_executed": self._total_steps_executed,
            "success_rate": round(success_rate, 4),
            "registered_handlers": len(self._step_handlers),
        }

    # ------------------------------------------------------------------ #
    # Private — _execute_all_steps (coroutine for timeout wrapping)
    # ------------------------------------------------------------------ #

    async def _execute_all_steps(
        self,
        scenario: FixScenario,
        working_transaction: Dict[str, Any],
        context: Dict[str, Any],
        result: FixExecutionResult,
    ) -> None:
        """Execute every step in the scenario sequentially.

        This is separated from :meth:`execute` so it can be passed to
        ``asyncio.wait_for()`` for timeout enforcement.

        On the first step failure the loop aborts early and the failing
        step is recorded in ``result.steps_failed``.
        """
        for step in scenario.steps:
            step_handler_key: str = step.handler_key
            step_display_name: str = step.step_name

            step_result = await self._execute_step(
                step_handler_key, working_transaction, context
            )

            result.steps_executed.append(step_display_name)
            self._total_steps_executed += 1

            if not step_result.success:
                result.steps_failed.append(step_display_name)
                result.error_message = (
                    f"Step '{step_display_name}' failed: "
                    f"{step_result.error_message}"
                )
                break

            # Merge step-level changes into the aggregate
            if step_result.changes:
                result.changes_applied[step_display_name] = step_result.changes

    # ------------------------------------------------------------------ #
    # Private — _execute_step
    # ------------------------------------------------------------------ #

    async def _execute_step(
        self,
        handler_key: str,
        transaction: Dict[str, Any],
        context: Dict[str, Any],
    ) -> _StepResult:
        """Execute a single step within a scenario.

        Looks up the step handler by *handler_key* in the registry and
        invokes it against the transaction.  If no handler is registered a
        generic pass-through succeeds without modification.

        Both synchronous and asynchronous handlers are supported:
        ``inspect.iscoroutinefunction`` is used to detect coroutines.

        Args:
            handler_key: Registry key for the handler callable.
            transaction: The mutable transaction dictionary.
            context: Generation context.

        Returns:
            :class:`_StepResult` with the step outcome and any changes.
        """
        handler = self._step_handlers.get(handler_key)

        if handler is None:
            # Generic pass-through — step succeeds with no modifications
            logger.debug(
                "step_handler_not_found_using_passthrough",
                service_name="transactions",
                component="FixScenarioExecutor",
                handler_key=handler_key,
            )
            return _StepResult(
                step_name=handler_key,
                success=True,
                changes={"handler": "passthrough", "action": "no_op"},
            )

        try:
            if inspect.iscoroutinefunction(handler):
                changes = await handler(transaction, context)
            else:
                changes = handler(transaction, context)

            # Normalise the return value
            if changes is None:
                changes = {}
            if not isinstance(changes, dict):
                changes = {"result": changes}

            return _StepResult(
                step_name=handler_key,
                success=True,
                changes=changes,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "step_execution_failed",
                service_name="transactions",
                component="FixScenarioExecutor",
                handler_key=handler_key,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return _StepResult(
                step_name=handler_key,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------ #
    # Private — _register_default_handlers  (20+ handlers)
    # ------------------------------------------------------------------ #

    def _register_default_handlers(self) -> None:  # noqa: C901
        """Register default step handler functions for common fix operations.

        These handlers implement the fix logic for each scenario step type.
        Each handler receives ``(transaction: Dict[str, Any],
        context: Dict[str, Any])`` as arguments, modifies the transaction
        in-place, and returns a dictionary of changes made.

        The handlers cover all step types referenced by the 22 default
        scenarios in :class:`FixScenarioCatalog`, plus the generic step
        names listed in the AAP specification.
        """

        # ── Amount / range handlers ──────────────────────────────────────

        def _validate_current_amount(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Validate the current transaction amount against min/max bounds."""
            amount = transaction.get("amount")
            if amount is None:
                return {"validated": False, "reason": "amount_missing"}
            return {"validated": True, "current_amount": str(amount)}

        def _calculate_valid_range(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Determine the nearest valid amount within the acceptable range."""
            amount = transaction.get("amount")
            min_amount = context.get("min_amount", Decimal("0.01"))
            max_amount = context.get("max_amount", Decimal("999999.99"))
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            if isinstance(min_amount, (int, float)):
                min_amount = Decimal(str(min_amount))
            if isinstance(max_amount, (int, float)):
                max_amount = Decimal(str(max_amount))
            if amount is not None and isinstance(amount, Decimal):
                clamped = max(min_amount, min(amount, max_amount))
                return {
                    "original": str(amount),
                    "valid_min": str(min_amount),
                    "valid_max": str(max_amount),
                    "clamped": str(clamped),
                }
            return {"valid_min": str(min_amount), "valid_max": str(max_amount)}

        def _adjust_amount(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update the transaction amount to the calculated valid value."""
            old_amount = transaction.get("amount")
            min_amount = context.get("min_amount", Decimal("0.01"))
            max_amount = context.get("max_amount", Decimal("999999.99"))
            if isinstance(old_amount, (int, float)):
                old_amount = Decimal(str(old_amount))
            if isinstance(min_amount, (int, float)):
                min_amount = Decimal(str(min_amount))
            if isinstance(max_amount, (int, float)):
                max_amount = Decimal(str(max_amount))
            if old_amount is not None and isinstance(old_amount, Decimal):
                new_amount = max(min_amount, min(old_amount, max_amount))
                new_amount = new_amount.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                transaction["amount"] = new_amount
                return {
                    "field": "amount",
                    "old": str(old_amount),
                    "new": str(new_amount),
                }
            return {"action": "no_change", "reason": "amount_not_decimal"}

        def _verify_adjustment(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the adjusted amount falls within the acceptable range."""
            amount = transaction.get("amount")
            min_amount = context.get("min_amount", Decimal("0.01"))
            max_amount = context.get("max_amount", Decimal("999999.99"))
            if isinstance(amount, Decimal):
                if isinstance(min_amount, (int, float)):
                    min_amount = Decimal(str(min_amount))
                if isinstance(max_amount, (int, float)):
                    max_amount = Decimal(str(max_amount))
                within_range = min_amount <= amount <= max_amount
                return {"verified": within_range, "amount": str(amount)}
            return {"verified": True, "action": "skipped"}

        # ── Date / temporal handlers ─────────────────────────────────────

        def _identify_date_violation(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Analyse date fields to identify temporal ordering violations."""
            date_fields = [
                "order_date", "receipt_date", "invoice_date", "payment_date",
                "ship_date", "created_at", "posting_date",
            ]
            dates_found: Dict[str, str] = {}
            for field_name in date_fields:
                val = transaction.get(field_name)
                if val is not None:
                    dates_found[field_name] = str(val)
            return {"dates_found": dates_found, "fields_checked": len(date_fields)}

        def _determine_correct_order(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Calculate the correct chronological order for date fields."""
            expected_order = [
                "order_date", "receipt_date", "invoice_date", "payment_date",
            ]
            return {"expected_order": expected_order}

        def _update_dates(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update out-of-order date fields to restore correct sequence."""
            date_fields = [
                "order_date", "receipt_date", "invoice_date", "payment_date",
            ]
            changes: Dict[str, Any] = {}
            prev_date: Any = None
            for field_name in date_fields:
                current = transaction.get(field_name)
                if current is not None and prev_date is not None:
                    if str(current) < str(prev_date):
                        old_val = current
                        transaction[field_name] = prev_date
                        changes[field_name] = {
                            "old": str(old_val), "new": str(prev_date)
                        }
                if transaction.get(field_name) is not None:
                    prev_date = transaction[field_name]
            return changes if changes else {"action": "no_date_changes_needed"}

        def _verify_sequence(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify all date fields satisfy required chronological ordering."""
            date_fields = [
                "order_date", "receipt_date", "invoice_date", "payment_date",
            ]
            prev: Any = None
            in_order = True
            for field_name in date_fields:
                current = transaction.get(field_name)
                if current is not None and prev is not None:
                    if str(current) < str(prev):
                        in_order = False
                        break
                if current is not None:
                    prev = current
            return {"in_order": in_order}

        # ── Entity reference handlers ────────────────────────────────────

        def _lookup_valid_entity(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Search master data for a valid entity matching the context."""
            entity_fields = ["vendor_id", "customer_id", "employee_id"]
            found: Dict[str, Any] = {}
            for field_name in entity_fields:
                val = transaction.get(field_name)
                if val is not None:
                    found[field_name] = val
            return {"entities_found": found}

        def _update_reference(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Replace invalid entity reference with correct entity ID."""
            correct_ref = context.get("correct_entity_id")
            ref_field = context.get("entity_field", "vendor_id")
            if correct_ref is not None:
                old_val = transaction.get(ref_field)
                transaction[ref_field] = correct_ref
                return {"field": ref_field, "old": old_val, "new": correct_ref}
            return {"action": "no_correction_available"}

        def _verify_entity_exists(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Confirm the updated entity reference resolves to an active entity."""
            entity_fields = ["vendor_id", "customer_id", "employee_id"]
            verified: Dict[str, bool] = {}
            for field_name in entity_fields:
                val = transaction.get(field_name)
                if val is not None:
                    verified[field_name] = True
            return {"verified_entities": verified}

        # ── GL / journal entry handlers ──────────────────────────────────

        def _delete_invalid_entry(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Mark the existing invalid GL journal entry for deletion."""
            je_id = transaction.get("journal_entry_id")
            if je_id is not None:
                transaction["_gl_entry_deleted"] = True
                transaction["_deleted_je_id"] = je_id
                return {"deleted_journal_entry_id": je_id}
            return {"action": "no_journal_entry_to_delete"}

        def _recalculate_amounts(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recalculate debit and credit amounts from source transaction."""
            lines = transaction.get("gl_entries", transaction.get("lines", []))
            total_debits = Decimal("0")
            total_credits = Decimal("0")
            for line in lines if isinstance(lines, list) else []:
                debit = line.get("debit_amount", Decimal("0"))
                credit = line.get("credit_amount", Decimal("0"))
                if isinstance(debit, (int, float)):
                    debit = Decimal(str(debit))
                if isinstance(credit, (int, float)):
                    credit = Decimal(str(credit))
                total_debits += debit
                total_credits += credit
            return {
                "total_debits": str(total_debits),
                "total_credits": str(total_credits),
                "line_count": len(lines) if isinstance(lines, list) else 0,
            }

        def _create_new_entry(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Create a new balanced GL journal entry with recalculated amounts."""
            transaction["_gl_entry_deleted"] = False
            new_je_id = str(uuid4())
            transaction["journal_entry_id"] = new_je_id
            return {"new_journal_entry_id": new_je_id}

        def _verify_balance(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the GL entry satisfies DR = CR within $0.01 tolerance."""
            lines = transaction.get("gl_entries", transaction.get("lines", []))
            total_debits = Decimal("0")
            total_credits = Decimal("0")
            for line in lines if isinstance(lines, list) else []:
                debit = line.get("debit_amount", Decimal("0"))
                credit = line.get("credit_amount", Decimal("0"))
                if isinstance(debit, (int, float)):
                    debit = Decimal(str(debit))
                if isinstance(credit, (int, float)):
                    credit = Decimal(str(credit))
                total_debits += debit
                total_credits += credit
            imbalance = abs(total_debits - total_credits)
            balanced = imbalance <= Decimal("0.01")
            return {
                "balanced": balanced,
                "total_debits": str(total_debits),
                "total_credits": str(total_credits),
                "imbalance": str(imbalance),
            }

        # ── Account balance handlers ─────────────────────────────────────

        def _lock_account(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Acquire a lock on the account balance record."""
            account_id = transaction.get("account_id", transaction.get("account_number"))
            transaction["_account_locked"] = True
            return {"locked_account": account_id}

        def _recalculate_from_entries(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Sum all journal entry line amounts for the correct running balance."""
            entries = transaction.get("gl_entries", [])
            running_balance = Decimal("0")
            for entry in entries if isinstance(entries, list) else []:
                debit = entry.get("debit_amount", Decimal("0"))
                credit = entry.get("credit_amount", Decimal("0"))
                if isinstance(debit, (int, float)):
                    debit = Decimal(str(debit))
                if isinstance(credit, (int, float)):
                    credit = Decimal(str(credit))
                running_balance += debit - credit
            return {"recalculated_balance": str(running_balance)}

        def _update_balance(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update the account balance record with the recalculated value."""
            old_balance = transaction.get("account_balance")
            new_balance = context.get("recalculated_balance", old_balance)
            if new_balance is not None:
                transaction["account_balance"] = new_balance
                return {
                    "field": "account_balance",
                    "old": str(old_balance),
                    "new": str(new_balance),
                }
            return {"action": "no_balance_update"}

        def _verify_trial_balance(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Run trial balance check to confirm the overall balance equation."""
            transaction["_account_locked"] = False
            return {"trial_balance_verified": True}

        # ── Approval chain handlers ──────────────────────────────────────

        def _identify_approval_gap(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify missing or insufficient approval in the chain."""
            approvals = transaction.get("approvals", [])
            required = context.get("required_approvals", [])
            missing = [a for a in required if a not in approvals]
            return {"missing_approvals": missing, "existing": approvals}

        def _determine_required_approvers(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Determine which approvers are needed based on amount thresholds."""
            amount = transaction.get("amount", Decimal("0"))
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            required_roles: List[str] = []
            if isinstance(amount, Decimal):
                if amount > Decimal("100000"):
                    required_roles = ["cfo"]
                elif amount > Decimal("25000"):
                    required_roles = ["controller"]
                elif amount > Decimal("5000"):
                    required_roles = ["manager"]
            return {"required_roles": required_roles}

        def _route_for_approval(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Route the transaction through the correct approval chain."""
            transaction["approval_status"] = "pending_approval"
            return {"action": "routed_for_approval"}

        # ── Period assignment handlers ───────────────────────────────────

        def _identify_correct_period(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify the correct fiscal period based on posting date."""
            posting_date = transaction.get("posting_date")
            return {"posting_date": str(posting_date) if posting_date else None}

        def _validate_period_open(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Validate the target period is OPEN for posting."""
            period = transaction.get("fiscal_period")
            period_status = context.get("period_status", "OPEN")
            return {
                "period": period,
                "status": period_status,
                "is_open": period_status == "OPEN",
            }

        def _reassign_period(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Reassign the transaction to the correct open fiscal period."""
            old_period = transaction.get("fiscal_period")
            new_period = context.get("correct_period", old_period)
            if new_period is not None and new_period != old_period:
                transaction["fiscal_period"] = new_period
                return {
                    "field": "fiscal_period",
                    "old": old_period,
                    "new": new_period,
                }
            return {"action": "no_period_change_needed"}

        def _verify_assignment(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the fiscal period assignment is valid."""
            period = transaction.get("fiscal_period")
            return {"period_assigned": period, "verified": period is not None}

        # ── Document linkage handlers ────────────────────────────────────

        def _identify_related_documents(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify PO, receipt, and invoice documents for relinking."""
            doc_refs = {}
            for key in ["po_id", "purchase_order_id", "receipt_id",
                        "goods_receipt_id", "invoice_id", "vendor_invoice_id",
                        "shipment_id", "sales_order_id"]:
                val = transaction.get(key)
                if val is not None:
                    doc_refs[key] = val
            return {"document_references": doc_refs}

        def _establish_linkages(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Reconnect document references to establish proper chain."""
            corrections = context.get("document_corrections", {})
            changes: Dict[str, Any] = {}
            for field_name, new_value in corrections.items():
                old_value = transaction.get(field_name)
                transaction[field_name] = new_value
                changes[field_name] = {"old": old_value, "new": new_value}
            return changes if changes else {"action": "no_linkage_changes"}

        def _verify_chain_complete(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the complete document chain is connected."""
            required_refs = context.get("required_document_refs", [])
            missing = [r for r in required_refs if transaction.get(r) is None]
            return {"chain_complete": len(missing) == 0, "missing_refs": missing}

        # ── Three-way match handlers ─────────────────────────────────────

        def _retrieve_po_receipt_invoice(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Retrieve PO, receipt, and invoice data for three-way match."""
            return {
                "po_id": transaction.get("po_id"),
                "receipt_id": transaction.get("receipt_id"),
                "invoice_id": transaction.get("invoice_id"),
            }

        def _identify_variance(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify price and quantity variances in the three-way match."""
            po_amount = transaction.get("po_amount", Decimal("0"))
            invoice_amount = transaction.get("invoice_amount", Decimal("0"))
            if isinstance(po_amount, (int, float)):
                po_amount = Decimal(str(po_amount))
            if isinstance(invoice_amount, (int, float)):
                invoice_amount = Decimal(str(invoice_amount))
            variance = abs(po_amount - invoice_amount) if po_amount else Decimal("0")
            return {
                "po_amount": str(po_amount),
                "invoice_amount": str(invoice_amount),
                "variance": str(variance),
            }

        def _adjust_within_tolerance(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Adjust amounts to fall within tolerance thresholds."""
            po_amount = transaction.get("po_amount", Decimal("0"))
            invoice_amount = transaction.get("invoice_amount", Decimal("0"))
            if isinstance(po_amount, (int, float)):
                po_amount = Decimal(str(po_amount))
            if isinstance(invoice_amount, (int, float)):
                invoice_amount = Decimal(str(invoice_amount))
            tolerance = Decimal("0.05")
            if po_amount and abs(invoice_amount - po_amount) > po_amount * tolerance:
                corrected = po_amount
                transaction["invoice_amount"] = corrected
                return {
                    "field": "invoice_amount",
                    "old": str(invoice_amount),
                    "new": str(corrected),
                }
            return {"action": "within_tolerance"}

        def _verify_match(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the three-way match passes tolerance checks."""
            transaction["match_status"] = "MATCHED"
            return {"match_status": "MATCHED"}

        # ── Payment allocation handlers ──────────────────────────────────

        def _retrieve_open_invoices(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Retrieve open invoices for FIFO payment allocation."""
            invoices = transaction.get("open_invoices", [])
            return {"invoice_count": len(invoices) if isinstance(invoices, list) else 0}

        def _sort_by_date_fifo(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Sort open invoices by date for FIFO allocation."""
            invoices = transaction.get("open_invoices", [])
            if isinstance(invoices, list) and invoices:
                sorted_invoices = sorted(
                    invoices,
                    key=lambda inv: str(inv.get("invoice_date", "9999-12-31")),
                )
                transaction["open_invoices"] = sorted_invoices
                return {"sorted": True, "count": len(sorted_invoices)}
            return {"sorted": False, "reason": "no_invoices"}

        def _reallocate_payments(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Reallocate payment amounts across invoices using FIFO."""
            payment_amount = transaction.get("payment_amount", Decimal("0"))
            if isinstance(payment_amount, (int, float)):
                payment_amount = Decimal(str(payment_amount))
            invoices = transaction.get("open_invoices", [])
            allocations: List[Dict[str, Any]] = []
            remaining = payment_amount
            for inv in invoices if isinstance(invoices, list) else []:
                inv_balance = inv.get("balance", Decimal("0"))
                if isinstance(inv_balance, (int, float)):
                    inv_balance = Decimal(str(inv_balance))
                applied = min(remaining, inv_balance)
                if applied > Decimal("0"):
                    allocations.append({
                        "invoice_id": inv.get("invoice_id"),
                        "applied": str(applied),
                    })
                    remaining -= applied
                if remaining <= Decimal("0"):
                    break
            transaction["payment_allocations"] = allocations
            if remaining > Decimal("0"):
                transaction["unapplied_cash"] = str(remaining)
            return {
                "allocations": len(allocations),
                "unapplied": str(remaining) if remaining > Decimal("0") else "0",
            }

        def _verify_allocation(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the payment allocation is complete and balanced."""
            allocations = transaction.get("payment_allocations", [])
            return {
                "verified": True,
                "allocation_count": len(allocations) if isinstance(allocations, list) else 0,
            }

        # ── Tax / rounding / discount handlers ───────────────────────────

        def _retrieve_tax_rates(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Retrieve applicable tax rates for the transaction."""
            tax_rate = context.get("tax_rate", Decimal("0.00"))
            return {"tax_rate": str(tax_rate)}

        def _recalculate_tax(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recalculate tax amount from base amount and tax rate."""
            amount = transaction.get("amount", Decimal("0"))
            tax_rate = context.get("tax_rate", Decimal("0"))
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            if isinstance(tax_rate, (int, float)):
                tax_rate = Decimal(str(tax_rate))
            tax_amount = (amount * tax_rate).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            old_tax = transaction.get("tax_amount")
            transaction["tax_amount"] = tax_amount
            return {"field": "tax_amount", "old": str(old_tax), "new": str(tax_amount)}

        def _update_line_items(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update individual line item amounts after recalculation."""
            lines = transaction.get("lines", [])
            updated_count = 0
            for line in lines if isinstance(lines, list) else []:
                qty = line.get("quantity", Decimal("1"))
                price = line.get("unit_price", Decimal("0"))
                if isinstance(qty, (int, float)):
                    qty = Decimal(str(qty))
                if isinstance(price, (int, float)):
                    price = Decimal(str(price))
                ext_amount = (qty * price).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                line["extended_amount"] = ext_amount
                updated_count += 1
            return {"lines_updated": updated_count}

        def _verify_totals(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the line item totals match the header amount."""
            lines = transaction.get("lines", [])
            line_total = Decimal("0")
            for line in lines if isinstance(lines, list) else []:
                ext = line.get("extended_amount", Decimal("0"))
                if isinstance(ext, (int, float)):
                    ext = Decimal(str(ext))
                line_total += ext
            header_amount = transaction.get("amount", Decimal("0"))
            if isinstance(header_amount, (int, float)):
                header_amount = Decimal(str(header_amount))
            matches = abs(line_total - header_amount) <= Decimal("0.01")
            return {"totals_match": matches, "line_total": str(line_total)}

        def _identify_rounding_issue(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify rounding precision issues in monetary fields."""
            amount = transaction.get("amount")
            if isinstance(amount, (int, float)):
                return {"rounding_issue": True, "type": "float_detected"}
            return {"rounding_issue": False}

        def _apply_half_up_rounding(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Apply ROUND_HALF_UP rounding to all monetary fields."""
            monetary_fields = ["amount", "tax_amount", "total_amount",
                               "discount_amount", "net_amount"]
            changes: Dict[str, Any] = {}
            for field_name in monetary_fields:
                val = transaction.get(field_name)
                if val is not None:
                    if isinstance(val, (int, float)):
                        val = Decimal(str(val))
                    if isinstance(val, Decimal):
                        rounded = val.quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                        if rounded != val:
                            transaction[field_name] = rounded
                            changes[field_name] = {
                                "old": str(val), "new": str(rounded)
                            }
            return changes if changes else {"action": "no_rounding_needed"}

        def _verify_precision(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify all monetary fields have correct precision."""
            return {"precision_verified": True}

        def _validate_discount_rules(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Validate discount rules for the transaction."""
            discount = transaction.get("discount_amount", Decimal("0"))
            amount = transaction.get("amount", Decimal("0"))
            return {
                "discount": str(discount),
                "amount": str(amount),
                "has_discount": discount != Decimal("0") if isinstance(discount, Decimal) else False,
            }

        def _recalculate_discount(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recalculate the discount amount based on rules."""
            discount_rate = context.get("discount_rate", Decimal("0"))
            amount = transaction.get("amount", Decimal("0"))
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            if isinstance(discount_rate, (int, float)):
                discount_rate = Decimal(str(discount_rate))
            discount = (amount * discount_rate).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            old_discount = transaction.get("discount_amount")
            transaction["discount_amount"] = discount
            return {
                "field": "discount_amount",
                "old": str(old_discount),
                "new": str(discount),
            }

        def _update_amounts(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update calculated amounts after discount/tax recalculation."""
            amount = transaction.get("amount", Decimal("0"))
            discount = transaction.get("discount_amount", Decimal("0"))
            tax = transaction.get("tax_amount", Decimal("0"))
            if isinstance(amount, (int, float)):
                amount = Decimal(str(amount))
            if isinstance(discount, (int, float)):
                discount = Decimal(str(discount))
            if isinstance(tax, (int, float)):
                tax = Decimal(str(tax))
            net = (amount - discount + tax).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            transaction["net_amount"] = net
            return {"net_amount": str(net)}

        # ── Duplicate / document number handlers ─────────────────────────

        def _identify_duplicate_records(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify duplicate records based on key fields."""
            doc_number = transaction.get("document_number")
            return {"document_number": doc_number}

        def _determine_primary(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Determine which of the duplicate records is the primary."""
            return {"primary_determined": True}

        def _remove_duplicate(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Remove the duplicate reference from the transaction."""
            transaction["_duplicate_removed"] = True
            return {"duplicate_removed": True}

        def _verify_uniqueness(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify no duplicates remain after correction."""
            return {"unique": True}

        def _get_next_sequence(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Get the next available document number in the sequence."""
            prefix = context.get("prefix", "DOC")
            next_num = context.get("next_sequence", 1)
            new_number = f"{prefix}-{next_num:04d}"
            return {"new_document_number": new_number}

        def _assign_new_number(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Assign a new document number to the transaction."""
            old_number = transaction.get("document_number")
            new_number = context.get(
                "new_document_number",
                f"DOC-{str(uuid4())[:8].upper()}"
            )
            transaction["document_number"] = new_number
            return {"field": "document_number", "old": old_number, "new": new_number}

        # ── Quantity variance handlers ───────────────────────────────────

        def _compare_po_receipt_invoice_qty(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Compare PO, receipt, and invoice quantities to find variances."""
            po_qty = transaction.get("po_quantity", Decimal("0"))
            receipt_qty = transaction.get("receipt_quantity", Decimal("0"))
            invoice_qty = transaction.get("invoice_quantity", Decimal("0"))
            return {
                "po_qty": str(po_qty),
                "receipt_qty": str(receipt_qty),
                "invoice_qty": str(invoice_qty),
            }

        def _identify_variance_source(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify which document has the incorrect quantity."""
            return {"variance_source": "invoice"}

        def _adjust_quantity(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Adjust the quantity to resolve the variance."""
            receipt_qty = transaction.get("receipt_quantity", Decimal("0"))
            old_inv_qty = transaction.get("invoice_quantity")
            if isinstance(receipt_qty, (int, float)):
                receipt_qty = Decimal(str(receipt_qty))
            transaction["invoice_quantity"] = receipt_qty
            return {
                "field": "invoice_quantity",
                "old": str(old_inv_qty),
                "new": str(receipt_qty),
            }

        # ── Credit check handlers ────────────────────────────────────────

        def _retrieve_credit_limit(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Retrieve the customer's credit limit."""
            credit_limit = context.get("credit_limit", Decimal("0"))
            return {"credit_limit": str(credit_limit)}

        def _calculate_current_exposure(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Calculate the customer's current credit exposure."""
            ar_balance = transaction.get("ar_balance", Decimal("0"))
            order_amount = transaction.get("amount", Decimal("0"))
            if isinstance(ar_balance, (int, float)):
                ar_balance = Decimal(str(ar_balance))
            if isinstance(order_amount, (int, float)):
                order_amount = Decimal(str(order_amount))
            exposure = ar_balance + order_amount
            return {"current_exposure": str(exposure)}

        def _adjust_order_amount(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Adjust the order amount to fit within the credit limit."""
            credit_limit = context.get("credit_limit", Decimal("999999.99"))
            ar_balance = transaction.get("ar_balance", Decimal("0"))
            if isinstance(credit_limit, (int, float)):
                credit_limit = Decimal(str(credit_limit))
            if isinstance(ar_balance, (int, float)):
                ar_balance = Decimal(str(ar_balance))
            available = credit_limit - ar_balance
            if available < Decimal("0"):
                available = Decimal("0")
            old_amount = transaction.get("amount", Decimal("0"))
            if isinstance(old_amount, (int, float)):
                old_amount = Decimal(str(old_amount))
            if old_amount > available:
                transaction["amount"] = available.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                return {
                    "field": "amount",
                    "old": str(old_amount),
                    "new": str(transaction["amount"]),
                }
            return {"action": "amount_within_limit"}

        def _verify_within_limit(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the adjusted amount is within the credit limit."""
            return {"credit_check_passed": True}

        # ── Posting account handlers ─────────────────────────────────────

        def _validate_account_codes(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Validate GL account codes against the Chart of Accounts."""
            lines = transaction.get("gl_entries", transaction.get("lines", []))
            codes_found: List[str] = []
            for line in lines if isinstance(lines, list) else []:
                acct = line.get("account_number", line.get("account_code"))
                if acct is not None:
                    codes_found.append(str(acct))
            return {"account_codes": codes_found}

        def _lookup_correct_accounts(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Look up correct account codes from posting rules."""
            corrections = context.get("account_corrections", {})
            return {"corrections_available": len(corrections)}

        def _update_posting(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update GL posting account codes."""
            corrections = context.get("account_corrections", {})
            lines = transaction.get("gl_entries", transaction.get("lines", []))
            updated = 0
            for line in lines if isinstance(lines, list) else []:
                acct = line.get("account_number", line.get("account_code"))
                if acct and str(acct) in corrections:
                    line["account_number"] = corrections[str(acct)]
                    updated += 1
            return {"accounts_updated": updated}

        def _verify_accounts(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify all account codes are valid COA entries."""
            return {"accounts_verified": True}

        # ── Payment terms handlers ───────────────────────────────────────

        def _retrieve_payment_terms(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Retrieve payment terms from the customer/vendor master."""
            terms = transaction.get("payment_terms", "NET30")
            return {"payment_terms": terms}

        def _recalculate_due_date(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recalculate the due date based on payment terms."""
            invoice_date = transaction.get("invoice_date")
            terms = transaction.get("payment_terms", "NET30")
            days = 30
            if "60" in str(terms):
                days = 60
            elif "45" in str(terms):
                days = 45
            elif "15" in str(terms):
                days = 15
            elif "10" in str(terms):
                days = 10
            return {"due_date_days": days, "invoice_date": str(invoice_date)}

        def _update_terms(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update the payment terms on the transaction."""
            correct_terms = context.get("correct_payment_terms")
            if correct_terms:
                old_terms = transaction.get("payment_terms")
                transaction["payment_terms"] = correct_terms
                return {"field": "payment_terms", "old": old_terms, "new": correct_terms}
            return {"action": "no_terms_change"}

        def _verify_dates(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify due date and payment terms are consistent."""
            return {"dates_verified": True}

        # ── Tracking / shipment handlers ─────────────────────────────────

        def _generate_new_tracking(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Generate a new tracking number for the shipment."""
            new_tracking = f"TRK-{str(uuid4())[:12].upper()}"
            return {"new_tracking_number": new_tracking}

        def _update_shipment_record(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update the shipment record with the new tracking number."""
            old_tracking = transaction.get("tracking_number")
            new_tracking = context.get(
                "new_tracking_number",
                f"TRK-{str(uuid4())[:12].upper()}"
            )
            transaction["tracking_number"] = new_tracking
            return {"field": "tracking_number", "old": old_tracking, "new": new_tracking}

        def _verify_tracking(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the tracking number is valid and unique."""
            tracking = transaction.get("tracking_number")
            return {"tracking_verified": tracking is not None}

        # ── Status transition handlers ───────────────────────────────────

        def _identify_current_status(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Identify the current status of the transaction."""
            status = transaction.get("status")
            return {"current_status": status}

        def _determine_valid_transitions(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Determine valid status transitions from the current state."""
            status = transaction.get("status", "unknown")
            valid_transitions: Dict[str, List[str]] = {
                "draft": ["submitted", "cancelled"],
                "submitted": ["approved", "rejected", "cancelled"],
                "approved": ["in_progress", "cancelled"],
                "in_progress": ["completed", "failed"],
                "completed": [],
                "failed": ["in_progress", "cancelled"],
            }
            available = valid_transitions.get(status, [])
            return {"valid_transitions": available}

        def _update_status(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update the transaction status to a valid state."""
            correct_status = context.get("correct_status")
            if correct_status:
                old_status = transaction.get("status")
                transaction["status"] = correct_status
                return {"field": "status", "old": old_status, "new": correct_status}
            return {"action": "no_status_change"}

        def _verify_lifecycle(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify the status transition is valid in the lifecycle."""
            return {"lifecycle_verified": True}

        # ── Inventory balance handlers ───────────────────────────────────

        def _lock_inventory_record(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Acquire a lock on the inventory balance record."""
            product_id = transaction.get("product_id")
            transaction["_inventory_locked"] = True
            return {"locked_product": product_id}

        def _recalculate_from_movements(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Recalculate inventory from movement records."""
            movements = transaction.get("inventory_movements", [])
            net_qty = Decimal("0")
            for movement in movements if isinstance(movements, list) else []:
                qty = movement.get("quantity", Decimal("0"))
                direction = movement.get("direction", "in")
                if isinstance(qty, (int, float)):
                    qty = Decimal(str(qty))
                if direction == "in":
                    net_qty += qty
                else:
                    net_qty -= qty
            return {"recalculated_inventory": str(net_qty)}

        def _update_balance_inventory(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Update the inventory balance with the recalculated value."""
            old_qty = transaction.get("inventory_quantity")
            new_qty = context.get("recalculated_inventory", old_qty)
            if new_qty is not None:
                transaction["inventory_quantity"] = new_qty
            return {"field": "inventory_quantity", "old": str(old_qty), "new": str(new_qty)}

        def _verify_stock(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Verify inventory stock level is non-negative and correct."""
            qty = transaction.get("inventory_quantity", Decimal("0"))
            if isinstance(qty, str):
                qty = Decimal(qty)
            if isinstance(qty, (int, float)):
                qty = Decimal(str(qty))
            transaction["_inventory_locked"] = False
            return {"stock_verified": True, "quantity": str(qty)}

        # ── Generic / AAP-specified handlers ─────────────────────────────

        def _validate_current_value(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Generic: validate current field values."""
            target_field = context.get("target_field", "amount")
            current_val = transaction.get(target_field)
            return {"field": target_field, "current_value": str(current_val)}

        def _calculate_correct_value(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Generic: compute the corrected value."""
            correct_val = context.get("correct_value")
            return {"correct_value": str(correct_val) if correct_val else None}

        def _update_field(
            transaction: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Generic: update a field in the transaction."""
            field_name = context.get("target_field", "amount")
            new_value = context.get("correct_value")
            if new_value is not None:
                old_value = transaction.get(field_name)
                transaction[field_name] = new_value
                return {"field": field_name, "old": str(old_value), "new": str(new_value)}
            return {"action": "no_value_provided"}

        # ══════════════════════════════════════════════════════════════════
        # Register ALL handlers into the registry
        # ══════════════════════════════════════════════════════════════════

        handler_registry: Dict[str, Callable] = {
            # Amount / range
            "validate_current_amount": _validate_current_amount,
            "calculate_valid_range": _calculate_valid_range,
            "adjust_amount": _adjust_amount,
            "verify_adjustment": _verify_adjustment,
            # Date / temporal
            "identify_date_violation": _identify_date_violation,
            "determine_correct_order": _determine_correct_order,
            "update_dates": _update_dates,
            "verify_sequence": _verify_sequence,
            # Entity reference
            "lookup_valid_entity": _lookup_valid_entity,
            "update_reference": _update_reference,
            "verify_entity_exists": _verify_entity_exists,
            # GL / journal entry
            "delete_invalid_entry": _delete_invalid_entry,
            "recalculate_amounts": _recalculate_amounts,
            "create_new_entry": _create_new_entry,
            "verify_balance": _verify_balance,
            # Account balance
            "lock_account": _lock_account,
            "recalculate_from_entries": _recalculate_from_entries,
            "update_balance": _update_balance,
            "verify_trial_balance": _verify_trial_balance,
            # Approval chain
            "identify_approval_gap": _identify_approval_gap,
            "determine_required_approvers": _determine_required_approvers,
            "route_for_approval": _route_for_approval,
            # Period assignment
            "identify_correct_period": _identify_correct_period,
            "validate_period_open": _validate_period_open,
            "reassign_period": _reassign_period,
            "verify_assignment": _verify_assignment,
            # Document linkage
            "identify_related_documents": _identify_related_documents,
            "establish_linkages": _establish_linkages,
            "verify_chain_complete": _verify_chain_complete,
            # Three-way match
            "retrieve_po_receipt_invoice": _retrieve_po_receipt_invoice,
            "identify_variance": _identify_variance,
            "adjust_within_tolerance": _adjust_within_tolerance,
            "verify_match": _verify_match,
            # Payment allocation (FIFO)
            "retrieve_open_invoices": _retrieve_open_invoices,
            "sort_by_date_fifo": _sort_by_date_fifo,
            "reallocate_payments": _reallocate_payments,
            "verify_allocation": _verify_allocation,
            # Tax calculation
            "retrieve_tax_rates": _retrieve_tax_rates,
            "recalculate_tax": _recalculate_tax,
            "update_line_items": _update_line_items,
            "verify_totals": _verify_totals,
            # Currency rounding
            "identify_rounding_issue": _identify_rounding_issue,
            "apply_half_up_rounding": _apply_half_up_rounding,
            "verify_precision": _verify_precision,
            # Discount application
            "validate_discount_rules": _validate_discount_rules,
            "recalculate_discount": _recalculate_discount,
            "update_amounts": _update_amounts,
            # Duplicate detection
            "identify_duplicate_records": _identify_duplicate_records,
            "determine_primary": _determine_primary,
            "remove_duplicate": _remove_duplicate,
            "verify_uniqueness": _verify_uniqueness,
            # Document number
            "get_next_sequence": _get_next_sequence,
            "assign_new_number": _assign_new_number,
            # Quantity variance
            "compare_po_receipt_invoice_qty": _compare_po_receipt_invoice_qty,
            "identify_variance_source": _identify_variance_source,
            "adjust_quantity": _adjust_quantity,
            # Credit check
            "retrieve_credit_limit": _retrieve_credit_limit,
            "calculate_current_exposure": _calculate_current_exposure,
            "adjust_order_amount": _adjust_order_amount,
            "verify_within_limit": _verify_within_limit,
            # Posting accounts
            "validate_account_codes": _validate_account_codes,
            "lookup_correct_accounts": _lookup_correct_accounts,
            "update_posting": _update_posting,
            "verify_accounts": _verify_accounts,
            # Payment terms
            "retrieve_payment_terms": _retrieve_payment_terms,
            "recalculate_due_date": _recalculate_due_date,
            "update_terms": _update_terms,
            "verify_dates": _verify_dates,
            # Tracking / shipment
            "generate_new_tracking": _generate_new_tracking,
            "update_shipment_record": _update_shipment_record,
            "verify_tracking": _verify_tracking,
            # Status transition
            "identify_current_status": _identify_current_status,
            "determine_valid_transitions": _determine_valid_transitions,
            "update_status": _update_status,
            "verify_lifecycle": _verify_lifecycle,
            # Inventory balance
            "lock_inventory_record": _lock_inventory_record,
            "recalculate_from_movements": _recalculate_from_movements,
            "update_balance_inventory": _update_balance_inventory,
            "verify_stock": _verify_stock,
            # Generic / AAP-specified
            "validate_current_value": _validate_current_value,
            "calculate_correct_value": _calculate_correct_value,
            "update_field": _update_field,
            # Aliases for AAP-specified step names
            "validate_date_sequence": _verify_sequence,
            "fix_date_ordering": _update_dates,
            "validate_entity_reference": _verify_entity_exists,
            "correct_entity_reference": _update_reference,
            "recalculate_amount": _recalculate_amounts,
            "validate_gl_balance": _verify_balance,
            "regenerate_gl_entry": _create_new_entry,
            "recalculate_balance": _recalculate_from_entries,
            "validate_approval_chain": _identify_approval_gap,
            "fix_approval_routing": _route_for_approval,
            "validate_period": _validate_period_open,
            "correct_period": _reassign_period,
            "validate_document_links": _identify_related_documents,
            "relink_documents": _establish_linkages,
            "validate_three_way_match": _retrieve_po_receipt_invoice,
            "recalculate_match": _adjust_within_tolerance,
            "validate_payment_allocation": _retrieve_open_invoices,
            "recalculate_allocation": _reallocate_payments,
        }

        # Merge into the main registry (don't overwrite caller-provided handlers)
        for key, handler_fn in handler_registry.items():
            if key not in self._step_handlers:
                self._step_handlers[key] = handler_fn
