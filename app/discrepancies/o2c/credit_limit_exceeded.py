"""O2C-003: Credit Limit Exceeded discrepancy type.

Simulates a sales order or invoice that was processed even though the
customer's outstanding AR balance plus the new order amount exceeds their
configured credit limit.

In the normal O2C flow, the SalesOrderGenerator performs a credit check:
    credit_limit >= current_ar_balance + new_order_amount

This discrepancy bypasses or overrides that check, allowing the order to
proceed beyond the customer's creditworthiness.

Detection Method: ``credit_check`` — comparing customer's AR balance + order
amount against their credit limit.

Difficulty: ``easy`` — standard credit limit validation.

Parameter Bounds:
    excess_percent: int, range [1, 50] — percentage by which the order
        exceeds the credit limit.

References:
    - AAP Section 0.5.1 Group 3: SalesOrderGenerator credit check
    - AAP Section 0.5.1 Group 5: O2C-003 Credit Limit Exceeded
    - AAP Section 0.7.5: Discrepancy Injection Rules
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, ClassVar, Dict, List, Tuple

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — JSON to stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["CreditLimitExceeded"]


class CreditLimitExceeded(BaseDiscrepancy):
    """O2C-003: Credit Limit Exceeded discrepancy.

    Simulates a sales order processed despite the customer's outstanding AR
    balance plus the new order amount exceeding their configured credit limit.
    The discrepancy increases the order amount so that the total exposure
    (``current_ar_balance + order_amount``) exceeds ``credit_limit`` by a
    configurable percentage (``excess_percent``), and marks the credit check
    as bypassed.

    Class-Level Attributes:
        type_code: ``"O2C-003"``
        category: ``"o2c"``
        difficulty: ``"easy"``
        name: ``"Credit Limit Exceeded"``
        description: Explains the credit limit bypass scenario.
        detection_method: ``"credit_check"``
        PARAMETER_BOUNDS: ``excess_percent`` in ``[1, 50]``.

    Usage::

        injector = CreditLimitExceeded()
        modified_txn, ground_truth = injector.inject(
            transaction=sales_order_data,
            params={"excess_percent": 25},
            rng=random.Random(42),
        )
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-003"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Credit Limit Exceeded"
    description: ClassVar[str] = (
        "Sales order processed despite customer's outstanding AR balance plus "
        "order amount exceeding their configured credit limit."
    )
    detection_method: ClassVar[str] = "credit_check"

    # ------------------------------------------------------------------
    # Parameter bounds for this discrepancy type
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "excess_percent": {"min": 1, "max": 50, "type": "int"},
    }

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a credit-limit-exceeded discrepancy into a sales order.

        The method increases the order amount so that the customer's total
        exposure (current AR balance + new order amount) exceeds the credit
        limit by ``excess_percent``.  The credit check flag is set to
        ``False`` (bypassed), and the original credit check status is
        preserved in the ground truth record.

        Args:
            transaction: Sales order transaction dictionary.  Expected keys:

                - ``"transaction_id"`` (str): Unique transaction identifier.
                - ``"order_amount"`` (str | Decimal | float): Current order
                  amount.
                - ``"credit_limit"`` (str | Decimal | float): Customer's
                  configured credit limit.
                - ``"current_ar_balance"`` (str | Decimal | float, optional):
                  Customer's outstanding AR balance.  Defaults to
                  ``Decimal("0.00")`` if absent.
                - ``"credit_check_passed"`` (bool, optional): Whether the
                  credit check originally passed.  Defaults to ``True``.

            params: Injection parameters.  Recognised keys:

                - ``"excess_percent"`` (int): Percentage by which the order
                  should exceed the available credit.  Range ``[1, 50]``.
                  If not provided, a random value in ``[1, 50]`` is
                  generated using *rng*.

            rng: A seeded :class:`random.Random` instance.  MUST be used
                for all non-deterministic operations.  NEVER use the
                module-level ``random`` functions.

        Returns:
            A 2-tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction is missing the
                ``credit_limit`` field or if the credit limit is zero or
                negative (cannot meaningfully exceed a non-positive limit).
        """
        try:
            # 1. Deep copy transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate and clamp parameters against bounds
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # 3. Extract excess_percent — default via seeded RNG if absent
            excess_percent: int = int(
                validated_params.get("excess_percent", rng.randint(1, 50))
            )

            # 4. Extract credit_limit from transaction data
            raw_credit_limit = modified.get("credit_limit")
            if raw_credit_limit is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required 'credit_limit' field for "
                    "O2C-003 Credit Limit Exceeded injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                        "missing_field": "credit_limit",
                    },
                )

            credit_limit: Decimal = Decimal(str(raw_credit_limit))
            if credit_limit <= Decimal("0"):
                raise DiscrepancyInjectionError(
                    "Customer credit_limit must be positive for O2C-003 "
                    "Credit Limit Exceeded injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                        "credit_limit": str(credit_limit),
                    },
                )

            # 5. Extract current AR balance (default 0 if absent)
            current_ar_balance: Decimal = Decimal(
                str(modified.get("current_ar_balance", "0.00"))
            )

            # Capture original values before modification
            original_order_amount: Decimal = Decimal(
                str(modified.get("order_amount", "0.00"))
            )
            original_credit_check_passed: bool = modified.get(
                "credit_check_passed", True
            )

            # 6. Calculate the excess amount
            #    excess_amount = credit_limit * excess_percent / 100
            excess_amount: Decimal = (
                credit_limit
                * Decimal(str(excess_percent))
                / Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # 7. Calculate the new order amount that will exceed the credit limit
            #    available_credit = credit_limit - current_ar_balance
            #    new_order_amount = available_credit + excess_amount
            #    This ensures: current_ar_balance + new_order_amount > credit_limit
            available_credit: Decimal = credit_limit - current_ar_balance
            if available_credit < Decimal("0"):
                # AR balance already exceeds credit limit;
                # set order to at least the excess amount
                available_credit = Decimal("0")

            new_order_amount: Decimal = (
                available_credit + excess_amount
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Ensure the new order amount is at least $1.00 (avoid trivial amounts)
            if new_order_amount < Decimal("1.00"):
                new_order_amount = Decimal("1.00") + excess_amount

            # 8. Modify the transaction
            modified["order_amount"] = str(new_order_amount)
            modified["credit_check_passed"] = False
            modified["credit_check_bypassed"] = True
            modified["credit_limit_excess_amount"] = str(excess_amount)
            modified["credit_limit_excess_percent"] = excess_percent

            # Compute the actual financial impact — the amount exceeding credit limit
            total_exposure: Decimal = current_ar_balance + new_order_amount
            financial_impact: Decimal = (
                total_exposure - credit_limit
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Ensure financial impact is positive
            if financial_impact < Decimal("0"):
                financial_impact = excess_amount

            # 9. Build ground truth data
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=["order_amount", "credit_check_passed"],
                original_values={
                    "order_amount": str(original_order_amount),
                    "credit_check_passed": original_credit_check_passed,
                },
                modified_values={
                    "order_amount": str(new_order_amount),
                    "credit_check_passed": False,
                    "credit_check_bypassed": True,
                },
                financial_impact=financial_impact,
                description=(
                    f"Order exceeds credit limit by {excess_percent}% — "
                    f"excess amount ${excess_amount}, total exposure "
                    f"${total_exposure} vs credit limit ${credit_limit}"
                ),
                extra_metadata={
                    "excess_percent": excess_percent,
                    "credit_limit": str(credit_limit),
                    "current_ar_balance": str(current_ar_balance),
                    "available_credit": str(available_credit),
                    "total_exposure": str(total_exposure),
                },
            )

            # 10. Log the injection event
            self._log_injection(
                transaction_id=str(
                    modified.get("transaction_id", "unknown")
                ),
                financial_impact=financial_impact,
                context={
                    "simulation_id": modified.get("simulation_id"),
                    "trace_id": modified.get("trace_id"),
                },
            )

            logger.debug(
                "credit_limit_exceeded_injected",
                service_name="transactions",
                component="CreditLimitExceeded",
                type_code=self.type_code,
                transaction_id=str(
                    modified.get("transaction_id", "unknown")
                ),
                excess_percent=excess_percent,
                credit_limit=str(credit_limit),
                original_order_amount=str(original_order_amount),
                new_order_amount=str(new_order_amount),
                financial_impact=str(financial_impact),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors without wrapping
            raise
        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError for
            # consistent error handling per AAP §0.7.4
            raise DiscrepancyInjectionError(
                f"Unexpected error during O2C-003 Credit Limit Exceeded "
                f"injection: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
