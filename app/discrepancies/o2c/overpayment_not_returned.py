"""O2C-005: Overpayment Not Returned discrepancy type.

Simulates a customer overpayment where the excess amount is neither returned
to the customer nor properly recorded as Unapplied Cash.

In the normal O2C flow (AAP §0.1.2), when Payment > Invoice:
    - Invoice balance is set to $0 (NEVER negative)
    - Excess is created as an Unapplied Cash record

This discrepancy simulates the failure of this control — the overpayment is
absorbed without creating an Unapplied Cash record, effectively hiding the
customer's credit balance.

Detection Method: ``payment_matching`` — comparing payment totals against
invoice amounts and checking for corresponding Unapplied Cash records.

Difficulty: ``medium`` — requires cross-referencing payment allocations with
unapplied cash records.

Parameter Bounds:
    overpay_pct: float, range [0.01, 0.20] — fractional overpayment above
        the invoice amount (e.g. 0.10 = 10%).

References:
    - AAP Section 0.1.2: Overpayment handling rule
    - AAP Section 0.5.1 Group 3: CustomerPaymentProcessor
    - AAP Section 0.5.1 Group 5: O2C-005 Overpayment Not Returned
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
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Two-decimal quantizer used for all financial rounding
# ---------------------------------------------------------------------------
_TWO_PLACES = Decimal("0.01")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["OverpaymentNotReturned"]


class OverpaymentNotReturned(BaseDiscrepancy):
    """O2C-005: Overpayment Not Returned discrepancy.

    Simulates a customer overpayment scenario where the excess payment
    amount is absorbed without creating the required Unapplied Cash record
    or returning the excess to the customer.

    In a properly functioning O2C system (per AAP §0.1.2):
        - Payment > Invoice → Unapplied Cash record is created
        - Invoice balance is set to exactly $0.00 (NEVER negative)

    This discrepancy injects the *failure* of that control:
        - Payment is inflated by ``overpay_pct`` above the invoice amount
        - ``unapplied_cash_created`` is set to ``False``
        - ``unapplied_cash_amount`` is set to ``Decimal("0.00")``
        - ``overpayment_returned`` is set to ``False``
        - ``invoice_balance`` remains at ``Decimal("0.00")`` — NEVER negative

    The financial impact equals the overpayment amount — cash the customer
    is owed but has not been returned or properly tracked.

    Attributes:
        type_code: ``"O2C-005"``
        category: ``"o2c"``
        difficulty: ``"medium"``
        name: ``"Overpayment Not Returned"``
        description: Human-readable summary of the discrepancy.
        detection_method: ``"payment_matching"``
        PARAMETER_BOUNDS: ``{"overpay_pct": {"min": 0.01, "max": 0.20, "type": "float"}}``
    """

    # ------------------------------------------------------------------
    # Class-level attributes required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-005"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Overpayment Not Returned"
    description: ClassVar[str] = (
        "Customer overpayment not returned or recorded as Unapplied Cash. "
        "Excess payment absorbed without proper accounting treatment."
    )
    detection_method: ClassVar[str] = "payment_matching"

    # ------------------------------------------------------------------
    # Parameter Bounds — validated by _validate_params()
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "overpay_pct": {"min": 0.01, "max": 0.20, "type": "float"},
    }

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an overpayment-not-returned discrepancy into the transaction.

        Inflates the payment amount above the invoice amount by
        ``overpay_pct`` (fractional), then suppresses the Unapplied Cash
        record that should normally be created.

        Steps:
            1. Deep-copy the transaction to preserve the original.
            2. Validate parameters against ``PARAMETER_BOUNDS``.
            3. Determine ``overpay_pct`` (from params or generated via
               ``rng.uniform(0.01, 0.20)``).
            4. Extract ``invoice_amount`` from transaction data.
            5. Compute the overpayment amount using ``Decimal`` with
               ``ROUND_HALF_UP`` rounding to 2 decimal places.
            6. Set ``payment_amount`` = invoice + overpayment.
            7. Suppress the Unapplied Cash record (the discrepancy).
            8. Ensure ``invoice_balance`` is ``Decimal("0.00")`` — **NEVER**
               negative per AAP §0.1.2.
            9. Build the ground truth data for ``GroundTruthGenerator``.
            10. Log the injection event via ``_log_injection()``.

        Args:
            transaction: Original transaction data dictionary.  Must
                contain at minimum an ``"invoice_amount"`` key with a
                numeric value (``int``, ``float``, ``str``, or
                ``Decimal``).  Should also contain ``"transaction_id"``
                for logging.
            params: Injection parameters.  Recognised key:
                ``"overpay_pct"`` (float, range [0.01, 0.20]).  If absent,
                a random value within bounds is generated using *rng*.
            rng: Seeded :class:`random.Random` instance for deterministic
                reproducibility.  **NEVER** use module-level ``random``.

        Returns:
            A 2-tuple of:
                - **modified_transaction** (Dict[str, Any]): Transaction
                  with inflated ``payment_amount``, suppressed unapplied
                  cash, and zero invoice balance.
                - **ground_truth_data** (Dict[str, Any]): Ground truth
                  record for ``GroundTruthGenerator``.

        Raises:
            DiscrepancyInjectionError: If ``invoice_amount`` is missing
                or non-positive, or if any calculation error occurs.
        """
        try:
            # ----------------------------------------------------------
            # 1. Deep-copy the transaction to preserve the original
            # ----------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ----------------------------------------------------------
            # 2. Validate parameters against bounds (auto-adjust=True)
            # ----------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # ----------------------------------------------------------
            # 3. Determine overpay_pct (fractional, e.g. 0.10 = 10%)
            # ----------------------------------------------------------
            overpay_pct: float = validated_params.get(
                "overpay_pct", rng.uniform(0.01, 0.20)
            )

            # ----------------------------------------------------------
            # 4. Extract the invoice amount from the transaction
            # ----------------------------------------------------------
            raw_invoice_amount = transaction.get("invoice_amount")
            if raw_invoice_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required 'invoice_amount' field "
                    "for O2C-005 overpayment injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                    },
                )

            invoice_amount = Decimal(str(raw_invoice_amount))

            if invoice_amount <= Decimal("0"):
                raise DiscrepancyInjectionError(
                    f"Invoice amount must be positive for O2C-005 injection, "
                    f"got {invoice_amount}",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "invoice_amount": str(invoice_amount),
                    },
                )

            # ----------------------------------------------------------
            # 5. Calculate the overpayment amount (Decimal, ROUND_HALF_UP)
            # ----------------------------------------------------------
            overpayment = (
                invoice_amount
                * Decimal(str(overpay_pct))
            ).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)

            # ----------------------------------------------------------
            # 6. Compute inflated payment_amount
            # ----------------------------------------------------------
            payment_amount = (invoice_amount + overpayment).quantize(
                _TWO_PLACES, rounding=ROUND_HALF_UP
            )

            # ----------------------------------------------------------
            # 7 & 8. Modify the transaction — suppress Unapplied Cash
            #         and ensure invoice_balance is NEVER negative
            # ----------------------------------------------------------
            # Store original payment value for ground truth
            original_payment = str(
                transaction.get("payment_amount", invoice_amount)
            )

            modified["payment_amount"] = payment_amount
            # The discrepancy: Unapplied Cash record NOT created
            modified["unapplied_cash_created"] = False
            modified["unapplied_cash_amount"] = Decimal("0.00")
            # CRITICAL: Invoice balance is NEVER negative (AAP §0.1.2)
            modified["invoice_balance"] = Decimal("0.00")
            # The excess was NOT returned to the customer
            modified["overpayment_returned"] = False
            # Mark the overpayment amount for audit trail
            modified["overpayment_amount"] = overpayment

            # ----------------------------------------------------------
            # 9. Build ground truth data via base class helper
            # ----------------------------------------------------------
            affected_fields: List[str] = [
                "payment_amount",
                "unapplied_cash_created",
                "unapplied_cash_amount",
                "overpayment_returned",
            ]

            original_values: Dict[str, Any] = {
                "payment_amount": str(invoice_amount),
                "unapplied_cash_created": True,
                "unapplied_cash_amount": str(overpayment),
                "overpayment_returned": True,
            }

            modified_values: Dict[str, Any] = {
                "payment_amount": str(payment_amount),
                "unapplied_cash_created": False,
                "unapplied_cash_amount": "0.00",
                "overpayment_returned": False,
            }

            financial_impact = overpayment

            description = (
                f"Overpayment of {overpay_pct * 100:.1f}% (${overpayment}) not "
                f"returned or recorded as Unapplied Cash"
            )

            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=description,
                extra_metadata={
                    "overpay_pct": overpay_pct,
                    "invoice_amount": str(invoice_amount),
                    "payment_amount": str(payment_amount),
                    "overpayment_amount": str(overpayment),
                    "original_payment_amount": original_payment,
                },
            )

            # ----------------------------------------------------------
            # 10. Log the successful injection
            # ----------------------------------------------------------
            transaction_id = str(
                transaction.get("transaction_id", "unknown")
            )
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "simulation_id": transaction.get("simulation_id"),
                    "trace_id": transaction.get("trace_id"),
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Unexpected error during O2C-005 injection: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
