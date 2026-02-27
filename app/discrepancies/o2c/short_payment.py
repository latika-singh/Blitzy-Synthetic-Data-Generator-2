"""O2C-004: Short Payment discrepancy type.

Simulates a customer payment that is less than the full invoice amount
without proper authorization, documentation, or dispute resolution for
the shortfall.

In the normal O2C flow, the CustomerPaymentProcessor uses FIFO allocation
(oldest invoice first). Short payments may be intentional (authorized
deductions, disputes) or discrepant (unexplained underpayment).

This discrepancy injects an unauthorized shortfall — the payment is reduced
by a configurable percentage, creating an open balance on the invoice that
lacks proper dispute documentation.

Detection Method: ``payment_matching`` — comparing payment amount against
invoice balance and checking for authorized deduction documentation.

Difficulty: ``easy`` — standard payment-to-invoice matching.

Parameter Bounds:
    short_percent: int, range [1, 20] — percentage below invoice amount.

References:
    - AAP Section 0.5.1 Group 3: CustomerPaymentProcessor FIFO allocation
    - AAP Section 0.5.1 Group 5: O2C-004 Short Payment
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.1.2: Payment Allocation is FIFO
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
# Module-level structured logger (AAP Section 0.7.7 — stdout JSON only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["ShortPayment"]

# ---------------------------------------------------------------------------
# Two-digit quantizer for rounding monetary values to cents
# ---------------------------------------------------------------------------
_TWO_PLACES = Decimal("0.01")


class ShortPayment(BaseDiscrepancy):
    """O2C-004: Short Payment discrepancy.

    Simulates a customer payment that is less than the full invoice amount
    without proper authorization, creating an unexplained open balance on
    the customer's account.

    In the normal O2C cycle (AAP §0.5.1 Group 3), the
    ``CustomerPaymentProcessor`` allocates payments via FIFO (oldest invoice
    first).  When a payment is intentionally short, an authorized deduction
    record with supporting dispute documentation is created.

    This discrepancy bypasses that control — the payment amount is reduced
    by a configurable percentage (``short_percent``, 1–20%), and the
    resulting shortfall is recorded **without** an authorized deduction
    or dispute document.

    Class Attributes:
        type_code: ``"O2C-004"``
        category: ``"o2c"``
        difficulty: ``"easy"``
        name: ``"Short Payment"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"payment_matching"``
        PARAMETER_BOUNDS: ``short_percent`` in ``[1, 20]``.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-004"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Short Payment"
    description: ClassVar[str] = (
        "Customer payment less than invoice amount without authorized deduction "
        "documentation, creating an unexplained open balance."
    )
    detection_method: ClassVar[str] = "payment_matching"

    # ------------------------------------------------------------------
    # Parameter bounds for discrepancy injection validation
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "short_pct": {"min": 0.01, "max": 0.25, "type": "float"},
    }

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a short payment discrepancy into the transaction.

        Reduces the payment amount by a configurable percentage of the
        invoice amount, and marks the transaction as lacking authorized
        deduction documentation.  The resulting shortfall creates an
        unexplained open balance on the customer's account.

        Args:
            transaction: The original transaction data dictionary.  Expected
                keys include ``"invoice_amount"`` (or ``"payment_amount"``
                as a fallback), ``"transaction_id"``, and optionally
                ``"customer_id"``.
            params: Injection parameters.  Supported keys:

                - ``"short_percent"`` (int): Percentage to reduce payment
                  by.  Must be in ``[1, 20]``.  If omitted, a random
                  value within bounds is generated via *rng*.
            rng: A seeded :class:`random.Random` instance for deterministic
                behavior.  **CRITICAL**: MUST use this RNG, NEVER the
                module-level ``random`` functions.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``:

            - **modified_transaction**: Deep copy of *transaction* with
              ``payment_amount`` reduced, ``short_pay_amount`` set to the
              shortfall, ``authorized_deduction`` set to ``False``, and
              ``dispute_documented`` set to ``False``.
            - **ground_truth_data**: Dictionary containing type metadata,
              affected fields, original/modified values, financial impact,
              and a human-readable description.

        Raises:
            DiscrepancyInjectionError: If the transaction is missing the
                ``"invoice_amount"`` field or if the invoice amount is
                not positive, preventing meaningful short-pay calculation.
        """
        # ── Step 1: Deep copy to preserve original ───────────────────
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # ── Step 2: Validate params against bounds ───────────────────
        validated_params: Dict[str, Any] = self._validate_params(
            params, self.PARAMETER_BOUNDS
        )

        # ── Step 3: Determine short_pct (fractional 0.01–0.25) ────────
        # Use the provided value or generate a default via the seeded RNG.
        # Parameter aligned to YAML o2c_discrepancies.yaml O2C-004 semantics:
        # fractional value where 0.01 = 1%, 0.25 = 25%.
        short_pct: float = validated_params.get(
            "short_pct", rng.uniform(0.01, 0.25)
        )

        # ── Step 4: Extract invoice amount ───────────────────────────
        # The transaction should carry an "invoice_amount" field.  As a
        # fallback, accept "payment_amount" (if the payment was already
        # set to the full invoice amount before injection).
        raw_amount = transaction.get(
            "invoice_amount",
            transaction.get("payment_amount"),
        )
        if raw_amount is None:
            raise DiscrepancyInjectionError(
                "Transaction missing 'invoice_amount' field required for "
                "short payment injection",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "available_keys": list(transaction.keys()),
                },
            )

        # Convert to Decimal for financial-grade arithmetic
        try:
            invoice_amount: Decimal = Decimal(str(raw_amount))
        except Exception as exc:
            raise DiscrepancyInjectionError(
                f"Cannot convert invoice_amount to Decimal: {raw_amount!r}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "raw_value": str(raw_amount),
                    "error": str(exc),
                },
            ) from exc

        if invoice_amount <= Decimal("0"):
            raise DiscrepancyInjectionError(
                f"Invoice amount must be positive, got {invoice_amount}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "invoice_amount": str(invoice_amount),
                },
            )

        # ── Step 5: Calculate shortfall ──────────────────────────────
        # shortfall = invoice_amount * short_pct, rounded to
        # 2 decimal places using ROUND_HALF_UP per AAP §0.7.2.
        shortfall: Decimal = (
            invoice_amount * Decimal(str(short_pct))
        ).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)

        # Ensure shortfall is at least $0.01 so the discrepancy is visible
        if shortfall < Decimal("0.01"):
            shortfall = Decimal("0.01")

        # ── Step 6: Compute new payment amount ───────────────────────
        new_payment: Decimal = (invoice_amount - shortfall).quantize(
            _TWO_PLACES, rounding=ROUND_HALF_UP
        )

        # ── Step 7: Update modified transaction ──────────────────────
        # Record the original payment amount for audit trail
        original_payment_str: str = str(
            transaction.get("payment_amount", str(invoice_amount))
        )

        modified["payment_amount"] = str(new_payment)
        modified["short_pay_amount"] = str(shortfall)
        modified["authorized_deduction"] = False
        modified["dispute_documented"] = False
        # Preserve the invoice_amount so downstream consumers can see
        # the gap between invoice and payment
        modified["invoice_amount"] = str(invoice_amount)

        # ── Step 8: Financial impact ─────────────────────────────────
        # The financial impact is the shortfall — the uninvoiced open
        # balance left on the customer's account.
        financial_impact: Decimal = shortfall

        # ── Step 9: Build ground truth data ──────────────────────────
        ground_truth: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=[
                "payment_amount",
                "short_pay_amount",
                "authorized_deduction",
            ],
            original_values={
                "payment_amount": original_payment_str,
                "authorized_deduction": True,
            },
            modified_values={
                "payment_amount": str(new_payment),
                "short_pay_amount": str(shortfall),
                "authorized_deduction": False,
            },
            financial_impact=financial_impact,
            description=(
                f"Short payment of {short_pct:.0%} "
                f"\u2014 underpayment of ${shortfall}"
            ),
            extra_metadata={
                "short_pct": short_pct,
                "invoice_amount": str(invoice_amount),
                "dispute_documented": False,
                "allocation_method": "FIFO",
            },
        )

        # ── Step 10: Log injection and return ────────────────────────
        transaction_id: str = str(
            modified.get("transaction_id", "unknown")
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
