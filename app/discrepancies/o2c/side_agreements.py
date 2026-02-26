"""O2C-010: Side Agreements Not Disclosed discrepancy type.

Simulates transactions with undisclosed side agreements — secret contractual
terms between the company and a customer that modify effective pricing,
payment conditions, or return rights without proper documentation or
accounting treatment.

Side agreements may include:
- Undisclosed discounts or rebates not recorded in the system
- Extended payment terms beyond standard customer terms
- Right of return agreements not reflected in revenue recognition
- Price protection guarantees not accrued as liabilities
- Conditional sales terms that affect revenue timing

These agreements represent a control weakness and potential fraud indicator
because the actual economics of the transaction differ from what is recorded
in the ERP system.

Detection Method: ``document_review`` — comparing recorded transaction terms
with actual contractual documents and identifying discrepancies between
stated and effective pricing.

Difficulty: ``medium`` — requires analysis of pricing patterns and comparison
with documented terms.

Parameter Bounds:
    discount_percent: int, range [5, 30] — the undisclosed discount percentage
        that reduces the effective price below the recorded invoice amount.

References:
    - AAP Section 0.5.1 Group 5: O2C-010 Side Agreements Not Disclosed
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Financial Integrity Rules
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
__all__ = ["SideAgreements"]

# ---------------------------------------------------------------------------
# Side agreement type choices for deterministic selection via rng.choice
# ---------------------------------------------------------------------------
_SIDE_AGREEMENT_TYPES: List[str] = [
    "undisclosed_discount",
    "extended_terms",
    "return_right",
    "price_protection",
]

# Two-decimal quantizer for financial rounding
_TWO_PLACES = Decimal("0.01")

# Hundred constant for percentage calculations
_HUNDRED = Decimal("100")


class SideAgreements(BaseDiscrepancy):
    """O2C-010: Side Agreements Not Disclosed discrepancy.

    Simulates an undisclosed side agreement that modifies the effective
    pricing of a customer transaction without proper documentation or
    accounting treatment.  The recorded amount in the ERP system remains
    at the full invoice value while a secret discount reduces the
    effective price the customer actually pays.

    The discrepancy creates a gap between the *recorded_amount* (what the
    ERP shows) and the *effective_price* (what the customer actually
    pays), with the difference being the undisclosed discount.

    Class Attributes:
        type_code: ``"O2C-010"`` — unique discrepancy identifier.
        category: ``"o2c"`` — Order-to-Cash category.
        difficulty: ``"medium"`` — requires document review and pricing
            pattern analysis for detection.
        name: ``"Side Agreements Not Disclosed"`` — human-readable name.
        description: Summary of the discrepancy's nature.
        detection_method: ``"document_review"`` — primary detection approach.
        PARAMETER_BOUNDS: Bounds for ``discount_percent`` parameter
            (min=5, max=30).

    Example::

        from app.discrepancies.o2c.side_agreements import SideAgreements
        import random

        discrepancy = SideAgreements()
        rng = random.Random(42)
        transaction = {
            "transaction_id": "INV-2024-0001",
            "invoice_amount": "10000.00",
            "customer_id": "CUST-001",
        }
        modified, ground_truth = discrepancy.inject(
            transaction=transaction,
            params={"discount_percent": 15},
            rng=rng,
        )
        assert modified["has_side_agreement"] is True
        assert modified["effective_price"] == "8500.00"
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-010"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Side Agreements Not Disclosed"
    description: ClassVar[str] = (
        "Undisclosed side agreements modifying effective pricing, payment terms, "
        "or return rights without proper documentation or accounting treatment."
    )
    detection_method: ClassVar[str] = "document_review"

    # ------------------------------------------------------------------
    # Parameter bounds — discount_percent [5, 30]
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "discount_percent": {"min": 5, "max": 30, "type": "int"},
    }

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection implementation
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an undisclosed side agreement into the transaction.

        Modifies the transaction to include secret side agreement terms
        that create a discrepancy between the recorded invoice amount
        and the effective price the customer pays.  The recorded amount
        stays at the original face value — this *is* the discrepancy.

        The method:
        1. Creates a deep copy of the transaction to preserve the original.
        2. Validates and extracts the ``discount_percent`` parameter.
        3. Computes the undisclosed discount and effective price using
           :class:`Decimal` arithmetic with ``ROUND_HALF_UP``.
        4. Adds side agreement fields to the transaction.
        5. Generates structured ground truth data for the
           ``GroundTruthGenerator``.

        Args:
            transaction: The original transaction data dictionary.  Must
                contain either ``"invoice_amount"`` or ``"order_amount"``
                as a numeric string or :class:`Decimal`.
            params: Injection parameters.  Recognised key:

                - ``"discount_percent"`` (int, 5–30): The undisclosed
                  discount percentage.  If absent, a random value in
                  [5, 30] is generated via *rng*.

            rng: A seeded :class:`random.Random` instance for
                deterministic reproducibility.  NEVER use module-level
                RNG.

        Returns:
            A 2-tuple of:

            - **modified_transaction** (Dict[str, Any]) — The transaction
              with side agreement fields injected.
            - **ground_truth_data** (Dict[str, Any]) — Structured data
              for ground truth record creation.

        Raises:
            DiscrepancyInjectionError: If the transaction lacks a valid
                amount field (``invoice_amount`` or ``order_amount``) or
                if any other injection logic error occurs.
        """
        try:
            # ---------------------------------------------------------------
            # Step 1: Deep copy transaction to preserve the original
            # ---------------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ---------------------------------------------------------------
            # Step 2: Validate params against PARAMETER_BOUNDS
            # ---------------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # ---------------------------------------------------------------
            # Step 3: Extract discount_percent (default via rng if absent)
            # ---------------------------------------------------------------
            discount_percent: int = int(
                validated_params.get(
                    "discount_percent", rng.randint(5, 30)
                )
            )

            # ---------------------------------------------------------------
            # Step 4: Get invoice/order amount from transaction
            # ---------------------------------------------------------------
            raw_amount = transaction.get(
                "invoice_amount",
                transaction.get("order_amount"),
            )
            if raw_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing 'invoice_amount' or 'order_amount' "
                    "field required for side agreement injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "available_keys": list(transaction.keys()),
                    },
                )

            invoice_amount: Decimal = Decimal(str(raw_amount))

            # ---------------------------------------------------------------
            # Step 5: Calculate undisclosed discount using Decimal arithmetic
            # ---------------------------------------------------------------
            undisclosed_discount: Decimal = (
                invoice_amount
                * Decimal(str(discount_percent))
                / _HUNDRED
            ).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)

            # ---------------------------------------------------------------
            # Step 6: Calculate effective price
            # ---------------------------------------------------------------
            effective_price: Decimal = (
                invoice_amount - undisclosed_discount
            ).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)

            # ---------------------------------------------------------------
            # Step 7: Select a side agreement type deterministically via rng
            # ---------------------------------------------------------------
            agreement_type: str = rng.choice(_SIDE_AGREEMENT_TYPES)

            # ---------------------------------------------------------------
            # Step 8: Modify the transaction with side agreement fields
            # ---------------------------------------------------------------
            # The recorded amount stays at face value — that IS the discrepancy.
            modified["has_side_agreement"] = True
            modified["undisclosed_discount_amount"] = str(undisclosed_discount)
            modified["effective_price"] = str(effective_price)
            modified["side_agreement_type"] = agreement_type
            modified["recorded_amount"] = str(invoice_amount)
            modified["side_agreement_documented"] = False
            modified["discount_percent"] = discount_percent

            # ---------------------------------------------------------------
            # Step 9: Financial impact is the undisclosed discount
            # ---------------------------------------------------------------
            financial_impact: Decimal = undisclosed_discount

            # ---------------------------------------------------------------
            # Step 10: Build ground truth data
            # ---------------------------------------------------------------
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[
                    "has_side_agreement",
                    "undisclosed_discount_amount",
                    "effective_price",
                    "side_agreement_documented",
                ],
                original_values={
                    "has_side_agreement": False,
                    "undisclosed_discount_amount": "0.00",
                    "effective_price": str(invoice_amount),
                    "side_agreement_documented": True,
                },
                modified_values={
                    "has_side_agreement": True,
                    "undisclosed_discount_amount": str(undisclosed_discount),
                    "effective_price": str(effective_price),
                    "side_agreement_type": agreement_type,
                    "side_agreement_documented": False,
                    "discount_percent": discount_percent,
                },
                financial_impact=financial_impact,
                description=(
                    f"Undisclosed side agreement with {discount_percent}% "
                    f"discount \u2014 effective price ${effective_price} vs "
                    f"recorded ${invoice_amount}"
                ),
                extra_metadata={
                    "side_agreement_type": agreement_type,
                    "discount_percent": discount_percent,
                    "recorded_amount": str(invoice_amount),
                    "effective_price": str(effective_price),
                    "undisclosed_discount": str(undisclosed_discount),
                },
            )

            # ---------------------------------------------------------------
            # Step 11: Log the injection
            # ---------------------------------------------------------------
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

            logger.debug(
                "side_agreement_injected",
                service_name="transactions",
                component="SideAgreements",
                type_code=self.type_code,
                transaction_id=transaction_id,
                discount_percent=discount_percent,
                agreement_type=agreement_type,
                invoice_amount=str(invoice_amount),
                effective_price=str(effective_price),
                undisclosed_discount=str(undisclosed_discount),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Failed to inject side agreement discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
