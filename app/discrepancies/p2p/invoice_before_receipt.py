"""P2P-006: Invoice Before Receipt discrepancy type implementation.

Modifies the invoice date to occur before the goods receipt date, creating
a temporal sequence violation. In a normal P2P cycle, the sequence is:
PO → Goods Receipt → Vendor Invoice. An invoice dated before its receipt
indicates either premature billing or potentially fraudulent timing.

Configurable Parameters:
    days_before (int): Number of days the invoice date should precede the receipt.
        Bounds: 1–30 days. Default: 3 days.

Detection Method: date_sequence
    Detected by comparing invoice_date < receipt_date.

Difficulty: easy

Financial Impact: Full invoice amount (temporal integrity violation).

References:
    - AAP Section 0.5.1 Group 5: P2P-006 Invoice Before Receipt
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - AAP Section 0.7.1: Deterministic reproducibility (seeded RNG)
"""

from __future__ import annotations

import copy
import random
from datetime import timedelta
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["InvoiceBeforeReceipt"]

# ---------------------------------------------------------------------------
# Receipt date field name candidates — ordered by priority.
# The inject() method will search the transaction dict for the first field
# that is present and non-None.
# ---------------------------------------------------------------------------
_RECEIPT_DATE_FIELDS: Tuple[str, ...] = (
    "receipt_date",
    "goods_receipt_date",
    "gr_date",
)

# ---------------------------------------------------------------------------
# Invoice date field name candidates — ordered by priority.
# ---------------------------------------------------------------------------
_INVOICE_DATE_FIELDS: Tuple[str, ...] = (
    "invoice_date",
    "vendor_invoice_date",
    "inv_date",
)


class InvoiceBeforeReceipt(BaseDiscrepancy):
    """P2P-006: Invoice Before Receipt discrepancy.

    Modifies the invoice date to precede the goods receipt date, creating a
    temporal sequence violation.  The ``days_before`` parameter controls how
    many days *before* the receipt date the invoice will be dated.

    In legitimate procurement, the vendor invoice is always dated on or after
    the goods receipt date.  An invoice dated before receipt indicates either
    premature billing (vendor sent invoice before delivery) or potentially
    fraudulent timing (backdated invoice to accelerate payment).

    Class Attributes:
        type_code: ``"P2P-006"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"Invoice Before Receipt"``
        description: Human-readable summary of the temporal sequence violation.
        detection_method: ``"date_sequence"``
        PARAMETER_BOUNDS: Single-parameter bounds for ``days_before``
            (min=1, max=30, default=3).

    Usage::

        injector = InvoiceBeforeReceipt()
        modified_txn, ground_truth = injector.inject(
            transaction={"invoice_date": date(2024, 3, 15),
                         "receipt_date": date(2024, 3, 10),
                         "total_amount": Decimal("5000.00"),
                         "transaction_id": "INV-2024-0001"},
            params={"days_before": 5},
            rng=random.Random(42),
        )
        # modified_txn["invoice_date"] == date(2024, 3, 5)  (5 days before receipt)
    """

    # ------------------------------------------------------------------
    # Class-level attributes — MUST match schema exports exactly
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-006"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Invoice Before Receipt"
    description: ClassVar[str] = (
        "Invoice date precedes goods receipt date "
        "(temporal sequence violation)"
    )
    detection_method: ClassVar[str] = "date_sequence"

    # ------------------------------------------------------------------
    # Parameter bounds for days_before (1–30 days, default 3)
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "days_before": {
            "min": 1,
            "max": 30,
            "type": "int",
            "default": 3,
        },
    }

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject P2P-006 discrepancy: set invoice date before receipt date.

        Creates a temporal sequence violation by setting the invoice date to
        ``receipt_date - timedelta(days=days_before)``.  The original invoice
        date is preserved in the ground truth record for auditing.

        Args:
            transaction: Transaction data dictionary containing at least:
                - A receipt date field (one of ``receipt_date``,
                  ``goods_receipt_date``, or ``gr_date``).
                - An invoice date field (one of ``invoice_date``,
                  ``vendor_invoice_date``, or ``inv_date``).
                - ``total_amount`` (Decimal) — used for financial impact.
                - ``transaction_id`` (str) — used for logging.
            params: Injection parameters.  Recognised key:
                - ``days_before`` (int): Number of days the invoice should
                  precede the receipt.  Validated against
                  ``PARAMETER_BOUNDS`` (1–30, default 3).
            rng: Seeded :class:`random.Random` instance for deterministic
                behaviour.  Not used directly by this discrepancy type
                (the shift is fully deterministic from ``days_before``),
                but accepted per the ``inject()`` contract.

        Returns:
            Tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction is missing a
                receipt date field, an invoice date field, or the injection
                logic encounters an unrecoverable error.
        """
        try:
            # Step 1: Deep-copy the transaction to preserve the original.
            modified = self._copy_transaction(transaction)

            # Step 2: Validate injection parameters against bounds.
            validated_params = self._validate_params(params)

            # Step 3: Extract the validated days_before parameter.
            days_before: int = int(validated_params.get("days_before", 3))

            # Step 4: Locate the receipt date in the transaction.
            receipt_date = None
            receipt_date_field: Optional[str] = None
            for field_name in _RECEIPT_DATE_FIELDS:
                if field_name in modified and modified[field_name] is not None:
                    receipt_date = modified[field_name]
                    receipt_date_field = field_name
                    break

            if receipt_date is None:
                raise DiscrepancyInjectionError(
                    f"Transaction missing receipt date field "
                    f"(checked: {', '.join(_RECEIPT_DATE_FIELDS)})",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "checked_fields": list(_RECEIPT_DATE_FIELDS),
                    },
                )

            # Step 5: Locate the invoice date in the transaction.
            original_invoice_date = None
            invoice_date_field: Optional[str] = None
            for field_name in _INVOICE_DATE_FIELDS:
                if field_name in modified and modified[field_name] is not None:
                    original_invoice_date = modified[field_name]
                    invoice_date_field = field_name
                    break

            if original_invoice_date is None:
                raise DiscrepancyInjectionError(
                    f"Transaction missing invoice date field "
                    f"(checked: {', '.join(_INVOICE_DATE_FIELDS)})",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "checked_fields": list(_INVOICE_DATE_FIELDS),
                    },
                )

            # Step 6: Calculate the modified invoice date.
            # The invoice date is set to days_before days BEFORE the receipt,
            # guaranteeing invoice_date < receipt_date.
            modified_invoice_date = receipt_date - timedelta(days=days_before)

            # Apply the modification to the transaction copy.
            modified[invoice_date_field] = modified_invoice_date

            # Step 7: Determine financial impact — full invoice amount
            # represents the temporal integrity violation exposure.
            raw_total = transaction.get(
                "total_amount",
                transaction.get(
                    "invoice_amount",
                    transaction.get("amount", Decimal("0")),
                ),
            )
            financial_impact = Decimal(str(raw_total))

            # Step 8: Build the ground truth record.
            ground_truth = self._create_ground_truth_data(
                affected_fields=[invoice_date_field],
                original_values={
                    invoice_date_field: str(original_invoice_date),
                },
                modified_values={
                    invoice_date_field: str(modified_invoice_date),
                },
                financial_impact=financial_impact,
                description=(
                    f"Invoice dated {days_before} day(s) before goods receipt"
                ),
                extra_metadata={
                    "days_before": days_before,
                    "receipt_date": str(receipt_date),
                    "receipt_date_field": receipt_date_field,
                    "invoice_date_field": invoice_date_field,
                    "original_invoice_date": str(original_invoice_date),
                    "modified_invoice_date": str(modified_invoice_date),
                },
            )

            # Step 9: Log the successful injection.
            self._log_injection(
                transaction_id=str(
                    transaction.get("transaction_id", "unknown")
                ),
                financial_impact=financial_impact,
                context={
                    "simulation_id": transaction.get("simulation_id"),
                    "trace_id": transaction.get("trace_id"),
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise domain-specific errors without wrapping.
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError for
            # consistent upstream handling per AAP §0.7.4.
            logger.error(
                "invoice_before_receipt_injection_failed",
                service_name="transactions",
                component="InvoiceBeforeReceipt",
                type_code=self.type_code,
                error=str(exc),
                transaction_id=str(
                    transaction.get("transaction_id", "unknown")
                ),
            )
            raise DiscrepancyInjectionError(
                f"P2P-006 injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "root_cause": str(exc),
                },
            ) from exc
