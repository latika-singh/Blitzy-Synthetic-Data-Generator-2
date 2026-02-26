"""P2P-010: Payment Before Invoice Date discrepancy type implementation.

Modifies the payment date to occur before the invoice date, creating a
temporal anomaly. In a normal P2P cycle, the sequence is:
PO → Goods Receipt → Vendor Invoice → Vendor Payment. A payment dated
before its invoice indicates either premature payment or potentially
backdated transactions.

Configurable Parameters:
    days_before (int): Number of days the payment date should precede the invoice.
        Bounds: 1–30 days. Default: 3 days.

Detection Method: date_sequence
    Detected by comparing payment_date < invoice_date.

Difficulty: easy

Financial Impact: Full payment amount (temporal integrity violation).

References:
    - AAP Section 0.5.1 Group 5: P2P-010 Payment Before Invoice Date
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
__all__ = ["PaymentBeforeInvoice"]


class PaymentBeforeInvoice(BaseDiscrepancy):
    """P2P-010: Payment Before Invoice Date discrepancy.

    Injects a temporal anomaly by shifting the vendor payment date to occur
    *before* the associated invoice date.  This violates the expected P2P
    chronological ordering (PO → Receipt → Invoice → Payment) and may
    indicate premature payment release, backdated transactions, or system
    control circumvention.

    The ``days_before`` parameter controls how many calendar days the
    payment date should precede the invoice date.  The resulting payment
    date is computed as ``invoice_date - timedelta(days=days_before)``.

    Class Attributes:
        type_code: ``"P2P-010"`` — unique code for this discrepancy type.
        category: ``"p2p"`` — procure-to-pay domain.
        difficulty: ``"easy"`` — straightforward date comparison detection.
        name: Human-readable name.
        description: Detailed discrepancy description.
        detection_method: ``"date_sequence"`` — date ordering validation.
        PARAMETER_BOUNDS: Bounds for ``days_before`` (1–30, default 3).
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-010"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Payment Before Invoice Date"
    description: ClassVar[str] = (
        "Payment date precedes the vendor invoice date (temporal anomaly)"
    )
    detection_method: ClassVar[str] = "date_sequence"

    # ------------------------------------------------------------------
    # Parameter bounds for this discrepancy type
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
    # Candidate field names — tried in priority order
    # ------------------------------------------------------------------
    _INVOICE_DATE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "invoice_date",
        "vendor_invoice_date",
        "inv_date",
    )
    _PAYMENT_DATE_FIELDS: ClassVar[Tuple[str, ...]] = (
        "payment_date",
        "vendor_payment_date",
        "pay_date",
    )
    _AMOUNT_FIELDS: ClassVar[Tuple[str, ...]] = (
        "payment_amount",
        "total_amount",
        "amount",
        "pay_amount",
        "invoice_amount",
    )

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a Payment Before Invoice Date discrepancy.

        Modifies the payment date so that it precedes the invoice date by
        ``days_before`` calendar days, creating a temporal anomaly that is
        detectable via simple date-ordering checks.

        Args:
            transaction: Original transaction data dictionary.  Must contain
                at least an invoice date field and a payment date field.
                Recognised invoice date keys (tried in order):
                ``"invoice_date"``, ``"vendor_invoice_date"``, ``"inv_date"``.
                Recognised payment date keys (tried in order):
                ``"payment_date"``, ``"vendor_payment_date"``, ``"pay_date"``.
            params: Injection parameters — may contain ``"days_before"``
                (int, 1–30).  Missing or out-of-bounds values are handled
                by :meth:`_validate_params` (auto-adjusted or defaulted).
            rng: Seeded :class:`random.Random` instance for deterministic
                behavior.  Not directly used by this discrepancy (no
                random choices), but accepted for API consistency.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction is missing both
                invoice date and payment date fields, or if any
                unrecoverable error occurs during injection.
        """
        try:
            # ── Step 1: Deep-copy the transaction ─────────────────────
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ── Step 2: Validate params and extract days_before ───────
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )
            days_before: int = int(validated_params.get("days_before", 3))

            # ── Step 3: Locate invoice_date field ─────────────────────
            invoice_date_key: Optional[str] = None
            invoice_date: Any = None
            for candidate in self._INVOICE_DATE_FIELDS:
                if candidate in modified and modified[candidate] is not None:
                    invoice_date_key = candidate
                    invoice_date = modified[candidate]
                    break

            if invoice_date is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing invoice_date field required for "
                    "P2P-010 (Payment Before Invoice Date)",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                        "tried_fields": list(self._INVOICE_DATE_FIELDS),
                    },
                )

            # ── Step 4: Locate and store original payment_date ────────
            payment_date_key: Optional[str] = None
            original_payment_date: Any = None
            for candidate in self._PAYMENT_DATE_FIELDS:
                if candidate in modified and modified[candidate] is not None:
                    payment_date_key = candidate
                    original_payment_date = modified[candidate]
                    break

            if original_payment_date is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing payment_date field required for "
                    "P2P-010 (Payment Before Invoice Date)",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                        "tried_fields": list(self._PAYMENT_DATE_FIELDS),
                    },
                )

            # ── Step 5: Compute modified payment date ─────────────────
            # Payment date = invoice_date minus days_before
            # This ensures payment_date < invoice_date (the anomaly).
            modified_payment_date = invoice_date - timedelta(days=days_before)
            modified[payment_date_key] = modified_payment_date

            # ── Step 6: Calculate financial impact ────────────────────
            # Financial impact is the full payment amount — the entire
            # payment is considered a temporal integrity violation.
            financial_impact: Decimal = Decimal("0")
            for amount_key in self._AMOUNT_FIELDS:
                raw_amount = modified.get(amount_key)
                if raw_amount is not None:
                    try:
                        financial_impact = abs(Decimal(str(raw_amount)))
                    except Exception:
                        continue
                    break

            # ── Step 7: Build ground truth data ───────────────────────
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[payment_date_key],
                original_values={
                    payment_date_key: str(original_payment_date),
                },
                modified_values={
                    payment_date_key: str(modified_payment_date),
                },
                financial_impact=financial_impact,
                description=(
                    f"Payment dated {days_before} day(s) before invoice date"
                ),
                extra_metadata={
                    "days_before": days_before,
                    "invoice_date": str(invoice_date),
                    "invoice_date_field": invoice_date_key,
                    "payment_date_field": payment_date_key,
                    "original_payment_date": str(original_payment_date),
                    "modified_payment_date": str(modified_payment_date),
                },
            )

            # ── Step 8: Log the injection event ───────────────────────
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

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise injection-specific errors directly
            raise
        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            logger.error(
                "payment_before_invoice_injection_failed",
                service_name="transactions",
                component="PaymentBeforeInvoice",
                type_code=self.type_code,
                error=str(exc),
                transaction_id=str(
                    transaction.get("transaction_id", "unknown")
                ),
            )
            raise DiscrepancyInjectionError(
                f"Failed to inject P2P-010 discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "root_cause": str(exc),
                },
            ) from exc
