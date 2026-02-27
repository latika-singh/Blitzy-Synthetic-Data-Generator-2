"""O2C-002: Invoice Without Shipment discrepancy type.

Simulates a customer invoice created without a corresponding shipment record.
In normal O2C flow, goods must be shipped before an invoice is generated.
This discrepancy represents a control weakness where revenue is recognized
before delivery.

Detection Method: ``document_linkage`` — verifying that every customer invoice
has a linked shipment record in the shipment table.

Difficulty: ``easy`` — straightforward document linkage check.

Parameter Bounds: None — this discrepancy has no configurable parameters.

References:
    - AAP Section 0.5.1 Group 5: O2C-002 Invoice Without Shipment
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Financial Integrity Rules
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Tuple

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
__all__ = ["InvoiceWithoutShipment"]


class InvoiceWithoutShipment(BaseDiscrepancy):
    """O2C-002: Invoice Without Shipment discrepancy.

    Simulates the creation of a customer invoice when no shipment has been
    recorded for the associated sales order.  This is a common control
    weakness in Order-to-Cash processes where revenue is recognized before
    physical delivery of goods.

    In a properly functioning O2C cycle, the sequence is:

        Sales Order → Shipment → Customer Invoice → Customer Payment

    This discrepancy removes the shipment linkage from an invoice, making
    it appear that goods were invoiced without ever being shipped.

    The financial impact equals the full invoice amount because the entire
    revenue has been recognized without a corresponding delivery obligation
    being satisfied.

    Class Attributes:
        type_code: ``"O2C-002"`` — unique discrepancy type code.
        category: ``"o2c"`` — Order-to-Cash category.
        difficulty: ``"easy"`` — straightforward document linkage check.
        name: Human-readable name for reports and ground truth records.
        description: Detailed description of the discrepancy scenario.
        detection_method: ``"document_linkage"`` — expected detection
            approach for audit systems.
        PARAMETER_BOUNDS: Empty dictionary — this discrepancy type has
            no configurable injection parameters.
    """

    # ------------------------------------------------------------------
    # Class-level attributes (required by BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-002"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Invoice Without Shipment"
    description: ClassVar[str] = (
        "Customer invoice created without a corresponding shipment record, "
        "indicating revenue recognition before delivery."
    )
    detection_method: ClassVar[str] = "document_linkage"

    # No configurable parameters for this discrepancy type
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an invoice-without-shipment discrepancy into a transaction.

        Removes or nullifies all shipment reference fields from the
        transaction data, simulating an invoice that was created before
        goods were shipped.  The original invoice data (invoice number,
        amount, customer, line items) is preserved.

        Steps:
            1. Deep-copy the transaction via :meth:`_copy_transaction`.
            2. Capture original shipment reference values for ground truth.
            3. Nullify ``shipment_id``, ``shipment_date``, and
               ``tracking_number`` (if present).
            4. Mark ``shipment_verified`` as ``False``.
            5. Calculate ``financial_impact`` as the full invoice amount
               (revenue recognised without delivery) using :class:`Decimal`.
            6. Build ground truth data via :meth:`_create_ground_truth_data`.
            7. Log the injection via :meth:`_log_injection`.
            8. Return ``(modified_transaction, ground_truth_data)``.

        Args:
            transaction: The original transaction data dictionary.  Expected
                keys include ``shipment_id``, ``shipment_date``, and
                optionally ``tracking_number``.  Invoice amount is read
                from ``invoice_amount``, ``total_amount``, or ``amount``
                (checked in that order).
            params: Injection parameters — ignored for this discrepancy
                type since ``PARAMETER_BOUNDS`` is empty.
            rng: A seeded :class:`random.Random` instance for deterministic
                behaviour.  This discrepancy has minimal randomness but
                the parameter is part of the :class:`BaseDiscrepancy`
                contract.

        Returns:
            A tuple of two dictionaries:

            - **modified_transaction** — Transaction data with shipment
              references nullified.
            - **ground_truth_data** — Dictionary for
              :class:`GroundTruthGenerator` containing affected fields,
              original/modified values, detection method, financial
              impact, and a human-readable description.

        Raises:
            DiscrepancyInjectionError: If injection fails due to missing
                transaction data or unexpected errors.
        """
        try:
            # ---- Step 1: Deep-copy transaction to preserve original ----
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ---- Step 2: Capture original shipment reference values ----
            original_shipment_id: Any = modified.get("shipment_id")
            original_shipment_date: Any = modified.get("shipment_date")
            original_tracking_number: Any = modified.get("tracking_number")

            # ---- Step 3: Nullify shipment references -------------------
            modified["shipment_id"] = None
            modified["shipment_date"] = None

            # Nullify tracking number if present in the transaction data
            if "tracking_number" in modified:
                modified["tracking_number"] = None

            # ---- Step 4: Mark invoice as unverified against shipment ---
            modified["shipment_verified"] = False

            # ---- Step 5: Calculate financial impact --------------------
            # The full invoice amount represents revenue recognized
            # without a corresponding delivery obligation.  Check multiple
            # candidate keys in priority order.
            raw_amount: Any = modified.get(
                "invoice_amount",
                modified.get(
                    "total_amount",
                    modified.get("amount", Decimal("0.00")),
                ),
            )
            financial_impact: Decimal = Decimal(str(raw_amount))

            # ---- Determine transaction ID for logging -----------------
            transaction_id: str = str(
                modified.get(
                    "transaction_id",
                    modified.get(
                        "invoice_id",
                        modified.get("id", "unknown"),
                    ),
                )
            )

            # ---- Step 6: Build affected-fields list --------------------
            affected_fields: List[str] = ["shipment_id", "shipment_date"]
            if original_tracking_number is not None:
                affected_fields.append("tracking_number")

            # ---- Build original values dictionary ----------------------
            original_values: Dict[str, Any] = {
                "shipment_id": (
                    str(original_shipment_id)
                    if original_shipment_id is not None
                    else None
                ),
                "shipment_date": (
                    original_shipment_date.isoformat()
                    if hasattr(original_shipment_date, "isoformat")
                    else (
                        str(original_shipment_date)
                        if original_shipment_date is not None
                        else None
                    )
                ),
            }
            if original_tracking_number is not None:
                original_values["tracking_number"] = str(
                    original_tracking_number
                )

            # ---- Build modified values dictionary ----------------------
            modified_values: Dict[str, Any] = {
                "shipment_id": None,
                "shipment_date": None,
                "shipment_verified": False,
            }
            if original_tracking_number is not None:
                modified_values["tracking_number"] = None

            # ---- Create ground truth data via helper -------------------
            ground_truth_data: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=(
                    "Customer invoice created without corresponding "
                    "shipment record"
                ),
                extra_metadata={
                    "original_shipment_id": (
                        str(original_shipment_id)
                        if original_shipment_id is not None
                        else None
                    ),
                    "invoice_amount": str(financial_impact),
                    "revenue_recognized_without_delivery": True,
                },
            )

            # ---- Step 7: Log the successful injection ------------------
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context=transaction.get("context")
                if isinstance(transaction.get("context"), dict)
                else None,
            )

            logger.debug(
                "invoice_without_shipment_injected",
                service_name="transactions",
                component="InvoiceWithoutShipment",
                type_code=self.type_code,
                transaction_id=transaction_id,
                financial_impact=str(financial_impact),
                original_shipment_id=(
                    str(original_shipment_id)
                    if original_shipment_id is not None
                    else None
                ),
            )

            # ---- Step 8: Return modified transaction and ground truth ---
            return modified, ground_truth_data

        except DiscrepancyInjectionError:
            # Re-raise already-structured injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError per
            # AAP Section 0.7.4 error handling conventions
            raise DiscrepancyInjectionError(
                f"Failed to inject {self.type_code} discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "category": self.category,
                    "difficulty": self.difficulty,
                    "error": str(exc),
                    "transaction_id": str(
                        transaction.get(
                            "transaction_id",
                            transaction.get("invoice_id", "unknown"),
                        )
                    ),
                },
            ) from exc
