"""O2C-001: Duplicate Customer Invoice discrepancy type.

Simulates the creation of duplicate customer invoices — invoices that are
issued more than once for the same sales order or shipment, potentially
resulting in double-billing of the customer.

Duplicate customer invoices can be:
- **Exact duplicates**: Same invoice number submitted twice
- **Near-duplicates**: Different invoice numbers for the same order within
  a configurable time window (days_apart parameter: 1–90 days)

Detection Method: ``duplicate_check`` — comparing invoice numbers, customer IDs,
amounts, and dates within the configured proximity window.

Difficulty: ``easy`` — detectable through standard duplicate invoice reports.

Parameter Bounds:
    days_apart: int, range [1, 90] — maximum days between duplicate invoices

References:
    - AAP Section 0.5.1 Group 5: O2C-001 Duplicate Customer Invoice
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.1: Deterministic reproducibility, structlog, Decimal
"""

from __future__ import annotations

import copy
import random
from datetime import timedelta
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
__all__ = ["DuplicateCustomerInvoice"]


class DuplicateCustomerInvoice(BaseDiscrepancy):
    """O2C-001: Duplicate Customer Invoice discrepancy.

    Simulates duplication of a customer invoice within the Order-to-Cash
    flow.  The duplicate invoice retains the same customer, sales order,
    and line-item details as the original but receives a new invoice number
    and a date offset within the ``days_apart`` parameter window.

    The financial impact equals the full invoice amount because the
    duplicate represents a double-billing risk — the customer may be
    charged twice for the same goods or services.

    Class-level attributes required by :class:`BaseDiscrepancy`:
        - ``type_code``  = ``"O2C-001"``
        - ``category``   = ``"o2c"``
        - ``difficulty``  = ``"easy"``
        - ``name``       = ``"Duplicate Customer Invoice"``
        - ``description`` — human-readable description of the discrepancy
        - ``detection_method`` = ``"duplicate_check"``

    PARAMETER_BOUNDS:
        ``days_apart``: int in ``[1, 90]`` — maximum days between the
        original and duplicate invoice dates.  When not explicitly provided
        in ``params``, a random value within bounds is generated using the
        seeded ``rng`` instance.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — overriding BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-001"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Duplicate Customer Invoice"
    description: ClassVar[str] = (
        "Duplicate customer invoice issued for the same sales order or shipment, "
        "potentially resulting in double-billing."
    )
    detection_method: ClassVar[str] = "duplicate_check"

    # ------------------------------------------------------------------
    # Parameter bounds for injection
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "days_apart": {"min": 1, "max": 90, "type": "int"},
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
        """Inject a duplicate customer invoice discrepancy into *transaction*.

        Creates a near-duplicate of the original customer invoice by generating
        a new invoice number and offsetting the invoice date by a random number
        of days (within the ``days_apart`` parameter window).  The duplicate
        retains the same customer, sales order, line items, and monetary
        amounts as the original — this is the double-billing scenario.

        Args:
            transaction: The original customer invoice transaction data dict.
                Expected keys (all optional with safe fallbacks):

                - ``"transaction_id"`` or ``"invoice_id"`` — unique identifier
                - ``"invoice_number"`` — original invoice number string
                - ``"invoice_date"`` — original date (ISO 8601 str or date obj)
                - ``"total_amount"`` or ``"invoice_amount"`` — monetary total
                - ``"customer_id"`` — customer reference
                - ``"sales_order_id"`` — linked sales order reference
                - ``"lines"`` — list of line-item dicts (preserved as-is)

            params: Injection parameters.  Recognised keys:

                - ``"days_apart"`` (int, default generated via *rng*):
                  Maximum number of days between the original and the
                  duplicate invoice dates.  Will be validated and clamped
                  to ``[1, 90]`` if ``auto_adjust`` is enabled.

            rng: Seeded :class:`random.Random` instance for deterministic
                generation.  MUST use this — NEVER ``random.random()`` or
                ``random.choice()`` on the module-level RNG.

        Returns:
            A 2-tuple of ``(modified_transaction, ground_truth_data)``.

            **modified_transaction** contains a ``"duplicate_invoice"`` key
            with the full duplicate invoice record, plus a
            ``"has_duplicate"`` flag set to ``True``.

            **ground_truth_data** is suitable for
            :class:`GroundTruthGenerator` with all required fields.

        Raises:
            DiscrepancyInjectionError: If required transaction fields are
                missing or if internal injection logic fails.
        """
        try:
            # 1. Deep-copy the transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate and clamp injection parameters
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # 3. Determine days_apart — use provided value or generate random
            if "days_apart" in validated_params:
                max_days_apart: int = int(validated_params["days_apart"])
            else:
                max_days_apart = rng.randint(
                    self.PARAMETER_BOUNDS["days_apart"]["min"],
                    self.PARAMETER_BOUNDS["days_apart"]["max"],
                )

            # Generate the actual days offset (at least 1 day, up to max)
            actual_days_apart: int = rng.randint(1, max(1, max_days_apart))

            # 4. Extract original invoice fields with safe fallbacks
            original_invoice_number: str = str(
                modified.get("invoice_number", modified.get("invoice_id", "INV-UNKNOWN"))
            )
            original_invoice_date: Any = modified.get(
                "invoice_date", modified.get("date", None)
            )

            # Resolve the total amount — try multiple common key names
            raw_amount = modified.get(
                "total_amount",
                modified.get(
                    "invoice_amount",
                    modified.get("amount", Decimal("0.00")),
                ),
            )
            # Ensure we have a Decimal for financial precision
            if isinstance(raw_amount, Decimal):
                invoice_amount: Decimal = raw_amount
            else:
                invoice_amount = Decimal(str(raw_amount))

            transaction_id: str = str(
                modified.get(
                    "transaction_id",
                    modified.get("invoice_id", "unknown"),
                )
            )
            customer_id: str = str(modified.get("customer_id", "unknown"))
            sales_order_id: str = str(modified.get("sales_order_id", ""))

            # 5. Generate the duplicate invoice record
            duplicate_invoice_number: str = f"{original_invoice_number}-DUP"

            # Calculate the duplicate invoice date
            duplicate_invoice_date: Any
            original_date_str: str
            duplicate_date_str: str

            if original_invoice_date is not None:
                # Handle both string dates and date/datetime objects
                if hasattr(original_invoice_date, "isoformat"):
                    # It's a date or datetime object
                    duplicate_invoice_date = (
                        original_invoice_date + timedelta(days=actual_days_apart)
                    )
                    original_date_str = original_invoice_date.isoformat()
                    duplicate_date_str = duplicate_invoice_date.isoformat()
                else:
                    # It's a string — store offset info for audit
                    original_date_str = str(original_invoice_date)
                    duplicate_date_str = (
                        f"{original_date_str} + {actual_days_apart} days"
                    )
                    duplicate_invoice_date = duplicate_date_str
            else:
                original_date_str = "N/A"
                duplicate_date_str = f"original + {actual_days_apart} days"
                duplicate_invoice_date = duplicate_date_str

            # Build the duplicate invoice data record — mirrors original
            duplicate_record: Dict[str, Any] = {
                "invoice_number": duplicate_invoice_number,
                "invoice_date": duplicate_invoice_date,
                "customer_id": customer_id,
                "sales_order_id": sales_order_id,
                "total_amount": invoice_amount,
                "lines": copy.deepcopy(modified.get("lines", [])),
                "is_duplicate": True,
                "original_invoice_number": original_invoice_number,
                "days_after_original": actual_days_apart,
            }

            # 6. Inject the duplicate into the modified transaction
            modified["duplicate_invoice"] = duplicate_record
            modified["has_duplicate"] = True

            # 7. Financial impact is the full duplicate invoice amount
            #    (double-billing risk: customer may pay both invoices)
            financial_impact: Decimal = invoice_amount

            # 8. Build ground truth data via base class helper
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=["invoice_number", "invoice_date"],
                original_values={
                    "invoice_number": original_invoice_number,
                    "invoice_date": original_date_str,
                },
                modified_values={
                    "invoice_number": duplicate_invoice_number,
                    "invoice_date": duplicate_date_str,
                },
                financial_impact=financial_impact,
                description=(
                    f"Duplicate customer invoice created {actual_days_apart} "
                    f"days after original invoice {original_invoice_number}"
                ),
                extra_metadata={
                    "original_invoice_number": original_invoice_number,
                    "duplicate_invoice_number": duplicate_invoice_number,
                    "days_apart_max": max_days_apart,
                    "days_apart_actual": actual_days_apart,
                    "customer_id": customer_id,
                    "sales_order_id": sales_order_id,
                    "invoice_amount": str(invoice_amount),
                },
            )

            # 9. Log the successful injection
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "customer_id": customer_id,
                    "days_apart": actual_days_apart,
                    "original_invoice": original_invoice_number,
                    "duplicate_invoice": duplicate_invoice_number,
                },
            )

            logger.debug(
                "duplicate_customer_invoice_injected",
                service_name="transactions",
                component="DuplicateCustomerInvoice",
                transaction_id=transaction_id,
                original_invoice_number=original_invoice_number,
                duplicate_invoice_number=duplicate_invoice_number,
                days_apart=actual_days_apart,
                invoice_amount=str(invoice_amount),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors directly
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError per AAP §0.7.4
            error_transaction_id = str(
                transaction.get(
                    "transaction_id",
                    transaction.get("invoice_id", "unknown"),
                )
            )
            logger.error(
                "duplicate_customer_invoice_injection_failed",
                service_name="transactions",
                component="DuplicateCustomerInvoice",
                transaction_id=error_transaction_id,
                error=str(exc),
            )
            raise DiscrepancyInjectionError(
                f"Failed to inject duplicate customer invoice for "
                f"transaction {error_transaction_id}: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": error_transaction_id,
                    "category": self.category,
                    "difficulty": self.difficulty,
                    "error": str(exc),
                },
            ) from exc
