"""P2P-015: Ghost Expense discrepancy type implementation.

Removes or nullifies all supporting documentation references from a vendor
invoice or expense transaction, creating an expense with no verifiable paper
trail. Ghost expenses lack purchase orders, delivery receipts, or other
supporting documents that would normally validate the expense.

Configurable Parameters:
    None — all supporting documentation references are simply removed.

Detection Method: document_review
    Detected during periodic expense reviews when no supporting documents
    can be found for the transaction.

Difficulty: medium
    Requires document completeness review, which is often a manual process.

Financial Impact: Full transaction amount (unverifiable expense).

References:
    - AAP Section 0.5.1 Group 5: P2P-015 Ghost Expense
"""

from __future__ import annotations

import copy
import random
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
__all__ = ["GhostExpense"]


class GhostExpense(BaseDiscrepancy):
    """P2P-015: Ghost Expense discrepancy (no supporting documentation).

    Removes or nullifies all supporting documentation references from
    an expense or vendor invoice transaction.  Ghost expenses are a
    classic procurement fraud technique where an insider creates an
    expense record without any supporting documentation — no purchase
    order, no delivery receipt, no contract, no attachments.

    The financial impact equals the full transaction amount because the
    entire expense is unverifiable.  Even when the transaction has no
    documentation fields to clear, the metadata flags are still set to
    mark the transaction as lacking documentation, making the
    discrepancy injection valid regardless of initial field presence.

    Attributes:
        type_code: ``"P2P-015"``
        category: ``"p2p"``
        difficulty: ``"medium"``
        name: ``"Ghost Expense"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"document_review"``
        PARAMETER_BOUNDS: Empty dict — no configurable parameters.
        _DOCUMENTATION_FIELDS: List of field names representing
            supporting documentation references that should normally
            be present on a valid expense.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy contract
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-015"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Ghost Expense"
    description: ClassVar[str] = (
        "Expense or invoice with no supporting documentation"
    )
    detection_method: ClassVar[str] = "document_review"

    # No configurable parameters — all documentation is simply removed.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Documentation fields that should be present for valid expenses.
    # Each field name represents a reference to supporting documentation
    # that the inject() method will nullify or remove.
    # ------------------------------------------------------------------
    _DOCUMENTATION_FIELDS: ClassVar[List[str]] = [
        "po_number",
        "po_reference",
        "receipt_number",
        "goods_receipt_id",
        "delivery_note",
        "contract_reference",
        "supporting_documents",
        "attachment_ids",
        "receipt_confirmed",
        "three_way_match_status",
    ]

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a ghost expense discrepancy into the transaction.

        Creates an expense with no verifiable paper trail by
        systematically removing or nullifying every supporting
        documentation reference found in the transaction data.

        Steps:
            1. Deep-copy the transaction to preserve the original.
            2. Validate parameters (no-op since PARAMETER_BOUNDS is
               empty, but upholds the contract).
            3. Record original values of all documentation fields that
               exist in the transaction.
            4. Set each documentation field to ``None``.
            5. Clear nested ``documents`` and ``attachments`` lists.
            6. Set explicit metadata flags:
               ``has_supporting_documents = False``,
               ``documentation_status = "missing"``.
            7. Calculate financial impact as the full transaction amount.
            8. Build ground truth data via
               :meth:`_create_ground_truth_data`.
            9. Log the injection event.
           10. Return ``(modified_transaction, ground_truth_data)``.

        Args:
            transaction: The original transaction data dictionary.
            params: Injection parameters (ignored — no parameters for
                this discrepancy type).
            rng: Seeded :class:`random.Random` instance for deterministic
                behaviour (not actively used by this type since there are
                no random decisions, but accepted per the contract).

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If injection fails due to
                unexpected data issues.
        """
        try:
            # ----------------------------------------------------------
            # 1. Deep-copy the transaction
            # ----------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ----------------------------------------------------------
            # 2. Validate parameters (no-op for empty bounds)
            # ----------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # ----------------------------------------------------------
            # 3. Record original values and nullify documentation fields
            # ----------------------------------------------------------
            original_values: Dict[str, Any] = {}
            modified_values: Dict[str, Any] = {}
            cleared_fields: List[str] = []

            for field_name in self._DOCUMENTATION_FIELDS:
                if field_name in modified:
                    original_values[field_name] = modified[field_name]
                    modified[field_name] = None
                    modified_values[field_name] = None
                    cleared_fields.append(field_name)

            # ----------------------------------------------------------
            # 4. Clear nested documents / attachments lists
            # ----------------------------------------------------------
            _NESTED_LIST_FIELDS: List[str] = [
                "documents",
                "attachments",
                "supporting_docs",
                "linked_documents",
            ]
            for nested_field in _NESTED_LIST_FIELDS:
                if nested_field in modified:
                    original_val = modified[nested_field]
                    if original_val is not None:
                        original_values[nested_field] = original_val
                        modified[nested_field] = []
                        modified_values[nested_field] = []
                        cleared_fields.append(nested_field)

            # ----------------------------------------------------------
            # 5. Set explicit metadata flags
            # ----------------------------------------------------------
            # Record originals for the metadata flags if they existed.
            if "has_supporting_documents" in modified:
                original_values["has_supporting_documents"] = modified[
                    "has_supporting_documents"
                ]
            if "documentation_status" in modified:
                original_values["documentation_status"] = modified[
                    "documentation_status"
                ]

            modified["has_supporting_documents"] = False
            modified["documentation_status"] = "missing"
            modified_values["has_supporting_documents"] = False
            modified_values["documentation_status"] = "missing"

            # Always include these metadata flags in affected fields.
            if "has_supporting_documents" not in cleared_fields:
                cleared_fields.append("has_supporting_documents")
            if "documentation_status" not in cleared_fields:
                cleared_fields.append("documentation_status")

            # ----------------------------------------------------------
            # 6. Warn if no documentation fields were present
            # ----------------------------------------------------------
            doc_field_count = sum(
                1
                for f in cleared_fields
                if f not in ("has_supporting_documents", "documentation_status")
            )
            if doc_field_count == 0:
                logger.warning(
                    "ghost_expense_no_doc_fields",
                    service_name="transactions",
                    component="GhostExpense",
                    type_code=self.type_code,
                    message=(
                        "Transaction had no documentation fields to clear; "
                        "metadata flags still set"
                    ),
                    transaction_id=str(
                        modified.get("transaction_id", "unknown")
                    ),
                )

            # ----------------------------------------------------------
            # 7. Calculate financial impact — full transaction amount
            # ----------------------------------------------------------
            raw_amount = modified.get(
                "total_amount",
                modified.get(
                    "invoice_amount",
                    modified.get("amount", Decimal("0")),
                ),
            )
            # Ensure Decimal — coerce from str/int/float safely.
            try:
                financial_impact = Decimal(str(raw_amount))
            except Exception:
                financial_impact = Decimal("0")

            # ----------------------------------------------------------
            # 8. Build human-readable description
            # ----------------------------------------------------------
            invoice_number = str(
                modified.get(
                    "invoice_number",
                    modified.get("transaction_id", "unknown"),
                )
            )
            description_text = (
                f"Ghost expense: {doc_field_count} supporting document "
                f"reference(s) removed from invoice {invoice_number}"
            )

            # ----------------------------------------------------------
            # 9. Create ground truth data
            # ----------------------------------------------------------
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=cleared_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=description_text,
                extra_metadata={
                    "cleared_field_count": doc_field_count,
                    "documentation_status": "missing",
                    "cleared_fields": [
                        f
                        for f in cleared_fields
                        if f
                        not in (
                            "has_supporting_documents",
                            "documentation_status",
                        )
                    ],
                },
            )

            # ----------------------------------------------------------
            # 10. Log the injection
            # ----------------------------------------------------------
            self._log_injection(
                transaction_id=str(
                    modified.get("transaction_id", "unknown")
                ),
                financial_impact=financial_impact,
                context={
                    k: v
                    for k, v in modified.items()
                    if k in ("simulation_id", "trace_id")
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise domain exceptions directly
            raise

        except Exception as exc:
            raise DiscrepancyInjectionError(
                f"Failed to inject ghost expense discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error": str(exc),
                },
            ) from exc
