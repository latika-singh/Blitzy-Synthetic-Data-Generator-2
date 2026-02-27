"""P2P-004: Missing Purchase Order discrepancy type implementation.

Removes or nullifies the purchase order reference from a vendor invoice,
creating an invoice that cannot be matched to any PO in the three-way
matching process. This is a common control violation in procurement.

In a normal Procure-to-Pay cycle, every vendor invoice MUST reference a
valid purchase order for three-way matching (PO ↔ Receipt ↔ Invoice).
When the PO reference is missing, the AP system cannot perform the
standard matching validation, leaving the invoice amount entirely
unvalidated against any authorised procurement commitment.

Configurable Parameters:
    None — this discrepancy has no configurable parameters. The PO reference
    is simply removed/nullified across all header and line-level fields.

Detection Method: document_linkage
    Detected when the AP system attempts to match the invoice to a PO
    and finds no valid PO reference or the reference is null/empty.

Difficulty: easy
    Document linkage validation is a basic ERP control check.

Financial Impact:
    The full invoice amount represents the unvalidated exposure, since
    no PO commitment can be verified against the invoice.

References:
    - AAP Section 0.5.1 Group 5: P2P-004 Missing Purchase Order
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - app/orchestration/transaction_orchestrator.py: REQUIRED_ARTIFACTS
      for vendor_invoice includes three_way_match which requires PO linkage
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
# Constants — PO reference field names to search for and nullify
# ---------------------------------------------------------------------------

# Header-level PO reference field names in order of precedence.
# Multiple names are checked because different transaction generators
# may use different naming conventions for the PO reference.
_HEADER_PO_FIELDS: Tuple[str, ...] = (
    "po_number",
    "po_id",
    "purchase_order_id",
    "po_reference",
    "po_link",
    "purchase_order_number",
    "purchase_order_reference",
)

# Line-level PO reference field names to nullify on each invoice line.
_LINE_PO_FIELDS: Tuple[str, ...] = (
    "po_line_number",
    "po_line_id",
    "purchase_order_line_id",
    "po_line_reference",
)

# Possible field names for invoice line items within the transaction dict.
_LINE_ITEM_KEYS: Tuple[str, ...] = (
    "lines",
    "invoice_lines",
    "line_items",
    "items",
)


class MissingPO(BaseDiscrepancy):
    """P2P-004: Missing Purchase Order discrepancy.

    Removes or nullifies all PO reference fields from a vendor invoice
    transaction at both the header and line-item level.  This simulates
    an invoice arriving without any purchase order linkage, which is a
    fundamental control violation in Procure-to-Pay workflows.

    The discrepancy has no configurable parameters — the PO reference
    fields are simply set to ``None`` on every occurrence found in the
    transaction data dictionary.

    Attributes:
        type_code: ``"P2P-004"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"Missing Purchase Order"``
        description: Vendor invoice without a valid purchase order reference
        detection_method: ``"document_linkage"``
        PARAMETER_BOUNDS: Empty dict (no configurable parameters)
    """

    # ------------------------------------------------------------------
    # Required ClassVar attributes (overrides from BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-004"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Missing Purchase Order"
    description: ClassVar[str] = (
        "Vendor invoice without a valid purchase order reference"
    )
    detection_method: ClassVar[str] = "document_linkage"

    # No configurable parameters for this discrepancy type.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a Missing PO discrepancy into a vendor invoice transaction.

        Removes or nullifies all PO reference fields from the transaction
        at both the header and line-item level.  The full invoice amount
        is reported as the financial impact since the entire invoice is
        unvalidated without a PO commitment.

        Args:
            transaction: The original vendor invoice transaction data
                dictionary.  Expected to contain fields such as
                ``"po_number"``, ``"po_id"``, ``"total_amount"``, and
                optionally ``"lines"`` with per-line PO references.
            params: Injection parameters — ignored for this discrepancy
                type (no configurable parameters).  Passed through
                ``_validate_params()`` as a no-op validation.
            rng: A seeded :class:`random.Random` instance for
                deterministic behaviour.  Not used by this discrepancy
                type (no random choices needed) but accepted to satisfy
                the ``BaseDiscrepancy.inject()`` contract.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``:

            - **modified_transaction**: Deep copy of the input with all
              PO reference fields set to ``None``.
            - **ground_truth_data**: Dictionary describing the injection
              for the ``GroundTruthGenerator``, including affected fields,
              original/modified values, financial impact, and metadata.

        Raises:
            DiscrepancyInjectionError: If the transaction data is
                malformed in a way that prevents safe injection (e.g.,
                the transaction object is ``None``).
        """
        try:
            # Step 1 — Deep copy the transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # Step 2 — Validate params (no-op for empty PARAMETER_BOUNDS)
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # Step 3 — Identify and store original PO reference values
            # at the header level
            original_values: Dict[str, Any] = {}
            affected_fields: List[str] = []
            modified_values: Dict[str, Any] = {}

            for field_name in _HEADER_PO_FIELDS:
                if field_name in modified:
                    original_values[field_name] = modified[field_name]
                    affected_fields.append(field_name)
                    # Step 4 — Nullify the PO reference field
                    modified[field_name] = None
                    modified_values[field_name] = None

            # Step 5 — Clear PO references at the line-item level
            lines: Optional[List[Dict[str, Any]]] = None
            lines_key_used: Optional[str] = None
            for key in _LINE_ITEM_KEYS:
                if key in modified and isinstance(modified[key], list):
                    lines = modified[key]
                    lines_key_used = key
                    break

            line_fields_cleared: int = 0
            if lines is not None:
                for line_idx, line in enumerate(lines):
                    if not isinstance(line, dict):
                        continue
                    for line_field in _LINE_PO_FIELDS:
                        if line_field in line:
                            line_key = f"lines[{line_idx}].{line_field}"
                            original_values[line_key] = line[line_field]
                            affected_fields.append(line_key)
                            line[line_field] = None
                            modified_values[line_key] = None
                            line_fields_cleared += 1

            # Log a warning if no PO reference fields were found at all.
            # This is not an error — the transaction simply had no PO
            # references to remove — but the discrepancy is still valid
            # because the *absence* of PO references is the discrepancy.
            total_fields_cleared = len(
                [f for f in affected_fields if not f.startswith("lines[")]
            ) + line_fields_cleared

            if total_fields_cleared == 0:
                logger.warning(
                    "no_po_reference_fields_found",
                    service_name="transactions",
                    component="MissingPO",
                    type_code=self.type_code,
                    transaction_id=str(
                        modified.get("transaction_id", "unknown")
                    ),
                    message=(
                        "Transaction has no identifiable PO reference "
                        "fields; discrepancy still applied as missing-PO "
                        "indicator"
                    ),
                )
                # Even with no fields to clear, mark the invoice as
                # having no PO by setting canonical fields to None so
                # downstream systems can detect the missing linkage.
                for default_field in ("po_number", "po_id"):
                    if default_field not in modified:
                        modified[default_field] = None
                        affected_fields.append(default_field)
                        original_values[default_field] = None
                        modified_values[default_field] = None

            # Step 6 — Calculate financial impact.
            # The full invoice total_amount represents the unvalidated
            # exposure.  If no total_amount is present, default to zero.
            raw_amount = modified.get(
                "total_amount",
                modified.get("amount", modified.get("invoice_amount", 0)),
            )
            try:
                financial_impact = abs(Decimal(str(raw_amount)))
            except Exception:
                financial_impact = Decimal("0")

            # Extract invoice identifiers for the description.
            invoice_number = str(
                modified.get(
                    "invoice_number",
                    modified.get(
                        "invoice_id",
                        modified.get("transaction_id", "unknown"),
                    ),
                )
            )

            # Step 7 — Create ground truth data
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=(
                    f"Invoice {invoice_number} has no purchase order "
                    f"reference — full amount ${financial_impact} is "
                    f"unvalidated exposure"
                ),
                extra_metadata={
                    "header_fields_cleared": len(
                        [
                            f
                            for f in affected_fields
                            if not f.startswith("lines[")
                        ]
                    ),
                    "line_fields_cleared": line_fields_cleared,
                    "total_fields_cleared": total_fields_cleared,
                    "invoice_number": invoice_number,
                    "lines_key_used": lines_key_used,
                    "has_line_items": lines is not None,
                },
            )

            # Step 8 — Log the successful injection
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
            # Re-raise known discrepancy injection errors as-is
            raise
        except Exception as exc:
            # Wrap any unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Failed to inject P2P-004 (Missing PO) discrepancy: "
                f"{exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                        if isinstance(transaction, dict)
                        else "unknown"
                    ),
                    "category": self.category,
                    "difficulty": self.difficulty,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
