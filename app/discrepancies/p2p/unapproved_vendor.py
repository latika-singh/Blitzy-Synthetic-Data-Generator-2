"""P2P-011: Unapproved Vendor discrepancy type implementation.

Modifies the vendor reference on a transaction to point to a vendor that
is not in the approved vendor list. Purchasing from unapproved vendors
violates procurement controls and may indicate circumvention of the
vendor qualification process.

Configurable Parameters:
    None — the vendor is simply changed to one not in the approved list.

Detection Method: vendor_validation
    Detected by checking the vendor_id against the approved vendor master list.

Difficulty: medium
    Requires maintaining and checking against an approved vendor list,
    which may not be consistently enforced in all organizations.

Financial Impact: Full transaction amount (control violation exposure).

References:
    - AAP Section 0.5.1 Group 5: P2P-011 Vendor Not in Approved List
    - app/statistical/selection_models.py: Vendor selection patterns
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
__all__ = ["UnapprovedVendor"]

# ---------------------------------------------------------------------------
# Possible unapproved vendor status values for realistic injection
# ---------------------------------------------------------------------------
_UNAPPROVED_STATUSES: Tuple[str, ...] = (
    "unapproved",
    "inactive",
    "pending_approval",
    "suspended",
    "deactivated",
)

# ---------------------------------------------------------------------------
# Temporary vendor name prefixes for generating fake unapproved vendor names
# ---------------------------------------------------------------------------
_TEMP_VENDOR_PREFIXES: Tuple[str, ...] = (
    "Temp Vendor",
    "Unregistered Supplier",
    "Provisional Vendor",
    "Ad-Hoc Supplier",
    "Non-Qualified Vendor",
)


class UnapprovedVendor(BaseDiscrepancy):
    """P2P-011: Vendor Not in Approved List discrepancy.

    Modifies a transaction's vendor reference so that the vendor is no
    longer in the approved vendor list.  The injection performs three
    coordinated modifications:

    1. **Vendor ID replacement** — Replaces the ``vendor_id`` with a
       clearly-distinguishable unapproved vendor identifier prefixed
       with ``"UNAPP-"`` followed by a deterministic random suffix.

    2. **Vendor status change** — Sets ``vendor_status`` to one of
       several unapproved status values (``"unapproved"``,
       ``"inactive"``, ``"pending_approval"``, ``"suspended"``,
       ``"deactivated"``), selected deterministically via the seeded
       RNG.

    3. **Approval flag** — Sets ``is_approved`` to ``False`` to
       explicitly mark the vendor as unapproved in any boolean-flag
       checks.

    Additionally, when a ``vendor_name`` field exists, it is replaced
    with a temporary vendor name to maintain data consistency.

    Financial impact is the full transaction amount, representing the
    total control-violation exposure from procuring through an
    unauthorized vendor.

    ClassVar Attributes:
        type_code: ``"P2P-011"``
        category: ``"p2p"``
        difficulty: ``"medium"``
        name: ``"Unapproved Vendor"``
        description: Describes the vendor-list violation.
        detection_method: ``"vendor_validation"``
        PARAMETER_BOUNDS: Empty dict — no configurable parameters.
    """

    # ------------------------------------------------------------------
    # Class-level attributes (required by BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-011"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Unapproved Vendor"
    description: ClassVar[str] = (
        "Transaction references a vendor not in the approved vendor list"
    )
    detection_method: ClassVar[str] = "vendor_validation"

    # No configurable injection parameters for this discrepancy type.
    # The vendor is simply replaced with an unapproved one.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an unapproved-vendor discrepancy into the transaction.

        Replaces the vendor reference on the transaction with one that is
        not in the approved vendor list, simulating procurement from an
        unauthorized source.

        The method performs the following steps:

        1. Deep-copies the transaction to preserve the original.
        2. Validates injection parameters (no-op for this type since
           ``PARAMETER_BOUNDS`` is empty).
        3. Captures original vendor field values for the ground truth.
        4. Replaces vendor fields (``vendor_id``, ``vendor_name``,
           ``vendor_status``, ``is_approved``) with unapproved values.
        5. Calculates financial impact as the full transaction amount.
        6. Constructs the ground truth data record.
        7. Logs the injection event.

        Args:
            transaction: The original transaction data dictionary.  Must
                contain at least ``vendor_id`` and ``total_amount`` (or
                ``amount``) fields.
            params: Injection parameters — expected to be empty or
                contain context keys.  No parameter bounds are enforced
                for this type.
            rng: A seeded :class:`random.Random` instance for
                deterministic status and name selection.  CRITICAL: only
                this RNG must be used; never ``random.random()`` or
                module-level ``random``.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``
            where:

            - **modified_transaction** contains the vendor fields changed
              to reflect an unapproved vendor.
            - **ground_truth_data** contains all 12 standard ground truth
              fields documenting the injection.

        Raises:
            DiscrepancyInjectionError: If the transaction is missing
                a ``vendor_id`` field or has no identifiable transaction
                amount.
        """
        try:
            # ---------------------------------------------------------- #
            # Step 1: Deep copy the transaction to preserve the original
            # ---------------------------------------------------------- #
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ---------------------------------------------------------- #
            # Step 2: Validate injection parameters (no-op — empty bounds)
            # ---------------------------------------------------------- #
            validated_params: Dict[str, Any] = self._validate_params(params)

            # ---------------------------------------------------------- #
            # Step 3: Extract and validate required transaction fields
            # ---------------------------------------------------------- #
            original_vendor_id: Optional[str] = transaction.get("vendor_id")
            if original_vendor_id is None:
                raise DiscrepancyInjectionError(
                    f"Transaction missing required 'vendor_id' field for "
                    f"{self.type_code} injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "missing_field": "vendor_id",
                    },
                )

            # Resolve transaction amount — try multiple common field names
            raw_amount = (
                transaction.get("total_amount")
                or transaction.get("amount")
                or transaction.get("payment_amount")
                or transaction.get("invoice_amount")
                or Decimal("0")
            )
            total_amount: Decimal = Decimal(str(raw_amount))

            # ---------------------------------------------------------- #
            # Step 4: Capture original vendor field values
            # ---------------------------------------------------------- #
            original_vendor_name: Optional[str] = transaction.get("vendor_name")
            original_vendor_status: Optional[str] = transaction.get(
                "vendor_status"
            )
            original_is_approved: Optional[bool] = transaction.get(
                "is_approved"
            )

            # Build the map of fields that existed before injection
            original_values: Dict[str, Any] = {
                "vendor_id": original_vendor_id,
            }
            if original_vendor_name is not None:
                original_values["vendor_name"] = original_vendor_name
            if original_vendor_status is not None:
                original_values["vendor_status"] = original_vendor_status
            if original_is_approved is not None:
                original_values["is_approved"] = original_is_approved

            # ---------------------------------------------------------- #
            # Step 5: Apply unapproved-vendor modifications
            # ---------------------------------------------------------- #

            # 5a. Generate a new unapproved vendor ID with a distinctive
            #     prefix so it is clearly distinguishable from approved
            #     vendor IDs in downstream analysis.
            random_suffix: int = rng.randint(10000, 99999)
            new_vendor_id: str = f"UNAPP-{random_suffix}"
            modified["vendor_id"] = new_vendor_id

            # 5b. Select an unapproved vendor status deterministically
            new_status: str = rng.choice(_UNAPPROVED_STATUSES)
            modified["vendor_status"] = new_status

            # 5c. Explicitly mark the vendor as unapproved
            modified["is_approved"] = False

            # 5d. Replace the vendor name with a temporary name when
            #     the original transaction includes a vendor_name field
            if "vendor_name" in transaction or original_vendor_name is not None:
                prefix: str = rng.choice(_TEMP_VENDOR_PREFIXES)
                new_vendor_name: str = f"{prefix} - {random_suffix}"
                modified["vendor_name"] = new_vendor_name
            else:
                new_vendor_name = ""

            # ---------------------------------------------------------- #
            # Step 6: Build modified values map for ground truth
            # ---------------------------------------------------------- #
            modified_values: Dict[str, Any] = {
                "vendor_id": new_vendor_id,
                "vendor_status": new_status,
                "is_approved": False,
            }
            if new_vendor_name:
                modified_values["vendor_name"] = new_vendor_name

            # ---------------------------------------------------------- #
            # Step 7: Determine affected fields list
            # ---------------------------------------------------------- #
            affected_fields: List[str] = ["vendor_id", "vendor_status", "is_approved"]
            if new_vendor_name:
                affected_fields.append("vendor_name")

            # ---------------------------------------------------------- #
            # Step 8: Calculate financial impact — full transaction amount
            # ---------------------------------------------------------- #
            financial_impact: Decimal = abs(total_amount)

            # ---------------------------------------------------------- #
            # Step 9: Build the ground truth data record
            # ---------------------------------------------------------- #
            description: str = (
                f"Transaction uses unapproved vendor "
                f"(was: {original_vendor_id}, "
                f"now: {new_vendor_id}, "
                f"status: {new_status})"
            )

            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=description,
                extra_metadata={
                    "original_vendor_id": original_vendor_id,
                    "new_vendor_id": new_vendor_id,
                    "unapproved_status": new_status,
                    "original_vendor_name": original_vendor_name or "",
                    "new_vendor_name": new_vendor_name,
                },
            )

            # ---------------------------------------------------------- #
            # Step 10: Log the successful injection
            # ---------------------------------------------------------- #
            transaction_id: str = str(
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
            # Re-raise domain-specific errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError for
            # consistent handling by the DiscrepancyInjector caller.
            raise DiscrepancyInjectionError(
                f"Unexpected error during {self.type_code} injection: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
