"""CTL-003: Approval Limit Exceeded — simulates approval authority violation.

Implements the Approval Limit Exceeded discrepancy type that modifies transaction
data so that the approver's role has insufficient authority for the transaction
amount. For example, setting a Purchasing Manager as the approver on a $60K PO
when their authority limit is $25K.

This discrepancy leverages the approval threshold structure defined in
``config/workflows/approval_thresholds.yaml`` and enforced by
``app/orchestration/approval_system.py``:

    Purchase Orders:
        - <$5K: no approval needed
        - $5K-$25K: purchasing_manager
        - $25K-$100K: controller
        - >$100K: cfo

    Vendor Invoices:
        - <$10K: no approval needed
        - $10K-$50K: ap_manager
        - $50K-$100K: controller
        - >$100K: cfo

    Journal Entries:
        - <$50K: senior_accountant
        - >$50K: controller

Catalog Entry:
    Type Code: CTL-003
    Category: control
    Difficulty: easy
    Detection Method: approval_check
    Parameters: excess_percent (1-100) — how much the amount exceeds the
        approver's authority limit, expressed as a percentage.

References:
    - AAP Section 0.5.1 Group 5: CTL-003 Approval Limit Exceeded
    - AAP Section 0.7.5: Discrepancy Injection Rules (parameter bounds)
    - config/workflows/approval_thresholds.yaml: Threshold tier definitions
    - app/orchestration/approval_system.py: APPROVAL_THRESHOLDS
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
# Approval Authority Limits (matching APPROVAL_THRESHOLDS)
# ---------------------------------------------------------------------------
# Maximum approval authority per role per transaction type.
# Amounts use Decimal to follow AAP §0.7.2 financial integrity rules.
#
# These values mirror the thresholds defined in:
#   - config/workflows/approval_thresholds.yaml
#   - app/orchestration/approval_system.py (APPROVAL_THRESHOLDS)
#
# Each role's limit represents the maximum transaction amount it may approve.
# The CFO/controller-for-JE are effectively unlimited for MVP purposes.
ROLE_AUTHORITY_LIMITS: Dict[str, Dict[str, Decimal]] = {
    "purchase_order": {
        "purchasing_manager": Decimal("25000"),
        "controller": Decimal("100000"),
        "cfo": Decimal("999999999"),  # Effectively unlimited for MVP
    },
    "vendor_invoice": {
        "ap_manager": Decimal("50000"),
        "controller": Decimal("100000"),
        "cfo": Decimal("999999999"),
    },
    "journal_entry": {
        "senior_accountant": Decimal("50000"),
        "controller": Decimal("999999999"),
    },
}


# ---------------------------------------------------------------------------
# ApprovalLimitExceeded — CTL-003 discrepancy implementation
# ---------------------------------------------------------------------------


class ApprovalLimitExceeded(BaseDiscrepancy):
    """CTL-003: Approval Limit Exceeded.

    Injects an approval limit violation by modifying the transaction so that:
    - The amount exceeds the approver's authority limit by excess_percent, OR
    - The approved_by role is downgraded to a role with insufficient authority

    This is an 'easy' difficulty discrepancy because the detection method is
    straightforward: compare the transaction amount against the approver's
    authority limit from the threshold configuration.

    Injection Approaches:
        **Approach A — Increase Amount:**
            Increase the transaction amount to
            ``authority_limit * (1 + excess_percent / 100)``.
            For example, if the approver's limit is $25K and
            ``excess_percent`` is 20, the new amount becomes $30K.

        **Approach B — Downgrade Approver:**
            Change the ``approved_by_role`` to a lower-authority role
            whose limit is below the current transaction amount.
            For example, on a $60K PO, change the controller ($100K
            limit) to purchasing_manager ($25K limit).

        When both approaches are available, one is selected at random
        using the provided seeded ``rng`` instance for deterministic
        reproducibility.

    Attributes:
        type_code: ``"CTL-003"``
        category: ``"control"``
        difficulty: ``"easy"``
        name: ``"Approval Limit Exceeded"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"approval_check"``
    """

    # ------------------------------------------------------------------
    # ClassVar attributes — override BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "CTL-003"
    category: ClassVar[str] = "control"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Approval Limit Exceeded"
    description: ClassVar[str] = (
        "Transaction approved by a role whose approval authority is "
        "below the transaction amount"
    )
    detection_method: ClassVar[str] = "approval_check"

    # Parameter bounds — excess_percent controls how much the amount
    # exceeds the approver's authority limit.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "excess_percent": {"min": 1, "max": 100, "type": "int"},
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
        """Inject an approval limit exceeded violation.

        Strategy: Either increase the transaction amount beyond the approver's
        authority limit, or downgrade the approver to a lower-authority role.
        The approach is selected based on data availability and, when both are
        possible, by deterministic random choice via ``rng``.

        Args:
            transaction: Transaction data dictionary.  Required keys:
                ``"transaction_type"`` (one of ``"purchase_order"``,
                ``"vendor_invoice"``, ``"journal_entry"``), and either
                ``"amount"`` or ``"total_amount"`` (numeric).  Optional:
                ``"approved_by_role"`` or ``"approver_role"``.
            params: Injection parameters.  Recognised key:
                ``"excess_percent"`` (int, 1–100).  If absent, a random
                value is generated via ``rng.randint(1, 100)``.
            rng: Seeded :class:`random.Random` instance for deterministic
                behavior.  CRITICAL: must use ONLY this instance, NEVER
                module-level ``random``.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If required fields are missing
                (``transaction_type``, ``amount``/``total_amount``) or
                the transaction type is not supported.
        """
        # ---- 1. Deep copy to preserve original transaction ----
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # ---- 2. Validate and extract excess_percent parameter ----
        validated_params: Dict[str, Any] = self._validate_params(
            params, self.PARAMETER_BOUNDS
        )
        excess_percent: int = validated_params.get("excess_percent")  # type: ignore[assignment]
        if excess_percent is None:
            excess_percent = rng.randint(1, 100)

        # ---- 3. Extract transaction_type (required) ----
        txn_type: Optional[str] = modified.get("transaction_type")
        if not txn_type:
            raise DiscrepancyInjectionError(
                "Missing 'transaction_type' field required for CTL-003 "
                "approval limit exceeded injection",
                details={
                    "discrepancy_type": self.type_code,
                    "missing_field": "transaction_type",
                },
            )

        # ---- 4. Extract monetary amount (required) ----
        amount_field: Optional[str] = None
        original_amount: Optional[Decimal] = None
        for field_name in ("amount", "total_amount"):
            if field_name in modified:
                amount_field = field_name
                try:
                    original_amount = Decimal(str(modified[field_name]))
                except Exception as exc:
                    raise DiscrepancyInjectionError(
                        f"Cannot parse monetary value from "
                        f"'{field_name}': {modified[field_name]}",
                        details={
                            "discrepancy_type": self.type_code,
                            "field_name": field_name,
                            "raw_value": str(modified[field_name]),
                        },
                    ) from exc
                break

        if amount_field is None or original_amount is None:
            raise DiscrepancyInjectionError(
                "Missing amount field ('amount' or 'total_amount') "
                "required for CTL-003 injection",
                details={
                    "discrepancy_type": self.type_code,
                    "missing_field": "amount",
                    "available_keys": list(modified.keys()),
                },
            )

        # ---- 5. Extract current approver role (optional) ----
        approver_field: Optional[str] = None
        approver_role: Optional[str] = None
        for field_name in ("approved_by_role", "approver_role"):
            if field_name in modified:
                approver_field = field_name
                approver_role = str(modified[field_name])
                break

        # ---- 6. Look up authority roles for this transaction type ----
        type_roles: Optional[Dict[str, Decimal]] = ROLE_AUTHORITY_LIMITS.get(
            txn_type
        )
        if not type_roles:
            raise DiscrepancyInjectionError(
                f"Transaction type '{txn_type}' not found in "
                f"ROLE_AUTHORITY_LIMITS.  Supported types: "
                f"{list(ROLE_AUTHORITY_LIMITS.keys())}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_type": txn_type,
                    "supported_types": list(ROLE_AUTHORITY_LIMITS.keys()),
                },
            )

        # ---- 7. Build sorted roles (ascending by authority limit) ----
        sorted_roles: List[Tuple[str, Decimal]] = sorted(
            type_roles.items(), key=lambda item: item[1]
        )

        # Roles whose authority limit is below the current transaction amount
        lower_roles: List[Tuple[str, Decimal]] = [
            (role, limit)
            for role, limit in sorted_roles
            if limit < original_amount
        ]

        # ---- 8. Determine available injection approaches ----
        approaches: List[str] = []

        # Approach A: Increase amount — requires a known approver with a
        # known authority limit so we can calculate new_amount = limit * (1 + %)
        if approver_role and approver_role in type_roles:
            approaches.append("increase_amount")

        # Approach B: Downgrade approver — requires at least one role whose
        # authority limit is below the current transaction amount
        if lower_roles:
            approaches.append("downgrade_approver")

        # Fallback: if neither approach is available (e.g. no known approver
        # AND amount is below all role limits), force Approach A using the
        # lowest-authority role's limit as reference and assign that role.
        if not approaches:
            if sorted_roles:
                approaches.append("increase_amount")
            else:
                raise DiscrepancyInjectionError(
                    "Cannot determine injection approach: no roles defined "
                    f"for transaction type '{txn_type}'",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_type": txn_type,
                        "amount": str(original_amount),
                    },
                )

        approach: str = (
            rng.choice(approaches) if len(approaches) > 1 else approaches[0]
        )

        # ---- 9. Apply the selected injection approach ----
        affected_fields: List[str]
        original_values: Dict[str, Any]
        modified_values: Dict[str, Any]
        financial_impact: Decimal
        authority_limit: Decimal

        if approach == "increase_amount":
            # Determine which authority limit to use as the basis
            if approver_role and approver_role in type_roles:
                authority_limit = type_roles[approver_role]
            else:
                # No known approver — pick the lowest-authority role and
                # assign it as the approver so the violation is explicit.
                target_role, authority_limit = sorted_roles[0]
                if approver_field is None:
                    approver_field = "approved_by_role"
                modified[approver_field] = target_role
                approver_role = target_role

            # Calculate the new amount: limit * (1 + excess_percent / 100)
            excess_factor: Decimal = (
                Decimal("1") + Decimal(str(excess_percent)) / Decimal("100")
            )
            new_amount: Decimal = authority_limit * excess_factor

            # Ensure we never *decrease* the amount — if the original amount
            # already exceeds the limit, apply the excess_percent to the
            # original amount instead, so the result always goes up.
            if new_amount <= original_amount:
                new_amount = original_amount * excess_factor

            modified[amount_field] = new_amount

            affected_fields = [amount_field]
            original_values = {amount_field: str(original_amount)}
            modified_values = {amount_field: str(new_amount)}
            financial_impact = new_amount - original_amount

        else:
            # Approach B: Downgrade the approver to a lower-authority role
            selected_role, authority_limit = rng.choice(lower_roles)

            if approver_field is None:
                approver_field = "approved_by_role"

            original_approver: Optional[str] = modified.get(approver_field)
            modified[approver_field] = selected_role

            affected_fields = [approver_field]
            original_values = {approver_field: original_approver}
            modified_values = {approver_field: selected_role}
            # Process risk only — no monetary amount changed
            financial_impact = Decimal("0")

        # ---- 10. Build human-readable description ----
        display_amount = modified.get(amount_field, original_amount)
        display_role = (
            modified.get(approver_field, approver_role)
            if approver_field
            else approver_role
        )
        description: str = (
            f"Transaction of ${display_amount} approved by "
            f"{display_role} whose limit is ${authority_limit}"
        )

        # ---- 11. Create structured ground truth data ----
        ground_truth: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=affected_fields,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=financial_impact,
            description=description,
            extra_metadata={
                "excess_percent": excess_percent,
                "authority_limit": str(authority_limit),
                "transaction_type": txn_type,
                "injection_approach": approach,
            },
        )

        # ---- 12. Log the injection event ----
        transaction_id: str = str(
            modified.get("transaction_id", modified.get("id", "unknown"))
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "simulation_id": modified.get("simulation_id"),
                "trace_id": modified.get("trace_id"),
            },
        )

        return modified, ground_truth
