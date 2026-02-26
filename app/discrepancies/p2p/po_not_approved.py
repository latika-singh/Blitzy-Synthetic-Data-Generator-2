"""P2P-005: PO Not Approved discrepancy type implementation.

Modifies a purchase order's approval status to simulate either a complete
bypass of the approval process or an approval at an insufficient authority
level relative to the PO amount.

PO Approval Thresholds (from config/workflows/approval_thresholds.yaml):
    - < $5,000: No approval required
    - $5,000–$25,000: purchasing_manager approval required
    - $25,000–$100,000: controller approval required
    - > $100,000: cfo approval required

This discrepancy sets the approval_status to 'not_approved', 'pending', or
sets the approver_role to a lower level than required for the PO amount.

Configurable Parameters:
    None — approval bypass is determined by the PO amount vs. threshold.

Detection Method: approval_check

Difficulty: easy

Financial Impact: Full PO amount (unapproved exposure).

References:
    - AAP Section 0.5.1 Group 5: P2P-005 PO Not Approved
    - config/workflows/approval_thresholds.yaml: PO approval thresholds
    - app/orchestration/approval_system.py: APPROVAL_THRESHOLDS mapping
    - app/agents/specialized/purchasing_agent.py: PO approval routing
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
__all__ = ["PONotApproved"]


# ---------------------------------------------------------------------------
# Approval Role Hierarchy (lowest → highest authority)
# Used for determining "one level below required" in insufficient-level bypass
# ---------------------------------------------------------------------------
_ROLE_HIERARCHY: List[str] = [
    "none",                # < $5K — no approval needed
    "purchasing_manager",  # $5K – $25K
    "controller",          # $25K – $100K
    "cfo",                 # > $100K
]


class PONotApproved(BaseDiscrepancy):
    """P2P-005: PO Not Approved discrepancy.

    Injects a discrepancy by modifying a purchase order's approval state to
    simulate either:

    **Option A — Complete Bypass:**
        The approval process is entirely bypassed.  ``approval_status`` is set
        to ``"not_approved"`` or ``"bypassed"``, and ``approved_by`` /
        ``approver_role`` are cleared.  This simulates a PO that was never
        submitted for approval despite exceeding the monetary threshold.

    **Option B — Insufficient Level:**
        The PO is approved by a role one level *below* the required authority.
        For example, a $50,000 PO that requires controller approval is instead
        approved by a purchasing_manager.  The ``approval_status`` remains
        ``"approved"`` but the ``approver_role`` is downgraded.

    The bypass method is selected deterministically via the seeded ``rng``
    instance to maintain reproducibility.

    Attributes:
        type_code: ``"P2P-005"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"PO Not Approved"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"approval_check"``
        PARAMETER_BOUNDS: Empty dict (no configurable parameters).
        _APPROVAL_THRESHOLDS: List of ``(Decimal, str)`` tuples mapping
            monetary thresholds to required approver roles.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-005"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "PO Not Approved"
    description: ClassVar[str] = (
        "Purchase order bypassed or received insufficient approval"
    )
    detection_method: ClassVar[str] = "approval_check"

    # No configurable injection parameters — bypass is determined
    # entirely by the PO amount vs. the approval threshold hierarchy.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # Approval threshold constants matching the specification in
    # config/workflows/approval_thresholds.yaml and
    # app/orchestration/approval_system.py APPROVAL_THRESHOLDS mapping.
    #
    # Each entry is (minimum_amount, required_approver_role).
    # The list is ordered ascending by threshold.  The required role is
    # the FIRST entry whose threshold is <= the PO amount.
    #
    # For POs below $5,000 no approval is required (no entry needed;
    # the _get_required_role() method handles this implicitly).
    _APPROVAL_THRESHOLDS: ClassVar[List[Tuple[Decimal, str]]] = [
        (Decimal("5000"), "purchasing_manager"),
        (Decimal("25000"), "controller"),
        (Decimal("100000"), "cfo"),
    ]

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a PO-not-approved discrepancy into the transaction.

        Args:
            transaction: The original purchase order transaction data.
                Expected keys include ``total_amount`` (numeric/Decimal),
                ``approval_status``, ``approved_by``, ``approver_role``,
                ``approval_date``, ``approval_chain``, ``po_number``, and
                ``transaction_id`` (or ``po_id``).
            params: Injection parameters (unused for P2P-005 — empty bounds).
            rng: Seeded :class:`random.Random` instance for deterministic
                bypass method selection.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction is missing
                critical fields (``total_amount``).
        """
        try:
            # 1. Deep-copy the transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate params (no-op for empty PARAMETER_BOUNDS, but
            #    ensures the standard validation path is exercised)
            validated_params = self._validate_params(params)

            # 3. Extract the PO total amount (required field)
            raw_amount = modified.get("total_amount")
            if raw_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required 'total_amount' field "
                    "for P2P-005 PO Not Approved discrepancy",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id")
                            or modified.get("po_id", "unknown")
                        ),
                    },
                )

            total_amount = Decimal(str(raw_amount))

            # 4. Determine the required approval role for this amount
            required_role = self._get_required_role(total_amount)

            # 5. Store original approval fields before modification
            original_approval_status: Optional[str] = modified.get(
                "approval_status"
            )
            original_approved_by: Any = modified.get("approved_by")
            original_approver_role: Optional[str] = modified.get(
                "approver_role"
            )
            original_approval_date: Any = modified.get("approval_date")
            original_approval_chain: Any = modified.get("approval_chain")

            original_values: Dict[str, Any] = {
                "approval_status": original_approval_status,
                "approved_by": original_approved_by,
                "approver_role": original_approver_role,
                "approval_date": (
                    str(original_approval_date)
                    if original_approval_date is not None
                    else None
                ),
                "approval_chain": original_approval_chain,
            }

            # 6. Select bypass method deterministically via seeded RNG
            #    "complete" = full approval bypass
            #    "insufficient_level" = approved at a lower authority level
            #
            #    If the required role is "none" (PO < $5K), there's nothing
            #    to downgrade, so we always use complete bypass.
            if required_role == "none":
                bypass_type = "complete"
            else:
                bypass_type = rng.choice(["complete", "insufficient_level"])

            # 7. Apply the selected bypass method
            affected_fields: List[str] = []
            modified_values: Dict[str, Any] = {}
            actual_role: Optional[str] = None

            if bypass_type == "complete":
                # Option A: Complete approval bypass
                new_status = rng.choice(["not_approved", "bypassed", "pending"])
                modified["approval_status"] = new_status
                modified["approved_by"] = None
                modified["approver_role"] = None
                modified["approval_date"] = None
                modified["approval_chain"] = []

                affected_fields = [
                    "approval_status",
                    "approved_by",
                    "approver_role",
                    "approval_date",
                    "approval_chain",
                ]
                modified_values = {
                    "approval_status": new_status,
                    "approved_by": None,
                    "approver_role": None,
                    "approval_date": None,
                    "approval_chain": [],
                }
                actual_role = None
            else:
                # Option B: Insufficient approval level
                insufficient_role = self._get_insufficient_role(
                    required_role, rng
                )
                modified["approval_status"] = "approved"
                modified["approver_role"] = insufficient_role

                # If there was no approved_by, generate a placeholder
                if not modified.get("approved_by"):
                    modified["approved_by"] = (
                        f"auto_{insufficient_role}_{rng.randint(1000, 9999)}"
                    )

                affected_fields = ["approval_status", "approver_role"]
                modified_values = {
                    "approval_status": "approved",
                    "approver_role": insufficient_role,
                }
                actual_role = insufficient_role

            # 8. Financial impact: full PO amount as unapproved exposure
            financial_impact = total_amount

            # 9. Build human-readable description
            po_number = modified.get("po_number", modified.get("po_id", "N/A"))
            if bypass_type == "complete":
                desc = (
                    f"PO {po_number} for ${total_amount:,.2f} completely "
                    f"bypassed approval (required: {required_role})"
                )
            else:
                desc = (
                    f"PO {po_number} for ${total_amount:,.2f} approved at "
                    f"insufficient level (required: {required_role}, "
                    f"actual: {actual_role})"
                )

            # 10. Create ground truth data via base-class helper
            ground_truth = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=desc,
                extra_metadata={
                    "bypass_type": bypass_type,
                    "required_role": required_role,
                    "actual_role": actual_role,
                    "po_number": str(po_number),
                    "total_amount": str(total_amount),
                },
            )

            # 11. Log the injection event via base-class helper
            transaction_id = str(
                modified.get("transaction_id")
                or modified.get("po_id", "unknown")
            )
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "bypass_type": bypass_type,
                    "required_role": required_role,
                    "actual_role": actual_role,
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise domain-specific errors as-is
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            logger.error(
                "po_not_approved_injection_failed",
                service_name="transactions",
                component="PONotApproved",
                type_code=self.type_code,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise DiscrepancyInjectionError(
                f"P2P-005 PO Not Approved injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_required_role(amount: Decimal) -> str:
        """Determine the required approval role for the given PO amount.

        Evaluates the PO amount against the approval threshold hierarchy:

        - ``< $5,000``: ``"none"`` (no approval required)
        - ``$5,000 – $24,999.99``: ``"purchasing_manager"``
        - ``$25,000 – $99,999.99``: ``"controller"``
        - ``≥ $100,000``: ``"cfo"``

        Args:
            amount: The PO total amount as a :class:`Decimal`.

        Returns:
            The role string for the minimum required approver, or
            ``"none"`` if no approval is required.
        """
        required_role = "none"
        for threshold, role in PONotApproved._APPROVAL_THRESHOLDS:
            if amount >= threshold:
                required_role = role
            else:
                break
        return required_role

    @staticmethod
    def _get_insufficient_role(
        required_role: str,
        rng: random.Random,
    ) -> str:
        """Return a role one level below the required approval level.

        Uses the ``_ROLE_HIERARCHY`` module-level list to find the index of
        the required role and returns the role at ``index - 1``.

        Special cases:
            - If the required role is ``"purchasing_manager"`` (the lowest
              actual approval level), the "insufficient" role is ``"none"``
              which is represented as ``"auto_approved"`` to indicate a PO
              that was processed without any human approval.
            - If the required role is not found in the hierarchy (should
              never happen), the role ``"purchasing_agent"`` is returned
              as a safe fallback.

        Args:
            required_role: The correct approval role for the PO amount.
            rng: Seeded RNG (unused here but accepted for interface
                consistency with potential future enhancements).

        Returns:
            A role string that is one level below ``required_role`` in
            the approval hierarchy.
        """
        try:
            idx = _ROLE_HIERARCHY.index(required_role)
        except ValueError:
            # Unknown role — return a generic low-authority fallback
            return "purchasing_agent"

        if idx <= 0:
            # Already at the lowest level; return "auto_approved"
            return "auto_approved"

        return _ROLE_HIERARCHY[idx - 1]
