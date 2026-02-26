"""CTL-002: Same User Created and Approved — simulates self-approval control failure.

Implements the Self-Approval discrepancy type that modifies transaction data so
that the approver_id (or equivalent approval field) equals the creator_id (or
equivalent creation field). This simulates the scenario where the same person
both creates and approves a transaction, bypassing the maker-checker control.

This is a common audit finding in ERP systems: the person who initiated a
purchase order should not be the same person who approves it.

Catalog Entry:
    Type Code: CTL-002
    Category: control
    Difficulty: easy
    Detection Method: approval_check
    Parameters: None (no configurable parameters)

References:
    - AAP Section 0.5.1 Group 5: CTL-002 Same User Created and Approved
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - app/orchestration/approval_system.py: ApprovalRequest model
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
__all__ = ["SelfApproval", "CREATOR_APPROVER_FIELDS"]


# ---------------------------------------------------------------------------
# Field pairs for creator-approver detection
# ---------------------------------------------------------------------------

# Each tuple is (creator_field, approver_field) representing a pair of
# transaction dictionary keys where the *first* field identifies the person
# who created/initiated the transaction and the *second* identifies the
# person who approved/authorized it.  A self-approval discrepancy is
# injected by setting the approver field value equal to the creator value.
#
# The list covers naming conventions used across P2P, O2C, GL, and generic
# ERP transaction layouts (purchase orders, vendor invoices, journal entries,
# payment requests, etc.).
CREATOR_APPROVER_FIELDS: List[Tuple[str, str]] = [
    ("created_by", "approved_by"),
    ("requested_by", "approved_by"),
    ("submitted_by", "approved_by"),
    ("created_by", "authorized_by"),
    ("processed_by", "approved_by"),
    ("initiated_by", "approved_by"),
]


# ---------------------------------------------------------------------------
# SelfApproval — CTL-002 discrepancy implementation
# ---------------------------------------------------------------------------


class SelfApproval(BaseDiscrepancy):
    """CTL-002: Same User Created and Approved.

    Injects a self-approval violation by modifying the transaction so that
    the approver field matches the creator field — the same user both
    created and approved the transaction.

    This is an 'easy' difficulty discrepancy because the detection method
    is straightforward: compare creator_id against approver_id.

    Injection logic:
        1. Scan the transaction dictionary for known creator-approver field
           pairs (see :data:`CREATOR_APPROVER_FIELDS`).
        2. If multiple eligible pairs exist, use the seeded RNG to select one.
        3. Set the approver field value to the creator field value.
        4. If no full pair is found but a lone ``created_by`` exists, add an
           ``approved_by`` field with the same value (and vice versa).
        5. If no suitable fields are found at all, raise
           :class:`DiscrepancyInjectionError`.

    The financial impact is always ``Decimal("0")`` because self-approval is
    a **process risk** (control deficiency), not a direct financial
    misstatement.

    Attributes:
        type_code: ``"CTL-002"``
        category: ``"control"``
        difficulty: ``"easy"``
        name: ``"Self-Approval"``
        description: Human-readable explanation of the discrepancy.
        detection_method: ``"approval_check"``
    """

    # ------------------------------------------------------------------
    # Class-level attributes (MUST match catalog entry exactly)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "CTL-002"
    category: ClassVar[str] = "control"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Self-Approval"
    description: ClassVar[str] = (
        "Same user created and approved the transaction, "
        "bypassing maker-checker control"
    )
    detection_method: ClassVar[str] = "approval_check"

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a self-approval violation into the transaction.

        Sets the approver field to the same value as the creator field,
        simulating a scenario where the same user both created and approved
        the transaction.

        Args:
            transaction: Transaction data dictionary containing creator and
                approver fields (e.g. ``created_by``, ``approved_by``).
            params: Injection parameters.  CTL-002 has **no** configurable
                parameters — this argument is accepted for interface
                compatibility but is not used.
            rng: Seeded :class:`random.Random` instance for deterministic
                behavior.  Used when multiple eligible field pairs exist
                to select one via ``rng.choice()``.

        Returns:
            A 2-tuple of:

            - **modified_transaction** — Deep copy of *transaction* with
              the approver field set to the creator's value.
            - **ground_truth_data** — Dictionary describing what was changed,
              suitable for the ``GroundTruthGenerator``.

        Raises:
            DiscrepancyInjectionError: If no creator-approver field pair
                can be identified in the transaction data.
        """
        # 1. Deep copy — preserve the original transaction unmodified.
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # 2. Identify eligible (creator_field, approver_field) pairs where
        #    BOTH keys already exist in the transaction dictionary.
        eligible_pairs: List[Tuple[str, str]] = [
            (creator_field, approver_field)
            for creator_field, approver_field in CREATOR_APPROVER_FIELDS
            if creator_field in modified and approver_field in modified
        ]

        # Track injection details for ground truth
        creator_field: str
        approver_field: str
        creator_value: Any
        original_approver_value: Any
        already_self_approved: bool = False

        if eligible_pairs:
            # 3a. Multiple eligible pairs → deterministic random selection.
            chosen_pair: Tuple[str, str] = (
                rng.choice(eligible_pairs) if len(eligible_pairs) > 1
                else eligible_pairs[0]
            )
            creator_field, approver_field = chosen_pair
            creator_value = modified[creator_field]
            original_approver_value = modified[approver_field]

            # Check for pre-existing self-approval (same value already).
            if creator_value == original_approver_value:
                already_self_approved = True

            # 4. Set the approver field equal to the creator field value.
            modified[approver_field] = copy.deepcopy(creator_value)

        else:
            # 3b. No full pair found — attempt fallback with lone fields.
            #
            # Fallback A: If a creator field exists without an approver field,
            #   add the standard ``approved_by`` field with the same value.
            #
            # Fallback B: If an approver field exists without a creator field,
            #   add the standard ``created_by`` field with the same value.

            lone_creator_field: str | None = None
            lone_approver_field: str | None = None

            # Scan for any known creator field that exists in the transaction.
            for c_field, _ in CREATOR_APPROVER_FIELDS:
                if c_field in modified:
                    lone_creator_field = c_field
                    break

            # Scan for any known approver field that exists in the transaction.
            for _, a_field in CREATOR_APPROVER_FIELDS:
                if a_field in modified:
                    lone_approver_field = a_field
                    break

            if lone_creator_field is not None:
                # Fallback A — add ``approved_by`` set to the creator's value.
                creator_field = lone_creator_field
                approver_field = "approved_by"
                creator_value = modified[creator_field]
                original_approver_value = None  # field did not exist before
                modified[approver_field] = copy.deepcopy(creator_value)

            elif lone_approver_field is not None:
                # Fallback B — add ``created_by`` set to the approver's value.
                approver_field = lone_approver_field
                creator_field = "created_by"
                original_approver_value = modified[approver_field]
                creator_value = original_approver_value
                modified[creator_field] = copy.deepcopy(original_approver_value)

            else:
                # No suitable fields found at all — injection is impossible.
                logger.warning(
                    "self_approval_injection_failed",
                    service_name="transactions",
                    component="SelfApproval",
                    type_code=self.type_code,
                    reason="no_creator_approver_fields",
                    available_keys=sorted(modified.keys())[:20],
                )
                raise DiscrepancyInjectionError(
                    "No creator-approver fields found in transaction "
                    "for self-approval injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "available_keys": sorted(modified.keys())[:20],
                    },
                )

        # 5. Compute financial impact — self-approval is a process risk,
        #    not a direct financial discrepancy.
        financial_impact: Decimal = Decimal("0")

        # 6. Build a human-readable description for the ground truth record.
        description: str = (
            f"User {creator_value} both created and approved the transaction"
        )

        # 7. Assemble extra metadata.
        extra_metadata: Dict[str, Any] = {
            "creator_field": creator_field,
            "approver_field": approver_field,
            "user_id": str(creator_value),
            "already_self_approved": already_self_approved,
        }

        # 8. Create ground truth data via the inherited helper.
        ground_truth_data: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=[approver_field],
            original_values={approver_field: original_approver_value},
            modified_values={approver_field: creator_value},
            financial_impact=financial_impact,
            description=description,
            extra_metadata=extra_metadata,
        )

        # 9. Log the successful injection event.
        transaction_id: str = str(
            modified.get("transaction_id", modified.get("id", "unknown"))
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "creator_field": creator_field,
                "approver_field": approver_field,
                "user_id": str(creator_value),
            },
        )

        # 10. Return the modified transaction and ground truth data.
        return modified, ground_truth_data
