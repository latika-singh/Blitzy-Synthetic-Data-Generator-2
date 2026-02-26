"""GL-002: Journal Entry Without Approval — simulates missing approval control.

Implements the Journal Entry Without Approval discrepancy type that removes,
clears, or nullifies the approval reference on a journal entry that requires
approval per the threshold configuration.

Per the approval system (``app/orchestration/approval_system.py``):
    - ALL journal entries require at minimum ``senior_accountant`` approval
    - Journal entries >$50K require ``controller`` approval
    - The approval workflow is:
      1. Accountant creates JE → routes to senior_accountant
      2. Senior accountant approves (if amount ≤ $50K) → posted
      3. Senior accountant escalates to controller (if amount > $50K)
      4. Controller approves → posted

This discrepancy simulates the scenario where a journal entry bypasses this
approval chain entirely — the JE is posted without any approval record.

Catalog Entry:
    Type Code: GL-002
    Category: gl
    Difficulty: easy
    Detection Method: approval_check
    Parameters: None (no configurable parameters)

References:
    - AAP Section 0.5.1 Group 5: GL-002 Journal Entry Without Approval
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - config/workflows/approval_thresholds.yaml: JE approval thresholds
    - app/orchestration/approval_system.py: APPROVAL_THRESHOLDS, ApprovalRequest
    - app/agents/specialized/accountant_agent.py: JE creation and routing
    - app/agents/specialized/senior_accountant_agent.py: JE approval authority
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
# Approval field definitions
# ---------------------------------------------------------------------------

# Fields that indicate approval status on a journal entry.
# These are the fields that a properly-approved JE would have populated.
APPROVAL_FIELDS: list[str] = [
    "approved_by",
    "approver_id",
    "approval_status",
    "approval_date",
    "approved_by_role",
    "approval_request_id",
    "approval_chain",
]

# Fields that indicate the JE has been properly reviewed (secondary indicators).
REVIEW_FIELDS: list[str] = [
    "reviewed_by",
    "reviewer_id",
    "review_status",
    "review_date",
]

# Values that clearly signal "no approval" when found in approval fields.
NO_APPROVAL_VALUES: list[Any] = [
    None,
    "",
    "none",
    "not_approved",
    "pending",
    "skipped",
]

# Journal entry approval thresholds matching the ApprovalSystem
# (app/orchestration/approval_system.py lines 169-172) and
# config/workflows/approval_thresholds.yaml lines 104-112.
#
# Key rule: ALL journal entries require at minimum senior_accountant approval.
# Journal entries >$50K require controller approval.
JE_APPROVAL_THRESHOLDS: Dict[str, Decimal] = {
    "senior_accountant": Decimal("0"),       # All JEs need senior_accountant
    "controller": Decimal("50000"),          # >$50K needs controller
}

# ---------------------------------------------------------------------------
# Strategy names for the three approval-removal approaches
# ---------------------------------------------------------------------------
_STRATEGY_CLEAR_ALL = "clear_all"
_STRATEGY_SET_SKIPPED = "set_skipped"
_STRATEGY_REMOVE_FIELDS = "remove_fields"

_ALL_STRATEGIES: list[str] = [
    _STRATEGY_CLEAR_ALL,
    _STRATEGY_SET_SKIPPED,
    _STRATEGY_REMOVE_FIELDS,
]

# Fields where amount data may appear on a journal entry transaction dict
_AMOUNT_FIELDS: list[str] = [
    "amount",
    "total_amount",
    "je_amount",
    "entry_amount",
    "total_debit",
]


class JournalNoApproval(BaseDiscrepancy):
    """GL-002: Journal Entry Without Approval.

    Injects a missing approval violation by removing or clearing approval
    fields on a journal entry.  This simulates the scenario where a JE is
    posted without proper authorization.

    This is an 'easy' difficulty discrepancy because the detection method
    is straightforward: check whether the journal entry has a valid
    approval record (``approved_by`` is not null, ``approval_status`` is not
    ``'pending'`` or ``'skipped'``).

    Three injection strategies are available, selected deterministically
    via the seeded ``rng``:

    - **clear_all** — Set all found approval fields to ``None``
    - **set_skipped** — Set ``approval_status`` to ``"skipped"`` and clear
      the ``approved_by`` / ``approver_id`` fields
    - **remove_fields** — Delete all approval-related keys from the
      transaction dictionary entirely

    When a transaction has no pre-existing approval fields, the injector
    adds ``approval_status`` and ``approved_by`` fields with bypass values
    to create a detectable pattern (rather than raising an error), since
    the discrepancy represents the *absence* of approval.
    """

    # ------------------------------------------------------------------
    # Class-level attributes (all 6 + detection_difficulty via base)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "GL-002"
    category: ClassVar[str] = "gl"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Journal Entry Without Approval"
    description: ClassVar[str] = (
        "Journal entry posted without required approval, bypassing the "
        "approval chain (senior_accountant for all JEs, controller for >$50K)"
    )
    detection_method: ClassVar[str] = "approval_check"

    # GL-002 has no configurable parameters.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject()
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a missing approval on a journal entry.

        Removes or nullifies approval fields to simulate posting without
        authorization.  The specific manipulation strategy is selected
        deterministically using the seeded ``rng``.

        Args:
            transaction: Journal entry transaction data.  May or may not
                already contain approval fields.
            params: Injection parameters.  GL-002 has no configurable
                parameters — this dict is ignored (but accepted for
                contract compatibility).
            rng: Seeded :class:`random.Random` instance for deterministic
                reproducibility per AAP §0.7.1.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If injection logic fails
                unexpectedly (defensive — should not normally occur).
        """
        # 1. Deep-copy the transaction to preserve the original
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # 2. Identify all approval-related fields present in the transaction
        found_approval_fields: List[str] = [
            field for field in APPROVAL_FIELDS if field in modified
        ]
        found_review_fields: List[str] = [
            field for field in REVIEW_FIELDS if field in modified
        ]
        all_found_fields: List[str] = found_approval_fields + found_review_fields

        # 3. Record the original values of ALL approval/review fields
        original_values: Dict[str, Any] = {
            field: modified.get(field) for field in all_found_fields
        }

        # 4. Determine if the transaction has any approval fields to manipulate
        has_approval_fields: bool = len(all_found_fields) > 0

        # 5. Choose a manipulation strategy using the seeded rng
        strategy: str = rng.choice(_ALL_STRATEGIES)

        # 6. Track the affected fields and their new values
        affected_fields: List[str] = []
        modified_values: Dict[str, Any] = {}

        if has_approval_fields:
            # Transaction has approval fields — manipulate them
            affected_fields, modified_values = self._apply_strategy(
                modified=modified,
                strategy=strategy,
                found_approval_fields=found_approval_fields,
                found_review_fields=found_review_fields,
                rng=rng,
            )
        else:
            # Transaction has NO approval fields.  Create the discrepancy
            # pattern by adding bypass indicators.  This represents a JE
            # that was posted without ever entering the approval workflow.
            modified["approval_status"] = "skipped"
            modified["approved_by"] = None

            affected_fields = ["approval_status", "approved_by"]
            original_values = {
                "approval_status": "NOT_PRESENT",
                "approved_by": "NOT_PRESENT",
            }
            modified_values = {
                "approval_status": "skipped",
                "approved_by": None,
            }
            # Override strategy for ground truth clarity
            strategy = "create_bypass_indicators"

        # 7. Determine financial impact from the transaction amount
        financial_impact: Decimal = self._extract_financial_impact(modified)

        # 8. Determine the required approver based on amount
        required_approver: str = self._determine_required_approver(
            financial_impact
        )

        # 9. Build a human-readable description
        if financial_impact > Decimal("0"):
            description_text = (
                f"Journal entry of ${financial_impact} posted without "
                f"{required_approver} approval"
            )
        else:
            description_text = (
                f"Journal entry posted without {required_approver} approval "
                f"(amount unknown)"
            )

        # 10. Create ground truth data using base class helper
        ground_truth: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=affected_fields,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=financial_impact,
            description=description_text,
            extra_metadata={
                "required_approver": required_approver,
                "amount": str(financial_impact),
                "strategy": strategy,
                "cleared_fields": affected_fields,
                "has_original_approval_fields": has_approval_fields,
            },
        )

        # 11. Log the injection event
        transaction_id: str = str(
            modified.get("transaction_id")
            or modified.get("je_id")
            or modified.get("journal_entry_id")
            or modified.get("id")
            or "unknown"
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "simulation_id": modified.get("simulation_id"),
                "trace_id": modified.get("trace_id"),
            },
        )

        # 12. Return modified transaction and ground truth
        return modified, ground_truth

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_strategy(
        *,
        modified: Dict[str, Any],
        strategy: str,
        found_approval_fields: List[str],
        found_review_fields: List[str],
        rng: random.Random,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Apply the selected approval-removal strategy to the transaction.

        Mutates *modified* in-place and returns the list of affected field
        names and their new values for ground truth tracking.

        Args:
            modified: The deep-copied transaction dict to mutate.
            strategy: One of ``_STRATEGY_CLEAR_ALL``,
                ``_STRATEGY_SET_SKIPPED``, or ``_STRATEGY_REMOVE_FIELDS``.
            found_approval_fields: Approval fields present in the
                transaction.
            found_review_fields: Review fields present in the transaction.
            rng: Seeded RNG (used by ``set_skipped`` for choosing the
                bypass value).

        Returns:
            Tuple of (affected_fields_list, modified_values_dict).
        """
        all_fields: List[str] = found_approval_fields + found_review_fields
        affected: List[str] = []
        new_values: Dict[str, Any] = {}

        if strategy == _STRATEGY_CLEAR_ALL:
            # Strategy A: Set all approval and review fields to None
            for field in all_fields:
                modified[field] = None
                affected.append(field)
                new_values[field] = None

        elif strategy == _STRATEGY_SET_SKIPPED:
            # Strategy B: Set approval_status to a bypass value and clear
            # the approver identity fields
            bypass_value: str = rng.choice(["skipped", "pending", "not_approved"])

            if "approval_status" in modified:
                modified["approval_status"] = bypass_value
                affected.append("approval_status")
                new_values["approval_status"] = bypass_value
            else:
                # Add the field if not present
                modified["approval_status"] = bypass_value
                affected.append("approval_status")
                new_values["approval_status"] = bypass_value

            # Clear the identity fields
            identity_fields = [
                f for f in found_approval_fields
                if f in ("approved_by", "approver_id", "approved_by_role")
            ]
            for field in identity_fields:
                modified[field] = None
                affected.append(field)
                new_values[field] = None

            # Also clear review fields
            for field in found_review_fields:
                modified[field] = None
                affected.append(field)
                new_values[field] = None

        elif strategy == _STRATEGY_REMOVE_FIELDS:
            # Strategy C: Remove all approval and review keys entirely
            for field in all_fields:
                if field in modified:
                    del modified[field]
                    affected.append(field)
                    new_values[field] = "REMOVED"

        return affected, new_values

    @staticmethod
    def _extract_financial_impact(transaction: Dict[str, Any]) -> Decimal:
        """Extract the financial impact from the transaction amount.

        Searches a prioritised list of amount field names and returns the
        first non-zero value found, coerced to :class:`Decimal`.  If no
        amount field is found, returns ``Decimal("0")``.

        Args:
            transaction: The transaction data dictionary.

        Returns:
            The journal entry amount as a :class:`Decimal`, or
            ``Decimal("0")`` if no amount is available.
        """
        for field_name in _AMOUNT_FIELDS:
            raw_value = transaction.get(field_name)
            if raw_value is not None:
                try:
                    amount = Decimal(str(raw_value))
                    if amount > Decimal("0"):
                        return amount
                except Exception:
                    # Non-numeric value — skip to next candidate field
                    continue
        return Decimal("0")

    @staticmethod
    def _determine_required_approver(amount: Decimal) -> str:
        """Determine which approver role is required for the given amount.

        Applies the journal entry approval thresholds from the
        ``ApprovalSystem`` (``approval_system.py`` lines 169-172):
            - All JEs (amount >= $0): ``senior_accountant``
            - JEs > $50K: ``controller``

        Args:
            amount: The journal entry amount as a :class:`Decimal`.

        Returns:
            The required approver role string.
        """
        controller_threshold = JE_APPROVAL_THRESHOLDS["controller"]
        if amount >= controller_threshold:
            return "controller"
        return "senior_accountant"
