"""CTL-001: Segregation of Duties Violation — simulates SoD control failures.

Implements the SoD Violation discrepancy type that modifies transaction data
to simulate scenarios where the same user/agent is assigned to incompatible
roles within a transaction workflow.

Typical SoD violation patterns:
    - Same person creates AND approves a purchase order
    - Same person processes a vendor invoice AND releases payment
    - Same person creates a journal entry AND posts it
    - Requisitioner is also the purchasing agent for the same PO

IMPORTANT (AAP §0.6.2): This class SIMULATES SoD violations as training data
for audit detection models. It does NOT enforce SoD rules — enforcement is
explicitly out of scope for MVP.

Catalog Entry:
    Type Code: CTL-001
    Category: control
    Difficulty: easy
    Detection Method: access_review
    Parameters: None (no configurable parameters)

References:
    - AAP Section 0.5.1 Group 5: CTL-001 Segregation of Duties Violation
    - AAP Section 0.6.2: SoD simulation, NOT enforcement
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - app/orchestration/workflow_orchestrator.py: ROLE_MAPPING
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
__all__: List[str] = ["SoDViolation", "SOD_CONFLICT_PAIRS"]

# ---------------------------------------------------------------------------
# SoD Conflict Pair Definitions
# ---------------------------------------------------------------------------
# Incompatible role pairs — if the same user holds both roles within a single
# transaction, it constitutes a Segregation of Duties violation.  Each pair
# is (role_a, role_b) where the two roles MUST be held by different persons.
#
# These abstract role names are mapped to concrete transaction field names
# via _ROLE_FIELD_MAP below.
SOD_CONFLICT_PAIRS: List[Tuple[str, str]] = [
    ("creator", "approver"),                        # Creator AND approver on same transaction
    ("requisitioner", "purchaser"),                  # Requisitioner AND purchasing agent
    ("ap_clerk", "payment_releaser"),                # AP clerk AND payment releaser
    ("accountant", "journal_poster"),                # Accountant AND journal poster
    ("warehouse_clerk", "inventory_adjuster"),        # Warehouse clerk AND inventory adjuster
    ("ar_clerk", "credit_approver"),                 # AR clerk AND credit limit approver
]

# ---------------------------------------------------------------------------
# Internal mapping: abstract role name → candidate transaction field names
# ---------------------------------------------------------------------------
# Maps each abstract role referenced in SOD_CONFLICT_PAIRS to the actual
# field names that can appear in transaction dictionaries.  The first match
# found in the transaction data is used.
_ROLE_FIELD_MAP: Dict[str, List[str]] = {
    "creator": ["created_by", "initiated_by", "submitted_by"],
    "approver": ["approved_by", "authorized_by"],
    "requisitioner": ["requested_by", "requisitioned_by"],
    "purchaser": ["purchased_by", "purchasing_agent_id", "buyer_id"],
    "ap_clerk": ["processed_by", "ap_processor_id"],
    "payment_releaser": ["released_by", "payment_released_by"],
    "accountant": ["prepared_by", "accountant_id"],
    "journal_poster": ["posted_by", "journal_posted_by"],
    "warehouse_clerk": ["received_by", "warehouse_processor_id"],
    "inventory_adjuster": ["adjusted_by", "inventory_adjusted_by"],
    "ar_clerk": ["ar_processed_by", "billed_by"],
    "credit_approver": ["credit_approved_by", "credit_reviewer_id"],
}

# ---------------------------------------------------------------------------
# Known role-like field name suffixes for generic fallback detection
# ---------------------------------------------------------------------------
_ROLE_FIELD_SUFFIXES: Tuple[str, ...] = (
    "_by",
    "_agent_id",
    "_processor_id",
    "_reviewer_id",
    "_clerk_id",
)


def _is_role_field(field_name: str) -> bool:
    """Check whether a field name looks like a role assignment field.

    A field is considered role-like if it ends with one of the known
    suffixes (``_by``, ``_agent_id``, etc.) that denote person/agent
    assignments in ERP transaction records.

    Args:
        field_name: The transaction dictionary key to evaluate.

    Returns:
        ``True`` if the field name matches a role-like pattern.
    """
    return any(field_name.endswith(suffix) for suffix in _ROLE_FIELD_SUFFIXES)


def _resolve_role_field(
    role_name: str,
    transaction: Dict[str, Any],
) -> str | None:
    """Resolve an abstract role name to a concrete field present in the transaction.

    Iterates through the candidate field names for *role_name* (from
    :data:`_ROLE_FIELD_MAP`) and returns the first one that exists as a
    key in *transaction*.

    Args:
        role_name: Abstract role identifier from :data:`SOD_CONFLICT_PAIRS`.
        transaction: The transaction data dictionary.

    Returns:
        The first matching field name, or ``None`` if no candidate exists.
    """
    candidates = _ROLE_FIELD_MAP.get(role_name, [])
    for candidate in candidates:
        if candidate in transaction:
            return candidate
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SoDViolation — CTL-001
# ═══════════════════════════════════════════════════════════════════════════


class SoDViolation(BaseDiscrepancy):
    """CTL-001: Segregation of Duties Violation.

    Injects an SoD violation by modifying a transaction so that the same
    user/agent ID appears in two incompatible role fields.  For example,
    setting ``approved_by`` equal to ``created_by``.

    This is an 'easy' difficulty discrepancy because the detection method
    (access_review / role comparison) is straightforward — simply check
    whether the same ``user_id`` appears in incompatible role fields.

    IMPORTANT: This class **simulates** SoD violations as training data for
    audit detection models.  It does **NOT** enforce segregation of duties —
    enforcement is explicitly out of scope for MVP (AAP §0.6.2).

    Attributes:
        type_code: ``"CTL-001"``
        category: ``"control"``
        difficulty: ``"easy"``
        name: ``"Segregation of Duties Violation"``
        description: Human-readable explanation of the discrepancy.
        detection_method: ``"access_review"``
    """

    # ------------------------------------------------------------------
    # Class-level attributes — override BaseDiscrepancy defaults
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "CTL-001"
    category: ClassVar[str] = "control"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Segregation of Duties Violation"
    description: ClassVar[str] = (
        "Same user assigned to incompatible roles within a transaction "
        "(e.g., creator and approver are the same person)"
    )
    detection_method: ClassVar[str] = "access_review"

    # No configurable parameters for CTL-001 — PARAMETER_BOUNDS stays as
    # the empty dict inherited from BaseDiscrepancy.

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an SoD violation into the transaction.

        Modifies the transaction so that the same ``user_id`` appears in
        two incompatible role fields (e.g., ``created_by == approved_by``).

        The method attempts three strategies in order:

        1. **Mapped conflict pairs** — iterate :data:`SOD_CONFLICT_PAIRS`,
           resolve both roles to concrete fields via :data:`_ROLE_FIELD_MAP`,
           and use the first pair where both fields exist in the transaction.
        2. **Generic role-field pairing** — collect all role-like fields
           (keys ending with ``_by``, ``_agent_id``, etc.) and pair any two
           with distinct values.
        3. **Error** — raise :class:`DiscrepancyInjectionError` if no role
           fields exist at all.

        Args:
            transaction: Transaction data dictionary containing role fields
                such as ``created_by``, ``approved_by``, ``posted_by``, etc.
            params: Injection parameters.  CTL-001 has **no** configurable
                parameters; this argument is accepted for contract compliance
                but is not inspected.
            rng: Seeded :class:`random.Random` instance for deterministic
                behaviour.  CRITICAL: all random selections MUST use this
                instance — NEVER the module-level ``random`` functions.

        Returns:
            A 2-tuple of:

            - **modified_transaction** (``Dict[str, Any]``) — deep copy of
              the input with one role field overwritten so that two
              incompatible fields share the same user/agent ID.
            - **ground_truth_data** (``Dict[str, Any]``) — structured
              ground truth record for :class:`GroundTruthGenerator`.

        Raises:
            DiscrepancyInjectionError: If the transaction contains no role
                fields at all, making SoD violation injection impossible.
        """
        # 1. Deep copy — never mutate the original transaction
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # 2. Attempt to find an eligible conflict pair from SOD_CONFLICT_PAIRS
        eligible_pairs: List[Tuple[str, str, str, str]] = []
        #    Each entry: (role_a_name, role_b_name, field_a, field_b)
        for role_a, role_b in SOD_CONFLICT_PAIRS:
            field_a = _resolve_role_field(role_a, modified)
            field_b = _resolve_role_field(role_b, modified)
            if field_a is not None and field_b is not None:
                eligible_pairs.append((role_a, role_b, field_a, field_b))

        role_1: str
        role_2: str
        field_1: str
        field_2: str

        if eligible_pairs:
            # Pick one eligible conflict pair deterministically via rng
            role_1, role_2, field_1, field_2 = rng.choice(eligible_pairs)
        else:
            # 3. Fallback — gather all role-like fields and pair any two
            role_fields: List[str] = [
                key for key in modified if _is_role_field(key)
            ]

            if len(role_fields) < 2:
                # Cannot form a pair — no SoD violation possible
                raise DiscrepancyInjectionError(
                    "No role fields found in transaction for SoD violation injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_keys": list(modified.keys()),
                        "role_fields_found": role_fields,
                    },
                )

            # Deterministic pair selection from available role fields
            pair = _select_generic_pair(role_fields, rng)
            field_1, field_2 = pair
            # Use field names as role labels for ground truth
            role_1 = field_1
            role_2 = field_2

        # 4. Record original values
        original_value_1 = modified.get(field_1)
        original_value_2 = modified.get(field_2)

        # 5. Apply the SoD violation: set field_2 equal to field_1
        #    (same user_id in both incompatible role fields)
        modified[field_2] = modified[field_1]

        user_id = str(modified[field_1])

        # 6. Financial impact — SoD violations are process risks, NOT
        #    direct financial discrepancies.  Impact is always zero.
        financial_impact = Decimal("0")

        # 7. Build ground truth record
        ground_truth = self._create_ground_truth_data(
            affected_fields=[field_1, field_2],
            original_values={
                field_1: original_value_1,
                field_2: original_value_2,
            },
            modified_values={
                field_1: modified[field_1],
                field_2: modified[field_2],
            },
            financial_impact=financial_impact,
            description=(
                f"Same user {user_id} assigned as both "
                f"{role_1} and {role_2}"
            ),
            extra_metadata={
                "conflict_type": "sod_violation",
                "incompatible_roles": [role_1, role_2],
                "conflicting_fields": [field_1, field_2],
                "shared_user_id": user_id,
            },
        )

        # 8. Structured logging per AAP §0.7.7
        transaction_id = str(
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

        logger.debug(
            "sod_violation_injected",
            service_name="transactions",
            component="SoDViolation",
            type_code=self.type_code,
            transaction_id=transaction_id,
            field_1=field_1,
            field_2=field_2,
            role_1=role_1,
            role_2=role_2,
            shared_user_id=user_id,
        )

        return modified, ground_truth


# ---------------------------------------------------------------------------
# Module-level helper (private)
# ---------------------------------------------------------------------------


def _select_generic_pair(
    role_fields: List[str],
    rng: random.Random,
) -> Tuple[str, str]:
    """Select two distinct role fields from a list for generic SoD injection.

    Shuffles the list deterministically (using the seeded *rng*) and returns
    the first two elements.  This guarantees a consistent selection for a
    given seed.

    Args:
        role_fields: List of at least 2 role-like field names found in the
            transaction data.
        rng: Seeded :class:`random.Random` instance.

    Returns:
        A 2-tuple of distinct field names ``(field_a, field_b)``.
    """
    # Work on a copy to avoid mutating the caller's list
    shuffled = list(role_fields)
    rng.shuffle(shuffled)
    return (shuffled[0], shuffled[1])
