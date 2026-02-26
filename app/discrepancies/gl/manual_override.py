"""GL-005: Manual Entry Overriding System Entry — simulates manual GL overrides.

Implements the Manual Override discrepancy type that modifies a journal entry
to appear as a manual override of a system-generated (automated) entry. This
simulates the scenario where a user manually creates a journal entry that
reverses, adjusts, or replaces an automated posting.

Manual overrides of system entries are a significant audit concern because:
    - They bypass standard transaction workflow controls
    - They may indicate intentional manipulation of financial records
    - System-generated entries follow deterministic rules; overrides break that chain
    - Pattern of manual overrides may indicate control weaknesses

The discrepancy modifies the journal entry's source/origin metadata to indicate
it is a manual entry that references (overrides) a prior system-generated entry.

Catalog Entry:
    Type Code: GL-005
    Category: gl
    Difficulty: medium
    Detection Method: system_log_review
    Parameters: None (no configurable parameters)

References:
    - AAP Section 0.5.1 Group 5: GL-005 Manual Entry Overriding System Entry
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - app/orchestration/transaction_orchestrator.py: journal_entry artifacts
    - app/agents/specialized/accountant_agent.py: JE creation workflow
"""

from __future__ import annotations

import copy
import random
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Source Type Definitions
# ---------------------------------------------------------------------------

# Journal entry source types indicating system/automated origin.
# These are the values typically assigned by an ERP system to entries that
# are generated programmatically (e.g., from sub-ledger postings, batch
# processing, or interface imports).
SYSTEM_SOURCE_TYPES: list[str] = [
    "system",
    "auto",
    "automated",
    "system_generated",
    "batch",
    "interface",
    "subledger",
]

# Journal entry source types indicating manual/user origin.
# These values signal that the entry was created by a human user rather
# than an automated process — precisely the pattern we inject.
MANUAL_SOURCE_TYPES: list[str] = [
    "manual",
    "manual_entry",
    "user_entry",
    "manual_override",
    "manual_adjustment",
]

# Fields that indicate the source/origin of a journal entry.
# We iterate these to find an existing source indicator to modify.
SOURCE_FIELDS: list[str] = [
    "source",
    "source_type",
    "entry_source",
    "origin",
    "created_by_system",
    "is_system_generated",
    "je_source",
]

# Fields that may reference a prior journal entry being overridden.
# We pick one of these to add as the override reference field.
OVERRIDE_REFERENCE_FIELDS: list[str] = [
    "overrides_je_id",
    "reverses_je_id",
    "reference_je_id",
    "original_je_id",
    "supersedes_je_id",
]

# Override reason values — human-readable explanations for the override.
# These mimic the kind of justifications a user might provide when
# manually overriding a system-generated entry.
OVERRIDE_REASONS: list[str] = [
    "Manual correction of system posting",
    "Override automated accrual entry",
    "Manual adjustment to system-generated entry",
    "Reverse and correct system posting",
    "Manual reclassification of system entry",
    "Override depreciation calculation",
    "Manual fix for automated GL posting",
]


# ---------------------------------------------------------------------------
# ManualOverride Discrepancy Class
# ---------------------------------------------------------------------------


class ManualOverride(BaseDiscrepancy):
    """GL-005: Manual Entry Overriding System Entry.

    Injects a manual override indicator into a journal entry, making it
    appear as though a manual entry was created to override, reverse, or
    adjust a prior system-generated entry.

    This is a 'medium' difficulty discrepancy because detection requires:
    1. Identifying the source/origin of the journal entry (system vs. manual)
    2. Checking for references to prior system-generated entries
    3. System log review to confirm the override relationship
    4. Contextual analysis of whether the override is justified

    GL-005 has no configurable parameters — it always modifies the
    source/origin metadata and adds override reference information.

    Attributes:
        type_code: ``"GL-005"``
        category: ``"gl"``
        difficulty: ``"medium"``
        name: ``"Manual Override of System Entry"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"system_log_review"``
    """

    # ------------------------------------------------------------------
    # Class-level attributes (MUST be set by every BaseDiscrepancy subclass)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "GL-005"
    category: ClassVar[str] = "gl"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Manual Override of System Entry"
    description: ClassVar[str] = (
        "Manual journal entry that overrides, reverses, or adjusts a "
        "prior system-generated (automated) entry"
    )
    detection_method: ClassVar[str] = "system_log_review"

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a manual override indicator into a journal entry.

        Modifies the source metadata to indicate this is a manual entry
        that overrides a system-generated entry. Adds an override reference
        to a fake prior system journal entry, along with override reason
        and metadata fields.

        Args:
            transaction: Journal entry transaction data dictionary.
            params: Injection parameters (GL-005 has no configurable params;
                this argument is accepted for contract compliance but ignored).
            rng: Seeded :class:`random.Random` instance for deterministic
                reproducibility.  CRITICAL: All random selections use this
                instance — NEVER module-level ``random``.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)`` where:

            - **modified_transaction** has its source metadata changed to
              indicate a manual override of a system entry.
            - **ground_truth_data** contains the full discrepancy record
              for the GroundTruthGenerator.

        Raises:
            DiscrepancyInjectionError: If the transaction cannot be modified
                to appear as a manual override (e.g., critical data missing).
        """
        try:
            # 1. Deep copy to preserve original transaction data
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Tracking dictionaries for ground truth
            affected_fields: List[str] = []
            original_values: Dict[str, Any] = {}
            modified_values: Dict[str, Any] = {}

            # 3. Select the manual source type and system source type
            manual_source_type: str = rng.choice(MANUAL_SOURCE_TYPES)
            original_system_source: str = rng.choice(SYSTEM_SOURCE_TYPES)

            # 4. Modify the source/origin of the journal entry
            source_field_found: bool = False
            source_field_name: str = "source_type"  # default if none found

            for field_name in SOURCE_FIELDS:
                if field_name in modified:
                    # Record original value and overwrite
                    original_values[field_name] = modified[field_name]
                    modified[field_name] = manual_source_type
                    modified_values[field_name] = manual_source_type
                    affected_fields.append(field_name)
                    source_field_name = field_name
                    source_field_found = True
                    break

            if not source_field_found:
                # No existing source field — add one
                source_field_name = "source_type"
                original_values[source_field_name] = "NOT_PRESENT"
                modified[source_field_name] = manual_source_type
                modified_values[source_field_name] = manual_source_type
                affected_fields.append(source_field_name)

            # 5. Add override reference to a fake prior system journal entry
            override_ref_field: str = rng.choice(OVERRIDE_REFERENCE_FIELDS)
            fake_je_reference: str = self._generate_je_reference(rng)

            original_values[override_ref_field] = modified.get(
                override_ref_field, "NOT_PRESENT"
            )
            modified[override_ref_field] = fake_je_reference
            modified_values[override_ref_field] = fake_je_reference
            affected_fields.append(override_ref_field)

            # 6. Add override metadata fields
            override_reason: str = rng.choice(OVERRIDE_REASONS)

            # override_reason field
            original_values["override_reason"] = modified.get(
                "override_reason", "NOT_PRESENT"
            )
            modified["override_reason"] = override_reason
            modified_values["override_reason"] = override_reason
            affected_fields.append("override_reason")

            # is_manual_override flag
            original_values["is_manual_override"] = modified.get(
                "is_manual_override", "NOT_PRESENT"
            )
            modified["is_manual_override"] = True
            modified_values["is_manual_override"] = True
            affected_fields.append("is_manual_override")

            # original_source — what the system source was before override
            original_values["original_source"] = modified.get(
                "original_source", "NOT_PRESENT"
            )
            modified["original_source"] = original_system_source
            modified_values["original_source"] = original_system_source
            affected_fields.append("original_source")

            # 7. Add an override timestamp
            override_timestamp: str = datetime.now(timezone.utc).isoformat()
            original_values["override_timestamp"] = modified.get(
                "override_timestamp", "NOT_PRESENT"
            )
            modified["override_timestamp"] = override_timestamp
            modified_values["override_timestamp"] = override_timestamp
            affected_fields.append("override_timestamp")

            # 8. If boolean system-generated flags exist, set them to False
            for bool_field in ("created_by_system", "is_system_generated"):
                if bool_field in modified:
                    original_val = modified[bool_field]
                    if original_val is not False:
                        original_values[bool_field] = original_val
                        modified[bool_field] = False
                        modified_values[bool_field] = False
                        if bool_field not in affected_fields:
                            affected_fields.append(bool_field)

            # 9. Calculate financial impact from the transaction amount
            financial_impact: Decimal = Decimal("0")
            for amount_field in ("amount", "total_amount", "entry_amount"):
                if amount_field in transaction:
                    raw_amount = transaction[amount_field]
                    try:
                        financial_impact = abs(Decimal(str(raw_amount)))
                    except Exception:
                        financial_impact = Decimal("0")
                    break

            # 10. Extract transaction_id for logging
            transaction_id: str = str(
                transaction.get(
                    "transaction_id",
                    transaction.get(
                        "je_id",
                        transaction.get("journal_entry_id", "unknown"),
                    ),
                )
            )

            # 11. Build ground truth data via inherited helper
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=(
                    f"Manual journal entry overriding system-generated "
                    f"entry: {override_reason}"
                ),
                extra_metadata={
                    "original_source_type": original_system_source,
                    "new_source_type": manual_source_type,
                    "override_reason": override_reason,
                    "referenced_je_id": fake_je_reference,
                    "override_reference_field": override_ref_field,
                    "source_field_modified": source_field_name,
                    "source_field_existed": source_field_found,
                    "override_timestamp": override_timestamp,
                },
            )

            # 12. Log the injection event
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
            # Re-raise known injection errors without wrapping
            raise

        except Exception as exc:
            raise DiscrepancyInjectionError(
                f"Failed to inject GL-005 manual override discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "category": self.category,
                    "difficulty": self.difficulty,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Helper: _generate_je_reference()
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_je_reference(rng: random.Random) -> str:
        """Generate a realistic-looking journal entry reference number.

        Produces references in the format ``"JE-YYYY-NNNN"`` where ``YYYY``
        is a reasonable fiscal year (2024–2026) and ``NNNN`` is a zero-padded
        sequential number (0001–9999).

        This reference is used to simulate a reference to a prior system-
        generated journal entry that the manual override is replacing.
        The referenced JE does not need to exist — it serves as metadata
        indicating the override relationship.

        Args:
            rng: Seeded :class:`random.Random` instance for deterministic
                generation.  Using the seeded RNG ensures that identical
                seeds produce identical reference numbers.

        Returns:
            A string in the format ``"JE-YYYY-NNNN"``, e.g. ``"JE-2025-0042"``.
        """
        year: int = rng.randint(2024, 2026)
        sequence_number: int = rng.randint(1, 9999)
        return f"JE-{year}-{sequence_number:04d}"
