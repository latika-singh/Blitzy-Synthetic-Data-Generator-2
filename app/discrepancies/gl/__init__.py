"""General Ledger Discrepancy Types sub-package — 5 GL discrepancy implementations (GL-001 through GL-005).

This sub-package contains discrepancy types that target General Ledger operations
within the ERP system. These discrepancies simulate accounting anomalies and
control weaknesses in journal entry processing, account management, and
period-end adjustments.

Discrepancy Types:
    GL-001: UnbalancedJournal
        Unbalanced Journal Entry — introduces a debit/credit imbalance into a
        journal entry so that SUM(debits) ≠ SUM(credits). Violates the fundamental
        GL balance invariant (AAP §0.7.2). Difficulty: easy.
        Detection method: balance_check.
        Parameters: imbalance_amount (Decimal 0.02–1000.00).

    GL-002: JournalNoApproval
        Journal Entry Without Approval — removes or clears the approval reference
        on a journal entry that requires approval per the threshold configuration
        (all JEs need senior_accountant approval; >$50K need controller).
        Difficulty: easy. Detection method: approval_check.
        Parameters: None.

    GL-003: SuspiciousAdjusting
        Period-End Adjusting Entry — creates or modifies a journal entry so that
        it falls within the last few days before period close, a common pattern
        for suspicious period-end manipulations. Difficulty: medium.
        Detection method: period_analysis.
        Parameters: days_before_close (1–5).

    GL-004: UnusualAccountCombo
        Unusual Account Combination — modifies journal entry line items to use
        an account combination that is unusual or unexpected (e.g., debiting
        Revenue and crediting an Asset). Difficulty: medium.
        Detection method: pattern_analysis.
        Parameters: None.

    GL-005: ManualOverride
        Manual Entry Overriding System Entry — marks or modifies a journal entry
        to appear as a manual override of an automated/system-generated entry.
        Difficulty: medium. Detection method: system_log_review.
        Parameters: None.

All classes extend ``BaseDiscrepancy`` from ``app.discrepancies.base_discrepancy``
and implement the ``inject()`` method returning
``Tuple[Dict[str, Any], Dict[str, Any]]`` (modified_transaction, ground_truth_data).

Architecture:
    - Constructor injection (ADR-003) for all dependencies
    - Seeded ``random.Random`` for deterministic reproducibility (NEVER module-level RNG)
    - ``Decimal`` for all financial calculations (prec=28, ROUND_HALF_UP)
    - ``structlog`` for JSON structured logging to stdout
    - Pydantic V2 data contracts at boundaries

References:
    - AAP Section 0.5.1 Group 5: GL discrepancies (GL-001 through GL-005)
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Financial Integrity Rules (GL Balance Invariant)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GL-001: Unbalanced Journal Entry
# ---------------------------------------------------------------------------
from app.discrepancies.gl.unbalanced_journal import UnbalancedJournal

# ---------------------------------------------------------------------------
# GL-002: Journal Entry Without Approval
# ---------------------------------------------------------------------------
from app.discrepancies.gl.journal_no_approval import JournalNoApproval

# ---------------------------------------------------------------------------
# GL-003: Period-End Adjusting Entry (Suspicious)
# ---------------------------------------------------------------------------
from app.discrepancies.gl.suspicious_adjusting import SuspiciousAdjusting

# ---------------------------------------------------------------------------
# GL-004: Unusual Account Combination
# ---------------------------------------------------------------------------
from app.discrepancies.gl.unusual_account_combo import UnusualAccountCombo

# ---------------------------------------------------------------------------
# GL-005: Manual Entry Overriding System Entry
# ---------------------------------------------------------------------------
from app.discrepancies.gl.manual_override import ManualOverride

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__: list[str] = [
    "UnbalancedJournal",
    "JournalNoApproval",
    "SuspiciousAdjusting",
    "UnusualAccountCombo",
    "ManualOverride",
]
