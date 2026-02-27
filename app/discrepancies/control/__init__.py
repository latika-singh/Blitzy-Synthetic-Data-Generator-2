"""Control Discrepancy Types sub-package — 5 control discrepancy implementations (CTL-001 through CTL-005).

This sub-package contains discrepancy types that simulate internal control
failures within the ERP system. Unlike P2P, O2C, and GL discrepancies that
target transactional data, control discrepancies target the *process* and
*governance* surrounding transactions.

IMPORTANT (AAP §0.6.2): These discrepancy types *simulate* control failures
as synthetic training data for audit detection models. They do NOT enforce
segregation of duties or approval controls — enforcement is explicitly out of
scope for MVP.

Discrepancy Types:
    CTL-001: SoDViolation
        Segregation of Duties Violation — assigns same user to roles that should
        be separated (e.g., requestor and approver). Difficulty: easy.
        Detection method: access_review.

    CTL-002: SelfApproval
        Same User Created and Approved — sets the approver_id to match the
        creator_id on a transaction. Difficulty: easy.
        Detection method: approval_check.

    CTL-003: ApprovalLimitExceeded
        Approval Limit Exceeded — modifies approved_by to a role whose
        approval authority is below the transaction amount. Difficulty: easy.
        Detection method: approval_check.
        Parameters: excess_percent (1-100).

    CTL-004: BackdatedTransaction
        Backdated Transaction — sets the transaction date to a past date
        relative to the posting date. Difficulty: medium.
        Detection method: date_check.
        Parameters: days_back (1-90).

    CTL-005: HolidayTransaction
        Transaction on Holiday/Weekend — sets the transaction processing date
        to a weekend or US Federal holiday. Difficulty: easy.
        Detection method: date_check.

All classes extend ``BaseDiscrepancy`` from ``app.discrepancies.base_discrepancy``
and implement the ``inject()`` method.

Architecture:
    - Constructor injection (ADR-003) for all dependencies
    - Seeded ``random.Random`` for deterministic reproducibility (NEVER module-level RNG)
    - ``Decimal`` for all financial calculations (prec=28, ROUND_HALF_UP)
    - ``structlog`` for JSON structured logging to stdout
    - Pydantic V2 data contracts at boundaries

References:
    - AAP Section 0.5.1 Group 5: Control discrepancies (CTL-001 through CTL-005)
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.6.2: CTL-001 simulates SoD, does NOT enforce
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CTL-001: Segregation of Duties Violation
# ---------------------------------------------------------------------------
from app.discrepancies.control.sod_violation import SoDViolation

# ---------------------------------------------------------------------------
# CTL-002: Same User Created and Approved
# ---------------------------------------------------------------------------
from app.discrepancies.control.self_approval import SelfApproval

# ---------------------------------------------------------------------------
# CTL-003: Approval Limit Exceeded
# ---------------------------------------------------------------------------
from app.discrepancies.control.approval_limit_exceeded import ApprovalLimitExceeded

# ---------------------------------------------------------------------------
# CTL-004: Backdated Transaction
# ---------------------------------------------------------------------------
from app.discrepancies.control.backdated_transaction import BackdatedTransaction

# ---------------------------------------------------------------------------
# CTL-005: Transaction on Holiday/Weekend
# ---------------------------------------------------------------------------
from app.discrepancies.control.holiday_transaction import HolidayTransaction

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__: list[str] = [
    "SoDViolation",
    "SelfApproval",
    "ApprovalLimitExceeded",
    "BackdatedTransaction",
    "HolidayTransaction",
]
