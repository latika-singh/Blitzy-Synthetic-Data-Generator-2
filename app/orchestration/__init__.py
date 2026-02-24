"""
Orchestration package (F-003 Workflow Orchestration, F-004 Time Controller).

Manages workflow routing, approval chains, transaction lifecycle tracking,
and simulation time progression. This package bridges the Agent System and
Event System to coordinate business process execution.

Core components:
    WorkflowOrchestrator
        Routes transactions to appropriate agents via role mapping with
        idle-agent selection (smallest queue depth).  Supports approval
        chain integration and event-driven coordination.
    TransactionOrchestrator
        Tracks required artifacts per transaction type and manages the
        full transaction lifecycle: register → add_artifact →
        check_completeness → mark_complete.
    ApprovalSystem
        Enforces monetary threshold-based approval chains for purchase
        orders ($5K/$25K/$100K), vendor invoices ($10K/$50K/$100K), and
        journal entries ($50K).
    TimeController
        Manages simulation time progression by integrating
        BusinessCalendar and FiscalCalendar for day advancement,
        working-hour queries, and fiscal-period transitions.
    BusinessCalendar
        US Federal holidays (2024-2026) via the ``holidays`` library,
        company holidays, half days, and standard working hours
        (8:00-17:00 with 12:00-13:00 lunch break).
    FiscalCalendar
        Configurable fiscal year start month with monthly/quarterly
        periods and an open → closing → closed lifecycle per period.

Design Decisions:
    - All imports are direct (no lazy try/except guards) because every
      submodule in this package is a required component of the
      orchestration layer.
    - No objects are instantiated at import time — only class/enum
      references are re-exported (AAP Section 0.7.1, constructor
      injection).
    - No I/O is performed at import time (no Redis connections, no
      file reads, no network calls).

References:
    - AAP Section 0.5.1 Group 6 (Orchestration Layer)
    - AAP Section 0.7.1 (constructor injection, EventBus notifications)
    - README.md lines 277-338 (orchestration specification)
"""

from app.orchestration.approval_system import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalSystem,
)
from app.orchestration.business_calendar import BusinessCalendar
from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod
from app.orchestration.time_controller import TimeController
from app.orchestration.transaction_orchestrator import (
    TransactionOrchestrator,
    TransactionState,
)
from app.orchestration.workflow_orchestrator import (
    WorkflowConfig,
    WorkflowInstance,
    WorkflowOrchestrator,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalSystem",
    "BusinessCalendar",
    "FiscalCalendar",
    "FiscalPeriod",
    "TimeController",
    "TransactionOrchestrator",
    "TransactionState",
    "WorkflowConfig",
    "WorkflowInstance",
    "WorkflowOrchestrator",
]
