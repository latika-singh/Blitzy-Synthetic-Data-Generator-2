"""
Orchestration package (F-003 Workflow Orchestration, F-004 Time Controller).

Manages workflow routing, approval chains, transaction lifecycle tracking,
and simulation time progression. This package bridges the Agent System and
Event System to coordinate business process execution.

Core components:
- WorkflowOrchestrator: Routes transactions to appropriate agents via role mapping
- TransactionOrchestrator: Tracks transaction artifact completeness and lifecycle
- ApprovalSystem: Enforces monetary threshold-based approval chains
- TimeController: Manages simulation time with business and fiscal calendars
- BusinessCalendar: US Federal holidays, working hours, and half days
- FiscalCalendar: Configurable fiscal periods with open/closing/closed lifecycle
"""

from __future__ import annotations

# --- Available submodule imports ---
# BusinessCalendar is always available (foundational, no internal deps)
from app.orchestration.business_calendar import BusinessCalendar

# --- Lazy imports for submodules created by other agents ---
# These use try/except so the package remains importable even when
# sibling modules have not yet been created by their respective agents.

try:
    from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod
except ImportError:  # pragma: no cover
    pass

try:
    from app.orchestration.time_controller import TimeController
except ImportError:  # pragma: no cover
    pass

try:
    from app.orchestration.approval_system import (
        ApprovalSystem,
        ApprovalRequest,
        ApprovalDecision,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from app.orchestration.transaction_orchestrator import (
        TransactionOrchestrator,
        TransactionState,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from app.orchestration.workflow_orchestrator import (
        WorkflowOrchestrator,
        WorkflowConfig,
        WorkflowInstance,
    )
except ImportError:  # pragma: no cover
    pass

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
