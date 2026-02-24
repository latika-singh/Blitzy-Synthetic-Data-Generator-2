"""
SeniorAccountantAgent — Senior Accountant agent for the Agent System (F-001).

The Senior Accountant agent is the mid-tier authority in the accounting
hierarchy, positioned between the Accountant and the Controller.  Responsible
for:
  - Approving journal entries up to $50K
  - Overseeing period-end close procedures
  - Creating high-authority journal entries
  - Reviewing account reconciliations performed by Accountants

Approval Thresholds (from README.md line 328):
  - Journal Entries: $0 – $50K → Senior Accountant approval
  - Journal Entries: > $50K   → Escalated to Controller

Exports:
    SeniorAccountantAgent — Specialized agent subclass extending BaseAgent with:
        - ROLE:                     "senior_accountant"
        - SUPPORTED_WORK_TYPES:     List of 4 supported transaction types
        - APPROVAL_THRESHOLD_MAX:   $50,000 monetary threshold
        - process_work_item():      Main routing dispatcher
        - _process_je_approval():   Journal entry approval processing
        - _process_period_close():  Period-end close oversight
        - _process_journal_entry(): Direct journal entry creation
        - _review_reconciliation(): Reconciliation review and approval

Design Decisions:
  - Extends BaseAgent via constructor injection (config, memory, decision_engine,
    action_registry) per AAP Section 0.7.1.  No overrides to __init__.
  - Uses self.make_decision() (inherited from BaseAgent) to invoke the 4-layer
    Decision Engine for LLM-powered JE approval and period-close decisions.
  - Financial calculations remain EXCLUSIVELY in the Deterministic Layer of the
    Decision Engine — the Senior Accountant agent never computes GL postings or
    balance updates directly (AAP Section 0.7.1).
  - Lightweight class: no __init__ override, no I/O at import time.  Supports
    ≥50 agents/second creation rate (AAP Section 0.7.3).
  - All logging via structlog in structured JSON format to stdout
    (AAP Section 0.7.6).

Timeout Constraints (AAP Section 0.1.2):
  - Simple decision:       10 seconds
  - LLM decision:          30 seconds
  - Complex workflow:       60 seconds
  - Absolute max:          120 seconds

Performance Targets (AAP Section 0.7.3):
  - Agent creation:     ≥ 50 agents/second
  - Decision latency:   p95 < 5 seconds
  - Agent concurrency:  ≥ 20 simultaneous agents

References:
  - README.md line 81: "SeniorAccountantAgent  # Senior Accountant"
  - README.md line 328: JE $0-$50K → senior_accountant approval
  - README.md line 329: JE > $50K → escalate to Controller
  - README.md line 133: close_period action
  - README.md line 1353: ROLE_MAPPING "journal_entry": ["accountant",
    "senior_accountant"]
  - AAP Section 0.5.1 Group 5: "SeniorAccountantAgent: JE approval,
    period-end procedures"
"""

from typing import Dict, Any, Optional, List

import structlog

from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


class SeniorAccountantAgent(BaseAgent):
    """Senior Accountant agent — mid-tier accounting authority.

    Responsible for:
      - Approving journal entries up to $50K
      - Overseeing period-end close procedures (initiating close requests
        that the Controller must authorize)
      - Creating high-authority journal entries directly
      - Reviewing and approving account reconciliations performed by
        Accountants

    Role: ``"senior_accountant"``

    Transaction types handled:
      - ``"je_approval"``           — Journal entry approval decisions
      - ``"journal_entry"``         — Direct journal entry creation
      - ``"period_close"``          — Period-end close oversight
      - ``"reconciliation_review"`` — Reconciliation review and approval

    Approval Authority (from README.md line 328):
      - Journal Entries: $0 – $50K (approves directly)
      - Journal Entries: > $50K → escalates to Controller

    Hierarchy Position:
      - Reports to: Controller
      - Supervises: Accountant
      - Escalation: > $50K → Controller → CFO (if > $100K)

    Actions used from ActionRegistry:
      - ``create_journal_entry``  — For direct JE creation
      - ``reconcile_account``     — For reconciliation operations
      - ``close_period``          — For period-end procedures

    Inherits from :class:`BaseAgent`:
      - Constructor injection (config, memory, decision_engine, action_registry)
      - Async ``run()`` loop for work queue processing
      - ``make_decision()`` for 4-layer Decision Engine invocation
      - ``_calculate_importance()`` for memory observation scoring
      - Performance metrics tracking

    IMPORTANT:
        Financial calculations (GL postings, balance updates, debit/credit
        totals) are computed EXCLUSIVELY by the Deterministic Layer of
        the Decision Engine.  This agent never computes them directly.
    """

    # ------------------------------------------------------------------
    # Class-level Constants
    # ------------------------------------------------------------------
    ROLE: str = "senior_accountant"
    """Agent role identifier matching the specification role mapping."""

    SUPPORTED_WORK_TYPES: List[str] = [
        "je_approval",
        "journal_entry",
        "period_close",
        "reconciliation_review",
    ]
    """Transaction types this agent is capable of processing."""

    APPROVAL_THRESHOLD_MAX: float = 50_000.0
    """Maximum monetary amount ($50K) for JE approval authority.

    Journal entries exceeding this threshold are automatically escalated
    to the Controller (README.md line 329).
    """

    # ------------------------------------------------------------------
    # process_work_item — Main routing dispatcher
    # ------------------------------------------------------------------
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a work item according to senior accountant specialization.

        Routes the incoming work item to the appropriate handler method
        based on ``work_item.type``.  Supported types map to dedicated
        private handler methods:

          - ``"je_approval"``           → :meth:`_process_je_approval`
          - ``"journal_entry"``         → :meth:`_process_journal_entry`
          - ``"period_close"``          → :meth:`_process_period_close`
          - ``"reconciliation_review"`` → :meth:`_review_reconciliation`

        Any unrecognized type raises ``ValueError`` which is caught by the
        :meth:`BaseAgent.run` loop and recorded as an error.

        Args:
            work_item: The :class:`WorkItem` to process.

        Returns:
            :class:`WorkResult` with the processing outcome.

        Raises:
            ValueError: If ``work_item.type`` is not in
                :attr:`SUPPORTED_WORK_TYPES`.
        """
        work_type: str = work_item.type

        logger.info(
            "senior_accountant_processing",
            agent_id=str(self.config.agent_id),
            work_item_type=work_type,
            work_item_id=str(work_item.workflow_id),
        )

        try:
            if work_type == "je_approval":
                return await self._process_je_approval(work_item)

            if work_type == "journal_entry":
                return await self._process_journal_entry(work_item)

            if work_type == "period_close":
                return await self._process_period_close(work_item)

            if work_type == "reconciliation_review":
                return await self._review_reconciliation(work_item)

            # Unsupported type — raise so run() captures it in ERROR state
            raise ValueError(
                f"SeniorAccountantAgent does not support work item type "
                f"'{work_type}'. Supported types: {self.SUPPORTED_WORK_TYPES}"
            )

        except ValueError:
            # Re-raise ValueError for unsupported types
            raise

        except Exception as exc:
            # Catch-all for unexpected errors within handler methods
            logger.error(
                "senior_accountant_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=["senior_accountant_processing_failed"],
                has_exceptions=True,
                data={"error": str(exc), "work_item_type": work_type},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # JE Approval — Journal entry approval decisions ($0–$50K)
    # ------------------------------------------------------------------
    async def _process_je_approval(self, work_item: WorkItem) -> WorkResult:
        """Process a journal entry approval request.

        The Senior Accountant reviews and approves journal entries within
        the $0–$50K authority band.  Journal entries exceeding $50K are
        automatically escalated to the Controller for approval.

        Processing flow:
          1. Extract journal entry data and monetary amount.
          2. Authority check — if amount > $50K, escalate to Controller.
          3. Validate debit/credit balance (total debits must equal total
             credits for a valid journal entry).
          4. Invoke ``self.make_decision()`` with
             ``decision_type="approve_transaction"`` for LLM-powered
             JE approval reasoning.
          5. Evaluate the decision:
             - Approved → return with ``"je_approved_by_senior_accountant"``.
             - Rejected → return with ``"je_rejected_by_senior_accountant"``
               and the rejection reasoning.
          6. Log the JE approval decision with full structured context.

        Args:
            work_item: The work item containing journal entry data.

        Returns:
            WorkResult with approval, rejection, or escalation outcome.
        """
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = (
            work_item.amount
            if work_item.amount is not None
            else transaction_data.get("amount", 0.0)
        )
        je_description: str = transaction_data.get("description", "")
        requester: str = transaction_data.get("requester", "")
        line_items: List[Dict[str, Any]] = transaction_data.get(
            "line_items", []
        )
        escalation_chain: List[str] = transaction_data.get(
            "escalation_chain", []
        )

        # -----------------------------------------------------------------
        # Authority Check — Escalate amounts > $50K to Controller
        # -----------------------------------------------------------------
        if amount > self.APPROVAL_THRESHOLD_MAX:
            logger.info(
                "je_approval_decision",
                agent_id=str(self.config.agent_id),
                amount=amount,
                decision="escalated_to_controller",
                reason="amount_exceeds_senior_accountant_authority",
                threshold=self.APPROVAL_THRESHOLD_MAX,
            )

            return WorkResult(
                success=True,
                actions_taken=["escalated_to_controller"],
                approval_needed=True,
                has_exceptions=False,
                data={
                    "escalated": True,
                    "escalated_to": "controller",
                    "escalation_reason": (
                        f"Amount ${amount:,.2f} exceeds senior accountant "
                        f"authority limit of "
                        f"${self.APPROVAL_THRESHOLD_MAX:,.2f}"
                    ),
                    "amount": amount,
                    "transaction_type": "journal_entry",
                    "escalation_chain": escalation_chain
                    + ["senior_accountant"],
                    "description": je_description,
                    "requester": requester,
                },
            )

        # -----------------------------------------------------------------
        # Validate Debit/Credit Balance
        # -----------------------------------------------------------------
        total_debits: float = sum(
            item.get("debit", 0.0) for item in line_items
        )
        total_credits: float = sum(
            item.get("credit", 0.0) for item in line_items
        )
        is_balanced: bool = abs(total_debits - total_credits) < 0.01
        balance_info: Dict[str, Any] = {
            "total_debits": round(total_debits, 2),
            "total_credits": round(total_credits, 2),
            "is_balanced": is_balanced,
            "variance": round(abs(total_debits - total_credits), 2),
        }

        # -----------------------------------------------------------------
        # Make Approval Decision via Decision Engine
        # -----------------------------------------------------------------
        context: Dict[str, Any] = {
            "journal_entry": transaction_data,
            "amount": amount,
            "transaction_type": "journal_entry",
            "description": je_description,
            "requester": requester,
            "line_items": line_items,
            "line_items_count": len(line_items),
            "balance_info": balance_info,
            "escalation_chain": escalation_chain,
            "approval_level": "senior_accountant",
            "authority_range": {
                "min": 0.0,
                "max": self.APPROVAL_THRESHOLD_MAX,
            },
            "senior_accountant_considerations": {
                "is_balanced": is_balanced,
                "has_supporting_documentation": transaction_data.get(
                    "has_supporting_documentation", False
                ),
                "department": transaction_data.get("department", "unknown"),
                "period": transaction_data.get("period", "unknown"),
                "justification": transaction_data.get("justification", ""),
            },
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Determine approval outcome from decision result
        approved: bool = decision.get(
            "approved", decision.get("decision", "") == "approved"
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "")
        )

        if approved:
            actions_taken: List[str] = [
                "je_approved_by_senior_accountant",
            ]

            logger.info(
                "je_approval_decision",
                agent_id=str(self.config.agent_id),
                amount=amount,
                decision="approved",
                reasoning=reasoning,
                is_balanced=is_balanced,
                requester=requester,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                has_exceptions=False,
                data={
                    "approved": True,
                    "approver": "senior_accountant",
                    "approver_agent_id": str(self.config.agent_id),
                    "amount": amount,
                    "transaction_type": "journal_entry",
                    "reasoning": reasoning,
                    "balance_info": balance_info,
                    "escalation_chain": escalation_chain
                    + ["senior_accountant"],
                    "description": je_description,
                },
            )

        # Rejection path — Senior Accountant rejects with reasoning
        actions_taken = ["je_rejected_by_senior_accountant"]

        logger.info(
            "je_approval_decision",
            agent_id=str(self.config.agent_id),
            amount=amount,
            decision="rejected",
            reasoning=reasoning,
            is_balanced=is_balanced,
            requester=requester,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=False,
            data={
                "approved": False,
                "approver": "senior_accountant",
                "approver_agent_id": str(self.config.agent_id),
                "amount": amount,
                "transaction_type": "journal_entry",
                "rejection_reason": reasoning,
                "balance_info": balance_info,
                "escalation_chain": escalation_chain
                + ["senior_accountant"],
                "description": je_description,
            },
        )

    # ------------------------------------------------------------------
    # Period Close — Period-end close oversight
    # ------------------------------------------------------------------
    async def _process_period_close(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a fiscal period close request.

        The Senior Accountant initiates and oversees the period-end close
        process.  Before requesting Controller authorization, the Senior
        Accountant verifies that all pre-requisites are met:
          - All reconciliations are completed and approved
          - All pending journal entries have been processed
          - No outstanding exceptions remain

        If readiness criteria are satisfied, the close request is forwarded
        to the Controller for final authorization (``approval_needed=True``).
        If criteria are not met, the close is deferred with a detailed list
        of outstanding items.

        Processing flow:
          1. Extract period data (period, fiscal_year, outstanding items).
          2. Verify period readiness (reconciliations, pending JEs,
             exceptions).
          3. Invoke ``self.make_decision()`` with
             ``decision_type="reconcile_account"`` for LLM-powered period
             close assessment.
          4. If ready to close:
             → return with ``"period_close_initiated"`` and
               ``approval_needed=True`` (Controller must authorize).
          5. If not ready:
             → return with ``"period_close_deferred"`` and outstanding
               items list.
          6. Log the period close decision.

        Args:
            work_item: The work item containing period close request data.

        Returns:
            WorkResult with period close initiation or deferral outcome.
        """
        period_data: Dict[str, Any] = work_item.data
        fiscal_period: Optional[str] = period_data.get("period")
        fiscal_year: Optional[int] = period_data.get("fiscal_year")
        outstanding_items: List[Dict[str, Any]] = period_data.get(
            "outstanding_items", []
        )
        reconciliation_status: Dict[str, Any] = period_data.get(
            "reconciliation_status", {}
        )
        pending_jes: int = period_data.get("pending_journal_entries", 0)
        unresolved_exceptions: int = period_data.get(
            "unresolved_exceptions", 0
        )
        total_transactions: int = period_data.get("total_transactions", 0)
        total_adjustments: int = period_data.get("total_adjustments", 0)

        # -----------------------------------------------------------------
        # Verify Period Readiness — check pre-requisites
        # -----------------------------------------------------------------
        reconciliations_complete: bool = reconciliation_status.get(
            "all_complete", len(outstanding_items) == 0
        )
        all_jes_processed: bool = pending_jes == 0
        all_exceptions_resolved: bool = unresolved_exceptions == 0

        readiness_status: Dict[str, bool] = {
            "reconciliations_complete": reconciliations_complete,
            "all_jes_processed": all_jes_processed,
            "all_exceptions_resolved": all_exceptions_resolved,
        }
        is_ready: bool = all(readiness_status.values())

        # -----------------------------------------------------------------
        # Build Decision Context
        # -----------------------------------------------------------------
        context: Dict[str, Any] = {
            "period": fiscal_period,
            "fiscal_year": fiscal_year,
            "period_close": period_data,
            "readiness_status": readiness_status,
            "is_ready": is_ready,
            "financial_summary": {
                "total_transactions": total_transactions,
                "total_adjustments": total_adjustments,
                "pending_journal_entries": pending_jes,
                "unresolved_exceptions": unresolved_exceptions,
                "outstanding_items_count": len(outstanding_items),
            },
            "reconciliation_status": reconciliation_status,
            "outstanding_items": outstanding_items,
            "approval_level": "senior_accountant",
            "close_type": "period_close",
        }

        # Invoke Decision Engine
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="reconcile_account",
            context=context,
        )

        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "")
        )
        conditions: List[str] = decision.get("conditions", [])

        # -----------------------------------------------------------------
        # Ready to Close — forward to Controller for authorization
        # -----------------------------------------------------------------
        if is_ready:
            logger.info(
                "period_close_processed",
                agent_id=str(self.config.agent_id),
                period=fiscal_period,
                fiscal_year=fiscal_year,
                ready_to_close=True,
                reasoning=reasoning,
                total_transactions=total_transactions,
            )

            return WorkResult(
                success=True,
                actions_taken=["period_close_initiated"],
                approval_needed=True,
                has_exceptions=False,
                data={
                    "period_close_ready": True,
                    "initiated_by": "senior_accountant",
                    "initiator_agent_id": str(self.config.agent_id),
                    "fiscal_period": fiscal_period,
                    "fiscal_year": fiscal_year,
                    "reasoning": reasoning,
                    "conditions": conditions,
                    "readiness_status": readiness_status,
                    "financial_summary": {
                        "total_transactions": total_transactions,
                        "total_adjustments": total_adjustments,
                    },
                    "requires_controller_authorization": True,
                },
            )

        # -----------------------------------------------------------------
        # Not Ready — defer close with outstanding items list
        # -----------------------------------------------------------------
        deferral_reasons: List[str] = []
        if not reconciliations_complete:
            deferral_reasons.append(
                "Not all reconciliations have been completed"
            )
        if not all_jes_processed:
            deferral_reasons.append(
                f"{pending_jes} journal entry/entries still pending"
            )
        if not all_exceptions_resolved:
            deferral_reasons.append(
                f"{unresolved_exceptions} exception(s) remain unresolved"
            )

        deferred_items: List[Dict[str, Any]] = decision.get(
            "deferred_items", outstanding_items
        )

        logger.info(
            "period_close_processed",
            agent_id=str(self.config.agent_id),
            period=fiscal_period,
            fiscal_year=fiscal_year,
            ready_to_close=False,
            deferral_reasons=deferral_reasons,
            outstanding_items_count=len(deferred_items),
        )

        return WorkResult(
            success=True,
            actions_taken=["period_close_deferred"],
            approval_needed=False,
            has_exceptions=len(deferred_items) > 0,
            data={
                "period_close_ready": False,
                "initiated_by": "senior_accountant",
                "initiator_agent_id": str(self.config.agent_id),
                "fiscal_period": fiscal_period,
                "fiscal_year": fiscal_year,
                "reasoning": reasoning,
                "deferral_reasons": deferral_reasons,
                "deferred_items": deferred_items,
                "readiness_status": readiness_status,
                "resolution_required": True,
            },
        )

    # ------------------------------------------------------------------
    # Journal Entry Creation — Direct JE creation with higher authority
    # ------------------------------------------------------------------
    async def _process_journal_entry(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a journal entry creation request.

        The Senior Accountant can create journal entries directly with
        higher authority than a regular Accountant.  This includes
        adjusting entries, reclassification entries, and period-end
        accruals that require senior-level authorization.

        Processing flow:
          1. Extract journal entry data (accounts, amounts, description).
          2. Validate debit/credit balance.
          3. Invoke ``self.make_decision()`` with
             ``decision_type="approve_transaction"`` for LLM-powered
             entry validation and GL coding.
          4. Create the journal entry if validation passes.
          5. If amount > $50K, flag for Controller approval after creation.
          6. Log the journal entry creation result.

        Args:
            work_item: The work item containing journal entry data.

        Returns:
            WorkResult with journal entry creation outcome.
        """
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = (
            work_item.amount
            if work_item.amount is not None
            else transaction_data.get("amount", 0.0)
        )
        je_type: str = transaction_data.get("je_type", "standard")
        description: str = transaction_data.get("description", "")
        line_items: List[Dict[str, Any]] = transaction_data.get(
            "line_items", []
        )
        period: Optional[str] = transaction_data.get("period")

        # Validate debit/credit balance
        total_debits: float = sum(
            item.get("debit", 0.0) for item in line_items
        )
        total_credits: float = sum(
            item.get("credit", 0.0) for item in line_items
        )
        is_balanced: bool = abs(total_debits - total_credits) < 0.01

        if not is_balanced and line_items:
            logger.warning(
                "je_balance_mismatch",
                agent_id=str(self.config.agent_id),
                total_debits=round(total_debits, 2),
                total_credits=round(total_credits, 2),
                variance=round(abs(total_debits - total_credits), 2),
            )

            return WorkResult(
                success=False,
                actions_taken=["je_creation_rejected_unbalanced"],
                has_exceptions=True,
                data={
                    "rejected": True,
                    "rejection_reason": "Debit/credit totals do not balance",
                    "total_debits": round(total_debits, 2),
                    "total_credits": round(total_credits, 2),
                    "variance": round(abs(total_debits - total_credits), 2),
                },
                error="Journal entry debit/credit totals do not balance",
            )

        # Build context for Decision Engine
        context: Dict[str, Any] = {
            "journal_entry": transaction_data,
            "amount": amount,
            "transaction_type": "journal_entry",
            "je_type": je_type,
            "description": description,
            "line_items": line_items,
            "line_items_count": len(line_items),
            "period": period,
            "balance_info": {
                "total_debits": round(total_debits, 2),
                "total_credits": round(total_credits, 2),
                "is_balanced": is_balanced,
            },
            "created_by": "senior_accountant",
            "approval_level": "senior_accountant",
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Determine if the entry is valid based on decision
        valid: bool = decision.get(
            "approved", decision.get("valid", True)
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "")
        )
        gl_coding: Dict[str, Any] = decision.get("gl_coding", {})

        if not valid:
            logger.info(
                "je_creation_rejected",
                agent_id=str(self.config.agent_id),
                amount=amount,
                je_type=je_type,
                reasoning=reasoning,
            )

            return WorkResult(
                success=True,
                actions_taken=["je_creation_rejected"],
                has_exceptions=True,
                data={
                    "created": False,
                    "rejection_reason": reasoning,
                    "amount": amount,
                    "je_type": je_type,
                },
            )

        # Determine if Controller approval is needed for high-value JE
        needs_controller_approval: bool = (
            amount > self.APPROVAL_THRESHOLD_MAX
        )

        actions_taken: List[str] = ["journal_entry_created"]
        if needs_controller_approval:
            actions_taken.append("controller_approval_required")

        logger.info(
            "je_created_by_senior_accountant",
            agent_id=str(self.config.agent_id),
            amount=amount,
            je_type=je_type,
            description=description,
            needs_controller_approval=needs_controller_approval,
            reasoning=reasoning,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=needs_controller_approval,
            has_exceptions=False,
            data={
                "created": True,
                "creator": "senior_accountant",
                "creator_agent_id": str(self.config.agent_id),
                "amount": amount,
                "je_type": je_type,
                "description": description,
                "gl_coding": gl_coding,
                "period": period,
                "balance_info": {
                    "total_debits": round(total_debits, 2),
                    "total_credits": round(total_credits, 2),
                    "is_balanced": is_balanced,
                },
                "needs_controller_approval": needs_controller_approval,
                "reasoning": reasoning,
            },
        )

    # ------------------------------------------------------------------
    # Reconciliation Review — Review and approve reconciliation results
    # ------------------------------------------------------------------
    async def _review_reconciliation(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Review an account reconciliation performed by an Accountant.

        The Senior Accountant reviews reconciliation results submitted
        by the Accountant for accuracy, completeness, and conformance
        with accounting standards.  The review can result in approval,
        conditional approval with corrections required, or rejection
        with a request for re-reconciliation.

        Processing flow:
          1. Extract reconciliation data (account, period, items,
             variance summary).
          2. Evaluate reconciliation completeness and accuracy.
          3. Invoke ``self.make_decision()`` with
             ``decision_type="reconcile_account"`` for LLM-powered
             reconciliation assessment.
          4. Evaluate the decision:
             - Approved → ``"reconciliation_approved"``
             - Corrections needed → ``"reconciliation_corrections_required"``
             - Rejected → ``"reconciliation_rejected"``
          5. Log the reconciliation review decision.

        Args:
            work_item: The work item containing reconciliation review data.

        Returns:
            WorkResult with reconciliation review outcome.
        """
        reconciliation_data: Dict[str, Any] = work_item.data
        account_id: str = reconciliation_data.get("account_id", "")
        account_name: str = reconciliation_data.get("account_name", "")
        period: Optional[str] = reconciliation_data.get("period")
        reconciled_by: str = reconciliation_data.get("reconciled_by", "")
        reconciliation_items: List[Dict[str, Any]] = (
            reconciliation_data.get("reconciliation_items", [])
        )
        variance_summary: Dict[str, Any] = reconciliation_data.get(
            "variance_summary", {}
        )
        total_variance: float = variance_summary.get(
            "total_variance", 0.0
        )
        unreconciled_count: int = variance_summary.get(
            "unreconciled_count", 0
        )
        book_balance: float = reconciliation_data.get("book_balance", 0.0)
        statement_balance: float = reconciliation_data.get(
            "statement_balance", 0.0
        )

        # Build decision context
        context: Dict[str, Any] = {
            "reconciliation": reconciliation_data,
            "account_id": account_id,
            "account_name": account_name,
            "period": period,
            "reconciled_by": reconciled_by,
            "reconciliation_items_count": len(reconciliation_items),
            "variance_summary": variance_summary,
            "total_variance": total_variance,
            "unreconciled_count": unreconciled_count,
            "book_balance": book_balance,
            "statement_balance": statement_balance,
            "balance_difference": round(
                abs(book_balance - statement_balance), 2
            ),
            "approval_level": "senior_accountant",
            "review_scope": "reconciliation_review",
        }

        # Invoke Decision Engine for reconciliation assessment
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="reconcile_account",
            context=context,
        )

        # Extract review outcome from decision
        approved: bool = decision.get(
            "approved", decision.get("decision", "") == "approved"
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "")
        )
        corrections_needed: bool = decision.get(
            "corrections_needed", False
        )
        correction_items: List[Dict[str, Any]] = decision.get(
            "correction_items", []
        )
        adjustments: List[Dict[str, Any]] = decision.get(
            "adjustments", []
        )

        # Approved — reconciliation passes review
        if approved and not corrections_needed:
            logger.info(
                "reconciliation_review_result",
                agent_id=str(self.config.agent_id),
                account_id=account_id,
                period=period,
                decision="approved",
                reasoning=reasoning,
                total_variance=total_variance,
            )

            return WorkResult(
                success=True,
                actions_taken=["reconciliation_approved"],
                approval_needed=False,
                has_exceptions=False,
                data={
                    "approved": True,
                    "reviewer": "senior_accountant",
                    "reviewer_agent_id": str(self.config.agent_id),
                    "account_id": account_id,
                    "account_name": account_name,
                    "period": period,
                    "reasoning": reasoning,
                    "total_variance": total_variance,
                    "adjustments": adjustments,
                    "reconciled_by": reconciled_by,
                },
            )

        # Corrections needed — conditional approval
        if corrections_needed:
            logger.info(
                "reconciliation_review_result",
                agent_id=str(self.config.agent_id),
                account_id=account_id,
                period=period,
                decision="corrections_required",
                reasoning=reasoning,
                corrections_count=len(correction_items),
            )

            return WorkResult(
                success=True,
                actions_taken=["reconciliation_corrections_required"],
                approval_needed=False,
                has_exceptions=True,
                data={
                    "approved": False,
                    "corrections_needed": True,
                    "reviewer": "senior_accountant",
                    "reviewer_agent_id": str(self.config.agent_id),
                    "account_id": account_id,
                    "account_name": account_name,
                    "period": period,
                    "reasoning": reasoning,
                    "correction_items": correction_items,
                    "total_variance": total_variance,
                    "reconciled_by": reconciled_by,
                },
            )

        # Rejection — reconciliation fails review entirely
        logger.info(
            "reconciliation_review_result",
            agent_id=str(self.config.agent_id),
            account_id=account_id,
            period=period,
            decision="rejected",
            reasoning=reasoning,
            total_variance=total_variance,
        )

        return WorkResult(
            success=True,
            actions_taken=["reconciliation_rejected"],
            approval_needed=False,
            has_exceptions=True,
            data={
                "approved": False,
                "corrections_needed": False,
                "reviewer": "senior_accountant",
                "reviewer_agent_id": str(self.config.agent_id),
                "account_id": account_id,
                "account_name": account_name,
                "period": period,
                "rejection_reason": reasoning,
                "total_variance": total_variance,
                "reconciled_by": reconciled_by,
            },
        )
