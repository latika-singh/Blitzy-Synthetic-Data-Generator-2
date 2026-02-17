"""Controller Agent — High-value approvals and period close oversight.

This module implements the :class:`ControllerAgent`, the second-highest
approval authority in the ERP simulation, responsible for high-value
transaction approvals, fiscal period close oversight, and escalation
review from managers and senior accountants.

The ControllerAgent sits **below the CFO** and **above departmental
managers** in the approval hierarchy.  Its monetary authority is defined
per transaction type:

    * **Purchase Orders**: $25,000 – $100,000
    * **Vendor Invoices**: $50,000 – $100,000
    * **Journal Entries**: $50,000+ (controller is the terminal approver
      for journal entries — no CFO escalation for JE)
    * Amounts **above $100,000** on POs and invoices are **escalated to
      the CFO** automatically.

Specification references:
    * README.md line 80  — ControllerAgent role definition
    * README.md line 318 — PO approval: $25K–$100K → controller
    * README.md line 324 — Invoice approval: $50K–$100K → controller
    * README.md line 329 — JE approval: $50K+ → controller
    * AAP Section 0.5.1 Group 5 — "ControllerAgent: high-value approvals,
      period close oversight"
    * AAP Section 0.7.1 — Must extend ``BaseAgent``, constructor injection,
      ``structlog`` logging

Exports:
    ControllerAgent — Specialized agent class for controller role.
"""

import asyncio
from typing import Dict, Any, Optional, List

import structlog

from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# ControllerAgent — High-value approvals and period close oversight
# ---------------------------------------------------------------------------
class ControllerAgent(BaseAgent):
    """Controller agent for high-value approvals and period close oversight.

    The Controller is the **second-highest** approval authority in the
    ERP hierarchy, positioned below the CFO and above departmental
    managers (Purchasing Manager, AP Manager, Senior Accountant).

    Responsibilities:
        - **High-value transaction approvals** across all transaction types
          within the controller's monetary authority band.
        - **Period close oversight** — final review and approval before
          fiscal periods transition from ``closing`` to ``closed``.
        - **Escalation review** — resolving issues escalated by managers
          and senior accountants that exceed their authority or require
          cross-departmental judgment.
        - **CFO escalation** — automatically escalating POs and invoices
          exceeding $100K to the CFO for strategic approval.

    Role: ``"controller"``

    Transaction types handled:
        - ``"high_value_approval"`` — generic high-value approval routing
        - ``"po_approval"`` — purchase order approval ($25K–$100K)
        - ``"invoice_approval"`` — vendor invoice approval ($50K–$100K)
        - ``"je_approval"`` — journal entry approval ($50K+)
        - ``"period_close_approval"`` — fiscal period close authorization
        - ``"escalation_review"`` — manager/accountant escalation review

    Approval authority (from README.md lines 318, 324, 329):

        +---------------------+----------+----------+
        | Transaction Type    | Min ($)  | Max ($)  |
        +=====================+==========+==========+
        | Purchase Order      |  25,000  | 100,000  |
        +---------------------+----------+----------+
        | Vendor Invoice      |  50,000  | 100,000  |
        +---------------------+----------+----------+
        | Journal Entry       |  50,000  |   ∞      |
        +---------------------+----------+----------+

        Amounts exceeding $100K on POs and invoices are **escalated to
        the CFO**. Journal entries have **no upper limit** because the
        controller is the terminal approver for JEs.

    Inherits from :class:`BaseAgent`:
        - Constructor injection (config, memory, decision_engine, action_registry)
        - Async ``run()`` loop for continuous work item processing
        - ``make_decision()`` for 4-layer Decision Engine delegation
        - ``_calculate_importance()`` for memory scoring
        - Performance metrics tracking

    Example::

        controller = ControllerAgent(
            config=controller_config,
            memory=controller_memory,
            decision_engine=decision_engine,
            action_registry=action_registry,
        )
        # Enqueue a high-value PO for controller approval
        await controller.work_queue.put(WorkItem(
            type="po_approval",
            data={"transaction_type": "purchase_order", "amount": 75000.0},
            amount=75000.0,
        ))
        await controller.run()
    """

    # ------------------------------------------------------------------
    # Class Constants
    # ------------------------------------------------------------------

    ROLE: str = "controller"
    """Agent role identifier. Must match ``"controller"`` in VALID_AGENT_ROLES
    (agent_config.py line 118)."""

    SUPPORTED_WORK_TYPES: List[str] = [
        "high_value_approval",
        "period_close_approval",
        "escalation_review",
        "po_approval",
        "invoice_approval",
        "je_approval",
    ]
    """Work item types this agent can process. Used by the
    WorkflowOrchestrator for routing validation."""

    CFO_THRESHOLD: float = 100_000.0
    """Monetary threshold ($100K) above which the Controller **must**
    escalate PO and Invoice approvals to the CFO.
    From README.md lines 319 (PO > $100K → CFO) and 325
    (Invoice > $100K → CFO)."""

    AUTHORITY: Dict[str, Dict[str, float]] = {
        "purchase_order": {"min": 25_000.0, "max": 100_000.0},
        "vendor_invoice": {"min": 50_000.0, "max": 100_000.0},
        "journal_entry": {"min": 50_000.0, "max": float("inf")},
    }
    """Per-transaction-type monetary authority ranges for the controller.

    * Purchase Orders: $25K – $100K (README.md line 318)
    * Vendor Invoices: $50K – $100K (README.md line 324)
    * Journal Entries: $50K+ with no upper limit (README.md line 329)

    Amounts above the ``max`` for POs and invoices are escalated to CFO.
    Journal entries have ``max=inf`` because the controller is the
    terminal approver for JEs.
    """

    # ------------------------------------------------------------------
    # process_work_item — Main routing dispatcher (abstract method impl)
    # ------------------------------------------------------------------
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Route a work item to the appropriate controller processing method.

        Dispatches based on ``work_item.type``:
          - ``"high_value_approval"`` | ``"po_approval"`` | ``"invoice_approval"``
            | ``"je_approval"``
            → :meth:`_process_approval`
          - ``"period_close_approval"``
            → :meth:`_process_period_close_approval`
          - ``"escalation_review"``
            → :meth:`_review_escalation`

        Unsupported types raise :class:`ValueError` so the base-class
        ``run()`` loop can capture and log the error.

        Args:
            work_item: The work item received from the orchestration layer.

        Returns:
            WorkResult indicating the approval decision, actions taken,
            and any supplementary result data.

        Raises:
            ValueError: If ``work_item.type`` is not in
                :attr:`SUPPORTED_WORK_TYPES`.
        """
        logger.info(
            "controller_processing",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            work_item_type=work_item.type,
            amount=work_item.amount,
            description=work_item.description,
        )

        work_type: str = work_item.type

        try:
            if work_type in (
                "high_value_approval",
                "po_approval",
                "invoice_approval",
                "je_approval",
            ):
                return await self._process_approval(work_item)

            if work_type == "period_close_approval":
                return await self._process_period_close_approval(work_item)

            if work_type == "escalation_review":
                return await self._review_escalation(work_item)

            # Unsupported type — raise so run() captures it in ERROR state
            raise ValueError(
                f"ControllerAgent does not support work item type '{work_type}'. "
                f"Supported types: {self.SUPPORTED_WORK_TYPES}"
            )

        except ValueError:
            # Re-raise ValueError for unsupported types
            raise

        except Exception as exc:
            # Catch-all for unexpected errors within handler methods
            logger.error(
                "controller_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=["controller_processing_failed"],
                has_exceptions=True,
                data={"error": str(exc), "work_item_type": work_type},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # High-Value Approval — Transaction decisions ($25K–$100K+)
    # ------------------------------------------------------------------
    async def _process_approval(self, work_item: WorkItem) -> WorkResult:
        """Process a high-value transaction approval request.

        The Controller reviews transactions that exceed departmental manager
        limits but fall within the controller's authority band.  If the
        amount exceeds the CFO threshold ($100K) for POs or invoices, the
        request is **automatically escalated** to the CFO.

        Processing flow:
          1. Extract transaction data and monetary amount from the work item.
          2. Determine the transaction type and check against authority limits.
          3. If amount >= $100K and transaction type is PO or invoice:
             → escalate to CFO (return with ``approval_needed=True``).
          4. Otherwise, invoke ``self.make_decision()`` with
             ``decision_type="approve_transaction"`` for LLM-powered
             approval reasoning.
          5. Evaluate the decision result:
             - Approved → return success with ``"transaction_approved_by_controller"``.
             - Rejected → return success with ``"transaction_rejected_by_controller"``
               and the rejection reasoning.
          6. Log the decision with full structured context.

        Args:
            work_item: The work item containing transaction data for approval.

        Returns:
            WorkResult with approval, rejection, or escalation outcome.
        """
        # Extract key fields from work item data
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = work_item.amount if work_item.amount is not None else (
            transaction_data.get("amount", 0.0)
        )
        transaction_type: str = transaction_data.get(
            "transaction_type", work_item.type
        )
        escalation_chain: List[str] = transaction_data.get(
            "escalation_chain", []
        )
        vendor_name: str = transaction_data.get("vendor_name", "")
        requester: str = transaction_data.get("requester", "")

        # -----------------------------------------------------------------
        # CFO Escalation Check — PO or Invoice amounts >= $100K
        # -----------------------------------------------------------------
        if (
            amount >= self.CFO_THRESHOLD
            and transaction_type in ("purchase_order", "vendor_invoice")
        ):
            logger.info(
                "controller_escalated_to_cfo",
                agent_id=str(self.config.agent_id),
                amount=amount,
                transaction_type=transaction_type,
                reason="amount_exceeds_controller_authority",
                cfo_threshold=self.CFO_THRESHOLD,
                vendor_name=vendor_name,
            )

            return WorkResult(
                success=True,
                actions_taken=["escalated_to_cfo"],
                approval_needed=True,
                has_exceptions=False,
                data={
                    "escalated": True,
                    "escalated_to": "cfo",
                    "escalation_reason": (
                        f"Amount ${amount:,.2f} exceeds controller authority "
                        f"limit of ${self.CFO_THRESHOLD:,.2f} for "
                        f"{transaction_type}"
                    ),
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "escalation_chain": escalation_chain + ["controller"],
                    "vendor_name": vendor_name,
                    "requester": requester,
                    "original_work_item_type": work_item.type,
                },
            )

        # -----------------------------------------------------------------
        # Within Authority — Make approval decision via Decision Engine
        # -----------------------------------------------------------------
        context: Dict[str, Any] = {
            "transaction": transaction_data,
            "amount": amount,
            "transaction_type": transaction_type,
            "escalation_chain": escalation_chain,
            "vendor_name": vendor_name,
            "requester": requester,
            "approval_level": "controller",
            "is_terminal_approver": transaction_type == "journal_entry",
            "authority_range": self.AUTHORITY.get(
                transaction_type,
                {"min": 0.0, "max": self.CFO_THRESHOLD},
            ),
            "controller_considerations": {
                "compliance_risk": transaction_data.get(
                    "compliance_risk", "unknown"
                ),
                "budget_impact": transaction_data.get(
                    "budget_impact", "unknown"
                ),
                "department": transaction_data.get("department", "unknown"),
                "justification": transaction_data.get("justification", ""),
            },
        }

        # Invoke Decision Engine via inherited make_decision()
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
            actions_taken: List[str] = ["transaction_approved_by_controller"]

            logger.info(
                "controller_approval_decision",
                agent_id=str(self.config.agent_id),
                amount=amount,
                transaction_type=transaction_type,
                decision="approved",
                reasoning=reasoning,
                vendor_name=vendor_name,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                has_exceptions=False,
                data={
                    "approved": True,
                    "approver": "controller",
                    "approver_agent_id": str(self.config.agent_id),
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "reasoning": reasoning,
                    "escalation_chain": escalation_chain + ["controller"],
                    "is_terminal_decision": (
                        transaction_type == "journal_entry"
                    ),
                },
            )

        # Rejection path — Controller rejects with reasoning
        actions_taken = ["transaction_rejected_by_controller"]

        logger.info(
            "controller_approval_decision",
            agent_id=str(self.config.agent_id),
            amount=amount,
            transaction_type=transaction_type,
            decision="rejected",
            reasoning=reasoning,
            vendor_name=vendor_name,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=False,
            data={
                "approved": False,
                "approver": "controller",
                "approver_agent_id": str(self.config.agent_id),
                "amount": amount,
                "transaction_type": transaction_type,
                "rejection_reason": reasoning,
                "escalation_chain": escalation_chain + ["controller"],
                "is_terminal_decision": (
                    transaction_type == "journal_entry"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Period Close Approval — Fiscal period close oversight
    # ------------------------------------------------------------------
    async def _process_period_close_approval(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a fiscal period close approval request.

        The Controller provides the final review and sign-off required
        before a fiscal period transitions from ``closing`` to ``closed``.
        Period close requests are typically initiated by the Senior
        Accountant after completing reconciliations and adjustments.

        Processing flow:
          1. Extract period data, outstanding items, and reconciliation
             status from the work item.
          2. Verify pre-requisites:
             - All reconciliations approved
             - All pending transactions processed
             - All exceptions resolved
          3. Build context with period financial summaries and readiness
             indicators.
          4. Invoke ``self.make_decision()`` with
             ``decision_type="approve_transaction"`` for LLM-powered
             period close assessment.
          5. Evaluate the decision:
             - Approved → return ``"period_close_approved"`` with the
               authorization.
             - Deferred → return ``"period_close_deferred"`` with the
               reasons and outstanding items requiring resolution.
          6. Log the period close decision.

        Args:
            work_item: The work item containing period close request data.

        Returns:
            WorkResult with approval/deferral outcome and details.
        """
        period_data: Dict[str, Any] = work_item.data
        fiscal_period: Optional[str] = period_data.get("fiscal_period")
        fiscal_year: Optional[int] = period_data.get("fiscal_year")
        outstanding_items: List[Dict[str, Any]] = period_data.get(
            "outstanding_items", []
        )
        reconciliation_status: Dict[str, Any] = period_data.get(
            "reconciliation_status", {}
        )
        pending_transactions: int = period_data.get(
            "pending_transactions", 0
        )
        unresolved_exceptions: int = period_data.get(
            "unresolved_exceptions", 0
        )
        total_transactions: int = period_data.get("total_transactions", 0)
        total_adjustments: int = period_data.get("total_adjustments", 0)

        # Assess readiness — check pre-requisites
        reconciliations_approved: bool = reconciliation_status.get(
            "all_approved", len(outstanding_items) == 0
        )
        all_transactions_processed: bool = pending_transactions == 0
        all_exceptions_resolved: bool = unresolved_exceptions == 0

        readiness_status: Dict[str, bool] = {
            "reconciliations_approved": reconciliations_approved,
            "all_transactions_processed": all_transactions_processed,
            "all_exceptions_resolved": all_exceptions_resolved,
        }
        is_ready: bool = all(readiness_status.values())

        # Build decision context for period close assessment
        context: Dict[str, Any] = {
            "period_close": period_data,
            "fiscal_period": fiscal_period,
            "fiscal_year": fiscal_year,
            "readiness_status": readiness_status,
            "is_ready": is_ready,
            "financial_summary": {
                "total_transactions": total_transactions,
                "total_adjustments": total_adjustments,
                "pending_transactions": pending_transactions,
                "unresolved_exceptions": unresolved_exceptions,
                "outstanding_items_count": len(outstanding_items),
            },
            "reconciliation_status": reconciliation_status,
            "outstanding_items": outstanding_items,
            "approval_level": "controller",
            "close_type": "period_close",
        }

        # Invoke Decision Engine
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        approved: bool = decision.get(
            "approved", decision.get("decision", "") == "approved"
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "")
        )
        conditions: List[str] = decision.get("conditions", [])

        if approved and is_ready:
            logger.info(
                "period_close_approval",
                agent_id=str(self.config.agent_id),
                fiscal_period=fiscal_period,
                fiscal_year=fiscal_year,
                decision="approved",
                reasoning=reasoning,
                total_transactions=total_transactions,
            )

            return WorkResult(
                success=True,
                actions_taken=["period_close_approved"],
                approval_needed=False,
                has_exceptions=False,
                data={
                    "close_authorized": True,
                    "authorizer": "controller",
                    "authorizer_agent_id": str(self.config.agent_id),
                    "fiscal_period": fiscal_period,
                    "fiscal_year": fiscal_year,
                    "reasoning": reasoning,
                    "conditions": conditions,
                    "readiness_status": readiness_status,
                    "financial_summary": {
                        "total_transactions": total_transactions,
                        "total_adjustments": total_adjustments,
                    },
                },
            )

        # Deferral path — period close conditions not yet met
        deferral_reasons: List[str] = []
        if not reconciliations_approved:
            deferral_reasons.append(
                "Not all reconciliations have been approved"
            )
        if not all_transactions_processed:
            deferral_reasons.append(
                f"{pending_transactions} transaction(s) still pending"
            )
        if not all_exceptions_resolved:
            deferral_reasons.append(
                f"{unresolved_exceptions} exception(s) remain unresolved"
            )
        if not approved:
            deferral_reasons.append(
                f"Decision engine declined: {reasoning}"
            )

        deferred_items: List[Dict[str, Any]] = decision.get(
            "deferred_items", outstanding_items
        )

        logger.info(
            "period_close_approval",
            agent_id=str(self.config.agent_id),
            fiscal_period=fiscal_period,
            fiscal_year=fiscal_year,
            decision="deferred",
            reasoning=reasoning,
            deferral_reasons=deferral_reasons,
            outstanding_items_count=len(deferred_items),
        )

        return WorkResult(
            success=True,
            actions_taken=["period_close_deferred"],
            approval_needed=False,
            has_exceptions=len(deferred_items) > 0,
            data={
                "close_authorized": False,
                "authorizer": "controller",
                "authorizer_agent_id": str(self.config.agent_id),
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
    # Escalation Review — Resolve issues escalated from managers
    # ------------------------------------------------------------------
    async def _review_escalation(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Review and resolve an issue escalated from a manager or senior
        accountant.

        Escalations arrive when a departmental manager (AP Manager,
        Purchasing Manager, AR Manager) or the Senior Accountant encounters
        a situation that exceeds their authority or requires cross-department
        judgment — for example, vendor disputes above the manager's
        threshold, unresolved discrepancies, or policy exceptions.

        Processing flow:
          1. Extract escalation data including the original issue,
             escalation source, previous attempts, and context.
          2. Build context with the escalation history, original
             transaction details, and the controller's oversight
             perspective.
          3. Invoke ``self.make_decision()`` with
             ``decision_type="handle_exception"`` for LLM-powered
             escalation resolution.
          4. Evaluate the resolution:
             - Resolved → return with appropriate resolution actions.
             - Further escalation needed → flag for CFO review with
               ``approval_needed=True``.
          5. Log the escalation review outcome.

        Args:
            work_item: The work item containing escalation data.

        Returns:
            WorkResult with escalation resolution or further escalation.
        """
        escalation_data: Dict[str, Any] = work_item.data
        escalation_source: str = escalation_data.get(
            "escalation_source", "unknown"
        )
        original_issue: str = escalation_data.get(
            "original_issue", "unspecified"
        )
        previous_attempts: List[Dict[str, Any]] = escalation_data.get(
            "previous_attempts", []
        )
        original_transaction: Dict[str, Any] = escalation_data.get(
            "original_transaction", {}
        )
        escalation_reason: str = escalation_data.get(
            "escalation_reason", ""
        )
        severity: str = escalation_data.get("severity", "medium")
        amount: float = work_item.amount if work_item.amount is not None else (
            escalation_data.get("amount", 0.0)
        )

        # Build decision context for escalation review
        context: Dict[str, Any] = {
            "escalation": escalation_data,
            "escalation_source": escalation_source,
            "original_issue": original_issue,
            "escalation_reason": escalation_reason,
            "severity": severity,
            "amount": amount,
            "previous_attempts": previous_attempts,
            "previous_attempts_count": len(previous_attempts),
            "original_transaction": original_transaction,
            "approval_level": "controller",
            "review_scope": "escalation_review",
        }

        # Invoke Decision Engine — use handle_exception for escalations
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="handle_exception",
            context=context,
        )

        # Extract resolution details
        resolved: bool = decision.get(
            "resolved", decision.get("resolution", "") != ""
        )
        resolution: str = decision.get(
            "resolution", decision.get("recommendation", "")
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "")
        )
        needs_cfo: bool = decision.get("escalate_to_cfo", False)
        action_items: List[Dict[str, Any]] = decision.get(
            "action_items", []
        )

        # Determine if further escalation to CFO is required
        if needs_cfo or (
            severity == "critical"
            and not resolved
        ):
            logger.info(
                "escalation_review",
                agent_id=str(self.config.agent_id),
                escalation_source=escalation_source,
                original_issue=original_issue,
                severity=severity,
                decision="escalated_to_cfo",
                reasoning=reasoning,
            )

            return WorkResult(
                success=True,
                actions_taken=[
                    "escalation_reviewed",
                    "escalated_to_cfo",
                ],
                approval_needed=True,
                has_exceptions=True,
                data={
                    "escalated": True,
                    "escalated_to": "cfo",
                    "reviewer": "controller",
                    "reviewer_agent_id": str(self.config.agent_id),
                    "escalation_source": escalation_source,
                    "original_issue": original_issue,
                    "severity": severity,
                    "reasoning": reasoning,
                    "resolution_attempted": resolution,
                    "previous_attempts": previous_attempts,
                    "amount": amount,
                },
            )

        # Controller resolves the escalation
        actions_taken: List[str] = ["escalation_reviewed"]
        if resolved:
            actions_taken.append("escalation_resolved")
        else:
            actions_taken.append("escalation_deferred")

        logger.info(
            "escalation_review",
            agent_id=str(self.config.agent_id),
            escalation_source=escalation_source,
            original_issue=original_issue,
            severity=severity,
            decision="resolved" if resolved else "deferred",
            reasoning=reasoning,
            resolution=resolution,
            action_items_count=len(action_items),
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=not resolved,
            data={
                "resolved": resolved,
                "reviewer": "controller",
                "reviewer_agent_id": str(self.config.agent_id),
                "escalation_source": escalation_source,
                "original_issue": original_issue,
                "severity": severity,
                "resolution": resolution,
                "reasoning": reasoning,
                "action_items": action_items,
                "amount": amount,
            },
        )
