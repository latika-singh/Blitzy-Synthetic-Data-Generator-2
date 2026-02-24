"""
CFOAgent — Chief Financial Officer agent for the Agent System (F-001).

The CFO agent is the APEX of the approval hierarchy, responsible for:
  - Strategic approvals for transactions exceeding $100K (terminal approver)
  - Year-end fiscal close final authorization
  - Financial oversight across all transaction types

There is NO escalation path above the CFO — this is the final decision-maker
in the approval chain. All unresolved escalations from the Controller terminate
here.

Approval Thresholds (from README.md lines 319, 325):
  - Purchase Orders: > $100K → CFO approval (terminal)
  - Vendor Invoices: > $100K → CFO approval (terminal)
  - Any escalation from Controller → CFO final review

Exports:
    CFOAgent — Specialized agent subclass extending BaseAgent with:
        - ROLE:                     "cfo"
        - SUPPORTED_WORK_TYPES:     List of 5 supported transaction types
        - STRATEGIC_THRESHOLD:      $100,000 monetary threshold
        - process_work_item():      Main routing dispatcher
        - _process_strategic_approval(): High-value transaction approvals
        - _process_year_end_close():     Fiscal year close authorization
        - _process_financial_oversight(): General financial oversight reviews

Design Decisions:
  - Extends BaseAgent via constructor injection (config, memory, decision_engine,
    action_registry) per AAP Section 0.7.1. No overrides to __init__.
  - Uses self.make_decision() (inherited from BaseAgent) to invoke the 4-layer
    Decision Engine for LLM-powered strategic approval reasoning.
  - Financial calculations remain EXCLUSIVELY in the Deterministic Layer of the
    Decision Engine — the CFO agent never computes GL postings or balance updates.
  - Lightweight class: no __init__ override, no I/O at import time. Supports
    ≥50 agents/second creation rate (AAP Section 0.7.3).
  - All logging via structlog in structured JSON format to stdout (AAP Section 0.7.6).

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
  - README.md line 81: "CFOAgent # CFO (strategic decisions)"
  - README.md lines 319, 325: Approval thresholds (>$100K → CFO)
  - AAP Section 0.5.1 Group 5: "CFOAgent: strategic approvals (>$100K),
    financial oversight"
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


class CFOAgent(BaseAgent):
    """Chief Financial Officer (CFO) agent — apex of the approval hierarchy.

    Responsible for:
      - Strategic approvals for transactions exceeding $100K
      - Financial oversight across all transaction types
      - Year-end fiscal close final approval
      - Final authority in the approval chain (no escalation above CFO)

    Role: ``"cfo"``

    Transaction types handled:
      - ``"strategic_approval"``  — General high-value strategic approvals
      - ``"year_end_close"``      — Fiscal year close final sign-off
      - ``"financial_oversight"`` — Periodic financial oversight reviews
      - ``"po_approval"``         — Purchase order approvals (> $100K)
      - ``"invoice_approval"``    — Vendor invoice approvals (> $100K)

    Approval Authority (from README.md lines 319, 325):
      - Purchase Orders:  > $100K (terminal approver)
      - Vendor Invoices:  > $100K (terminal approver)
      - Any escalation from Controller

    IMPORTANT:
        CFO is the **apex** of the approval hierarchy. There is no further
        escalation path beyond the CFO. All decisions made here are final.

    Inherits from :class:`BaseAgent`:
      - Constructor injection (config, memory, decision_engine, action_registry)
      - Async ``run()`` loop for continuous work item processing
      - ``make_decision()`` for 4-layer Decision Engine delegation
      - ``_calculate_importance()`` for memory scoring
      - Performance metrics tracking

    Example::

        cfo = CFOAgent(
            config=cfo_config,
            memory=cfo_memory,
            decision_engine=decision_engine,
            action_registry=action_registry,
        )
        # Enqueue a high-value PO for CFO approval
        await cfo.work_queue.put(WorkItem(
            type="strategic_approval",
            data={"transaction_type": "purchase_order", "amount": 150000.0},
            amount=150000.0,
        ))
        await cfo.run()
    """

    # ------------------------------------------------------------------
    # Class Constants
    # ------------------------------------------------------------------

    ROLE: str = "cfo"
    """Agent role identifier. Must match ``"cfo"`` in VALID_AGENT_ROLES
    (agent_config.py line 118)."""

    SUPPORTED_WORK_TYPES: List[str] = [
        "strategic_approval",
        "year_end_close",
        "financial_oversight",
        "po_approval",
        "invoice_approval",
    ]
    """Work item types this agent can process. Used by the
    WorkflowOrchestrator for routing validation."""

    STRATEGIC_THRESHOLD: float = 100_000.0
    """Monetary threshold ($100K) above which the CFO's approval is required.
    From README.md lines 319 (PO > $100K → CFO) and 325 (Invoice > $100K → CFO).
    """

    # ------------------------------------------------------------------
    # process_work_item — Main routing dispatcher (abstract method impl)
    # ------------------------------------------------------------------
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Route a work item to the appropriate CFO processing method.

        Dispatches based on ``work_item.type``:
          - ``"strategic_approval"`` | ``"po_approval"`` | ``"invoice_approval"``
            → :meth:`_process_strategic_approval`
          - ``"year_end_close"``
            → :meth:`_process_year_end_close`
          - ``"financial_oversight"``
            → :meth:`_process_financial_oversight`

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
            "cfo_processing",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            work_item_type=work_item.type,
            amount=work_item.amount,
            description=work_item.description,
        )

        work_type: str = work_item.type

        try:
            if work_type in (
                "strategic_approval",
                "po_approval",
                "invoice_approval",
            ):
                return await self._process_strategic_approval(work_item)

            if work_type == "year_end_close":
                return await self._process_year_end_close(work_item)

            if work_type == "financial_oversight":
                return await self._process_financial_oversight(work_item)

            # Unsupported type — raise so run() captures it in ERROR state
            raise ValueError(
                f"CFOAgent does not support work item type '{work_type}'. "
                f"Supported types: {self.SUPPORTED_WORK_TYPES}"
            )

        except ValueError:
            # Re-raise ValueError for unsupported types
            raise

        except Exception as exc:
            # Catch-all for unexpected errors within handler methods
            logger.error(
                "cfo_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=["cfo_processing_failed"],
                has_exceptions=True,
                data={"error": str(exc), "work_item_type": work_type},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Strategic Approval — High-value transaction decisions (>$100K)
    # ------------------------------------------------------------------
    async def _process_strategic_approval(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a high-value strategic approval request.

        The CFO is the **terminal approver** — there is NO escalation path
        above this agent. Decisions are either approved or rejected; never
        escalated.

        Processing flow:
          1. Extract transaction data and monetary amount from the work item.
          2. Build context dictionary with transaction details, amount,
             escalation chain history, and the CFO's strategic perspective.
          3. Invoke ``self.make_decision()`` with ``decision_type="approve_transaction"``
             to engage the 4-layer Decision Engine for LLM-powered reasoning.
          4. Evaluate the decision result:
             - Approved → return success with ``"transaction_approved_by_cfo"``.
             - Rejected → return success with ``"transaction_rejected_by_cfo"``
               and the rejection reasoning.
          5. Log the decision with full structured context.

        IMPORTANT: The CFO **never escalates**. Every request terminates here.

        Args:
            work_item: The work item containing transaction data for approval.

        Returns:
            WorkResult with approval/rejection outcome and reasoning.
        """
        # Extract key fields from work item data
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = work_item.amount if work_item.amount is not None else (
            transaction_data.get("amount", 0.0)
        )
        transaction_type: str = transaction_data.get("transaction_type", work_item.type)
        escalation_chain: List[str] = transaction_data.get("escalation_chain", [])
        vendor_name: str = transaction_data.get("vendor_name", "")
        requester: str = transaction_data.get("requester", "")

        # Build decision context for the 4-layer Decision Engine
        context: Dict[str, Any] = {
            "transaction": transaction_data,
            "amount": amount,
            "transaction_type": transaction_type,
            "escalation_chain": escalation_chain,
            "vendor_name": vendor_name,
            "requester": requester,
            "approval_level": "cfo",
            "is_terminal_approver": True,
            "threshold": self.STRATEGIC_THRESHOLD,
            "strategic_considerations": {
                "budget_impact": transaction_data.get("budget_impact", "unknown"),
                "strategic_alignment": transaction_data.get(
                    "strategic_alignment", "unknown"
                ),
                "risk_assessment": transaction_data.get("risk_assessment", "unknown"),
            },
        }

        # Invoke Decision Engine via inherited make_decision()
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Determine approval outcome from decision result
        approved: bool = decision.get("approved", decision.get("decision", "") == "approved")
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))

        if approved:
            actions_taken: List[str] = ["transaction_approved_by_cfo"]

            logger.info(
                "cfo_strategic_approval",
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
                    "approver": "cfo",
                    "approver_agent_id": str(self.config.agent_id),
                    "amount": amount,
                    "transaction_type": transaction_type,
                    "reasoning": reasoning,
                    "escalation_chain": escalation_chain + ["cfo"],
                    "is_terminal_decision": True,
                },
            )

        # Rejection path — CFO rejects with reasoning
        actions_taken = ["transaction_rejected_by_cfo"]

        logger.info(
            "cfo_strategic_approval",
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
                "approver": "cfo",
                "approver_agent_id": str(self.config.agent_id),
                "amount": amount,
                "transaction_type": transaction_type,
                "rejection_reason": reasoning,
                "escalation_chain": escalation_chain + ["cfo"],
                "is_terminal_decision": True,
            },
        )

    # ------------------------------------------------------------------
    # Year-End Close — Fiscal year close final authorization
    # ------------------------------------------------------------------
    async def _process_year_end_close(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a year-end fiscal close authorization request.

        The CFO provides the final sign-off required to close a fiscal year.
        This involves reviewing a summary of the year's financial activity
        and determining whether conditions are met for closure.

        Processing flow:
          1. Extract fiscal year, period data, and year-end summary from
             the work item.
          2. Build context with financial summaries, outstanding items,
             and compliance status.
          3. Invoke ``self.make_decision()`` for LLM-powered year-end
             assessment.
          4. Evaluate the decision:
             - Approved → return ``"year_end_close_approved"`` with the
               authorization.
             - Deferred → return ``"year_end_close_deferred"`` with the
               list of outstanding items requiring resolution before
               the close can proceed.
          5. Log the year-end close decision.

        Args:
            work_item: The work item containing year-end close data.

        Returns:
            WorkResult with approval/deferral outcome and details.
        """
        year_end_data: Dict[str, Any] = work_item.data
        fiscal_year: Optional[int] = year_end_data.get("fiscal_year")
        fiscal_period: Optional[str] = year_end_data.get("fiscal_period")
        outstanding_items: List[Dict[str, Any]] = year_end_data.get(
            "outstanding_items", []
        )
        total_transactions: int = year_end_data.get("total_transactions", 0)
        total_revenue: float = year_end_data.get("total_revenue", 0.0)
        total_expenses: float = year_end_data.get("total_expenses", 0.0)
        unreconciled_count: int = year_end_data.get("unreconciled_count", 0)

        # Build decision context for year-end close assessment
        context: Dict[str, Any] = {
            "year_end_close": year_end_data,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "financial_summary": {
                "total_transactions": total_transactions,
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "net_income": total_revenue - total_expenses,
                "outstanding_items_count": len(outstanding_items),
                "unreconciled_count": unreconciled_count,
            },
            "outstanding_items": outstanding_items,
            "approval_level": "cfo",
            "close_type": "year_end",
        }

        # Invoke Decision Engine
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        approved: bool = decision.get("approved", decision.get("decision", "") == "approved")
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))
        conditions: List[str] = decision.get("conditions", [])

        if approved:
            logger.info(
                "year_end_close",
                agent_id=str(self.config.agent_id),
                fiscal_year=fiscal_year,
                decision="approved",
                reasoning=reasoning,
                total_transactions=total_transactions,
            )

            return WorkResult(
                success=True,
                actions_taken=["year_end_close_approved"],
                approval_needed=False,
                has_exceptions=False,
                data={
                    "close_authorized": True,
                    "authorizer": "cfo",
                    "authorizer_agent_id": str(self.config.agent_id),
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "reasoning": reasoning,
                    "conditions": conditions,
                    "financial_summary": {
                        "total_transactions": total_transactions,
                        "total_revenue": total_revenue,
                        "total_expenses": total_expenses,
                        "net_income": total_revenue - total_expenses,
                    },
                },
            )

        # Deferral path — year-end close conditions not yet met
        deferred_items: List[Dict[str, Any]] = decision.get(
            "deferred_items", outstanding_items
        )

        logger.info(
            "year_end_close",
            agent_id=str(self.config.agent_id),
            fiscal_year=fiscal_year,
            decision="deferred",
            reasoning=reasoning,
            outstanding_items_count=len(deferred_items),
        )

        return WorkResult(
            success=True,
            actions_taken=["year_end_close_deferred"],
            approval_needed=False,
            has_exceptions=len(deferred_items) > 0,
            data={
                "close_authorized": False,
                "authorizer": "cfo",
                "authorizer_agent_id": str(self.config.agent_id),
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "reasoning": reasoning,
                "deferred_items": deferred_items,
                "resolution_required": True,
            },
        )

    # ------------------------------------------------------------------
    # Financial Oversight — General financial oversight reviews
    # ------------------------------------------------------------------
    async def _process_financial_oversight(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a financial oversight review request.

        The CFO periodically reviews financial reports, budget compliance,
        and operational metrics to ensure organizational financial health.
        This method handles general oversight work items that do not fall
        under strategic approval or year-end close categories.

        Processing flow:
          1. Extract oversight data including report type, period, and
             financial metrics from the work item.
          2. Build context with budget compliance data, variance analysis,
             and risk indicators.
          3. Invoke ``self.make_decision()`` for LLM-powered oversight
             assessment.
          4. Compile findings and recommendations from the decision result.
          5. Log the oversight review outcome.

        Args:
            work_item: The work item containing financial oversight data.

        Returns:
            WorkResult with oversight findings and recommendations.
        """
        oversight_data: Dict[str, Any] = work_item.data
        report_type: str = oversight_data.get("report_type", "general")
        review_period: Optional[str] = oversight_data.get("review_period")
        budget_data: Dict[str, Any] = oversight_data.get("budget_data", {})
        variance_data: Dict[str, Any] = oversight_data.get("variance_data", {})
        risk_indicators: List[Dict[str, Any]] = oversight_data.get(
            "risk_indicators", []
        )
        compliance_status: Dict[str, Any] = oversight_data.get(
            "compliance_status", {}
        )

        # Build decision context for oversight assessment
        context: Dict[str, Any] = {
            "oversight_review": oversight_data,
            "report_type": report_type,
            "review_period": review_period,
            "budget_compliance": budget_data,
            "variance_analysis": variance_data,
            "risk_indicators": risk_indicators,
            "compliance_status": compliance_status,
            "approval_level": "cfo",
            "review_scope": "financial_oversight",
        }

        # Invoke Decision Engine
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Extract findings and recommendations from decision
        findings: List[str] = decision.get("findings", [])
        recommendations: List[str] = decision.get("recommendations", [])
        risk_level: str = decision.get("risk_level", "normal")
        action_items: List[Dict[str, Any]] = decision.get("action_items", [])
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))

        # Determine if oversight findings flag any exceptions
        has_concerns: bool = (
            risk_level in ("high", "critical")
            or len(action_items) > 0
            or any(
                indicator.get("severity", "low") in ("high", "critical")
                for indicator in risk_indicators
            )
        )

        actions_taken: List[str] = ["financial_oversight_completed"]
        if has_concerns:
            actions_taken.append("oversight_concerns_flagged")

        logger.info(
            "financial_oversight",
            agent_id=str(self.config.agent_id),
            report_type=report_type,
            review_period=review_period,
            risk_level=risk_level,
            findings_count=len(findings),
            recommendations_count=len(recommendations),
            has_concerns=has_concerns,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=has_concerns,
            data={
                "reviewer": "cfo",
                "reviewer_agent_id": str(self.config.agent_id),
                "report_type": report_type,
                "review_period": review_period,
                "risk_level": risk_level,
                "findings": findings,
                "recommendations": recommendations,
                "action_items": action_items,
                "reasoning": reasoning,
                "has_concerns": has_concerns,
            },
        )
