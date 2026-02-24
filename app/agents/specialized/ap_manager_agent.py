"""Accounts Payable Manager agent implementation.

This module provides the APManagerAgent class — the Accounts Payable manager
responsible for invoice approval, exception handling, and escalation within
the AP workflow.

The AP Manager sits above the AP Clerk in the organizational hierarchy and
handles invoice approval decisions within their monetary authority band
($10K–$50K), exception resolution for invoices flagged during 3-way matching,
and direct vendor invoice processing when routed via the WorkflowOrchestrator.

Key responsibilities:
    - Approving vendor invoices within authority ($10K–$50K)
    - Handling invoice exceptions and escalations from AP Clerk
    - Reviewing 3-way match variances (quantity, price, receipt discrepancies)
    - Escalating high-value invoices (>$50K) to Controller or CFO
    - Processing vendor invoices directly when assigned via role mapping

Approval thresholds (README.md lines 321–325)::

    vendor_invoice:
        $0–$10K:     no approval needed (auto-approved by AP Clerk)
        $10K–$50K:   ap_manager approval
        $50K–$100K:  controller approval
        $100K+:      CFO approval

Role mapping (README.md line 1349)::

    "vendor_invoice": ["ap_clerk", "ap_manager"]

Decision types used:
    - approve_transaction: For invoice approval decisions with reasoning
    - handle_exception: For exception resolution (resolve, return, escalate)
    - process_vendor_invoice: For direct vendor invoice review and coding

All logging uses structlog to stdout in structured JSON format per AAP
Section 0.7.6. Every decision produces a structured log entry including
agent_id, agent_role, decision_type, context summary, decision result,
and duration_seconds.
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


class APManagerAgent(BaseAgent):
    """Accounts Payable Manager agent.

    Responsible for:
        - Approving vendor invoices within authority ($10K–$50K)
        - Handling invoice exceptions and escalations from AP Clerk
        - Reviewing 3-way match variances (quantity, price, receipt)
        - Escalating high-value invoices to Controller or CFO
        - Processing vendor invoices directly when role-mapped

    Role: ``"ap_manager"``

    Transaction types handled:
        ``"invoice_approval"``, ``"invoice_exception"``, ``"vendor_invoice"``

    Actions used:
        ``approve_invoice``, ``process_vendor_invoice``

    Approval authority band (README.md lines 321–325):
        - Minimum: $10,000 — invoices below this are auto-approved by AP Clerk
        - Maximum: $50,000 — invoices above this must be escalated to Controller

    Escalation rules:
        - Amount > $50K and <= $100K → Controller
        - Amount > $100K → CFO
        - Unresolvable exceptions → Controller review

    The AP Manager uses the 4-layer DecisionEngine pipeline for all decisions:
        Statistical Layer → LLM Layer → Validation Layer → Deterministic Layer
    Financial calculations are exclusively computed in the Deterministic Layer
    and are never LLM-generated.

    Constructor injection pattern (inherited from BaseAgent):
        config:          AgentConfig — agent identity and personality traits
        memory:          AgentMemory — dual-stream observation/reflection memory
        decision_engine: DecisionEngine — 4-layer hybrid decision pipeline
        action_registry: ActionRegistry — registry of 14 ERP action types
    """

    # ------------------------------------------------------------------ #
    # Class-level constants
    # ------------------------------------------------------------------ #

    ROLE: str = "ap_manager"
    """Agent role identifier matching VALID_AGENT_ROLES in agent_config.py
    and ROLE_MAPPING in WorkflowOrchestrator."""

    SUPPORTED_WORK_TYPES: List[str] = [
        "invoice_approval",
        "invoice_exception",
        "vendor_invoice",
    ]
    """Work-item types this agent is capable of processing."""

    APPROVAL_THRESHOLD_MIN: float = 10000.0
    """Minimum dollar amount for AP Manager approval authority ($10K).
    Invoices below this amount are auto-approved by the AP Clerk and do
    not require AP Manager review under normal circumstances."""

    APPROVAL_THRESHOLD_MAX: float = 50000.0
    """Maximum dollar amount for AP Manager approval authority ($50K).
    Invoices exceeding this amount must be escalated to the Controller
    (up to $100K) or CFO (above $100K)."""

    # ------------------------------------------------------------------ #
    # process_work_item — required by BaseAgent ABC
    # ------------------------------------------------------------------ #

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Route an incoming work item to the appropriate AP Manager handler.

        Dispatches work items based on their ``type`` field to specialized
        handler methods. Each handler uses the DecisionEngine (via
        ``self.make_decision()``) for LLM-powered decisions and returns a
        WorkResult with the outcome.

        Routing table:
            - ``"invoice_approval"``  → ``_process_invoice_approval()``
            - ``"invoice_exception"`` → ``_handle_invoice_exception()``
            - ``"vendor_invoice"``    → ``_process_vendor_invoice()``

        Args:
            work_item: The work item to process. Must have a ``type``
                matching one of ``SUPPORTED_WORK_TYPES``.

        Returns:
            WorkResult containing the processing outcome, actions taken,
            and any escalation or exception flags.

        Raises:
            ValueError: If ``work_item.type`` is not in
                ``SUPPORTED_WORK_TYPES``.
        """
        logger.info(
            "ap_manager_processing",
            agent_id=str(self.config.agent_id),
            agent_role=self.config.role,
            work_item_type=work_item.type,
            workflow_id=str(work_item.workflow_id),
        )

        try:
            if work_item.type == "invoice_approval":
                return await self._process_invoice_approval(work_item)
            elif work_item.type == "invoice_exception":
                return await self._handle_invoice_exception(work_item)
            elif work_item.type == "vendor_invoice":
                return await self._process_vendor_invoice(work_item)
            else:
                raise ValueError(
                    f"APManagerAgent does not support work item type: "
                    f"{work_item.type!r}. Supported types: "
                    f"{self.SUPPORTED_WORK_TYPES}"
                )
        except ValueError:
            # Re-raise ValueError for unsupported work item types so the
            # caller (BaseAgent.run loop) can log and handle appropriately.
            raise
        except Exception as exc:
            logger.error(
                "ap_manager_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_item.type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=["processing_failed"],
                has_exceptions=True,
                error=f"AP Manager processing error: {str(exc)}",
                data={"work_item_type": work_item.type},
            )

    # ------------------------------------------------------------------ #
    # Invoice Approval — Authority band $10K–$50K
    # ------------------------------------------------------------------ #

    async def _process_invoice_approval(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process an invoice approval request within AP Manager authority.

        Evaluates whether the invoice falls within the AP Manager's approval
        authority band ($10K–$50K). If the amount exceeds the maximum
        threshold, the invoice is escalated to the Controller. Otherwise,
        the 4-layer DecisionEngine pipeline makes the approve/reject
        decision using the ``approve_transaction`` decision type.

        Decision outcomes:
            - **approve / approved** — Invoice approved, actions include
              ``"invoice_approved"``.
            - **reject / rejected** — Invoice rejected with reasoning,
              actions include ``"invoice_rejected"``.
            - **escalate / escalated** — Forwarded up the approval chain,
              ``approval_needed=True``.
            - **Decision engine error** — Returns failure with exception
              flag set.

        Args:
            work_item: Work item containing invoice data for approval.
                Expected keys in ``work_item.data``:
                    - ``amount`` (float): Invoice total amount
                    - ``invoice_id`` (str): Unique invoice identifier
                    - ``vendor_id`` (str): Vendor who submitted the invoice
                    - ``match_result`` (dict): 3-way match outcome from
                      AP Clerk processing

        Returns:
            WorkResult with approval decision. The ``approval_needed`` flag
            is set to True when escalation to Controller or CFO is required.
        """
        invoice_data: Dict[str, Any] = work_item.data
        amount: float = float(invoice_data.get("amount", work_item.amount or 0.0))
        invoice_id: str = str(invoice_data.get("invoice_id", "unknown"))
        vendor_id: str = str(invoice_data.get("vendor_id", "unknown"))

        logger.info(
            "invoice_approval_started",
            agent_id=str(self.config.agent_id),
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            amount=amount,
        )

        # ---- Authority check: escalate invoices above $50K --------------- #
        if amount > self.APPROVAL_THRESHOLD_MAX:
            escalation_target: str = "cfo" if amount > 100000.0 else "controller"
            escalation_reason: str = (
                f"Amount ${amount:,.2f} exceeds AP Manager authority limit "
                f"of ${self.APPROVAL_THRESHOLD_MAX:,.2f}. "
                f"Escalating to {escalation_target}."
            )
            logger.info(
                "invoice_approval_escalated",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                escalation_target=escalation_target,
                reason="exceeds_authority",
            )
            return WorkResult(
                success=True,
                actions_taken=[f"escalated_to_{escalation_target}"],
                approval_needed=True,
                data={
                    "escalation_reason": escalation_reason,
                    "escalation_target": escalation_target,
                    "amount": amount,
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                },
            )

        # ---- Build approval context for DecisionEngine ------------------- #
        match_result: Dict[str, Any] = invoice_data.get("match_result", {})
        approval_context: Dict[str, Any] = {
            "transaction": invoice_data,
            "amount": amount,
            "invoice_id": invoice_id,
            "vendor_id": vendor_id,
            "match_result": match_result,
            "approver_role": self.ROLE,
            "threshold_min": self.APPROVAL_THRESHOLD_MIN,
            "threshold_max": self.APPROVAL_THRESHOLD_MAX,
        }

        # ---- Make approval decision via 4-layer DecisionEngine ----------- #
        try:
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="approve_transaction",
                context=approval_context,
            )
        except Exception as exc:
            logger.error(
                "invoice_approval_decision_error",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["approval_decision_failed"],
                has_exceptions=True,
                error=f"Decision engine error during invoice approval: {str(exc)}",
                data={
                    "invoice_id": invoice_id,
                    "amount": amount,
                    "vendor_id": vendor_id,
                },
            )

        # ---- Process decision outcome ------------------------------------ #
        decision_result: str = decision.get("decision", "").lower()
        reasoning: str = decision.get("reasoning", "")

        if decision_result in ("approve", "approved"):
            logger.info(
                "invoice_approval_decision",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                decision="approved",
                reasoning=reasoning,
            )
            return WorkResult(
                success=True,
                actions_taken=["invoice_approved"],
                data={
                    "approved": True,
                    "approver": self.ROLE,
                    "approver_agent_id": str(self.config.agent_id),
                    "amount": amount,
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "reasoning": reasoning,
                    "match_result": match_result,
                },
            )

        elif decision_result in ("reject", "rejected"):
            logger.info(
                "invoice_approval_decision",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                decision="rejected",
                reasoning=reasoning,
            )
            return WorkResult(
                success=True,
                actions_taken=["invoice_rejected"],
                data={
                    "approved": False,
                    "approver": self.ROLE,
                    "approver_agent_id": str(self.config.agent_id),
                    "amount": amount,
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "rejection_reason": reasoning,
                },
            )

        elif decision_result in ("escalate", "escalated"):
            logger.info(
                "invoice_approval_decision",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                decision="escalated",
                reasoning=reasoning,
            )
            return WorkResult(
                success=True,
                actions_taken=["escalated_to_controller"],
                approval_needed=True,
                data={
                    "escalation_reason": reasoning or "Decision engine recommended escalation",
                    "escalation_target": "controller",
                    "amount": amount,
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                },
            )

        else:
            # Unrecognised decision outcome — treat as needing escalation
            # to avoid silently dropping an invoice.
            logger.warning(
                "invoice_approval_unknown_decision",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                raw_decision=decision_result,
            )
            return WorkResult(
                success=True,
                actions_taken=["escalated_to_controller"],
                approval_needed=True,
                has_exceptions=True,
                data={
                    "escalation_reason": (
                        f"Unrecognised decision outcome: {decision_result!r}. "
                        "Escalating for manual review."
                    ),
                    "escalation_target": "controller",
                    "amount": amount,
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "raw_decision": decision_result,
                },
            )

    # ------------------------------------------------------------------ #
    # Invoice Exception Handling
    # ------------------------------------------------------------------ #

    async def _handle_invoice_exception(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Handle an invoice exception escalated from the AP Clerk.

        Exception types include 3-way match failures, pricing discrepancies,
        quantity mismatches, missing documentation, and duplicate invoice
        detection. The handler uses the ``handle_exception`` decision type
        to determine the appropriate resolution path.

        Resolution paths:
            - **resolve** — Exception resolved at manager level; mark complete
              and return ``WorkResult`` with ``"exception_resolved"`` action.
            - **return_to_clerk** — Additional information or rework needed
              from AP Clerk; return ``WorkResult`` with requeue instructions.
            - **escalate** — Exception requires Controller or higher authority;
              set ``approval_needed=True`` for Controller review.
            - **reject** — Invoice rejected due to irreconcilable exception;
              mark as rejected with reasoning.
            - **Decision engine error** — Returns failure with exception
              flag set.

        Args:
            work_item: Work item containing exception data.
                Expected keys in ``work_item.data``:
                    - ``exception_type`` (str): Category of exception
                    - ``invoice_id`` (str): Related invoice identifier
                    - ``vendor_id`` (str): Vendor associated with the invoice
                    - ``amount`` (float): Invoice amount
                    - ``match_result`` (dict): 3-way match details
                    - ``original_error`` (str): Description of the exception

        Returns:
            WorkResult with exception resolution outcome.
        """
        exception_data: Dict[str, Any] = work_item.data
        exception_type: str = str(exception_data.get("exception_type", "unknown"))
        invoice_id: str = str(exception_data.get("invoice_id", "unknown"))
        vendor_id: str = str(exception_data.get("vendor_id", "unknown"))
        amount: float = float(exception_data.get("amount", work_item.amount or 0.0))
        original_error: str = str(exception_data.get("original_error", ""))

        logger.info(
            "invoice_exception_started",
            agent_id=str(self.config.agent_id),
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            exception_type=exception_type,
            amount=amount,
        )

        # ---- Build exception context for DecisionEngine ------------------ #
        exception_context: Dict[str, Any] = {
            "exception": exception_data,
            "exception_type": exception_type,
            "invoice_id": invoice_id,
            "vendor_id": vendor_id,
            "amount": amount,
            "original_error": original_error,
            "match_result": exception_data.get("match_result", {}),
            "handler_role": self.ROLE,
        }

        # ---- Make exception handling decision via DecisionEngine --------- #
        try:
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="handle_exception",
                context=exception_context,
            )
        except Exception as exc:
            logger.error(
                "invoice_exception_decision_error",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                exception_type=exception_type,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["exception_handling_failed"],
                has_exceptions=True,
                error=f"Decision engine error during exception handling: {str(exc)}",
                data={
                    "invoice_id": invoice_id,
                    "exception_type": exception_type,
                    "amount": amount,
                },
            )

        # ---- Route based on decision result ------------------------------ #
        resolution: str = decision.get("resolution", decision.get("decision", "")).lower()
        reasoning: str = decision.get("reasoning", "")
        resolution_notes: str = decision.get("resolution_notes", reasoning)

        if resolution in ("resolve", "resolved"):
            logger.info(
                "exception_handled",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                resolution="resolved",
                exception_type=exception_type,
            )
            return WorkResult(
                success=True,
                actions_taken=["exception_resolved"],
                data={
                    "resolution": "resolved",
                    "resolver": self.ROLE,
                    "resolver_agent_id": str(self.config.agent_id),
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "exception_type": exception_type,
                    "resolution_notes": resolution_notes,
                },
            )

        elif resolution in ("return_to_clerk", "return", "rework"):
            logger.info(
                "exception_handled",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                resolution="return_to_clerk",
                exception_type=exception_type,
            )
            return WorkResult(
                success=True,
                actions_taken=["returned_to_clerk"],
                data={
                    "resolution": "return_to_clerk",
                    "requeue_to": "ap_clerk",
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "exception_type": exception_type,
                    "instructions": resolution_notes or "Requires additional review or documentation",
                },
            )

        elif resolution in ("escalate", "escalated"):
            logger.info(
                "exception_handled",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                resolution="escalated",
                exception_type=exception_type,
            )
            return WorkResult(
                success=True,
                actions_taken=["exception_escalated_to_controller"],
                approval_needed=True,
                has_exceptions=True,
                data={
                    "resolution": "escalated",
                    "escalation_target": "controller",
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "exception_type": exception_type,
                    "escalation_reason": resolution_notes or "Requires higher authority review",
                    "amount": amount,
                },
            )

        elif resolution in ("reject", "rejected"):
            logger.info(
                "exception_handled",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                resolution="rejected",
                exception_type=exception_type,
            )
            return WorkResult(
                success=True,
                actions_taken=["invoice_rejected_due_to_exception"],
                data={
                    "resolution": "rejected",
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "exception_type": exception_type,
                    "rejection_reason": resolution_notes or "Exception could not be resolved",
                },
            )

        else:
            # Unrecognised resolution — escalate to Controller for safety.
            logger.warning(
                "exception_unknown_resolution",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                exception_type=exception_type,
                raw_resolution=resolution,
            )
            return WorkResult(
                success=True,
                actions_taken=["exception_escalated_to_controller"],
                approval_needed=True,
                has_exceptions=True,
                data={
                    "resolution": "escalated",
                    "escalation_target": "controller",
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "exception_type": exception_type,
                    "escalation_reason": (
                        f"Unrecognised resolution outcome: {resolution!r}. "
                        "Escalating for manual review."
                    ),
                    "raw_resolution": resolution,
                    "amount": amount,
                },
            )

    # ------------------------------------------------------------------ #
    # Vendor Invoice Processing (Manager-level review)
    # ------------------------------------------------------------------ #

    async def _process_vendor_invoice(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a vendor invoice with manager-level review.

        The AP Manager can be directly assigned vendor invoices via the
        role mapping (README.md line 1349). When processing directly, the
        manager performs a higher-level review than the AP Clerk, focusing
        on overall accuracy, GL coding appropriateness, and match result
        evaluation.

        Processing steps:
            1. Extract and validate invoice data (vendor_id, amount, invoice_id).
            2. Check if the amount exceeds AP Manager authority — escalate
               to Controller/CFO if needed.
            3. Use the ``process_vendor_invoice`` decision type via the
               4-layer DecisionEngine pipeline to obtain GL coding,
               match assessment, and processing notes.
            4. Evaluate the match result for any variances that require
               exception handling.
            5. Return WorkResult with appropriate actions and data.

        Args:
            work_item: Work item containing vendor invoice data.
                Expected keys in ``work_item.data``:
                    - ``invoice_id`` (str): Unique invoice identifier
                    - ``vendor_id`` (str): Submitting vendor identifier
                    - ``amount`` (float): Invoice total amount
                    - ``purchase_order`` (dict): Related PO data
                    - ``goods_receipt`` (dict): Related receipt data
                    - ``line_items`` (list): Invoice line items

        Returns:
            WorkResult with invoice processing outcome including GL coding,
            match results, and approval/escalation flags.
        """
        invoice_data: Dict[str, Any] = work_item.data
        invoice_id: str = str(invoice_data.get("invoice_id", "unknown"))
        vendor_id: str = str(invoice_data.get("vendor_id", "unknown"))
        amount: float = float(invoice_data.get("amount", work_item.amount or 0.0))

        logger.info(
            "vendor_invoice_processing_started",
            agent_id=str(self.config.agent_id),
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            amount=amount,
        )

        # ---- Validate required fields ------------------------------------ #
        if not invoice_id or invoice_id == "unknown":
            logger.warning(
                "vendor_invoice_missing_id",
                agent_id=str(self.config.agent_id),
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Missing required field: invoice_id"},
                error="Missing invoice_id in vendor invoice data",
            )

        # ---- Authority check — escalate if above $50K -------------------- #
        if amount > self.APPROVAL_THRESHOLD_MAX:
            escalation_target = "cfo" if amount > 100000.0 else "controller"
            logger.info(
                "vendor_invoice_escalated",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                escalation_target=escalation_target,
            )
            return WorkResult(
                success=True,
                actions_taken=[f"escalated_to_{escalation_target}"],
                approval_needed=True,
                data={
                    "escalation_reason": (
                        f"Invoice amount ${amount:,.2f} exceeds AP Manager authority. "
                        f"Escalated to {escalation_target}."
                    ),
                    "escalation_target": escalation_target,
                    "amount": amount,
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                },
            )

        # ---- Build context for DecisionEngine ---------------------------- #
        po_data: Dict[str, Any] = invoice_data.get("purchase_order", {})
        receipt_data: Dict[str, Any] = invoice_data.get("goods_receipt", {})

        vendor_invoice_context: Dict[str, Any] = {
            "invoice": invoice_data,
            "po": po_data,
            "receipt": receipt_data,
            "amount": amount,
            "invoice_id": invoice_id,
            "vendor_id": vendor_id,
            "reviewer_role": self.ROLE,
            "line_items": invoice_data.get("line_items", []),
        }

        # ---- Make vendor invoice decision via DecisionEngine ------------- #
        try:
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="process_vendor_invoice",
                context=vendor_invoice_context,
            )
        except Exception as exc:
            logger.error(
                "vendor_invoice_decision_error",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_id,
                amount=amount,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["vendor_invoice_processing_failed"],
                has_exceptions=True,
                error=f"Decision engine error during vendor invoice processing: {str(exc)}",
                data={
                    "invoice_id": invoice_id,
                    "vendor_id": vendor_id,
                    "amount": amount,
                },
            )

        # ---- Extract decision outputs ------------------------------------ #
        gl_coding: Dict[str, Any] = decision.get("gl_coding", {})
        match_status: str = decision.get("match_status", "unknown")
        processing_notes: str = decision.get("processing_notes", "")
        recommendation: str = decision.get("recommendation", "approve").lower()

        # ---- Evaluate match result for variances ------------------------- #
        has_variance: bool = match_status not in ("matched", "full_match")
        actions_taken: List[str] = ["gl_coding_reviewed", "three_way_match_reviewed"]
        approval_needed: bool = False
        has_exceptions: bool = False

        if has_variance and recommendation in ("escalate", "reject"):
            # Significant variance detected — escalate or reject
            actions_taken.append("variance_detected")
            if recommendation == "escalate":
                actions_taken.append("escalated_to_controller")
                approval_needed = True
                has_exceptions = True
            else:
                actions_taken.append("invoice_rejected")
                has_exceptions = True
        elif has_variance:
            # Minor variance — manager accepts with note
            actions_taken.append("variance_accepted_with_note")
        else:
            # Clean match — approve
            actions_taken.append("invoice_approved")

        logger.info(
            "vendor_invoice_processed",
            agent_id=str(self.config.agent_id),
            invoice_id=invoice_id,
            vendor_id=vendor_id,
            amount=amount,
            match_status=match_status,
            has_variance=has_variance,
            recommendation=recommendation,
            actions_taken=actions_taken,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            has_exceptions=has_exceptions,
            data={
                "gl_coding": gl_coding,
                "match_status": match_status,
                "processing_notes": processing_notes,
                "recommendation": recommendation,
                "invoice_id": invoice_id,
                "vendor_id": vendor_id,
                "amount": amount,
                "reviewer": self.ROLE,
                "reviewer_agent_id": str(self.config.agent_id),
            },
        )
