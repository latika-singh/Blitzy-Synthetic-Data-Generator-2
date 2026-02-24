"""Purchasing Manager Agent — PO approval and vendor management.

This module implements the :class:`PurchasingManagerAgent`, the managerial
authority for the purchasing function within the ERP simulation.  The
Purchasing Manager is responsible for:

    * **Purchase order approval** within the $5,000 – $25,000 authority band.
    * **Vendor management** — reviewing vendor performance, evaluating vendor
      relationships, and making strategic vendor decisions.
    * **PO escalation** — automatically escalating POs above $25K to the
      Controller for higher-level approval.
    * **Purchase order processing** — manager-level PO creation for higher-value
      or strategic purchases.

Approval authority per the specification:
    * POs < $5,000: No approval required (auto-approved by PurchasingAgent).
    * POs $5,000 – $25,000: **Purchasing Manager approval** (this agent).
    * POs > $25,000: Escalated to Controller.
    * POs > $100,000: Escalated to CFO (via Controller).

Specification references:
    * README.md line 317 — PO $5K–$25K requires purchasing_manager
    * README.md line 318 — PO > $25K escalated to controller
    * README.md line 1350 — ROLE_MAPPING: "purchase_order": ["purchasing_agent",
      "purchasing_manager"]
    * AAP Section 0.5.1 Group 5 — "PurchasingManagerAgent: PO approval,
      vendor management"
    * AAP Section 0.7.1 — Must extend ``BaseAgent``, constructor injection,
      ``structlog`` logging

Exports:
    PurchasingManagerAgent — Specialized agent class for purchasing manager role.
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
# PurchasingManagerAgent — PO approval and vendor management
# ---------------------------------------------------------------------------
class PurchasingManagerAgent(BaseAgent):
    """Purchasing Manager agent for PO approval and vendor management.

    The Purchasing Manager sits **above the Purchasing Agent** and **below
    the Controller** in the approval hierarchy.  Its primary authority
    band is $5,000 – $25,000 for purchase order approvals.  Transactions
    exceeding $25,000 are escalated to the Controller.

    Responsibilities:
        - **PO approval** — reviewing and approving purchase orders in the
          $5K–$25K range based on vendor risk, budget impact, and compliance.
        - **PO processing** — manager-level purchase order creation for
          strategic or high-value procurements within authority.
        - **Vendor review** — evaluating vendor performance data, making
          decisions on vendor status (approved/probation/suspended), and
          managing ongoing vendor relationships.
        - **Escalation routing** — automatically escalating amounts above
          $25K to the Controller for higher-level approval.

    Role: ``"purchasing_manager"``

    Transaction types handled:
        - ``"po_approval"`` — purchase order approval within authority band
        - ``"purchase_order"`` — manager-level PO creation/processing
        - ``"vendor_review"`` — vendor performance review and status decisions

    Approval authority (from README.md lines 315–320):

        +---------------------+----------+----------+
        | Authority Level     | Min ($)  | Max ($)  |
        +=====================+==========+==========+
        | Purchase Order      |  5,000   |  25,000  |
        +---------------------+----------+----------+

        Amounts below $5K require no manager approval (handled by
        PurchasingAgent).  Amounts above $25K are escalated to the
        Controller.

    Inherits from :class:`BaseAgent`:
        - Constructor injection (config, memory, decision_engine, action_registry)
        - Async ``run()`` loop for continuous work item processing
        - ``make_decision()`` for 4-layer Decision Engine delegation
        - ``_calculate_importance()`` for memory scoring
        - Performance metrics tracking

    Example::

        pm = PurchasingManagerAgent(
            config=pm_config,
            memory=pm_memory,
            decision_engine=decision_engine,
            action_registry=action_registry,
        )
        # Enqueue a PO approval request
        await pm.work_queue.put(WorkItem(
            type="po_approval",
            data={"po_id": "PO-001", "vendor_id": "V-100", "amount": 15000.0},
            amount=15000.0,
        ))
        await pm.run()
    """

    # ------------------------------------------------------------------
    # Class Constants
    # ------------------------------------------------------------------

    ROLE: str = "purchasing_manager"
    """Agent role identifier. Must match ``"purchasing_manager"`` in
    VALID_AGENT_ROLES (agent_config.py line 112)."""

    SUPPORTED_WORK_TYPES: List[str] = [
        "po_approval",
        "purchase_order",
        "vendor_review",
    ]
    """Work item types this agent can process. Used by the
    WorkflowOrchestrator for routing validation."""

    APPROVAL_THRESHOLD_MIN: float = 5_000.0
    """Minimum monetary threshold ($5K) for Purchasing Manager approval.
    POs below this amount are auto-approved by PurchasingAgent.
    From README.md line 317."""

    APPROVAL_THRESHOLD_MAX: float = 25_000.0
    """Maximum monetary threshold ($25K) for Purchasing Manager authority.
    POs above this amount are escalated to the Controller.
    From README.md line 318."""

    # ------------------------------------------------------------------
    # process_work_item — Main routing dispatcher (abstract method impl)
    # ------------------------------------------------------------------
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Route a work item to the appropriate purchasing manager method.

        Dispatches based on ``work_item.type``:
          - ``"po_approval"``
            → :meth:`_process_po_approval`
          - ``"purchase_order"``
            → :meth:`_process_purchase_order`
          - ``"vendor_review"``
            → :meth:`_review_vendor`

        Unsupported types raise :class:`ValueError` so the base-class
        ``run()`` loop can capture and log the error.

        Args:
            work_item: The work item received from the orchestration layer.

        Returns:
            WorkResult indicating the processing outcome, actions taken,
            and any supplementary result data.

        Raises:
            ValueError: If ``work_item.type`` is not in
                :attr:`SUPPORTED_WORK_TYPES`.
        """
        logger.info(
            "purchasing_manager_processing",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            work_item_type=work_item.type,
            amount=work_item.amount,
            description=work_item.description,
        )

        work_type: str = work_item.type

        try:
            if work_type == "po_approval":
                return await self._process_po_approval(work_item)

            if work_type == "purchase_order":
                return await self._process_purchase_order(work_item)

            if work_type == "vendor_review":
                return await self._review_vendor(work_item)

            # Unsupported type — raise so run() captures it in ERROR state
            raise ValueError(
                f"PurchasingManagerAgent does not support work item type "
                f"'{work_type}'. Supported types: {self.SUPPORTED_WORK_TYPES}"
            )

        except ValueError:
            # Re-raise ValueError for unsupported types
            raise

        except Exception as exc:
            # Catch-all for unexpected errors within handler methods
            logger.error(
                "purchasing_manager_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=["purchasing_manager_processing_failed"],
                has_exceptions=True,
                data={"error": str(exc), "work_item_type": work_type},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # PO Approval — Purchase order approval within authority ($5K–$25K)
    # ------------------------------------------------------------------
    async def _process_po_approval(self, work_item: WorkItem) -> WorkResult:
        """Process a purchase order approval request.

        The Purchasing Manager reviews POs that fall within its authority
        band ($5K–$25K).  If the amount exceeds $25K, the request is
        automatically escalated to the Controller.

        Processing flow:
          1. Extract PO data and monetary amount from the work item.
          2. If amount > $25K (APPROVAL_THRESHOLD_MAX):
             → escalate to Controller (return with ``approval_needed=True``).
          3. Build decision context with transaction data, vendor info,
             compliance traits, and budget considerations.
          4. Invoke ``self.make_decision()`` with
             ``decision_type="approve_transaction"`` for Decision Engine
             evaluation.
          5. Validate action inputs via ``action_registry.validate_inputs()``.
          6. Execute the ``approve_purchase_order`` action via the
             ``action_registry.execute()`` method.
          7. Evaluate the decision result and return appropriate WorkResult
             (approved, rejected, or escalated).
          8. Log the decision with full structured context including amount,
             decision outcome, and vendor name.

        Args:
            work_item: The work item containing PO data for approval.

        Returns:
            WorkResult with approval, rejection, or escalation outcome.
        """
        # Extract key fields from work item data
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = work_item.amount if work_item.amount is not None else (
            transaction_data.get("amount", 0.0)
        )
        po_id: str = transaction_data.get("po_id", "")
        vendor_id: str = transaction_data.get("vendor_id", "")
        vendor_name: str = transaction_data.get("vendor_name", "")
        requester: str = transaction_data.get("requester", "")
        escalation_chain: List[str] = transaction_data.get(
            "escalation_chain", []
        )

        # -----------------------------------------------------------------
        # Controller Escalation Check — POs above $25K
        # -----------------------------------------------------------------
        if amount > self.APPROVAL_THRESHOLD_MAX:
            logger.info(
                "purchasing_manager_escalated_to_controller",
                agent_id=str(self.config.agent_id),
                amount=amount,
                po_id=po_id,
                reason="amount_exceeds_purchasing_manager_authority",
                threshold_max=self.APPROVAL_THRESHOLD_MAX,
                vendor_name=vendor_name,
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
                        f"PO amount ${amount:,.2f} exceeds purchasing manager "
                        f"authority limit of ${self.APPROVAL_THRESHOLD_MAX:,.2f}"
                    ),
                    "amount": amount,
                    "po_id": po_id,
                    "vendor_id": vendor_id,
                    "vendor_name": vendor_name,
                    "requester": requester,
                    "escalation_chain": escalation_chain + ["purchasing_manager"],
                    "original_work_item_type": work_item.type,
                },
            )

        # -----------------------------------------------------------------
        # Within Authority — Make approval decision via Decision Engine
        # -----------------------------------------------------------------
        context: Dict[str, Any] = {
            "transaction": transaction_data,
            "amount": amount,
            "transaction_type": "purchase_order",
            "vendor": transaction_data.get("vendor", {}),
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "po_id": po_id,
            "requester": requester,
            "approval_level": "purchasing_manager",
            "authority_range": {
                "min": self.APPROVAL_THRESHOLD_MIN,
                "max": self.APPROVAL_THRESHOLD_MAX,
            },
            "manager_considerations": {
                "budget_impact": transaction_data.get(
                    "budget_impact", "unknown"
                ),
                "urgency": transaction_data.get("urgency", "normal"),
                "department": transaction_data.get("department", "unknown"),
                "justification": transaction_data.get("justification", ""),
                "vendor_performance": transaction_data.get(
                    "vendor_performance", {}
                ),
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
            # Validate and execute the approve_purchase_order action
            action_inputs: Dict[str, Any] = {
                "po_id": po_id,
                "approver_id": str(self.config.agent_id),
                "decision": "approve",
                "comments": reasoning,
            }

            validation_errors: List[str] = self.action_registry.validate_inputs(
                "approve_purchase_order", action_inputs
            )

            if not validation_errors:
                action_result = await self.action_registry.execute(
                    action_id="approve_purchase_order",
                    agent=self,
                    inputs=action_inputs,
                )

                logger.info(
                    "po_approval_decision",
                    agent_id=str(self.config.agent_id),
                    amount=amount,
                    po_id=po_id,
                    decision="approved",
                    reasoning=reasoning,
                    vendor_name=vendor_name,
                    action_success=action_result.success,
                )

                return WorkResult(
                    success=True,
                    actions_taken=[
                        "po_reviewed_by_purchasing_manager",
                        "po_approved",
                    ],
                    approval_needed=False,
                    has_exceptions=False,
                    data={
                        "approved": True,
                        "approver": "purchasing_manager",
                        "approver_agent_id": str(self.config.agent_id),
                        "amount": amount,
                        "po_id": po_id,
                        "vendor_id": vendor_id,
                        "vendor_name": vendor_name,
                        "reasoning": reasoning,
                        "escalation_chain": (
                            escalation_chain + ["purchasing_manager"]
                        ),
                        "action_result": action_result.outputs,
                    },
                )
            else:
                # Validation failure — log and return with exceptions flag
                logger.warning(
                    "po_approval_validation_failed",
                    agent_id=str(self.config.agent_id),
                    po_id=po_id,
                    validation_errors=validation_errors,
                )

                return WorkResult(
                    success=False,
                    actions_taken=["po_approval_validation_failed"],
                    has_exceptions=True,
                    data={
                        "validation_errors": validation_errors,
                        "po_id": po_id,
                        "amount": amount,
                    },
                    error=(
                        f"Action input validation failed: "
                        f"{'; '.join(validation_errors)}"
                    ),
                )

        # -----------------------------------------------------------------
        # Rejection path — PO rejected with reasoning
        # -----------------------------------------------------------------
        # Execute rejection action
        reject_inputs: Dict[str, Any] = {
            "po_id": po_id,
            "approver_id": str(self.config.agent_id),
            "decision": "reject",
            "comments": reasoning,
        }

        reject_validation_errors: List[str] = self.action_registry.validate_inputs(
            "approve_purchase_order", reject_inputs
        )

        if not reject_validation_errors:
            await self.action_registry.execute(
                action_id="approve_purchase_order",
                agent=self,
                inputs=reject_inputs,
            )

        logger.info(
            "po_approval_decision",
            agent_id=str(self.config.agent_id),
            amount=amount,
            po_id=po_id,
            decision="rejected",
            reasoning=reasoning,
            vendor_name=vendor_name,
        )

        return WorkResult(
            success=True,
            actions_taken=[
                "po_reviewed_by_purchasing_manager",
                "po_rejected",
            ],
            approval_needed=False,
            has_exceptions=False,
            data={
                "approved": False,
                "approver": "purchasing_manager",
                "approver_agent_id": str(self.config.agent_id),
                "amount": amount,
                "po_id": po_id,
                "vendor_id": vendor_id,
                "vendor_name": vendor_name,
                "rejection_reason": reasoning,
                "escalation_chain": (
                    escalation_chain + ["purchasing_manager"]
                ),
            },
        )

    # ------------------------------------------------------------------
    # Purchase Order Processing — Manager-level PO creation
    # ------------------------------------------------------------------
    async def _process_purchase_order(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a manager-level purchase order creation request.

        The Purchasing Manager can create purchase orders directly,
        typically for strategic or higher-value procurements that warrant
        manager oversight from inception rather than being initiated by
        a Purchasing Agent and routed for approval.

        Processing flow:
          1. Extract PO request data including vendor, items, and amount.
          2. Build decision context with vendor selection rationale,
             budget context, and strategic considerations.
          3. Invoke ``self.make_decision()`` with
             ``decision_type="approve_transaction"`` for the Decision
             Engine to evaluate whether to proceed with PO creation.
          4. If the decision is to proceed:
             a. Validate inputs via ``action_registry.validate_inputs()``.
             b. Execute ``create_purchase_order`` action.
             c. If amount > APPROVAL_THRESHOLD_MAX, flag for additional
                approval from Controller.
          5. Log the PO creation with full structured context.

        Args:
            work_item: The work item containing PO creation data.

        Returns:
            WorkResult with PO creation outcome and approval needs.
        """
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = work_item.amount if work_item.amount is not None else (
            transaction_data.get("total_amount", 0.0)
        )
        vendor_id: str = transaction_data.get("vendor_id", "")
        vendor_name: str = transaction_data.get("vendor_name", "")
        items: List[Dict[str, Any]] = transaction_data.get("items", [])
        company_id: str = transaction_data.get(
            "company_id", str(self.config.company_id) if self.config.company_id else ""
        )

        # Build context for the Decision Engine
        context: Dict[str, Any] = {
            "transaction": transaction_data,
            "amount": amount,
            "transaction_type": "purchase_order",
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "items": items,
            "approval_level": "purchasing_manager",
            "strategic_context": {
                "vendor_relationship": transaction_data.get(
                    "vendor_relationship", "standard"
                ),
                "contract_reference": transaction_data.get(
                    "contract_reference", ""
                ),
                "budget_category": transaction_data.get(
                    "budget_category", "operational"
                ),
                "urgency": transaction_data.get("urgency", "normal"),
            },
        }

        # Invoke Decision Engine for PO creation assessment
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        proceed: bool = decision.get(
            "approved", decision.get("decision", "") != "rejected"
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "PO creation reviewed")
        )

        if not proceed:
            logger.info(
                "purchase_order_creation_declined",
                agent_id=str(self.config.agent_id),
                amount=amount,
                vendor_id=vendor_id,
                reasoning=reasoning,
            )

            return WorkResult(
                success=True,
                actions_taken=["po_creation_reviewed", "po_creation_declined"],
                approval_needed=False,
                has_exceptions=False,
                data={
                    "created": False,
                    "manager_agent_id": str(self.config.agent_id),
                    "amount": amount,
                    "vendor_id": vendor_id,
                    "vendor_name": vendor_name,
                    "decline_reason": reasoning,
                },
            )

        # Validate and execute PO creation action
        action_inputs: Dict[str, Any] = {
            "vendor_id": vendor_id,
            "items": items if items else [{"description": "Manager-initiated procurement", "quantity": 1, "unit_price": amount}],
            "total_amount": float(amount),
            "company_id": company_id,
            "notes": reasoning,
        }

        validation_errors: List[str] = self.action_registry.validate_inputs(
            "create_purchase_order", action_inputs
        )

        if validation_errors:
            logger.warning(
                "purchase_order_creation_validation_failed",
                agent_id=str(self.config.agent_id),
                vendor_id=vendor_id,
                amount=amount,
                validation_errors=validation_errors,
            )

            return WorkResult(
                success=False,
                actions_taken=["po_creation_validation_failed"],
                has_exceptions=True,
                data={
                    "validation_errors": validation_errors,
                    "vendor_id": vendor_id,
                    "amount": amount,
                },
                error=(
                    f"PO creation validation failed: "
                    f"{'; '.join(validation_errors)}"
                ),
            )

        action_result = await self.action_registry.execute(
            action_id="create_purchase_order",
            agent=self,
            inputs=action_inputs,
        )

        if not action_result.success:
            logger.error(
                "purchase_order_creation_action_failed",
                agent_id=str(self.config.agent_id),
                vendor_id=vendor_id,
                amount=amount,
                error=action_result.error,
            )

            return WorkResult(
                success=False,
                actions_taken=["po_creation_attempted", "po_creation_action_failed"],
                has_exceptions=True,
                data={
                    "vendor_id": vendor_id,
                    "amount": amount,
                    "action_error": action_result.error,
                },
                error=action_result.error,
            )

        # Determine if additional approval is needed from Controller
        needs_controller_approval: bool = amount > self.APPROVAL_THRESHOLD_MAX
        actions_taken: List[str] = ["po_created_by_purchasing_manager"]

        if needs_controller_approval:
            actions_taken.append("flagged_for_controller_approval")

        logger.info(
            "purchase_order_created",
            agent_id=str(self.config.agent_id),
            amount=amount,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
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
                "manager_agent_id": str(self.config.agent_id),
                "amount": amount,
                "vendor_id": vendor_id,
                "vendor_name": vendor_name,
                "items_count": len(items) if items else 1,
                "company_id": company_id,
                "reasoning": reasoning,
                "needs_controller_approval": needs_controller_approval,
                "action_result": action_result.outputs,
            },
        )

    # ------------------------------------------------------------------
    # Vendor Review — Vendor performance evaluation and status management
    # ------------------------------------------------------------------
    async def _review_vendor(self, work_item: WorkItem) -> WorkResult:
        """Review vendor performance and make status decisions.

        The Purchasing Manager evaluates vendor performance data across
        multiple dimensions (delivery timeliness, quality, pricing,
        communication responsiveness) and decides on the vendor's
        operational status.

        Processing flow:
          1. Extract vendor data, performance metrics, and review context
             from the work item.
          2. Build a comprehensive decision context including:
             - Vendor historical performance data (on-time delivery rate,
               quality score, pricing competitiveness)
             - Current contract terms and compliance status
             - Recent interaction history (disputes, returns, credits)
             - Spend volume and strategic importance tier
          3. Invoke ``self.make_decision()`` to evaluate the vendor's
             standing using the Decision Engine.
          4. Determine the vendor status recommendation:
             - ``"approved"`` — vendor in good standing, continue business
             - ``"probation"`` — vendor underperforming, increased monitoring
             - ``"suspended"`` — vendor relationship suspended pending review
             - ``"preferred"`` — top-performing vendor, increase allocation
          5. Log the vendor review outcome with structured context.

        Args:
            work_item: The work item containing vendor review data.

        Returns:
            WorkResult with vendor status recommendation and review details.
        """
        transaction_data: Dict[str, Any] = work_item.data
        vendor_id: str = transaction_data.get("vendor_id", "")
        vendor_name: str = transaction_data.get("vendor_name", "")
        current_status: str = transaction_data.get("current_status", "approved")

        # Extract vendor performance metrics
        performance_metrics: Dict[str, Any] = transaction_data.get(
            "performance_metrics", {}
        )
        on_time_delivery_rate: float = performance_metrics.get(
            "on_time_delivery_rate", 0.0
        )
        quality_score: float = performance_metrics.get(
            "quality_score", 0.0
        )
        pricing_competitiveness: float = performance_metrics.get(
            "pricing_competitiveness", 0.0
        )
        dispute_count: int = performance_metrics.get(
            "dispute_count", 0
        )
        total_spend: float = performance_metrics.get(
            "total_spend", 0.0
        )
        tier: str = transaction_data.get("tier", "standard")

        # Build comprehensive decision context for the Decision Engine
        context: Dict[str, Any] = {
            "transaction": transaction_data,
            "transaction_type": "vendor_review",
            "vendor_id": vendor_id,
            "vendor_name": vendor_name,
            "current_status": current_status,
            "performance_metrics": performance_metrics,
            "review_dimensions": {
                "on_time_delivery_rate": on_time_delivery_rate,
                "quality_score": quality_score,
                "pricing_competitiveness": pricing_competitiveness,
                "dispute_count": dispute_count,
                "total_spend": total_spend,
                "tier": tier,
            },
            "contract_info": transaction_data.get("contract_info", {}),
            "recent_interactions": transaction_data.get(
                "recent_interactions", []
            ),
            "review_period": transaction_data.get(
                "review_period", "quarterly"
            ),
            "approval_level": "purchasing_manager",
        }

        # Invoke Decision Engine for vendor assessment
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Determine vendor status from decision result
        # Default to maintaining current status if decision is unclear
        recommended_status: str = decision.get(
            "vendor_status",
            decision.get("recommendation", current_status),
        )
        reasoning: str = decision.get(
            "reasoning", decision.get("notes", "Vendor review completed")
        )

        # Validate the recommended status is one of the known statuses
        valid_statuses: List[str] = [
            "approved", "probation", "suspended", "preferred",
        ]
        if recommended_status not in valid_statuses:
            recommended_status = current_status

        # Determine if there is a status change
        status_changed: bool = recommended_status != current_status

        actions_taken: List[str] = ["vendor_reviewed_by_purchasing_manager"]
        if status_changed:
            actions_taken.append(
                f"vendor_status_changed_to_{recommended_status}"
            )

        logger.info(
            "vendor_review_completed",
            agent_id=str(self.config.agent_id),
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            current_status=current_status,
            recommended_status=recommended_status,
            status_changed=status_changed,
            on_time_delivery_rate=on_time_delivery_rate,
            quality_score=quality_score,
            total_spend=total_spend,
            reasoning=reasoning,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=False,
            data={
                "vendor_id": vendor_id,
                "vendor_name": vendor_name,
                "previous_status": current_status,
                "recommended_status": recommended_status,
                "status_changed": status_changed,
                "reasoning": reasoning,
                "performance_summary": {
                    "on_time_delivery_rate": on_time_delivery_rate,
                    "quality_score": quality_score,
                    "pricing_competitiveness": pricing_competitiveness,
                    "dispute_count": dispute_count,
                    "total_spend": total_spend,
                    "tier": tier,
                },
                "reviewer_agent_id": str(self.config.agent_id),
                "review_period": transaction_data.get(
                    "review_period", "quarterly"
                ),
            },
        )
