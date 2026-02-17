"""Purchasing Agent (Buyer) — Specialized ERP agent for purchase order creation and vendor selection.

This module implements the PurchasingAgent class, which extends BaseAgent to handle
purchase order creation and purchase requisition processing within the ERP simulation.

The PurchasingAgent is responsible for:
- Creating purchase orders with vendor selection via Pareto 80/20 distribution
  (handled by the Statistical Layer of the DecisionEngine)
- Generating PO descriptions via the LLM Layer of the DecisionEngine
- Routing POs through the approval chain based on monetary thresholds:
    - < $5K: No approval needed
    - $5K - $25K: Purchasing Manager approval
    - $25K - $100K: Controller approval
    - > $100K: CFO approval
- Converting purchase requisitions into purchase orders

Role: "purchasing_agent"
Transaction types: "purchase_order", "purchase_requisition"
Actions used: create_purchase_order (from ActionRegistry)

Reference: README.md lines 315-320 (approval thresholds),
           lines 520-525 (PO amount distribution),
           lines 571-575 (vendor selection — Pareto 80/20),
           line 1350 (ROLE_MAPPING: purchase_order → purchasing_agent, purchasing_manager)
"""

import asyncio
from typing import Dict, Any, Optional, List

import structlog

from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry

# Module-level structured logger per AAP Section 0.7.6
logger = structlog.get_logger(__name__)


class PurchasingAgent(BaseAgent):
    """Purchasing Agent (Buyer).

    Responsible for:
    - Creating purchase orders
    - Selecting vendors based on Pareto 80/20 distribution
      (top 20% of vendors receive 80% of business, driven by the
       Statistical Layer of the DecisionEngine)
    - Determining PO amounts via log-normal distribution
      (mean ~$5,000, std ~$10,000, min $100, max $500K)
    - Generating natural-language PO descriptions via the LLM Layer
    - Routing POs for approval based on amount thresholds

    Role: "purchasing_agent"
    Transaction types handled: "purchase_order", "purchase_requisition"
    Actions used: create_purchase_order

    PO Approval thresholds (README.md lines 315-320):
    - < $5K: no approval needed
    - $5K - $25K: purchasing_manager
    - $25K - $100K: controller
    - > $100K: CFO

    Constructor parameters are injected via BaseAgent:
        config: AgentConfig — agent identity, role, personality traits
        memory: AgentMemory — dual-stream observation/reflection memory
        decision_engine: DecisionEngine — 4-layer hybrid decision pipeline
        action_registry: ActionRegistry — registry of 14 ERP action types
    """

    # ------------------------------------------------------------------ #
    #  Class-level constants                                              #
    # ------------------------------------------------------------------ #

    ROLE: str = "purchasing_agent"
    """Agent role identifier matching ROLE_MAPPING in WorkflowOrchestrator."""

    SUPPORTED_WORK_TYPES: List[str] = ["purchase_order", "purchase_requisition"]
    """Work item types this agent can process."""

    # Approval threshold boundaries (from README.md lines 315-320)
    NO_APPROVAL_THRESHOLD: float = 5_000.0
    """PO amounts below this value require no approval ($5K)."""

    MANAGER_APPROVAL_THRESHOLD: float = 25_000.0
    """PO amounts at or above $5K but below this value require purchasing_manager approval ($25K)."""

    CONTROLLER_APPROVAL_THRESHOLD: float = 100_000.0
    """PO amounts at or above $25K but below this value require controller approval ($100K).
    Amounts at or above this threshold require CFO approval."""

    # ------------------------------------------------------------------ #
    #  Constructor — inherited from BaseAgent                             #
    # ------------------------------------------------------------------ #
    # BaseAgent.__init__(self, config, memory, decision_engine, action_registry)
    # is used as-is; no additional initialization is required for the
    # PurchasingAgent, keeping agent creation lightweight (≥50 agents/sec).

    # ------------------------------------------------------------------ #
    #  Core work-processing entry point                                   #
    # ------------------------------------------------------------------ #

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a purchasing-related work item.

        Routes the work item to the appropriate handler based on its type:
        - ``"purchase_order"``  → :meth:`_create_purchase_order`
        - ``"purchase_requisition"`` → :meth:`_process_requisition`

        Args:
            work_item: The incoming work item containing type, data, amount,
                       and workflow context.

        Returns:
            A :class:`WorkResult` describing the outcome including actions
            taken, approval requirements, and any PO data produced.

        Raises:
            ValueError: If ``work_item.type`` is not in
                :attr:`SUPPORTED_WORK_TYPES`.
        """
        logger.info(
            "purchasing_agent_processing",
            agent_id=self.config.agent_id,
            agent_role=self.config.role,
            work_item_type=work_item.type,
            workflow_id=str(work_item.workflow_id),
        )

        if work_item.type == "purchase_order":
            return await self._create_purchase_order(work_item)
        elif work_item.type == "purchase_requisition":
            return await self._process_requisition(work_item)
        else:
            error_msg = (
                f"Unsupported work item type for PurchasingAgent: "
                f"'{work_item.type}'. "
                f"Supported types: {self.SUPPORTED_WORK_TYPES}"
            )
            logger.error(
                "purchasing_agent_unsupported_type",
                agent_id=self.config.agent_id,
                work_item_type=work_item.type,
                error=error_msg,
            )
            raise ValueError(error_msg)

    # ------------------------------------------------------------------ #
    #  Purchase order creation                                            #
    # ------------------------------------------------------------------ #

    async def _create_purchase_order(self, work_item: WorkItem) -> WorkResult:
        """Create a purchase order from the supplied work item data.

        Workflow:
        1. Extract PO data and identify the vendor (or request vendor
           selection from the DecisionEngine's Statistical Layer which
           applies Pareto 80/20 distribution).
        2. Use the DecisionEngine (via ``self.make_decision``) to generate
           a PO description and any supplementary context.
        3. Determine the total amount from the work item or the decision.
        4. Check whether approval is needed based on the PO amount
           thresholds and identify the required approver role.
        5. Assemble and return the :class:`WorkResult`.

        Args:
            work_item: Work item with ``type="purchase_order"`` and
                       associated PO data in ``work_item.data``.

        Returns:
            A :class:`WorkResult` containing the created PO data,
            actions taken, and approval routing information.
        """
        po_data: Dict[str, Any] = work_item.data
        vendor_id: Any = po_data.get("vendor_id")
        vendor_info: Dict[str, Any] = po_data.get("vendor", {})

        # ------------------------------------------------------------- #
        # Step 1: Vendor selection (Pareto 80/20 via DecisionEngine)     #
        # ------------------------------------------------------------- #
        # If the work item does not already specify a vendor, delegate
        # vendor selection to the DecisionEngine.  The Statistical Layer
        # applies the Pareto distribution to choose a vendor, and the
        # LLM Layer may generate a natural-language PO description.
        try:
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="generate_description",
                context={
                    "po_data": po_data,
                    "vendor": vendor_info,
                    "agent_role": self.ROLE,
                    "action": "create_purchase_order",
                },
            )
        except Exception as exc:
            logger.error(
                "purchasing_agent_decision_error",
                agent_id=self.config.agent_id,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["decision_failed"],
                data={"workflow_id": str(work_item.workflow_id)},
                error=f"Decision engine error during PO creation: {exc}",
            )

        # If a vendor was selected by the decision engine, use it
        if not vendor_id:
            vendor_id = decision.get("vendor_id", vendor_info.get("id"))

        # ------------------------------------------------------------- #
        # Step 2: Determine the total PO amount                          #
        # ------------------------------------------------------------- #
        # Amount may come from the work item directly, from po_data,
        # or from the decision engine's statistical sampling.
        total_amount: float = 0.0
        if work_item.amount is not None and work_item.amount > 0:
            total_amount = float(work_item.amount)
        elif po_data.get("total_amount") is not None:
            total_amount = float(po_data["total_amount"])
        elif po_data.get("amount") is not None:
            total_amount = float(po_data["amount"])
        elif decision.get("total_amount") is not None:
            total_amount = float(decision["total_amount"])
        elif decision.get("amount") is not None:
            total_amount = float(decision["amount"])

        # ------------------------------------------------------------- #
        # Step 3: Check approval requirements                            #
        # ------------------------------------------------------------- #
        approval_needed: bool = self._check_approval_needed(total_amount)
        approver_role: Optional[str] = self._get_approver_role(total_amount)

        # ------------------------------------------------------------- #
        # Step 4: Build actions_taken list                               #
        # ------------------------------------------------------------- #
        actions_taken: List[str] = ["purchase_order_created"]
        if approval_needed and approver_role:
            actions_taken.append("approval_requested")

        # ------------------------------------------------------------- #
        # Step 5: Assemble result data                                   #
        # ------------------------------------------------------------- #
        result_data: Dict[str, Any] = {
            "po_data": po_data,
            "vendor_id": vendor_id,
            "total_amount": total_amount,
            "description": decision.get("description", ""),
            "processing_notes": decision.get("processing_notes", ""),
        }
        if approval_needed and approver_role:
            result_data["approver_role"] = approver_role
            result_data["approval_reason"] = (
                f"PO amount ${total_amount:,.2f} requires {approver_role} approval"
            )

        logger.info(
            "purchase_order_created",
            agent_id=self.config.agent_id,
            vendor_id=vendor_id,
            amount=total_amount,
            approval_needed=approval_needed,
            approver_role=approver_role,
            workflow_id=str(work_item.workflow_id),
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            data=result_data,
        )

    # ------------------------------------------------------------------ #
    #  Purchase requisition processing                                    #
    # ------------------------------------------------------------------ #

    async def _process_requisition(self, work_item: WorkItem) -> WorkResult:
        """Convert a purchase requisition into a purchase order.

        A purchase requisition is an internal request for goods or services
        that must be converted into a formal purchase order.  This method
        validates the requisition data, enriches it, and delegates to the
        PO creation workflow.

        Args:
            work_item: Work item with ``type="purchase_requisition"`` and
                       requisition data in ``work_item.data``.

        Returns:
            A :class:`WorkResult` containing the resulting PO data and
            approval routing information.
        """
        requisition_data: Dict[str, Any] = work_item.data

        # ------------------------------------------------------------- #
        # Step 1: Validate requisition data                              #
        # ------------------------------------------------------------- #
        requisition_id: Any = requisition_data.get("requisition_id")
        requestor: Any = requisition_data.get("requestor")
        items: List[Dict[str, Any]] = requisition_data.get("items", [])

        if not items and not requisition_data.get("description"):
            logger.warning(
                "purchasing_agent_empty_requisition",
                agent_id=self.config.agent_id,
                requisition_id=requisition_id,
            )
            return WorkResult(
                success=False,
                actions_taken=["requisition_validation_failed"],
                data={"requisition_id": requisition_id},
                error="Requisition contains no items or description",
            )

        logger.info(
            "purchasing_agent_processing_requisition",
            agent_id=self.config.agent_id,
            requisition_id=requisition_id,
            requestor=requestor,
            item_count=len(items),
        )

        # ------------------------------------------------------------- #
        # Step 2: Enrich requisition into PO-compatible work item        #
        # ------------------------------------------------------------- #
        # Build a synthetic PO work item from the requisition, carrying
        # over all relevant fields and marking the source as a requisition.
        po_data: Dict[str, Any] = {
            **requisition_data,
            "source": "purchase_requisition",
            "requisition_id": requisition_id,
            "requestor": requestor,
        }

        # Create an equivalent work item with purchase_order semantics
        po_work_item = WorkItem(
            type="purchase_order",
            data=po_data,
            workflow_id=work_item.workflow_id,
            priority=work_item.priority,
            amount=work_item.amount,
            created_at=work_item.created_at,
            description=work_item.description or f"PO from requisition {requisition_id}",
        )

        # ------------------------------------------------------------- #
        # Step 3: Delegate to PO creation workflow                       #
        # ------------------------------------------------------------- #
        result: WorkResult = await self._create_purchase_order(po_work_item)

        # Augment the actions list to reflect the requisition conversion
        if "purchase_order_created" in result.actions_taken:
            enriched_actions: List[str] = ["requisition_converted"] + result.actions_taken
        else:
            enriched_actions = result.actions_taken

        # Merge requisition metadata into result data
        enriched_data: Dict[str, Any] = {
            **result.data,
            "requisition_id": requisition_id,
            "requestor": requestor,
            "source": "purchase_requisition",
        }

        logger.info(
            "purchase_requisition_processed",
            agent_id=self.config.agent_id,
            requisition_id=requisition_id,
            po_created=result.success,
            approval_needed=result.approval_needed,
        )

        return WorkResult(
            success=result.success,
            actions_taken=enriched_actions,
            approval_needed=result.approval_needed,
            has_exceptions=result.has_exceptions,
            data=enriched_data,
            error=result.error,
            processing_time_seconds=result.processing_time_seconds,
        )

    # ------------------------------------------------------------------ #
    #  Approval helpers                                                   #
    # ------------------------------------------------------------------ #

    def _check_approval_needed(self, amount: float) -> bool:
        """Determine whether the given PO amount requires approval.

        Any PO with a total amount at or above the
        :attr:`NO_APPROVAL_THRESHOLD` ($5,000) requires approval from at
        least a purchasing manager.

        Args:
            amount: The total purchase order amount in USD.

        Returns:
            ``True`` if the amount meets or exceeds the no-approval
            threshold; ``False`` otherwise.
        """
        return amount >= self.NO_APPROVAL_THRESHOLD

    def _get_approver_role(self, amount: float) -> Optional[str]:
        """Determine the required approver role for a given PO amount.

        Approval thresholds (README.md lines 315-320):
        - < $5K   → ``None`` (no approval needed)
        - $5K–$25K  → ``"purchasing_manager"``
        - $25K–$100K → ``"controller"``
        - ≥ $100K   → ``"cfo"``

        Args:
            amount: The total purchase order amount in USD.

        Returns:
            The role string of the required approver, or ``None`` if no
            approval is needed.
        """
        if amount < self.NO_APPROVAL_THRESHOLD:
            return None
        elif amount < self.MANAGER_APPROVAL_THRESHOLD:
            return "purchasing_manager"
        elif amount < self.CONTROLLER_APPROVAL_THRESHOLD:
            return "controller"
        else:
            return "cfo"
