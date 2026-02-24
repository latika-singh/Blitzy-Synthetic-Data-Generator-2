"""Warehouse Manager agent implementation.

This module provides the WarehouseManagerAgent class, a specialized agent
responsible for overseeing warehouse operations within the ERP simulation.
The Warehouse Manager reviews and approves operational decisions escalated
by the WarehouseClerkAgent, manages receipt discrepancies, handles shipment
exceptions, and conducts periodic inventory reviews.

Role: "warehouse_manager"
Escalation path: Controller (for significant issues requiring financial oversight)

Transaction types handled:
    - warehouse_approval: Approval of warehouse operations (goods receipts, adjustments)
    - receipt_discrepancy: Review of goods receipt discrepancies flagged by clerk
    - shipment_exception: Resolution of shipment issues (delays, damages, wrong items)
    - inventory_review: Periodic inventory count reviews and adjustment approvals

Architecture:
    Extends BaseAgent with constructor injection of AgentConfig, AgentMemory,
    DecisionEngine, and ActionRegistry. All LLM-assisted decisions flow through
    the inherited make_decision() method which invokes the 4-layer DecisionEngine
    pipeline (Statistical → LLM → Validation → Deterministic).
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


class WarehouseManagerAgent(BaseAgent):
    """Warehouse Manager agent responsible for warehouse operations oversight.

    The Warehouse Manager oversees all warehouse operations including goods
    receipt approvals, shipment exception management, inventory adjustment
    reviews, and discrepancy resolution. This role acts as the operational
    authority for warehouse processes, reviewing work escalated by the
    WarehouseClerkAgent and escalating significant issues to the Controller.

    Responsibilities:
        - Overseeing warehouse operations and approving operational decisions
        - Approving goods receipt discrepancies (quantity variances, damage claims)
        - Managing shipment exceptions (delays, damages, wrong items, shortages)
        - Reviewing inventory adjustments and count variances
        - Escalating significant warehouse issues to Controller for financial review

    Role: "warehouse_manager"
    Transaction types handled:
        - "warehouse_approval": General warehouse operation approvals
        - "receipt_discrepancy": Goods receipt discrepancy resolution
        - "shipment_exception": Shipment issue management
        - "inventory_review": Inventory count and adjustment review

    Note:
        No explicit monetary approval thresholds are defined for this role in
        the specification. The Warehouse Manager operates at an operational
        oversight level, using LLM-assisted decisions for complex judgment calls
        and escalating to Controller when financial impact is significant.

    Example:
        >>> config = AgentConfig(agent_id="wm-001", role="warehouse_manager", ...)
        >>> agent = WarehouseManagerAgent(config, memory, decision_engine, action_registry)
        >>> result = await agent.process_work_item(work_item)
    """

    # Class-level role identifier matching the agent role configuration
    ROLE: str = "warehouse_manager"

    # Supported work item types that this agent can process
    SUPPORTED_WORK_TYPES: List[str] = [
        "warehouse_approval",
        "receipt_discrepancy",
        "shipment_exception",
        "inventory_review",
    ]

    # Escalation threshold: discrepancies above this value are escalated to Controller
    _ESCALATION_VALUE_THRESHOLD: float = 25000.0

    # Maximum acceptable quantity variance percentage before escalation
    _MAX_QUANTITY_VARIANCE_PCT: float = 0.10  # 10% variance triggers escalation

    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        decision_engine: DecisionEngine,
        action_registry: ActionRegistry,
    ) -> None:
        """Initialize the WarehouseManagerAgent with injected dependencies.

        All dependencies are passed through to the BaseAgent constructor,
        which manages the agent lifecycle, work queue, and state transitions.

        Args:
            config: Agent configuration including agent_id, role, personality
                traits (thoroughness, risk_tolerance, efficiency, compliance),
                and work schedule settings.
            memory: Dual-stream memory system for recording observations and
                reflections during warehouse operations processing.
            decision_engine: 4-layer hybrid decision pipeline used for
                approval decisions and exception handling via make_decision().
            action_registry: Registry of ERP action types available for
                executing warehouse-related actions (receive_goods, ship_order).
        """
        super().__init__(
            config=config,
            memory=memory,
            decision_engine=decision_engine,
            action_registry=action_registry,
        )
        # Verify essential warehouse actions are available in the registry
        self._warehouse_actions: List[str] = ["receive_goods", "ship_order"]
        logger.info(
            "warehouse_manager_agent_initialized",
            agent_id=str(self.config.agent_id),
            role=self.config.role,
            action_registry_available=self.action_registry is not None,
            decision_engine_available=self.decision_engine is not None,
        )

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a warehouse management work item by routing to the appropriate handler.

        Routes incoming work items to specialized handler methods based on the
        work item type. Each handler implements the specific business logic for
        that warehouse operation type, using LLM-assisted decisions where
        appropriate.

        Args:
            work_item: The work item to process, containing:
                - type: One of SUPPORTED_WORK_TYPES
                - data: Payload with operation-specific details
                - amount: Optional monetary value of the operation
                - description: Optional human-readable description

        Returns:
            WorkResult with processing outcome including success status,
            actions taken, whether further approval is needed, exception
            flags, and detailed result data.

        Raises:
            ValueError: If work_item.type is not in SUPPORTED_WORK_TYPES.
        """
        logger.info(
            "warehouse_manager_processing",
            agent_id=str(self.config.agent_id),
            work_item_type=work_item.type,
            description=work_item.description,
            amount=work_item.amount,
            agent_state=str(self.state),
        )

        try:
            if work_item.type == "warehouse_approval":
                return await self._process_warehouse_approval(work_item)
            elif work_item.type == "receipt_discrepancy":
                return await self._handle_receipt_discrepancy(work_item)
            elif work_item.type == "shipment_exception":
                return await self._handle_shipment_exception(work_item)
            elif work_item.type == "inventory_review":
                return await self._process_inventory_review(work_item)
            else:
                raise ValueError(
                    f"Unsupported work item type for WarehouseManagerAgent: "
                    f"'{work_item.type}'. Supported types: {self.SUPPORTED_WORK_TYPES}"
                )
        except ValueError:
            # Re-raise ValueError for unsupported work types
            raise
        except Exception as exc:
            logger.error(
                "warehouse_manager_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_item.type,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["processing_failed"],
                approval_needed=False,
                has_exceptions=True,
                data={"error_type": type(exc).__name__, "error_detail": str(exc)},
                error=f"Warehouse manager processing failed: {str(exc)}",
            )

    async def _process_warehouse_approval(self, work_item: WorkItem) -> WorkResult:
        """Process a warehouse operation requiring manager approval.

        Reviews warehouse operations escalated by the WarehouseClerkAgent or
        submitted for managerial oversight. Uses the DecisionEngine to evaluate
        the operation and determine whether to approve, reject, or escalate.

        Approval scenarios include:
            - Goods receipt approvals for high-value deliveries
            - Warehouse procedure change approvals
            - Equipment or resource allocation approvals
            - Special handling instruction approvals

        Args:
            work_item: Work item containing warehouse operation details in data,
                including operation_type, requestor, justification, and value.

        Returns:
            WorkResult indicating approval decision with detailed reasoning.
        """
        operation_data: Dict[str, Any] = work_item.data
        operation_type: str = operation_data.get("operation_type", "general")
        operation_value: float = work_item.amount or operation_data.get("value", 0.0)

        logger.info(
            "warehouse_approval_started",
            agent_id=str(self.config.agent_id),
            operation_type=operation_type,
            operation_value=operation_value,
        )

        # Check if the operation value warrants escalation to Controller
        if operation_value > self._ESCALATION_VALUE_THRESHOLD:
            logger.info(
                "warehouse_approval_escalated",
                agent_id=str(self.config.agent_id),
                operation_type=operation_type,
                operation_value=operation_value,
                escalation_reason="value_exceeds_threshold",
                threshold=self._ESCALATION_VALUE_THRESHOLD,
            )
            return WorkResult(
                success=True,
                actions_taken=["warehouse_operation_reviewed", "escalated_to_controller"],
                approval_needed=True,
                has_exceptions=False,
                data={
                    "approved": False,
                    "escalated": True,
                    "escalation_target": "controller",
                    "escalation_reason": (
                        f"Operation value ${operation_value:,.2f} exceeds warehouse "
                        f"manager threshold of ${self._ESCALATION_VALUE_THRESHOLD:,.2f}"
                    ),
                    "operation_type": operation_type,
                    "operation_value": operation_value,
                },
                error=None,
            )

        # Use the DecisionEngine for approval decision via inherited make_decision()
        context: Dict[str, Any] = {
            "warehouse_operation": operation_data,
            "operation_type": operation_type,
            "operation_value": operation_value,
            "agent_role": self.ROLE,
            "agent_traits": self.config.traits if hasattr(self.config, "traits") else {},
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Extract approval outcome from decision
        approved: bool = decision.get("approved", decision.get("decision", "") == "approve")
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))

        if approved:
            actions_taken: List[str] = ["warehouse_operation_reviewed", "warehouse_operation_approved"]
            # Record observation in agent memory for future context
            await self.memory.add_observation(
                f"Approved warehouse {operation_type} operation "
                f"valued at ${operation_value:,.2f}"
            )
            logger.info(
                "warehouse_approval_granted",
                agent_id=str(self.config.agent_id),
                operation_type=operation_type,
                operation_value=operation_value,
                reasoning=reasoning,
            )
        else:
            actions_taken = ["warehouse_operation_reviewed", "warehouse_operation_rejected"]
            await self.memory.add_observation(
                f"Rejected warehouse {operation_type} operation "
                f"valued at ${operation_value:,.2f}: {reasoning}"
            )
            logger.info(
                "warehouse_approval_rejected",
                agent_id=str(self.config.agent_id),
                operation_type=operation_type,
                operation_value=operation_value,
                reasoning=reasoning,
            )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=False,
            data={
                "approved": approved,
                "operation_type": operation_type,
                "operation_value": operation_value,
                "reasoning": reasoning,
                "decision_details": decision,
            },
            error=None,
        )

    async def _handle_receipt_discrepancy(self, work_item: WorkItem) -> WorkResult:
        """Handle goods receipt discrepancies flagged by the WarehouseClerkAgent.

        Reviews discrepancies between expected and actual goods receipts,
        including quantity variances, quality issues, and damaged goods. The
        manager decides whether to accept the discrepancy (adjust inventory),
        reject the shipment (return to vendor), or escalate to Controller.

        Resolution options:
            - accept: Accept discrepancy and adjust inventory records accordingly
            - reject: Reject the receipt and initiate return-to-vendor process
            - partial_accept: Accept usable portion, reject the remainder
            - escalate: Escalate to Controller for significant financial impact

        Args:
            work_item: Work item containing discrepancy details including
                po_id, expected quantities, received quantities, variance
                amounts, and any damage or quality notes.

        Returns:
            WorkResult with resolution decision and updated inventory actions.
        """
        discrepancy_data: Dict[str, Any] = work_item.data
        po_id: str = discrepancy_data.get("po_id", "unknown")
        expected_qty: float = discrepancy_data.get("expected_quantity", 0.0)
        received_qty: float = discrepancy_data.get("received_quantity", 0.0)
        discrepancy_value: float = work_item.amount or discrepancy_data.get(
            "discrepancy_value", 0.0
        )
        discrepancy_type: str = discrepancy_data.get(
            "discrepancy_type", "quantity_variance"
        )

        # Calculate quantity variance percentage
        quantity_variance_pct: float = 0.0
        if expected_qty > 0:
            quantity_variance_pct = abs(expected_qty - received_qty) / expected_qty

        logger.info(
            "receipt_discrepancy_review_started",
            agent_id=str(self.config.agent_id),
            po_id=po_id,
            expected_quantity=expected_qty,
            received_quantity=received_qty,
            quantity_variance_pct=round(quantity_variance_pct, 4),
            discrepancy_value=discrepancy_value,
            discrepancy_type=discrepancy_type,
        )

        # Auto-escalate if discrepancy value or variance is significant
        needs_escalation: bool = (
            discrepancy_value > self._ESCALATION_VALUE_THRESHOLD
            or quantity_variance_pct > self._MAX_QUANTITY_VARIANCE_PCT
        )

        if needs_escalation and discrepancy_value > self._ESCALATION_VALUE_THRESHOLD:
            logger.info(
                "receipt_discrepancy_escalated",
                agent_id=str(self.config.agent_id),
                po_id=po_id,
                discrepancy_value=discrepancy_value,
                escalation_reason="high_value_discrepancy",
            )
            return WorkResult(
                success=True,
                actions_taken=[
                    "receipt_discrepancy_reviewed",
                    "escalated_to_controller",
                ],
                approval_needed=True,
                has_exceptions=True,
                data={
                    "resolution": "escalate",
                    "escalation_target": "controller",
                    "po_id": po_id,
                    "discrepancy_type": discrepancy_type,
                    "discrepancy_value": discrepancy_value,
                    "quantity_variance_pct": round(quantity_variance_pct, 4),
                    "escalation_reason": (
                        f"Discrepancy value ${discrepancy_value:,.2f} requires "
                        f"controller oversight"
                    ),
                },
                error=None,
            )

        # Use DecisionEngine for discrepancy resolution
        context: Dict[str, Any] = {
            "discrepancy": discrepancy_data,
            "po_id": po_id,
            "expected_quantity": expected_qty,
            "received_quantity": received_qty,
            "quantity_variance_pct": quantity_variance_pct,
            "discrepancy_value": discrepancy_value,
            "discrepancy_type": discrepancy_type,
            "agent_role": self.ROLE,
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="handle_exception",
            context=context,
        )

        # Parse resolution from decision
        resolution: str = decision.get("resolution", "accept")
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))
        actions_taken: List[str] = ["receipt_discrepancy_reviewed"]

        if resolution == "accept":
            actions_taken.append("discrepancy_accepted")
            actions_taken.append("inventory_adjusted")
            has_exceptions = False
        elif resolution == "reject":
            actions_taken.append("discrepancy_rejected")
            actions_taken.append("return_to_vendor_initiated")
            has_exceptions = True
        elif resolution == "partial_accept":
            actions_taken.append("partial_receipt_accepted")
            actions_taken.append("partial_return_initiated")
            has_exceptions = True
        elif resolution == "escalate":
            actions_taken.append("escalated_to_controller")
            has_exceptions = True
        else:
            # Default to accept for unrecognized resolutions
            actions_taken.append("discrepancy_accepted")
            actions_taken.append("inventory_adjusted")
            has_exceptions = False

        approval_needed: bool = resolution == "escalate"

        # Record observation in agent memory
        await self.memory.add_observation(
            f"Receipt discrepancy for PO {po_id}: resolution={resolution}, "
            f"variance={quantity_variance_pct:.1%}, value=${discrepancy_value:,.2f}"
        )

        logger.info(
            "receipt_discrepancy_resolved",
            agent_id=str(self.config.agent_id),
            po_id=po_id,
            resolution=resolution,
            discrepancy_value=discrepancy_value,
            quantity_variance_pct=round(quantity_variance_pct, 4),
            reasoning=reasoning,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            has_exceptions=has_exceptions,
            data={
                "resolution": resolution,
                "po_id": po_id,
                "discrepancy_type": discrepancy_type,
                "discrepancy_value": discrepancy_value,
                "expected_quantity": expected_qty,
                "received_quantity": received_qty,
                "quantity_variance_pct": round(quantity_variance_pct, 4),
                "reasoning": reasoning,
                "decision_details": decision,
            },
            error=None,
        )

    async def _handle_shipment_exception(self, work_item: WorkItem) -> WorkResult:
        """Handle shipment exceptions such as delays, damages, or wrong items.

        Manages shipment issues reported during the outbound shipping process,
        including carrier delays, in-transit damage, incorrect items shipped,
        quantity shortages, and address issues. The manager evaluates the
        exception and determines the appropriate resolution.

        Resolution options:
            - reship: Initiate reshipment of correct/replacement items
            - credit: Issue credit to customer for the affected shipment
            - partial_reship: Reship only the missing or damaged items
            - carrier_claim: File a claim with the carrier for damages
            - escalate: Escalate to Controller for significant financial impact

        Args:
            work_item: Work item containing shipment exception details including
                order_id, shipment_id, exception_type, affected items, and
                any carrier or customer communication.

        Returns:
            WorkResult with exception resolution actions and outcome data.
        """
        exception_data: Dict[str, Any] = work_item.data
        order_id: str = exception_data.get("order_id", "unknown")
        shipment_id: str = exception_data.get("shipment_id", "unknown")
        exception_type: str = exception_data.get(
            "exception_type", "general_exception"
        )
        exception_value: float = work_item.amount or exception_data.get("value", 0.0)

        logger.info(
            "shipment_exception_review_started",
            agent_id=str(self.config.agent_id),
            order_id=order_id,
            shipment_id=shipment_id,
            exception_type=exception_type,
            exception_value=exception_value,
        )

        # Use DecisionEngine to determine resolution
        context: Dict[str, Any] = {
            "shipment_exception": exception_data,
            "order_id": order_id,
            "shipment_id": shipment_id,
            "exception_type": exception_type,
            "exception_value": exception_value,
            "agent_role": self.ROLE,
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="handle_exception",
            context=context,
        )

        # Parse resolution from decision
        resolution: str = decision.get("resolution", "reship")
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))
        actions_taken: List[str] = ["shipment_exception_reviewed"]
        approval_needed: bool = False
        has_exceptions: bool = True

        if resolution == "reship":
            actions_taken.append("reshipment_initiated")
        elif resolution == "credit":
            actions_taken.append("customer_credit_issued")
        elif resolution == "partial_reship":
            actions_taken.append("partial_reshipment_initiated")
        elif resolution == "carrier_claim":
            actions_taken.append("carrier_claim_filed")
        elif resolution == "escalate":
            actions_taken.append("escalated_to_controller")
            approval_needed = True
        else:
            # Default to reshipment for unrecognized resolutions
            actions_taken.append("reshipment_initiated")

        # If the exception value is high, also flag for controller awareness
        if exception_value > self._ESCALATION_VALUE_THRESHOLD and resolution != "escalate":
            actions_taken.append("controller_notified")

        # Record observation in agent memory
        await self.memory.add_observation(
            f"Shipment exception for order {order_id} (shipment {shipment_id}): "
            f"type={exception_type}, resolution={resolution}, "
            f"value=${exception_value:,.2f}"
        )

        logger.info(
            "shipment_exception_handled",
            agent_id=str(self.config.agent_id),
            order_id=order_id,
            shipment_id=shipment_id,
            exception_type=exception_type,
            resolution=resolution,
            exception_value=exception_value,
            reasoning=reasoning,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            has_exceptions=has_exceptions,
            data={
                "resolution": resolution,
                "order_id": order_id,
                "shipment_id": shipment_id,
                "exception_type": exception_type,
                "exception_value": exception_value,
                "reasoning": reasoning,
                "decision_details": decision,
            },
            error=None,
        )

    async def _process_inventory_review(self, work_item: WorkItem) -> WorkResult:
        """Process inventory review requests including count adjustments and audits.

        Reviews inventory adjustments, cycle count variances, and periodic
        physical inventory results. The manager evaluates whether adjustments
        are within acceptable thresholds and approves or rejects them. Large
        adjustments or suspicious patterns are escalated to Controller.

        Review scenarios:
            - Cycle count variance approvals
            - Physical inventory adjustment reviews
            - Warehouse transfer reconciliation
            - Shrinkage and loss investigation results

        Args:
            work_item: Work item containing inventory review details including
                warehouse_id, item details, current vs counted quantities,
                adjustment values, and any investigation notes.

        Returns:
            WorkResult with inventory review decision and adjustment actions.
        """
        review_data: Dict[str, Any] = work_item.data
        warehouse_id: str = review_data.get("warehouse_id", "unknown")
        review_type: str = review_data.get("review_type", "cycle_count")
        adjustment_value: float = work_item.amount or review_data.get(
            "adjustment_value", 0.0
        )
        items_reviewed: int = review_data.get("items_reviewed", 0)
        items_with_variance: int = review_data.get("items_with_variance", 0)

        logger.info(
            "inventory_review_started",
            agent_id=str(self.config.agent_id),
            warehouse_id=warehouse_id,
            review_type=review_type,
            adjustment_value=adjustment_value,
            items_reviewed=items_reviewed,
            items_with_variance=items_with_variance,
        )

        # Use DecisionEngine for review decision
        context: Dict[str, Any] = {
            "inventory_review": review_data,
            "warehouse_id": warehouse_id,
            "review_type": review_type,
            "adjustment_value": adjustment_value,
            "items_reviewed": items_reviewed,
            "items_with_variance": items_with_variance,
            "agent_role": self.ROLE,
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="approve_transaction",
            context=context,
        )

        # Parse decision outcome
        approved: bool = decision.get("approved", decision.get("decision", "") == "approve")
        reasoning: str = decision.get("reasoning", decision.get("notes", ""))
        actions_taken: List[str] = ["inventory_review_completed"]
        has_exceptions: bool = False
        approval_needed: bool = False

        if approved:
            actions_taken.append("inventory_adjustments_approved")
            # Check if adjustment value warrants controller notification
            if abs(adjustment_value) > self._ESCALATION_VALUE_THRESHOLD:
                actions_taken.append("controller_notified")
                approval_needed = True
        else:
            actions_taken.append("inventory_adjustments_rejected")
            actions_taken.append("recount_requested")
            has_exceptions = True

        # Flag variance patterns that may indicate systemic issues
        variance_rate: float = 0.0
        if items_reviewed > 0:
            variance_rate = items_with_variance / items_reviewed

        if variance_rate > 0.20:
            # More than 20% of items have variances — potential systemic issue
            actions_taken.append("variance_investigation_initiated")
            has_exceptions = True

        # Record observation in agent memory
        await self.memory.add_observation(
            f"Inventory review at {warehouse_id}: type={review_type}, "
            f"approved={approved}, adjustment=${adjustment_value:,.2f}, "
            f"variance_rate={variance_rate:.1%}"
        )

        logger.info(
            "inventory_review_completed",
            agent_id=str(self.config.agent_id),
            warehouse_id=warehouse_id,
            review_type=review_type,
            approved=approved,
            adjustment_value=adjustment_value,
            variance_rate=round(variance_rate, 4),
            reasoning=reasoning,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            has_exceptions=has_exceptions,
            data={
                "approved": approved,
                "warehouse_id": warehouse_id,
                "review_type": review_type,
                "adjustment_value": adjustment_value,
                "items_reviewed": items_reviewed,
                "items_with_variance": items_with_variance,
                "variance_rate": round(variance_rate, 4),
                "reasoning": reasoning,
                "decision_details": decision,
            },
            error=None,
        )
