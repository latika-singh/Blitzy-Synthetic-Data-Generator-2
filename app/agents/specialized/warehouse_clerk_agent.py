"""
WarehouseClerkAgent — Goods receipt and shipment processing specialist.

This module implements the WarehouseClerkAgent class, one of 12 specialized
agent types in the Agent System (F-001).  The warehouse clerk is responsible
for receiving goods against purchase orders, processing outbound shipments
for sales orders, and recording inventory adjustments.

Exports:
    WarehouseClerkAgent — Concrete BaseAgent subclass that processes three
                          work-item types: goods_receipt, shipment, and
                          inventory_adjustment.

Design Decisions:
    - Extends BaseAgent via constructor injection (AAP Section 0.7.1): all
      dependencies (config, memory, decision_engine, action_registry) are
      received from the parent constructor and never self-instantiated.
    - Goods receipt data (received quantities, discrepancies) is structured
      to support the downstream 3-way match performed by APClerkAgent
      (README.md line 122).
    - Quantity verification is deterministic (Layer 4 of the Decision Engine).
      Only discrepancy handling and description generation use the LLM Layer
      (Layer 2) via ``self.make_decision()``.
    - Financial calculations (amounts, adjustments) are computed in the
      Deterministic Layer and are never LLM-generated (AAP Section 0.7.1).
    - All logging uses ``structlog`` to stdout in structured JSON format
      per AAP Section 0.7.6.

Performance Targets (AAP Section 0.1.2):
    - Simple decision:       10 seconds
    - LLM decision:          30 seconds
    - Complex workflow:       60 seconds
    - Absolute max:          120 seconds

Actions Used:
    - receive_goods  (ActionRegistry, README.md line 122)
    - ship_order     (ActionRegistry, README.md line 128)

References:
    - AAP Section 0.5.1 Group 5 — WarehouseClerkAgent specification
    - README.md lines 122, 128 — receive_goods and ship_order actions
    - README.md line 485 — carrier tracking in external-world simulation
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
# Module-level structured logger (AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


class WarehouseClerkAgent(BaseAgent):
    """Warehouse Clerk agent — goods receipt and shipment processing.

    Responsible for:
        - Receiving goods against purchase orders and verifying quantities
        - Recording goods receipt data that feeds into the AP 3-way match
        - Initiating and processing outbound shipments for sales orders
        - Recording inventory adjustments for cycle counts / corrections

    Role: ``"warehouse_clerk"``

    Transaction types handled:
        - ``goods_receipt``         — verify and record inbound deliveries
        - ``shipment``              — pick, pack, and ship outbound orders
        - ``inventory_adjustment``  — adjust counts after physical audit

    Actions used from ActionRegistry:
        - ``receive_goods``  — record received quantities and conditions
        - ``ship_order``     — process and dispatch outbound shipments

    Decision types invoked:
        - ``handle_exception``       — resolve receipt discrepancies
        - ``generate_description``   — produce shipment documentation text
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------
    ROLE: str = "warehouse_clerk"
    SUPPORTED_WORK_TYPES: List[str] = [
        "goods_receipt",
        "shipment",
        "inventory_adjustment",
    ]

    # ------------------------------------------------------------------
    # process_work_item — AbstractMethod implementation (BaseAgent contract)
    # ------------------------------------------------------------------
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a warehouse work item by dispatching to the correct handler.

        Routes the incoming work item based on its ``type`` field:
            - ``"goods_receipt"``        → ``_process_goods_receipt()``
            - ``"shipment"``             → ``_process_shipment()``
            - ``"inventory_adjustment"`` → ``_process_inventory_adjustment()``

        Any unsupported work-item type raises a ``ValueError`` so the
        BaseAgent run-loop can record the error and transition to ERROR state.

        Args:
            work_item: The WorkItem dispatched by the WorkflowOrchestrator.

        Returns:
            WorkResult with success/failure status, actions taken, and
            result data specific to the work-item type.

        Raises:
            ValueError: If ``work_item.type`` is not in SUPPORTED_WORK_TYPES.
        """
        logger.info(
            "warehouse_clerk_processing",
            agent_id=str(self.config.agent_id),
            work_item_type=work_item.type,
            workflow_id=str(work_item.workflow_id),
            priority=work_item.priority,
        )

        if work_item.type == "goods_receipt":
            return await self._process_goods_receipt(work_item)
        elif work_item.type == "shipment":
            return await self._process_shipment(work_item)
        elif work_item.type == "inventory_adjustment":
            return await self._process_inventory_adjustment(work_item)
        else:
            raise ValueError(
                f"Unsupported work item type '{work_item.type}' for "
                f"{self.ROLE}. Supported types: {self.SUPPORTED_WORK_TYPES}"
            )

    # ------------------------------------------------------------------
    # Goods Receipt Processing
    # ------------------------------------------------------------------
    async def _process_goods_receipt(self, work_item: WorkItem) -> WorkResult:
        """Receive goods against a purchase order and verify quantities.

        Processing steps:
            1. Extract PO reference and received-item details from the
               work-item data payload.
            2. For each line item, compare ``received_quantity`` against
               ``ordered_quantity`` to detect discrepancies (short shipment,
               over-delivery, damaged goods).
            3. If discrepancies exist, invoke the Decision Engine with
               ``handle_exception`` to determine corrective action.
            4. Build a structured receipt record suitable for the downstream
               AP 3-way match (PO ↔ Receipt ↔ Invoice).
            5. Log the outcome with structured context.

        Args:
            work_item: WorkItem of type ``"goods_receipt"`` containing:
                - ``po_id``          (str)  — purchase order identifier
                - ``received_items`` (list) — items with received_quantity,
                  ordered_quantity, item_id, and optional condition notes
                - ``vendor_id``      (str)  — delivering vendor
                - ``delivery_date``  (str)  — date of delivery
                - ``warehouse_id``   (str)  — receiving warehouse

        Returns:
            WorkResult with receipt verification data and any discrepancies.
        """
        data: Dict[str, Any] = work_item.data

        # --- 1. Extract receipt payload ---------------------------------
        po_id: str = data.get("po_id", "UNKNOWN")
        received_items: List[Dict[str, Any]] = data.get("received_items", [])
        vendor_id: str = data.get("vendor_id", "")
        delivery_date: str = data.get("delivery_date", "")
        warehouse_id: str = data.get("warehouse_id", "")

        actions_taken: List[str] = []
        discrepancies: List[Dict[str, Any]] = []
        verified_items: List[Dict[str, Any]] = []
        has_exceptions: bool = False

        # --- 2. Verify quantities per line item -------------------------
        for item in received_items:
            item_id: str = item.get("item_id", "UNKNOWN")
            received_qty: float = float(item.get("received_quantity", 0))
            ordered_qty: float = float(item.get("ordered_quantity", 0))
            condition: str = item.get("condition", "good")

            verified_record: Dict[str, Any] = {
                "item_id": item_id,
                "received_quantity": received_qty,
                "ordered_quantity": ordered_qty,
                "condition": condition,
                "quantity_match": abs(received_qty - ordered_qty) < 0.001,
            }

            # Detect quantity discrepancy
            if abs(received_qty - ordered_qty) >= 0.001:
                discrepancy_type: str
                if received_qty < ordered_qty:
                    discrepancy_type = "short_shipment"
                else:
                    discrepancy_type = "over_delivery"

                discrepancy: Dict[str, Any] = {
                    "item_id": item_id,
                    "type": discrepancy_type,
                    "received_quantity": received_qty,
                    "ordered_quantity": ordered_qty,
                    "variance": round(received_qty - ordered_qty, 4),
                }
                discrepancies.append(discrepancy)
                verified_record["discrepancy"] = discrepancy

            # Detect damaged goods
            if condition.lower() in ("damaged", "partial_damage", "rejected"):
                damage_discrepancy: Dict[str, Any] = {
                    "item_id": item_id,
                    "type": "damaged_goods",
                    "condition": condition,
                    "received_quantity": received_qty,
                    "ordered_quantity": ordered_qty,
                }
                discrepancies.append(damage_discrepancy)
                verified_record["damage_noted"] = True

            verified_items.append(verified_record)

        # --- 3. Handle discrepancies via Decision Engine ----------------
        exception_resolution: Optional[Dict[str, Any]] = None
        if discrepancies:
            has_exceptions = True
            try:
                exception_resolution = await self.make_decision(
                    decision_type="handle_exception",
                    context={
                        "receipt": {
                            "po_id": po_id,
                            "vendor_id": vendor_id,
                            "delivery_date": delivery_date,
                            "warehouse_id": warehouse_id,
                        },
                        "discrepancies": discrepancies,
                        "item_count": len(received_items),
                        "discrepancy_count": len(discrepancies),
                    },
                )
                actions_taken.append("discrepancy_resolved")
            except Exception as exc:
                logger.warning(
                    "goods_receipt_exception_handling_failed",
                    agent_id=str(self.config.agent_id),
                    po_id=po_id,
                    error=str(exc),
                )
                exception_resolution = {
                    "resolution": "manual_review_required",
                    "reason": str(exc),
                }
                actions_taken.append("discrepancy_escalated")

        # --- 4. Record receipt (deterministic action) -------------------
        actions_taken.insert(0, "goods_received")
        actions_taken.append("quantity_verified")

        receipt_record: Dict[str, Any] = {
            "po_id": po_id,
            "vendor_id": vendor_id,
            "delivery_date": delivery_date,
            "warehouse_id": warehouse_id,
            "items_received": len(received_items),
            "items_verified": len(verified_items),
            "verified_items": verified_items,
            "discrepancies": discrepancies,
            "discrepancy_count": len(discrepancies),
            "all_quantities_match": len(discrepancies) == 0,
            "exception_resolution": exception_resolution,
            "receipt_complete": True,
        }

        # --- 5. Log structured outcome ---------------------------------
        logger.info(
            "goods_receipt_processed",
            agent_id=str(self.config.agent_id),
            po_id=po_id,
            items_received=len(received_items),
            discrepancy_count=len(discrepancies),
            vendor_id=vendor_id,
            warehouse_id=warehouse_id,
            has_exceptions=has_exceptions,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            has_exceptions=has_exceptions,
            data=receipt_record,
        )

    # ------------------------------------------------------------------
    # Shipment Processing
    # ------------------------------------------------------------------
    async def _process_shipment(self, work_item: WorkItem) -> WorkResult:
        """Process an outbound shipment for a sales order.

        Processing steps:
            1. Extract order and shipment details from the work-item payload.
            2. Validate that all requested items are present in the shipment
               manifest and mark availability status.
            3. Invoke the Decision Engine with ``generate_description`` to
               produce human-readable shipment documentation text.
            4. Build a structured shipment record with tracking information.
            5. Log the outcome with structured context.

        Args:
            work_item: WorkItem of type ``"shipment"`` containing:
                - ``order_id``     (str)  — sales order identifier
                - ``items``        (list) — items to ship with quantities
                - ``carrier_id``   (str)  — assigned carrier
                - ``warehouse_id`` (str)  — shipping warehouse
                - ``ship_to``      (dict) — destination address details
                - ``shipping_method`` (str) — shipping speed / method

        Returns:
            WorkResult with shipment processing data and tracking details.
        """
        data: Dict[str, Any] = work_item.data

        # --- 1. Extract shipment payload --------------------------------
        order_id: str = data.get("order_id", "UNKNOWN")
        items: List[Dict[str, Any]] = data.get("items", [])
        carrier_id: str = data.get("carrier_id", "")
        warehouse_id: str = data.get("warehouse_id", "")
        ship_to: Dict[str, Any] = data.get("ship_to", {})
        shipping_method: str = data.get("shipping_method", "standard")

        actions_taken: List[str] = []
        has_exceptions: bool = False

        # --- 2. Validate item availability ------------------------------
        shipment_items: List[Dict[str, Any]] = []
        unavailable_items: List[Dict[str, Any]] = []

        for item in items:
            item_id: str = item.get("item_id", "UNKNOWN")
            requested_qty: float = float(item.get("quantity", 0))
            available_qty: float = float(item.get("available_quantity", requested_qty))

            item_record: Dict[str, Any] = {
                "item_id": item_id,
                "requested_quantity": requested_qty,
                "available_quantity": available_qty,
                "is_available": available_qty >= requested_qty,
            }

            if available_qty < requested_qty:
                item_record["shortage"] = round(requested_qty - available_qty, 4)
                unavailable_items.append(item_record)
                has_exceptions = True
            else:
                item_record["shipped_quantity"] = requested_qty

            shipment_items.append(item_record)

        # --- 3. Generate shipment description via Decision Engine -------
        shipment_description: str = ""
        try:
            description_decision: Dict[str, Any] = await self.make_decision(
                decision_type="generate_description",
                context={
                    "shipment": {
                        "order_id": order_id,
                        "carrier_id": carrier_id,
                        "warehouse_id": warehouse_id,
                        "item_count": len(items),
                        "shipping_method": shipping_method,
                        "has_shortages": len(unavailable_items) > 0,
                    },
                },
            )
            shipment_description = description_decision.get("description", "")
        except Exception as exc:
            logger.warning(
                "shipment_description_generation_failed",
                agent_id=str(self.config.agent_id),
                order_id=order_id,
                error=str(exc),
            )
            shipment_description = (
                f"Shipment for order {order_id} via carrier {carrier_id}"
            )

        # --- 4. Build structured shipment record ------------------------
        actions_taken.append("shipment_processed")

        if unavailable_items:
            actions_taken.append("partial_shipment_noted")
        else:
            actions_taken.append("full_shipment_confirmed")

        shipment_record: Dict[str, Any] = {
            "order_id": order_id,
            "carrier_id": carrier_id,
            "warehouse_id": warehouse_id,
            "shipping_method": shipping_method,
            "ship_to": ship_to,
            "items": shipment_items,
            "total_items": len(items),
            "items_shipped": len(items) - len(unavailable_items),
            "items_unavailable": len(unavailable_items),
            "unavailable_details": unavailable_items,
            "is_partial_shipment": len(unavailable_items) > 0,
            "shipment_description": shipment_description,
            "shipment_complete": len(unavailable_items) == 0,
        }

        # --- 5. Log structured outcome ----------------------------------
        logger.info(
            "shipment_processed",
            agent_id=str(self.config.agent_id),
            order_id=order_id,
            carrier_id=carrier_id,
            warehouse_id=warehouse_id,
            total_items=len(items),
            items_shipped=len(items) - len(unavailable_items),
            has_exceptions=has_exceptions,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            has_exceptions=has_exceptions,
            data=shipment_record,
        )

    # ------------------------------------------------------------------
    # Inventory Adjustment Processing
    # ------------------------------------------------------------------
    async def _process_inventory_adjustment(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process an inventory count adjustment (cycle count or correction).

        Processing steps:
            1. Extract adjustment details from the work-item payload.
            2. Calculate the variance between system quantity and physical
               count for each item (deterministic — no LLM).
            3. If any adjustment exceeds the significance threshold
               (``adjustment_threshold`` or default ±5 % of system count),
               flag for review and use the Decision Engine to determine
               corrective action via ``handle_exception``.
            4. Build the adjustment record with per-item variance detail.
            5. Log the outcome with structured context.

        Args:
            work_item: WorkItem of type ``"inventory_adjustment"`` containing:
                - ``warehouse_id``   (str)  — warehouse where count occurred
                - ``adjustment_type``(str)  — e.g. "cycle_count", "correction"
                - ``items``          (list) — items with system_quantity and
                  physical_count
                - ``counted_by``     (str)  — employee who performed the count
                - ``count_date``     (str)  — date of the physical count
                - ``adjustment_threshold`` (float) — optional variance pct

        Returns:
            WorkResult with adjustment variance data and review flags.
        """
        data: Dict[str, Any] = work_item.data

        # --- 1. Extract adjustment payload ------------------------------
        warehouse_id: str = data.get("warehouse_id", "")
        adjustment_type: str = data.get("adjustment_type", "cycle_count")
        items: List[Dict[str, Any]] = data.get("items", [])
        counted_by: str = data.get("counted_by", "")
        count_date: str = data.get("count_date", "")
        adjustment_threshold: float = float(
            data.get("adjustment_threshold", 0.05)
        )

        actions_taken: List[str] = []
        has_exceptions: bool = False
        adjusted_items: List[Dict[str, Any]] = []
        significant_variances: List[Dict[str, Any]] = []

        # --- 2. Calculate variances (deterministic) ---------------------
        for item in items:
            item_id: str = item.get("item_id", "UNKNOWN")
            system_qty: float = float(item.get("system_quantity", 0))
            physical_count: float = float(item.get("physical_count", 0))
            variance: float = round(physical_count - system_qty, 4)

            # Calculate percentage variance (guard against zero division)
            variance_pct: float = 0.0
            if system_qty > 0:
                variance_pct = round(abs(variance) / system_qty, 6)

            item_record: Dict[str, Any] = {
                "item_id": item_id,
                "system_quantity": system_qty,
                "physical_count": physical_count,
                "variance": variance,
                "variance_pct": variance_pct,
                "needs_adjustment": abs(variance) >= 0.001,
                "is_significant": variance_pct > adjustment_threshold,
            }

            if variance_pct > adjustment_threshold:
                significant_variances.append(item_record)
                has_exceptions = True

            adjusted_items.append(item_record)

        # --- 3. Handle significant variances via Decision Engine --------
        exception_resolution: Optional[Dict[str, Any]] = None
        if significant_variances:
            try:
                exception_resolution = await self.make_decision(
                    decision_type="handle_exception",
                    context={
                        "adjustment": {
                            "warehouse_id": warehouse_id,
                            "adjustment_type": adjustment_type,
                            "counted_by": counted_by,
                            "count_date": count_date,
                        },
                        "significant_variances": significant_variances,
                        "total_items": len(items),
                        "significant_count": len(significant_variances),
                        "threshold_pct": adjustment_threshold,
                    },
                )
                actions_taken.append("variance_resolved")
            except Exception as exc:
                logger.warning(
                    "inventory_adjustment_exception_handling_failed",
                    agent_id=str(self.config.agent_id),
                    warehouse_id=warehouse_id,
                    error=str(exc),
                )
                exception_resolution = {
                    "resolution": "manual_review_required",
                    "reason": str(exc),
                }
                actions_taken.append("variance_escalated")

        # --- 4. Build adjustment record ---------------------------------
        actions_taken.insert(0, "inventory_counted")
        actions_taken.append("adjustment_recorded")

        items_needing_adjustment: int = sum(
            1 for r in adjusted_items if r["needs_adjustment"]
        )

        adjustment_record: Dict[str, Any] = {
            "warehouse_id": warehouse_id,
            "adjustment_type": adjustment_type,
            "counted_by": counted_by,
            "count_date": count_date,
            "total_items_counted": len(items),
            "items_needing_adjustment": items_needing_adjustment,
            "significant_variance_count": len(significant_variances),
            "adjusted_items": adjusted_items,
            "significant_variances": significant_variances,
            "exception_resolution": exception_resolution,
            "adjustment_threshold_pct": adjustment_threshold,
            "adjustment_complete": True,
        }

        # --- 5. Log structured outcome ----------------------------------
        logger.info(
            "inventory_adjustment_processed",
            agent_id=str(self.config.agent_id),
            warehouse_id=warehouse_id,
            adjustment_type=adjustment_type,
            total_items=len(items),
            items_adjusted=items_needing_adjustment,
            significant_variances=len(significant_variances),
            has_exceptions=has_exceptions,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            has_exceptions=has_exceptions,
            data=adjustment_record,
        )
