"""Accounts Receivable Clerk agent implementation.

This module provides the ARClerkAgent class, which specialises BaseAgent
for the three core AR-clerk transaction types defined in the specification:

* **sales_order** — create a new sales order from a customer interaction,
  generating an LLM-powered description via the ``generate_description``
  decision type and optionally initiating shipment.
* **customer_payment** — apply an incoming customer payment against an
  outstanding invoice, using the ``match_documents`` decision type for
  automated matching, and detecting short-pay / over-payment conditions
  that require AR-Manager escalation.
* **customer_invoice** — generate a customer invoice from an existing
  sales order, including an LLM-generated description.

Role mapping (README.md lines 1351-1352)::

    "sales_order":       ["ar_clerk"]
    "customer_payment":  ["ar_clerk"]

AR actions registered in ActionRegistry::

    create_sales_order, ship_order, create_customer_invoice, apply_payment

All logging follows AAP Section 0.7.6 (structured JSON via ``structlog``).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import structlog

from app.agents.action_registry import ActionRegistry
from app.agents.agent_config import AgentConfig
from app.agents.agent_memory import AgentMemory
from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.decision_engine import DecisionEngine

# ---------------------------------------------------------------------------
# Module-level logger (AAP Section 0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


class ARClerkAgent(BaseAgent):
    """Accounts Receivable Clerk agent.

    Responsible for:
    - Processing sales orders from customers
    - Creating and sending customer invoices
    - Applying customer payments to invoices
    - Initiating shipment processing for fulfilled orders

    Role: ``"ar_clerk"``

    Transaction types handled:
        ``"sales_order"``, ``"customer_payment"``, ``"customer_invoice"``

    Actions used:
        ``create_sales_order``, ``create_customer_invoice``,
        ``apply_payment``, ``ship_order``

    Customer order sizing tiers (README.md lines 526-528):
        Small $500-$5 K, Medium $2 K-$50 K, Large $10 K-$500 K

    Notes:
        * Short-payment detection flags the work-item for AR-Manager review.
        * Over-payments trigger a credit-memo action.
        * LLM decisions are delegated through ``self.make_decision()``
          which invokes the 4-layer DecisionEngine pipeline
          (Statistical → LLM → Validation → Deterministic).
    """

    # ------------------------------------------------------------------ #
    # Class-level constants
    # ------------------------------------------------------------------ #

    ROLE: str = "ar_clerk"
    """Agent role identifier matching ROLE_MAPPING in WorkflowOrchestrator."""

    SUPPORTED_WORK_TYPES: List[str] = [
        "sales_order",
        "customer_payment",
        "customer_invoice",
    ]
    """Work-item types this agent is capable of processing."""

    # Tolerance for payment matching — a 2 % margin handles minor
    # rounding differences between invoiced amounts and remittance advice.
    _PAYMENT_MATCH_TOLERANCE: float = 0.02

    # ------------------------------------------------------------------ #
    # process_work_item — required by BaseAgent ABC
    # ------------------------------------------------------------------ #

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Route an incoming work item to the appropriate handler.

        Parameters
        ----------
        work_item:
            A ``WorkItem`` instance whose ``.type`` indicates the
            transaction category.

        Returns
        -------
        WorkResult
            The outcome of processing the work item.

        Raises
        ------
        ValueError
            If ``work_item.type`` is not in ``SUPPORTED_WORK_TYPES``.
        """
        logger.info(
            "ar_clerk_processing",
            agent_id=str(self.config.agent_id),
            agent_role=self.config.role,
            work_item_type=work_item.type,
        )

        if work_item.type == "sales_order":
            return await self._process_sales_order(work_item)
        elif work_item.type == "customer_payment":
            return await self._process_customer_payment(work_item)
        elif work_item.type == "customer_invoice":
            return await self._create_customer_invoice(work_item)
        else:
            raise ValueError(
                f"ARClerkAgent does not support work item type: "
                f"{work_item.type!r}. Supported types: "
                f"{self.SUPPORTED_WORK_TYPES}"
            )

    # ------------------------------------------------------------------ #
    # Sales Order Processing
    # ------------------------------------------------------------------ #

    async def _process_sales_order(self, work_item: WorkItem) -> WorkResult:
        """Process a sales order from a customer.

        Steps:
        1. Extract and validate order data (customer_id, items, amount).
        2. Request an LLM-generated description via ``generate_description``.
        3. Build the list of actions taken.
        4. If the order requires shipment, mark ``shipment_initiated``.

        Parameters
        ----------
        work_item:
            WorkItem containing order data in ``work_item.data``.

        Returns
        -------
        WorkResult
            Success result with order metadata and generated description.
        """
        order_data: Dict[str, Any] = work_item.data
        customer_id: str = str(order_data.get("customer_id", "unknown"))
        items: List[Any] = order_data.get("items", [])
        total_amount: float = float(order_data.get("total_amount", 0.0))

        # ----- validation ------------------------------------------------ #
        if not customer_id or customer_id == "unknown":
            logger.warning(
                "sales_order_missing_customer",
                agent_id=str(self.config.agent_id),
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Missing required field: customer_id"},
                error="Missing customer_id in sales order data",
            )

        if not items:
            logger.warning(
                "sales_order_no_items",
                agent_id=str(self.config.agent_id),
                customer_id=customer_id,
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Sales order has no line items"},
                error="Empty items list in sales order",
            )

        # ----- LLM decision for order description ----------------------- #
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="generate_description",
            context={
                "order": order_data,
                "customer": order_data.get("customer", {}),
                "items": items,
                "total_amount": total_amount,
            },
        )

        description: str = decision.get("description", "")

        # ----- build action list ----------------------------------------- #
        actions_taken: List[str] = ["sales_order_created"]

        # Determine if the order should trigger a shipment request.
        requires_shipment: bool = order_data.get("requires_shipment", True)
        if requires_shipment and items:
            actions_taken.append("shipment_initiated")

        logger.info(
            "sales_order_processed",
            agent_id=str(self.config.agent_id),
            customer_id=customer_id,
            amount=total_amount,
            item_count=len(items),
            shipment_initiated=requires_shipment,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            data={
                "order_data": order_data,
                "description": description,
                "customer_id": customer_id,
                "total_amount": total_amount,
                "item_count": len(items),
            },
        )

    # ------------------------------------------------------------------ #
    # Customer Payment Processing
    # ------------------------------------------------------------------ #

    async def _process_customer_payment(self, work_item: WorkItem) -> WorkResult:
        """Apply a customer payment against an outstanding invoice.

        The method uses the ``match_documents`` decision type to let the
        4-layer DecisionEngine determine how the payment maps to the
        referenced invoice.  Based on the comparison of the payment
        amount to the invoice amount the result is classified as:

        * **full_match** — payment equals invoice (within tolerance):
          applied immediately.
        * **short_pay** — payment < invoice: flagged as exception,
          escalated to AR Manager for review.
        * **overpayment** — payment > invoice: applied and a credit-memo
          action is recorded.

        Parameters
        ----------
        work_item:
            WorkItem containing payment data including ``payment_amount``,
            ``invoice_id``, ``invoice_amount``, and optional ``invoice``
            sub-dict.

        Returns
        -------
        WorkResult
            Outcome including ``match_type`` and whether approval is needed.
        """
        payment_data: Dict[str, Any] = work_item.data
        payment_amount: float = float(payment_data.get("payment_amount", 0.0))
        invoice_id: str = str(payment_data.get("invoice_id", ""))
        invoice_amount: float = float(payment_data.get("invoice_amount", 0.0))

        # ----- basic validation ------------------------------------------ #
        if not invoice_id:
            logger.warning(
                "payment_missing_invoice_id",
                agent_id=str(self.config.agent_id),
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Missing required field: invoice_id"},
                error="Missing invoice_id in payment data",
            )

        if payment_amount <= 0:
            logger.warning(
                "payment_invalid_amount",
                agent_id=str(self.config.agent_id),
                payment_amount=payment_amount,
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Payment amount must be positive"},
                error="Invalid payment amount",
            )

        # ----- LLM matching decision ------------------------------------ #
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="match_documents",
            context={
                "payment": payment_data,
                "invoice": payment_data.get("invoice", {}),
                "payment_amount": payment_amount,
                "invoice_amount": invoice_amount,
                "invoice_id": invoice_id,
            },
        )

        # ----- classify the payment match -------------------------------- #
        amount_difference: float = payment_amount - invoice_amount
        tolerance_amount: float = invoice_amount * self._PAYMENT_MATCH_TOLERANCE

        actions_taken: List[str] = []
        approval_needed: bool = False
        has_exceptions: bool = False
        match_type: str

        if invoice_amount <= 0:
            # No invoice amount to compare — treat as a generic application.
            match_type = "no_invoice_amount"
            actions_taken = ["payment_applied"]
        elif abs(amount_difference) <= tolerance_amount:
            # Full match — amount within tolerance.
            match_type = "full_match"
            actions_taken = ["payment_applied"]
        elif amount_difference < 0:
            # Short payment — need AR Manager review.
            match_type = "short_pay"
            actions_taken = ["payment_applied_partial"]
            has_exceptions = True
            approval_needed = True
        else:
            # Overpayment — apply and flag credit memo.
            match_type = "overpayment"
            actions_taken = ["payment_applied", "credit_memo_needed"]

        logger.info(
            "payment_processed",
            agent_id=str(self.config.agent_id),
            invoice_id=invoice_id,
            match_type=match_type,
            payment_amount=payment_amount,
            invoice_amount=invoice_amount,
            amount_difference=round(amount_difference, 2),
            approval_needed=approval_needed,
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            has_exceptions=has_exceptions,
            data={
                "match_type": match_type,
                "payment_amount": payment_amount,
                "invoice_amount": invoice_amount,
                "invoice_id": invoice_id,
                "amount_difference": round(amount_difference, 2),
                "match_decision": decision,
            },
        )

    # ------------------------------------------------------------------ #
    # Customer Invoice Creation
    # ------------------------------------------------------------------ #

    async def _create_customer_invoice(self, work_item: WorkItem) -> WorkResult:
        """Generate a customer invoice from order / shipment data.

        An LLM-generated description is produced via the
        ``generate_description`` decision type, and the invoice data is
        assembled from the incoming work-item payload.

        Parameters
        ----------
        work_item:
            WorkItem containing invoice context data such as ``order_id``,
            ``customer_id``, ``amount``, and ``line_items``.

        Returns
        -------
        WorkResult
            Success result with assembled invoice data and description.
        """
        invoice_context: Dict[str, Any] = work_item.data
        order_id: str = str(invoice_context.get("order_id", ""))
        customer_id: str = str(invoice_context.get("customer_id", "unknown"))
        amount: float = float(invoice_context.get("amount", 0.0))
        line_items: List[Any] = invoice_context.get("line_items", [])
        payment_terms: str = str(invoice_context.get("payment_terms", "Net 30"))

        # ----- validation ------------------------------------------------ #
        if not customer_id or customer_id == "unknown":
            logger.warning(
                "invoice_missing_customer",
                agent_id=str(self.config.agent_id),
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Missing required field: customer_id"},
                error="Missing customer_id for invoice creation",
            )

        if amount <= 0:
            logger.warning(
                "invoice_invalid_amount",
                agent_id=str(self.config.agent_id),
                customer_id=customer_id,
                amount=amount,
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                data={"error_detail": "Invoice amount must be positive"},
                error="Invalid invoice amount",
            )

        # ----- LLM decision for invoice description ---------------------- #
        decision: Dict[str, Any] = await self.make_decision(
            decision_type="generate_description",
            context={
                "invoice_context": invoice_context,
                "customer_id": customer_id,
                "order_id": order_id,
                "amount": amount,
                "line_items": line_items,
            },
        )

        description: str = decision.get("description", "")

        # ----- assemble invoice data ------------------------------------- #
        invoice_data: Dict[str, Any] = {
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "line_items": line_items,
            "payment_terms": payment_terms,
            "description": description,
        }

        logger.info(
            "customer_invoice_created",
            agent_id=str(self.config.agent_id),
            customer_id=customer_id,
            order_id=order_id,
            amount=amount,
            line_item_count=len(line_items),
        )

        return WorkResult(
            success=True,
            actions_taken=["customer_invoice_created"],
            data={
                "invoice_data": invoice_data,
                "description": description,
            },
        )
