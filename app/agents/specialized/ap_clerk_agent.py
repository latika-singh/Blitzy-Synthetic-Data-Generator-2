"""Accounts Payable Clerk agent implementation.

This module provides the APClerkAgent class, which specialises BaseAgent
for the core AP-clerk transaction types defined in the specification
(README.md lines 892-978):

* **vendor_invoice** — process an incoming vendor invoice, invoke the 4-layer
  DecisionEngine for GL coding and processing notes, then perform a 3-way
  match (PO ↔ Goods Receipt ↔ Invoice) with a 5 % price-variance tolerance
  and an exact quantity comparison.  Variances exceeding tolerance set the
  ``approval_needed`` flag on the returned ``WorkResult`` so the
  WorkflowOrchestrator can escalate to the AP Manager.
* **invoice_exception** — handle an invoice that could not be processed
  normally (missing PO, unresolvable variance, duplicate, etc.).  An LLM
  decision of type ``handle_exception`` determines resolution, escalation,
  or rejection.

Role mapping (README.md line 1349)::

    "vendor_invoice": ["ap_clerk"]

AP actions registered in ActionRegistry::

    process_vendor_invoice, match_three_way, approve_invoice, schedule_payment

All logging follows AAP Section 0.7.6 (structured JSON via ``structlog``
to stdout).

References:
    - README.md lines 892-978 (Specialized Agent Example — AP Clerk)
    - README.md line 524 (95 % match PO ±5 % tolerance)
    - README.md lines 1554-1609 (Agent error handling pattern)
    - AAP Section 0.7.1 (MUST extend BaseAgent, 4-layer pipeline inviolable)
    - AAP Section 0.7.3 (≥50 agents/second creation, p95 < 5 s decisions)
    - AAP Section 0.7.6 (structlog for all logging with agent_id context)
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


class APClerkAgent(BaseAgent):
    """Accounts Payable Clerk agent.

    Responsible for:
    - Processing vendor invoices
    - Performing 3-way matching (PO, receipt, invoice)
    - GL account coding
    - Handling invoice exceptions

    Role: ``"ap_clerk"``

    Transaction types handled:
        ``"vendor_invoice"``, ``"invoice_exception"``

    Actions used:
        ``process_vendor_invoice``, ``match_three_way``, ``approve_invoice``

    Variance tolerance:
        5 % price tolerance for 3-way match (README.md line 524: 95 % match
        PO ±5 %).  Quantity comparison is exact (integer equality).

    Notes:
        * LLM decisions are delegated through ``self.make_decision()``
          which invokes the 4-layer DecisionEngine pipeline
          (Statistical → LLM → Validation → Deterministic).
        * GL coding and financial calculations are produced exclusively by the
          Deterministic Layer of the DecisionEngine — never by LLM output
          directly (AAP Section 0.7.1).
        * Memory observations are recorded automatically by ``BaseAgent.run()``
          after ``process_work_item()`` completes.
    """

    # ------------------------------------------------------------------ #
    # Class-level constants
    # ------------------------------------------------------------------ #

    ROLE: str = "ap_clerk"
    """Agent role identifier matching ROLE_MAPPING in WorkflowOrchestrator."""

    SUPPORTED_WORK_TYPES: List[str] = ["vendor_invoice", "invoice_exception"]
    """Work-item types this agent is capable of processing."""

    VARIANCE_TOLERANCE: float = 0.05
    """5 % tolerance for 3-way price matching.

    Derived from README.md line 524 (vendor invoice matching 95 % within
    ±5 %).  A price difference *less than* ``unit_price * VARIANCE_TOLERANCE``
    is considered a match; otherwise the variance triggers an approval
    escalation to the AP Manager.
    """

    # ------------------------------------------------------------------ #
    # process_work_item — required by BaseAgent ABC
    # ------------------------------------------------------------------ #

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Route an incoming work item to the appropriate handler.

        Dispatches based on ``work_item.type``:
            * ``"vendor_invoice"`` → :meth:`_process_vendor_invoice`
            * ``"invoice_exception"`` → :meth:`_handle_invoice_exception`

        Any unsupported type raises ``ValueError`` so the BaseAgent ``run()``
        loop catches it and records an error observation.

        Args:
            work_item: The work item to process.

        Returns:
            A :class:`WorkResult` produced by the delegated handler.

        Raises:
            ValueError: If ``work_item.type`` is not in
                :attr:`SUPPORTED_WORK_TYPES`.
        """
        logger.info(
            "ap_clerk_processing",
            agent_id=str(self.config.agent_id),
            agent_role=self.config.role,
            work_item_type=work_item.type,
            work_item_id=str(work_item.workflow_id),
        )

        if work_item.type == "vendor_invoice":
            return await self._process_vendor_invoice(work_item)
        elif work_item.type == "invoice_exception":
            return await self._handle_invoice_exception(work_item)
        else:
            raise ValueError(f"Unknown work item type: {work_item.type}")

    # ------------------------------------------------------------------ #
    # Private: vendor invoice processing (README.md lines 907-951)
    # ------------------------------------------------------------------ #

    async def _process_vendor_invoice(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Process a vendor invoice through GL coding and 3-way matching.

        Processing steps (README.md lines 907-951):
            1. Extract invoice data from the work item payload.
            2. Invoke the 4-layer DecisionEngine via ``self.make_decision()``
               with decision type ``"process_vendor_invoice"`` and context
               containing the invoice, PO, and goods-receipt data.
            3. Extract GL coding from the decision output.
            4. Perform 3-way match (PO ↔ receipt ↔ invoice).
            5. Determine whether approval is needed based on variance.
            6. Build and return a ``WorkResult``.

        Args:
            work_item: Work item with ``type == "vendor_invoice"`` and
                ``data`` containing invoice details plus optional nested
                ``"purchase_order"`` and ``"goods_receipt"`` dictionaries.

        Returns:
            A :class:`WorkResult` with:
                * ``success=True`` on normal completion
                * ``actions_taken`` including ``"gl_coding_assigned"``,
                  ``"three_way_match_performed"``, and optionally
                  ``"approval_requested"``
                * ``approval_needed`` set when 3-way match variance exceeds
                  the :attr:`VARIANCE_TOLERANCE`
                * ``data`` containing ``gl_coding``, ``match_result``,
                  and ``processing_notes``
        """
        # Step 1 — Extract invoice data
        invoice_data: Dict[str, Any] = work_item.data

        logger.info(
            "processing_vendor_invoice",
            agent_id=str(self.config.agent_id),
            invoice_id=invoice_data.get("invoice_id", "unknown"),
            vendor_id=invoice_data.get("vendor_id", "unknown"),
            amount=invoice_data.get("amount"),
        )

        try:
            # Step 2 — Make LLM-assisted decision via DecisionEngine
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="process_vendor_invoice",
                context={
                    "invoice": invoice_data,
                    "po": invoice_data.get("purchase_order"),
                    "receipt": invoice_data.get("goods_receipt"),
                },
            )

            # Step 3 — Extract GL coding from decision output
            gl_coding: Dict[str, Any] = decision.get("gl_coding", {})

            # Step 4 — Perform 3-way match
            match_result: Dict[str, Any] = await self._perform_three_way_match(
                invoice_data, decision
            )

            # Step 5 — Determine if approval is needed
            approval_needed: bool = match_result.get(
                "variance_exceeds_tolerance", False
            )

            # Step 6 — Build WorkResult
            actions_taken: List[str] = [
                "gl_coding_assigned",
                "three_way_match_performed",
            ]
            if approval_needed:
                actions_taken.append("approval_requested")

            result = WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=approval_needed,
                data={
                    "gl_coding": gl_coding,
                    "match_result": match_result,
                    "processing_notes": decision.get("processing_notes", ""),
                },
            )

            logger.info(
                "vendor_invoice_processed",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_data.get("invoice_id", "unknown"),
                match_status=match_result.get("match_status"),
                approval_needed=approval_needed,
                actions_taken=actions_taken,
            )

            return result

        except Exception as exc:
            logger.error(
                "vendor_invoice_processing_error",
                agent_id=str(self.config.agent_id),
                invoice_id=invoice_data.get("invoice_id", "unknown"),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                approval_needed=False,
                data={"invoice_id": invoice_data.get("invoice_id", "unknown")},
                error=f"Vendor invoice processing failed: {exc}",
            )

    # ------------------------------------------------------------------ #
    # Private: 3-way match (README.md lines 953-978)
    # ------------------------------------------------------------------ #

    async def _perform_three_way_match(
        self,
        invoice_data: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform three-way matching: PO ↔ Goods Receipt ↔ Invoice.

        Comparison logic (README.md lines 965-978):
            * **Quantity match** — exact integer equality between the goods
              receipt quantity and the invoice quantity.
            * **Price match** — the absolute price difference between the PO
              unit price and the invoice unit price must be less than
              ``po_unit_price * VARIANCE_TOLERANCE`` (5 %).

        If both quantity and price match the status is ``"matched"``;
        otherwise ``"variance"`` is reported and
        ``variance_exceeds_tolerance`` is set to ``True``.

        Args:
            invoice_data: Full invoice payload (may contain nested
                ``"purchase_order"`` and ``"goods_receipt"`` dicts).
            decision: Decision output from the DecisionEngine (not directly
                consumed by the match logic but kept for future enrichment).

        Returns:
            Dictionary with keys:
                ``match_status``, ``quantity_match``, ``price_match``,
                ``quantity_variance``, ``price_variance``,
                ``variance_exceeds_tolerance``
        """
        po: Dict[str, Any] = invoice_data.get("purchase_order", {})
        receipt: Dict[str, Any] = invoice_data.get("goods_receipt", {})
        invoice: Dict[str, Any] = invoice_data

        # --- Quantity comparison (exact) --------------------------------- #
        receipt_qty = receipt.get("quantity")
        invoice_qty = invoice.get("quantity")

        # Treat both-None as matched (no quantity data to compare)
        if receipt_qty is None and invoice_qty is None:
            quantity_match: bool = True
        else:
            quantity_match = receipt_qty == invoice_qty

        quantity_variance: float = abs(
            (receipt.get("quantity", 0) or 0) - (invoice.get("quantity", 0) or 0)
        )

        # --- Price comparison (within tolerance) ------------------------- #
        po_unit_price: float = po.get("unit_price", 0) or 0
        invoice_unit_price: float = invoice.get("unit_price", 0) or 0

        price_diff: float = abs(po_unit_price - invoice_unit_price)
        # Tolerance threshold: use po_unit_price; fall back to 1 to avoid
        # division-by-zero while still performing a meaningful comparison.
        tolerance_base: float = po_unit_price if po_unit_price > 0 else 1.0
        price_match: bool = price_diff < (tolerance_base * self.VARIANCE_TOLERANCE)

        price_variance: float = price_diff

        # --- Overall match status ---------------------------------------- #
        match_status: str = (
            "matched" if (quantity_match and price_match) else "variance"
        )
        variance_exceeds: bool = not (quantity_match and price_match)

        logger.info(
            "three_way_match_result",
            agent_id=str(self.config.agent_id),
            match_status=match_status,
            quantity_match=quantity_match,
            price_match=price_match,
            quantity_variance=quantity_variance,
            price_variance=round(price_variance, 4),
        )

        return {
            "match_status": match_status,
            "quantity_match": quantity_match,
            "price_match": price_match,
            "quantity_variance": quantity_variance,
            "price_variance": price_variance,
            "variance_exceeds_tolerance": variance_exceeds,
        }

    # ------------------------------------------------------------------ #
    # Private: invoice exception handling
    # ------------------------------------------------------------------ #

    async def _handle_invoice_exception(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Handle an invoice exception that could not be processed normally.

        Exception scenarios include missing PO references, unresolvable
        3-way match variances, duplicate invoice submissions, and vendor
        data mismatches.

        The method delegates to the DecisionEngine with decision type
        ``"handle_exception"`` to determine the appropriate resolution.
        The LLM layer produces a recommendation that is one of:
            * ``"resolve"`` — the exception can be cleared automatically.
            * ``"escalate"`` — the exception must be escalated to the
              AP Manager for manual review (sets ``approval_needed=True``).
            * ``"reject"`` — the invoice should be rejected outright.

        Args:
            work_item: Work item with ``type == "invoice_exception"`` and
                ``data`` containing exception details plus optional nested
                ``"invoice"`` dictionary.

        Returns:
            A :class:`WorkResult` with:
                * ``success=True`` on successful resolution or escalation
                * ``actions_taken`` listing the resolution action taken
                * ``approval_needed=True`` if escalation is required
                * ``data`` containing ``resolution``, ``exception_type``,
                  ``resolution_notes``, and ``original_exception``
        """
        exception_data: Dict[str, Any] = work_item.data
        exception_type: str = exception_data.get("exception_type", "unknown")
        invoice_info: Dict[str, Any] = exception_data.get("invoice", {})

        logger.info(
            "handling_invoice_exception",
            agent_id=str(self.config.agent_id),
            exception_type=exception_type,
            invoice_id=invoice_info.get("invoice_id", "unknown"),
            work_item_id=str(work_item.workflow_id),
        )

        try:
            # Invoke the DecisionEngine for exception resolution
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="handle_exception",
                context={
                    "exception": exception_data,
                    "invoice": invoice_info,
                },
            )

            # Determine resolution action from the decision output
            resolution: str = decision.get("resolution", "escalate")
            resolution_notes: str = decision.get("resolution_notes", "")

            # Map resolution to work-result flags
            approval_needed: bool = False
            actions_taken: List[str] = []

            if resolution == "resolve":
                actions_taken.append("exception_resolved")
            elif resolution == "escalate":
                approval_needed = True
                actions_taken.append("exception_escalated")
            elif resolution == "reject":
                actions_taken.append("invoice_rejected")
            else:
                # Unrecognised resolution — default to escalation for safety
                approval_needed = True
                actions_taken.append("exception_escalated")
                resolution = "escalate"
                resolution_notes = (
                    f"Unrecognised resolution '{decision.get('resolution')}'; "
                    "defaulting to escalation."
                )

            result = WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=approval_needed,
                has_exceptions=True,
                data={
                    "resolution": resolution,
                    "exception_type": exception_type,
                    "resolution_notes": resolution_notes,
                    "original_exception": exception_data,
                },
            )

            logger.info(
                "invoice_exception_handled",
                agent_id=str(self.config.agent_id),
                resolution=resolution,
                approval_needed=approval_needed,
                exception_type=exception_type,
                invoice_id=invoice_info.get("invoice_id", "unknown"),
            )

            return result

        except Exception as exc:
            logger.error(
                "invoice_exception_handling_error",
                agent_id=str(self.config.agent_id),
                exception_type=exception_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=[],
                approval_needed=True,
                has_exceptions=True,
                data={
                    "resolution": "escalate",
                    "exception_type": exception_type,
                    "original_exception": exception_data,
                },
                error=f"Invoice exception handling failed: {exc}",
            )
