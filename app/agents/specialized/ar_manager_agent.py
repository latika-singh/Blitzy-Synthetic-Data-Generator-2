"""Accounts Receivable Manager agent implementation.

This module provides the ARManagerAgent class — the Accounts Receivable manager
responsible for AR approval, customer payment dispute resolution, credit memo
processing, and payment exception handling.

The AR Manager sits above the AR Clerk in the organizational hierarchy and handles
escalated items including customer disputes, short payment reviews, credit memo
approvals, and payment exception resolutions. Unlike some other manager agents,
the AR Manager has no explicit monetary approval thresholds defined in the
specification; instead, it manages AR-related approvals and dispute resolution
with escalation to Controller for significant cases.

Key responsibilities:
    - Approving AR-related transactions escalated from AR Clerk
    - Resolving customer payment disputes (accept, reject, partial credit, escalate)
    - Reviewing and approving credit memo requests
    - Handling payment exceptions (short payments, overpayments, duplicates)
    - Escalating significant issues to Controller for further review

Decision types used:
    - approve_transaction: For AR approval decisions
    - handle_exception: For dispute resolution and payment exception handling

All logging uses structlog to stdout in structured JSON format per AAP Section 0.7.6.
"""

import asyncio
from typing import Dict, Any, Optional, List

import structlog

from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry

# Module-level structured logger for AR Manager agent events
logger = structlog.get_logger(__name__)


class ARManagerAgent(BaseAgent):
    """Accounts Receivable Manager agent.

    Responsible for:
        - Approving AR-related transactions escalated from AR Clerk
        - Resolving customer payment disputes with multiple resolution paths
        - Reviewing and approving credit memo requests
        - Handling payment exceptions (short payments, overpayments, duplicates)
        - Managing customer collections escalation to Controller

    Role: "ar_manager"
    Transaction types handled: "ar_approval", "dispute_resolution",
                               "credit_memo", "payment_exception"

    The AR Manager uses the 4-layer DecisionEngine pipeline for all decisions:
        Statistical Layer → LLM Layer → Validation Layer → Deterministic Layer
    Financial calculations are exclusively computed in the Deterministic Layer
    and are never LLM-generated.

    Dispute resolution outcomes include:
        - accept_dispute: Full credit memo issued to customer
        - reject_dispute: Original amount maintained
        - partial_credit: Partial credit memo issued
        - escalate: Forwarded to Controller for further review

    Constructor injection pattern (inherited from BaseAgent):
        config: AgentConfig — agent identity and personality traits
        memory: AgentMemory — dual-stream observation/reflection memory
        decision_engine: DecisionEngine — 4-layer hybrid decision pipeline
        action_registry: ActionRegistry — registry of 14 ERP action types
    """

    # Agent role identifier matching VALID_AGENT_ROLES in agent_config.py
    ROLE: str = "ar_manager"

    # Work item types this agent can process
    SUPPORTED_WORK_TYPES: List[str] = [
        "ar_approval",
        "dispute_resolution",
        "credit_memo",
        "payment_exception",
    ]

    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process an AR manager work item by routing to the appropriate handler.

        Routes work items based on their type to specialized handler methods.
        Each handler uses the DecisionEngine (via self.make_decision()) for
        LLM-powered decisions and returns a WorkResult with the outcome.

        Args:
            work_item: The work item to process. Must have a type matching
                one of SUPPORTED_WORK_TYPES.

        Returns:
            WorkResult containing the processing outcome, actions taken,
            and any escalation or exception flags.

        Raises:
            ValueError: If work_item.type is not in SUPPORTED_WORK_TYPES.
        """
        logger.info(
            "ar_manager_processing",
            agent_id=self.config.agent_id,
            agent_role=self.config.role,
            work_item_type=work_item.type,
            workflow_id=work_item.workflow_id,
        )

        try:
            if work_item.type == "ar_approval":
                return await self._process_ar_approval(work_item)
            elif work_item.type == "dispute_resolution":
                return await self._resolve_dispute(work_item)
            elif work_item.type == "credit_memo":
                return await self._process_credit_memo(work_item)
            elif work_item.type == "payment_exception":
                return await self._handle_payment_exception(work_item)
            else:
                raise ValueError(
                    f"Unsupported work item type for AR Manager: {work_item.type}. "
                    f"Supported types: {self.SUPPORTED_WORK_TYPES}"
                )
        except ValueError:
            # Re-raise ValueError for unsupported types
            raise
        except Exception as exc:
            logger.error(
                "ar_manager_processing_error",
                agent_id=self.config.agent_id,
                work_item_type=work_item.type,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["processing_failed"],
                has_exceptions=True,
                error=f"AR Manager processing error: {str(exc)}",
                data={"work_item_type": work_item.type},
            )

    async def _process_ar_approval(self, work_item: WorkItem) -> WorkResult:
        """Process an AR approval request.

        Reviews AR-related transactions that require manager-level approval.
        This includes customer invoice approvals, large credit applications,
        and other AR transactions escalated from the AR Clerk.

        The decision is made via the 4-layer DecisionEngine pipeline using
        the 'approve_transaction' decision type, which considers the
        transaction context, amount, and agent personality traits.

        Args:
            work_item: Work item containing AR transaction data for approval.
                Expected data keys: transaction details, amount, customer info.

        Returns:
            WorkResult with approval decision outcome. Possible outcomes:
                - Approved: success=True, actions_taken includes "ar_transaction_approved"
                - Rejected: success=True, actions_taken includes "ar_transaction_rejected"
                - Escalated: success=True, approval_needed=True for Controller review
        """
        transaction_data: Dict[str, Any] = work_item.data
        amount: float = transaction_data.get("amount", work_item.amount or 0.0)
        transaction_type: str = transaction_data.get("transaction_type", "ar_transaction")
        customer_id: str = str(transaction_data.get("customer_id", "unknown"))

        logger.info(
            "ar_approval_processing",
            agent_id=self.config.agent_id,
            amount=amount,
            transaction_type=transaction_type,
            customer_id=customer_id,
        )

        # Build approval context for DecisionEngine
        approval_context: Dict[str, Any] = {
            "transaction": transaction_data,
            "amount": amount,
            "transaction_type": transaction_type,
            "customer_id": customer_id,
            "approver_role": self.ROLE,
        }

        try:
            # Use the 4-layer DecisionEngine via BaseAgent.make_decision()
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="approve_transaction",
                context=approval_context,
            )
        except Exception as exc:
            logger.error(
                "ar_approval_decision_error",
                agent_id=self.config.agent_id,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["ar_approval_decision_failed"],
                has_exceptions=True,
                error=f"Decision engine error during AR approval: {str(exc)}",
                data={"transaction_type": transaction_type, "amount": amount},
            )

        # Extract the decision outcome
        decision_result: str = decision.get("decision", "").lower()
        reasoning: str = decision.get("reasoning", "")

        if decision_result in ("approve", "approved"):
            actions_taken: List[str] = ["ar_transaction_approved"]
            result_data: Dict[str, Any] = {
                "approved": True,
                "approver": self.ROLE,
                "approver_agent_id": self.config.agent_id,
                "amount": amount,
                "transaction_type": transaction_type,
                "reasoning": reasoning,
            }

            logger.info(
                "ar_approval_decision",
                agent_id=self.config.agent_id,
                decision="approved",
                amount=amount,
                transaction_type=transaction_type,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        elif decision_result in ("reject", "rejected"):
            actions_taken = ["ar_transaction_rejected"]
            result_data = {
                "approved": False,
                "approver": self.ROLE,
                "approver_agent_id": self.config.agent_id,
                "amount": amount,
                "transaction_type": transaction_type,
                "rejection_reason": reasoning,
            }

            logger.info(
                "ar_approval_decision",
                agent_id=self.config.agent_id,
                decision="rejected",
                amount=amount,
                transaction_type=transaction_type,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        else:
            # Escalate to Controller for further review
            actions_taken = ["escalated_to_controller"]
            result_data = {
                "approved": False,
                "escalated": True,
                "escalation_reason": reasoning or "Requires higher-level approval",
                "amount": amount,
                "transaction_type": transaction_type,
                "original_approver": self.ROLE,
            }

            logger.info(
                "ar_approval_decision",
                agent_id=self.config.agent_id,
                decision="escalated",
                amount=amount,
                transaction_type=transaction_type,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=True,
                data=result_data,
            )

    async def _resolve_dispute(self, work_item: WorkItem) -> WorkResult:
        """Resolve a customer payment dispute.

        Handles customer disputes including billing disagreements, service
        quality issues, pricing disputes, and quantity discrepancies. The
        AR Manager evaluates the dispute context and determines the
        appropriate resolution path.

        Resolution outcomes:
            - accept_dispute: Full credit memo issued to customer
            - reject_dispute: Original invoice amount maintained
            - partial_credit: Partial credit memo issued
            - escalate: Forwarded to Controller for further review

        The dispute_rate behavior profile parameter (from external world
        simulation) drives the frequency of disputes entering this handler.

        Args:
            work_item: Work item containing dispute details.
                Expected data keys: customer_id, invoice_id, dispute_reason,
                disputed_amount, original_amount, customer (customer context).

        Returns:
            WorkResult with dispute resolution outcome including resolution
            type, credit amount (if applicable), and escalation flags.
        """
        dispute_data: Dict[str, Any] = work_item.data
        customer_id: str = str(dispute_data.get("customer_id", "unknown"))
        invoice_id: str = str(dispute_data.get("invoice_id", "unknown"))
        dispute_reason: str = dispute_data.get("dispute_reason", "unspecified")
        disputed_amount: float = dispute_data.get(
            "disputed_amount", work_item.amount or 0.0
        )
        original_amount: float = dispute_data.get("original_amount", disputed_amount)

        logger.info(
            "dispute_resolution_processing",
            agent_id=self.config.agent_id,
            customer_id=customer_id,
            invoice_id=invoice_id,
            dispute_reason=dispute_reason,
            disputed_amount=disputed_amount,
        )

        # Build dispute context for DecisionEngine
        dispute_context: Dict[str, Any] = {
            "dispute": dispute_data,
            "customer": dispute_data.get("customer", {}),
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "dispute_reason": dispute_reason,
            "disputed_amount": disputed_amount,
            "original_amount": original_amount,
        }

        try:
            # Use handle_exception decision type for dispute resolution
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="handle_exception",
                context=dispute_context,
            )
        except Exception as exc:
            logger.error(
                "dispute_resolution_decision_error",
                agent_id=self.config.agent_id,
                customer_id=customer_id,
                invoice_id=invoice_id,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["dispute_resolution_failed"],
                has_exceptions=True,
                error=f"Decision engine error during dispute resolution: {str(exc)}",
                data={
                    "customer_id": customer_id,
                    "invoice_id": invoice_id,
                    "disputed_amount": disputed_amount,
                },
            )

        # Extract resolution type from decision
        resolution: str = decision.get("resolution", "escalate").lower()
        resolution_reasoning: str = decision.get("reasoning", "")
        credit_amount: float = decision.get("credit_amount", 0.0)

        if resolution in ("accept_dispute", "accept", "full_credit"):
            # Accept the dispute — issue full credit memo to customer
            credit_amount = disputed_amount
            actions_taken: List[str] = ["dispute_accepted", "credit_memo_issued"]
            result_data: Dict[str, Any] = {
                "resolution_type": "accept_dispute",
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "disputed_amount": disputed_amount,
                "credit_amount": credit_amount,
                "original_amount": original_amount,
                "adjusted_amount": original_amount - credit_amount,
                "reasoning": resolution_reasoning,
            }

            logger.info(
                "dispute_resolved",
                agent_id=self.config.agent_id,
                resolution_type="accept_dispute",
                customer_id=customer_id,
                invoice_id=invoice_id,
                disputed_amount=disputed_amount,
                credit_amount=credit_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        elif resolution in ("reject_dispute", "reject", "deny"):
            # Reject the dispute — maintain original invoice amount
            actions_taken = ["dispute_rejected"]
            result_data = {
                "resolution_type": "reject_dispute",
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "disputed_amount": disputed_amount,
                "credit_amount": 0.0,
                "original_amount": original_amount,
                "adjusted_amount": original_amount,
                "reasoning": resolution_reasoning,
            }

            logger.info(
                "dispute_resolved",
                agent_id=self.config.agent_id,
                resolution_type="reject_dispute",
                customer_id=customer_id,
                invoice_id=invoice_id,
                disputed_amount=disputed_amount,
                credit_amount=0.0,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        elif resolution in ("partial_credit", "partial", "compromise"):
            # Issue partial credit — use credit_amount from decision or half
            if credit_amount <= 0.0 or credit_amount >= disputed_amount:
                credit_amount = disputed_amount * 0.5  # Default to 50% if not specified
            actions_taken = ["partial_credit_issued"]
            result_data = {
                "resolution_type": "partial_credit",
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "disputed_amount": disputed_amount,
                "credit_amount": credit_amount,
                "original_amount": original_amount,
                "adjusted_amount": original_amount - credit_amount,
                "reasoning": resolution_reasoning,
            }

            logger.info(
                "dispute_resolved",
                agent_id=self.config.agent_id,
                resolution_type="partial_credit",
                customer_id=customer_id,
                invoice_id=invoice_id,
                disputed_amount=disputed_amount,
                credit_amount=credit_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        else:
            # Escalate to Controller for further review
            actions_taken = ["dispute_escalated_to_controller"]
            result_data = {
                "resolution_type": "escalate",
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "disputed_amount": disputed_amount,
                "credit_amount": 0.0,
                "original_amount": original_amount,
                "escalation_reason": resolution_reasoning or "Requires Controller review",
                "escalated_by": self.ROLE,
            }

            logger.info(
                "dispute_resolved",
                agent_id=self.config.agent_id,
                resolution_type="escalate",
                customer_id=customer_id,
                invoice_id=invoice_id,
                disputed_amount=disputed_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=True,
                data=result_data,
            )

    async def _process_credit_memo(self, work_item: WorkItem) -> WorkResult:
        """Process a credit memo request.

        Reviews credit memo requests for customer accounts. Credit memos
        may originate from dispute resolutions, return authorizations,
        pricing corrections, or service credits. The AR Manager evaluates
        the request and decides whether to approve, reject, or escalate.

        Args:
            work_item: Work item containing credit memo details.
                Expected data keys: customer_id, invoice_id, credit_amount,
                credit_reason, original_invoice_amount.

        Returns:
            WorkResult with credit memo processing outcome. Possible outcomes:
                - Approved: Credit memo issued
                - Rejected: Credit memo denied
                - Escalated: Forwarded to Controller for high-value review
        """
        memo_data: Dict[str, Any] = work_item.data
        customer_id: str = str(memo_data.get("customer_id", "unknown"))
        invoice_id: str = str(memo_data.get("invoice_id", "unknown"))
        credit_amount: float = memo_data.get(
            "credit_amount", work_item.amount or 0.0
        )
        credit_reason: str = memo_data.get("credit_reason", "unspecified")
        original_invoice_amount: float = memo_data.get(
            "original_invoice_amount", 0.0
        )

        logger.info(
            "credit_memo_processing",
            agent_id=self.config.agent_id,
            customer_id=customer_id,
            invoice_id=invoice_id,
            credit_amount=credit_amount,
            credit_reason=credit_reason,
        )

        # Build approval context for credit memo
        memo_context: Dict[str, Any] = {
            "transaction": memo_data,
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "credit_amount": credit_amount,
            "credit_reason": credit_reason,
            "original_invoice_amount": original_invoice_amount,
            "approver_role": self.ROLE,
        }

        try:
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="approve_transaction",
                context=memo_context,
            )
        except Exception as exc:
            logger.error(
                "credit_memo_decision_error",
                agent_id=self.config.agent_id,
                customer_id=customer_id,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["credit_memo_processing_failed"],
                has_exceptions=True,
                error=f"Decision engine error during credit memo processing: {str(exc)}",
                data={
                    "customer_id": customer_id,
                    "invoice_id": invoice_id,
                    "credit_amount": credit_amount,
                },
            )

        decision_result: str = decision.get("decision", "").lower()
        reasoning: str = decision.get("reasoning", "")

        if decision_result in ("approve", "approved"):
            actions_taken: List[str] = ["credit_memo_approved", "credit_memo_issued"]
            result_data: Dict[str, Any] = {
                "approved": True,
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "credit_amount": credit_amount,
                "credit_reason": credit_reason,
                "original_invoice_amount": original_invoice_amount,
                "adjusted_invoice_amount": original_invoice_amount - credit_amount,
                "approver": self.ROLE,
                "approver_agent_id": self.config.agent_id,
                "reasoning": reasoning,
            }

            logger.info(
                "credit_memo_processed",
                agent_id=self.config.agent_id,
                decision="approved",
                customer_id=customer_id,
                credit_amount=credit_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        elif decision_result in ("reject", "rejected"):
            actions_taken = ["credit_memo_rejected"]
            result_data = {
                "approved": False,
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "credit_amount": credit_amount,
                "rejection_reason": reasoning,
                "approver": self.ROLE,
            }

            logger.info(
                "credit_memo_processed",
                agent_id=self.config.agent_id,
                decision="rejected",
                customer_id=customer_id,
                credit_amount=credit_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=False,
                data=result_data,
            )

        else:
            # Escalate to Controller for high-value or complex cases
            actions_taken = ["credit_memo_escalated_to_controller"]
            result_data = {
                "approved": False,
                "escalated": True,
                "customer_id": customer_id,
                "invoice_id": invoice_id,
                "credit_amount": credit_amount,
                "escalation_reason": reasoning or "Requires Controller review",
                "original_approver": self.ROLE,
            }

            logger.info(
                "credit_memo_processed",
                agent_id=self.config.agent_id,
                decision="escalated",
                customer_id=customer_id,
                credit_amount=credit_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=True,
                data=result_data,
            )

    async def _handle_payment_exception(self, work_item: WorkItem) -> WorkResult:
        """Handle a payment exception case.

        Processes payment exceptions including short payments, overpayments,
        duplicate payments, and unidentified payment receipts. The AR Manager
        determines the appropriate resolution for each exception type.

        Exception types handled:
            - short_payment: Customer paid less than invoice amount
            - overpayment: Customer paid more than invoice amount
            - duplicate_payment: Same payment received twice
            - unidentified_payment: Payment cannot be matched to an invoice
            - misapplied_payment: Payment applied to wrong invoice

        Args:
            work_item: Work item containing payment exception details.
                Expected data keys: exception_type, customer_id, payment_amount,
                invoice_amount, invoice_id, payment_id.

        Returns:
            WorkResult with exception resolution outcome including resolution
            actions, adjustment amounts, and any escalation flags.
        """
        exception_data: Dict[str, Any] = work_item.data
        exception_type: str = exception_data.get("exception_type", "unknown")
        customer_id: str = str(exception_data.get("customer_id", "unknown"))
        payment_amount: float = exception_data.get(
            "payment_amount", work_item.amount or 0.0
        )
        invoice_amount: float = exception_data.get("invoice_amount", 0.0)
        invoice_id: str = str(exception_data.get("invoice_id", "unknown"))
        payment_id: str = str(exception_data.get("payment_id", "unknown"))

        logger.info(
            "payment_exception_processing",
            agent_id=self.config.agent_id,
            exception_type=exception_type,
            customer_id=customer_id,
            payment_amount=payment_amount,
            invoice_amount=invoice_amount,
        )

        # Build exception context for DecisionEngine
        exception_context: Dict[str, Any] = {
            "exception": exception_data,
            "exception_type": exception_type,
            "customer_id": customer_id,
            "payment_amount": payment_amount,
            "invoice_amount": invoice_amount,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
            "variance": payment_amount - invoice_amount,
        }

        try:
            # Use handle_exception decision type for payment exceptions
            decision: Dict[str, Any] = await self.make_decision(
                decision_type="handle_exception",
                context=exception_context,
            )
        except Exception as exc:
            logger.error(
                "payment_exception_decision_error",
                agent_id=self.config.agent_id,
                exception_type=exception_type,
                customer_id=customer_id,
                error=str(exc),
            )
            return WorkResult(
                success=False,
                actions_taken=["payment_exception_resolution_failed"],
                has_exceptions=True,
                error=f"Decision engine error during payment exception handling: {str(exc)}",
                data={
                    "exception_type": exception_type,
                    "customer_id": customer_id,
                    "payment_amount": payment_amount,
                },
            )

        # Extract resolution from decision
        resolution: str = decision.get("resolution", "escalate").lower()
        resolution_reasoning: str = decision.get("reasoning", "")
        adjustment_amount: float = decision.get("adjustment_amount", 0.0)

        # Determine actions based on exception type and resolution
        actions_taken: List[str] = []
        result_data: Dict[str, Any] = {
            "exception_type": exception_type,
            "customer_id": customer_id,
            "invoice_id": invoice_id,
            "payment_id": payment_id,
            "payment_amount": payment_amount,
            "invoice_amount": invoice_amount,
            "resolution": resolution,
            "reasoning": resolution_reasoning,
        }

        if resolution in ("write_off", "accept_short_payment"):
            # Write off the difference for short payments
            write_off_amount: float = abs(invoice_amount - payment_amount)
            actions_taken = ["short_payment_accepted", "write_off_recorded"]
            result_data["write_off_amount"] = write_off_amount
            result_data["adjusted_balance"] = 0.0

        elif resolution in ("apply_credit", "issue_refund"):
            # Issue credit or refund for overpayments
            refund_amount: float = abs(payment_amount - invoice_amount)
            actions_taken = ["overpayment_processed", "credit_applied"]
            result_data["refund_amount"] = refund_amount
            result_data["credit_amount"] = refund_amount

        elif resolution in ("reverse_duplicate", "reverse"):
            # Reverse the duplicate payment
            actions_taken = ["duplicate_payment_identified", "payment_reversed"]
            result_data["reversed_amount"] = payment_amount

        elif resolution in ("reapply", "correct_application"):
            # Reapply misapplied payment to correct invoice
            correct_invoice_id: str = decision.get("correct_invoice_id", "")
            actions_taken = ["payment_reapplied"]
            result_data["correct_invoice_id"] = correct_invoice_id

        elif resolution in ("collect_remaining", "pursue_collection"):
            # Pursue collection for remaining balance (short payments)
            remaining_balance: float = abs(invoice_amount - payment_amount)
            actions_taken = ["collection_initiated"]
            result_data["remaining_balance"] = remaining_balance

        elif resolution in ("escalate", "refer_to_controller"):
            # Escalate to Controller for complex or high-value exceptions
            actions_taken = ["payment_exception_escalated_to_controller"]
            result_data["escalated"] = True
            result_data["escalation_reason"] = (
                resolution_reasoning or "Requires Controller review"
            )
            result_data["escalated_by"] = self.ROLE

            logger.info(
                "payment_exception_handled",
                agent_id=self.config.agent_id,
                exception_type=exception_type,
                resolution="escalated",
                customer_id=customer_id,
                payment_amount=payment_amount,
            )

            return WorkResult(
                success=True,
                actions_taken=actions_taken,
                approval_needed=True,
                data=result_data,
            )

        else:
            # Default resolution: record the exception and flag for review
            actions_taken = ["payment_exception_recorded"]
            result_data["requires_review"] = True
            if adjustment_amount != 0.0:
                result_data["adjustment_amount"] = adjustment_amount
                actions_taken.append("adjustment_recorded")

        logger.info(
            "payment_exception_handled",
            agent_id=self.config.agent_id,
            exception_type=exception_type,
            resolution=resolution,
            customer_id=customer_id,
            payment_amount=payment_amount,
            actions_count=len(actions_taken),
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=resolution not in (
                "write_off",
                "accept_short_payment",
                "apply_credit",
                "issue_refund",
                "reverse_duplicate",
                "reverse",
                "reapply",
                "correct_application",
            ),
            data=result_data,
        )
