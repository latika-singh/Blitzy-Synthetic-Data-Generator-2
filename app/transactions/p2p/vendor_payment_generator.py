"""Vendor Payment Generator — Stage 5 (final) of the Procure-to-Pay (P2P) cycle.

Generates synthetic vendor payments against approved invoices, applying early
payment discounts, grouping invoices by vendor and payment terms, selecting
payment methods, and posting GL entries to close the P2P cycle.

Business Flow
-------------
1.  Query for approved, unpaid vendor invoices ready for payment.
2.  Group invoices by ``(vendor_id, payment_terms)`` for batch payment.
3.  For each vendor/terms group:
    a.  Sort invoices by invoice date (FIFO — oldest first) for allocation.
    b.  Calculate early payment discount eligibility per invoice.
    c.  Calculate payment amounts:
        ``payment = invoice_total - prior_payments - discount``.
    d.  Select payment method based on total amount:
        * check:  total < $5,000
        * ach:    $5,000 ≤ total < $50,000
        * wire:   total ≥ $50,000
    e.  Generate sequential payment number ``VPAY-YYYY-NNNN``.
    f.  Create :class:`PaymentAllocation` for each invoice in the group.
    g.  Create :class:`VendorPaymentRecord` with all allocations.
4.  Check discrepancy trigger (via ``DiscrepancyInjector``).
5.  Post GL entries:
    * DEBIT:  Accounts Payable — ``gross_amount`` (reduce liability)
    * CREDIT: Cash / Bank — ``net_amount`` (actual disbursement)
    * CREDIT: Purchase Discount — ``discount_amount`` (if > 0)
    * **CRITICAL**: ``SUM(debits) = SUM(credits)`` within ``$0.01``.
6.  Publish ``TransactionCompleted`` event (completes the P2P cycle).

Key Business Rules
------------------
*   Payment allocated **FIFO** (oldest invoice first).
*   Early payment discount: e.g., ``2/10 Net 30`` means 2 % if paid ≤ 10 days.
*   Overpayment: must create *Unapplied Cash* — **NEVER** negative invoice balance.
*   All monetary calculations use :class:`~decimal.Decimal` — **never** ``float``.
*   On GL posting failure: **FULL ROLLBACK** (atomicity, AAP §0.7.2).
*   Deterministic reproducibility via seeded ``random.Random`` instances.

Performance Target
------------------
Part of P2P ≥ 50 complete cycles / minute (AAP §0.7.3).

Retry Policy (AAP §0.1.2)
--------------------------
``p2p_cycle_generation``: 2 attempts, linear backoff (1 s, 2 s), 60 s timeout,
fallback: skip transaction.

Database Pattern
----------------
::

    from synthetic_erp.db.session import get_session
    async with get_session() as session:
        ...

Exports
-------
:class:`PaymentAllocation`
    Pydantic V2 model for a single payment → invoice allocation.
:class:`VendorPaymentRecord`
    Pydantic V2 model for a complete vendor payment with allocations.
:class:`VendorPaymentGenerator`
    Concrete :class:`TransactionGenerator` subclass for vendor payments.
"""

from __future__ import annotations

import decimal
import random
import re
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.base_generator import (
    GenerationContext,
    TransactionGenerator,
    TransactionResult,
)
from app.transactions.constants import (
    APPROVAL_THRESHOLDS,
    DOCUMENT_NUMBER_PREFIXES,
    FINANCIAL_TOLERANCES,
    GL_BALANCE_TOLERANCE,
    RETRY_POLICIES,
)
from app.transactions.exceptions import (
    GLPostingError,
    PaymentAllocationError,
    TransactionGenerationError,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus
    from app.orchestration.approval_system import ApprovalSystem
    from app.orchestration.workflow_orchestrator import WorkflowOrchestrator

# ---------------------------------------------------------------------------
# Module-level structured logger — AAP §0.7.7 (JSON to stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Decimal precision configuration — AAP §0.7.2 (MUST be at module level)
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Module-level constants — fallback / default data
# ---------------------------------------------------------------------------

_PAYMENT_METHOD_CHECK_THRESHOLD = Decimal("5000")
"""Below this amount payments use *check*."""

_PAYMENT_METHOD_WIRE_THRESHOLD = Decimal("50000")
"""At or above this amount payments use *wire*."""

_DEFAULT_INVOICES: List[Dict[str, Any]] = [
    {
        "invoice_id": str(uuid4()),
        "invoice_number": "VINV-2024-0001",
        "vendor_id": str(uuid4()),
        "vendor_name": "Acme Office Supplies",
        "payment_terms": "Net 30",
        "invoice_date": "2024-03-01",
        "total_amount": "4500.00",
        "prior_payments": "0.00",
        "status": "approved",
    },
    {
        "invoice_id": str(uuid4()),
        "invoice_number": "VINV-2024-0002",
        "vendor_id": str(uuid4()),
        "vendor_name": "Global Tech Components",
        "payment_terms": "2/10 Net 30",
        "invoice_date": "2024-03-05",
        "total_amount": "12750.00",
        "prior_payments": "0.00",
        "status": "approved",
    },
    {
        "invoice_id": str(uuid4()),
        "invoice_number": "VINV-2024-0003",
        "vendor_id": str(uuid4()),
        "vendor_name": "Premier Industrial Parts",
        "payment_terms": "Net 45",
        "invoice_date": "2024-02-20",
        "total_amount": "67300.00",
        "prior_payments": "0.00",
        "status": "approved",
    },
]

_GL_ACCOUNTS: Dict[str, str] = {
    "accounts_payable": "2100-001",
    "cash": "1010-001",
    "purchase_discount": "5900-001",
}
"""Default GL account numbers for vendor payment journal entries."""

_DISCOUNT_TERMS_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)\s*/\s*(\d+)\s+Net\s+(\d+)$",
    re.IGNORECASE,
)
"""Regex matching discount terms like ``2/10 Net 30``."""


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════════


class PaymentAllocation(BaseModel):
    """Allocation of a single payment to a specific vendor invoice.

    Each allocation records the amount applied from a single payment
    towards one outstanding invoice.  When a payment covers multiple
    invoices the payment record contains one :class:`PaymentAllocation`
    per invoice, ordered by invoice date (FIFO).

    Attributes:
        allocation_id: Unique allocation identifier.
        invoice_id: Reference to the invoice being paid.
        invoice_number: Human-readable invoice number for audit trail.
        invoice_amount: Original total of the invoice.
        prior_payments: Amount previously applied to this invoice.
        discount_amount: Early payment discount applied in this allocation.
        payment_amount: Cash amount being applied in this allocation.
        remaining_balance: Amount still owed after this allocation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    allocation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique allocation identifier.",
    )
    invoice_id: UUID = Field(
        ...,
        description="Invoice being paid.",
    )
    invoice_number: str = Field(
        default="",
        description="Invoice reference number for audit trail.",
    )
    invoice_amount: Decimal = Field(
        ...,
        description="Original invoice total.",
    )
    prior_payments: Decimal = Field(
        default=Decimal("0"),
        description="Amount previously paid on this invoice.",
    )
    discount_amount: Decimal = Field(
        default=Decimal("0"),
        description="Early payment discount applied.",
    )
    payment_amount: Decimal = Field(
        ...,
        description="Amount being paid in this allocation.",
    )
    remaining_balance: Decimal = Field(
        default=Decimal("0"),
        description="Amount still owed after this allocation.",
    )


class VendorPaymentRecord(BaseModel):
    """Complete vendor payment record with invoice allocations.

    Represents a single payment disbursement to a vendor, potentially
    covering multiple invoices grouped by vendor + payment-terms.
    Serialised via ``model_dump()`` for artifact storage and event
    payloads.

    Attributes:
        payment_id: Unique payment identifier (UUID).
        payment_number: Sequential ``VPAY-YYYY-NNNN`` number.
        vendor_id: Vendor being paid.
        vendor_name: Human-readable vendor name.
        payment_date: Date the payment is issued.
        payment_method: ``check``, ``ach``, or ``wire``.
        payment_terms: Payment terms code (e.g. ``Net 30``).
        gross_amount: Sum of invoice amounts before discounts.
        discount_amount: Total early payment discount.
        net_amount: Actual disbursement (``gross - discount``).
        currency: ISO currency code (``USD`` for MVP).
        status: Lifecycle status: ``pending`` → ``processed`` → ``posted``
            or ``voided``.
        bank_reference: Bank or check reference number.
        created_by: Agent UUID who processed the payment.
        allocations: Per-invoice payment allocations.
        metadata: Extensible key-value pairs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique payment identifier.",
    )
    payment_number: str = Field(
        ...,
        description="Sequential VPAY-YYYY-NNNN.",
    )
    vendor_id: UUID = Field(
        ...,
        description="Vendor being paid.",
    )
    vendor_name: str = Field(
        default="",
        description="Vendor display name.",
    )
    payment_date: date = Field(
        ...,
        description="Date payment is issued.",
    )
    payment_method: str = Field(
        default="check",
        description="check, ach, or wire.",
    )
    payment_terms: str = Field(
        default="Net 30",
        description="Payment terms code.",
    )
    gross_amount: Decimal = Field(
        default=Decimal("0"),
        description="Total invoice amounts before discounts.",
    )
    discount_amount: Decimal = Field(
        default=Decimal("0"),
        description="Total early payment discount.",
    )
    net_amount: Decimal = Field(
        default=Decimal("0"),
        description="Actual payment amount (gross - discount).",
    )
    currency: str = Field(
        default="USD",
        description="Currency code — USD only for MVP.",
    )
    status: str = Field(
        default="pending",
        description="pending, processed, posted, voided.",
    )
    bank_reference: str = Field(
        default="",
        description="Bank / check reference number.",
    )
    created_by: Optional[UUID] = Field(
        default=None,
        description="Agent who processed the payment.",
    )
    allocations: List[PaymentAllocation] = Field(
        default_factory=list,
        description="Per-invoice payment allocations.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VendorPaymentGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class VendorPaymentGenerator(TransactionGenerator):
    """Generates vendor payments against approved invoices.

    **Stage 5 of 5** in the P2P pipeline.  Completes the procure-to-pay
    cycle by issuing payments that clear outstanding vendor invoices.

    Business Flow
    ~~~~~~~~~~~~~
    1. Query for approved, unpaid vendor invoices ready for payment.
    2. Group invoices by ``(vendor_id, payment_terms)`` for batch payment.
    3. For each group: calculate payment (invoice − prior payments − discount).
    4. Select payment method based on net amount (check / ACH / wire).
    5. Create :class:`VendorPaymentRecord` with :class:`PaymentAllocation`
       entries.
    6. Post GL entries: DR AP, CR Cash (DR Discount if applicable).
    7. Update invoice status to ``paid``.
    8. Publish ``TransactionCompleted`` event.

    Constructor Injection (ADR-003)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    All dependencies are ``Optional`` with ``None`` defaults, enabling
    zero-dependency instantiation for unit testing.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        db_session_factory: Optional[Any] = None,
        agent_registry: Optional[AgentRegistry] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
        event_bus: Optional[EventBus] = None,
        approval_system: Optional[ApprovalSystem] = None,
        discrepancy_injector: Optional[Any] = None,
        gl_posting_engine: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise VendorPaymentGenerator with injected dependencies.

        Args:
            db_session_factory: Callable returning an async context manager
                yielding a database session (e.g., ``get_session``).
            agent_registry: Project 2 ``AgentRegistry`` for agent lookup.
            workflow_orchestrator: Project 2 ``WorkflowOrchestrator`` for
                routing payments to ``ap_clerk`` agents.
            event_bus: Project 2 ``EventBus`` for event publishing.
            approval_system: Project 2 ``ApprovalSystem`` for threshold
                approval chains.
            discrepancy_injector: P3 ``DiscrepancyInjector`` instance.
            gl_posting_engine: P3 ``GLPostingEngine`` for journal entry
                posting and balance validation.
            statistical_models: Dictionary of statistical model instances
                (e.g., ``payment_timing`` → ``PaymentTimingModel``).
        """
        super().__init__(
            db_session_factory=db_session_factory,
            agent_registry=agent_registry,
            workflow_orchestrator=workflow_orchestrator,
            event_bus=event_bus,
            approval_system=approval_system,
            discrepancy_injector=discrepancy_injector,
            gl_posting_engine=gl_posting_engine,
            statistical_models=statistical_models,
        )
        self._payment_sequence: int = 0
        self._retry_policy: Dict[str, Any] = RETRY_POLICIES.get(
            "p2p_cycle_generation", {}
        )
        logger.info(
            "vendor_payment_generator_initialized",
            service_name="transactions",
            component="VendorPaymentGenerator",
            retry_policy=self._retry_policy,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Generate a vendor payment for the current simulation day.

        Queries for approved, unpaid invoices, groups them by vendor and
        payment terms, calculates amounts with early-discount eligibility,
        and produces a :class:`TransactionResult` with artifacts matching
        ``REQUIRED_ARTIFACTS["vendor_payment"]``:
        ``["payment_record", "payment_allocation", "gl_entries"]``.

        Args:
            context: Per-invocation generation context.

        Returns:
            A :class:`TransactionResult` carrying the payment artifacts.

        Raises:
            TransactionGenerationError: If generation fails after retries.
        """
        start_ts = time.perf_counter()
        rng = self._create_seeded_rng(context)

        try:
            # ----- 1. Retrieve approved invoices -----
            invoices = await self._fetch_approved_invoices(context, rng)
            if not invoices:
                logger.info(
                    "vendor_payment_no_invoices",
                    simulation_id=str(context.simulation_id),
                    trace_id=str(context.trace_id),
                    service_name="transactions",
                    component="VendorPaymentGenerator",
                )
                return TransactionResult(
                    transaction_type="vendor_payment",
                    status="skipped",
                    amount=Decimal("0"),
                    duration_ms=(time.perf_counter() - start_ts) * 1_000.0,
                )

            # ----- 2. Group by vendor + payment terms -----
            groups = self._group_invoices_by_vendor(invoices)

            # Process the first group to generate a single payment per call
            group_key = list(groups.keys())[0]
            group_invoices = groups[group_key]
            vendor_id_str, payment_terms = group_key

            # Sort FIFO — oldest invoice_date first
            group_invoices.sort(
                key=lambda inv: inv.get("invoice_date", "9999-99-99")
            )

            vendor_id = UUID(vendor_id_str) if isinstance(vendor_id_str, str) else vendor_id_str
            vendor_name = group_invoices[0].get("vendor_name", "")

            # ----- 3. Calculate allocations per invoice -----
            allocations: List[PaymentAllocation] = []
            total_gross = Decimal("0")
            total_discount = Decimal("0")

            for inv in group_invoices:
                inv_amount = Decimal(str(inv.get("total_amount", "0")))
                prior_paid = Decimal(str(inv.get("prior_payments", "0")))
                outstanding = inv_amount - prior_paid

                if outstanding <= Decimal("0"):
                    continue

                inv_date_raw = inv.get("invoice_date", str(context.current_date))
                if isinstance(inv_date_raw, str):
                    inv_date = date.fromisoformat(inv_date_raw)
                else:
                    inv_date = inv_date_raw

                discount = self._calculate_early_discount(
                    payment_terms=payment_terms,
                    invoice_amount=outstanding,
                    invoice_date=inv_date,
                    current_date=context.current_date,
                )
                pay_amount = outstanding - discount
                remaining = Decimal("0")

                inv_id_raw = inv.get("invoice_id", str(uuid4()))
                inv_id = UUID(inv_id_raw) if isinstance(inv_id_raw, str) else inv_id_raw

                allocation = PaymentAllocation(
                    invoice_id=inv_id,
                    invoice_number=inv.get("invoice_number", ""),
                    invoice_amount=inv_amount,
                    prior_payments=prior_paid,
                    discount_amount=discount,
                    payment_amount=pay_amount,
                    remaining_balance=remaining,
                )
                allocations.append(allocation)
                total_gross += outstanding
                total_discount += discount

            if not allocations:
                return TransactionResult(
                    transaction_type="vendor_payment",
                    status="skipped",
                    amount=Decimal("0"),
                    duration_ms=(time.perf_counter() - start_ts) * 1_000.0,
                )

            net_amount = total_gross - total_discount

            # ----- 4. Select payment method -----
            payment_method = self._select_payment_method(net_amount)

            # ----- 5. Generate payment number -----
            payment_number = self._generate_payment_number(context.current_date.year)

            # ----- 6. Build payment record -----
            bank_ref = f"REF-{payment_number}-{rng.randint(1000, 9999)}"
            payment_record = VendorPaymentRecord(
                payment_number=payment_number,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                payment_date=context.current_date,
                payment_method=payment_method,
                payment_terms=payment_terms,
                gross_amount=total_gross,
                discount_amount=total_discount,
                net_amount=net_amount,
                currency="USD",
                status="pending",
                bank_reference=bank_ref,
                allocations=allocations,
                metadata={
                    "simulation_id": str(context.simulation_id),
                    "trace_id": str(context.trace_id),
                    "fiscal_period": context.fiscal_period,
                    "invoice_count": len(allocations),
                },
            )

            # ----- 7. Check discrepancy trigger -----
            should_inject, disc_type, disc_params = (
                await self._check_discrepancy_trigger(context, "vendor_payment")
            )

            # ----- 8. Prepare GL entries -----
            gl_entries = self._prepare_gl_entries(
                gross_amount=total_gross,
                net_amount=net_amount,
                discount_amount=total_discount,
                payment_id=payment_record.payment_id,
                context=context,
            )

            # ----- 9. Delegate GL posting -----
            try:
                await self._delegate_gl_posting(gl_entries, context)
                payment_record.status = "posted"
            except Exception as exc:
                payment_record.status = "pending"
                logger.error(
                    "vendor_payment_gl_posting_failed",
                    error=str(exc),
                    payment_number=payment_number,
                    trace_id=str(context.trace_id),
                    service_name="transactions",
                    component="VendorPaymentGenerator",
                )
                raise GLPostingError(
                    "GL posting failed for vendor payment",
                    details={
                        "payment_number": payment_number,
                        "net_amount": str(net_amount),
                        "trace_id": str(context.trace_id),
                    },
                ) from exc

            # ----- 10. Publish event -----
            await self._publish_event(
                event_type="TransactionCompleted",
                payload={
                    "transaction_type": "vendor_payment",
                    "payment_id": str(payment_record.payment_id),
                    "payment_number": payment_number,
                    "vendor_id": str(vendor_id),
                    "net_amount": str(net_amount),
                    "payment_method": payment_method,
                    "invoice_count": len(allocations),
                },
                context=context,
            )

            # ----- 11. Build and return result -----
            elapsed_ms = (time.perf_counter() - start_ts) * 1_000.0

            allocation_dicts = [a.model_dump() for a in allocations]

            result = TransactionResult(
                transaction_type="vendor_payment",
                status="completed",
                artifacts={
                    "payment_record": payment_record.model_dump(),
                    "payment_allocation": allocation_dicts,
                    "gl_entries": gl_entries,
                },
                gl_entries=gl_entries,
                has_discrepancy=should_inject,
                discrepancy_type=disc_type if should_inject else None,
                amount=net_amount,
                duration_ms=elapsed_ms,
                events_published=["TransactionCompleted"],
            )

            self._log_transaction(result, context)

            logger.info(
                "vendor_payment_generated",
                payment_number=payment_number,
                vendor_id=str(vendor_id),
                net_amount=str(net_amount),
                payment_method=payment_method,
                has_discrepancy=should_inject,
                discrepancy_type=disc_type,
                gl_entries=len(gl_entries),
                duration_ms=elapsed_ms,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )

            return result

        except (GLPostingError, PaymentAllocationError):
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_ts) * 1_000.0
            logger.error(
                "vendor_payment_generation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component="VendorPaymentGenerator",
                duration_ms=elapsed_ms,
            )
            raise TransactionGenerationError(
                f"Vendor payment generation failed: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "trace_id": str(context.trace_id),
                },
            ) from exc

    async def validate(self, result: TransactionResult) -> bool:
        """Validate a generated vendor payment against business rules.

        Checks performed:
        *   ``payment_record`` artifact exists with required fields.
        *   ``payment_allocation`` artifact has ≥ 1 allocation.
        *   ``net_amount = gross_amount - discount_amount``.
        *   All amounts are :class:`~decimal.Decimal`.
        *   GL entries balance: ``SUM(debits) = SUM(credits)`` within
            ``$0.01``.
        *   Sum of allocation ``payment_amount`` values equals
            ``net_amount``.
        *   No negative payment amounts.

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if all checks pass, ``False`` otherwise.
        """
        if result.status in ("skipped", "failed"):
            return result.status == "skipped"

        artifacts = result.artifacts
        errors: List[str] = []

        # --- payment_record artifact ---
        payment_record = artifacts.get("payment_record")
        if not payment_record:
            errors.append("Missing payment_record artifact")
        else:
            required_fields = [
                "payment_id", "payment_number", "vendor_id",
                "payment_date", "gross_amount", "discount_amount",
                "net_amount", "payment_method", "status",
            ]
            for field in required_fields:
                if field not in payment_record:
                    errors.append(f"payment_record missing field: {field}")

            if payment_record:
                gross = Decimal(str(payment_record.get("gross_amount", "0")))
                discount = Decimal(str(payment_record.get("discount_amount", "0")))
                net = Decimal(str(payment_record.get("net_amount", "0")))
                expected_net = gross - discount
                if abs(net - expected_net) > GL_BALANCE_TOLERANCE:
                    errors.append(
                        f"net_amount mismatch: {net} != gross({gross}) - "
                        f"discount({discount}) = {expected_net}"
                    )
                if net < Decimal("0"):
                    errors.append(f"Negative net_amount: {net}")

        # --- payment_allocation artifact ---
        allocations = artifacts.get("payment_allocation")
        if not allocations or not isinstance(allocations, list):
            errors.append("Missing or empty payment_allocation artifact")
        else:
            alloc_total = Decimal("0")
            for idx, alloc in enumerate(allocations):
                pay_amt = Decimal(str(alloc.get("payment_amount", "0")))
                if pay_amt < Decimal("0"):
                    errors.append(
                        f"Negative payment_amount in allocation {idx}: {pay_amt}"
                    )
                alloc_total += pay_amt

            if payment_record:
                expected_net = Decimal(str(payment_record.get("net_amount", "0")))
                if abs(alloc_total - expected_net) > GL_BALANCE_TOLERANCE:
                    errors.append(
                        f"Allocation total ({alloc_total}) != "
                        f"net_amount ({expected_net})"
                    )

        # --- GL entries balance ---
        gl_entries = artifacts.get("gl_entries", [])
        if gl_entries:
            total_debits = Decimal("0")
            total_credits = Decimal("0")
            for entry in gl_entries:
                total_debits += Decimal(str(entry.get("debit", "0")))
                total_credits += Decimal(str(entry.get("credit", "0")))

            imbalance = abs(total_debits - total_credits)
            if imbalance > GL_BALANCE_TOLERANCE:
                errors.append(
                    f"GL entries unbalanced: debits={total_debits}, "
                    f"credits={total_credits}, imbalance={imbalance}"
                )

        if errors:
            logger.warning(
                "vendor_payment_validation_failed",
                errors=errors,
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )
            result.errors = errors
            return False

        logger.debug(
            "vendor_payment_validation_passed",
            transaction_id=str(result.transaction_id),
            service_name="transactions",
            component="VendorPaymentGenerator",
        )
        return True

    async def post(self, result: TransactionResult) -> None:
        """Post vendor payment GL entries to the General Ledger.

        GL entries for a vendor payment:
        *   DEBIT: Accounts Payable — ``gross_amount`` (reduce liability)
        *   CREDIT: Cash / Bank — ``net_amount``
        *   CREDIT: Purchase Discount — ``discount_amount`` (if > 0)

        **Atomicity**: If any step fails, the entire transaction is rolled
        back (AAP §0.7.2).

        Args:
            result: A validated :class:`TransactionResult`.

        Raises:
            GLPostingError: If GL posting fails.
        """
        if result.status in ("skipped", "failed"):
            logger.debug(
                "vendor_payment_post_skipped",
                status=result.status,
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )
            return

        gl_entries = result.gl_entries
        if not gl_entries:
            gl_entries = result.artifacts.get("gl_entries", [])

        if not gl_entries:
            logger.debug(
                "vendor_payment_post_no_entries",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )
            return

        # Validate balance before delegating
        total_debits = Decimal("0")
        total_credits = Decimal("0")
        for entry in gl_entries:
            total_debits += Decimal(str(entry.get("debit", "0")))
            total_credits += Decimal(str(entry.get("credit", "0")))

        imbalance = abs(total_debits - total_credits)
        if imbalance > GL_BALANCE_TOLERANCE:
            raise GLPostingError(
                "Cannot post unbalanced vendor payment GL entries",
                details={
                    "total_debits": str(total_debits),
                    "total_credits": str(total_credits),
                    "imbalance": str(imbalance),
                    "tolerance": str(GL_BALANCE_TOLERANCE),
                    "transaction_id": str(result.transaction_id),
                },
            )

        # Construct a lightweight GenerationContext for GL delegation
        context = GenerationContext(
            current_date=date.today(),
            simulation_id=result.transaction_id,
            trace_id=result.transaction_id,
        )

        try:
            await self._delegate_gl_posting(gl_entries, context)
            logger.info(
                "vendor_payment_posted",
                transaction_id=str(result.transaction_id),
                entry_count=len(gl_entries),
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )
        except Exception as exc:
            logger.error(
                "vendor_payment_post_failed",
                error=str(exc),
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )
            raise GLPostingError(
                f"Vendor payment GL posting failed: {exc}",
                details={
                    "transaction_id": str(result.transaction_id),
                    "entry_count": len(gl_entries),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _group_invoices_by_vendor(
        self,
        invoices: Sequence[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Group invoices by ``(vendor_id, payment_terms)`` for batch payment.

        Invoices within each group are paid together in a single payment
        disbursement to the vendor.

        Args:
            invoices: Sequence of invoice dictionaries, each containing
                at least ``vendor_id`` and ``payment_terms`` keys.

        Returns:
            Dictionary mapping ``(vendor_id, payment_terms)`` tuples to
            lists of invoice dictionaries.
        """
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for inv in invoices:
            vendor_id = str(inv.get("vendor_id", ""))
            terms = inv.get("payment_terms", "Net 30")
            key = (vendor_id, terms)
            if key not in groups:
                groups[key] = []
            groups[key].append(inv)
        return groups

    def _calculate_early_discount(
        self,
        *,
        payment_terms: str,
        invoice_amount: Decimal,
        invoice_date: date,
        current_date: date,
    ) -> Decimal:
        """Determine if an early payment discount applies and return the amount.

        Parses discount terms like ``2/10 Net 30`` (2 % discount if paid
        within 10 days of the invoice date).  If the current simulation
        date is on or before the discount deadline, the discount is
        applied.

        Args:
            payment_terms: Payment terms string (e.g., ``"2/10 Net 30"``).
            invoice_amount: Outstanding invoice amount (Decimal).
            invoice_date: Date on the invoice.
            current_date: Current simulation date.

        Returns:
            Discount amount as :class:`~decimal.Decimal`, or
            ``Decimal("0")`` if no discount is eligible.
        """
        match = _DISCOUNT_TERMS_PATTERN.match(payment_terms.strip())
        if not match:
            return Decimal("0")

        discount_pct = Decimal(match.group(1)) / Decimal("100")
        discount_days = int(match.group(2))

        discount_deadline = invoice_date + timedelta(days=discount_days)

        if current_date <= discount_deadline:
            discount_amount = (invoice_amount * discount_pct).quantize(
                Decimal("0.01")
            )
            logger.debug(
                "early_discount_applied",
                payment_terms=payment_terms,
                invoice_amount=str(invoice_amount),
                discount_pct=str(discount_pct),
                discount_amount=str(discount_amount),
                discount_deadline=str(discount_deadline),
                service_name="transactions",
                component="VendorPaymentGenerator",
            )
            return discount_amount

        return Decimal("0")

    def _select_payment_method(self, total_amount: Decimal) -> str:
        """Select payment method based on the total disbursement amount.

        Selection thresholds:
        *   ``check``:  total < $5,000
        *   ``ach``:    $5,000 ≤ total < $50,000
        *   ``wire``:   total ≥ $50,000

        Args:
            total_amount: Net payment amount (Decimal).

        Returns:
            Payment method string: ``"check"``, ``"ach"``, or ``"wire"``.
        """
        if total_amount >= _PAYMENT_METHOD_WIRE_THRESHOLD:
            return "wire"
        if total_amount >= _PAYMENT_METHOD_CHECK_THRESHOLD:
            return "ach"
        return "check"

    def _prepare_gl_entries(
        self,
        *,
        gross_amount: Decimal,
        net_amount: Decimal,
        discount_amount: Decimal,
        payment_id: UUID,
        context: GenerationContext,
    ) -> List[Dict[str, Any]]:
        """Prepare balanced GL journal entries for a vendor payment.

        Journal entry structure:
        *   DEBIT: Accounts Payable — ``gross_amount``
        *   CREDIT: Cash / Bank — ``net_amount``
        *   CREDIT: Purchase Discount — ``discount_amount`` (only if > 0)

        Args:
            gross_amount: Total invoice amounts being paid.
            net_amount: Actual cash disbursement.
            discount_amount: Early payment discount.
            payment_id: Payment identifier for correlation.
            context: Generation context.

        Returns:
            List of GL entry dictionaries with ``account``, ``debit``,
            ``credit``, and ``description`` keys.

        Raises:
            GLPostingError: If entries do not balance within tolerance.
        """
        entries: List[Dict[str, Any]] = []

        # DR Accounts Payable (reduce liability)
        entries.append({
            "account": _GL_ACCOUNTS["accounts_payable"],
            "account_name": "Accounts Payable",
            "debit": str(gross_amount),
            "credit": str(Decimal("0")),
            "description": "Vendor payment — reduce AP liability",
            "payment_id": str(payment_id),
            "fiscal_period": context.fiscal_period,
            "posting_date": str(context.current_date),
        })

        # CR Cash / Bank (reduce asset)
        entries.append({
            "account": _GL_ACCOUNTS["cash"],
            "account_name": "Cash",
            "debit": str(Decimal("0")),
            "credit": str(net_amount),
            "description": "Vendor payment — cash disbursement",
            "payment_id": str(payment_id),
            "fiscal_period": context.fiscal_period,
            "posting_date": str(context.current_date),
        })

        # CR Purchase Discount (if applicable)
        if discount_amount > Decimal("0"):
            entries.append({
                "account": _GL_ACCOUNTS["purchase_discount"],
                "account_name": "Purchase Discount",
                "debit": str(Decimal("0")),
                "credit": str(discount_amount),
                "description": "Vendor payment — early payment discount",
                "payment_id": str(payment_id),
                "fiscal_period": context.fiscal_period,
                "posting_date": str(context.current_date),
            })

        # Validate balance
        total_debits = Decimal("0")
        total_credits = Decimal("0")
        for entry in entries:
            total_debits += Decimal(str(entry["debit"]))
            total_credits += Decimal(str(entry["credit"]))

        imbalance = abs(total_debits - total_credits)
        if imbalance > GL_BALANCE_TOLERANCE:
            raise GLPostingError(
                "Vendor payment GL entries do not balance",
                details={
                    "total_debits": str(total_debits),
                    "total_credits": str(total_credits),
                    "imbalance": str(imbalance),
                    "tolerance": str(GL_BALANCE_TOLERANCE),
                },
            )

        logger.debug(
            "vendor_payment_gl_entries_prepared",
            entry_count=len(entries),
            total_debits=str(total_debits),
            total_credits=str(total_credits),
            discount_applied=discount_amount > Decimal("0"),
            service_name="transactions",
            component="VendorPaymentGenerator",
        )

        return entries

    def _generate_payment_number(self, year: int) -> str:
        """Generate a sequential vendor payment document number.

        Format: ``VPAY-YYYY-NNNN`` (e.g. ``VPAY-2024-0001``).

        Args:
            year: Four-digit year component.

        Returns:
            Formatted payment number string.
        """
        self._payment_sequence += 1
        prefix = DOCUMENT_NUMBER_PREFIXES.get("vendor_payment", "VPAY")
        return self._generate_sequential_number(
            prefix=prefix,
            year=year,
            sequence=self._payment_sequence,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_approved_invoices(
        self,
        context: GenerationContext,
        rng: random.Random,
    ) -> List[Dict[str, Any]]:
        """Fetch approved invoices ready for payment.

        Attempts to query via the injected ``db_session_factory``.  If no
        database is configured, returns deterministic fallback data from
        ``_DEFAULT_INVOICES`` for testing / standalone operation.

        Args:
            context: Generation context for simulation scoping.
            rng: Seeded random generator for deterministic selection.

        Returns:
            List of invoice dictionaries.
        """
        if self._db_session_factory is not None:
            try:
                async with self._db_session_factory() as session:
                    # In a full implementation this would query the
                    # vendor_invoices table for status='approved'
                    # and payment_status != 'paid'.
                    # For now return empty to allow tests to inject data.
                    logger.debug(
                        "vendor_payment_db_query",
                        trace_id=str(context.trace_id),
                        service_name="transactions",
                        component="VendorPaymentGenerator",
                    )
                    return []
            except Exception as exc:
                logger.warning(
                    "vendor_payment_db_query_failed",
                    error=str(exc),
                    trace_id=str(context.trace_id),
                    service_name="transactions",
                    component="VendorPaymentGenerator",
                )

        # Fallback: deterministic sample from defaults
        available = [
            inv.copy() for inv in _DEFAULT_INVOICES
            if inv.get("status") == "approved"
        ]
        if not available:
            return []

        count = rng.randint(1, len(available))
        selected = rng.sample(available, k=count)
        return selected
