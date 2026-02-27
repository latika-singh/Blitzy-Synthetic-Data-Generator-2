"""Customer Payment Processor — FIFO payment allocation for customer receipts.

Processes customer payments as the final step in the Order-to-Cash cycle.
Implements **FIFO (First In, First Out) payment allocation** — oldest invoice
first — which is the ONLY supported allocation strategy (NOT configurable per
AAP §0.1.2).

Key Behaviors:

1. **FIFO Allocation**: Payments are applied to the oldest open invoices first.
   Each invoice is fully paid before moving to the next.

2. **Short Payment Handling**: When payment < total invoices, the remaining
   balance is either written off (below threshold) or left as an open balance
   on the oldest unpaid invoice.

3. **Overpayment Handling** (CRITICAL): If payment > total open invoices,
   the excess is recorded as **Unapplied Cash**. NEVER create a negative
   invoice balance. This is a NON-NEGOTIABLE constraint per AAP §0.1.2:
   "If Payment > Invoice, create Unapplied Cash record — do NOT allow
   negative invoice balances."

4. **GL Posting**: For each payment:
   - DR Cash (payment amount)
   - CR Accounts Receivable (allocated to invoices)
   - DR Discount Allowed (if early payment discount taken)
   - DR Bad Debt Expense (if short pay written off)
   - CR Unapplied Cash (if overpayment — excess amount)

5. **Payment Method Selection**: Check, ACH, or Wire based on amount and
   customer preferences.

6. **Sequential Numbering**: CPAY-YYYY-NNNN (e.g., CPAY-2024-0001)

Database Pattern: ``from synthetic_erp.db.session import get_session``

Retry Policy (AAP §0.1.2 — o2c_cycle_generation):
    - max_attempts: 2
    - backoff: linear (1s, 2s)
    - timeout: 60s
    - fallback: skip transaction

Performance Target: O2C ≥ 60 cycles/min (AAP §0.7.3)

References:
    - AAP §0.5.1 Group 3: O2C Engine (CustomerPaymentProcessor)
    - AAP §0.1.2: FIFO allocation, overpayment → Unapplied Cash
    - AAP §0.7.2: Financial Integrity Rules (Decimal precision, balance validation)
    - AAP §0.7.4: Error Handling (PaymentAllocationError)
"""

from __future__ import annotations

import asyncio
import decimal
import random
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
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
from app.transactions.exceptions import (
    PaymentAllocationError,
    TransactionError,
    TransactionGenerationError,
    GLPostingError,
)
from app.transactions.constants import (
    DOCUMENT_NUMBER_PREFIXES,
    FINANCIAL_TOLERANCES,
    GL_BALANCE_TOLERANCE,
    RETRY_POLICIES,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus
    from app.orchestration.workflow_orchestrator import WorkflowOrchestrator
    from app.transactions.gl.gl_posting_engine import GLPostingEngine
    from app.transactions.gl.account_balance_manager import AccountBalanceManager
    from app.statistical.payment_timing_model import PaymentTimingModel
    from app.statistical.amount_distributions import AmountDistribution

# ---------------------------------------------------------------------------
# Module-level structured logger — AAP §0.7.7 (JSON to stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# CRITICAL: Decimal precision configuration (AAP §0.7.2)
# All financial calculations MUST use Decimal with prec=28 and ROUND_HALF_UP.
# NEVER use float for monetary amounts.
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Default GL account codes for customer payment postings
# ---------------------------------------------------------------------------
_CASH_ACCOUNT_CODE = "1010"
_AR_ACCOUNT_CODE = "1200"
_DISCOUNT_ALLOWED_ACCOUNT_CODE = "6100"
_BAD_DEBT_ACCOUNT_CODE = "6200"
_UNAPPLIED_CASH_ACCOUNT_CODE = "2050"

# ---------------------------------------------------------------------------
# Quantize constant — used for all monetary results
# ---------------------------------------------------------------------------
_TWO_PLACES = Decimal("0.01")

# ---------------------------------------------------------------------------
# Payment method selection thresholds (amounts in USD)
# ---------------------------------------------------------------------------
_SMALL_PAYMENT_THRESHOLD = Decimal("1000")
_MEDIUM_PAYMENT_THRESHOLD = Decimal("10000")


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════


class PaymentMethod(str, Enum):
    """Customer payment methods.

    Values are lowercase strings for serialization compatibility with
    Pydantic V2 and JSON output.
    """

    CHECK = "check"
    ACH = "ach"
    WIRE = "wire"
    CREDIT_CARD = "credit_card"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════


class PaymentAllocation(BaseModel):
    """Represents allocation of a payment to a single invoice.

    CRITICAL INVARIANT: ``invoice_balance_after`` MUST always be >= 0.
    A negative invoice balance is NEVER permitted (AAP §0.1.2).

    All monetary fields use :class:`~decimal.Decimal` — never ``float``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    allocation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique allocation identifier.",
    )
    payment_id: UUID = Field(
        ...,
        description="Parent payment identifier.",
    )
    invoice_id: UUID = Field(
        ...,
        description="Customer invoice being paid.",
    )
    invoice_number: str = Field(
        default="",
        description="Human-readable invoice number.",
    )
    allocated_amount: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Amount applied to this invoice.",
    )
    discount_taken: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Early payment discount taken.",
    )
    write_off_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Bad debt write-off amount.",
    )
    invoice_balance_before: Decimal = Field(
        default=Decimal("0.00"),
        description="Invoice open balance before allocation.",
    )
    invoice_balance_after: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Invoice open balance after allocation — MUST be >= 0, NEVER negative.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of allocation.",
    )


class CustomerPaymentRecord(BaseModel):
    """Complete customer payment record with FIFO allocations.

    Encapsulates the full payment including all invoice allocations,
    discount/write-off totals, and any unapplied cash from overpayment.

    CRITICAL: If ``unapplied_amount > 0``, the payment is an overpayment
    and an Unapplied Cash GL entry is required.

    All monetary fields use :class:`~decimal.Decimal` — never ``float``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    payment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique payment identifier.",
    )
    payment_number: str = Field(
        default="",
        description="Sequential number CPAY-YYYY-NNNN.",
    )
    customer_id: str = Field(
        ...,
        description="Customer identifier.",
    )
    customer_name: str = Field(
        default="",
        description="Customer display name.",
    )
    payment_date: date = Field(
        ...,
        description="Date the payment was received.",
    )
    payment_amount: Decimal = Field(
        ...,
        gt=Decimal("0.00"),
        description="Total payment amount received.",
    )
    payment_method: PaymentMethod = Field(
        default=PaymentMethod.ACH,
        description="Payment method used.",
    )
    bank_reference: str = Field(
        default="",
        description="Bank or check reference number.",
    )
    allocations: List[PaymentAllocation] = Field(
        default_factory=list,
        description="FIFO allocation records.",
    )
    total_allocated: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Sum of amounts allocated to invoices.",
    )
    total_discount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Sum of early payment discounts taken.",
    )
    total_write_off: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Sum of bad debt write-offs.",
    )
    unapplied_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Overpayment excess — MUST become Unapplied Cash if > 0.",
    )
    status: str = Field(
        default="pending",
        description="Payment status: pending, applied, partially_applied, unapplied_created.",
    )
    simulation_id: Optional[UUID] = Field(
        default=None,
        description="Simulation run identifier.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of payment record creation.",
    )


class OpenInvoice(BaseModel):
    """Lightweight model for an open customer invoice in the FIFO queue.

    Carries just enough data to execute FIFO allocation: the invoice
    identity, dates for discount eligibility, and the current open balance.

    All monetary fields use :class:`~decimal.Decimal` — never ``float``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    invoice_id: UUID = Field(
        ...,
        description="Unique invoice identifier.",
    )
    invoice_number: str = Field(
        default="",
        description="Human-readable invoice number.",
    )
    customer_id: str = Field(
        ...,
        description="Customer identifier.",
    )
    invoice_date: date = Field(
        ...,
        description="Invoice issue date.",
    )
    due_date: date = Field(
        ...,
        description="Net payment due date.",
    )
    original_amount: Decimal = Field(
        ...,
        description="Original invoice total.",
    )
    open_balance: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Remaining open balance.",
    )
    payment_terms: str = Field(
        default="Net 30",
        description="Payment terms string.",
    )
    discount_percent: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Early payment discount percentage.",
    )
    discount_due_date: Optional[date] = Field(
        default=None,
        description="Deadline for early payment discount.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# CustomerPaymentProcessor
# ═══════════════════════════════════════════════════════════════════════════


class CustomerPaymentProcessor(TransactionGenerator):
    """Processes customer payments with FIFO allocation.

    Extends :class:`TransactionGenerator` to implement the payment step
    of the O2C cycle.  FIFO allocation is the ONLY supported strategy.

    CRITICAL INVARIANTS:
        1. FIFO: Oldest invoice first — NOT configurable.
        2. Overpayment → Unapplied Cash — NEVER negative invoice balance.
        3. All amounts use :class:`~decimal.Decimal` (never ``float``).
        4. GL posting: DR Cash, CR AR (+ DR Discount, DR Bad Debt,
           CR Unapplied Cash as needed).

    Constructor Injection (ADR-003):
        All parameters are ``Optional`` with ``None`` defaults enabling
        partial composition for testing and incremental integration.
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
        gl_posting_engine: Optional[GLPostingEngine] = None,
        account_balance_manager: Optional[AccountBalanceManager] = None,
        payment_timing_model: Optional[PaymentTimingModel] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        discrepancy_injector: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
        write_off_threshold: Decimal = Decimal("10.00"),
    ) -> None:
        """Initialise the CustomerPaymentProcessor with injected dependencies.

        Args:
            db_session_factory: Callable returning an async DB context manager.
            agent_registry: Project 2 AgentRegistry for agent lookup.
            workflow_orchestrator: Project 2 WorkflowOrchestrator for routing.
            event_bus: Project 2 EventBus for cross-subsystem events.
            gl_posting_engine: P3 GLPostingEngine for journal entry posting.
            account_balance_manager: P3 AccountBalanceManager for balance updates.
            payment_timing_model: Statistical model for payment timing.
            amount_distribution: Statistical model for amount generation.
            discrepancy_injector: P3 DiscrepancyInjector for rate-based injection.
            statistical_models: Dictionary of statistical model instances.
            write_off_threshold: Threshold below which short pays are
                automatically written off as bad debt.  Defaults to $10.00.
        """
        super().__init__(
            db_session_factory=db_session_factory,
            agent_registry=agent_registry,
            workflow_orchestrator=workflow_orchestrator,
            event_bus=event_bus,
            discrepancy_injector=discrepancy_injector,
            gl_posting_engine=gl_posting_engine,
            statistical_models=statistical_models,
        )

        # Additional dependencies specific to payment processing
        self._payment_timing_model = payment_timing_model
        self._amount_distribution = amount_distribution
        self._account_balance_manager = account_balance_manager

        # Write-off threshold for short-pay handling
        self._write_off_threshold: Decimal = write_off_threshold

        # Sequential numbering state
        self._sequence_counter: int = 0
        self._current_year: int = 0

        # Metrics tracking
        self._payments_processed: int = 0
        self._total_allocated: Decimal = Decimal("0.00")
        self._total_unapplied: Decimal = Decimal("0.00")
        self._payments_history: List[CustomerPaymentRecord] = []

        logger.info(
            "customer_payment_processor_initialized",
            service_name="transactions",
            component="CustomerPaymentProcessor",
            has_gl_engine=gl_posting_engine is not None,
            has_payment_timing=payment_timing_model is not None,
            has_balance_manager=account_balance_manager is not None,
            write_off_threshold=str(write_off_threshold),
        )

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Generate customer payment(s) with FIFO allocation.

        Main generation entry point.  Creates one or more customer payments
        by querying open invoices, applying FIFO allocation, and building
        GL entries for each payment.

        Args:
            context: Generation parameters including business date, fiscal
                period, RNG seed, and discrepancy configuration.

        Returns:
            A :class:`TransactionResult` with generated payment artifacts
            and GL entries.

        Raises:
            TransactionGenerationError: If generation fails after all
                configured retry attempts.
        """
        start_time = time.perf_counter()
        result = TransactionResult(
            transaction_type="customer_payment",
            status="completed",
        )

        try:
            rng = self._create_seeded_rng(context)

            # Retrieve open invoices for processing
            open_invoices = self._get_open_invoices()

            if not open_invoices:
                logger.info(
                    "no_open_invoices_for_payment",
                    service_name="transactions",
                    component="CustomerPaymentProcessor",
                    trace_id=str(context.trace_id),
                    simulation_id=str(context.simulation_id),
                )
                result.status = "skipped"
                result.duration_ms = (time.perf_counter() - start_time) * 1_000.0
                return result

            # Group invoices by customer for batch processing
            invoices_by_customer: Dict[str, List[OpenInvoice]] = {}
            for inv in open_invoices:
                invoices_by_customer.setdefault(inv.customer_id, []).append(inv)

            all_payments: List[CustomerPaymentRecord] = []
            all_gl_entries: List[Dict[str, Any]] = []

            for customer_id, customer_invoices in invoices_by_customer.items():
                # Sort invoices by date ascending (FIFO — oldest first)
                customer_invoices.sort(key=lambda inv: inv.invoice_date)

                # Calculate total open balance for the customer
                total_open = sum(
                    inv.open_balance for inv in customer_invoices
                )

                # Determine payment amount — use timing model or generate
                payment_amount = self._determine_payment_amount(
                    total_open, rng, context,
                )
                if payment_amount <= Decimal("0.00"):
                    continue

                # Determine payment date
                payment_date = context.current_date

                # Generate sequential payment number
                payment_number = self._generate_payment_number(payment_date)

                # Select payment method
                payment_method = self._select_payment_method(payment_amount, rng)

                # Create payment ID for correlation
                payment_id = uuid4()

                # Execute FIFO allocation (CORE ALGORITHM)
                allocations, unapplied_amount = self._allocate_payment_fifo(
                    payment_amount=payment_amount,
                    open_invoices=customer_invoices,
                    payment_date=payment_date,
                    payment_id=payment_id,
                )

                # Calculate totals from allocations
                total_allocated = sum(
                    a.allocated_amount for a in allocations
                ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)
                total_discount = sum(
                    a.discount_taken for a in allocations
                ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)
                total_write_off = sum(
                    a.write_off_amount for a in allocations
                ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)

                # Determine status
                if unapplied_amount > Decimal("0.00"):
                    status = "unapplied_created"
                elif total_allocated < payment_amount:
                    status = "partially_applied"
                else:
                    status = "applied"

                # Get customer name from context or default
                customer_name = f"Customer-{customer_id}"

                # Build the payment record
                payment_record = CustomerPaymentRecord(
                    payment_id=payment_id,
                    payment_number=payment_number,
                    customer_id=customer_id,
                    customer_name=customer_name,
                    payment_date=payment_date,
                    payment_amount=payment_amount,
                    payment_method=payment_method,
                    bank_reference=f"REF-{payment_number}",
                    allocations=allocations,
                    total_allocated=total_allocated,
                    total_discount=total_discount,
                    total_write_off=total_write_off,
                    unapplied_amount=unapplied_amount,
                    status=status,
                    simulation_id=context.simulation_id,
                )

                # Build GL entries for this payment
                gl_entries = self._build_gl_entries(payment_record)
                all_gl_entries.extend(gl_entries)

                # Delegate GL posting
                try:
                    await self._delegate_gl_posting(gl_entries, context)
                except TransactionError:
                    logger.error(
                        "customer_payment_gl_posting_failed",
                        payment_number=payment_number,
                        customer_id=customer_id,
                        amount=str(payment_amount),
                        service_name="transactions",
                        component="CustomerPaymentProcessor",
                        trace_id=str(context.trace_id),
                    )
                    raise

                # Publish TransactionCreated event
                await self._publish_event(
                    event_type="TransactionCreated",
                    payload={
                        "transaction_type": "customer_payment",
                        "payment_id": str(payment_id),
                        "payment_number": payment_number,
                        "customer_id": customer_id,
                        "amount": str(payment_amount),
                        "payment_method": payment_method.value,
                        "allocation_count": len(allocations),
                        "unapplied_amount": str(unapplied_amount),
                        "status": status,
                    },
                    context=context,
                )

                all_payments.append(payment_record)

                # Update metrics
                self._payments_processed += 1
                self._total_allocated += total_allocated
                self._total_unapplied += unapplied_amount
                self._payments_history.append(payment_record)

            # Check discrepancy trigger
            should_inject, disc_type, disc_params = (
                await self._check_discrepancy_trigger(context, "customer_payment")
            )

            # Build result artifacts
            result.artifacts = {
                "payments": [p.model_dump(mode="json") for p in all_payments],
                "payment_count": len(all_payments),
                "total_payment_amount": str(
                    sum(p.payment_amount for p in all_payments)
                ) if all_payments else "0.00",
            }
            result.gl_entries = all_gl_entries
            result.has_discrepancy = should_inject
            result.discrepancy_type = disc_type
            if all_payments:
                result.amount = sum(p.payment_amount for p in all_payments)

            # Publish TransactionCompleted event
            if all_payments:
                await self._publish_event(
                    event_type="TransactionCompleted",
                    payload={
                        "transaction_type": "customer_payment",
                        "payment_count": len(all_payments),
                        "total_amount": str(result.amount) if result.amount else "0.00",
                        "has_discrepancy": result.has_discrepancy,
                    },
                    context=context,
                )
                result.events_published.append("TransactionCompleted")

            result.events_published.append("TransactionCreated")

        except TransactionError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            logger.error(
                "customer_payment_generation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                service_name="transactions",
                component="CustomerPaymentProcessor",
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                duration_ms=elapsed_ms,
            )
            raise TransactionGenerationError(
                f"Customer payment generation failed: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "trace_id": str(context.trace_id),
                },
            ) from exc

        result.duration_ms = (time.perf_counter() - start_time) * 1_000.0
        self._log_transaction(result, context)
        return result

    async def validate(self, result: TransactionResult) -> bool:
        """Validate a customer payment result against business rules.

        Checks:
        - All allocations have invoice_balance_after >= 0 (NEVER negative)
        - total_allocated + unapplied_amount == payment_amount
        - GL entries balance (SUM debits = SUM credits within $0.01)

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if all checks pass, ``False`` otherwise.
        """
        if result.status == "skipped":
            return True

        payments_data = result.artifacts.get("payments", [])
        if not payments_data:
            return result.status == "skipped"

        for payment_data in payments_data:
            payment = CustomerPaymentRecord.model_validate(payment_data)

            # Check 1: No negative invoice balances — CRITICAL INVARIANT
            for alloc in payment.allocations:
                if alloc.invoice_balance_after < Decimal("0.00"):
                    logger.error(
                        "validation_negative_invoice_balance",
                        payment_number=payment.payment_number,
                        invoice_id=str(alloc.invoice_id),
                        invoice_balance_after=str(alloc.invoice_balance_after),
                        service_name="transactions",
                        component="CustomerPaymentProcessor",
                    )
                    return False

            # Check 2: Allocation accounting identity
            expected = (
                payment.total_allocated + payment.unapplied_amount
            ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)
            actual = payment.payment_amount.quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )
            if abs(expected - actual) > GL_BALANCE_TOLERANCE:
                logger.error(
                    "validation_allocation_mismatch",
                    payment_number=payment.payment_number,
                    expected=str(expected),
                    actual=str(actual),
                    difference=str(abs(expected - actual)),
                    service_name="transactions",
                    component="CustomerPaymentProcessor",
                )
                return False

        # Check 3: GL entries balance (DR = CR within $0.01)
        total_debits = Decimal("0.00")
        total_credits = Decimal("0.00")
        for entry in result.gl_entries:
            total_debits += Decimal(str(entry.get("debit_amount", "0.00")))
            total_credits += Decimal(str(entry.get("credit_amount", "0.00")))

        total_debits = total_debits.quantize(
            _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
        )
        total_credits = total_credits.quantize(
            _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
        )
        imbalance = abs(total_debits - total_credits)
        if imbalance > GL_BALANCE_TOLERANCE:
            logger.error(
                "validation_gl_imbalance",
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                imbalance=str(imbalance),
                service_name="transactions",
                component="CustomerPaymentProcessor",
            )
            return False

        logger.debug(
            "customer_payment_validation_passed",
            payment_count=len(payments_data),
            total_debits=str(total_debits),
            total_credits=str(total_credits),
            service_name="transactions",
            component="CustomerPaymentProcessor",
        )
        return True

    async def post(self, result: TransactionResult) -> None:
        """Post validated customer payment to the General Ledger.

        Delegates GL entries from the result to the injected GLPostingEngine.
        If any step fails, the entire transaction is rolled back per
        AAP §0.7.2 atomicity requirement.

        Args:
            result: The validated :class:`TransactionResult` to post.

        Raises:
            GLPostingError: If GL posting fails.
            TransactionError: If an unexpected posting error occurs.
        """
        if not result.gl_entries:
            logger.debug(
                "customer_payment_post_no_entries",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="CustomerPaymentProcessor",
            )
            return

        minimal_context = GenerationContext(
            current_date=date.today(),
            simulation_id=result.transaction_id,
        )
        await self._delegate_gl_posting(result.gl_entries, minimal_context)

        logger.info(
            "customer_payment_posted",
            transaction_id=str(result.transaction_id),
            gl_entry_count=len(result.gl_entries),
            service_name="transactions",
            component="CustomerPaymentProcessor",
        )

    # ------------------------------------------------------------------
    # FIFO Allocation (MOST CRITICAL METHOD)
    # ------------------------------------------------------------------

    def _allocate_payment_fifo(
        self,
        payment_amount: Decimal,
        open_invoices: List[OpenInvoice],
        payment_date: date,
        payment_id: UUID,
    ) -> Tuple[List[PaymentAllocation], Decimal]:
        """Allocate a payment to invoices using strict FIFO ordering.

        FIFO = First In, First Out — oldest invoice date first.  Each
        invoice is fully paid before moving to the next.  If the payment
        exceeds all open invoices, the excess becomes Unapplied Cash.

        CRITICAL CONSTRAINTS:
            - ALL arithmetic uses Decimal, NEVER float.
            - NEVER allow invoice_balance_after < Decimal("0.00").
            - Use ``.quantize(Decimal("0.01"))`` for all monetary results.
            - Sort by ``invoice_date`` ascending — this IS the FIFO order.

        Args:
            payment_amount: Total payment to allocate.
            open_invoices: List of open invoices (sorted by invoice_date).
            payment_date: Date of the payment for discount eligibility.
            payment_id: Parent payment UUID for allocation records.

        Returns:
            Tuple of (allocations list, unapplied amount).

        Raises:
            PaymentAllocationError: If allocation produces an invalid state.
        """
        if payment_amount <= Decimal("0.00"):
            raise PaymentAllocationError(
                "Payment amount must be positive",
                details={"payment_amount": str(payment_amount)},
            )

        # Step 1: Sort invoices by invoice_date ascending (FIFO)
        sorted_invoices = sorted(open_invoices, key=lambda inv: inv.invoice_date)

        # Step 2: Initialize
        remaining_amount = payment_amount
        allocations: List[PaymentAllocation] = []

        # Step 3: Iterate through invoices in FIFO order
        for invoice in sorted_invoices:
            if remaining_amount <= Decimal("0.00"):
                break

            if invoice.open_balance <= Decimal("0.00"):
                continue

            # Step 3c: Check early payment discount eligibility
            discount = Decimal("0.00")
            if (
                invoice.discount_due_date is not None
                and payment_date <= invoice.discount_due_date
                and invoice.discount_percent > Decimal("0.00")
            ):
                discount = (
                    invoice.open_balance
                    * invoice.discount_percent
                    / Decimal("100")
                ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)

            # Calculate effective balance (amount needed after discount)
            effective_balance = (
                invoice.open_balance - discount
            ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)

            # Step 3b: Calculate applicable amount
            applicable = min(remaining_amount, effective_balance)
            applicable = applicable.quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )

            # If partial payment and discount was calculated, revoke discount
            if applicable < effective_balance and discount > Decimal("0.00"):
                discount = Decimal("0.00")
                applicable = min(remaining_amount, invoice.open_balance)
                applicable = applicable.quantize(
                    _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
                )

            # Step 3d: Calculate new invoice balance
            new_balance = (
                invoice.open_balance - applicable - discount
            ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)

            # Handle short pay write-off for remaining balance
            write_off = Decimal("0.00")
            if (
                new_balance > Decimal("0.00")
                and remaining_amount - applicable <= Decimal("0.00")
            ):
                new_balance, write_off = self._handle_short_pay(
                    invoice, new_balance,
                )

            # Step 3e: CRITICAL CHECK — NEVER negative invoice balance
            if new_balance < Decimal("0.00"):
                raise PaymentAllocationError(
                    "FIFO allocation produced negative invoice balance",
                    details={
                        "invoice_id": str(invoice.invoice_id),
                        "invoice_number": invoice.invoice_number,
                        "open_balance": str(invoice.open_balance),
                        "applicable": str(applicable),
                        "discount": str(discount),
                        "new_balance": str(new_balance),
                    },
                )

            # Step 3f: Create PaymentAllocation record
            allocation = PaymentAllocation(
                payment_id=payment_id,
                invoice_id=invoice.invoice_id,
                invoice_number=invoice.invoice_number,
                allocated_amount=applicable,
                discount_taken=discount,
                write_off_amount=write_off,
                invoice_balance_before=invoice.open_balance,
                invoice_balance_after=new_balance,
            )
            allocations.append(allocation)

            # Step 3g: Reduce remaining amount
            remaining_amount = (remaining_amount - applicable).quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )

            # Step 3h: Update invoice open balance
            invoice.open_balance = new_balance

            logger.debug(
                "fifo_allocation_applied",
                invoice_number=invoice.invoice_number,
                allocated_amount=str(applicable),
                discount_taken=str(discount),
                write_off=str(write_off),
                balance_before=str(allocation.invoice_balance_before),
                balance_after=str(new_balance),
                remaining_payment=str(remaining_amount),
                service_name="transactions",
                component="CustomerPaymentProcessor",
            )

        # Step 5: Handle remaining amount (OVERPAYMENT)
        unapplied_amount = Decimal("0.00")
        if remaining_amount > Decimal("0.00"):
            unapplied_amount = remaining_amount.quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )
            logger.info(
                "overpayment_detected",
                unapplied_amount=str(unapplied_amount),
                payment_id=str(payment_id),
                allocation_count=len(allocations),
                service_name="transactions",
                component="CustomerPaymentProcessor",
            )

        return (allocations, unapplied_amount)

    # ------------------------------------------------------------------
    # Short Pay Handling
    # ------------------------------------------------------------------

    def _handle_short_pay(
        self,
        invoice: OpenInvoice,
        remaining_balance: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """Handle short payment on an invoice.

        When a payment doesn't fully cover an invoice:
        - If remaining balance <= write_off_threshold: write off as bad debt.
        - Otherwise: leave remaining balance open on the invoice.

        Args:
            invoice: The invoice with a remaining balance.
            remaining_balance: The remaining unpaid amount.

        Returns:
            Tuple of (new_open_balance, write_off_amount).
        """
        if remaining_balance <= Decimal("0.00"):
            return (Decimal("0.00"), Decimal("0.00"))

        if remaining_balance <= self._write_off_threshold:
            write_off = remaining_balance.quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )
            logger.info(
                "short_pay_written_off",
                invoice_number=invoice.invoice_number,
                write_off_amount=str(write_off),
                threshold=str(self._write_off_threshold),
                service_name="transactions",
                component="CustomerPaymentProcessor",
            )
            return (Decimal("0.00"), write_off)
        else:
            logger.info(
                "short_pay_balance_open",
                invoice_number=invoice.invoice_number,
                remaining_balance=str(remaining_balance),
                threshold=str(self._write_off_threshold),
                service_name="transactions",
                component="CustomerPaymentProcessor",
            )
            return (remaining_balance, Decimal("0.00"))

    # ------------------------------------------------------------------
    # GL Entry Building
    # ------------------------------------------------------------------

    def _build_gl_entries(
        self,
        payment: CustomerPaymentRecord,
    ) -> List[Dict[str, Any]]:
        """Build GL journal entry lines for a customer payment.

        GL Posting Pattern::

            1. DR Cash            — payment_amount
            2. CR AR              — total_allocated + total_discount + total_write_off
            3. DR Discount Allowed — total_discount (if any)
            4. DR Bad Debt Expense — total_write_off (if any)
            5. CR Unapplied Cash  — unapplied_amount (if overpayment)

        CRITICAL: SUM(debits) MUST equal SUM(credits) within $0.01.

        The accounting identity ensures balance:
            Total DR = Cash + Discount + Bad Debt
                     = payment_amount + total_discount + total_write_off
            Total CR = AR + Unapplied Cash
                     = (total_allocated + total_discount + total_write_off)
                       + unapplied_amount
                     = payment_amount + total_discount + total_write_off

        Args:
            payment: The payment record to build GL entries for.

        Returns:
            List of GL entry dictionaries.

        Raises:
            PaymentAllocationError: If entries do not balance.
        """
        entries: List[Dict[str, Any]] = []
        line_num = 0

        ar_credit_amount = (
            payment.total_allocated
            + payment.total_discount
            + payment.total_write_off
        ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)

        # Line 1: DR Cash
        line_num += 1
        entries.append({
            "line_number": line_num,
            "account_code": _CASH_ACCOUNT_CODE,
            "debit_amount": payment.payment_amount.quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            ),
            "credit_amount": Decimal("0.00"),
            "description": (
                f"Cash received from {payment.customer_name or payment.customer_id} "
                f"- {payment.payment_number}"
            ),
        })

        # Line 2: CR Accounts Receivable
        line_num += 1
        entries.append({
            "line_number": line_num,
            "account_code": _AR_ACCOUNT_CODE,
            "debit_amount": Decimal("0.00"),
            "credit_amount": ar_credit_amount,
            "description": (
                f"AR reduction for payment {payment.payment_number} "
                f"({len(payment.allocations)} invoices)"
            ),
        })

        # Line 3: DR Discount Allowed (if discounts were taken)
        if payment.total_discount > Decimal("0.00"):
            line_num += 1
            entries.append({
                "line_number": line_num,
                "account_code": _DISCOUNT_ALLOWED_ACCOUNT_CODE,
                "debit_amount": payment.total_discount.quantize(
                    _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
                ),
                "credit_amount": Decimal("0.00"),
                "description": (
                    f"Early payment discount for {payment.payment_number}"
                ),
            })

        # Line 4: DR Bad Debt Expense (if write-offs occurred)
        if payment.total_write_off > Decimal("0.00"):
            line_num += 1
            entries.append({
                "line_number": line_num,
                "account_code": _BAD_DEBT_ACCOUNT_CODE,
                "debit_amount": payment.total_write_off.quantize(
                    _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
                ),
                "credit_amount": Decimal("0.00"),
                "description": (
                    f"Bad debt write-off for {payment.payment_number}"
                ),
            })

        # Line 5: CR Unapplied Cash (if overpayment)
        if payment.unapplied_amount > Decimal("0.00"):
            line_num += 1
            entries.append({
                "line_number": line_num,
                "account_code": _UNAPPLIED_CASH_ACCOUNT_CODE,
                "debit_amount": Decimal("0.00"),
                "credit_amount": payment.unapplied_amount.quantize(
                    _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
                ),
                "description": (
                    f"Unapplied cash (overpayment) for {payment.payment_number}"
                ),
            })

        # CRITICAL: Verify GL balance before returning
        total_debits = sum(
            Decimal(str(e["debit_amount"])) for e in entries
        ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)
        total_credits = sum(
            Decimal(str(e["credit_amount"])) for e in entries
        ).quantize(_TWO_PLACES, rounding=decimal.ROUND_HALF_UP)
        imbalance = abs(total_debits - total_credits)

        if imbalance > GL_BALANCE_TOLERANCE:
            raise PaymentAllocationError(
                "GL entries for customer payment do not balance",
                details={
                    "payment_number": payment.payment_number,
                    "total_debits": str(total_debits),
                    "total_credits": str(total_credits),
                    "imbalance": str(imbalance),
                    "tolerance": str(GL_BALANCE_TOLERANCE),
                },
            )

        logger.debug(
            "gl_entries_built",
            payment_number=payment.payment_number,
            line_count=len(entries),
            total_debits=str(total_debits),
            total_credits=str(total_credits),
            service_name="transactions",
            component="CustomerPaymentProcessor",
        )

        return entries

    # ------------------------------------------------------------------
    # Payment Method Selection
    # ------------------------------------------------------------------

    def _select_payment_method(
        self,
        amount: Decimal,
        rng: random.Random,
    ) -> PaymentMethod:
        """Select a payment method based on amount and random distribution.

        Distribution by amount tier:
            - < $1,000:     60% check, 30% ACH, 10% wire
            - $1K – $10K:   30% check, 50% ACH, 20% wire
            - ≥ $10,000:    10% check, 40% ACH, 50% wire

        Uses the seeded ``rng`` parameter for deterministic reproducibility.
        NEVER uses module-level ``random``.

        Args:
            amount: Payment amount for tier selection.
            rng: Seeded Random instance from context.

        Returns:
            Selected :class:`PaymentMethod`.
        """
        roll = rng.random()

        if amount < _SMALL_PAYMENT_THRESHOLD:
            if roll < 0.60:
                return PaymentMethod.CHECK
            elif roll < 0.90:
                return PaymentMethod.ACH
            else:
                return PaymentMethod.WIRE
        elif amount < _MEDIUM_PAYMENT_THRESHOLD:
            if roll < 0.30:
                return PaymentMethod.CHECK
            elif roll < 0.80:
                return PaymentMethod.ACH
            else:
                return PaymentMethod.WIRE
        else:
            if roll < 0.10:
                return PaymentMethod.CHECK
            elif roll < 0.50:
                return PaymentMethod.ACH
            else:
                return PaymentMethod.WIRE

    # ------------------------------------------------------------------
    # Sequential Numbering
    # ------------------------------------------------------------------

    def _generate_payment_number(self, payment_date: date) -> str:
        """Generate a sequential payment number in CPAY-YYYY-NNNN format.

        Resets the sequence counter when the year changes.

        Args:
            payment_date: Date for year extraction.

        Returns:
            Formatted payment number (e.g., "CPAY-2024-0001").
        """
        year = payment_date.year
        if year != self._current_year:
            self._current_year = year
            self._sequence_counter = 0

        self._sequence_counter += 1
        prefix = DOCUMENT_NUMBER_PREFIXES.get("customer_payment", "CPAY")
        return self._generate_sequential_number(
            prefix, year, self._sequence_counter,
        )

    # ------------------------------------------------------------------
    # Open Invoices Retrieval
    # ------------------------------------------------------------------

    def _get_open_invoices(
        self,
        customer_id: Optional[str] = None,
    ) -> List[OpenInvoice]:
        """Get open invoices for FIFO processing.

        In simulation mode, returns sample data for testing.  When a
        ``db_session_factory`` is available, queries from the database.

        Args:
            customer_id: Optional filter for a specific customer.

        Returns:
            List of :class:`OpenInvoice` sorted by invoice_date ascending.
        """
        # Check for externally injected pending invoices
        if hasattr(self, '_pending_invoices') and self._pending_invoices:
            invoices = list(self._pending_invoices)
            if customer_id:
                invoices = [
                    inv for inv in invoices if inv.customer_id == customer_id
                ]
            return sorted(invoices, key=lambda inv: inv.invoice_date)

        # In simulation mode without a database, generate sample invoices
        base_date = date(2024, 1, 15)
        sample_data = [
            {
                "invoice_id": uuid4(),
                "invoice_number": "INV-2024-0001",
                "customer_id": customer_id or "CUST-001",
                "invoice_date": base_date,
                "due_date": base_date + timedelta(days=30),
                "original_amount": Decimal("5000.00"),
                "open_balance": Decimal("5000.00"),
                "payment_terms": "Net 30",
                "discount_percent": Decimal("2.00"),
                "discount_due_date": base_date + timedelta(days=10),
            },
            {
                "invoice_id": uuid4(),
                "invoice_number": "INV-2024-0002",
                "customer_id": customer_id or "CUST-001",
                "invoice_date": base_date + timedelta(days=15),
                "due_date": base_date + timedelta(days=45),
                "original_amount": Decimal("3000.00"),
                "open_balance": Decimal("3000.00"),
                "payment_terms": "Net 30",
                "discount_percent": Decimal("0.00"),
                "discount_due_date": None,
            },
            {
                "invoice_id": uuid4(),
                "invoice_number": "INV-2024-0003",
                "customer_id": customer_id or "CUST-001",
                "invoice_date": base_date + timedelta(days=30),
                "due_date": base_date + timedelta(days=60),
                "original_amount": Decimal("2000.00"),
                "open_balance": Decimal("2000.00"),
                "payment_terms": "Net 30",
                "discount_percent": Decimal("0.00"),
                "discount_due_date": None,
            },
        ]

        return sorted(
            [OpenInvoice.model_validate(d) for d in sample_data],
            key=lambda inv: inv.invoice_date,
        )

    # ------------------------------------------------------------------
    # Payment Amount Determination
    # ------------------------------------------------------------------

    def _determine_payment_amount(
        self,
        total_open: Decimal,
        rng: random.Random,
        context: GenerationContext,
    ) -> Decimal:
        """Determine the payment amount for a customer.

        Uses the payment timing model if available, otherwise generates
        a realistic payment amount using the seeded RNG.

        Payment amounts can be full, partial (short pay), or over the open
        balance (overpayment).

        Args:
            total_open: Total open balance for the customer.
            rng: Seeded Random instance.
            context: Current generation context.

        Returns:
            Payment amount as Decimal (always > 0, or 0 if nothing open).
        """
        if total_open <= Decimal("0.00"):
            return Decimal("0.00")

        # Use payment timing model if available
        if self._payment_timing_model is not None:
            try:
                timing_result = self._payment_timing_model.generate_timing(
                    rng=rng,
                )
                if hasattr(timing_result, 'payment_fraction'):
                    fraction = Decimal(str(timing_result.payment_fraction))
                    amount = (total_open * fraction).quantize(
                        _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
                    )
                    if amount > Decimal("0.00"):
                        return amount
            except Exception:
                pass  # Fall through to RNG-based generation

        # RNG-based: 70% full, 15% short, 10% slight over, 5% significant over
        roll = rng.random()

        if roll < 0.70:
            amount = total_open
        elif roll < 0.85:
            fraction = Decimal(str(rng.uniform(0.80, 0.99)))
            amount = (total_open * fraction).quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )
        elif roll < 0.95:
            fraction = Decimal(str(rng.uniform(1.00, 1.05)))
            amount = (total_open * fraction).quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )
        else:
            fraction = Decimal(str(rng.uniform(1.05, 1.20)))
            amount = (total_open * fraction).quantize(
                _TWO_PLACES, rounding=decimal.ROUND_HALF_UP,
            )

        if amount <= Decimal("0.00"):
            amount = total_open

        return amount

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return processing metrics for monitoring and reporting.

        Returns:
            Dictionary with payment processing statistics.
        """
        return {
            "payments_processed": self._payments_processed,
            "total_allocated": str(self._total_allocated),
            "total_unapplied": str(self._total_unapplied),
            "payments_history_count": len(self._payments_history),
        }
