"""Customer Invoice Generator — creates invoices from shipped orders.

Generates customer invoices as the third step in the Order-to-Cash cycle.
Invoices are created from SHIPMENT data (not order data), meaning:

**CRITICAL**: Invoice amounts are calculated from **shipment quantities**,
NOT from order quantities. If a sales order was partially shipped, the
invoice reflects only what was actually shipped.

Key Behaviors:

1. **Invoice from Shipment**: Links invoice to the shipment record and
   the originating sales order. Quantities and amounts come from the
   shipment lines, NOT the SO lines.

2. **Payment Terms**: Applied from the customer master data record.
   Common terms: Net 30, Net 60, 2/10 Net 30 (2% discount if paid
   within 10 days, otherwise full amount in 30 days).

3. **Due Date Calculation**: ``invoice_date + payment_term_days``.
   For discount terms (e.g., 2/10 Net 30), both a discount due date
   and a net due date are calculated.

4. **GL Posting**:
   - DR Accounts Receivable (invoice total)
   - CR Revenue (invoice total)

5. **Agent Routing**: Invoices are routed to 'ar_clerk' agents via the
   WorkflowOrchestrator's ROLE_MAPPING
   (``"customer_invoice": ["ar_clerk"]``).

6. **Sequential Numbering**: INV-YYYY-NNNN (e.g., INV-2024-0001)

7. **Required Artifacts** (from REQUIRED_ARTIFACTS mapping):
   - invoice_header
   - invoice_lines
   - gl_entries

Database Pattern: ``from synthetic_erp.db.session import get_session``

Retry Policy (AAP §0.1.2 — o2c_cycle_generation):
    - max_attempts: 2, backoff: linear (1s, 2s), timeout: 60s, fallback: skip

Performance Target: O2C ≥ 60 cycles/min (AAP §0.7.3)

References:
    - AAP §0.5.1 Group 3: O2C Engine (CustomerInvoiceGenerator)
    - AAP §0.1.2: Invoice amounts from SHIPMENT quantities, NOT order quantities
    - AAP §0.7.2: Financial Integrity Rules (Decimal, balance validation)
"""

from __future__ import annotations

import asyncio
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
from app.transactions.exceptions import (
    GLPostingError,
    TransactionError,
    TransactionGenerationError,
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
    from app.statistical.amount_distributions import AmountDistribution
    from app.statistical.payment_timing_model import PaymentTimingModel
    from app.transactions.gl.gl_posting_engine import GLPostingEngine

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
# Pre-compiled regex for parsing payment terms
# Matches patterns like "2/10 Net 30", "1/15 Net 45", etc.
# ---------------------------------------------------------------------------
_DISCOUNT_TERMS_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*/\s*(\d+)\s+[Nn]et\s+(\d+)$"
)
_NET_TERMS_RE = re.compile(r"^[Nn]et\s+(\d+)$")

# ---------------------------------------------------------------------------
# Default GL account codes for customer invoice postings
# ---------------------------------------------------------------------------
_AR_ACCOUNT_CODE = "1200"
_REVENUE_ACCOUNT_CODE = "4000"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════


class InvoiceLine(BaseModel):
    """A single line item on a customer invoice.

    All monetary fields use :class:`~decimal.Decimal` — never ``float``.
    The ``shipped_quantity`` originates from the shipment record, ensuring
    that invoice amounts reflect what was actually shipped rather than what
    was ordered.

    Attributes:
        line_number: 1-based line number within the invoice.
        product_id: Product identifier from the product master.
        product_name: Human-readable product name.
        shipped_quantity: Quantity from SHIPMENT — **not** order quantity.
        unit_price: Unit selling price (Decimal).
        line_total: ``shipped_quantity * unit_price`` before discounts.
        discount_percent: Percentage discount applied to this line.
        discount_amount: Computed discount amount.
        net_amount: ``line_total - discount_amount``.
        shipment_line_reference: Reference to the originating shipment line.
        sales_order_line_reference: Reference to the originating SO line.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_number: int = Field(..., ge=1, description="1-based line number")
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(default="", description="Product name")
    shipped_quantity: Decimal = Field(
        ...,
        gt=Decimal("0.00"),
        description="Quantity from SHIPMENT, not order",
    )
    unit_price: Decimal = Field(
        ...,
        gt=Decimal("0.00"),
        description="Unit selling price",
    )
    line_total: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="shipped_quantity * unit_price",
    )
    discount_percent: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Percentage discount on this line",
    )
    discount_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Computed discount amount",
    )
    net_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="line_total - discount_amount",
    )
    shipment_line_reference: Optional[str] = Field(
        default=None, description="Reference to shipment line"
    )
    sales_order_line_reference: Optional[str] = Field(
        default=None, description="Reference to SO line"
    )


class CustomerInvoiceRecord(BaseModel):
    """Complete customer invoice header with lines.

    Represents a single customer invoice generated from a shipped order.
    The ``open_balance`` starts equal to ``invoice_total`` and is reduced
    as payments are applied by the downstream
    :class:`~app.transactions.o2c.customer_payment_processor.CustomerPaymentProcessor`.

    All monetary fields use :class:`~decimal.Decimal`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    invoice_id: UUID = Field(
        default_factory=uuid4, description="Unique invoice identifier"
    )
    invoice_number: str = Field(
        default="", description="Sequential number INV-YYYY-NNNN"
    )
    customer_id: str = Field(..., description="Customer identifier")
    customer_name: str = Field(default="", description="Customer name")
    sales_order_id: Optional[UUID] = Field(
        default=None, description="Originating sales order"
    )
    sales_order_number: str = Field(
        default="", description="SO number reference"
    )
    shipment_id: Optional[UUID] = Field(
        default=None, description="Source shipment"
    )
    shipment_number: str = Field(
        default="", description="Shipment number reference"
    )
    invoice_date: date = Field(..., description="Invoice issue date")
    payment_terms: str = Field(
        default="Net 30",
        description="Payment terms e.g. 'Net 30', '2/10 Net 30'",
    )
    payment_term_days: int = Field(
        default=30, ge=0, description="Net payment term in days"
    )
    due_date: date = Field(..., description="Net payment due date")
    discount_due_date: Optional[date] = Field(
        default=None,
        description="Early payment discount deadline (e.g. 2/10 Net 30)",
    )
    discount_percent: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Early payment discount percentage",
    )
    lines: List[InvoiceLine] = Field(
        default_factory=list, description="Invoice line items"
    )
    subtotal: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="SUM of line_total across all lines",
    )
    total_discount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="SUM of discount_amount across all lines",
    )
    invoice_total: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="subtotal - total_discount",
    )
    open_balance: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Remaining open balance, starts = invoice_total",
    )
    status: str = Field(
        default="draft",
        description="draft, posted, partially_paid, paid, cancelled",
    )
    simulation_id: Optional[UUID] = Field(
        default=None, description="Parent simulation run identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation timestamp",
    )


class ShippedOrderInfo(BaseModel):
    """Information about a shipped order ready for invoicing.

    This is the input data model consumed by the
    :class:`CustomerInvoiceGenerator` to create a customer invoice.
    The ``shipped_lines`` carry product-level detail from the shipment
    record, **not** from the sales order.

    Attributes:
        sales_order_id: UUID of the originating sales order.
        sales_order_number: Human-readable SO number.
        shipment_id: UUID of the shipment that fulfilled this order.
        shipment_number: Human-readable shipment number.
        customer_id: Customer identifier.
        customer_name: Customer name.
        ship_date: Date the shipment was dispatched.
        payment_terms: Payment terms from the customer master.
        shipped_lines: List of dicts, each with keys ``product_id``,
            ``product_name``, ``quantity`` (shipped), ``unit_price``,
            and optionally ``discount_percent``, ``shipment_line_ref``,
            ``so_line_ref``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sales_order_id: UUID = Field(
        ..., description="Originating sales order UUID"
    )
    sales_order_number: str = Field(default="", description="SO number")
    shipment_id: UUID = Field(..., description="Source shipment UUID")
    shipment_number: str = Field(default="", description="Shipment number")
    customer_id: str = Field(..., description="Customer identifier")
    customer_name: str = Field(default="", description="Customer name")
    ship_date: date = Field(..., description="Shipment dispatch date")
    payment_terms: str = Field(
        default="Net 30", description="Customer payment terms"
    )
    shipped_lines: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Shipment line details: product_id, product_name, "
            "quantity, unit_price, discount_percent (optional)"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CustomerInvoiceGenerator
# ═══════════════════════════════════════════════════════════════════════════


class CustomerInvoiceGenerator(TransactionGenerator):
    """Generates customer invoices from shipped orders.

    **CRITICAL**: Invoice amounts are calculated from **SHIPMENT** quantities,
    NOT from order quantities.  This is the third step in the O2C cycle
    (SalesOrder → Shipment → **CustomerInvoice** → CustomerPayment).

    Constructor Injection (ADR-003):
        All dependencies are ``Optional`` with ``None`` defaults, enabling
        test isolation and incremental integration.

    Agent Routing:
        Invoices are routed to ``ar_clerk`` agents via
        ``WorkflowOrchestrator.ROLE_MAPPING["customer_invoice"]``.

    GL Posting:
        DR Accounts Receivable — invoice total
        CR Revenue             — invoice total

    Required Artifacts (per ``REQUIRED_ARTIFACTS``):
        ``invoice_header``, ``invoice_lines``, ``gl_entries``

    Sequential Numbering:
        ``INV-YYYY-NNNN`` (e.g. ``INV-2024-0001``)
    """

    # ------------------------------------------------------------------
    # Construction (ADR-003 — constructor injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        db_session_factory: Optional[Any] = None,
        agent_registry: Optional[AgentRegistry] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
        event_bus: Optional[EventBus] = None,
        gl_posting_engine: Optional[GLPostingEngine] = None,
        payment_timing_model: Optional[PaymentTimingModel] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        discrepancy_injector: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the CustomerInvoiceGenerator.

        Args:
            db_session_factory: Callable returning async context manager
                yielding a database session (``get_session``).
            agent_registry: Project 2 :class:`AgentRegistry` for agent
                lookup during invoice routing.
            workflow_orchestrator: Project 2 :class:`WorkflowOrchestrator`
                for routing invoices to ``ar_clerk`` agents.
            event_bus: Project 2 :class:`EventBus` for ``TransactionCreated``
                / ``TransactionCompleted`` event publication.
            gl_posting_engine: P3 :class:`GLPostingEngine` for journal entry
                creation (DR AR, CR Revenue) with balance validation.
            payment_timing_model: Statistical model for payment term analysis.
            amount_distribution: Statistical model for amount generation.
            discrepancy_injector: P3 ``DiscrepancyInjector`` for rate-based
                discrepancy injection into invoices.
            statistical_models: Additional statistical model instances.
        """
        super().__init__(
            db_session_factory=db_session_factory,
            agent_registry=agent_registry,
            workflow_orchestrator=workflow_orchestrator,
            event_bus=event_bus,
            gl_posting_engine=gl_posting_engine,
            discrepancy_injector=discrepancy_injector,
            statistical_models=statistical_models,
        )

        # Additional domain-specific dependencies
        self._payment_timing_model = payment_timing_model
        self._amount_distribution = amount_distribution

        # Sequential numbering state
        self._sequence_counter: int = 0
        self._current_year: int = 0

        # Metrics counters
        self._invoices_generated: int = 0
        self._total_invoice_amount: Decimal = Decimal("0.00")
        self._invoice_history: List[CustomerInvoiceRecord] = []

        logger.info(
            "customer_invoice_generator_initialized",
            service_name="transactions",
            component="CustomerInvoiceGenerator",
            has_db_session=db_session_factory is not None,
            has_agent_registry=agent_registry is not None,
            has_workflow_orchestrator=workflow_orchestrator is not None,
            has_event_bus=event_bus is not None,
            has_gl_engine=gl_posting_engine is not None,
            has_payment_timing=payment_timing_model is not None,
            has_amount_distribution=amount_distribution is not None,
            has_discrepancy_injector=discrepancy_injector is not None,
        )

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(
        self, context: GenerationContext
    ) -> TransactionResult:
        """Generate customer invoices from shipped orders.

        For each shipped order in the current context:
        1. Build invoice lines from **SHIPMENT** lines (not order lines).
        2. Calculate line-level totals, discounts, and net amounts.
        3. Parse payment terms and compute due date / discount due date.
        4. Generate sequential invoice number (INV-YYYY-NNNN).
        5. Build GL entries (DR AR, CR Revenue).
        6. Delegate GL posting to the injected ``GLPostingEngine``.
        7. Publish ``TransactionCreated`` event.
        8. Check for discrepancy injection.

        The entire generation is wrapped with ``asyncio.wait_for`` using
        the O2C cycle timeout from ``RETRY_POLICIES`` (60s default).

        Args:
            context: Generation parameters including business date, fiscal
                period, RNG seed, and discrepancy configuration.

        Returns:
            :class:`TransactionResult` carrying invoice artifacts, GL entries,
            and metadata.

        Raises:
            TransactionGenerationError: If invoice generation fails after
                all configured retry attempts or times out.
        """
        # Apply timeout from retry policy (AAP §0.1.2 — 60s for O2C)
        retry_cfg = RETRY_POLICIES.get("o2c_cycle_generation", {})
        timeout_seconds = retry_cfg.get("timeout_seconds", 60)
        try:
            return await asyncio.wait_for(
                self._generate_invoices(context),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise TransactionGenerationError(
                f"Customer invoice generation timed out after {timeout_seconds}s",
                details={
                    "timeout_seconds": timeout_seconds,
                    "trace_id": str(context.trace_id),
                    "simulation_id": str(context.simulation_id),
                },
            )

    async def _generate_invoices(
        self, context: GenerationContext
    ) -> TransactionResult:
        """Internal invoice generation logic (wrapped by timeout).

        Separated from :meth:`generate` to enable ``asyncio.wait_for``
        timeout wrapping at the entry point.

        Args:
            context: Generation parameters.

        Returns:
            :class:`TransactionResult` carrying invoice artifacts.
        """
        start_time = time.perf_counter()
        rng = self._create_seeded_rng(context)

        # Retrieve shipped orders ready for invoicing
        shipped_orders = self._get_shipped_orders(context)

        all_invoices: List[CustomerInvoiceRecord] = []
        all_gl_entries: List[Dict[str, Any]] = []
        events_published: List[str] = []
        has_discrepancy = False
        discrepancy_type: Optional[str] = None

        for shipped_order in shipped_orders:
            try:
                # Create the invoice from shipment data (CRITICAL: shipment
                # quantities, NOT order quantities)
                invoice = self._create_invoice_from_shipment(
                    shipped_order=shipped_order,
                    invoice_date=context.current_date,
                    rng=rng,
                )
                invoice.simulation_id = context.simulation_id

                # Build GL entries for this invoice
                gl_entries = self._build_gl_entries(invoice)

                # Delegate GL posting
                try:
                    await self._delegate_gl_posting(gl_entries, context)
                    invoice.status = "posted"
                except TransactionError as gl_exc:
                    logger.error(
                        "invoice_gl_posting_failed",
                        service_name="transactions",
                        component="CustomerInvoiceGenerator",
                        invoice_number=invoice.invoice_number,
                        error=str(gl_exc),
                        trace_id=str(context.trace_id),
                        simulation_id=str(context.simulation_id),
                    )
                    invoice.status = "draft"

                # Publish TransactionCreated event
                await self._publish_event(
                    event_type="TransactionCreated",
                    payload={
                        "transaction_type": "customer_invoice",
                        "invoice_id": str(invoice.invoice_id),
                        "invoice_number": invoice.invoice_number,
                        "customer_id": invoice.customer_id,
                        "customer_name": invoice.customer_name,
                        "invoice_total": str(invoice.invoice_total),
                        "sales_order_id": str(invoice.sales_order_id)
                        if invoice.sales_order_id
                        else None,
                        "shipment_id": str(invoice.shipment_id)
                        if invoice.shipment_id
                        else None,
                    },
                    context=context,
                )
                events_published.append("TransactionCreated")

                all_invoices.append(invoice)
                all_gl_entries.extend(gl_entries)

                # Update metrics
                self._invoices_generated += 1
                self._total_invoice_amount += invoice.invoice_total
                self._invoice_history.append(invoice)

            except Exception as exc:
                logger.error(
                    "invoice_generation_error",
                    service_name="transactions",
                    component="CustomerInvoiceGenerator",
                    shipped_order_id=str(shipped_order.shipment_id),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    trace_id=str(context.trace_id),
                    simulation_id=str(context.simulation_id),
                )
                continue

        # Check discrepancy injection trigger
        should_inject, disc_type, disc_params = (
            await self._check_discrepancy_trigger(
                context, "customer_invoice"
            )
        )
        if should_inject:
            has_discrepancy = True
            discrepancy_type = disc_type

        # Compute total amount across all invoices
        total_amount = sum(
            (inv.invoice_total for inv in all_invoices), Decimal("0.00")
        )

        # Build TransactionResult with required artifacts
        result = TransactionResult(
            transaction_type="customer_invoice",
            status="completed" if all_invoices else "skipped",
            artifacts={
                "invoice_header": [
                    inv.model_dump(mode="json") for inv in all_invoices
                ],
                "invoice_lines": [
                    [line.model_dump(mode="json") for line in inv.lines]
                    for inv in all_invoices
                ],
                "gl_entries": all_gl_entries,
            },
            gl_entries=all_gl_entries,
            events_published=events_published,
            has_discrepancy=has_discrepancy,
            discrepancy_type=discrepancy_type,
            amount=total_amount if total_amount > Decimal("0.00") else None,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
        result.duration_ms = elapsed_ms

        # Log the completed transaction
        self._log_transaction(result, context)

        return result

    async def validate(self, result: TransactionResult) -> bool:
        """Validate generated customer invoice(s).

        Checks:
        - At least one invoice was generated (or result is ``skipped``).
        - Each invoice total is > ``Decimal("0.00")``.
        - All shipped quantities are positive Decimals.
        - GL entries balance: SUM(debits) = SUM(credits) within $0.01.
        - Each invoice references a valid customer.

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if all validations pass, ``False`` otherwise.
        """
        if result.status == "skipped":
            return True

        if result.status == "failed":
            return False

        # Validate GL entries balance using financial tolerances
        # GL_BALANCE_TOLERANCE is the specific $0.01 tolerance, also
        # available via FINANCIAL_TOLERANCES["gl_balance_tolerance"]
        balance_tolerance = FINANCIAL_TOLERANCES.get(
            "gl_balance_tolerance", GL_BALANCE_TOLERANCE
        )

        gl_entries = result.gl_entries
        if gl_entries:
            total_debits = sum(
                (
                    Decimal(str(entry.get("debit_amount", "0.00")))
                    for entry in gl_entries
                ),
                Decimal("0.00"),
            )
            total_credits = sum(
                (
                    Decimal(str(entry.get("credit_amount", "0.00")))
                    for entry in gl_entries
                ),
                Decimal("0.00"),
            )
            imbalance = abs(total_debits - total_credits)
            if imbalance > balance_tolerance:
                logger.warning(
                    "invoice_gl_balance_validation_failed",
                    service_name="transactions",
                    component="CustomerInvoiceGenerator",
                    total_debits=str(total_debits),
                    total_credits=str(total_credits),
                    imbalance=str(imbalance),
                    tolerance=str(GL_BALANCE_TOLERANCE),
                )
                return False

        # Validate invoice artifacts
        invoice_headers = result.artifacts.get("invoice_header", [])
        if not invoice_headers:
            logger.warning(
                "invoice_validation_no_headers",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
            )
            return False

        for header in invoice_headers:
            invoice_total = Decimal(str(header.get("invoice_total", "0.00")))
            if invoice_total <= Decimal("0.00"):
                logger.warning(
                    "invoice_validation_zero_total",
                    service_name="transactions",
                    component="CustomerInvoiceGenerator",
                    invoice_number=header.get("invoice_number", ""),
                    invoice_total=str(invoice_total),
                )
                return False

            customer_id = header.get("customer_id", "")
            if not customer_id:
                logger.warning(
                    "invoice_validation_missing_customer",
                    service_name="transactions",
                    component="CustomerInvoiceGenerator",
                    invoice_number=header.get("invoice_number", ""),
                )
                return False

        return True

    async def post(self, result: TransactionResult) -> None:
        """Post validated customer invoice GL entries.

        Delegates to the injected :class:`GLPostingEngine` for journal
        entry creation (DR Accounts Receivable, CR Revenue) with balance
        validation.

        If any step fails, the entire transaction MUST be fully rolled
        back (atomicity requirement per AAP §0.7.2).

        Args:
            result: The validated :class:`TransactionResult` to post.

        Raises:
            GLPostingError: If GL posting fails.
            TransactionError: On unexpected posting failures.
        """
        if result.status in ("skipped", "failed"):
            logger.debug(
                "invoice_post_skipped",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
                status=result.status,
            )
            return

        gl_entries = result.gl_entries
        if not gl_entries:
            logger.debug(
                "invoice_post_no_gl_entries",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
                transaction_id=str(result.transaction_id),
            )
            return

        # Build a minimal GenerationContext for posting delegation
        # (the actual context should be passed through in a full pipeline,
        # but post() is designed to work with the result alone)
        from datetime import date as _date_type

        posting_context = GenerationContext(
            current_date=_date_type.today(),
        )

        try:
            await self._delegate_gl_posting(gl_entries, posting_context)
            logger.info(
                "invoice_posted",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
                transaction_id=str(result.transaction_id),
                gl_entry_count=len(gl_entries),
            )
        except TransactionError:
            raise
        except Exception as exc:
            raise GLPostingError(
                f"Invoice GL posting failed: {exc}",
                details={
                    "transaction_id": str(result.transaction_id),
                    "gl_entry_count": len(gl_entries),
                    "error_type": type(exc).__name__,
                },
            ) from exc

    # ------------------------------------------------------------------
    # Invoice Creation Methods
    # ------------------------------------------------------------------

    def _create_invoice_from_shipment(
        self,
        shipped_order: ShippedOrderInfo,
        invoice_date: date,
        rng: random.Random,
    ) -> CustomerInvoiceRecord:
        """Build a customer invoice from shipment data.

        **CRITICAL**: Invoice amounts are calculated from SHIPMENT quantities,
        NOT from order quantities.  Each invoice line's ``shipped_quantity``
        comes directly from the shipment line record.

        Args:
            shipped_order: Shipped order information containing shipment lines.
            invoice_date: Date to assign to the invoice.
            rng: Seeded random number generator for deterministic decisions.

        Returns:
            A fully populated :class:`CustomerInvoiceRecord`.
        """
        # Parse payment terms
        term_days, discount_days, discount_pct = self._parse_payment_terms(
            shipped_order.payment_terms
        )

        # Calculate due date and optional discount due date
        due_date = self._calculate_due_date(invoice_date, term_days)
        discount_due_date: Optional[date] = None
        if discount_days is not None:
            discount_due_date = self._calculate_due_date(
                invoice_date, discount_days
            )

        # Generate invoice number
        invoice_number = self._generate_invoice_number(invoice_date)

        # Build invoice lines from SHIPMENT lines (NOT order lines)
        invoice_lines: List[InvoiceLine] = []
        subtotal = Decimal("0.00")
        total_discount = Decimal("0.00")

        for idx, shipped_line in enumerate(
            shipped_order.shipped_lines, start=1
        ):
            # Extract shipped quantity and unit price from shipment line
            quantity_raw = shipped_line.get("quantity", shipped_line.get("shipped_quantity", 0))
            quantity = Decimal(str(quantity_raw)).quantize(Decimal("0.01"))
            if quantity <= Decimal("0.00"):
                continue

            price_raw = shipped_line.get("unit_price", 0)
            unit_price = Decimal(str(price_raw)).quantize(Decimal("0.01"))
            if unit_price <= Decimal("0.00"):
                continue

            # Line total = shipped_quantity * unit_price (Decimal)
            line_total = (quantity * unit_price).quantize(Decimal("0.01"))

            # Apply line-level discount if specified
            line_discount_pct_raw = shipped_line.get("discount_percent", "0.00")
            line_discount_pct = Decimal(str(line_discount_pct_raw))

            discount_amount = Decimal("0.00")
            if line_discount_pct > Decimal("0.00"):
                discount_amount = (
                    line_total * line_discount_pct / Decimal("100")
                ).quantize(Decimal("0.01"))

            net_amount = (line_total - discount_amount).quantize(
                Decimal("0.01")
            )

            invoice_line = InvoiceLine(
                line_number=idx,
                product_id=str(shipped_line.get("product_id", "")),
                product_name=str(shipped_line.get("product_name", "")),
                shipped_quantity=quantity,
                unit_price=unit_price,
                line_total=line_total,
                discount_percent=line_discount_pct,
                discount_amount=discount_amount,
                net_amount=net_amount,
                shipment_line_reference=shipped_line.get(
                    "shipment_line_ref"
                ),
                sales_order_line_reference=shipped_line.get("so_line_ref"),
            )

            invoice_lines.append(invoice_line)
            subtotal += line_total
            total_discount += discount_amount

        # Compute invoice total and open balance
        invoice_total = (subtotal - total_discount).quantize(Decimal("0.01"))
        open_balance = invoice_total  # Initially full amount is owed

        # Build the invoice record
        invoice = CustomerInvoiceRecord(
            invoice_number=invoice_number,
            customer_id=shipped_order.customer_id,
            customer_name=shipped_order.customer_name,
            sales_order_id=shipped_order.sales_order_id,
            sales_order_number=shipped_order.sales_order_number,
            shipment_id=shipped_order.shipment_id,
            shipment_number=shipped_order.shipment_number,
            invoice_date=invoice_date,
            payment_terms=shipped_order.payment_terms,
            payment_term_days=term_days,
            due_date=due_date,
            discount_due_date=discount_due_date,
            discount_percent=discount_pct,
            lines=invoice_lines,
            subtotal=subtotal.quantize(Decimal("0.01")),
            total_discount=total_discount.quantize(Decimal("0.01")),
            invoice_total=invoice_total,
            open_balance=open_balance,
            status="draft",
        )

        logger.debug(
            "invoice_created_from_shipment",
            service_name="transactions",
            component="CustomerInvoiceGenerator",
            invoice_number=invoice_number,
            customer_id=shipped_order.customer_id,
            shipment_number=shipped_order.shipment_number,
            line_count=len(invoice_lines),
            subtotal=str(subtotal),
            total_discount=str(total_discount),
            invoice_total=str(invoice_total),
            payment_terms=shipped_order.payment_terms,
            term_days=term_days,
        )

        return invoice

    # ------------------------------------------------------------------
    # Payment Terms Parsing
    # ------------------------------------------------------------------

    def _parse_payment_terms(
        self, terms: str
    ) -> Tuple[int, Optional[int], Decimal]:
        """Parse a payment terms string into structured components.

        Supported formats:
            - ``"Net 30"``       → (30, None, Decimal("0.00"))
            - ``"Net 60"``       → (60, None, Decimal("0.00"))
            - ``"2/10 Net 30"``  → (30, 10, Decimal("2.00"))
            - ``"1/10 Net 45"``  → (45, 10, Decimal("1.00"))

        Args:
            terms: Payment terms string from the customer master.

        Returns:
            Tuple of ``(net_days, discount_days, discount_percent)``:
            - ``net_days``: Number of days until full payment is due.
            - ``discount_days``: Number of days for early payment discount
              eligibility (``None`` if no discount).
            - ``discount_percent``: Percentage discount if paid within
              the discount window (``Decimal("0.00")`` if no discount).
        """
        terms = terms.strip()

        # Try discount terms pattern first: "X/Y Net Z"
        discount_match = _DISCOUNT_TERMS_RE.match(terms)
        if discount_match:
            discount_pct = Decimal(discount_match.group(1))
            discount_days = int(discount_match.group(2))
            net_days = int(discount_match.group(3))
            logger.debug(
                "payment_terms_parsed_discount",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
                terms=terms,
                net_days=net_days,
                discount_days=discount_days,
                discount_percent=str(discount_pct),
            )
            return (net_days, discount_days, discount_pct)

        # Try simple Net N pattern
        net_match = _NET_TERMS_RE.match(terms)
        if net_match:
            net_days = int(net_match.group(1))
            logger.debug(
                "payment_terms_parsed_net",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
                terms=terms,
                net_days=net_days,
            )
            return (net_days, None, Decimal("0.00"))

        # Fallback: default to Net 30 for unrecognised terms
        logger.warning(
            "payment_terms_unrecognized_defaulting",
            service_name="transactions",
            component="CustomerInvoiceGenerator",
            terms=terms,
            default="Net 30",
        )
        return (30, None, Decimal("0.00"))

    # ------------------------------------------------------------------
    # Due Date Calculation
    # ------------------------------------------------------------------

    def _calculate_due_date(self, invoice_date: date, term_days: int) -> date:
        """Calculate the due date from invoice date and term days.

        Args:
            invoice_date: The invoice issue date.
            term_days: Number of days until payment is due.

        Returns:
            The computed due date.
        """
        return invoice_date + timedelta(days=term_days)

    # ------------------------------------------------------------------
    # GL Entry Building
    # ------------------------------------------------------------------

    def _build_gl_entries(
        self, invoice: CustomerInvoiceRecord
    ) -> List[Dict[str, Any]]:
        """Build balanced GL journal entries for a customer invoice.

        Creates a two-line journal entry:
            Line 1: DR Accounts Receivable — ``invoice.invoice_total``
            Line 2: CR Revenue             — ``invoice.invoice_total``

        **CRITICAL**: ``SUM(debits)`` MUST equal ``SUM(credits)`` within
        ``Decimal("0.01")``.  Since both sides use the same
        ``invoice_total``, they are exactly equal.

        Args:
            invoice: The customer invoice record to create GL entries for.

        Returns:
            List of GL entry dicts with ``line_number``, ``account_code``,
            ``debit_amount``, ``credit_amount``, ``description``, and
            ``reference`` keys.
        """
        invoice_total = invoice.invoice_total

        entries: List[Dict[str, Any]] = [
            {
                "line_number": 1,
                "account_code": _AR_ACCOUNT_CODE,
                "account_name": "Accounts Receivable",
                "debit_amount": invoice_total,
                "credit_amount": Decimal("0.00"),
                "description": (
                    f"AR - Customer invoice {invoice.invoice_number} "
                    f"for {invoice.customer_name}"
                ),
                "reference": invoice.invoice_number,
                "source_document_type": "customer_invoice",
                "source_document_id": str(invoice.invoice_id),
            },
            {
                "line_number": 2,
                "account_code": _REVENUE_ACCOUNT_CODE,
                "account_name": "Revenue",
                "debit_amount": Decimal("0.00"),
                "credit_amount": invoice_total,
                "description": (
                    f"Revenue - Customer invoice {invoice.invoice_number} "
                    f"for {invoice.customer_name}"
                ),
                "reference": invoice.invoice_number,
                "source_document_type": "customer_invoice",
                "source_document_id": str(invoice.invoice_id),
            },
        ]

        # Verify balance (defensive — should always be zero imbalance)
        total_debits = sum(
            Decimal(str(e["debit_amount"])) for e in entries
        )
        total_credits = sum(
            Decimal(str(e["credit_amount"])) for e in entries
        )
        imbalance = abs(total_debits - total_credits)

        if imbalance > GL_BALANCE_TOLERANCE:
            logger.error(
                "invoice_gl_entries_imbalanced",
                service_name="transactions",
                component="CustomerInvoiceGenerator",
                invoice_number=invoice.invoice_number,
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                imbalance=str(imbalance),
            )

        return entries

    # ------------------------------------------------------------------
    # Sequential Numbering
    # ------------------------------------------------------------------

    def _generate_invoice_number(self, invoice_date: date) -> str:
        """Generate a sequential invoice number in INV-YYYY-NNNN format.

        Resets the sequence counter when the year changes, ensuring that
        numbering restarts at ``0001`` for each new year.

        Args:
            invoice_date: The invoice date used for the year component.

        Returns:
            Formatted invoice number, e.g. ``"INV-2024-0001"``.
        """
        year = invoice_date.year

        # Reset counter on year change
        if year != self._current_year:
            self._current_year = year
            self._sequence_counter = 0

        self._sequence_counter += 1

        prefix = DOCUMENT_NUMBER_PREFIXES.get("customer_invoice", "INV")
        return self._generate_sequential_number(
            prefix, year, self._sequence_counter
        )

    # ------------------------------------------------------------------
    # Shipped Order Query
    # ------------------------------------------------------------------

    def _get_shipped_orders(
        self, context: GenerationContext
    ) -> Sequence[ShippedOrderInfo]:
        """Retrieve shipped orders ready for invoicing.

        In a full pipeline integration, this queries the database for
        shipments in ``"in_transit"`` or ``"delivered"`` status that have
        not yet been invoiced.  For standalone or test use, returns sample
        data from the context's ``additional_params`` or ``day_context``.

        Args:
            context: The current :class:`GenerationContext` that may carry
                shipped order data in ``additional_params["shipped_orders"]``
                or ``day_context``.

        Returns:
            List of :class:`ShippedOrderInfo` objects ready for invoicing.
        """
        # Check if shipped orders were provided in context
        shipped_orders_data = context.additional_params.get(
            "shipped_orders", []
        )
        if shipped_orders_data:
            orders: List[ShippedOrderInfo] = []
            for data in shipped_orders_data:
                if isinstance(data, ShippedOrderInfo):
                    orders.append(data)
                elif isinstance(data, dict):
                    orders.append(ShippedOrderInfo.model_validate(data))
            return orders

        # Check day_context for shipped orders
        if context.day_context is not None:
            day_ctx = context.day_context
            if hasattr(day_ctx, "shipped_orders_for_invoicing"):
                raw_orders = day_ctx.shipped_orders_for_invoicing
                orders = []
                for data in raw_orders:
                    if isinstance(data, ShippedOrderInfo):
                        orders.append(data)
                    elif isinstance(data, dict):
                        orders.append(ShippedOrderInfo.model_validate(data))
                return orders

        # No shipped orders available — return empty list
        logger.debug(
            "no_shipped_orders_for_invoicing",
            service_name="transactions",
            component="CustomerInvoiceGenerator",
            trace_id=str(context.trace_id),
            simulation_id=str(context.simulation_id),
        )
        return []

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return customer invoice generation metrics.

        Returns:
            Dictionary with:
            - ``invoices_generated``: Total number of invoices created.
            - ``total_invoice_amount``: Sum of all invoice totals (str).
            - ``average_invoice_amount``: Mean invoice amount (str).
            - ``invoice_history_count``: Number of invoices in history.
        """
        avg_amount = Decimal("0.00")
        if self._invoices_generated > 0:
            avg_amount = (
                self._total_invoice_amount / Decimal(str(self._invoices_generated))
            ).quantize(Decimal("0.01"))

        return {
            "invoices_generated": self._invoices_generated,
            "total_invoice_amount": str(self._total_invoice_amount),
            "average_invoice_amount": str(avg_amount),
            "invoice_history_count": len(self._invoice_history),
        }
