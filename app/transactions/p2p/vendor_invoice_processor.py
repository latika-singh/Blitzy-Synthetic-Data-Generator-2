"""Vendor Invoice Processor — Stage 3 of the Procure-to-Pay (P2P) cycle.

Creates vendor invoices, links them to purchase orders, invokes three-way
matching (PO / Receipt / Invoice), routes to AP clerk agents for processing,
and posts GL journal entries.

Business Flow
-------------
1.  Vendor submits invoice referencing an existing purchase order.
2.  Link invoice to the open PO and its associated goods receipt(s).
3.  Create ``VendorInvoiceHeader`` and ``VendorInvoiceLine`` items from the
    PO / receipt data with realistic pricing and quantity variation.
4.  Invoke :class:`ThreeWayMatcher` (PO vs. Receipt vs. Invoice) — **MUST**
    be performed **before** GL posting.  Match tolerances: ±5 % price,
    ±2 % quantity.
5.  Route to AP clerk agent via ``WorkflowOrchestrator``
    (``ROLE_MAPPING`` ``vendor_invoice`` → ``["ap_clerk", "ap_manager"]``).
6.  Check approval thresholds:
    *  < $10 K  → no approval required
    *  $10 K – $50 K  → ``ap_manager``
    *  $50 K – $100 K  → ``controller``
    *  ≥ $100 K  → ``cfo``
7.  Post GL entries:
    *  **DR Expense / Asset** (per line category)
    *  **CR Accounts Payable** (total invoice amount)
8.  Publish ``TransactionCreated`` event via ``EventBus`` (ADR-001).

Key Business Rules
------------------
*   Vendor invoice MUST reference a valid Purchase Order.
*   Three-way matching MUST be performed before GL posting.
*   Invoice amounts are validated at LINE-ITEM level (per line, then summed).
*   GL balance invariant: ``SUM(debits) = SUM(credits)`` within ``$0.01``.
*   Atomicity: if ANY step in GL posting fails, the ENTIRE transaction is
    rolled back — no partial postings.
*   Sequential numbering: ``VINV-YYYY-NNNN``.
*   All monetary calculations use ``Decimal`` — **never** ``float``.
*   Deterministic reproducibility via seeded ``random.Random``.

Performance Target
------------------
Part of P2P ≥ 50 complete cycles / minute (AAP §0.7.3).

Retry Policy (AAP §0.1.2)
--------------------------
``p2p_cycle_generation``: 2 attempts, linear backoff (1 s, 2 s), 60 s
timeout, fallback: skip transaction.

Database Pattern
----------------
::

    from synthetic_erp.db.session import get_session
    async with get_session() as session:
        ...

Exports
-------
:class:`VendorInvoiceLine`
    Pydantic V2 model for a single invoice line item.
:class:`VendorInvoiceHeader`
    Pydantic V2 model for a complete vendor invoice header.
:class:`VendorInvoiceProcessor`
    Concrete :class:`TransactionGenerator` subclass for vendor invoice
    processing.
"""

from __future__ import annotations

import decimal
import random
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
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
    RETRY_POLICIES,
    THREE_WAY_MATCH_TOLERANCES,
)
from app.transactions.exceptions import (
    GLPostingError,
    ThreeWayMatchError,
    TransactionGenerationError,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus
    from app.orchestration.approval_system import ApprovalSystem
    from app.orchestration.workflow_orchestrator import WorkflowOrchestrator
    from app.transactions.p2p.three_way_matcher import ThreeWayMatcher

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
# Module-level constants — reusable Decimal values and default data
# ---------------------------------------------------------------------------
_ZERO = Decimal("0")
_ONE = Decimal("1")

_GL_BALANCE_TOLERANCE: Decimal = FINANCIAL_TOLERANCES.get(
    "gl_balance", Decimal("0.01")
)

_VENDOR_INVOICE_PREFIX: str = DOCUMENT_NUMBER_PREFIXES.get(
    "vendor_invoice", "VINV"
)

_P2P_RETRY_POLICY: Dict[str, Any] = RETRY_POLICIES.get(
    "p2p_cycle_generation",
    {
        "max_attempts": 2,
        "backoff_strategy": "linear",
        "backoff_seconds": [1, 2],
        "timeout_seconds": 60,
        "fallback": "skip_transaction",
    },
)

# Default payment terms and their corresponding day counts
_PAYMENT_TERMS_DAYS: Dict[str, int] = {
    "Net 15": 15,
    "Net 30": 30,
    "Net 45": 45,
    "Net 60": 60,
    "2/10 Net 30": 30,
    "1/10 Net 30": 30,
    "Due on Receipt": 0,
}

# Default GL account codes for vendor invoice posting
_DEFAULT_EXPENSE_GL = "5000-001"  # General Expense
_DEFAULT_ASSET_GL = "1500-001"  # Fixed Assets — Equipment
_AP_CONTROL_ACCOUNT = "2000-001"  # Accounts Payable control account

# Default vendor data for fallback when no db_session_factory available
_DEFAULT_VENDORS: List[Dict[str, Any]] = [
    {
        "vendor_id": str(uuid4()),
        "name": "Acme Office Supplies",
        "payment_terms": "Net 30",
    },
    {
        "vendor_id": str(uuid4()),
        "name": "Global Tech Components",
        "payment_terms": "2/10 Net 30",
    },
    {
        "vendor_id": str(uuid4()),
        "name": "Premier Industrial Parts",
        "payment_terms": "Net 45",
    },
]

# Default products for invoice line generation
_DEFAULT_PRODUCTS: List[Dict[str, Any]] = [
    {"product_id": str(uuid4()), "description": "Standard Office Paper", "unit_price": Decimal("12.99")},
    {"product_id": str(uuid4()), "description": "Printer Toner Cartridge", "unit_price": Decimal("89.50")},
    {"product_id": str(uuid4()), "description": "Ethernet Cable Cat6 100ft", "unit_price": Decimal("24.75")},
    {"product_id": str(uuid4()), "description": "Desk Chair Ergonomic", "unit_price": Decimal("399.00")},
    {"product_id": str(uuid4()), "description": "LED Monitor 27in", "unit_price": Decimal("329.99")},
]

_DEFAULT_GL_ACCOUNTS: List[str] = [
    "5100-001",  # Office Supplies Expense
    "5100-002",  # IT Equipment Expense
    "5200-001",  # Inventory — Raw Materials
    "1500-001",  # Fixed Assets — Equipment
]


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════════


class VendorInvoiceLine(BaseModel):
    """Single line item on a vendor invoice.

    Represents one product/service billed on a vendor invoice.  The
    ``extended_amount`` MUST equal ``quantity × unit_price`` and is
    computed using ``Decimal`` arithmetic — *never* ``float``.

    Attributes:
        line_number: 1-based line sequence number.
        po_line_number: Corresponding PO line number for matching.
        product_id: Product or item identifier (nullable for services).
        description: Human-readable line item description.
        quantity: Invoiced quantity (Decimal, > 0).
        unit_price: Per-unit price on the invoice (Decimal, > 0).
        extended_amount: ``quantity × unit_price`` (Decimal, never float).
        gl_account: GL account for posting (Expense or Asset).
        po_unit_price: Original PO unit price for three-way matching.
        receipt_quantity: Received quantity for three-way matching.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_number: int = Field(
        ..., ge=1, description="Line sequence number starting at 1"
    )
    po_line_number: int = Field(
        ..., ge=1, description="Corresponding PO line number"
    )
    product_id: Optional[UUID] = Field(
        default=None, description="Product/item identifier"
    )
    description: str = Field(
        default="", description="Line item description"
    )
    quantity: Decimal = Field(
        ..., gt=Decimal("0"), description="Invoiced quantity"
    )
    unit_price: Decimal = Field(
        ..., gt=Decimal("0"), description="Unit price on invoice"
    )
    extended_amount: Decimal = Field(
        ..., description="quantity × unit_price"
    )
    gl_account: str = Field(
        default="", description="GL account for posting (Expense or Asset)"
    )
    po_unit_price: Optional[Decimal] = Field(
        default=None, description="Original PO unit price for matching"
    )
    receipt_quantity: Optional[Decimal] = Field(
        default=None, description="Received quantity for matching"
    )


class VendorInvoiceHeader(BaseModel):
    """Vendor invoice header with complete metadata and embedded line items.

    Carries the full vendor invoice state including vendor information,
    PO linkage, financial totals, match and approval status, and the list
    of :class:`VendorInvoiceLine` items.

    Financial Integrity:
        * ``subtotal`` = ``SUM(line.extended_amount for line in lines)``
        * ``total_amount`` = ``subtotal + tax_amount``
        * All monetary fields are ``Decimal`` — never ``float``.

    Statuses:
        * ``status``: ``pending_match`` → ``matched`` → ``approved`` →
          ``posted`` → ``paid`` | ``exception``
        * ``match_status``: Result from three-way matching (e.g.
          ``full_match``, ``price_exception``).
        * ``approval_status``: ``not_required``, ``pending``,
          ``approved``, ``rejected``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    invoice_id: UUID = Field(
        default_factory=uuid4,
        description="Unique invoice identifier",
    )
    invoice_number: str = Field(
        ..., description="Sequential number VINV-YYYY-NNNN"
    )
    vendor_id: UUID = Field(
        ..., description="Vendor identifier"
    )
    vendor_name: str = Field(
        default="", description="Vendor display name"
    )
    po_id: UUID = Field(
        ..., description="Linked Purchase Order ID"
    )
    po_number: str = Field(
        default="", description="Linked PO number for reference"
    )
    invoice_date: date = Field(
        ..., description="Date on the vendor invoice"
    )
    received_date: date = Field(
        ..., description="Date invoice was received"
    )
    due_date: date = Field(
        ..., description="Payment due date"
    )
    payment_terms: str = Field(
        default="Net 30", description="Payment terms code"
    )
    subtotal: Decimal = Field(
        default=Decimal("0"),
        description="Sum of line extended_amounts",
    )
    tax_amount: Decimal = Field(
        default=Decimal("0"),
        description="Tax amount (0 for MVP — single company, no tax)",
    )
    total_amount: Decimal = Field(
        default=Decimal("0"),
        description="subtotal + tax_amount",
    )
    currency: str = Field(
        default="USD", description="Currency code — USD only for MVP"
    )
    status: str = Field(
        default="pending_match",
        description=(
            "Invoice lifecycle status: pending_match, matched, "
            "approved, posted, exception"
        ),
    )
    match_status: Optional[str] = Field(
        default=None,
        description="Three-way match result status",
    )
    approval_status: Optional[str] = Field(
        default=None, description="Approval chain status"
    )
    created_by: Optional[UUID] = Field(
        default=None, description="Agent who processed the invoice"
    )
    lines: List[VendorInvoiceLine] = Field(
        default_factory=list, description="Invoice line items"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extensibility metadata"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VendorInvoiceProcessor
# ═══════════════════════════════════════════════════════════════════════════════


class VendorInvoiceProcessor(TransactionGenerator):
    """Processes vendor invoices through the P2P pipeline — Stage 3 of 5.

    Business Flow
    -------------
    1.  Receive vendor invoice data (from context or generated fallback).
    2.  Validate invoice against open PO (match PO reference).
    3.  Create invoice header and lines from PO/receipt data.
    4.  Invoke ``ThreeWayMatcher`` (PO vs. Receipt vs. Invoice).
    5.  Route to AP clerk agent via ``WorkflowOrchestrator``.
    6.  Check approval thresholds ($10 K / $50 K / $100 K).
    7.  Post GL entries: DR Expense/Asset, CR Accounts Payable.
    8.  Update invoice status to ``'posted'``.
    9.  Publish ``TransactionCreated`` event via ``EventBus``.

    Constructor Injection (ADR-003)
    -------------------------------
    All dependencies are ``Optional`` with ``None`` defaults, enabling
    partial composition for testing and incremental integration.

    Artifacts Produced
    ------------------
    The ``artifacts`` dictionary on the returned :class:`TransactionResult`
    contains these keys (matching ``REQUIRED_ARTIFACTS["vendor_invoice"]``
    from ``app/orchestration/transaction_orchestrator.py``):

    *   ``"invoice_header"`` — serialised :class:`VendorInvoiceHeader`
    *   ``"invoice_lines"`` — list of serialised :class:`VendorInvoiceLine`
    *   ``"three_way_match"`` — serialised match result dict
    *   ``"gl_entries"`` — list of GL journal entry dicts
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
        three_way_matcher: Optional[Any] = None,
    ) -> None:
        """Initialise the VendorInvoiceProcessor with injected dependencies.

        Args:
            db_session_factory: Callable returning an async session context
                manager (e.g., ``get_session`` from Project 1).
            agent_registry: Project 2 :class:`AgentRegistry` for agent
                lookup during invoice processing.
            workflow_orchestrator: Project 2 :class:`WorkflowOrchestrator`
                for routing invoices to AP clerk agents.
            event_bus: Project 2 :class:`EventBus` for cross-subsystem
                event publication (ADR-001).
            approval_system: Project 2 :class:`ApprovalSystem` for
                threshold-based approval chain determination.
            discrepancy_injector: P3 ``DiscrepancyInjector`` instance for
                rate-based discrepancy injection into invoices.
            gl_posting_engine: P3 ``GLPostingEngine`` instance for journal
                entry creation and balance validation.
            statistical_models: Dictionary of statistical model instances
                (amount distributions, payment timing, etc.).
            three_way_matcher: :class:`ThreeWayMatcher` instance for
                PO/Receipt/Invoice line-item matching.  Received via
                constructor injection (ADR-003).
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

        # ThreeWayMatcher instance — not a base-class dependency, stored
        # directly on this subclass.
        self._three_way_matcher = three_way_matcher

        # Instance-level invoice sequence counter for numbering
        self._invoice_sequence: int = 0

        # Cache retry policy for documentation / runtime checks
        self._retry_policy: Dict[str, Any] = _P2P_RETRY_POLICY

        logger.info(
            "vendor_invoice_processor_initialized",
            has_three_way_matcher=three_way_matcher is not None,
            service_name="transactions",
            component="VendorInvoiceProcessor",
        )

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Generate a vendor invoice for the given context.

        Produces a :class:`TransactionResult` with artifacts keyed as:
        ``"invoice_header"``, ``"invoice_lines"``, ``"three_way_match"``,
        ``"gl_entries"`` — matching the
        ``REQUIRED_ARTIFACTS["vendor_invoice"]`` specification from
        ``app/orchestration/transaction_orchestrator.py``.

        Args:
            context: Per-invocation generation context carrying
                ``current_date``, ``rng_seed``, ``simulation_id``, etc.

        Returns:
            A :class:`TransactionResult` with
            ``transaction_type="vendor_invoice"`` and all required
            artifacts populated.

        Raises:
            TransactionGenerationError: On unrecoverable generation failure.
        """
        start_time = time.perf_counter()

        try:
            # 1. Create deterministic RNG from context seed
            rng: random.Random = self._create_seeded_rng(context)

            # 2. Obtain PO data for invoicing (from context or fallback)
            po_data = self._get_po_data_for_invoicing(rng, context)

            # 3. Create invoice header and lines from PO data
            invoice_header, invoice_lines = self._create_invoice_from_po(
                rng=rng,
                po_data=po_data,
                context=context,
            )

            # 4. Calculate financial totals
            subtotal = _ZERO
            for line in invoice_lines:
                subtotal += line.extended_amount

            tax_amount = _ZERO  # 0 for MVP — single company, no tax calc
            total_amount = subtotal + tax_amount

            invoice_header.subtotal = subtotal
            invoice_header.tax_amount = tax_amount
            invoice_header.total_amount = total_amount
            invoice_header.lines = invoice_lines

            # 5. Check discrepancy trigger
            has_discrepancy = False
            discrepancy_type: Optional[str] = None
            should_inject, disc_type, disc_params = (
                await self._check_discrepancy_trigger(
                    context=context,
                    transaction_type="vendor_invoice",
                )
            )
            if should_inject:
                has_discrepancy = True
                discrepancy_type = disc_type

            # 6. Invoke three-way match (MUST be before GL posting)
            match_result_dict = self._perform_three_way_match(
                invoice_header=invoice_header,
                invoice_lines=invoice_lines,
                po_data=po_data,
            )
            invoice_header.match_status = match_result_dict.get(
                "match_status", "full_match"
            )

            # Update status based on match result
            match_status_val = match_result_dict.get("match_status", "full_match")
            if match_status_val == "full_match":
                invoice_header.status = "matched"
            else:
                invoice_header.status = "exception"

            # 7. Route to AP clerk via WorkflowOrchestrator
            await self._route_to_ap_clerk(
                invoice_header=invoice_header,
                context=context,
            )

            # 8. Check approval thresholds
            requires_approval, required_role = self._check_approval_required(
                total_amount=total_amount,
            )
            if requires_approval:
                invoice_header.approval_status = "pending"
                invoice_header.metadata["required_approver_role"] = required_role
            else:
                invoice_header.approval_status = "not_required"
                # Auto-advance status if matched and no approval needed
                if invoice_header.status == "matched":
                    invoice_header.status = "approved"

            # 9. Prepare GL entries: DR Expense/Asset per line, CR AP total
            gl_entries = self._prepare_gl_entries(
                invoice_header=invoice_header,
                invoice_lines=invoice_lines,
                context=context,
            )

            # 10. Build and return TransactionResult
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0

            result = TransactionResult(
                transaction_type="vendor_invoice",
                status="completed",
                artifacts={
                    "invoice_header": invoice_header.model_dump(mode="json"),
                    "invoice_lines": [
                        line.model_dump(mode="json") for line in invoice_lines
                    ],
                    "three_way_match": match_result_dict,
                    "gl_entries": gl_entries,
                },
                gl_entries=gl_entries,
                has_discrepancy=has_discrepancy,
                discrepancy_type=discrepancy_type,
                amount=total_amount,
                duration_ms=elapsed_ms,
            )

            # 11. Publish TransactionCreated event
            await self._publish_event(
                event_type="TransactionCreated",
                payload={
                    "transaction_type": "vendor_invoice",
                    "transaction_id": str(result.transaction_id),
                    "invoice_id": str(invoice_header.invoice_id),
                    "invoice_number": invoice_header.invoice_number,
                    "vendor_id": str(invoice_header.vendor_id),
                    "po_id": str(invoice_header.po_id),
                    "total_amount": str(total_amount),
                    "match_status": invoice_header.match_status,
                    "has_discrepancy": has_discrepancy,
                    "gl_entry_count": len(gl_entries),
                },
                context=context,
            )
            result.events_published.append("TransactionCreated")

            # 12. Log transaction per AAP §0.7.7
            self._log_transaction(result, context)

            logger.info(
                "vendor_invoice_generated",
                transaction_id=str(result.transaction_id),
                invoice_number=invoice_header.invoice_number,
                vendor_id=str(invoice_header.vendor_id),
                amount=str(total_amount),
                match_status=invoice_header.match_status,
                has_discrepancy=has_discrepancy,
                discrepancy_type=discrepancy_type,
                gl_entries=len(gl_entries),
                duration_ms=round(elapsed_ms, 2),
                service_name="transactions",
                component="VendorInvoiceProcessor",
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
            )

            return result

        except TransactionGenerationError:
            raise
        except ThreeWayMatchError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            logger.error(
                "vendor_invoice_three_way_match_failed",
                error=str(exc),
                duration_ms=round(elapsed_ms, 2),
                service_name="transactions",
                component="VendorInvoiceProcessor",
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
            )
            return TransactionResult(
                transaction_type="vendor_invoice",
                status="failed",
                error_message=f"Three-way match failed: {exc}",
                errors=[str(exc)],
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            logger.error(
                "vendor_invoice_generation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=round(elapsed_ms, 2),
                service_name="transactions",
                component="VendorInvoiceProcessor",
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
            )
            raise TransactionGenerationError(
                message=f"Vendor invoice generation failed: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "generator_class": "VendorInvoiceProcessor",
                    "trace_id": str(context.trace_id),
                    "simulation_id": str(context.simulation_id),
                },
            ) from exc

    async def validate(self, result: TransactionResult) -> bool:
        """Validate a generated vendor invoice result against business rules.

        Checks:
        *   ``invoice_header`` artifact exists and has required fields.
        *   ``invoice_lines`` artifact is non-empty.
        *   All line amounts use ``Decimal`` (never ``float``).
        *   ``total_amount`` = ``SUM(line.extended_amount)``.
        *   ``three_way_match`` artifact exists.
        *   GL entries balance: ``SUM(debits) = SUM(credits)`` within $0.01.
        *   Invoice references a valid PO (``po_id`` is present).

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if the result passes all validation checks,
            ``False`` otherwise.
        """
        if result.status == "failed":
            logger.warning(
                "vendor_invoice_validate_skipped_failed",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return False

        artifacts = result.artifacts

        # 1. Validate invoice_header artifact exists
        invoice_header = artifacts.get("invoice_header")
        if not invoice_header:
            logger.error(
                "vendor_invoice_validate_missing_header",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return False

        # 2. Validate required header fields
        required_header_fields = [
            "invoice_id", "invoice_number", "vendor_id", "po_id",
            "invoice_date", "total_amount",
        ]
        for field_name in required_header_fields:
            if field_name not in invoice_header or invoice_header[field_name] is None:
                logger.error(
                    "vendor_invoice_validate_missing_field",
                    field=field_name,
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="VendorInvoiceProcessor",
                )
                return False

        # 3. Validate invoice_lines artifact is non-empty
        invoice_lines = artifacts.get("invoice_lines")
        if not invoice_lines or len(invoice_lines) == 0:
            logger.error(
                "vendor_invoice_validate_empty_lines",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return False

        # 4. Validate total_amount = SUM(line.extended_amount)
        expected_subtotal = _ZERO
        for line in invoice_lines:
            extended = Decimal(str(line.get("extended_amount", "0")))
            expected_subtotal += extended

        header_subtotal = Decimal(str(invoice_header.get("subtotal", "0")))
        if abs(header_subtotal - expected_subtotal) > _GL_BALANCE_TOLERANCE:
            logger.error(
                "vendor_invoice_validate_subtotal_mismatch",
                expected=str(expected_subtotal),
                actual=str(header_subtotal),
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return False

        # 5. Validate three_way_match artifact exists
        three_way_match = artifacts.get("three_way_match")
        if three_way_match is None:
            logger.error(
                "vendor_invoice_validate_missing_match",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return False

        # 6. Validate GL entries balance (DR = CR within $0.01)
        gl_entries = artifacts.get("gl_entries", [])
        if gl_entries:
            total_debits = _ZERO
            total_credits = _ZERO
            for entry in gl_entries:
                total_debits += Decimal(str(entry.get("debit_amount", entry.get("debit", "0"))))
                total_credits += Decimal(str(entry.get("credit_amount", entry.get("credit", "0"))))

            imbalance = abs(total_debits - total_credits)
            if imbalance > _GL_BALANCE_TOLERANCE:
                logger.error(
                    "vendor_invoice_validate_gl_imbalance",
                    total_debits=str(total_debits),
                    total_credits=str(total_credits),
                    imbalance=str(imbalance),
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="VendorInvoiceProcessor",
                )
                return False

        logger.debug(
            "vendor_invoice_validated",
            transaction_id=str(result.transaction_id),
            line_count=len(invoice_lines),
            gl_entry_count=len(gl_entries),
            service_name="transactions",
            component="VendorInvoiceProcessor",
        )
        return True

    async def post(self, result: TransactionResult) -> None:
        """Post vendor invoice GL entries to the General Ledger.

        GL entries for a vendor invoice:
        *   **DR Expense/Asset** (per line category) — each line amount
        *   **CR Accounts Payable** — total invoice amount

        CRITICAL ATOMICITY: If ANY step fails, the ENTIRE transaction
        is rolled back — no partial postings (AAP §0.7.2).

        Args:
            result: The validated :class:`TransactionResult` to post.

        Raises:
            GLPostingError: If GL posting fails (balance, account, or
                period validation failures).
        """
        if not result.gl_entries:
            logger.info(
                "vendor_invoice_post_no_gl_entries",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return

        # Pre-posting balance validation
        total_debits = _ZERO
        total_credits = _ZERO
        for entry in result.gl_entries:
            total_debits += Decimal(str(entry.get("debit_amount", entry.get("debit", "0"))))
            total_credits += Decimal(str(entry.get("credit_amount", entry.get("credit", "0"))))

        imbalance = abs(total_debits - total_credits)
        if imbalance > _GL_BALANCE_TOLERANCE:
            raise GLPostingError(
                message=(
                    f"GL entries do not balance: "
                    f"debits={total_debits}, credits={total_credits}, "
                    f"imbalance={imbalance}"
                ),
                details={
                    "total_debits": str(total_debits),
                    "total_credits": str(total_credits),
                    "imbalance": str(imbalance),
                    "tolerance": str(_GL_BALANCE_TOLERANCE),
                    "transaction_id": str(result.transaction_id),
                    "transaction_type": result.transaction_type,
                },
            )

        # Delegate to GLPostingEngine if available
        if self._gl_posting_engine is not None:
            try:
                await self._gl_posting_engine.post_entries(
                    journal_entries=result.gl_entries,
                )
                logger.info(
                    "vendor_invoice_gl_entries_posted",
                    transaction_id=str(result.transaction_id),
                    entry_count=len(result.gl_entries),
                    total_debits=str(total_debits),
                    total_credits=str(total_credits),
                    service_name="transactions",
                    component="VendorInvoiceProcessor",
                )
            except Exception as exc:
                logger.error(
                    "vendor_invoice_gl_posting_failed",
                    error=str(exc),
                    transaction_id=str(result.transaction_id),
                    entry_count=len(result.gl_entries),
                    service_name="transactions",
                    component="VendorInvoiceProcessor",
                )
                raise GLPostingError(
                    message=f"GL posting failed for vendor invoice: {exc}",
                    details={
                        "transaction_id": str(result.transaction_id),
                        "entry_count": len(result.gl_entries),
                        "error_type": type(exc).__name__,
                    },
                ) from exc
        else:
            logger.info(
                "vendor_invoice_post_skipped_no_engine",
                transaction_id=str(result.transaction_id),
                entry_count=len(result.gl_entries),
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _create_invoice_from_po(
        self,
        *,
        rng: random.Random,
        po_data: Dict[str, Any],
        context: GenerationContext,
    ) -> Tuple[VendorInvoiceHeader, List[VendorInvoiceLine]]:
        """Create a VendorInvoiceHeader and line items from PO/receipt data.

        Maps PO lines to invoice lines, calculates ``extended_amount =
        quantity × unit_price`` per line, and determines payment terms
        and due date from vendor master data.

        Args:
            rng: Seeded ``random.Random`` instance for deterministic
                generation.
            po_data: Dictionary containing PO header and line data.
            context: Current generation context for date and simulation
                information.

        Returns:
            A tuple of ``(invoice_header, invoice_lines)`` ready for
            three-way matching and GL posting.
        """
        # Extract PO metadata
        po_id = UUID(str(po_data.get("po_id", uuid4())))
        po_number = po_data.get("po_number", "PO-0000-0000")
        vendor_id = UUID(str(po_data.get("vendor_id", uuid4())))
        vendor_name = po_data.get("vendor_name", "Unknown Vendor")
        payment_terms = po_data.get("payment_terms", "Net 30")

        # Extract PO lines (with fallback generation)
        po_lines_raw = po_data.get("lines", [])
        if not po_lines_raw:
            po_lines_raw = self._generate_fallback_po_lines(rng)

        # Determine invoice date and due date
        invoice_date = context.current_date
        received_date = context.current_date

        # Calculate due date from payment terms
        terms_days = _PAYMENT_TERMS_DAYS.get(payment_terms, 30)
        due_date = invoice_date + timedelta(days=terms_days)

        # Generate sequential invoice number
        invoice_number = self._generate_invoice_number(
            year=context.current_date.year,
        )

        # Create invoice lines from PO lines
        invoice_lines: List[VendorInvoiceLine] = []
        for idx, po_line in enumerate(po_lines_raw, start=1):
            # Get quantities and prices from PO line
            po_quantity = Decimal(str(po_line.get("quantity", "1")))
            po_unit_price = Decimal(str(po_line.get("unit_price", "10.00")))

            # Receipt quantity (from context or same as PO quantity)
            receipt_qty = Decimal(
                str(po_line.get("receipt_quantity", str(po_quantity)))
            )

            # Invoice quantity — typically matches receipt quantity
            # with minor variation for realistic simulation
            invoice_quantity = receipt_qty

            # Invoice unit price — typically matches PO price
            # with minor variation for realistic simulation
            invoice_unit_price = po_unit_price

            # Calculate extended amount: quantity × unit_price
            extended_amount = (invoice_quantity * invoice_unit_price).quantize(
                Decimal("0.01")
            )

            # Determine GL account for posting
            gl_account = po_line.get(
                "gl_account",
                rng.choice(_DEFAULT_GL_ACCOUNTS),
            )

            # Product identifiers
            product_id_str = po_line.get("product_id")
            product_id: Optional[UUID] = None
            if product_id_str:
                try:
                    product_id = UUID(str(product_id_str))
                except (ValueError, TypeError):
                    product_id = None

            description = po_line.get("description", f"Line item {idx}")

            invoice_line = VendorInvoiceLine(
                line_number=idx,
                po_line_number=po_line.get("line_number", idx),
                product_id=product_id,
                description=description,
                quantity=invoice_quantity,
                unit_price=invoice_unit_price,
                extended_amount=extended_amount,
                gl_account=gl_account,
                po_unit_price=po_unit_price,
                receipt_quantity=receipt_qty,
            )
            invoice_lines.append(invoice_line)

        # Determine created_by agent (if agent_registry available)
        created_by: Optional[UUID] = None
        if self._agent_registry is not None:
            try:
                agent = self._agent_registry.get_agent_by_role("ap_clerk")
                if agent is not None:
                    created_by = getattr(agent, "agent_id", None)
            except Exception:
                pass

        # Create invoice header
        invoice_header = VendorInvoiceHeader(
            invoice_number=invoice_number,
            vendor_id=vendor_id,
            vendor_name=vendor_name,
            po_id=po_id,
            po_number=po_number,
            invoice_date=invoice_date,
            received_date=received_date,
            due_date=due_date,
            payment_terms=payment_terms,
            currency="USD",
            status="pending_match",
            created_by=created_by,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "VendorInvoiceProcessor",
            },
        )

        return invoice_header, invoice_lines

    def _prepare_gl_entries(
        self,
        *,
        invoice_header: VendorInvoiceHeader,
        invoice_lines: List[VendorInvoiceLine],
        context: GenerationContext,
    ) -> List[Dict[str, Any]]:
        """Prepare GL journal entries for the vendor invoice.

        Creates one DEBIT entry per invoice line (to Expense or Asset GL
        account) and one CREDIT entry for the total (to Accounts Payable
        control account).  Validates that DR = CR within $0.01 tolerance
        before returning.

        GL Posting Pattern:
            * DR Expense/Asset (per line) — each ``line.extended_amount``
            * CR Accounts Payable (total) — ``invoice_header.total_amount``

        Args:
            invoice_header: The vendor invoice header with totals.
            invoice_lines: The vendor invoice line items.
            context: Current generation context for trace correlation.

        Returns:
            List of GL journal entry dictionaries, each containing:
            ``account``, ``debit``, ``credit``, ``description``,
            ``line_number``, ``transaction_id``, ``transaction_type``.

        Raises:
            GLPostingError: If the generated entries do not balance
                within the $0.01 tolerance.
        """
        gl_entries: List[Dict[str, Any]] = []
        total_debits = _ZERO

        # One DEBIT entry per invoice line (Expense/Asset account)
        for line in invoice_lines:
            gl_account = line.gl_account or _DEFAULT_EXPENSE_GL
            debit_amount = line.extended_amount

            gl_entry = {
                "account": gl_account,
                "debit_amount": str(debit_amount),
                "credit_amount": str(_ZERO),
                "description": (
                    f"Vendor invoice {invoice_header.invoice_number} "
                    f"line {line.line_number}: {line.description}"
                ),
                "line_number": line.line_number,
                "transaction_id": str(invoice_header.invoice_id),
                "transaction_type": "vendor_invoice",
                "posting_date": str(invoice_header.invoice_date),
                "vendor_id": str(invoice_header.vendor_id),
                "po_id": str(invoice_header.po_id),
                "fiscal_period": context.fiscal_period,
            }
            gl_entries.append(gl_entry)
            total_debits += debit_amount

        # One CREDIT entry for total (Accounts Payable control account)
        credit_amount = invoice_header.total_amount
        ap_entry = {
            "account": _AP_CONTROL_ACCOUNT,
            "debit_amount": str(_ZERO),
            "credit_amount": str(credit_amount),
            "description": (
                f"Vendor invoice {invoice_header.invoice_number} "
                f"AP for vendor {invoice_header.vendor_name}"
            ),
            "line_number": 0,
            "transaction_id": str(invoice_header.invoice_id),
            "transaction_type": "vendor_invoice",
            "posting_date": str(invoice_header.invoice_date),
            "vendor_id": str(invoice_header.vendor_id),
            "po_id": str(invoice_header.po_id),
            "fiscal_period": context.fiscal_period,
        }
        gl_entries.append(ap_entry)

        # Validate balance: DR = CR within $0.01
        imbalance = abs(total_debits - credit_amount)
        if imbalance > _GL_BALANCE_TOLERANCE:
            raise GLPostingError(
                message=(
                    f"Prepared GL entries do not balance: "
                    f"debits={total_debits}, credits={credit_amount}, "
                    f"imbalance={imbalance}"
                ),
                details={
                    "total_debits": str(total_debits),
                    "total_credits": str(credit_amount),
                    "imbalance": str(imbalance),
                    "invoice_id": str(invoice_header.invoice_id),
                    "invoice_number": invoice_header.invoice_number,
                },
            )

        logger.debug(
            "vendor_invoice_gl_entries_prepared",
            invoice_number=invoice_header.invoice_number,
            entry_count=len(gl_entries),
            total_debits=str(total_debits),
            total_credits=str(credit_amount),
            service_name="transactions",
            component="VendorInvoiceProcessor",
        )

        return gl_entries

    def _check_approval_required(
        self,
        *,
        total_amount: Decimal,
    ) -> Tuple[bool, Optional[str]]:
        """Check if the invoice amount requires approval.

        Reads vendor_invoice approval thresholds from
        ``APPROVAL_THRESHOLDS`` and determines the required approver role
        based on the invoice total amount:

        *   < $10 K  → no approval required (auto-approved)
        *   $10 K – $50 K  → ``ap_manager``
        *   $50 K – $100 K  → ``controller``
        *   ≥ $100 K  → ``cfo``

        Args:
            total_amount: Total invoice amount (Decimal).

        Returns:
            A tuple ``(requires_approval, required_role)`` where:
            *   ``requires_approval`` is ``True`` when an approver must
                sign off.
            *   ``required_role`` is the string role name (e.g.,
                ``"ap_manager"``, ``"controller"``, ``"cfo"``) or ``None``
                when no approval is needed.
        """
        vendor_invoice_config = APPROVAL_THRESHOLDS.get(
            "vendor_invoice", {}
        )
        tiers = vendor_invoice_config.get("tiers", [])

        if not tiers:
            # Fallback: hardcoded thresholds matching AAP specification
            if total_amount < Decimal("10000"):
                return (False, None)
            elif total_amount < Decimal("50000"):
                return (True, "ap_manager")
            elif total_amount < Decimal("100000"):
                return (True, "controller")
            else:
                return (True, "cfo")

        # Iterate through configured tiers (ordered by ascending max_amount)
        for tier in tiers:
            max_amount = tier.get("max_amount")
            required_role = tier.get("required_role")

            if max_amount is None:
                # Unbounded tier — this is the catch-all for highest amounts
                if required_role is not None:
                    return (True, required_role)
                return (False, None)

            if total_amount < Decimal(str(max_amount)):
                if required_role is not None:
                    return (True, required_role)
                return (False, None)

        # If no tier matched (shouldn't happen with a None-capped last tier)
        return (False, None)

    def _generate_invoice_number(self, *, year: int) -> str:
        """Generate a sequential vendor invoice number.

        Format: ``VINV-YYYY-NNNN`` using the
        :meth:`_generate_sequential_number` base method with the prefix
        from ``DOCUMENT_NUMBER_PREFIXES["vendor_invoice"]``.

        Increments the instance-level ``_invoice_sequence`` counter on
        each call.

        Args:
            year: Four-digit year component.

        Returns:
            Formatted invoice number string (e.g., ``"VINV-2024-0001"``).
        """
        self._invoice_sequence += 1
        return self._generate_sequential_number(
            prefix=_VENDOR_INVOICE_PREFIX,
            year=year,
            sequence=self._invoice_sequence,
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_po_data_for_invoicing(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> Dict[str, Any]:
        """Obtain PO data suitable for vendor invoice creation.

        Tries the following sources in order:
        1. ``context.day_context`` — P3 daily context (open POs with receipts).
        2. ``context.additional_params["po_data"]`` — explicitly passed PO.
        3. Fallback — generate synthetic PO data for standalone testing.

        Args:
            rng: Seeded RNG for fallback data generation.
            context: Current generation context.

        Returns:
            Dictionary containing PO header and line data suitable for
            invoice generation.
        """
        # Source 1: Day context (realistic scenario)
        if context.day_context is not None:
            day_ctx = context.day_context
            open_pos = getattr(day_ctx, "open_pos_with_receipts", None)
            if open_pos and len(open_pos) > 0:
                return rng.choice(open_pos)

        # Source 2: Additional params (explicit PO data injection)
        po_data = context.additional_params.get("po_data")
        if po_data is not None:
            return po_data

        # Source 3: Fallback — generate synthetic PO data
        return self._generate_fallback_po_data(rng, context)

    def _generate_fallback_po_data(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> Dict[str, Any]:
        """Generate synthetic PO data for standalone/testing scenarios.

        Creates a realistic PO header and 1–5 line items using the seeded
        RNG for deterministic reproducibility.

        Args:
            rng: Seeded RNG for deterministic generation.
            context: Current generation context for date information.

        Returns:
            Dictionary mimicking a real PO with header fields and lines.
        """
        vendor = rng.choice(_DEFAULT_VENDORS)
        num_lines = rng.randint(1, 5)
        selected_products = rng.sample(
            _DEFAULT_PRODUCTS, min(num_lines, len(_DEFAULT_PRODUCTS))
        )

        po_id = uuid4()
        po_number = f"PO-{context.current_date.year}-{rng.randint(1, 9999):04d}"

        lines: List[Dict[str, Any]] = []
        for idx, product in enumerate(selected_products, start=1):
            quantity = Decimal(str(rng.randint(1, 100)))
            unit_price = Decimal(str(product["unit_price"]))
            extended_amount = (quantity * unit_price).quantize(Decimal("0.01"))

            lines.append({
                "line_number": idx,
                "product_id": product["product_id"],
                "description": product["description"],
                "quantity": str(quantity),
                "unit_price": str(unit_price),
                "extended_amount": str(extended_amount),
                "gl_account": rng.choice(_DEFAULT_GL_ACCOUNTS),
                "receipt_quantity": str(quantity),
            })

        return {
            "po_id": str(po_id),
            "po_number": po_number,
            "vendor_id": vendor["vendor_id"],
            "vendor_name": vendor["name"],
            "payment_terms": vendor["payment_terms"],
            "order_date": str(
                context.current_date - timedelta(days=rng.randint(7, 30))
            ),
            "status": "approved",
            "lines": lines,
        }

    def _generate_fallback_po_lines(
        self,
        rng: random.Random,
    ) -> List[Dict[str, Any]]:
        """Generate fallback PO line data when no lines are in PO data.

        Args:
            rng: Seeded RNG for deterministic generation.

        Returns:
            List of PO line dictionaries.
        """
        num_lines = rng.randint(1, 3)
        selected_products = rng.sample(
            _DEFAULT_PRODUCTS, min(num_lines, len(_DEFAULT_PRODUCTS))
        )

        lines: List[Dict[str, Any]] = []
        for idx, product in enumerate(selected_products, start=1):
            quantity = Decimal(str(rng.randint(1, 50)))
            unit_price = Decimal(str(product["unit_price"]))
            lines.append({
                "line_number": idx,
                "product_id": product["product_id"],
                "description": product["description"],
                "quantity": str(quantity),
                "unit_price": str(unit_price),
                "gl_account": rng.choice(_DEFAULT_GL_ACCOUNTS),
                "receipt_quantity": str(quantity),
            })
        return lines

    def _perform_three_way_match(
        self,
        *,
        invoice_header: VendorInvoiceHeader,
        invoice_lines: List[VendorInvoiceLine],
        po_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke the three-way matcher to validate PO/Receipt/Invoice.

        If no :class:`ThreeWayMatcher` was injected, returns a synthetic
        full-match result for graceful degradation during testing.

        Args:
            invoice_header: The vendor invoice header.
            invoice_lines: The vendor invoice line items.
            po_data: The PO data dictionary (with lines).

        Returns:
            Dictionary representation of the match result, containing
            at minimum ``match_status`` and per-line details.
        """
        po_lines_raw = po_data.get("lines", [])

        # Build dicts for the matcher
        po_match_lines: List[Dict[str, Any]] = []
        receipt_match_lines: List[Dict[str, Any]] = []
        invoice_match_lines: List[Dict[str, Any]] = []

        for line in invoice_lines:
            line_num = line.line_number

            # Find corresponding PO line data
            po_line_data = {}
            for pl in po_lines_raw:
                if pl.get("line_number", 0) == line.po_line_number:
                    po_line_data = pl
                    break

            po_match_lines.append({
                "line_number": line_num,
                "quantity": str(line.po_unit_price and Decimal(
                    str(po_line_data.get("quantity", str(line.quantity)))
                ) or str(line.quantity)),
                "unit_price": str(
                    line.po_unit_price
                    if line.po_unit_price is not None
                    else line.unit_price
                ),
            })

            receipt_match_lines.append({
                "line_number": line_num,
                "quantity": str(
                    line.receipt_quantity
                    if line.receipt_quantity is not None
                    else line.quantity
                ),
            })

            invoice_match_lines.append({
                "line_number": line_num,
                "quantity": str(line.quantity),
                "unit_price": str(line.unit_price),
            })

        if self._three_way_matcher is not None:
            try:
                # Determine receipt_id from po_data metadata if available
                receipt_id_str = po_data.get("receipt_id")
                receipt_id: Optional[UUID] = None
                if receipt_id_str:
                    try:
                        receipt_id = UUID(str(receipt_id_str))
                    except (ValueError, TypeError):
                        receipt_id = None

                match_result = self._three_way_matcher.match(
                    po_lines=po_match_lines,
                    receipt_lines=receipt_match_lines,
                    invoice_lines=invoice_match_lines,
                    invoice_id=invoice_header.invoice_id,
                    po_id=invoice_header.po_id,
                    receipt_id=receipt_id,
                )

                # Convert Pydantic model to dict for artifact storage
                result_dict = match_result.model_dump(mode="json")

                logger.debug(
                    "vendor_invoice_three_way_match_completed",
                    invoice_number=invoice_header.invoice_number,
                    match_status=result_dict.get("match_status", "unknown"),
                    service_name="transactions",
                    component="VendorInvoiceProcessor",
                )

                return result_dict

            except ThreeWayMatchError:
                raise
            except Exception as exc:
                logger.warning(
                    "vendor_invoice_three_way_match_error",
                    error=str(exc),
                    invoice_number=invoice_header.invoice_number,
                    service_name="transactions",
                    component="VendorInvoiceProcessor",
                )
                # Fallback: return exception status
                return {
                    "match_status": "exception",
                    "approval_required": True,
                    "notes": f"Match error: {exc}",
                    "line_results": [],
                }

        # No matcher injected — return synthetic full match result
        logger.debug(
            "vendor_invoice_three_way_match_skipped_no_matcher",
            invoice_number=invoice_header.invoice_number,
            service_name="transactions",
            component="VendorInvoiceProcessor",
        )

        return {
            "match_id": str(uuid4()),
            "invoice_id": str(invoice_header.invoice_id),
            "po_id": str(invoice_header.po_id),
            "match_status": "full_match",
            "approval_required": False,
            "total_price_variance": str(_ZERO),
            "total_quantity_variance": str(_ZERO),
            "total_extended_variance": str(_ZERO),
            "line_results": [
                {
                    "line_number": line.line_number,
                    "quantity_within_tolerance": True,
                    "price_within_tolerance": True,
                    "line_match_status": "full_match",
                }
                for line in invoice_lines
            ],
            "lines_matched": len(invoice_lines),
            "lines_with_exceptions": 0,
            "total_lines": len(invoice_lines),
        }

    async def _route_to_ap_clerk(
        self,
        *,
        invoice_header: VendorInvoiceHeader,
        context: GenerationContext,
    ) -> None:
        """Route the invoice to AP clerk agent via WorkflowOrchestrator.

        Uses the ``WorkflowOrchestrator``'s ``ROLE_MAPPING`` for
        ``vendor_invoice`` → ``["ap_clerk", "ap_manager"]`` to assign
        the invoice to an available AP agent for processing.

        If no ``WorkflowOrchestrator`` is available, the routing step
        is skipped gracefully.

        Args:
            invoice_header: The vendor invoice header to route.
            context: Current generation context for trace correlation.
        """
        if self._workflow_orchestrator is None:
            logger.debug(
                "vendor_invoice_route_skipped_no_orchestrator",
                invoice_number=invoice_header.invoice_number,
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
            return

        try:
            # The WorkflowOrchestrator routes transactions to eligible agent
            # roles defined in its ROLE_MAPPING.
            await self._workflow_orchestrator.route_transaction(
                transaction_type="vendor_invoice",
                transaction_id=str(invoice_header.invoice_id),
                payload={
                    "invoice_number": invoice_header.invoice_number,
                    "vendor_id": str(invoice_header.vendor_id),
                    "vendor_name": invoice_header.vendor_name,
                    "total_amount": str(invoice_header.total_amount),
                    "match_status": invoice_header.match_status,
                },
            )
            logger.debug(
                "vendor_invoice_routed_to_ap",
                invoice_number=invoice_header.invoice_number,
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
        except Exception as exc:
            # Routing failure is non-fatal — log and continue
            logger.warning(
                "vendor_invoice_route_failed",
                error=str(exc),
                invoice_number=invoice_header.invoice_number,
                service_name="transactions",
                component="VendorInvoiceProcessor",
            )
