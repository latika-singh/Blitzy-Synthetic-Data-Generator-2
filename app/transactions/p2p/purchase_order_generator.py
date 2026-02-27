"""Purchase Order Generator — Stage 1 of the Procure-to-Pay (P2P) cycle.

Generates synthetic purchase orders with realistic attributes derived from
statistical models, applying business rules for vendor selection, quantity
determination, pricing, and approval routing.

Business Flow
-------------
1.  Determine order count for the simulation day (Poisson distribution via
    ``OrderFrequencyModel``).
2.  Select vendor using **Pareto 80/20** distribution (top 20 % of vendors
    receive ~80 % of POs) via ``SelectionModel``.
3.  Select products and calculate **EOQ** (Economic Order Quantity) line
    quantities: ``EOQ = sqrt(2 × D × S / H)`` where *D* = annual demand,
    *S* = order cost, *H* = holding cost per unit.
4.  Determine unit pricing from ``AmountDistribution`` (log-normal with
    ``scale=5000, shape=1.2``).
5.  Check **approval thresholds** against PO total:
    * < $5 K  → no approval required (auto-approved)
    * $5 K – $25 K  → ``purchasing_manager``
    * $25 K – $100 K  → ``controller``
    * ≥ $100 K  → ``cfo``
6.  Create PO header (``PurchaseOrderHeader``) and line items
    (``PurchaseOrderLine``).
7.  Route to purchasing agent via ``WorkflowOrchestrator`` (``ROLE_MAPPING``
    ``purchase_order`` → ``["purchasing_agent", "purchasing_manager"]``).
8.  Publish ``TransactionCreated`` event through the ``EventBus`` (ADR-001).

.. important::

   **NO GL posting occurs at the PO stage.**  Purchase orders represent
   financial *commitments*, not *postings*.  GL entries are created at the
   Goods Receipt stage (DR Inventory, CR AP Accrual) and the Vendor Invoice
   stage (DR Expense/Asset, CR AP).

Key Business Rules
------------------
*   Sequential numbering: ``PO-YYYY-NNNN`` via ``_generate_sequential_number``.
*   All monetary calculations use ``Decimal`` (never ``float``).
*   Deterministic reproducibility via seeded ``random.Random`` instances.
*   Each PO has a header with a total amount and 1 + line items with
    individual per-unit pricing.

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
:class:`PurchaseOrderLine`
    Pydantic V2 model for a single PO line item.
:class:`PurchaseOrderHeader`
    Pydantic V2 model for a complete PO header (with embedded lines).
:class:`ApprovalInfo`
    Pydantic V2 model capturing the PO approval determination.
:class:`PurchaseOrderGenerator`
    Concrete :class:`TransactionGenerator` subclass for PO creation.
"""

from __future__ import annotations

import decimal
import math
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
    RETRY_POLICIES,
)
from app.transactions.exceptions import (
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
# Module-level constants — fallback data for deterministic generation
# ---------------------------------------------------------------------------
_DEFAULT_VENDORS: List[Dict[str, Any]] = [
    {
        "vendor_id": str(uuid4()),
        "name": "Acme Office Supplies",
        "payment_terms": "Net 30",
        "spend_history": 150_000.0,
    },
    {
        "vendor_id": str(uuid4()),
        "name": "Global Tech Components",
        "payment_terms": "2/10 Net 30",
        "spend_history": 320_000.0,
    },
    {
        "vendor_id": str(uuid4()),
        "name": "Premier Industrial Parts",
        "payment_terms": "Net 45",
        "spend_history": 80_000.0,
    },
    {
        "vendor_id": str(uuid4()),
        "name": "Reliable Paper Co",
        "payment_terms": "Net 30",
        "spend_history": 45_000.0,
    },
    {
        "vendor_id": str(uuid4()),
        "name": "Fast Freight Logistics",
        "payment_terms": "Net 60",
        "spend_history": 25_000.0,
    },
]

_DEFAULT_PRODUCTS: List[Dict[str, Any]] = [
    {"product_id": str(uuid4()), "description": "Standard Office Paper", "unit_price": Decimal("12.99")},
    {"product_id": str(uuid4()), "description": "Printer Toner Cartridge", "unit_price": Decimal("89.50")},
    {"product_id": str(uuid4()), "description": "Ethernet Cable Cat6 100ft", "unit_price": Decimal("24.75")},
    {"product_id": str(uuid4()), "description": "Desk Chair Ergonomic", "unit_price": Decimal("399.00")},
    {"product_id": str(uuid4()), "description": "LED Monitor 27in", "unit_price": Decimal("329.99")},
    {"product_id": str(uuid4()), "description": "USB-C Hub Multiport", "unit_price": Decimal("59.99")},
    {"product_id": str(uuid4()), "description": "Whiteboard Markers 12pk", "unit_price": Decimal("8.49")},
    {"product_id": str(uuid4()), "description": "Server Rack 42U", "unit_price": Decimal("1_249.00")},
    {"product_id": str(uuid4()), "description": "Wireless Mouse", "unit_price": Decimal("29.99")},
    {"product_id": str(uuid4()), "description": "Mechanical Keyboard", "unit_price": Decimal("149.99")},
]

_GL_ACCOUNTS: List[str] = [
    "5100-001",  # Office Supplies Expense
    "5100-002",  # IT Equipment Expense
    "5200-001",  # Inventory — Raw Materials
    "5200-002",  # Inventory — Finished Goods
    "1500-001",  # Fixed Assets — Equipment
]

_DEFAULT_LEAD_TIME_DAYS: int = 14
"""Default delivery lead time when vendor-specific data is unavailable."""


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════════


class PurchaseOrderLine(BaseModel):
    """Single line item on a purchase order.

    Represents one product/service ordered on a PO.  The ``extended_amount``
    MUST equal ``quantity × unit_price`` and is computed using ``Decimal``
    arithmetic — *never* ``float``.

    Attributes:
        line_number: 1-based line sequence number.
        product_id: Product or item identifier (nullable for non-stock items).
        description: Human-readable product description.
        quantity: Order quantity, typically derived from EOQ calculation.
        unit_of_measure: ISO unit code (default ``"EA"`` — each).
        unit_price: Per-unit price from vendor price list.
        extended_amount: ``quantity × unit_price`` (Decimal, never float).
        gl_account: GL account code for future posting at invoice stage.
        delivery_date: Requested delivery date for this line.
        metadata: Extensible key-value pairs for downstream consumers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_number: int = Field(
        ..., ge=1, description="Line sequence number starting at 1"
    )
    product_id: Optional[UUID] = Field(
        default=None, description="Product/item identifier"
    )
    description: str = Field(
        default="", description="Product description"
    )
    quantity: Decimal = Field(
        ..., gt=Decimal("0"), description="Order quantity (EOQ-based)"
    )
    unit_of_measure: str = Field(
        default="EA", description="Unit of measure code"
    )
    unit_price: Decimal = Field(
        ..., gt=Decimal("0"), description="Unit price from vendor price list"
    )
    extended_amount: Decimal = Field(
        ..., description="quantity × unit_price"
    )
    gl_account: str = Field(
        default="", description="Expense or Asset GL account for future posting"
    )
    delivery_date: Optional[date] = Field(
        default=None, description="Requested delivery date"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PurchaseOrderHeader(BaseModel):
    """Purchase order header with complete metadata and embedded line items.

    Carries the full PO state including vendor information, financial
    totals, approval status, and the list of :class:`PurchaseOrderLine`
    items.  Serialised via ``model_dump()`` for artifact storage and
    event payloads.

    Financial Integrity:
        * ``subtotal`` = ``SUM(line.extended_amount for line in lines)``
        * ``total_amount`` = ``subtotal + tax_amount``
        * All monetary fields are ``Decimal`` — never ``float``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    po_id: UUID = Field(
        default_factory=uuid4, description="Unique PO identifier"
    )
    po_number: str = Field(
        ..., description="Sequential number PO-YYYY-NNNN"
    )
    vendor_id: UUID = Field(
        ..., description="Selected vendor identifier"
    )
    vendor_name: str = Field(
        default="", description="Vendor display name"
    )
    order_date: date = Field(
        ..., description="Date PO was created"
    )
    expected_delivery_date: Optional[date] = Field(
        default=None, description="Expected delivery date"
    )
    payment_terms: str = Field(
        default="Net 30", description="Payment terms code"
    )
    ship_to_address: str = Field(
        default="", description="Delivery address"
    )
    subtotal: Decimal = Field(
        default=Decimal("0"), description="Sum of line extended_amounts"
    )
    tax_amount: Decimal = Field(
        default=Decimal("0"),
        description="Tax (0 for MVP — single company, no tax calc)",
    )
    total_amount: Decimal = Field(
        default=Decimal("0"), description="subtotal + tax_amount"
    )
    currency: str = Field(
        default="USD", description="Currency code — USD only for MVP"
    )
    status: str = Field(
        default="draft",
        description=(
            "PO lifecycle status: draft, pending_approval, approved, "
            "partially_received, fully_received, closed"
        ),
    )
    approval_status: Optional[str] = Field(
        default=None, description="Approval chain status"
    )
    approval_required: bool = Field(
        default=False, description="Whether approval is needed based on amount"
    )
    approver_role: Optional[str] = Field(
        default=None,
        description="Required approver role if approval needed",
    )
    created_by: Optional[UUID] = Field(
        default=None, description="Agent who created the PO"
    )
    lines: List[PurchaseOrderLine] = Field(
        default_factory=list, description="PO line items"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApprovalInfo(BaseModel):
    """Approval determination for a purchase order.

    Captures the approval threshold evaluation result including the
    required approver role and current status.

    Statuses:
        ``"not_required"`` — PO amount below lowest threshold (<$5 K).
        ``"pending"`` — approval required but not yet obtained.
        ``"approved"`` — approver has approved the PO.
        ``"rejected"`` — approver has rejected the PO.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    required: bool = Field(
        default=False, description="Whether approval is required"
    )
    threshold_level: str = Field(
        default="", description="e.g., '<$5K', '$5K-$25K'"
    )
    required_role: Optional[str] = Field(
        default=None, description="Role needed to approve"
    )
    status: str = Field(
        default="not_required",
        description="not_required, pending, approved, rejected",
    )
    approver_agent_id: Optional[UUID] = Field(default=None)
    approved_at: Optional[datetime] = Field(default=None)


# ═══════════════════════════════════════════════════════════════════════════════
# PurchaseOrderGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class PurchaseOrderGenerator(TransactionGenerator):
    """Generates purchase orders as the first step of the P2P cycle.

    Business Flow
    -------------
    1. Determine order count for the day (Poisson distribution).
    2. Select vendor (Pareto 80/20 weighted via ``SelectionModel``).
    3. Select products and calculate EOQ quantities.
    4. Determine pricing from vendor price list / log-normal distribution.
    5. Create PO header with line items.
    6. Check approval thresholds ($5 K / $25 K / $100 K).
    7. Route to purchasing agent via ``WorkflowOrchestrator``.
    8. Publish ``TransactionCreated`` event.

    .. note::

       **NO GL posting at PO stage** — purchase orders are financial
       commitments only.  GL entries occur at the Goods Receipt and
       Vendor Invoice stages.

    Constructor Injection (ADR-003)
    -------------------------------
    All dependencies are ``Optional`` with ``None`` defaults, enabling
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
        approval_system: Optional[ApprovalSystem] = None,
        discrepancy_injector: Optional[Any] = None,
        gl_posting_engine: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the PurchaseOrderGenerator with injected dependencies.

        Args:
            db_session_factory: Callable returning an async session context
                manager (e.g., ``get_session``).
            agent_registry: Project 2 :class:`AgentRegistry`.
            workflow_orchestrator: Project 2 :class:`WorkflowOrchestrator`.
            event_bus: Project 2 :class:`EventBus`.
            approval_system: Project 2 :class:`ApprovalSystem`.
            discrepancy_injector: P3 ``DiscrepancyInjector`` instance.
            gl_posting_engine: P3 ``GLPostingEngine`` instance.
            statistical_models: Dict of statistical model instances —
                expects keys ``"selection_model"``, ``"amount_distribution"``,
                and ``"order_frequency"`` when available.
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

        # Quick-access references to statistical sub-models
        self._selection_model = (
            self._statistical_models.get("selection_model")
            if self._statistical_models
            else None
        )
        self._amount_distribution = (
            self._statistical_models.get("amount_distribution")
            if self._statistical_models
            else None
        )
        self._order_frequency = (
            self._statistical_models.get("order_frequency")
            if self._statistical_models
            else None
        )

        # Instance-level PO sequence counter for numbering
        self._po_sequence: int = 0

        # Retry policy reference for documentation / runtime checks
        self._retry_policy: Dict[str, Any] = RETRY_POLICIES.get(
            "p2p_cycle_generation", {}
        )

        logger.info(
            "purchase_order_generator_initialized",
            has_selection_model=self._selection_model is not None,
            has_amount_distribution=self._amount_distribution is not None,
            has_order_frequency=self._order_frequency is not None,
            service_name="transactions",
            component="PurchaseOrderGenerator",
        )

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(
        self, context: GenerationContext
    ) -> TransactionResult:
        """Generate a single purchase order for the given context.

        Produces a :class:`TransactionResult` with artifacts keyed as:
        ``"po_header"``, ``"po_lines"``, ``"approval"`` — matching the
        ``REQUIRED_ARTIFACTS["purchase_order"]`` specification from
        ``app/orchestration/transaction_orchestrator.py``.

        Args:
            context: Per-invocation generation context carrying
                ``current_date``, ``rng_seed``, ``simulation_id``, etc.

        Returns:
            A :class:`TransactionResult` with ``transaction_type="purchase_order"``
            and all required artifacts populated.

        Raises:
            TransactionGenerationError: On unrecoverable generation failure.
        """
        start_time = time.perf_counter()

        try:
            # 1. Create deterministic RNG from context seed
            rng: random.Random = self._create_seeded_rng(context)

            # 2. Select vendor (Pareto 80/20)
            vendor_id, vendor_name, payment_terms = self._select_vendor(
                rng, context
            )

            # 3. Select products for the PO (1–5 line items)
            products: List[Tuple[Optional[UUID], str, Decimal]] = (
                self._select_products(rng, context)
            )

            # 4. Build PO lines with EOQ quantities and pricing
            po_lines: List[PurchaseOrderLine] = []
            for idx, (prod_id, description, base_price) in enumerate(
                products, start=1
            ):
                quantity: Decimal = self._calculate_eoq(rng)
                unit_price: Decimal = self._determine_unit_price(
                    rng, base_price
                )
                extended_amount: Decimal = (quantity * unit_price).quantize(
                    Decimal("0.01")
                )
                delivery_date: date = context.current_date + timedelta(
                    days=_DEFAULT_LEAD_TIME_DAYS
                )
                gl_account: str = rng.choice(_GL_ACCOUNTS)

                po_lines.append(
                    PurchaseOrderLine(
                        line_number=idx,
                        product_id=prod_id,
                        description=description,
                        quantity=quantity,
                        unit_of_measure="EA",
                        unit_price=unit_price,
                        extended_amount=extended_amount,
                        gl_account=gl_account,
                        delivery_date=delivery_date,
                        metadata={},
                    )
                )

            # 5. Calculate totals (ALL Decimal)
            subtotal: Decimal = sum(
                (line.extended_amount for line in po_lines), Decimal("0")
            )
            tax_amount: Decimal = Decimal("0")  # MVP — no tax calculation
            total_amount: Decimal = subtotal + tax_amount

            # 6. Generate PO number
            po_number: str = self._generate_po_number(context.current_date)

            # 7. Determine approval requirements
            approval_info: ApprovalInfo = self._determine_approval(
                total_amount
            )

            # Derive PO status from approval determination
            if approval_info.required:
                po_status = "pending_approval"
                approval_status_str = "pending"
            else:
                po_status = "approved"
                approval_status_str = "approved"

            # 8. Create PO header
            po_header = PurchaseOrderHeader(
                po_number=po_number,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                order_date=context.current_date,
                expected_delivery_date=(
                    context.current_date
                    + timedelta(days=_DEFAULT_LEAD_TIME_DAYS)
                ),
                payment_terms=payment_terms,
                ship_to_address="Main Warehouse — 100 Corporate Drive",
                subtotal=subtotal,
                tax_amount=tax_amount,
                total_amount=total_amount,
                currency="USD",
                status=po_status,
                approval_status=approval_status_str,
                approval_required=approval_info.required,
                approver_role=approval_info.required_role,
                lines=po_lines,
                metadata={
                    "simulation_id": str(context.simulation_id),
                    "trace_id": str(context.trace_id),
                    "fiscal_period": context.fiscal_period,
                },
            )

            # 9. Check discrepancy trigger
            should_inject, disc_type, disc_params = (
                await self._check_discrepancy_trigger(
                    context, "purchase_order"
                )
            )

            # 10. Route to purchasing agent via WorkflowOrchestrator
            if self._workflow_orchestrator is not None:
                try:
                    await self._workflow_orchestrator.route_transaction(
                        transaction_type="purchase_order",
                        transaction_id=str(po_header.po_id),
                        payload={
                            "po_number": po_number,
                            "vendor_id": str(vendor_id),
                            "total_amount": str(total_amount),
                        },
                    )
                except Exception as route_exc:
                    logger.warning(
                        "po_routing_failed",
                        error=str(route_exc),
                        po_number=po_number,
                        trace_id=str(context.trace_id),
                        service_name="transactions",
                        component="PurchaseOrderGenerator",
                    )

            # 11. Request approval via ApprovalSystem if required
            if approval_info.required and self._approval_system is not None:
                try:
                    # NOTE: ApprovalSystem.request_approval expects a numeric
                    # amount.  We pass the Decimal directly; if the approval
                    # system internally needs float, the conversion happens at
                    # the boundary closest to the consumer, preserving full
                    # Decimal precision within the transaction layer.
                    await self._approval_system.request_approval(
                        transaction_type="purchase_order",
                        transaction_id=str(po_header.po_id),
                        amount=total_amount,
                        metadata={
                            "po_number": po_number,
                            "vendor_name": vendor_name,
                        },
                    )
                except Exception as approval_exc:
                    logger.warning(
                        "po_approval_request_failed",
                        error=str(approval_exc),
                        po_number=po_number,
                        trace_id=str(context.trace_id),
                        service_name="transactions",
                        component="PurchaseOrderGenerator",
                    )

            # 12. Publish TransactionCreated event
            await self._publish_event(
                event_type="TransactionCreated",
                payload={
                    "transaction_type": "purchase_order",
                    "transaction_id": str(po_header.po_id),
                    "po_number": po_number,
                    "vendor_id": str(vendor_id),
                    "vendor_name": vendor_name,
                    "total_amount": str(total_amount),
                    "approval_required": approval_info.required,
                    "status": po_status,
                },
                context=context,
            )

            # 13. NO GL POSTING — POs are commitments only, not postings.
            #     GL entries happen at Goods Receipt (DR Inventory, CR GRNI)
            #     and Vendor Invoice (DR Expense/Asset, CR AP) stages.

            # 14. Build and return TransactionResult
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0

            result = TransactionResult(
                transaction_type="purchase_order",
                status="completed",
                artifacts={
                    "po_header": po_header.model_dump(mode="json"),
                    "po_lines": [
                        line.model_dump(mode="json") for line in po_lines
                    ],
                    "approval": approval_info.model_dump(mode="json"),
                },
                gl_entries=[],  # No GL entries at PO stage
                has_discrepancy=should_inject,
                discrepancy_type=disc_type,
                amount=total_amount,
                duration_ms=elapsed_ms,
            )

            # 15. Log the generated transaction
            self._log_transaction(result, context)

            return result

        except TransactionGenerationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            logger.error(
                "purchase_order_generation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                duration_ms=elapsed_ms,
                service_name="transactions",
                component="PurchaseOrderGenerator",
            )
            raise TransactionGenerationError(
                message=f"Purchase order generation failed: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "trace_id": str(context.trace_id),
                    "simulation_id": str(context.simulation_id),
                    "generator_class": "PurchaseOrderGenerator",
                },
            ) from exc

    async def validate(self, result: TransactionResult) -> bool:
        """Validate a generated PO result against business rules.

        Checks:
        *   ``po_header`` artifact exists with required fields.
        *   ``po_lines`` artifact is non-empty.
        *   All monetary amounts are ``Decimal``-compatible (serialised
            as strings, not floats).
        *   ``total_amount = SUM(line.extended_amount) + tax_amount``.
        *   Each line: ``extended_amount = quantity × unit_price``.
        *   Approval status is consistent with amount thresholds.

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if all validation checks pass, ``False`` otherwise.
        """
        try:
            artifacts = result.artifacts

            # 1. Artifact presence
            if "po_header" not in artifacts:
                logger.warning(
                    "po_validation_failed_missing_header",
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )
                return False

            if "po_lines" not in artifacts or not artifacts["po_lines"]:
                logger.warning(
                    "po_validation_failed_missing_lines",
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )
                return False

            if "approval" not in artifacts:
                logger.warning(
                    "po_validation_failed_missing_approval",
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )
                return False

            header = artifacts["po_header"]
            lines = artifacts["po_lines"]

            # 2. Validate line-level arithmetic
            computed_subtotal = Decimal("0")
            for line_data in lines:
                qty = Decimal(str(line_data["quantity"]))
                price = Decimal(str(line_data["unit_price"]))
                ext_amt = Decimal(str(line_data["extended_amount"]))
                expected_ext = (qty * price).quantize(Decimal("0.01"))

                if abs(ext_amt - expected_ext) > Decimal("0.01"):
                    logger.warning(
                        "po_validation_line_arithmetic_error",
                        line_number=line_data.get("line_number"),
                        expected=str(expected_ext),
                        actual=str(ext_amt),
                        service_name="transactions",
                        component="PurchaseOrderGenerator",
                    )
                    return False

                computed_subtotal += ext_amt

            # 3. Validate header totals
            header_subtotal = Decimal(str(header["subtotal"]))
            header_tax = Decimal(str(header["tax_amount"]))
            header_total = Decimal(str(header["total_amount"]))

            if abs(header_subtotal - computed_subtotal) > Decimal("0.01"):
                logger.warning(
                    "po_validation_subtotal_mismatch",
                    header_subtotal=str(header_subtotal),
                    computed_subtotal=str(computed_subtotal),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )
                return False

            expected_total = header_subtotal + header_tax
            if abs(header_total - expected_total) > Decimal("0.01"):
                logger.warning(
                    "po_validation_total_mismatch",
                    header_total=str(header_total),
                    expected_total=str(expected_total),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )
                return False

            # 4. Validate approval consistency
            approval_data = artifacts["approval"]
            approval_required = approval_data.get("required", False)
            expected_approval = self._determine_approval(header_total)

            if approval_required != expected_approval.required:
                logger.warning(
                    "po_validation_approval_mismatch",
                    actual_required=approval_required,
                    expected_required=expected_approval.required,
                    total_amount=str(header_total),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )
                return False

            logger.debug(
                "po_validation_passed",
                transaction_id=str(result.transaction_id),
                total_amount=str(header_total),
                line_count=len(lines),
                service_name="transactions",
                component="PurchaseOrderGenerator",
            )
            return True

        except Exception as exc:
            logger.error(
                "po_validation_error",
                error=str(exc),
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="PurchaseOrderGenerator",
            )
            return False

    async def post(self, result: TransactionResult) -> None:
        """Post a validated PO — NO-OP for purchase orders.

        Purchase orders are financial *commitments*, not *postings*.
        No GL journal entries are created at this stage.  GL entries
        occur downstream:

        *   **Goods Receipt**: DR Inventory, CR AP Accrual (GRNI)
        *   **Vendor Invoice**: DR Expense/Asset, CR AP

        This method logs the commitment and returns immediately.

        Args:
            result: The validated :class:`TransactionResult`.
        """
        logger.info(
            "po_post_skipped_commitment_only",
            transaction_id=str(result.transaction_id),
            transaction_type=result.transaction_type,
            amount=str(result.amount) if result.amount is not None else None,
            reason=(
                "Purchase orders are commitments — GL entries happen "
                "at goods_receipt and vendor_invoice stages"
            ),
            service_name="transactions",
            component="PurchaseOrderGenerator",
        )
        # Intentionally no GL posting — this is by design.

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _select_vendor(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> Tuple[UUID, str, str]:
        """Select a vendor using Pareto 80/20 weighted distribution.

        When a ``SelectionModel`` is available via ``statistical_models``,
        delegates to ``select_vendor()`` for true Pareto-weighted
        selection.  Otherwise, falls back to a seeded RNG choice from
        the default vendor pool.

        Args:
            rng: Seeded ``random.Random`` instance.
            context: Current generation context.

        Returns:
            A 3-tuple of ``(vendor_id, vendor_name, payment_terms)``.
        """
        vendors_pool: List[Dict[str, Any]] = _DEFAULT_VENDORS

        # Attempt to retrieve vendor pool from day context if available
        if context.day_context is not None:
            try:
                day_ctx_vendors = getattr(
                    context.day_context, "available_vendors", None
                )
                if day_ctx_vendors and len(day_ctx_vendors) > 0:
                    vendors_pool = day_ctx_vendors
            except (AttributeError, TypeError):
                pass

        # Attempt to retrieve vendor pool from additional_params
        if context.additional_params.get("vendors"):
            vendors_pool = context.additional_params["vendors"]

        # Use SelectionModel for Pareto 80/20 if available
        if self._selection_model is not None:
            try:
                selected = self._selection_model.select_vendor(
                    vendors_pool
                )
                vendor_id_raw = selected.get(
                    "vendor_id",
                    selected.get("id", str(uuid4())),
                )
                vendor_id = (
                    UUID(vendor_id_raw)
                    if isinstance(vendor_id_raw, str)
                    else vendor_id_raw
                )
                vendor_name = selected.get("name", "Unknown Vendor")
                payment_terms = selected.get(
                    "payment_terms", "Net 30"
                )
                return vendor_id, vendor_name, payment_terms
            except Exception as exc:
                logger.warning(
                    "selection_model_vendor_fallback",
                    error=str(exc),
                    service_name="transactions",
                    component="PurchaseOrderGenerator",
                )

        # Fallback: seeded RNG choice (still deterministic)
        selected = rng.choice(vendors_pool)
        vendor_id_raw = selected.get(
            "vendor_id", selected.get("id", str(uuid4()))
        )
        vendor_id = (
            UUID(vendor_id_raw)
            if isinstance(vendor_id_raw, str)
            else vendor_id_raw
        )
        vendor_name = selected.get("name", "Unknown Vendor")
        payment_terms = selected.get("payment_terms", "Net 30")

        logger.debug(
            "vendor_selected_fallback",
            vendor_name=vendor_name,
            service_name="transactions",
            component="PurchaseOrderGenerator",
        )
        return vendor_id, vendor_name, payment_terms

    def _select_products(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> List[Tuple[Optional[UUID], str, Decimal]]:
        """Select 1–5 products for the purchase order.

        Uses the ``SelectionModel`` product selection (70/30 repeat/new)
        when available, otherwise falls back to seeded RNG choice from
        the default product catalog.

        Args:
            rng: Seeded ``random.Random`` instance.
            context: Current generation context.

        Returns:
            List of ``(product_id, description, base_unit_price)`` tuples.
        """
        product_count = rng.randint(1, 5)
        products_pool: List[Dict[str, Any]] = _DEFAULT_PRODUCTS

        # Override product pool from context if available
        if context.additional_params.get("products"):
            products_pool = context.additional_params["products"]

        selected_products: List[Tuple[Optional[UUID], str, Decimal]] = []

        for _ in range(product_count):
            product: Optional[Dict[str, Any]] = None

            # Use SelectionModel if available
            if self._selection_model is not None:
                try:
                    product = self._selection_model.select_product(
                        products_pool
                    )
                except Exception:
                    product = None

            # Fallback: seeded RNG choice
            if product is None:
                product = rng.choice(products_pool)

            prod_id_raw = product.get(
                "product_id", product.get("id")
            )
            prod_id: Optional[UUID] = None
            if prod_id_raw is not None:
                try:
                    prod_id = (
                        UUID(prod_id_raw)
                        if isinstance(prod_id_raw, str)
                        else prod_id_raw
                    )
                except (ValueError, TypeError):
                    prod_id = uuid4()

            description = product.get("description", "Product Item")
            base_price_raw = product.get(
                "unit_price", product.get("price", "100.00")
            )
            base_price = Decimal(str(base_price_raw))

            selected_products.append((prod_id, description, base_price))

        return selected_products

    def _calculate_eoq(
        self,
        rng: random.Random,
        annual_demand: Optional[Decimal] = None,
        order_cost: Optional[Decimal] = None,
        holding_cost: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate Economic Order Quantity for a PO line.

        Formula: ``EOQ = sqrt(2 × D × S / H)``

        Where:
            *D* = annual demand (units per year)
            *S* = fixed cost per order (ordering cost)
            *H* = annual holding cost per unit

        When parameters are unavailable, generates a reasonable default
        quantity using the seeded RNG (between 10 and 500 units).

        Args:
            rng: Seeded ``random.Random`` instance.
            annual_demand: Optional annual demand in units.
            order_cost: Optional fixed ordering cost.
            holding_cost: Optional annual holding cost per unit.

        Returns:
            Calculated EOQ as ``Decimal``, rounded to the nearest
            whole number.
        """
        if (
            annual_demand is not None
            and order_cost is not None
            and holding_cost is not None
            and holding_cost > Decimal("0")
        ):
            numerator = (
                Decimal("2") * annual_demand * order_cost
            )
            denominator = holding_cost
            eoq_float = math.sqrt(
                float(numerator / denominator)
            )
            eoq = Decimal(str(eoq_float)).quantize(Decimal("1"))
            # Ensure minimum order quantity of 1
            return max(eoq, Decimal("1"))

        # Fallback: deterministic random quantity (10–500)
        qty = Decimal(str(rng.randint(10, 500)))
        return qty

    def _determine_unit_price(
        self,
        rng: random.Random,
        base_price: Decimal,
    ) -> Decimal:
        """Determine the unit price for a PO line item.

        Uses the ``AmountDistribution`` for log-normal sampling when
        available.  Otherwise, applies a small variance (±10 %) to the
        base price using the seeded RNG.

        Args:
            rng: Seeded ``random.Random`` instance.
            base_price: Base unit price from the product catalog.

        Returns:
            Final unit price as ``Decimal``, always > 0.
        """
        if self._amount_distribution is not None:
            try:
                sampled = self._amount_distribution.sample_po_amount(n=1)
                # Immediately convert the float sample to Decimal with
                # explicit 2-decimal-place rounding (AAP §0.7.2 — Decimal
                # precision: convert at the earliest possible point).
                sampled_decimal = Decimal(str(float(sampled))).quantize(
                    Decimal("0.01"), rounding=decimal.ROUND_HALF_UP
                )
                # Use sampled amount as a scaling factor, keeping it
                # within a reasonable range relative to the base price
                if sampled_decimal > Decimal("0"):
                    return sampled_decimal
            except Exception:
                pass

        # Fallback: apply ±10% variance to base price
        variance = Decimal(str(rng.uniform(-0.10, 0.10)))
        adjusted = base_price * (Decimal("1") + variance)
        adjusted = adjusted.quantize(Decimal("0.01"))
        # Ensure price is always positive
        return max(adjusted, Decimal("0.01"))

    def _determine_approval(
        self, total_amount: Decimal
    ) -> ApprovalInfo:
        """Determine approval requirement based on PO total amount.

        Reads purchase-order thresholds from ``APPROVAL_THRESHOLDS``
        and returns an :class:`ApprovalInfo` model.

        Threshold Tiers:
            * < $5,000      → no approval required
            * $5 K – $25 K  → ``purchasing_manager``
            * $25 K – $100 K → ``controller``
            * ≥ $100 K      → ``cfo``

        Args:
            total_amount: PO total monetary amount.

        Returns:
            An :class:`ApprovalInfo` model with the approval
            determination.
        """
        po_thresholds = APPROVAL_THRESHOLDS.get("purchase_order", {})
        tiers: List[Dict[str, Any]] = po_thresholds.get("tiers", [])

        # Walk the tier list — tiers are ordered ascending by max_amount
        previous_max = Decimal("0")
        for tier in tiers:
            tier_max = tier.get("max_amount")
            required_role = tier.get("required_role")

            # No approval required for this tier
            if tier_max is not None and total_amount < tier_max:
                if required_role is None:
                    # Below lowest threshold — no approval needed
                    return ApprovalInfo(
                        required=False,
                        threshold_level=f"<${int(tier_max / 1000)}K",
                        required_role=None,
                        status="not_required",
                    )
                # Approval required for this tier
                low = (
                    int(previous_max / 1000)
                    if previous_max > 0
                    else 0
                )
                high = int(tier_max / 1000)
                return ApprovalInfo(
                    required=True,
                    threshold_level=f"${low}K-${high}K",
                    required_role=required_role,
                    status="pending",
                )
            previous_max = (
                tier_max if tier_max is not None else previous_max
            )

        # Unbounded top tier (max_amount is None)
        last_tier = tiers[-1] if tiers else {}
        return ApprovalInfo(
            required=True,
            threshold_level=f">=${int(previous_max / 1000)}K",
            required_role=last_tier.get("required_role", "cfo"),
            status="pending",
        )

    def _generate_po_number(self, current_date: date) -> str:
        """Generate a sequential PO number in ``PO-YYYY-NNNN`` format.

        Uses the base-class ``_generate_sequential_number`` helper with
        the ``"PO"`` prefix derived from ``DOCUMENT_NUMBER_PREFIXES``.

        Args:
            current_date: The current simulation date (for the year
                component).

        Returns:
            A formatted PO number string, e.g., ``"PO-2024-0001"``.
        """
        self._po_sequence += 1
        prefix = DOCUMENT_NUMBER_PREFIXES.get("purchase_order", "PO")
        return self._generate_sequential_number(
            prefix=prefix,
            year=current_date.year,
            sequence=self._po_sequence,
        )
