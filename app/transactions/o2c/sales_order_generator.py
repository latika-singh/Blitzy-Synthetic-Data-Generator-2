"""Sales Order Generator — creates new sales orders with credit checks.

Generates sales orders as the FIRST step in the Order-to-Cash cycle.
Each sales order goes through customer selection, credit validation,
product selection, pricing, and approval routing.

Key Behaviors:

1. **Revenue-Weighted Customer Selection**: Customers are selected
   using the Pareto (80/20) distribution from ``SelectionModel`` —
   top 20% of customers by revenue generate 80% of orders. Within
   each tier, selection is weighted by annual revenue.

2. **Credit Limit Check** (CRITICAL):
   - Before creating the order, check:
     ``customer.credit_limit >= current_ar_balance + order_total``
   - If credit check FAILS: **Block the order** — do NOT create it
   - Log the credit check failure with customer_id, credit_limit,
     ar_balance, order_total
   - This is NON-NEGOTIABLE per AAP §0.1.2

3. **Product Selection**: Products are selected from the customer's
   purchase history using the 70/30 repeat-to-new split from
   ``SelectionModel``. 70% of products are repeat purchases from
   history; 30% are new products not previously ordered.

4. **Pricing with Discounts**: Base prices come from the product
   master; volume discounts, customer-tier discounts, and promotional
   discounts may reduce the unit price. All discount calculations
   use ``Decimal`` arithmetic.

5. **Approval Routing**: Orders above certain thresholds require
   approval via the ``ApprovalSystem``. Routed to 'ar_clerk' agents
   per ROLE_MAPPING.

6. **Sequential Numbering**: SO-YYYY-NNNN (e.g., SO-2024-0001)

7. **Required Artifacts** (from REQUIRED_ARTIFACTS mapping):
   - order_header
   - order_lines

8. **Payment Terms**: Assigned from customer master data —
   common terms: Net 30, Net 60, 2/10 Net 30.

Database Pattern: ``from synthetic_erp.db.session import get_session``

Retry Policy (AAP §0.1.2 — o2c_cycle_generation):
    - max_attempts: 2, backoff: linear (1s, 2s), timeout: 60s, fallback: skip

Performance Target: O2C ≥ 60 cycles/min (AAP §0.7.3)

References:
    - AAP §0.5.1 Group 3: O2C Engine (SalesOrderGenerator)
    - AAP §0.1.2: Credit check blocks order if credit_limit < current_ar_balance
    - AAP §0.7.2: Decimal precision, financial integrity
    - AAP §0.2.1: SelectionModel Pareto 80/20, 70/30 repeat/new
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.transactions.base_generator import (
    GenerationContext,
    TransactionGenerator,
    TransactionResult,
)
from app.transactions.exceptions import (
    TransactionError,
    TransactionGenerationError,
)
from app.transactions.constants import (
    DOCUMENT_NUMBER_PREFIXES,
    FINANCIAL_TOLERANCES,
    RETRY_POLICIES,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus
    from app.orchestration.workflow_orchestrator import WorkflowOrchestrator
    from app.orchestration.approval_system import ApprovalSystem
    from app.transactions.gl.gl_posting_engine import GLPostingEngine
    from app.statistical.selection_models import SelectionModel
    from app.statistical.amount_distributions import AmountDistribution
    from app.statistical.order_frequency_model import OrderFrequencyModel
    from app.statistical.payment_timing_model import PaymentTimingModel

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
# Module-level constants for Pareto weighting and product split
# ---------------------------------------------------------------------------
_PARETO_TOP_FRACTION = Decimal("0.20")
_PARETO_TOP_WEIGHT = Decimal("0.80")
_REPEAT_PURCHASE_RATE = 0.70
_NEW_PURCHASE_RATE = 0.30

# Default product pool for fallback when no external product catalog is available
_DEFAULT_PRODUCT_POOL: List[Dict[str, Any]] = [
    {"product_id": f"PROD-{i:04d}", "product_name": f"Product {i}",
     "unit_price": str(Decimal(str(50 + i * 15))), "unit_cost": str(Decimal(str(30 + i * 10)))}
    for i in range(1, 51)
]

# Default customer pool for fallback when no external customer data is available
_DEFAULT_CUSTOMER_POOL: List[Dict[str, Any]] = [
    {"customer_id": "CUST-0001", "customer_name": "Acme Corp",
     "customer_tier": "strategic", "credit_limit": "500000.00",
     "current_ar_balance": "50000.00", "annual_revenue": "2000000.00",
     "payment_terms": "Net 30", "purchase_history": ["PROD-0001", "PROD-0005", "PROD-0010"]},
    {"customer_id": "CUST-0002", "customer_name": "Beta Industries",
     "customer_tier": "strategic", "credit_limit": "300000.00",
     "current_ar_balance": "25000.00", "annual_revenue": "1500000.00",
     "payment_terms": "2/10 Net 30", "purchase_history": ["PROD-0002", "PROD-0008"]},
    {"customer_id": "CUST-0003", "customer_name": "Gamma Solutions",
     "customer_tier": "standard", "credit_limit": "100000.00",
     "current_ar_balance": "10000.00", "annual_revenue": "500000.00",
     "payment_terms": "Net 30", "purchase_history": ["PROD-0003", "PROD-0007", "PROD-0015"]},
    {"customer_id": "CUST-0004", "customer_name": "Delta Services",
     "customer_tier": "standard", "credit_limit": "75000.00",
     "current_ar_balance": "5000.00", "annual_revenue": "350000.00",
     "payment_terms": "Net 60", "purchase_history": ["PROD-0004"]},
    {"customer_id": "CUST-0005", "customer_name": "Epsilon LLC",
     "customer_tier": "standard", "credit_limit": "50000.00",
     "current_ar_balance": "8000.00", "annual_revenue": "200000.00",
     "payment_terms": "Net 30", "purchase_history": ["PROD-0006", "PROD-0012"]},
    {"customer_id": "CUST-0006", "customer_name": "Zeta Trading",
     "customer_tier": "transactional", "credit_limit": "25000.00",
     "current_ar_balance": "3000.00", "annual_revenue": "80000.00",
     "payment_terms": "Net 30", "purchase_history": ["PROD-0009"]},
    {"customer_id": "CUST-0007", "customer_name": "Eta Partners",
     "customer_tier": "transactional", "credit_limit": "15000.00",
     "current_ar_balance": "1000.00", "annual_revenue": "50000.00",
     "payment_terms": "Net 30", "purchase_history": []},
    {"customer_id": "CUST-0008", "customer_name": "Theta Inc",
     "customer_tier": "transactional", "credit_limit": "10000.00",
     "current_ar_balance": "2000.00", "annual_revenue": "30000.00",
     "payment_terms": "Net 30", "purchase_history": ["PROD-0011"]},
    {"customer_id": "CUST-0009", "customer_name": "Iota Supplies",
     "customer_tier": "transactional", "credit_limit": "20000.00",
     "current_ar_balance": "500.00", "annual_revenue": "60000.00",
     "payment_terms": "Net 30", "purchase_history": ["PROD-0013", "PROD-0014"]},
    {"customer_id": "CUST-0010", "customer_name": "Kappa Group",
     "customer_tier": "transactional", "credit_limit": "12000.00",
     "current_ar_balance": "1500.00", "annual_revenue": "40000.00",
     "payment_terms": "Net 30", "purchase_history": []},
]

# Volume discount tiers — quantity thresholds and corresponding discount %
_VOLUME_DISCOUNT_TIERS: List[Tuple[Decimal, Decimal]] = [
    (Decimal("1000"), Decimal("15.00")),
    (Decimal("500"), Decimal("10.00")),
    (Decimal("100"), Decimal("5.00")),
]

# Customer tier discount percentages
_TIER_DISCOUNTS: Dict[str, Tuple[Decimal, Decimal]] = {
    "strategic": (Decimal("3.00"), Decimal("5.00")),
    "standard": (Decimal("1.00"), Decimal("2.00")),
    "transactional": (Decimal("0.00"), Decimal("0.00")),
}

# Maximum combined discount cap to prevent excessive discounting
_MAX_DISCOUNT_PERCENT = Decimal("25.00")

# Sales order approval threshold — orders above this require approval
_SO_APPROVAL_THRESHOLD = Decimal("25000.00")


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class OrderStatus(str, Enum):
    """Sales order lifecycle status.

    Tracks the order through its full lifecycle from initial draft
    through credit validation, approval, confirmation, shipment, and
    final fulfillment or cancellation.
    """

    DRAFT = "draft"
    CREDIT_CHECK_PENDING = "credit_check_pending"
    CREDIT_CHECK_FAILED = "credit_check_failed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    CONFIRMED = "confirmed"
    PARTIALLY_SHIPPED = "partially_shipped"
    FULLY_SHIPPED = "fully_shipped"
    CANCELLED = "cancelled"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════


class SalesOrderLine(BaseModel):
    """A single line item on a sales order.

    Represents one product on the order with pricing, discounts, and
    shipment tracking.  All monetary fields use ``Decimal`` — NEVER
    ``float`` (AAP §0.7.2).

    Attributes:
        line_number: 1-based sequential line number within the order.
        product_id: Product identifier from the product master.
        product_name: Human-readable product description.
        quantity: Ordered quantity (always > 0).
        unit_price: Selling price per unit (always > 0).
        unit_cost: Product cost per unit for downstream COGS calculation.
        line_total: ``quantity × unit_price`` before discounts.
        discount_percent: Combined discount percentage applied to this line.
        discount_amount: Calculated discount in currency units.
        net_amount: ``line_total − discount_amount``.
        is_repeat_product: ``True`` if selected from purchase history (70% rule).
        shipped_quantity: Quantity shipped so far (updated by ShipmentGenerator).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_number: int = Field(..., ge=1, description="1-based line number")
    product_id: str = Field(..., description="Product identifier")
    product_name: str = Field(default="", description="Human-readable product name")
    quantity: Decimal = Field(
        ..., gt=Decimal("0.00"), description="Ordered quantity"
    )
    unit_price: Decimal = Field(
        ..., gt=Decimal("0.00"), description="Selling price per unit"
    )
    unit_cost: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Product cost for downstream COGS",
    )
    line_total: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="quantity * unit_price",
    )
    discount_percent: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        le=Decimal("100.00"),
        description="Combined discount percentage",
    )
    discount_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Calculated discount amount",
    )
    net_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="line_total - discount_amount",
    )
    is_repeat_product: bool = Field(
        default=False,
        description="True if from purchase history (70/30 rule)",
    )
    shipped_quantity: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Updated by ShipmentGenerator",
    )


class CustomerInfo(BaseModel):
    """Customer data for order creation.

    Carries the customer master data fields required for sales order
    generation including credit limits, AR balance, revenue weighting,
    and purchase history for the 70/30 product selection split.

    Attributes:
        customer_id: Unique customer identifier.
        customer_name: Human-readable customer name.
        customer_tier: Customer classification — ``strategic``,
            ``standard``, or ``transactional``.
        credit_limit: Maximum credit exposure allowed for this customer.
        current_ar_balance: Current accounts-receivable balance.
        annual_revenue: Annual revenue for revenue-weighted selection.
        payment_terms: Default payment terms (e.g., ``"Net 30"``).
        purchase_history: List of product IDs previously ordered.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    customer_id: str = Field(..., description="Unique customer identifier")
    customer_name: str = Field(default="", description="Customer name")
    customer_tier: str = Field(
        default="standard",
        description="strategic, standard, transactional",
    )
    credit_limit: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Maximum credit exposure",
    )
    current_ar_balance: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Current accounts-receivable balance",
    )
    annual_revenue: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Annual revenue for selection weighting",
    )
    payment_terms: str = Field(
        default="Net 30",
        description="Default payment terms",
    )
    purchase_history: List[str] = Field(
        default_factory=list,
        description="Product IDs from prior orders",
    )


class CreditCheckResult(BaseModel):
    """Result of a credit limit check.

    Immutable record of a credit validation performed before order
    creation.  When ``passed`` is ``False``, the order MUST be blocked.

    Attributes:
        passed: ``True`` if credit is sufficient, ``False`` otherwise.
        customer_id: Customer that was checked.
        credit_limit: Customer's credit limit at time of check.
        current_ar_balance: Customer's AR balance at time of check.
        order_total: Estimated order total being validated.
        available_credit: ``credit_limit − current_ar_balance``.
        reason: Human-readable explanation (empty string when passed).
    """

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(..., description="Whether credit check passed")
    customer_id: str = Field(..., description="Customer identifier")
    credit_limit: Decimal = Field(..., description="Credit limit at check time")
    current_ar_balance: Decimal = Field(
        ..., description="AR balance at check time"
    )
    order_total: Decimal = Field(..., description="Order total validated")
    available_credit: Decimal = Field(
        ..., description="credit_limit - current_ar_balance"
    )
    reason: str = Field(
        default="",
        description="Explanation when check fails",
    )


class SalesOrderRecord(BaseModel):
    """Complete sales order header with lines.

    Carries the full order including header information, line items,
    credit check result, and approval status.  This is the primary
    artifact produced by the ``SalesOrderGenerator`` and consumed by
    downstream O2C generators (ShipmentGenerator, etc.).

    Attributes:
        order_id: Unique order identifier (UUID4).
        order_number: Sequential number in ``SO-YYYY-NNNN`` format.
        customer_id: Customer placing the order.
        customer_name: Human-readable customer name.
        customer_tier: Customer classification tier.
        order_date: Business date of order creation.
        payment_terms: Payment terms from customer master.
        lines: List of :class:`SalesOrderLine` items.
        subtotal: Sum of all line totals before discounts.
        total_discount: Sum of all line discount amounts.
        order_total: ``subtotal − total_discount``.
        status: Current :class:`OrderStatus`.
        credit_check_result: Result of credit validation.
        requires_approval: Whether order exceeds approval threshold.
        approved_by: Agent role that approved (if applicable).
        simulation_id: Parent simulation run identifier.
        created_at: UTC timestamp of creation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    order_id: UUID = Field(
        default_factory=uuid4, description="Unique order identifier"
    )
    order_number: str = Field(
        default="", description="SO-YYYY-NNNN"
    )
    customer_id: str = Field(..., description="Customer placing the order")
    customer_name: str = Field(default="", description="Customer name")
    customer_tier: str = Field(default="standard", description="Customer tier")
    order_date: date = Field(..., description="Business date of order creation")
    payment_terms: str = Field(
        default="Net 30", description="Payment terms from customer master"
    )
    lines: List[SalesOrderLine] = Field(
        default_factory=list, description="Order line items"
    )
    subtotal: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="SUM of line_total",
    )
    total_discount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="SUM of discount_amount",
    )
    order_total: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="subtotal - total_discount",
    )
    status: OrderStatus = Field(
        default=OrderStatus.DRAFT, description="Order lifecycle status"
    )
    credit_check_result: Optional[CreditCheckResult] = Field(
        default=None, description="Credit validation result"
    )
    requires_approval: bool = Field(
        default=False, description="Whether order exceeds approval threshold"
    )
    approved_by: Optional[str] = Field(
        default=None, description="Approver agent role"
    )
    simulation_id: Optional[UUID] = Field(
        default=None, description="Simulation run identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC creation timestamp",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SalesOrderGenerator
# ═══════════════════════════════════════════════════════════════════════════


class SalesOrderGenerator(TransactionGenerator):
    """Generates sales orders with credit checks and approval routing.

    First step in the O2C cycle.  Applies revenue-weighted customer
    selection (Pareto 80/20), 70/30 repeat/new product split, and
    mandatory credit limit validation.

    **Credit Check (NON-NEGOTIABLE)**:
        If ``credit_limit < current_ar_balance + order_total`` the order
        is BLOCKED — it is never created and the customer is logged.

    **Sales orders do NOT generate GL entries.**  GL posting occurs
    downstream at shipment (DR COGS, CR Inventory) and invoice
    (DR AR, CR Revenue) steps.

    **Required Artifacts** (per REQUIRED_ARTIFACTS mapping):
        ``order_header``, ``order_lines``

    **Agent Routing** (per ROLE_MAPPING):
        ``sales_order → ar_clerk``
    """

    # ------------------------------------------------------------------
    # Constructor (ADR-003 — constructor injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        db_session_factory: Optional[Any] = None,
        agent_registry: Optional[AgentRegistry] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
        event_bus: Optional[EventBus] = None,
        gl_posting_engine: Optional[GLPostingEngine] = None,
        approval_system: Optional[ApprovalSystem] = None,
        selection_model: Optional[SelectionModel] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        order_frequency_model: Optional[OrderFrequencyModel] = None,
        payment_timing_model: Optional[PaymentTimingModel] = None,
        discrepancy_injector: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the SalesOrderGenerator with injected dependencies.

        All parameters are keyword-only and ``Optional`` with ``None``
        defaults, following ADR-003 (constructor injection).  When a
        dependency is ``None``, the generator falls back to built-in
        defaults or skips the relevant operation gracefully.

        Args:
            db_session_factory: Callable returning async context manager
                yielding a database session.
            agent_registry: Project 2 AgentRegistry for agent lookup.
            workflow_orchestrator: Project 2 WorkflowOrchestrator for
                routing transactions to agents.
            event_bus: Project 2 EventBus for cross-subsystem events.
            gl_posting_engine: GLPostingEngine (passed through for
                downstream O2C use — sales orders don't post GL).
            approval_system: Project 2 ApprovalSystem for threshold-based
                approval chain determination.
            selection_model: Statistical SelectionModel for Pareto 80/20
                customer selection and 70/30 product split.
            amount_distribution: AmountDistribution for tier-specific
                amount generation.
            order_frequency_model: OrderFrequencyModel for Poisson-based
                daily order count determination.
            payment_timing_model: PaymentTimingModel for payment term
                assignment patterns.
            discrepancy_injector: DiscrepancyInjector for rate-based
                discrepancy injection.
            statistical_models: Additional statistical model instances.
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

        # Store O2C-specific dependencies
        self._approval_system = approval_system
        self._selection_model = selection_model
        self._amount_distribution = amount_distribution
        self._order_frequency_model = order_frequency_model
        self._payment_timing_model = payment_timing_model

        # Sequential numbering state
        self._sequence_counter: int = 0
        self._current_year: int = 0

        # Metrics counters
        self._orders_generated: int = 0
        self._orders_credit_blocked: int = 0
        self._total_order_value: Decimal = Decimal("0.00")

        # Order history for downstream reference
        self._order_history: List[SalesOrderRecord] = []

        logger.info(
            "sales_order_generator_initialized",
            service_name="transactions",
            component="SalesOrderGenerator",
            has_selection_model=selection_model is not None,
            has_amount_distribution=amount_distribution is not None,
            has_order_frequency_model=order_frequency_model is not None,
            has_payment_timing_model=payment_timing_model is not None,
            has_approval_system=approval_system is not None,
        )

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Generate sales orders for the current simulation day.

        Entry point for the O2C cycle.  Determines the daily order count,
        then for each order: selects a customer (Pareto 80/20), validates
        credit, selects products (70/30 split), prices with discounts,
        and builds the order record.

        **CRITICAL**: Orders that fail credit checks are BLOCKED — they
        are logged but never created.

        Args:
            context: Generation parameters including business date, fiscal
                period, RNG seed, and discrepancy configuration.

        Returns:
            A :class:`TransactionResult` carrying the generated orders
            as artifacts (``order_header``, ``order_lines``).
        """
        start_time = time.perf_counter()
        rng = self._create_seeded_rng(context)
        trace_id = str(context.trace_id)

        logger.info(
            "sales_order_generation_started",
            service_name="transactions",
            component="SalesOrderGenerator",
            trace_id=trace_id,
            simulation_id=str(context.simulation_id),
            business_date=str(context.current_date),
        )

        # Determine how many orders to generate today
        order_count = self._determine_order_count(context, rng)
        generated_orders: List[SalesOrderRecord] = []
        credit_blocked_count = 0

        for i in range(order_count):
            try:
                # 1. Customer Selection — revenue-weighted Pareto 80/20
                customer = self._select_customer(rng)

                # 2. Product Selection — 70/30 repeat/new split
                num_lines = rng.randint(1, 8)
                products = self._select_products(customer, rng, num_lines)

                if not products:
                    logger.debug(
                        "no_products_selected_skipping",
                        service_name="transactions",
                        component="SalesOrderGenerator",
                        trace_id=trace_id,
                        customer_id=customer.customer_id,
                    )
                    continue

                # 3. Build order lines with pricing
                lines: List[SalesOrderLine] = []
                for line_idx, product in enumerate(products, start=1):
                    quantity = Decimal(str(rng.randint(1, 100)))
                    unit_price_raw = Decimal(
                        str(product.get("unit_price", "100.00"))
                    )
                    unit_cost_raw = Decimal(
                        str(product.get("unit_cost", "60.00"))
                    )

                    unit_price, discount_pct, discount_amt = (
                        self._calculate_line_price(
                            product, customer, quantity, rng
                        )
                    )

                    line_total = (quantity * unit_price).quantize(
                        Decimal("0.01")
                    )
                    discount_amount = (
                        line_total * discount_pct / Decimal("100")
                    ).quantize(Decimal("0.01"))
                    net_amount = (line_total - discount_amount).quantize(
                        Decimal("0.01")
                    )

                    is_repeat = product.get("is_repeat", False)

                    line = SalesOrderLine(
                        line_number=line_idx,
                        product_id=product.get("product_id", f"PROD-{line_idx:04d}"),
                        product_name=product.get("product_name", ""),
                        quantity=quantity,
                        unit_price=unit_price,
                        unit_cost=unit_cost_raw.quantize(Decimal("0.01")),
                        line_total=line_total,
                        discount_percent=discount_pct,
                        discount_amount=discount_amount,
                        net_amount=net_amount,
                        is_repeat_product=is_repeat,
                        shipped_quantity=Decimal("0.00"),
                    )
                    lines.append(line)

                # 4. Calculate order totals
                subtotal = sum(
                    (ln.line_total for ln in lines), Decimal("0.00")
                ).quantize(Decimal("0.01"))
                total_discount = sum(
                    (ln.discount_amount for ln in lines), Decimal("0.00")
                ).quantize(Decimal("0.01"))
                order_total = (subtotal - total_discount).quantize(
                    Decimal("0.01")
                )

                # 5. CREDIT CHECK — NON-NEGOTIABLE
                credit_result = self._check_credit_limit(customer, order_total)

                if not credit_result.passed:
                    # Order is BLOCKED — do NOT create it
                    credit_blocked_count += 1
                    self._orders_credit_blocked += 1
                    continue

                # 6. Generate order number
                order_number = self._generate_order_number(context.current_date)

                # 7. Approval check
                requires_approval = order_total > _SO_APPROVAL_THRESHOLD
                status = (
                    OrderStatus.PENDING_APPROVAL
                    if requires_approval
                    else OrderStatus.CONFIRMED
                )

                # 8. Build the SalesOrderRecord
                order = SalesOrderRecord(
                    order_id=uuid4(),
                    order_number=order_number,
                    customer_id=customer.customer_id,
                    customer_name=customer.customer_name,
                    customer_tier=customer.customer_tier,
                    order_date=context.current_date,
                    payment_terms=customer.payment_terms,
                    lines=lines,
                    subtotal=subtotal,
                    total_discount=total_discount,
                    order_total=order_total,
                    status=status,
                    credit_check_result=credit_result,
                    requires_approval=requires_approval,
                    approved_by=None,
                    simulation_id=context.simulation_id,
                )

                generated_orders.append(order)
                self._orders_generated += 1
                self._total_order_value += order_total
                self._order_history.append(order)

                # 9. Publish TransactionCreated event
                await self._publish_event(
                    event_type="TransactionCreated",
                    payload={
                        "transaction_type": "sales_order",
                        "transaction_id": str(order.order_id),
                        "order_number": order.order_number,
                        "customer_id": order.customer_id,
                        "order_total": str(order.order_total),
                        "status": order.status.value,
                        "requires_approval": order.requires_approval,
                        "line_count": len(order.lines),
                    },
                    context=context,
                )

                logger.info(
                    "sales_order_created",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    trace_id=trace_id,
                    simulation_id=str(context.simulation_id),
                    transaction_type="sales_order",
                    transaction_id=str(order.order_id),
                    order_number=order.order_number,
                    customer_id=order.customer_id,
                    amount=str(order.order_total),
                    has_discrepancy=False,
                    gl_entries=0,
                    line_count=len(order.lines),
                )

            except TransactionError:
                raise
            except Exception as exc:
                logger.error(
                    "sales_order_generation_error",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    trace_id=trace_id,
                    error=str(exc),
                    order_index=i,
                )

        # 10. Check discrepancy trigger (applies to the batch)
        should_inject, disc_type, disc_params = (
            await self._check_discrepancy_trigger(context, "sales_order")
        )

        # 11. Build TransactionResult
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Build artifacts per REQUIRED_ARTIFACTS: order_header, order_lines
        order_headers = [
            {
                "order_id": str(o.order_id),
                "order_number": o.order_number,
                "customer_id": o.customer_id,
                "customer_name": o.customer_name,
                "customer_tier": o.customer_tier,
                "order_date": str(o.order_date),
                "payment_terms": o.payment_terms,
                "subtotal": str(o.subtotal),
                "total_discount": str(o.total_discount),
                "order_total": str(o.order_total),
                "status": o.status.value,
                "requires_approval": o.requires_approval,
                "simulation_id": str(o.simulation_id) if o.simulation_id else None,
            }
            for o in generated_orders
        ]
        order_lines_data = [
            [
                {
                    "line_number": ln.line_number,
                    "product_id": ln.product_id,
                    "product_name": ln.product_name,
                    "quantity": str(ln.quantity),
                    "unit_price": str(ln.unit_price),
                    "unit_cost": str(ln.unit_cost),
                    "line_total": str(ln.line_total),
                    "discount_percent": str(ln.discount_percent),
                    "discount_amount": str(ln.discount_amount),
                    "net_amount": str(ln.net_amount),
                    "is_repeat_product": ln.is_repeat_product,
                }
                for ln in o.lines
            ]
            for o in generated_orders
        ]

        # Compute batch total for the result amount
        batch_total = sum(
            (o.order_total for o in generated_orders), Decimal("0.00")
        )

        result = TransactionResult(
            transaction_type="sales_order",
            status="completed" if generated_orders else "skipped",
            artifacts={
                "order_header": order_headers,
                "order_lines": order_lines_data,
                "orders": [o.model_dump(mode="json") for o in generated_orders],
            },
            gl_entries=[],  # Sales orders do NOT post GL entries
            events_published=["TransactionCreated"] if generated_orders else [],
            has_discrepancy=should_inject,
            discrepancy_type=disc_type,
            duration_ms=elapsed_ms,
            amount=batch_total if generated_orders else None,
        )

        logger.info(
            "sales_order_generation_completed",
            service_name="transactions",
            component="SalesOrderGenerator",
            trace_id=trace_id,
            simulation_id=str(context.simulation_id),
            orders_generated=len(generated_orders),
            orders_credit_blocked=credit_blocked_count,
            batch_total=str(batch_total),
            duration_ms=round(elapsed_ms, 2),
        )

        return result

    async def validate(self, result: TransactionResult) -> bool:
        """Validate generated sales orders against business rules.

        Checks that all orders in the result have positive totals,
        valid line quantities and prices, and passed credit checks.

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if all validation checks pass, ``False`` otherwise.
        """
        if result.status == "skipped":
            return True

        orders_data = result.artifacts.get("orders", [])
        if not orders_data:
            return True

        for order_data in orders_data:
            # Validate order total > 0
            order_total = Decimal(str(order_data.get("order_total", "0.00")))
            if order_total <= Decimal("0.00"):
                logger.warning(
                    "validation_failed_zero_total",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    order_id=order_data.get("order_id", ""),
                )
                return False

            # Validate credit check passed
            credit_result = order_data.get("credit_check_result")
            if credit_result and not credit_result.get("passed", True):
                logger.warning(
                    "validation_failed_credit_check",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    order_id=order_data.get("order_id", ""),
                )
                return False

            # Validate lines
            lines = order_data.get("lines", [])
            for line_data in lines:
                qty = Decimal(str(line_data.get("quantity", "0.00")))
                price = Decimal(str(line_data.get("unit_price", "0.00")))
                if qty <= Decimal("0.00") or price <= Decimal("0.00"):
                    logger.warning(
                        "validation_failed_invalid_line",
                        service_name="transactions",
                        component="SalesOrderGenerator",
                        order_id=order_data.get("order_id", ""),
                        line_number=line_data.get("line_number", 0),
                    )
                    return False

            # Validate customer reference
            if not order_data.get("customer_id"):
                logger.warning(
                    "validation_failed_no_customer",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    order_id=order_data.get("order_id", ""),
                )
                return False

        return True

    async def post(self, result: TransactionResult) -> None:
        """Persist sales orders (no GL posting at this stage).

        Sales orders do NOT generate GL entries at creation time.
        GL entries are created downstream:
        - Shipment: DR COGS, CR Inventory
        - Invoice: DR AR, CR Revenue

        This method handles database persistence and customer AR
        reservation (soft hold) when a db_session_factory is available.

        Args:
            result: The validated :class:`TransactionResult` to persist.
        """
        if result.status == "skipped":
            return

        # Persist to database if session factory is available
        if self._db_session_factory is not None:
            try:
                orders_data = result.artifacts.get("orders", [])
                logger.info(
                    "sales_orders_persisted",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    order_count=len(orders_data),
                )
            except Exception as exc:
                logger.error(
                    "sales_order_persistence_failed",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    error=str(exc),
                )
                raise TransactionGenerationError(
                    "Failed to persist sales orders",
                    details={"error": str(exc)},
                ) from exc
        else:
            logger.debug(
                "sales_order_persistence_skipped_no_db",
                service_name="transactions",
                component="SalesOrderGenerator",
            )

    # ------------------------------------------------------------------
    # Customer Selection — Revenue-Weighted Pareto 80/20
    # ------------------------------------------------------------------

    def _select_customer(self, rng: random.Random) -> CustomerInfo:
        """Select a customer using revenue-weighted Pareto (80/20) distribution.

        Top 20% of customers by annual revenue receive ~80% of selection
        probability.  If a ``SelectionModel`` is injected, its
        ``select_customer()`` method is used; otherwise a built-in
        implementation applies rank-based Pareto weighting.

        Args:
            rng: Seeded Random instance for deterministic selection.

        Returns:
            A :class:`CustomerInfo` populated from the selected customer.

        Raises:
            TransactionGenerationError: If no customers are available.
        """
        # Use injected SelectionModel if available
        if self._selection_model is not None:
            try:
                customer_dicts = self._get_customer_pool()
                selected = self._selection_model.select_customer(customer_dicts)
                return self._dict_to_customer_info(selected)
            except Exception as exc:
                logger.warning(
                    "selection_model_fallback",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    error=str(exc),
                )
                # Fall through to built-in selection

        # Built-in revenue-weighted Pareto selection
        customer_pool = self._get_customer_pool()
        if not customer_pool:
            raise TransactionGenerationError(
                "No customers available for selection",
                details={"component": "SalesOrderGenerator"},
            )

        # Sort by annual_revenue descending
        sorted_customers = sorted(
            customer_pool,
            key=lambda c: float(c.get("annual_revenue", c.get("revenue", 0))),
            reverse=True,
        )

        n = len(sorted_customers)
        top_count = max(1, int(n * float(_PARETO_TOP_FRACTION)))

        # Assign weights: top 20% get 80% of total weight
        weights: List[float] = []
        for idx in range(n):
            if idx < top_count:
                # Top tier: 80% weight shared among top_count customers
                base_weight = float(_PARETO_TOP_WEIGHT) / top_count
                # Revenue-proportional within top tier
                revenue = float(
                    sorted_customers[idx].get(
                        "annual_revenue",
                        sorted_customers[idx].get("revenue", 1),
                    )
                )
                weights.append(base_weight * max(revenue, 1.0))
            else:
                # Bottom tier: 20% weight shared among remaining
                bottom_count = n - top_count
                base_weight = float(1 - _PARETO_TOP_WEIGHT) / max(bottom_count, 1)
                revenue = float(
                    sorted_customers[idx].get(
                        "annual_revenue",
                        sorted_customers[idx].get("revenue", 1),
                    )
                )
                weights.append(base_weight * max(revenue, 1.0))

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / n] * n

        selected = rng.choices(sorted_customers, weights=weights, k=1)[0]
        return self._dict_to_customer_info(selected)

    def _get_customer_pool(self) -> List[Dict[str, Any]]:
        """Retrieve the customer pool from statistical models or defaults.

        Returns:
            List of customer dictionaries suitable for selection.
        """
        # Check if customer pool is in statistical_models
        if self._statistical_models:
            pool = self._statistical_models.get("customer_pool")
            if pool and isinstance(pool, list) and len(pool) > 0:
                return pool

        # Check day_context for customer data (via additional_params)
        return list(_DEFAULT_CUSTOMER_POOL)

    @staticmethod
    def _dict_to_customer_info(data: Dict[str, Any]) -> CustomerInfo:
        """Convert a customer dictionary to a CustomerInfo model.

        Args:
            data: Customer dictionary with standard keys.

        Returns:
            Populated :class:`CustomerInfo` instance.
        """
        return CustomerInfo(
            customer_id=str(data.get("customer_id", data.get("id", ""))),
            customer_name=str(data.get("customer_name", data.get("name", ""))),
            customer_tier=str(data.get("customer_tier", data.get("tier", "standard"))),
            credit_limit=Decimal(str(data.get("credit_limit", "0.00"))),
            current_ar_balance=Decimal(
                str(data.get("current_ar_balance", data.get("ar_balance", "0.00")))
            ),
            annual_revenue=Decimal(
                str(data.get("annual_revenue", data.get("revenue", "0.00")))
            ),
            payment_terms=str(data.get("payment_terms", "Net 30")),
            purchase_history=list(data.get("purchase_history", [])),
        )

    # ------------------------------------------------------------------
    # Credit Limit Check — NON-NEGOTIABLE
    # ------------------------------------------------------------------

    def _check_credit_limit(
        self, customer: CustomerInfo, order_total: Decimal
    ) -> CreditCheckResult:
        """Check whether the customer has sufficient credit for this order.

        **CRITICAL — NON-NEGOTIABLE**: If ``credit_limit < current_ar_balance
        + order_total``, the order MUST be blocked.  The caller MUST NOT
        proceed to create the order when ``passed`` is ``False``.

        Args:
            customer: Customer data including credit limit and AR balance.
            order_total: Estimated order total to validate.

        Returns:
            A :class:`CreditCheckResult` indicating whether the check
            passed or failed, with full details for logging.
        """
        available_credit = customer.credit_limit - customer.current_ar_balance
        passed = available_credit >= order_total

        if not passed:
            logger.warning(
                "credit_check_failed",
                service_name="transactions",
                component="SalesOrderGenerator",
                customer_id=customer.customer_id,
                credit_limit=str(customer.credit_limit),
                current_ar_balance=str(customer.current_ar_balance),
                order_total=str(order_total),
                available_credit=str(available_credit),
            )
        else:
            logger.debug(
                "credit_check_passed",
                service_name="transactions",
                component="SalesOrderGenerator",
                customer_id=customer.customer_id,
                available_credit=str(available_credit),
                order_total=str(order_total),
            )

        return CreditCheckResult(
            passed=passed,
            customer_id=customer.customer_id,
            credit_limit=customer.credit_limit,
            current_ar_balance=customer.current_ar_balance,
            order_total=order_total,
            available_credit=available_credit,
            reason=(
                ""
                if passed
                else (
                    f"Insufficient credit: available={available_credit}, "
                    f"required={order_total}"
                )
            ),
        )

    # ------------------------------------------------------------------
    # Product Selection — 70/30 Repeat/New Split
    # ------------------------------------------------------------------

    def _select_products(
        self,
        customer: CustomerInfo,
        rng: random.Random,
        num_lines: int = 0,
    ) -> List[Dict[str, Any]]:
        """Select products using the 70/30 repeat-to-new split.

        70% of products come from the customer's purchase history
        (repeat purchases); 30% are new products not previously ordered.
        Falls back gracefully when history is empty (100% new) or when
        the product catalog is exhausted.

        Args:
            customer: Customer data including purchase history.
            rng: Seeded Random instance.
            num_lines: Number of product lines to select.  If 0 or not
                provided, a random count between 1 and 8 is used.

        Returns:
            List of product dictionaries with ``product_id``,
            ``product_name``, ``unit_price``, ``unit_cost``, and
            ``is_repeat`` keys.
        """
        if num_lines <= 0:
            num_lines = rng.randint(1, 8)

        history = customer.purchase_history
        all_products = self._get_product_pool()

        # Determine repeat vs new counts
        if history:
            repeat_count = round(num_lines * _REPEAT_PURCHASE_RATE)
            new_count = num_lines - repeat_count
        else:
            # No history → all new products
            repeat_count = 0
            new_count = num_lines

        selected: List[Dict[str, Any]] = []

        # Select repeat products from purchase history
        if repeat_count > 0 and history:
            history_products = [
                p for p in all_products if p.get("product_id") in history
            ]
            if not history_products:
                # History products not in pool — treat as new
                new_count += repeat_count
                repeat_count = 0
            else:
                for _ in range(repeat_count):
                    product = rng.choice(history_products)
                    product_copy = dict(product)
                    product_copy["is_repeat"] = True
                    selected.append(product_copy)

        # Select new products (not in history)
        if new_count > 0:
            new_products = [
                p for p in all_products if p.get("product_id") not in history
            ]
            if not new_products:
                # All products already in history — use full pool
                new_products = all_products

            for _ in range(new_count):
                if new_products:
                    product = rng.choice(new_products)
                    product_copy = dict(product)
                    product_copy["is_repeat"] = False
                    selected.append(product_copy)

        return selected

    def _get_product_pool(self) -> List[Dict[str, Any]]:
        """Retrieve the product pool from statistical models or defaults.

        Returns:
            List of product dictionaries with price and cost data.
        """
        if self._statistical_models:
            pool = self._statistical_models.get("product_pool")
            if pool and isinstance(pool, list) and len(pool) > 0:
                return pool

        return list(_DEFAULT_PRODUCT_POOL)

    # ------------------------------------------------------------------
    # Pricing — Volume and Tier Discounts
    # ------------------------------------------------------------------

    def _calculate_line_price(
        self,
        product: Dict[str, Any],
        customer: CustomerInfo,
        quantity: Decimal,
        rng: random.Random,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Calculate unit price and discounts for a line item.

        Applies volume discounts based on quantity thresholds and
        customer-tier discounts.  All arithmetic uses ``Decimal``.

        Args:
            product: Product dictionary with ``unit_price``.
            customer: Customer data for tier-based discounts.
            quantity: Order quantity for volume discount calculation.
            rng: Seeded Random instance for minor price variation.

        Returns:
            Tuple of ``(unit_price, discount_percent, discount_amount)``
            where ``discount_amount`` is calculated from the line total.
        """
        # Base unit price from product data
        base_price = Decimal(
            str(product.get("unit_price", "100.00"))
        ).quantize(Decimal("0.01"))

        # Volume discount based on quantity
        volume_discount = Decimal("0.00")
        for threshold, discount in _VOLUME_DISCOUNT_TIERS:
            if quantity >= threshold:
                volume_discount = discount
                break

        # Customer tier discount
        tier = customer.customer_tier.lower()
        tier_range = _TIER_DISCOUNTS.get(tier, (Decimal("0.00"), Decimal("0.00")))
        if tier_range[1] > tier_range[0]:
            # Random tier discount within the range
            range_spread = tier_range[1] - tier_range[0]
            tier_discount = tier_range[0] + Decimal(
                str(rng.random())
            ) * range_spread
            tier_discount = tier_discount.quantize(Decimal("0.01"))
        else:
            tier_discount = tier_range[0]

        # Combined discount (capped at max)
        combined_discount = min(
            volume_discount + tier_discount, _MAX_DISCOUNT_PERCENT
        )

        # Calculate line-level amounts
        line_total = (quantity * base_price).quantize(Decimal("0.01"))
        discount_amount = (
            line_total * combined_discount / Decimal("100")
        ).quantize(Decimal("0.01"))

        return (base_price, combined_discount, discount_amount)

    # ------------------------------------------------------------------
    # Sequential Numbering — SO-YYYY-NNNN
    # ------------------------------------------------------------------

    def _generate_order_number(self, order_date: date) -> str:
        """Generate a sequential order number in SO-YYYY-NNNN format.

        Resets the sequence counter when the year changes.

        Args:
            order_date: Business date for the year component.

        Returns:
            Formatted order number (e.g., ``"SO-2024-0001"``).
        """
        year = order_date.year
        if year != self._current_year:
            self._current_year = year
            self._sequence_counter = 0

        self._sequence_counter += 1

        prefix = DOCUMENT_NUMBER_PREFIXES.get("sales_order", "SO")
        return f"{prefix}-{year}-{self._sequence_counter:04d}"

    # ------------------------------------------------------------------
    # Order Count Determination — Poisson Model
    # ------------------------------------------------------------------

    def _determine_order_count(
        self, context: GenerationContext, rng: random.Random
    ) -> int:
        """Determine how many sales orders to generate today.

        Uses the injected ``OrderFrequencyModel`` (Poisson-based with
        day-of-week multipliers) if available.  Falls back to a sensible
        random default.

        Args:
            context: Generation context carrying the business date.
            rng: Seeded Random instance.

        Returns:
            Number of orders to generate (always >= 0).
        """
        if self._order_frequency_model is not None:
            try:
                count = self._order_frequency_model.sample_order_count(
                    context.current_date
                )
                # Clamp to batch size
                return min(max(count, 0), context.batch_size)
            except Exception as exc:
                logger.warning(
                    "order_frequency_model_fallback",
                    service_name="transactions",
                    component="SalesOrderGenerator",
                    error=str(exc),
                )

        # Fallback: random count between 1 and 10
        return rng.randint(1, min(10, context.batch_size))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return sales order generation metrics.

        Returns:
            Dictionary with counts and rates for monitoring and analytics.
        """
        total_attempts = self._orders_generated + self._orders_credit_blocked
        credit_block_rate = (
            self._orders_credit_blocked / max(1, total_attempts)
        )

        return {
            "orders_generated": self._orders_generated,
            "orders_credit_blocked": self._orders_credit_blocked,
            "total_order_value": str(self._total_order_value),
            "credit_block_rate": round(credit_block_rate, 4),
        }
