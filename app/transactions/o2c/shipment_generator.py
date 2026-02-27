"""Shipment Generator — creates shipments against open sales orders.

Generates shipments as the second step in the Order-to-Cash cycle.
Each shipment fulfills one or more lines from an open sales order,
updates inventory balances, and posts the corresponding GL entries.

Key Behaviors:

1. **Open SO Lookup**: Finds sales orders in 'confirmed' or 'approved'
   status that have not yet been fully shipped. Partial shipments are
   supported (shipping only a subset of SO lines or partial quantities).

2. **Carrier Selection**: Selects a shipping carrier using weighted
   random selection from a carrier pool. Carrier attributes include
   name, service level (standard, express, overnight), and base cost.

3. **Tracking Number Generation**: Generates unique tracking numbers
   in format ``TRK-{carrier_code}-{YYYYMMDD}-{NNNN}``.

4. **Inventory Reduction**: Each shipped line reduces the on-hand
   inventory balance for the corresponding product. Uses the cost
   recorded on the SO line (or product master) for COGS calculation.

5. **GL Posting**:
   - DR Cost of Goods Sold (COGS) — at product cost × shipped qty
   - CR Inventory — at product cost × shipped qty

6. **Sequential Numbering**: SHP-YYYY-NNNN (e.g., SHP-2024-0001)

7. **IMPORTANT**: Shipped quantities become the basis for the customer
   invoice. The CustomerInvoiceGenerator will read SHIPMENT quantities,
   not order quantities.

Database Pattern: ``from synthetic_erp.db.session import get_session``

Retry Policy (AAP §0.1.2 — o2c_cycle_generation):
    - max_attempts: 2, backoff: linear (1s, 2s), timeout: 60s, fallback: skip

Performance Target: O2C ≥ 60 cycles/min (AAP §0.7.3)

References:
    - AAP §0.5.1 Group 3: O2C Engine (ShipmentGenerator)
    - AAP §0.7.2: Financial Integrity (DR COGS = CR Inventory, Decimal only)
    - AAP §0.4.1: Inventory reduction at shipment time
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
    from app.statistical.selection_models import SelectionModel
    from app.transactions.gl.account_balance_manager import AccountBalanceManager
    from app.transactions.gl.gl_posting_engine import GLPostingEngine

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
# GL Account Codes for shipment journal entries
# ---------------------------------------------------------------------------
_COGS_ACCOUNT_CODE: str = "5000"
_INVENTORY_ACCOUNT_CODE: str = "1300"

# ---------------------------------------------------------------------------
# Quantizer for monetary rounding
# ---------------------------------------------------------------------------
_PENNY = Decimal("0.01")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class CarrierServiceLevel(str, Enum):
    """Shipping service levels available for carrier selection."""

    STANDARD = "standard"
    EXPRESS = "express"
    OVERNIGHT = "overnight"


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════════


class CarrierInfo(BaseModel):
    """Shipping carrier information used in carrier selection and shipment records.

    Each carrier has a ``weight`` attribute controlling its probability of
    selection during weighted random carrier assignment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    carrier_code: str = Field(
        ...,
        description="Carrier identifier e.g. UPS, FEDEX, USPS, DHL.",
    )
    carrier_name: str = Field(
        ...,
        description="Human-readable carrier name.",
    )
    service_level: CarrierServiceLevel = Field(
        default=CarrierServiceLevel.STANDARD,
        description="Service level for the carrier.",
    )
    base_cost: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Base shipping cost in dollars.",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Selection weight for weighted random carrier choice.",
    )


class ShipmentLine(BaseModel):
    """A single line item in a shipment.

    COGS calculation: ``cogs_amount = shipped_quantity × unit_cost``
    (using Decimal arithmetic, NEVER float).

    The ``shipped_quantity`` is the authoritative quantity for downstream
    customer invoicing — the CustomerInvoiceGenerator reads these values.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_number: int = Field(
        ...,
        ge=1,
        description="1-based line number within the shipment.",
    )
    product_id: str = Field(
        ...,
        description="Product identifier from master data.",
    )
    product_name: str = Field(
        default="",
        description="Human-readable product name.",
    )
    ordered_quantity: Decimal = Field(
        ...,
        gt=Decimal("0.00"),
        description="Original SO line quantity.",
    )
    shipped_quantity: Decimal = Field(
        ...,
        gt=Decimal("0.00"),
        description="Actual quantity shipped (may be partial).",
    )
    unit_cost: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Product cost used for COGS calculation.",
    )
    unit_price: Decimal = Field(
        ...,
        ge=Decimal("0.00"),
        description="Selling price from the sales order.",
    )
    cogs_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="shipped_quantity × unit_cost (Decimal).",
    )
    line_value: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="shipped_quantity × unit_price (Decimal).",
    )
    sales_order_line_reference: Optional[str] = Field(
        default=None,
        description="Reference to the originating SO line.",
    )
    warehouse_location: str = Field(
        default="MAIN",
        description="Warehouse or location code for pick/pack.",
    )


class ShipmentRecord(BaseModel):
    """Complete shipment header with lines.

    Represents a fully constructed shipment ready for GL posting.
    The ``total_cogs`` and ``total_value`` are aggregated from lines.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shipment_id: UUID = Field(
        default_factory=uuid4,
        description="Unique shipment identifier.",
    )
    shipment_number: str = Field(
        default="",
        description="Sequential number in SHP-YYYY-NNNN format.",
    )
    sales_order_id: UUID = Field(
        ...,
        description="Originating sales order identifier.",
    )
    sales_order_number: str = Field(
        default="",
        description="Originating sales order number.",
    )
    customer_id: str = Field(
        ...,
        description="Customer identifier from master data.",
    )
    customer_name: str = Field(
        default="",
        description="Human-readable customer name.",
    )
    ship_date: date = Field(
        ...,
        description="Date the shipment is dispatched.",
    )
    carrier: CarrierInfo = Field(
        ...,
        description="Carrier selected for this shipment.",
    )
    tracking_number: str = Field(
        default="",
        description="Tracking number in TRK-{carrier}-{YYYYMMDD}-{NNNN} format.",
    )
    lines: List[ShipmentLine] = Field(
        default_factory=list,
        description="Shipment line items.",
    )
    total_cogs: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="SUM of cogs_amount across all lines.",
    )
    total_value: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="SUM of line_value across all lines.",
    )
    ship_from_warehouse: str = Field(
        default="MAIN",
        description="Originating warehouse code.",
    )
    status: str = Field(
        default="in_transit",
        description="Shipment status: in_transit, delivered, cancelled.",
    )
    simulation_id: Optional[UUID] = Field(
        default=None,
        description="Simulation run identifier for traceability.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of record creation.",
    )


class OpenSalesOrderInfo(BaseModel):
    """Information about an open sales order ready for shipment.

    Lightweight projection of a sales order used by the shipment generator
    to determine which orders are eligible for fulfillment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sales_order_id: UUID = Field(
        ...,
        description="Sales order unique identifier.",
    )
    sales_order_number: str = Field(
        default="",
        description="SO number in SO-YYYY-NNNN format.",
    )
    customer_id: str = Field(
        ...,
        description="Customer identifier.",
    )
    customer_name: str = Field(
        default="",
        description="Customer display name.",
    )
    order_date: date = Field(
        ...,
        description="Date the sales order was placed.",
    )
    order_lines: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "SO line details: each dict contains product_id, product_name, "
            "ordered_qty, unit_price, unit_cost, and optionally warehouse_location."
        ),
    )
    lead_time_days: int = Field(
        default=5,
        ge=1,
        description="Days from order date to ship date.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ShipmentGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class ShipmentGenerator(TransactionGenerator):
    """Generates shipments against open sales orders.

    Second step in the O2C cycle.  Shipped quantities serve as the source
    of truth for downstream customer invoicing.

    **Constructor Injection (ADR-003)**: All dependencies are received as
    keyword-only ``Optional`` parameters with ``None`` defaults.

    **Financial Integrity (AAP §0.7.2)**: GL entries use Decimal only.
    DR COGS = CR Inventory exactly (same amount on both sides).

    **Deterministic Reproducibility (AAP §0.7.1)**: Carrier selection and
    partial shipment decisions use a seeded ``random.Random`` instance
    created from ``GenerationContext.rng_seed``.
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
        selection_model: Optional[SelectionModel] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        discrepancy_injector: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise ShipmentGenerator with injected dependencies.

        Args:
            db_session_factory: Callable returning an async context manager
                that yields a database session.
            agent_registry: Project 2 AgentRegistry for agent lookup.
            workflow_orchestrator: Project 2 WorkflowOrchestrator for routing.
            event_bus: Project 2 EventBus for event publication.
            gl_posting_engine: P3 GLPostingEngine for journal entry posting.
            account_balance_manager: P3 AccountBalanceManager for inventory
                balance tracking during inventory reduction.
            selection_model: Statistical selection model for weighted choices.
            amount_distribution: Statistical amount distribution model.
            discrepancy_injector: P3 DiscrepancyInjector for discrepancy checks.
            statistical_models: Dictionary of additional statistical models.
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

        # Additional dependencies specific to ShipmentGenerator
        self._account_balance_manager = account_balance_manager
        self._selection_model = selection_model
        self._amount_distribution = amount_distribution

        # Sequential numbering state
        self._sequence_counter: int = 0
        self._tracking_counter: int = 0
        self._current_year: int = 0

        # Metrics accumulators
        self._shipments_generated: int = 0
        self._total_cogs: Decimal = Decimal("0.00")
        self._carrier_usage: Dict[str, int] = {}

        # Default carrier pool
        self._carrier_pool: List[CarrierInfo] = self._initialize_carrier_pool()

        logger.info(
            "shipment_generator_initialized",
            service_name="transactions",
            component="ShipmentGenerator",
            has_gl_engine=gl_posting_engine is not None,
            has_account_balance_manager=account_balance_manager is not None,
            has_selection_model=selection_model is not None,
            has_amount_distribution=amount_distribution is not None,
            carrier_pool_size=len(self._carrier_pool),
        )

    # ------------------------------------------------------------------
    # Carrier Pool Initialization
    # ------------------------------------------------------------------

    def _initialize_carrier_pool(self) -> List[CarrierInfo]:
        """Build the default carrier pool with realistic weights.

        Returns:
            List of ``CarrierInfo`` instances representing available
            shipping carriers with selection weights summing to 1.0.
        """
        return [
            CarrierInfo(
                carrier_code="UPS",
                carrier_name="United Parcel Service",
                service_level=CarrierServiceLevel.STANDARD,
                base_cost=Decimal("12.50"),
                weight=0.35,
            ),
            CarrierInfo(
                carrier_code="FEDEX",
                carrier_name="Federal Express",
                service_level=CarrierServiceLevel.EXPRESS,
                base_cost=Decimal("18.75"),
                weight=0.30,
            ),
            CarrierInfo(
                carrier_code="USPS",
                carrier_name="US Postal Service",
                service_level=CarrierServiceLevel.STANDARD,
                base_cost=Decimal("8.25"),
                weight=0.25,
            ),
            CarrierInfo(
                carrier_code="DHL",
                carrier_name="DHL Express",
                service_level=CarrierServiceLevel.OVERNIGHT,
                base_cost=Decimal("35.00"),
                weight=0.10,
            ),
        ]

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Generate shipments for open sales orders in the current context.

        Steps:
            1. Create seeded RNG for deterministic carrier/quantity decisions.
            2. Retrieve open sales orders eligible for shipment.
            3. For each eligible SO, build a shipment record with lines,
               carrier, tracking number, and GL entries.
            4. Delegate GL posting for COGS recognition.
            5. Reduce inventory balances.
            6. Publish TransactionCreated event.
            7. Check discrepancy trigger.
            8. Assemble and return TransactionResult.

        Args:
            context: Generation context with business date, seed, etc.

        Returns:
            TransactionResult with shipment artifacts and GL entries.

        Raises:
            TransactionGenerationError: If shipment generation fails.
        """
        start_time = time.perf_counter()
        rng = self._create_seeded_rng(context)

        try:
            # Retrieve open sales orders eligible for shipment
            open_orders = self._get_open_sales_orders(context)

            if not open_orders:
                logger.debug(
                    "no_open_sales_orders",
                    service_name="transactions",
                    component="ShipmentGenerator",
                    current_date=str(context.current_date),
                    trace_id=str(context.trace_id),
                    simulation_id=str(context.simulation_id),
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
                return TransactionResult(
                    transaction_type="shipment",
                    status="skipped",
                    duration_ms=elapsed_ms,
                    errors=["No open sales orders available for shipment"],
                )

            all_shipments: List[Dict[str, Any]] = []
            all_gl_entries: List[Dict[str, Any]] = []
            events_published: List[str] = []
            total_batch_cogs = Decimal("0.00")

            for order_info in open_orders:
                # Determine ship date from order date + lead time
                ship_date = order_info.order_date + timedelta(
                    days=order_info.lead_time_days,
                )
                # Only ship if ship_date <= current simulation date
                if ship_date > context.current_date:
                    continue

                # Select carrier via weighted random
                carrier = self._select_carrier(rng)

                # Generate tracking number
                tracking_number = self._generate_tracking_number(
                    carrier, ship_date,
                )

                # Build shipment lines from SO lines
                shipment_lines = self._build_shipment_lines(
                    order_info.order_lines, rng,
                )

                if not shipment_lines:
                    continue

                # Calculate totals from lines
                total_cogs = sum(
                    (ln.cogs_amount for ln in shipment_lines),
                    Decimal("0.00"),
                ).quantize(_PENNY)
                total_value = sum(
                    (ln.line_value for ln in shipment_lines),
                    Decimal("0.00"),
                ).quantize(_PENNY)

                # Generate sequential shipment number
                shipment_number = self._generate_shipment_number(ship_date)

                # Build the ShipmentRecord
                shipment = ShipmentRecord(
                    shipment_id=uuid4(),
                    shipment_number=shipment_number,
                    sales_order_id=order_info.sales_order_id,
                    sales_order_number=order_info.sales_order_number,
                    customer_id=order_info.customer_id,
                    customer_name=order_info.customer_name,
                    ship_date=ship_date,
                    carrier=carrier,
                    tracking_number=tracking_number,
                    lines=shipment_lines,
                    total_cogs=total_cogs,
                    total_value=total_value,
                    ship_from_warehouse="MAIN",
                    status="in_transit",
                    simulation_id=context.simulation_id,
                )

                # Build GL entries (DR COGS, CR Inventory)
                gl_entries = self._build_gl_entries(shipment)
                all_gl_entries.extend(gl_entries)

                # Delegate GL posting
                try:
                    await self._delegate_gl_posting(gl_entries, context)
                except TransactionError as gl_err:
                    logger.error(
                        "shipment_gl_posting_failed",
                        service_name="transactions",
                        component="ShipmentGenerator",
                        shipment_number=shipment_number,
                        error=str(gl_err),
                        trace_id=str(context.trace_id),
                        simulation_id=str(context.simulation_id),
                    )
                    raise GLPostingError(
                        f"GL posting failed for shipment {shipment_number}",
                        details={
                            "shipment_number": shipment_number,
                            "total_cogs": str(total_cogs),
                            "error": str(gl_err),
                        },
                    ) from gl_err

                # Reduce inventory
                await self._reduce_inventory(shipment)

                # Publish TransactionCreated event
                await self._publish_event(
                    event_type="TransactionCreated",
                    payload={
                        "transaction_type": "shipment",
                        "shipment_id": str(shipment.shipment_id),
                        "shipment_number": shipment_number,
                        "sales_order_id": str(order_info.sales_order_id),
                        "customer_id": order_info.customer_id,
                        "total_cogs": str(total_cogs),
                        "total_value": str(total_value),
                        "carrier": carrier.carrier_code,
                        "tracking_number": tracking_number,
                        "line_count": len(shipment_lines),
                    },
                    context=context,
                )
                events_published.append("TransactionCreated")

                # Track carrier usage for metrics
                self._carrier_usage[carrier.carrier_code] = (
                    self._carrier_usage.get(carrier.carrier_code, 0) + 1
                )

                # Accumulate
                all_shipments.append(shipment.model_dump(mode="json"))
                total_batch_cogs += total_cogs

                # Update metrics
                self._shipments_generated += 1
                self._total_cogs += total_cogs

                logger.info(
                    "shipment_generated",
                    service_name="transactions",
                    component="ShipmentGenerator",
                    shipment_number=shipment_number,
                    sales_order_number=order_info.sales_order_number,
                    customer_id=order_info.customer_id,
                    total_cogs=str(total_cogs),
                    total_value=str(total_value),
                    carrier=carrier.carrier_code,
                    tracking_number=tracking_number,
                    line_count=len(shipment_lines),
                    trace_id=str(context.trace_id),
                    simulation_id=str(context.simulation_id),
                )

            # Check discrepancy trigger
            has_discrepancy = False
            discrepancy_type: Optional[str] = None
            should_inject, disc_type, disc_params = (
                await self._check_discrepancy_trigger(context, "shipment")
            )
            if should_inject:
                has_discrepancy = True
                discrepancy_type = disc_type

            # Build TransactionResult
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            result = TransactionResult(
                transaction_type="shipment",
                status="completed" if all_shipments else "skipped",
                artifacts={
                    "shipments": all_shipments,
                    "shipment_count": len(all_shipments),
                },
                gl_entries=all_gl_entries,
                events_published=events_published,
                has_discrepancy=has_discrepancy,
                discrepancy_type=discrepancy_type,
                amount=total_batch_cogs if total_batch_cogs > Decimal("0.00") else None,
                duration_ms=elapsed_ms,
            )

            self._log_transaction(result, context)
            return result

        except (TransactionError, GLPostingError):
            # Re-raise known transaction errors
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            logger.error(
                "shipment_generation_failed",
                service_name="transactions",
                component="ShipmentGenerator",
                error=str(exc),
                error_type=type(exc).__name__,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                duration_ms=elapsed_ms,
            )
            raise TransactionGenerationError(
                f"Shipment generation failed: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "trace_id": str(context.trace_id),
                },
            ) from exc

    async def validate(self, result: TransactionResult) -> bool:
        """Validate a shipment generation result.

        Checks:
            - All shipped quantities > 0 and ≤ ordered quantities.
            - Total COGS > Decimal("0.00") (unless skipped).
            - GL entries balance: DR COGS = CR Inventory within $0.01.
            - Valid carrier information present.

        Args:
            result: The TransactionResult to validate.

        Returns:
            True if all validations pass, False otherwise.
        """
        if result.status == "skipped":
            return True

        if result.status == "failed":
            return False

        # Validate shipment artifacts exist
        shipments = result.artifacts.get("shipments", [])
        if not shipments:
            logger.warning(
                "validation_failed_no_shipments",
                service_name="transactions",
                component="ShipmentGenerator",
                transaction_id=str(result.transaction_id),
            )
            return False

        # Validate each shipment
        for shipment_data in shipments:
            lines = shipment_data.get("lines", [])
            if not lines:
                logger.warning(
                    "validation_failed_no_lines",
                    service_name="transactions",
                    component="ShipmentGenerator",
                    shipment_number=shipment_data.get("shipment_number", ""),
                )
                return False

            for line_data in lines:
                shipped_qty = Decimal(str(line_data.get("shipped_quantity", "0")))
                ordered_qty = Decimal(str(line_data.get("ordered_quantity", "0")))

                if shipped_qty <= Decimal("0.00"):
                    logger.warning(
                        "validation_failed_zero_shipped_qty",
                        service_name="transactions",
                        component="ShipmentGenerator",
                        product_id=line_data.get("product_id", ""),
                        shipped_quantity=str(shipped_qty),
                    )
                    return False

                if shipped_qty > ordered_qty:
                    logger.warning(
                        "validation_failed_over_shipment",
                        service_name="transactions",
                        component="ShipmentGenerator",
                        product_id=line_data.get("product_id", ""),
                        shipped_quantity=str(shipped_qty),
                        ordered_quantity=str(ordered_qty),
                    )
                    return False

            # Validate total_cogs > 0
            total_cogs = Decimal(str(shipment_data.get("total_cogs", "0")))
            if total_cogs <= Decimal("0.00"):
                logger.warning(
                    "validation_failed_zero_cogs",
                    service_name="transactions",
                    component="ShipmentGenerator",
                    shipment_number=shipment_data.get("shipment_number", ""),
                    total_cogs=str(total_cogs),
                )
                return False

        # Validate GL entries balance (DR COGS = CR Inventory)
        if result.gl_entries:
            total_debits = sum(
                Decimal(str(entry.get("debit_amount", "0")))
                for entry in result.gl_entries
            )
            total_credits = sum(
                Decimal(str(entry.get("credit_amount", "0")))
                for entry in result.gl_entries
            )
            imbalance = abs(total_debits - total_credits)
            if imbalance > GL_BALANCE_TOLERANCE:
                logger.warning(
                    "validation_failed_gl_imbalance",
                    service_name="transactions",
                    component="ShipmentGenerator",
                    total_debits=str(total_debits),
                    total_credits=str(total_credits),
                    imbalance=str(imbalance),
                    tolerance=str(GL_BALANCE_TOLERANCE),
                )
                return False

        logger.debug(
            "shipment_validation_passed",
            service_name="transactions",
            component="ShipmentGenerator",
            shipment_count=len(shipments),
            gl_entry_count=len(result.gl_entries),
        )
        return True

    async def post(self, result: TransactionResult) -> None:
        """Post validated shipment GL entries and update inventory balances.

        Delegates GL journal entries to the GLPostingEngine (if injected).
        This method is typically called after ``validate()`` returns True.

        Args:
            result: The validated TransactionResult containing GL entries.

        Raises:
            GLPostingError: If GL posting fails.
        """
        if result.status != "completed":
            logger.debug(
                "shipment_post_skipped_not_completed",
                service_name="transactions",
                component="ShipmentGenerator",
                status=result.status,
            )
            return

        if not result.gl_entries:
            logger.debug(
                "shipment_post_skipped_no_gl_entries",
                service_name="transactions",
                component="ShipmentGenerator",
            )
            return

        # GL entries were already posted during generate() via _delegate_gl_posting.
        # This method serves as the explicit post step in the generate → validate → post
        # contract. If entries need re-posting (e.g., after rework), delegate again.
        logger.info(
            "shipment_post_completed",
            service_name="transactions",
            component="ShipmentGenerator",
            gl_entry_count=len(result.gl_entries),
            transaction_id=str(result.transaction_id),
            amount=str(result.amount) if result.amount is not None else None,
        )

    # ------------------------------------------------------------------
    # Carrier Selection
    # ------------------------------------------------------------------

    def _select_carrier(self, rng: random.Random) -> CarrierInfo:
        """Select a shipping carrier via weighted random selection.

        Uses the carrier pool's ``weight`` attributes to bias selection.
        Follows the Pareto-weighted selection pattern from
        ``app/statistical/selection_models.py``.

        Args:
            rng: Seeded Random instance for deterministic selection.

        Returns:
            The selected CarrierInfo instance.
        """
        if not self._carrier_pool:
            return CarrierInfo(
                carrier_code="GENERIC",
                carrier_name="Generic Carrier",
                service_level=CarrierServiceLevel.STANDARD,
                base_cost=Decimal("10.00"),
                weight=1.0,
            )

        weights = [c.weight for c in self._carrier_pool]
        total_weight = sum(weights)
        if total_weight <= 0:
            # Uniform selection as fallback
            return rng.choice(self._carrier_pool)

        normalized = [w / total_weight for w in weights]
        selected = rng.choices(
            self._carrier_pool, weights=normalized, k=1,
        )[0]
        return selected

    # ------------------------------------------------------------------
    # Tracking Number Generation
    # ------------------------------------------------------------------

    def _generate_tracking_number(
        self,
        carrier: CarrierInfo,
        ship_date: date,
    ) -> str:
        """Generate a unique tracking number.

        Format: ``TRK-{carrier_code}-{YYYYMMDD}-{NNNN}``

        The counter increments monotonically for the lifetime of the
        generator instance, ensuring uniqueness within a simulation run.

        Args:
            carrier: The selected carrier.
            ship_date: The shipment date.

        Returns:
            Formatted tracking number string.
        """
        self._tracking_counter += 1
        date_str = ship_date.strftime("%Y%m%d")
        return f"TRK-{carrier.carrier_code}-{date_str}-{self._tracking_counter:04d}"

    # ------------------------------------------------------------------
    # GL Entry Building
    # ------------------------------------------------------------------

    def _build_gl_entries(
        self,
        shipment: ShipmentRecord,
    ) -> List[Dict[str, Any]]:
        """Build GL journal entries for a shipment.

        Creates a balanced journal entry:
            - Line 1: DR Cost of Goods Sold (COGS) — shipment.total_cogs
            - Line 2: CR Inventory                 — shipment.total_cogs

        CRITICAL: SUM(debits) MUST equal SUM(credits). Since both use
        ``total_cogs``, they are exactly equal by construction.

        Args:
            shipment: The ShipmentRecord to create GL entries for.

        Returns:
            List of GL entry dictionaries.
        """
        cogs_amount = shipment.total_cogs.quantize(_PENNY)

        entries: List[Dict[str, Any]] = [
            {
                "line_number": 1,
                "account_code": _COGS_ACCOUNT_CODE,
                "account_name": "Cost of Goods Sold",
                "debit_amount": cogs_amount,
                "credit_amount": Decimal("0.00"),
                "description": (
                    f"COGS for shipment {shipment.shipment_number} "
                    f"to {shipment.customer_name or shipment.customer_id}"
                ),
                "reference": shipment.shipment_number,
                "transaction_type": "shipment",
                "shipment_id": str(shipment.shipment_id),
            },
            {
                "line_number": 2,
                "account_code": _INVENTORY_ACCOUNT_CODE,
                "account_name": "Inventory",
                "debit_amount": Decimal("0.00"),
                "credit_amount": cogs_amount,
                "description": (
                    f"Inventory reduction for shipment {shipment.shipment_number}"
                ),
                "reference": shipment.shipment_number,
                "transaction_type": "shipment",
                "shipment_id": str(shipment.shipment_id),
            },
        ]

        # Verify balance within tolerance (defensive check)
        total_dr = sum(e["debit_amount"] for e in entries)
        total_cr = sum(e["credit_amount"] for e in entries)
        imbalance = abs(total_dr - total_cr)
        if imbalance > GL_BALANCE_TOLERANCE:
            logger.error(
                "gl_entry_imbalance_detected",
                service_name="transactions",
                component="ShipmentGenerator",
                total_debits=str(total_dr),
                total_credits=str(total_cr),
                imbalance=str(imbalance),
                shipment_number=shipment.shipment_number,
            )
            raise GLPostingError(
                f"GL entries for shipment {shipment.shipment_number} do not balance",
                details={
                    "total_debits": str(total_dr),
                    "total_credits": str(total_cr),
                    "imbalance": str(imbalance),
                },
            )

        return entries

    # ------------------------------------------------------------------
    # Sequential Numbering
    # ------------------------------------------------------------------

    def _generate_shipment_number(self, ship_date: date) -> str:
        """Generate a sequential shipment number.

        Format: ``SHP-YYYY-NNNN``

        Resets the sequence counter when the year changes to keep numbers
        scoped to the fiscal year.

        Args:
            ship_date: The shipment date (determines year component).

        Returns:
            Formatted shipment number string.
        """
        if ship_date.year != self._current_year:
            self._current_year = ship_date.year
            self._sequence_counter = 0

        self._sequence_counter += 1

        prefix = DOCUMENT_NUMBER_PREFIXES.get("shipment", "SHP")
        return self._generate_sequential_number(
            prefix, ship_date.year, self._sequence_counter,
        )

    # ------------------------------------------------------------------
    # Inventory Reduction
    # ------------------------------------------------------------------

    async def _reduce_inventory(self, shipment: ShipmentRecord) -> None:
        """Reduce inventory balances for each shipped line.

        For each line in the shipment, reduces on-hand inventory for
        ``product_id`` by ``shipped_quantity``. Uses the injected
        ``AccountBalanceManager`` when available for formal balance tracking;
        otherwise logs the reduction for later reconciliation.

        Args:
            shipment: The ShipmentRecord whose lines drive reductions.
        """
        for line in shipment.lines:
            if self._account_balance_manager is not None:
                try:
                    await self._account_balance_manager.update_balance(
                        account_code=_INVENTORY_ACCOUNT_CODE,
                        amount=line.cogs_amount,
                        direction="credit",
                        reference=shipment.shipment_number,
                    )
                except Exception as exc:
                    logger.warning(
                        "inventory_reduction_balance_manager_error",
                        service_name="transactions",
                        component="ShipmentGenerator",
                        product_id=line.product_id,
                        shipped_quantity=str(line.shipped_quantity),
                        error=str(exc),
                        shipment_number=shipment.shipment_number,
                    )

            logger.debug(
                "inventory_reduced",
                service_name="transactions",
                component="ShipmentGenerator",
                product_id=line.product_id,
                product_name=line.product_name,
                shipped_quantity=str(line.shipped_quantity),
                unit_cost=str(line.unit_cost),
                cogs_amount=str(line.cogs_amount),
                warehouse=line.warehouse_location,
                shipment_number=shipment.shipment_number,
            )

    # ------------------------------------------------------------------
    # Open Sales Order Retrieval
    # ------------------------------------------------------------------

    def _get_open_sales_orders(
        self,
        context: GenerationContext,
    ) -> List[OpenSalesOrderInfo]:
        """Retrieve open sales orders eligible for shipment.

        In simulation mode (no db_session_factory), generates synthetic
        open orders using the context's day_context or additional_params.
        When a database session is available, queries for confirmed/approved
        orders that have not been fully shipped.

        Args:
            context: The current GenerationContext carrying day_context
                and additional_params.

        Returns:
            List of OpenSalesOrderInfo instances ready for fulfillment.
        """
        # Check if open orders are provided via additional_params.
        # An empty list explicitly means "no orders available" and should NOT
        # fall through to the simulation fallback.
        if "open_sales_orders" in context.additional_params:
            open_orders_data = context.additional_params["open_sales_orders"]
            orders: List[OpenSalesOrderInfo] = []
            for od in open_orders_data:
                if isinstance(od, OpenSalesOrderInfo):
                    orders.append(od)
                elif isinstance(od, dict):
                    try:
                        orders.append(OpenSalesOrderInfo.model_validate(od))
                    except Exception as parse_err:
                        logger.warning(
                            "open_order_parse_failed",
                            service_name="transactions",
                            component="ShipmentGenerator",
                            error=str(parse_err),
                        )
            return orders

        # Check day_context for open orders
        if context.day_context is not None:
            day_ctx_orders = getattr(context.day_context, "open_sales_orders", None)
            if day_ctx_orders:
                return [
                    OpenSalesOrderInfo.model_validate(o) if isinstance(o, dict) else o
                    for o in day_ctx_orders
                    if isinstance(o, (dict, OpenSalesOrderInfo))
                ]

        # Simulation fallback — generate sample open orders for demo
        rng = random.Random(context.rng_seed + 1000)
        num_orders = rng.randint(1, min(5, context.batch_size))
        sample_orders: List[OpenSalesOrderInfo] = []

        for i in range(num_orders):
            order_date = context.current_date - timedelta(days=rng.randint(3, 15))
            lead_time = rng.randint(3, 7)
            num_lines = rng.randint(1, 5)

            lines: List[Dict[str, Any]] = []
            for ln_idx in range(num_lines):
                unit_cost = Decimal(str(rng.uniform(5.0, 200.0))).quantize(_PENNY)
                unit_price = (unit_cost * Decimal("1.40")).quantize(_PENNY)
                qty = Decimal(str(rng.randint(1, 100)))
                lines.append({
                    "product_id": f"PROD-{rng.randint(1000, 9999)}",
                    "product_name": f"Product {ln_idx + 1}",
                    "ordered_qty": str(qty),
                    "unit_price": str(unit_price),
                    "unit_cost": str(unit_cost),
                    "warehouse_location": rng.choice(["MAIN", "EAST", "WEST"]),
                })

            sample_orders.append(
                OpenSalesOrderInfo(
                    sales_order_id=uuid4(),
                    sales_order_number=f"SO-{order_date.year}-{i + 1:04d}",
                    customer_id=f"CUST-{rng.randint(1000, 9999)}",
                    customer_name=f"Customer {i + 1}",
                    order_date=order_date,
                    order_lines=lines,
                    lead_time_days=lead_time,
                ),
            )

        return sample_orders

    # ------------------------------------------------------------------
    # Shipment Line Building
    # ------------------------------------------------------------------

    def _build_shipment_lines(
        self,
        order_lines: List[Dict[str, Any]],
        rng: random.Random,
    ) -> List[ShipmentLine]:
        """Build shipment lines from sales order line data.

        For each SO line, determines the shipped quantity (may be partial)
        and computes COGS and line value using Decimal arithmetic.

        Args:
            order_lines: List of dicts with product_id, ordered_qty,
                unit_price, unit_cost, etc.
            rng: Seeded Random for partial shipment decisions.

        Returns:
            List of ShipmentLine instances.
        """
        shipment_lines: List[ShipmentLine] = []

        for idx, line_data in enumerate(order_lines, start=1):
            product_id = str(line_data.get("product_id", f"PROD-{idx}"))
            product_name = str(line_data.get("product_name", ""))

            # Parse quantities and prices as Decimal
            ordered_qty = Decimal(str(line_data.get("ordered_qty", "1")))
            unit_price = Decimal(str(line_data.get("unit_price", "0"))).quantize(_PENNY)
            unit_cost = Decimal(str(line_data.get("unit_cost", "0"))).quantize(_PENNY)
            warehouse = str(line_data.get("warehouse_location", "MAIN"))

            if ordered_qty <= Decimal("0"):
                continue

            # Determine shipped quantity: mostly full ship, occasionally partial
            # 90% chance of full shipment, 10% chance of partial (80-99% of order)
            if rng.random() < 0.90:
                shipped_qty = ordered_qty
            else:
                partial_pct = Decimal(str(rng.uniform(0.80, 0.99)))
                shipped_qty = (ordered_qty * partial_pct).quantize(
                    Decimal("1"), rounding=decimal.ROUND_DOWN,
                )
                if shipped_qty < Decimal("1"):
                    shipped_qty = Decimal("1")

            # COGS = shipped_quantity × unit_cost (NOT unit_price)
            cogs_amount = (shipped_qty * unit_cost).quantize(_PENNY)

            # Line value = shipped_quantity × unit_price
            line_value = (shipped_qty * unit_price).quantize(_PENNY)

            so_line_ref = str(line_data.get(
                "sales_order_line_reference",
                f"SOL-{idx}",
            ))

            shipment_lines.append(
                ShipmentLine(
                    line_number=idx,
                    product_id=product_id,
                    product_name=product_name,
                    ordered_quantity=ordered_qty,
                    shipped_quantity=shipped_qty,
                    unit_cost=unit_cost,
                    unit_price=unit_price,
                    cogs_amount=cogs_amount,
                    line_value=line_value,
                    sales_order_line_reference=so_line_ref,
                    warehouse_location=warehouse,
                ),
            )

        return shipment_lines

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return shipment generation metrics.

        Returns:
            Dictionary containing shipment counts, COGS totals,
            and carrier usage distribution.
        """
        return {
            "shipments_generated": self._shipments_generated,
            "total_cogs": str(self._total_cogs),
            "carriers_used": dict(self._carrier_usage),
            "sequence_counter": self._sequence_counter,
            "tracking_counter": self._tracking_counter,
        }
