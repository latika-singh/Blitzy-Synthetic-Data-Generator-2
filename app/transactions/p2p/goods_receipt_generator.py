"""Goods Receipt Generator — Stage 2 of the Procure-to-Pay (P2P) cycle.

Generates synthetic goods receipts against open purchase orders, handles
quantity variance, updates inventory balances, and posts GL journal entries.

Business Flow
-------------
1.  Find open, approved POs awaiting receipt (from context or fallback data).
2.  Determine receipt date (PO date + product/vendor lead time, 5–15
    business days by default using seeded RNG).
3.  Validate that the receipt date falls within an OPEN fiscal period.
4.  For each PO line, determine received quantity:
    *  ~80 % of receipts: ``received_quantity = ordered_quantity`` (exact).
    *  ~20 % of receipts: slight variance (over/under by up to ±5 %) using
       the seeded RNG.
    *  Partial receipts possible (receive a subset of lines).
5.  Calculate per-line ``extended_cost = received_quantity × unit_cost``.
6.  Calculate per-line ``variance_quantity = received - ordered`` and
    ``variance_pct``.
7.  Generate sequential receipt number: ``GR-YYYY-NNNN``.
8.  Create :class:`GoodsReceiptHeader` with embedded
    :class:`GoodsReceiptLine` items.
9.  Check discrepancy injection trigger via base class.
10. Prepare GL journal entries.
11. Delegate GL posting to ``GLPostingEngine`` via base class.
12. Route to warehouse clerk via ``WorkflowOrchestrator``
    (``ROLE_MAPPING`` ``goods_receipt → ["warehouse_clerk"]``).
13. Publish ``TransactionCreated`` event through the ``EventBus`` (ADR-001).

Key Business Rules
------------------
*   Goods receipt **MUST** reference a valid, approved Purchase Order.
*   Receipt date = PO date + lead time (configurable per product/vendor).
*   Quantity received may differ from PO quantity (over-receipt, under-receipt,
    partial receipt).
*   Inventory balance increases by received quantity × unit cost.
*   GL posting:

    *   **DR Inventory** — increase asset (per line ``extended_cost``).
    *   **CR AP Accrual / GRNI** — increase liability accrual (total cost).
    *   GRNI = Goods Received Not Invoiced — liability accrual cleared when
        the vendor invoice is processed.

*   Balance validation: ``SUM(debits) = SUM(credits)`` within ``$0.01``.
*   Atomicity: if **any** step in GL posting fails, the **entire** transaction
    is rolled back — no partial postings.
*   Sequential numbering: ``GR-YYYY-NNNN``.
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
:class:`GoodsReceiptLine`
    Pydantic V2 model for a single receipt line item.
:class:`GoodsReceiptHeader`
    Pydantic V2 model for a complete receipt header (with embedded lines).
:class:`GoodsReceiptGenerator`
    Concrete :class:`TransactionGenerator` subclass for goods receipt creation.
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
    DOCUMENT_NUMBER_PREFIXES,
    FINANCIAL_TOLERANCES,
    GL_BALANCE_TOLERANCE,
    RETRY_POLICIES,
)
from app.transactions.exceptions import (
    GLPostingError,
    TransactionGenerationError,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus
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
# Module-level constants — lead time and GL account mappings
# ---------------------------------------------------------------------------

_DEFAULT_LEAD_TIME_MIN_DAYS: int = 5
"""Minimum lead time in business days for goods delivery."""

_DEFAULT_LEAD_TIME_MAX_DAYS: int = 15
"""Maximum lead time in business days for goods delivery."""

_VARIANCE_PROBABILITY: float = 0.20
"""Probability of a receipt having quantity variance (~20 % of receipts)."""

_MAX_VARIANCE_PCT: Decimal = Decimal("0.05")
"""Maximum ±5 % variance applied to receipt quantities."""

_PARTIAL_RECEIPT_PROBABILITY: float = 0.10
"""Probability of a partial receipt (only subset of PO lines received)."""

_GL_INVENTORY_ACCOUNT: str = "1400-001"
"""Default Inventory GL account (DR side of goods receipt posting)."""

_GL_AP_ACCRUAL_ACCOUNT: str = "2100-001"
"""Default AP Accrual / GRNI GL account (CR side of goods receipt posting)."""

_DEFAULT_STORAGE_LOCATIONS: List[str] = [
    "WH-A-01",
    "WH-A-02",
    "WH-B-01",
    "WH-B-02",
    "WH-C-01",
]
"""Fallback warehouse storage location codes."""


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════════


class GoodsReceiptLine(BaseModel):
    """Single line item on a goods receipt.

    Represents a received quantity against a corresponding PO line.
    ``extended_cost`` **MUST** equal ``received_quantity × unit_cost`` and is
    computed using ``Decimal`` arithmetic — *never* ``float``.

    Variance fields track over-receipt and under-receipt:
    *   ``variance_quantity = received_quantity - ordered_quantity``
    *   ``variance_pct = (variance_quantity / ordered_quantity) * 100``

    Attributes:
        line_number: 1-based line sequence number on this receipt.
        po_line_number: Corresponding line number on the source PO.
        product_id: Product/item identifier (nullable for non-stock items).
        description: Human-readable product description from PO line.
        ordered_quantity: Quantity originally ordered on the PO line.
        received_quantity: Quantity actually received at the warehouse.
        unit_of_measure: ISO unit code (default ``"EA"`` — each).
        unit_cost: Cost per unit from the PO line pricing.
        extended_cost: ``received_quantity × unit_cost`` (Decimal, never float).
        variance_quantity: ``received - ordered`` (positive=over, negative=under).
        variance_pct: Variance as a percentage of ordered quantity.
        storage_location: Warehouse storage location code.
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
        default="", description="Product description"
    )
    ordered_quantity: Decimal = Field(
        ..., gt=Decimal("0"), description="Quantity ordered on PO"
    )
    received_quantity: Decimal = Field(
        ..., ge=Decimal("0"), description="Quantity actually received"
    )
    unit_of_measure: str = Field(
        default="EA", description="Unit of measure code"
    )
    unit_cost: Decimal = Field(
        ..., gt=Decimal("0"), description="Cost per unit from PO"
    )
    extended_cost: Decimal = Field(
        ..., description="received_quantity × unit_cost"
    )
    variance_quantity: Decimal = Field(
        default=Decimal("0"),
        description="received - ordered (positive = over, negative = under)",
    )
    variance_pct: Decimal = Field(
        default=Decimal("0"), description="Variance as percentage"
    )
    storage_location: str = Field(
        default="", description="Warehouse storage location"
    )


class GoodsReceiptHeader(BaseModel):
    """Goods receipt header with all metadata and embedded line items.

    Carries the full receipt state including PO reference, vendor info,
    financial totals, receipt status, and the list of
    :class:`GoodsReceiptLine` items.

    Financial Integrity:
        * ``total_cost`` = ``SUM(line.extended_cost for line in lines)``
        * All monetary fields are ``Decimal`` — never ``float``.

    Status Values:
        ``"received"`` — goods physically received at warehouse.
        ``"inspected"`` — quality inspection completed.
        ``"accepted"`` — goods accepted into inventory.
        ``"rejected"`` — goods rejected (quality failure, wrong items).

    Attributes:
        receipt_id: Unique receipt identifier (UUID).
        receipt_number: Sequential document number ``GR-YYYY-NNNN``.
        po_id: Linked Purchase Order identifier.
        po_number: PO number string for cross-reference.
        vendor_id: Vendor who shipped the goods.
        vendor_name: Vendor display name.
        receipt_date: Date goods were received at the warehouse.
        delivery_note: Vendor's delivery/packing note reference.
        total_cost: Sum of all line ``extended_cost`` values.
        currency: Currency code (``"USD"`` for MVP).
        status: Receipt lifecycle status.
        is_partial: Whether this receipt covers only a subset of PO lines.
        received_by: Warehouse clerk agent who processed the receipt.
        lines: Embedded list of :class:`GoodsReceiptLine` items.
        metadata: Extensible key-value pairs for downstream consumers.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    receipt_id: UUID = Field(
        default_factory=uuid4, description="Unique receipt identifier"
    )
    receipt_number: str = Field(
        ..., description="Sequential number GR-YYYY-NNNN"
    )
    po_id: UUID = Field(
        ..., description="Linked Purchase Order ID"
    )
    po_number: str = Field(
        default="", description="PO number for reference"
    )
    vendor_id: UUID = Field(
        ..., description="Vendor identifier"
    )
    vendor_name: str = Field(
        default="", description="Vendor display name"
    )
    receipt_date: date = Field(
        ..., description="Date goods were received"
    )
    delivery_note: str = Field(
        default="", description="Vendor's delivery note reference"
    )
    total_cost: Decimal = Field(
        default=Decimal("0"), description="Sum of line extended_costs"
    )
    currency: str = Field(
        default="USD", description="Currency code — USD only for MVP"
    )
    status: str = Field(
        default="received",
        description="received, inspected, accepted, rejected",
    )
    is_partial: bool = Field(
        default=False, description="Whether this is a partial receipt"
    )
    received_by: Optional[UUID] = Field(
        default=None, description="Warehouse clerk agent who received"
    )
    lines: List[GoodsReceiptLine] = Field(
        default_factory=list, description="Receipt line items"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extensible metadata"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GoodsReceiptGenerator
# ═══════════════════════════════════════════════════════════════════════════════


class GoodsReceiptGenerator(TransactionGenerator):
    """Generates goods receipts against open purchase orders.

    Stage 2 of the P2P pipeline (PO → **Receipt** → Invoice → Match → Payment).

    Business Flow
    -------------
    1.  Query for open, approved POs awaiting receipt.
    2.  Determine receipt date (PO date + product lead time).
    3.  Generate receipt quantities (may include variance).
    4.  Create receipt header and lines.
    5.  Calculate inventory impact.
    6.  Route to warehouse clerk via ``WorkflowOrchestrator``.
    7.  Post GL entries: DR Inventory, CR AP Accrual (GRNI).
    8.  Publish ``TransactionCreated`` event.

    GL Posting
    ----------
    *   **DR Inventory** (asset account 1400-001) — per line extended_cost
    *   **CR AP Accrual / GRNI** (liability account 2100-001) — total receipt cost
    *   GRNI = Goods Received Not Invoiced — liability accrual that gets cleared
        when the vendor invoice is processed.

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
        approval_system: Optional[Any] = None,
        discrepancy_injector: Optional[Any] = None,
        gl_posting_engine: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the GoodsReceiptGenerator with injected dependencies.

        Args:
            db_session_factory: Callable returning an async session context
                manager (e.g., ``get_session``).
            agent_registry: Project 2 :class:`AgentRegistry`.
            workflow_orchestrator: Project 2 :class:`WorkflowOrchestrator`.
            event_bus: Project 2 :class:`EventBus`.
            approval_system: Project 2 approval system instance.
            discrepancy_injector: P3 ``DiscrepancyInjector`` instance.
            gl_posting_engine: P3 ``GLPostingEngine`` instance.
            statistical_models: Dict of statistical model instances —
                may carry ``"amount_distribution"``, ``"selection_model"``,
                etc.
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

        # Instance-level receipt sequence counter for numbering
        self._receipt_sequence: int = 0

        # Retry policy reference for runtime inspection
        self._retry_policy: Dict[str, Any] = RETRY_POLICIES.get(
            "p2p_cycle_generation", {}
        )

        logger.info(
            "goods_receipt_generator_initialized",
            service_name="transactions",
            component="GoodsReceiptGenerator",
        )

    # ------------------------------------------------------------------
    # Abstract Method Implementations
    # ------------------------------------------------------------------

    async def generate(
        self, context: GenerationContext
    ) -> TransactionResult:
        """Generate a goods receipt against an open purchase order.

        Produces a :class:`TransactionResult` with artifacts keyed as:
        ``"receipt_header"``, ``"receipt_lines"`` — matching the
        ``REQUIRED_ARTIFACTS["goods_receipt"]`` specification from
        ``app/orchestration/transaction_orchestrator.py``.

        Args:
            context: Per-invocation generation context carrying
                ``current_date``, ``rng_seed``, ``simulation_id``, etc.

        Returns:
            A :class:`TransactionResult` with
            ``transaction_type="goods_receipt"`` and all required artifacts.

        Raises:
            TransactionGenerationError: On unrecoverable generation failure.
        """
        start_time = time.perf_counter()

        try:
            # 1. Create deterministic RNG from context seed
            rng: random.Random = self._create_seeded_rng(context)

            # 2. Obtain a PO to receive against (from context or fallback)
            po_data = self._resolve_purchase_order(rng, context)
            po_id: UUID = po_data["po_id"]
            po_number: str = po_data["po_number"]
            vendor_id: UUID = po_data["vendor_id"]
            vendor_name: str = po_data["vendor_name"]
            po_order_date: date = po_data["order_date"]
            po_lines_data: List[Dict[str, Any]] = po_data["lines"]

            # 3. Determine receipt date (PO date + lead time)
            lead_time: timedelta = self._calculate_lead_time(rng, context)
            receipt_date: date = po_order_date + lead_time

            # Ensure receipt_date does not exceed current simulation date
            if receipt_date > context.current_date:
                receipt_date = context.current_date

            # 4. Validate receipt date is in an open fiscal period
            if context.fiscal_period_status not in ("OPEN", ""):
                logger.warning(
                    "goods_receipt_period_not_open",
                    fiscal_period=context.fiscal_period,
                    fiscal_period_status=context.fiscal_period_status,
                    trace_id=str(context.trace_id),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )

            # 5. Determine whether this is a partial receipt
            is_partial: bool = rng.random() < _PARTIAL_RECEIPT_PROBABILITY

            # 6. Determine receipt quantities per PO line
            receipt_quantities: List[Decimal] = self._determine_receipt_quantities(
                rng, po_lines_data, is_partial
            )

            # 7. Build receipt lines
            receipt_lines: List[GoodsReceiptLine] = []
            for idx, (po_line, recv_qty) in enumerate(
                zip(po_lines_data, receipt_quantities), start=1
            ):
                if recv_qty <= Decimal("0"):
                    # Skip lines with zero received quantity (partial receipt)
                    continue

                ordered_qty = Decimal(str(po_line.get("quantity", po_line.get("ordered_quantity", "1"))))
                unit_cost = Decimal(str(po_line.get("unit_price", po_line.get("unit_cost", "1.00"))))
                extended_cost: Decimal = (recv_qty * unit_cost).quantize(
                    Decimal("0.01")
                )
                variance_quantity: Decimal = recv_qty - ordered_qty
                variance_pct: Decimal = (
                    (variance_quantity / ordered_qty * Decimal("100")).quantize(
                        Decimal("0.01")
                    )
                    if ordered_qty > Decimal("0")
                    else Decimal("0")
                )

                product_id_raw = po_line.get("product_id")
                product_id: Optional[UUID] = None
                if product_id_raw is not None:
                    product_id = (
                        UUID(product_id_raw)
                        if isinstance(product_id_raw, str)
                        else product_id_raw
                    )

                storage_location: str = rng.choice(_DEFAULT_STORAGE_LOCATIONS)
                description: str = po_line.get("description", "")

                receipt_lines.append(
                    GoodsReceiptLine(
                        line_number=len(receipt_lines) + 1,
                        po_line_number=po_line.get("line_number", idx),
                        product_id=product_id,
                        description=description,
                        ordered_quantity=ordered_qty,
                        received_quantity=recv_qty,
                        unit_of_measure=po_line.get("unit_of_measure", "EA"),
                        unit_cost=unit_cost,
                        extended_cost=extended_cost,
                        variance_quantity=variance_quantity,
                        variance_pct=variance_pct,
                        storage_location=storage_location,
                    )
                )

            # Handle edge case: no lines received (skip transaction)
            if not receipt_lines:
                elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
                return TransactionResult(
                    transaction_type="goods_receipt",
                    status="skipped",
                    artifacts={},
                    gl_entries=[],
                    amount=Decimal("0"),
                    duration_ms=elapsed_ms,
                    errors=["No lines received — receipt skipped"],
                    error_message="No lines received — receipt skipped",
                )

            # 8. Calculate total cost
            total_cost: Decimal = sum(
                (line.extended_cost for line in receipt_lines), Decimal("0")
            )

            # 9. Generate receipt number GR-YYYY-NNNN
            receipt_number: str = self._generate_receipt_number(
                receipt_date
            )

            # 10. Assign warehouse clerk agent if available
            received_by: Optional[UUID] = None
            if self._agent_registry is not None:
                try:
                    agents = self._agent_registry.get_agents_by_role(
                        "warehouse_clerk"
                    )
                    if agents:
                        selected_agent = rng.choice(agents)
                        received_by = getattr(
                            selected_agent, "agent_id", None
                        )
                except Exception as agent_exc:
                    logger.warning(
                        "warehouse_clerk_assignment_failed",
                        error=str(agent_exc),
                        trace_id=str(context.trace_id),
                        service_name="transactions",
                        component="GoodsReceiptGenerator",
                    )

            # 11. Determine partial receipt status
            actual_is_partial: bool = len(receipt_lines) < len(po_lines_data)

            # 12. Generate delivery note reference
            delivery_note: str = f"DN-{rng.randint(100000, 999999)}"

            # 13. Create receipt header
            receipt_header = GoodsReceiptHeader(
                receipt_number=receipt_number,
                po_id=po_id,
                po_number=po_number,
                vendor_id=vendor_id,
                vendor_name=vendor_name,
                receipt_date=receipt_date,
                delivery_note=delivery_note,
                total_cost=total_cost,
                currency="USD",
                status="received",
                is_partial=actual_is_partial,
                received_by=received_by,
                lines=receipt_lines,
                metadata={
                    "simulation_id": str(context.simulation_id),
                    "trace_id": str(context.trace_id),
                    "fiscal_period": context.fiscal_period,
                    "lead_time_days": lead_time.days,
                    "variance_lines": sum(
                        1
                        for line in receipt_lines
                        if line.variance_quantity != Decimal("0")
                    ),
                },
            )

            # 14. Check discrepancy trigger
            should_inject, disc_type, disc_params = (
                await self._check_discrepancy_trigger(
                    context, "goods_receipt"
                )
            )

            # 15. Prepare GL entries
            gl_entries: List[Dict[str, Any]] = self._prepare_gl_entries(
                receipt_header, context
            )

            # 16. Delegate GL posting to GLPostingEngine
            try:
                await self._delegate_gl_posting(gl_entries, context)
            except GLPostingError as gl_exc:
                # Atomicity: full rollback — do not persist partial postings
                logger.error(
                    "goods_receipt_gl_posting_failed",
                    receipt_number=receipt_number,
                    error=str(gl_exc),
                    trace_id=str(context.trace_id),
                    simulation_id=str(context.simulation_id),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
                raise TransactionGenerationError(
                    message=f"GL posting failed for goods receipt {receipt_number}: {gl_exc}",
                    details={
                        "receipt_number": receipt_number,
                        "total_cost": str(total_cost),
                        "gl_entry_count": len(gl_entries),
                        "error_type": "GLPostingError",
                        "trace_id": str(context.trace_id),
                    },
                ) from gl_exc

            # 17. Route to warehouse clerk via WorkflowOrchestrator
            if self._workflow_orchestrator is not None:
                try:
                    await self._workflow_orchestrator.route_transaction(
                        transaction_type="goods_receipt",
                        transaction_id=str(receipt_header.receipt_id),
                        payload={
                            "receipt_number": receipt_number,
                            "po_number": po_number,
                            "vendor_id": str(vendor_id),
                            "total_cost": str(total_cost),
                        },
                    )
                except Exception as route_exc:
                    logger.warning(
                        "goods_receipt_routing_failed",
                        error=str(route_exc),
                        receipt_number=receipt_number,
                        trace_id=str(context.trace_id),
                        service_name="transactions",
                        component="GoodsReceiptGenerator",
                    )

            # 18. Publish TransactionCreated event
            await self._publish_event(
                event_type="TransactionCreated",
                payload={
                    "transaction_type": "goods_receipt",
                    "transaction_id": str(receipt_header.receipt_id),
                    "receipt_number": receipt_number,
                    "po_id": str(po_id),
                    "po_number": po_number,
                    "vendor_id": str(vendor_id),
                    "vendor_name": vendor_name,
                    "total_cost": str(total_cost),
                    "is_partial": actual_is_partial,
                    "line_count": len(receipt_lines),
                    "status": "received",
                },
                context=context,
            )

            # 19. Build and return TransactionResult
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0

            result = TransactionResult(
                transaction_type="goods_receipt",
                status="completed",
                artifacts={
                    "receipt_header": receipt_header.model_dump(mode="json"),
                    "receipt_lines": [
                        line.model_dump(mode="json") for line in receipt_lines
                    ],
                },
                gl_entries=gl_entries,
                has_discrepancy=should_inject,
                discrepancy_type=disc_type,
                amount=total_cost,
                duration_ms=elapsed_ms,
            )

            # 20. Log the generated transaction
            self._log_transaction(result, context)

            return result

        except TransactionGenerationError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1_000.0
            logger.error(
                "goods_receipt_generation_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                duration_ms=elapsed_ms,
                service_name="transactions",
                component="GoodsReceiptGenerator",
            )
            raise TransactionGenerationError(
                message=f"Goods receipt generation failed: {exc}",
                details={
                    "error_type": type(exc).__name__,
                    "trace_id": str(context.trace_id),
                    "simulation_id": str(context.simulation_id),
                    "generator_class": "GoodsReceiptGenerator",
                },
            ) from exc

    async def validate(self, result: TransactionResult) -> bool:
        """Validate a generated goods receipt against business rules.

        Checks:
        *   ``receipt_header`` artifact exists with required fields.
        *   ``receipt_lines`` artifact is non-empty.
        *   All quantities and costs are ``Decimal``-compatible.
        *   ``total_cost = SUM(line.extended_cost)``.
        *   Per-line: ``extended_cost = received_quantity × unit_cost``.
        *   GL entries balance: ``SUM(debits) = SUM(credits)`` within $0.01.
        *   Receipt references a valid PO (``po_id`` is present).

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if all validation checks pass, ``False`` otherwise.
        """
        try:
            artifacts = result.artifacts

            # 1. Artifact presence — receipt_header
            if "receipt_header" not in artifacts:
                logger.warning(
                    "gr_validation_failed_missing_header",
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
                return False

            # 2. Artifact presence — receipt_lines
            if "receipt_lines" not in artifacts or not artifacts["receipt_lines"]:
                logger.warning(
                    "gr_validation_failed_missing_lines",
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
                return False

            header = artifacts["receipt_header"]
            lines = artifacts["receipt_lines"]

            # 3. PO reference validation
            if not header.get("po_id"):
                logger.warning(
                    "gr_validation_failed_missing_po_ref",
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
                return False

            # 4. Validate line-level arithmetic
            computed_total = Decimal("0")
            for line_data in lines:
                recv_qty = Decimal(str(line_data["received_quantity"]))
                unit_cost = Decimal(str(line_data["unit_cost"]))
                ext_cost = Decimal(str(line_data["extended_cost"]))
                expected_ext = (recv_qty * unit_cost).quantize(Decimal("0.01"))

                if abs(ext_cost - expected_ext) > GL_BALANCE_TOLERANCE:
                    logger.warning(
                        "gr_validation_line_arithmetic_error",
                        line_number=line_data.get("line_number"),
                        expected=str(expected_ext),
                        actual=str(ext_cost),
                        service_name="transactions",
                        component="GoodsReceiptGenerator",
                    )
                    return False

                computed_total += ext_cost

            # 5. Validate header total matches sum of lines
            header_total = Decimal(str(header["total_cost"]))
            if abs(header_total - computed_total) > GL_BALANCE_TOLERANCE:
                logger.warning(
                    "gr_validation_total_mismatch",
                    header_total=str(header_total),
                    computed_total=str(computed_total),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
                return False

            # 6. Validate GL entries balance (DR = CR within $0.01)
            if result.gl_entries:
                total_debits = Decimal("0")
                total_credits = Decimal("0")
                for entry in result.gl_entries:
                    total_debits += Decimal(str(entry.get("debit", "0")))
                    total_credits += Decimal(str(entry.get("credit", "0")))

                imbalance = abs(total_debits - total_credits)
                if imbalance > GL_BALANCE_TOLERANCE:
                    logger.warning(
                        "gr_validation_gl_imbalance",
                        total_debits=str(total_debits),
                        total_credits=str(total_credits),
                        imbalance=str(imbalance),
                        tolerance=str(GL_BALANCE_TOLERANCE),
                        transaction_id=str(result.transaction_id),
                        service_name="transactions",
                        component="GoodsReceiptGenerator",
                    )
                    return False

            # 7. Validate variance calculations
            for line_data in lines:
                ordered_qty = Decimal(str(line_data["ordered_quantity"]))
                recv_qty = Decimal(str(line_data["received_quantity"]))
                variance_qty = Decimal(str(line_data["variance_quantity"]))
                expected_variance = recv_qty - ordered_qty

                if abs(variance_qty - expected_variance) > Decimal("0.001"):
                    logger.warning(
                        "gr_validation_variance_error",
                        line_number=line_data.get("line_number"),
                        expected_variance=str(expected_variance),
                        actual_variance=str(variance_qty),
                        service_name="transactions",
                        component="GoodsReceiptGenerator",
                    )
                    return False

            logger.debug(
                "gr_validation_passed",
                transaction_id=str(result.transaction_id),
                total_cost=str(header_total),
                line_count=len(lines),
                service_name="transactions",
                component="GoodsReceiptGenerator",
            )
            return True

        except Exception as exc:
            logger.error(
                "gr_validation_error",
                error=str(exc),
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="GoodsReceiptGenerator",
            )
            return False

    async def post(self, result: TransactionResult) -> None:
        """Post goods receipt GL entries to the General Ledger.

        GL entries for a goods receipt:
        *   **DR Inventory** (asset account) — per line extended_cost
        *   **CR AP Accrual / GRNI** (liability account) — total receipt cost

        GRNI = Goods Received Not Invoiced — the credit side creates a
        liability accrual that gets cleared when the vendor invoice is
        processed at the Invoice Processor stage.

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
                "gr_post_no_gl_entries",
                transaction_id=str(result.transaction_id),
                service_name="transactions",
                component="GoodsReceiptGenerator",
            )
            return

        # Pre-posting balance validation
        total_debits = Decimal("0")
        total_credits = Decimal("0")
        for entry in result.gl_entries:
            total_debits += Decimal(str(entry.get("debit", "0")))
            total_credits += Decimal(str(entry.get("credit", "0")))

        imbalance = abs(total_debits - total_credits)
        if imbalance > GL_BALANCE_TOLERANCE:
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
                    "tolerance": str(GL_BALANCE_TOLERANCE),
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
                    "gr_gl_entries_posted",
                    transaction_id=str(result.transaction_id),
                    entry_count=len(result.gl_entries),
                    total_debits=str(total_debits),
                    total_credits=str(total_credits),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
            except Exception as exc:
                logger.error(
                    "gr_gl_posting_failed",
                    error=str(exc),
                    transaction_id=str(result.transaction_id),
                    service_name="transactions",
                    component="GoodsReceiptGenerator",
                )
                raise GLPostingError(
                    message=f"GL posting failed for goods receipt: {exc}",
                    details={
                        "transaction_id": str(result.transaction_id),
                        "entry_count": len(result.gl_entries),
                        "error_type": type(exc).__name__,
                    },
                ) from exc
        else:
            logger.info(
                "gr_post_skipped_no_engine",
                transaction_id=str(result.transaction_id),
                entry_count=len(result.gl_entries),
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                service_name="transactions",
                component="GoodsReceiptGenerator",
            )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _determine_receipt_quantities(
        self,
        rng: random.Random,
        po_lines: List[Dict[str, Any]],
        is_partial: bool,
    ) -> List[Decimal]:
        """Determine received quantities for each PO line.

        Applies the following logic per line:
        *   ~80 % of lines: exact match (``received = ordered``).
        *   ~20 % of lines: slight variance (±5 % max) using seeded RNG.
        *   Partial receipts: some lines get zero quantity (not received yet).

        All quantities use ``Decimal`` — never ``float``.

        Args:
            rng: Seeded ``random.Random`` instance for deterministic output.
            po_lines: List of PO line dicts with ``quantity`` or
                ``ordered_quantity`` key.
            is_partial: If ``True``, some lines may receive zero quantity.

        Returns:
            List of ``Decimal`` received quantities (one per PO line,
            in the same order as ``po_lines``).
        """
        quantities: List[Decimal] = []

        for po_line in po_lines:
            ordered_qty = Decimal(
                str(po_line.get("quantity", po_line.get("ordered_quantity", "1")))
            )

            # Partial receipt: randomly skip some lines (30 % skip probability)
            if is_partial and rng.random() < 0.30:
                quantities.append(Decimal("0"))
                continue

            # Determine if this line has quantity variance
            if rng.random() < _VARIANCE_PROBABILITY:
                # Apply variance: ±5 % range
                variance_factor = Decimal(
                    str(rng.uniform(
                        float(Decimal("1") - _MAX_VARIANCE_PCT),
                        float(Decimal("1") + _MAX_VARIANCE_PCT),
                    ))
                )
                received_qty = (ordered_qty * variance_factor).quantize(
                    Decimal("0.01")
                )
                # Ensure received quantity is never negative
                if received_qty < Decimal("0"):
                    received_qty = Decimal("0.01")
            else:
                # Exact match — most common case
                received_qty = ordered_qty

            quantities.append(received_qty)

        return quantities

    def _calculate_lead_time(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> timedelta:
        """Calculate goods delivery lead time.

        Default range: 5–15 business days, determined using the seeded RNG.
        May use vendor-specific or product-specific lead times from context
        ``additional_params`` if available.

        Args:
            rng: Seeded ``random.Random`` instance.
            context: Current generation context (may carry vendor-specific
                lead time in ``additional_params``).

        Returns:
            A :class:`timedelta` representing the delivery lead time.
        """
        # Check for context-provided lead time override
        custom_lead_time = context.additional_params.get("lead_time_days")
        if custom_lead_time is not None:
            try:
                return timedelta(days=int(custom_lead_time))
            except (TypeError, ValueError):
                pass

        # Default: random lead time within configured range
        lead_time_days: int = rng.randint(
            _DEFAULT_LEAD_TIME_MIN_DAYS, _DEFAULT_LEAD_TIME_MAX_DAYS
        )
        return timedelta(days=lead_time_days)

    def _prepare_gl_entries(
        self,
        receipt_header: GoodsReceiptHeader,
        context: GenerationContext,
    ) -> List[Dict[str, Any]]:
        """Create GL journal entries for the goods receipt.

        Generates balanced journal entries per the GRNI accounting model:
        *   **DR Inventory** (asset) — per line ``extended_cost``
        *   **CR AP Accrual / GRNI** (liability) — total receipt cost

        Balance validation: ``SUM(debits) = SUM(credits)`` within $0.01
        (``GL_BALANCE_TOLERANCE``).

        The DR side is broken into per-line entries to maintain line-level
        audit trail, while the CR side is a single summary entry for the
        total cost.

        Args:
            receipt_header: The :class:`GoodsReceiptHeader` to create
                GL entries for.
            context: Current generation context for trace correlation.

        Returns:
            List of GL entry dicts, each with ``account``, ``debit``,
            ``credit``, ``description``, and metadata fields.
        """
        gl_entries: List[Dict[str, Any]] = []
        total_debit = Decimal("0")

        # DR Inventory — one entry per receipt line for audit trail
        for line in receipt_header.lines:
            debit_amount = line.extended_cost

            # Check for custom inventory account from context
            inventory_account = context.additional_params.get(
                "gl_inventory_account", _GL_INVENTORY_ACCOUNT
            )

            gl_entries.append({
                "account": inventory_account,
                "debit": str(debit_amount),
                "credit": str(Decimal("0")),
                "description": (
                    f"Inventory receipt — {line.description or 'Line ' + str(line.line_number)} "
                    f"(GR: {receipt_header.receipt_number}, PO: {receipt_header.po_number})"
                ),
                "transaction_type": "goods_receipt",
                "transaction_id": str(receipt_header.receipt_id),
                "receipt_number": receipt_header.receipt_number,
                "line_number": line.line_number,
                "fiscal_period": context.fiscal_period,
                "posting_date": str(receipt_header.receipt_date),
                "trace_id": str(context.trace_id),
            })
            total_debit += debit_amount

        # CR AP Accrual / GRNI — single summary entry
        ap_accrual_account = context.additional_params.get(
            "gl_ap_accrual_account", _GL_AP_ACCRUAL_ACCOUNT
        )
        gl_entries.append({
            "account": ap_accrual_account,
            "debit": str(Decimal("0")),
            "credit": str(total_debit),
            "description": (
                f"AP Accrual (GRNI) — {receipt_header.receipt_number} "
                f"(Vendor: {receipt_header.vendor_name})"
            ),
            "transaction_type": "goods_receipt",
            "transaction_id": str(receipt_header.receipt_id),
            "receipt_number": receipt_header.receipt_number,
            "fiscal_period": context.fiscal_period,
            "posting_date": str(receipt_header.receipt_date),
            "trace_id": str(context.trace_id),
        })

        # Balance validation: DR = CR within tolerance
        total_credit = total_debit  # By construction, credit equals debit sum
        imbalance = abs(total_debit - total_credit)
        if imbalance > GL_BALANCE_TOLERANCE:
            logger.error(
                "gr_gl_entries_imbalanced",
                total_debit=str(total_debit),
                total_credit=str(total_credit),
                imbalance=str(imbalance),
                receipt_number=receipt_header.receipt_number,
                service_name="transactions",
                component="GoodsReceiptGenerator",
            )

        logger.debug(
            "gr_gl_entries_prepared",
            entry_count=len(gl_entries),
            total_debit=str(total_debit),
            total_credit=str(total_credit),
            receipt_number=receipt_header.receipt_number,
            trace_id=str(context.trace_id),
            service_name="transactions",
            component="GoodsReceiptGenerator",
        )

        return gl_entries

    def _generate_receipt_number(self, receipt_date: date) -> str:
        """Generate a sequential goods receipt number: ``GR-YYYY-NNNN``.

        Uses the base class :meth:`_generate_sequential_number` with the
        ``"GR"`` prefix from ``DOCUMENT_NUMBER_PREFIXES["goods_receipt"]``.

        Args:
            receipt_date: The receipt date (used for the YYYY component).

        Returns:
            Formatted receipt number string (e.g., ``"GR-2024-0001"``).
        """
        self._receipt_sequence += 1
        prefix: str = DOCUMENT_NUMBER_PREFIXES.get("goods_receipt", "GR")
        return self._generate_sequential_number(
            prefix=prefix,
            year=receipt_date.year,
            sequence=self._receipt_sequence,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _resolve_purchase_order(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> Dict[str, Any]:
        """Resolve a purchase order to generate a receipt against.

        Attempts to retrieve PO data from:
        1.  ``context.additional_params["purchase_order"]`` — direct PO data
            injected by the simulation engine or test harness.
        2.  ``context.day_context`` — day-level state with open PO list.
        3.  Fallback — generates a synthetic PO reference for standalone
            testing when no upstream data source is available.

        Args:
            rng: Seeded ``random.Random`` instance.
            context: Current generation context.

        Returns:
            Dictionary with keys: ``po_id``, ``po_number``, ``vendor_id``,
            ``vendor_name``, ``order_date``, ``lines``.
        """
        # Attempt 1: Direct PO data from additional_params
        po_data = context.additional_params.get("purchase_order")
        if po_data is not None:
            return self._normalize_po_data(po_data)

        # Attempt 2: Day context with open POs
        if context.day_context is not None:
            try:
                open_pos = getattr(context.day_context, "open_purchase_orders", None)
                if open_pos and len(open_pos) > 0:
                    selected_po = rng.choice(open_pos)
                    return self._normalize_po_data(selected_po)
            except (AttributeError, TypeError, IndexError):
                pass

        # Attempt 3: Additional params list of POs
        po_list = context.additional_params.get("purchase_orders")
        if po_list and len(po_list) > 0:
            selected_po = rng.choice(po_list)
            return self._normalize_po_data(selected_po)

        # Fallback: generate synthetic PO reference
        return self._generate_fallback_po(rng, context)

    def _normalize_po_data(self, po_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize PO data dictionary to ensure consistent field names.

        Handles variations in field naming between different PO sources
        (PurchaseOrderHeader.model_dump output, database records, etc.).

        Args:
            po_data: Raw PO dictionary from any source.

        Returns:
            Normalized dictionary with canonical field names.
        """
        po_id_raw = po_data.get("po_id", po_data.get("id", str(uuid4())))
        po_id = UUID(po_id_raw) if isinstance(po_id_raw, str) else po_id_raw

        vendor_id_raw = po_data.get("vendor_id", str(uuid4()))
        vendor_id = (
            UUID(vendor_id_raw) if isinstance(vendor_id_raw, str) else vendor_id_raw
        )

        order_date_raw = po_data.get("order_date")
        if isinstance(order_date_raw, str):
            order_date = date.fromisoformat(order_date_raw)
        elif isinstance(order_date_raw, date):
            order_date = order_date_raw
        else:
            order_date = date.today()

        # Normalize lines — support both "lines" and "po_lines" keys
        lines = po_data.get("lines", po_data.get("po_lines", []))

        return {
            "po_id": po_id,
            "po_number": po_data.get("po_number", ""),
            "vendor_id": vendor_id,
            "vendor_name": po_data.get("vendor_name", "Unknown Vendor"),
            "order_date": order_date,
            "lines": lines,
        }

    def _generate_fallback_po(
        self,
        rng: random.Random,
        context: GenerationContext,
    ) -> Dict[str, Any]:
        """Generate a synthetic PO reference for standalone testing.

        Creates a minimal but realistic PO structure when no upstream
        PO data is available (e.g., during unit tests or initial system
        bootstrapping).

        Args:
            rng: Seeded ``random.Random`` instance.
            context: Current generation context.

        Returns:
            Dictionary with synthetic PO data matching the normalized schema.
        """
        po_id = uuid4()
        vendor_id = uuid4()
        num_lines = rng.randint(1, 5)

        fallback_products = [
            ("Standard Office Paper", Decimal("12.99")),
            ("Printer Toner Cartridge", Decimal("89.50")),
            ("Ethernet Cable Cat6 100ft", Decimal("24.75")),
            ("Desk Chair Ergonomic", Decimal("399.00")),
            ("LED Monitor 27in", Decimal("329.99")),
            ("USB-C Hub Multiport", Decimal("59.99")),
            ("Whiteboard Markers 12pk", Decimal("8.49")),
            ("Server Rack 42U", Decimal("1249.00")),
            ("Wireless Mouse", Decimal("29.99")),
            ("Mechanical Keyboard", Decimal("149.99")),
        ]

        lines: List[Dict[str, Any]] = []
        for idx in range(1, num_lines + 1):
            product_desc, unit_price = rng.choice(fallback_products)
            quantity = Decimal(str(rng.randint(1, 100)))
            lines.append({
                "line_number": idx,
                "product_id": str(uuid4()),
                "description": product_desc,
                "quantity": str(quantity),
                "unit_price": str(unit_price),
                "unit_of_measure": "EA",
                "extended_amount": str(
                    (quantity * unit_price).quantize(Decimal("0.01"))
                ),
            })

        # PO order date is earlier than current simulation date
        days_back = rng.randint(
            _DEFAULT_LEAD_TIME_MIN_DAYS, _DEFAULT_LEAD_TIME_MAX_DAYS
        )
        po_order_date = context.current_date - timedelta(days=days_back)

        logger.debug(
            "goods_receipt_using_fallback_po",
            po_id=str(po_id),
            line_count=num_lines,
            po_order_date=str(po_order_date),
            trace_id=str(context.trace_id),
            service_name="transactions",
            component="GoodsReceiptGenerator",
        )

        return {
            "po_id": po_id,
            "po_number": f"PO-{context.current_date.year}-{rng.randint(1, 9999):04d}",
            "vendor_id": vendor_id,
            "vendor_name": rng.choice([
                "Acme Office Supplies",
                "Global Tech Components",
                "Premier Industrial Parts",
                "Reliable Paper Co",
                "Fast Freight Logistics",
            ]),
            "order_date": po_order_date,
            "lines": lines,
        }
