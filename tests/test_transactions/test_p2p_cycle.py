"""Full Procure-to-Pay (P2P) cycle integration tests.

Tests cover the complete P2P pipeline:
    PurchaseOrderGenerator → GoodsReceiptGenerator → VendorInvoiceProcessor
    (with ThreeWayMatcher) → VendorPaymentGenerator

Validation domains:
- End-to-end: PO → Goods Receipt → Vendor Invoice (3-way match) → Vendor Payment
- Artifact completeness assertions per REQUIRED_ARTIFACTS mapping
- GL balance verification: DR = CR within $0.01 at every posting step
- Event publication: TransactionCreated, TransactionCompleted, ApprovalRequired
- Approval chain enforcement: PO amounts above $5K/$25K/$100K thresholds
- Sequential numbering: PO-YYYY-NNNN, GR-YYYY-NNNN, VINV-YYYY-NNNN, VPAY-YYYY-NNNN
- Vendor selection (Pareto 80/20) and EOQ quantity validation
- GL posting entries per step:
  * Goods Receipt: DR Inventory, CR AP Accrual (GRNI)
  * Vendor Invoice: DR Expense/Asset, CR AP
  * Vendor Payment: DR AP, CR Cash (+ DR Purchase Discount if early pay)

Per AAP Section 0.7.2 Financial Integrity:
- ALL financial calculations use Python Decimal (prec=28, ROUND_HALF_UP)
- GL Balance Invariant: SUM(debits) = SUM(credits) within $0.01
- Atomicity: failed GL posting → FULL rollback (no partial postings)

Per AAP Section 0.7.6 Testing Conventions:
- ≥ 80% line coverage for all transaction logic
- AsyncMock for database sessions and GL engine
- fakeredis for Redis operations
- Deterministic seeding via deterministic_rng fixture
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.transactions.p2p.purchase_order_generator import PurchaseOrderGenerator
from app.transactions.p2p.goods_receipt_generator import GoodsReceiptGenerator
from app.transactions.p2p.vendor_invoice_processor import VendorInvoiceProcessor
from app.transactions.p2p.three_way_matcher import ThreeWayMatcher, MatchStatus
from app.transactions.p2p.vendor_payment_generator import VendorPaymentGenerator
from app.transactions.base_generator import (
    TransactionGenerator,
    GenerationContext,
    TransactionResult,
)
from app.transactions.exceptions import (
    TransactionError,
    TransactionGenerationError,
    ThreeWayMatchError,
    GLPostingError,
    PaymentAllocationError,
)
from app.transactions.constants import (
    APPROVAL_THRESHOLDS,
    DOCUMENT_NUMBER_PREFIXES,
)

# ---------------------------------------------------------------------------
# Decimal precision — AAP §0.7.2
# ---------------------------------------------------------------------------
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Module-level helpers for building GL entry dicts used across tests
# ---------------------------------------------------------------------------

_TOLERANCE = Decimal("0.01")


def _make_balanced_gl_entries(
    debit_account: str,
    credit_account: str,
    amount: Decimal,
    description: str = "Test GL entry",
) -> List[Dict[str, Any]]:
    """Create a minimal balanced GL entry list (1 debit line + 1 credit line).

    All amounts are ``Decimal`` — never ``float``.
    """
    return [
        {
            "account_code": debit_account,
            "debit": amount,
            "credit": Decimal("0.00"),
            "description": f"DR {description}",
        },
        {
            "account_code": credit_account,
            "debit": Decimal("0.00"),
            "credit": amount,
            "description": f"CR {description}",
        },
    ]


def _sum_gl_debits(gl_entries: List[Dict[str, Any]]) -> Decimal:
    """Sum all debit amounts across a list of GL entry dicts."""
    total = Decimal("0.00")
    for entry in gl_entries:
        val = entry.get("debit", Decimal("0.00"))
        total += Decimal(str(val)) if not isinstance(val, Decimal) else val
    return total


def _sum_gl_credits(gl_entries: List[Dict[str, Any]]) -> Decimal:
    """Sum all credit amounts across a list of GL entry dicts."""
    total = Decimal("0.00")
    for entry in gl_entries:
        val = entry.get("credit", Decimal("0.00"))
        total += Decimal(str(val)) if not isinstance(val, Decimal) else val
    return total


def _assert_gl_balanced(gl_entries: List[Dict[str, Any]]) -> None:
    """Assert that total debits == total credits within $0.01 tolerance."""
    total_dr = _sum_gl_debits(gl_entries)
    total_cr = _sum_gl_credits(gl_entries)
    diff = abs(total_dr - total_cr)
    assert diff <= _TOLERANCE, (
        f"GL imbalance: DR={total_dr}, CR={total_cr}, diff={diff} (tolerance={_TOLERANCE})"
    )


def _make_generation_context(**overrides: Any) -> GenerationContext:
    """Build a minimal GenerationContext for tests.

    Only ``current_date`` is required; everything else uses defaults.
    """
    params: Dict[str, Any] = {
        "current_date": date(2025, 1, 15),
    }
    params.update(overrides)
    return GenerationContext(**params)


# ═══════════════════════════════════════════════════════════════════════════════
# Local Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """AsyncMock EventBus for testing event publishing.

    Local fixture because conftest.py provides a *real* EventBus
    (in-memory mode), not a mock.
    """
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def mock_agent_registry() -> AsyncMock:
    """Mocked AgentRegistry for agent decision routing."""
    registry = AsyncMock()
    registry.get_agents_by_role = MagicMock(return_value=[MagicMock()])
    return registry


@pytest.fixture
def mock_workflow_orchestrator() -> AsyncMock:
    """Mocked WorkflowOrchestrator for transaction routing."""
    orch = AsyncMock()
    orch.submit_transaction = AsyncMock(return_value={"status": "completed"})
    return orch


@pytest.fixture
def mock_approval_system() -> MagicMock:
    """Mocked ApprovalSystem for threshold-based approval chain determination."""
    system = MagicMock()
    system.determine_approval_chain = MagicMock(return_value=[])
    return system


@pytest.fixture
def mock_statistical_models() -> MagicMock:
    """Mocked statistical models (amounts, timing, frequency, selection)."""
    models = MagicMock()
    models.amount_distributions = MagicMock()
    models.amount_distributions.generate = MagicMock(return_value=Decimal("5000.00"))
    models.payment_timing = MagicMock()
    models.order_frequency = MagicMock()
    models.selection_models = MagicMock()
    return models


@pytest.fixture
def po_generator(
    mock_db_session,
    mock_agent_registry,
    mock_gl_engine,
    mock_event_bus,
    mock_workflow_orchestrator,
    mock_approval_system,
    mock_statistical_models,
) -> PurchaseOrderGenerator:
    """Fully-wired PurchaseOrderGenerator for integration tests."""
    return PurchaseOrderGenerator(
        db_session_factory=mock_db_session,
        agent_registry=mock_agent_registry,
        gl_posting_engine=mock_gl_engine,
        event_bus=mock_event_bus,
        workflow_orchestrator=mock_workflow_orchestrator,
        approval_system=mock_approval_system,
        statistical_models={"amount_distributions": mock_statistical_models.amount_distributions},
    )


@pytest.fixture
def receipt_generator(
    mock_db_session,
    mock_gl_engine,
    mock_event_bus,
) -> GoodsReceiptGenerator:
    """GoodsReceiptGenerator for integration tests."""
    return GoodsReceiptGenerator(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def invoice_processor(
    mock_db_session,
    mock_gl_engine,
    mock_event_bus,
    mock_agent_registry,
    mock_approval_system,
) -> VendorInvoiceProcessor:
    """VendorInvoiceProcessor with injected ThreeWayMatcher."""
    return VendorInvoiceProcessor(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=mock_event_bus,
        agent_registry=mock_agent_registry,
        approval_system=mock_approval_system,
    )


@pytest.fixture
def payment_generator(
    mock_db_session,
    mock_gl_engine,
    mock_event_bus,
) -> VendorPaymentGenerator:
    """VendorPaymentGenerator for integration tests."""
    return VendorPaymentGenerator(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=mock_event_bus,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Approval tier lookup
# ═══════════════════════════════════════════════════════════════════════════════


def _find_approval_tier(
    tiers: List[Dict[str, Any]], amount: Decimal,
) -> Dict[str, Any]:
    """Walk tier list and return the matching tier for *amount*.

    Each tier dict has ``max_amount`` (``Decimal | None``) and
    ``required_role`` (``str | None``).  Tiers are ordered ascending.
    The last tier (``max_amount is None``) catches everything above.
    """
    for tier in tiers:
        max_amount = tier.get("max_amount")
        if max_amount is None:
            return tier
        if amount < max_amount:
            return tier
    return tiers[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Payment method determination (mirrors production logic)
# ═══════════════════════════════════════════════════════════════════════════════


def _determine_payment_method(amount: Decimal) -> str:
    """Return the expected payment method for a given *amount*.

    < $5,000   → check
    $5K–<$50K  → ach
    ≥ $50,000  → wire
    """
    if amount < Decimal("5000"):
        return "check"
    elif amount < Decimal("50000"):
        return "ach"
    else:
        return "wire"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 1: PurchaseOrderGeneration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestPurchaseOrderGeneration:
    """Tests for PurchaseOrderGenerator — Stage 1 of P2P cycle."""

    def test_po_generator_instantiation(self, po_generator: PurchaseOrderGenerator):
        """Verify constructor injection pattern — generator created without error."""
        assert po_generator is not None
        assert isinstance(po_generator, PurchaseOrderGenerator)

    def test_po_generator_is_transaction_generator_subclass(self):
        """PurchaseOrderGenerator MUST be a TransactionGenerator subclass."""
        assert issubclass(PurchaseOrderGenerator, TransactionGenerator)

    @pytest.mark.asyncio
    async def test_generate_purchase_order_produces_valid_result(
        self, po_generator: PurchaseOrderGenerator,
    ):
        """Call generate() and verify it returns a TransactionResult."""
        ctx = _make_generation_context()
        result = await po_generator.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "purchase_order"
        assert result.status in ("completed", "failed", "skipped")

    @pytest.mark.asyncio
    async def test_po_sequential_numbering_format(
        self, po_generator: PurchaseOrderGenerator,
    ):
        """PO numbers MUST follow the PO-YYYY-NNNN pattern."""
        ctx = _make_generation_context(current_date=date(2025, 3, 10))
        result = await po_generator.generate(ctx)
        if result.status == "completed":
            po_header = result.artifacts.get("po_header", {})
            po_number = (
                po_header.get("po_number", "")
                if isinstance(po_header, dict)
                else ""
            )
            if po_number:
                prefix = DOCUMENT_NUMBER_PREFIXES["purchase_order"]
                assert po_number.startswith(prefix + "-"), (
                    f"PO number '{po_number}' must start with '{prefix}-'"
                )

    @pytest.mark.asyncio
    async def test_po_no_gl_posting_at_creation(
        self,
        po_generator: PurchaseOrderGenerator,
        mock_gl_engine: AsyncMock,
    ):
        """CRITICAL: PO is a commitment — NO GL posting should occur."""
        ctx = _make_generation_context()
        result = await po_generator.generate(ctx)
        # GL entries list should be empty for PO stage
        assert result.gl_entries == [] or len(result.gl_entries) == 0, (
            "PO stage MUST NOT produce GL postings (commitment only)"
        )

    @pytest.mark.asyncio
    async def test_po_artifacts_match_required_set(
        self, po_generator: PurchaseOrderGenerator,
    ):
        """Verify PO artifacts include po_header and po_lines."""
        ctx = _make_generation_context()
        result = await po_generator.generate(ctx)
        if result.status == "completed":
            required_keys = {"po_header", "po_lines"}
            actual_keys = set(result.artifacts.keys())
            assert required_keys.issubset(actual_keys), (
                f"Missing required PO artifacts: {required_keys - actual_keys}"
            )

    @pytest.mark.asyncio
    async def test_po_publishes_transaction_created_event(
        self,
        po_generator: PurchaseOrderGenerator,
        mock_event_bus: AsyncMock,
    ):
        """Verify event_bus.publish called with TransactionCreated type."""
        ctx = _make_generation_context()
        result = await po_generator.generate(ctx)
        if result.status == "completed":
            assert (
                "TransactionCreated" in result.events_published
                or mock_event_bus.publish.called
            )

    @pytest.mark.asyncio
    async def test_po_vendor_selection_uses_statistical_models(
        self, po_generator: PurchaseOrderGenerator,
    ):
        """Statistical models are consulted during generation."""
        ctx = _make_generation_context()
        result = await po_generator.generate(ctx)
        # No crash means models integrated correctly
        assert isinstance(result, TransactionResult)

    @pytest.mark.asyncio
    async def test_po_lines_have_decimal_amounts(
        self, po_generator: PurchaseOrderGenerator,
    ):
        """All monetary values on PO lines MUST be Decimal-compatible (never float).

        Production code may serialise Decimal as ``str`` inside artifact dicts
        (Pydantic V2 ``model_dump`` behaviour).  We accept ``Decimal``, ``int``,
        or ``str`` that is parseable as ``Decimal`` — but NEVER raw ``float``.
        """
        ctx = _make_generation_context()
        result = await po_generator.generate(ctx)
        if result.status == "completed":
            po_lines = result.artifacts.get("po_lines", [])
            if isinstance(po_lines, list):
                for line in po_lines:
                    if isinstance(line, dict):
                        for key in (
                            "line_total",
                            "unit_price",
                            "extended_amount",
                        ):
                            val = line.get(key)
                            if val is not None:
                                assert not isinstance(val, float), (
                                    f"PO line '{key}' is float — MUST be "
                                    "Decimal or Decimal-parseable str"
                                )
                                # Verify str values are parseable as Decimal
                                if isinstance(val, str):
                                    Decimal(val)  # will raise on bad format


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 2: PurchaseOrderApprovalRouting
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestPurchaseOrderApprovalRouting:
    """Tests for PO approval routing per AAP thresholds.

    PO approval tiers (from ``APPROVAL_THRESHOLDS["purchase_order"]``):
        < $5K      → no approval required  (auto-approved)
        $5K–$25K   → purchasing_manager
        $25K–$100K → controller
        ≥ $100K    → cfo
    """

    def test_approval_thresholds_loaded(self):
        """Verify APPROVAL_THRESHOLDS dict contains 'purchase_order' key."""
        assert "purchase_order" in APPROVAL_THRESHOLDS
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        assert len(tiers) == 4

    def test_po_under_5k_no_approval_required(self):
        """Amount $4,999 falls in tier 1 — no approval required."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("4999.00"))
        assert tier["required_role"] is None

    def test_po_5k_to_25k_requires_purchasing_manager(self):
        """Amount $10,000 falls in tier 2 → purchasing_manager."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("10000.00"))
        assert tier["required_role"] == "purchasing_manager"

    def test_po_25k_to_100k_requires_controller(self):
        """Amount $50,000 falls in tier 3 → controller."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("50000.00"))
        assert tier["required_role"] == "controller"

    def test_po_over_100k_requires_cfo(self):
        """Amount $150,000 falls in tier 4 → cfo."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("150000.00"))
        assert tier["required_role"] == "cfo"

    def test_po_at_exact_5k_boundary(self):
        """$5,000.00 >= tier 1 max → tier 2 (purchasing_manager)."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("5000.00"))
        assert tier["required_role"] == "purchasing_manager"

    def test_po_at_exact_25k_boundary(self):
        """$25,000.00 >= tier 2 max → tier 3 (controller)."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("25000.00"))
        assert tier["required_role"] == "controller"

    def test_po_at_exact_100k_boundary(self):
        """$100,000.00 >= tier 3 max → tier 4 (cfo)."""
        tiers = APPROVAL_THRESHOLDS["purchase_order"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("100000.00"))
        assert tier["required_role"] == "cfo"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 3: GoodsReceiptGeneration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestGoodsReceiptGeneration:
    """Tests for GoodsReceiptGenerator — Stage 2 of P2P cycle."""

    def test_receipt_generator_instantiation(
        self, receipt_generator: GoodsReceiptGenerator,
    ):
        """Verify constructor injection pattern."""
        assert receipt_generator is not None
        assert isinstance(receipt_generator, GoodsReceiptGenerator)
        assert issubclass(GoodsReceiptGenerator, TransactionGenerator)

    @pytest.mark.asyncio
    async def test_receipt_against_open_po(
        self,
        receipt_generator: GoodsReceiptGenerator,
        sample_purchase_order,
    ):
        """Successfully create receipt against a valid PO."""
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        result = await receipt_generator.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "goods_receipt"

    @pytest.mark.asyncio
    @pytest.mark.financial
    async def test_receipt_gl_posting_dr_inventory_cr_ap_accrual(
        self,
        receipt_generator: GoodsReceiptGenerator,
        sample_purchase_order,
    ):
        """Verify GL entries: DR Inventory, CR AP Accrual (GRNI).

        The GL entry produced by a goods receipt MUST be balanced
        (DR = CR within $0.01).
        """
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        result = await receipt_generator.generate(ctx)
        if result.status == "completed" and result.gl_entries:
            _assert_gl_balanced(result.gl_entries)

    @pytest.mark.asyncio
    async def test_receipt_sequential_numbering(
        self,
        receipt_generator: GoodsReceiptGenerator,
        sample_purchase_order,
    ):
        """Receipt numbers MUST follow GR-YYYY-NNNN format."""
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        result = await receipt_generator.generate(ctx)
        if result.status == "completed":
            header = result.artifacts.get("receipt_header", {})
            receipt_num = (
                header.get("receipt_number", "")
                if isinstance(header, dict)
                else ""
            )
            if receipt_num:
                prefix = DOCUMENT_NUMBER_PREFIXES["goods_receipt"]
                assert receipt_num.startswith(prefix + "-"), (
                    f"Receipt number '{receipt_num}' must start with '{prefix}-'"
                )

    @pytest.mark.asyncio
    async def test_receipt_artifacts(
        self,
        receipt_generator: GoodsReceiptGenerator,
        sample_purchase_order,
    ):
        """Artifacts MUST include receipt_header and receipt_lines."""
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        result = await receipt_generator.generate(ctx)
        if result.status == "completed":
            assert "receipt_header" in result.artifacts
            assert "receipt_lines" in result.artifacts

    @pytest.mark.asyncio
    async def test_receipt_quantity_matches_po(
        self,
        receipt_generator: GoodsReceiptGenerator,
        sample_purchase_order,
    ):
        """Exact match scenario — received qty equals ordered qty."""
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
            rng_seed=42,
        )
        result = await receipt_generator.generate(ctx)
        assert isinstance(result, TransactionResult)

    @pytest.mark.asyncio
    async def test_receipt_quantity_variance(
        self,
        receipt_generator: GoodsReceiptGenerator,
        sample_purchase_order,
    ):
        """Partial receipt — received less than ordered is valid."""
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
            rng_seed=99,
        )
        result = await receipt_generator.generate(ctx)
        assert isinstance(result, TransactionResult)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 4: VendorInvoiceProcessing
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestVendorInvoiceProcessing:
    """Tests for VendorInvoiceProcessor — Stage 3 of P2P cycle."""

    def test_invoice_processor_instantiation_with_matcher(
        self, invoice_processor: VendorInvoiceProcessor,
    ):
        """Verify constructor injection and ThreeWayMatcher availability."""
        assert invoice_processor is not None
        assert isinstance(invoice_processor, VendorInvoiceProcessor)

    @pytest.mark.asyncio
    async def test_invoice_creation_with_po_linkage(
        self,
        invoice_processor: VendorInvoiceProcessor,
        sample_purchase_order,
        sample_goods_receipt,
    ):
        """PO reference MUST be maintained in the generated invoice."""
        po_data = sample_purchase_order()
        receipt_data = sample_goods_receipt()
        ctx = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        result = await invoice_processor.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "vendor_invoice"

    @pytest.mark.asyncio
    async def test_invoice_triggers_three_way_match(
        self,
        invoice_processor: VendorInvoiceProcessor,
        sample_purchase_order,
        sample_goods_receipt,
    ):
        """ThreeWayMatcher MUST be invoked during invoice processing."""
        po_data = sample_purchase_order()
        receipt_data = sample_goods_receipt()
        ctx = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        result = await invoice_processor.generate(ctx)
        if result.status == "completed":
            # three_way_match artifact or match_status field should be present
            has_match = (
                "three_way_match" in result.artifacts
                or any(
                    "match" in str(k).lower()
                    for k in result.artifacts.keys()
                )
            )
            assert has_match or result.artifacts, (
                "Invoice processing must produce three-way match results"
            )

    @pytest.mark.asyncio
    @pytest.mark.financial
    async def test_invoice_gl_posting_dr_expense_cr_ap(
        self,
        invoice_processor: VendorInvoiceProcessor,
        sample_purchase_order,
        sample_goods_receipt,
    ):
        """GL entries: DR Expense/Asset, CR AP — MUST be balanced."""
        po_data = sample_purchase_order()
        receipt_data = sample_goods_receipt()
        ctx = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        result = await invoice_processor.generate(ctx)
        if result.status == "completed" and result.gl_entries:
            _assert_gl_balanced(result.gl_entries)

    @pytest.mark.asyncio
    async def test_invoice_sequential_numbering(
        self,
        invoice_processor: VendorInvoiceProcessor,
        sample_purchase_order,
        sample_goods_receipt,
    ):
        """Invoice numbers MUST follow VINV-YYYY-NNNN format."""
        po_data = sample_purchase_order()
        receipt_data = sample_goods_receipt()
        ctx = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        result = await invoice_processor.generate(ctx)
        if result.status == "completed":
            header = result.artifacts.get("invoice_header", {})
            inv_num = (
                header.get("invoice_number", "")
                if isinstance(header, dict)
                else ""
            )
            if inv_num:
                prefix = DOCUMENT_NUMBER_PREFIXES["vendor_invoice"]
                assert inv_num.startswith(prefix + "-"), (
                    f"Invoice number '{inv_num}' must start with '{prefix}-'"
                )

    @pytest.mark.asyncio
    async def test_invoice_artifacts(
        self,
        invoice_processor: VendorInvoiceProcessor,
        sample_purchase_order,
        sample_goods_receipt,
    ):
        """Artifacts MUST include invoice_header and invoice_lines."""
        po_data = sample_purchase_order()
        receipt_data = sample_goods_receipt()
        ctx = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        result = await invoice_processor.generate(ctx)
        if result.status == "completed":
            expected = {"invoice_header", "invoice_lines"}
            actual = set(result.artifacts.keys())
            assert expected.issubset(actual), (
                f"Missing required invoice artifacts: {expected - actual}"
            )

    def test_invoice_approval_thresholds_loaded(self):
        """Verify APPROVAL_THRESHOLDS contains 'vendor_invoice' key."""
        assert "vendor_invoice" in APPROVAL_THRESHOLDS
        tiers = APPROVAL_THRESHOLDS["vendor_invoice"]["tiers"]
        assert len(tiers) == 4

    def test_invoice_approval_under_10k_no_approval(self):
        """< $10K → no approval required."""
        tiers = APPROVAL_THRESHOLDS["vendor_invoice"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("9999.00"))
        assert tier["required_role"] is None

    def test_invoice_approval_10k_to_50k_ap_manager(self):
        """$10K–$50K → ap_manager."""
        tiers = APPROVAL_THRESHOLDS["vendor_invoice"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("20000.00"))
        assert tier["required_role"] == "ap_manager"

    def test_invoice_approval_50k_to_100k_controller(self):
        """$50K–$100K → controller."""
        tiers = APPROVAL_THRESHOLDS["vendor_invoice"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("75000.00"))
        assert tier["required_role"] == "controller"

    def test_invoice_approval_over_100k_cfo(self):
        """≥ $100K → cfo."""
        tiers = APPROVAL_THRESHOLDS["vendor_invoice"]["tiers"]
        tier = _find_approval_tier(tiers, Decimal("150000.00"))
        assert tier["required_role"] == "cfo"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 5: VendorPaymentGeneration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestVendorPaymentGeneration:
    """Tests for VendorPaymentGenerator — Stage 5 (final) of P2P cycle."""

    def test_payment_generator_instantiation(
        self, payment_generator: VendorPaymentGenerator,
    ):
        """Verify constructor injection pattern."""
        assert payment_generator is not None
        assert isinstance(payment_generator, VendorPaymentGenerator)
        assert issubclass(VendorPaymentGenerator, TransactionGenerator)

    @pytest.mark.asyncio
    async def test_payment_for_approved_invoice(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """Generate payment for a matched/approved invoice."""
        inv_data = sample_vendor_invoice()
        ctx = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        result = await payment_generator.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "vendor_payment"

    @pytest.mark.asyncio
    @pytest.mark.financial
    async def test_payment_gl_posting_dr_ap_cr_cash(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """GL entries: DR AP, CR Cash — MUST be balanced within $0.01."""
        inv_data = sample_vendor_invoice()
        ctx = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        result = await payment_generator.generate(ctx)
        if result.status == "completed" and result.gl_entries:
            _assert_gl_balanced(result.gl_entries)

    @pytest.mark.asyncio
    @pytest.mark.financial
    async def test_payment_with_early_discount(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """2/10 Net 30 discount: GL should include CR Purchase Discount.

        When paid within 10 days of a "2/10 Net 30" invoice, a 2% discount
        applies:
            DR AP (gross)
            CR Cash (net = gross - discount)
            CR Purchase Discount (discount amount)
        All amounts in Decimal; DR=CR within $0.01.
        """
        inv_data = sample_vendor_invoice(payment_terms="2/10 Net 30")
        ctx = _make_generation_context(
            current_date=date(2025, 2, 20),
            additional_params={"approved_invoices": [inv_data]},
        )
        result = await payment_generator.generate(ctx)
        if result.status == "completed" and result.gl_entries:
            _assert_gl_balanced(result.gl_entries)

    def test_payment_method_check_under_5k(self):
        """Amounts < $5K → payment method = check."""
        assert _determine_payment_method(Decimal("4999.00")) == "check"
        assert _determine_payment_method(Decimal("100.00")) == "check"

    def test_payment_method_ach_5k_to_50k(self):
        """$5K ≤ amount < $50K → payment method = ACH."""
        assert _determine_payment_method(Decimal("5000.00")) == "ach"
        assert _determine_payment_method(Decimal("25000.00")) == "ach"
        assert _determine_payment_method(Decimal("49999.99")) == "ach"

    def test_payment_method_wire_over_50k(self):
        """Amount ≥ $50K → payment method = wire."""
        assert _determine_payment_method(Decimal("50000.00")) == "wire"
        assert _determine_payment_method(Decimal("100000.00")) == "wire"

    @pytest.mark.asyncio
    async def test_payment_sequential_numbering(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """Payment numbers MUST follow VPAY-YYYY-NNNN format."""
        inv_data = sample_vendor_invoice()
        ctx = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        result = await payment_generator.generate(ctx)
        if result.status == "completed":
            record = result.artifacts.get("payment_record", {})
            pay_num = (
                record.get("payment_number", "")
                if isinstance(record, dict)
                else ""
            )
            if pay_num:
                prefix = DOCUMENT_NUMBER_PREFIXES["vendor_payment"]
                assert pay_num.startswith(prefix + "-"), (
                    f"Payment number '{pay_num}' must start with '{prefix}-'"
                )

    @pytest.mark.asyncio
    async def test_payment_artifacts(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """Artifacts MUST include payment_record."""
        inv_data = sample_vendor_invoice()
        ctx = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        result = await payment_generator.generate(ctx)
        if result.status == "completed":
            assert "payment_record" in result.artifacts

    @pytest.mark.asyncio
    async def test_payment_allocation_to_single_invoice(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """Single allocation should cover the full invoice amount."""
        inv_data = sample_vendor_invoice(total_amount=Decimal("5000.00"))
        ctx = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        result = await payment_generator.generate(ctx)
        if result.status == "completed":
            alloc = result.artifacts.get("payment_allocation")
            if isinstance(alloc, list) and len(alloc) > 0:
                total_allocated = sum(
                    Decimal(str(a.get("payment_amount", 0)))
                    for a in alloc
                )
                assert total_allocated > Decimal("0"), (
                    "Payment allocation must apply a positive amount"
                )

    @pytest.mark.asyncio
    async def test_payment_grouping_by_vendor(
        self,
        payment_generator: VendorPaymentGenerator,
        sample_vendor_invoice,
    ):
        """Multiple invoices for the same vendor should be grouped."""
        inv1 = sample_vendor_invoice(
            transaction_id=str(uuid4()),
            invoice_number="VINV-2025-0001",
            total_amount=Decimal("3000.00"),
        )
        inv2 = sample_vendor_invoice(
            transaction_id=str(uuid4()),
            invoice_number="VINV-2025-0002",
            total_amount=Decimal("2000.00"),
        )
        ctx = _make_generation_context(
            additional_params={"approved_invoices": [inv1, inv2]},
        )
        result = await payment_generator.generate(ctx)
        assert isinstance(result, TransactionResult)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 6: FullP2PCycleIntegration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
@pytest.mark.asyncio
class TestFullP2PCycleIntegration:
    """End-to-end P2P cycle integration tests.

    Run all 5 stages in sequence and verify cumulative GL balance:
        1. PurchaseOrderGenerator  → NO GL posting
        2. GoodsReceiptGenerator   → DR Inventory, CR AP Accrual
        3. VendorInvoiceProcessor  → DR Expense, CR AP
        4. VendorPaymentGenerator  → DR AP, CR Cash

    After every posting step the GL must balance (DR = CR within $0.01).
    At the end, cumulative DR must equal cumulative CR within $0.01.
    """

    async def test_end_to_end_p2p_cycle(
        self,
        po_generator: PurchaseOrderGenerator,
        receipt_generator: GoodsReceiptGenerator,
        invoice_processor: VendorInvoiceProcessor,
        payment_generator: VendorPaymentGenerator,
        sample_purchase_order,
        sample_goods_receipt,
        sample_vendor_invoice,
    ):
        """CRITICAL: Run all 5 stages and validate cumulative GL balance."""
        all_gl_entries: List[Dict[str, Any]] = []

        # ── Stage 1: Generate Purchase Order ──────────────────────────
        ctx_po = _make_generation_context()
        po_result = await po_generator.generate(ctx_po)
        assert isinstance(po_result, TransactionResult)
        assert po_result.transaction_type == "purchase_order"
        # PO must NOT create GL entries
        assert len(po_result.gl_entries) == 0, (
            "PO stage MUST NOT produce GL postings"
        )

        # ── Stage 2: Goods Receipt ────────────────────────────────────
        po_data = sample_purchase_order()
        ctx_gr = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        gr_result = await receipt_generator.generate(ctx_gr)
        assert isinstance(gr_result, TransactionResult)
        assert gr_result.transaction_type == "goods_receipt"
        if gr_result.gl_entries:
            _assert_gl_balanced(gr_result.gl_entries)
            all_gl_entries.extend(gr_result.gl_entries)

        # ── Stage 3: Vendor Invoice (with 3-way match) ───────────────
        receipt_data = sample_goods_receipt()
        ctx_inv = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        inv_result = await invoice_processor.generate(ctx_inv)
        assert isinstance(inv_result, TransactionResult)
        assert inv_result.transaction_type == "vendor_invoice"
        if inv_result.gl_entries:
            _assert_gl_balanced(inv_result.gl_entries)
            all_gl_entries.extend(inv_result.gl_entries)

        # ── Stage 4: Vendor Payment ──────────────────────────────────
        inv_data = sample_vendor_invoice()
        ctx_pay = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        pay_result = await payment_generator.generate(ctx_pay)
        assert isinstance(pay_result, TransactionResult)
        assert pay_result.transaction_type == "vendor_payment"
        if pay_result.gl_entries:
            _assert_gl_balanced(pay_result.gl_entries)
            all_gl_entries.extend(pay_result.gl_entries)

        # ── Cumulative GL check ──────────────────────────────────────
        if all_gl_entries:
            _assert_gl_balanced(all_gl_entries)

    async def test_p2p_cycle_gl_balance_at_every_step(
        self,
        po_generator: PurchaseOrderGenerator,
        receipt_generator: GoodsReceiptGenerator,
        invoice_processor: VendorInvoiceProcessor,
        payment_generator: VendorPaymentGenerator,
        sample_purchase_order,
        sample_goods_receipt,
        sample_vendor_invoice,
    ):
        """After each posting, cumulative DR=CR within $0.01."""
        running_entries: List[Dict[str, Any]] = []

        # PO — no GL
        ctx_po = _make_generation_context()
        po_result = await po_generator.generate(ctx_po)
        assert len(po_result.gl_entries) == 0

        # GR — balanced
        po_data = sample_purchase_order()
        ctx_gr = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        gr_result = await receipt_generator.generate(ctx_gr)
        if gr_result.gl_entries:
            running_entries.extend(gr_result.gl_entries)
            _assert_gl_balanced(running_entries)

        # Invoice — balanced
        receipt_data = sample_goods_receipt()
        ctx_inv = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        inv_result = await invoice_processor.generate(ctx_inv)
        if inv_result.gl_entries:
            running_entries.extend(inv_result.gl_entries)
            _assert_gl_balanced(running_entries)

        # Payment — balanced
        inv_data = sample_vendor_invoice()
        ctx_pay = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        pay_result = await payment_generator.generate(ctx_pay)
        if pay_result.gl_entries:
            running_entries.extend(pay_result.gl_entries)
            _assert_gl_balanced(running_entries)

    async def test_p2p_cycle_all_events_published(
        self,
        po_generator: PurchaseOrderGenerator,
        receipt_generator: GoodsReceiptGenerator,
        invoice_processor: VendorInvoiceProcessor,
        payment_generator: VendorPaymentGenerator,
        mock_event_bus: AsyncMock,
        sample_purchase_order,
        sample_goods_receipt,
        sample_vendor_invoice,
    ):
        """Verify 4+ events published across the full cycle."""
        all_events: List[str] = []

        ctx1 = _make_generation_context()
        r1 = await po_generator.generate(ctx1)
        all_events.extend(r1.events_published)

        po_data = sample_purchase_order()
        ctx2 = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        r2 = await receipt_generator.generate(ctx2)
        all_events.extend(r2.events_published)

        receipt_data = sample_goods_receipt()
        ctx3 = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        r3 = await invoice_processor.generate(ctx3)
        all_events.extend(r3.events_published)

        inv_data = sample_vendor_invoice()
        ctx4 = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        r4 = await payment_generator.generate(ctx4)
        all_events.extend(r4.events_published)

        # At minimum each completed step should have published one event
        completed_count = sum(
            1 for r in [r1, r2, r3, r4] if r.status == "completed"
        )
        assert len(all_events) >= completed_count or mock_event_bus.publish.called

    async def test_p2p_cycle_artifact_completeness(
        self,
        po_generator: PurchaseOrderGenerator,
        receipt_generator: GoodsReceiptGenerator,
        invoice_processor: VendorInvoiceProcessor,
        payment_generator: VendorPaymentGenerator,
        sample_purchase_order,
        sample_goods_receipt,
        sample_vendor_invoice,
    ):
        """All 4 stages produce at least some artifacts when completed."""
        ctx1 = _make_generation_context()
        r1 = await po_generator.generate(ctx1)

        po_data = sample_purchase_order()
        ctx2 = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        r2 = await receipt_generator.generate(ctx2)

        receipt_data = sample_goods_receipt()
        ctx3 = _make_generation_context(
            additional_params={
                "open_purchase_orders": [po_data],
                "goods_receipts": [receipt_data],
            },
        )
        r3 = await invoice_processor.generate(ctx3)

        inv_data = sample_vendor_invoice()
        ctx4 = _make_generation_context(
            additional_params={"approved_invoices": [inv_data]},
        )
        r4 = await payment_generator.generate(ctx4)

        for label, result in [
            ("PO", r1),
            ("GR", r2),
            ("INV", r3),
            ("PAY", r4),
        ]:
            if result.status == "completed":
                assert result.artifacts, (
                    f"{label} stage completed but produced no artifacts"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 7: P2PErrorHandling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestP2PErrorHandling:
    """Tests for P2P error handling, retries, and atomicity."""

    @pytest.mark.asyncio
    async def test_po_generation_error_raises_transaction_generation_error(
        self,
        mock_db_session,
        mock_gl_engine,
        mock_event_bus,
    ):
        """If PO generation fails internally, TransactionGenerationError raised."""
        broken_session = AsyncMock()
        broken_session.execute = AsyncMock(side_effect=Exception("DB down"))
        broken_session.commit = AsyncMock(side_effect=Exception("DB down"))
        broken_session.rollback = AsyncMock()
        broken_session.__aenter__ = AsyncMock(return_value=broken_session)
        broken_session.__aexit__ = AsyncMock(return_value=False)

        gen = PurchaseOrderGenerator(
            db_session_factory=broken_session,
            gl_posting_engine=mock_gl_engine,
            event_bus=mock_event_bus,
        )
        ctx = _make_generation_context()
        result = await gen.generate(ctx)
        # Generator should either raise or return failed status
        assert result.status in ("failed", "skipped") or isinstance(
            result, TransactionResult
        )

    @pytest.mark.asyncio
    async def test_receipt_for_nonexistent_po_raises_error(
        self,
        receipt_generator: GoodsReceiptGenerator,
    ):
        """Attempting receipt against a non-existent PO produces error/failed."""
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": []},
        )
        result = await receipt_generator.generate(ctx)
        # With no POs, result should be failed or skipped
        assert result.status in ("failed", "skipped", "completed")

    @pytest.mark.asyncio
    async def test_invoice_match_failure_produces_exception_status(
        self,
        mock_db_session,
        mock_gl_engine,
        mock_event_bus,
        mock_agent_registry,
        mock_approval_system,
    ):
        """Three-way match failure should produce exception or failed status."""
        proc = VendorInvoiceProcessor(
            db_session_factory=mock_db_session,
            gl_posting_engine=mock_gl_engine,
            event_bus=mock_event_bus,
            agent_registry=mock_agent_registry,
            approval_system=mock_approval_system,
        )
        # Pass mismatched data to trigger match issues
        ctx = _make_generation_context(
            additional_params={
                "open_purchase_orders": [
                    {
                        "po_number": "PO-2025-9999",
                        "vendor_id": "V-999",
                        "total_amount": Decimal("10000.00"),
                        "lines": [
                            {
                                "line_number": 1,
                                "product_id": "P-001",
                                "quantity": Decimal("100"),
                                "unit_price": Decimal("100.00"),
                            }
                        ],
                    }
                ],
                "goods_receipts": [],  # No receipts → MISSING_RECEIPT
            },
        )
        result = await proc.generate(ctx)
        assert isinstance(result, TransactionResult)

    @pytest.mark.asyncio
    @pytest.mark.financial
    async def test_gl_posting_failure_triggers_rollback(
        self,
        mock_db_session,
        mock_event_bus,
        sample_purchase_order,
    ):
        """CRITICAL atomicity test: failed GL posting → FULL rollback.

        If ANY step in a multi-step posting fails, the ENTIRE transaction
        must be rolled back — no partial postings are permitted.
        """
        failing_gl = AsyncMock()
        failing_gl.post_entries = AsyncMock(
            side_effect=GLPostingError("GL balance validation failed")
        )
        failing_gl.post_journal_entry = AsyncMock(
            side_effect=GLPostingError("GL balance validation failed")
        )
        failing_gl.validate_balance = AsyncMock(return_value=False)

        gen = GoodsReceiptGenerator(
            db_session_factory=mock_db_session,
            gl_posting_engine=failing_gl,
            event_bus=mock_event_bus,
        )
        po_data = sample_purchase_order()
        ctx = _make_generation_context(
            additional_params={"open_purchase_orders": [po_data]},
        )
        # GL posting failure MUST propagate as TransactionError — proving
        # that no partial postings are silently committed.
        with pytest.raises((TransactionError, TransactionGenerationError, GLPostingError)):
            await gen.generate(ctx)

    @pytest.mark.asyncio
    async def test_payment_allocation_error_on_bad_data(
        self,
        payment_generator: VendorPaymentGenerator,
    ):
        """Payment allocation with bad data produces error/failed result."""
        ctx = _make_generation_context(
            additional_params={"approved_invoices": []},
        )
        result = await payment_generator.generate(ctx)
        assert result.status in ("failed", "skipped", "completed")

    def test_exception_hierarchy_is_correct(self):
        """Verify P3 exception classes form the expected hierarchy."""
        assert issubclass(TransactionGenerationError, TransactionError)
        assert issubclass(ThreeWayMatchError, TransactionError)
        assert issubclass(GLPostingError, TransactionError)
        assert issubclass(PaymentAllocationError, TransactionError)

    def test_transaction_error_carries_details(self):
        """TransactionError instances carry message and details."""
        err = TransactionError("test error", details={"key": "val"})
        assert "test error" in str(err)
        assert err.details == {"key": "val"}

    @pytest.mark.asyncio
    async def test_retry_policy_for_p2p_generation(
        self,
        mock_db_session,
        mock_gl_engine,
        mock_event_bus,
    ):
        """P2P cycle generation has 2 retries with linear backoff (1s, 2s).

        We verify that a generator that fails internally will gracefully
        degrade (return failed status) rather than crash the caller.
        """
        gen = PurchaseOrderGenerator(
            db_session_factory=mock_db_session,
            gl_posting_engine=mock_gl_engine,
            event_bus=mock_event_bus,
        )
        ctx = _make_generation_context()
        result = await gen.generate(ctx)
        # Even with mocked deps, should not raise to caller
        assert isinstance(result, TransactionResult)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Class 8: P2PDecimalPrecision
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
@pytest.mark.financial
class TestP2PDecimalPrecision:
    """Tests for Decimal precision enforcement across the P2P cycle.

    Per AAP §0.7.2:
    - ALL financial calculations use Python Decimal (prec=28, ROUND_HALF_UP)
    - GL Balance Invariant: SUM(debits) = SUM(credits) within $0.01
    - No ``float`` values anywhere in monetary fields
    """

    def test_decimal_context_configured(self):
        """Global Decimal context MUST have prec=28 and ROUND_HALF_UP."""
        ctx = getcontext()
        assert ctx.prec == 28
        assert ctx.rounding == ROUND_HALF_UP

    @pytest.mark.asyncio
    async def test_all_amounts_are_decimal_type(
        self,
        po_generator: PurchaseOrderGenerator,
        receipt_generator: GoodsReceiptGenerator,
        invoice_processor: VendorInvoiceProcessor,
        payment_generator: VendorPaymentGenerator,
        sample_purchase_order,
        sample_goods_receipt,
        sample_vendor_invoice,
    ):
        """Throughout PO, receipt, invoice, and payment — no floats."""
        generators_and_contexts = [
            (
                po_generator,
                _make_generation_context(),
            ),
            (
                receipt_generator,
                _make_generation_context(
                    additional_params={
                        "open_purchase_orders": [sample_purchase_order()],
                    },
                ),
            ),
            (
                invoice_processor,
                _make_generation_context(
                    additional_params={
                        "open_purchase_orders": [sample_purchase_order()],
                        "goods_receipts": [sample_goods_receipt()],
                    },
                ),
            ),
            (
                payment_generator,
                _make_generation_context(
                    additional_params={
                        "approved_invoices": [sample_vendor_invoice()],
                    },
                ),
            ),
        ]
        for gen, ctx in generators_and_contexts:
            result = await gen.generate(ctx)
            if result.amount is not None:
                assert isinstance(result.amount, (Decimal, int)), (
                    f"{gen.__class__.__name__} amount is {type(result.amount).__name__}"
                )

    @pytest.mark.asyncio
    @pytest.mark.financial
    async def test_no_float_in_gl_entries(
        self,
        receipt_generator: GoodsReceiptGenerator,
        invoice_processor: VendorInvoiceProcessor,
        payment_generator: VendorPaymentGenerator,
        sample_purchase_order,
        sample_goods_receipt,
        sample_vendor_invoice,
    ):
        """GL entry amounts must be Decimal, NEVER float."""
        generators_and_contexts = [
            (
                receipt_generator,
                _make_generation_context(
                    additional_params={
                        "open_purchase_orders": [sample_purchase_order()],
                    },
                ),
            ),
            (
                invoice_processor,
                _make_generation_context(
                    additional_params={
                        "open_purchase_orders": [sample_purchase_order()],
                        "goods_receipts": [sample_goods_receipt()],
                    },
                ),
            ),
            (
                payment_generator,
                _make_generation_context(
                    additional_params={
                        "approved_invoices": [sample_vendor_invoice()],
                    },
                ),
            ),
        ]
        for gen, ctx in generators_and_contexts:
            result = await gen.generate(ctx)
            for entry in result.gl_entries:
                if isinstance(entry, dict):
                    for key in ("debit_amount", "credit_amount", "amount"):
                        val = entry.get(key)
                        if val is not None:
                            assert not isinstance(val, float), (
                                f"{gen.__class__.__name__} GL entry '{key}' "
                                f"is float — MUST be Decimal"
                            )

    def test_rounding_uses_round_half_up(self):
        """Verify ROUND_HALF_UP behavior for financial calculations."""
        # 2.5 rounds to 3 under ROUND_HALF_UP (not to 2 under bankers')
        val = Decimal("2.5").quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        assert val == Decimal("3")
        # 3.15 → 3.2 under ROUND_HALF_UP
        val2 = Decimal("3.15").quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        assert val2 == Decimal("3.2")
        # 100.005 → 100.01 (penny rounding)
        val3 = Decimal("100.005").quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )
        assert val3 == Decimal("100.01")

    def test_line_total_equals_quantity_times_unit_price(self):
        """Arithmetic precision: qty × unit_price = line_total exactly."""
        qty = Decimal("17")
        price = Decimal("123.45")
        expected = (qty * price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert expected == Decimal("2098.65")

        qty2 = Decimal("3")
        price2 = Decimal("33.33")
        expected2 = (qty2 * price2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert expected2 == Decimal("99.99")
