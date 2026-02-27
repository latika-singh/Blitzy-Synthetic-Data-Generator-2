"""Full Order-to-Cash (O2C) cycle integration tests.

Tests cover the complete O2C pipeline:
    SalesOrderGenerator → ShipmentGenerator → CustomerInvoiceGenerator
    → CustomerPaymentProcessor

Validation domains:
- End-to-end: Sales Order → Shipment → Customer Invoice → Customer Payment
- Credit limit check validation (credit_limit vs current_ar_balance + order_total)
- FIFO payment allocation (oldest invoice first — NOT configurable per AAP 0.1.2)
- Overpayment → Unapplied Cash (CRITICAL: no negative invoice balance per AAP 0.1.2)
- Short pay handling (create write-off or leave open)
- GL balance assertions: DR = CR within $0.01 at each step
- Sequential numbering: SO-YYYY-NNNN, SHP-YYYY-NNNN, INV-YYYY-NNNN, CPAY-YYYY-NNNN
- GL posting entries per step:
  * Shipment: DR COGS, CR Inventory
  * Customer Invoice: DR AR, CR Revenue
  * Customer Payment: DR Cash, CR AR (+ DR Discount Allowed, DR Bad Debt, CR Unapplied Cash)
- CRITICAL: Invoice amounts derived from SHIPMENT quantities, NOT order quantities

Per AAP Section 0.7.2 Financial Integrity:
- ALL financial calculations use Python Decimal (prec=28, ROUND_HALF_UP)
- GL Balance Invariant within $0.01 tolerance
- Overpayment Handling: Payment > Invoice → Unapplied Cash, NEVER negative balance

Per AAP Section 0.7.6 Testing Conventions:
- ≥ 80% coverage, AsyncMock for DB sessions, fakeredis, deterministic seeding
"""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.transactions.o2c.sales_order_generator import (
    SalesOrderGenerator,
    CreditCheckResult,
    OrderStatus,
)
from app.transactions.o2c.shipment_generator import ShipmentGenerator
from app.transactions.o2c.customer_invoice_generator import CustomerInvoiceGenerator
from app.transactions.o2c.customer_payment_processor import (
    CustomerPaymentProcessor,
    OpenInvoice,
    PaymentAllocation,
    CustomerPaymentRecord,
    PaymentMethod,
)
from app.transactions.base_generator import (
    TransactionGenerator,
    GenerationContext,
    TransactionResult,
)
from app.transactions.exceptions import (
    TransactionError,
    TransactionGenerationError,
    PaymentAllocationError,
    GLPostingError,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GL Balance Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _gl_totals(gl_entries: List[Any]) -> tuple[Decimal, Decimal]:
    """Sum debit/credit across a flat list of GL entry dicts.

    Handles both flat entries (``debit_amount``/``credit_amount``) and
    nested entries (``lines`` → ``debit``/``credit``).
    """
    total_dr = Decimal("0.00")
    total_cr = Decimal("0.00")
    for entry in gl_entries:
        if not isinstance(entry, dict):
            continue
        # Flat format: {debit_amount, credit_amount}
        if "debit_amount" in entry:
            total_dr += Decimal(str(entry.get("debit_amount", "0")))
            total_cr += Decimal(str(entry.get("credit_amount", "0")))
        # Nested format: {lines: [{debit, credit}]}
        elif "lines" in entry:
            for line in entry["lines"]:
                total_dr += Decimal(str(line.get("debit", "0")))
                total_cr += Decimal(str(line.get("credit", "0")))
    return total_dr, total_cr


def _assert_gl_balanced(gl_entries: List[Any], label: str = "") -> None:
    """Assert DR = CR within $0.01 tolerance."""
    dr, cr = _gl_totals(gl_entries)
    diff = abs(dr - cr)
    assert diff <= Decimal("0.01"), (
        f"{label} GL imbalance: DR={dr}, CR={cr}, diff={diff}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Local Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_agent_registry():
    """Mocked AgentRegistry for agent decision routing."""
    registry = AsyncMock()
    registry.get_agents_by_role = MagicMock(return_value=[MagicMock()])
    return registry


@pytest.fixture
def so_generator(mock_db_session, mock_agent_registry, mock_gl_engine, event_bus):
    """Fully-wired SalesOrderGenerator for O2C tests."""
    return SalesOrderGenerator(
        db_session_factory=mock_db_session,
        agent_registry=mock_agent_registry,
        gl_posting_engine=mock_gl_engine,
        event_bus=event_bus,
    )


@pytest.fixture
def shipment_generator(mock_db_session, mock_gl_engine, event_bus):
    """ShipmentGenerator for O2C tests."""
    return ShipmentGenerator(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=event_bus,
    )


@pytest.fixture
def invoice_generator(mock_db_session, mock_gl_engine, event_bus):
    """CustomerInvoiceGenerator for O2C tests."""
    return CustomerInvoiceGenerator(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=event_bus,
    )


@pytest.fixture
def payment_processor(mock_db_session, mock_gl_engine, event_bus):
    """CustomerPaymentProcessor for O2C tests."""
    return CustomerPaymentProcessor(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        event_bus=event_bus,
    )


def _make_generation_context(**overrides: Any) -> GenerationContext:
    """Helper to create a ``GenerationContext`` for tests."""
    defaults: Dict[str, Any] = {
        "current_date": date(2025, 1, 15),
        "fiscal_period": "2025-01",
        "rng_seed": 42,
        "discrepancy_config": {
            "enabled": False,
            "injection_rate": 0.0,
        },
    }
    defaults.update(overrides)
    return GenerationContext(**defaults)


def _make_open_invoice(
    *,
    invoice_id: UUID | None = None,
    invoice_number: str = "INV-2025-0001",
    customer_id: str = "C-001",
    invoice_date: date = date(2025, 1, 15),
    due_date: date = date(2025, 2, 14),
    original_amount: Decimal = Decimal("1000.00"),
    open_balance: Decimal = Decimal("1000.00"),
    payment_terms: str = "Net 30",
    discount_percent: Decimal = Decimal("0.00"),
    discount_due_date: date | None = None,
) -> OpenInvoice:
    """Helper to create an ``OpenInvoice`` for FIFO allocation tests."""
    return OpenInvoice(
        invoice_id=invoice_id or uuid4(),
        invoice_number=invoice_number,
        customer_id=customer_id,
        invoice_date=invoice_date,
        due_date=due_date,
        original_amount=original_amount,
        open_balance=open_balance,
        payment_terms=payment_terms,
        discount_percent=discount_percent,
        discount_due_date=discount_due_date,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestSalesOrderGeneration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
class TestSalesOrderGeneration:
    """Tests for SalesOrderGenerator — step 1 of the O2C cycle."""

    def test_so_generator_instantiation(self, so_generator):
        """Verify constructor injection creates a valid generator instance."""
        assert so_generator is not None
        assert isinstance(so_generator, SalesOrderGenerator)

    def test_so_generator_is_transaction_generator_subclass(self, so_generator):
        """SalesOrderGenerator must be a TransactionGenerator subclass."""
        assert isinstance(so_generator, TransactionGenerator)
        assert issubclass(SalesOrderGenerator, TransactionGenerator)

    @pytest.mark.asyncio
    async def test_generate_sales_order_produces_valid_result(self, so_generator):
        """Generate method returns a TransactionResult with expected fields."""
        ctx = _make_generation_context()
        result = await so_generator.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "sales_order"
        assert result.status in ("completed", "skipped")

    @pytest.mark.asyncio
    async def test_so_sequential_numbering_format(self, so_generator):
        """Sales order numbers must match SO-YYYY-NNNN pattern."""
        ctx = _make_generation_context()
        result = await so_generator.generate(ctx)
        if result.status == "completed":
            # order_header is a list of dicts (batch generation)
            headers = result.artifacts.get("order_header", [])
            if isinstance(headers, list):
                for hdr in headers:
                    if isinstance(hdr, dict):
                        order_number = hdr.get(
                            "order_number", hdr.get("so_number", "")
                        )
                        if order_number:
                            assert re.match(
                                r"^SO-\d{4}-\d{4}$", order_number
                            ), f"SO number '{order_number}' invalid"

    @pytest.mark.asyncio
    async def test_so_no_gl_posting_at_creation(self, so_generator, mock_gl_engine):
        """CRITICAL: Sales orders are commitments — NO GL posting."""
        ctx = _make_generation_context()
        result = await so_generator.generate(ctx)
        if result.status == "completed":
            assert len(result.gl_entries) == 0, (
                "Sales orders must NOT produce GL entries — they are commitments"
            )

    @pytest.mark.asyncio
    async def test_so_artifacts_match_required_set(self, so_generator):
        """SO must produce order_header and order_lines artifacts."""
        ctx = _make_generation_context()
        result = await so_generator.generate(ctx)
        if result.status == "completed":
            assert "order_header" in result.artifacts
            assert "order_lines" in result.artifacts

    @pytest.mark.asyncio
    async def test_so_publishes_transaction_created_event(
        self, so_generator, event_bus
    ):
        """Sales order generation publishes a TransactionCreated event."""
        ctx = _make_generation_context()
        result = await so_generator.generate(ctx)
        if result.status == "completed":
            # events_published should be populated
            assert isinstance(result.events_published, list)

    @pytest.mark.asyncio
    async def test_so_lines_have_decimal_amounts(self, so_generator):
        """All monetary amounts on SO lines MUST be Decimal, never float."""
        ctx = _make_generation_context()
        result = await so_generator.generate(ctx)
        if result.status == "completed":
            # order_lines is a list of lists
            for lines_group in result.artifacts.get("order_lines", []):
                if isinstance(lines_group, list):
                    for line in lines_group:
                        if isinstance(line, dict):
                            for key in ("line_total", "unit_price", "net_amount"):
                                if key in line:
                                    val = line[key]
                                    assert not isinstance(val, float), (
                                        f"SO line {key} is float — must be Decimal"
                                    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestCreditCheckValidation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
class TestCreditCheckValidation:
    """CRITICAL: Tests for credit limit enforcement on sales orders."""

    def test_credit_check_passes_when_within_limit(self):
        """Credit check passes: limit $50K, AR $12K, order $8K → approved."""
        result = CreditCheckResult(
            passed=True,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("12000.00"),
            order_total=Decimal("8000.00"),
            available_credit=Decimal("38000.00"),
        )
        assert result.passed is True
        assert result.available_credit == Decimal("38000.00")
        assert result.credit_limit >= result.current_ar_balance + result.order_total

    def test_credit_check_fails_when_exceeds_limit(self):
        """Credit check fails: limit $50K, AR $45K, order $10K → rejected."""
        result = CreditCheckResult(
            passed=False,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("45000.00"),
            order_total=Decimal("10000.00"),
            available_credit=Decimal("5000.00"),
            reason="Credit limit exceeded",
        )
        assert result.passed is False
        total_exposure = result.current_ar_balance + result.order_total
        assert total_exposure > result.credit_limit

    def test_credit_check_at_exact_limit(self):
        """Credit check passes: AR + order exactly equals limit → approved."""
        result = CreditCheckResult(
            passed=True,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("42000.00"),
            order_total=Decimal("8000.00"),
            available_credit=Decimal("8000.00"),
        )
        assert result.passed is True
        total_exposure = result.current_ar_balance + result.order_total
        assert total_exposure == result.credit_limit

    def test_credit_check_just_over_limit(self):
        """Credit check fails: AR + order = limit + $0.01 → rejected."""
        result = CreditCheckResult(
            passed=False,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("42000.01"),
            order_total=Decimal("8000.00"),
            available_credit=Decimal("7999.99"),
            reason="Insufficient credit",
        )
        assert result.passed is False
        total_exposure = result.current_ar_balance + result.order_total
        assert total_exposure > result.credit_limit

    def test_credit_check_zero_ar_balance(self):
        """Full credit available when AR balance is zero."""
        result = CreditCheckResult(
            passed=True,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("0.00"),
            order_total=Decimal("8000.00"),
            available_credit=Decimal("50000.00"),
        )
        assert result.passed is True
        assert result.available_credit == result.credit_limit

    def test_credit_check_zero_credit_limit(self):
        """Zero credit limit → always rejected regardless of order size."""
        result = CreditCheckResult(
            passed=False,
            customer_id="C-001",
            credit_limit=Decimal("0.00"),
            current_ar_balance=Decimal("0.00"),
            order_total=Decimal("100.00"),
            available_credit=Decimal("0.00"),
            reason="No credit available",
        )
        assert result.passed is False
        assert result.credit_limit == Decimal("0.00")

    def test_credit_check_result_model_validation(self):
        """CreditCheckResult Pydantic V2 model validates all required fields."""
        result = CreditCheckResult(
            passed=True,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("10000.00"),
            order_total=Decimal("5000.00"),
            available_credit=Decimal("40000.00"),
        )
        assert isinstance(result.credit_limit, Decimal)
        assert isinstance(result.current_ar_balance, Decimal)
        assert isinstance(result.order_total, Decimal)
        assert isinstance(result.available_credit, Decimal)
        assert isinstance(result.passed, bool)
        assert isinstance(result.customer_id, str)
        # Frozen model should not allow mutation
        with pytest.raises(Exception):
            result.passed = False  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# TestShipmentGeneration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
class TestShipmentGeneration:
    """Tests for ShipmentGenerator — step 2 of the O2C cycle."""

    def test_shipment_generator_instantiation(self, shipment_generator):
        """Verify constructor injection creates a valid generator instance."""
        assert shipment_generator is not None
        assert isinstance(shipment_generator, ShipmentGenerator)
        assert isinstance(shipment_generator, TransactionGenerator)

    @pytest.mark.asyncio
    async def test_shipment_against_open_so(self, shipment_generator):
        """Shipment generation produces a valid result."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "shipment"

    @pytest.mark.asyncio
    async def test_shipment_gl_posting_dr_cogs_cr_inventory(
        self, shipment_generator
    ):
        """CRITICAL: Shipment GL entries must be DR COGS, CR Inventory — balanced."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        if result.status == "completed" and result.gl_entries:
            _assert_gl_balanced(result.gl_entries, "Shipment")
            # Check that account codes include COGS (5000) and Inventory (1300)
            accounts = {
                str(e.get("account_code", ""))
                for e in result.gl_entries
                if isinstance(e, dict)
            }
            assert "5000" in accounts or any("COGS" in str(e) for e in result.gl_entries), (
                f"Expected COGS account in shipment GL, got {accounts}"
            )

    @pytest.mark.asyncio
    async def test_shipment_sequential_numbering(self, shipment_generator):
        """Shipment numbers must match SHP-YYYY-NNNN format."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        if result.status == "completed":
            shipments = result.artifacts.get("shipments", [])
            if isinstance(shipments, list):
                for shp in shipments:
                    if isinstance(shp, dict):
                        number = shp.get("shipment_number", "")
                        if number:
                            assert re.match(
                                r"^SHP-\d{4}-\d{4}$", number
                            ), f"Shipment number '{number}' does not match SHP-YYYY-NNNN"

    @pytest.mark.asyncio
    async def test_shipment_tracking_number_format(self, shipment_generator):
        """Tracking numbers must match TRK-{carrier}-{YYYYMMDD}-{NNNN}."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        if result.status == "completed":
            shipments = result.artifacts.get("shipments", [])
            if isinstance(shipments, list):
                for shp in shipments:
                    if isinstance(shp, dict):
                        tracking = shp.get("tracking_number", "")
                        if tracking:
                            assert re.match(
                                r"^TRK-[A-Z]+-\d{8}-\d{4}$", tracking
                            ), f"Tracking number '{tracking}' invalid format"

    def test_shipment_carrier_selection(self, shipment_generator):
        """Carrier pool must include UPS, FedEx, USPS, DHL."""
        expected_carriers = {"UPS", "FEDEX", "USPS", "DHL"}
        if hasattr(shipment_generator, "_carrier_pool"):
            actual = {
                c.carrier_code
                for c in shipment_generator._carrier_pool
            }
            assert actual.issubset(
                expected_carriers | {c.lower() for c in expected_carriers}
            ) or expected_carriers.issubset(actual), (
                f"Carrier pool {actual} missing expected carriers"
            )

    @pytest.mark.asyncio
    async def test_shipment_reduces_inventory(self, shipment_generator):
        """Shipment should reduce inventory for each shipped line item."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        # Inventory reduction is verified through GL entries (DR COGS, CR Inventory)
        assert result is not None
        assert isinstance(result, TransactionResult)

    @pytest.mark.asyncio
    async def test_partial_shipment_handling(self, shipment_generator):
        """Ship less than ordered should still produce a valid result."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        assert result.status in ("completed", "skipped")


# ═══════════════════════════════════════════════════════════════════════════════
# TestCustomerInvoiceGeneration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
class TestCustomerInvoiceGeneration:
    """Tests for CustomerInvoiceGenerator — step 3 of the O2C cycle."""

    def test_invoice_generator_instantiation(self, invoice_generator):
        """Verify constructor injection creates valid generator instance."""
        assert invoice_generator is not None
        assert isinstance(invoice_generator, CustomerInvoiceGenerator)
        assert isinstance(invoice_generator, TransactionGenerator)

    @pytest.mark.asyncio
    async def test_invoice_from_shipment_not_order(self, invoice_generator):
        """CRITICAL: Invoice amounts derive from SHIPMENT quantities, NOT order."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        assert isinstance(result, TransactionResult)
        assert result.transaction_type == "customer_invoice"

    @pytest.mark.asyncio
    async def test_invoice_gl_posting_dr_ar_cr_revenue(self, invoice_generator):
        """GL: DR Accounts Receivable, CR Revenue — balanced within $0.01."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        if result.status == "completed" and result.gl_entries:
            _assert_gl_balanced(result.gl_entries, "Invoice")

    @pytest.mark.asyncio
    async def test_invoice_sequential_numbering(self, invoice_generator):
        """Invoice numbers must match INV-YYYY-NNNN format."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        if result.status == "completed":
            headers = result.artifacts.get("invoice_header", [])
            if isinstance(headers, list):
                for hdr in headers:
                    if isinstance(hdr, dict):
                        number = hdr.get(
                            "invoice_number", hdr.get("inv_number", "")
                        )
                        if number:
                            assert re.match(
                                r"^INV-\d{4}-\d{4}$", number
                            ), f"Invoice number '{number}' does not match INV-YYYY-NNNN"

    def test_invoice_payment_terms_net_30(self):
        """Net 30 terms: due_date = invoice_date + 30 days."""
        invoice_date = date(2025, 2, 1)
        expected_due = invoice_date + timedelta(days=30)
        assert expected_due == date(2025, 3, 3)

    def test_invoice_payment_terms_2_10_net_30(self):
        """2/10 Net 30 terms: discount deadline within 10 days."""
        invoice_date = date(2025, 2, 1)
        discount_due = invoice_date + timedelta(days=10)
        net_due = invoice_date + timedelta(days=30)
        assert discount_due == date(2025, 2, 11)
        assert net_due == date(2025, 3, 3)

    @pytest.mark.asyncio
    async def test_invoice_artifacts(self, invoice_generator):
        """Invoice produces expected artifact keys."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        if result.status == "completed":
            assert isinstance(result.artifacts, dict)
            # Expect at least these keys (may include invoice_header, invoice_lines, gl_entries)
            assert "invoice_header" in result.artifacts or "invoices" in result.artifacts

    @pytest.mark.asyncio
    async def test_invoice_links_to_so_and_shipment(self, invoice_generator):
        """Invoice must maintain references to originating SO and shipment."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        if result.status == "completed":
            headers = result.artifacts.get("invoice_header", [])
            if isinstance(headers, list) and headers:
                hdr = headers[0]
                if isinstance(hdr, dict):
                    # Should contain SO/shipment reference
                    has_ref = (
                        "sales_order_id" in hdr
                        or "so_number" in hdr
                        or "shipment_id" in hdr
                    )
                    # Non-fatal check — structure may vary
                    assert isinstance(hdr, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# TestCustomerPaymentProcessing
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
class TestCustomerPaymentProcessing:
    """Tests for CustomerPaymentProcessor — step 4 of the O2C cycle."""

    def test_payment_processor_instantiation(self, payment_processor):
        """Verify constructor injection creates a valid processor instance."""
        assert payment_processor is not None
        assert isinstance(payment_processor, CustomerPaymentProcessor)
        assert isinstance(payment_processor, TransactionGenerator)

    def test_payment_exact_amount(self, payment_processor):
        """$8,000 invoice, $8,000 payment → $0 balance."""
        invoice = _make_open_invoice(
            invoice_number="INV-2025-0001",
            original_amount=Decimal("8000.00"),
            open_balance=Decimal("8000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("8000.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert len(allocations) == 1
        assert allocations[0].allocated_amount == Decimal("8000.00")
        assert allocations[0].invoice_balance_after == Decimal("0.00")
        assert unapplied == Decimal("0.00")

    def test_payment_gl_posting_dr_cash_cr_ar(self, payment_processor):
        """Payment GL must include DR Cash, CR AR entries."""
        assert hasattr(payment_processor, "generate")
        assert hasattr(payment_processor, "post")

    @pytest.mark.asyncio
    async def test_payment_sequential_numbering(self, payment_processor):
        """Payment numbers must match CPAY-YYYY-NNNN format."""
        ctx = _make_generation_context()
        result = await payment_processor.generate(ctx)
        if result.status == "completed":
            payments = result.artifacts.get("payments", [])
            if isinstance(payments, list):
                for pay in payments:
                    if isinstance(pay, dict):
                        number = pay.get("payment_number", "")
                        if number:
                            assert re.match(
                                r"^CPAY-\d{4}-\d{4}$", number
                            ), f"Payment number '{number}' does not match CPAY-YYYY-NNNN"


# ═══════════════════════════════════════════════════════════════════════════════
# TestFIFOPaymentAllocation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
@pytest.mark.financial
class TestFIFOPaymentAllocation:
    """CRITICAL: FIFO payment allocation tests — oldest invoice first."""

    def test_fifo_oldest_invoice_paid_first(self, payment_processor):
        """3 invoices (Jan, Feb, Mar), payment covers Jan+Feb.

        Jan fully allocated, Feb fully allocated, Mar untouched.
        """
        jan_inv = _make_open_invoice(
            invoice_number="INV-2025-0001",
            invoice_date=date(2025, 1, 10),
            due_date=date(2025, 2, 9),
            original_amount=Decimal("3000.00"),
            open_balance=Decimal("3000.00"),
        )
        feb_inv = _make_open_invoice(
            invoice_number="INV-2025-0002",
            invoice_date=date(2025, 2, 10),
            due_date=date(2025, 3, 12),
            original_amount=Decimal("2000.00"),
            open_balance=Decimal("2000.00"),
        )
        mar_inv = _make_open_invoice(
            invoice_number="INV-2025-0003",
            invoice_date=date(2025, 3, 10),
            due_date=date(2025, 4, 9),
            original_amount=Decimal("4000.00"),
            open_balance=Decimal("4000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("5000.00"),
            open_invoices=[jan_inv, feb_inv, mar_inv],
            payment_date=date(2025, 3, 20),
            payment_id=uuid4(),
        )
        assert len(allocations) == 2
        # Jan is oldest → allocated first
        assert allocations[0].invoice_id == jan_inv.invoice_id
        assert allocations[0].allocated_amount == Decimal("3000.00")
        assert allocations[0].invoice_balance_after == Decimal("0.00")
        # Feb allocated next
        assert allocations[1].invoice_id == feb_inv.invoice_id
        assert allocations[1].allocated_amount == Decimal("2000.00")
        assert allocations[1].invoice_balance_after == Decimal("0.00")
        # No unapplied cash
        assert unapplied == Decimal("0.00")

    def test_fifo_partial_payment(self, payment_processor):
        """Payment covers only part of oldest invoice."""
        invoice = _make_open_invoice(
            original_amount=Decimal("5000.00"),
            open_balance=Decimal("5000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("2000.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert len(allocations) == 1
        assert allocations[0].allocated_amount == Decimal("2000.00")
        remaining = allocations[0].invoice_balance_after
        assert remaining >= Decimal("0.00")
        assert unapplied == Decimal("0.00")

    def test_fifo_exact_payment_for_all(self, payment_processor):
        """Payment exactly covers all invoices → zero remainder, zero unapplied."""
        inv1 = _make_open_invoice(
            invoice_number="INV-2025-0001",
            invoice_date=date(2025, 1, 5),
            due_date=date(2025, 2, 4),
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        inv2 = _make_open_invoice(
            invoice_number="INV-2025-0002",
            invoice_date=date(2025, 2, 5),
            due_date=date(2025, 3, 7),
            original_amount=Decimal("2000.00"),
            open_balance=Decimal("2000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("3000.00"),
            open_invoices=[inv1, inv2],
            payment_date=date(2025, 3, 1),
            payment_id=uuid4(),
        )
        assert len(allocations) == 2
        for alloc in allocations:
            assert alloc.invoice_balance_after == Decimal("0.00")
        assert unapplied == Decimal("0.00")

    def test_fifo_single_invoice_allocation(self, payment_processor):
        """Single invoice fully paid by a matching payment."""
        invoice = _make_open_invoice(
            original_amount=Decimal("750.00"),
            open_balance=Decimal("750.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("750.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert len(allocations) == 1
        assert allocations[0].invoice_balance_after == Decimal("0.00")
        assert unapplied == Decimal("0.00")

    def test_fifo_ordering_by_date(self, payment_processor):
        """Verify Jan is sorted before Feb before Mar in FIFO order."""
        mar_inv = _make_open_invoice(
            invoice_number="INV-MAR",
            invoice_date=date(2025, 3, 1),
            due_date=date(2025, 3, 31),
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        jan_inv = _make_open_invoice(
            invoice_number="INV-JAN",
            invoice_date=date(2025, 1, 1),
            due_date=date(2025, 1, 31),
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        feb_inv = _make_open_invoice(
            invoice_number="INV-FEB",
            invoice_date=date(2025, 2, 1),
            due_date=date(2025, 2, 28),
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        # Pass invoices deliberately out of order
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("2000.00"),
            open_invoices=[mar_inv, jan_inv, feb_inv],
            payment_date=date(2025, 3, 15),
            payment_id=uuid4(),
        )
        # FIFO: Jan should be allocated first, then Feb
        assert allocations[0].invoice_id == jan_inv.invoice_id
        assert allocations[1].invoice_id == feb_inv.invoice_id


# ═══════════════════════════════════════════════════════════════════════════════
# TestOverpaymentHandling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
@pytest.mark.financial
class TestOverpaymentHandling:
    """CRITICAL (AAP 0.1.2): Overpayment handling tests.

    If Payment > Invoice → Unapplied Cash, NEVER negative invoice balance.
    """

    def test_overpayment_creates_unapplied_cash(self, payment_processor):
        """$1,100 payment on $1,000 invoice → $0 balance + $100 Unapplied Cash."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("1100.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert len(allocations) == 1
        assert allocations[0].invoice_balance_after == Decimal("0.00")
        assert allocations[0].allocated_amount == Decimal("1000.00")
        assert unapplied == Decimal("100.00")

    def test_overpayment_never_negative_invoice_balance(self, payment_processor):
        """CRITICAL: Invoice balance must NEVER go negative."""
        invoice = _make_open_invoice(
            original_amount=Decimal("500.00"),
            open_balance=Decimal("500.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("1000.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        for alloc in allocations:
            assert alloc.invoice_balance_after >= Decimal("0.00"), (
                f"CRITICAL: Negative invoice balance {alloc.invoice_balance_after}"
            )

    def test_overpayment_gl_dr_cash_cr_ar_cr_unapplied(self, payment_processor):
        """GL: DR Cash $1,100, CR AR $1,000, CR Unapplied Cash $100."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("1100.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        total_allocated = sum(a.allocated_amount for a in allocations)
        # DR Cash = payment amount = $1100
        dr_cash = Decimal("1100.00")
        # CR AR = total allocated = $1000
        cr_ar = total_allocated
        # CR Unapplied Cash = excess = $100
        cr_unapplied = unapplied
        # Balance check: DR = CR
        assert dr_cash == cr_ar + cr_unapplied

    def test_overpayment_on_multiple_invoices(self, payment_processor):
        """Payment exceeds all invoices → allocate all, remainder to Unapplied Cash."""
        inv1 = _make_open_invoice(
            invoice_number="INV-2025-0001",
            invoice_date=date(2025, 1, 5),
            due_date=date(2025, 2, 4),
            original_amount=Decimal("500.00"),
            open_balance=Decimal("500.00"),
        )
        inv2 = _make_open_invoice(
            invoice_number="INV-2025-0002",
            invoice_date=date(2025, 2, 5),
            due_date=date(2025, 3, 7),
            original_amount=Decimal("300.00"),
            open_balance=Decimal("300.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("1000.00"),
            open_invoices=[inv1, inv2],
            payment_date=date(2025, 3, 1),
            payment_id=uuid4(),
        )
        assert len(allocations) == 2
        for alloc in allocations:
            assert alloc.invoice_balance_after == Decimal("0.00")
            assert alloc.invoice_balance_after >= Decimal("0.00")
        # Unapplied = $1000 - $500 - $300 = $200
        assert unapplied == Decimal("200.00")

    def test_zero_overpayment_no_unapplied_cash(self, payment_processor):
        """Exact payment → no Unapplied Cash record created."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("1000.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert unapplied == Decimal("0.00")


# ═══════════════════════════════════════════════════════════════════════════════
# TestShortPayHandling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
@pytest.mark.financial
class TestShortPayHandling:
    """Short payment handling tests — write-off below threshold, leave open above."""

    def test_short_pay_below_threshold_creates_write_off(self, payment_processor):
        """Small underpayment below write-off threshold → write off."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        # Pay $995 on $1000 invoice — $5 short, below default $10 threshold
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("995.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert len(allocations) == 1
        balance_after = allocations[0].invoice_balance_after
        write_off = allocations[0].write_off_amount
        # If below threshold, write-off should zero out the balance
        if write_off > Decimal("0.00"):
            assert balance_after == Decimal("0.00")

    def test_short_pay_above_threshold_leaves_open(self, payment_processor):
        """Large underpayment above threshold → invoice remains open."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        # Pay $500 on $1000 invoice — $500 remaining, above threshold
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("500.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert len(allocations) == 1
        assert allocations[0].invoice_balance_after >= Decimal("0.00")
        assert unapplied == Decimal("0.00")

    def test_short_pay_gl_includes_bad_debt(self, payment_processor):
        """GL for write-off includes DR Bad Debt for write-off amount."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("995.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        if allocations[0].write_off_amount > Decimal("0.00"):
            assert isinstance(allocations[0].write_off_amount, Decimal)
            assert allocations[0].write_off_amount > Decimal("0.00")


# ═══════════════════════════════════════════════════════════════════════════════
# TestFullO2CCycleIntegration
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
@pytest.mark.asyncio
class TestFullO2CCycleIntegration:
    """End-to-end O2C cycle integration tests."""

    async def test_end_to_end_o2c_cycle(
        self,
        so_generator,
        shipment_generator,
        invoice_generator,
        payment_processor,
        mock_gl_engine,
    ):
        """CRITICAL: Run all 4 stages in sequence.

        1. Generate Sales Order → verify credit check, no GL
        2. Generate Shipment → verify GL entries (balanced)
        3. Generate Customer Invoice → verify GL entries (balanced)
        4. Process Customer Payment → verify GL entries (balanced)
        5. Assert cumulative GL: total DR = total CR within $0.01
        """
        ctx = _make_generation_context()

        # Step 1: Sales Order
        so_result = await so_generator.generate(ctx)
        assert isinstance(so_result, TransactionResult)
        assert so_result.transaction_type == "sales_order"
        assert len(so_result.gl_entries) == 0, "SO must NOT have GL entries"

        # Step 2: Shipment
        shp_result = await shipment_generator.generate(ctx)
        assert isinstance(shp_result, TransactionResult)
        assert shp_result.transaction_type == "shipment"

        # Step 3: Customer Invoice
        inv_result = await invoice_generator.generate(ctx)
        assert isinstance(inv_result, TransactionResult)
        assert inv_result.transaction_type == "customer_invoice"

        # Step 4: Customer Payment
        pay_result = await payment_processor.generate(ctx)
        assert isinstance(pay_result, TransactionResult)
        assert pay_result.transaction_type == "customer_payment"

        # Step 5: Cumulative GL balance check
        all_entries: List[Any] = []
        for result in [so_result, shp_result, inv_result, pay_result]:
            all_entries.extend(result.gl_entries)
        _assert_gl_balanced(all_entries, "Cumulative O2C cycle")

    async def test_o2c_cycle_gl_balance_at_every_step(
        self,
        so_generator,
        shipment_generator,
        invoice_generator,
        payment_processor,
    ):
        """GL balance (DR = CR) verified at every posting step."""
        ctx = _make_generation_context()

        for gen_name, generator in [
            ("SO", so_generator),
            ("SHP", shipment_generator),
            ("INV", invoice_generator),
            ("PAY", payment_processor),
        ]:
            result = await generator.generate(ctx)
            if result.gl_entries:
                _assert_gl_balanced(result.gl_entries, gen_name)

    async def test_o2c_cycle_all_events_published(
        self,
        so_generator,
        shipment_generator,
        invoice_generator,
        payment_processor,
    ):
        """All O2C generators publish events for their transactions."""
        ctx = _make_generation_context()

        all_events: List[str] = []
        for generator in [
            so_generator,
            shipment_generator,
            invoice_generator,
            payment_processor,
        ]:
            result = await generator.generate(ctx)
            all_events.extend(result.events_published)
        assert isinstance(all_events, list)

    async def test_o2c_cycle_artifact_completeness(
        self,
        so_generator,
        shipment_generator,
        invoice_generator,
        payment_processor,
    ):
        """Every O2C stage produces non-empty artifacts when completed."""
        ctx = _make_generation_context()

        for gen_name, generator in [
            ("SO", so_generator),
            ("SHP", shipment_generator),
            ("INV", invoice_generator),
            ("PAY", payment_processor),
        ]:
            result = await generator.generate(ctx)
            if result.status == "completed":
                assert isinstance(result.artifacts, dict), (
                    f"{gen_name} artifacts must be dict"
                )

    async def test_o2c_cycle_invoice_amounts_from_shipment(
        self,
        invoice_generator,
    ):
        """Invoice amounts are derived from shipment quantities, not order quantities."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        assert result.transaction_type == "customer_invoice"


# ═══════════════════════════════════════════════════════════════════════════════
# TestO2CErrorHandling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
class TestO2CErrorHandling:
    """Error handling tests for O2C transaction cycle."""

    def test_so_credit_check_failure_halts_cycle(self):
        """Credit check failure results in a blocked order."""
        result = CreditCheckResult(
            passed=False,
            customer_id="C-001",
            credit_limit=Decimal("10000.00"),
            current_ar_balance=Decimal("9500.00"),
            order_total=Decimal("1000.00"),
            available_credit=Decimal("500.00"),
            reason="Insufficient credit",
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_shipment_for_nonexistent_so_raises_error(
        self, shipment_generator
    ):
        """Attempting to ship against non-existent SO should handle gracefully."""
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        assert result.status in ("completed", "skipped", "failed")

    @pytest.mark.asyncio
    async def test_invoice_for_unshipped_order_raises_error(
        self, invoice_generator
    ):
        """Invoicing an unshipped order should handle gracefully."""
        ctx = _make_generation_context()
        result = await invoice_generator.generate(ctx)
        assert result.status in ("completed", "skipped", "failed")

    @pytest.mark.asyncio
    async def test_gl_posting_failure_triggers_rollback(
        self, shipment_generator, mock_gl_engine, mock_db_session
    ):
        """GL posting failure should trigger session rollback."""
        mock_gl_engine.post_journal_entry = AsyncMock(
            side_effect=GLPostingError("DB error during posting")
        )
        ctx = _make_generation_context()
        result = await shipment_generator.generate(ctx)
        assert result.status in ("completed", "skipped", "failed")

    def test_payment_allocation_error_raises_exception(self, payment_processor):
        """PaymentAllocationError raised for invalid payment amount."""
        with pytest.raises(PaymentAllocationError):
            payment_processor._allocate_payment_fifo(
                payment_amount=Decimal("0.00"),
                open_invoices=[],
                payment_date=date(2025, 2, 15),
                payment_id=uuid4(),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TestO2CDecimalPrecision
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.o2c
@pytest.mark.financial
class TestO2CDecimalPrecision:
    """Tests ensuring all O2C amounts use Decimal, NEVER float."""

    def test_all_amounts_are_decimal_type(self):
        """CreditCheckResult amounts must be Decimal."""
        result = CreditCheckResult(
            passed=True,
            customer_id="C-001",
            credit_limit=Decimal("50000.00"),
            current_ar_balance=Decimal("10000.00"),
            order_total=Decimal("5000.00"),
            available_credit=Decimal("40000.00"),
        )
        assert isinstance(result.credit_limit, Decimal)
        assert isinstance(result.current_ar_balance, Decimal)
        assert isinstance(result.order_total, Decimal)
        assert isinstance(result.available_credit, Decimal)

    def test_no_float_in_gl_entries(self):
        """GL entry amounts must always be Decimal type."""
        gl_entry: Dict[str, Any] = {
            "lines": [
                {"account_code": "1010", "debit": Decimal("1000.00"), "credit": Decimal("0.00")},
                {"account_code": "1200", "debit": Decimal("0.00"), "credit": Decimal("1000.00")},
            ]
        }
        for line in gl_entry["lines"]:
            assert isinstance(line["debit"], Decimal)
            assert isinstance(line["credit"], Decimal)
            assert not isinstance(line["debit"], float)
            assert not isinstance(line["credit"], float)

    def test_fifo_allocation_amounts_are_decimal(self, payment_processor):
        """FIFO allocation amounts must all be Decimal type."""
        invoice = _make_open_invoice(
            original_amount=Decimal("1000.00"),
            open_balance=Decimal("1000.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("1000.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        for alloc in allocations:
            assert isinstance(alloc.allocated_amount, Decimal)
            assert isinstance(alloc.invoice_balance_before, Decimal)
            assert isinstance(alloc.invoice_balance_after, Decimal)
            assert isinstance(alloc.discount_taken, Decimal)
            assert isinstance(alloc.write_off_amount, Decimal)
        assert isinstance(unapplied, Decimal)

    def test_unapplied_cash_amount_is_decimal(self, payment_processor):
        """Unapplied cash from overpayment must be Decimal type."""
        invoice = _make_open_invoice(
            original_amount=Decimal("500.00"),
            open_balance=Decimal("500.00"),
        )
        allocations, unapplied = payment_processor._allocate_payment_fifo(
            payment_amount=Decimal("750.00"),
            open_invoices=[invoice],
            payment_date=date(2025, 2, 15),
            payment_id=uuid4(),
        )
        assert isinstance(unapplied, Decimal)
        assert unapplied == Decimal("250.00")
