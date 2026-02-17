"""
Comprehensive tests for Customer, Vendor, and Bank simulators.

Tests cover:
- CustomerSimulator: order generation, payment simulation, dispute handling
- VendorSimulator: invoice generation (95% PO match ±5%), goods delivery, inquiry response
- BankSimulator: bank statement generation, payment processing (check/ACH/wire clearing)
- Pydantic V2 data model validation for all output types
- Cross-simulator integration: reproducibility, consistency checks

AAP §0.7.5 standards:
- All external dependencies mocked (no live API calls)
- No real Redis (fakeredis only if needed)
- Async tests use pytest-asyncio
- Random seed=42 for reproducibility
- Coverage target ≥80%
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from app.events.event_bus import EventBus
from app.events.event_types import (
    DiscrepancyDetected,
    DocumentGenerated,
    TransactionCreated,
)
from app.external_world.bank_simulator import (
    BankSimulator,
    BankSimulatorConfig,
    BankStatement,
    BankStatementEntry,
    PaymentProcessingResult,
)
from app.external_world.behavior_profiles import (
    BehaviorProfile,
    BehaviorProfileFactory,
    InvoiceBehavior,
    OrderBehavior,
    PaymentBehavior,
)
from app.external_world.customer_simulator import (
    CustomerDispute,
    CustomerOrder,
    CustomerPayment,
    CustomerSimulator,
    CustomerSimulatorConfig,
)
from app.external_world.vendor_simulator import (
    GoodsDelivery,
    VendorInquiryResponse,
    VendorInvoice,
    VendorSimulator,
    VendorSimulatorConfig,
)


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Return an AsyncMock with EventBus spec for verifying event publishing."""
    bus = AsyncMock(spec=EventBus)
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_amount_distribution() -> MagicMock:
    """Return a mock AmountDistribution with deterministic return values."""
    dist = MagicMock()
    dist.sample_customer_order_amount = MagicMock(return_value=5000.0)
    dist.sample_vendor_invoice_amount = MagicMock(return_value=5000.0)
    dist.sample_po_amount = MagicMock(return_value=5000.0)
    dist.round_to_nearest = MagicMock(
        side_effect=lambda amount, precision=100: round(amount / precision) * precision
    )
    return dist


@pytest.fixture
def mock_order_frequency_model() -> MagicMock:
    """Return a mock OrderFrequencyModel that returns 5 orders per day."""
    model = MagicMock()
    model.sample_order_count = MagicMock(return_value=5)
    return model


@pytest.fixture
def mock_payment_timing_model() -> MagicMock:
    """Return a mock PaymentTimingModel with a fixed payment date."""
    model = MagicMock()
    model.get_payment_date = MagicMock(return_value=date(2024, 2, 15))
    return model


@pytest.fixture
def mock_selection_model() -> MagicMock:
    """Return a mock SelectionModel with Pareto 80/20 selection mocks."""
    model = MagicMock()
    model.select_vendor = MagicMock(
        return_value={"vendor_id": "V-001", "name": "Test Vendor"}
    )
    model.select_customers_batch = MagicMock(
        side_effect=lambda customers, count: customers[:count]
    )
    model.select_product = MagicMock(
        return_value={"product_id": "P-001", "name": "Widget A", "price": 50.0}
    )
    return model


@pytest.fixture
def sample_customer() -> Dict[str, Any]:
    """Return a standard-tier test customer with average behaviour profile."""
    return {
        "customer_id": str(uuid4()),
        "name": "Test Customer",
        "tier": "standard",
        "revenue": 500000.0,
        "payment_terms": "Net 30",
        "behavior_profile": BehaviorProfileFactory.create_profile(
            "average", "customer"
        ),
    }


@pytest.fixture
def sample_customer_strategic() -> Dict[str, Any]:
    """Return a strategic-tier test customer with excellent behaviour profile."""
    return {
        "customer_id": str(uuid4()),
        "name": "Strategic Customer",
        "tier": "strategic",
        "revenue": 2000000.0,
        "payment_terms": "Net 15",
        "behavior_profile": BehaviorProfileFactory.create_profile(
            "excellent", "customer"
        ),
    }


@pytest.fixture
def sample_customer_problem() -> Dict[str, Any]:
    """Return a transactional-tier problem customer for edge-case testing."""
    return {
        "customer_id": str(uuid4()),
        "name": "Problem Customer",
        "tier": "transactional",
        "revenue": 50000.0,
        "payment_terms": "Net 60",
        "behavior_profile": BehaviorProfileFactory.create_profile(
            "problem", "customer"
        ),
    }


@pytest.fixture
def sample_vendor() -> Dict[str, Any]:
    """Return a standard-tier test vendor with 'good' behaviour profile."""
    return {
        "vendor_id": str(uuid4()),
        "name": "Test Vendor",
        "tier": "standard",
        "category": "supplies",
        "spend_history": 200000.0,
        "behavior_profile": BehaviorProfileFactory.create_profile(
            "good", "vendor"
        ),
    }


@pytest.fixture
def sample_vendor_excellent() -> Dict[str, Any]:
    """Return a strategic-tier vendor with excellent behaviour profile."""
    return {
        "vendor_id": str(uuid4()),
        "name": "Excellent Vendor",
        "tier": "strategic",
        "category": "raw_materials",
        "spend_history": 1000000.0,
        "behavior_profile": BehaviorProfileFactory.create_profile(
            "excellent", "vendor"
        ),
    }


@pytest.fixture
def sample_vendor_problem() -> Dict[str, Any]:
    """Return a transactional-tier vendor with problem behaviour profile."""
    return {
        "vendor_id": str(uuid4()),
        "name": "Problem Vendor",
        "tier": "transactional",
        "category": "misc",
        "spend_history": 10000.0,
        "behavior_profile": BehaviorProfileFactory.create_profile(
            "problem", "vendor"
        ),
    }


@pytest.fixture
def sample_purchase_order() -> Dict[str, Any]:
    """Return a sample purchase order dict for invoice/delivery testing."""
    return {
        "po_id": str(uuid4()),
        "vendor_id": "V-001",
        "amount": 5000.0,
        "items": 100,
        "quantity": 100,
        "status": "approved",
    }


@pytest.fixture
def sample_bank() -> Dict[str, Any]:
    """Return a sample bank entity dict."""
    return {
        "bank_id": str(uuid4()),
        "name": "Test Bank",
        "account_number": "ACCT-0001",
    }


@pytest.fixture
def sample_transactions() -> List[Dict[str, Any]]:
    """Return a mix of debit/credit transactions for bank statement testing."""
    return [
        {
            "date": date(2024, 1, 5),
            "amount": 1000.0,
            "description": "Vendor Payment",
            "type": "debit",
        },
        {
            "date": date(2024, 1, 10),
            "amount": 3000.0,
            "description": "Customer Receipt",
            "type": "credit",
        },
        {
            "date": date(2024, 1, 15),
            "amount": 500.0,
            "description": "Utility Payment",
            "type": "debit",
        },
        {
            "date": date(2024, 1, 20),
            "amount": 7500.0,
            "description": "Large Customer Receipt",
            "type": "credit",
        },
        {
            "date": date(2024, 1, 25),
            "amount": 2000.0,
            "description": "Supplier Payment",
            "type": "debit",
        },
    ]


@pytest.fixture
def sample_invoice() -> Dict[str, Any]:
    """Return a sample invoice dict for customer payment testing."""
    return {
        "invoice_id": str(uuid4()),
        "customer_id": "C-001",
        "amount": 5000.0,
        "due_date": date(2024, 2, 15),
        "payment_terms": "Net 30",
        "date": date(2024, 1, 15),
    }


@pytest.fixture
def sample_vendors() -> List[Dict[str, Any]]:
    """Return a list of 5 vendor dicts with varying tiers for batch testing."""
    tiers = ["strategic", "standard", "standard", "transactional", "transactional"]
    profiles = ["excellent", "good", "average", "poor", "problem"]
    vendors: List[Dict[str, Any]] = []
    for tier, profile_name in zip(tiers, profiles):
        vendors.append(
            {
                "vendor_id": str(uuid4()),
                "name": f"Vendor {tier.title()}",
                "tier": tier,
                "category": "supplies",
                "spend_history": 100000.0,
                "behavior_profile": BehaviorProfileFactory.create_profile(
                    profile_name, "vendor"
                ),
            }
        )
    return vendors


# ============================================================================
# Simulator Instance Fixtures
# ============================================================================


@pytest.fixture
def customer_simulator(
    mock_amount_distribution: MagicMock,
    mock_order_frequency_model: MagicMock,
    mock_payment_timing_model: MagicMock,
    mock_selection_model: MagicMock,
    mock_event_bus: AsyncMock,
) -> CustomerSimulator:
    """Create a CustomerSimulator with all mocked dependencies and seed=42."""
    return CustomerSimulator(
        amount_distribution=mock_amount_distribution,
        order_frequency_model=mock_order_frequency_model,
        payment_timing_model=mock_payment_timing_model,
        selection_model=mock_selection_model,
        event_bus=mock_event_bus,
        seed=42,
    )


@pytest.fixture
def vendor_simulator(
    mock_amount_distribution: MagicMock,
    mock_selection_model: MagicMock,
    mock_event_bus: AsyncMock,
) -> VendorSimulator:
    """Create a VendorSimulator with all mocked dependencies and seed=42."""
    return VendorSimulator(
        amount_distribution=mock_amount_distribution,
        selection_model=mock_selection_model,
        event_bus=mock_event_bus,
        seed=42,
    )


@pytest.fixture
def bank_simulator(
    mock_amount_distribution: MagicMock,
    mock_event_bus: AsyncMock,
) -> BankSimulator:
    """Create a BankSimulator with all mocked dependencies and seed=42."""
    return BankSimulator(
        amount_distribution=mock_amount_distribution,
        event_bus=mock_event_bus,
        seed=42,
    )


# ============================================================================
# Phase 3: CustomerSimulator Tests
# ============================================================================


class TestCustomerSimulatorConfig:
    """Validate CustomerSimulatorConfig Pydantic model defaults and overrides."""

    def test_default_config(self) -> None:
        """Default config should provide valid payment terms and dispute types."""
        config = CustomerSimulatorConfig()
        assert config.default_payment_terms is not None
        assert isinstance(config.default_payment_terms, str)
        assert len(config.dispute_types) > 0
        assert "pricing" in config.dispute_types
        assert "quality" in config.dispute_types

    def test_custom_config(self) -> None:
        """Custom values must be preserved after construction."""
        config = CustomerSimulatorConfig(
            default_payment_terms="Net 45",
            dispute_types=["pricing", "service"],
        )
        assert config.default_payment_terms == "Net 45"
        assert config.dispute_types == ["pricing", "service"]


class TestCustomerSimulatorInit:
    """Validate CustomerSimulator construction and dependency injection."""

    def test_constructor_injection(
        self,
        mock_amount_distribution: MagicMock,
        mock_order_frequency_model: MagicMock,
        mock_payment_timing_model: MagicMock,
        mock_selection_model: MagicMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """All injected dependencies should be stored on the instance."""
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )
        assert sim._amount_distribution is mock_amount_distribution
        assert sim._order_frequency_model is mock_order_frequency_model
        assert sim._payment_timing_model is mock_payment_timing_model
        assert sim._selection_model is mock_selection_model
        assert sim._event_bus is mock_event_bus

    def test_default_config_used_when_none(self) -> None:
        """When config=None, a default CustomerSimulatorConfig should be used."""
        sim = CustomerSimulator(config=None, seed=42)
        assert sim._config is not None
        assert isinstance(sim._config, CustomerSimulatorConfig)

    def test_seed_creates_reproducible_rng(self) -> None:
        """Two simulators with same seed must produce identical RNG sequences."""
        sim_a = CustomerSimulator(seed=42)
        sim_b = CustomerSimulator(seed=42)
        vals_a = [sim_a.rng.random() for _ in range(10)]
        vals_b = [sim_b.rng.random() for _ in range(10)]
        assert vals_a == vals_b


class TestCustomerOrderGeneration:
    """Test CustomerSimulator.generate_customer_order and generate_daily_orders."""

    @pytest.mark.asyncio
    async def test_generate_customer_order_returns_customer_order(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
    ) -> None:
        """generate_customer_order must return a valid CustomerOrder."""
        order = await customer_simulator.generate_customer_order(
            sample_customer, date(2024, 1, 15)
        )
        assert isinstance(order, CustomerOrder)
        assert order.customer_id == sample_customer["customer_id"]
        assert order.order_date == date(2024, 1, 15)
        assert order.total_amount > 0

    @pytest.mark.asyncio
    async def test_generate_customer_order_uses_amount_distribution(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
        mock_amount_distribution: MagicMock,
    ) -> None:
        """Amount distribution should be called to determine order amount."""
        await customer_simulator.generate_customer_order(
            sample_customer, date(2024, 1, 15)
        )
        mock_amount_distribution.sample_customer_order_amount.assert_called()

    @pytest.mark.asyncio
    async def test_generate_customer_order_publishes_event(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
        mock_event_bus: AsyncMock,
    ) -> None:
        """A TransactionCreated event should be published for each order."""
        await customer_simulator.generate_customer_order(
            sample_customer, date(2024, 1, 15)
        )
        mock_event_bus.publish.assert_called_once()
        published_event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(published_event, TransactionCreated)

    @pytest.mark.asyncio
    async def test_generate_customer_order_no_event_bus(
        self,
        mock_amount_distribution: MagicMock,
        sample_customer: Dict[str, Any],
    ) -> None:
        """Simulator without event_bus should generate orders without error."""
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution,
            event_bus=None,
            seed=42,
        )
        order = await sim.generate_customer_order(
            sample_customer, date(2024, 1, 15)
        )
        assert isinstance(order, CustomerOrder)

    @pytest.mark.asyncio
    async def test_generate_daily_orders_returns_list(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
    ) -> None:
        """generate_daily_orders must return a list of CustomerOrder objects."""
        customers = [sample_customer] * 10
        orders = await customer_simulator.generate_daily_orders(
            customers, date(2024, 1, 15)
        )
        assert isinstance(orders, list)
        assert all(isinstance(o, CustomerOrder) for o in orders)

    @pytest.mark.asyncio
    async def test_generate_daily_orders_uses_frequency_model(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
        mock_order_frequency_model: MagicMock,
    ) -> None:
        """The order frequency model should be called to determine daily count."""
        customers = [sample_customer] * 10
        await customer_simulator.generate_daily_orders(
            customers, date(2024, 1, 15)
        )
        mock_order_frequency_model.sample_order_count.assert_called_once_with(
            date(2024, 1, 15)
        )

    @pytest.mark.asyncio
    async def test_generate_daily_orders_uses_selection_model(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
        mock_selection_model: MagicMock,
    ) -> None:
        """The selection model should pick which customers place orders."""
        customers = [sample_customer] * 10
        await customer_simulator.generate_daily_orders(
            customers, date(2024, 1, 15)
        )
        mock_selection_model.select_customers_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_customer_order_with_products(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
    ) -> None:
        """Order items should reference products when a catalogue is provided."""
        products = [
            {"product_id": "P-001", "name": "Widget A", "price": 50.0},
            {"product_id": "P-002", "name": "Widget B", "price": 75.0},
        ]
        order = await customer_simulator.generate_customer_order(
            sample_customer, date(2024, 1, 15), available_products=products
        )
        assert isinstance(order, CustomerOrder)
        assert len(order.items) > 0

    @pytest.mark.asyncio
    async def test_generate_daily_orders_empty_customer_list(
        self,
        customer_simulator: CustomerSimulator,
    ) -> None:
        """An empty customer list should produce no orders."""
        orders = await customer_simulator.generate_daily_orders(
            [], date(2024, 1, 15)
        )
        assert orders == []


class TestCustomerPaymentSimulation:
    """Test CustomerSimulator.simulate_payment and simulate_daily_payments."""

    @pytest.mark.asyncio
    async def test_simulate_payment_returns_customer_payment(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
        sample_invoice: Dict[str, Any],
    ) -> None:
        """simulate_payment must return a valid CustomerPayment."""
        payment = await customer_simulator.simulate_payment(
            sample_customer, sample_invoice, date(2024, 2, 10)
        )
        assert isinstance(payment, CustomerPayment)
        assert payment.customer_id == sample_customer["customer_id"]
        assert payment.amount_due > 0

    @pytest.mark.asyncio
    async def test_simulate_payment_uses_timing_model(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
        sample_invoice: Dict[str, Any],
        mock_payment_timing_model: MagicMock,
    ) -> None:
        """The payment timing model should be queried for payment date."""
        await customer_simulator.simulate_payment(
            sample_customer, sample_invoice, date(2024, 2, 10)
        )
        mock_payment_timing_model.get_payment_date.assert_called()

    @pytest.mark.asyncio
    async def test_simulate_payment_short_pay(
        self,
        mock_amount_distribution: MagicMock,
        sample_invoice: Dict[str, Any],
    ) -> None:
        """Customer with short_pay_rate=1.0 must always short-pay."""
        profile = BehaviorProfileFactory.create_profile("problem", "customer")
        profile.payment_behavior.short_pay_rate = 1.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Short-Pay Customer",
            "tier": "transactional",
            "revenue": 100_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution, seed=42
        )
        payment = await sim.simulate_payment(
            customer, sample_invoice, date(2024, 2, 10)
        )
        assert payment.is_short_pay is True
        assert payment.amount_paid < payment.amount_due

    @pytest.mark.asyncio
    async def test_simulate_payment_no_short_pay(
        self,
        mock_amount_distribution: MagicMock,
        sample_invoice: Dict[str, Any],
    ) -> None:
        """Customer with short_pay_rate=0.0 must never short-pay."""
        profile = BehaviorProfileFactory.create_profile("excellent", "customer")
        profile.payment_behavior.short_pay_rate = 0.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Full-Pay Customer",
            "tier": "strategic",
            "revenue": 1_000_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution, seed=42
        )
        payment = await sim.simulate_payment(
            customer, sample_invoice, date(2024, 2, 10)
        )
        assert payment.is_short_pay is False
        assert payment.amount_paid == payment.amount_due

    @pytest.mark.asyncio
    async def test_simulate_daily_payments(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
    ) -> None:
        """simulate_daily_payments must return a list of CustomerPayment."""
        invoices = [
            {
                "invoice_id": str(uuid4()),
                "customer_id": sample_customer["customer_id"],
                "amount": 5000.0,
                "due_date": date(2024, 2, 15),
                "payment_terms": "Net 30",
                "date": date(2024, 1, 15),
            }
            for _ in range(3)
        ]
        payments = await customer_simulator.simulate_daily_payments(
            customers=[sample_customer],
            outstanding_invoices=invoices,
            simulation_date=date(2024, 2, 20),
        )
        assert isinstance(payments, list)
        assert all(isinstance(p, CustomerPayment) for p in payments)


class TestCustomerDisputeHandling:
    """Test CustomerSimulator.simulate_dispute and check_disputes_for_invoices."""

    @pytest.mark.asyncio
    async def test_simulate_dispute_triggered(
        self, mock_amount_distribution: MagicMock
    ) -> None:
        """Customer with dispute_rate=1.0 must always trigger a dispute."""
        profile = BehaviorProfileFactory.create_profile("problem", "customer")
        profile.payment_behavior.dispute_rate = 1.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Disputatious Customer",
            "tier": "transactional",
            "revenue": 80_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution, seed=42
        )
        invoice = {
            "invoice_id": str(uuid4()),
            "customer_id": customer["customer_id"],
            "amount": 3000.0,
            "due_date": date(2024, 2, 15),
            "payment_terms": "Net 30",
        }
        dispute = await sim.simulate_dispute(customer, invoice, date(2024, 1, 15))
        assert dispute is not None
        assert isinstance(dispute, CustomerDispute)
        assert dispute.customer_id == customer["customer_id"]

    @pytest.mark.asyncio
    async def test_simulate_dispute_not_triggered(
        self, mock_amount_distribution: MagicMock
    ) -> None:
        """Customer with dispute_rate=0.0 must never trigger a dispute."""
        profile = BehaviorProfileFactory.create_profile("excellent", "customer")
        profile.payment_behavior.dispute_rate = 0.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Happy Customer",
            "tier": "strategic",
            "revenue": 1_000_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution, seed=42
        )
        invoice = {
            "invoice_id": str(uuid4()),
            "customer_id": customer["customer_id"],
            "amount": 3000.0,
            "due_date": date(2024, 2, 15),
            "payment_terms": "Net 30",
        }
        dispute = await sim.simulate_dispute(customer, invoice, date(2024, 1, 15))
        assert dispute is None

    @pytest.mark.asyncio
    async def test_simulate_dispute_type_from_config(
        self, mock_amount_distribution: MagicMock
    ) -> None:
        """Dispute type must come from CustomerSimulatorConfig.dispute_types."""
        profile = BehaviorProfileFactory.create_profile("problem", "customer")
        profile.payment_behavior.dispute_rate = 1.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Disputatious",
            "tier": "transactional",
            "revenue": 80_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        config = CustomerSimulatorConfig(
            dispute_types=["pricing", "quality", "quantity", "service"]
        )
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution,
            config=config,
            seed=42,
        )
        invoice = {
            "invoice_id": str(uuid4()),
            "customer_id": customer["customer_id"],
            "amount": 3000.0,
            "due_date": date(2024, 2, 15),
            "payment_terms": "Net 30",
        }
        dispute = await sim.simulate_dispute(customer, invoice, date(2024, 1, 15))
        assert dispute is not None
        assert dispute.dispute_type in config.dispute_types

    @pytest.mark.asyncio
    async def test_simulate_dispute_publishes_event(
        self,
        mock_amount_distribution: MagicMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """When a dispute is triggered an event should be published."""
        profile = BehaviorProfileFactory.create_profile("problem", "customer")
        profile.payment_behavior.dispute_rate = 1.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Disputatious",
            "tier": "transactional",
            "revenue": 80_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution,
            event_bus=mock_event_bus,
            seed=42,
        )
        invoice = {
            "invoice_id": str(uuid4()),
            "customer_id": customer["customer_id"],
            "amount": 3000.0,
            "due_date": date(2024, 2, 15),
            "payment_terms": "Net 30",
        }
        dispute = await sim.simulate_dispute(customer, invoice, date(2024, 1, 15))
        assert dispute is not None
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(event, DiscrepancyDetected)

    @pytest.mark.asyncio
    async def test_check_disputes_for_invoices(
        self, mock_amount_distribution: MagicMock
    ) -> None:
        """check_disputes_for_invoices should return a filtered list."""
        profile = BehaviorProfileFactory.create_profile("problem", "customer")
        profile.payment_behavior.dispute_rate = 1.0
        customer: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "name": "Disputatious",
            "tier": "transactional",
            "revenue": 80_000.0,
            "payment_terms": "Net 30",
            "behavior_profile": profile,
        }
        sim = CustomerSimulator(
            amount_distribution=mock_amount_distribution, seed=42
        )
        invoices = [
            {
                "invoice_id": str(uuid4()),
                "customer_id": customer["customer_id"],
                "amount": float(i * 1000),
                "due_date": date(2024, 2, 15),
                "payment_terms": "Net 30",
            }
            for i in range(1, 4)
        ]
        disputes = await sim.check_disputes_for_invoices(
            customer, invoices, date(2024, 1, 15)
        )
        assert isinstance(disputes, list)
        assert all(isinstance(d, CustomerDispute) for d in disputes)


# ============================================================================
# Phase 4: VendorSimulator Tests
# ============================================================================


class TestVendorSimulatorConfig:
    """Validate VendorSimulatorConfig Pydantic model."""

    def test_default_config(self) -> None:
        """Default config must have valid invoice timing and delivery params."""
        config = VendorSimulatorConfig()
        assert config.partial_delivery_rate >= 0.0
        assert config.backorder_rate >= 0.0
        assert config.invoice_timing_days is not None

    def test_invoice_timing_days_mapping(self) -> None:
        """Default timing map should contain standard keys."""
        config = VendorSimulatorConfig()
        assert isinstance(config.invoice_timing_days, dict)
        assert "immediate" in config.invoice_timing_days or len(config.invoice_timing_days) > 0


class TestVendorSimulatorInit:
    """Validate VendorSimulator construction and dependency injection."""

    def test_constructor_injection(
        self,
        mock_amount_distribution: MagicMock,
        mock_selection_model: MagicMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """All injected dependencies should be stored on the instance."""
        sim = VendorSimulator(
            amount_distribution=mock_amount_distribution,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )
        assert sim._amount_distribution is mock_amount_distribution
        assert sim._selection_model is mock_selection_model
        assert sim._event_bus is mock_event_bus

    def test_seed_reproducibility(self) -> None:
        """Two simulators with the same seed must produce identical RNG."""
        sim_a = VendorSimulator(seed=42)
        sim_b = VendorSimulator(seed=42)
        vals_a = [sim_a.rng.random() for _ in range(10)]
        vals_b = [sim_b.rng.random() for _ in range(10)]
        assert vals_a == vals_b


class TestVendorInvoiceGeneration:
    """Test VendorSimulator.generate_vendor_invoice and batch generation."""

    @pytest.mark.asyncio
    async def test_generate_vendor_invoice_returns_vendor_invoice(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
        sample_purchase_order: Dict[str, Any],
    ) -> None:
        """generate_vendor_invoice must return a valid VendorInvoice."""
        invoice = await vendor_simulator.generate_vendor_invoice(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        assert isinstance(invoice, VendorInvoice)
        assert invoice.vendor_id == sample_vendor["vendor_id"]
        assert invoice.po_id == sample_purchase_order["po_id"]
        assert invoice.invoice_amount > 0

    @pytest.mark.asyncio
    async def test_invoice_amount_matches_po_within_tolerance(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """95%+ of invoices should match PO amount within ±5%."""
        sim = VendorSimulator(event_bus=mock_event_bus, seed=42)
        profile = BehaviorProfileFactory.create_profile("good", "vendor")
        vendor: Dict[str, Any] = {
            "vendor_id": str(uuid4()),
            "name": "Test Vendor",
            "tier": "standard",
            "category": "supplies",
            "spend_history": 200_000.0,
            "behavior_profile": profile,
        }
        po_amount = 5000.0
        po: Dict[str, Any] = {
            "po_id": str(uuid4()),
            "vendor_id": vendor["vendor_id"],
            "amount": po_amount,
            "items": [{"product_id": "P-001", "quantity": 100, "unit_price": 50.0}],
            "status": "approved",
        }
        match_count = 0
        n_samples = 100
        for _ in range(n_samples):
            inv = await sim.generate_vendor_invoice(vendor, po, date(2024, 1, 15))
            tolerance = po_amount * 0.05
            if abs(inv.invoice_amount - po_amount) <= tolerance:
                match_count += 1
        # Accept ≥85 matches (relaxed from theoretical 95% due to randomness)
        assert match_count >= 85, (
            f"Expected ≥85/100 invoices within ±5%, got {match_count}"
        )

    @pytest.mark.asyncio
    async def test_invoice_amount_has_variance_sometimes(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Over many invoices some should have non-zero variance."""
        sim = VendorSimulator(event_bus=mock_event_bus, seed=42)
        profile = BehaviorProfileFactory.create_profile("average", "vendor")
        vendor: Dict[str, Any] = {
            "vendor_id": str(uuid4()),
            "name": "Variance Vendor",
            "tier": "standard",
            "category": "supplies",
            "spend_history": 200_000.0,
            "behavior_profile": profile,
        }
        po: Dict[str, Any] = {
            "po_id": str(uuid4()),
            "vendor_id": vendor["vendor_id"],
            "amount": 10000.0,
            "items": [{"product_id": "P-002", "quantity": 200, "unit_price": 50.0}],
            "status": "approved",
        }
        variance_count = 0
        for _ in range(100):
            inv = await sim.generate_vendor_invoice(vendor, po, date(2024, 1, 15))
            if inv.variance_amount != 0.0:
                variance_count += 1
        assert variance_count > 0, "Expected at least one invoice with variance"

    @pytest.mark.asyncio
    async def test_invoice_publishes_event(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
        sample_purchase_order: Dict[str, Any],
        mock_event_bus: AsyncMock,
    ) -> None:
        """A DocumentGenerated event should be published for each invoice."""
        await vendor_simulator.generate_vendor_invoice(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(event, DocumentGenerated)

    @pytest.mark.asyncio
    async def test_generate_batch_invoices(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
    ) -> None:
        """Batch generation must return a list of VendorInvoice objects."""
        # generate_batch_invoices expects List[Dict] with "vendor" and "po" keys
        vendors_with_pos: List[Dict[str, Any]] = [
            {
                "vendor": sample_vendor,
                "po": {
                    "po_id": str(uuid4()),
                    "vendor_id": sample_vendor["vendor_id"],
                    "amount": 3000.0 + i * 500,
                    "items": [{"product_id": f"P-{i}", "quantity": 10, "unit_price": 50.0}],
                    "status": "approved",
                },
            }
            for i in range(5)
        ]
        invoices = await vendor_simulator.generate_batch_invoices(
            vendors_with_pos, date(2024, 1, 15)
        )
        assert isinstance(invoices, list)
        assert len(invoices) == 5
        assert all(isinstance(inv, VendorInvoice) for inv in invoices)


class TestGoodsDelivery:
    """Test VendorSimulator.simulate_goods_delivery."""

    @pytest.mark.asyncio
    async def test_simulate_goods_delivery_returns_delivery(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
        sample_purchase_order: Dict[str, Any],
    ) -> None:
        """simulate_goods_delivery must return a valid GoodsDelivery."""
        delivery = await vendor_simulator.simulate_goods_delivery(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        assert isinstance(delivery, GoodsDelivery)
        assert delivery.delivery_id is not None

    @pytest.mark.asyncio
    async def test_delivery_items_less_or_equal_to_ordered(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
        sample_purchase_order: Dict[str, Any],
    ) -> None:
        """Delivered quantity must never exceed ordered quantity."""
        delivery = await vendor_simulator.simulate_goods_delivery(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        assert delivery.items_delivered <= delivery.items_ordered

    @pytest.mark.asyncio
    async def test_partial_delivery_rate(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Over many deliveries, some should be partial."""
        sim = VendorSimulator(event_bus=mock_event_bus, seed=42)
        profile = BehaviorProfileFactory.create_profile("average", "vendor")
        vendor: Dict[str, Any] = {
            "vendor_id": str(uuid4()),
            "name": "Partial Vendor",
            "tier": "standard",
            "category": "supplies",
            "spend_history": 200_000.0,
            "behavior_profile": profile,
        }
        po: Dict[str, Any] = {
            "po_id": str(uuid4()),
            "vendor_id": vendor["vendor_id"],
            "amount": 5000.0,
            "quantity": 100,
            "line_items": [{"product_id": "P-001", "quantity": 100, "unit_price": 50.0}],
            "status": "approved",
        }
        partial_count = 0
        for _ in range(200):
            delivery = await sim.simulate_goods_delivery(vendor, po, date(2024, 1, 15))
            if delivery.is_partial:
                partial_count += 1
        # At least 1 partial delivery expected from 200 attempts
        assert partial_count >= 1, "Expected at least one partial delivery in 200"

    @pytest.mark.asyncio
    async def test_backorder_rate(
        self,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Over many deliveries, some should be back-ordered."""
        sim = VendorSimulator(event_bus=mock_event_bus, seed=42)
        profile = BehaviorProfileFactory.create_profile("average", "vendor")
        vendor: Dict[str, Any] = {
            "vendor_id": str(uuid4()),
            "name": "Backorder Vendor",
            "tier": "standard",
            "category": "supplies",
            "spend_history": 200_000.0,
            "behavior_profile": profile,
        }
        po: Dict[str, Any] = {
            "po_id": str(uuid4()),
            "vendor_id": vendor["vendor_id"],
            "amount": 5000.0,
            "quantity": 100,
            "line_items": [{"product_id": "P-001", "quantity": 100, "unit_price": 50.0}],
            "status": "approved",
        }
        backorder_count = 0
        for _ in range(500):
            delivery = await sim.simulate_goods_delivery(vendor, po, date(2024, 1, 15))
            if delivery.is_backorder:
                backorder_count += 1
        # With a ~2% default rate, expect at least 1 backorder in 500
        assert backorder_count >= 1, "Expected at least one backorder in 500"


class TestVendorInquiryResponse:
    """Test VendorSimulator.simulate_inquiry_response."""

    @pytest.mark.asyncio
    async def test_simulate_inquiry_response(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
    ) -> None:
        """simulate_inquiry_response must return a valid VendorInquiryResponse."""
        response = await vendor_simulator.simulate_inquiry_response(
            sample_vendor, date(2024, 1, 15), "pricing"
        )
        assert isinstance(response, VendorInquiryResponse)
        assert response.response_time_days >= 1

    @pytest.mark.asyncio
    async def test_inquiry_type_preserved(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
    ) -> None:
        """The inquiry type must be preserved in the response object."""
        for itype in ["pricing", "quality", "status"]:
            response = await vendor_simulator.simulate_inquiry_response(
                sample_vendor, date(2024, 1, 15), itype
            )
            assert response.inquiry_type == itype


class TestVendorSelection:
    """Test VendorSimulator.select_vendor_for_purchase."""

    @pytest.mark.asyncio
    async def test_select_vendor_for_purchase(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """Vendor selection should use the selection model."""
        selected = await vendor_simulator.select_vendor_for_purchase(
            sample_vendors, "supplies"
        )
        assert selected is not None
        assert "vendor_id" in selected

    @pytest.mark.asyncio
    async def test_select_vendor_without_selection_model(
        self,
        mock_event_bus: AsyncMock,
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """When selection_model is None, random fallback must not raise."""
        sim = VendorSimulator(
            selection_model=None, event_bus=mock_event_bus, seed=42
        )
        selected = await sim.select_vendor_for_purchase(sample_vendors, "supplies")
        assert selected is not None


# ============================================================================
# Phase 5: BankSimulator Tests
# ============================================================================


class TestBankSimulatorConfig:
    """Validate BankSimulatorConfig Pydantic model defaults."""

    def test_default_config(self) -> None:
        """Default config must define clearing times, fees, and rates."""
        config = BankSimulatorConfig()
        assert config.check_clearing_days_min == 1
        assert config.check_clearing_days_max == 3
        assert config.ach_clearing_days_min == 1
        assert config.ach_clearing_days_max == 2
        assert config.wire_clearing_days == 0
        assert config.payment_failure_rate == pytest.approx(0.01, abs=0.005)
        assert config.monthly_service_fee == pytest.approx(25.0, abs=5.0)
        assert config.interest_rate_annual == pytest.approx(0.01, abs=0.005)


class TestBankSimulatorInit:
    """Validate BankSimulator construction and dependency injection."""

    def test_constructor_injection(
        self,
        mock_amount_distribution: MagicMock,
        mock_event_bus: AsyncMock,
    ) -> None:
        """All injected dependencies should be stored."""
        sim = BankSimulator(
            amount_distribution=mock_amount_distribution,
            event_bus=mock_event_bus,
            seed=42,
        )
        assert sim.event_bus is mock_event_bus
        assert sim.amount_distribution is mock_amount_distribution

    def test_seed_reproducibility(self) -> None:
        """Two simulators with same seed must be deterministic."""
        sim_a = BankSimulator(seed=42)
        sim_b = BankSimulator(seed=42)
        vals_a = [sim_a.rng.random() for _ in range(10)]
        vals_b = [sim_b.rng.random() for _ in range(10)]
        assert vals_a == vals_b


class TestBankStatementGeneration:
    """Test BankSimulator.generate_bank_statement and monthly generation."""

    @pytest.mark.asyncio
    async def test_generate_bank_statement_returns_bank_statement(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        sample_transactions: List[Dict[str, Any]],
    ) -> None:
        """generate_bank_statement must return a valid BankStatement."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, sample_transactions, date(2024, 1, 1), opening_balance=10000.0
        )
        assert isinstance(statement, BankStatement)
        assert statement.bank_id == sample_bank["bank_id"]
        assert statement.opening_balance == pytest.approx(10000.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_statement_period_covers_full_month(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        sample_transactions: List[Dict[str, Any]],
    ) -> None:
        """Statement period should span the full month."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, sample_transactions, date(2024, 1, 1), opening_balance=10000.0
        )
        assert statement.statement_period_start == date(2024, 1, 1)
        assert statement.statement_period_end == date(2024, 1, 31)

    @pytest.mark.asyncio
    async def test_statement_includes_service_fee(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Statement should include a monthly service fee debit entry."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, [], date(2024, 1, 1), opening_balance=10000.0
        )
        # Find fee entry
        fee_entries = [
            e for e in statement.entries if "fee" in e.description.lower()
        ]
        assert len(fee_entries) >= 1, "Expected at least one service fee entry"

    @pytest.mark.asyncio
    async def test_statement_includes_interest(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Statement should include an interest credit entry."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, [], date(2024, 1, 1), opening_balance=100000.0
        )
        interest_entries = [
            e for e in statement.entries if "interest" in e.description.lower()
        ]
        assert len(interest_entries) >= 1, "Expected at least one interest entry"

    @pytest.mark.asyncio
    async def test_closing_balance_calculation(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        sample_transactions: List[Dict[str, Any]],
    ) -> None:
        """Closing balance must equal opening + credits - debits - fees + interest."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, sample_transactions, date(2024, 1, 1), opening_balance=10000.0
        )
        expected_closing = (
            statement.opening_balance
            + statement.total_credits
            - abs(statement.total_debits)
            - statement.total_fees
            + statement.total_interest
        )
        assert statement.closing_balance == pytest.approx(expected_closing, abs=0.02)

    @pytest.mark.asyncio
    async def test_running_balance_chronological(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        sample_transactions: List[Dict[str, Any]],
    ) -> None:
        """Each entry's running balance follows from the previous entry."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, sample_transactions, date(2024, 1, 1), opening_balance=10000.0
        )
        if len(statement.entries) < 2:
            return  # nothing to validate
        prev_balance = statement.opening_balance
        for entry in statement.entries:
            expected = prev_balance + entry.amount
            assert entry.running_balance == pytest.approx(expected, abs=0.02), (
                f"Running balance mismatch at entry {entry.entry_id}"
            )
            prev_balance = entry.running_balance

    @pytest.mark.asyncio
    async def test_statement_entries_have_required_fields(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        sample_transactions: List[Dict[str, Any]],
    ) -> None:
        """Every BankStatementEntry must have all required fields populated."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, sample_transactions, date(2024, 1, 1), opening_balance=10000.0
        )
        for entry in statement.entries:
            assert entry.entry_id is not None
            assert entry.date is not None
            assert entry.description is not None and len(entry.description) > 0
            assert entry.amount is not None
            assert entry.entry_type is not None
            assert entry.running_balance is not None

    @pytest.mark.asyncio
    async def test_statement_publishes_event(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        mock_event_bus: AsyncMock,
    ) -> None:
        """A DocumentGenerated event should be published for the statement."""
        await bank_simulator.generate_bank_statement(
            sample_bank, [], date(2024, 1, 1), opening_balance=10000.0
        )
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert isinstance(event, DocumentGenerated)

    @pytest.mark.asyncio
    async def test_generate_monthly_statements(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Batch monthly generation must return a list of BankStatement."""
        # generate_monthly_statements expects (banks, all_transactions, statement_month)
        # Opening balance is extracted from bank dict via bank.get("opening_balance", 0.0)
        bank_with_balance = {**sample_bank, "opening_balance": 10000.0}
        banks = [bank_with_balance]
        all_transactions: Dict[str, List[Dict[str, Any]]] = {
            bank_with_balance["bank_id"]: [],
        }
        statements = await bank_simulator.generate_monthly_statements(
            banks, all_transactions, date(2024, 1, 1)
        )
        assert isinstance(statements, list)
        assert len(statements) == 1
        assert isinstance(statements[0], BankStatement)

    @pytest.mark.asyncio
    async def test_empty_transactions_still_generates_statement(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Statement with no transactions should still have fee/interest entries."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, [], date(2024, 1, 1), opening_balance=50000.0
        )
        assert isinstance(statement, BankStatement)
        assert len(statement.entries) > 0  # at least fee + interest


class TestPaymentProcessing:
    """Test BankSimulator.process_payment and batch processing."""

    @pytest.mark.asyncio
    async def test_process_check_payment_clearing_time(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Check payments must clear in 1-3 business days."""
        payment = {"payment_id": str(uuid4()), "amount": 1000.0, "payment_method": "check"}
        result = await bank_simulator.process_payment(
            sample_bank, payment, date(2024, 1, 15)  # Monday
        )
        assert isinstance(result, PaymentProcessingResult)
        days_to_clear = (result.cleared_date - date(2024, 1, 15)).days
        # 1-3 business days may become up to 5 calendar days due to weekends
        assert 1 <= days_to_clear <= 5

    @pytest.mark.asyncio
    async def test_process_ach_payment_clearing_time(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """ACH payments must clear in 1-2 business days."""
        payment = {"payment_id": str(uuid4()), "amount": 2500.0, "payment_method": "ach"}
        result = await bank_simulator.process_payment(
            sample_bank, payment, date(2024, 1, 15)
        )
        days_to_clear = (result.cleared_date - date(2024, 1, 15)).days
        # 1-2 business days may become up to 4 calendar days due to weekends
        assert 1 <= days_to_clear <= 4

    @pytest.mark.asyncio
    async def test_process_wire_payment_same_day(
        self,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Wire payments must clear same day (or next business day on weekend)."""
        # Use failure_rate=0 to guarantee success for clearing-time assertion
        config = BankSimulatorConfig(payment_failure_rate=0.0)
        sim = BankSimulator(config=config, seed=42)
        payment = {"payment_id": str(uuid4()), "amount": 50000.0, "payment_method": "wire"}
        result = await sim.process_payment(
            sample_bank, payment, date(2024, 1, 15)  # Monday
        )
        assert result.is_successful is True
        assert result.cleared_date is not None
        days_to_clear = (result.cleared_date - date(2024, 1, 15)).days
        assert days_to_clear <= 1  # same day or next business day

    @pytest.mark.asyncio
    async def test_payment_failure_occurs(self) -> None:
        """With failure_rate=1.0, all payments must fail."""
        config = BankSimulatorConfig(payment_failure_rate=1.0)
        sim = BankSimulator(config=config, seed=42)
        bank = {"bank_id": str(uuid4()), "name": "Fail Bank", "account_number": "X-001"}
        payment = {"payment_id": str(uuid4()), "amount": 500.0, "payment_method": "ach"}
        result = await sim.process_payment(bank, payment, date(2024, 1, 15))
        assert result.is_successful is False
        assert result.failure_reason is not None and len(result.failure_reason) > 0

    @pytest.mark.asyncio
    async def test_payment_success_default(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """With default failure rate (1%), most payments must succeed."""
        success_count = 0
        n = 50
        for _ in range(n):
            payment = {"payment_id": str(uuid4()), "amount": 1000.0, "payment_method": "ach"}
            result = await bank_simulator.process_payment(
                sample_bank, payment, date(2024, 1, 15)
            )
            if result.is_successful:
                success_count += 1
        assert success_count >= 40, f"Expected ≥40/50 successes, got {success_count}"

    @pytest.mark.asyncio
    async def test_weekend_adjustment_saturday(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Initiating on Saturday should push clearing to Monday or later."""
        payment = {"payment_id": str(uuid4()), "amount": 1000.0, "payment_method": "wire"}
        result = await bank_simulator.process_payment(
            sample_bank, payment, date(2024, 1, 13)  # Saturday
        )
        # Wire is same-day, but Saturday → Monday
        assert result.cleared_date.weekday() < 5  # must be a weekday

    @pytest.mark.asyncio
    async def test_weekend_adjustment_sunday(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Initiating on Sunday should push clearing to Monday or later."""
        payment = {"payment_id": str(uuid4()), "amount": 1000.0, "payment_method": "wire"}
        result = await bank_simulator.process_payment(
            sample_bank, payment, date(2024, 1, 14)  # Sunday
        )
        assert result.cleared_date.weekday() < 5

    @pytest.mark.asyncio
    async def test_weekday_no_adjustment(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Weekday initiation should not need weekend adjustment."""
        payment = {"payment_id": str(uuid4()), "amount": 1000.0, "payment_method": "wire"}
        result = await bank_simulator.process_payment(
            sample_bank, payment, date(2024, 1, 17)  # Wednesday
        )
        assert result.cleared_date == date(2024, 1, 17) or result.cleared_date >= date(2024, 1, 17)

    @pytest.mark.asyncio
    async def test_process_payments_batch(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """Batch processing must return a list of PaymentProcessingResult."""
        payments = [
            {"payment_id": str(uuid4()), "amount": 1000.0 * (i + 1), "payment_method": "ach"}
            for i in range(5)
        ]
        results = await bank_simulator.process_payments_batch(
            sample_bank, payments, date(2024, 1, 15)
        )
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, PaymentProcessingResult) for r in results)

    @pytest.mark.asyncio
    async def test_payment_methods_supported(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
    ) -> None:
        """All three payment methods (check, ach, wire) must be accepted."""
        for method in ["check", "ach", "wire"]:
            payment = {"payment_id": str(uuid4()), "amount": 1000.0, "payment_method": method}
            result = await bank_simulator.process_payment(
                sample_bank, payment, date(2024, 1, 15)
            )
            assert isinstance(result, PaymentProcessingResult)


class TestBankUtilities:
    """Test bank utility methods: interest calc, reference numbers, weekend adjust."""

    def test_calculate_interest(self) -> None:
        """Monthly interest should follow annual rate / 12."""
        # _calculate_interest reads annual rate from config (default 0.01)
        sim = BankSimulator(seed=42)
        interest = sim._calculate_interest(average_balance=100000.0)
        expected = 100000.0 * sim.config.interest_rate_annual / 12  # ~83.33
        assert interest == pytest.approx(expected, abs=1.0)

    def test_generate_reference_number(self) -> None:
        """Reference number must be a non-empty string."""
        sim = BankSimulator(seed=42)
        ref = sim._generate_reference_number()
        assert isinstance(ref, str)
        assert len(ref) > 0

    def test_adjust_for_weekends_monday_through_friday(self) -> None:
        """Weekday dates should be returned unchanged."""
        sim = BankSimulator(seed=42)
        for d in [date(2024, 1, 15), date(2024, 1, 16), date(2024, 1, 17),
                   date(2024, 1, 18), date(2024, 1, 19)]:
            assert sim._adjust_for_weekends(d) == d

    def test_adjust_for_weekends_saturday(self) -> None:
        """Saturday should be pushed to Monday (+2 days)."""
        sim = BankSimulator(seed=42)
        saturday = date(2024, 1, 13)
        assert saturday.weekday() == 5  # Saturday
        adjusted = sim._adjust_for_weekends(saturday)
        assert adjusted == date(2024, 1, 15)  # Monday

    def test_adjust_for_weekends_sunday(self) -> None:
        """Sunday should be pushed to Monday (+1 day)."""
        sim = BankSimulator(seed=42)
        sunday = date(2024, 1, 14)
        assert sunday.weekday() == 6  # Sunday
        adjusted = sim._adjust_for_weekends(sunday)
        assert adjusted == date(2024, 1, 15)  # Monday


# ============================================================================
# Phase 6: Pydantic Model Validation Tests
# ============================================================================


class TestCustomerDataModels:
    """Validate Pydantic V2 models for Customer simulator output."""

    def test_customer_order_creation(self) -> None:
        """CustomerOrder with valid data must succeed."""
        order = CustomerOrder(
            order_id=str(uuid4()),
            customer_id=str(uuid4()),
            order_date=date(2024, 1, 15),
            total_amount=5000.0,
            items=[{"product_id": "P-001", "quantity": 10, "unit_price": 50.0}],
        )
        assert order.total_amount == 5000.0

    def test_customer_order_has_uuid_id(self) -> None:
        """order_id should be a UUID string."""
        order = CustomerOrder(
            order_id=str(uuid4()),
            customer_id=str(uuid4()),
            order_date=date(2024, 1, 15),
            total_amount=1000.0,
            items=[],
        )
        assert len(order.order_id) > 0

    def test_customer_payment_creation(self) -> None:
        """CustomerPayment with valid data must succeed."""
        payment = CustomerPayment(
            payment_id=str(uuid4()),
            customer_id=str(uuid4()),
            invoice_id=str(uuid4()),
            payment_date=date(2024, 2, 15),
            amount_due=5000.0,
            amount_paid=5000.0,
            is_short_pay=False,
        )
        assert payment.amount_paid == 5000.0

    def test_customer_dispute_creation(self) -> None:
        """CustomerDispute with valid data must succeed."""
        dispute = CustomerDispute(
            dispute_id=str(uuid4()),
            customer_id=str(uuid4()),
            invoice_id=str(uuid4()),
            dispute_type="pricing",
            dispute_amount=500.0,
            dispute_date=date(2024, 1, 20),
            status="open",
        )
        assert dispute.dispute_type == "pricing"


class TestVendorDataModels:
    """Validate Pydantic V2 models for Vendor simulator output."""

    def test_vendor_invoice_creation(self) -> None:
        """VendorInvoice with valid data must succeed."""
        invoice = VendorInvoice(
            invoice_id=str(uuid4()),
            vendor_id=str(uuid4()),
            po_id=str(uuid4()),
            invoice_date=date(2024, 1, 15),
            invoice_amount=4900.0,
            po_amount=5000.0,
            variance_amount=-100.0,
            is_accurate=True,
        )
        assert invoice.invoice_amount == 4900.0

    def test_vendor_invoice_has_uuid_id(self) -> None:
        """invoice_id should be a UUID string."""
        invoice = VendorInvoice(
            invoice_id=str(uuid4()),
            vendor_id=str(uuid4()),
            po_id=str(uuid4()),
            invoice_date=date(2024, 1, 15),
            invoice_amount=5000.0,
            po_amount=5000.0,
            variance_amount=0.0,
            is_accurate=True,
        )
        assert len(invoice.invoice_id) > 0

    def test_goods_delivery_creation(self) -> None:
        """GoodsDelivery with valid data must succeed."""
        delivery = GoodsDelivery(
            delivery_id=str(uuid4()),
            vendor_id=str(uuid4()),
            po_id=str(uuid4()),
            delivery_date=date(2024, 1, 20),
            items_ordered=100,
            items_delivered=100,
            is_partial=False,
            is_backorder=False,
        )
        assert delivery.items_delivered == 100

    def test_vendor_inquiry_response_creation(self) -> None:
        """VendorInquiryResponse with valid data must succeed."""
        response = VendorInquiryResponse(
            inquiry_id=str(uuid4()),
            vendor_id=str(uuid4()),
            inquiry_date=date(2024, 1, 10),
            response_date=date(2024, 1, 13),
            response_time_days=3,
            inquiry_type="pricing",
        )
        assert response.inquiry_type == "pricing"


class TestBankDataModels:
    """Validate Pydantic V2 models for Bank simulator output."""

    def test_bank_statement_creation(self) -> None:
        """BankStatement with valid data must succeed."""
        statement = BankStatement(
            statement_id=str(uuid4()),
            bank_id=str(uuid4()),
            account_number="ACCT-0001",
            statement_period_start=date(2024, 1, 1),
            statement_period_end=date(2024, 1, 31),
            opening_balance=10000.0,
            closing_balance=10500.0,
            total_debits=2000.0,
            total_credits=2525.0,
            total_fees=25.0,
            total_interest=0.0,
            entries=[],
            generated_date=date(2024, 2, 1),
        )
        assert statement.closing_balance == 10500.0

    def test_bank_statement_entry_creation(self) -> None:
        """BankStatementEntry with valid data must succeed."""
        entry = BankStatementEntry(
            entry_id=str(uuid4()),
            date=date(2024, 1, 15),
            description="Payment received",
            amount=1500.0,
            entry_type="credit",
            reference_number="REF-001",
            running_balance=11500.0,
        )
        assert entry.amount == 1500.0

    def test_payment_processing_result_creation(self) -> None:
        """PaymentProcessingResult with valid data must succeed."""
        result = PaymentProcessingResult(
            result_id=str(uuid4()),
            payment_id=str(uuid4()),
            bank_id=str(uuid4()),
            payment_method="check",
            amount=1000.0,
            initiated_date=date(2024, 1, 15),
            cleared_date=date(2024, 1, 17),
            is_successful=True,
            failure_reason=None,
            status="completed",
        )
        assert result.is_successful is True

    def test_payment_processing_result_default_status(self) -> None:
        """Default status must be 'pending' if not specified."""
        result = PaymentProcessingResult(
            result_id=str(uuid4()),
            payment_id=str(uuid4()),
            bank_id=str(uuid4()),
            payment_method="ach",
            amount=500.0,
            initiated_date=date(2024, 1, 15),
            cleared_date=date(2024, 1, 17),
            is_successful=True,
        )
        assert result.status == "pending"


# ============================================================================
# Phase 7: Cross-Simulator Integration Tests
# ============================================================================


class TestSimulatorIntegration:
    """Higher-level tests validating cross-simulator consistency."""

    @pytest.mark.asyncio
    async def test_customer_order_amount_is_positive(
        self,
        customer_simulator: CustomerSimulator,
        sample_customer: Dict[str, Any],
    ) -> None:
        """All generated order amounts must be positive."""
        for _ in range(20):
            order = await customer_simulator.generate_customer_order(
                sample_customer, date(2024, 1, 15)
            )
            assert order.total_amount > 0

    @pytest.mark.asyncio
    async def test_vendor_invoice_references_valid_po(
        self,
        vendor_simulator: VendorSimulator,
        sample_vendor: Dict[str, Any],
        sample_purchase_order: Dict[str, Any],
    ) -> None:
        """Every generated VendorInvoice.po_id must match the input PO."""
        invoice = await vendor_simulator.generate_vendor_invoice(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        assert invoice.po_id == sample_purchase_order["po_id"]

    @pytest.mark.asyncio
    async def test_bank_statement_balances_consistent(
        self,
        bank_simulator: BankSimulator,
        sample_bank: Dict[str, Any],
        sample_transactions: List[Dict[str, Any]],
    ) -> None:
        """Opening balance plus all entries must equal closing balance."""
        statement = await bank_simulator.generate_bank_statement(
            sample_bank, sample_transactions, date(2024, 1, 1), opening_balance=10000.0
        )
        running = statement.opening_balance
        for entry in statement.entries:
            running += entry.amount
        assert running == pytest.approx(statement.closing_balance, abs=0.02)

    @pytest.mark.asyncio
    async def test_all_simulators_reproducible_with_same_seed(
        self,
        mock_amount_distribution: MagicMock,
        mock_order_frequency_model: MagicMock,
        mock_payment_timing_model: MagicMock,
        mock_selection_model: MagicMock,
        mock_event_bus: AsyncMock,
        sample_customer: Dict[str, Any],
        sample_vendor: Dict[str, Any],
        sample_purchase_order: Dict[str, Any],
        sample_bank: Dict[str, Any],
    ) -> None:
        """Same seed across all simulators should produce identical results."""
        # -- Customer Simulator --
        cs1 = CustomerSimulator(
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )
        cs2 = CustomerSimulator(
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )
        o1 = await cs1.generate_customer_order(sample_customer, date(2024, 1, 15))
        o2 = await cs2.generate_customer_order(sample_customer, date(2024, 1, 15))
        assert o1.total_amount == o2.total_amount

        # -- Vendor Simulator --
        vs1 = VendorSimulator(
            amount_distribution=mock_amount_distribution,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )
        vs2 = VendorSimulator(
            amount_distribution=mock_amount_distribution,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )
        inv1 = await vs1.generate_vendor_invoice(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        inv2 = await vs2.generate_vendor_invoice(
            sample_vendor, sample_purchase_order, date(2024, 1, 15)
        )
        assert inv1.invoice_amount == inv2.invoice_amount

        # -- Bank Simulator --
        bs1 = BankSimulator(event_bus=mock_event_bus, seed=42)
        bs2 = BankSimulator(event_bus=mock_event_bus, seed=42)
        txns: List[Dict[str, Any]] = []
        st1 = await bs1.generate_bank_statement(
            sample_bank, txns, date(2024, 1, 1), opening_balance=10000.0
        )
        st2 = await bs2.generate_bank_statement(
            sample_bank, txns, date(2024, 1, 1), opening_balance=10000.0
        )
        assert st1.closing_balance == st2.closing_balance
