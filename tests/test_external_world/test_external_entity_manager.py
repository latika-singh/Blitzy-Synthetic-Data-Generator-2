"""
Comprehensive tests for ExternalWorldManager.

Covers:
- ExternalWorldConfig Pydantic model validation
- Entity pool initialization and management
- Tier assignment distribution (Strategic 10%, Standard 30%, Transactional 60%)
- Behavior profile assignment with tier-weighted probabilities
- Interaction generation delegation to sub-simulators
- Project 1 REST API integration (mocked with aioresponses)
- Metrics collection and reporting
- Edge cases and error handling

All async tests use pytest-asyncio.  No real HTTP or Redis calls — everything
is mocked.  Random generators are seeded (seed=42) for reproducibility.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

import aiohttp
import numpy as np
import pytest
import pytest_asyncio
from aioresponses import aioresponses
from pydantic import ValidationError

from app.events.event_bus import EventBus
from app.external_world.behavior_profiles import (
    BehaviorProfile,
    BehaviorProfileFactory,
    PROFILE_NAMES,
    TierDistribution,
)
from app.external_world.external_entity_manager import (
    ExternalWorldConfig,
    ExternalWorldManager,
    TIER_DISTRIBUTION,
    TIER_TO_PROFILE_WEIGHTS,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_customers() -> List[Dict[str, Any]]:
    """10 customer dicts ranked by revenue (descending)."""
    return [
        {
            "customer_id": str(uuid4()),
            "name": f"Customer {i}",
            "revenue": (10 - i) * 100_000,
            "payment_terms": "Net 30",
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_vendors() -> List[Dict[str, Any]]:
    """10 vendor dicts ranked by spend_history (descending)."""
    return [
        {
            "vendor_id": str(uuid4()),
            "name": f"Vendor {i}",
            "spend_history": (10 - i) * 50_000,
            "category": "supplies",
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_banks() -> List[Dict[str, Any]]:
    """3 bank dicts."""
    return [
        {
            "bank_id": str(uuid4()),
            "name": f"Bank {i}",
            "account_number": f"ACCT-{i:04d}",
        }
        for i in range(3)
    ]


@pytest.fixture
def sample_carriers() -> List[Dict[str, Any]]:
    """5 carrier dicts."""
    return [
        {
            "carrier_id": str(uuid4()),
            "name": f"Carrier {i}",
            "service_type": "ground" if i < 3 else "express",
        }
        for i in range(5)
    ]


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Mocked EventBus with publish and subscribe methods."""
    bus = AsyncMock(spec=EventBus)
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def mock_amount_distribution() -> MagicMock:
    """Mocked AmountDistribution with representative return values."""
    mock = MagicMock()
    mock.sample_customer_order_amount = MagicMock(return_value=5000.0)
    mock.sample_vendor_invoice_amount = MagicMock(return_value=4900.0)
    mock.sample = MagicMock(return_value=5000.0)
    return mock


@pytest.fixture
def mock_order_frequency_model() -> MagicMock:
    """Mocked OrderFrequencyModel returning a fixed order count."""
    mock = MagicMock()
    mock.sample_order_count = MagicMock(return_value=5)
    mock.sample = MagicMock(return_value=5)
    return mock


@pytest.fixture
def mock_payment_timing_model() -> MagicMock:
    """Mocked PaymentTimingModel returning a fixed date."""
    mock = MagicMock()
    mock.get_payment_date = MagicMock(return_value=date(2024, 2, 15))
    mock.sample = MagicMock(return_value=30)
    return mock


@pytest.fixture
def mock_selection_model() -> MagicMock:
    """Mocked SelectionModel with representative return values."""
    mock = MagicMock()
    mock.select_vendor = MagicMock(
        return_value={"vendor_id": "V-001", "name": "Selected Vendor"}
    )
    mock.select_customers_batch = MagicMock(return_value=[])
    mock.select = MagicMock(return_value=0)
    return mock


@pytest.fixture
def manager(
    mock_event_bus: AsyncMock,
    mock_amount_distribution: MagicMock,
    mock_order_frequency_model: MagicMock,
    mock_payment_timing_model: MagicMock,
    mock_selection_model: MagicMock,
) -> ExternalWorldManager:
    """ExternalWorldManager with fully-mocked dependencies and seed=42."""
    return ExternalWorldManager(
        config=ExternalWorldConfig(),
        amount_distribution=mock_amount_distribution,
        order_frequency_model=mock_order_frequency_model,
        payment_timing_model=mock_payment_timing_model,
        selection_model=mock_selection_model,
        event_bus=mock_event_bus,
        seed=42,
    )


@pytest.fixture
def manager_no_deps() -> ExternalWorldManager:
    """ExternalWorldManager with no injected dependencies — graceful degradation."""
    return ExternalWorldManager(seed=42)


# ============================================================================
# Phase 3 — ExternalWorldConfig Tests
# ============================================================================


class TestExternalWorldConfig:
    """Tests for the Pydantic V2 ExternalWorldConfig model."""

    def test_default_config(self) -> None:
        """Default constructor produces correct default values."""
        config = ExternalWorldConfig()

        assert config.tier_distribution == TIER_DISTRIBUTION
        assert config.enable_api_integration is False
        assert config.api_timeout_seconds == 10.0
        assert config.max_retries == 3
        assert config.project1_api_base_url is None

    def test_custom_config(self) -> None:
        """Custom values are stored correctly."""
        custom = ExternalWorldConfig(
            tier_distribution={"strategic": 0.20, "standard": 0.30, "transactional": 0.50},
            enable_api_integration=True,
            api_timeout_seconds=5.0,
            max_retries=1,
            project1_api_base_url="http://localhost:8000",
        )

        assert custom.tier_distribution["strategic"] == 0.20
        assert custom.tier_distribution["standard"] == 0.30
        assert custom.tier_distribution["transactional"] == 0.50
        assert custom.enable_api_integration is True
        assert custom.api_timeout_seconds == 5.0
        assert custom.max_retries == 1
        assert custom.project1_api_base_url == "http://localhost:8000"

    def test_tier_distribution_validation(self) -> None:
        """Config with invalid tier distribution (not summing to 1.0) raises ValidationError."""
        with pytest.raises(ValidationError):
            ExternalWorldConfig(
                tier_distribution={"strategic": 0.50, "standard": 0.30, "transactional": 0.60}
            )

    def test_tier_distribution_exact_sum(self) -> None:
        """Config with tier distribution summing to exactly 1.0 succeeds."""
        config = ExternalWorldConfig(
            tier_distribution={"strategic": 0.15, "standard": 0.35, "transactional": 0.50}
        )
        assert abs(sum(config.tier_distribution.values()) - 1.0) < 0.01


# ============================================================================
# Phase 4 — Entity Pool Management Tests
# ============================================================================


class TestEntityPoolManagement:
    """Tests for entity pool initialization and query methods."""

    def test_initialize_empty_pools(self, manager: ExternalWorldManager) -> None:
        """Before initialize_entity_pools, all pools are empty."""
        assert manager.customers == []
        assert manager.vendors == []
        assert manager.banks == []
        assert manager.carriers == []

    @pytest.mark.asyncio
    async def test_initialize_with_customers(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Customers are loaded into the pool after initialization."""
        await manager.initialize_entity_pools(customers=sample_customers)
        assert len(manager.customers) == 10

    @pytest.mark.asyncio
    async def test_initialize_with_all_entity_types(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
        sample_vendors: List[Dict[str, Any]],
        sample_banks: List[Dict[str, Any]],
        sample_carriers: List[Dict[str, Any]],
    ) -> None:
        """All four entity types are populated after initialization."""
        await manager.initialize_entity_pools(
            customers=sample_customers,
            vendors=sample_vendors,
            banks=sample_banks,
            carriers=sample_carriers,
        )
        assert len(manager.customers) == 10
        assert len(manager.vendors) == 10
        assert len(manager.banks) == 3
        assert len(manager.carriers) == 5

    @pytest.mark.asyncio
    async def test_entity_count(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """get_entity_count returns correct per-type counts."""
        await manager.initialize_entity_pools(
            customers=sample_customers, vendors=sample_vendors
        )
        counts = manager.get_entity_count()
        assert counts["customer"] == 10
        assert counts["vendor"] == 10
        assert counts["bank"] == 0
        assert counts["carrier"] == 0

    @pytest.mark.asyncio
    async def test_entity_count_specific_type(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """get_entity_count with a specific type returns only that count."""
        await manager.initialize_entity_pools(customers=sample_customers)
        counts = manager.get_entity_count("customer")
        assert counts == {"customer": 10}


# ============================================================================
# Phase 5 — Tier Assignment Tests (CRITICAL — 10/30/60 distribution)
# ============================================================================


class TestTierAssignment:
    """Tests verifying TIER_DISTRIBUTION: Strategic 10%, Standard 30%, Transactional 60%."""

    def test_tier_distribution_constant(self) -> None:
        """TIER_DISTRIBUTION constant has the specification values."""
        assert TIER_DISTRIBUTION["strategic"] == pytest.approx(0.10)
        assert TIER_DISTRIBUTION["standard"] == pytest.approx(0.30)
        assert TIER_DISTRIBUTION["transactional"] == pytest.approx(0.60)

    def test_tier_to_profile_weights_has_all_tiers(self) -> None:
        """TIER_TO_PROFILE_WEIGHTS defines weights for all three tiers."""
        assert "strategic" in TIER_TO_PROFILE_WEIGHTS
        assert "standard" in TIER_TO_PROFILE_WEIGHTS
        assert "transactional" in TIER_TO_PROFILE_WEIGHTS

    def test_tier_assignment_distribution_10_entities(
        self, manager: ExternalWorldManager
    ) -> None:
        """With 10 entities: 1 strategic, 3 standard, 6 transactional."""
        entities = [
            {"customer_id": str(uuid4()), "revenue": (10 - i) * 100_000}
            for i in range(10)
        ]
        result = manager.assign_entity_tier(entities)

        tier_counts = _count_tiers(result)
        assert tier_counts["strategic"] == 1
        assert tier_counts["standard"] == 3
        assert tier_counts["transactional"] == 6

    def test_tier_assignment_distribution_20_entities(
        self, manager: ExternalWorldManager
    ) -> None:
        """With 20 entities: 2 strategic, 6 standard, 12 transactional."""
        entities = [
            {"customer_id": str(uuid4()), "revenue": (20 - i) * 50_000}
            for i in range(20)
        ]
        result = manager.assign_entity_tier(entities)

        tier_counts = _count_tiers(result)
        assert tier_counts["strategic"] == 2
        assert tier_counts["standard"] == 6
        assert tier_counts["transactional"] == 12

    def test_tier_assignment_distribution_100_entities(
        self, manager: ExternalWorldManager
    ) -> None:
        """With 100 entities: 10 strategic, 30 standard, 60 transactional."""
        entities = [
            {"customer_id": str(uuid4()), "revenue": (100 - i) * 10_000}
            for i in range(100)
        ]
        result = manager.assign_entity_tier(entities)

        tier_counts = _count_tiers(result)
        assert tier_counts["strategic"] == 10
        assert tier_counts["standard"] == 30
        assert tier_counts["transactional"] == 60

    def test_tier_assignment_adds_tier_field(
        self, manager: ExternalWorldManager
    ) -> None:
        """Each entity dict receives a 'tier' key after assignment."""
        entities = [
            {"customer_id": str(uuid4()), "revenue": 100_000},
            {"customer_id": str(uuid4()), "revenue": 50_000},
        ]
        result = manager.assign_entity_tier(entities)
        for entity in result:
            assert "tier" in entity
            assert entity["tier"] in ("strategic", "standard", "transactional")

    def test_tier_assignment_ranked_by_revenue(
        self, manager: ExternalWorldManager
    ) -> None:
        """Highest-revenue entity gets strategic, lowest gets transactional."""
        entities = [
            {"customer_id": "low", "revenue": 10_000},
            {"customer_id": "high", "revenue": 1_000_000},
            {"customer_id": "mid", "revenue": 100_000},
        ]
        result = manager.assign_entity_tier(entities)

        # Sorted descending: high (rank 0) → strategic
        high_entity = next(e for e in result if e["customer_id"] == "high")
        assert high_entity["tier"] == "strategic"

        # Low entity (rank 2/3 = 0.667) → transactional
        low_entity = next(e for e in result if e["customer_id"] == "low")
        assert low_entity["tier"] == "transactional"

    def test_tier_assignment_ranked_by_spend_history(
        self, manager: ExternalWorldManager
    ) -> None:
        """Vendors are ranked by spend_history for tier assignment."""
        entities = [
            {"vendor_id": str(uuid4()), "spend_history": 500_000},
            {"vendor_id": str(uuid4()), "spend_history": 10_000},
        ]
        result = manager.assign_entity_tier(entities)

        # First entity (highest spend) should be ranked higher
        high_spend = next(e for e in result if e["spend_history"] == 500_000)
        low_spend = next(e for e in result if e["spend_history"] == 10_000)
        assert high_spend["tier"] == "strategic"
        assert low_spend["tier"] == "transactional"

    @pytest.mark.asyncio
    async def test_get_entities_by_tier(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """get_entities_by_tier returns correct filtered lists for 10 customers."""
        await manager.initialize_entity_pools(customers=sample_customers)

        strategic = manager.get_entities_by_tier("customer", "strategic")
        standard = manager.get_entities_by_tier("customer", "standard")
        transactional = manager.get_entities_by_tier("customer", "transactional")

        assert len(strategic) == 1   # 10% of 10
        assert len(standard) == 3    # 30% of 10
        assert len(transactional) == 6  # 60% of 10

    @pytest.mark.asyncio
    async def test_get_tier_distribution(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """get_tier_distribution returns per-type tier counts."""
        await manager.initialize_entity_pools(
            customers=sample_customers, vendors=sample_vendors
        )
        dist = manager.get_tier_distribution()

        assert dist["customer"]["strategic"] == 1
        assert dist["customer"]["standard"] == 3
        assert dist["customer"]["transactional"] == 6

        assert dist["vendor"]["strategic"] == 1
        assert dist["vendor"]["standard"] == 3
        assert dist["vendor"]["transactional"] == 6

    def test_tier_assignment_empty_list(
        self, manager: ExternalWorldManager
    ) -> None:
        """Empty entity list returns empty list without errors."""
        result = manager.assign_entity_tier([])
        assert result == []


# ============================================================================
# Phase 6 — Behavior Profile Assignment Tests
# ============================================================================


class TestBehaviorProfileAssignment:
    """Tests for tier-weighted behavior profile assignment."""

    @pytest.mark.asyncio
    async def test_profile_assigned_to_all_entities(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Every customer has a 'behavior_profile' key after initialization."""
        await manager.initialize_entity_pools(customers=sample_customers)
        for customer in manager.customers:
            assert "behavior_profile" in customer

    @pytest.mark.asyncio
    async def test_profile_is_behavior_profile_instance(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Each assigned profile is a BehaviorProfile instance."""
        await manager.initialize_entity_pools(customers=sample_customers)
        for customer in manager.customers:
            profile = customer["behavior_profile"]
            assert isinstance(profile, BehaviorProfile)

    @pytest.mark.asyncio
    async def test_profile_has_required_attributes(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Each profile has profile_name, entity_type, payment/order/invoice behavior."""
        await manager.initialize_entity_pools(customers=sample_customers)
        for customer in manager.customers:
            profile: BehaviorProfile = customer["behavior_profile"]
            assert profile.profile_name in PROFILE_NAMES
            assert profile.entity_type in ("customer", "vendor", "bank", "carrier")
            assert profile.payment_behavior is not None
            assert profile.order_behavior is not None
            assert profile.invoice_behavior is not None

    def test_strategic_tier_favors_excellent_profiles(self) -> None:
        """Strategic-tier entities get excellent/good profiles > 50% of the time."""
        mgr = ExternalWorldManager(seed=42)

        # Create 200 strategic-tier entities for statistical significance
        entities = [
            {"customer_id": str(uuid4()), "revenue": 1_000_000, "tier": "strategic"}
            for _ in range(200)
        ]
        profile_names: List[str] = []
        for entity in entities:
            result = mgr.assign_behavior_profile(entity)
            profile_names.append(result["profile_name"])

        # Per TIER_TO_PROFILE_WEIGHTS strategic: excellent=0.50, good=0.35
        excellent_good = sum(
            1 for p in profile_names if p in ("excellent", "good")
        )
        ratio = excellent_good / len(profile_names)
        assert ratio > 0.50, (
            f"Expected strategic tier to have > 50% excellent/good, got {ratio:.2%}"
        )

    def test_transactional_tier_favors_poor_profiles(self) -> None:
        """Transactional-tier entities get poor/problem profiles > 25% of the time."""
        mgr = ExternalWorldManager(seed=42)

        entities = [
            {"customer_id": str(uuid4()), "revenue": 10_000, "tier": "transactional"}
            for _ in range(200)
        ]
        profile_names: List[str] = []
        for entity in entities:
            result = mgr.assign_behavior_profile(entity)
            profile_names.append(result["profile_name"])

        # Per TIER_TO_PROFILE_WEIGHTS transactional: poor=0.30, problem=0.15
        poor_problem = sum(
            1 for p in profile_names if p in ("poor", "problem")
        )
        ratio = poor_problem / len(profile_names)
        assert ratio > 0.25, (
            f"Expected transactional tier to have > 25% poor/problem, got {ratio:.2%}"
        )

    def test_profile_assignment_deterministic_with_seed(self) -> None:
        """Two managers with seed=42 produce identical profile assignments."""
        entities_a = [
            {"customer_id": f"C-{i}", "revenue": (10 - i) * 100_000, "tier": "standard"}
            for i in range(10)
        ]
        entities_b = [dict(e) for e in entities_a]  # deep copy

        mgr1 = ExternalWorldManager(seed=42)
        mgr2 = ExternalWorldManager(seed=42)

        profiles_a = [mgr1.assign_behavior_profile(e)["profile_name"] for e in entities_a]
        profiles_b = [mgr2.assign_behavior_profile(e)["profile_name"] for e in entities_b]

        assert profiles_a == profiles_b

    def test_assign_behavior_profile_single_entity(
        self, manager: ExternalWorldManager
    ) -> None:
        """assign_behavior_profile adds profile to a single entity dict."""
        entity: Dict[str, Any] = {
            "customer_id": str(uuid4()),
            "revenue": 50_000,
            "tier": "standard",
        }
        result = manager.assign_behavior_profile(entity)

        assert "behavior_profile" in result
        assert isinstance(result["behavior_profile"], BehaviorProfile)
        assert "profile_name" in result
        assert result["profile_name"] in PROFILE_NAMES

    def test_profile_names_valid(self) -> None:
        """All assigned profile names belong to PROFILE_NAMES."""
        mgr = ExternalWorldManager(seed=42)
        for tier in ("strategic", "standard", "transactional"):
            entity = {"customer_id": str(uuid4()), "revenue": 100_000, "tier": tier}
            result = mgr.assign_behavior_profile(entity)
            assert result["profile_name"] in PROFILE_NAMES

    def test_profile_factory_available_profiles(self) -> None:
        """BehaviorProfileFactory.get_available_profiles returns non-empty mapping."""
        available = BehaviorProfileFactory.get_available_profiles()
        assert isinstance(available, dict)
        assert len(available) > 0

    def test_tier_distribution_model_get_tier_for_rank(self) -> None:
        """TierDistribution.get_tier_for_rank maps percentiles correctly."""
        td = TierDistribution(strategic=0.10, standard=0.30, transactional=0.60)

        assert td.get_tier_for_rank(0.0) == "strategic"
        assert td.get_tier_for_rank(0.05) == "strategic"
        assert td.get_tier_for_rank(0.10) == "standard"
        assert td.get_tier_for_rank(0.39) == "standard"
        assert td.get_tier_for_rank(0.40) == "transactional"
        assert td.get_tier_for_rank(0.99) == "transactional"


# ============================================================================
# Phase 7 — Interaction Generation Delegation Tests
# ============================================================================


class TestInteractionGeneration:
    """Tests verifying delegation to sub-simulators."""

    @pytest.mark.asyncio
    async def test_generate_customer_order_delegates(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """generate_customer_order delegates to _customer_simulator."""
        await manager.initialize_entity_pools(customers=sample_customers)

        mock_order = {"order_id": "ORD-001", "amount": 5000.0}
        manager._customer_simulator.generate_customer_order = AsyncMock(
            return_value=mock_order
        )

        customer = manager.customers[0]
        result = await manager.generate_customer_order(
            customer=customer, simulation_date=date(2024, 1, 15)
        )

        manager._customer_simulator.generate_customer_order.assert_awaited_once()
        assert result == mock_order

    @pytest.mark.asyncio
    async def test_generate_vendor_invoice_delegates(
        self,
        manager: ExternalWorldManager,
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """generate_vendor_invoice delegates to _vendor_simulator."""
        await manager.initialize_entity_pools(vendors=sample_vendors)

        mock_invoice = {"invoice_id": "INV-001", "amount": 4900.0}
        manager._vendor_simulator.generate_vendor_invoice = AsyncMock(
            return_value=mock_invoice
        )

        vendor = manager.vendors[0]
        po = {"po_id": "PO-001", "amount": 5000.0}
        result = await manager.generate_vendor_invoice(
            vendor=vendor, po=po, simulation_date=date(2024, 1, 15)
        )

        manager._vendor_simulator.generate_vendor_invoice.assert_awaited_once()
        assert result == mock_invoice

    @pytest.mark.asyncio
    async def test_generate_bank_statement_delegates(
        self,
        manager: ExternalWorldManager,
        sample_banks: List[Dict[str, Any]],
    ) -> None:
        """generate_bank_statement delegates to _bank_simulator."""
        await manager.initialize_entity_pools(banks=sample_banks)

        mock_statement = {"statement_id": "BS-001", "balance": 50_000.0}
        manager._bank_simulator.generate_bank_statement = AsyncMock(
            return_value=mock_statement
        )

        bank = manager.banks[0]
        result = await manager.generate_bank_statement(
            bank=bank,
            transactions=[],
            statement_month=date(2024, 1, 1),
            opening_balance=0.0,
        )

        manager._bank_simulator.generate_bank_statement.assert_awaited_once()
        assert result == mock_statement

    @pytest.mark.asyncio
    async def test_generate_customer_order_increments_metrics(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Each customer order generation increments the metric counter."""
        await manager.initialize_entity_pools(customers=sample_customers)
        manager._customer_simulator.generate_customer_order = AsyncMock(
            return_value={"order_id": "ORD-X"}
        )

        initial_count = manager._metrics["total_customer_orders"]
        await manager.generate_customer_order(
            customer=manager.customers[0], simulation_date=date(2024, 1, 15)
        )
        assert manager._metrics["total_customer_orders"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_generate_daily_interactions(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """generate_daily_interactions orchestrates all sub-simulators."""
        await manager.initialize_entity_pools(
            customers=sample_customers, vendors=sample_vendors
        )

        # Mock all daily methods on sub-simulators
        manager._customer_simulator.generate_daily_orders = AsyncMock(
            return_value=[{"order_id": "ORD-DAILY"}]
        )
        manager._customer_simulator.simulate_daily_payments = AsyncMock(
            return_value=[{"payment_id": "PMT-001"}]
        )
        manager._vendor_simulator.generate_batch_invoices = AsyncMock(
            return_value=[{"invoice_id": "INV-DAILY"}]
        )

        result = await manager.generate_daily_interactions(
            simulation_date=date(2024, 1, 15)
        )

        # Verify return structure
        assert "date" in result
        assert "customer_orders" in result
        assert "customer_payments" in result
        assert "vendor_invoices" in result

        # Verify data was populated
        assert len(result["customer_orders"]) == 1
        assert result["customer_orders"][0]["order_id"] == "ORD-DAILY"

    @pytest.mark.asyncio
    async def test_generate_daily_interactions_returns_dict_structure(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Result of generate_daily_interactions is a dict with list values."""
        await manager.initialize_entity_pools(customers=sample_customers)

        manager._customer_simulator.generate_daily_orders = AsyncMock(return_value=[])
        manager._customer_simulator.simulate_daily_payments = AsyncMock(return_value=[])
        manager._vendor_simulator.generate_batch_invoices = AsyncMock(return_value=[])

        result = await manager.generate_daily_interactions(
            simulation_date=date(2024, 1, 15)
        )

        assert isinstance(result, dict)
        assert isinstance(result["customer_orders"], list)
        assert isinstance(result["customer_payments"], list)
        assert isinstance(result["vendor_invoices"], list)
        assert result["date"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_generate_daily_interactions_empty_pools(
        self, manager: ExternalWorldManager
    ) -> None:
        """Daily interactions with empty pools produce empty result lists."""
        # No initialization → all pools empty
        manager._vendor_simulator.generate_batch_invoices = AsyncMock(return_value=[])

        result = await manager.generate_daily_interactions(
            simulation_date=date(2024, 1, 15)
        )
        assert result["customer_orders"] == []
        assert result["customer_payments"] == []
        assert result["vendor_invoices"] == []


# ============================================================================
# Phase 8 — Project 1 REST API Integration Tests (Mocked)
# ============================================================================


class TestApiIntegration:
    """Tests for Project 1 REST API integration using mocked HTTP calls."""

    @pytest.mark.asyncio
    async def test_api_integration_disabled_by_default(
        self, manager: ExternalWorldManager
    ) -> None:
        """Default config has enable_api_integration=False — no HTTP calls."""
        assert manager._config.enable_api_integration is False
        # initialize without explicit data — no API calls should be made
        await manager.initialize_entity_pools()
        assert manager.customers == []
        assert manager.vendors == []

    @pytest.mark.asyncio
    async def test_fetch_customers_from_api_when_enabled(
        self,
        mock_event_bus: AsyncMock,
        mock_amount_distribution: MagicMock,
        mock_order_frequency_model: MagicMock,
        mock_payment_timing_model: MagicMock,
        mock_selection_model: MagicMock,
    ) -> None:
        """When API integration is enabled, customers are fetched from the REST API."""
        config = ExternalWorldConfig(
            enable_api_integration=True,
            project1_api_base_url="http://localhost:8000",
            max_retries=1,
        )
        mgr = ExternalWorldManager(
            config=config,
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )

        api_customers = [
            {"customer_id": "C-API-1", "name": "API Customer 1", "revenue": 500_000},
            {"customer_id": "C-API-2", "name": "API Customer 2", "revenue": 200_000},
        ]

        with aioresponses() as mocked:
            mocked.get(
                "http://localhost:8000/api/customers",
                payload=api_customers,
            )
            # Also mock vendors endpoint (called during initialize_entity_pools)
            mocked.get(
                "http://localhost:8000/api/vendors",
                payload=[],
            )
            await mgr.initialize_entity_pools()

        assert len(mgr.customers) == 2
        # Verify customer data was loaded from API
        customer_ids = {c["customer_id"] for c in mgr.customers}
        assert "C-API-1" in customer_ids
        assert "C-API-2" in customer_ids

    @pytest.mark.asyncio
    async def test_fetch_vendors_from_api_when_enabled(
        self,
        mock_event_bus: AsyncMock,
        mock_amount_distribution: MagicMock,
        mock_order_frequency_model: MagicMock,
        mock_payment_timing_model: MagicMock,
        mock_selection_model: MagicMock,
    ) -> None:
        """When API integration is enabled, vendors are fetched from the REST API."""
        config = ExternalWorldConfig(
            enable_api_integration=True,
            project1_api_base_url="http://localhost:8000",
            max_retries=1,
        )
        mgr = ExternalWorldManager(
            config=config,
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )

        api_vendors = [
            {"vendor_id": "V-API-1", "name": "API Vendor 1", "spend_history": 100_000},
        ]

        with aioresponses() as mocked:
            mocked.get("http://localhost:8000/api/customers", payload=[])
            mocked.get("http://localhost:8000/api/vendors", payload=api_vendors)
            await mgr.initialize_entity_pools()

        assert len(mgr.vendors) == 1
        assert mgr.vendors[0]["vendor_id"] == "V-API-1"

    @pytest.mark.asyncio
    async def test_api_failure_graceful_degradation(
        self,
        mock_event_bus: AsyncMock,
        mock_amount_distribution: MagicMock,
        mock_order_frequency_model: MagicMock,
        mock_payment_timing_model: MagicMock,
        mock_selection_model: MagicMock,
    ) -> None:
        """When API is unavailable (ClientError), manager degrades gracefully."""
        config = ExternalWorldConfig(
            enable_api_integration=True,
            project1_api_base_url="http://localhost:8000",
            max_retries=1,
        )
        mgr = ExternalWorldManager(
            config=config,
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )

        with aioresponses() as mocked:
            mocked.get(
                "http://localhost:8000/api/customers",
                exception=aiohttp.ClientError("Connection refused"),
            )
            mocked.get(
                "http://localhost:8000/api/vendors",
                exception=aiohttp.ClientError("Connection refused"),
            )
            # Should NOT raise — graceful degradation
            await mgr.initialize_entity_pools()

        assert mgr.customers == []
        assert mgr.vendors == []

    @pytest.mark.asyncio
    async def test_api_timeout_handled(
        self,
        mock_event_bus: AsyncMock,
        mock_amount_distribution: MagicMock,
        mock_order_frequency_model: MagicMock,
        mock_payment_timing_model: MagicMock,
        mock_selection_model: MagicMock,
    ) -> None:
        """When API times out (TimeoutError), manager handles gracefully."""
        config = ExternalWorldConfig(
            enable_api_integration=True,
            project1_api_base_url="http://localhost:8000",
            max_retries=1,
        )
        mgr = ExternalWorldManager(
            config=config,
            amount_distribution=mock_amount_distribution,
            order_frequency_model=mock_order_frequency_model,
            payment_timing_model=mock_payment_timing_model,
            selection_model=mock_selection_model,
            event_bus=mock_event_bus,
            seed=42,
        )

        with aioresponses() as mocked:
            mocked.get(
                "http://localhost:8000/api/customers",
                exception=asyncio.TimeoutError(),
            )
            mocked.get(
                "http://localhost:8000/api/vendors",
                exception=asyncio.TimeoutError(),
            )
            await mgr.initialize_entity_pools()

        assert mgr.customers == []
        assert mgr.vendors == []


# ============================================================================
# Phase 9 — Metrics and Health Tests
# ============================================================================


class TestMetrics:
    """Tests for operational metrics collection."""

    @pytest.mark.asyncio
    async def test_get_metrics_returns_dict(
        self, manager: ExternalWorldManager
    ) -> None:
        """get_metrics returns a well-structured dictionary."""
        metrics = await manager.get_metrics()
        assert isinstance(metrics, dict)
        assert "entity_counts" in metrics
        assert "tier_distribution" in metrics
        assert "profile_distribution" in metrics
        assert "cumulative_interactions" in metrics
        assert "last_interaction_date" in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_after_initialization(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """After pool initialization, metrics reflect entity counts and tiers."""
        await manager.initialize_entity_pools(
            customers=sample_customers, vendors=sample_vendors
        )
        metrics = await manager.get_metrics()

        # Entity counts
        assert metrics["entity_counts"]["customer"] == 10
        assert metrics["entity_counts"]["vendor"] == 10

        # Tier distribution
        assert metrics["tier_distribution"]["customer"]["strategic"] == 1
        assert metrics["tier_distribution"]["customer"]["standard"] == 3
        assert metrics["tier_distribution"]["customer"]["transactional"] == 6

        # Profile distribution should have at least some entries
        assert len(metrics["profile_distribution"]["customer"]) > 0

        # Cumulative interaction counts start at zero
        assert metrics["cumulative_interactions"]["total_customer_orders"] == 0
        assert metrics["cumulative_interactions"]["total_vendor_invoices"] == 0

    @pytest.mark.asyncio
    async def test_metrics_updated_after_interactions(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """Metrics track cumulative interaction counts after daily interactions."""
        await manager.initialize_entity_pools(
            customers=sample_customers, vendors=sample_vendors
        )

        # Mock simulators for daily interactions
        manager._customer_simulator.generate_daily_orders = AsyncMock(
            return_value=[{"order_id": "O1"}, {"order_id": "O2"}]
        )
        manager._customer_simulator.simulate_daily_payments = AsyncMock(
            return_value=[]
        )
        manager._vendor_simulator.generate_batch_invoices = AsyncMock(
            return_value=[{"invoice_id": "I1"}]
        )

        await manager.generate_daily_interactions(simulation_date=date(2024, 1, 15))

        metrics = await manager.get_metrics()
        assert metrics["cumulative_interactions"]["total_customer_orders"] == 2
        assert metrics["cumulative_interactions"]["total_vendor_invoices"] == 1
        assert metrics["cumulative_interactions"]["total_daily_interactions"] == 1
        assert metrics["last_interaction_date"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_get_metrics_empty_pools(
        self, manager: ExternalWorldManager
    ) -> None:
        """Metrics on empty (pre-init) pools report all zeroes."""
        metrics = await manager.get_metrics()
        assert metrics["entity_counts"]["customer"] == 0
        assert metrics["entity_counts"]["vendor"] == 0
        assert metrics["cumulative_interactions"]["total_customer_orders"] == 0
        assert metrics["last_interaction_date"] is None


# ============================================================================
# Phase 10 — Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_entity_pools(
        self, manager: ExternalWorldManager
    ) -> None:
        """Initializing with empty lists is handled gracefully."""
        await manager.initialize_entity_pools(
            customers=[], vendors=[], banks=[], carriers=[]
        )
        assert manager.customers == []
        assert manager.vendors == []
        assert manager.banks == []
        assert manager.carriers == []

    @pytest.mark.asyncio
    async def test_single_entity_pool(
        self, manager: ExternalWorldManager
    ) -> None:
        """A pool with 1 entity assigns it to 'strategic' (top 10% of 1 = 1)."""
        single_customer = [
            {"customer_id": str(uuid4()), "name": "Only Customer", "revenue": 500_000}
        ]
        await manager.initialize_entity_pools(customers=single_customer)

        assert len(manager.customers) == 1
        # rank_percentile = 0/1 = 0.0 → strategic
        assert manager.customers[0]["tier"] == "strategic"
        # Profile should be assigned
        assert "behavior_profile" in manager.customers[0]

    @pytest.mark.asyncio
    async def test_two_entity_pool(
        self, manager: ExternalWorldManager
    ) -> None:
        """A pool with 2 entities: highest → strategic, lowest → transactional."""
        two_customers = [
            {"customer_id": "C-HIGH", "name": "High Rev", "revenue": 1_000_000},
            {"customer_id": "C-LOW", "name": "Low Rev", "revenue": 10_000},
        ]
        await manager.initialize_entity_pools(customers=two_customers)

        assert len(manager.customers) == 2
        high = next(c for c in manager.customers if c["customer_id"] == "C-HIGH")
        low = next(c for c in manager.customers if c["customer_id"] == "C-LOW")

        # rank 0/2 = 0.0 → strategic, rank 1/2 = 0.5 → transactional
        assert high["tier"] == "strategic"
        assert low["tier"] == "transactional"

    def test_manager_without_dependencies(
        self, manager_no_deps: ExternalWorldManager
    ) -> None:
        """ExternalWorldManager with no dependencies initialises cleanly."""
        assert manager_no_deps.customers == []
        assert manager_no_deps.vendors == []
        assert manager_no_deps.banks == []
        assert manager_no_deps.carriers == []

        # Entity count works on empty pools
        counts = manager_no_deps.get_entity_count()
        assert counts["customer"] == 0

    def test_constructor_injection_stores_simulators(
        self, manager: ExternalWorldManager
    ) -> None:
        """Constructor injection wires sub-simulators on the manager instance."""
        assert manager._customer_simulator is not None
        assert manager._vendor_simulator is not None
        assert manager._bank_simulator is not None

    def test_constructor_injection_stores_event_bus(
        self,
        manager: ExternalWorldManager,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Injected event bus is accessible on the manager instance."""
        assert manager._event_bus is mock_event_bus

    @pytest.mark.asyncio
    async def test_initialize_entity_pools_idempotent(
        self,
        manager: ExternalWorldManager,
        sample_customers: List[Dict[str, Any]],
    ) -> None:
        """Calling initialize_entity_pools twice replaces the pool cleanly."""
        await manager.initialize_entity_pools(customers=sample_customers)
        assert len(manager.customers) == 10

        new_customers = [
            {"customer_id": str(uuid4()), "name": "New Cust", "revenue": 999_999}
        ]
        await manager.initialize_entity_pools(customers=new_customers)
        assert len(manager.customers) == 1

    def test_assign_entity_tier_preserves_existing_fields(
        self, manager: ExternalWorldManager
    ) -> None:
        """Tier assignment does not remove existing entity fields."""
        entities = [
            {
                "customer_id": "C-001",
                "name": "Customer One",
                "revenue": 500_000,
                "custom_field": "preserve_me",
            }
        ]
        result = manager.assign_entity_tier(entities)
        assert result[0]["custom_field"] == "preserve_me"
        assert "tier" in result[0]

    def test_assign_entity_tier_no_ranking_key(
        self, manager: ExternalWorldManager
    ) -> None:
        """Entities without ranking keys get default rank=0.0 and valid tiers."""
        entities = [
            {"customer_id": "C-NO-REV-1"},
            {"customer_id": "C-NO-REV-2"},
            {"customer_id": "C-NO-REV-3"},
        ]
        result = manager.assign_entity_tier(entities)
        for entity in result:
            assert "tier" in entity
            assert entity["tier"] in ("strategic", "standard", "transactional")

    @pytest.mark.asyncio
    async def test_generate_vendor_invoice_increments_metrics(
        self,
        manager: ExternalWorldManager,
        sample_vendors: List[Dict[str, Any]],
    ) -> None:
        """Vendor invoice generation increments the metric counter."""
        await manager.initialize_entity_pools(vendors=sample_vendors)
        manager._vendor_simulator.generate_vendor_invoice = AsyncMock(
            return_value={"invoice_id": "INV-X"}
        )

        initial = manager._metrics["total_vendor_invoices"]
        await manager.generate_vendor_invoice(
            vendor=manager.vendors[0],
            po={"po_id": "PO-1", "amount": 5000},
            simulation_date=date(2024, 1, 15),
        )
        assert manager._metrics["total_vendor_invoices"] == initial + 1

    @pytest.mark.asyncio
    async def test_generate_bank_statement_increments_metrics(
        self,
        manager: ExternalWorldManager,
        sample_banks: List[Dict[str, Any]],
    ) -> None:
        """Bank statement generation increments the metric counter."""
        await manager.initialize_entity_pools(banks=sample_banks)
        manager._bank_simulator.generate_bank_statement = AsyncMock(
            return_value={"statement_id": "BS-X"}
        )

        initial = manager._metrics["total_bank_statements"]
        await manager.generate_bank_statement(
            bank=manager.banks[0],
            transactions=[],
            statement_month=date(2024, 1, 1),
        )
        assert manager._metrics["total_bank_statements"] == initial + 1

    def test_rng_seed_creates_deterministic_state(self) -> None:
        """Two managers with the same seed produce the same numpy RNG state."""
        mgr1 = ExternalWorldManager(seed=42)
        mgr2 = ExternalWorldManager(seed=42)

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        # Verify that RNG outputs match
        assert rng1.rand() == rng2.rand()


# ============================================================================
# Helpers (private — used by tests, not exported)
# ============================================================================


def _count_tiers(entities: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count entities by tier for assertion convenience."""
    counts: Dict[str, int] = {"strategic": 0, "standard": 0, "transactional": 0}
    for entity in entities:
        tier = entity.get("tier", "unknown")
        if tier in counts:
            counts[tier] += 1
    return counts
