"""
External World Manager — Central coordinator for all external entity simulation.

This module implements the ``ExternalWorldManager`` class, the primary entry point
for the ``app.external_world`` subsystem (F-005).  It manages **entity pools**
with a tiered distribution and delegates specific simulation work to
``CustomerSimulator``, ``VendorSimulator``, and ``BankSimulator``.

Tier Distribution (README.md lines 470-474):
    - **Strategic** (10 %): High interaction frequency, excellent/good profiles.
    - **Standard** (30 %): Medium interaction frequency, good/average profiles.
    - **Transactional** (60 %): Low interaction frequency, average/poor/problem profiles.

Entity Pools:
    - ``customers`` — customer entities with payment terms, tier, and profile.
    - ``vendors`` — vendor entities with categories, spend history, and profile.
    - ``banks`` — bank entities for statement generation.
    - ``carriers`` — carrier entities for shipping simulation.

Behavior profiles are drawn from ``BehaviorProfileFactory`` with tier-weighted
probabilities.  Five quality levels are supported: *excellent*, *good*, *average*,
*poor*, and *problem*.

Project 1 REST API Integration (AAP §0.4.1):
    Optional read-only HTTP queries via ``aiohttp`` for master data (customers,
    vendors, products).  Controlled by ``ExternalWorldConfig.enable_api_integration``.

All public methods are ``async`` coroutines with no blocking I/O (AAP §0.7.3).
Constructor injection is used for all dependencies (AAP §0.7.1).  Logging uses
``structlog`` to stdout in structured JSON format (AAP §0.7.6).  Reproducible
seeding is achieved via ``numpy.random.RandomState``.

References:
    - README.md lines 456-486: ExternalWorldManager specification
    - AAP §0.4.1: Project 1 REST API integration (read-only, aiohttp)
    - AAP §0.4.3: Dependency injection wiring
    - AAP §0.5.1 Group 7: External World Simulation module
    - AAP §0.7.1: Constructor injection, Pydantic V2 at subsystem boundaries
    - AAP §0.7.3: Async with no blocking I/O
    - AAP §0.7.6: Structured logging via structlog
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import structlog
from pydantic import BaseModel, Field, field_validator

from app.external_world.behavior_profiles import (
    BehaviorProfile,
    BehaviorProfileFactory,
    TierDistribution,
)
from app.external_world.customer_simulator import CustomerSimulator
from app.external_world.vendor_simulator import VendorSimulator
from app.external_world.bank_simulator import BankSimulator

if TYPE_CHECKING:
    from app.statistical.amount_distributions import AmountDistribution
    from app.statistical.order_frequency_model import OrderFrequencyModel
    from app.statistical.payment_timing_model import PaymentTimingModel
    from app.statistical.selection_models import SelectionModel
    from app.events.event_bus import EventBus

# Optional aiohttp import — degrade gracefully when absent so that the module
# can still be loaded in environments that do not require REST API integration.
try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ===========================================================================
# Constants
# ===========================================================================

TIER_DISTRIBUTION: Dict[str, float] = {
    "strategic": 0.10,       # 10 % — high interaction frequency
    "standard": 0.30,        # 30 % — medium interaction frequency
    "transactional": 0.60,   # 60 % — low interaction frequency
}
"""Default tier distribution for entity pools.

The three fractions **must** sum to exactly ``1.00``.  Strategic entities are
the highest-importance / highest-interaction subset, while transactional
entities represent the long tail.  (README.md lines 470-474.)
"""

TIER_TO_PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "strategic": {
        "excellent": 0.50,
        "good": 0.35,
        "average": 0.10,
        "poor": 0.05,
        "problem": 0.00,
    },
    "standard": {
        "excellent": 0.15,
        "good": 0.35,
        "average": 0.35,
        "poor": 0.10,
        "problem": 0.05,
    },
    "transactional": {
        "excellent": 0.05,
        "good": 0.15,
        "average": 0.35,
        "poor": 0.30,
        "problem": 0.15,
    },
}
"""Mapping of entity tier to weighted probability of each behavior profile.

Strategic entities skew heavily toward excellent/good profiles, while
transactional entities are more likely to have average/poor/problem profiles.
Weights within each tier sum to ``1.00``.
"""


# ===========================================================================
# ExternalWorldConfig — Pydantic V2 configuration model (AAP §0.7.1)
# ===========================================================================


class ExternalWorldConfig(BaseModel):
    """Configuration model for the :class:`ExternalWorldManager`.

    Uses **Pydantic V2** ``BaseModel`` at the subsystem boundary, as mandated
    by AAP §0.7.1.  All fields carry sensible defaults so that
    ``ExternalWorldConfig()`` produces a usable configuration out of the box.

    Attributes:
        tier_distribution: Mapping of tier name to fraction.  Must sum to 1.0.
        project1_api_base_url: Optional base URL for Project 1 REST API.
        api_timeout_seconds: Timeout in seconds for REST API calls.
        max_retries: Maximum number of retry attempts for API calls.
        enable_api_integration: Whether to query Project 1 REST API for
            master data at startup.
    """

    tier_distribution: Dict[str, float] = Field(
        default_factory=lambda: dict(TIER_DISTRIBUTION),
        description="Tier name to fraction mapping.  Must sum to 1.0.",
    )
    project1_api_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for Project 1 REST API (e.g. 'http://localhost:8000').",
    )
    api_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Timeout in seconds for REST API calls.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for API calls.",
    )
    enable_api_integration: bool = Field(
        default=False,
        description="Whether to fetch master data from Project 1 REST API.",
    )

    @field_validator("tier_distribution")
    @classmethod
    def _validate_tier_distribution(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Ensure that tier distribution values sum to approximately 1.0."""
        total = sum(value.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"tier_distribution values must sum to 1.0, got {total:.6f}"
            )
        for tier_name, fraction in value.items():
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(
                    f"tier_distribution['{tier_name}'] must be in [0.0, 1.0], "
                    f"got {fraction}"
                )
        return value


# ===========================================================================
# ExternalWorldManager — Central coordinator (README.md lines 460-486)
# ===========================================================================


class ExternalWorldManager:
    """Central coordinator for all external entity simulation.

    Manages entity pools (customers, vendors, banks, carriers) with tiered
    distribution, assigns behavior profiles, and delegates specific interaction
    generation to ``CustomerSimulator``, ``VendorSimulator``, and
    ``BankSimulator``.

    All dependencies are injected via the constructor (AAP §0.7.1).  The
    ``SimulationEngine`` acts as the composition root that wires this class
    with its statistical models and event bus.

    Args:
        config: External world configuration.  Uses defaults when ``None``.
        amount_distribution: Injected ``AmountDistribution`` for transaction
            amount sampling (log-normal distributions by tier).
        order_frequency_model: Injected ``OrderFrequencyModel`` for Poisson
            daily order counts with day-of-week effects.
        payment_timing_model: Injected ``PaymentTimingModel`` for 5-segment
            mixture model payment dates.
        selection_model: Injected ``SelectionModel`` for Pareto 80/20 entity
            selection and revenue-weighted customer selection.
        event_bus: Injected ``EventBus`` for publishing entity interaction
            events (TransactionCreated, DocumentGenerated, etc.).
        seed: Optional RNG seed for reproducible simulation runs.

    Example::

        manager = ExternalWorldManager(
            amount_distribution=amount_dist,
            order_frequency_model=freq_model,
            payment_timing_model=timing_model,
            selection_model=sel_model,
            event_bus=bus,
            seed=42,
        )
        await manager.initialize_entity_pools(customers=[...], vendors=[...])
        interactions = await manager.generate_daily_interactions(today)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[ExternalWorldConfig] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        order_frequency_model: Optional[OrderFrequencyModel] = None,
        payment_timing_model: Optional[PaymentTimingModel] = None,
        selection_model: Optional[SelectionModel] = None,
        event_bus: Optional[EventBus] = None,
        seed: Optional[int] = None,
    ) -> None:
        # Configuration
        self._config: ExternalWorldConfig = config or ExternalWorldConfig()
        self._seed: Optional[int] = seed
        self.rng: np.random.RandomState = np.random.RandomState(seed)

        # Injected statistical models
        self._amount_distribution = amount_distribution
        self._order_frequency_model = order_frequency_model
        self._payment_timing_model = payment_timing_model
        self._selection_model = selection_model

        # Injected event bus
        self._event_bus = event_bus

        # Tier distribution helper (Pydantic V2 model)
        self._tier_distribution = TierDistribution(
            strategic=self._config.tier_distribution.get("strategic", 0.10),
            standard=self._config.tier_distribution.get("standard", 0.30),
            transactional=self._config.tier_distribution.get("transactional", 0.60),
        )

        # Entity pools (initialised empty, populated via initialize_entity_pools)
        self._customers: List[Dict[str, Any]] = []
        self._vendors: List[Dict[str, Any]] = []
        self._banks: List[Dict[str, Any]] = []
        self._carriers: List[Dict[str, Any]] = []

        # Metrics tracking
        self._metrics: Dict[str, Any] = {
            "total_customer_orders": 0,
            "total_vendor_invoices": 0,
            "total_bank_statements": 0,
            "total_daily_interactions": 0,
            "last_interaction_date": None,
        }

        # Sub-simulators — created with constructor injection
        self._customer_simulator = CustomerSimulator(
            amount_distribution=amount_distribution,
            order_frequency_model=order_frequency_model,
            payment_timing_model=payment_timing_model,
            selection_model=selection_model,
            event_bus=event_bus,
            seed=seed,
        )
        self._vendor_simulator = VendorSimulator(
            amount_distribution=amount_distribution,
            selection_model=selection_model,
            event_bus=event_bus,
            seed=seed,
        )
        self._bank_simulator = BankSimulator(
            amount_distribution=amount_distribution,
            event_bus=event_bus,
            seed=seed,
        )

        logger.info(
            "external_world_manager_initialized",
            seed=seed,
            enable_api_integration=self._config.enable_api_integration,
            tier_distribution=self._config.tier_distribution,
            has_amount_distribution=amount_distribution is not None,
            has_order_frequency_model=order_frequency_model is not None,
            has_payment_timing_model=payment_timing_model is not None,
            has_selection_model=selection_model is not None,
            has_event_bus=event_bus is not None,
        )

    # ------------------------------------------------------------------
    # Properties — entity pool access (README.md lines 463-467)
    # ------------------------------------------------------------------

    @property
    def customers(self) -> List[Dict[str, Any]]:
        """Return the current customer entity pool."""
        return self._customers

    @property
    def vendors(self) -> List[Dict[str, Any]]:
        """Return the current vendor entity pool."""
        return self._vendors

    @property
    def banks(self) -> List[Dict[str, Any]]:
        """Return the current bank entity pool."""
        return self._banks

    @property
    def carriers(self) -> List[Dict[str, Any]]:
        """Return the current carrier entity pool."""
        return self._carriers

    # ------------------------------------------------------------------
    # Entity Pool Initialization
    # ------------------------------------------------------------------

    async def initialize_entity_pools(
        self,
        customers: Optional[List[Dict[str, Any]]] = None,
        vendors: Optional[List[Dict[str, Any]]] = None,
        banks: Optional[List[Dict[str, Any]]] = None,
        carriers: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Populate and configure all entity pools.

        For each entity type the caller may supply records directly.  If none
        are provided **and** ``enable_api_integration`` is ``True``, the
        manager attempts to fetch master data from the Project 1 REST API.

        After loading, each entity pool is:
        1. Sorted by a ranking metric (spend, revenue, or insertion order).
        2. Assigned a tier (strategic / standard / transactional) based on
           ``TIER_DISTRIBUTION``.
        3. Assigned a ``BehaviorProfile`` based on tier-weighted probabilities
           from ``TIER_TO_PROFILE_WEIGHTS``.

        Args:
            customers: Pre-loaded customer records.
            vendors: Pre-loaded vendor records.
            banks: Pre-loaded bank records.
            carriers: Pre-loaded carrier records.
        """
        # --- Load entity data -----------------------------------------------
        if customers is not None:
            self._customers = [dict(c) for c in customers]
        elif self._config.enable_api_integration:
            self._customers = await self._fetch_customers_from_api()

        if vendors is not None:
            self._vendors = [dict(v) for v in vendors]
        elif self._config.enable_api_integration:
            self._vendors = await self._fetch_vendors_from_api()

        if banks is not None:
            self._banks = [dict(b) for b in banks]

        if carriers is not None:
            self._carriers = [dict(c) for c in carriers]

        # --- Assign tiers ---------------------------------------------------
        self._customers = self.assign_entity_tier(self._customers)
        self._vendors = self.assign_entity_tier(self._vendors)
        self._banks = self.assign_entity_tier(self._banks)
        self._carriers = self.assign_entity_tier(self._carriers)

        # --- Assign behavior profiles ---------------------------------------
        for i, customer in enumerate(self._customers):
            self._customers[i] = self.assign_behavior_profile(customer)
        for i, vendor in enumerate(self._vendors):
            self._vendors[i] = self.assign_behavior_profile(vendor)
        for i, bank in enumerate(self._banks):
            self._banks[i] = self.assign_behavior_profile(bank)
        for i, carrier in enumerate(self._carriers):
            self._carriers[i] = self.assign_behavior_profile(carrier)

        logger.info(
            "entity_pools_initialized",
            customer_count=len(self._customers),
            vendor_count=len(self._vendors),
            bank_count=len(self._banks),
            carrier_count=len(self._carriers),
            tier_distribution=self._summarize_tier_counts(),
        )

    # ------------------------------------------------------------------
    # Tier Assignment (README.md lines 469-474)
    # ------------------------------------------------------------------

    def assign_entity_tier(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assign interaction tiers to entities based on ranking metrics.

        Entities are sorted by a ranking metric (``spend_history``, ``revenue``,
        ``total_spend``, or insertion order as a fallback) in descending order.
        The top 10 % are assigned ``"strategic"``, the next 30 % ``"standard"``,
        and the remaining 60 % ``"transactional"``.

        Args:
            entities: List of entity dictionaries.

        Returns:
            The same list with a ``"tier"`` field added to each entity.
        """
        if not entities:
            return entities

        # Sort by a ranking metric (descending) — use the first found
        def _rank_key(entity: Dict[str, Any]) -> float:
            for key in ("spend_history", "revenue", "total_spend", "annual_revenue"):
                val = entity.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
            return 0.0

        sorted_entities = sorted(entities, key=_rank_key, reverse=True)

        total = len(sorted_entities)
        for idx, entity in enumerate(sorted_entities):
            rank_percentile = idx / max(total, 1)
            entity["tier"] = self._tier_distribution.get_tier_for_rank(rank_percentile)

        return sorted_entities

    # ------------------------------------------------------------------
    # Behavior Profile Assignment (README.md lines 477-479)
    # ------------------------------------------------------------------

    def assign_behavior_profile(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Assign a behavior profile to an entity based on its tier.

        Uses ``TIER_TO_PROFILE_WEIGHTS`` to probabilistically select one of the
        five named profiles (excellent, good, average, poor, problem), then
        creates a ``BehaviorProfile`` instance via ``BehaviorProfileFactory``
        with entity-level ±10 % variation for realism.

        Args:
            entity: Entity dictionary with at least a ``"tier"`` field.

        Returns:
            The entity dictionary with a ``"behavior_profile"`` field attached.
        """
        tier: str = entity.get("tier", "transactional")
        weights_map: Dict[str, float] = TIER_TO_PROFILE_WEIGHTS.get(
            tier, TIER_TO_PROFILE_WEIGHTS["transactional"]
        )

        # Decompose to parallel lists for rng.choice
        profile_names: List[str] = list(weights_map.keys())
        weights: List[float] = list(weights_map.values())

        # Normalise weights to ensure they sum to 1.0 (defensive)
        weight_sum = sum(weights)
        if weight_sum > 0:
            probabilities = [w / weight_sum for w in weights]
        else:
            probabilities = [1.0 / len(weights)] * len(weights)

        # Select profile name based on weighted probability
        selected_profile_name: str = str(
            self.rng.choice(profile_names, p=probabilities)
        )

        # Determine entity type for the profile factory
        entity_type: str = self._infer_entity_type(entity)

        # Create profile with ±10 % variation for entity-level realism
        profile: BehaviorProfile = BehaviorProfileFactory.create_profile_with_variation(
            profile_name=selected_profile_name,
            entity_type=entity_type,
            rng=self.rng,
        )

        entity["behavior_profile"] = profile
        entity["profile_name"] = selected_profile_name

        logger.debug(
            "behavior_profile_assigned",
            entity_type=entity_type,
            tier=tier,
            profile_name=selected_profile_name,
        )

        return entity

    # ------------------------------------------------------------------
    # Interaction Generation — Delegates to sub-simulators
    # ------------------------------------------------------------------

    async def generate_vendor_invoice(
        self,
        vendor: Dict[str, Any],
        po: Dict[str, Any],
        simulation_date: date,
    ) -> Any:
        """Generate a single vendor invoice for a purchase order.

        Delegates to ``VendorSimulator.generate_vendor_invoice()``.

        Args:
            vendor: Vendor entity dictionary.
            po: Purchase order dictionary with ``"po_id"`` and ``"amount"``.
            simulation_date: Current simulation date.

        Returns:
            A ``VendorInvoice`` model instance.
        """
        result = await self._vendor_simulator.generate_vendor_invoice(
            vendor=vendor, po=po, simulation_date=simulation_date
        )
        self._metrics["total_vendor_invoices"] += 1
        return result

    async def generate_customer_order(
        self,
        customer: Dict[str, Any],
        simulation_date: date,
        available_products: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Generate a single customer purchase order.

        Delegates to ``CustomerSimulator.generate_customer_order()``.

        Args:
            customer: Customer entity dictionary.
            simulation_date: Current simulation date.
            available_products: Optional product catalogue for line items.

        Returns:
            A ``CustomerOrder`` model instance.
        """
        result = await self._customer_simulator.generate_customer_order(
            customer=customer,
            simulation_date=simulation_date,
            available_products=available_products,
        )
        self._metrics["total_customer_orders"] += 1
        return result

    async def generate_bank_statement(
        self,
        bank: Dict[str, Any],
        transactions: List[Dict[str, Any]],
        statement_month: date,
        opening_balance: float = 0.0,
    ) -> Any:
        """Generate a monthly bank statement.

        Delegates to ``BankSimulator.generate_bank_statement()``.

        Args:
            bank: Bank entity dictionary.
            transactions: Transaction records for the statement period.
            statement_month: Any date within the target month.
            opening_balance: Opening balance carried from previous period.

        Returns:
            A ``BankStatement`` model instance.
        """
        result = await self._bank_simulator.generate_bank_statement(
            bank=bank,
            transactions=transactions,
            statement_month=statement_month,
            opening_balance=opening_balance,
        )
        self._metrics["total_bank_statements"] += 1
        return result

    async def generate_daily_interactions(
        self,
        simulation_date: date,
        available_products: Optional[List[Dict[str, Any]]] = None,
        outstanding_invoices: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate all external entity interactions for a single simulation day.

        This is the **main daily orchestration method** invoked by the
        ``SimulationEngine`` during its daily processing loop.  It coordinates:

        1. Customer order generation via ``CustomerSimulator.generate_daily_orders()``.
        2. Customer payment simulation via ``CustomerSimulator.simulate_daily_payments()``.
        3. Vendor batch invoice generation via ``VendorSimulator.generate_batch_invoices()``.

        Performance target: < 2 seconds for external entity response.

        Args:
            simulation_date: The business date being simulated.
            available_products: Optional product catalogue for order line items.
            outstanding_invoices: Optional list of outstanding invoice dicts for
                payment simulation.  Defaults to an empty list.

        Returns:
            Dictionary with keys ``"customer_orders"``, ``"customer_payments"``,
            ``"vendor_invoices"``, and ``"date"``.
        """
        start_time = datetime.now(timezone.utc)

        # --- Customer orders ------------------------------------------------
        customer_orders: List[Any] = []
        if self._customers:
            customer_orders = await self._customer_simulator.generate_daily_orders(
                customers=self._customers,
                simulation_date=simulation_date,
                available_products=available_products,
            )

        # --- Customer payments -----------------------------------------------
        customer_payments: List[Any] = []
        if self._customers:
            customer_payments = await self._customer_simulator.simulate_daily_payments(
                customers=self._customers,
                outstanding_invoices=outstanding_invoices if outstanding_invoices is not None else [],
                simulation_date=simulation_date,
            )

        # --- Vendor invoices -------------------------------------------------
        vendor_invoices: List[Any] = []
        if self._vendors:
            vendor_invoices = await self._vendor_simulator.generate_batch_invoices(
                vendors_with_pos=self._vendors,
                simulation_date=simulation_date,
            )

        # --- Update metrics --------------------------------------------------
        self._metrics["total_customer_orders"] += len(customer_orders)
        self._metrics["total_vendor_invoices"] += len(vendor_invoices)
        self._metrics["total_daily_interactions"] += 1
        self._metrics["last_interaction_date"] = simulation_date.isoformat()

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            "daily_interactions_generated",
            date=simulation_date.isoformat(),
            customer_orders=len(customer_orders),
            customer_payments=len(customer_payments),
            vendor_invoices=len(vendor_invoices),
            elapsed_seconds=round(elapsed, 3),
        )

        return {
            "date": simulation_date.isoformat(),
            "customer_orders": customer_orders,
            "customer_payments": customer_payments,
            "vendor_invoices": vendor_invoices,
        }

    # ------------------------------------------------------------------
    # Entity Query Methods
    # ------------------------------------------------------------------

    def get_entities_by_tier(
        self, entity_type: str, tier: str
    ) -> List[Dict[str, Any]]:
        """Filter an entity pool by tier.

        Args:
            entity_type: One of ``"customer"``, ``"vendor"``, ``"bank"``,
                ``"carrier"``.
            tier: One of ``"strategic"``, ``"standard"``, ``"transactional"``.

        Returns:
            Filtered list of entity dictionaries matching the tier.
        """
        pool = self._get_pool_by_type(entity_type)
        return [e for e in pool if e.get("tier") == tier]

    def get_entity_count(
        self, entity_type: Optional[str] = None
    ) -> Dict[str, int]:
        """Return entity counts, either per type or for a specific type.

        Args:
            entity_type: Optional entity type to filter.  When ``None``,
                returns counts for all types.

        Returns:
            Dictionary mapping entity type names to counts.
        """
        counts: Dict[str, int] = {
            "customer": len(self._customers),
            "vendor": len(self._vendors),
            "bank": len(self._banks),
            "carrier": len(self._carriers),
        }
        if entity_type is not None:
            type_lower = entity_type.lower()
            return {type_lower: counts.get(type_lower, 0)}
        return counts

    def get_tier_distribution(self) -> Dict[str, Dict[str, int]]:
        """Return actual tier counts per entity type.

        Useful for validating that the tier assignment produced the expected
        distribution ratios.

        Returns:
            Nested dictionary: ``{entity_type: {tier: count}}``.
        """
        result: Dict[str, Dict[str, int]] = {}
        for type_name, pool in (
            ("customer", self._customers),
            ("vendor", self._vendors),
            ("bank", self._banks),
            ("carrier", self._carriers),
        ):
            tier_counts: Dict[str, int] = {
                "strategic": 0,
                "standard": 0,
                "transactional": 0,
            }
            for entity in pool:
                tier = entity.get("tier", "transactional")
                if tier in tier_counts:
                    tier_counts[tier] += 1
                else:
                    tier_counts[tier] = 1
            result[type_name] = tier_counts
        return result

    # ------------------------------------------------------------------
    # Metrics and Health
    # ------------------------------------------------------------------

    async def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the external world subsystem.

        Returns:
            Dictionary containing entity counts, tier distribution, profile
            distribution, and cumulative interaction counts.
        """
        profile_distribution: Dict[str, Dict[str, int]] = {}
        for type_name, pool in (
            ("customer", self._customers),
            ("vendor", self._vendors),
            ("bank", self._banks),
            ("carrier", self._carriers),
        ):
            profile_counts: Dict[str, int] = {}
            for entity in pool:
                p_name = entity.get("profile_name", "unknown")
                profile_counts[p_name] = profile_counts.get(p_name, 0) + 1
            profile_distribution[type_name] = profile_counts

        return {
            "entity_counts": self.get_entity_count(),
            "tier_distribution": self.get_tier_distribution(),
            "profile_distribution": profile_distribution,
            "cumulative_interactions": {
                "total_customer_orders": self._metrics["total_customer_orders"],
                "total_vendor_invoices": self._metrics["total_vendor_invoices"],
                "total_bank_statements": self._metrics["total_bank_statements"],
                "total_daily_interactions": self._metrics["total_daily_interactions"],
            },
            "last_interaction_date": self._metrics["last_interaction_date"],
        }

    # ------------------------------------------------------------------
    # Project 1 REST API Integration (AAP §0.4.1 — read-only, aiohttp)
    # ------------------------------------------------------------------

    async def _fetch_customers_from_api(self) -> List[Dict[str, Any]]:
        """Fetch customer master data from the Project 1 REST API.

        Performs a read-only HTTP GET to ``{base_url}/api/customers``.  If the
        API is unavailable or ``aiohttp`` is not installed, an empty list is
        returned and a warning is logged.

        Returns:
            List of customer dictionaries, or empty list on failure.
        """
        if aiohttp is None:
            logger.warning(
                "aiohttp_not_available",
                message="Cannot fetch customers — aiohttp not installed.",
            )
            return []

        base_url = self._config.project1_api_base_url
        if not base_url:
            logger.debug("api_integration_skipped", reason="no_base_url_configured")
            return []

        url = f"{base_url.rstrip('/')}/api/customers"
        return await self._fetch_from_api(url, "customers")

    async def _fetch_vendors_from_api(self) -> List[Dict[str, Any]]:
        """Fetch vendor master data from the Project 1 REST API.

        Performs a read-only HTTP GET to ``{base_url}/api/vendors``.

        Returns:
            List of vendor dictionaries, or empty list on failure.
        """
        if aiohttp is None:
            logger.warning(
                "aiohttp_not_available",
                message="Cannot fetch vendors — aiohttp not installed.",
            )
            return []

        base_url = self._config.project1_api_base_url
        if not base_url:
            logger.debug("api_integration_skipped", reason="no_base_url_configured")
            return []

        url = f"{base_url.rstrip('/')}/api/vendors"
        return await self._fetch_from_api(url, "vendors")

    async def _fetch_products_from_api(self) -> List[Dict[str, Any]]:
        """Fetch product catalogue from the Project 1 REST API.

        Performs a read-only HTTP GET to ``{base_url}/api/products``.

        Returns:
            List of product dictionaries, or empty list on failure.
        """
        if aiohttp is None:
            logger.warning(
                "aiohttp_not_available",
                message="Cannot fetch products — aiohttp not installed.",
            )
            return []

        base_url = self._config.project1_api_base_url
        if not base_url:
            logger.debug("api_integration_skipped", reason="no_base_url_configured")
            return []

        url = f"{base_url.rstrip('/')}/api/products"
        return await self._fetch_from_api(url, "products")

    async def _fetch_from_api(
        self, url: str, entity_label: str
    ) -> List[Dict[str, Any]]:
        """Internal helper to perform a read-only GET from the Project 1 API.

        Implements retry logic with ``max_retries`` attempts and per-request
        timeouts from the configuration.

        Args:
            url: Full URL to fetch.
            entity_label: Human-readable label for logging (e.g. ``"customers"``).

        Returns:
            Parsed JSON list, or empty list on failure.
        """
        timeout = aiohttp.ClientTimeout(total=self._config.api_timeout_seconds)
        last_error: Optional[Exception] = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, list):
                                logger.info(
                                    "api_fetch_success",
                                    entity=entity_label,
                                    count=len(data),
                                    url=url,
                                )
                                return data
                            # If the response wraps items under a key, try
                            # common patterns: "results", "data", "items"
                            if isinstance(data, dict):
                                for key in ("results", "data", "items", entity_label):
                                    if key in data and isinstance(data[key], list):
                                        logger.info(
                                            "api_fetch_success",
                                            entity=entity_label,
                                            count=len(data[key]),
                                            url=url,
                                            wrapper_key=key,
                                        )
                                        return data[key]
                            logger.warning(
                                "api_unexpected_format",
                                entity=entity_label,
                                url=url,
                                status=response.status,
                            )
                            return []
                        else:
                            logger.warning(
                                "api_fetch_failed",
                                entity=entity_label,
                                url=url,
                                status=response.status,
                                attempt=attempt,
                            )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "api_fetch_error",
                    entity=entity_label,
                    url=url,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < self._config.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 10))

        logger.error(
            "api_fetch_exhausted",
            entity=entity_label,
            url=url,
            max_retries=self._config.max_retries,
            last_error=str(last_error) if last_error else None,
        )
        return []

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_pool_by_type(self, entity_type: str) -> List[Dict[str, Any]]:
        """Return the entity pool for a given entity type string.

        Args:
            entity_type: One of ``"customer"``, ``"vendor"``, ``"bank"``,
                ``"carrier"``.

        Returns:
            The corresponding entity list.

        Raises:
            ValueError: If *entity_type* is not recognised.
        """
        type_lower = entity_type.lower()
        mapping: Dict[str, List[Dict[str, Any]]] = {
            "customer": self._customers,
            "vendor": self._vendors,
            "bank": self._banks,
            "carrier": self._carriers,
        }
        if type_lower not in mapping:
            raise ValueError(
                f"Unknown entity_type '{entity_type}'. "
                f"Must be one of: {list(mapping.keys())}"
            )
        return mapping[type_lower]

    @staticmethod
    def _infer_entity_type(entity: Dict[str, Any]) -> str:
        """Infer the entity type from dictionary keys.

        Checks for well-known keys (``entity_type``, ``customer_id``,
        ``vendor_id``, ``bank_id``, ``carrier_id``) to determine the
        entity type.  Falls back to ``"customer"`` if ambiguous.

        Args:
            entity: Entity dictionary.

        Returns:
            One of ``"customer"``, ``"vendor"``, ``"bank"``, ``"carrier"``.
        """
        # Explicit field
        explicit_type = entity.get("entity_type", "").lower()
        if explicit_type in ("customer", "vendor", "bank", "carrier"):
            return explicit_type

        # Infer from presence of well-known ID keys
        if "customer_id" in entity:
            return "customer"
        if "vendor_id" in entity:
            return "vendor"
        if "bank_id" in entity:
            return "bank"
        if "carrier_id" in entity:
            return "carrier"

        return "customer"

    def _summarize_tier_counts(self) -> Dict[str, Dict[str, int]]:
        """Generate a summary of tier counts per entity type for logging."""
        return self.get_tier_distribution()
