"""
Entity Selection Models for the Synthetic ERP Data Generation Platform.

This module implements weighted selection algorithms for vendors, customers, and products
using statistical distributions that produce realistic ERP entity selection patterns:

- **Vendor Selection (Pareto 80/20)**: Top 20% of vendors receive ~80% of business,
  weighted by spend history, with category-aware filtering. Uses rank-based Pareto
  weighting to model heavy-tail concentration in procurement.

- **Customer Selection (Revenue-Weighted)**: Larger customers generate more frequent
  orders via revenue-proportional weighting. Strategic tier customers (10%) appear
  more frequently than transactional tier (60%).

- **Product Selection (70/30 Repeat/New)**: 70% repeat purchases from customer history,
  30% new products from the available catalog. Falls back gracefully when history is
  empty or all products have been purchased.

All selection methods use numpy.random.RandomState for reproducible seeding and are
vectorized for batch operations achieving >= 10,000 selections/second.

This is a pure-function module with NO internal app dependencies.

References:
    - README.md lines 566-586: SelectionModel specification
    - AAP Section 0.5.1 Group 3: Statistical Models
    - AAP Section 0.7.1: Pydantic V2 at subsystem boundaries
    - AAP Section 0.7.6: structlog JSON logging
"""

from __future__ import annotations

from scipy import stats
import numpy as np
from typing import Optional, Dict, List, Any, Sequence
from pydantic import BaseModel, Field, model_validator
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration Model (Pydantic V2)
# ---------------------------------------------------------------------------


class SelectionConfig(BaseModel):
    """Configuration for entity selection distribution parameters.

    Controls the Pareto shape for vendor concentration (80/20 rule),
    repeat-vs-new product purchase split, and top-vendor thresholds
    used for validation and analytics.

    Attributes:
        pareto_shape: Pareto distribution shape parameter. A value of ~1.16
            produces the classic 80/20 concentration where the top 20% of
            entities capture ~80% of selections. Higher values increase
            concentration; lower values flatten the distribution.
        repeat_purchase_rate: Probability of selecting a product from
            the customer's existing purchase history (default 0.70 = 70%).
        new_purchase_rate: Probability of selecting a new product not in
            the customer's history (default 0.30 = 30%). Must equal
            ``1.0 - repeat_purchase_rate``.
        top_vendor_pct: The fraction of vendors considered "top" for
            concentration ratio analytics (default 0.20 = top 20%).
        top_vendor_business_pct: The expected share of business captured
            by the top vendor fraction (default 0.80 = 80%). Used as a
            validation target, not an enforced constraint.
    """

    pareto_shape: float = Field(
        default=1.16,
        gt=0.0,
        description=(
            "Pareto shape parameter controlling vendor concentration. "
            "~1.16 yields the 80/20 rule."
        ),
    )
    repeat_purchase_rate: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Probability of selecting a repeat product from history (70%).",
    )
    new_purchase_rate: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Probability of selecting a new product from catalog (30%).",
    )
    top_vendor_pct: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="Top vendor fraction for analytics (20%).",
    )
    top_vendor_business_pct: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Expected business share for top vendors (80%).",
    )

    @model_validator(mode="after")
    def _validate_purchase_rates(self) -> "SelectionConfig":
        """Ensure repeat_purchase_rate + new_purchase_rate == 1.0."""
        total = self.repeat_purchase_rate + self.new_purchase_rate
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"repeat_purchase_rate ({self.repeat_purchase_rate}) + "
                f"new_purchase_rate ({self.new_purchase_rate}) must equal 1.0, "
                f"got {total}"
            )
        return self


# ---------------------------------------------------------------------------
# SelectionModel — Core Selection Engine
# ---------------------------------------------------------------------------


class SelectionModel:
    """Entity selection engine using weighted statistical distributions.

    Implements three primary selection algorithms:

    1. **Vendor selection** — Pareto 80/20 rule where top 20% of vendors
       (by spend history) capture ~80% of selections.  Uses rank-based
       weighting with the configured ``pareto_shape`` exponent so that
       even when spend values are unavailable, the heavy-tail property
       is preserved.

    2. **Customer selection** — Revenue-weighted random selection where
       larger customers generate proportionally more frequent orders.

    3. **Product selection** — 70/30 repeat/new split where 70% of
       selections come from the customer's purchase history and 30%
       introduce new products from the available catalog.

    All methods accept entity dictionaries and return entity dictionaries,
    making them agnostic to the upstream data schema.

    Args:
        config: Selection parameters.  Defaults to ``SelectionConfig()``
            which produces the standard 80/20 / 70-30 configuration.
        seed: Optional integer seed for ``numpy.random.RandomState``
            to enable reproducible selection sequences.

    Example::

        model = SelectionModel(seed=42)
        vendor = model.select_vendor(vendors, category="office_supplies")
        customer = model.select_customer(customers)
        product = model.select_product(catalog, purchase_history=history)
    """

    def __init__(
        self,
        config: Optional[SelectionConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config if config is not None else SelectionConfig()
        self.rng = np.random.RandomState(seed)
        self._seed = seed
        logger.info(
            "selection_model_initialized",
            pareto_shape=self.config.pareto_shape,
            repeat_purchase_rate=self.config.repeat_purchase_rate,
            new_purchase_rate=self.config.new_purchase_rate,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(entity: Dict[str, Any]) -> str:
        """Extract an identifier from an entity dict.

        Looks for common identifier keys in order: ``id``, ``entity_id``,
        ``vendor_id``, ``customer_id``, ``product_id``, ``name``.
        Falls back to ``str(hash(frozenset(entity.items())))`` if none found.
        """
        for key in ("id", "entity_id", "vendor_id", "customer_id", "product_id", "name"):
            if key in entity:
                return str(entity[key])
        # Deterministic fallback — use sorted items for consistency
        try:
            return str(hash(frozenset(sorted(entity.items()))))
        except TypeError:
            return str(id(entity))

    def _compute_pareto_weights(self, spend_values: np.ndarray) -> np.ndarray:
        """Compute Pareto-distributed selection weights from spend history.

        Uses ``scipy.stats.pareto`` survival function to assign rank-based
        weights that produce the classic heavy-tail concentration.  When
        spend values are available and non-zero, entities are ranked by
        spend (descending) and the Pareto survival function at each rank
        position determines the selection weight.  When all spend values
        are zero or missing, a pure rank-based Pareto fallback is used.

        The resulting array is normalised to form a valid probability
        distribution that sums to 1.0.

        Args:
            spend_values: 1-D array of non-negative spend history values,
                one per entity.  Order must match the entity list from which
                selections will be made.

        Returns:
            Normalised weight array of the same length as *spend_values*.
        """
        n = len(spend_values)
        if n == 0:
            return np.array([], dtype=np.float64)
        if n == 1:
            return np.array([1.0], dtype=np.float64)

        shape = self.config.pareto_shape

        # Build the Pareto distribution for rank-based weighting.
        # stats.pareto(b=shape) has survival function S(x) = 1/x^b for x >= 1,
        # which naturally produces the heavy-tail 80/20 concentration.
        pareto_dist = stats.pareto(b=shape)

        # Sort indices by spend descending to establish rank order
        sorted_indices = np.argsort(-spend_values)

        # Check if any spend values are positive
        has_positive_spend = bool(np.any(spend_values > 0))

        raw_weights = np.zeros(n, dtype=np.float64)

        if has_positive_spend:
            # Rank positions 1..n (Pareto x >= 1)
            rank_positions = np.arange(1, n + 1, dtype=np.float64)
            # Pareto survival function gives decreasing weights by rank
            rank_weights = pareto_dist.sf(rank_positions)
            # Ensure minimum weight so no entity is completely ignored
            rank_weights = np.maximum(rank_weights, 1e-10)

            for rank, idx in enumerate(sorted_indices):
                # Combine spend magnitude with Pareto rank weight
                spend_component = spend_values[idx] + 1e-10
                raw_weights[idx] = (spend_component ** shape) * rank_weights[rank]
        else:
            # Pure rank-based fallback using Pareto survival function
            rank_positions = np.arange(1, n + 1, dtype=np.float64)
            pareto_weights = pareto_dist.sf(rank_positions)
            pareto_weights = np.maximum(pareto_weights, 1e-10)
            for rank, idx in enumerate(sorted_indices):
                raw_weights[idx] = pareto_weights[rank]

        total = np.sum(raw_weights)
        if total <= 0:
            # Ultimate fallback — uniform
            return np.ones(n, dtype=np.float64) / n

        return raw_weights / total

    def _prepare_vendor_list(
        self,
        vendors: List[Dict[str, Any]],
        category: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Filter vendors by category with fallback to full list."""
        if not vendors:
            raise ValueError("Vendor list must not be empty.")

        if category is not None:
            filtered = [
                v for v in vendors
                if v.get("category") == category
                or v.get("procurement_category") == category
            ]
            if filtered:
                return filtered
            # Fall back to all vendors when category filter yields nothing
            logger.debug(
                "vendor_category_filter_empty_fallback",
                category=category,
                total_vendors=len(vendors),
            )

        return vendors

    def _get_vendor_weights(
        self, vendors: Sequence[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract spend values and compute Pareto weights for vendors."""
        spend_values = np.array(
            [
                float(v.get("spend_history", v.get("spend", v.get("total_spend", 0.0))))
                for v in vendors
            ],
            dtype=np.float64,
        )
        return self._compute_pareto_weights(spend_values)

    # ------------------------------------------------------------------
    # Vendor Selection — Pareto 80/20
    # ------------------------------------------------------------------

    def select_vendor(
        self,
        vendors: List[Dict[str, Any]],
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select a single vendor using Pareto-weighted probabilities.

        Implements the 80/20 rule: the top 20% of vendors (ranked by spend
        history) are selected approximately 80% of the time.

        If *category* is provided, only vendors matching that procurement
        category are considered.  When no vendors match the category, the
        full vendor list is used as a fallback.

        Args:
            vendors: Non-empty list of vendor dictionaries.  Each vendor
                should contain a ``"spend_history"`` (or ``"spend"`` /
                ``"total_spend"``) key for weighted selection; if missing,
                rank-based weighting is used.
            category: Optional procurement category string to filter
                vendors.

        Returns:
            A single vendor dictionary selected from *vendors*.

        Raises:
            ValueError: If *vendors* is empty.
        """
        eligible = self._prepare_vendor_list(vendors, category)
        weights = self._get_vendor_weights(eligible)
        idx = self.rng.choice(len(eligible), p=weights)
        selected = eligible[idx]
        logger.debug(
            "vendor_selected",
            vendor_id=self._extract_id(selected),
            category=category,
            pool_size=len(eligible),
        )
        return selected

    def select_vendors_batch(
        self,
        vendors: List[Dict[str, Any]],
        n: int,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Select *n* vendors (with replacement) using Pareto-weighted probabilities.

        Vectorised for performance >= 10,000 selections/second.

        Args:
            vendors: Non-empty list of vendor dictionaries.
            n: Number of vendors to select.
            category: Optional procurement category filter.

        Returns:
            List of *n* vendor dictionaries (may contain duplicates).

        Raises:
            ValueError: If *vendors* is empty or *n* < 1.
        """
        if n < 1:
            raise ValueError(f"Batch size n must be >= 1, got {n}.")

        eligible = self._prepare_vendor_list(vendors, category)
        weights = self._get_vendor_weights(eligible)

        # Vectorised batch selection
        indices = self.rng.choice(len(eligible), size=n, replace=True, p=weights)
        selected = [eligible[i] for i in indices]

        logger.debug(
            "vendors_batch_selected",
            batch_size=n,
            pool_size=len(eligible),
            category=category,
        )
        return selected

    # ------------------------------------------------------------------
    # Customer Selection — Revenue-Weighted
    # ------------------------------------------------------------------

    def _get_customer_weights(self, customers: List[Dict[str, Any]]) -> np.ndarray:
        """Extract revenue values and compute normalised selection weights.

        Looks for ``"revenue"`` or ``"annual_revenue"`` keys.  Falls back
        to uniform weighting when no revenue data is available.
        """
        revenues = np.array(
            [
                float(
                    c.get("revenue", c.get("annual_revenue", c.get("total_revenue", 0.0)))
                )
                for c in customers
            ],
            dtype=np.float64,
        )

        # Ensure non-negative
        revenues = np.maximum(revenues, 0.0)

        if np.sum(revenues) <= 0:
            # Uniform fallback
            n = len(customers)
            return np.ones(n, dtype=np.float64) / n

        return revenues / np.sum(revenues)

    def select_customer(
        self,
        customers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Select a single customer using revenue-weighted probabilities.

        Larger customers (higher ``"revenue"`` or ``"annual_revenue"``)
        are selected proportionally more often, ensuring that strategic
        tier customers generate more frequent orders than transactional
        tier customers.

        Args:
            customers: Non-empty list of customer dictionaries.

        Returns:
            A single customer dictionary.

        Raises:
            ValueError: If *customers* is empty.
        """
        if not customers:
            raise ValueError("Customer list must not be empty.")

        weights = self._get_customer_weights(customers)
        idx = self.rng.choice(len(customers), p=weights)
        selected = customers[idx]
        logger.debug(
            "customer_selected",
            customer_id=self._extract_id(selected),
            pool_size=len(customers),
        )
        return selected

    def select_customers_batch(
        self,
        customers: List[Dict[str, Any]],
        n: int,
    ) -> List[Dict[str, Any]]:
        """Select *n* customers (with replacement) using revenue-weighted probabilities.

        Vectorised for performance >= 10,000 selections/second.

        Args:
            customers: Non-empty list of customer dictionaries.
            n: Number of customers to select.

        Returns:
            List of *n* customer dictionaries.

        Raises:
            ValueError: If *customers* is empty or *n* < 1.
        """
        if not customers:
            raise ValueError("Customer list must not be empty.")
        if n < 1:
            raise ValueError(f"Batch size n must be >= 1, got {n}.")

        weights = self._get_customer_weights(customers)
        indices = self.rng.choice(len(customers), size=n, replace=True, p=weights)
        selected = [customers[i] for i in indices]

        logger.debug(
            "customers_batch_selected",
            batch_size=n,
            pool_size=len(customers),
        )
        return selected

    # ------------------------------------------------------------------
    # Product Selection — 70/30 Repeat/New Split
    # ------------------------------------------------------------------

    def select_product(
        self,
        available_products: List[Dict[str, Any]],
        purchase_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Select a single product using the 70/30 repeat/new split.

        When the customer has purchase history:
        - With probability ``repeat_purchase_rate`` (70%): a product is
          selected uniformly from *purchase_history*.
        - With probability ``new_purchase_rate`` (30%): a product is
          selected uniformly from *available_products* that are **not**
          in the purchase history (by ID).

        If no new products remain (all already purchased), the selection
        falls back to the purchase history.  If no purchase history is
        provided, a random product from the catalog is returned.

        Args:
            available_products: Non-empty list of product dictionaries.
            purchase_history: Optional list of previously purchased product
                dictionaries.

        Returns:
            A single product dictionary.

        Raises:
            ValueError: If *available_products* is empty.
        """
        if not available_products:
            raise ValueError("Available products list must not be empty.")

        # No history — select randomly from catalog
        if not purchase_history:
            idx = self.rng.randint(0, len(available_products))
            return available_products[idx]

        # Determine repeat vs new via random draw
        draw = self.rng.random()

        if draw < self.config.repeat_purchase_rate:
            # --- Repeat purchase (70%) ---
            idx = self.rng.randint(0, len(purchase_history))
            selected = purchase_history[idx]
            logger.debug(
                "product_selected_repeat",
                product_id=self._extract_id(selected),
                history_size=len(purchase_history),
            )
            return selected

        # --- New purchase (30%) ---
        history_ids = {self._extract_id(p) for p in purchase_history}
        new_products = [
            p for p in available_products
            if self._extract_id(p) not in history_ids
        ]

        if not new_products:
            # All products already purchased — fall back to history
            idx = self.rng.randint(0, len(purchase_history))
            selected = purchase_history[idx]
            logger.debug(
                "product_selected_repeat_fallback",
                product_id=self._extract_id(selected),
                reason="no_new_products_available",
            )
            return selected

        idx = self.rng.randint(0, len(new_products))
        selected = new_products[idx]
        logger.debug(
            "product_selected_new",
            product_id=self._extract_id(selected),
            new_pool_size=len(new_products),
        )
        return selected

    def select_products_batch(
        self,
        available_products: List[Dict[str, Any]],
        n: int,
        purchase_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Select *n* products using the 70/30 repeat/new split.

        Each selection is independent; the batch may contain duplicates.

        Args:
            available_products: Non-empty list of product dictionaries.
            n: Number of products to select.
            purchase_history: Optional list of previously purchased products.

        Returns:
            List of *n* product dictionaries.

        Raises:
            ValueError: If *available_products* is empty or *n* < 1.
        """
        if not available_products:
            raise ValueError("Available products list must not be empty.")
        if n < 1:
            raise ValueError(f"Batch size n must be >= 1, got {n}.")

        # Pre-compute new products pool once for efficiency
        if purchase_history:
            history_ids = {self._extract_id(p) for p in purchase_history}
            new_products = [
                p for p in available_products
                if self._extract_id(p) not in history_ids
            ]
        else:
            new_products = list(available_products)

        results: List[Dict[str, Any]] = []
        # Vectorised random draws for repeat-vs-new decisions
        draws = self.rng.random(size=n)

        for draw in draws:
            if not purchase_history:
                # No history — always pick from catalog
                idx = self.rng.randint(0, len(available_products))
                results.append(available_products[idx])
            elif draw < self.config.repeat_purchase_rate:
                # Repeat purchase
                idx = self.rng.randint(0, len(purchase_history))
                results.append(purchase_history[idx])
            elif new_products:
                # New product
                idx = self.rng.randint(0, len(new_products))
                results.append(new_products[idx])
            else:
                # Fallback to history
                idx = self.rng.randint(0, len(purchase_history))
                results.append(purchase_history[idx])

        logger.debug(
            "products_batch_selected",
            batch_size=n,
            catalog_size=len(available_products),
            history_size=len(purchase_history) if purchase_history else 0,
        )
        return results

    # ------------------------------------------------------------------
    # Utility / Analytics Methods
    # ------------------------------------------------------------------

    def compute_selection_distribution(
        self,
        entities: List[Dict[str, Any]],
        n_samples: int = 10000,
    ) -> Dict[str, float]:
        """Simulate vendor selections and return the frequency distribution.

        Runs *n_samples* Pareto-weighted selections over *entities* and
        returns a mapping from entity identifier to observed selection
        frequency (0.0–1.0).  Useful for validating that the top 20%
        of entities capture approximately 80% of selections.

        Args:
            entities: List of entity dictionaries (treated as vendors
                for weighting purposes).
            n_samples: Number of Monte Carlo samples (default 10,000).

        Returns:
            Dict mapping entity ID strings to selection frequency floats.
        """
        if not entities:
            return {}

        weights = self._get_vendor_weights(entities)
        indices = self.rng.choice(len(entities), size=n_samples, replace=True, p=weights)

        counts = np.zeros(len(entities), dtype=np.int64)
        for idx in indices:
            counts[idx] += 1

        distribution: Dict[str, float] = {}
        for i, entity in enumerate(entities):
            entity_id = self._extract_id(entity)
            distribution[entity_id] = float(counts[i]) / n_samples

        return distribution

    def get_concentration_ratio(
        self,
        entities: List[Dict[str, Any]],
        top_pct: float = 0.20,
        n_samples: int = 10000,
    ) -> float:
        """Calculate the share of selections captured by the top entities.

        With default Pareto configuration (shape=1.16, top_pct=0.20),
        this should return approximately 0.80, confirming the 80/20 rule.

        Args:
            entities: List of entity dictionaries (treated as vendors
                for weighting purposes).
            top_pct: Fraction of entities considered "top" (default 0.20).
            n_samples: Number of Monte Carlo samples (default 10,000).

        Returns:
            Float in [0.0, 1.0] representing the selection share of the
            top fraction.
        """
        if not entities or top_pct <= 0.0:
            return 0.0

        distribution = self.compute_selection_distribution(entities, n_samples)
        # Sort frequencies descending
        frequencies = sorted(distribution.values(), reverse=True)
        top_count = max(1, int(len(frequencies) * top_pct))
        top_share = sum(frequencies[:top_count])

        logger.info(
            "concentration_ratio_computed",
            total_entities=len(entities),
            top_pct=top_pct,
            top_count=top_count,
            concentration_ratio=round(top_share, 4),
            n_samples=n_samples,
        )
        return top_share
