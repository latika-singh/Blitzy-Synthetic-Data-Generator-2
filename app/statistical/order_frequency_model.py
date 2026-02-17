"""
Order Frequency Model — Poisson Process with Day-of-Week Effects.

This module implements the OrderFrequencyModel class, which generates realistic
order frequencies for an ERP simulation using a Poisson process with configurable
day-of-week multipliers and optional seasonal adjustments.

Key specifications (from README.md lines 554-558 and AAP Section 0.5.1 Group 3):
    - Base model: Poisson process for order generation
    - Day-of-week multipliers:
        Monday:    0.85×
        Tuesday:   1.00×
        Wednesday: 1.00×
        Thursday:  1.10×
        Friday:    1.20×
        Saturday:  0.00× (non-business day)
        Sunday:    0.00× (non-business day)
    - Performance: ≥ 10,000 samples/second (vectorized batch operations)
    - Reproducible seeding via numpy.random.RandomState
    - Configurable via config/statistical/ YAML files (loaded externally)

This is a pure-function module with NO internal app dependencies.
All data contracts use Pydantic V2 models per AAP Section 0.7.1.
Logging uses structlog to stdout per AAP Section 0.7.6.

Exports:
    OrderFrequencyConfig — Pydantic V2 configuration model
    OrderFrequencyModel  — Core sampling and analytics class
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field
from scipy import stats

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6: structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default day-of-week multipliers (Python weekday convention: 0=Mon … 6=Sun)
# Spec: Mon=0.85, Tue=1.0, Wed=1.0, Thu=1.1, Fri=1.2, Sat/Sun=0.0
# ---------------------------------------------------------------------------
_DEFAULT_DAY_OF_WEEK_MULTIPLIERS: Dict[int, float] = {
    0: 0.85,  # Monday
    1: 1.00,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.10,  # Thursday
    4: 1.20,  # Friday
    5: 0.00,  # Saturday  (non-business day)
    6: 0.00,  # Sunday    (non-business day)
}

# Mapping from weekday integer to human-readable name for analytics output
_DAY_NAMES: Dict[int, str] = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

# Customer-type multipliers for customer-specific frequency sampling
# Strategic tier (10% of entities, AAP 0.1.1 F-005): 2× base rate
# Standard tier (30%): 1× base rate
# Transactional tier (60%): 0.5× base rate
_CUSTOMER_TYPE_MULTIPLIERS: Dict[str, float] = {
    "strategic": 2.0,
    "standard": 1.0,
    "transactional": 0.5,
}


# ============================================================================
# Configuration Model — Pydantic V2 (AAP Section 0.7.1)
# ============================================================================

class OrderFrequencyConfig(BaseModel):
    """Pydantic V2 configuration for the Order Frequency Model.

    All subsystem-boundary data contracts use Pydantic V2 models for validation
    per AAP Section 0.7.1.  This config can be loaded from
    ``config/statistical/`` YAML files by the calling layer.

    Attributes:
        base_rate: Average orders per business day (Poisson λ before
            day-of-week and seasonal adjustments).  Must be > 0.
        day_of_week_multipliers: Mapping of Python ``date.weekday()`` values
            (0=Monday … 6=Sunday) to multiplicative factors applied to
            ``base_rate``.  Saturday/Sunday default to 0.0.
        seasonal_multipliers: Optional mapping of calendar month (1-12) to
            multiplicative factors.  ``None`` disables seasonal adjustment.
        min_orders: Floor clamp applied after Poisson sampling.  Must be ≥ 0.
        max_orders: Ceiling clamp applied after Poisson sampling.
    """

    base_rate: float = Field(
        default=5.0,
        gt=0,
        description="Base Poisson lambda — average orders per business day",
    )
    day_of_week_multipliers: Dict[int, float] = Field(
        default_factory=lambda: dict(_DEFAULT_DAY_OF_WEEK_MULTIPLIERS),
        description=(
            "Day-of-week multipliers keyed by Python weekday int "
            "(0=Mon … 6=Sun).  Saturday/Sunday default to 0.0."
        ),
    )
    seasonal_multipliers: Optional[Dict[int, float]] = Field(
        default=None,
        description=(
            "Optional monthly multipliers keyed by calendar month (1-12). "
            "None disables seasonal adjustment."
        ),
    )
    min_orders: int = Field(
        default=0,
        ge=0,
        description="Minimum orders per day (floor clamp after sampling)",
    )
    max_orders: int = Field(
        default=50,
        description="Maximum orders per day (ceiling clamp after sampling)",
    )


# ============================================================================
# OrderFrequencyModel — Core Sampling and Analytics
# ============================================================================

class OrderFrequencyModel:
    """Poisson-based order frequency model with day-of-week effects.

    Generates realistic daily order counts using a Poisson process whose rate
    parameter (λ) is modulated by:

    1. **Day-of-week multipliers** — captures the empirical pattern of lower
       Monday order volume ramping up to Friday, with zero orders on weekends.
    2. **Seasonal multipliers** — optional per-month adjustment for seasonal
       business cycles.

    The model also provides:

    * Vectorized batch sampling (``sample_order_counts_batch``) achieving
      ≥ 10,000 samples/second for high-throughput simulation.
    * Intra-day order-time generation (``sample_order_times``).
    * Customer-type–specific frequency sampling aligned to the three-tier
      entity model (Strategic / Standard / Transactional).
    * Analytical helpers for effective-rate inspection and weekly distribution.

    All random state is managed through ``numpy.random.RandomState`` for full
    reproducibility across simulation runs.

    Args:
        config: Optional configuration override.  Defaults to
            ``OrderFrequencyConfig()`` when ``None``.
        seed: Optional RNG seed for reproducibility.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[OrderFrequencyConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._config: OrderFrequencyConfig = config or OrderFrequencyConfig()
        self._rng: np.random.RandomState = np.random.RandomState(seed)

        # Validate that all weekday keys 0-6 are present in multipliers.
        # Missing keys are filled with a safe default of 0.0 (no orders).
        for day_idx in range(7):
            if day_idx not in self._config.day_of_week_multipliers:
                self._config.day_of_week_multipliers[day_idx] = 0.0

        # Pre-compute an ordered numpy array of multipliers for vectorized ops
        # Index i corresponds to weekday i (0=Monday … 6=Sunday)
        self._dow_array: np.ndarray = np.array(
            [self._config.day_of_week_multipliers[i] for i in range(7)],
            dtype=np.float64,
        )

        logger.info(
            "order_frequency_model_initialized",
            base_rate=self._config.base_rate,
            day_of_week_multipliers=self._config.day_of_week_multipliers,
            seasonal_multipliers=self._config.seasonal_multipliers,
            min_orders=self._config.min_orders,
            max_orders=self._config.max_orders,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_lambda(self, target_date: date) -> float:
        """Compute the effective Poisson λ for a specific calendar date.

        Applies day-of-week and (optionally) seasonal multipliers to the
        configured ``base_rate``.

        Args:
            target_date: The calendar date to compute the rate for.

        Returns:
            Effective λ (non-negative float).  Returns 0.0 for weekend days
            when their multiplier is 0.0.
        """
        weekday: int = target_date.weekday()  # 0=Mon … 6=Sun
        day_multiplier: float = self._config.day_of_week_multipliers.get(
            weekday, 0.0
        )

        # Short-circuit for zero-multiplier days (weekends)
        if day_multiplier == 0.0:
            return 0.0

        effective: float = self._config.base_rate * day_multiplier

        # Apply optional seasonal (monthly) multiplier
        if self._config.seasonal_multipliers is not None:
            month: int = target_date.month  # 1-12
            seasonal_mult: float = self._config.seasonal_multipliers.get(
                month, 1.0
            )
            effective *= seasonal_mult

        # Ensure non-negative (should always be, but defensive)
        return max(effective, 0.0)

    # ------------------------------------------------------------------
    # Core Sampling Methods
    # ------------------------------------------------------------------

    def sample_order_count(self, target_date: date) -> int:
        """Sample a single day's order count from the Poisson model.

        The effective Poisson λ is ``base_rate × day_of_week_multiplier
        × seasonal_multiplier`` (if configured).  The raw Poisson sample
        is then clipped to ``[min_orders, max_orders]``.

        Args:
            target_date: The business date to sample for.

        Returns:
            Integer order count clipped to the configured bounds.
            Returns 0 for weekend dates (Saturday/Sunday) when their
            multiplier is 0.0.
        """
        effective_lam: float = self._effective_lambda(target_date)

        # Weekend / zero-rate fast path
        if effective_lam == 0.0:
            return 0

        # Sample from Poisson distribution using numpy RNG for reproducibility
        raw_count: int = int(self._rng.poisson(lam=effective_lam))

        # Clip to configured bounds
        clipped: int = int(np.clip(raw_count, self._config.min_orders, self._config.max_orders))
        return clipped

    def sample_order_counts_batch(self, dates: List[date]) -> List[int]:
        """Vectorized batch sampling of order counts for multiple dates.

        Computes effective λ values for all supplied dates as a numpy array,
        samples from Poisson in a single vectorized call, and clips results.
        Designed for ≥ 10,000 samples/second throughput.

        Args:
            dates: List of calendar dates to sample order counts for.

        Returns:
            List of integer order counts, one per input date, each clipped
            to ``[min_orders, max_orders]``.
        """
        if not dates:
            return []

        n: int = len(dates)

        # Build effective lambda array — vectorized computation
        lambdas: np.ndarray = np.zeros(n, dtype=np.float64)
        for idx, d in enumerate(dates):
            lambdas[idx] = self._effective_lambda(d)

        # Poisson sampling: for zero-lambda entries numpy returns 0
        raw_counts: np.ndarray = self._rng.poisson(lam=lambdas)

        # Clip to configured bounds
        clipped: np.ndarray = np.clip(
            raw_counts,
            self._config.min_orders,
            self._config.max_orders,
        )

        return clipped.tolist()

    # ------------------------------------------------------------------
    # Effective Rate Calculation (analytics / monitoring)
    # ------------------------------------------------------------------

    def get_effective_rate(self, target_date: date) -> float:
        """Return the effective Poisson λ for a given calendar date.

        Useful for analytics, monitoring dashboards, and configuration
        validation.  Applies day-of-week and seasonal multipliers without
        actually sampling.

        Args:
            target_date: Calendar date to inspect.

        Returns:
            Effective Poisson λ (float ≥ 0).
        """
        return self._effective_lambda(target_date)

    def get_weekly_distribution(
        self,
        base_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """Return expected order rates for each day of a typical week.

        Maps human-readable day names to their effective Poisson λ values.
        Useful for configuration validation and visual inspection.

        Seasonal multipliers are **not** applied here because this represents
        a generic (seasonless) week.

        Args:
            base_rate: Override base rate.  Uses the configured
                ``config.base_rate`` when ``None``.

        Returns:
            Dictionary ``{"Monday": λ_mon, "Tuesday": λ_tue, …}`` with
            seven entries.
        """
        rate: float = (
            base_rate if base_rate is not None else self._config.base_rate
        )

        distribution: Dict[str, float] = {}
        for day_idx in range(7):
            day_name: str = _DAY_NAMES[day_idx]
            multiplier: float = self._config.day_of_week_multipliers.get(
                day_idx, 0.0
            )
            distribution[day_name] = rate * multiplier

        return distribution

    # ------------------------------------------------------------------
    # Intra-Day Order Timing
    # ------------------------------------------------------------------

    def sample_order_times(
        self,
        count: int,
        business_start_hour: int = 8,
        business_end_hour: int = 17,
    ) -> List[float]:
        """Sample intra-day order arrival times within business hours.

        Given the number of orders that will occur on a day, this method
        generates uniformly distributed arrival times within the business-hour
        window and returns them sorted chronologically.

        Working-hours spec: 8 AM – 5 PM (AAP Section 0.1.1 F-004).

        Args:
            count: Number of orders to place during the day.  If ≤ 0, an
                empty list is returned.
            business_start_hour: Start of business hours (inclusive), default 8.
            business_end_hour: End of business hours (exclusive), default 17.

        Returns:
            Sorted list of float hours (e.g., ``9.5`` = 9:30 AM).
            Length equals ``max(count, 0)``.
        """
        if count <= 0:
            return []

        if business_start_hour >= business_end_hour:
            logger.warning(
                "invalid_business_hours",
                start=business_start_hour,
                end=business_end_hour,
            )
            return []

        # Uniform sampling within [start, end) and sort chronologically
        times: np.ndarray = self._rng.uniform(
            low=float(business_start_hour),
            high=float(business_end_hour),
            size=count,
        )
        times.sort()

        return times.tolist()

    # ------------------------------------------------------------------
    # Customer-Specific Frequency
    # ------------------------------------------------------------------

    def sample_customer_order_frequency(
        self,
        customer_type: str = "standard",
    ) -> int:
        """Sample an order count adjusted for customer tier.

        Customer tiers are aligned with the External World Simulation
        entity-pool model (AAP Section 0.1.1 F-005):

        * **strategic** (10% of entities): 2× base rate — high-value
          customers with frequent large orders.
        * **standard** (30%): 1× base rate — typical order patterns.
        * **transactional** (60%): 0.5× base rate — low-frequency,
          smaller orders.

        The adjusted rate is used as the Poisson λ, and the result is
        clipped to ``[min_orders, max_orders]``.

        Args:
            customer_type: One of ``"strategic"``, ``"standard"``, or
                ``"transactional"``.  Unrecognised values fall back to
                ``"standard"`` with a warning log.

        Returns:
            Integer order count for the customer, clipped to bounds.
        """
        type_lower: str = customer_type.lower().strip()

        multiplier: float = _CUSTOMER_TYPE_MULTIPLIERS.get(type_lower, -1.0)
        if multiplier < 0.0:
            logger.warning(
                "unknown_customer_type_fallback_to_standard",
                customer_type=customer_type,
                known_types=list(_CUSTOMER_TYPE_MULTIPLIERS.keys()),
            )
            multiplier = _CUSTOMER_TYPE_MULTIPLIERS["standard"]

        adjusted_rate: float = self._config.base_rate * multiplier

        # Ensure non-negative lambda
        if adjusted_rate <= 0.0:
            return 0

        raw_count: int = int(self._rng.poisson(lam=adjusted_rate))

        clipped: int = int(
            np.clip(raw_count, self._config.min_orders, self._config.max_orders)
        )
        return clipped
