"""
Amount Distribution Models for ERP Transaction Simulation.

This module provides statistical distribution models for generating realistic
ERP transaction amounts using log-normal distributions parameterized to match
real-world financial patterns. It is part of the Statistical Models subsystem
(F-006) of the Synthetic ERP Data Generation Platform (Project 2).

Key capabilities:
- Purchase order amount sampling via log-normal distribution (s=1.2, scale=5000)
- Vendor invoice amount generation with configurable match rates (95% exact match)
- Customer order sizing by tier (small/medium/large) with tier-specific distributions
- Intelligent rounding to business-appropriate precision levels
- Early-payment discount application (e.g., 2/10 Net 30 terms)
- Unified batch sampling interface achieving >=10,000 samples/second

This is a pure-function module with NO internal application dependencies.
All distribution parameters are configurable via the AmountConfig Pydantic V2 model.
Reproducible seeding is supported via numpy.random.RandomState.

References:
    - README.md lines 512-538: Amount Distribution Parameters specification
    - AAP Section 0.5.1 Group 3: Statistical Models definition
    - AAP Section 0.7.1: Pydantic V2 at subsystem boundaries
    - AAP Section 0.7.6: structlog for structured JSON logging
"""

from scipy import stats
import numpy as np
from typing import Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
import structlog

# Module-level structured logger for JSON logging to stdout (AAP Section 0.7.6)
logger = structlog.get_logger(__name__)

# Default customer tier definitions matching README.md specification lines 526-528
_DEFAULT_CUSTOMER_TIERS: Dict[str, Dict[str, float]] = {
    "small": {
        "min": 500.0,
        "max": 5_000.0,
        "shape": 0.8,
        "scale": 1_500.0,
    },
    "medium": {
        "min": 2_000.0,
        "max": 50_000.0,
        "shape": 1.0,
        "scale": 8_000.0,
    },
    "large": {
        "min": 10_000.0,
        "max": 500_000.0,
        "shape": 1.2,
        "scale": 50_000.0,
    },
}


class AmountConfig(BaseModel):
    """Pydantic V2 configuration model for amount distribution parameters.

    Validates and provides defaults for all configurable parameters used by
    the AmountDistribution class. Serves as the subsystem boundary contract
    per AAP Section 0.7.1.

    Attributes:
        min_amount: Global minimum transaction amount in USD. Default $100.
        max_amount: Global maximum transaction amount in USD. Default $500,000.
        po_shape: Log-normal shape parameter (s) for PO distribution. Default 1.2.
        po_scale: Log-normal scale parameter for PO distribution. Default $5,000.
        invoice_match_rate: Fraction of invoices that exactly match PO amount.
            Default 0.95 (95%).
        invoice_variance_pct: Maximum percentage variance for non-matching invoices.
            Default 0.05 (±5%).
        customer_tiers: Dictionary mapping tier names to their distribution
            parameters (min, max, shape, scale). Defaults to small/medium/large
            tiers matching the README.md specification.
    """

    min_amount: float = Field(
        default=100.0,
        ge=0,
        description="Minimum transaction amount in USD ($100)",
    )
    max_amount: float = Field(
        default=500_000.0,
        gt=0,
        description="Maximum transaction amount in USD ($500K)",
    )
    po_shape: float = Field(
        default=1.2,
        gt=0,
        description="Log-normal shape parameter s for purchase orders",
    )
    po_scale: float = Field(
        default=5_000.0,
        gt=0,
        description="Log-normal scale parameter for purchase orders",
    )
    invoice_match_rate: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Fraction of invoices matching PO amount exactly (95%)",
    )
    invoice_variance_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum percentage variance for non-matching invoices (±5%)",
    )
    customer_tiers: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: _DEFAULT_CUSTOMER_TIERS.copy(),
        description="Tier definitions mapping tier name to {min, max, shape, scale}",
    )


class AmountDistribution:
    """Statistical models for generating realistic ERP transaction amounts.

    Implements log-normal distributions for purchase orders, vendor invoice
    matching logic, and tier-based customer order sizing. All sampling uses
    numpy.random.RandomState for reproducible results across simulation runs.

    The distribution parameters match real-world ERP financial patterns:
    - Purchase orders follow a log-normal with s=1.2, scale=$5,000
    - 95% of vendor invoices match the PO amount exactly; 5% have ±5% variance
    - Customer orders are sized by tier (small/medium/large) with tier-specific
      log-normal distributions

    Performance target: >=10,000 samples/second via vectorized numpy operations.

    Attributes:
        config: AmountConfig instance with all distribution parameters.
        rng: numpy RandomState for reproducible sampling.

    Example:
        >>> dist = AmountDistribution(seed=42)
        >>> po_amount = dist.sample_po_amount()
        >>> invoice_amount = dist.sample_vendor_invoice_amount(po_amount)
        >>> order_amount = dist.sample_customer_order_amount(tier="medium")
    """

    def __init__(
        self,
        config: Optional[AmountConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the AmountDistribution with configuration and random seed.

        Args:
            config: Optional AmountConfig with distribution parameters. If None,
                defaults are used (PO: s=1.2, scale=5000; tiers: small/medium/large).
            seed: Optional integer seed for numpy.random.RandomState to enable
                reproducible sampling across simulation runs.
        """
        self.config = config if config is not None else AmountConfig()
        self.rng = np.random.RandomState(seed)

        # Build the purchase order log-normal distribution
        # Per README.md line 520: stats.lognorm(s=1.2, scale=5000)
        self._po_distribution = stats.lognorm(
            s=self.config.po_shape,
            scale=self.config.po_scale,
        )

        # Build customer tier distributions from configuration
        # Each tier maps to a scipy log-normal frozen distribution and its clip bounds
        self._tier_distributions: Dict[str, Dict[str, Any]] = {}
        for tier_name, tier_params in self.config.customer_tiers.items():
            shape = tier_params.get("shape", 1.0)
            scale = tier_params.get("scale", 5_000.0)
            tier_min = tier_params.get("min", self.config.min_amount)
            tier_max = tier_params.get("max", self.config.max_amount)
            self._tier_distributions[tier_name] = {
                "distribution": stats.lognorm(s=shape, scale=scale),
                "min": tier_min,
                "max": tier_max,
            }

        logger.info(
            "amount_distribution_initialized",
            po_shape=self.config.po_shape,
            po_scale=self.config.po_scale,
            min_amount=self.config.min_amount,
            max_amount=self.config.max_amount,
            invoice_match_rate=self.config.invoice_match_rate,
            customer_tiers=list(self._tier_distributions.keys()),
            seed=seed,
        )

    def sample_po_amount(self, n: int = 1) -> Union[np.ndarray, float]:
        """Sample purchase order amounts from the log-normal distribution.

        Generates realistic PO amounts using a log-normal distribution with
        parameters s=1.2, scale=$5,000 (configurable). Results are clipped
        to [min_amount, max_amount] and rounded to business-appropriate precision.

        Performance: vectorized via scipy.stats.rvs for >=10,000 samples/second.

        Args:
            n: Number of samples to generate. Default 1.

        Returns:
            If n == 1, returns a single float amount.
            If n > 1, returns a numpy ndarray of float amounts.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError(f"Sample count n must be >= 1, got {n}")

        # Vectorized sampling from the frozen log-normal distribution
        raw_samples = self._po_distribution.rvs(size=n, random_state=self.rng)

        # Clip to configured amount bounds
        clipped = np.clip(raw_samples, self.config.min_amount, self.config.max_amount)

        # Apply business-appropriate rounding to each sample
        if n == 1:
            scalar_val = float(clipped.item()) if isinstance(clipped, np.ndarray) else float(clipped)
            return self.round_to_nearest(scalar_val)

        # Vectorized rounding for batch samples
        rounded = np.array([self.round_to_nearest(float(val)) for val in clipped])
        return rounded

    def sample_vendor_invoice_amount(self, po_amount: float) -> float:
        """Generate a vendor invoice amount based on a purchase order amount.

        Per README.md lines 522-524:
        - 95% of invoices match the PO amount exactly (configurable match rate)
        - 5% of invoices have variances (quantity or price differences) within ±5%

        The variance is sampled uniformly from [-invoice_variance_pct, +invoice_variance_pct].

        Args:
            po_amount: The purchase order amount in USD that the invoice references.

        Returns:
            The vendor invoice amount in USD, clipped to [min_amount, max_amount].

        Raises:
            ValueError: If po_amount is negative.
        """
        if po_amount < 0:
            raise ValueError(f"PO amount must be non-negative, got {po_amount}")

        # Determine whether this invoice matches the PO exactly
        match_roll = self.rng.random()

        if match_roll < self.config.invoice_match_rate:
            # 95% case: exact match — invoice equals PO amount
            invoice_amount = po_amount
        else:
            # 5% case: introduce a variance within ±invoice_variance_pct
            variance_factor = self.rng.uniform(
                -self.config.invoice_variance_pct,
                self.config.invoice_variance_pct,
            )
            invoice_amount = po_amount * (1.0 + variance_factor)

        # Clip to global amount bounds
        invoice_amount = float(
            np.clip(invoice_amount, self.config.min_amount, self.config.max_amount)
        )

        return invoice_amount

    def sample_customer_order_amount(
        self, tier: str = "medium", n: int = 1
    ) -> Union[np.ndarray, float]:
        """Sample customer order amounts based on customer size tier.

        Per README.md lines 526-528, customer orders follow tier-specific
        log-normal distributions:
        - Small:  s=0.8, scale=$1,500, range $500–$5,000
        - Medium: s=1.0, scale=$8,000, range $2,000–$50,000
        - Large:  s=1.2, scale=$50,000, range $10,000–$500,000

        If the specified tier is not found in configuration, falls back to "medium".

        Args:
            tier: Customer size tier name ("small", "medium", or "large").
                Defaults to "medium".
            n: Number of samples to generate. Default 1.

        Returns:
            If n == 1, returns a single float amount.
            If n > 1, returns a numpy ndarray of float amounts.

        Raises:
            ValueError: If n < 1.
        """
        if n < 1:
            raise ValueError(f"Sample count n must be >= 1, got {n}")

        # Look up tier; fall back to "medium" if not found
        tier_key = tier.lower()
        if tier_key not in self._tier_distributions:
            logger.warning(
                "unknown_customer_tier_falling_back",
                requested_tier=tier,
                fallback_tier="medium",
                available_tiers=list(self._tier_distributions.keys()),
            )
            tier_key = "medium"

        tier_data = self._tier_distributions[tier_key]
        distribution = tier_data["distribution"]
        tier_min = tier_data["min"]
        tier_max = tier_data["max"]

        # Vectorized sampling from the tier-specific log-normal distribution
        raw_samples = distribution.rvs(size=n, random_state=self.rng)

        # Clip to tier-specific bounds
        clipped = np.clip(raw_samples, tier_min, tier_max)

        # Apply rounding
        if n == 1:
            scalar_val = float(clipped.item()) if isinstance(clipped, np.ndarray) else float(clipped)
            return self.round_to_nearest(scalar_val)

        rounded = np.array([self.round_to_nearest(float(val)) for val in clipped])
        return rounded

    def round_to_nearest(
        self, amount: float, precision: Optional[int] = None
    ) -> float:
        """Round an amount to the nearest business-appropriate precision level.

        Per README.md lines 530-533:
        - $4,847.32 → $4,800 (round to $100 for amounts in $1K–$10K range)
        - $124,389  → $125,000 (round to $5,000 for amounts >= $100K)

        Auto-precision tiers when precision is None:
        - amount < $1,000:   round to nearest $10
        - amount < $10,000:  round to nearest $100
        - amount < $100,000: round to nearest $1,000
        - amount >= $100,000: round to nearest $5,000

        The result is guaranteed to be at least min_amount.

        Args:
            amount: The raw amount in USD to round.
            precision: Optional explicit rounding precision. If None, auto-determined
                from amount magnitude.

        Returns:
            The rounded amount as a float, guaranteed >= min_amount.
        """
        if precision is None:
            # Auto-determine precision based on amount magnitude
            abs_amount = abs(amount)
            if abs_amount < 1_000.0:
                precision = 10
            elif abs_amount < 10_000.0:
                precision = 100
            elif abs_amount < 100_000.0:
                precision = 1_000
            else:
                precision = 5_000

        if precision <= 0:
            precision = 1

        # Standard rounding to nearest precision unit
        rounded = round(amount / precision) * precision

        # Ensure result respects the configured minimum amount
        rounded = max(rounded, self.config.min_amount)

        return float(rounded)

    def apply_discount(
        self,
        amount: float,
        discount_rate: float,
        days_to_pay: int,
        discount_days: int = 10,
    ) -> float:
        """Apply an early-payment discount to a transaction amount.

        Per README.md lines 535-537:
        - Common terms: 2/10 Net 30 — 2% discount if paid within 10 days
        - If payment is made within the discount period, the discount is applied
        - Otherwise, the full amount is due

        Args:
            amount: The original transaction amount in USD.
            discount_rate: The discount rate as a decimal (e.g., 0.02 for 2%).
            days_to_pay: Number of days until payment is made.
            discount_days: Maximum days to qualify for discount. Default 10.

        Returns:
            The discounted amount (if eligible) or the full amount, rounded
            to business-appropriate precision.

        Raises:
            ValueError: If amount is negative, discount_rate is outside [0, 1],
                or days_to_pay is negative.
        """
        if amount < 0:
            raise ValueError(f"Amount must be non-negative, got {amount}")
        if not 0.0 <= discount_rate <= 1.0:
            raise ValueError(
                f"Discount rate must be between 0.0 and 1.0, got {discount_rate}"
            )
        if days_to_pay < 0:
            raise ValueError(f"Days to pay must be non-negative, got {days_to_pay}")

        if days_to_pay <= discount_days:
            # Payment qualifies for early-payment discount
            discounted = amount * (1.0 - discount_rate)
        else:
            # Full amount is due (no discount)
            discounted = amount

        # Round the result to business-appropriate precision
        return self.round_to_nearest(discounted)

    def sample_batch(
        self,
        distribution_type: str,
        n: int,
        **kwargs: Any,
    ) -> np.ndarray:
        """Unified batch sampling interface for all distribution types.

        Provides a single entry point for generating batches of amounts from
        any supported distribution. Optimized for >=10,000 samples/second
        via delegation to the vectorized sample_* methods.

        Args:
            distribution_type: One of "purchase_order", "vendor_invoice",
                or "customer_order".
            n: Number of samples to generate. Must be >= 1.
            **kwargs: Additional keyword arguments passed to the underlying
                sampling method:
                - For "purchase_order": no extra kwargs needed.
                - For "vendor_invoice": requires 'po_amount' (float) or
                  'po_amounts' (array-like of floats).
                - For "customer_order": optional 'tier' (str, default "medium").

        Returns:
            A numpy ndarray of n sampled amounts.

        Raises:
            ValueError: If distribution_type is not recognized or n < 1.
        """
        if n < 1:
            raise ValueError(f"Sample count n must be >= 1, got {n}")

        if distribution_type == "purchase_order":
            # Delegate to vectorized PO sampling
            result = self.sample_po_amount(n=n)
            if isinstance(result, (int, float)):
                return np.array([result])
            return np.asarray(result)

        elif distribution_type == "vendor_invoice":
            # For vendor invoices, we need PO amounts to match against
            po_amounts = kwargs.get("po_amounts")
            po_amount = kwargs.get("po_amount")

            if po_amounts is not None:
                # Batch: generate invoice for each PO amount
                po_arr = np.asarray(po_amounts)
                results = np.array(
                    [self.sample_vendor_invoice_amount(float(pa)) for pa in po_arr[:n]]
                )
                # Pad with additional samples if po_amounts is shorter than n
                if len(results) < n:
                    extra = np.array(
                        [
                            self.sample_vendor_invoice_amount(
                                float(po_arr[i % len(po_arr)])
                            )
                            for i in range(len(results), n)
                        ]
                    )
                    results = np.concatenate([results, extra])
                return results
            elif po_amount is not None:
                # Single PO amount: generate n invoices from it
                return np.array(
                    [self.sample_vendor_invoice_amount(float(po_amount)) for _ in range(n)]
                )
            else:
                # No PO amount provided: sample PO amounts first, then invoice
                po_samples = self.sample_po_amount(n=n)
                if isinstance(po_samples, (int, float)):
                    po_samples = np.array([po_samples])
                return np.array(
                    [
                        self.sample_vendor_invoice_amount(float(pa))
                        for pa in po_samples
                    ]
                )

        elif distribution_type == "customer_order":
            tier = kwargs.get("tier", "medium")
            result = self.sample_customer_order_amount(tier=tier, n=n)
            if isinstance(result, (int, float)):
                return np.array([result])
            return np.asarray(result)

        else:
            raise ValueError(
                f"Unknown distribution_type '{distribution_type}'. "
                f"Supported types: 'purchase_order', 'vendor_invoice', 'customer_order'"
            )
