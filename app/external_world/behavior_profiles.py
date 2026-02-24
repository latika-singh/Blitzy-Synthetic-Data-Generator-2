"""
Entity Behavior Profiles — Multi-dimensional behavioral configuration for external entities.

This module defines the ``BehaviorProfile`` data model and associated sub-models that
characterize how external entities (customers, vendors, banks, carriers) interact with
the simulated ERP system.  Each entity is assigned a profile that drives:

* **Payment behavior** (customers): payment timing segment, day-variance from terms,
  short-pay rate, and dispute rate.
* **Order behavior** (customers): order frequency, order-size consistency, and monthly
  seasonality multipliers.
* **Invoice behavior** (vendors): invoice-timing speed, accuracy percentage, and inquiry
  response turnaround.

Profiles are organized into five quality tiers — ``excellent``, ``good``, ``average``,
``poor``, and ``problem`` — and entities are classified into three interaction tiers —
``strategic`` (10 %), ``standard`` (30 %), and ``transactional`` (60 %).

The module is a **pure data-definition layer** with zero internal-app imports; it is
consumed by ``CustomerSimulator``, ``VendorSimulator``, ``ExternalWorldManager``, and
other downstream modules.

All models use **Pydantic V2** ``BaseModel`` for validation at subsystem boundaries,
and all logging uses **structlog** to stdout in structured JSON format.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
import structlog

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ===========================================================================
# Enum-like constants
# ===========================================================================

PAYMENT_SEGMENTS: List[str] = ["early", "prompt", "on_time", "late", "problem"]
"""Valid payment-timing segments aligned with ``PaymentTimingModel``."""

ORDER_FREQUENCIES: List[str] = ["daily", "weekly", "monthly"]
"""Valid order-frequency categories."""

ORDER_SIZE_PATTERNS: List[str] = ["consistent", "variable"]
"""Valid order-size consistency patterns."""

INVOICE_TIMINGS: List[str] = ["immediate", "prompt", "slow"]
"""Valid vendor invoice-timing categories."""

PROFILE_NAMES: List[str] = ["excellent", "good", "average", "poor", "problem"]
"""Named quality profiles applied to entities based on tier and type."""

ENTITY_TYPES: List[str] = ["customer", "vendor", "bank", "carrier"]
"""Supported external-entity categories."""

# ===========================================================================
# Sub-models
# ===========================================================================


class PaymentBehavior(BaseModel):
    """Customer payment-behavior configuration.

    Attributes:
        payment_segment: Timing segment (``early`` | ``prompt`` | ``on_time`` |
            ``late`` | ``problem``).
        payment_variance: Standard-deviation in *days* from the nominal payment
            terms for this segment.
        short_pay_rate: Fraction (0.0–1.0) of invoices that the customer pays
            below the invoiced amount.
        dispute_rate: Fraction (0.0–1.0) of invoices that the customer disputes.
    """

    payment_segment: str = "on_time"
    payment_variance: float = Field(default=3.0, ge=0.0, description="Days variance from payment terms")
    short_pay_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of invoices paid short (0.0–1.0)"
    )
    dispute_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of invoices disputed (0.0–1.0)"
    )

    @field_validator("payment_segment")
    @classmethod
    def _validate_payment_segment(cls, value: str) -> str:
        if value not in PAYMENT_SEGMENTS:
            raise ValueError(
                f"payment_segment must be one of {PAYMENT_SEGMENTS}, got '{value}'"
            )
        return value


class OrderBehavior(BaseModel):
    """Customer order-behavior configuration.

    Attributes:
        order_frequency: How often the customer places orders (``daily`` |
            ``weekly`` | ``monthly``).
        order_size_pattern: Whether order sizes are ``consistent`` or ``variable``.
        seasonality: Monthly multipliers keyed 1–12 (January–December).  A value
            of ``1.0`` means no seasonal effect for that month.
    """

    order_frequency: str = "weekly"
    order_size_pattern: str = "consistent"
    seasonality: Dict[int, float] = Field(
        default_factory=lambda: {m: 1.0 for m in range(1, 13)},
        description="Monthly seasonality multipliers keyed 1-12",
    )

    @field_validator("order_frequency")
    @classmethod
    def _validate_order_frequency(cls, value: str) -> str:
        if value not in ORDER_FREQUENCIES:
            raise ValueError(
                f"order_frequency must be one of {ORDER_FREQUENCIES}, got '{value}'"
            )
        return value

    @field_validator("order_size_pattern")
    @classmethod
    def _validate_order_size_pattern(cls, value: str) -> str:
        if value not in ORDER_SIZE_PATTERNS:
            raise ValueError(
                f"order_size_pattern must be one of {ORDER_SIZE_PATTERNS}, got '{value}'"
            )
        return value

    @field_validator("seasonality")
    @classmethod
    def _validate_seasonality(cls, value: Dict[int, float]) -> Dict[int, float]:
        for month, multiplier in value.items():
            if not (1 <= month <= 12):
                raise ValueError(
                    f"Seasonality keys must be 1-12, got {month}"
                )
            if multiplier <= 0:
                raise ValueError(
                    f"Seasonality multiplier for month {month} must be > 0, got {multiplier}"
                )
        return value


class InvoiceBehavior(BaseModel):
    """Vendor invoice-behavior configuration.

    Attributes:
        invoice_timing: Speed at which the vendor sends invoices after delivery
            (``immediate`` | ``prompt`` | ``slow``).
        invoice_accuracy: Fraction (0.0–1.0) of invoices that are accurate
            (match the PO within tolerance).
        response_time: Business days the vendor takes to respond to inquiries.
    """

    invoice_timing: str = "prompt"
    invoice_accuracy: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Fraction of accurate invoices"
    )
    response_time: int = Field(
        default=5, ge=1, description="Days to respond to inquiries"
    )

    @field_validator("invoice_timing")
    @classmethod
    def _validate_invoice_timing(cls, value: str) -> str:
        if value not in INVOICE_TIMINGS:
            raise ValueError(
                f"invoice_timing must be one of {INVOICE_TIMINGS}, got '{value}'"
            )
        return value


# ===========================================================================
# Main BehaviorProfile model
# ===========================================================================


class BehaviorProfile(BaseModel):
    """Multi-dimensional behavior profile for an external entity.

    Composes ``PaymentBehavior``, ``OrderBehavior``, and ``InvoiceBehavior``
    sub-models, plus top-level metadata (``profile_name`` and ``entity_type``).

    Convenience ``@property`` accessors are provided so that callers can write
    ``profile.payment_segment`` instead of
    ``profile.payment_behavior.payment_segment``, matching the flat-field
    interface described in the specification (README.md lines 490–508).
    """

    profile_name: str = "average"
    entity_type: str = "customer"
    payment_behavior: PaymentBehavior = Field(default_factory=PaymentBehavior)
    order_behavior: OrderBehavior = Field(default_factory=OrderBehavior)
    invoice_behavior: InvoiceBehavior = Field(default_factory=InvoiceBehavior)

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("profile_name")
    @classmethod
    def _validate_profile_name(cls, value: str) -> str:
        if value not in PROFILE_NAMES:
            raise ValueError(
                f"profile_name must be one of {PROFILE_NAMES}, got '{value}'"
            )
        return value

    @field_validator("entity_type")
    @classmethod
    def _validate_entity_type(cls, value: str) -> str:
        if value not in ENTITY_TYPES:
            raise ValueError(
                f"entity_type must be one of {ENTITY_TYPES}, got '{value}'"
            )
        return value

    # ------------------------------------------------------------------
    # Convenience properties — flat access (README.md §490-508)
    # ------------------------------------------------------------------

    @property
    def payment_segment(self) -> str:
        """Shortcut to ``payment_behavior.payment_segment``."""
        return self.payment_behavior.payment_segment

    @property
    def payment_variance(self) -> float:
        """Shortcut to ``payment_behavior.payment_variance``."""
        return self.payment_behavior.payment_variance

    @property
    def short_pay_rate(self) -> float:
        """Shortcut to ``payment_behavior.short_pay_rate``."""
        return self.payment_behavior.short_pay_rate

    @property
    def dispute_rate(self) -> float:
        """Shortcut to ``payment_behavior.dispute_rate``."""
        return self.payment_behavior.dispute_rate

    @property
    def order_frequency(self) -> str:
        """Shortcut to ``order_behavior.order_frequency``."""
        return self.order_behavior.order_frequency

    @property
    def order_size_pattern(self) -> str:
        """Shortcut to ``order_behavior.order_size_pattern``."""
        return self.order_behavior.order_size_pattern

    @property
    def seasonality(self) -> Dict[int, float]:
        """Shortcut to ``order_behavior.seasonality``."""
        return self.order_behavior.seasonality

    @property
    def invoice_timing(self) -> str:
        """Shortcut to ``invoice_behavior.invoice_timing``."""
        return self.invoice_behavior.invoice_timing

    @property
    def invoice_accuracy(self) -> float:
        """Shortcut to ``invoice_behavior.invoice_accuracy``."""
        return self.invoice_behavior.invoice_accuracy

    @property
    def response_time(self) -> int:
        """Shortcut to ``invoice_behavior.response_time``."""
        return self.invoice_behavior.response_time


# ===========================================================================
# Predefined Profile Templates
# ===========================================================================

CUSTOMER_PROFILE_TEMPLATES: Dict[str, BehaviorProfile] = {
    "excellent": BehaviorProfile(
        profile_name="excellent",
        entity_type="customer",
        payment_behavior=PaymentBehavior(
            payment_segment="early",
            payment_variance=1.0,
            short_pay_rate=0.0,
            dispute_rate=0.01,
        ),
        order_behavior=OrderBehavior(
            order_frequency="daily",
            order_size_pattern="consistent",
        ),
        invoice_behavior=InvoiceBehavior(),
    ),
    "good": BehaviorProfile(
        profile_name="good",
        entity_type="customer",
        payment_behavior=PaymentBehavior(
            payment_segment="prompt",
            payment_variance=2.0,
            short_pay_rate=0.01,
            dispute_rate=0.02,
        ),
        order_behavior=OrderBehavior(
            order_frequency="weekly",
            order_size_pattern="consistent",
        ),
        invoice_behavior=InvoiceBehavior(),
    ),
    "average": BehaviorProfile(
        profile_name="average",
        entity_type="customer",
        payment_behavior=PaymentBehavior(
            payment_segment="on_time",
            payment_variance=3.0,
            short_pay_rate=0.03,
            dispute_rate=0.05,
        ),
        order_behavior=OrderBehavior(
            order_frequency="weekly",
            order_size_pattern="variable",
        ),
        invoice_behavior=InvoiceBehavior(),
    ),
    "poor": BehaviorProfile(
        profile_name="poor",
        entity_type="customer",
        payment_behavior=PaymentBehavior(
            payment_segment="late",
            payment_variance=5.0,
            short_pay_rate=0.05,
            dispute_rate=0.10,
        ),
        order_behavior=OrderBehavior(
            order_frequency="monthly",
            order_size_pattern="variable",
        ),
        invoice_behavior=InvoiceBehavior(),
    ),
    "problem": BehaviorProfile(
        profile_name="problem",
        entity_type="customer",
        payment_behavior=PaymentBehavior(
            payment_segment="problem",
            payment_variance=10.0,
            short_pay_rate=0.10,
            dispute_rate=0.20,
        ),
        order_behavior=OrderBehavior(
            order_frequency="monthly",
            order_size_pattern="variable",
        ),
        invoice_behavior=InvoiceBehavior(),
    ),
}
"""Pre-built customer behavior profiles keyed by quality tier name."""

VENDOR_PROFILE_TEMPLATES: Dict[str, BehaviorProfile] = {
    "excellent": BehaviorProfile(
        profile_name="excellent",
        entity_type="vendor",
        payment_behavior=PaymentBehavior(),
        order_behavior=OrderBehavior(),
        invoice_behavior=InvoiceBehavior(
            invoice_timing="immediate",
            invoice_accuracy=0.99,
            response_time=1,
        ),
    ),
    "good": BehaviorProfile(
        profile_name="good",
        entity_type="vendor",
        payment_behavior=PaymentBehavior(),
        order_behavior=OrderBehavior(),
        invoice_behavior=InvoiceBehavior(
            invoice_timing="immediate",
            invoice_accuracy=0.97,
            response_time=2,
        ),
    ),
    "average": BehaviorProfile(
        profile_name="average",
        entity_type="vendor",
        payment_behavior=PaymentBehavior(),
        order_behavior=OrderBehavior(),
        invoice_behavior=InvoiceBehavior(
            invoice_timing="prompt",
            invoice_accuracy=0.95,
            response_time=5,
        ),
    ),
    "poor": BehaviorProfile(
        profile_name="poor",
        entity_type="vendor",
        payment_behavior=PaymentBehavior(),
        order_behavior=OrderBehavior(),
        invoice_behavior=InvoiceBehavior(
            invoice_timing="slow",
            invoice_accuracy=0.90,
            response_time=10,
        ),
    ),
    "problem": BehaviorProfile(
        profile_name="problem",
        entity_type="vendor",
        payment_behavior=PaymentBehavior(),
        order_behavior=OrderBehavior(),
        invoice_behavior=InvoiceBehavior(
            invoice_timing="slow",
            invoice_accuracy=0.80,
            response_time=20,
        ),
    ),
}
"""Pre-built vendor behavior profiles keyed by quality tier name."""


# ===========================================================================
# BehaviorProfileFactory
# ===========================================================================


class BehaviorProfileFactory:
    """Factory for creating and customizing ``BehaviorProfile`` instances.

    Provides:
    * ``create_profile`` — deterministic lookup from predefined templates.
    * ``create_profile_with_variation`` — template + small random perturbation
      on numeric fields for entity-level realism.
    * ``get_available_profiles`` — introspection of registered profiles per
      entity type.
    """

    @staticmethod
    def create_profile(profile_name: str, entity_type: str) -> BehaviorProfile:
        """Return a ``BehaviorProfile`` for *profile_name* and *entity_type*.

        If *entity_type* is ``"customer"`` or ``"vendor"`` and a matching
        template exists, it is returned directly.  For other entity types (or
        unrecognised profile names) a sensible default profile is constructed.

        Args:
            profile_name: One of ``PROFILE_NAMES``.
            entity_type: One of ``ENTITY_TYPES``.

        Returns:
            A fully-populated ``BehaviorProfile`` instance.
        """
        templates: Optional[Dict[str, BehaviorProfile]] = None
        if entity_type == "customer":
            templates = CUSTOMER_PROFILE_TEMPLATES
        elif entity_type == "vendor":
            templates = VENDOR_PROFILE_TEMPLATES

        if templates is not None and profile_name in templates:
            profile = templates[profile_name].model_copy(deep=True)
        else:
            # Construct a generic default profile for banks, carriers, or
            # unrecognised profile names.
            safe_profile_name = profile_name if profile_name in PROFILE_NAMES else "average"
            safe_entity_type = entity_type if entity_type in ENTITY_TYPES else "customer"
            profile = BehaviorProfile(
                profile_name=safe_profile_name,
                entity_type=safe_entity_type,
            )

        logger.debug(
            "behavior_profile_created",
            profile_name=profile.profile_name,
            entity_type=profile.entity_type,
        )
        return profile

    @staticmethod
    def create_profile_with_variation(
        profile_name: str,
        entity_type: str,
        rng: Any = None,
    ) -> BehaviorProfile:
        """Create a profile with ±10 % random perturbation on numeric fields.

        This produces entity-level variation so that two "good" customers do
        not behave identically.

        Args:
            profile_name: One of ``PROFILE_NAMES``.
            entity_type: One of ``ENTITY_TYPES``.
            rng: A ``numpy.random.RandomState`` for reproducibility.  If
                *None* a fresh one is created.

        Returns:
            A perturbed ``BehaviorProfile`` instance.
        """
        base = BehaviorProfileFactory.create_profile(profile_name, entity_type)

        if rng is None:
            try:
                import numpy as np  # noqa: WPS433 — lazy import for optional dep
                rng = np.random.RandomState()
            except ImportError:
                # If numpy is somehow unavailable fall back to unvaried profile.
                logger.warning("numpy_unavailable_for_variation")
                return base

        def _vary(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
            """Apply ±10 % jitter, clamped to *[lo, hi]*."""
            factor = 1.0 + rng.uniform(-0.10, 0.10)
            return max(lo, min(hi, round(value * factor, 6)))

        # --- Payment behavior ---------------------------------------------------
        varied_payment = PaymentBehavior(
            payment_segment=base.payment_behavior.payment_segment,
            payment_variance=max(0.0, round(base.payment_behavior.payment_variance * (1.0 + rng.uniform(-0.10, 0.10)), 2)),
            short_pay_rate=_vary(base.payment_behavior.short_pay_rate),
            dispute_rate=_vary(base.payment_behavior.dispute_rate),
        )

        # --- Order behavior ------------------------------------------------------
        varied_seasonality = {
            m: max(0.01, round(v * (1.0 + rng.uniform(-0.10, 0.10)), 4))
            for m, v in base.order_behavior.seasonality.items()
        }
        varied_order = OrderBehavior(
            order_frequency=base.order_behavior.order_frequency,
            order_size_pattern=base.order_behavior.order_size_pattern,
            seasonality=varied_seasonality,
        )

        # --- Invoice behavior ----------------------------------------------------
        varied_invoice = InvoiceBehavior(
            invoice_timing=base.invoice_behavior.invoice_timing,
            invoice_accuracy=_vary(base.invoice_behavior.invoice_accuracy),
            response_time=max(1, int(round(base.invoice_behavior.response_time * (1.0 + rng.uniform(-0.10, 0.10))))),
        )

        profile = BehaviorProfile(
            profile_name=base.profile_name,
            entity_type=base.entity_type,
            payment_behavior=varied_payment,
            order_behavior=varied_order,
            invoice_behavior=varied_invoice,
        )

        logger.debug(
            "behavior_profile_created_with_variation",
            profile_name=profile.profile_name,
            entity_type=profile.entity_type,
        )
        return profile

    @staticmethod
    def get_available_profiles() -> Dict[str, List[str]]:
        """Return the set of pre-defined profiles available per entity type.

        Returns:
            Mapping of entity-type → list of profile-name strings.

        Example::

            >>> BehaviorProfileFactory.get_available_profiles()
            {
                "customer": ["excellent", "good", "average", "poor", "problem"],
                "vendor": ["excellent", "good", "average", "poor", "problem"],
            }
        """
        return {
            "customer": list(CUSTOMER_PROFILE_TEMPLATES.keys()),
            "vendor": list(VENDOR_PROFILE_TEMPLATES.keys()),
        }


# ===========================================================================
# TierDistribution model
# ===========================================================================


class TierDistribution(BaseModel):
    """Tiered entity distribution configuration.

    Specifies the fraction of entities assigned to each interaction tier.
    The three fractions **must** sum to ``1.0`` (within a tolerance of
    ``0.001``).

    Default values follow the specification (README.md lines 470–474):
    * ``strategic``:     10 %
    * ``standard``:      30 %
    * ``transactional``: 60 %
    """

    strategic: float = Field(
        default=0.10, ge=0.0, le=1.0, description="Fraction of strategic entities"
    )
    standard: float = Field(
        default=0.30, ge=0.0, le=1.0, description="Fraction of standard entities"
    )
    transactional: float = Field(
        default=0.60, ge=0.0, le=1.0, description="Fraction of transactional entities"
    )

    @field_validator("transactional")
    @classmethod
    def _validate_sum_to_one(cls, value: float, info: Any) -> float:
        """Ensure the three tier fractions sum to approximately 1.0."""
        strategic = info.data.get("strategic", 0.10)
        standard = info.data.get("standard", 0.30)
        total = strategic + standard + value
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"strategic + standard + transactional must equal 1.0 "
                f"(got {strategic} + {standard} + {value} = {total})"
            )
        return value

    def get_tier_for_rank(self, rank_percentile: float) -> str:
        """Map a rank percentile (0.0–1.0) to a tier label.

        Lower percentiles correspond to higher-importance tiers::

            0.00 – strategic boundary  → ``"strategic"``
            strategic – strategic+standard  → ``"standard"``
            remainder – 1.00           → ``"transactional"``

        Args:
            rank_percentile: A value in ``[0.0, 1.0]``.

        Returns:
            ``"strategic"``, ``"standard"``, or ``"transactional"``.

        Raises:
            ValueError: If *rank_percentile* is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= rank_percentile <= 1.0:
            raise ValueError(
                f"rank_percentile must be between 0.0 and 1.0, got {rank_percentile}"
            )
        if rank_percentile < self.strategic:
            return "strategic"
        if rank_percentile < self.strategic + self.standard:
            return "standard"
        return "transactional"
