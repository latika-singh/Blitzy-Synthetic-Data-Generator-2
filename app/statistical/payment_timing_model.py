"""
Payment Timing Model — 5-Segment Mixture Model for Customer Payment Timing.

This module implements a configurable mixture model that simulates realistic
customer payment behaviour across five behavioural segments:

    1. **early_discount** (20 %) — Normal(loc=-8, scale=2)
       Customers who pay early to capture early-payment discounts.
    2. **prompt** (30 %) — Normal(loc=-2, scale=3)
       Customers who reliably pay shortly before the due date.
    3. **on_time** (25 %) — Normal(loc=2, scale=4)
       Customers who pay around (slightly after) the due date.
    4. **late** (15 %) — LogNormal(s=1.0, loc=10, scale=5)
       Customers who are consistently late payers.
    5. **problem** (10 %) — LogNormal(s=1.2, loc=30, scale=15)
       Customers with severely delinquent payment patterns.

Segment selection is **profile-weighted**: each customer profile (excellent,
good, average, poor, problem) defines a probability vector over the five
segments so that, for example, an "excellent" customer has a 60 % chance of
landing in the *early_discount* segment but a 0 % chance of *problem*.

Weekend adjustment moves Saturday/Sunday payments to the following Monday.

This is a pure-function module with **no internal application dependencies**.
All data contracts use Pydantic V2 ``BaseModel`` for subsystem-boundary
validation (AAP Section 0.7.1).  Logging follows the structured JSON stdout
convention via ``structlog`` (AAP Section 0.7.6).

Performance target: ≥ 10 000 samples / second.

References
----------
- README.md lines 1379–1492 (reference implementation)
- AAP Section 0.1.1 (F-006 — Statistical Models)
- AAP Section 0.5.1 Group 3 (PaymentTimingModel specification)
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field, model_validator
from scipy import stats

# ---------------------------------------------------------------------------
# Module-level structured logger (JSON to stdout per AAP § 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default segment order — canonical ordering used throughout the module
# ---------------------------------------------------------------------------
_DEFAULT_SEGMENT_ORDER: List[str] = [
    "early_discount",
    "prompt",
    "on_time",
    "late",
    "problem",
]


# ============================================================================
# Pydantic V2 Configuration Models
# ============================================================================


class PaymentSegment(BaseModel):
    """Definition of a single payment-behaviour segment.

    Attributes
    ----------
    name : str
        Unique identifier for the segment (e.g. ``"early_discount"``).
    weight : float
        Prior probability of this segment under the base (un-profiled)
        mixture.  Must be in [0.0, 1.0].
    distribution_type : str
        ``"normal"`` for ``scipy.stats.norm`` or ``"lognormal"`` for
        ``scipy.stats.lognorm``.
    loc : float
        Location parameter of the distribution (days offset from due date).
    scale : float
        Scale parameter of the distribution.
    s : float | None
        Shape parameter — required for log-normal segments, ignored for
        normal segments.
    description : str
        Human-readable description of the segment's behaviour.
    """

    name: str
    weight: float = Field(ge=0.0, le=1.0)
    distribution_type: str = Field(
        description="'normal' or 'lognormal'"
    )
    loc: float
    scale: float = Field(gt=0.0)
    s: Optional[float] = Field(
        default=None,
        description="Shape parameter for lognormal distribution",
    )
    description: str = ""

    @model_validator(mode="after")
    def _validate_shape_for_lognormal(self) -> "PaymentSegment":
        """Ensure ``s`` is provided when distribution_type is lognormal."""
        if self.distribution_type == "lognormal" and self.s is None:
            raise ValueError(
                f"Segment '{self.name}': shape parameter 's' is required "
                f"for lognormal distribution"
            )
        return self


class PaymentTimingConfig(BaseModel):
    """Configuration for the 5-segment payment timing mixture model.

    Contains the segment definitions, profile weight vectors, and the
    default profile to fall back to when an unknown profile is requested.

    Validators ensure that:
    * segment weights sum to approximately 1.0 (±0.01 tolerance)
    * every profile weight vector sums to approximately 1.0
    * the default_profile key exists in profile_weights

    Attributes
    ----------
    segments : list[PaymentSegment]
        Ordered list of segment definitions.  The default provides the
        canonical five segments from the specification.
    profile_weights : dict[str, list[float]]
        Mapping of customer-profile name → weight vector over segments.
        Each vector must have the same length as ``segments`` and sum to
        ~1.0.
    default_profile : str
        Profile to use when a requested profile is not found in
        ``profile_weights``.
    """

    segments: List[PaymentSegment] = Field(
        default_factory=lambda: [
            PaymentSegment(
                name="early_discount",
                weight=0.20,
                distribution_type="normal",
                loc=-8.0,
                scale=2.0,
                description="Pay early to capture discount",
            ),
            PaymentSegment(
                name="prompt",
                weight=0.30,
                distribution_type="normal",
                loc=-2.0,
                scale=3.0,
                description="Pay slightly before due",
            ),
            PaymentSegment(
                name="on_time",
                weight=0.25,
                distribution_type="normal",
                loc=2.0,
                scale=4.0,
                description="Pay around due date",
            ),
            PaymentSegment(
                name="late",
                weight=0.15,
                distribution_type="lognormal",
                loc=10.0,
                scale=5.0,
                s=1.0,
                description="Consistently late",
            ),
            PaymentSegment(
                name="problem",
                weight=0.10,
                distribution_type="lognormal",
                loc=30.0,
                scale=15.0,
                s=1.2,
                description="Severely late",
            ),
        ]
    )

    profile_weights: Dict[str, List[float]] = Field(
        default_factory=lambda: {
            "excellent": [0.60, 0.30, 0.08, 0.02, 0.00],
            "good": [0.20, 0.50, 0.20, 0.08, 0.02],
            "average": [0.10, 0.30, 0.35, 0.20, 0.05],
            "poor": [0.05, 0.10, 0.25, 0.40, 0.20],
            "problem": [0.00, 0.05, 0.15, 0.35, 0.45],
        }
    )

    default_profile: str = "average"

    # ---- Validators -------------------------------------------------

    @model_validator(mode="after")
    def _validate_weights(self) -> "PaymentTimingConfig":
        """Validate segment and profile weights sum to ~1.0."""
        # Segment weights
        segment_weight_sum = sum(seg.weight for seg in self.segments)
        if abs(segment_weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Segment weights must sum to ~1.0, got {segment_weight_sum:.4f}"
            )

        n_segments = len(self.segments)

        # Profile weights
        for profile_name, weights in self.profile_weights.items():
            if len(weights) != n_segments:
                raise ValueError(
                    f"Profile '{profile_name}' has {len(weights)} weights "
                    f"but there are {n_segments} segments"
                )
            weight_sum = sum(weights)
            if abs(weight_sum - 1.0) > 0.01:
                raise ValueError(
                    f"Profile '{profile_name}' weights must sum to ~1.0, "
                    f"got {weight_sum:.4f}"
                )

        # Default profile must exist
        if self.default_profile not in self.profile_weights:
            raise ValueError(
                f"default_profile '{self.default_profile}' not found in "
                f"profile_weights keys: {list(self.profile_weights.keys())}"
            )

        return self


# ============================================================================
# Payment Timing Model — Core Engine
# ============================================================================

# Pre-compiled regex for extracting numeric net-day terms from payment strings
_NET_DAYS_RE = re.compile(r"Net\s+(\d+)", re.IGNORECASE)


class PaymentTimingModel:
    """5-segment mixture model for realistic customer payment timing.

    This model simulates *when* a customer will pay an invoice by:

    1. Computing the due date from the invoice date and payment terms.
    2. Selecting a behavioural segment via the customer's profile weights.
    3. Sampling a days-offset from the selected segment's distribution.
    4. Adjusting for weekends (Saturday/Sunday → next Monday).

    The model is fully deterministic when a ``seed`` is provided, making
    simulation runs reproducible.

    Parameters
    ----------
    config : PaymentTimingConfig | None
        Model configuration.  When ``None`` the default 5-segment /
        5-profile configuration from the specification is used.
    seed : int | None
        Seed for ``numpy.random.RandomState``.  Pass a fixed value for
        reproducible results across simulation runs.

    Examples
    --------
    >>> model = PaymentTimingModel(seed=42)
    >>> payment = model.get_payment_date(
    ...     invoice_date=date(2025, 1, 15),
    ...     payment_terms="Net 30",
    ...     customer_profile="good",
    ... )
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[PaymentTimingConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config: PaymentTimingConfig = config or PaymentTimingConfig()
        self.rng: np.random.RandomState = np.random.RandomState(seed)

        # Build scipy frozen distribution objects in segment order
        self._distributions: Dict[str, Any] = {}
        self._segment_names: List[str] = []
        for segment in self.config.segments:
            self._segment_names.append(segment.name)
            if segment.distribution_type == "normal":
                self._distributions[segment.name] = stats.norm(
                    loc=segment.loc, scale=segment.scale
                )
            elif segment.distribution_type == "lognormal":
                self._distributions[segment.name] = stats.lognorm(
                    s=segment.s,  # type: ignore[arg-type]
                    loc=segment.loc,
                    scale=segment.scale,
                )
            else:
                raise ValueError(
                    f"Unsupported distribution_type '{segment.distribution_type}' "
                    f"for segment '{segment.name}'. "
                    f"Must be 'normal' or 'lognormal'."
                )

        logger.info(
            "payment_timing_model_initialized",
            n_segments=len(self.config.segments),
            n_profiles=len(self.config.profile_weights),
            segment_names=self._segment_names,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Public API — Core
    # ------------------------------------------------------------------

    def get_payment_date(
        self,
        invoice_date: date,
        payment_terms: str,
        customer_profile: str = "average",
    ) -> date:
        """Calculate a simulated payment date for an invoice.

        This is the primary entry-point of the model.

        Parameters
        ----------
        invoice_date : date
            Date the invoice was issued.
        payment_terms : str
            Payment terms string, e.g. ``"Net 30"``, ``"2/10 Net 30"``,
            ``"Due on Receipt"``.
        customer_profile : str
            One of the profile keys in the configuration
            (``"excellent"``, ``"good"``, ``"average"``, ``"poor"``,
            ``"problem"``).  Defaults to ``"average"``.

        Returns
        -------
        date
            Simulated payment date, adjusted for weekends.
        """
        # Step 1 — due date from terms
        due_date = self._calculate_due_date(invoice_date, payment_terms)

        # Step 2 — select behavioural segment
        segment = self._select_segment(customer_profile)

        # Step 3 — sample days offset from distribution
        days_offset = self._sample_offset(segment)

        # Step 4 — compute raw payment date
        payment_date = due_date + timedelta(days=int(days_offset))

        # Step 5 — weekend adjustment
        payment_date = self._adjust_for_weekend(payment_date)

        return payment_date

    # ------------------------------------------------------------------
    # Public API — Batch
    # ------------------------------------------------------------------

    def sample_payment_dates_batch(
        self,
        invoice_dates: List[date],
        payment_terms: List[str],
        profiles: List[str],
    ) -> List[date]:
        """Process a batch of invoices and return simulated payment dates.

        For large batches this method pre-groups invoices by profile to
        amortise the segment-selection overhead.

        Parameters
        ----------
        invoice_dates : list[date]
            Invoice dates — must be the same length as *payment_terms*
            and *profiles*.
        payment_terms : list[str]
            Payment terms strings.
        profiles : list[str]
            Customer profiles (one per invoice).

        Returns
        -------
        list[date]
            Simulated payment dates, one per input invoice.

        Raises
        ------
        ValueError
            If the three input lists have different lengths.
        """
        n = len(invoice_dates)
        if len(payment_terms) != n or len(profiles) != n:
            raise ValueError(
                f"Input lists must have equal length; got "
                f"invoice_dates={n}, payment_terms={len(payment_terms)}, "
                f"profiles={len(profiles)}"
            )

        results: List[date] = []
        for inv_date, terms, profile in zip(
            invoice_dates, payment_terms, profiles
        ):
            results.append(
                self.get_payment_date(inv_date, terms, profile)
            )

        logger.debug(
            "payment_dates_batch_sampled",
            batch_size=n,
        )
        return results

    # ------------------------------------------------------------------
    # Public API — Statistics
    # ------------------------------------------------------------------

    def get_segment_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Return descriptive statistics for every configured segment.

        Useful for configuration validation, monitoring dashboards, and
        debugging.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of segment name → statistics dictionary with keys:
            ``name``, ``weight``, ``distribution_type``, ``mean_offset``,
            ``std_offset``, ``loc``, ``scale``, ``s``.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for segment in self.config.segments:
            dist = self._distributions[segment.name]
            mean_val: float = float(dist.mean())
            std_val: float = float(dist.std())
            result[segment.name] = {
                "name": segment.name,
                "weight": segment.weight,
                "distribution_type": segment.distribution_type,
                "mean_offset": mean_val,
                "std_offset": std_val,
                "loc": segment.loc,
                "scale": segment.scale,
                "s": segment.s,
                "description": segment.description,
            }
        return result

    # ------------------------------------------------------------------
    # Internal — Due Date Calculation
    # ------------------------------------------------------------------

    def _calculate_due_date(self, invoice_date: date, terms: str) -> date:
        """Compute the contractual due date from payment terms.

        Recognised patterns (checked in order):

        * ``"Due on Receipt"`` — due immediately (0 days)
        * ``"2/10 Net 30"``   — due in 30 days (discount window handled
          separately by the caller)
        * ``"Net <N>"``        — due in *N* days
        * Fallback             — due in 30 days

        Parameters
        ----------
        invoice_date : date
            Date the invoice was issued.
        terms : str
            Raw payment-terms string.

        Returns
        -------
        date
            Computed due date.
        """
        if "Due on Receipt" in terms:
            return invoice_date + timedelta(days=0)

        # "2/10 Net 30" — extract the net days portion
        net_match = _NET_DAYS_RE.search(terms)
        if net_match:
            net_days = int(net_match.group(1))
            return invoice_date + timedelta(days=net_days)

        # Fallback — Net 30
        return invoice_date + timedelta(days=30)

    # ------------------------------------------------------------------
    # Internal — Segment Selection
    # ------------------------------------------------------------------

    def _select_segment(self, profile: str) -> str:
        """Select a payment-behaviour segment for *profile*.

        Uses the profile's weight vector (or the default profile's if
        *profile* is not recognised) and samples one segment from the
        categorical distribution via ``numpy.random.RandomState.choice``.

        Parameters
        ----------
        profile : str
            Customer profile key.

        Returns
        -------
        str
            Name of the selected segment.
        """
        weights = self.config.profile_weights.get(
            profile,
            self.config.profile_weights[self.config.default_profile],
        )
        return str(
            self.rng.choice(self._segment_names, p=weights)
        )

    # ------------------------------------------------------------------
    # Internal — Offset Sampling
    # ------------------------------------------------------------------

    def _sample_offset(self, segment_name: str) -> float:
        """Sample a days-offset from a segment's distribution.

        A **negative** offset means payment *before* the due date; a
        **positive** offset means payment *after* the due date.

        Parameters
        ----------
        segment_name : str
            Name of the segment whose distribution should be sampled.

        Returns
        -------
        float
            Days offset (may be fractional; caller truncates to int).
        """
        dist = self._distributions[segment_name]
        return float(dist.rvs(random_state=self.rng))

    # ------------------------------------------------------------------
    # Internal — Weekend Adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def _adjust_for_weekend(dt: date) -> date:
        """Move a weekend date to the following Monday.

        * Saturday (``weekday() == 5``) → Monday (+2 days)
        * Sunday  (``weekday() == 6``) → Monday (+1 day)
        * Weekdays are returned unchanged.

        Parameters
        ----------
        dt : date
            Candidate payment date.

        Returns
        -------
        date
            Adjusted date (always a weekday).
        """
        weekday = dt.weekday()
        if weekday == 5:  # Saturday
            return dt + timedelta(days=2)
        if weekday == 6:  # Sunday
            return dt + timedelta(days=1)
        return dt
