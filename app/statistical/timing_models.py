"""
Approval and Processing Timing Models for ERP Workflow Durations.

This module provides statistical timing models for non-payment-related ERP workflow
operations, including approval processing times and general transaction processing
durations. These models are distinct from the payment timing model (which handles
customer payment behavior) and focus on internal operational timing.

Key models:
    - ApprovalTimingModel: Normal(μ=4hrs, σ=2hrs) distribution for approval delays,
      with +1 business day (8 hours) escalation surcharge.
    - ProcessingTimingModel: Normal distributions for invoice processing, PO creation,
      and goods receipt durations.

All models use numpy.random.RandomState for reproducible seeding and scipy.stats.norm
for distribution sampling. Configuration is validated via Pydantic V2 BaseModel classes
at all subsystem boundaries.

Performance target: ≥ 10,000 samples/second for batch operations.

This is a pure-function module with NO internal application dependencies.

References:
    - README.md lines 560-563: Approval processing time specification
    - AAP Section 0.5.1 Group 3: Statistical Models layer
    - AAP Section 0.7.1: Pydantic V2 for subsystem boundary contracts
    - AAP Section 0.7.6: structlog for structured JSON logging
"""

from scipy import stats
import numpy as np
from datetime import timedelta
from typing import Optional, Dict, List

from pydantic import BaseModel, Field
import structlog

# Module-level structured logger for initialization and sampling operations
logger = structlog.get_logger(__name__)


class ApprovalTimingConfig(BaseModel):
    """Configuration for approval processing time distribution.

    Models the time required for an approver to review and process an approval
    request using a Normal distribution. Escalated items incur an additional
    delay representing the overhead of routing to a higher authority.

    Attributes:
        mean_hours: Mean approval processing time in hours (μ = 4.0).
        std_hours: Standard deviation of approval time in hours (σ = 2.0).
        min_hours: Minimum approval time floor (0.5 hours = 30 minutes).
        max_hours: Maximum approval time ceiling (48 hours = 2 business days).
        escalation_additional_hours: Extra hours added for escalated items
            (8.0 hours = 1 business day).
        business_hours_per_day: Number of working hours per business day (8.0).
    """

    mean_hours: float = Field(
        default=4.0,
        gt=0,
        description="Mean approval processing time in hours (μ parameter)",
    )
    std_hours: float = Field(
        default=2.0,
        gt=0,
        description="Standard deviation of approval time in hours (σ parameter)",
    )
    min_hours: float = Field(
        default=0.5,
        ge=0,
        description="Minimum approval processing time in hours (30 minutes floor)",
    )
    max_hours: float = Field(
        default=48.0,
        gt=0,
        description="Maximum approval processing time in hours (2 business day ceiling)",
    )
    escalation_additional_hours: float = Field(
        default=8.0,
        ge=0,
        description="Additional hours for escalated approvals (+1 business day)",
    )
    business_hours_per_day: float = Field(
        default=8.0,
        gt=0,
        description="Number of working hours per business day",
    )


class ProcessingTimingConfig(BaseModel):
    """Configuration for general ERP workflow processing time distributions.

    Defines Normal distribution parameters for common ERP processing operations
    including invoice processing, purchase order creation, and goods receipt
    handling. Each operation has independent mean and standard deviation
    parameters reflecting its typical duration profile.

    Attributes:
        invoice_processing_mean_hours: Mean time to process a vendor invoice (2.0 hrs).
        invoice_processing_std_hours: Std dev for invoice processing (1.0 hrs).
        po_creation_mean_hours: Mean time to create a purchase order (1.5 hrs).
        po_creation_std_hours: Std dev for PO creation (0.75 hrs).
        goods_receipt_mean_hours: Mean time to process goods receipt (3.0 hrs).
        goods_receipt_std_hours: Std dev for goods receipt processing (1.5 hrs).
    """

    invoice_processing_mean_hours: float = Field(
        default=2.0,
        gt=0,
        description="Mean invoice processing time in hours",
    )
    invoice_processing_std_hours: float = Field(
        default=1.0,
        gt=0,
        description="Standard deviation of invoice processing time in hours",
    )
    po_creation_mean_hours: float = Field(
        default=1.5,
        gt=0,
        description="Mean purchase order creation time in hours",
    )
    po_creation_std_hours: float = Field(
        default=0.75,
        gt=0,
        description="Standard deviation of PO creation time in hours",
    )
    goods_receipt_mean_hours: float = Field(
        default=3.0,
        gt=0,
        description="Mean goods receipt processing time in hours",
    )
    goods_receipt_std_hours: float = Field(
        default=1.5,
        gt=0,
        description="Standard deviation of goods receipt processing time in hours",
    )


class ApprovalTimingModel:
    """Statistical model for approval processing time estimation.

    Implements a Normal(μ=4hrs, σ=2hrs) distribution for modeling the time
    required for an approver to review and act on an approval request. Escalated
    items receive an additional +1 business day (8 hours) surcharge on top of
    the base sampled time.

    All sampled values are clipped to the configured [min_hours, max_hours]
    range to prevent unrealistic negative or excessively long processing times.

    The model supports:
        - Single approval time sampling with optional escalation
        - Conversion to timedelta objects for business time arithmetic
        - Vectorized batch sampling for high-throughput simulation
        - Business hours ↔ business days conversion utilities

    Performance: ≥ 10,000 samples/second for batch operations.

    Example:
        >>> model = ApprovalTimingModel(seed=42)
        >>> hours = model.sample_approval_time(is_escalated=False)
        >>> print(f"Approval will take {hours:.1f} hours")
        >>> td = model.sample_approval_time_as_timedelta(is_escalated=True)
        >>> print(f"Escalated approval: {td}")
    """

    def __init__(
        self,
        config: Optional[ApprovalTimingConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the approval timing model.

        Args:
            config: Configuration parameters for the approval timing distribution.
                If None, default values are used (Normal μ=4hrs, σ=2hrs).
            seed: Random seed for reproducible sampling. If None, sampling is
                non-deterministic.
        """
        self._config = config if config is not None else ApprovalTimingConfig()
        self.rng = np.random.RandomState(seed)

        # Build the Normal distribution for approval processing times
        self._distribution = stats.norm(
            loc=self._config.mean_hours,
            scale=self._config.std_hours,
        )

        logger.info(
            "approval_timing_model_initialized",
            mean_hours=self._config.mean_hours,
            std_hours=self._config.std_hours,
            min_hours=self._config.min_hours,
            max_hours=self._config.max_hours,
            escalation_additional_hours=self._config.escalation_additional_hours,
            seed=seed,
        )

    def sample_approval_time(self, is_escalated: bool = False) -> float:
        """Sample a single approval processing time from the Normal distribution.

        Draws a random value from Normal(μ, σ), clips it to [min_hours, max_hours],
        and optionally adds the escalation surcharge for items requiring higher-level
        review.

        Args:
            is_escalated: If True, adds escalation_additional_hours (default +8hrs,
                representing 1 business day) to the sampled base time.

        Returns:
            Approval processing time in hours as a positive float, clipped to
            the configured bounds. Escalated times may exceed max_hours by up
            to the escalation surcharge amount.
        """
        # Sample base approval time from Normal(μ=4, σ=2)
        raw_hours: float = float(self._distribution.rvs(random_state=self.rng))

        # Clip to enforce positive, bounded processing time
        hours = float(np.clip(raw_hours, self._config.min_hours, self._config.max_hours))

        # Apply escalation surcharge: +1 business day (8 hours)
        if is_escalated:
            hours += self._config.escalation_additional_hours

        return hours

    def sample_approval_time_as_timedelta(self, is_escalated: bool = False) -> timedelta:
        """Sample an approval time and return it as a timedelta object.

        Convenience method that wraps sample_approval_time() and converts the
        result to a datetime.timedelta, suitable for direct use in date/time
        arithmetic for business calendar operations.

        Args:
            is_escalated: If True, adds escalation surcharge to the sampled time.

        Returns:
            A timedelta representing the sampled approval processing duration.
        """
        hours = self.sample_approval_time(is_escalated=is_escalated)
        return timedelta(hours=hours)

    def sample_approval_times_batch(
        self,
        n: int,
        escalated_flags: Optional[List[bool]] = None,
    ) -> List[float]:
        """Sample a batch of approval processing times using vectorized operations.

        Efficiently samples n approval times from the Normal distribution using
        numpy vectorization, applies per-item escalation flags, and clips all
        results to the configured bounds.

        Args:
            n: Number of approval times to sample. Must be positive.
            escalated_flags: Optional list of boolean flags indicating which items
                are escalated. If provided, must have length n. If None, no items
                are treated as escalated.

        Returns:
            List of approval processing times in hours. Each value is clipped to
            [min_hours, max_hours] before escalation surcharge is applied.

        Raises:
            ValueError: If n is not positive, or if escalated_flags has wrong length.
        """
        if n <= 0:
            raise ValueError(f"Batch size n must be positive, got {n}")

        if escalated_flags is not None and len(escalated_flags) != n:
            raise ValueError(
                f"escalated_flags length ({len(escalated_flags)}) must match n ({n})"
            )

        # Vectorized sampling from Normal distribution
        raw_samples = self._distribution.rvs(size=n, random_state=self.rng)

        # Clip all samples to configured bounds
        clipped = np.clip(raw_samples, self._config.min_hours, self._config.max_hours)

        # Apply escalation surcharge where flagged
        if escalated_flags is not None:
            escalation_mask = np.array(escalated_flags, dtype=bool)
            clipped[escalation_mask] += self._config.escalation_additional_hours

        return clipped.tolist()

    def hours_to_business_days(self, hours: float, hours_per_day: float = 8.0) -> float:
        """Convert hours to fractional business days.

        Utility method for translating approval processing hours into business
        day equivalents for calendar scheduling and reporting.

        Args:
            hours: Number of hours to convert.
            hours_per_day: Working hours per business day. Defaults to 8.0.

        Returns:
            Fractional business days (e.g., 12.0 hours → 1.5 business days).
        """
        if hours_per_day <= 0:
            raise ValueError(f"hours_per_day must be positive, got {hours_per_day}")
        return hours / hours_per_day

    def business_days_to_hours(self, days: float, hours_per_day: float = 8.0) -> float:
        """Convert fractional business days to hours.

        Inverse of hours_to_business_days. Useful for converting business day
        estimates into hour-level granularity for simulation scheduling.

        Args:
            days: Number of business days to convert.
            hours_per_day: Working hours per business day. Defaults to 8.0.

        Returns:
            Number of hours (e.g., 1.5 business days → 12.0 hours).
        """
        if hours_per_day <= 0:
            raise ValueError(f"hours_per_day must be positive, got {hours_per_day}")
        return days * hours_per_day


class ProcessingTimingModel:
    """Statistical model for general ERP workflow processing durations.

    Provides Normal-distribution-based timing models for common ERP operations
    beyond approval processing. Each supported process type has its own
    parameterized Normal distribution reflecting the typical duration profile
    of that operation.

    Supported process types:
        - "invoice_processing": Vendor invoice review and entry (Normal μ=2hrs, σ=1hr)
        - "po_creation": Purchase order creation workflow (Normal μ=1.5hrs, σ=0.75hrs)
        - "goods_receipt": Goods receipt and inspection (Normal μ=3hrs, σ=1.5hrs)

    All sampled values are clipped to [0.25, 24.0] hours (15 minutes to 24 hours)
    to prevent unrealistic durations.

    Performance: ≥ 10,000 samples/second for batch operations.

    Example:
        >>> model = ProcessingTimingModel(seed=42)
        >>> hours = model.sample_processing_time("invoice_processing")
        >>> print(f"Invoice processing will take {hours:.1f} hours")
    """

    # Global bounds for processing times: minimum 15 min, maximum 24 hours
    _MIN_PROCESSING_HOURS: float = 0.25
    _MAX_PROCESSING_HOURS: float = 24.0

    def __init__(
        self,
        config: Optional[ProcessingTimingConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the processing timing model.

        Builds Normal distributions for each supported process type using the
        provided configuration parameters.

        Args:
            config: Configuration with mean/std for each process type. If None,
                default parameters are used.
            seed: Random seed for reproducible sampling. If None, sampling is
                non-deterministic.
        """
        self._config = config if config is not None else ProcessingTimingConfig()
        self.rng = np.random.RandomState(seed)

        # Build per-process-type Normal distributions from configuration
        self._distributions: Dict[str, stats.rv_continuous] = {
            "invoice_processing": stats.norm(
                loc=self._config.invoice_processing_mean_hours,
                scale=self._config.invoice_processing_std_hours,
            ),
            "po_creation": stats.norm(
                loc=self._config.po_creation_mean_hours,
                scale=self._config.po_creation_std_hours,
            ),
            "goods_receipt": stats.norm(
                loc=self._config.goods_receipt_mean_hours,
                scale=self._config.goods_receipt_std_hours,
            ),
        }

        logger.info(
            "processing_timing_model_initialized",
            process_types=list(self._distributions.keys()),
            invoice_processing_mean=self._config.invoice_processing_mean_hours,
            po_creation_mean=self._config.po_creation_mean_hours,
            goods_receipt_mean=self._config.goods_receipt_mean_hours,
            seed=seed,
        )

    def sample_processing_time(self, process_type: str) -> float:
        """Sample a single processing time for the given operation type.

        Draws from the Normal distribution associated with the specified process
        type and clips the result to [0.25, 24.0] hours.

        Args:
            process_type: The type of processing operation. Must be one of:
                "invoice_processing", "po_creation", "goods_receipt".

        Returns:
            Processing time in hours as a positive float, clipped to bounds.

        Raises:
            ValueError: If process_type is not a recognized operation type.
        """
        distribution = self._distributions.get(process_type)
        if distribution is None:
            valid_types = list(self._distributions.keys())
            raise ValueError(
                f"Unknown process_type '{process_type}'. "
                f"Valid types: {valid_types}"
            )

        # Sample from the process-specific Normal distribution
        raw_hours: float = float(distribution.rvs(random_state=self.rng))

        # Clip to global processing time bounds
        hours = float(
            np.clip(raw_hours, self._MIN_PROCESSING_HOURS, self._MAX_PROCESSING_HOURS)
        )

        return hours

    def sample_processing_times_batch(self, process_type: str, n: int) -> List[float]:
        """Sample a batch of processing times for a specific operation type.

        Uses vectorized numpy/scipy operations for high-throughput sampling.
        All results are clipped to [0.25, 24.0] hours.

        Args:
            process_type: The type of processing operation. Must be one of:
                "invoice_processing", "po_creation", "goods_receipt".
            n: Number of processing times to sample. Must be positive.

        Returns:
            List of processing times in hours, each clipped to bounds.

        Raises:
            ValueError: If process_type is unrecognized or n is not positive.
        """
        if n <= 0:
            raise ValueError(f"Batch size n must be positive, got {n}")

        distribution = self._distributions.get(process_type)
        if distribution is None:
            valid_types = list(self._distributions.keys())
            raise ValueError(
                f"Unknown process_type '{process_type}'. "
                f"Valid types: {valid_types}"
            )

        # Vectorized sampling from the process-specific distribution
        raw_samples = distribution.rvs(size=n, random_state=self.rng)

        # Clip all samples to global bounds
        clipped = np.clip(raw_samples, self._MIN_PROCESSING_HOURS, self._MAX_PROCESSING_HOURS)

        return clipped.tolist()

    def hours_to_business_days(self, hours: float, hours_per_day: float = 8.0) -> float:
        """Convert hours to fractional business days.

        Args:
            hours: Number of hours to convert.
            hours_per_day: Working hours per business day. Defaults to 8.0.

        Returns:
            Fractional business days.
        """
        if hours_per_day <= 0:
            raise ValueError(f"hours_per_day must be positive, got {hours_per_day}")
        return hours / hours_per_day

    def business_days_to_hours(self, days: float, hours_per_day: float = 8.0) -> float:
        """Convert fractional business days to hours.

        Args:
            days: Number of business days to convert.
            hours_per_day: Working hours per business day. Defaults to 8.0.

        Returns:
            Number of hours.
        """
        if hours_per_day <= 0:
            raise ValueError(f"hours_per_day must be positive, got {hours_per_day}")
        return days * hours_per_day
