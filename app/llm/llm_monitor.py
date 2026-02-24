"""LLM usage monitoring, cost tracking, latency statistics, and budget enforcement.

Provides LLMMonitor for comprehensive metrics and LLMBudgetManager for hard budget
cap management with configurable warning/block thresholds.

Key features:
- LLMBudgetManager: $100/month hard cap, 80% warning, 100% block
- LLMMonitor: total_requests, successful, failed, total_tokens, cost, avg/p95 latency
- AlertConfig: Pydantic V2 model for alert threshold configuration
- All logging uses structlog to stdout in structured JSON format
"""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

# Module-level structured logger — stdout only per AAP Section 0.7.6
logger = structlog.get_logger(__name__)


class AlertConfig(BaseModel):
    """Pydantic V2 configuration model for alert thresholds.

    Defines configurable thresholds for high latency alerts, high error rate
    alerts, and budget warning/block levels. Used to parametrise LLMMonitor
    and LLMBudgetManager without hard-coding magic numbers.

    Attributes:
        high_latency_threshold: Seconds above which a latency alert fires (default 10.0s).
        high_error_rate_threshold: Fraction (0–1) above which error-rate alert fires (default 0.05 = 5%).
        budget_warning_threshold: Fraction of monthly budget that triggers a warning (default 0.80 = 80%).
        budget_block_threshold: Fraction of monthly budget that triggers a hard block (default 1.0 = 100%).
    """

    high_latency_threshold: float = Field(
        default=10.0,
        ge=0.0,
        description="Latency in seconds above which a high-latency alert is raised (README.md line 269).",
    )
    high_error_rate_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Error rate fraction (0–1) above which a high-error-rate alert is raised (README.md line 270).",
    )
    budget_warning_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Fraction of monthly budget at which a warning is logged (AAP: 80%).",
    )
    budget_block_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Fraction of monthly budget at which requests are blocked (AAP: 100%).",
    )

    model_config = {
        "str_strip_whitespace": True,
        "validate_default": True,
    }


class LLMBudgetManager:
    """Manage LLM spending against a monthly budget with hard-cap enforcement.

    Implements the budget control pattern from README.md lines 1838-1873 and
    AAP Section 0.1.2:
      - Monthly budget default: $100 USD
      - Warning at 80% utilisation ($80)
      - Hard block at 100% utilisation ($100)

    The manager performs automatic monthly rollover: when a new calendar month
    is detected (UTC), counters reset transparently.

    Attributes:
        monthly_budget: The monthly budget ceiling in USD.
        current_spend: Accumulated spend in the current month.
        budget_warning_threshold: Fraction triggering a warning log (default 0.80).
        block_threshold: Fraction triggering a hard block (default 1.0).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        monthly_budget_usd: float = 100.0,
        warning_threshold: float = 0.80,
        block_threshold: float = 1.0,
    ) -> None:
        """Initialise the budget manager.

        Args:
            monthly_budget_usd: Maximum spend per calendar month in USD.
            warning_threshold: Fraction of budget that triggers a warning (0–1).
            block_threshold: Fraction of budget that triggers a hard block (0–1).
        """
        self.monthly_budget: float = monthly_budget_usd
        self.current_spend: float = 0.0
        self.budget_warning_threshold: float = warning_threshold
        self.block_threshold: float = block_threshold

        # Internal bookkeeping for monthly rollover
        self._month_start: datetime = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        self._warning_issued: bool = False

        logger.info(
            "budget_manager_initialized",
            monthly_budget=self.monthly_budget,
            warning_threshold=self.budget_warning_threshold,
            block_threshold=self.block_threshold,
        )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def check_budget_before_request(self, estimated_cost: float) -> bool:
        """Check whether a request with *estimated_cost* should be allowed.

        Behaviour (per README.md lines 1847-1868):
        1. Detect calendar-month rollover and auto-reset.
        2. If projected spend exceeds the block threshold → BLOCK (return False).
        3. If projected spend exceeds the warning threshold → log a WARNING (once).
        4. Otherwise → ALLOW (return True).

        Args:
            estimated_cost: Estimated cost of the upcoming LLM request in USD.

        Returns:
            True if the request is within budget, False if it must be blocked.
        """
        # Auto-detect month rollover
        self._check_month_rollover()

        projected_spend = self.current_spend + estimated_cost

        # --- Hard block at block_threshold (default 100%) ---
        if projected_spend > self.monthly_budget * self.block_threshold:
            logger.error(
                "llm_budget_exceeded",
                current_spend=self.current_spend,
                monthly_budget=self.monthly_budget,
                estimated_cost=estimated_cost,
                block_threshold=self.block_threshold,
            )
            return False

        # --- Warning at budget_warning_threshold (default 80%) ---
        if projected_spend > self.monthly_budget * self.budget_warning_threshold:
            if not self._warning_issued:
                percent_used = (
                    (projected_spend / self.monthly_budget * 100)
                    if self.monthly_budget > 0
                    else 0.0
                )
                logger.warning(
                    "llm_budget_warning",
                    current_spend=self.current_spend,
                    monthly_budget=self.monthly_budget,
                    percent_used=round(percent_used, 2),
                )
                self._warning_issued = True

        return True

    # ------------------------------------------------------------------
    # Spend recording
    # ------------------------------------------------------------------

    def record_spend(self, cost: float) -> None:
        """Record actual spend for a completed LLM request.

        Args:
            cost: The actual cost in USD of the completed request.
        """
        self.current_spend += cost
        remaining = self.get_remaining_budget()
        logger.debug(
            "llm_spend_recorded",
            cost=cost,
            current_spend=self.current_spend,
            remaining_budget=remaining,
        )

    # ------------------------------------------------------------------
    # Budget queries
    # ------------------------------------------------------------------

    def get_remaining_budget(self) -> float:
        """Return the remaining budget for the current month in USD."""
        return max(0.0, self.monthly_budget - self.current_spend)

    def get_budget_utilization(self) -> float:
        """Return the current budget utilisation as a fraction (0.0–1.0+)."""
        if self.monthly_budget <= 0:
            return 0.0
        return self.current_spend / self.monthly_budget

    # ------------------------------------------------------------------
    # Reset / rollover
    # ------------------------------------------------------------------

    def reset_monthly(self) -> None:
        """Manually reset the budget for a new month."""
        previous_spend = self.current_spend
        self.current_spend = 0.0
        self._warning_issued = False
        self._month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        logger.info(
            "budget_reset",
            previous_spend=previous_spend,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of budget metrics.

        Returns:
            Dictionary containing monthly_budget, current_spend,
            remaining_budget, utilization_percent, and warning_issued.
        """
        utilization = self.get_budget_utilization()
        return {
            "monthly_budget": self.monthly_budget,
            "current_spend": self.current_spend,
            "remaining_budget": self.get_remaining_budget(),
            "utilization_percent": round(utilization * 100, 2),
            "warning_issued": self._warning_issued,
            "month_start": self._month_start.isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_month_rollover(self) -> None:
        """Detect whether the calendar month has changed and auto-reset."""
        now = datetime.now(timezone.utc)
        current_month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        if current_month_start > self._month_start:
            logger.info(
                "budget_month_rollover_detected",
                previous_month_start=self._month_start.isoformat(),
                new_month_start=current_month_start.isoformat(),
                previous_spend=self.current_spend,
            )
            self.current_spend = 0.0
            self._warning_issued = False
            self._month_start = current_month_start


class LLMMonitor:
    """Monitor LLM usage, costs, and performance metrics.

    Tracks comprehensive metrics per README.md lines 248-271:
      - total_requests, successful_requests, failed_requests
      - total_tokens (prompt + completion), total_input_tokens, total_output_tokens
      - total_cost (USD)
      - average_latency, p95_latency
      - error_rate, success_rate

    Alerts (per README.md):
      - Budget warning at 80%, budget exceeded at 100%
      - High latency alert (> 10 s)
      - High error rate alert (> 5%)

    Integrates with an optional LLMBudgetManager for budget enforcement via
    constructor injection (AAP Section 0.7.1).

    Attributes:
        total_requests: Cumulative count of all LLM requests.
        successful_requests: Cumulative count of successful requests.
        failed_requests: Cumulative count of failed requests.
        total_tokens: Cumulative token count (input + output combined).
        total_input_tokens: Cumulative input/prompt token count.
        total_output_tokens: Cumulative output/completion token count.
        total_cost: Cumulative cost in USD.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        budget_manager: Optional[LLMBudgetManager] = None,
        alert_config: Optional[AlertConfig] = None,
    ) -> None:
        """Initialise the LLM monitor.

        Args:
            budget_manager: Optional budget manager for spend delegation.
            alert_config: Optional alert configuration; defaults apply if None.
        """
        # Counters
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.total_tokens: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0

        # Latency tracking — capped to prevent unbounded memory growth
        self._latencies: List[float] = []
        self._max_latency_samples: int = 10_000

        # Budget manager (constructor injection)
        self._budget_manager: Optional[LLMBudgetManager] = budget_manager

        # Alert thresholds
        _alert = alert_config or AlertConfig()
        self._high_latency_threshold: float = _alert.high_latency_threshold
        self._high_error_rate_threshold: float = _alert.high_error_rate_threshold

        logger.info("llm_monitor_initialized")

    # ------------------------------------------------------------------
    # Request recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency: float = 0.0,
        model: str = "",
        request_id: str = "",
    ) -> None:
        """Record the outcome of a single LLM request.

        This is the primary entry-point for instrumentation.  It updates all
        counters, records latency, delegates spend to the budget manager, and
        evaluates alert conditions.

        Args:
            success: Whether the request completed successfully.
            input_tokens: Number of prompt/input tokens consumed.
            output_tokens: Number of completion/output tokens generated.
            cost: Actual cost of the request in USD.
            latency: Wall-clock duration of the request in seconds.
            model: Model identifier (e.g. "claude-sonnet-4-20250514").
            request_id: Unique request identifier for correlation.
        """
        # --- Counters ---
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        # --- Token accounting ---
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens

        # --- Cost accounting ---
        self.total_cost += cost

        # --- Latency tracking ---
        self._latencies.append(latency)
        # Trim oldest samples when the buffer is full to keep memory bounded
        if len(self._latencies) > self._max_latency_samples:
            excess = len(self._latencies) - self._max_latency_samples
            self._latencies = self._latencies[excess:]

        # --- Budget delegation ---
        if self._budget_manager is not None:
            self._budget_manager.record_spend(cost)

        # --- Alert evaluation ---
        if latency > self._high_latency_threshold:
            logger.warning(
                "llm_high_latency",
                latency=latency,
                threshold=self._high_latency_threshold,
                model=model,
                request_id=request_id,
            )

        current_error_rate = self.error_rate
        if (
            self.total_requests > 0
            and current_error_rate > self._high_error_rate_threshold
        ):
            logger.warning(
                "llm_high_error_rate",
                error_rate=round(current_error_rate, 4),
                threshold=self._high_error_rate_threshold,
                total_requests=self.total_requests,
                failed_requests=self.failed_requests,
            )

        # --- Structured request log ---
        logger.debug(
            "llm_request_recorded",
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
            model=model,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Budget check delegation
    # ------------------------------------------------------------------

    def check_budget(self, estimated_cost: float) -> bool:
        """Check whether the upcoming request is within budget.

        Delegates to the injected LLMBudgetManager.  If no manager is
        configured, all requests are allowed.

        Args:
            estimated_cost: Estimated cost of the request in USD.

        Returns:
            True if the request should proceed, False if it must be blocked.
        """
        if self._budget_manager is not None:
            return self._budget_manager.check_budget_before_request(estimated_cost)
        return True

    # ------------------------------------------------------------------
    # Latency properties
    # ------------------------------------------------------------------

    @property
    def average_latency(self) -> float:
        """Return the mean latency across all recorded requests."""
        if not self._latencies:
            return 0.0
        return statistics.mean(self._latencies)

    @property
    def p95_latency(self) -> float:
        """Return the 95th-percentile latency.

        Uses the sorted-array index approach for accurate percentile
        calculation.  Falls back to 0.0 when no data is available.
        """
        if not self._latencies:
            return 0.0
        n = len(self._latencies)
        if n == 1:
            return self._latencies[0]
        sorted_latencies = sorted(self._latencies)
        # Use interpolation-free "nearest-rank" method
        index = int(0.95 * (n - 1))
        # Linear interpolation between the two surrounding ranks for accuracy
        lower = sorted_latencies[index]
        if index + 1 < n:
            fractional = (0.95 * (n - 1)) - index
            upper = sorted_latencies[index + 1]
            return lower + fractional * (upper - lower)
        return lower

    # ------------------------------------------------------------------
    # Rate properties
    # ------------------------------------------------------------------

    @property
    def error_rate(self) -> float:
        """Return the error rate as a fraction (0.0–1.0)."""
        if self.total_requests <= 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Return the success rate as a fraction (0.0–1.0)."""
        if self.total_requests <= 0:
            return 0.0
        return self.successful_requests / self.total_requests

    # ------------------------------------------------------------------
    # Comprehensive metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return a comprehensive metrics snapshot.

        This dictionary feeds into the daily summary logs per
        AAP Section 0.7.6.

        Returns:
            Dictionary containing all tracked metrics plus budget metrics
            (when a budget manager is attached).
        """
        metrics: Dict[str, Any] = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "average_latency": round(self.average_latency, 4),
            "p95_latency": round(self.p95_latency, 4),
            "error_rate": round(self.error_rate, 4),
            "success_rate": round(self.success_rate, 4),
        }

        if self._budget_manager is not None:
            metrics["budget_metrics"] = self._budget_manager.get_metrics()

        return metrics

    # ------------------------------------------------------------------
    # Daily summary (for SimulationEngine daily logs)
    # ------------------------------------------------------------------

    def get_daily_summary(self) -> Dict[str, Any]:
        """Return metrics formatted for the daily summary log entry.

        Per README.md lines 1744-1756 the daily summary includes
        ``llm_requests`` and ``llm_cost_usd``.

        Returns:
            Dictionary with keys aligned to the daily summary log spec.
        """
        return {
            "llm_requests": self.total_requests,
            "llm_cost_usd": round(self.total_cost, 4),
            "llm_successful_requests": self.successful_requests,
            "llm_failed_requests": self.failed_requests,
            "llm_total_tokens": self.total_tokens,
            "llm_average_latency": round(self.average_latency, 4),
            "llm_p95_latency": round(self.p95_latency, 4),
            "llm_error_rate": round(self.error_rate, 4),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all monitor counters and latency data.

        The attached budget manager state is **preserved** because budget
        tracking spans across monitor resets (e.g. daily resets do not
        affect monthly budget).
        """
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self._latencies = []

        logger.info("llm_monitor_reset")
