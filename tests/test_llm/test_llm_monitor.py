"""Comprehensive unit tests for LLMMonitor and LLMBudgetManager.

Tests verify:
- LLMBudgetManager: $100/month hard cap, 80% warning threshold, 100% block
  threshold, spend recording, monthly reset, budget queries, and metrics.
- LLMMonitor: metrics accumulation (total_requests, successful/failed,
  total_tokens, cost), latency statistics (average, p95), error/success
  rates, alert triggering (high latency >10s, high error rate >5%),
  budget delegation, daily summary, and reset.

Per AAP Section 0.5.1 Group 10: "Metrics accumulation, budget warning/block triggers."
Per AAP Section 0.7.3: "LLM budget enforcement: Hard cap at $100/month —
the LLMBudgetManager must block all requests when the budget is exhausted,
not merely warn."
Per AAP Section 0.7.5: "Unit test coverage target: >= 80% across all
source modules."
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import pytest

# ---------------------------------------------------------------------------
# Internal Imports — Module Under Test
# ---------------------------------------------------------------------------
from app.llm.llm_monitor import LLMBudgetManager, LLMMonitor


# =========================================================================
# Local Fixtures
# =========================================================================


@pytest.fixture
def budget_manager() -> LLMBudgetManager:
    """Standard $100/month budget manager with 80% warning, 100% block."""
    return LLMBudgetManager(
        monthly_budget_usd=100.0,
        warning_threshold=0.80,
        block_threshold=1.0,
    )


@pytest.fixture
def monitor() -> LLMMonitor:
    """Basic LLMMonitor without a budget manager (metrics only)."""
    return LLMMonitor()


@pytest.fixture
def monitor_with_budget() -> LLMMonitor:
    """LLMMonitor with an attached $100 budget manager."""
    budget = LLMBudgetManager(monthly_budget_usd=100.0)
    return LLMMonitor(budget_manager=budget)


@pytest.fixture
def small_budget_manager() -> LLMBudgetManager:
    """Small $10 budget manager for easier threshold testing."""
    return LLMBudgetManager(monthly_budget_usd=10.0)


# =========================================================================
# Section 1: LLMBudgetManager — Initialization Tests
# =========================================================================


class TestBudgetManagerInitialization:
    """Verify LLMBudgetManager default and custom initialisation."""

    def test_budget_manager_default_values(self) -> None:
        """Default constructor produces $100 budget, $0 spend, 80% warning."""
        manager = LLMBudgetManager()
        assert manager.monthly_budget == 100.0
        assert manager.current_spend == 0.0
        assert manager.budget_warning_threshold == 0.80
        assert manager.block_threshold == 1.0

    def test_budget_manager_custom_values(self) -> None:
        """Custom parameters are stored correctly."""
        manager = LLMBudgetManager(
            monthly_budget_usd=50.0,
            warning_threshold=0.70,
            block_threshold=0.90,
        )
        assert manager.monthly_budget == 50.0
        assert manager.budget_warning_threshold == 0.70
        assert manager.block_threshold == 0.90

    def test_budget_manager_initial_spend_is_zero(self, budget_manager: LLMBudgetManager) -> None:
        """A newly created budget manager has zero spend."""
        assert budget_manager.current_spend == 0.0

    def test_budget_manager_warning_not_issued_initially(self, budget_manager: LLMBudgetManager) -> None:
        """Internal warning flag is False on creation."""
        assert budget_manager._warning_issued is False


# =========================================================================
# Section 2: LLMBudgetManager — check_budget_before_request Tests
# CRITICAL: Per AAP, "HARD cap at $100/month — must BLOCK all requests
#           when budget exhausted, not merely warn."
# =========================================================================


class TestBudgetManagerBudgetCheck:
    """Verify budget enforcement: warnings and hard blocks."""

    def test_check_budget_allows_request_under_budget(self, budget_manager: LLMBudgetManager) -> None:
        """Requests well under budget are allowed."""
        assert budget_manager.check_budget_before_request(estimated_cost=1.0) is True

    def test_check_budget_allows_request_at_warning_boundary(self, budget_manager: LLMBudgetManager) -> None:
        """Projected spend below warning threshold is allowed without warning."""
        budget_manager.current_spend = 79.0
        # projected = 79.0 + 0.5 = 79.5, which is < 80.0 (80% of $100)
        assert budget_manager.check_budget_before_request(estimated_cost=0.5) is True

    def test_check_budget_warns_at_80_percent(self, budget_manager: LLMBudgetManager) -> None:
        """Projected spend crossing 80% logs a warning but still allows the request."""
        budget_manager.current_spend = 79.0
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            result = budget_manager.check_budget_before_request(estimated_cost=2.0)
            # projected = 79 + 2 = 81 > 80 → warning logged
            assert result is True  # Warning only, NOT a block
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_budget_warning"
            ]
            assert len(warning_calls) == 1

    def test_check_budget_blocks_at_100_percent(self, budget_manager: LLMBudgetManager) -> None:
        """CRITICAL: Projected spend exceeding budget is a HARD STOP (returns False)."""
        budget_manager.current_spend = 99.0
        # projected = 99 + 2 = 101 > 100 → BLOCKED
        assert budget_manager.check_budget_before_request(estimated_cost=2.0) is False

    def test_check_budget_blocks_exactly_at_budget_plus_tiny_amount(
        self, budget_manager: LLMBudgetManager
    ) -> None:
        """Even tiny amounts blocked when current spend is at budget ceiling."""
        budget_manager.current_spend = 100.0
        # projected = 100.0 + 0.01 = 100.01 > 100.0 → BLOCKED
        assert budget_manager.check_budget_before_request(estimated_cost=0.01) is False

    def test_check_budget_blocks_when_already_over(self, budget_manager: LLMBudgetManager) -> None:
        """Requests blocked when spend already exceeds budget, even zero-cost."""
        budget_manager.current_spend = 105.0
        # projected = 105.0 + 0.0 = 105.0 > 100.0 → BLOCKED
        assert budget_manager.check_budget_before_request(estimated_cost=0.0) is False

    def test_check_budget_allows_zero_cost_at_exact_limit(self, budget_manager: LLMBudgetManager) -> None:
        """Zero-cost request at exact budget boundary is allowed (not > budget).

        Implementation uses strict greater-than: projected_spend > budget,
        so 100.0 + 0.0 = 100.0, and 100.0 > 100.0 is False → allowed.
        """
        budget_manager.current_spend = 100.0
        result = budget_manager.check_budget_before_request(estimated_cost=0.0)
        # 100.0 is NOT > 100.0 → allowed
        assert result is True

    def test_check_budget_warning_only_logged_once(self, budget_manager: LLMBudgetManager) -> None:
        """Warning is logged once; subsequent above-threshold calls don't re-warn."""
        budget_manager.current_spend = 79.0
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            # First call — should warn
            budget_manager.check_budget_before_request(estimated_cost=2.0)
            first_count = sum(
                1 for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_budget_warning"
            )
            assert first_count == 1

            # Second call — should NOT warn again
            budget_manager.check_budget_before_request(estimated_cost=2.0)
            second_count = sum(
                1 for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_budget_warning"
            )
            assert second_count == 1  # Still only one warning total

    def test_small_budget_blocks_earlier(self, small_budget_manager: LLMBudgetManager) -> None:
        """Small $10 budget blocks at proportionally lower spend levels."""
        small_budget_manager.current_spend = 9.0
        # projected = 9 + 2 = 11 > 10 → BLOCKED
        assert small_budget_manager.check_budget_before_request(estimated_cost=2.0) is False

    def test_check_budget_block_logs_error(self, budget_manager: LLMBudgetManager) -> None:
        """When budget is blocked, an error is logged with details."""
        budget_manager.current_spend = 99.0
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            budget_manager.check_budget_before_request(estimated_cost=2.0)
            error_calls = [
                c for c in mock_logger.error.call_args_list
                if c[0][0] == "llm_budget_exceeded"
            ]
            assert len(error_calls) == 1


# =========================================================================
# Section 3: LLMBudgetManager — record_spend Tests
# =========================================================================


class TestBudgetManagerRecordSpend:
    """Verify spend accumulation."""

    def test_record_spend_updates_current_spend(self, budget_manager: LLMBudgetManager) -> None:
        """Spend is accumulated correctly across multiple calls."""
        budget_manager.record_spend(5.0)
        assert budget_manager.current_spend == 5.0
        budget_manager.record_spend(3.0)
        assert budget_manager.current_spend == 8.0

    def test_record_spend_accumulates_many_small_amounts(self, budget_manager: LLMBudgetManager) -> None:
        """Ten $1.00 spends accumulate to $10.00."""
        for _ in range(10):
            budget_manager.record_spend(1.0)
        assert budget_manager.current_spend == pytest.approx(10.0)

    def test_record_spend_zero_does_not_change_spend(self, budget_manager: LLMBudgetManager) -> None:
        """Recording zero cost leaves spend unchanged."""
        budget_manager.record_spend(0.0)
        assert budget_manager.current_spend == 0.0


# =========================================================================
# Section 4: LLMBudgetManager — Budget Utility Methods
# =========================================================================


class TestBudgetManagerUtilities:
    """Verify remaining budget and utilisation calculations."""

    def test_get_remaining_budget(self, budget_manager: LLMBudgetManager) -> None:
        """Remaining budget = monthly_budget - current_spend."""
        budget_manager.current_spend = 30.0
        assert budget_manager.get_remaining_budget() == 70.0

    def test_get_remaining_budget_when_over(self, budget_manager: LLMBudgetManager) -> None:
        """Remaining budget floors at 0.0 when over budget."""
        budget_manager.current_spend = 110.0
        assert budget_manager.get_remaining_budget() == 0.0

    def test_get_remaining_budget_at_zero_spend(self, budget_manager: LLMBudgetManager) -> None:
        """Full budget remains when nothing has been spent."""
        assert budget_manager.get_remaining_budget() == 100.0

    def test_get_budget_utilization(self, budget_manager: LLMBudgetManager) -> None:
        """Utilisation = current_spend / monthly_budget."""
        budget_manager.current_spend = 50.0
        assert budget_manager.get_budget_utilization() == pytest.approx(0.5)

    def test_get_budget_utilization_at_zero(self, budget_manager: LLMBudgetManager) -> None:
        """Zero spend yields zero utilisation."""
        assert budget_manager.get_budget_utilization() == 0.0

    def test_get_budget_utilization_over_100_percent(self, budget_manager: LLMBudgetManager) -> None:
        """Over-budget spend produces utilisation > 1.0."""
        budget_manager.current_spend = 120.0
        assert budget_manager.get_budget_utilization() == pytest.approx(1.2)

    def test_get_budget_utilization_zero_budget(self) -> None:
        """Zero monthly budget yields zero utilisation (division protection)."""
        manager = LLMBudgetManager(monthly_budget_usd=0.0)
        assert manager.get_budget_utilization() == 0.0


# =========================================================================
# Section 5: LLMBudgetManager — Monthly Reset Tests
# =========================================================================


class TestBudgetManagerReset:
    """Verify monthly reset clears spend and warning state."""

    def test_reset_monthly_clears_spend(self, budget_manager: LLMBudgetManager) -> None:
        """Monthly reset sets current_spend to zero."""
        budget_manager.current_spend = 75.0
        budget_manager.reset_monthly()
        assert budget_manager.current_spend == 0.0

    def test_reset_monthly_clears_warning_flag(self, budget_manager: LLMBudgetManager) -> None:
        """After reset, the warning can fire again on subsequent calls."""
        # First: trigger a warning (spend crosses 80%)
        budget_manager.current_spend = 79.0
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            budget_manager.check_budget_before_request(estimated_cost=2.0)
            first_warnings = sum(
                1 for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_budget_warning"
            )
            assert first_warnings == 1

        # Reset clears the flag
        budget_manager.reset_monthly()

        # Second: same scenario should warn again because flag was cleared
        budget_manager.current_spend = 79.0
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            budget_manager.check_budget_before_request(estimated_cost=2.0)
            renewed_warnings = sum(
                1 for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_budget_warning"
            )
            assert renewed_warnings == 1

    def test_reset_monthly_updates_month_start(self, budget_manager: LLMBudgetManager) -> None:
        """Reset updates the internal month-start timestamp."""
        old_month_start = budget_manager._month_start
        budget_manager.reset_monthly()
        new_month_start = budget_manager._month_start
        # Both should be the first day of the current UTC month
        assert new_month_start.day == 1
        assert new_month_start.tzinfo is not None


# =========================================================================
# Section 6: LLMBudgetManager — Metrics Tests
# =========================================================================


class TestBudgetManagerMetrics:
    """Verify get_metrics returns a complete and accurate snapshot."""

    def test_get_metrics_returns_complete_dict(self, budget_manager: LLMBudgetManager) -> None:
        """Metrics dictionary contains all expected keys with correct values."""
        budget_manager.current_spend = 25.0
        metrics = budget_manager.get_metrics()
        assert metrics["monthly_budget"] == 100.0
        assert metrics["current_spend"] == 25.0
        assert metrics["remaining_budget"] == 75.0
        assert metrics["utilization_percent"] == 25.0
        assert "warning_issued" in metrics
        assert "month_start" in metrics

    def test_get_metrics_at_zero_spend(self, budget_manager: LLMBudgetManager) -> None:
        """Metrics at zero spend show full remaining budget."""
        metrics = budget_manager.get_metrics()
        assert metrics["current_spend"] == 0.0
        assert metrics["remaining_budget"] == 100.0
        assert metrics["utilization_percent"] == 0.0
        assert metrics["warning_issued"] is False

    def test_get_metrics_reflects_warning_state(self, budget_manager: LLMBudgetManager) -> None:
        """Metrics reflect the warning_issued flag after a warning fires."""
        budget_manager.current_spend = 79.0
        with patch("app.llm.llm_monitor.logger"):
            budget_manager.check_budget_before_request(estimated_cost=2.0)
        metrics = budget_manager.get_metrics()
        assert metrics["warning_issued"] is True


# =========================================================================
# Section 7: LLMMonitor — Initialization Tests
# =========================================================================


class TestMonitorInitialization:
    """Verify LLMMonitor initialises with zero counters."""

    def test_monitor_initial_metrics(self, monitor: LLMMonitor) -> None:
        """All counters start at zero for a fresh monitor."""
        assert monitor.total_requests == 0
        assert monitor.successful_requests == 0
        assert monitor.failed_requests == 0
        assert monitor.total_tokens == 0
        assert monitor.total_cost == 0.0

    def test_monitor_with_budget_manager(self, monitor_with_budget: LLMMonitor) -> None:
        """Budget delegation works when a manager is attached."""
        assert monitor_with_budget._budget_manager is not None
        assert monitor_with_budget.check_budget(1.0) is True

    def test_monitor_without_budget_manager(self, monitor: LLMMonitor) -> None:
        """Monitor without a budget manager has _budget_manager as None."""
        assert monitor._budget_manager is None


# =========================================================================
# Section 8: LLMMonitor — record_request Tests
# =========================================================================


class TestMonitorRecordRequest:
    """Verify that record_request correctly updates all counters."""

    def test_record_successful_request(self, monitor: LLMMonitor) -> None:
        """Successful request updates all relevant counters."""
        monitor.record_request(
            success=True,
            input_tokens=500,
            output_tokens=100,
            cost=0.01,
            latency=1.5,
        )
        assert monitor.total_requests == 1
        assert monitor.successful_requests == 1
        assert monitor.failed_requests == 0
        assert monitor.total_tokens == 600
        assert monitor.total_cost == pytest.approx(0.01)

    def test_record_failed_request(self, monitor: LLMMonitor) -> None:
        """Failed request increments total and failed counters only."""
        monitor.record_request(success=False, latency=5.0)
        assert monitor.total_requests == 1
        assert monitor.successful_requests == 0
        assert monitor.failed_requests == 1

    def test_record_multiple_requests(self, monitor: LLMMonitor) -> None:
        """Multiple mixed requests accumulate correctly."""
        for _ in range(5):
            monitor.record_request(success=True, latency=1.0)
        for _ in range(2):
            monitor.record_request(success=False, latency=2.0)
        assert monitor.total_requests == 7
        assert monitor.successful_requests == 5
        assert monitor.failed_requests == 2

    def test_record_request_accumulates_tokens(self, monitor: LLMMonitor) -> None:
        """Token counts accumulate across multiple requests."""
        monitor.record_request(success=True, input_tokens=200, output_tokens=50)
        monitor.record_request(success=True, input_tokens=300, output_tokens=80)
        assert monitor.total_tokens == 630  # 200+50+300+80

    def test_record_request_tracks_input_output_separately(self, monitor: LLMMonitor) -> None:
        """Input and output token totals are tracked independently."""
        monitor.record_request(success=True, input_tokens=200, output_tokens=50)
        monitor.record_request(success=True, input_tokens=300, output_tokens=80)
        assert monitor.total_input_tokens == 500
        assert monitor.total_output_tokens == 130

    def test_record_request_accumulates_cost(self, monitor: LLMMonitor) -> None:
        """Costs accumulate across requests with floating-point precision."""
        monitor.record_request(success=True, cost=0.005)
        monitor.record_request(success=True, cost=0.010)
        assert monitor.total_cost == pytest.approx(0.015)

    def test_record_request_with_budget_integration(self, monitor_with_budget: LLMMonitor) -> None:
        """Cost is delegated to the attached budget manager via record_spend."""
        monitor_with_budget.record_request(success=True, cost=5.0)
        assert monitor_with_budget._budget_manager.current_spend == pytest.approx(5.0)

    def test_record_request_default_values(self, monitor: LLMMonitor) -> None:
        """Default parameters produce minimal side effects."""
        monitor.record_request(success=True)
        assert monitor.total_requests == 1
        assert monitor.total_tokens == 0
        assert monitor.total_cost == 0.0


# =========================================================================
# Section 9: LLMMonitor — Latency Tests
# =========================================================================


class TestMonitorLatency:
    """Verify average and p95 latency calculations."""

    def test_average_latency_single_request(self, monitor: LLMMonitor) -> None:
        """Single-request average equals the recorded latency."""
        monitor.record_request(success=True, latency=2.0)
        assert monitor.average_latency == pytest.approx(2.0)

    def test_average_latency_multiple_requests(self, monitor: LLMMonitor) -> None:
        """Average across multiple requests is the arithmetic mean."""
        for lat in [1.0, 2.0, 3.0]:
            monitor.record_request(success=True, latency=lat)
        assert monitor.average_latency == pytest.approx(2.0)

    def test_average_latency_no_requests(self, monitor: LLMMonitor) -> None:
        """No data yields zero average latency."""
        assert monitor.average_latency == 0.0

    def test_p95_latency_calculation(self, monitor: LLMMonitor) -> None:
        """P95 across 100 evenly-spaced latencies matches expected value.

        With latencies 0.1, 0.2, ..., 10.0 (100 values):
        n=100, index = int(0.95 * 99) = 94
        sorted[94]=9.5, sorted[95]=9.6, frac=0.05
        result = 9.5 + 0.05 * 0.1 = 9.505
        """
        for i in range(1, 101):
            monitor.record_request(success=True, latency=i * 0.1)
        assert monitor.p95_latency == pytest.approx(9.505, rel=0.01)

    def test_p95_latency_no_requests(self, monitor: LLMMonitor) -> None:
        """No data yields zero p95 latency."""
        assert monitor.p95_latency == 0.0

    def test_p95_latency_single_request(self, monitor: LLMMonitor) -> None:
        """Single-request p95 equals that request's latency."""
        monitor.record_request(success=True, latency=3.0)
        assert monitor.p95_latency == pytest.approx(3.0)

    def test_p95_latency_two_requests(self, monitor: LLMMonitor) -> None:
        """P95 with two data points interpolates correctly."""
        monitor.record_request(success=True, latency=1.0)
        monitor.record_request(success=True, latency=5.0)
        # n=2, index = int(0.95 * 1) = 0, frac = 0.95
        # sorted = [1.0, 5.0], result = 1.0 + 0.95 * (5.0 - 1.0) = 4.8
        assert monitor.p95_latency == pytest.approx(4.8, rel=0.01)

    def test_latency_includes_failed_requests(self, monitor: LLMMonitor) -> None:
        """Failed requests also contribute to latency statistics."""
        monitor.record_request(success=True, latency=1.0)
        monitor.record_request(success=False, latency=10.0)
        assert monitor.average_latency == pytest.approx(5.5)


# =========================================================================
# Section 10: LLMMonitor — Error/Success Rate Tests
# =========================================================================


class TestMonitorRates:
    """Verify error rate and success rate calculations."""

    def test_error_rate_calculation(self, monitor: LLMMonitor) -> None:
        """Error rate = failed / total."""
        for _ in range(8):
            monitor.record_request(success=True, latency=1.0)
        for _ in range(2):
            monitor.record_request(success=False, latency=1.0)
        assert monitor.error_rate == pytest.approx(0.2)

    def test_error_rate_no_requests(self, monitor: LLMMonitor) -> None:
        """No data yields zero error rate."""
        assert monitor.error_rate == 0.0

    def test_error_rate_all_failures(self, monitor: LLMMonitor) -> None:
        """100% failure rate when all requests fail."""
        for _ in range(5):
            monitor.record_request(success=False, latency=1.0)
        assert monitor.error_rate == pytest.approx(1.0)

    def test_success_rate_calculation(self, monitor: LLMMonitor) -> None:
        """Success rate = successful / total."""
        for _ in range(8):
            monitor.record_request(success=True, latency=1.0)
        for _ in range(2):
            monitor.record_request(success=False, latency=1.0)
        assert monitor.success_rate == pytest.approx(0.8)

    def test_success_rate_no_requests(self, monitor: LLMMonitor) -> None:
        """No data yields zero success rate."""
        assert monitor.success_rate == 0.0

    def test_error_plus_success_rate_equals_one(self, monitor: LLMMonitor) -> None:
        """Error rate + success rate = 1.0 for any non-empty dataset."""
        for _ in range(7):
            monitor.record_request(success=True, latency=1.0)
        for _ in range(3):
            monitor.record_request(success=False, latency=1.0)
        assert monitor.error_rate + monitor.success_rate == pytest.approx(1.0)


# =========================================================================
# Section 11: LLMMonitor — Alert Tests
# Per README.md line 269: "High latency > 10s"
# Per README.md line 270: "High error rate > 5%"
# =========================================================================


class TestMonitorAlerts:
    """Verify alert triggering for high latency and high error rate."""

    def test_high_latency_alert(self, monitor: LLMMonitor) -> None:
        """Latency > 10s triggers 'llm_high_latency' warning."""
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            monitor.record_request(success=True, latency=11.0)
            high_lat_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_high_latency"
            ]
            assert len(high_lat_calls) >= 1

    def test_no_latency_alert_under_threshold(self, monitor: LLMMonitor) -> None:
        """Latency <= 10s does not trigger a high-latency warning."""
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            monitor.record_request(success=True, latency=9.0)
            high_lat_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_high_latency"
            ]
            assert len(high_lat_calls) == 0

    def test_latency_alert_at_exact_threshold(self, monitor: LLMMonitor) -> None:
        """Latency exactly at 10.0s does not trigger (threshold is strict >)."""
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            monitor.record_request(success=True, latency=10.0)
            high_lat_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_high_latency"
            ]
            assert len(high_lat_calls) == 0

    def test_high_error_rate_alert(self, monitor: LLMMonitor) -> None:
        """Error rate > 5% triggers 'llm_high_error_rate' warning."""
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            monitor.record_request(success=True, latency=1.0)
            monitor.record_request(success=False, latency=1.0)
            # 1 fail / 2 total = 50% > 5%
            high_err_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_high_error_rate"
            ]
            assert len(high_err_calls) >= 1

    def test_no_error_rate_alert_when_within_threshold(self, monitor: LLMMonitor) -> None:
        """Error rate <= 5% does not trigger an alert."""
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            # 1 fail among 100 total = 1% < 5%
            for _ in range(99):
                monitor.record_request(success=True, latency=1.0)
            monitor.record_request(success=False, latency=1.0)
            # The error rate after the last call is 1/100 = 1%
            high_err_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_high_error_rate"
            ]
            assert len(high_err_calls) == 0

    def test_high_latency_alert_includes_latency_value(self, monitor: LLMMonitor) -> None:
        """High latency alert log entry includes the actual latency value."""
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            monitor.record_request(success=True, latency=15.0)
            high_lat_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_high_latency"
            ]
            assert len(high_lat_calls) == 1
            call_kwargs = high_lat_calls[0][1]
            assert call_kwargs["latency"] == 15.0
            assert call_kwargs["threshold"] == 10.0


# =========================================================================
# Section 12: LLMMonitor — check_budget Tests
# =========================================================================


class TestMonitorCheckBudget:
    """Verify budget check delegation to the attached budget manager."""

    def test_check_budget_without_manager(self, monitor: LLMMonitor) -> None:
        """No budget manager → all requests allowed regardless of cost."""
        assert monitor.check_budget(1.0) is True
        assert monitor.check_budget(1_000_000.0) is True

    def test_check_budget_with_manager_under_budget(self, monitor_with_budget: LLMMonitor) -> None:
        """Under-budget check returns True."""
        assert monitor_with_budget.check_budget(1.0) is True

    def test_check_budget_with_manager_over_budget(self, monitor_with_budget: LLMMonitor) -> None:
        """Exhausted budget check returns False (hard block)."""
        # Exhaust the budget via record_request
        monitor_with_budget.record_request(success=True, cost=100.0)
        # 100.0 + 1.0 = 101.0 > 100.0 → BLOCKED
        assert monitor_with_budget.check_budget(1.0) is False


# =========================================================================
# Section 13: LLMMonitor — get_metrics and Utility Tests
# =========================================================================


class TestMonitorMetrics:
    """Verify get_metrics, get_daily_summary, and reset."""

    def test_get_metrics_complete(self, monitor: LLMMonitor) -> None:
        """Metrics dictionary contains all required keys."""
        monitor.record_request(
            success=True,
            input_tokens=100,
            output_tokens=50,
            cost=0.01,
            latency=1.0,
        )
        metrics = monitor.get_metrics()
        required_keys = [
            "total_requests",
            "successful_requests",
            "failed_requests",
            "total_tokens",
            "total_cost",
            "average_latency",
            "p95_latency",
            "error_rate",
            "success_rate",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_get_metrics_values_are_correct(self, monitor: LLMMonitor) -> None:
        """Metric values match expected calculations."""
        monitor.record_request(success=True, input_tokens=100, output_tokens=50, cost=0.01, latency=2.0)
        monitor.record_request(success=False, input_tokens=0, output_tokens=0, cost=0.0, latency=5.0)
        metrics = monitor.get_metrics()
        assert metrics["total_requests"] == 2
        assert metrics["successful_requests"] == 1
        assert metrics["failed_requests"] == 1
        assert metrics["total_tokens"] == 150
        assert metrics["total_cost"] == pytest.approx(0.01)

    def test_get_metrics_with_budget_manager(self, monitor_with_budget: LLMMonitor) -> None:
        """Metrics include budget_metrics when a budget manager is attached."""
        monitor_with_budget.record_request(success=True, cost=10.0)
        metrics = monitor_with_budget.get_metrics()
        assert "budget_metrics" in metrics
        assert metrics["budget_metrics"]["current_spend"] == pytest.approx(10.0)

    def test_get_daily_summary(self, monitor: LLMMonitor) -> None:
        """Daily summary contains llm-prefixed fields per spec."""
        monitor.record_request(success=True, cost=0.01, latency=1.0)
        summary = monitor.get_daily_summary()
        required_keys = [
            "llm_requests",
            "llm_cost_usd",
            "llm_successful_requests",
            "llm_failed_requests",
            "llm_total_tokens",
            "llm_average_latency",
            "llm_p95_latency",
            "llm_error_rate",
        ]
        for key in required_keys:
            assert key in summary, f"Missing daily summary key: {key}"

    def test_get_daily_summary_values(self, monitor: LLMMonitor) -> None:
        """Daily summary values match the recorded data."""
        monitor.record_request(success=True, cost=0.05, latency=2.0)
        monitor.record_request(success=True, cost=0.03, latency=1.0)
        summary = monitor.get_daily_summary()
        assert summary["llm_requests"] == 2
        assert summary["llm_cost_usd"] == pytest.approx(0.08, abs=0.001)
        assert summary["llm_successful_requests"] == 2
        assert summary["llm_failed_requests"] == 0

    def test_monitor_reset(self, monitor: LLMMonitor) -> None:
        """Reset clears all counters and the latency buffer."""
        monitor.record_request(
            success=True, cost=1.0, latency=5.0, input_tokens=100, output_tokens=50,
        )
        monitor.record_request(success=False, latency=3.0)
        monitor.reset()
        assert monitor.total_requests == 0
        assert monitor.successful_requests == 0
        assert monitor.failed_requests == 0
        assert monitor.total_tokens == 0
        assert monitor.total_cost == 0.0
        assert monitor.average_latency == 0.0
        assert monitor.p95_latency == 0.0

    def test_monitor_reset_preserves_budget_manager(self, monitor_with_budget: LLMMonitor) -> None:
        """Reset does NOT clear the budget manager state (per specification)."""
        monitor_with_budget.record_request(success=True, cost=50.0)
        monitor_with_budget.reset()
        # Budget manager still has the accumulated spend
        assert monitor_with_budget._budget_manager.current_spend == pytest.approx(50.0)
        # Monitor counters are zeroed
        assert monitor_with_budget.total_requests == 0
        assert monitor_with_budget.total_cost == 0.0


# =========================================================================
# Section 14: Integration with conftest Fixtures
# Verify that conftest-provided fixtures work correctly.
# =========================================================================


class TestConftestFixtureIntegration:
    """Verify conftest-provided fixtures are functional and compatible."""

    def test_conftest_llm_monitor_fixture(self, llm_monitor: LLMMonitor) -> None:
        """Conftest llm_monitor provides a working monitor with budget."""
        assert llm_monitor.total_requests == 0
        assert llm_monitor._budget_manager is not None
        # The budget manager from conftest has $100 budget
        assert llm_monitor._budget_manager.monthly_budget == 100.0

    def test_conftest_llm_budget_manager_fixture(self, llm_budget_manager: LLMBudgetManager) -> None:
        """Conftest llm_budget_manager provides a standard $100 manager."""
        assert llm_budget_manager.monthly_budget == 100.0
        assert llm_budget_manager.current_spend == 0.0
        assert llm_budget_manager.budget_warning_threshold == 0.80
        assert llm_budget_manager.block_threshold == 1.0

    def test_conftest_llm_monitor_records_request(self, llm_monitor: LLMMonitor) -> None:
        """Conftest monitor can record requests and delegate to budget."""
        llm_monitor.record_request(success=True, cost=10.0, latency=1.0)
        assert llm_monitor.total_requests == 1
        assert llm_monitor._budget_manager.current_spend == pytest.approx(10.0)

    def test_conftest_budget_manager_blocks_at_limit(self, llm_budget_manager: LLMBudgetManager) -> None:
        """Conftest budget manager enforces the hard block at $100."""
        llm_budget_manager.current_spend = 99.0
        assert llm_budget_manager.check_budget_before_request(estimated_cost=2.0) is False


# =========================================================================
# Section 15: Edge Cases and Additional Coverage
# =========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions for comprehensive coverage."""

    def test_monitor_handles_very_large_request_count(self, monitor: LLMMonitor) -> None:
        """Monitor handles a high volume of requests without error."""
        for _ in range(500):
            monitor.record_request(success=True, latency=0.1)
        assert monitor.total_requests == 500
        assert monitor.average_latency == pytest.approx(0.1)

    def test_budget_manager_fractional_costs(self) -> None:
        """Budget manager handles very small fractional costs correctly."""
        manager = LLMBudgetManager(monthly_budget_usd=100.0)
        for _ in range(1000):
            manager.record_spend(0.001)
        assert manager.current_spend == pytest.approx(1.0, abs=0.01)

    def test_monitor_cost_and_tokens_after_mixed_operations(self, monitor: LLMMonitor) -> None:
        """Cost and token tracking remains consistent through mixed ops."""
        monitor.record_request(success=True, input_tokens=100, output_tokens=50, cost=0.01, latency=1.0)
        monitor.record_request(success=False, input_tokens=200, output_tokens=0, cost=0.005, latency=2.0)
        monitor.record_request(success=True, input_tokens=0, output_tokens=150, cost=0.02, latency=0.5)
        assert monitor.total_tokens == 500  # 150 + 200 + 150
        assert monitor.total_cost == pytest.approx(0.035)
        assert monitor.total_input_tokens == 300
        assert monitor.total_output_tokens == 200

    def test_budget_manager_month_start_is_utc(self, budget_manager: LLMBudgetManager) -> None:
        """The internal month start timestamp is timezone-aware (UTC)."""
        assert budget_manager._month_start.tzinfo is not None

    def test_monitor_p95_with_identical_latencies(self, monitor: LLMMonitor) -> None:
        """P95 with all identical latencies equals that latency value."""
        for _ in range(50):
            monitor.record_request(success=True, latency=2.5)
        assert monitor.p95_latency == pytest.approx(2.5)

    def test_small_budget_manager_warns_at_80_percent(self, small_budget_manager: LLMBudgetManager) -> None:
        """Small $10 budget warns at $8 (80% of $10)."""
        small_budget_manager.current_spend = 7.0
        with patch("app.llm.llm_monitor.logger") as mock_logger:
            # projected = 7 + 2 = 9 > 8.0 (80% of 10) → warning
            result = small_budget_manager.check_budget_before_request(estimated_cost=2.0)
            assert result is True
            warning_calls = [
                c for c in mock_logger.warning.call_args_list
                if c[0][0] == "llm_budget_warning"
            ]
            assert len(warning_calls) == 1
