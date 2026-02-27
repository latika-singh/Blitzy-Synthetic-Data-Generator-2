"""Rework Loop Engine — Orchestrates the intelligent rework loop cycle.

The ``ReworkLoopEngine`` is the central orchestrator for the intelligent rework
loop (Project 3). It classifies validation failures, selects fix scenarios from
a catalog sorted by success rate, applies corrections, re-validates the result,
and escalates unresolvable errors for human review.

Rework Cycle:
    1. **Classify** — FailureClassifier determines if failure is planned
       discrepancy (within/outside parameters) or unplanned error
    2. **Select Fix** — FixScenarioCatalog selects highest-success-rate
       applicable fix scenario, excluding previously failed scenarios
    3. **Apply Fix** — FixScenarioExecutor applies the selected scenario
       steps to the transaction (10s per-scenario timeout)
    4. **Re-validate** — Run validation again on the fixed transaction
    5. **Escalate** — If max attempts exceeded (3) or all scenarios fail,
       escalate to admin for human review

Constraints:
    - Max 3 attempts per transaction (configurable via REWORK_LOOP_MAX_ATTEMPTS)
    - Escalation threshold: >5% failure rate (configurable via REWORK_ESCALATION_THRESHOLD)
    - Circuit breaker: 50 failures → open circuit, 120s recovery timeout
    - Retry policy: 3 retries, Linear backoff (1s, 2s, 3s), 30s timeout
    - Fallback: Escalate to admin
    - Rework loop completion: Within 30 seconds per transaction

Design Decisions:
    - Constructor injection (ADR-003): All dependencies via __init__, all Optional with None default
    - EventBus (ADR-001): Publishes rework events (attempt, success, escalation)
    - structlog: JSON to stdout, includes service_name, component, trace_id, simulation_id
    - Deterministic: Seeded random.Random, never module-level RNG
    - Pydantic V2: Data contracts for ReworkAttempt and ReworkResult

References:
    - AAP Section 0.5.1 Group 6: Intelligent Rework Loop
    - AAP Section 0.1.2: Retry Policy Table (rework_loop_fix: 3 attempts, Linear 1s/2s/3s, 30s timeout)
    - AAP Section 0.1.2: Circuit Breaker (rework_loop: 50 failures, 120s recovery)
    - AAP Section 0.7.4: Error Handling Conventions
    - AAP Section 0.7.7: Logging Specification (rework_loop at INFO dev / WARNING prod)
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.exceptions import ReworkLoopError, TransactionError
from app.transactions.constants import (
    CIRCUIT_BREAKER_CONFIG,
    RETRY_POLICIES,
    REWORK_LOOP_CONFIG,
)

# ---------------------------------------------------------------------------
# TYPE_CHECKING-only imports — prevents circular runtime dependencies with
# the rework subsystem modules and the EventBus.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from app.events.event_bus import EventBus
    from app.rework.failure_classifier import ClassificationResult, FailureClassifier
    from app.rework.fix_scenario_catalog import FixScenario, FixScenarioCatalog
    from app.rework.fix_scenario_executor import FixExecutionResult, FixScenarioExecutor

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only, JSON format)
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = ["ReworkLoopEngine", "ReworkAttempt", "ReworkResult"]


# ═══════════════════════════════════════════════════════════════════════════
# Credential Scrubbing Helper
# ═══════════════════════════════════════════════════════════════════════════

_SENSITIVE_KEYS = frozenset({
    "api_key",
    "apikey",
    "api_secret",
    "authorization",
    "auth_token",
    "token",
    "secret",
    "password",
    "credential",
})


def _scrub_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *context* with sensitive keys redacted.

    Prevents API keys and other credentials from appearing in structured
    log output (AAP §0.7.4).

    Args:
        context: The raw context dictionary.

    Returns:
        A shallow copy with sensitive values replaced by ``"***REDACTED***"``.
    """
    scrubbed: Dict[str, Any] = {}
    for key, value in context.items():
        if key.lower() in _SENSITIVE_KEYS:
            scrubbed[key] = "***REDACTED***"
        else:
            scrubbed[key] = value
    return scrubbed


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Models — Cross-subsystem contracts
# ═══════════════════════════════════════════════════════════════════════════


class ReworkAttempt(BaseModel):
    """Record of a single rework attempt on a transaction.

    Each attempt captures the fix scenario applied, the steps executed,
    whether the fix resolved the issue, and timing information.

    Attributes:
        attempt_number: 1-based attempt counter (max 3 per transaction).
        scenario_name: Name of the fix scenario that was applied.
        scenario_steps: List of step names executed within the scenario.
        success: Whether the fix resolved the validation failure.
        duration_ms: Wall-clock duration of this attempt in milliseconds.
        error_message: Error description if the fix failed, else ``None``.
        timestamp: UTC timestamp when this attempt was executed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    attempt_number: int = Field(
        ..., ge=1, le=3, description="Attempt number (1-3)"
    )
    scenario_name: str = Field(
        ..., description="Fix scenario that was applied"
    )
    scenario_steps: List[str] = Field(
        default_factory=list, description="Steps executed in this attempt"
    )
    success: bool = Field(
        default=False, description="Whether the fix resolved the issue"
    )
    duration_ms: float = Field(
        default=0.0, ge=0.0, description="Duration in milliseconds"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if fix failed"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class ReworkResult(BaseModel):
    """Complete result of rework loop processing for a transaction.

    Produced by :meth:`ReworkLoopEngine.process_validation_failure` and
    consumed by the simulation engine to decide whether to persist the
    fixed transaction or escalate for human review.

    Attributes:
        transaction_id: UUID of the transaction being reworked.
        transaction_type: Business type of the transaction (e.g. ``"purchase_order"``).
        classification: Classification result — ``"planned_discrepancy"`` or
            ``"unplanned_error"``.
        attempts: Ordered list of all rework attempts made.
        resolved: Whether the validation failure was successfully resolved.
        escalated: Whether the failure was escalated to admin for human review.
        final_status: Terminal status — ``"resolved"``, ``"escalated"``, or ``"failed"``.
        total_duration_ms: Total wall-clock time for the entire rework loop.
        modified_transaction: The corrected transaction data, or ``None`` if unresolved.
        escalation_reason: Human-readable reason for escalation, or ``None``.
        simulation_id: Simulation run identifier for tracing.
        trace_id: Distributed trace identifier for cross-service correlation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transaction_id: UUID = Field(
        ..., description="Transaction being reworked"
    )
    transaction_type: str = Field(
        default="", description="Transaction type"
    )
    classification: str = Field(
        default="", description="planned_discrepancy or unplanned_error"
    )
    attempts: List[ReworkAttempt] = Field(
        default_factory=list, description="All rework attempts"
    )
    resolved: bool = Field(
        default=False, description="Whether the issue was resolved"
    )
    escalated: bool = Field(
        default=False, description="Whether escalated to admin"
    )
    final_status: str = Field(
        default="pending", description="resolved, escalated, or failed"
    )
    total_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total rework duration"
    )
    modified_transaction: Optional[Dict[str, Any]] = Field(
        default=None, description="Fixed transaction data"
    )
    escalation_reason: Optional[str] = Field(
        default=None, description="Reason for escalation"
    )
    simulation_id: Optional[UUID] = Field(
        default=None, description="Simulation run identifier"
    )
    trace_id: Optional[UUID] = Field(
        default=None, description="Distributed trace identifier"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker — Internal state machine for rework loop protection
# ═══════════════════════════════════════════════════════════════════════════


class CircuitState(str, Enum):
    """Circuit breaker states for rework loop protection.

    Members:
        CLOSED: Normal operation — rework loop processes all failures.
        OPEN: Failures exceeded threshold — skip rework, escalate directly.
        HALF_OPEN: Recovery probe — allow a single rework attempt to test
            whether the underlying issue has been resolved.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _CircuitBreaker:
    """Internal circuit breaker for rework loop operations.

    Implements the standard three-state circuit breaker pattern to protect
    the rework loop from cascading failures.  When the failure count exceeds
    the configured threshold, the circuit opens and all subsequent rework
    requests are immediately escalated.  After the recovery timeout elapses,
    the circuit enters half-open state and allows a single probe request.

    Configuration from AAP §0.1.2:
        - failure_threshold: 50 (rework_loop specific)
        - recovery_timeout_seconds: 120 (rework_loop specific)
        - fallback: escalate_to_admin

    Args:
        failure_threshold: Number of consecutive failures before opening
            the circuit.
        recovery_timeout_seconds: Seconds to wait before transitioning
            from OPEN to HALF_OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 50,
        recovery_timeout_seconds: float = 120.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._state = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._success_count: int = 0

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> CircuitState:
        """Current circuit breaker state."""
        return self._state

    # ------------------------------------------------------------------ #
    # State Transitions
    # ------------------------------------------------------------------ #

    def can_execute(self) -> bool:
        """Determine whether the rework loop is allowed to process a request.

        Returns ``True`` if the circuit is CLOSED or HALF_OPEN (probe).
        If the circuit is OPEN and the recovery timeout has elapsed,
        transitions to HALF_OPEN and returns ``True``.

        Returns:
            ``True`` if execution is permitted, ``False`` otherwise.
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    previous_state = self._state
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        "circuit_breaker_state_transition",
                        service_name="transactions",
                        component="ReworkLoopEngine",
                        previous_state=previous_state.value,
                        new_state=self._state.value,
                        reason="recovery_timeout_elapsed",
                        elapsed_seconds=round(elapsed, 2),
                        recovery_timeout=self._recovery_timeout,
                    )
                    return True
            return False

        # HALF_OPEN — allow one probe attempt
        return True

    def record_success(self) -> None:
        """Record a successful rework execution.

        Resets the failure counter and transitions back to CLOSED state
        if the circuit was in HALF_OPEN (probe succeeded).
        """
        previous_state = self._state
        self._failure_count = 0
        self._success_count += 1

        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            logger.info(
                "circuit_breaker_state_transition",
                service_name="transactions",
                component="ReworkLoopEngine",
                previous_state=previous_state.value,
                new_state=CircuitState.CLOSED.value,
                reason="success_recorded",
                success_count=self._success_count,
            )

    def record_failure(self) -> None:
        """Record a failed rework execution.

        Increments the failure counter.  If the count reaches the threshold,
        the circuit opens.  If the circuit was HALF_OPEN (probe failed),
        it transitions back to OPEN.
        """
        self._failure_count += 1
        previous_state = self._state

        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — re-open the circuit
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
            logger.warning(
                "circuit_breaker_state_transition",
                service_name="transactions",
                component="ReworkLoopEngine",
                previous_state=previous_state.value,
                new_state=CircuitState.OPEN.value,
                reason="half_open_probe_failed",
                failure_count=self._failure_count,
                failure_threshold=self._failure_threshold,
            )
        elif self._failure_count >= self._failure_threshold:
            # Threshold exceeded — open the circuit
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
            logger.warning(
                "circuit_breaker_state_transition",
                service_name="transactions",
                component="ReworkLoopEngine",
                previous_state=previous_state.value,
                new_state=CircuitState.OPEN.value,
                reason="failure_threshold_exceeded",
                failure_count=self._failure_count,
                failure_threshold=self._failure_threshold,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Return circuit breaker metrics for monitoring.

        Returns:
            Dictionary containing state, failure count, success count,
            and last failure time.
        """
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_seconds": self._recovery_timeout,
            "last_failure_time": self._last_failure_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# ReworkLoopEngine — Core orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class ReworkLoopEngine:
    """Orchestrates the intelligent rework loop cycle.

    The rework loop engine is the central entry point for processing
    validation failures in the P3 transaction workflow system.  It
    coordinates the full classify → select fix → apply → re-validate
    → escalate cycle for each failed transaction.

    Receives all dependencies via constructor injection (ADR-003).  Every
    parameter is ``Optional`` with ``None`` default to enable testing
    flexibility and incremental integration.

    Args:
        failure_classifier: Optional classifier for determining whether a
            failure is a planned discrepancy or an unplanned error.
        fix_catalog: Optional catalog of fix scenarios sorted by success rate.
        fix_executor: Optional executor for applying fix scenario steps.
        event_bus: Optional EventBus for publishing rework lifecycle events.
        max_attempts: Maximum number of fix attempts per transaction (default 3).
        escalation_threshold: Failure rate above which escalation is triggered
            (default 0.05 = 5%).
        timeout_seconds: Per-transaction rework timeout in seconds (default 30).
    """

    def __init__(
        self,
        *,
        failure_classifier: Optional[FailureClassifier] = None,
        fix_catalog: Optional[FixScenarioCatalog] = None,
        fix_executor: Optional[FixScenarioExecutor] = None,
        event_bus: Optional[EventBus] = None,
        max_attempts: int = REWORK_LOOP_CONFIG["max_attempts_per_transaction"],
        escalation_threshold: float = float(
            REWORK_LOOP_CONFIG["escalation_failure_rate_threshold"]
        ),
        timeout_seconds: float = float(
            REWORK_LOOP_CONFIG["timeout_seconds"]
        ),
    ) -> None:
        # Injected dependencies (all Optional per ADR-003)
        self._failure_classifier = failure_classifier
        self._fix_catalog = fix_catalog
        self._fix_executor = fix_executor
        self._event_bus = event_bus

        # Configuration
        self._max_attempts: int = max_attempts
        self._escalation_threshold: float = escalation_threshold
        self._timeout_seconds: float = timeout_seconds

        # Circuit breaker — configured from AAP §0.1.2
        _cb_config = CIRCUIT_BREAKER_CONFIG.get("rework_loop", {})
        self._circuit_breaker = _CircuitBreaker(
            failure_threshold=int(_cb_config.get("failure_threshold", 50)),
            recovery_timeout_seconds=float(
                _cb_config.get("recovery_timeout_seconds", 120.0)
            ),
        )

        # Aggregate metrics
        self._total_processed: int = 0
        self._total_resolved: int = 0
        self._total_escalated: int = 0
        self._total_failed: int = 0
        self._scenario_success_counts: Dict[str, int] = {}
        self._scenario_failure_counts: Dict[str, int] = {}

        logger.info(
            "rework_loop_engine_initialized",
            service_name="transactions",
            component="ReworkLoopEngine",
            max_attempts=self._max_attempts,
            escalation_threshold=self._escalation_threshold,
            timeout_seconds=self._timeout_seconds,
            has_classifier=failure_classifier is not None,
            has_catalog=fix_catalog is not None,
            has_executor=fix_executor is not None,
            has_event_bus=event_bus is not None,
        )

    # ------------------------------------------------------------------ #
    # Core Method — process_validation_failure
    # ------------------------------------------------------------------ #

    async def process_validation_failure(
        self,
        transaction: Dict[str, Any],
        validation_errors: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> ReworkResult:
        """Process a validation failure through the rework loop.

        Orchestrates the full classify → select fix → apply → re-validate
        → escalate cycle for a single transaction.

        Args:
            transaction: The failed transaction data dictionary.  Must
                contain a ``transaction_id`` key (UUID or str).
            validation_errors: List of validation error detail dicts, each
                containing at minimum ``type`` and ``message`` keys.
            context: Generation context carrying ``simulation_id``,
                ``trace_id``, ``current_date``, and other simulation state.

        Returns:
            :class:`ReworkResult` with resolution status and attempt history.

        Raises:
            ReworkLoopError: If the rework loop itself fails fatally
                (e.g. internal state corruption).
        """
        # Step 1: Start timing
        start_time = time.monotonic()

        # Step 2: Build initial result
        raw_txn_id = transaction.get("transaction_id")
        if isinstance(raw_txn_id, UUID):
            txn_id = raw_txn_id
        elif isinstance(raw_txn_id, str):
            try:
                txn_id = UUID(raw_txn_id)
            except ValueError:
                txn_id = uuid4()
        else:
            txn_id = uuid4()

        raw_sim_id = context.get("simulation_id")
        sim_id: Optional[UUID] = None
        if isinstance(raw_sim_id, UUID):
            sim_id = raw_sim_id
        elif isinstance(raw_sim_id, str):
            try:
                sim_id = UUID(raw_sim_id)
            except ValueError:
                sim_id = None

        raw_trace_id = context.get("trace_id")
        trace_id: Optional[UUID] = None
        if isinstance(raw_trace_id, UUID):
            trace_id = raw_trace_id
        elif isinstance(raw_trace_id, str):
            try:
                trace_id = UUID(raw_trace_id)
            except ValueError:
                trace_id = None

        result = ReworkResult(
            transaction_id=txn_id,
            transaction_type=transaction.get("transaction_type", ""),
            simulation_id=sim_id,
            trace_id=trace_id,
        )

        safe_ctx = _scrub_context(context)

        logger.info(
            "rework_loop_started",
            service_name="transactions",
            component="ReworkLoopEngine",
            transaction_id=str(txn_id),
            transaction_type=result.transaction_type,
            validation_error_count=len(validation_errors),
            simulation_id=str(sim_id) if sim_id else None,
            trace_id=str(trace_id) if trace_id else None,
        )

        try:
            # Step 3: Circuit breaker check
            if not self._circuit_breaker.can_execute():
                result.escalated = True
                result.final_status = "escalated"
                result.escalation_reason = "circuit_breaker_open"
                self._total_processed += 1
                self._total_escalated += 1
                result.total_duration_ms = (
                    time.monotonic() - start_time
                ) * 1000

                logger.warning(
                    "rework_loop_circuit_breaker_open",
                    service_name="transactions",
                    component="ReworkLoopEngine",
                    transaction_id=str(txn_id),
                    circuit_state=self._circuit_breaker.state.value,
                    simulation_id=str(sim_id) if sim_id else None,
                    trace_id=str(trace_id) if trace_id else None,
                )

                await self._publish_rework_event(result, context)
                return result

            # Step 4: Classify failure
            classification_type = "unplanned_error"
            if self._failure_classifier is not None:
                try:
                    classification = await self._failure_classifier.classify(
                        validation_errors, transaction, context
                    )
                    classification_type = classification.classification_type
                except Exception as classify_exc:
                    logger.warning(
                        "rework_loop_classification_failed",
                        service_name="transactions",
                        component="ReworkLoopEngine",
                        transaction_id=str(txn_id),
                        error=str(classify_exc),
                        simulation_id=str(sim_id) if sim_id else None,
                        trace_id=str(trace_id) if trace_id else None,
                    )
                    classification_type = "unplanned_error"

            result.classification = classification_type

            # Step 5: Wrap the attempt loop in asyncio.wait_for for timeout
            try:
                await asyncio.wait_for(
                    self._execute_rework_cycle(
                        result, transaction, validation_errors, context
                    ),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                result.escalated = True
                result.final_status = "escalated"
                result.escalation_reason = "rework_timeout"
                logger.warning(
                    "rework_loop_timeout",
                    service_name="transactions",
                    component="ReworkLoopEngine",
                    transaction_id=str(txn_id),
                    timeout_seconds=self._timeout_seconds,
                    attempts_completed=len(result.attempts),
                    simulation_id=str(sim_id) if sim_id else None,
                    trace_id=str(trace_id) if trace_id else None,
                )

        except ReworkLoopError:
            # Re-raise domain-specific fatal errors
            raise
        except TransactionError as txn_exc:
            # Handle other transaction errors gracefully
            result.escalated = True
            result.final_status = "escalated"
            result.escalation_reason = f"transaction_error: {txn_exc.message}"
            logger.error(
                "rework_loop_transaction_error",
                service_name="transactions",
                component="ReworkLoopEngine",
                transaction_id=str(txn_id),
                error=str(txn_exc),
                simulation_id=str(sim_id) if sim_id else None,
                trace_id=str(trace_id) if trace_id else None,
            )
        except Exception as exc:
            # Catch-all: wrap as ReworkLoopError for unexpected failures
            result.escalated = True
            result.final_status = "escalated"
            result.escalation_reason = f"unexpected_error: {str(exc)}"
            logger.error(
                "rework_loop_unexpected_error",
                service_name="transactions",
                component="ReworkLoopEngine",
                transaction_id=str(txn_id),
                error=str(exc),
                error_type=type(exc).__name__,
                simulation_id=str(sim_id) if sim_id else None,
                trace_id=str(trace_id) if trace_id else None,
            )

        # Step 13: Record metrics
        self._total_processed += 1
        if result.resolved:
            self._total_resolved += 1
            self._circuit_breaker.record_success()
        elif result.escalated:
            self._total_escalated += 1
            self._circuit_breaker.record_failure()
        else:
            self._total_failed += 1
            self._circuit_breaker.record_failure()

        # Step 15: Finalize timing
        result.total_duration_ms = (time.monotonic() - start_time) * 1000

        # Step 16: Publish event
        await self._publish_rework_event(result, context)

        # Step 17: Log result
        logger.info(
            "rework_loop_completed",
            service_name="transactions",
            component="ReworkLoopEngine",
            transaction_id=str(result.transaction_id),
            classification=result.classification,
            resolved=result.resolved,
            escalated=result.escalated,
            attempts=len(result.attempts),
            total_duration_ms=round(result.total_duration_ms, 2),
            final_status=result.final_status,
            simulation_id=str(result.simulation_id) if result.simulation_id else None,
            trace_id=str(result.trace_id) if result.trace_id else None,
        )

        return result

    # ------------------------------------------------------------------ #
    # Internal — Rework Cycle Execution
    # ------------------------------------------------------------------ #

    async def _execute_rework_cycle(
        self,
        result: ReworkResult,
        transaction: Dict[str, Any],
        validation_errors: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> None:
        """Execute the rework attempt loop (steps 6–12).

        This method is wrapped in ``asyncio.wait_for()`` by the caller to
        enforce the 30-second per-transaction timeout.

        Args:
            result: The :class:`ReworkResult` to populate with attempt data.
            transaction: The failed transaction data dictionary.
            validation_errors: List of validation error details.
            context: Generation context.
        """
        used_scenarios: List[str] = []
        current_transaction = dict(transaction)  # Work on a shallow copy

        for attempt_num in range(1, self._max_attempts + 1):
            attempt_start = time.monotonic()
            scenario_name = "none"
            scenario_steps: List[str] = []
            success = False
            error_message: Optional[str] = None

            try:
                # Step 7: Select fix scenario
                if self._fix_catalog is None:
                    error_message = "no_fix_catalog_available"
                    result.escalated = True
                    result.final_status = "escalated"
                    result.escalation_reason = "no_fix_catalog_available"

                    attempt_record = ReworkAttempt(
                        attempt_number=attempt_num,
                        scenario_name=scenario_name,
                        scenario_steps=scenario_steps,
                        success=False,
                        duration_ms=(time.monotonic() - attempt_start) * 1000,
                        error_message=error_message,
                    )
                    result.attempts.append(attempt_record)
                    return

                error_type = self._extract_error_type(validation_errors)
                scenario = self._fix_catalog.select_scenario(
                    error_type,
                    excluded_scenarios=set(used_scenarios),
                )

                if scenario is None:
                    # No applicable scenario available — escalate
                    error_message = "no_applicable_scenario"
                    result.escalated = True
                    result.final_status = "escalated"
                    result.escalation_reason = "no_applicable_fix_scenario"

                    logger.info(
                        "rework_loop_no_scenario",
                        service_name="transactions",
                        component="ReworkLoopEngine",
                        transaction_id=str(result.transaction_id),
                        error_type=error_type,
                        excluded_count=len(used_scenarios),
                        attempt_number=attempt_num,
                    )

                    attempt_record = ReworkAttempt(
                        attempt_number=attempt_num,
                        scenario_name=scenario_name,
                        scenario_steps=scenario_steps,
                        success=False,
                        duration_ms=(time.monotonic() - attempt_start) * 1000,
                        error_message=error_message,
                    )
                    result.attempts.append(attempt_record)
                    return

                # Extract scenario metadata
                scenario_name = scenario.name
                scenario_steps = [
                    step.step_name for step in scenario.steps
                ]

                # Step 9: Track used scenario (prevent re-selection)
                used_scenarios.append(scenario_name)

                # Step 8: Apply fix
                if self._fix_executor is None:
                    error_message = "no_fix_executor_available"
                    self._track_scenario_failure(scenario_name)

                    attempt_record = ReworkAttempt(
                        attempt_number=attempt_num,
                        scenario_name=scenario_name,
                        scenario_steps=scenario_steps,
                        success=False,
                        duration_ms=(time.monotonic() - attempt_start) * 1000,
                        error_message=error_message,
                    )
                    result.attempts.append(attempt_record)

                    result.escalated = True
                    result.final_status = "escalated"
                    result.escalation_reason = "no_fix_executor_available"
                    return

                fix_result = await self._fix_executor.execute(
                    scenario, current_transaction, context
                )

                # Step 10: Re-validate / check result
                if fix_result.success:
                    success = True
                    result.resolved = True
                    result.final_status = "resolved"
                    result.modified_transaction = fix_result.modified_transaction

                    # Update working copy for potential future use
                    if fix_result.modified_transaction is not None:
                        current_transaction = dict(
                            fix_result.modified_transaction
                        )

                    self._track_scenario_success(scenario_name)

                    logger.info(
                        "rework_loop_fix_success",
                        service_name="transactions",
                        component="ReworkLoopEngine",
                        transaction_id=str(result.transaction_id),
                        scenario_name=scenario_name,
                        attempt_number=attempt_num,
                    )
                else:
                    # Step 11: Fix failed — track and continue
                    error_message = (
                        fix_result.error_message or "fix_scenario_failed"
                    )
                    self._track_scenario_failure(scenario_name)

                    logger.info(
                        "rework_loop_fix_failed",
                        service_name="transactions",
                        component="ReworkLoopEngine",
                        transaction_id=str(result.transaction_id),
                        scenario_name=scenario_name,
                        attempt_number=attempt_num,
                        error=error_message,
                    )

            except asyncio.CancelledError:
                # Propagate cancellation (from timeout)
                raise
            except Exception as exc:
                error_message = str(exc)
                if scenario_name != "none":
                    self._track_scenario_failure(scenario_name)

                logger.warning(
                    "rework_loop_attempt_exception",
                    service_name="transactions",
                    component="ReworkLoopEngine",
                    transaction_id=str(result.transaction_id),
                    scenario_name=scenario_name,
                    attempt_number=attempt_num,
                    error=error_message,
                    error_type=type(exc).__name__,
                )

            # Record attempt
            attempt_duration_ms = (time.monotonic() - attempt_start) * 1000
            attempt_record = ReworkAttempt(
                attempt_number=attempt_num,
                scenario_name=scenario_name,
                scenario_steps=scenario_steps,
                success=success,
                duration_ms=round(attempt_duration_ms, 2),
                error_message=error_message if not success else None,
            )
            result.attempts.append(attempt_record)

            if success:
                break

            # Linear backoff between retry attempts (AAP §0.1.2:
            # rework_loop_fix — Linear 1s, 2s, 3s).  The delay equals the
            # attempt number: 1s after attempt 1, 2s after attempt 2, etc.
            # Only sleep if there are remaining attempts to avoid wasted time.
            if attempt_num < self._max_attempts:
                await asyncio.sleep(attempt_num)

        # Step 12: After loop — if not resolved, escalate
        if not result.resolved and not result.escalated:
            result.escalated = True
            result.final_status = "escalated"
            result.escalation_reason = "max_attempts_exceeded"

            logger.info(
                "rework_loop_max_attempts_exceeded",
                service_name="transactions",
                component="ReworkLoopEngine",
                transaction_id=str(result.transaction_id),
                max_attempts=self._max_attempts,
                attempts_made=len(result.attempts),
            )

    # ------------------------------------------------------------------ #
    # Helper — Extract error type from validation errors
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_error_type(
        validation_errors: List[Dict[str, Any]],
    ) -> str:
        """Extract the primary error type string from validation errors.

        Checks the ``type`` and ``error_type`` keys of the first validation
        error.  Falls back to ``"unknown_error"`` if no type can be extracted.

        Args:
            validation_errors: List of validation error detail dicts.

        Returns:
            A string identifying the error type for scenario selection.
        """
        if not validation_errors:
            return "unknown_error"
        first_error = validation_errors[0]
        error_type = first_error.get(
            "type",
            first_error.get("error_type", "unknown_error"),
        )
        return str(error_type) if error_type else "unknown_error"

    # ------------------------------------------------------------------ #
    # Helper — Publish rework event via EventBus
    # ------------------------------------------------------------------ #

    async def _publish_rework_event(
        self,
        result: ReworkResult,
        context: Dict[str, Any],
    ) -> None:
        """Publish a rework event via EventBus.

        Constructs a ``DiscrepancyDetected`` event carrying the rework result
        payload and publishes it through the injected EventBus.  Follows the
        safe publish pattern from ``app/orchestration/approval_system.py`` —
        wraps the publish call in ``try/except`` and tolerates EventBus
        failures without propagating exceptions.

        Args:
            result: The completed :class:`ReworkResult` to publish.
            context: Generation context for simulation_id extraction.
        """
        if self._event_bus is None:
            logger.debug(
                "rework_event_skipped_no_event_bus",
                service_name="transactions",
                component="ReworkLoopEngine",
                transaction_id=str(result.transaction_id),
            )
            return

        try:
            # Lazy import at runtime to avoid circular dependencies
            from app.events.event_types import Event, EventType

            # Build event payload
            payload: Dict[str, Any] = {
                "transaction_id": str(result.transaction_id),
                "transaction_type": result.transaction_type,
                "classification": result.classification,
                "resolved": result.resolved,
                "escalated": result.escalated,
                "final_status": result.final_status,
                "attempts_count": len(result.attempts),
                "total_duration_ms": round(result.total_duration_ms, 2),
                "escalation_reason": result.escalation_reason,
                "rework_source": "rework_loop_engine",
            }

            # Determine simulation_id for the event
            raw_sim_id = context.get("simulation_id")
            event_sim_id: UUID
            if isinstance(raw_sim_id, UUID):
                event_sim_id = raw_sim_id
            elif isinstance(raw_sim_id, str):
                try:
                    event_sim_id = UUID(raw_sim_id)
                except ValueError:
                    event_sim_id = uuid4()
            else:
                event_sim_id = uuid4()

            event = Event(
                simulation_id=event_sim_id,
                event_type=EventType.DISCREPANCY_DETECTED.value,
                payload=payload,
            )

            await self._event_bus.publish(event)

            logger.debug(
                "rework_event_published",
                service_name="transactions",
                component="ReworkLoopEngine",
                event_id=str(event.event_id),
                transaction_id=str(result.transaction_id),
                final_status=result.final_status,
            )

        except Exception as exc:
            # Tolerate EventBus failures — log but do not propagate
            logger.warning(
                "rework_event_publish_failed",
                service_name="transactions",
                component="ReworkLoopEngine",
                transaction_id=str(result.transaction_id),
                error=str(exc),
                error_type=type(exc).__name__,
            )

    # ------------------------------------------------------------------ #
    # Helper — Track scenario success/failure
    # ------------------------------------------------------------------ #

    def _track_scenario_success(self, scenario_name: str) -> None:
        """Track a successful fix scenario application.

        Increments the success counter for the named scenario in the
        aggregated metrics.

        Args:
            scenario_name: Name of the scenario that succeeded.
        """
        self._scenario_success_counts[scenario_name] = (
            self._scenario_success_counts.get(scenario_name, 0) + 1
        )

    def _track_scenario_failure(self, scenario_name: str) -> None:
        """Track a failed fix scenario application.

        Increments the failure counter for the named scenario in the
        aggregated metrics.

        Args:
            scenario_name: Name of the scenario that failed.
        """
        self._scenario_failure_counts[scenario_name] = (
            self._scenario_failure_counts.get(scenario_name, 0) + 1
        )

    # ------------------------------------------------------------------ #
    # Helper — Escalation threshold check
    # ------------------------------------------------------------------ #

    def _check_escalation_threshold(self) -> bool:
        """Check if the overall failure rate exceeds the escalation threshold.

        Computes the combined escalation + failure rate across all processed
        transactions and compares against the configured threshold (default 5%).

        Returns:
            ``True`` if the failure rate exceeds the threshold, ``False``
            otherwise.  Returns ``False`` if no transactions have been
            processed yet.
        """
        if self._total_processed == 0:
            return False
        failure_rate = (
            (self._total_escalated + self._total_failed) / self._total_processed
        )
        return failure_rate > self._escalation_threshold

    # ------------------------------------------------------------------ #
    # Metrics — get_metrics
    # ------------------------------------------------------------------ #

    def get_metrics(self) -> Dict[str, Any]:
        """Return comprehensive rework loop metrics for monitoring.

        Provides aggregate statistics on rework processing including
        resolution rates, escalation rates, per-scenario performance,
        circuit breaker state, and configuration parameters.

        Returns:
            Dictionary containing all rework loop metrics.
        """
        resolution_rate = (
            self._total_resolved / self._total_processed
            if self._total_processed > 0
            else 0.0
        )
        escalation_rate = (
            self._total_escalated / self._total_processed
            if self._total_processed > 0
            else 0.0
        )
        failure_rate = (
            self._total_failed / self._total_processed
            if self._total_processed > 0
            else 0.0
        )

        return {
            "total_processed": self._total_processed,
            "total_resolved": self._total_resolved,
            "total_escalated": self._total_escalated,
            "total_failed": self._total_failed,
            "resolution_rate": round(resolution_rate, 4),
            "escalation_rate": round(escalation_rate, 4),
            "failure_rate": round(failure_rate, 4),
            "scenario_success_counts": dict(self._scenario_success_counts),
            "scenario_failure_counts": dict(self._scenario_failure_counts),
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_metrics": self._circuit_breaker.get_metrics(),
            "max_attempts": self._max_attempts,
            "escalation_threshold": self._escalation_threshold,
            "timeout_seconds": self._timeout_seconds,
            "above_escalation_threshold": self._check_escalation_threshold(),
        }
