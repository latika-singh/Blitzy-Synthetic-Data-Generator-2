"""Discrepancy Injector — Rate-based injection trigger and orchestration.

The ``DiscrepancyInjector`` is the single entry point for all discrepancy injection
into generated transactions. It decides whether a discrepancy should be injected
(rate-based trigger), selects the type (weighted by category), validates parameters
against configured bounds, orchestrates the injection via the appropriate concrete
``BaseDiscrepancy`` implementation, and creates ground truth records via
``GroundTruthGenerator``.

Circuit Breaker Configuration (AAP §0.1.2):
    - Failure threshold: 20 consecutive failures → circuit OPEN
    - Recovery timeout: 30 seconds before attempting half-open probe
    - Fallback: Skip discrepancy (transaction proceeds without injection)

Retry Policy (AAP §0.1.2):
    - Max attempts: 1 (no retry)
    - Backoff strategy: None
    - Timeout: 5 seconds
    - Fallback: Skip discrepancy

Discrepancy Injection Rules (AAP §0.7.5):
    - Rate: Actual injection rate within ±1% of configured target (default 2%)
    - Difficulty: Easy (70%) / Medium (30%) / Hard (0%) — ±5% tolerance
    - Parameter Bounds: ALL parameters within configured bounds
    - Auto-Adjust: When enabled, clamp out-of-bounds parameters to nearest bound
    - Ground Truth: 100% coverage — every injection has a ground truth record
    - Transaction Linkage: Every ground truth references valid transaction IDs

References:
    - AAP Section 0.5.1 Group 5: Discrepancy Injection System
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.4: Error Handling Conventions (circuit breaker, retry)
    - AAP Section 0.7.1: Constructor injection (ADR-003)
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
    Dict,
    List,
    Optional,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.exceptions import DiscrepancyInjectionError
from app.transactions.constants import (
    CIRCUIT_BREAKER_CONFIG,
    DISCREPANCY_DEFAULTS,
    RETRY_POLICIES,
)

if TYPE_CHECKING:
    from app.discrepancies.base_discrepancy import BaseDiscrepancy
    from app.discrepancies.discrepancy_catalog import DiscrepancyCatalog
    from app.discrepancies.ground_truth_generator import GroundTruthGenerator
    from app.events.event_bus import EventBus

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Transaction-type to discrepancy-category mapping
# ---------------------------------------------------------------------------
# Maps transaction_type strings emitted by P2P and O2C generators to the
# discrepancy catalog categories used by DiscrepancyCatalog.
# Unknown transaction types default to querying ALL categories.

_TRANSACTION_TYPE_TO_CATEGORY: Dict[str, str] = {
    "purchase_order": "p2p",
    "goods_receipt": "p2p",
    "vendor_invoice": "p2p",
    "vendor_payment": "p2p",
    "three_way_match": "p2p",
    "sales_order": "o2c",
    "shipment": "o2c",
    "customer_invoice": "o2c",
    "customer_payment": "o2c",
    "journal_entry": "gl",
}


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Configuration Models
# ═══════════════════════════════════════════════════════════════════════════


class DiscrepancyConfig(BaseModel):
    """Configuration for the DiscrepancyInjector.

    All financial values use :class:`~decimal.Decimal` — NEVER ``float``
    (AAP §0.7.2).  Defaults are drawn from
    :data:`~app.transactions.constants.DISCREPANCY_DEFAULTS`.

    Attributes:
        injection_rate: Target injection rate (0.0–1.0). Default 0.02 (2%).
        rate_tolerance: Acceptable ±tolerance on the actual injection rate.
        difficulty_distribution: Weighted distribution mapping difficulty
            level names to probabilities (must sum to 1.0).
        difficulty_tolerance: Acceptable ±tolerance on each difficulty bucket.
        auto_adjust_to_bounds: When ``True``, out-of-bounds discrepancy
            parameters are silently clamped to the nearest bound value.
            When ``False``, out-of-bounds parameters raise
            :class:`DiscrepancyInjectionError`.
        enabled: Master on/off switch. When ``False`` the injector returns
            ``InjectionResult(injected=False)`` immediately.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    injection_rate: Decimal = Field(
        default=Decimal("0.02"),
        description="Target injection rate (0.0-1.0)",
    )
    rate_tolerance: Decimal = Field(
        default=Decimal("0.01"),
        description="Acceptable ±tolerance on rate",
    )
    difficulty_distribution: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "easy": Decimal("0.70"),
            "medium": Decimal("0.30"),
            "hard": Decimal("0.00"),
        },
        description="Difficulty distribution (must sum to 1.0)",
    )
    difficulty_tolerance: Decimal = Field(
        default=Decimal("0.05"),
        description="Acceptable ±tolerance on distribution",
    )
    auto_adjust_to_bounds: bool = Field(
        default=True,
        description="Clamp out-of-bounds parameters",
    )
    enabled: bool = Field(
        default=True,
        description="Master switch for injection",
    )


class InjectionResult(BaseModel):
    """Result of a discrepancy injection attempt.

    Returned by :meth:`DiscrepancyInjector.check_and_inject` to communicate
    the outcome of a single injection check — whether a discrepancy was
    injected, the type and difficulty selected, the ground truth record ID,
    the modified transaction data, and any error or skip reason.

    Attributes:
        injected: ``True`` if a discrepancy was actually injected.
        discrepancy_type: Type code (e.g. ``'P2P-001'``), or ``None``.
        difficulty: ``'easy'`` or ``'medium'``, or ``None``.
        ground_truth_id: UUID of the created ground truth record, or ``None``.
        modified_transaction: The transaction data after injection.
        original_values: Field values before modification.
        modified_values: Field values after modification.
        financial_impact: Dollar impact of the discrepancy.
        error_message: Human-readable error string if injection failed.
        skipped_reason: Reason string if injection was skipped.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    injected: bool = Field(
        default=False,
        description="Whether a discrepancy was actually injected",
    )
    discrepancy_type: Optional[str] = Field(
        default=None,
        description="Type code (e.g., 'P2P-001')",
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="'easy' or 'medium'",
    )
    ground_truth_id: Optional[UUID] = Field(
        default=None,
        description="Ground truth record ID",
    )
    modified_transaction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Modified transaction data",
    )
    original_values: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Original values before modification",
    )
    modified_values: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Modified values after injection",
    )
    financial_impact: Optional[Decimal] = Field(
        default=None,
        description="Financial impact of discrepancy",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error if injection failed",
    )
    skipped_reason: Optional[str] = Field(
        default=None,
        description="Reason if injection was skipped",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker State
# ═══════════════════════════════════════════════════════════════════════════


class CircuitState(str, Enum):
    """Three-state circuit breaker model for the discrepancy injection subsystem.

    States:
        CLOSED:    Normal operation — injections proceed.
        OPEN:      Failures exceeded the 20-failure threshold — all
                   injections are skipped until ``recovery_timeout`` elapses.
        HALF_OPEN: Recovery probe — a single injection attempt is permitted;
                   success resets to CLOSED, failure returns to OPEN.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class _CircuitBreaker:
    """Internal circuit breaker for discrepancy injection.

    Configuration from AAP §0.1.2 and
    :data:`~app.transactions.constants.CIRCUIT_BREAKER_CONFIG`:

        - ``failure_threshold``: 20 consecutive failures before opening.
        - ``recovery_timeout_seconds``: 30 seconds before half-open probe.
        - ``fallback``: ``skip_discrepancy``.

    The breaker tracks a monotonic last-failure timestamp so that wall-clock
    adjustments do not affect recovery timing.

    This class is intentionally *not* thread-safe; it is accessed only from
    the single-threaded asyncio event loop.
    """

    def __init__(
        self,
        failure_threshold: int = 20,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialise the circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before the
                circuit transitions from CLOSED to OPEN.
            recovery_timeout_seconds: Seconds to wait in OPEN state before
                transitioning to HALF_OPEN for a probe attempt.
        """
        self._failure_threshold: int = failure_threshold
        self._recovery_timeout: float = recovery_timeout_seconds
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._success_count: int = 0
        self._total_trips: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Return the current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Return the current consecutive failure count."""
        return self._failure_count

    # ------------------------------------------------------------------
    # State Transition Methods
    # ------------------------------------------------------------------

    def can_execute(self) -> bool:
        """Determine whether the next injection attempt should proceed.

        Returns ``True`` when the circuit is CLOSED or HALF_OPEN (probe).
        When OPEN, checks whether ``recovery_timeout`` has elapsed; if so,
        transitions to HALF_OPEN and returns ``True`` for a single probe
        attempt.

        Returns:
            ``True`` if the next injection should proceed, ``False`` if
            the circuit is OPEN and recovery has not elapsed.
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check whether the recovery timeout has elapsed
            if self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.warning(
                        "circuit_breaker_state_change",
                        service_name="transactions",
                        component="DiscrepancyInjector",
                        previous_state=CircuitState.OPEN.value,
                        new_state=CircuitState.HALF_OPEN.value,
                        elapsed_seconds=round(elapsed, 2),
                        recovery_timeout=self._recovery_timeout,
                    )
                    return True
            return False

        # HALF_OPEN — allow exactly one probe attempt
        return True

    def record_success(self) -> None:
        """Record a successful injection and reset the breaker to CLOSED.

        Called after a discrepancy injection completes without error.
        Resets the consecutive failure counter and transitions the circuit
        back to CLOSED (if it was HALF_OPEN or already CLOSED).
        """
        previous_state = self._state
        self._failure_count = 0
        self._last_failure_time = None
        self._success_count += 1

        if self._state != CircuitState.CLOSED:
            self._state = CircuitState.CLOSED
            logger.warning(
                "circuit_breaker_state_change",
                service_name="transactions",
                component="DiscrepancyInjector",
                previous_state=previous_state.value,
                new_state=CircuitState.CLOSED.value,
                reason="successful_probe",
            )

    def record_failure(self) -> None:
        """Record a failed injection attempt and potentially trip the breaker.

        Increments the consecutive failure counter.  If the counter reaches
        ``failure_threshold``, transitions the circuit to OPEN and records
        the monotonic timestamp for recovery timeout calculation.

        If the circuit is HALF_OPEN when a failure occurs, it immediately
        returns to OPEN state with a fresh recovery timeout.
        """
        self._failure_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # Probe failed — return to OPEN with fresh timeout
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
            self._total_trips += 1
            logger.warning(
                "circuit_breaker_state_change",
                service_name="transactions",
                component="DiscrepancyInjector",
                previous_state=CircuitState.HALF_OPEN.value,
                new_state=CircuitState.OPEN.value,
                reason="probe_failed",
                failure_count=self._failure_count,
            )
            return

        if self._failure_count >= self._failure_threshold:
            previous_state = self._state
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
            self._total_trips += 1
            logger.warning(
                "circuit_breaker_state_change",
                service_name="transactions",
                component="DiscrepancyInjector",
                previous_state=previous_state.value,
                new_state=CircuitState.OPEN.value,
                reason="failure_threshold_exceeded",
                failure_count=self._failure_count,
                failure_threshold=self._failure_threshold,
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Return circuit breaker metrics for diagnostics.

        Returns:
            Dictionary with state, failure count, success count, and trip count.
        """
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_trips": self._total_trips,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout_seconds": self._recovery_timeout,
        }


# ═══════════════════════════════════════════════════════════════════════════
# DiscrepancyInjector — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class DiscrepancyInjector:
    """Rate-based discrepancy injection orchestrator.

    The ``DiscrepancyInjector`` is the single entry point consumed by
    transaction generators (P2P, O2C, GL) to inject discrepancies into
    generated transactions.  It implements the full injection pipeline:

    1. **Rate check** — compare ``rng.random()`` against the configured
       injection rate to decide whether to inject.
    2. **Difficulty selection** — weighted random choice from the configured
       difficulty distribution (easy: 70%, medium: 30%, hard: 0%).
    3. **Type selection** — weighted random choice from eligible catalog
       entries matching the transaction category and difficulty.
    4. **Parameter validation** — validate (and optionally auto-adjust)
       parameters against the catalog's declared bounds.
    5. **Injection execution** — delegate to the concrete
       :class:`BaseDiscrepancy` subclass obtained from the catalog.
    6. **Ground truth creation** — create a 16-field ground truth record
       via :class:`GroundTruthGenerator`.
    7. **Event publication** — publish a ``DiscrepancyDetected`` event
       via :class:`EventBus`.

    All dependencies are received via constructor injection (ADR-003).
    Every parameter is ``Optional`` with ``None`` default to enable testing
    and incremental integration.

    Circuit breaker and timeout enforcement protect the injection pipeline
    from cascading failures (AAP §0.7.4).

    Attributes:
        _config: Injection configuration (rate, distribution, toggles).
        _catalog: Discrepancy type catalog for lookups.
        _ground_truth_generator: Creates ground truth records.
        _event_bus: Publishes DiscrepancyDetected events.
        _circuit_breaker: Circuit breaker guarding the injection pipeline.
    """

    # ------------------------------------------------------------------
    # Construction (ADR-003: Constructor Injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        catalog: Optional[DiscrepancyCatalog] = None,
        ground_truth_generator: Optional[GroundTruthGenerator] = None,
        event_bus: Optional[EventBus] = None,
        config: Optional[DiscrepancyConfig] = None,
    ) -> None:
        """Initialise the DiscrepancyInjector with injected dependencies.

        Args:
            catalog: Central registry of 35+ discrepancy type codes and
                their implementation classes.  When ``None``, type selection
                returns ``None`` and injection is skipped.
            ground_truth_generator: Creates 16-field ground truth records
                for every successful injection.  When ``None``, ground truth
                creation is silently skipped (violates AAP §0.7.5 in
                production but enables unit testing).
            event_bus: Dual-mode event bus for publishing
                ``DiscrepancyDetected`` events.  When ``None``, event
                publication is silently skipped.
            config: Injection configuration.  When ``None``, default
                :class:`DiscrepancyConfig` is used.
        """
        self._config: DiscrepancyConfig = config or DiscrepancyConfig()
        self._catalog = catalog
        self._ground_truth_generator = ground_truth_generator
        self._event_bus = event_bus

        # Circuit breaker — use constants from CIRCUIT_BREAKER_CONFIG if available
        cb_config = CIRCUIT_BREAKER_CONFIG.get("discrepancy_injection", {})
        self._circuit_breaker = _CircuitBreaker(
            failure_threshold=int(cb_config.get("failure_threshold", 20)),
            recovery_timeout_seconds=float(
                cb_config.get("recovery_timeout_seconds", 30.0)
            ),
        )

        # Retry/timeout policy from constants
        retry_policy = RETRY_POLICIES.get("discrepancy_injection", {})
        self._timeout_seconds: float = float(
            retry_policy.get("timeout_seconds", 5)
        )

        # --------------- Metrics counters --------------------------------
        self._injection_count: int = 0
        self._skip_count: int = 0
        self._error_count: int = 0
        self._total_checks: int = 0

        # Per-difficulty counters
        self._difficulty_counts: Dict[str, int] = {
            "easy": 0,
            "medium": 0,
            "hard": 0,
        }

        # Per-category counters
        self._category_counts: Dict[str, int] = {
            "p2p": 0,
            "o2c": 0,
            "gl": 0,
            "control": 0,
        }

        logger.info(
            "discrepancy_injector_initialized",
            service_name="transactions",
            component="DiscrepancyInjector",
            injection_rate=str(self._config.injection_rate),
            auto_adjust=self._config.auto_adjust_to_bounds,
            enabled=self._config.enabled,
            has_catalog=catalog is not None,
            has_ground_truth=ground_truth_generator is not None,
            has_event_bus=event_bus is not None,
        )

    # ------------------------------------------------------------------
    # Core Method: check_and_inject
    # ------------------------------------------------------------------

    async def check_and_inject(
        self,
        transaction: Dict[str, Any],
        transaction_type: str,
        context: Dict[str, Any],
        rng: random.Random,
    ) -> InjectionResult:
        """Check if a discrepancy should be injected and execute if so.

        This is the single entry point called by transaction generators
        for every generated transaction.  It performs the full injection
        pipeline: rate check → difficulty selection → type selection →
        parameter validation → injection execution → ground truth creation
        → event publication.

        The entire injection pipeline is wrapped in a 5-second
        ``asyncio.wait_for`` timeout per AAP §0.1.2 retry policy table.

        Args:
            transaction: The transaction data to potentially modify.
            transaction_type: Transaction type string, e.g.
                ``"purchase_order"``, ``"vendor_invoice"``,
                ``"sales_order"``.
            context: Generation context dictionary containing at minimum
                ``simulation_id`` and ``trace_id``.
            rng: Seeded :class:`random.Random` instance for deterministic
                behavior.  CRITICAL: MUST use this RNG, NEVER module-level
                ``random``.

        Returns:
            :class:`InjectionResult` with injection status and, if injected,
            the ground truth record ID, modified transaction, and financial
            impact.
        """
        self._total_checks += 1
        sim_id = context.get("simulation_id", "unknown")
        trace_id = context.get("trace_id", "unknown")

        # ---- Gate 1: Master switch ----------------------------------------
        if not self._config.enabled:
            self._skip_count += 1
            logger.debug(
                "discrepancy_injection_skipped",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="injection_disabled",
            )
            return InjectionResult(
                injected=False,
                skipped_reason="injection_disabled",
            )

        # ---- Gate 2: Circuit breaker --------------------------------------
        if not self._circuit_breaker.can_execute():
            self._skip_count += 1
            logger.debug(
                "discrepancy_injection_skipped",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="circuit_breaker_open",
                circuit_state=self._circuit_breaker.state.value,
            )
            return InjectionResult(
                injected=False,
                skipped_reason="circuit_breaker_open",
            )

        # ---- Gate 3: Rate check -------------------------------------------
        roll = rng.random()
        if roll > float(self._config.injection_rate):
            self._skip_count += 1
            logger.debug(
                "discrepancy_injection_skipped",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="rate_check_not_triggered",
                roll=round(roll, 6),
                injection_rate=str(self._config.injection_rate),
            )
            return InjectionResult(
                injected=False,
                skipped_reason="rate_check_not_triggered",
            )

        # ---- Rate check passed — attempt injection with timeout -----------
        logger.debug(
            "discrepancy_injection_triggered",
            service_name="transactions",
            component="DiscrepancyInjector",
            simulation_id=str(sim_id),
            trace_id=str(trace_id),
            transaction_type=transaction_type,
            roll=round(roll, 6),
        )

        try:
            result = await asyncio.wait_for(
                self._execute_injection(
                    transaction=transaction,
                    transaction_type=transaction_type,
                    context=context,
                    rng=rng,
                ),
                timeout=self._timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            self._circuit_breaker.record_failure()
            self._error_count += 1
            logger.warning(
                "discrepancy_injection_failed",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="timeout",
                timeout_seconds=self._timeout_seconds,
            )
            return InjectionResult(
                injected=False,
                error_message=f"Injection timed out after {self._timeout_seconds}s",
            )
        except DiscrepancyInjectionError as exc:
            self._circuit_breaker.record_failure()
            self._error_count += 1
            logger.warning(
                "discrepancy_injection_failed",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="discrepancy_injection_error",
                error=exc.message,
                details=exc.details,
            )
            return InjectionResult(
                injected=False,
                error_message=exc.message,
            )
        except Exception as exc:
            self._circuit_breaker.record_failure()
            self._error_count += 1
            logger.warning(
                "discrepancy_injection_failed",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="unexpected_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return InjectionResult(
                injected=False,
                error_message=f"Unexpected error: {type(exc).__name__}: {exc}",
            )

    # ------------------------------------------------------------------
    # Private: Injection Execution Pipeline
    # ------------------------------------------------------------------

    async def _execute_injection(
        self,
        transaction: Dict[str, Any],
        transaction_type: str,
        context: Dict[str, Any],
        rng: random.Random,
    ) -> InjectionResult:
        """Execute the full injection pipeline (steps 5–12 of check_and_inject).

        This coroutine is wrapped by :func:`asyncio.wait_for` in the caller
        with a 5-second timeout.

        Steps:
            1. Select difficulty (weighted)
            2. Select discrepancy type from catalog (weighted by base_rate)
            3. Retrieve and validate/adjust parameters
            4. Get the concrete BaseDiscrepancy implementation from catalog
            5. Call discrepancy.inject(transaction, params, rng)
            6. Create ground truth record
            7. Publish DiscrepancyDetected event
            8. Record circuit breaker success
            9. Return populated InjectionResult
        """
        sim_id = context.get("simulation_id", "unknown")
        trace_id = context.get("trace_id", "unknown")

        # Step 1: Select difficulty
        difficulty = self._select_difficulty(rng)

        # Step 2: Select discrepancy type
        type_code = self._select_discrepancy_type(
            transaction_type=transaction_type,
            difficulty=difficulty,
            rng=rng,
        )

        if type_code is None:
            # No eligible type found — skip injection gracefully
            self._skip_count += 1
            logger.debug(
                "discrepancy_injection_skipped",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="no_eligible_type",
                transaction_type=transaction_type,
                difficulty=difficulty,
            )
            return InjectionResult(
                injected=False,
                difficulty=difficulty,
                skipped_reason="no_eligible_type",
            )

        logger.debug(
            "discrepancy_type_selected",
            service_name="transactions",
            component="DiscrepancyInjector",
            simulation_id=str(sim_id),
            trace_id=str(trace_id),
            type_code=type_code,
            difficulty=difficulty,
            transaction_type=transaction_type,
        )

        # Step 3: Retrieve parameters from catalog entry and validate/adjust
        entry = self._catalog.get_entry(type_code) if self._catalog else None
        raw_params: Dict[str, Any] = {}
        if entry is not None and entry.parameter_bounds:
            # Generate default parameters from bounds using the seeded RNG
            raw_params = self._generate_default_params(
                entry.parameter_bounds, rng
            )

        validated_params = self._validate_and_adjust_params(
            type_code=type_code,
            params=raw_params,
        )

        # Step 4: Get the concrete BaseDiscrepancy implementation
        impl_class = (
            self._catalog.get_implementation_class(type_code)
            if self._catalog
            else None
        )
        if impl_class is None:
            self._skip_count += 1
            logger.debug(
                "discrepancy_injection_skipped",
                service_name="transactions",
                component="DiscrepancyInjector",
                simulation_id=str(sim_id),
                trace_id=str(trace_id),
                reason="no_implementation_class",
                type_code=type_code,
            )
            return InjectionResult(
                injected=False,
                discrepancy_type=type_code,
                difficulty=difficulty,
                skipped_reason="no_implementation_class",
            )

        # Step 5: Instantiate and call inject()
        discrepancy_instance: BaseDiscrepancy = impl_class()
        modified_transaction, ground_truth_data = discrepancy_instance.inject(
            transaction, validated_params, rng
        )

        # Extract ground truth fields from the returned data
        original_values: Dict[str, Any] = ground_truth_data.get(
            "original_values", {}
        )
        modified_values: Dict[str, Any] = ground_truth_data.get(
            "modified_values", {}
        )
        affected_fields: List[str] = ground_truth_data.get(
            "affected_fields", []
        )
        financial_impact: Decimal = Decimal(
            str(ground_truth_data.get("financial_impact", Decimal("0.00")))
        )
        detection_method: str = ground_truth_data.get("detection_method", "")
        detection_difficulty: str = ground_truth_data.get(
            "detection_difficulty", difficulty
        )
        description: str = ground_truth_data.get(
            "description",
            f"Discrepancy {type_code} injected into {transaction_type}",
        )
        gt_metadata: Dict[str, Any] = ground_truth_data.get("metadata", {})

        # Step 6: Create ground truth record (100% coverage per AAP §0.7.5)
        ground_truth_id: Optional[UUID] = None
        transaction_ids = self._extract_transaction_ids(transaction, context)

        # Derive category from entry or type code
        category = ""
        if entry is not None:
            category = entry.category
        elif type_code.startswith("P2P"):
            category = "p2p"
        elif type_code.startswith("O2C"):
            category = "o2c"
        elif type_code.startswith("GL"):
            category = "gl"
        elif type_code.startswith("CTL"):
            category = "control"

        if self._ground_truth_generator is not None:
            gt_record = self._ground_truth_generator.create_record(
                type_code=type_code,
                category=category,
                difficulty=difficulty,
                transaction_ids=transaction_ids,
                affected_fields=affected_fields if affected_fields else ["unknown"],
                original_values=original_values,
                modified_values=modified_values,
                detection_method=detection_method or "automated",
                detection_difficulty=detection_difficulty or difficulty,
                financial_impact=financial_impact,
                description=description,
                metadata=gt_metadata,
            )
            ground_truth_id = gt_record.discrepancy_id

        # Step 7: Publish DiscrepancyDetected event via EventBus
        await self._publish_discrepancy_event(
            type_code=type_code,
            transaction_type=transaction_type,
            context=context,
            ground_truth_id=ground_truth_id,
        )

        # Step 8: Record circuit breaker success
        self._circuit_breaker.record_success()

        # Step 9: Update metrics
        self._injection_count += 1
        if difficulty in self._difficulty_counts:
            self._difficulty_counts[difficulty] += 1
        if category in self._category_counts:
            self._category_counts[category] += 1

        logger.info(
            "discrepancy_injected",
            service_name="transactions",
            component="DiscrepancyInjector",
            simulation_id=str(sim_id),
            trace_id=str(trace_id),
            type_code=type_code,
            difficulty=difficulty,
            category=category,
            transaction_type=transaction_type,
            financial_impact=str(financial_impact),
            ground_truth_id=str(ground_truth_id) if ground_truth_id else None,
        )

        return InjectionResult(
            injected=True,
            discrepancy_type=type_code,
            difficulty=difficulty,
            ground_truth_id=ground_truth_id,
            modified_transaction=modified_transaction,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=financial_impact,
        )

    # ------------------------------------------------------------------
    # Helper: Difficulty Selection
    # ------------------------------------------------------------------

    def _select_difficulty(self, rng: random.Random) -> str:
        """Select difficulty level based on configured distribution.

        Uses weighted random selection with the configured distribution:
        easy: 0.70, medium: 0.30, hard: 0.00.

        Filters out difficulties with zero weight before selection.

        CRITICAL: Uses the seeded ``rng`` parameter, NEVER module-level
        ``random``, to ensure deterministic reproducibility (AAP §0.7.1).

        Args:
            rng: Seeded :class:`random.Random` instance.

        Returns:
            Selected difficulty string (``'easy'`` or ``'medium'``).
        """
        distribution = self._config.difficulty_distribution

        # Filter out difficulties with zero weight
        eligible: List[str] = []
        weights: List[float] = []
        for diff, weight in distribution.items():
            weight_float = float(weight)
            if weight_float > 0:
                eligible.append(diff)
                weights.append(weight_float)

        if not eligible:
            # Fallback to 'easy' if all weights are zero (should not happen
            # with valid config, but handle gracefully)
            return "easy"

        selected = rng.choices(eligible, weights=weights, k=1)[0]
        return selected

    # ------------------------------------------------------------------
    # Helper: Type Selection
    # ------------------------------------------------------------------

    def _select_discrepancy_type(
        self,
        transaction_type: str,
        difficulty: str,
        rng: random.Random,
    ) -> Optional[str]:
        """Select a discrepancy type code from the catalog.

        Filters by transaction category (derived from ``transaction_type``)
        and ``difficulty``, then uses weighted random selection by
        ``base_rate`` across eligible catalog entries.

        Also queries the ``'control'`` category (CTL-001 through CTL-005)
        since control discrepancies apply to all transaction types.

        CRITICAL: Uses the seeded ``rng`` parameter, NEVER module-level
        ``random``.

        Args:
            transaction_type: Transaction type string (e.g.
                ``"purchase_order"``).
            difficulty: Selected difficulty level (``'easy'`` or
                ``'medium'``).
            rng: Seeded :class:`random.Random` instance.

        Returns:
            Selected type code string (e.g. ``'P2P-001'``), or ``None``
            if no eligible types exist in the catalog.
        """
        if self._catalog is None:
            return None

        # Map transaction type to discrepancy category
        category = _TRANSACTION_TYPE_TO_CATEGORY.get(transaction_type)

        # Collect eligible entries from the primary category
        eligible_entries: list = []
        if category is not None:
            eligible_entries.extend(
                self._catalog.get_by_category_and_difficulty(
                    category, difficulty
                )
            )

        # Also include control discrepancies (applicable to all types)
        eligible_entries.extend(
            self._catalog.get_by_category_and_difficulty(
                "control", difficulty
            )
        )

        if not eligible_entries:
            return None

        # Weighted selection by base_rate
        type_codes: List[str] = []
        base_rates: List[float] = []
        for entry in eligible_entries:
            type_codes.append(entry.type_code)
            base_rates.append(float(entry.base_rate))

        # Handle case where all base_rates are zero
        if sum(base_rates) <= 0:
            return rng.choice(type_codes) if type_codes else None

        selected = rng.choices(type_codes, weights=base_rates, k=1)[0]
        return selected

    # ------------------------------------------------------------------
    # Helper: Parameter Generation
    # ------------------------------------------------------------------

    def _generate_default_params(
        self,
        parameter_bounds: Dict[str, Dict[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Generate default parameter values from the declared bounds.

        For each parameter, picks a value within [min, max] using the
        seeded RNG, respecting the declared type.

        Args:
            parameter_bounds: Mapping from parameter name to
                ``{"min": ..., "max": ..., "type": ...}`` bound spec.
            rng: Seeded :class:`random.Random` instance.

        Returns:
            Dictionary of generated parameter values.
        """
        params: Dict[str, Any] = {}
        for param_name, bounds in parameter_bounds.items():
            min_val = bounds.get("min", 0)
            max_val = bounds.get("max", min_val)
            param_type = bounds.get("type", "float")

            if param_type == "bool":
                params[param_name] = rng.random() < 0.5
            elif param_type == "int":
                params[param_name] = rng.randint(int(min_val), int(max_val))
            elif param_type == "decimal":
                min_d = Decimal(str(min_val))
                max_d = Decimal(str(max_val))
                range_d = max_d - min_d
                factor = Decimal(str(rng.random()))
                params[param_name] = min_d + range_d * factor
            else:
                # float — generate uniform in [min, max]
                params[param_name] = rng.uniform(float(min_val), float(max_val))

        return params

    # ------------------------------------------------------------------
    # Helper: Parameter Validation and Auto-Adjust
    # ------------------------------------------------------------------

    def _validate_and_adjust_params(
        self,
        type_code: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate discrepancy parameters against bounds and auto-adjust.

        For each parameter, checks the value against the catalog's declared
        bounds.  Behaviour when a parameter falls outside bounds:

        - If ``auto_adjust_to_bounds`` is ``True``: clamp to the nearest
          bound value (min or max).
        - If ``auto_adjust_to_bounds`` is ``False``: raise
          :class:`DiscrepancyInjectionError`.

        Args:
            type_code: The discrepancy type code.
            params: Dictionary of parameter name → value.

        Returns:
            A new dictionary with validated (and possibly clamped) values.

        Raises:
            DiscrepancyInjectionError: If ``auto_adjust_to_bounds`` is
                ``False`` and a parameter is outside its declared bounds.
        """
        if self._catalog is None:
            return dict(params)

        bounds = self._catalog.get_parameter_bounds(type_code)
        if not bounds:
            return dict(params)

        adjusted: Dict[str, Any] = dict(params)

        for param_name, param_value in adjusted.items():
            if param_name not in bounds:
                # Parameter not in bounds spec — pass through
                continue

            bound_spec = bounds[param_name]
            min_val = bound_spec.get("min")
            max_val = bound_spec.get("max")
            param_type = bound_spec.get("type", "float")

            # Skip boolean parameters — no numeric bounds
            if param_type == "bool":
                continue

            # Convert to comparable numeric type
            if param_type == "decimal":
                cmp_val = Decimal(str(param_value))
                cmp_min = Decimal(str(min_val)) if min_val is not None else None
                cmp_max = Decimal(str(max_val)) if max_val is not None else None
            elif param_type == "int":
                cmp_val = int(param_value) if not isinstance(param_value, int) else param_value
                cmp_min = int(min_val) if min_val is not None else None
                cmp_max = int(max_val) if max_val is not None else None
            else:
                cmp_val = float(param_value) if not isinstance(param_value, float) else param_value
                cmp_min = float(min_val) if min_val is not None else None
                cmp_max = float(max_val) if max_val is not None else None

            # Check and adjust
            out_of_bounds = False
            if cmp_min is not None and cmp_val < cmp_min:
                out_of_bounds = True
                if self._config.auto_adjust_to_bounds:
                    adjusted[param_name] = cmp_min
                    logger.debug(
                        "parameter_auto_adjusted",
                        service_name="transactions",
                        component="DiscrepancyInjector",
                        type_code=type_code,
                        param_name=param_name,
                        original_value=str(param_value),
                        adjusted_to=str(cmp_min),
                        bound="min",
                    )
                else:
                    raise DiscrepancyInjectionError(
                        f"Parameter '{param_name}' value {param_value} is below "
                        f"minimum bound {min_val} for type {type_code}",
                        details={
                            "type_code": type_code,
                            "param_name": param_name,
                            "value": str(param_value),
                            "min_bound": str(min_val),
                            "max_bound": str(max_val),
                        },
                    )

            if cmp_max is not None and cmp_val > cmp_max:
                out_of_bounds = True
                if self._config.auto_adjust_to_bounds:
                    adjusted[param_name] = cmp_max
                    logger.debug(
                        "parameter_auto_adjusted",
                        service_name="transactions",
                        component="DiscrepancyInjector",
                        type_code=type_code,
                        param_name=param_name,
                        original_value=str(param_value),
                        adjusted_to=str(cmp_max),
                        bound="max",
                    )
                else:
                    raise DiscrepancyInjectionError(
                        f"Parameter '{param_name}' value {param_value} exceeds "
                        f"maximum bound {max_val} for type {type_code}",
                        details={
                            "type_code": type_code,
                            "param_name": param_name,
                            "value": str(param_value),
                            "min_bound": str(min_val),
                            "max_bound": str(max_val),
                        },
                    )

        return adjusted

    # ------------------------------------------------------------------
    # Helper: Transaction ID Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_transaction_ids(
        transaction: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[UUID]:
        """Extract transaction UUIDs from transaction data and context.

        Looks for ``transaction_id`` in both the transaction dictionary
        and the context.  Ensures at least one UUID is returned (generates
        one if none is found).

        Args:
            transaction: The transaction data dictionary.
            context: The generation context dictionary.

        Returns:
            Non-empty list of transaction UUIDs.
        """
        ids: List[UUID] = []

        for source in (transaction, context):
            raw_id = source.get("transaction_id")
            if raw_id is not None:
                if isinstance(raw_id, UUID):
                    ids.append(raw_id)
                else:
                    try:
                        ids.append(UUID(str(raw_id)))
                    except (ValueError, AttributeError):
                        pass

        # Check for additional IDs in arrays
        for source in (transaction, context):
            raw_ids = source.get("transaction_ids")
            if isinstance(raw_ids, (list, tuple)):
                for rid in raw_ids:
                    if isinstance(rid, UUID):
                        ids.append(rid)
                    else:
                        try:
                            ids.append(UUID(str(rid)))
                        except (ValueError, AttributeError):
                            pass

        # Ensure at least one ID exists (generate if needed)
        if not ids:
            ids.append(uuid4())

        # Deduplicate while preserving order
        seen: set = set()
        unique_ids: List[UUID] = []
        for uid in ids:
            if uid not in seen:
                seen.add(uid)
                unique_ids.append(uid)

        return unique_ids

    # ------------------------------------------------------------------
    # Helper: Event Publication
    # ------------------------------------------------------------------

    async def _publish_discrepancy_event(
        self,
        type_code: str,
        transaction_type: str,
        context: Dict[str, Any],
        ground_truth_id: Optional[UUID],
    ) -> None:
        """Publish a ``DiscrepancyDetected`` event via EventBus.

        Follows the lazy import pattern from
        ``app/orchestration/approval_system.py`` — the
        ``DiscrepancyDetected`` event class is imported inside this method
        rather than at module level.

        If ``event_bus`` is ``None``, logs a debug message and returns
        silently.  Exceptions from ``EventBus.publish()`` are caught and
        logged — event publication failures MUST NOT block the injection
        pipeline.

        Args:
            type_code: Discrepancy type code (e.g. ``'P2P-001'``).
            transaction_type: Transaction type string.
            context: Generation context dictionary.
            ground_truth_id: UUID of the ground truth record, or ``None``.
        """
        if self._event_bus is None:
            logger.debug(
                "discrepancy_event_skipped",
                service_name="transactions",
                component="DiscrepancyInjector",
                reason="no_event_bus",
                type_code=type_code,
            )
            return

        sim_id = context.get("simulation_id")
        try:
            from app.events.event_types import DiscrepancyDetected

            # Determine severity based on type code prefix
            severity = "medium"
            if type_code.startswith("CTL"):
                severity = "high"
            elif type_code.startswith("GL"):
                severity = "high"

            event = DiscrepancyDetected(
                payload={
                    "discrepancy_type": type_code,
                    "transaction_id": str(
                        context.get("transaction_id", "")
                    ),
                    "severity": severity,
                    "details": (
                        f"Discrepancy {type_code} injected into "
                        f"{transaction_type} transaction"
                    ),
                    "detected_by_agent_id": str(
                        context.get("agent_id", "system")
                    ),
                    "ground_truth_id": (
                        str(ground_truth_id) if ground_truth_id else None
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                simulation_id=(
                    sim_id if isinstance(sim_id, UUID) else uuid4()
                ),
            )
            await self._event_bus.publish(event)

        except Exception as exc:
            # Log but do NOT block the injection pipeline on event failure
            logger.warning(
                "discrepancy_event_publish_failed",
                service_name="transactions",
                component="DiscrepancyInjector",
                event_type="DiscrepancyDetected",
                type_code=type_code,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Public: Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return comprehensive injection metrics for operational monitoring.

        Provides counters, actual rates, actual difficulty distribution, and
        circuit breaker state.  These metrics are consumed by
        ``SimulationMetrics`` for daily/monthly aggregation.

        Returns:
            Dictionary with keys: ``total_checks``, ``injection_count``,
            ``skip_count``, ``error_count``, ``actual_injection_rate``,
            ``actual_difficulty_distribution``, ``category_counts``,
            ``circuit_breaker``, ``config``.
        """
        actual_rate = Decimal("0.00")
        if self._total_checks > 0:
            actual_rate = Decimal(str(self._injection_count)) / Decimal(
                str(self._total_checks)
            )

        # Compute actual difficulty distribution
        total_injections = max(self._injection_count, 1)
        actual_difficulty_dist: Dict[str, str] = {}
        for diff, count in self._difficulty_counts.items():
            actual_difficulty_dist[diff] = str(
                Decimal(str(count)) / Decimal(str(total_injections))
            )

        return {
            "total_checks": self._total_checks,
            "injection_count": self._injection_count,
            "skip_count": self._skip_count,
            "error_count": self._error_count,
            "actual_injection_rate": str(actual_rate),
            "actual_difficulty_distribution": actual_difficulty_dist,
            "category_counts": dict(self._category_counts),
            "circuit_breaker": self._circuit_breaker.get_metrics(),
            "config": {
                "injection_rate": str(self._config.injection_rate),
                "rate_tolerance": str(self._config.rate_tolerance),
                "auto_adjust_to_bounds": self._config.auto_adjust_to_bounds,
                "enabled": self._config.enabled,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "DiscrepancyConfig",
    "InjectionResult",
    "CircuitState",
    "DiscrepancyInjector",
]
