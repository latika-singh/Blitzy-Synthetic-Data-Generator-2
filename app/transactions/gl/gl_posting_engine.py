"""GL Posting Engine — journal entry creation, balance validation, and trial balance.

The ``GLPostingEngine`` is the **foundational posting layer** consumed by every P2P
and O2C transaction generator. It enforces the financial integrity rules that are
non-negotiable per AAP §0.7.2:

1. **Balance Validation**: Every journal entry MUST satisfy
   ``SUM(debits) = SUM(credits)`` within ``Decimal("0.01")``.
   Unbalanced entries are REJECTED before persistence.

2. **Account Validation**: Every account referenced in a journal entry must
   exist in the Chart of Accounts and have ``is_posting = True``.

3. **Period Validation**: Journal entries can only be posted to fiscal periods
   with status ``OPEN``. Posting to ``CLOSING`` or ``CLOSED`` periods raises
   ``PeriodClosedError``.

4. **Sequential Numbering**: Journal entries receive sequential numbers in
   the format ``JE-YYYY-NNNN`` (e.g., JE-2024-0001).

5. **Balance Update Delegation**: After posting, the engine delegates balance
   updates to the ``AccountBalanceManager``.

6. **Continuous Trial Balance**: After every posting batch, the cumulative
   trial balance is verified to equal zero within ``Decimal("0.01")``.

7. **Atomicity**: If ANY step in a multi-step posting fails, the ENTIRE
   transaction is rolled back (SQLAlchemy session rollback).

Retry Policy (AAP §0.1.2):
    - max_attempts: 3
    - backoff: exponential (1s, 2s, 4s)
    - timeout: 30s
    - fallback: rollback transaction

Circuit Breaker (AAP §0.1.2):
    - failure_threshold: 10 failures → open
    - recovery_timeout: 60s
    - fallback: rollback transaction

Decimal Configuration (AAP §0.7.2):
    - decimal.getcontext().prec = 28
    - decimal.getcontext().rounding = ROUND_HALF_UP
    - NEVER use float for monetary amounts

References:
    - AAP §0.5.1 Group 4: GL Integration (GLPostingEngine)
    - AAP §0.7.2: Financial Integrity Rules
    - AAP §0.7.3: Performance Requirements (GL posting ≥200/min)
    - AAP §0.7.4: Error Handling Conventions (retry, circuit breaker)
    - AAP §0.7.7: Logging Specification (GL Posting: INFO dev / WARNING prod)
"""

from __future__ import annotations

import asyncio
import decimal
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.transactions.exceptions import (
    BalanceError,
    GLPostingError,
    PeriodClosedError,
    ConcurrencyError,
    TransactionError,
)
from app.transactions.constants import (
    FINANCIAL_TOLERANCES,
    GL_BALANCE_TOLERANCE,
    TRIAL_BALANCE_TOLERANCE,
    CIRCUIT_BREAKER_CONFIG,
    RETRY_POLICIES,
    DOCUMENT_NUMBER_PREFIXES,
    PERIOD_STATUS_OPEN,
    PERIOD_STATUS_CLOSING,
    PERIOD_STATUS_CLOSED,
    DEBIT_NORMAL_ACCOUNT_TYPES,
    CREDIT_NORMAL_ACCOUNT_TYPES,
)

if TYPE_CHECKING:
    from app.events.event_bus import EventBus
    from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod
    from app.transactions.gl.account_balance_manager import AccountBalanceManager

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only, JSON format)
# GL Posting log level: INFO (dev) / WARNING (prod)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# CRITICAL: Decimal precision configuration (AAP §0.7.2)
# All financial calculations MUST use Decimal with prec=28 and ROUND_HALF_UP.
# NEVER use float for monetary amounts.
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Models
# ═══════════════════════════════════════════════════════════════════════════


class JournalEntryLine(BaseModel):
    """A single line within a journal entry.

    Each line represents a debit or credit to a specific GL account.
    Exactly one of ``debit_amount`` or ``credit_amount`` should be non-zero
    for a well-formed line, though both being zero is allowed (no-op line).

    Attributes:
        line_number: 1-based sequential line number within the parent entry.
        account_code: Chart of Accounts account code (e.g., ``"1010"``).
        account_name: Human-readable account name (resolved during posting).
        debit_amount: Debit amount (always >= 0, Decimal).
        credit_amount: Credit amount (always >= 0, Decimal).
        description: Line-level description / memo text.
        department: Optional department code for cost allocation.
        cost_center: Optional cost center code for cost allocation.
        reference: Source document reference (e.g., PO number, invoice number).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    line_number: int = Field(
        ..., ge=1, description="Line number within the entry"
    )
    account_code: str = Field(
        ..., description="Chart of Accounts account code"
    )
    account_name: Optional[str] = Field(
        default=None, description="Human-readable account name"
    )
    debit_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Debit amount (always >= 0)",
    )
    credit_amount: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Credit amount (always >= 0)",
    )
    description: str = Field(
        default="", description="Line item description"
    )
    department: Optional[str] = Field(
        default=None, description="Department code for cost allocation"
    )
    cost_center: Optional[str] = Field(
        default=None, description="Cost center code for cost allocation"
    )
    reference: Optional[str] = Field(
        default=None, description="Source document reference"
    )


class JournalEntry(BaseModel):
    """A complete journal entry with header and lines.

    Represents a balanced set of debits and credits that record a financial
    transaction in the General Ledger.  The entry is considered *balanced*
    when ``abs(total_debits - total_credits) <= Decimal("0.01")``.

    Sequential numbering follows the format ``JE-YYYY-NNNN`` (e.g.,
    ``JE-2024-0001``), assigned by the ``GLPostingEngine`` during posting.

    Attributes:
        entry_id: Unique journal entry identifier (UUID4).
        entry_number: Sequential number assigned during posting (JE-YYYY-NNNN).
        posting_date: Business date the entry is posted to.
        period_id: Fiscal period identifier (e.g., ``"2024-01"``).
        description: Entry-level description / memo.
        source_document_type: Type of originating document (e.g.,
            ``"purchase_order"``, ``"vendor_invoice"``).
        source_document_id: ID of the originating document.
        lines: List of :class:`JournalEntryLine` objects.
        total_debits: Calculated sum of all debit amounts.
        total_credits: Calculated sum of all credit amounts.
        is_balanced: Whether DR = CR within tolerance.
        status: Lifecycle status (``"draft"``, ``"posted"``, ``"reversed"``,
            ``"failed"``).
        created_by: Agent or system identifier that created this entry.
        simulation_id: Parent simulation run identifier.
        created_at: UTC timestamp of creation.
        posted_at: UTC timestamp of posting (set when status becomes ``"posted"``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry_id: UUID = Field(
        default_factory=uuid4, description="Unique journal entry ID"
    )
    entry_number: str = Field(
        default="", description="Sequential number JE-YYYY-NNNN"
    )
    posting_date: date = Field(
        ..., description="Date of posting"
    )
    period_id: Optional[str] = Field(
        default=None, description="Fiscal period identifier"
    )
    description: str = Field(
        default="", description="Entry description/memo"
    )
    source_document_type: Optional[str] = Field(
        default=None,
        description="e.g. 'purchase_order', 'vendor_invoice'",
    )
    source_document_id: Optional[str] = Field(
        default=None, description="Source document reference"
    )
    lines: List[JournalEntryLine] = Field(
        default_factory=list, description="Entry lines"
    )
    total_debits: Decimal = Field(
        default=Decimal("0.00"), ge=Decimal("0.00")
    )
    total_credits: Decimal = Field(
        default=Decimal("0.00"), ge=Decimal("0.00")
    )
    is_balanced: bool = Field(
        default=False, description="Whether DR = CR within tolerance"
    )
    status: str = Field(
        default="draft",
        description="draft, posted, reversed, failed",
    )
    created_by: Optional[str] = Field(
        default=None,
        description="Agent or system that created this entry",
    )
    simulation_id: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    posted_at: Optional[datetime] = Field(default=None)


class PostingResult(BaseModel):
    """Result of a journal entry posting attempt.

    Encapsulates the outcome of :meth:`GLPostingEngine.post_journal_entry`,
    including success/failure status, the trial balance after posting, and
    any error message on failure.

    Attributes:
        entry_id: The journal entry's unique identifier.
        entry_number: The sequential entry number (JE-YYYY-NNNN).
        success: Whether the posting succeeded.
        posted_at: UTC timestamp of posting (only set on success).
        trial_balance_after: Cumulative trial balance after this posting.
        error_message: Human-readable error description (only on failure).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry_id: UUID = Field(default_factory=uuid4)
    entry_number: str = Field(default="")
    success: bool = Field(default=False)
    posted_at: Optional[datetime] = Field(default=None)
    trial_balance_after: Decimal = Field(default=Decimal("0.00"))
    error_message: Optional[str] = Field(default=None)


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════


class CircuitBreakerState(str, Enum):
    """State machine for the GL posting circuit breaker.

    Transitions:
        CLOSED  → OPEN       (after ``failure_threshold`` consecutive failures)
        OPEN    → HALF_OPEN  (after ``recovery_timeout_seconds`` elapsed)
        HALF_OPEN → CLOSED   (on successful test call)
        HALF_OPEN → OPEN     (on failed test call)
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple circuit breaker implementation for GL posting.

    Protects the GL posting pipeline from cascading failures by tracking
    consecutive failure counts and temporarily rejecting requests when
    the failure threshold is exceeded.

    Configuration (AAP §0.1.2 — gl_posting):
        - failure_threshold: 10 consecutive failures → OPEN state
        - recovery_timeout_seconds: 60s before testing recovery (HALF_OPEN)

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout_seconds: Seconds to wait before attempting recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout_seconds: float = 60.0,
    ) -> None:
        self._state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self._failure_count: int = 0
        self._failure_threshold: int = failure_threshold
        self._recovery_timeout: float = recovery_timeout_seconds
        self._last_failure_time: Optional[float] = None
        self._success_count: int = 0
        self._total_calls: int = 0

    def can_execute(self) -> bool:
        """Check if a call is allowed through the circuit breaker.

        Returns:
            ``True`` if the call is permitted, ``False`` if the circuit
            is open and the recovery timeout has not yet elapsed.

        State Transitions:
            - CLOSED: Always allows execution.
            - OPEN: Checks if ``recovery_timeout`` has elapsed since the
              last failure.  If so, transitions to HALF_OPEN and allows
              exactly one test call.
            - HALF_OPEN: Allows one test call (the result determines the
              next state transition via :meth:`record_success` or
              :meth:`record_failure`).
        """
        self._total_calls += 1

        if self._state == CircuitBreakerState.CLOSED:
            return True

        if self._state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self._recovery_timeout:
                    # Transition to HALF_OPEN for a test call
                    self._state = CircuitBreakerState.HALF_OPEN
                    logger.info(
                        "circuit_breaker_half_open",
                        service_name="transactions",
                        component="CircuitBreaker",
                        elapsed_seconds=round(elapsed, 2),
                        recovery_timeout=self._recovery_timeout,
                    )
                    return True
            return False

        # HALF_OPEN — allow exactly one test call
        if self._state == CircuitBreakerState.HALF_OPEN:
            return True

        return False  # pragma: no cover — defensive fallback

    def record_success(self) -> None:
        """Record a successful call and potentially close the circuit.

        If the circuit is in HALF_OPEN state, a success transitions it
        back to CLOSED, resetting all failure counters.
        """
        self._success_count += 1

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Recovery confirmed — close the circuit
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(
                "circuit_breaker_closed",
                service_name="transactions",
                component="CircuitBreaker",
                reason="half_open_success",
                total_successes=self._success_count,
            )
        elif self._state == CircuitBreakerState.CLOSED:
            # Reset consecutive failure counter on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit.

        Increments the consecutive failure counter.  When the counter
        reaches ``failure_threshold``, the circuit transitions to OPEN.
        If already in HALF_OPEN, a single failure re-opens the circuit.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Recovery test failed — re-open the circuit
            self._state = CircuitBreakerState.OPEN
            logger.warning(
                "circuit_breaker_reopened",
                service_name="transactions",
                component="CircuitBreaker",
                reason="half_open_failure",
                failure_count=self._failure_count,
            )
        elif (
            self._state == CircuitBreakerState.CLOSED
            and self._failure_count >= self._failure_threshold
        ):
            # Threshold reached — open the circuit
            self._state = CircuitBreakerState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                service_name="transactions",
                component="CircuitBreaker",
                failure_count=self._failure_count,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )

    @property
    def state(self) -> CircuitBreakerState:
        """Current state of the circuit breaker."""
        return self._state


# ═══════════════════════════════════════════════════════════════════════════
# GLPostingEngine
# ═══════════════════════════════════════════════════════════════════════════


class GLPostingEngine:
    """General Ledger posting engine — the foundational posting layer.

    Every P2P and O2C transaction generator delegates journal entry creation,
    balance validation, and trial balance checking to this class.

    Constructor Injection (ADR-003):
        All dependencies are received as keyword-only ``Optional`` parameters
        with ``None`` defaults.  ``None`` values silently disable the
        corresponding capability, enabling test isolation and incremental
        integration.

    Financial Integrity (AAP §0.7.2):
        - ALL calculations use ``Decimal``.  ``float`` is NEVER used.
        - Every journal entry satisfies SUM(debits) = SUM(credits)
          within ``Decimal("0.01")``.
        - Continuous trial balance = zero within ``Decimal("0.01")``.
        - Full rollback on ANY posting failure (atomicity).

    Retry Policy (AAP §0.1.2 — gl_posting):
        max_attempts=3, exponential backoff (1s, 2s, 4s), timeout=30s,
        fallback=rollback transaction.

    Circuit Breaker (AAP §0.1.2 — gl_posting):
        failure_threshold=10, recovery_timeout=60s,
        fallback=rollback transaction.

    Performance Target (AAP §0.7.3):
        ≥200 journal entries posted per minute.
    """

    # ------------------------------------------------------------------
    # Construction (ADR-003 — constructor injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        account_balance_manager: Optional[AccountBalanceManager] = None,
        fiscal_calendar: Optional[FiscalCalendar] = None,
        event_bus: Optional[EventBus] = None,
        db_session_factory: Optional[Any] = None,
        chart_of_accounts: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the GLPostingEngine.

        Args:
            account_balance_manager: Delegates balance updates to this
                manager after successful posting.  When ``None``, balance
                updates are skipped (logged as warning).
            fiscal_calendar: Used for period validation — journal entries
                can only be posted to OPEN periods.  When ``None``, period
                validation is skipped.
            event_bus: Project 2 EventBus for publishing ``DocumentGenerated``
                events on successful posting.  When ``None``, events are
                silently skipped.
            db_session_factory: Callable returning an async context manager
                wrapping a database session (``get_session``).  When ``None``,
                all persistence is in-memory only.
            chart_of_accounts: Dictionary mapping account codes to account
                metadata (at minimum ``{"is_posting": bool}``).  When
                ``None``, account validation is skipped.
        """
        # Injected dependencies (all Optional per ADR-003)
        self._account_balance_manager: Optional[AccountBalanceManager] = (
            account_balance_manager
        )
        self._fiscal_calendar: Optional[FiscalCalendar] = fiscal_calendar
        self._event_bus: Optional[EventBus] = event_bus
        self._db_session_factory: Optional[Any] = db_session_factory
        self._chart_of_accounts: Optional[Dict[str, Dict[str, Any]]] = (
            chart_of_accounts
        )

        # Circuit breaker — initialised from CIRCUIT_BREAKER_CONFIG
        _gl_cb_config = CIRCUIT_BREAKER_CONFIG.get("gl_posting", {})
        self._circuit_breaker: CircuitBreaker = CircuitBreaker(
            failure_threshold=int(
                _gl_cb_config.get("failure_threshold", 10)
            ),
            recovery_timeout_seconds=float(
                _gl_cb_config.get("recovery_timeout_seconds", 60)
            ),
        )

        # Sequential numbering (JE-YYYY-NNNN)
        self._sequence_counter: int = 0
        self._current_year: int = 0

        # Posted entries history (in-memory ledger)
        self._posted_entries: List[JournalEntry] = []

        # Cumulative debit/credit totals for fast trial balance calculation
        self._cumulative_debits: Decimal = Decimal("0.00")
        self._cumulative_credits: Decimal = Decimal("0.00")

        # Metrics counters
        self._entries_posted: int = 0
        self._entries_rejected: int = 0
        self._balance_checks_passed: int = 0
        self._balance_checks_failed: int = 0
        self._trial_balance_checks_passed: int = 0
        self._start_time: float = time.perf_counter()

        logger.info(
            "gl_posting_engine_initialized",
            service_name="transactions",
            component="GLPostingEngine",
            has_balance_manager=account_balance_manager is not None,
            has_fiscal_calendar=fiscal_calendar is not None,
            has_event_bus=event_bus is not None,
            has_db_session=db_session_factory is not None,
            has_chart_of_accounts=chart_of_accounts is not None,
            circuit_breaker_threshold=int(
                _gl_cb_config.get("failure_threshold", 10)
            ),
            circuit_breaker_recovery=float(
                _gl_cb_config.get("recovery_timeout_seconds", 60)
            ),
        )

    # ------------------------------------------------------------------
    # Core Posting Methods
    # ------------------------------------------------------------------

    async def post_journal_entry(
        self,
        entry: JournalEntry,
        simulation_id: Optional[UUID] = None,
    ) -> PostingResult:
        """Post a single journal entry to the General Ledger.

        Implements the 11-step posting pipeline in strict order:

        1. Circuit breaker check — reject if OPEN
        2. Balance validation — SUM(DR) = SUM(CR) within $0.01
        3. Account validation — each account in COA, is_posting=True
        4. Period validation — posting_date falls in an OPEN period
        5. Sequential numbering — assign JE-YYYY-NNNN
        6. Post entry lines (persist journal entry header + lines)
        7. Balance update — delegate to AccountBalanceManager
        8. Trial balance check — cumulative must be zero within $0.01
        9. Set status → "posted", posted_at → now(UTC)
        10. Publish DocumentGenerated event via EventBus
        11. Record success on circuit breaker, increment metrics

        On ANY failure: FULL ROLLBACK (atomicity), record failure, raise.

        Args:
            entry: The journal entry to post.
            simulation_id: Parent simulation run ID for event correlation.

        Returns:
            :class:`PostingResult` with success status and trial balance.

        Raises:
            GLPostingError: If the circuit breaker is OPEN or a non-balance
                posting failure occurs.
            BalanceError: If the entry is unbalanced or trial balance fails.
            PeriodClosedError: If the target period is CLOSING or CLOSED.
            ConcurrencyError: If balance update locking fails.
            TransactionError: Catch-all for unexpected failures.
        """
        start_ts = time.perf_counter()
        entry_sim_id = simulation_id or entry.simulation_id

        # --- Step 1: Circuit Breaker Check ---
        if not self._circuit_breaker.can_execute():
            self._entries_rejected += 1
            error_msg = (
                "GL posting circuit breaker is OPEN — rejecting entry. "
                f"Recovery in {self._circuit_breaker._recovery_timeout}s."
            )
            logger.error(
                "gl_posting_circuit_breaker_open",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                circuit_breaker_state=self._circuit_breaker.state.value,
                simulation_id=str(entry_sim_id) if entry_sim_id else None,
            )
            raise GLPostingError(
                error_msg,
                details={
                    "entry_id": str(entry.entry_id),
                    "circuit_breaker_state": self._circuit_breaker.state.value,
                },
            )

        # --- Retry wrapper using tenacity (AAP §0.7.4) ---
        # GL posting: 3 attempts, exponential backoff (1s, 2s, 4s), 30s timeout
        gl_retry_config = RETRY_POLICIES.get("gl_posting", {})
        max_attempts: int = int(gl_retry_config.get("max_attempts", 3))
        timeout_seconds: float = float(
            gl_retry_config.get("timeout_seconds", 30)
        )

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=4),
                retry=retry_if_exception_type(
                    (GLPostingError, ConcurrencyError, asyncio.TimeoutError),
                ),
                reraise=True,
            ):
                with attempt:
                    attempt_number = attempt.retry_state.attempt_number
                    try:
                        result = await asyncio.wait_for(
                            self._execute_posting(
                                entry=entry,
                                simulation_id=entry_sim_id,
                                start_ts=start_ts,
                            ),
                            timeout=timeout_seconds,
                        )
                        return result

                    except (BalanceError, PeriodClosedError):
                        # Non-retryable errors — rollback and propagate immediately
                        self._rollback_entry(entry)
                        self._circuit_breaker.record_failure()
                        self._entries_rejected += 1
                        raise

                    except asyncio.TimeoutError:
                        logger.warning(
                            "gl_posting_timeout",
                            service_name="transactions",
                            component="GLPostingEngine",
                            entry_id=str(entry.entry_id),
                            attempt=attempt_number,
                            max_attempts=max_attempts,
                            timeout_seconds=timeout_seconds,
                            simulation_id=str(entry_sim_id) if entry_sim_id else None,
                        )
                        raise

                    except (GLPostingError, ConcurrencyError, TransactionError) as exc:
                        logger.warning(
                            "gl_posting_attempt_failed",
                            service_name="transactions",
                            component="GLPostingEngine",
                            entry_id=str(entry.entry_id),
                            attempt=attempt_number,
                            max_attempts=max_attempts,
                            error=str(exc),
                            simulation_id=str(entry_sim_id) if entry_sim_id else None,
                        )
                        raise

                    except Exception as exc:
                        logger.error(
                            "gl_posting_unexpected_error",
                            service_name="transactions",
                            component="GLPostingEngine",
                            entry_id=str(entry.entry_id),
                            attempt=attempt_number,
                            max_attempts=max_attempts,
                            error=str(exc),
                            error_type=type(exc).__name__,
                            simulation_id=str(entry_sim_id) if entry_sim_id else None,
                        )
                        raise

        except RetryError as retry_exc:
            # All attempts exhausted — record failure and rollback
            self._circuit_breaker.record_failure()
            self._entries_rejected += 1
            self._rollback_entry(entry)

            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            last_exception = retry_exc.last_attempt.exception()
            logger.error(
                "gl_posting_all_attempts_exhausted",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                max_attempts=max_attempts,
                duration_ms=round(elapsed_ms, 2),
                simulation_id=str(entry_sim_id) if entry_sim_id else None,
            )

            if isinstance(last_exception, (GLPostingError, TransactionError)):
                raise last_exception
            raise GLPostingError(
                f"GL posting failed after {max_attempts} attempts: "
                f"{last_exception}",
                details={
                    "entry_id": str(entry.entry_id),
                    "attempts": max_attempts,
                    "last_error": str(last_exception),
                },
            ) from last_exception

        except (BalanceError, PeriodClosedError):
            # These propagate through tenacity (non-retryable)
            raise

    async def _execute_posting(
        self,
        *,
        entry: JournalEntry,
        simulation_id: Optional[UUID],
        start_ts: float,
    ) -> PostingResult:
        """Execute the core posting pipeline (steps 2-11).

        This method is called within the retry loop and timeout wrapper.
        On any failure, the entry is rolled back and the appropriate
        exception is raised.

        Args:
            entry: The journal entry to post.
            simulation_id: Simulation context for event correlation.
            start_ts: Start timestamp for duration measurement.

        Returns:
            :class:`PostingResult` on success.
        """
        # --- Step 2: Balance Validation (CRITICAL) ---
        self._validate_balance(entry)

        # --- Step 3: Account Validation ---
        self._validate_accounts(entry)

        # --- Step 4: Period Validation ---
        self._validate_period(entry)

        # --- Step 5: Sequential Numbering ---
        entry.entry_number = self._generate_entry_number(entry.posting_date)

        # --- Step 6: Persist journal entry header and lines ---
        # In-memory persistence (database persistence via db_session_factory
        # is deferred to production integration)
        entry.simulation_id = simulation_id

        # --- Step 7: Balance Update Delegation ---
        try:
            await self._update_balances(entry)
        except (ConcurrencyError, TransactionError):
            # Rollback on balance update failure
            self._rollback_entry(entry)
            raise

        # --- Step 8: Trial Balance Check ---
        try:
            trial_balance = await self._check_trial_balance()
        except BalanceError:
            # CRITICAL: Rollback the entry AND the balance updates
            self._rollback_entry(entry)
            await self._reverse_balance_updates(entry)
            raise

        # --- Step 9: Set status to posted ---
        now_utc = datetime.now(timezone.utc)
        entry.status = "posted"
        entry.posted_at = now_utc

        # Add to posted entries list
        self._posted_entries.append(entry)

        # Update cumulative totals
        self._cumulative_debits += entry.total_debits
        self._cumulative_credits += entry.total_credits

        # --- Step 10: Publish DocumentGenerated event ---
        await self._publish_document_event(entry, simulation_id=simulation_id)

        # --- Step 11: Record success on circuit breaker + metrics ---
        self._circuit_breaker.record_success()
        self._entries_posted += 1

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        logger.info(
            "gl_entry_posted",
            service_name="transactions",
            component="GLPostingEngine",
            entry_id=str(entry.entry_id),
            entry_number=entry.entry_number,
            posting_date=str(entry.posting_date),
            total_debits=str(entry.total_debits),
            total_credits=str(entry.total_credits),
            source_document_type=entry.source_document_type,
            source_document_id=entry.source_document_id,
            lines_count=len(entry.lines),
            trial_balance_after=str(trial_balance),
            duration_ms=round(elapsed_ms, 2),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        return PostingResult(
            entry_id=entry.entry_id,
            entry_number=entry.entry_number,
            success=True,
            posted_at=now_utc,
            trial_balance_after=trial_balance,
            error_message=None,
        )

    async def post_entries_batch(
        self,
        entries: List[JournalEntry],
        simulation_id: Optional[UUID] = None,
    ) -> List[PostingResult]:
        """Post multiple journal entries as a batch.

        Posts each entry individually via :meth:`post_journal_entry`.
        If any single entry fails, the batch continues with remaining
        entries (does NOT abort the entire batch).  After all entries
        are processed, a final trial balance check is performed.

        Args:
            entries: List of journal entries to post.
            simulation_id: Parent simulation run ID.

        Returns:
            List of :class:`PostingResult` objects (one per input entry,
            in the same order).
        """
        if not entries:
            return []

        results: List[PostingResult] = []
        batch_start_ts = time.perf_counter()

        for entry in entries:
            try:
                result = await self.post_journal_entry(
                    entry, simulation_id=simulation_id
                )
                results.append(result)
            except (
                GLPostingError,
                BalanceError,
                PeriodClosedError,
                ConcurrencyError,
                TransactionError,
            ) as exc:
                # Record failure but continue with remaining entries
                results.append(
                    PostingResult(
                        entry_id=entry.entry_id,
                        entry_number=entry.entry_number,
                        success=False,
                        posted_at=None,
                        trial_balance_after=Decimal("0.00"),
                        error_message=str(exc),
                    )
                )
            except Exception as exc:
                # Catch-all for unexpected errors
                results.append(
                    PostingResult(
                        entry_id=entry.entry_id,
                        entry_number=entry.entry_number,
                        success=False,
                        posted_at=None,
                        trial_balance_after=Decimal("0.00"),
                        error_message=f"Unexpected error: {exc}",
                    )
                )

        # Final trial balance check after batch completes
        try:
            batch_trial_balance = await self._check_trial_balance()
        except BalanceError as exc:
            logger.error(
                "batch_trial_balance_failed",
                service_name="transactions",
                component="GLPostingEngine",
                batch_size=len(entries),
                success_count=sum(1 for r in results if r.success),
                error=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            batch_trial_balance = self._calculate_raw_trial_balance()

        batch_elapsed_ms = (time.perf_counter() - batch_start_ts) * 1000.0
        success_count = sum(1 for r in results if r.success)

        logger.info(
            "gl_batch_posting_completed",
            service_name="transactions",
            component="GLPostingEngine",
            batch_size=len(entries),
            success_count=success_count,
            failure_count=len(entries) - success_count,
            trial_balance_after=str(batch_trial_balance),
            duration_ms=round(batch_elapsed_ms, 2),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        return results

    # ------------------------------------------------------------------
    # Validation Methods (CRITICAL for financial integrity)
    # ------------------------------------------------------------------

    def _validate_balance(self, entry: JournalEntry) -> None:
        """Validate that the journal entry is balanced.

        Calculates ``total_debits`` and ``total_credits`` from the entry
        lines and verifies that ``abs(total_debits - total_credits)`` is
        within ``GL_BALANCE_TOLERANCE`` (``Decimal("0.01")``).

        Updates ``entry.total_debits``, ``entry.total_credits``, and
        ``entry.is_balanced`` as side effects.

        Args:
            entry: The journal entry to validate.

        Raises:
            BalanceError: If the entry is unbalanced beyond tolerance.
        """
        # CRITICAL: ALL math uses Decimal, NEVER float
        total_debits = Decimal("0.00")
        total_credits = Decimal("0.00")

        for line in entry.lines:
            total_debits += line.debit_amount
            total_credits += line.credit_amount

        difference = abs(total_debits - total_credits)

        # Update entry totals
        entry.total_debits = total_debits
        entry.total_credits = total_credits

        if difference <= GL_BALANCE_TOLERANCE:
            entry.is_balanced = True
            self._balance_checks_passed += 1
            logger.debug(
                "gl_balance_validated",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                difference=str(difference),
                tolerance=str(GL_BALANCE_TOLERANCE),
            )
        else:
            entry.is_balanced = False
            self._balance_checks_failed += 1
            logger.error(
                "gl_balance_violation",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                total_debits=str(total_debits),
                total_credits=str(total_credits),
                difference=str(difference),
                tolerance=str(GL_BALANCE_TOLERANCE),
            )
            raise BalanceError(
                f"Journal entry is unbalanced: "
                f"debits={total_debits}, credits={total_credits}, "
                f"difference={difference} exceeds tolerance "
                f"{GL_BALANCE_TOLERANCE}",
                details={
                    "journal_entry_id": str(entry.entry_id),
                    "total_debits": str(total_debits),
                    "total_credits": str(total_credits),
                    "imbalance": str(difference),
                    "tolerance": str(GL_BALANCE_TOLERANCE),
                },
            )

    def _validate_accounts(self, entry: JournalEntry) -> None:
        """Validate that all accounts in the entry exist in the COA.

        Verifies each line's ``account_code`` exists in the injected
        ``chart_of_accounts`` dictionary and that the account has
        ``is_posting == True`` (cannot post to summary/control accounts).

        When ``chart_of_accounts`` is ``None`` (not injected), validation
        is silently skipped.

        Args:
            entry: The journal entry to validate.

        Raises:
            GLPostingError: If any account is invalid or not a posting account.
        """
        if self._chart_of_accounts is None:
            # No COA injected — skip validation
            logger.debug(
                "gl_account_validation_skipped",
                service_name="transactions",
                component="GLPostingEngine",
                reason="chart_of_accounts_not_injected",
                entry_id=str(entry.entry_id),
            )
            return

        for line in entry.lines:
            account_code = line.account_code
            account_data = self._chart_of_accounts.get(account_code)

            if account_data is None:
                logger.error(
                    "gl_account_not_found",
                    service_name="transactions",
                    component="GLPostingEngine",
                    entry_id=str(entry.entry_id),
                    account_code=account_code,
                    line_number=line.line_number,
                )
                raise GLPostingError(
                    f"Account '{account_code}' not found in Chart of Accounts",
                    details={
                        "journal_entry_id": str(entry.entry_id),
                        "account_code": account_code,
                        "line_number": line.line_number,
                        "error_category": "account_validation",
                    },
                )

            is_posting = account_data.get("is_posting", False)
            if not is_posting:
                logger.error(
                    "gl_account_not_posting",
                    service_name="transactions",
                    component="GLPostingEngine",
                    entry_id=str(entry.entry_id),
                    account_code=account_code,
                    line_number=line.line_number,
                )
                raise GLPostingError(
                    f"Account '{account_code}' is not a posting account "
                    f"(is_posting=False)",
                    details={
                        "journal_entry_id": str(entry.entry_id),
                        "account_code": account_code,
                        "line_number": line.line_number,
                        "is_posting": False,
                        "error_category": "account_validation",
                    },
                )

            # Populate account_name from COA if available
            if line.account_name is None:
                account_name = account_data.get("name") or account_data.get(
                    "account_name"
                )
                if account_name:
                    line.account_name = account_name

    def _validate_period(self, entry: JournalEntry) -> None:
        """Validate that the entry's posting date falls in an OPEN period.

        Queries the injected ``FiscalCalendar`` to find the period
        containing the entry's ``posting_date``.  The period must have
        status ``OPEN``; ``CLOSING`` and ``CLOSED`` periods reject postings.

        When ``fiscal_calendar`` is ``None`` (not injected), validation
        is silently skipped.

        Args:
            entry: The journal entry to validate.

        Raises:
            PeriodClosedError: If the period is CLOSING or CLOSED.
            GLPostingError: If no period covers the posting date.
        """
        if self._fiscal_calendar is None:
            # No fiscal calendar injected — skip validation
            logger.debug(
                "gl_period_validation_skipped",
                service_name="transactions",
                component="GLPostingEngine",
                reason="fiscal_calendar_not_injected",
                entry_id=str(entry.entry_id),
            )
            return

        period = self._fiscal_calendar.get_period_for_date(entry.posting_date)

        if period is None:
            logger.error(
                "gl_period_not_found",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                posting_date=str(entry.posting_date),
            )
            raise GLPostingError(
                f"No fiscal period found for posting date "
                f"{entry.posting_date}",
                details={
                    "journal_entry_id": str(entry.entry_id),
                    "posting_date": str(entry.posting_date),
                    "error_category": "period_validation",
                },
            )

        # Compare period status (case-insensitive for compatibility with
        # FiscalPeriod PeriodStatus enum values "open"/"closing"/"closed"
        # and constants PERIOD_STATUS_OPEN="OPEN"/CLOSING/CLOSED)
        period_status_upper = str(period.status).upper()

        if period_status_upper == PERIOD_STATUS_OPEN:
            # Period is OPEN — posting is allowed
            entry.period_id = (
                f"{period.fiscal_year}-{period.period_number:02d}"
            )
            logger.debug(
                "gl_period_validated",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                posting_date=str(entry.posting_date),
                period_id=entry.period_id,
                period_status=period_status_upper,
            )
            return

        if period_status_upper == PERIOD_STATUS_CLOSING:
            logger.error(
                "gl_posting_to_closing_period",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                posting_date=str(entry.posting_date),
                period_status="CLOSING",
            )
            raise PeriodClosedError(
                "Cannot post to closing period",
                details={
                    "journal_entry_id": str(entry.entry_id),
                    "fiscal_period": (
                        f"{period.fiscal_year}-{period.period_number:02d}"
                    ),
                    "period_status": "CLOSING",
                    "posting_date": str(entry.posting_date),
                },
            )

        if period_status_upper == PERIOD_STATUS_CLOSED:
            logger.error(
                "gl_posting_to_closed_period",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                posting_date=str(entry.posting_date),
                period_status="CLOSED",
            )
            raise PeriodClosedError(
                "Cannot post to closed period",
                details={
                    "journal_entry_id": str(entry.entry_id),
                    "fiscal_period": (
                        f"{period.fiscal_year}-{period.period_number:02d}"
                    ),
                    "period_status": "CLOSED",
                    "posting_date": str(entry.posting_date),
                },
            )

        # Unknown period status — reject defensively
        raise GLPostingError(
            f"Unknown period status '{period.status}' for posting date "
            f"{entry.posting_date}",
            details={
                "journal_entry_id": str(entry.entry_id),
                "period_status": str(period.status),
                "posting_date": str(entry.posting_date),
                "error_category": "period_validation",
            },
        )

    # ------------------------------------------------------------------
    # Trial Balance & Balance Updates
    # ------------------------------------------------------------------

    async def _check_trial_balance(self) -> Decimal:
        """Check the cumulative trial balance across all posted entries.

        The trial balance is the difference between cumulative debits and
        cumulative credits.  It MUST equal zero within
        ``TRIAL_BALANCE_TOLERANCE`` (``Decimal("0.01")``).

        If an ``AccountBalanceManager`` is available, delegates to its
        :meth:`calculate_trial_balance` method.  Otherwise, computes
        from the in-memory ``_posted_entries`` list.

        Returns:
            The absolute trial balance difference (Decimal).

        Raises:
            BalanceError: If the trial balance exceeds tolerance.
        """
        if self._account_balance_manager is not None:
            try:
                report = await self._account_balance_manager.calculate_trial_balance()
                difference = report.difference
                if abs(difference) <= TRIAL_BALANCE_TOLERANCE:
                    self._trial_balance_checks_passed += 1
                    return difference
                else:
                    raise BalanceError(
                        f"Trial balance check failed: difference="
                        f"{difference} exceeds tolerance "
                        f"{TRIAL_BALANCE_TOLERANCE}",
                        details={
                            "total_debits": str(report.total_debits),
                            "total_credits": str(report.total_credits),
                            "difference": str(difference),
                            "tolerance": str(TRIAL_BALANCE_TOLERANCE),
                        },
                    )
            except BalanceError:
                raise
            except Exception as exc:
                logger.warning(
                    "trial_balance_manager_fallback",
                    service_name="transactions",
                    component="GLPostingEngine",
                    error=str(exc),
                )
                # Fall through to in-memory calculation

        # In-memory trial balance from posted entries
        difference = self._calculate_raw_trial_balance()

        if abs(difference) <= TRIAL_BALANCE_TOLERANCE:
            self._trial_balance_checks_passed += 1
            logger.debug(
                "trial_balance_passed",
                service_name="transactions",
                component="GLPostingEngine",
                difference=str(difference),
                tolerance=str(TRIAL_BALANCE_TOLERANCE),
                posted_entries_count=len(self._posted_entries),
            )
            return difference

        logger.error(
            "trial_balance_failed",
            service_name="transactions",
            component="GLPostingEngine",
            difference=str(difference),
            tolerance=str(TRIAL_BALANCE_TOLERANCE),
            posted_entries_count=len(self._posted_entries),
        )
        raise BalanceError(
            f"Trial balance check failed: difference={difference} "
            f"exceeds tolerance {TRIAL_BALANCE_TOLERANCE}",
            details={
                "total_debits": str(self._cumulative_debits),
                "total_credits": str(self._cumulative_credits),
                "difference": str(difference),
                "tolerance": str(TRIAL_BALANCE_TOLERANCE),
            },
        )

    def _calculate_raw_trial_balance(self) -> Decimal:
        """Calculate the raw trial balance from cumulative totals.

        Uses the running cumulative debit/credit counters for O(1) performance,
        rather than re-summing all posted entries.

        Returns:
            ``cumulative_debits - cumulative_credits`` (Decimal).
        """
        return self._cumulative_debits - self._cumulative_credits

    async def _update_balances(self, entry: JournalEntry) -> None:
        """Delegate balance updates to the AccountBalanceManager.

        For each line in the journal entry, creates a balance update
        request and sends it to the ``AccountBalanceManager``.

        When ``AccountBalanceManager`` is ``None``, logs a warning and
        skips the update (still allows posting for test isolation).

        Args:
            entry: The posted journal entry whose line items drive updates.

        Raises:
            ConcurrencyError: Propagated from AccountBalanceManager.
            TransactionError: Propagated from AccountBalanceManager.
        """
        if self._account_balance_manager is None:
            logger.warning(
                "gl_balance_update_skipped",
                service_name="transactions",
                component="GLPostingEngine",
                reason="account_balance_manager_not_injected",
                entry_id=str(entry.entry_id),
                lines_count=len(entry.lines),
            )
            return

        # Import BalanceUpdateRequest from the manager's module
        # (lazy import to avoid circular dependency at module level)
        from app.transactions.gl.account_balance_manager import (
            BalanceUpdateRequest,
        )

        for line in entry.lines:
            request = BalanceUpdateRequest(
                account_code=line.account_code,
                debit_amount=line.debit_amount,
                credit_amount=line.credit_amount,
                posting_date=entry.posting_date,
                period_id=entry.period_id,
                journal_entry_id=entry.entry_id,
                description=line.description or entry.description,
            )
            result = await self._account_balance_manager.update_balance(request)
            if not result.success:
                raise GLPostingError(
                    f"Balance update failed for account "
                    f"{line.account_code}: {result.error_message}",
                    details={
                        "journal_entry_id": str(entry.entry_id),
                        "account_code": line.account_code,
                        "line_number": line.line_number,
                        "error_message": result.error_message,
                    },
                )

    async def _reverse_balance_updates(self, entry: JournalEntry) -> None:
        """Reverse balance updates for a failed posting (rollback support).

        Swaps debit and credit amounts for each line and re-applies the
        update to the ``AccountBalanceManager``, effectively reversing
        the original balance changes.

        Args:
            entry: The journal entry whose balance updates should be reversed.
        """
        if self._account_balance_manager is None:
            return

        from app.transactions.gl.account_balance_manager import (
            BalanceUpdateRequest,
        )

        for line in entry.lines:
            # Swap debit ↔ credit to reverse the original update
            reversal_request = BalanceUpdateRequest(
                account_code=line.account_code,
                debit_amount=line.credit_amount,  # Swapped
                credit_amount=line.debit_amount,  # Swapped
                posting_date=entry.posting_date,
                period_id=entry.period_id,
                journal_entry_id=entry.entry_id,
                description=f"REVERSAL: {line.description or entry.description}",
            )
            try:
                await self._account_balance_manager.update_balance(
                    reversal_request
                )
            except Exception as exc:
                # Log but do not raise — we are already in error recovery
                logger.error(
                    "balance_reversal_failed",
                    service_name="transactions",
                    component="GLPostingEngine",
                    entry_id=str(entry.entry_id),
                    account_code=line.account_code,
                    error=str(exc),
                )

    def _rollback_entry(self, entry: JournalEntry) -> None:
        """Mark an entry as failed and reset its posting state.

        Does NOT reverse balance updates — that must be done separately
        via :meth:`_reverse_balance_updates` when appropriate.

        Args:
            entry: The journal entry to roll back.
        """
        entry.status = "failed"
        entry.posted_at = None
        logger.debug(
            "gl_entry_rolled_back",
            service_name="transactions",
            component="GLPostingEngine",
            entry_id=str(entry.entry_id),
            entry_number=entry.entry_number,
        )

    # ------------------------------------------------------------------
    # Sequential Numbering
    # ------------------------------------------------------------------

    def _generate_entry_number(self, posting_date: date) -> str:
        """Generate a sequential journal entry number.

        Format: ``JE-YYYY-NNNN`` where:
            - ``JE`` is from ``DOCUMENT_NUMBER_PREFIXES["journal_entry"]``
            - ``YYYY`` is the posting year
            - ``NNNN`` is a zero-padded 4-digit sequence number

        The sequence counter resets when the year changes.

        Args:
            posting_date: The posting date (determines the year component).

        Returns:
            Sequential entry number string (e.g., ``"JE-2024-0001"``).
        """
        posting_year = posting_date.year

        # Reset sequence counter when year changes
        if posting_year != self._current_year:
            self._current_year = posting_year
            self._sequence_counter = 0

        self._sequence_counter += 1
        prefix = DOCUMENT_NUMBER_PREFIXES.get("journal_entry", "JE")

        return f"{prefix}-{posting_year}-{self._sequence_counter:04d}"

    # ------------------------------------------------------------------
    # Event Publishing
    # ------------------------------------------------------------------

    async def _publish_document_event(
        self,
        entry: JournalEntry,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Publish a DocumentGenerated event for a posted journal entry.

        Creates a ``DocumentGenerated`` event (from ``app.events.event_types``)
        with the journal entry's metadata and publishes it via the EventBus.

        When ``EventBus`` is ``None`` (not injected), the event is silently
        skipped with a debug log.

        Args:
            entry: The posted journal entry.
            simulation_id: Simulation context for event correlation.
        """
        if self._event_bus is None:
            logger.debug(
                "gl_event_publish_skipped",
                service_name="transactions",
                component="GLPostingEngine",
                reason="event_bus_not_injected",
                entry_id=str(entry.entry_id),
            )
            return

        try:
            from app.events.event_types import DocumentGenerated

            event = DocumentGenerated(
                simulation_id=simulation_id or entry.simulation_id or uuid4(),
                payload={
                    "document_id": str(entry.entry_id),
                    "document_type": "journal_entry",
                    "transaction_id": entry.source_document_id or str(entry.entry_id),
                    "entry_number": entry.entry_number,
                    "posting_date": str(entry.posting_date),
                    "total_debits": str(entry.total_debits),
                    "total_credits": str(entry.total_credits),
                    "lines_count": len(entry.lines),
                    "source_document_type": entry.source_document_type,
                },
            )
            await self._event_bus.publish(event)

            logger.debug(
                "gl_document_event_published",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                entry_number=entry.entry_number,
                event_id=str(event.event_id),
            )
        except Exception as exc:
            # Event publishing failure should NOT fail the posting
            logger.warning(
                "gl_event_publish_failed",
                service_name="transactions",
                component="GLPostingEngine",
                entry_id=str(entry.entry_id),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Metrics and Inspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics for the GL posting engine.

        Returns:
            Dictionary containing counters, rates, and circuit breaker state.
        """
        elapsed_seconds = max(
            time.perf_counter() - self._start_time, 0.001
        )
        elapsed_minutes = elapsed_seconds / 60.0
        posting_rate = (
            self._entries_posted / elapsed_minutes
            if elapsed_minutes > 0
            else 0.0
        )

        return {
            "entries_posted": self._entries_posted,
            "entries_rejected": self._entries_rejected,
            "balance_checks_passed": self._balance_checks_passed,
            "balance_checks_failed": self._balance_checks_failed,
            "trial_balance_checks_passed": self._trial_balance_checks_passed,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_failure_count": self._circuit_breaker._failure_count,
            "posting_rate_per_minute": round(posting_rate, 2),
            "cumulative_debits": str(self._cumulative_debits),
            "cumulative_credits": str(self._cumulative_credits),
            "posted_entries_count": len(self._posted_entries),
        }

    def get_posted_entries(self) -> List[JournalEntry]:
        """Return the list of all successfully posted journal entries.

        Returns:
            Shallow copy of the posted entries list.
        """
        return list(self._posted_entries)

    def get_trial_balance(self) -> Decimal:
        """Return the current cumulative trial balance.

        This is a synchronous O(1) operation using the running totals.

        Returns:
            ``cumulative_debits - cumulative_credits`` (Decimal).
        """
        return self._calculate_raw_trial_balance()

    # ------------------------------------------------------------------
    # Batch Convenience — SimulationEngine Integration
    # ------------------------------------------------------------------

    async def post_pending_entries(
        self,
        *,
        day_context: Any = None,
    ) -> Dict[str, Any]:
        """Post pending GL journal entries from the day context.

        This is a convenience wrapper invoked by :class:`SimulationEngine`
        during its daily pipeline (Step 2c).  It extracts any unposted GL
        journal entries accumulated in the ``DayContext.unposted_gl_entries``
        list, posts them via :meth:`post_entries_batch`, and returns a
        summary dictionary suitable for metrics collection.

        When *day_context* is ``None`` or has no pending entries, the method
        returns immediately with ``entries_posted = 0``.

        Args:
            day_context: A :class:`~app.simulation.day_context.DayContext`
                instance (or any object with an ``unposted_gl_entries``
                attribute and an optional ``simulation_id`` attribute).

        Returns:
            Dictionary with keys ``entries_posted`` (int),
            ``entries_failed`` (int), and ``trial_balance_after`` (str).
        """
        entries_to_post: List[JournalEntry] = []
        simulation_id: Optional[UUID] = None

        if day_context is not None:
            raw_entries = getattr(day_context, "unposted_gl_entries", []) or []
            for raw in raw_entries:
                if isinstance(raw, JournalEntry):
                    entries_to_post.append(raw)
                elif isinstance(raw, dict):
                    try:
                        entries_to_post.append(JournalEntry(**raw))
                    except Exception:
                        logger.warning(
                            "gl_pending_entry_parse_failed",
                            service_name="transactions",
                            component="GLPostingEngine",
                            entry_data=str(raw)[:200],
                        )
            raw_sim_id = getattr(day_context, "simulation_id", None)
            if isinstance(raw_sim_id, UUID):
                simulation_id = raw_sim_id
            elif isinstance(raw_sim_id, str):
                try:
                    simulation_id = UUID(raw_sim_id)
                except ValueError:
                    simulation_id = None

        if not entries_to_post:
            logger.debug(
                "gl_post_pending_noop",
                service_name="transactions",
                component="GLPostingEngine",
                reason="no_pending_entries",
            )
            return {
                "entries_posted": 0,
                "entries_failed": 0,
                "trial_balance_after": str(self._calculate_raw_trial_balance()),
            }

        results = await self.post_entries_batch(
            entries_to_post, simulation_id=simulation_id
        )

        posted_count = sum(1 for r in results if r.success)
        failed_count = len(results) - posted_count

        # Clear posted entries from day context to prevent re-processing
        if day_context is not None and hasattr(day_context, "unposted_gl_entries"):
            day_context.unposted_gl_entries = []

        return {
            "entries_posted": posted_count,
            "entries_failed": failed_count,
            "trial_balance_after": str(self._calculate_raw_trial_balance()),
        }
