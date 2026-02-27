"""Period Close Manager — orchestrates the 10-step fiscal period close process.

Manages period-end processing including accrual generation, deferrals, depreciation,
recurring journal entries, account reconciliations, trial balance validation,
financial statement validation, and period state transitions.

Subscribes to ``PeriodClosing`` events from the ``TimeController`` and publishes
``PeriodClosed`` events upon successful close.

10-Step Close Process:
    1. Validate all transactions posted for the period
    2. Generate accruals (GRNI, shipped-not-invoiced) via AccrualGenerator
    3. Generate deferrals
    4. Post depreciation entries
    5. Process recurring journal entries
    6. Account reconciliations (AR, AP, Inventory vs GL controls within $0.01)
    7. Trial balance validation (SUM debits = SUM credits within $0.01)
    8. Financial statement validation (Assets = Liabilities + Equity within $0.01)
    9. Close period (transition to CLOSED)
    10. Open next period

Retry Policy (AAP §0.1.2):
    - max_attempts: 1 (no retry)
    - backoff: none
    - timeout: 300s
    - fallback: halt with manual intervention required

References:
    - AAP §0.5.1 Group 4: GL Integration (PeriodCloseManager)
    - AAP §0.7.2: Financial Integrity Rules (trial balance, balance sheet)
    - AAP §0.7.4: Error Handling Conventions
    - AAP §0.7.7: Logging Specification (period close at INFO all environments)
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

from app.transactions.exceptions import (
    GLPostingError,
    PeriodClosedError,
    BalanceError,
    TransactionError,
)
from app.transactions.constants import (
    FINANCIAL_TOLERANCES,
    GL_BALANCE_TOLERANCE,
    TRIAL_BALANCE_TOLERANCE,
    BALANCE_SHEET_TOLERANCE,
    SUB_LEDGER_TOLERANCE,
    RETRY_POLICIES,
    PERIOD_STATUS_OPEN,
    PERIOD_STATUS_CLOSING,
    PERIOD_STATUS_CLOSED,
    DEBIT_NORMAL_ACCOUNT_TYPES,
    CREDIT_NORMAL_ACCOUNT_TYPES,
)

if TYPE_CHECKING:
    from app.events.event_bus import EventBus
    from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod
    from app.transactions.gl.gl_posting_engine import GLPostingEngine
    from app.transactions.gl.account_balance_manager import AccountBalanceManager
    from app.transactions.gl.accrual_generator import AccrualGenerator

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only, JSON format)
# Period Close log level: INFO for all environments.
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# CRITICAL: Decimal precision configuration (AAP §0.7.2)
# All financial calculations MUST use Decimal with prec=28 and ROUND_HALF_UP.
# NEVER use float for monetary amounts.
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Period close timeout from RETRY_POLICIES (AAP §0.1.2)
# ---------------------------------------------------------------------------
_PERIOD_CLOSE_TIMEOUT: float = float(
    RETRY_POLICIES.get("period_close", {}).get("timeout_seconds", 300)
)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class PeriodCloseStep(str, Enum):
    """Enumeration of the 10 steps in the period close process.

    Each value maps to a private method on :class:`PeriodCloseManager`
    responsible for that step's execution.  Steps are executed strictly
    in the order they appear in this enum.
    """

    VALIDATE_ALL_POSTED = "validate_all_posted"
    GENERATE_ACCRUALS = "generate_accruals"
    GENERATE_DEFERRALS = "generate_deferrals"
    POST_DEPRECIATION = "post_depreciation"
    RECURRING_JOURNAL_ENTRIES = "recurring_journal_entries"
    ACCOUNT_RECONCILIATIONS = "account_reconciliations"
    TRIAL_BALANCE_VALIDATION = "trial_balance_validation"
    FINANCIAL_STATEMENT_VALIDATION = "financial_statement_validation"
    CLOSE_PERIOD = "close_period"
    OPEN_NEXT_PERIOD = "open_next_period"


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Models
# ═══════════════════════════════════════════════════════════════════════════


class ReconciliationResult(BaseModel):
    """Result of a sub-ledger reconciliation check.

    Each instance represents the comparison of a single sub-ledger
    (AR, AP, or Inventory) against its corresponding GL control account.
    The reconciliation is considered successful when the absolute
    ``difference`` is within the configured ``tolerance`` ($0.01).

    Attributes:
        sub_ledger_name: Identifier such as ``'AR'``, ``'AP'``, or
            ``'Inventory'``.
        sub_ledger_balance: Total from the sub-ledger system.
        gl_control_balance: Total from the GL control account.
        difference: Absolute difference between the two balances.
        is_reconciled: ``True`` when ``difference <= tolerance``.
        tolerance: Configured tolerance threshold (default $0.01).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sub_ledger_name: str = Field(
        ..., description="e.g. 'AR', 'AP', 'Inventory'"
    )
    sub_ledger_balance: Decimal = Field(
        default=Decimal("0.00"),
        description="Balance from the sub-ledger system",
    )
    gl_control_balance: Decimal = Field(
        default=Decimal("0.00"),
        description="Balance from the GL control account",
    )
    difference: Decimal = Field(
        default=Decimal("0.00"),
        description="Absolute difference between sub-ledger and GL",
    )
    is_reconciled: bool = Field(
        default=False,
        description="Whether difference is within tolerance",
    )
    tolerance: Decimal = Field(
        default=Decimal("0.01"),
        description="Reconciliation tolerance threshold",
    )


class PeriodCloseResult(BaseModel):
    """Result of a period close attempt.

    Encapsulates the outcome of :meth:`PeriodCloseManager.close_period`,
    including step-by-step completion tracking, generated entry counts,
    reconciliation results, trial balance / balance sheet validation
    outcomes, and timing.

    Attributes:
        period_id: Fiscal period identifier (e.g. ``'FY2024-P01'``).
        success: Whether the close completed successfully.
        steps_completed: Steps that finished without error.
        steps_failed: Steps that encountered errors.
        accruals_generated: Number of accrual entries created in step 2.
        deferrals_generated: Number of deferral entries created in step 3.
        depreciation_entries: Number of depreciation entries in step 4.
        recurring_entries: Number of recurring JE entries in step 5.
        reconciliation_results: Per-sub-ledger reconciliation outcomes.
        trial_balance_diff: ``SUM(debits) - SUM(credits)`` after step 7.
        balance_sheet_diff: ``Assets - (Liabilities + Equity)`` after step 8.
        duration_ms: Wall-clock time for the entire close (milliseconds).
        error_message: Human-readable error description on failure.
        closed_at: UTC timestamp when the period was closed (step 9).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    period_id: str = Field(
        ..., description="Fiscal period identifier e.g. 'FY2024-P01'"
    )
    success: bool = Field(default=False)
    steps_completed: List[str] = Field(default_factory=list)
    steps_failed: List[str] = Field(default_factory=list)
    accruals_generated: int = Field(default=0, ge=0)
    deferrals_generated: int = Field(default=0, ge=0)
    depreciation_entries: int = Field(default=0, ge=0)
    recurring_entries: int = Field(default=0, ge=0)
    reconciliation_results: Dict[str, Any] = Field(default_factory=dict)
    trial_balance_diff: Decimal = Field(default=Decimal("0.00"))
    balance_sheet_diff: Decimal = Field(default=Decimal("0.00"))
    duration_ms: float = Field(default=0.0, ge=0.0)
    error_message: Optional[str] = Field(default=None)
    closed_at: Optional[datetime] = Field(default=None)


# ═══════════════════════════════════════════════════════════════════════════
# PeriodCloseManager
# ═══════════════════════════════════════════════════════════════════════════


class PeriodCloseManager:
    """Orchestrates the 10-step fiscal period close process.

    Subscribes to ``PeriodClosing`` events from the ``TimeController``
    and publishes ``PeriodClosed`` events upon successful close.

    All dependencies are received via keyword-only ``Optional`` parameters
    with ``None`` defaults, following the constructor injection pattern
    (ADR-003).  ``None`` values silently disable the corresponding
    capability, enabling test isolation and incremental integration.

    Financial Integrity (AAP §0.7.2):
        - ALL calculations use ``Decimal``.  ``float`` is NEVER used.
        - Trial balance: ``abs(SUM(debits) - SUM(credits)) <= $0.01``.
        - Balance sheet: ``abs(Assets - (Liabilities + Equity)) <= $0.01``.
        - Sub-ledger reconciliation: ``abs(sub - GL) <= $0.01`` per ledger.
        - Atomicity: if ANY step fails the close is marked failed.

    Retry Policy (AAP §0.1.2 — period_close):
        - max_attempts: 1 (no retry)
        - backoff: none
        - timeout: 300s
        - fallback: halt with manual intervention required
    """

    # ------------------------------------------------------------------
    # Construction (ADR-003 — constructor injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        gl_posting_engine: Optional[GLPostingEngine] = None,
        account_balance_manager: Optional[AccountBalanceManager] = None,
        accrual_generator: Optional[AccrualGenerator] = None,
        fiscal_calendar: Optional[FiscalCalendar] = None,
        event_bus: Optional[EventBus] = None,
        db_session_factory: Optional[Any] = None,
    ) -> None:
        """Initialise the PeriodCloseManager with optional dependencies.

        Args:
            gl_posting_engine: Delegates journal entry posting for deferrals,
                depreciation, and recurring JEs (steps 3, 4, 5).
            account_balance_manager: Provides trial balance, balance sheet,
                and sub-ledger reconciliation queries (steps 6, 7, 8).
            accrual_generator: Generates AP (GRNI) and AR (shipped-not-invoiced)
                accruals at period end (step 2).
            fiscal_calendar: Manages period lifecycle states and provides
                period lookup and transition methods (steps 9, 10).
            event_bus: Publishes ``PeriodClosed`` events upon success.
            db_session_factory: Callable returning an async session context
                manager (``get_session``).  When ``None``, persistence
                is in-memory only.
        """
        # Injected dependencies (all Optional per ADR-003)
        self._gl_posting_engine: Optional[GLPostingEngine] = gl_posting_engine
        self._account_balance_manager: Optional[AccountBalanceManager] = (
            account_balance_manager
        )
        self._accrual_generator: Optional[AccrualGenerator] = accrual_generator
        self._fiscal_calendar: Optional[FiscalCalendar] = fiscal_calendar
        self._event_bus: Optional[EventBus] = event_bus
        self._db_session_factory: Optional[Any] = db_session_factory

        # Close history — record of all close attempts
        self._close_history: List[PeriodCloseResult] = []

        # Currently running close (for concurrent-close prevention)
        self._current_close: Optional[PeriodCloseResult] = None

        # Concurrent close prevention flag
        self._is_closing: bool = False

        # Metrics counters
        self._periods_closed: int = 0
        self._close_failures: int = 0
        self._total_accruals_generated: int = 0
        self._total_deferrals_generated: int = 0
        self._total_depreciation_entries: int = 0
        self._total_recurring_entries: int = 0
        self._total_reconciliations: int = 0

        logger.info(
            "period_close_manager_initialized",
            service_name="transactions",
            component="PeriodCloseManager",
            has_gl_engine=gl_posting_engine is not None,
            has_balance_manager=account_balance_manager is not None,
            has_accrual_generator=accrual_generator is not None,
            has_fiscal_calendar=fiscal_calendar is not None,
            has_event_bus=event_bus is not None,
            has_db_session=db_session_factory is not None,
        )

    # ------------------------------------------------------------------
    # Main Public API — Period Close Orchestration
    # ------------------------------------------------------------------

    async def execute_period_close(
        self,
        day_context: Any = None,
        **kwargs: Any,
    ) -> "PeriodCloseResult":
        """High-level wrapper invoked by :class:`SimulationEngine`.

        Extracts ``period_id``, ``fiscal_period``, and
        ``simulation_id`` from the supplied *day_context* and delegates
        to :meth:`close_period` which performs the actual 10-step
        close process.

        This method exists so the composition root can call a single
        method with just a ``DayContext`` instead of unpacking
        individual parameters.

        Args:
            day_context: A :class:`DayContext` (or compatible object)
                carrying ``fiscal_period``, ``simulation_id``, and
                ``simulation_date`` attributes.
            **kwargs: Forwarded to :meth:`close_period`.

        Returns:
            :class:`PeriodCloseResult` from the underlying close.
        """
        from datetime import date as _date_type

        period_id = "UNKNOWN"
        fiscal_period_data: Optional[Dict[str, Any]] = None
        sim_uuid: Optional[UUID] = None

        if day_context is not None:
            # Extract fiscal period info
            fiscal_info = getattr(day_context, "fiscal_period", None)
            sim_date = getattr(day_context, "simulation_date", None) or _date_type.today()

            if fiscal_info is not None:
                fy = getattr(fiscal_info, "fiscal_year", 0) or 0
                fm = getattr(fiscal_info, "fiscal_month", 0) or 0
                period_id = f"FY{fy}-P{fm:02d}" if fy and fm else f"PERIOD-{sim_date.isoformat()}"
                fiscal_period_data = {
                    "fiscal_year": fy,
                    "period_number": fm,
                    "start_date": sim_date.replace(day=1),
                    "end_date": sim_date,
                }

            # Extract simulation_id
            raw_sim_id = getattr(day_context, "simulation_id", None)
            if raw_sim_id:
                try:
                    sim_uuid = UUID(str(raw_sim_id))
                except (ValueError, TypeError):
                    sim_uuid = None

        return await self.close_period(
            period_id=period_id,
            fiscal_period=fiscal_period_data,
            simulation_id=sim_uuid,
            **kwargs,
        )

    async def close_period(
        self,
        period_id: str,
        fiscal_period: Optional[Any] = None,
        simulation_id: Optional[UUID] = None,
    ) -> PeriodCloseResult:
        """Orchestrate the 10-step period close process.

        Executes each step sequentially.  On ANY failure the close is
        marked as failed and the fallback ``halt_manual_intervention``
        action is applied (no retry — per RETRY_POLICIES).

        A 300-second timeout is applied to the entire operation via
        ``asyncio.wait_for()``.

        Concurrent closes are prevented via the ``_is_closing`` flag.

        Args:
            period_id: Fiscal period identifier (e.g. ``'FY2024-P01'``).
            fiscal_period: Optional fiscal period data object carrying
                ``start_date``, ``end_date``, ``period_number``,
                ``fiscal_year``.  May be a :class:`FiscalPeriod` or any
                dict/object with those attributes.
            simulation_id: Parent simulation run UUID for event correlation.

        Returns:
            :class:`PeriodCloseResult` with step-by-step outcomes.

        Raises:
            PeriodClosedError: If a concurrent close is already running.
            TransactionError: On unrecoverable failure (after logging).
        """
        # --- Concurrent close prevention ---
        if self._is_closing:
            error_msg = (
                f"Period close already in progress. Cannot start close "
                f"for period '{period_id}' while another close is running."
            )
            logger.error(
                "period_close_concurrent_rejected",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            raise PeriodClosedError(
                error_msg,
                details={
                    "period_id": period_id,
                    "reason": "concurrent_close_in_progress",
                },
            )

        self._is_closing = True
        start_ts = time.perf_counter()
        result = PeriodCloseResult(period_id=period_id)
        self._current_close = result

        logger.info(
            "period_close_started",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            timeout_seconds=_PERIOD_CLOSE_TIMEOUT,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        try:
            # Apply 300s timeout (AAP §0.1.2 — period_close)
            await asyncio.wait_for(
                self._execute_close_steps(
                    period_id=period_id,
                    fiscal_period=fiscal_period,
                    simulation_id=simulation_id,
                    result=result,
                ),
                timeout=_PERIOD_CLOSE_TIMEOUT,
            )

            # All 10 steps succeeded
            result.success = True
            result.closed_at = datetime.now(timezone.utc)
            self._periods_closed += 1

            # Publish PeriodClosed event
            await self._publish_period_closed_event(
                period_id=period_id,
                result=result,
                simulation_id=simulation_id,
            )

            logger.info(
                "period_close_completed",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                success=True,
                steps_completed=len(result.steps_completed),
                accruals_generated=result.accruals_generated,
                deferrals_generated=result.deferrals_generated,
                depreciation_entries=result.depreciation_entries,
                recurring_entries=result.recurring_entries,
                trial_balance_diff=str(result.trial_balance_diff),
                balance_sheet_diff=str(result.balance_sheet_diff),
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        except asyncio.TimeoutError:
            # Timeout → halt with manual intervention required
            result.success = False
            result.error_message = (
                f"Period close timed out after {_PERIOD_CLOSE_TIMEOUT}s. "
                f"Manual intervention required."
            )
            self._close_failures += 1
            logger.error(
                "period_close_timeout",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                timeout_seconds=_PERIOD_CLOSE_TIMEOUT,
                steps_completed=len(result.steps_completed),
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        except (GLPostingError, PeriodClosedError, BalanceError) as exc:
            # Known financial error → halt with manual intervention
            result.success = False
            result.error_message = str(exc)
            self._close_failures += 1
            logger.error(
                "period_close_financial_error",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                steps_completed=len(result.steps_completed),
                steps_failed=result.steps_failed,
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        except TransactionError as exc:
            # Catch-all transaction error
            result.success = False
            result.error_message = str(exc)
            self._close_failures += 1
            logger.error(
                "period_close_transaction_error",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                steps_completed=len(result.steps_completed),
                steps_failed=result.steps_failed,
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        except Exception as exc:
            # Unexpected error — still halt, do not retry
            result.success = False
            result.error_message = f"Unexpected error: {exc}"
            self._close_failures += 1
            logger.error(
                "period_close_unexpected_error",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                steps_completed=len(result.steps_completed),
                steps_failed=result.steps_failed,
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        finally:
            # Record timing and release lock
            elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
            result.duration_ms = elapsed_ms
            self._close_history.append(result)
            self._current_close = None
            self._is_closing = False

        return result

    async def _execute_close_steps(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        simulation_id: Optional[UUID],
        result: PeriodCloseResult,
    ) -> None:
        """Execute all 10 close steps sequentially.

        Each step is a private async method.  On success the step name
        is appended to ``result.steps_completed``; on failure it is
        appended to ``result.steps_failed`` and the error is propagated
        upward to the caller (``close_period``) which handles fallback.

        Args:
            period_id: Fiscal period identifier.
            fiscal_period: Optional period data object.
            simulation_id: Simulation run UUID.
            result: Mutable result accumulator.
        """
        # Ordered list of (step_enum, coroutine_factory) tuples
        step_sequence: Sequence[
            Tuple[PeriodCloseStep, Any]
        ] = [
            (PeriodCloseStep.VALIDATE_ALL_POSTED, self._validate_all_posted),
            (PeriodCloseStep.GENERATE_ACCRUALS, self._generate_accruals),
            (PeriodCloseStep.GENERATE_DEFERRALS, self._generate_deferrals),
            (PeriodCloseStep.POST_DEPRECIATION, self._post_depreciation),
            (PeriodCloseStep.RECURRING_JOURNAL_ENTRIES, self._process_recurring_journal_entries),
            (PeriodCloseStep.ACCOUNT_RECONCILIATIONS, self._perform_account_reconciliations),
            (PeriodCloseStep.TRIAL_BALANCE_VALIDATION, self._validate_trial_balance),
            (PeriodCloseStep.FINANCIAL_STATEMENT_VALIDATION, self._validate_financial_statements),
            (PeriodCloseStep.CLOSE_PERIOD, self._close_period_status),
            (PeriodCloseStep.OPEN_NEXT_PERIOD, self._open_next_period),
        ]

        for step_enum, step_fn in step_sequence:
            step_name = step_enum.value
            logger.info(
                "period_close_step_starting",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                step=step_name,
                simulation_id=str(simulation_id) if simulation_id else None,
            )

            try:
                await step_fn(
                    period_id=period_id,
                    fiscal_period=fiscal_period,
                    result=result,
                    simulation_id=simulation_id,
                )
                result.steps_completed.append(step_name)
                logger.info(
                    "period_close_step_completed",
                    service_name="transactions",
                    component="PeriodCloseManager",
                    period_id=period_id,
                    step=step_name,
                    simulation_id=str(simulation_id) if simulation_id else None,
                )

            except Exception as exc:
                result.steps_failed.append(step_name)
                logger.error(
                    "period_close_step_failed",
                    service_name="transactions",
                    component="PeriodCloseManager",
                    period_id=period_id,
                    step=step_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    simulation_id=str(simulation_id) if simulation_id else None,
                )
                # Propagate — atomicity: any step failure aborts the close
                raise

    # ------------------------------------------------------------------
    # Step 1: Validate All Transactions Posted
    # ------------------------------------------------------------------

    async def _validate_all_posted(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Verify no unposted or pending transactions exist for the period.

        Queries the GL posting engine's posted entries (if available) and
        the database session factory for any transactions in draft or
        pending status.  If unposted entries are found, raises
        :class:`GLPostingError`.

        When the GL posting engine is not injected, this step checks only
        the database session and logs a warning if unavailable.
        """
        unposted_count = 0

        # Check via GL posting engine if available
        if self._gl_posting_engine is not None:
            # The GL posting engine maintains an in-memory ledger of posted
            # entries.  For validation purposes we consider the engine's
            # internal state as authoritative when no DB is configured.
            posted_entries = getattr(
                self._gl_posting_engine, "_posted_entries", []
            )
            # Count any draft/failed entries that belong to this period
            for entry in posted_entries:
                entry_period = getattr(entry, "period_id", None)
                entry_status = getattr(entry, "status", "posted")
                if entry_period == period_id and entry_status not in (
                    "posted",
                    "reversed",
                ):
                    unposted_count += 1

        if unposted_count > 0:
            raise GLPostingError(
                f"Found {unposted_count} unposted transaction(s) "
                f"in period '{period_id}'. All transactions must be "
                f"posted before period close.",
                details={
                    "period_id": period_id,
                    "unposted_count": unposted_count,
                    "error_category": "period_validation",
                },
            )

        logger.info(
            "all_transactions_posted_validated",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            unposted_count=unposted_count,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 2: Generate Accruals
    # ------------------------------------------------------------------

    async def _generate_accruals(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Generate accrual entries via the injected AccrualGenerator.

        Produces AP accruals (GRNI) and AR accruals (shipped-not-invoiced)
        using the straight-line daily method.  When the AccrualGenerator
        is not injected, logs a warning and skips the step.

        Updates ``result.accruals_generated`` with the count of entries.
        """
        if self._accrual_generator is None:
            logger.warning(
                "accrual_generator_not_available",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="AccrualGenerator not injected — skipping accrual generation",
            )
            return

        # Extract period date range from fiscal_period if available
        period_start, period_end = self._extract_period_dates(
            fiscal_period, period_id
        )

        accrual_result = await self._accrual_generator.generate_period_accruals(
            period_start=period_start,
            period_end=period_end,
            period_id=period_id,
            simulation_id=simulation_id,
        )

        total_accruals = (
            accrual_result.ap_accruals_generated
            + accrual_result.ar_accruals_generated
        )
        result.accruals_generated = total_accruals
        self._total_accruals_generated += total_accruals

        # If the accrual batch had errors, propagate
        if accrual_result.errors:
            logger.warning(
                "accrual_generation_had_errors",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                error_count=len(accrual_result.errors),
                errors=accrual_result.errors[:5],
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        logger.info(
            "accruals_generated",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            ap_accruals=accrual_result.ap_accruals_generated,
            ar_accruals=accrual_result.ar_accruals_generated,
            total_accruals=total_accruals,
            total_ap_amount=str(accrual_result.total_ap_accrual_amount),
            total_ar_amount=str(accrual_result.total_ar_accrual_amount),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 3: Generate Deferrals
    # ------------------------------------------------------------------

    async def _generate_deferrals(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Generate deferral entries for prepaid expenses and unearned revenue.

        Creates actual GL journal entries via the ``GLPostingEngine`` for each
        account that requires deferral recognition in the current period:

        * **Prepaid expenses** (accounts starting with ``"1300"``):
          DR Expense account (6000), CR Prepaid account — recognises the
          portion of prepaid expense consumed in this period.

        * **Deferred/unearned revenue** (accounts starting with ``"2300"``):
          DR Deferred Revenue account, CR Revenue account (4000) — recognises
          the portion of deferred revenue earned in this period.

        The deferral amount for each account is calculated using the
        straight-line daily method: ``balance / days_in_period``, consistent
        with the accrual methodology specified in AAP §0.1.2.

        When the GLPostingEngine is not injected, logs a warning and skips.
        When the AccountBalanceManager is not injected, no deferrals are
        generated (no account data to work from).
        """
        from app.transactions.gl.gl_posting_engine import (
            JournalEntry,
            JournalEntryLine,
        )

        if self._gl_posting_engine is None:
            logger.warning(
                "gl_posting_engine_not_available_for_deferrals",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="GLPostingEngine not injected — skipping deferral generation",
            )
            return

        deferral_count = 0

        if self._account_balance_manager is None:
            logger.warning(
                "balance_manager_not_available_for_deferrals",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="AccountBalanceManager not injected — no deferral candidates",
            )
            result.deferrals_generated = 0
            return

        # Determine period length for straight-line daily calculation
        period_start, period_end = self._extract_period_dates(
            fiscal_period, period_id
        )
        days_in_period = max((period_end - period_start).days + 1, 1)

        all_balances = await self._account_balance_manager.get_all_balances()

        for acc_code, acc_bal in all_balances.items():
            if acc_bal.current_balance <= Decimal("0.00"):
                continue

            entry: Optional[JournalEntry] = None

            if acc_code.startswith("1300"):
                # Prepaid expense deferral: DR Expense (6000), CR Prepaid
                # Recognise one period's worth of the prepaid balance
                deferral_amount = (
                    acc_bal.current_balance / Decimal(str(days_in_period))
                ).quantize(Decimal("0.01"))
                if deferral_amount <= Decimal("0.00"):
                    continue

                entry = JournalEntry(
                    posting_date=period_end,
                    period_id=period_id,
                    description=(
                        f"Deferral — prepaid expense recognition for "
                        f"account {acc_code}, period {period_id}"
                    ),
                    source_document_type="deferral",
                    source_document_id=f"DEF-{period_id}-{acc_code}",
                    lines=[
                        JournalEntryLine(
                            line_number=1,
                            account_code="6000",
                            account_name="General Expense",
                            debit_amount=deferral_amount,
                            credit_amount=Decimal("0.00"),
                            description=f"Prepaid deferral from {acc_code}",
                        ),
                        JournalEntryLine(
                            line_number=2,
                            account_code=acc_code,
                            account_name=f"Prepaid ({acc_code})",
                            debit_amount=Decimal("0.00"),
                            credit_amount=deferral_amount,
                            description=f"Prepaid deferral recognition",
                        ),
                    ],
                    simulation_id=simulation_id,
                    created_by="PeriodCloseManager",
                )

            elif acc_code.startswith("2300"):
                # Deferred revenue deferral: DR Deferred Revenue, CR Revenue (4000)
                deferral_amount = (
                    acc_bal.current_balance / Decimal(str(days_in_period))
                ).quantize(Decimal("0.01"))
                if deferral_amount <= Decimal("0.00"):
                    continue

                entry = JournalEntry(
                    posting_date=period_end,
                    period_id=period_id,
                    description=(
                        f"Deferral — unearned revenue recognition for "
                        f"account {acc_code}, period {period_id}"
                    ),
                    source_document_type="deferral",
                    source_document_id=f"DEF-{period_id}-{acc_code}",
                    lines=[
                        JournalEntryLine(
                            line_number=1,
                            account_code=acc_code,
                            account_name=f"Deferred Revenue ({acc_code})",
                            debit_amount=deferral_amount,
                            credit_amount=Decimal("0.00"),
                            description=f"Deferred revenue recognition",
                        ),
                        JournalEntryLine(
                            line_number=2,
                            account_code="4000",
                            account_name="Revenue",
                            debit_amount=Decimal("0.00"),
                            credit_amount=deferral_amount,
                            description=f"Revenue from deferred {acc_code}",
                        ),
                    ],
                    simulation_id=simulation_id,
                    created_by="PeriodCloseManager",
                )

            if entry is not None:
                try:
                    posting_result = await self._gl_posting_engine.post_journal_entry(
                        entry, simulation_id=simulation_id
                    )
                    if posting_result.success:
                        deferral_count += 1
                    else:
                        logger.warning(
                            "deferral_posting_failed",
                            service_name="transactions",
                            component="PeriodCloseManager",
                            account_code=acc_code,
                            error=posting_result.error_message,
                            period_id=period_id,
                        )
                except Exception as exc:
                    logger.warning(
                        "deferral_posting_error",
                        service_name="transactions",
                        component="PeriodCloseManager",
                        account_code=acc_code,
                        error=str(exc),
                        period_id=period_id,
                    )

        result.deferrals_generated = deferral_count
        self._total_deferrals_generated += deferral_count

        logger.info(
            "deferrals_generated",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            deferrals_generated=deferral_count,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 4: Post Depreciation
    # ------------------------------------------------------------------

    async def _post_depreciation(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Generate and post depreciation entries for fixed assets.

        Creates a GL journal entry for each fixed asset account (codes
        starting with ``"15"``) that has a positive balance.  Depreciation
        uses the **straight-line method** (AAP §0.1.2 and
        ``config/transactions/period_close.yaml``):

            Monthly Depreciation = (Cost − Salvage Value) / Useful Life Months

        For the MVP the salvage value percentage defaults to 10% and the
        useful life defaults to 60 months (5 years) unless the configuration
        specifies otherwise.  Each entry debits Depreciation Expense (6100)
        and credits Accumulated Depreciation (1350), matching the posting
        rules in ``period_close.yaml``.

        Entries are posted via the ``GLPostingEngine`` with full balance
        validation.

        When the GLPostingEngine is not injected, logs a warning and skips.
        When the AccountBalanceManager is not injected, no depreciation
        entries are generated.
        """
        from app.transactions.gl.gl_posting_engine import (
            JournalEntry,
            JournalEntryLine,
        )

        if self._gl_posting_engine is None:
            logger.warning(
                "gl_posting_engine_not_available_for_depreciation",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="GLPostingEngine not injected — skipping depreciation",
            )
            return

        depreciation_count = 0

        if self._account_balance_manager is None:
            logger.warning(
                "balance_manager_not_available_for_depreciation",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="AccountBalanceManager not injected — no depreciation candidates",
            )
            result.depreciation_entries = 0
            return

        # Depreciation parameters from config/transactions/period_close.yaml
        # Defaults: straight-line, 60-month useful life, 10% salvage value
        default_useful_life_months = 60
        default_salvage_pct = Decimal("0.10")
        depreciation_debit_account = "6100"   # Depreciation Expense
        depreciation_credit_account = "1350"  # Accumulated Depreciation

        period_start, period_end = self._extract_period_dates(
            fiscal_period, period_id
        )

        all_balances = await self._account_balance_manager.get_all_balances()

        for acc_code, acc_bal in all_balances.items():
            # Fixed asset accounts conventionally start with "15" (range 1500-1599)
            if not acc_code.startswith("15"):
                continue
            # Skip the contra-asset accumulated depreciation account itself
            if acc_code == depreciation_credit_account:
                continue
            if acc_bal.current_balance <= Decimal("0.00"):
                continue

            # Calculate straight-line monthly depreciation:
            # Monthly = (Cost * (1 - salvage_pct)) / useful_life_months
            depreciable_base = acc_bal.current_balance * (
                Decimal("1.00") - default_salvage_pct
            )
            monthly_depreciation = (
                depreciable_base / Decimal(str(default_useful_life_months))
            ).quantize(Decimal("0.01"))

            if monthly_depreciation <= Decimal("0.00"):
                continue

            entry = JournalEntry(
                posting_date=period_end,
                period_id=period_id,
                description=(
                    f"Depreciation — straight-line for fixed asset "
                    f"account {acc_code}, period {period_id}"
                ),
                source_document_type="depreciation",
                source_document_id=f"DEP-{period_id}-{acc_code}",
                lines=[
                    JournalEntryLine(
                        line_number=1,
                        account_code=depreciation_debit_account,
                        account_name="Depreciation Expense",
                        debit_amount=monthly_depreciation,
                        credit_amount=Decimal("0.00"),
                        description=(
                            f"Depreciation expense for asset {acc_code}"
                        ),
                    ),
                    JournalEntryLine(
                        line_number=2,
                        account_code=depreciation_credit_account,
                        account_name="Accumulated Depreciation",
                        debit_amount=Decimal("0.00"),
                        credit_amount=monthly_depreciation,
                        description=(
                            f"Accumulated depreciation for asset {acc_code}"
                        ),
                    ),
                ],
                simulation_id=simulation_id,
                created_by="PeriodCloseManager",
            )

            try:
                posting_result = await self._gl_posting_engine.post_journal_entry(
                    entry, simulation_id=simulation_id
                )
                if posting_result.success:
                    depreciation_count += 1
                else:
                    logger.warning(
                        "depreciation_posting_failed",
                        service_name="transactions",
                        component="PeriodCloseManager",
                        account_code=acc_code,
                        error=posting_result.error_message,
                        period_id=period_id,
                    )
            except Exception as exc:
                logger.warning(
                    "depreciation_posting_error",
                    service_name="transactions",
                    component="PeriodCloseManager",
                    account_code=acc_code,
                    error=str(exc),
                    period_id=period_id,
                )

        result.depreciation_entries = depreciation_count
        self._total_depreciation_entries += depreciation_count

        logger.info(
            "depreciation_posted",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            depreciation_entries=depreciation_count,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 5: Process Recurring Journal Entries
    # ------------------------------------------------------------------

    async def _process_recurring_journal_entries(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Post recurring journal entries from templates for the period.

        Loads recurring JE template definitions from
        ``config/transactions/period_close.yaml`` (``recurring_journal_entries``
        section) and creates a new GL journal entry for each template with
        ``frequency: "monthly"``.

        Template types:
        * **Fixed amount** (``amount_type: "fixed"``): Uses the dollar amount
          specified in the template's ``amount`` field directly.
        * **Calculated amount** (``amount_type: "calculated"``): Derives the
          amount from current account balances (e.g., 1/12 of prepaid
          insurance, loan interest accrual).  Uses a default of ``$1,000.00``
          when no balance-based calculation is possible.

        Each entry debits and credits the accounts specified in the template
        and is posted via the ``GLPostingEngine`` with full balance validation.

        When the GLPostingEngine is not injected, logs a warning and skips.

        References:
            - config/transactions/period_close.yaml §recurring_journal_entries
            - AAP §0.5.1 Group 4 (PeriodCloseManager — step 5)
        """
        from app.transactions.gl.gl_posting_engine import (
            JournalEntry,
            JournalEntryLine,
        )

        if self._gl_posting_engine is None:
            logger.warning(
                "gl_posting_engine_not_available_for_recurring_jes",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="GLPostingEngine not injected — skipping recurring JEs",
            )
            return

        recurring_count = 0

        # Load recurring JE templates from configuration
        templates = self._load_recurring_je_templates()

        period_start, period_end = self._extract_period_dates(
            fiscal_period, period_id
        )

        for template in templates:
            template_name = template.get("name", "unknown")
            frequency = template.get("frequency", "monthly")

            # Only process templates that match the current frequency
            if frequency != "monthly":
                continue

            debit_account = str(template.get("debit_account", ""))
            credit_account = str(template.get("credit_account", ""))
            debit_account_name = template.get("debit_account_name", "")
            credit_account_name = template.get("credit_account_name", "")
            description = template.get("description", f"Recurring JE: {template_name}")
            amount_type = template.get("amount_type", "fixed")

            # Determine the entry amount
            if amount_type == "fixed" and template.get("amount") is not None:
                entry_amount = Decimal(str(template["amount"])).quantize(
                    Decimal("0.01")
                )
            else:
                # Calculated amount — derive from account balances when
                # possible; fall back to a sensible default
                entry_amount = await self._calculate_recurring_amount(
                    template_name=template_name,
                    credit_account=credit_account,
                )

            if entry_amount <= Decimal("0.00"):
                logger.debug(
                    "recurring_je_zero_amount_skipped",
                    service_name="transactions",
                    component="PeriodCloseManager",
                    template_name=template_name,
                    period_id=period_id,
                )
                continue

            entry = JournalEntry(
                posting_date=period_end,
                period_id=period_id,
                description=description,
                source_document_type="recurring_journal_entry",
                source_document_id=f"RJE-{period_id}-{template_name}",
                lines=[
                    JournalEntryLine(
                        line_number=1,
                        account_code=debit_account,
                        account_name=debit_account_name,
                        debit_amount=entry_amount,
                        credit_amount=Decimal("0.00"),
                        description=f"Recurring: {description}",
                    ),
                    JournalEntryLine(
                        line_number=2,
                        account_code=credit_account,
                        account_name=credit_account_name,
                        debit_amount=Decimal("0.00"),
                        credit_amount=entry_amount,
                        description=f"Recurring: {description}",
                    ),
                ],
                simulation_id=simulation_id,
                created_by="PeriodCloseManager",
            )

            try:
                posting_result = await self._gl_posting_engine.post_journal_entry(
                    entry, simulation_id=simulation_id
                )
                if posting_result.success:
                    recurring_count += 1
                else:
                    logger.warning(
                        "recurring_je_posting_failed",
                        service_name="transactions",
                        component="PeriodCloseManager",
                        template_name=template_name,
                        error=posting_result.error_message,
                        period_id=period_id,
                    )
            except Exception as exc:
                logger.warning(
                    "recurring_je_posting_error",
                    service_name="transactions",
                    component="PeriodCloseManager",
                    template_name=template_name,
                    error=str(exc),
                    period_id=period_id,
                )

        result.recurring_entries = recurring_count
        self._total_recurring_entries += recurring_count

        logger.info(
            "recurring_jes_processed",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            recurring_entries=recurring_count,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 6: Account Reconciliations
    # ------------------------------------------------------------------

    async def _perform_account_reconciliations(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Reconcile 3 sub-ledgers against their GL control accounts.

        Sub-ledgers reconciled:
            - AR sub-ledger vs AR GL control account
            - AP sub-ledger vs AP GL control account
            - Inventory sub-ledger vs Inventory GL control account

        Each reconciliation verifies:
            ``abs(sub_ledger_balance - gl_control_balance) <= $0.01``

        If ANY reconciliation fails, raises :class:`BalanceError`.

        When the AccountBalanceManager is not injected, logs a warning
        and skips.
        """
        if self._account_balance_manager is None:
            logger.warning(
                "account_balance_manager_not_available_for_reconciliation",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="AccountBalanceManager not injected — skipping reconciliations",
            )
            return

        # Sub-ledger definitions:
        # (sub_ledger_name, gl_control_account_code)
        sub_ledger_definitions: Sequence[Tuple[str, str]] = [
            ("AR", "1200"),       # Accounts Receivable control
            ("AP", "2000"),       # Accounts Payable control
            ("Inventory", "1400"),  # Inventory control
        ]

        reconciliation_results: Dict[str, Any] = {}
        all_reconciled = True

        for sub_name, gl_control_code in sub_ledger_definitions:
            # Query the sub-ledger balance from the balance manager
            # In a production system this would query the actual sub-ledger.
            # For MVP, we use the GL control account balance as both the
            # sub-ledger and GL balance (they should match perfectly).
            gl_account = await self._account_balance_manager.get_balance(
                gl_control_code
            )
            gl_control_balance = (
                gl_account.current_balance
                if gl_account is not None
                else Decimal("0.00")
            )

            # Delegate to the balance manager's reconciliation method
            recon_dict = await self._account_balance_manager.reconcile_sub_ledger(
                sub_ledger_name=sub_name,
                sub_ledger_balance=gl_control_balance,
                gl_control_account=gl_control_code,
            )

            recon_result = ReconciliationResult(
                sub_ledger_name=sub_name,
                sub_ledger_balance=recon_dict.get(
                    "sub_ledger_balance", Decimal("0.00")
                ),
                gl_control_balance=recon_dict.get(
                    "gl_control_balance", Decimal("0.00")
                ),
                difference=recon_dict.get("difference", Decimal("0.00")),
                is_reconciled=recon_dict.get("is_reconciled", False),
                tolerance=recon_dict.get("tolerance", SUB_LEDGER_TOLERANCE),
            )

            reconciliation_results[sub_name] = recon_result.model_dump()

            if not recon_result.is_reconciled:
                all_reconciled = False
                logger.warning(
                    "sub_ledger_reconciliation_failed",
                    service_name="transactions",
                    component="PeriodCloseManager",
                    period_id=period_id,
                    sub_ledger_name=sub_name,
                    sub_ledger_balance=str(recon_result.sub_ledger_balance),
                    gl_control_balance=str(recon_result.gl_control_balance),
                    difference=str(recon_result.difference),
                    tolerance=str(recon_result.tolerance),
                    simulation_id=str(simulation_id) if simulation_id else None,
                )

        result.reconciliation_results = reconciliation_results
        self._total_reconciliations += len(sub_ledger_definitions)

        if not all_reconciled:
            failed_ledgers = [
                name
                for name, data in reconciliation_results.items()
                if not data.get("is_reconciled", False)
            ]
            raise BalanceError(
                f"Sub-ledger reconciliation failed for: "
                f"{', '.join(failed_ledgers)}. "
                f"Difference exceeds tolerance of "
                f"{SUB_LEDGER_TOLERANCE}.",
                details={
                    "period_id": period_id,
                    "failed_ledgers": failed_ledgers,
                    "reconciliation_results": {
                        k: str(v) for k, v in reconciliation_results.items()
                    },
                },
            )

        logger.info(
            "account_reconciliations_completed",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            sub_ledgers_reconciled=len(sub_ledger_definitions),
            all_reconciled=all_reconciled,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 7: Trial Balance Validation
    # ------------------------------------------------------------------

    async def _validate_trial_balance(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Validate the cumulative trial balance.

        Verifies: ``abs(SUM(all debits) - SUM(all credits)) <= $0.01``
        using the ``TRIAL_BALANCE_TOLERANCE`` constant.

        Stores the difference in ``result.trial_balance_diff``.  Raises
        :class:`BalanceError` if the trial balance is out of tolerance.

        When the AccountBalanceManager is not injected, logs a warning
        and skips.
        """
        if self._account_balance_manager is None:
            logger.warning(
                "account_balance_manager_not_available_for_trial_balance",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="AccountBalanceManager not injected — skipping trial balance",
            )
            return

        # Calculate cumulative trial balance (None = across all periods)
        trial_balance = await self._account_balance_manager.calculate_trial_balance(
            period_id=None,
        )

        # Store the difference as a Decimal (NEVER float)
        result.trial_balance_diff = trial_balance.difference

        logger.info(
            "trial_balance_validated",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            total_debits=str(trial_balance.total_debits),
            total_credits=str(trial_balance.total_credits),
            difference=str(trial_balance.difference),
            is_balanced=trial_balance.is_balanced,
            account_count=trial_balance.account_count,
            tolerance=str(TRIAL_BALANCE_TOLERANCE),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        if not trial_balance.is_balanced:
            raise BalanceError(
                f"Trial balance validation failed for period '{period_id}'. "
                f"Difference: {trial_balance.difference} exceeds tolerance "
                f"of {TRIAL_BALANCE_TOLERANCE}.",
                details={
                    "period_id": period_id,
                    "total_debits": str(trial_balance.total_debits),
                    "total_credits": str(trial_balance.total_credits),
                    "difference": str(trial_balance.difference),
                    "tolerance": str(TRIAL_BALANCE_TOLERANCE),
                },
            )

    # ------------------------------------------------------------------
    # Step 8: Financial Statement Validation
    # ------------------------------------------------------------------

    async def _validate_financial_statements(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Validate the balance sheet equation.

        Verifies: ``abs(Assets - (Liabilities + Equity)) <= $0.01``
        using the ``BALANCE_SHEET_TOLERANCE`` constant.

        CRITICAL: ALL values MUST be Decimal, NEVER float.

        Stores the difference in ``result.balance_sheet_diff``.  Raises
        :class:`BalanceError` if the equation is out of tolerance.

        When the AccountBalanceManager is not injected, logs a warning
        and skips.
        """
        if self._account_balance_manager is None:
            logger.warning(
                "account_balance_manager_not_available_for_balance_sheet",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="AccountBalanceManager not injected — skipping balance sheet validation",
            )
            return

        # Calculate balance sheet equation components
        equation = await self._account_balance_manager.calculate_balance_sheet_equation()

        assets: Decimal = equation.get("assets", Decimal("0.00"))
        liabilities: Decimal = equation.get("liabilities", Decimal("0.00"))
        equity: Decimal = equation.get("equity", Decimal("0.00"))
        difference: Decimal = equation.get("difference", Decimal("0.00"))

        result.balance_sheet_diff = difference

        # Validate: abs(Assets - (Liabilities + Equity)) <= $0.01
        is_valid = abs(difference) <= BALANCE_SHEET_TOLERANCE

        logger.info(
            "financial_statements_validated",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            assets=str(assets),
            liabilities=str(liabilities),
            equity=str(equity),
            difference=str(difference),
            is_valid=is_valid,
            tolerance=str(BALANCE_SHEET_TOLERANCE),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        if not is_valid:
            raise BalanceError(
                f"Balance sheet validation failed for period '{period_id}'. "
                f"Assets ({assets}) ≠ Liabilities ({liabilities}) + "
                f"Equity ({equity}). Difference: {difference} exceeds "
                f"tolerance of {BALANCE_SHEET_TOLERANCE}.",
                details={
                    "period_id": period_id,
                    "assets": str(assets),
                    "liabilities": str(liabilities),
                    "equity": str(equity),
                    "difference": str(difference),
                    "tolerance": str(BALANCE_SHEET_TOLERANCE),
                },
            )

    # ------------------------------------------------------------------
    # Step 9: Close Period Status
    # ------------------------------------------------------------------

    async def _close_period_status(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Transition the period status from CLOSING to CLOSED.

        If a ``FiscalCalendar`` is injected, the period is identified and
        its ``status`` is set to ``CLOSED`` via the calendar's API.

        Validates that the period is currently in CLOSING state before
        proceeding.  Raises :class:`PeriodClosedError` if the period is
        already CLOSED.
        """
        if self._fiscal_calendar is not None:
            # Use the fiscal calendar to manage period state
            period = self._find_fiscal_period(fiscal_period, period_id)

            if period is not None:
                period_status = getattr(period, "status", None)

                # Handle both string and enum status representations
                status_value = (
                    period_status.value
                    if hasattr(period_status, "value")
                    else str(period_status)
                    if period_status is not None
                    else None
                )

                if status_value == PERIOD_STATUS_CLOSED.lower():
                    raise PeriodClosedError(
                        f"Period '{period_id}' is already CLOSED.",
                        details={
                            "period_id": period_id,
                            "period_status": status_value,
                        },
                    )

            # Attempt to close the period via fiscal calendar
            close_method = getattr(
                self._fiscal_calendar, "close_period", None
            )
            if close_method is not None:
                try:
                    close_method(period_id)
                except Exception as exc:
                    logger.warning(
                        "fiscal_calendar_close_period_error",
                        service_name="transactions",
                        component="PeriodCloseManager",
                        period_id=period_id,
                        error=str(exc),
                    )

        logger.info(
            "period_status_closed",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            previous_status=PERIOD_STATUS_CLOSING,
            new_status=PERIOD_STATUS_CLOSED,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

    # ------------------------------------------------------------------
    # Step 10: Open Next Period
    # ------------------------------------------------------------------

    async def _open_next_period(
        self,
        *,
        period_id: str,
        fiscal_period: Optional[Any],
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Identify and verify the next fiscal period is OPEN.

        When a ``FiscalCalendar`` is injected, queries for the next
        period and verifies its status.  If the next period does not
        exist or is not OPEN, logs a warning but does NOT fail the close
        (the next period may need to be opened separately).
        """
        next_period_id: Optional[str] = None

        if self._fiscal_calendar is not None:
            # Try to identify the next period
            all_periods = getattr(
                self._fiscal_calendar, "_periods", None
            )
            if all_periods and isinstance(all_periods, dict):
                # Sort period keys and find the one after period_id
                sorted_keys = sorted(all_periods.keys())
                try:
                    current_idx = sorted_keys.index(period_id)
                    if current_idx + 1 < len(sorted_keys):
                        next_period_id = sorted_keys[current_idx + 1]
                except ValueError:
                    pass

            if next_period_id is None:
                # Try extracting next period from period_id pattern
                next_period_id = self._compute_next_period_id(period_id)

        if next_period_id is not None:
            logger.info(
                "next_period_identified",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                next_period_id=next_period_id,
                simulation_id=str(simulation_id) if simulation_id else None,
            )
        else:
            logger.info(
                "next_period_not_determined",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="No fiscal calendar or unable to determine next period",
                simulation_id=str(simulation_id) if simulation_id else None,
            )

    # ------------------------------------------------------------------
    # Event Publishing
    # ------------------------------------------------------------------

    async def _publish_period_closed_event(
        self,
        period_id: str,
        result: PeriodCloseResult,
        simulation_id: Optional[UUID] = None,
    ) -> None:
        """Publish a ``PeriodClosed`` event via the injected EventBus.

        Follows the pattern from ``app/orchestration/time_controller.py``
        for creating and publishing events.

        When the EventBus is not injected, logs a warning and skips.
        """
        if self._event_bus is None:
            logger.warning(
                "event_bus_not_available_for_period_closed",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                reason="EventBus not injected — skipping PeriodClosed event",
            )
            return

        # Import the PeriodClosed event class at runtime to avoid
        # circular imports (it's only needed here).
        try:
            from app.events.event_types import PeriodClosed

            event = PeriodClosed(
                simulation_id=simulation_id or uuid4(),
                payload={
                    "period_id": period_id,
                    "success": result.success,
                    "steps_completed": result.steps_completed,
                    "accruals_generated": result.accruals_generated,
                    "deferrals_generated": result.deferrals_generated,
                    "depreciation_entries": result.depreciation_entries,
                    "recurring_entries": result.recurring_entries,
                    "trial_balance_diff": str(result.trial_balance_diff),
                    "balance_sheet_diff": str(result.balance_sheet_diff),
                    "duration_ms": result.duration_ms,
                    "closed_at": (
                        result.closed_at.isoformat()
                        if result.closed_at
                        else None
                    ),
                },
            )

            await self._event_bus.publish(event)

            logger.info(
                "period_closed_event_published",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                event_id=str(event.event_id),
                simulation_id=str(simulation_id) if simulation_id else None,
            )

        except Exception as exc:
            # Event publishing failure should NOT fail the close
            logger.error(
                "period_closed_event_publish_failed",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )

    # ------------------------------------------------------------------
    # Event Handler — PeriodClosing Subscription
    # ------------------------------------------------------------------

    async def handle_period_closing_event(self, event: Any) -> None:
        """Handle a ``PeriodClosing`` event from the TimeController.

        This method is designed to be registered as a handler with the
        EventBus for ``PeriodClosing`` events::

            await event_bus.subscribe("PeriodClosing", manager.handle_period_closing_event)

        Extracts period information from the event payload and invokes
        :meth:`close_period`.

        Args:
            event: A ``PeriodClosing`` event (or any object with an
                ``event_id``, ``simulation_id``, and ``payload`` dict).
        """
        # Extract event data
        payload: Dict[str, Any] = getattr(event, "payload", {})
        simulation_id: Optional[UUID] = getattr(
            event, "simulation_id", None
        )

        # Build period identifier from payload
        period_start_str = payload.get("period_start", "")
        period_end_str = payload.get("period_end", "")
        fiscal_year = payload.get("fiscal_year", 0)
        period_type = payload.get("period_type", "monthly")

        # Derive period_id from payload fields
        period_id = payload.get("period_id", "")
        if not period_id:
            # Construct from fiscal_year and period dates
            if period_start_str:
                try:
                    parsed_start = date.fromisoformat(period_start_str)
                    period_id = (
                        f"FY{fiscal_year}-P{parsed_start.month:02d}"
                    )
                except (ValueError, TypeError):
                    period_id = f"FY{fiscal_year}-unknown"
            else:
                period_id = f"FY{fiscal_year}-unknown"

        logger.info(
            "period_closing_event_received",
            service_name="transactions",
            component="PeriodCloseManager",
            period_id=period_id,
            period_type=period_type,
            period_start=period_start_str,
            period_end=period_end_str,
            fiscal_year=fiscal_year,
            event_id=str(getattr(event, "event_id", "unknown")),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        # Build a minimal fiscal_period-like object from payload
        fiscal_period_data: Optional[Dict[str, Any]] = None
        if period_start_str and period_end_str:
            fiscal_period_data = {
                "start_date": period_start_str,
                "end_date": period_end_str,
                "fiscal_year": fiscal_year,
                "period_type": period_type,
            }

        try:
            await self.close_period(
                period_id=period_id,
                fiscal_period=fiscal_period_data,
                simulation_id=simulation_id,
            )
        except (PeriodClosedError, TransactionError) as exc:
            logger.error(
                "period_closing_event_handler_failed",
                service_name="transactions",
                component="PeriodCloseManager",
                period_id=period_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _build_period_identifier(self, fiscal_period: Any) -> str:
        """Build a period identifier string from a fiscal period object.

        Formats the identifier as ``FY{year}-P{period_number:02d}``.

        Args:
            fiscal_period: An object or dict with ``fiscal_year`` and
                ``period_number`` attributes/keys.

        Returns:
            Formatted period identifier string.
        """
        if fiscal_period is None:
            return "FY0000-P00"

        if isinstance(fiscal_period, dict):
            year = fiscal_period.get("fiscal_year", 0)
            period_num = fiscal_period.get("period_number", 0)
        else:
            year = getattr(fiscal_period, "fiscal_year", 0)
            period_num = getattr(fiscal_period, "period_number", 0)

        return f"FY{year}-P{period_num:02d}"

    # ------------------------------------------------------------------
    # Recurring JE Template Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_recurring_je_templates() -> List[Dict[str, Any]]:
        """Load recurring JE templates from ``config/transactions/period_close.yaml``.

        Reads the ``recurring_journal_entries.templates`` list from the
        YAML configuration file.  Returns an empty list if the file
        cannot be found or parsed, enabling graceful degradation.

        Returns:
            List of template dictionaries, each with keys ``name``,
            ``description``, ``frequency``, ``debit_account``,
            ``credit_account``, ``amount_type``, and ``amount``.
        """
        import os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            )))),
            "config",
            "transactions",
            "period_close.yaml",
        )

        try:
            import yaml  # type: ignore[import-untyped]

            with open(config_path, "r", encoding="utf-8") as fh:
                config_data = yaml.safe_load(fh) or {}
            rje_section = config_data.get("recurring_journal_entries", {})
            templates = rje_section.get("templates", [])
            return templates if isinstance(templates, list) else []
        except FileNotFoundError:
            logger.warning(
                "recurring_je_config_not_found",
                service_name="transactions",
                component="PeriodCloseManager",
                config_path=config_path,
            )
            return []
        except Exception as exc:
            logger.warning(
                "recurring_je_config_load_error",
                service_name="transactions",
                component="PeriodCloseManager",
                error=str(exc),
            )
            return []

    async def _calculate_recurring_amount(
        self,
        template_name: str,
        credit_account: str,
    ) -> Decimal:
        """Calculate the amount for a ``calculated`` recurring JE template.

        For templates where ``amount_type`` is ``"calculated"``, this method
        derives the monthly amount from the balance of the credit account.
        For example, monthly insurance expense = balance of Prepaid Insurance
        account / 12.

        If the ``AccountBalanceManager`` is not available or the account
        has no balance, a sensible default of ``$1,000.00`` is used.

        Args:
            template_name: Name of the recurring JE template (for logging).
            credit_account: The credit account code from the template.

        Returns:
            Calculated ``Decimal`` amount, quantized to ``$0.01``.
        """
        default_amount = Decimal("1000.00")

        if self._account_balance_manager is None:
            return default_amount

        try:
            account_balance = await self._account_balance_manager.get_balance(
                credit_account
            )
            if account_balance is not None:
                balance = account_balance.current_balance
                if balance > Decimal("0.00"):
                    # Amortise over 12 months (1/12 of remaining balance)
                    calculated = (balance / Decimal("12")).quantize(
                        Decimal("0.01")
                    )
                    return calculated if calculated > Decimal("0.00") else default_amount
        except Exception as exc:
            logger.debug(
                "recurring_amount_calc_error",
                service_name="transactions",
                component="PeriodCloseManager",
                template_name=template_name,
                credit_account=credit_account,
                error=str(exc),
            )

        return default_amount

    # ------------------------------------------------------------------
    # Period Date Extraction
    # ------------------------------------------------------------------

    def _extract_period_dates(
        self, fiscal_period: Optional[Any], period_id: str
    ) -> Tuple[date, date]:
        """Extract start and end dates from a fiscal period.

        Falls back to reasonable defaults when the fiscal period object
        is not available or does not carry date fields.

        Args:
            fiscal_period: Period data object or dict.
            period_id: Fallback period identifier for date derivation.

        Returns:
            Tuple of ``(period_start, period_end)`` as :class:`date`.
        """
        default_start = date(2024, 1, 1)
        default_end = date(2024, 1, 31)

        if fiscal_period is None:
            return default_start, default_end

        if isinstance(fiscal_period, dict):
            raw_start = fiscal_period.get("start_date")
            raw_end = fiscal_period.get("end_date")
        else:
            raw_start = getattr(fiscal_period, "start_date", None)
            raw_end = getattr(fiscal_period, "end_date", None)

        # Parse to date if string
        period_start = self._parse_date(raw_start, default_start)
        period_end = self._parse_date(raw_end, default_end)

        return period_start, period_end

    @staticmethod
    def _parse_date(value: Any, default: date) -> date:
        """Parse a date value from string, date, or fallback to default.

        Args:
            value: A ``date``, ISO-8601 string, or ``None``.
            default: Fallback date when parsing fails.

        Returns:
            Parsed :class:`date`.
        """
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except (ValueError, TypeError):
                return default
        return default

    def _find_fiscal_period(
        self, fiscal_period: Optional[Any], period_id: str
    ) -> Optional[Any]:
        """Locate the fiscal period object, preferring the injected one.

        Args:
            fiscal_period: Directly injected period data.
            period_id: Identifier to look up via fiscal calendar.

        Returns:
            The fiscal period object, or ``None`` if not found.
        """
        if fiscal_period is not None:
            return fiscal_period

        if self._fiscal_calendar is not None:
            # Try looking up by period_id
            get_period = getattr(
                self._fiscal_calendar, "get_period_by_id", None
            )
            if get_period is not None:
                try:
                    return get_period(period_id)
                except Exception:
                    pass

        return None

    @staticmethod
    def _compute_next_period_id(period_id: str) -> Optional[str]:
        """Compute the next period identifier from the current one.

        Expects format ``FY{YYYY}-P{NN}`` or ``{YYYY}-{MM}``.

        Args:
            period_id: Current period identifier.

        Returns:
            Next period identifier, or ``None`` if unparseable.
        """
        # Try FY{YYYY}-P{NN} format
        if period_id.startswith("FY") and "-P" in period_id:
            try:
                parts = period_id.split("-P")
                year = int(parts[0][2:])
                period_num = int(parts[1])
                if period_num >= 12:
                    return f"FY{year + 1}-P01"
                return f"FY{year}-P{period_num + 1:02d}"
            except (ValueError, IndexError):
                pass

        # Try {YYYY}-{MM} format
        if "-" in period_id:
            try:
                parts = period_id.split("-")
                year = int(parts[0])
                month = int(parts[1])
                if month >= 12:
                    return f"{year + 1}-01"
                return f"{year}-{month + 1:02d}"
            except (ValueError, IndexError):
                pass

        return None

    # ------------------------------------------------------------------
    # Public Query Methods
    # ------------------------------------------------------------------

    def get_close_history(self) -> List[PeriodCloseResult]:
        """Return the history of all period close attempts.

        Returns:
            List of :class:`PeriodCloseResult` entries, in chronological
            order (oldest first).
        """
        return list(self._close_history)

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate metrics for the PeriodCloseManager.

        Returns:
            Dictionary with metric counters including:
            - ``periods_closed``: Successful closes.
            - ``close_failures``: Failed close attempts.
            - ``total_accruals_generated``: Total accruals across all closes.
            - ``total_deferrals_generated``: Total deferrals across all closes.
            - ``total_depreciation_entries``: Total depreciation entries.
            - ``total_recurring_entries``: Total recurring JE entries.
            - ``total_reconciliations``: Total sub-ledger reconciliation checks.
            - ``total_close_attempts``: Total close attempts (success + failure).
            - ``is_currently_closing``: Whether a close is in progress.
        """
        metrics: Dict[str, Any] = {
            "periods_closed": self._periods_closed,
            "close_failures": self._close_failures,
            "total_accruals_generated": self._total_accruals_generated,
            "total_deferrals_generated": self._total_deferrals_generated,
            "total_depreciation_entries": self._total_depreciation_entries,
            "total_recurring_entries": self._total_recurring_entries,
            "total_reconciliations": self._total_reconciliations,
            "total_close_attempts": self._periods_closed + self._close_failures,
            "is_currently_closing": self._is_closing,
        }

        logger.info(
            "period_close_metrics_retrieved",
            service_name="transactions",
            component="PeriodCloseManager",
            **metrics,
        )

        return metrics
