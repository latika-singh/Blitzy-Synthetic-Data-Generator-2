"""Account Balance Manager — real-time balance maintenance with pessimistic locking.

Maintains running account balances across the Chart of Accounts, enforcing:

1. **SELECT FOR UPDATE Locking**: Uses pessimistic locking when reading balances
   to prevent race conditions during concurrent balance updates. This is the
   ONLY acceptable locking strategy per AAP §0.1.2.

2. **Normal Balance Direction Enforcement**:
   - **Asset and Expense** accounts: Debits INCREASE the balance
   - **Liability, Equity, and Revenue** accounts: Credits INCREASE the balance
   The balance manager enforces this invariant for every balance update.

3. **Running Balance Maintenance**: Each account maintains a running balance
   that is updated in real-time as journal entries are posted.

4. **Period-Based Balance Tracking**: Balances are tracked per fiscal period
   in addition to the cumulative running balance, enabling period-end
   reporting and trial balance calculations.

Database Session Pattern (AAP §0.1.2):
    ``from synthetic_erp.db.session import get_session``
    ``async with get_session() as session:``

Retry Policy (AAP §0.1.2 — balance_update):
    - max_attempts: 2
    - backoff: linear (500ms, 1s)
    - timeout: 10s
    - fallback: rollback transaction

References:
    - AAP §0.5.1 Group 4: GL Integration (AccountBalanceManager)
    - AAP §0.7.2: Financial Integrity Rules (normal balance direction, sub-ledger reconciliation)
    - AAP §0.1.2: SELECT FOR UPDATE for concurrent balance access
    - AAP §0.1.2: Retry Policy Table (balance_update)
"""

from __future__ import annotations

import asyncio
import decimal
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.exceptions import (
    BalanceError,
    ConcurrencyError,
    GLPostingError,
    TransactionError,
)
from app.transactions.constants import (
    FINANCIAL_TOLERANCES,
    GL_BALANCE_TOLERANCE,
    SUB_LEDGER_TOLERANCE,
    BALANCE_SHEET_TOLERANCE,
    RETRY_POLICIES,
    DEBIT_NORMAL_ACCOUNT_TYPES,
    CREDIT_NORMAL_ACCOUNT_TYPES,
    PERIOD_STATUS_OPEN,
)

if TYPE_CHECKING:
    from app.events.event_bus import EventBus

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# CRITICAL: Decimal precision configuration (AAP §0.7.2)
# All financial calculations MUST use Decimal with prec=28 and ROUND_HALF_UP.
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Lock acquisition timeout for simulating SELECT FOR UPDATE (seconds).
# Derived from RETRY_POLICIES["balance_update"]["timeout_seconds"].
# ---------------------------------------------------------------------------
_LOCK_ACQUISITION_TIMEOUT: float = float(
    RETRY_POLICIES.get("balance_update", {}).get("timeout_seconds", 10)
)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Models
# ═══════════════════════════════════════════════════════════════════════════


class AccountBalance(BaseModel):
    """Represents the current balance state of a single General Ledger account.

    Tracks both the cumulative running balance and the period-specific debit
    and credit totals.  The ``normal_balance_side`` field is automatically
    derived from the ``account_type`` during :meth:`AccountBalanceManager.register_account`.

    Attributes:
        account_code: Chart of Accounts account code (e.g., ``"1010"``).
        account_name: Human-readable display name for the account.
        account_type: One of ``"asset"``, ``"liability"``, ``"equity"``,
            ``"revenue"``, or ``"expense"``.
        normal_balance_side: ``"debit"`` for asset/expense accounts,
            ``"credit"`` for liability/equity/revenue accounts.
        current_balance: Running balance reflecting the natural direction.
        period_debits: Sum of debit amounts applied in the current period.
        period_credits: Sum of credit amounts applied in the current period.
        period_id: Fiscal period identifier (e.g., ``"2024-01"``).
        last_updated: UTC timestamp of the most recent balance change.
        version: Optimistic concurrency version counter.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    account_code: str = Field(..., description="Chart of Accounts account code")
    account_name: str = Field(default="", description="Human-readable account name")
    account_type: str = Field(
        ...,
        description="'asset', 'liability', 'equity', 'revenue', 'expense'",
    )
    normal_balance_side: str = Field(
        default="debit", description="'debit' or 'credit'"
    )
    current_balance: Decimal = Field(
        default=Decimal("0.00"), description="Running balance"
    )
    period_debits: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Total debits for current period",
    )
    period_credits: Decimal = Field(
        default=Decimal("0.00"),
        ge=Decimal("0.00"),
        description="Total credits for current period",
    )
    period_id: Optional[str] = Field(
        default=None, description="Current fiscal period"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    version: int = Field(
        default=0, ge=0, description="Optimistic version for concurrency"
    )


class BalanceUpdateRequest(BaseModel):
    """Request to apply a debit and/or credit to a specific GL account.

    At least one of ``debit_amount`` or ``credit_amount`` must be non-zero
    for the update to have an effect.  Both are validated to be >= 0.

    Attributes:
        account_code: Target GL account code.
        debit_amount: Amount to debit (always >= 0).
        credit_amount: Amount to credit (always >= 0).
        posting_date: Business date of the journal entry.
        period_id: Fiscal period for period-based tracking.
        journal_entry_id: UUID of the parent journal entry.
        description: Free-text description of the update.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    account_code: str = Field(...)
    debit_amount: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    credit_amount: Decimal = Field(default=Decimal("0.00"), ge=Decimal("0.00"))
    posting_date: date = Field(...)
    period_id: Optional[str] = Field(default=None)
    journal_entry_id: Optional[UUID] = Field(default=None)
    description: str = Field(default="")


class BalanceUpdateResult(BaseModel):
    """Result of a balance update operation.

    Contains the before/after balance values and a success indicator.
    When ``success`` is ``False``, ``error_message`` carries a human-readable
    description of the failure.

    Attributes:
        account_code: GL account that was updated.
        previous_balance: Balance before the update.
        new_balance: Balance after the update (only meaningful when success=True).
        debit_applied: Debit amount that was applied.
        credit_applied: Credit amount that was applied.
        success: Whether the update was applied successfully.
        error_message: Description of failure if success is False.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    account_code: str = Field(...)
    previous_balance: Decimal = Field(default=Decimal("0.00"))
    new_balance: Decimal = Field(default=Decimal("0.00"))
    debit_applied: Decimal = Field(default=Decimal("0.00"))
    credit_applied: Decimal = Field(default=Decimal("0.00"))
    success: bool = Field(default=False)
    error_message: Optional[str] = Field(default=None)


class TrialBalanceReport(BaseModel):
    """Trial balance report summarising debit and credit totals.

    The trial balance is *balanced* when the absolute ``difference``
    between ``total_debits`` and ``total_credits`` is within the
    ``GL_BALANCE_TOLERANCE`` ($0.01).

    Attributes:
        period_id: Fiscal period covered (``None`` for cumulative).
        total_debits: Aggregate period debits across all accounts.
        total_credits: Aggregate period credits across all accounts.
        difference: ``total_debits - total_credits``.
        is_balanced: ``True`` when ``abs(difference) <= $0.01``.
        account_count: Number of accounts included in the report.
        generated_at: UTC timestamp when the report was produced.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    period_id: Optional[str] = Field(default=None)
    total_debits: Decimal = Field(default=Decimal("0.00"))
    total_credits: Decimal = Field(default=Decimal("0.00"))
    difference: Decimal = Field(default=Decimal("0.00"))
    is_balanced: bool = Field(default=False)
    account_count: int = Field(default=0, ge=0)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ═══════════════════════════════════════════════════════════════════════════
# AccountBalanceManager
# ═══════════════════════════════════════════════════════════════════════════


class AccountBalanceManager:
    """Real-time GL account balance manager with pessimistic locking.

    Maintains an in-memory cache of account balances, enforces normal
    balance direction, provides period-based tracking, and performs
    trial balance / balance sheet / sub-ledger reconciliation.

    Constructor Injection (ADR-003):
        All dependencies are received as keyword-only ``Optional``
        parameters with ``None`` defaults.  ``None`` values silently
        disable the corresponding capability, allowing test isolation
        and incremental integration.

    Concurrency Model:
        An ``asyncio.Lock`` simulates the ``SELECT FOR UPDATE``
        pessimistic locking strategy.  Lock acquisition is bounded by
        ``_LOCK_ACQUISITION_TIMEOUT`` seconds; exceeding the timeout
        raises ``ConcurrencyError``.

    Financial Integrity (AAP §0.7.2):
        - ALL calculations use ``Decimal``.  ``float`` is NEVER used.
        - Normal balance direction: Asset/Expense → debit increases;
          Liability/Equity/Revenue → credit increases.
        - Trial balance tolerance: $0.01.
        - Balance sheet tolerance: $0.01.
        - Sub-ledger reconciliation tolerance: $0.01.

    Retry Policy (AAP §0.1.2 — balance_update):
        max_attempts=2, linear backoff (500ms, 1s), timeout=10s,
        fallback=rollback transaction.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        db_session_factory: Optional[Any] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialise the AccountBalanceManager.

        Args:
            db_session_factory: Callable returning an async context manager
                wrapping a database session (``get_session``).  When
                ``None``, all persistence is in-memory only.
            event_bus: Project 2 EventBus for publishing balance-related
                events.  When ``None``, events are silently skipped.
        """
        # Injected dependencies
        self._db_session_factory: Optional[Any] = db_session_factory
        self._event_bus: Optional[EventBus] = event_bus

        # In-memory balance caches
        self._balances: Dict[str, AccountBalance] = {}
        self._period_balances: Dict[str, Dict[str, AccountBalance]] = {}

        # Update history for auditing
        self._update_history: List[BalanceUpdateResult] = []

        # Concurrency: asyncio.Lock simulates SELECT FOR UPDATE
        self._lock: asyncio.Lock = asyncio.Lock()

        # Period management
        self._open_periods: Set[str] = set()
        self._closed_periods: Set[str] = set()

        # Metrics counters
        self._updates_applied: int = 0
        self._lock_acquisitions: int = 0
        self._lock_timeouts: int = 0
        self._balance_checks: int = 0
        self._batch_rollbacks: int = 0

        logger.info(
            "account_balance_manager_initialized",
            service_name="transactions",
            component="AccountBalanceManager",
            has_db_session=db_session_factory is not None,
            has_event_bus=event_bus is not None,
        )

    # ------------------------------------------------------------------
    # Core Balance Update Methods
    # ------------------------------------------------------------------

    async def update_balance(
        self, request: BalanceUpdateRequest
    ) -> BalanceUpdateResult:
        """Apply a debit/credit update to a single account balance.

        Acquires the internal ``asyncio.Lock`` (simulating ``SELECT FOR
        UPDATE``) before reading/writing the balance.  If the lock
        cannot be acquired within the configured timeout, a
        ``ConcurrencyError`` is raised.

        Normal balance direction:
            - **Debit-normal** (asset, expense):
              ``balance += debit_amount - credit_amount``
            - **Credit-normal** (liability, equity, revenue):
              ``balance += credit_amount - debit_amount``

        Args:
            request: The balance update request.

        Returns:
            A :class:`BalanceUpdateResult` indicating success or failure.

        Raises:
            ConcurrencyError: If the lock cannot be acquired in time.
            BalanceError: If the account type is invalid.
        """
        start_ts = time.perf_counter()
        account_code = request.account_code

        # --- Acquire lock (SELECT FOR UPDATE simulation) ---
        try:
            await asyncio.wait_for(
                self._lock.acquire(), timeout=_LOCK_ACQUISITION_TIMEOUT
            )
            self._lock_acquisitions += 1
        except asyncio.TimeoutError:
            self._lock_timeouts += 1
            logger.warning(
                "lock_acquisition_timeout",
                service_name="transactions",
                component="AccountBalanceManager",
                account_code=account_code,
                timeout_seconds=_LOCK_ACQUISITION_TIMEOUT,
            )
            raise ConcurrencyError(
                f"Lock acquisition timed out for account {account_code}",
                details={
                    "account_code": account_code,
                    "timeout_seconds": _LOCK_ACQUISITION_TIMEOUT,
                    "retry_eligible": True,
                },
            )

        try:
            return self._apply_update_locked(request, start_ts)
        finally:
            self._lock.release()

    def _apply_update_locked(
        self,
        request: BalanceUpdateRequest,
        start_ts: float,
    ) -> BalanceUpdateResult:
        """Apply the balance update while the lock is held.

        This is a synchronous helper called under the ``_lock`` to keep
        the critical section short and avoid awaiting inside the lock.

        Args:
            request: The update request.
            start_ts: ``time.perf_counter()`` when the operation started.

        Returns:
            :class:`BalanceUpdateResult`.
        """
        account_code = request.account_code

        # --- Account lookup (unregistered accounts are rejected) ---
        account = self._balances.get(account_code)
        if account is None:
            # Raise BalanceError for unregistered accounts rather than
            # silently auto-registering with a default type.  Auto-
            # registration as "expense" could assign the wrong normal
            # balance direction for Revenue, Liability, or Equity accounts,
            # violating financial integrity invariants (AAP §0.7.2).
            # Callers MUST register accounts via register_account() first.
            logger.warning(
                "balance_update_unregistered_account",
                service_name="transactions",
                component="AccountBalanceManager",
                account_code=account_code,
            )
            error_msg = (
                f"Account '{account_code}' is not registered. "
                f"Call register_account() before updating balances."
            )
            return BalanceUpdateResult(
                account_code=account_code,
                previous_balance=Decimal("0.00"),
                new_balance=Decimal("0.00"),
                debit_applied=Decimal("0.00"),
                credit_applied=Decimal("0.00"),
                success=False,
                error_message=error_msg,
            )

        previous_balance = account.current_balance

        # --- Period validation ---
        period_id = request.period_id
        if period_id and period_id in self._closed_periods:
            error_msg = (
                f"Cannot update balance for closed period {period_id}"
            )
            logger.warning(
                "balance_update_closed_period",
                service_name="transactions",
                component="AccountBalanceManager",
                account_code=account_code,
                period_id=period_id,
            )
            return BalanceUpdateResult(
                account_code=account_code,
                previous_balance=previous_balance,
                new_balance=previous_balance,
                debit_applied=Decimal("0.00"),
                credit_applied=Decimal("0.00"),
                success=False,
                error_message=error_msg,
            )

        # --- Normal balance direction enforcement (AAP §0.7.2) ---
        debit = request.debit_amount
        credit = request.credit_amount

        if account.account_type in DEBIT_NORMAL_ACCOUNT_TYPES:
            # Asset / Expense: debits increase the balance
            balance_change = debit - credit
        elif account.account_type in CREDIT_NORMAL_ACCOUNT_TYPES:
            # Liability / Equity / Revenue: credits increase the balance
            balance_change = credit - debit
        else:
            error_msg = (
                f"Unknown account type '{account.account_type}' "
                f"for account {account_code}"
            )
            logger.error(
                "invalid_account_type",
                service_name="transactions",
                component="AccountBalanceManager",
                account_code=account_code,
                account_type=account.account_type,
            )
            return BalanceUpdateResult(
                account_code=account_code,
                previous_balance=previous_balance,
                new_balance=previous_balance,
                debit_applied=Decimal("0.00"),
                credit_applied=Decimal("0.00"),
                success=False,
                error_message=error_msg,
            )

        # --- Apply update ---
        new_balance = previous_balance + balance_change
        account.current_balance = new_balance
        account.period_debits = account.period_debits + debit
        account.period_credits = account.period_credits + credit
        account.last_updated = datetime.now(timezone.utc)
        account.version += 1

        if period_id:
            account.period_id = period_id

        # --- Period-specific balance tracking ---
        if period_id:
            self._update_period_balance(
                account_code=account_code,
                account=account,
                period_id=period_id,
                debit=debit,
                credit=credit,
                balance_change=balance_change,
            )

        # --- Record result ---
        result = BalanceUpdateResult(
            account_code=account_code,
            previous_balance=previous_balance,
            new_balance=new_balance,
            debit_applied=debit,
            credit_applied=credit,
            success=True,
            error_message=None,
        )
        self._update_history.append(result)
        self._updates_applied += 1

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        logger.debug(
            "balance_updated",
            service_name="transactions",
            component="AccountBalanceManager",
            account_code=account_code,
            previous_balance=str(previous_balance),
            new_balance=str(new_balance),
            debit_applied=str(debit),
            credit_applied=str(credit),
            balance_change=str(balance_change),
            period_id=period_id,
            version=account.version,
            duration_ms=round(elapsed_ms, 2),
        )

        return result

    def _update_period_balance(
        self,
        *,
        account_code: str,
        account: AccountBalance,
        period_id: str,
        debit: Decimal,
        credit: Decimal,
        balance_change: Decimal,
    ) -> None:
        """Maintain the period-specific balance snapshot.

        Creates a new ``AccountBalance`` for the period if none exists,
        or updates the existing period balance.
        """
        if period_id not in self._period_balances:
            self._period_balances[period_id] = {}

        period_accounts = self._period_balances[period_id]
        period_bal = period_accounts.get(account_code)

        if period_bal is None:
            period_bal = AccountBalance(
                account_code=account_code,
                account_name=account.account_name,
                account_type=account.account_type,
                normal_balance_side=account.normal_balance_side,
                current_balance=Decimal("0.00"),
                period_debits=Decimal("0.00"),
                period_credits=Decimal("0.00"),
                period_id=period_id,
            )
            period_accounts[account_code] = period_bal

        period_bal.current_balance = period_bal.current_balance + balance_change
        period_bal.period_debits = period_bal.period_debits + debit
        period_bal.period_credits = period_bal.period_credits + credit
        period_bal.last_updated = datetime.now(timezone.utc)
        period_bal.version += 1

    async def update_balances_batch(
        self, requests: List[BalanceUpdateRequest]
    ) -> List[BalanceUpdateResult]:
        """Apply multiple balance updates atomically.

        Acquires the lock once for the entire batch.  If **any** update
        within the batch fails, **all** updates in the batch are rolled
        back to preserve atomicity (AAP §0.7.2).

        Args:
            requests: Ordered list of balance update requests.

        Returns:
            List of :class:`BalanceUpdateResult` entries (one per request).
            On rollback, every result shows ``success=False``.

        Raises:
            ConcurrencyError: If the lock cannot be acquired in time.
        """
        if not requests:
            return []

        start_ts = time.perf_counter()

        # --- Acquire lock once for entire batch ---
        try:
            await asyncio.wait_for(
                self._lock.acquire(), timeout=_LOCK_ACQUISITION_TIMEOUT
            )
            self._lock_acquisitions += 1
        except asyncio.TimeoutError:
            self._lock_timeouts += 1
            logger.warning(
                "batch_lock_acquisition_timeout",
                service_name="transactions",
                component="AccountBalanceManager",
                batch_size=len(requests),
                timeout_seconds=_LOCK_ACQUISITION_TIMEOUT,
            )
            raise ConcurrencyError(
                "Lock acquisition timed out for batch update",
                details={
                    "batch_size": len(requests),
                    "timeout_seconds": _LOCK_ACQUISITION_TIMEOUT,
                    "retry_eligible": True,
                },
            )

        try:
            return self._apply_batch_locked(requests, start_ts)
        finally:
            self._lock.release()

    def _apply_batch_locked(
        self,
        requests: List[BalanceUpdateRequest],
        start_ts: float,
    ) -> List[BalanceUpdateResult]:
        """Process a batch of updates under lock with rollback support.

        Captures pre-update snapshots so that any failure can trigger a
        full rollback of all changes made within this batch.  Tracks
        accounts and period entries that existed before the batch so that
        any newly created entries during the batch can be removed on
        rollback.
        """
        # --- Snapshot for rollback ---
        snapshot: Dict[str, Tuple[Decimal, Decimal, Decimal, int]] = {}
        period_snapshot: Dict[
            str, Dict[str, Tuple[Decimal, Decimal, Decimal, int]]
        ] = {}

        # Track which account codes existed before the batch, so that
        # newly added accounts (if any) can be removed on rollback.
        pre_batch_account_codes: Set[str] = set(self._balances.keys())
        pre_batch_period_account_codes: Dict[str, Set[str]] = {
            pid: set(accounts.keys())
            for pid, accounts in self._period_balances.items()
        }

        for req in requests:
            code = req.account_code
            if code in self._balances and code not in snapshot:
                acc = self._balances[code]
                snapshot[code] = (
                    acc.current_balance,
                    acc.period_debits,
                    acc.period_credits,
                    acc.version,
                )
            # Period snapshot
            pid = req.period_id
            if pid and pid in self._period_balances:
                if pid not in period_snapshot:
                    period_snapshot[pid] = {}
                if code in self._period_balances[pid] and code not in period_snapshot[pid]:
                    pacc = self._period_balances[pid][code]
                    period_snapshot[pid][code] = (
                        pacc.current_balance,
                        pacc.period_debits,
                        pacc.period_credits,
                        pacc.version,
                    )

        results: List[BalanceUpdateResult] = []
        failed = False

        for req in requests:
            result = self._apply_update_locked(req, start_ts)
            results.append(result)
            if not result.success:
                failed = True
                break

        # --- Rollback on any failure (atomicity) ---
        if failed:
            self._rollback_batch(
                snapshot,
                period_snapshot,
                results,
                pre_batch_account_codes,
                pre_batch_period_account_codes,
            )
            self._batch_rollbacks += 1
            logger.warning(
                "batch_update_rolled_back",
                service_name="transactions",
                component="AccountBalanceManager",
                batch_size=len(requests),
                failed_at_index=len(results) - 1,
            )

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
        logger.info(
            "batch_update_completed",
            service_name="transactions",
            component="AccountBalanceManager",
            batch_size=len(requests),
            success_count=sum(1 for r in results if r.success),
            failed=failed,
            duration_ms=round(elapsed_ms, 2),
        )

        return results

    def _rollback_batch(
        self,
        snapshot: Dict[str, Tuple[Decimal, Decimal, Decimal, int]],
        period_snapshot: Dict[
            str, Dict[str, Tuple[Decimal, Decimal, Decimal, int]]
        ],
        results: List[BalanceUpdateResult],
        pre_batch_account_codes: Optional[Set[str]] = None,
        pre_batch_period_account_codes: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        """Restore account balances to their pre-batch state.

        Reverts the running balance cache and period-specific balances
        to the values captured before the batch started.  Also removes
        any accounts or period entries that were newly created during
        the failed batch (i.e., not present before the batch).  Marks
        **all** results in the batch as failed.

        Args:
            snapshot: Pre-batch cumulative balance snapshots.
            period_snapshot: Pre-batch period balance snapshots.
            results: Batch results to mark as failed.
            pre_batch_account_codes: Set of account codes that existed
                before the batch.  Accounts not in this set are removed.
            pre_batch_period_account_codes: Mapping of period_id to sets
                of account codes that existed before the batch.
        """
        # Restore cumulative balances
        for code, (bal, pd, pc, ver) in snapshot.items():
            if code in self._balances:
                acc = self._balances[code]
                acc.current_balance = bal
                acc.period_debits = pd
                acc.period_credits = pc
                acc.version = ver

        # Remove accounts that were newly created during the failed batch
        if pre_batch_account_codes is not None:
            new_accounts = set(self._balances.keys()) - pre_batch_account_codes
            for orphan_code in new_accounts:
                del self._balances[orphan_code]
                logger.debug(
                    "rollback_removed_orphaned_account",
                    service_name="transactions",
                    component="AccountBalanceManager",
                    account_code=orphan_code,
                )

        # Restore period balances
        for pid, accounts in period_snapshot.items():
            if pid in self._period_balances:
                for code, (bal, pd, pc, ver) in accounts.items():
                    if code in self._period_balances[pid]:
                        pacc = self._period_balances[pid][code]
                        pacc.current_balance = bal
                        pacc.period_debits = pd
                        pacc.period_credits = pc
                        pacc.version = ver

        # Remove period-level accounts created during the failed batch
        if pre_batch_period_account_codes is not None:
            for pid, period_accounts in self._period_balances.items():
                pre_batch_codes = pre_batch_period_account_codes.get(pid, set())
                new_period_accounts = set(period_accounts.keys()) - pre_batch_codes
                for orphan_code in new_period_accounts:
                    del period_accounts[orphan_code]

        # Mark all results as failed
        for result in results:
            if result.success:
                result.success = False
                result.error_message = "Rolled back due to batch failure"
                # Undo metrics count for reverted updates
                self._updates_applied = max(0, self._updates_applied - 1)

        # Remove rolled-back entries from update history
        rollback_count = len(results)
        if rollback_count > 0 and len(self._update_history) >= rollback_count:
            self._update_history = self._update_history[:-rollback_count]

    # ------------------------------------------------------------------
    # Balance Query Methods
    # ------------------------------------------------------------------

    async def get_balance(
        self, account_code: str
    ) -> Optional[AccountBalance]:
        """Retrieve the current cumulative balance for an account.

        Args:
            account_code: The GL account code to look up.

        Returns:
            The :class:`AccountBalance` if registered, otherwise ``None``.
        """
        self._balance_checks += 1
        return self._balances.get(account_code)

    async def get_period_balance(
        self, account_code: str, period_id: str
    ) -> Optional[AccountBalance]:
        """Retrieve the balance for a specific fiscal period.

        Args:
            account_code: Target GL account code.
            period_id: Fiscal period identifier (e.g., ``"2024-01"``).

        Returns:
            The period-specific :class:`AccountBalance` if it exists,
            otherwise ``None``.
        """
        self._balance_checks += 1
        period_accounts = self._period_balances.get(period_id)
        if period_accounts is None:
            return None
        return period_accounts.get(account_code)

    async def get_all_balances(self) -> Dict[str, AccountBalance]:
        """Return a shallow copy of all cumulative account balances.

        Returns:
            Dictionary mapping account codes to :class:`AccountBalance`.
        """
        self._balance_checks += 1
        return dict(self._balances)

    # ------------------------------------------------------------------
    # Trial Balance Methods
    # ------------------------------------------------------------------

    async def calculate_trial_balance(
        self, period_id: Optional[str] = None
    ) -> TrialBalanceReport:
        """Calculate a trial balance report.

        When ``period_id`` is provided, only the period-specific debit
        and credit totals are summed.  When ``None``, cumulative totals
        across all accounts are used.

        The trial balance is considered *balanced* when:
        ``abs(total_debits - total_credits) <= GL_BALANCE_TOLERANCE``

        Args:
            period_id: Optional fiscal period to scope the report to.

        Returns:
            A :class:`TrialBalanceReport` with totals and balance status.
        """
        self._balance_checks += 1

        total_debits = Decimal("0.00")
        total_credits = Decimal("0.00")
        account_count = 0

        if period_id is not None:
            # Period-specific trial balance
            period_accounts = self._period_balances.get(period_id, {})
            for acc in period_accounts.values():
                total_debits += acc.period_debits
                total_credits += acc.period_credits
                account_count += 1
        else:
            # Cumulative trial balance across all accounts
            for acc in self._balances.values():
                total_debits += acc.period_debits
                total_credits += acc.period_credits
                account_count += 1

        difference = total_debits - total_credits
        is_balanced = abs(difference) <= GL_BALANCE_TOLERANCE

        report = TrialBalanceReport(
            period_id=period_id,
            total_debits=total_debits,
            total_credits=total_credits,
            difference=difference,
            is_balanced=is_balanced,
            account_count=account_count,
        )

        logger.info(
            "trial_balance_calculated",
            service_name="transactions",
            component="AccountBalanceManager",
            period_id=period_id,
            total_debits=str(total_debits),
            total_credits=str(total_credits),
            difference=str(difference),
            is_balanced=is_balanced,
            account_count=account_count,
        )

        return report

    async def validate_trial_balance(
        self, period_id: Optional[str] = None
    ) -> bool:
        """Validate that the trial balance is within tolerance.

        Shorthand for :meth:`calculate_trial_balance` returning only the
        ``is_balanced`` flag.

        Args:
            period_id: Optional fiscal period to validate.

        Returns:
            ``True`` if ``abs(total_debits - total_credits) <= $0.01``.
        """
        report = await self.calculate_trial_balance(period_id=period_id)
        return report.is_balanced

    # ------------------------------------------------------------------
    # Balance Sheet Methods
    # ------------------------------------------------------------------

    async def calculate_balance_sheet_equation(
        self,
    ) -> Dict[str, Decimal]:
        """Calculate the fundamental accounting equation components.

        Computes:
            ``Assets = Liabilities + Equity``

        Returns a dictionary with ``"assets"``, ``"liabilities"``,
        ``"equity"``, and ``"difference"`` (should be within $0.01 of
        zero for a balanced set of books).

        Returns:
            Dict with Decimal values for each component.
        """
        self._balance_checks += 1

        assets = Decimal("0.00")
        liabilities = Decimal("0.00")
        equity = Decimal("0.00")

        for acc in self._balances.values():
            if acc.account_type == "asset":
                assets += acc.current_balance
            elif acc.account_type == "liability":
                liabilities += acc.current_balance
            elif acc.account_type == "equity":
                equity += acc.current_balance
            # Revenue and expense are income-statement accounts, excluded
            # from the balance sheet equation directly.

        difference = assets - (liabilities + equity)

        logger.info(
            "balance_sheet_equation_calculated",
            service_name="transactions",
            component="AccountBalanceManager",
            assets=str(assets),
            liabilities=str(liabilities),
            equity=str(equity),
            difference=str(difference),
        )

        return {
            "assets": assets,
            "liabilities": liabilities,
            "equity": equity,
            "difference": difference,
        }

    async def validate_balance_sheet(self) -> bool:
        """Validate the balance sheet equation within tolerance.

        Checks that ``abs(Assets - (Liabilities + Equity)) <= $0.01``.

        Returns:
            ``True`` if the equation holds within
            ``BALANCE_SHEET_TOLERANCE``.
        """
        equation = await self.calculate_balance_sheet_equation()
        is_valid = abs(equation["difference"]) <= BALANCE_SHEET_TOLERANCE

        if not is_valid:
            logger.warning(
                "balance_sheet_out_of_tolerance",
                service_name="transactions",
                component="AccountBalanceManager",
                difference=str(equation["difference"]),
                tolerance=str(BALANCE_SHEET_TOLERANCE),
            )

        return is_valid

    # ------------------------------------------------------------------
    # Sub-Ledger Reconciliation
    # ------------------------------------------------------------------

    async def reconcile_sub_ledger(
        self,
        sub_ledger_name: str,
        sub_ledger_balance: Decimal,
        gl_control_account: str,
    ) -> Dict[str, Any]:
        """Reconcile a sub-ledger against its GL control account.

        Compares the provided ``sub_ledger_balance`` (from an external
        sub-ledger like AP or AR) against the current balance of the
        specified GL control account.  The reconciliation is considered
        successful when the absolute difference is within
        ``SUB_LEDGER_TOLERANCE`` ($0.01).

        Args:
            sub_ledger_name: Descriptive name (e.g., ``"AP"``, ``"AR"``).
            sub_ledger_balance: Total from the sub-ledger.
            gl_control_account: GL account code for the control account.

        Returns:
            Dictionary with reconciliation details:
            ``sub_ledger_name``, ``sub_ledger_balance``,
            ``gl_control_balance``, ``difference``, ``is_reconciled``,
            ``tolerance``.
        """
        self._balance_checks += 1

        gl_account = self._balances.get(gl_control_account)
        gl_control_balance = (
            gl_account.current_balance if gl_account else Decimal("0.00")
        )

        difference = abs(sub_ledger_balance - gl_control_balance)
        is_reconciled = difference <= SUB_LEDGER_TOLERANCE

        reconciliation = {
            "sub_ledger_name": sub_ledger_name,
            "sub_ledger_balance": sub_ledger_balance,
            "gl_control_balance": gl_control_balance,
            "difference": difference,
            "is_reconciled": is_reconciled,
            "tolerance": SUB_LEDGER_TOLERANCE,
        }

        logger.info(
            "sub_ledger_reconciled",
            service_name="transactions",
            component="AccountBalanceManager",
            sub_ledger_name=sub_ledger_name,
            sub_ledger_balance=str(sub_ledger_balance),
            gl_control_balance=str(gl_control_balance),
            difference=str(difference),
            is_reconciled=is_reconciled,
        )

        return reconciliation

    # ------------------------------------------------------------------
    # Account Management
    # ------------------------------------------------------------------

    def register_account(
        self,
        account_code: str,
        account_name: str,
        account_type: str,
    ) -> AccountBalance:
        """Register a new GL account in the balance manager.

        Determines the ``normal_balance_side`` from the ``account_type``
        and creates an :class:`AccountBalance` with a zero starting
        balance.

        If the account is already registered, the existing balance is
        returned without modification.

        Args:
            account_code: GL account code.
            account_name: Human-readable account name.
            account_type: One of ``"asset"``, ``"liability"``,
                ``"equity"``, ``"revenue"``, ``"expense"``.

        Returns:
            The newly created or existing :class:`AccountBalance`.

        Raises:
            BalanceError: If ``account_type`` is not a recognised value.
        """
        # Return existing account if already registered
        existing = self._balances.get(account_code)
        if existing is not None:
            return existing

        # Determine normal balance side from account type
        if account_type in DEBIT_NORMAL_ACCOUNT_TYPES:
            normal_side = "debit"
        elif account_type in CREDIT_NORMAL_ACCOUNT_TYPES:
            normal_side = "credit"
        else:
            raise BalanceError(
                f"Unrecognised account type '{account_type}' for "
                f"account {account_code}",
                details={
                    "account_code": account_code,
                    "account_type": account_type,
                    "valid_types": sorted(
                        DEBIT_NORMAL_ACCOUNT_TYPES | CREDIT_NORMAL_ACCOUNT_TYPES
                    ),
                },
            )

        account = AccountBalance(
            account_code=account_code,
            account_name=account_name,
            account_type=account_type,
            normal_balance_side=normal_side,
            current_balance=Decimal("0.00"),
            period_debits=Decimal("0.00"),
            period_credits=Decimal("0.00"),
        )
        self._balances[account_code] = account

        logger.info(
            "account_registered",
            service_name="transactions",
            component="AccountBalanceManager",
            account_code=account_code,
            account_name=account_name,
            account_type=account_type,
            normal_balance_side=normal_side,
        )

        return account

    def get_accounts_by_type(
        self, account_type: str
    ) -> List[AccountBalance]:
        """Retrieve all registered accounts of a specific type.

        Args:
            account_type: Filter by this type (e.g., ``"asset"``).

        Returns:
            List of :class:`AccountBalance` instances matching the type.
        """
        return [
            acc
            for acc in self._balances.values()
            if acc.account_type == account_type
        ]

    # ------------------------------------------------------------------
    # Period Management
    # ------------------------------------------------------------------

    def open_period(self, period_id: str) -> None:
        """Open a new fiscal period for balance tracking.

        Initialises the period-specific balance dictionary and marks
        the period as open.  If the period is already open, this is a
        no-op.

        Args:
            period_id: Fiscal period identifier (e.g., ``"2024-01"``).
        """
        if period_id in self._open_periods:
            logger.debug(
                "period_already_open",
                service_name="transactions",
                component="AccountBalanceManager",
                period_id=period_id,
            )
            return

        # Remove from closed if being re-opened (Controller/CFO action)
        self._closed_periods.discard(period_id)

        self._open_periods.add(period_id)
        if period_id not in self._period_balances:
            self._period_balances[period_id] = {}

        logger.info(
            "period_opened",
            service_name="transactions",
            component="AccountBalanceManager",
            period_id=period_id,
        )

    def close_period(self, period_id: str) -> None:
        """Close a fiscal period, preventing further balance updates.

        Moves the period from the open set to the closed set.  Any
        subsequent :meth:`update_balance` or :meth:`update_balances_batch`
        calls targeting this period will fail with a closed-period error.

        Args:
            period_id: Fiscal period identifier (e.g., ``"2024-01"``).
        """
        self._open_periods.discard(period_id)
        self._closed_periods.add(period_id)

        logger.info(
            "period_closed",
            service_name="transactions",
            component="AccountBalanceManager",
            period_id=period_id,
            account_count=len(
                self._period_balances.get(period_id, {})
            ),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the balance manager.

        Returns:
            Dictionary containing counters and state information:
            ``updates_applied``, ``lock_acquisitions``,
            ``lock_timeouts``, ``balance_checks``, ``batch_rollbacks``,
            ``account_count``, ``open_periods``, ``closed_periods``,
            ``update_history_size``.
        """
        return {
            "updates_applied": self._updates_applied,
            "lock_acquisitions": self._lock_acquisitions,
            "lock_timeouts": self._lock_timeouts,
            "balance_checks": self._balance_checks,
            "batch_rollbacks": self._batch_rollbacks,
            "account_count": len(self._balances),
            "open_periods": len(self._open_periods),
            "closed_periods": len(self._closed_periods),
            "update_history_size": len(self._update_history),
        }
