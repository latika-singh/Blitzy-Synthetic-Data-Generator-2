"""General Ledger Integration package — foundational posting layer for Transaction Workflows.

This package implements the **GL Integration** module of the Synthetic ERP Data
Generation Platform (Project 3: Transaction Workflows & Discrepancies).  It is
the central financial posting layer consumed by every P2P and O2C transaction
generator, and it orchestrates period-end close processing.

Core components
---------------
GLPostingEngine
    Creates journal entries, enforces balance validation
    (``SUM(debits) = SUM(credits)`` within ``Decimal("0.01")``), validates
    accounts against the Chart of Accounts (``is_posting = True``), validates
    fiscal period status (``OPEN`` only), assigns sequential entry numbers
    (``JE-YYYY-NNNN``), delegates balance updates to
    :class:`AccountBalanceManager`, and performs continuous trial balance
    verification after every posting batch.

    Pydantic V2 data models:
        - :class:`JournalEntry` — header with lines, totals, and status
        - :class:`JournalEntryLine` — individual debit/credit line
        - :class:`PostingResult` — outcome of a posting operation

AccountBalanceManager
    Maintains real-time account balances with ``SELECT FOR UPDATE``
    pessimistic locking to prevent race conditions during concurrent
    updates.  Enforces normal balance direction (Asset/Expense accounts
    increase with debits; Liability/Equity/Revenue accounts increase
    with credits).  Provides period-based balance tracking, trial balance
    calculation, balance sheet equation validation
    (``Assets = Liabilities + Equity`` within ``Decimal("0.01")``), and
    sub-ledger reconciliation (AR, AP, Inventory within ``Decimal("0.01")``).

    Pydantic V2 data models:
        - :class:`AccountBalance` — per-account, per-period balance snapshot
        - :class:`BalanceUpdateRequest` — request to update an account balance
        - :class:`BalanceUpdateResult` — outcome of a balance update
        - :class:`TrialBalanceReport` — full trial balance with account details

AccrualGenerator
    Produces period-end AP accruals (GRNI — Goods Received Not Invoiced:
    DR Expense/Asset, CR AP Accrual) and AR accruals (shipped-not-invoiced:
    DR AR, CR Revenue) using the straight-line daily method
    (``Total / Days in Period``).  Generates corresponding reversing entries
    dated the first day of the next period.

    Pydantic V2 data models:
        - :class:`AccrualEntry` — individual accrual or reversing entry
        - :class:`AccrualBatchResult` — batch outcome with totals and errors

PeriodCloseManager
    Orchestrates the **10-step fiscal period close** process:
        1. Validate all transactions posted
        2. Generate accruals (GRNI, shipped-not-invoiced)
        3. Generate deferrals
        4. Post depreciation entries
        5. Process recurring journal entries
        6. Account reconciliations (AR/AP/Inventory within ``Decimal("0.01")``)
        7. Trial balance validation
        8. Financial statement validation (``Assets = Liabilities + Equity``)
        9. Close period (transition to CLOSED)
        10. Open next period

    Subscribes to ``PeriodClosing`` events from the ``TimeController`` and
    publishes ``PeriodClosed`` events upon successful close.

    Pydantic V2 data models and enums:
        - :class:`PeriodCloseResult` — full close outcome with step details
        - :class:`PeriodCloseStep` — enum of the 10 close steps
        - :class:`ReconciliationResult` — sub-ledger vs GL reconciliation

Financial Integrity Rules (AAP §0.7.2)
---------------------------------------
This package enforces the following non-negotiable financial invariants:

    * **GL Balance**: ``SUM(debits) = SUM(credits)`` within ``Decimal("0.01")``
      for every journal entry — unbalanced entries are rejected before persistence.
    * **Continuous Trial Balance**: cumulative trial balance equals zero within
      ``Decimal("0.01")`` after every posting batch.
    * **Balance Sheet Equation**: ``Assets = Liabilities + Equity`` within
      ``Decimal("0.01")`` — validated during period close.
    * **Sub-ledger Reconciliation**: AR, AP, and Inventory sub-ledgers reconcile
      to their respective GL control accounts within ``Decimal("0.01")``.
    * **Atomicity**: if ANY step in a multi-step posting fails, the ENTIRE
      transaction is rolled back via SQLAlchemy session rollback — no partial
      postings are permitted.
    * **Normal Balance Direction**: Asset and Expense accounts increase with
      debits; Liability, Equity, and Revenue accounts increase with credits.
    * **Decimal Precision**: ``decimal.getcontext().prec = 28`` and
      ``rounding = ROUND_HALF_UP`` — ``float`` is NEVER used for monetary amounts.

Architecture (AAP §0.7.1)
--------------------------
    * **Constructor Injection (ADR-003)**: all classes receive dependencies
      (``EventBus``, ``AccountBalanceManager``, ``AccrualGenerator``, etc.)
      through constructor parameters — no service locator, no global state.
    * **Pydantic V2**: all data models crossing subsystem boundaries are
      ``BaseModel`` subclasses with ``model_dump()`` / ``model_validate()``.
    * **EventBus (ADR-001)**: cross-subsystem async notifications flow through
      the existing ``EventBus``; direct calls are permitted only for
      synchronous data flow within the same processing pipeline.
    * **structlog**: all logging uses ``structlog`` with JSON output to stdout.
    * **Database Access**: ``from synthetic_erp.db.session import get_session``
      with ``async with get_session() as session:`` for all database operations.

Performance Target (AAP §0.7.3)
-------------------------------
    * GL posting rate ≥ 200 journal entries per minute.

This package has **no import-time side effects** — no objects are instantiated,
no I/O is performed, and no heavy external dependencies are loaded at import
time.  Only class and enum references are re-exported.

References:
    - AAP §0.5.1 Group 4: GL Integration
    - AAP §0.7.2: Financial Integrity Rules
    - AAP §0.7.3: Performance Requirements
    - AAP §0.7.4: Error Handling Conventions
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GL Posting Engine — journal entry creation, balance validation, trial balance
# ---------------------------------------------------------------------------

from app.transactions.gl.gl_posting_engine import (
    GLPostingEngine,
    JournalEntry,
    JournalEntryLine,
    PostingResult,
)

# ---------------------------------------------------------------------------
# Account Balance Manager — real-time balances with pessimistic locking
# ---------------------------------------------------------------------------

from app.transactions.gl.account_balance_manager import (
    AccountBalance,
    AccountBalanceManager,
    BalanceUpdateRequest,
    BalanceUpdateResult,
    TrialBalanceReport,
)

# ---------------------------------------------------------------------------
# Accrual Generator — GRNI, shipped-not-invoiced, and reversing entries
# ---------------------------------------------------------------------------

from app.transactions.gl.accrual_generator import (
    AccrualBatchResult,
    AccrualEntry,
    AccrualGenerator,
)

# ---------------------------------------------------------------------------
# Period Close Manager — 10-step fiscal period close orchestration
# ---------------------------------------------------------------------------

from app.transactions.gl.period_close_manager import (
    PeriodCloseManager,
    PeriodCloseResult,
    PeriodCloseStep,
    ReconciliationResult,
)

# ---------------------------------------------------------------------------
# Public API — all symbols re-exported for ``from app.transactions.gl import …``
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # GL Posting Engine
    "GLPostingEngine",
    "JournalEntry",
    "JournalEntryLine",
    "PostingResult",
    # Account Balance Manager
    "AccountBalanceManager",
    "AccountBalance",
    "BalanceUpdateRequest",
    "BalanceUpdateResult",
    "TrialBalanceReport",
    # Accrual Generator
    "AccrualGenerator",
    "AccrualEntry",
    "AccrualBatchResult",
    # Period Close Manager
    "PeriodCloseManager",
    "PeriodCloseResult",
    "PeriodCloseStep",
    "ReconciliationResult",
]
