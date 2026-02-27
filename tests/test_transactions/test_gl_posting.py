"""GL posting engine and account balance management tests.

Tests cover the foundational General Ledger posting layer:
    GLPostingEngine — journal entry creation, balance validation, trial balance
    AccountBalanceManager — real-time balance maintenance with SELECT FOR UPDATE

THIS FILE CONTAINS THE 5 MANDATORY CRITICAL FINANCIAL TEST SCENARIOS (AAP 0.7.6):
1. GL Balance Zero: SUM(Debits) - SUM(Credits) == $0.00 after 1,000 mixed transactions
2. Concurrent Posting: 20 agents post to 'Cash' simultaneously; final balance = sum of inputs
3. Period Close: Dec 31 transactions posted; Jan 1 blocked until period OPEN
4. Overpayment: $1,100 payment on $1,000 invoice → $0 Invoice + $100 Unapplied Cash
5. Rollback: DB error during 'Post Line 2' causes 'Post Line 1' to disappear (atomicity)

Validation domains:
- Balance validation: SUM(debits) = SUM(credits) within $0.01 tolerance (Decimal)
- Trial balance continuity: cumulative trial balance = 0 within $0.01
- Account validation against Chart of Accounts (is_posting=TRUE)
- Period validation: reject postings to CLOSED periods, allow OPEN only
- Concurrent posting safety: SELECT FOR UPDATE locking simulation
- Rollback atomicity: failed step 2 rolls back step 1
- Normal balance direction: Asset/Expense DR increases, Liability/Equity/Revenue CR increases
- Circuit breaker: 10 failures → OPEN, 60s recovery, fallback → rollback
- Sequential numbering: JE-YYYY-NNNN

Per AAP Section 0.7.2 Financial Integrity Rules:
- GL Balance Invariant: SUM(debits) = SUM(credits) within Decimal('0.01')
- Continuous Trial Balance after every batch
- Decimal precision: prec=28, ROUND_HALF_UP — NEVER float
- Atomicity: partial posting failure → FULL rollback
- Balance Sheet: Assets = Liabilities + Equity within $0.01
- Sub-ledger reconciliation: AR/AP/Inventory vs GL within $0.01
- Normal balance direction enforcement

Per AAP Section 0.7.4 Error Handling:
- Retry: 3 attempts, exponential backoff (1s, 2s, 4s), 30s timeout, fallback → rollback
- Circuit breaker: 10 failures → OPEN state, 60s recovery

Per AAP Section 0.7.6 Testing Conventions:
- Property-based testing with hypothesis for GL balance validation
- ≥ 80% coverage, AsyncMock for DB sessions
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import date, datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import UUID, uuid4

import pytest
from hypothesis import given, settings, strategies as st, assume

from app.transactions.gl.gl_posting_engine import (
    GLPostingEngine,
    JournalEntryLine,
    JournalEntry,
    PostingResult,
    CircuitBreakerState,
    CircuitBreaker,
)
from app.transactions.gl.account_balance_manager import (
    AccountBalanceManager,
    AccountBalance,
    BalanceUpdateRequest,
    BalanceUpdateResult,
    TrialBalanceReport,
)
from app.transactions.exceptions import (
    BalanceError,
    GLPostingError,
    PeriodClosedError,
    ConcurrencyError,
)
from app.transactions.constants import (
    FINANCIAL_TOLERANCES,
    CIRCUIT_BREAKER_CONFIG,
    DEBIT_NORMAL_ACCOUNT_TYPES,
    CREDIT_NORMAL_ACCOUNT_TYPES,
)

# ---------------------------------------------------------------------------
# Global Decimal configuration (AAP §0.7.2)
# ---------------------------------------------------------------------------
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


def make_je_line(
    account_code: str,
    debit: Decimal = Decimal("0.00"),
    credit: Decimal = Decimal("0.00"),
    line_number: int = 1,
    account_name: str = "Test Account",
    description: str = "",
) -> JournalEntryLine:
    """Create a single JournalEntryLine Pydantic model instance.

    Args:
        account_code: GL account code string.
        debit: Debit amount (Decimal, ≥0).
        credit: Credit amount (Decimal, ≥0).
        line_number: 1-based line number.
        account_name: Human-readable account name.
        description: Line description text.

    Returns:
        A fully-constructed JournalEntryLine Pydantic V2 model.
    """
    return JournalEntryLine(
        line_number=line_number,
        account_code=account_code,
        account_name=account_name,
        debit_amount=debit,
        credit_amount=credit,
        description=description,
    )


def make_balanced_je(
    total: Decimal = Decimal("1000.00"),
    debit_account: str = "1010",
    credit_account: str = "2010",
    posting_date: date = date(2024, 6, 15),
    description: str = "Test balanced journal entry",
    source_document_type: str = "test",
) -> JournalEntry:
    """Create a balanced journal entry with one debit line and one credit line.

    SUM(debits) = SUM(credits) = total, satisfying GL Balance Invariant.

    Args:
        total: Total amount for both DR and CR sides (Decimal).
        debit_account: Account code for the debit line.
        credit_account: Account code for the credit line.
        posting_date: Business date for the entry.
        description: Entry-level description.
        source_document_type: Source document type tag.

    Returns:
        A balanced JournalEntry Pydantic V2 model.
    """
    lines = [
        make_je_line(
            account_code=debit_account,
            debit=total,
            credit=Decimal("0.00"),
            line_number=1,
            account_name=f"DR Account {debit_account}",
            description=f"Debit {total}",
        ),
        make_je_line(
            account_code=credit_account,
            debit=Decimal("0.00"),
            credit=total,
            line_number=2,
            account_name=f"CR Account {credit_account}",
            description=f"Credit {total}",
        ),
    ]
    return JournalEntry(
        posting_date=posting_date,
        description=description,
        lines=lines,
        total_debits=total,
        total_credits=total,
        is_balanced=True,
        source_document_type=source_document_type,
    )


def make_unbalanced_je(
    debit: Decimal = Decimal("1000.00"),
    credit: Decimal = Decimal("900.00"),
    posting_date: date = date(2024, 6, 15),
) -> JournalEntry:
    """Create an intentionally unbalanced journal entry.

    Args:
        debit: Total debit amount.
        credit: Total credit amount (different from debit to create imbalance).
        posting_date: Business date for the entry.

    Returns:
        An unbalanced JournalEntry Pydantic V2 model.
    """
    lines = [
        make_je_line(
            account_code="1010",
            debit=debit,
            credit=Decimal("0.00"),
            line_number=1,
        ),
        make_je_line(
            account_code="2010",
            debit=Decimal("0.00"),
            credit=credit,
            line_number=2,
        ),
    ]
    return JournalEntry(
        posting_date=posting_date,
        description="Unbalanced test entry",
        lines=lines,
        total_debits=debit,
        total_credits=credit,
        is_balanced=False,
    )


def make_multi_line_balanced_je(
    debit_lines: List[Decimal],
    credit_lines: List[Decimal],
    posting_date: date = date(2024, 6, 15),
) -> JournalEntry:
    """Create a multi-line balanced journal entry.

    Asserts at construction that SUM(debit_lines) == SUM(credit_lines).

    Args:
        debit_lines: List of debit amounts.
        credit_lines: List of credit amounts.
        posting_date: Business date.

    Returns:
        A multi-line balanced JournalEntry.
    """
    lines: List[JournalEntryLine] = []
    line_num = 1
    for amount in debit_lines:
        lines.append(
            make_je_line(
                account_code=f"1{line_num:03d}",
                debit=amount,
                credit=Decimal("0.00"),
                line_number=line_num,
                account_name=f"Debit Account {line_num}",
            )
        )
        line_num += 1
    for amount in credit_lines:
        lines.append(
            make_je_line(
                account_code=f"2{line_num:03d}",
                debit=Decimal("0.00"),
                credit=amount,
                line_number=line_num,
                account_name=f"Credit Account {line_num}",
            )
        )
        line_num += 1

    total_dr = sum(debit_lines, Decimal("0.00"))
    total_cr = sum(credit_lines, Decimal("0.00"))

    return JournalEntry(
        posting_date=posting_date,
        description="Multi-line balanced entry",
        lines=lines,
        total_debits=total_dr,
        total_credits=total_cr,
        is_balanced=abs(total_dr - total_cr) <= Decimal("0.01"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Local Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_balance_manager() -> AsyncMock:
    """Mocked AccountBalanceManager with pre-configured responses.

    Returns an AsyncMock that simulates successful balance updates,
    balanced trial balance reports, and zero-balance responses.
    """
    manager = AsyncMock(spec=AccountBalanceManager)
    manager.update_balance = AsyncMock(
        return_value=BalanceUpdateResult(
            account_code="1010",
            previous_balance=Decimal("0.00"),
            new_balance=Decimal("1000.00"),
            debit_applied=Decimal("1000.00"),
            credit_applied=Decimal("0.00"),
            success=True,
            error_message=None,
        )
    )
    manager.get_balance = AsyncMock(
        return_value=AccountBalance(
            account_code="1010",
            account_name="Cash",
            account_type="asset",
            normal_balance_side="debit",
            current_balance=Decimal("10000.00"),
            period_debits=Decimal("10000.00"),
            period_credits=Decimal("0.00"),
        )
    )
    manager.calculate_trial_balance = AsyncMock(
        return_value=TrialBalanceReport(
            total_debits=Decimal("0.00"),
            total_credits=Decimal("0.00"),
            difference=Decimal("0.00"),
            is_balanced=True,
            account_count=0,
        )
    )
    manager.validate_trial_balance = AsyncMock(return_value=True)
    manager.calculate_balance_sheet_equation = AsyncMock(
        return_value={
            "assets": Decimal("0.00"),
            "liabilities": Decimal("0.00"),
            "equity": Decimal("0.00"),
            "difference": Decimal("0.00"),
        }
    )
    return manager


@pytest.fixture
def mock_fiscal_calendar() -> MagicMock:
    """Mocked FiscalCalendar with period state control."""
    cal = MagicMock()
    open_period = MagicMock()
    open_period.status = "open"
    open_period.fiscal_year = 2024
    open_period.period_number = 6
    open_period.start_date = date(2024, 6, 1)
    open_period.end_date = date(2024, 6, 30)
    cal.get_period_for_date = MagicMock(return_value=open_period)
    return cal


@pytest.fixture
def sample_chart_of_accounts() -> Dict[str, Dict[str, Any]]:
    """Sample Chart of Accounts for account validation tests."""
    return {
        "1010": {"name": "Cash", "account_type": "asset", "is_posting": True},
        "1100": {"name": "Accounts Receivable", "account_type": "asset", "is_posting": True},
        "1200": {"name": "Inventory", "account_type": "asset", "is_posting": True},
        "1500": {"name": "Fixed Assets", "account_type": "asset", "is_posting": True},
        "2010": {"name": "Accounts Payable", "account_type": "liability", "is_posting": True},
        "2100": {"name": "Accrued Liabilities", "account_type": "liability", "is_posting": True},
        "2500": {"name": "Unapplied Cash", "account_type": "liability", "is_posting": True},
        "3010": {"name": "Retained Earnings", "account_type": "equity", "is_posting": True},
        "4010": {"name": "Revenue", "account_type": "revenue", "is_posting": True},
        "5010": {"name": "Cost of Goods Sold", "account_type": "expense", "is_posting": True},
        "6100": {"name": "Office Supplies Expense", "account_type": "expense", "is_posting": True},
        "9000": {"name": "Total Assets (Summary)", "account_type": "asset", "is_posting": False},
    }


@pytest.fixture
def gl_engine(
    mock_balance_manager: AsyncMock,
    event_bus,
    mock_fiscal_calendar: MagicMock,
    sample_chart_of_accounts: Dict[str, Dict[str, Any]],
) -> GLPostingEngine:
    """Fully-wired GLPostingEngine for testing."""
    return GLPostingEngine(
        account_balance_manager=mock_balance_manager,
        event_bus=event_bus,
        fiscal_calendar=mock_fiscal_calendar,
        chart_of_accounts=sample_chart_of_accounts,
    )


@pytest.fixture
def gl_engine_no_deps() -> GLPostingEngine:
    """GLPostingEngine with NO dependencies (all None)."""
    return GLPostingEngine()


@pytest.fixture
def balance_manager() -> AccountBalanceManager:
    """Real AccountBalanceManager for unit testing (in-memory mode)."""
    return AccountBalanceManager()


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: GLPostingEngine Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestGLPostingEngineConstruction:
    """Tests for GLPostingEngine constructor injection per ADR-003."""

    def test_engine_instantiation_no_args(self) -> None:
        """GLPostingEngine can be created with no arguments."""
        engine = GLPostingEngine()
        assert engine is not None
        assert engine._account_balance_manager is None
        assert engine._fiscal_calendar is None
        assert engine._event_bus is None
        assert engine._db_session_factory is None
        assert engine._chart_of_accounts is None

    def test_engine_instantiation_with_all_deps(
        self,
        mock_balance_manager: AsyncMock,
        event_bus,
        mock_fiscal_calendar: MagicMock,
        sample_chart_of_accounts: Dict[str, Dict[str, Any]],
    ) -> None:
        """GLPostingEngine accepts all dependencies via constructor injection."""
        engine = GLPostingEngine(
            account_balance_manager=mock_balance_manager,
            event_bus=event_bus,
            fiscal_calendar=mock_fiscal_calendar,
            chart_of_accounts=sample_chart_of_accounts,
        )
        assert engine._account_balance_manager is mock_balance_manager
        assert engine._fiscal_calendar is mock_fiscal_calendar
        assert engine._event_bus is event_bus
        assert engine._chart_of_accounts is sample_chart_of_accounts

    def test_engine_initial_circuit_breaker_closed(self) -> None:
        """Circuit breaker starts in CLOSED state."""
        engine = GLPostingEngine()
        assert engine._circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_engine_initial_metrics_zero(self) -> None:
        """All metrics counters start at zero."""
        engine = GLPostingEngine()
        metrics = engine.get_metrics()
        assert metrics["entries_posted"] == 0
        assert metrics["entries_rejected"] == 0

    def test_engine_initial_cumulative_totals_zero(self) -> None:
        """Cumulative debit/credit totals start at Decimal('0.00')."""
        engine = GLPostingEngine()
        assert engine._cumulative_debits == Decimal("0.00")
        assert engine._cumulative_credits == Decimal("0.00")


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Balance Validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestBalanceValidation:
    """Balance validation: SUM(debits) = SUM(credits) within Decimal('0.01')."""

    def test_balanced_entry_passes_validation(self) -> None:
        """DR $1000 = CR $1000 → True."""
        engine = GLPostingEngine()
        entry = make_balanced_je(total=Decimal("1000.00"))
        engine._validate_balance(entry)
        assert entry.is_balanced is True

    def test_balanced_entry_within_tolerance(self) -> None:
        """DR $1000.00 vs CR $1000.009 → True (within $0.01)."""
        engine = GLPostingEngine()
        lines = [
            make_je_line("1010", debit=Decimal("1000.00"), line_number=1),
            make_je_line("2010", credit=Decimal("1000.009"), line_number=2),
        ]
        entry = JournalEntry(posting_date=date(2024, 6, 15), lines=lines)
        engine._validate_balance(entry)
        assert entry.is_balanced is True

    def test_unbalanced_entry_fails_validation(self) -> None:
        """DR $1000 != CR $999 → BalanceError."""
        engine = GLPostingEngine()
        entry = make_unbalanced_je(debit=Decimal("1000.00"), credit=Decimal("999.00"))
        with pytest.raises(BalanceError):
            engine._validate_balance(entry)

    def test_unbalanced_by_two_cents_fails(self) -> None:
        """DR $1000.00 vs CR $999.98 → BalanceError ($0.02 > $0.01)."""
        engine = GLPostingEngine()
        lines = [
            make_je_line("1010", debit=Decimal("1000.00"), line_number=1),
            make_je_line("2010", credit=Decimal("999.98"), line_number=2),
        ]
        entry = JournalEntry(posting_date=date(2024, 6, 15), lines=lines)
        with pytest.raises(BalanceError):
            engine._validate_balance(entry)

    def test_multi_line_balance_validation(self) -> None:
        """5 DR lines + 3 CR lines with matching sums → passes."""
        dr = [Decimal("200.00"), Decimal("150.00"), Decimal("100.00"), Decimal("300.00"), Decimal("250.00")]
        cr = [Decimal("400.00"), Decimal("350.00"), Decimal("250.00")]
        engine = GLPostingEngine()
        entry = make_multi_line_balanced_je(dr, cr)
        engine._validate_balance(entry)
        assert entry.is_balanced is True

    def test_zero_amount_journal_entry(self) -> None:
        """DR $0 = CR $0 → valid."""
        engine = GLPostingEngine()
        entry = make_balanced_je(total=Decimal("0.00"))
        engine._validate_balance(entry)
        assert entry.is_balanced is True

    def test_tolerance_uses_financial_tolerances_constant(self) -> None:
        """GL balance tolerance matches FINANCIAL_TOLERANCES."""
        assert FINANCIAL_TOLERANCES["gl_balance"] == Decimal("0.01")


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Account Validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestAccountValidation:
    """Account validation: is_posting=TRUE required."""

    def test_valid_posting_account_accepted(self, sample_chart_of_accounts: Dict) -> None:
        engine = GLPostingEngine(chart_of_accounts=sample_chart_of_accounts)
        entry = make_balanced_je(debit_account="1010", credit_account="2010")
        engine._validate_accounts(entry)

    def test_non_posting_account_rejected(self, sample_chart_of_accounts: Dict) -> None:
        engine = GLPostingEngine(chart_of_accounts=sample_chart_of_accounts)
        entry = make_balanced_je(debit_account="9000", credit_account="2010")
        with pytest.raises(GLPostingError):
            engine._validate_accounts(entry)

    def test_invalid_account_code_rejected(self, sample_chart_of_accounts: Dict) -> None:
        engine = GLPostingEngine(chart_of_accounts=sample_chart_of_accounts)
        entry = make_balanced_je(debit_account="9999", credit_account="2010")
        with pytest.raises(GLPostingError):
            engine._validate_accounts(entry)

    def test_account_validation_checks_all_lines(self, sample_chart_of_accounts: Dict) -> None:
        engine = GLPostingEngine(chart_of_accounts=sample_chart_of_accounts)
        lines = [
            make_je_line("1010", debit=Decimal("500.00"), line_number=1),
            make_je_line("INVALID", credit=Decimal("500.00"), line_number=2),
        ]
        entry = JournalEntry(posting_date=date(2024, 6, 15), lines=lines)
        with pytest.raises(GLPostingError):
            engine._validate_accounts(entry)

    def test_account_validation_skipped_when_no_coa(self) -> None:
        engine = GLPostingEngine()
        entry = make_balanced_je(debit_account="ANYTHING", credit_account="ELSE")
        engine._validate_accounts(entry)


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Period Validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestPeriodValidation:
    """Period validation: OPEN only."""

    def test_posting_to_open_period_succeeds(self, mock_fiscal_calendar: MagicMock) -> None:
        engine = GLPostingEngine(fiscal_calendar=mock_fiscal_calendar)
        entry = make_balanced_je(posting_date=date(2024, 6, 15))
        engine._validate_period(entry)
        assert entry.period_id is not None

    def test_posting_to_closed_period_raises_period_closed_error(self, mock_fiscal_calendar: MagicMock) -> None:
        closed = MagicMock()
        closed.status = "closed"
        closed.fiscal_year = 2024
        closed.period_number = 5
        mock_fiscal_calendar.get_period_for_date.return_value = closed
        engine = GLPostingEngine(fiscal_calendar=mock_fiscal_calendar)
        entry = make_balanced_je(posting_date=date(2024, 5, 15))
        with pytest.raises(PeriodClosedError):
            engine._validate_period(entry)

    def test_posting_to_closing_period_raises_error(self, mock_fiscal_calendar: MagicMock) -> None:
        closing = MagicMock()
        closing.status = "closing"
        closing.fiscal_year = 2024
        closing.period_number = 6
        mock_fiscal_calendar.get_period_for_date.return_value = closing
        engine = GLPostingEngine(fiscal_calendar=mock_fiscal_calendar)
        entry = make_balanced_je(posting_date=date(2024, 6, 30))
        with pytest.raises(PeriodClosedError):
            engine._validate_period(entry)

    def test_period_validation_skipped_when_no_calendar(self) -> None:
        engine = GLPostingEngine()
        entry = make_balanced_je()
        engine._validate_period(entry)

    def test_no_period_found_raises_gl_posting_error(self, mock_fiscal_calendar: MagicMock) -> None:
        mock_fiscal_calendar.get_period_for_date.return_value = None
        engine = GLPostingEngine(fiscal_calendar=mock_fiscal_calendar)
        entry = make_balanced_je(posting_date=date(2099, 1, 1))
        with pytest.raises(GLPostingError):
            engine._validate_period(entry)


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Sequential Numbering
# ═══════════════════════════════════════════════════════════════════════════


class TestSequentialNumbering:
    """Sequential entry numbering: JE-YYYY-NNNN."""

    def test_je_numbering_format(self) -> None:
        engine = GLPostingEngine()
        assert engine._generate_entry_number(date(2024, 3, 15)) == "JE-2024-0001"

    def test_je_numbers_increment_sequentially(self) -> None:
        engine = GLPostingEngine()
        nums = [engine._generate_entry_number(date(2024, 6, 15)) for _ in range(5)]
        assert nums == ["JE-2024-0001", "JE-2024-0002", "JE-2024-0003", "JE-2024-0004", "JE-2024-0005"]

    def test_je_numbering_year_rollover(self) -> None:
        engine = GLPostingEngine()
        engine._generate_entry_number(date(2024, 12, 31))
        engine._generate_entry_number(date(2024, 12, 31))
        assert engine._generate_entry_number(date(2025, 1, 1)) == "JE-2025-0001"


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Normal Balance Direction
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestNormalBalanceDirection:
    """Normal balance direction enforcement."""

    @pytest.mark.asyncio
    async def test_asset_account_dr_increases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("1010", "Cash", "asset")
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("500.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("500.00")

    @pytest.mark.asyncio
    async def test_asset_account_cr_decreases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("1010", "Cash", "asset")
        await mgr.update_balance(BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("1000.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("0.00"), credit_amount=Decimal("300.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("700.00")

    @pytest.mark.asyncio
    async def test_expense_account_dr_increases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("6100", "Office Supplies", "expense")
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="6100", debit_amount=Decimal("250.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("250.00")

    @pytest.mark.asyncio
    async def test_liability_account_cr_increases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("2010", "AP", "liability")
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="2010", debit_amount=Decimal("0.00"), credit_amount=Decimal("750.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("750.00")

    @pytest.mark.asyncio
    async def test_liability_account_dr_decreases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("2010", "AP", "liability")
        await mgr.update_balance(BalanceUpdateRequest(account_code="2010", debit_amount=Decimal("0.00"), credit_amount=Decimal("1000.00"), posting_date=date(2024, 6, 15)))
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="2010", debit_amount=Decimal("400.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("600.00")

    @pytest.mark.asyncio
    async def test_equity_account_cr_increases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("3010", "Retained Earnings", "equity")
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="3010", debit_amount=Decimal("0.00"), credit_amount=Decimal("5000.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("5000.00")

    @pytest.mark.asyncio
    async def test_revenue_account_cr_increases_balance(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("4010", "Sales Revenue", "revenue")
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="4010", debit_amount=Decimal("0.00"), credit_amount=Decimal("3000.00"), posting_date=date(2024, 6, 15)))
        assert result.success and result.new_balance == Decimal("3000.00")

    def test_debit_normal_account_types_constant(self) -> None:
        assert "asset" in DEBIT_NORMAL_ACCOUNT_TYPES
        assert "expense" in DEBIT_NORMAL_ACCOUNT_TYPES

    def test_credit_normal_account_types_constant(self) -> None:
        assert "liability" in CREDIT_NORMAL_ACCOUNT_TYPES
        assert "equity" in CREDIT_NORMAL_ACCOUNT_TYPES
        assert "revenue" in CREDIT_NORMAL_ACCOUNT_TYPES


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestCircuitBreaker:
    """Circuit breaker: 10 failures → OPEN, 60s recovery."""

    def test_circuit_breaker_starts_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_seconds=60.0)
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_opens_after_10_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_seconds=60.0)
        for _ in range(10):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_rejects_when_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_seconds=60.0)
        for _ in range(10):
            cb.record_failure()
        assert cb.can_execute() is False

    def test_circuit_breaker_transitions_to_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_seconds=0.01)
        for _ in range(10):
            cb.record_failure()
        time.sleep(0.02)
        assert cb.can_execute() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_closes_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_seconds=0.01)
        for _ in range(10):
            cb.record_failure()
        time.sleep(0.02)
        cb.can_execute()
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_reopens_on_failure_in_half_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=10, recovery_timeout_seconds=0.01)
        for _ in range(10):
            cb.record_failure()
        time.sleep(0.02)
        cb.can_execute()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_config_matches_constants(self) -> None:
        gl_config = CIRCUIT_BREAKER_CONFIG["gl_posting"]
        assert gl_config["failure_threshold"] == 10
        assert gl_config["recovery_timeout_seconds"] == 60

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback_is_rollback(self, gl_engine: GLPostingEngine) -> None:
        for _ in range(10):
            gl_engine._circuit_breaker.record_failure()
        entry = make_balanced_je()
        with pytest.raises(GLPostingError):
            await gl_engine.post_journal_entry(entry)


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Trial Balance
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestTrialBalance:
    """Trial balance: cumulative DR = CR within $0.01."""

    @pytest.mark.asyncio
    async def test_trial_balance_zero_after_balanced_postings(self, gl_engine: GLPostingEngine) -> None:
        entry = make_balanced_je(total=Decimal("5000.00"), debit_account="1010", credit_account="2010")
        result = await gl_engine.post_journal_entry(entry)
        assert result.success is True
        assert abs(gl_engine.get_trial_balance()) <= Decimal("0.01")

    @pytest.mark.asyncio
    async def test_trial_balance_report_model(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("1010", "Cash", "asset")
        mgr.register_account("2010", "AP", "liability")
        await mgr.update_balance(BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("500.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        await mgr.update_balance(BalanceUpdateRequest(account_code="2010", debit_amount=Decimal("0.00"), credit_amount=Decimal("500.00"), posting_date=date(2024, 6, 15)))
        report = await mgr.calculate_trial_balance()
        assert isinstance(report, TrialBalanceReport)
        assert report.is_balanced is True
        assert report.difference == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_continuous_trial_balance_check_after_each_batch(self, gl_engine: GLPostingEngine) -> None:
        for i in range(5):
            amount = Decimal(str((i + 1) * 100))
            entry = make_balanced_je(total=amount, debit_account="1010", credit_account="2010")
            result = await gl_engine.post_journal_entry(entry)
            assert result.success is True
            assert abs(gl_engine.get_trial_balance()) <= Decimal("0.01")


# ═══════════════════════════════════════════════════════════════════════════
# Test Class: Pydantic V2 Models
# ═══════════════════════════════════════════════════════════════════════════


class TestPydanticModels:
    """Pydantic V2 model validation for GL data contracts."""

    def test_journal_entry_line_model_validation(self) -> None:
        line = JournalEntryLine(line_number=1, account_code="1010", debit_amount=Decimal("500.00"), credit_amount=Decimal("0.00"))
        assert line.account_code == "1010"

    def test_journal_entry_model_validation(self) -> None:
        entry = make_balanced_je()
        assert isinstance(entry.entry_id, UUID)
        assert len(entry.lines) == 2

    def test_posting_result_model_validation(self) -> None:
        result = PostingResult(entry_id=uuid4(), entry_number="JE-2024-0001", success=True, posted_at=datetime.now(timezone.utc), trial_balance_after=Decimal("0.00"))
        assert result.success is True

    def test_account_balance_model_validation(self) -> None:
        balance = AccountBalance(account_code="1010", account_name="Cash", account_type="asset", current_balance=Decimal("10000.00"))
        assert balance.account_code == "1010"

    def test_trial_balance_report_model_validation(self) -> None:
        report = TrialBalanceReport(total_debits=Decimal("5000.00"), total_credits=Decimal("5000.00"), difference=Decimal("0.00"), is_balanced=True)
        assert report.is_balanced is True

    def test_balance_update_request_model(self) -> None:
        req = BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("100.00"), posting_date=date(2024, 6, 15))
        assert req.account_code == "1010"

    def test_balance_update_result_model(self) -> None:
        res = BalanceUpdateResult(account_code="1010", new_balance=Decimal("100.00"), success=True)
        assert res.success is True


# ═══════════════════════════════════════════════════════════════════════════
# MANDATORY CRITICAL FINANCIAL TEST SCENARIOS (AAP §0.7.6)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestCriticalFinancialScenarios:
    """5 MANDATORY Critical Financial Test Scenarios per AAP Section 0.7.6."""

    @pytest.mark.asyncio
    async def test_gl_balance_zero_after_1000_mixed_transactions(
        self, mock_balance_manager: AsyncMock, event_bus, mock_fiscal_calendar: MagicMock, sample_chart_of_accounts: Dict,
    ) -> None:
        """MANDATORY SCENARIO 1: SUM(Debits) - SUM(Credits) == $0.00 after 1,000 mixed txns."""
        engine = GLPostingEngine(
            account_balance_manager=mock_balance_manager, event_bus=event_bus,
            fiscal_calendar=mock_fiscal_calendar, chart_of_accounts=sample_chart_of_accounts,
        )
        rng = random.Random(42)
        cumulative_debits = Decimal("0.00")
        cumulative_credits = Decimal("0.00")
        tx_types = ["purchase_order", "vendor_invoice", "sales_order", "customer_payment", "journal_entry"]
        dr_accounts = ["1010", "1100", "1200", "5010", "6100"]
        cr_accounts = ["2010", "2100", "4010", "3010", "2010"]

        for i in range(1000):
            amount_cents = rng.randint(100, 9999999)
            amount = Decimal(str(amount_cents)) / Decimal("100")
            entry = make_balanced_je(
                total=amount, debit_account=rng.choice(dr_accounts), credit_account=rng.choice(cr_accounts),
                posting_date=date(2024, 6, 15), description=f"Txn #{i+1}", source_document_type=rng.choice(tx_types),
            )
            result = await engine.post_journal_entry(entry)
            assert result.success is True
            cumulative_debits += amount
            cumulative_credits += amount

        assert cumulative_debits - cumulative_credits == Decimal("0.00")
        assert abs(engine.get_trial_balance()) <= Decimal("0.01")
        assert engine.get_metrics()["entries_posted"] == 1000

    @pytest.mark.asyncio
    async def test_concurrent_posting_20_agents_cash_account(
        self, event_bus, mock_fiscal_calendar: MagicMock, sample_chart_of_accounts: Dict,
    ) -> None:
        """MANDATORY SCENARIO 2: 20 agents post to 'Cash'; final balance = sum of inputs."""
        real_mgr = AccountBalanceManager()
        real_mgr.register_account("1010", "Cash", "asset")
        real_mgr.register_account("4010", "Revenue", "revenue")
        engine = GLPostingEngine(
            account_balance_manager=real_mgr, event_bus=event_bus,
            fiscal_calendar=mock_fiscal_calendar, chart_of_accounts=sample_chart_of_accounts,
        )
        amounts = [Decimal(str(100 * (i + 1))) for i in range(20)]
        expected = sum(amounts, Decimal("0.00"))

        async def post_task(idx: int) -> PostingResult:
            entry = make_balanced_je(total=amounts[idx], debit_account="1010", credit_account="4010", posting_date=date(2024, 6, 15))
            return await engine.post_journal_entry(entry)

        results = await asyncio.gather(*[post_task(i) for i in range(20)])
        for r in results:
            assert r.success is True

        cash = await real_mgr.get_balance("1010")
        assert cash is not None
        assert cash.current_balance == expected

    @pytest.mark.asyncio
    async def test_period_close_dec31_posted_jan1_blocked(
        self, mock_balance_manager: AsyncMock, event_bus, sample_chart_of_accounts: Dict,
    ) -> None:
        """MANDATORY SCENARIO 3: Dec 31 posted; Jan 1 blocked until OPEN."""
        fiscal_cal = MagicMock()
        dec_period = MagicMock()
        dec_period.status = "open"
        dec_period.fiscal_year = 2024
        dec_period.period_number = 12

        jan_closed = MagicMock()
        jan_closed.status = "closed"
        jan_closed.fiscal_year = 2025
        jan_closed.period_number = 1

        jan_open = MagicMock()
        jan_open.status = "open"
        jan_open.fiscal_year = 2025
        jan_open.period_number = 1

        fiscal_cal.get_period_for_date = MagicMock(return_value=dec_period)
        engine = GLPostingEngine(
            account_balance_manager=mock_balance_manager, event_bus=event_bus,
            fiscal_calendar=fiscal_cal, chart_of_accounts=sample_chart_of_accounts,
        )

        # Dec 31 → success
        dec_entry = make_balanced_je(total=Decimal("1000.00"), debit_account="6100", credit_account="2010", posting_date=date(2024, 12, 31))
        assert (await engine.post_journal_entry(dec_entry)).success is True

        # Jan 1 → CLOSED → PeriodClosedError
        fiscal_cal.get_period_for_date.return_value = jan_closed
        jan_entry = make_balanced_je(total=Decimal("500.00"), debit_account="1010", credit_account="4010", posting_date=date(2025, 1, 1))
        with pytest.raises(PeriodClosedError):
            await engine.post_journal_entry(jan_entry)

        # Open January → success
        fiscal_cal.get_period_for_date.return_value = jan_open
        jan_entry2 = make_balanced_je(total=Decimal("500.00"), debit_account="1010", credit_account="4010", posting_date=date(2025, 1, 1))
        assert (await engine.post_journal_entry(jan_entry2)).success is True

    @pytest.mark.asyncio
    async def test_overpayment_1100_on_1000_invoice(
        self, event_bus, mock_fiscal_calendar: MagicMock, sample_chart_of_accounts: Dict,
    ) -> None:
        """MANDATORY SCENARIO 4: $1,100 on $1,000 → $0 Invoice + $100 Unapplied Cash."""
        real_mgr = AccountBalanceManager()
        real_mgr.register_account("1010", "Cash", "asset")
        real_mgr.register_account("1100", "Accounts Receivable", "asset")
        real_mgr.register_account("2500", "Unapplied Cash", "liability")
        real_mgr.register_account("4010", "Revenue", "revenue")
        engine = GLPostingEngine(
            account_balance_manager=real_mgr, event_bus=event_bus,
            fiscal_calendar=mock_fiscal_calendar, chart_of_accounts=sample_chart_of_accounts,
        )

        # Invoice: DR AR $1000, CR Revenue $1000
        inv = JournalEntry(posting_date=date(2024, 6, 1), description="Invoice", lines=[
            make_je_line("1100", debit=Decimal("1000.00"), line_number=1),
            make_je_line("4010", credit=Decimal("1000.00"), line_number=2),
        ])
        assert (await engine.post_journal_entry(inv)).success is True
        ar = await real_mgr.get_balance("1100")
        assert ar.current_balance == Decimal("1000.00")

        # Overpayment: DR Cash $1100, CR AR $1000, CR Unapplied $100
        pmt = JournalEntry(posting_date=date(2024, 6, 15), description="Payment w/ overpay", lines=[
            make_je_line("1010", debit=Decimal("1100.00"), line_number=1),
            make_je_line("1100", credit=Decimal("1000.00"), line_number=2),
            make_je_line("2500", credit=Decimal("100.00"), line_number=3),
        ])
        assert (await engine.post_journal_entry(pmt)).success is True

        ar_after = await real_mgr.get_balance("1100")
        assert ar_after.current_balance == Decimal("0.00")
        assert ar_after.current_balance >= Decimal("0")
        unapplied = await real_mgr.get_balance("2500")
        assert unapplied.current_balance == Decimal("100.00")
        cash = await real_mgr.get_balance("1010")
        assert cash.current_balance == Decimal("1100.00")

    @pytest.mark.asyncio
    async def test_rollback_db_error_line2_rolls_back_line1(
        self, mock_balance_manager: AsyncMock, event_bus, mock_fiscal_calendar: MagicMock, sample_chart_of_accounts: Dict,
    ) -> None:
        """MANDATORY SCENARIO 5: DB error Line 2 → Line 1 disappears (atomicity)."""
        call_count = 0

        async def fail_on_second(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return BalanceUpdateResult(account_code=request.account_code, previous_balance=Decimal("0.00"), new_balance=Decimal("1000.00"), debit_applied=request.debit_amount, credit_applied=request.credit_amount, success=True)
            raise GLPostingError("DB error on line 2", details={"account_code": request.account_code})

        mock_balance_manager.update_balance = AsyncMock(side_effect=fail_on_second)
        engine = GLPostingEngine(
            account_balance_manager=mock_balance_manager, event_bus=event_bus,
            fiscal_calendar=mock_fiscal_calendar, chart_of_accounts=sample_chart_of_accounts,
        )
        entry = make_balanced_je(total=Decimal("1000.00"), debit_account="1010", credit_account="2010", posting_date=date(2024, 6, 15))

        with pytest.raises((GLPostingError, ConcurrencyError)):
            await engine.post_journal_entry(entry)

        assert entry.status == "failed"
        assert len([e for e in engine.get_posted_entries() if e.entry_id == entry.entry_id]) == 0
        assert engine.get_metrics()["entries_posted"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Property-Based Tests with Hypothesis
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestGLPostingPropertyBased:
    """Property-based testing with hypothesis for GL balance invariants."""

    @given(amount=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("999999.99"), allow_nan=False, allow_infinity=False, places=2))
    @settings(max_examples=200, deadline=None)
    def test_any_balanced_je_passes_validation(self, amount: Decimal) -> None:
        """Property: balanced JE always passes validation."""
        engine = GLPostingEngine()
        entry = make_balanced_je(total=amount)
        engine._validate_balance(entry)
        assert entry.is_balanced is True

    @given(
        debit=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("999999.99"), allow_nan=False, allow_infinity=False, places=2),
        credit=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("999999.99"), allow_nan=False, allow_infinity=False, places=2),
    )
    @settings(max_examples=200, deadline=None)
    def test_unbalanced_je_always_rejected(self, debit: Decimal, credit: Decimal) -> None:
        """Property: unbalanced JE with |DR-CR| > $0.01 always fails."""
        assume(abs(debit - credit) > Decimal("0.01"))
        engine = GLPostingEngine()
        entry = make_unbalanced_je(debit=debit, credit=credit)
        with pytest.raises(BalanceError):
            engine._validate_balance(entry)

    @given(amount=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("999999.99"), allow_nan=False, allow_infinity=False, places=2))
    @settings(max_examples=100, deadline=None)
    def test_balanced_je_total_debits_equals_total_credits(self, amount: Decimal) -> None:
        """Property: after validation, total_debits == total_credits."""
        engine = GLPostingEngine()
        entry = make_balanced_je(total=amount)
        engine._validate_balance(entry)
        assert entry.total_debits == entry.total_credits


# ═══════════════════════════════════════════════════════════════════════════
# Full Posting Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestFullPostingPipeline:
    """Integration tests for the complete posting pipeline."""

    @pytest.mark.asyncio
    async def test_successful_posting_returns_posting_result(self, gl_engine: GLPostingEngine) -> None:
        entry = make_balanced_je(total=Decimal("5000.00"), debit_account="1010", credit_account="2010")
        result = await gl_engine.post_journal_entry(entry)
        assert result.success is True
        assert result.entry_number.startswith("JE-")

    @pytest.mark.asyncio
    async def test_posted_entry_status_is_posted(self, gl_engine: GLPostingEngine) -> None:
        entry = make_balanced_je()
        await gl_engine.post_journal_entry(entry)
        assert entry.status == "posted"

    @pytest.mark.asyncio
    async def test_metrics_incremented_after_posting(self, gl_engine: GLPostingEngine) -> None:
        entry = make_balanced_je()
        await gl_engine.post_journal_entry(entry)
        assert gl_engine.get_metrics()["entries_posted"] == 1

    @pytest.mark.asyncio
    async def test_batch_posting_processes_all_entries(self, gl_engine: GLPostingEngine) -> None:
        entries = [make_balanced_je(total=Decimal(str(i * 100 + 100))) for i in range(5)]
        results = await gl_engine.post_entries_batch(entries)
        assert len(results) == 5
        assert all(r.success for r in results)


# ═══════════════════════════════════════════════════════════════════════════
# AccountBalanceManager Unit Tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.financial
class TestAccountBalanceManagerUnit:
    """Unit tests for AccountBalanceManager."""

    @pytest.mark.asyncio
    async def test_unregistered_account_update_fails(self) -> None:
        mgr = AccountBalanceManager()
        result = await mgr.update_balance(BalanceUpdateRequest(account_code="9999", debit_amount=Decimal("100.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        assert result.success is False

    @pytest.mark.asyncio
    async def test_batch_update_atomic_rollback_on_failure(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("1010", "Cash", "asset")
        requests = [
            BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("500.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)),
            BalanceUpdateRequest(account_code="INVALID", debit_amount=Decimal("500.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)),
        ]
        results = await mgr.update_balances_batch(requests)
        for r in results:
            assert r.success is False
        cash = await mgr.get_balance("1010")
        assert cash.current_balance == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_balance_sheet_equation(self) -> None:
        mgr = AccountBalanceManager()
        mgr.register_account("1010", "Cash", "asset")
        mgr.register_account("2010", "AP", "liability")
        mgr.register_account("3010", "Equity", "equity")
        await mgr.update_balance(BalanceUpdateRequest(account_code="1010", debit_amount=Decimal("1000.00"), credit_amount=Decimal("0.00"), posting_date=date(2024, 6, 15)))
        await mgr.update_balance(BalanceUpdateRequest(account_code="2010", debit_amount=Decimal("0.00"), credit_amount=Decimal("600.00"), posting_date=date(2024, 6, 15)))
        await mgr.update_balance(BalanceUpdateRequest(account_code="3010", debit_amount=Decimal("0.00"), credit_amount=Decimal("400.00"), posting_date=date(2024, 6, 15)))
        bs = await mgr.calculate_balance_sheet_equation()
        assert abs(bs["difference"]) <= Decimal("0.01")

    @pytest.mark.asyncio
    async def test_manager_instantiation_no_args(self) -> None:
        mgr = AccountBalanceManager()
        assert mgr._db_session_factory is None
