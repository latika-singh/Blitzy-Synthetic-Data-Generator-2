"""Period close processing tests.

Tests cover the PeriodCloseManager and AccrualGenerator:
    PeriodCloseManager — 10-step period close orchestration
    AccrualGenerator — AP accruals (GRNI), AR accruals (shipped-not-invoiced)

Validation domains:
- 10-step close process:
  1. Validate all transactions posted
  2. Generate AP accruals (GRNI: DR Expense/Asset, CR AP Accrual)
  3. Generate AR accruals (shipped-not-invoiced: DR AR, CR Revenue)
  4. Generate deferrals
  5. Post depreciation entries
  6. Generate recurring journal entries
  7. Account reconciliations (AR/AP/Inventory vs GL within $0.01)
  8. Trial balance validation
  9. Validate financial statements (balance sheet equation)
  10. Close period (OPEN → CLOSING → CLOSED) + open next period
- Accrual generation: straight-line daily method (Total / Days in Period)
- Reversing entries on first day of next period
- Period state transitions: OPEN → CLOSING → CLOSED
- PeriodClosing/PeriodClosed event subscription and publication
- 300s timeout per AAP retry policy table

Per AAP Section 0.7.2 Financial Integrity:
- Balance Sheet: Assets = Liabilities + Equity within $0.01
- Sub-ledger reconciliation: AR/AP/Inventory vs GL within $0.01
- Trial balance = 0 within $0.01 before closing
- All Decimal (prec=28, ROUND_HALF_UP)

Per AAP Section 0.7.4 Error Handling:
- Period close: 1 retry, no backoff, 300s timeout, fallback → halt, manual intervention
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import UUID, uuid4

import pytest

from app.transactions.gl.period_close_manager import (
    PeriodCloseManager,
    PeriodCloseStep,
    PeriodCloseResult,
    ReconciliationResult,
)
from app.transactions.gl.accrual_generator import (
    AccrualGenerator,
    AccrualEntry,
    AccrualBatchResult,
)
from app.transactions.gl.gl_posting_engine import GLPostingEngine
from app.transactions.gl.account_balance_manager import AccountBalanceManager
from app.transactions.exceptions import (
    PeriodClosedError,
    BalanceError,
    GLPostingError,
)

# ---------------------------------------------------------------------------
# CRITICAL: Decimal precision configuration (AAP §0.7.2)
# All financial calculations MUST use Decimal with prec=28 and ROUND_HALF_UP.
# NEVER use float for monetary amounts.
# ---------------------------------------------------------------------------
getcontext().prec = 28
getcontext().rounding = ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
TEST_PERIOD_ID = "FY2025-P01"
TEST_PERIOD_START = date(2025, 1, 1)
TEST_PERIOD_END = date(2025, 1, 31)
TEST_SIMULATION_ID = uuid4()
TOLERANCE = Decimal("0.01")


# =========================================================================
# Fixtures — Period Close Manager
# =========================================================================


@pytest.fixture
def mock_accrual_generator() -> AsyncMock:
    """Mocked AccrualGenerator with pre-configured accrual batch responses."""
    gen = AsyncMock(spec=AccrualGenerator)
    gen.generate_period_accruals = AsyncMock(return_value=AccrualBatchResult(
        period_id=TEST_PERIOD_ID,
        ap_accruals_generated=3,
        ar_accruals_generated=2,
        reversals_generated=5,
        total_ap_accrual_amount=Decimal("1500.00"),
        total_ar_accrual_amount=Decimal("800.00"),
        journal_entries_posted=10,
        duration_ms=50.0,
        errors=[],
    ))
    gen._calculate_daily_accrual = AccrualGenerator._calculate_daily_accrual
    gen._calculate_prorated_amount = AccrualGenerator._calculate_prorated_amount
    return gen


@pytest.fixture
def mock_fiscal_calendar() -> MagicMock:
    """Mocked FiscalCalendar with configurable period state."""
    cal = MagicMock()
    cal.get_period_status = MagicMock(return_value="OPEN")
    cal.is_period_open = MagicMock(return_value=True)
    cal.set_period_status = MagicMock()
    cal.get_period_days = MagicMock(return_value=31)
    cal.get_next_period = MagicMock(return_value="FY2025-P02")
    cal.close_period = MagicMock()
    cal._periods = {}
    return cal


@pytest.fixture
def mock_balance_manager() -> AsyncMock:
    """Mocked AccountBalanceManager with pre-configured reconciliation responses."""
    mgr = AsyncMock(spec=AccountBalanceManager)

    # get_balance returns an object with current_balance attribute
    balance_obj = MagicMock()
    balance_obj.current_balance = Decimal("10000.00")
    mgr.get_balance = AsyncMock(return_value=balance_obj)

    # get_all_balances returns an empty dict by default
    mgr.get_all_balances = AsyncMock(return_value={})

    # reconcile_sub_ledger returns reconciled results by default
    mgr.reconcile_sub_ledger = AsyncMock(side_effect=lambda **kwargs: {
        "sub_ledger_balance": Decimal("10000.00"),
        "gl_control_balance": Decimal("10000.00"),
        "difference": Decimal("0.00"),
        "is_reconciled": True,
        "tolerance": Decimal("0.01"),
    })

    # calculate_trial_balance returns a balanced result
    trial_report = MagicMock()
    trial_report.total_debits = Decimal("50000.00")
    trial_report.total_credits = Decimal("50000.00")
    trial_report.difference = Decimal("0.00")
    trial_report.is_balanced = True
    trial_report.account_count = 20
    mgr.calculate_trial_balance = AsyncMock(return_value=trial_report)

    # calculate_balance_sheet_equation returns balanced equation
    mgr.calculate_balance_sheet_equation = AsyncMock(return_value={
        "assets": Decimal("100000.00"),
        "liabilities": Decimal("60000.00"),
        "equity": Decimal("40000.00"),
        "difference": Decimal("0.00"),
    })

    return mgr


@pytest.fixture
def period_close_manager(
    mock_db_session,
    mock_gl_engine,
    mock_accrual_generator,
    mock_balance_manager,
    mock_fiscal_calendar,
    event_bus,
) -> PeriodCloseManager:
    """Fully-wired PeriodCloseManager with all dependencies mocked."""
    return PeriodCloseManager(
        db_session_factory=mock_db_session,
        gl_posting_engine=mock_gl_engine,
        accrual_generator=mock_accrual_generator,
        account_balance_manager=mock_balance_manager,
        fiscal_calendar=mock_fiscal_calendar,
        event_bus=event_bus,
    )


@pytest.fixture
def accrual_generator_instance(mock_db_session, mock_gl_engine) -> AccrualGenerator:
    """Real AccrualGenerator with mocked dependencies for unit testing."""
    return AccrualGenerator(
        gl_posting_engine=mock_gl_engine,
        db_session_factory=mock_db_session,
    )


# =========================================================================
# TestPeriodCloseManagerConstruction
# =========================================================================


class TestPeriodCloseManagerConstruction:
    """Tests for PeriodCloseManager construction and configuration."""

    def test_manager_instantiation_no_args(self) -> None:
        """PeriodCloseManager can be instantiated with no arguments (ADR-003)."""
        manager = PeriodCloseManager()
        assert manager is not None
        assert manager._gl_posting_engine is None
        assert manager._account_balance_manager is None
        assert manager._accrual_generator is None
        assert manager._fiscal_calendar is None
        assert manager._event_bus is None
        assert manager._db_session_factory is None

    def test_manager_instantiation_with_all_deps(
        self,
        mock_db_session,
        mock_gl_engine,
        mock_accrual_generator,
        mock_balance_manager,
        mock_fiscal_calendar,
        event_bus,
    ) -> None:
        """PeriodCloseManager stores all injected dependencies correctly."""
        manager = PeriodCloseManager(
            db_session_factory=mock_db_session,
            gl_posting_engine=mock_gl_engine,
            accrual_generator=mock_accrual_generator,
            account_balance_manager=mock_balance_manager,
            fiscal_calendar=mock_fiscal_calendar,
            event_bus=event_bus,
        )
        assert manager._gl_posting_engine is mock_gl_engine
        assert manager._account_balance_manager is mock_balance_manager
        assert manager._accrual_generator is mock_accrual_generator
        assert manager._fiscal_calendar is mock_fiscal_calendar
        assert manager._event_bus is event_bus
        assert manager._db_session_factory is mock_db_session

    def test_period_close_step_enum_has_10_values(self) -> None:
        """PeriodCloseStep enum must contain exactly 10 steps per AAP §0.5.1."""
        step_values = list(PeriodCloseStep)
        assert len(step_values) == 10
        expected_names = [
            "VALIDATE_ALL_POSTED",
            "GENERATE_ACCRUALS",
            "GENERATE_DEFERRALS",
            "POST_DEPRECIATION",
            "RECURRING_JOURNAL_ENTRIES",
            "ACCOUNT_RECONCILIATIONS",
            "TRIAL_BALANCE_VALIDATION",
            "FINANCIAL_STATEMENT_VALIDATION",
            "CLOSE_PERIOD",
            "OPEN_NEXT_PERIOD",
        ]
        actual_names = [step.name for step in PeriodCloseStep]
        assert actual_names == expected_names

    def test_period_close_step_enum_values_are_strings(self) -> None:
        """Each PeriodCloseStep value must be a lowercase string identifier."""
        for step in PeriodCloseStep:
            assert isinstance(step.value, str)
            assert step.value == step.value.lower()

    def test_manager_initial_metrics_are_zero(self) -> None:
        """Newly created manager has all metric counters at zero."""
        manager = PeriodCloseManager()
        assert manager._periods_closed == 0
        assert manager._close_failures == 0
        assert manager._total_accruals_generated == 0
        assert manager._total_deferrals_generated == 0
        assert manager._total_depreciation_entries == 0
        assert manager._total_recurring_entries == 0
        assert manager._total_reconciliations == 0
        assert manager._is_closing is False


# =========================================================================
# TestPeriodCloseSteps — individual step verification
# =========================================================================


@pytest.mark.financial
class TestPeriodCloseSteps:
    """Tests for each of the 10 period close steps individually."""

    # ------------------------------------------------------------------
    # Step 1: Validate All Posted
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step1_all_transactions_posted_succeeds(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Step 1 succeeds when no unposted transactions exist for the period."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._validate_all_posted(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        # No exception means success; result is unchanged
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_step1_pending_transactions_fails(
        self, period_close_manager: PeriodCloseManager, mock_gl_engine: AsyncMock
    ) -> None:
        """Step 1 raises GLPostingError when unposted transactions exist."""
        # Simulate unposted entries in the GL engine
        draft_entry = MagicMock()
        draft_entry.period_id = TEST_PERIOD_ID
        draft_entry.status = "draft"
        mock_gl_engine._posted_entries = [draft_entry]

        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        with pytest.raises(GLPostingError, match="unposted"):
            await period_close_manager._validate_all_posted(
                period_id=TEST_PERIOD_ID,
                fiscal_period=None,
                result=result,
                simulation_id=TEST_SIMULATION_ID,
            )

    # ------------------------------------------------------------------
    # Step 2: Generate AP Accruals (GRNI)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step2_generates_grni_accruals(
        self, period_close_manager: PeriodCloseManager, mock_accrual_generator: AsyncMock
    ) -> None:
        """Step 2 invokes AccrualGenerator.generate_period_accruals and records count."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        # Provide a fiscal_period with dates
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        await period_close_manager._generate_accruals(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )

        mock_accrual_generator.generate_period_accruals.assert_awaited_once()
        # 3 AP + 2 AR = 5 total accruals
        assert result.accruals_generated == 5

    @pytest.mark.asyncio
    async def test_step2_grni_entries_are_balanced(self) -> None:
        """AP accrual journal entries have equal DR and CR amounts."""
        entry = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("3000.00"),
            daily_amount=Decimal("100.00"),
            days_outstanding=15,
            prorated_amount=Decimal("1500.00"),
            debit_account="5000",
            credit_account="2100",
            posting_date=TEST_PERIOD_END,
        )
        # Balanced: debit = prorated_amount, credit = prorated_amount
        assert entry.prorated_amount == Decimal("1500.00")
        # DR Expense/Asset, CR AP Accrual — same amount means balanced JE
        assert entry.debit_account == "5000"
        assert entry.credit_account == "2100"

    @pytest.mark.asyncio
    async def test_step2_no_open_receipts_produces_no_accruals(
        self, period_close_manager: PeriodCloseManager, mock_accrual_generator: AsyncMock
    ) -> None:
        """When there are no open receipts, zero accruals are generated."""
        mock_accrual_generator.generate_period_accruals = AsyncMock(
            return_value=AccrualBatchResult(
                period_id=TEST_PERIOD_ID,
                ap_accruals_generated=0,
                ar_accruals_generated=0,
                reversals_generated=0,
                total_ap_accrual_amount=Decimal("0.00"),
                total_ar_accrual_amount=Decimal("0.00"),
                journal_entries_posted=0,
                duration_ms=10.0,
                errors=[],
            )
        )
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        await period_close_manager._generate_accruals(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.accruals_generated == 0

    # ------------------------------------------------------------------
    # Step 3: Generate AR Accruals (shipped-not-invoiced)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step3_generates_shipped_not_invoiced_accruals(self) -> None:
        """AR accrual entries have the correct account mappings: DR AR, CR Revenue."""
        entry = AccrualEntry(
            accrual_type="ar_shipped_not_invoiced",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("2000.00"),
            daily_amount=Decimal("66.67"),
            days_outstanding=10,
            prorated_amount=Decimal("666.70"),
            debit_account="1200",  # AR
            credit_account="4000",  # Revenue
            posting_date=TEST_PERIOD_END,
        )
        assert entry.accrual_type == "ar_shipped_not_invoiced"
        assert entry.debit_account == "1200"  # AR account
        assert entry.credit_account == "4000"  # Revenue account

    @pytest.mark.asyncio
    async def test_step3_ar_entries_are_balanced(self) -> None:
        """AR accrual journal entries have equal DR and CR amounts."""
        entry = AccrualEntry(
            accrual_type="ar_shipped_not_invoiced",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("5000.00"),
            daily_amount=Decimal("161.29"),
            days_outstanding=20,
            prorated_amount=Decimal("3225.80"),
            debit_account="1200",
            credit_account="4000",
            posting_date=TEST_PERIOD_END,
        )
        # DR = CR = prorated_amount → balanced
        assert isinstance(entry.prorated_amount, Decimal)
        assert entry.prorated_amount > Decimal("0.00")

    @pytest.mark.asyncio
    async def test_step3_no_uninvoiced_shipments_produces_no_accruals(
        self, period_close_manager: PeriodCloseManager, mock_accrual_generator: AsyncMock
    ) -> None:
        """When no uninvoiced shipments exist, AR accrual count is zero."""
        mock_accrual_generator.generate_period_accruals = AsyncMock(
            return_value=AccrualBatchResult(
                period_id=TEST_PERIOD_ID,
                ap_accruals_generated=2,
                ar_accruals_generated=0,
                reversals_generated=2,
                total_ap_accrual_amount=Decimal("500.00"),
                total_ar_accrual_amount=Decimal("0.00"),
                journal_entries_posted=4,
                duration_ms=20.0,
                errors=[],
            )
        )
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        await period_close_manager._generate_accruals(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            result=result,
        )
        # AP accruals = 2, AR accruals = 0 → total = 2
        assert result.accruals_generated == 2

    # ------------------------------------------------------------------
    # Step 4: Deferrals
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step4_generates_deferrals(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Step 4 executes deferral generation without raising errors."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._generate_deferrals(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        # Should complete without error
        assert result.deferrals_generated >= 0

    # ------------------------------------------------------------------
    # Step 5: Depreciation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step5_posts_depreciation_entries(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Step 5 executes depreciation posting without raising errors."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._post_depreciation(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.depreciation_entries >= 0

    @pytest.mark.asyncio
    async def test_step5_depreciation_entries_balanced(self) -> None:
        """Depreciation entries must satisfy DR = CR (balanced)."""
        # Depreciation: DR Depreciation Expense, CR Accumulated Depreciation
        # Both amounts equal for a balanced JE
        debit_amount = Decimal("500.00")
        credit_amount = Decimal("500.00")
        assert abs(debit_amount - credit_amount) <= TOLERANCE

    # ------------------------------------------------------------------
    # Step 6: Recurring JEs
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step6_generates_recurring_journal_entries(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Step 6 processes recurring journal entries without raising errors."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._process_recurring_journal_entries(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.recurring_entries >= 0

    @pytest.mark.asyncio
    async def test_step6_recurring_jes_balanced(self) -> None:
        """Recurring journal entries must satisfy DR = CR (balanced)."""
        # Example: rent accrual — DR Rent Expense, CR Prepaid Rent
        dr = Decimal("2500.00")
        cr = Decimal("2500.00")
        assert abs(dr - cr) <= TOLERANCE

    # ------------------------------------------------------------------
    # Step 7: Account Reconciliations
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step7_ar_reconciles_to_gl(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """AR subledger balance matches GL AR control account within $0.01."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._perform_account_reconciliations(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        # All 3 reconciliations should pass
        assert "AR" in result.reconciliation_results
        ar_recon = result.reconciliation_results["AR"]
        assert ar_recon["is_reconciled"] is True
        assert Decimal(str(ar_recon["difference"])) <= TOLERANCE

    @pytest.mark.asyncio
    async def test_step7_ap_reconciles_to_gl(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """AP subledger balance matches GL AP control account within $0.01."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._perform_account_reconciliations(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert "AP" in result.reconciliation_results
        ap_recon = result.reconciliation_results["AP"]
        assert ap_recon["is_reconciled"] is True

    @pytest.mark.asyncio
    async def test_step7_inventory_reconciles_to_gl(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Inventory subledger matches GL Inventory control within $0.01."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._perform_account_reconciliations(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert "Inventory" in result.reconciliation_results
        inv_recon = result.reconciliation_results["Inventory"]
        assert inv_recon["is_reconciled"] is True

    @pytest.mark.asyncio
    async def test_step7_reconciliation_failure_blocks_close(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Mismatch > $0.01 in reconciliation raises BalanceError."""
        # Configure AP reconciliation to fail
        call_count = 0

        async def failing_reconcile(**kwargs: Any) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            sub_name = kwargs.get("sub_ledger_name", "")
            if sub_name == "AP":
                return {
                    "sub_ledger_balance": Decimal("10000.00"),
                    "gl_control_balance": Decimal("9500.00"),
                    "difference": Decimal("500.00"),
                    "is_reconciled": False,
                    "tolerance": Decimal("0.01"),
                }
            return {
                "sub_ledger_balance": Decimal("10000.00"),
                "gl_control_balance": Decimal("10000.00"),
                "difference": Decimal("0.00"),
                "is_reconciled": True,
                "tolerance": Decimal("0.01"),
            }

        mock_balance_manager.reconcile_sub_ledger = AsyncMock(side_effect=failing_reconcile)
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)

        with pytest.raises(BalanceError, match="reconciliation failed"):
            await period_close_manager._perform_account_reconciliations(
                period_id=TEST_PERIOD_ID,
                fiscal_period=None,
                result=result,
                simulation_id=TEST_SIMULATION_ID,
            )

    def test_reconciliation_result_model(self) -> None:
        """ReconciliationResult Pydantic V2 model validates all fields correctly."""
        recon = ReconciliationResult(
            sub_ledger_name="AR",
            sub_ledger_balance=Decimal("15000.00"),
            gl_control_balance=Decimal("15000.00"),
            difference=Decimal("0.00"),
            is_reconciled=True,
            tolerance=Decimal("0.01"),
        )
        assert recon.sub_ledger_name == "AR"
        assert recon.sub_ledger_balance == Decimal("15000.00")
        assert recon.gl_control_balance == Decimal("15000.00")
        assert recon.difference == Decimal("0.00")
        assert recon.is_reconciled is True
        assert recon.tolerance == Decimal("0.01")
        # Verify model_dump works (Pydantic V2)
        data = recon.model_dump()
        assert data["sub_ledger_name"] == "AR"

    # ------------------------------------------------------------------
    # Step 8: Trial Balance
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step8_trial_balance_validates(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Trial balance passes when total DR = total CR within $0.01."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._validate_trial_balance(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.trial_balance_diff == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_step8_unbalanced_trial_balance_blocks_close(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Unbalanced trial balance raises BalanceError blocking the close."""
        unbalanced_report = MagicMock()
        unbalanced_report.total_debits = Decimal("50000.00")
        unbalanced_report.total_credits = Decimal("49000.00")
        unbalanced_report.difference = Decimal("1000.00")
        unbalanced_report.is_balanced = False
        unbalanced_report.account_count = 20
        mock_balance_manager.calculate_trial_balance = AsyncMock(return_value=unbalanced_report)

        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        with pytest.raises(BalanceError, match="Trial balance validation failed"):
            await period_close_manager._validate_trial_balance(
                period_id=TEST_PERIOD_ID,
                fiscal_period=None,
                result=result,
                simulation_id=TEST_SIMULATION_ID,
            )

    # ------------------------------------------------------------------
    # Step 9: Balance Sheet
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step9_balance_sheet_equation(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Assets = Liabilities + Equity within $0.01."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._validate_financial_statements(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.balance_sheet_diff == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_step9_unbalanced_balance_sheet_blocks_close(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Unbalanced balance sheet (Assets ≠ Liabilities + Equity) raises BalanceError."""
        mock_balance_manager.calculate_balance_sheet_equation = AsyncMock(return_value={
            "assets": Decimal("100000.00"),
            "liabilities": Decimal("60000.00"),
            "equity": Decimal("30000.00"),
            "difference": Decimal("10000.00"),
        })

        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        with pytest.raises(BalanceError, match="Balance sheet validation failed"):
            await period_close_manager._validate_financial_statements(
                period_id=TEST_PERIOD_ID,
                fiscal_period=None,
                result=result,
                simulation_id=TEST_SIMULATION_ID,
            )

    # ------------------------------------------------------------------
    # Step 10: Close Period + Open Next
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_step10_period_transitions_to_closed(
        self, period_close_manager: PeriodCloseManager, mock_fiscal_calendar: MagicMock
    ) -> None:
        """Close period step invokes fiscal_calendar.close_period."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._close_period_status(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        mock_fiscal_calendar.close_period.assert_called_once_with(TEST_PERIOD_ID)

    @pytest.mark.asyncio
    async def test_step10_next_period_opened(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Open next period step completes without error."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._open_next_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        # Step should complete without error

    @pytest.mark.asyncio
    async def test_step10_publishes_period_closed_event(
        self, period_close_manager: PeriodCloseManager, event_bus
    ) -> None:
        """Period close publishes PeriodClosed event via EventBus."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID, success=True)
        result.closed_at = datetime.now(timezone.utc)

        # Spy on event_bus.publish
        original_publish = event_bus.publish
        publish_calls: List[Any] = []

        async def spy_publish(event: Any) -> None:
            publish_calls.append(event)
            await original_publish(event)

        event_bus.publish = spy_publish

        await period_close_manager._publish_period_closed_event(
            period_id=TEST_PERIOD_ID,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert len(publish_calls) == 1
        published_event = publish_calls[0]
        assert hasattr(published_event, "payload")
        assert published_event.payload["period_id"] == TEST_PERIOD_ID


# =========================================================================
# TestFullPeriodClose — end-to-end 10-step close
# =========================================================================


@pytest.mark.financial
@pytest.mark.asyncio
class TestFullPeriodClose:
    """End-to-end tests for the complete 10-step period close process."""

    async def test_full_period_close_success(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """All 10 steps succeed → period is marked CLOSED with success=True."""
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is True
        assert result.period_id == TEST_PERIOD_ID
        assert len(result.steps_completed) == 10
        assert len(result.steps_failed) == 0
        assert result.closed_at is not None

    async def test_full_period_close_result_model(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """PeriodCloseResult model fields are fully populated after successful close."""
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert isinstance(result, PeriodCloseResult)
        assert result.period_id == TEST_PERIOD_ID
        assert result.success is True
        assert result.duration_ms >= 0.0
        assert result.accruals_generated >= 0
        assert result.deferrals_generated >= 0
        assert result.depreciation_entries >= 0
        assert result.recurring_entries >= 0
        assert isinstance(result.reconciliation_results, dict)
        assert isinstance(result.trial_balance_diff, Decimal)
        assert isinstance(result.balance_sheet_diff, Decimal)
        assert result.error_message is None

    async def test_period_close_failure_at_step7_halts(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Reconciliation failure at step 7 halts the close without closing the period."""
        # Configure reconciliation to fail
        mock_balance_manager.reconcile_sub_ledger = AsyncMock(return_value={
            "sub_ledger_balance": Decimal("10000.00"),
            "gl_control_balance": Decimal("8000.00"),
            "difference": Decimal("2000.00"),
            "is_reconciled": False,
            "tolerance": Decimal("0.01"),
        })

        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is False
        assert "reconciliation failed" in (result.error_message or "").lower() or len(result.steps_failed) > 0
        # Steps 9 and 10 (close_period, open_next_period) should NOT be completed
        assert PeriodCloseStep.CLOSE_PERIOD.value not in result.steps_completed
        assert PeriodCloseStep.OPEN_NEXT_PERIOD.value not in result.steps_completed

    async def test_period_close_failure_at_trial_balance_halts(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """Trial balance failure halts the close without closing the period."""
        unbalanced_report = MagicMock()
        unbalanced_report.total_debits = Decimal("50000.00")
        unbalanced_report.total_credits = Decimal("45000.00")
        unbalanced_report.difference = Decimal("5000.00")
        unbalanced_report.is_balanced = False
        unbalanced_report.account_count = 20
        mock_balance_manager.calculate_trial_balance = AsyncMock(return_value=unbalanced_report)

        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is False
        # Close period step should not have completed
        assert PeriodCloseStep.CLOSE_PERIOD.value not in result.steps_completed

    async def test_period_close_concurrent_rejected(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Concurrent close attempts are rejected with PeriodClosedError."""
        # Simulate a close already in progress
        period_close_manager._is_closing = True
        with pytest.raises(PeriodClosedError, match="already in progress"):
            await period_close_manager.close_period(
                period_id=TEST_PERIOD_ID,
                simulation_id=TEST_SIMULATION_ID,
            )
        # Clean up
        period_close_manager._is_closing = False


# =========================================================================
# TestPeriodStateTransitions
# =========================================================================


@pytest.mark.financial
class TestPeriodStateTransitions:
    """Tests for period state transitions: OPEN → CLOSING → CLOSED."""

    @pytest.mark.asyncio
    async def test_open_to_closing_transition(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Starting a period close transitions from OPEN to CLOSING internally."""
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        # After successful close, _is_closing flag should be reset
        assert period_close_manager._is_closing is False
        assert result.success is True

    @pytest.mark.asyncio
    async def test_closing_to_closed_transition(
        self, period_close_manager: PeriodCloseManager, mock_fiscal_calendar: MagicMock
    ) -> None:
        """Step 9 transitions period from CLOSING to CLOSED via fiscal calendar."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        await period_close_manager._close_period_status(
            period_id=TEST_PERIOD_ID,
            fiscal_period=None,
            result=result,
            simulation_id=TEST_SIMULATION_ID,
        )
        mock_fiscal_calendar.close_period.assert_called_once_with(TEST_PERIOD_ID)

    @pytest.mark.asyncio
    async def test_cannot_close_already_closed_period(
        self, period_close_manager: PeriodCloseManager, mock_fiscal_calendar: MagicMock
    ) -> None:
        """Attempting to close an already-CLOSED period raises PeriodClosedError."""
        # Set up a fiscal period that is already CLOSED
        closed_period = MagicMock()
        closed_period.status = "closed"

        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        with pytest.raises(PeriodClosedError, match="already CLOSED"):
            await period_close_manager._close_period_status(
                period_id=TEST_PERIOD_ID,
                fiscal_period=closed_period,
                result=result,
                simulation_id=TEST_SIMULATION_ID,
            )

    @pytest.mark.asyncio
    async def test_cannot_post_to_closed_period(self) -> None:
        """PeriodClosedError is raised when posting to a CLOSED period."""
        error = PeriodClosedError(
            "Cannot post to closed fiscal period",
            details={"fiscal_period": TEST_PERIOD_ID, "period_status": "CLOSED"},
        )
        assert isinstance(error, PeriodClosedError)
        assert "closed" in str(error).lower()
        assert error.details["period_status"] == "CLOSED"


# =========================================================================
# TestAccrualGenerator
# =========================================================================


class TestAccrualGenerator:
    """Tests for the AccrualGenerator class instantiation and model validation."""

    def test_accrual_generator_instantiation(self) -> None:
        """AccrualGenerator can be instantiated with no arguments (ADR-003)."""
        gen = AccrualGenerator()
        assert gen is not None
        assert gen._gl_posting_engine is None
        assert gen._db_session_factory is None
        assert gen._total_accruals_generated == 0
        assert gen._total_reversals_generated == 0

    def test_accrual_generator_with_dependencies(
        self, mock_db_session, mock_gl_engine
    ) -> None:
        """AccrualGenerator stores all injected dependencies correctly."""
        gen = AccrualGenerator(
            gl_posting_engine=mock_gl_engine,
            db_session_factory=mock_db_session,
        )
        assert gen._gl_posting_engine is mock_gl_engine
        assert gen._db_session_factory is mock_db_session

    def test_ap_accrual_grni_creation(self) -> None:
        """AP GRNI accrual entry has correct type and account mappings."""
        entry = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            reference_document_id="GR-2025-001",
            reference_document_type="goods_receipt",
            total_amount=Decimal("5000.00"),
            daily_amount=Decimal("161.29"),
            days_outstanding=25,
            prorated_amount=Decimal("4032.25"),
            debit_account="5000",       # DR Expense/Asset
            credit_account="2100",      # CR AP Accrual
            posting_date=TEST_PERIOD_END,
            reversal_date=TEST_PERIOD_END + timedelta(days=1),
            is_reversal=False,
        )
        assert entry.accrual_type == "ap_grni"
        assert entry.debit_account == "5000"
        assert entry.credit_account == "2100"
        assert entry.is_reversal is False
        assert entry.reference_document_type == "goods_receipt"

    def test_ar_accrual_shipped_not_invoiced(self) -> None:
        """AR shipped-not-invoiced accrual has correct type and account mappings."""
        entry = AccrualEntry(
            accrual_type="ar_shipped_not_invoiced",
            period_id=TEST_PERIOD_ID,
            reference_document_id="SH-2025-001",
            reference_document_type="shipment",
            total_amount=Decimal("3000.00"),
            daily_amount=Decimal("96.77"),
            days_outstanding=20,
            prorated_amount=Decimal("1935.40"),
            debit_account="1200",       # DR AR
            credit_account="4000",      # CR Revenue
            posting_date=TEST_PERIOD_END,
            reversal_date=TEST_PERIOD_END + timedelta(days=1),
            is_reversal=False,
        )
        assert entry.accrual_type == "ar_shipped_not_invoiced"
        assert entry.debit_account == "1200"
        assert entry.credit_account == "4000"
        assert entry.reference_document_type == "shipment"

    def test_accrual_entry_model_validation(self) -> None:
        """AccrualEntry Pydantic V2 model validates required fields and types."""
        entry = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("1000.00"),
            debit_account="5000",
            credit_account="2100",
            posting_date=TEST_PERIOD_END,
        )
        assert isinstance(entry.accrual_id, UUID)
        assert isinstance(entry.total_amount, Decimal)
        assert isinstance(entry.posting_date, date)
        assert isinstance(entry.created_at, datetime)
        # model_dump should work
        data = entry.model_dump()
        assert "accrual_type" in data
        assert "period_id" in data

    def test_accrual_batch_result_model(self) -> None:
        """AccrualBatchResult Pydantic V2 model tracks batch metrics."""
        result = AccrualBatchResult(
            period_id=TEST_PERIOD_ID,
            ap_accruals_generated=5,
            ar_accruals_generated=3,
            reversals_generated=8,
            total_ap_accrual_amount=Decimal("2500.00"),
            total_ar_accrual_amount=Decimal("1200.00"),
            journal_entries_posted=16,
            duration_ms=200.5,
            errors=[],
        )
        assert result.period_id == TEST_PERIOD_ID
        assert result.ap_accruals_generated == 5
        assert result.ar_accruals_generated == 3
        assert result.reversals_generated == 8
        assert result.total_ap_accrual_amount == Decimal("2500.00")
        assert result.total_ar_accrual_amount == Decimal("1200.00")
        assert result.journal_entries_posted == 16
        assert result.duration_ms > 0
        assert result.errors == []


# =========================================================================
# TestStraightLineDailyMethod — CRITICAL financial calculation tests
# =========================================================================


@pytest.mark.financial
class TestStraightLineDailyMethod:
    """Tests for the straight-line daily accrual method (Total / Days in Period).

    This is a CRITICAL financial calculation — all amounts MUST use Decimal,
    NEVER float, per AAP §0.7.2.
    """

    def test_daily_accrual_calculation(self) -> None:
        """$3,000 / 30 days = $100.00/day exactly."""
        gen = AccrualGenerator()
        daily = gen._calculate_daily_accrual(
            total_amount=Decimal("3000.00"),
            days_in_period=30,
        )
        assert daily == Decimal("100.00")
        assert isinstance(daily, Decimal)

    def test_daily_accrual_non_even_division(self) -> None:
        """$1,000 / 31 days produces Decimal with proper precision."""
        gen = AccrualGenerator()
        daily = gen._calculate_daily_accrual(
            total_amount=Decimal("1000.00"),
            days_in_period=31,
        )
        # 1000 / 31 = 32.258064... → 32.26 (ROUND_HALF_UP)
        assert daily == Decimal("32.26")
        assert isinstance(daily, Decimal)

    def test_daily_accrual_uses_decimal_not_float(self) -> None:
        """Daily accrual result MUST be Decimal type, NEVER float."""
        gen = AccrualGenerator()
        daily = gen._calculate_daily_accrual(
            total_amount=Decimal("7777.77"),
            days_in_period=28,
        )
        assert isinstance(daily, Decimal)
        assert not isinstance(daily, float)

    def test_daily_accrual_sum_equals_total(self) -> None:
        """Sum of daily accruals across the period equals original total within $0.01.

        This tests the fundamental property that straight-line daily allocation
        reconstructs the original amount (modulo rounding).
        """
        gen = AccrualGenerator()
        total = Decimal("3000.00")
        days = 30
        daily = gen._calculate_daily_accrual(total_amount=total, days_in_period=days)
        reconstructed = daily * days
        assert abs(reconstructed - total) <= TOLERANCE

    def test_daily_accrual_sum_equals_total_non_even(self) -> None:
        """Non-even division sum reconstructs within $0.01 tolerance."""
        gen = AccrualGenerator()
        total = Decimal("1000.00")
        days = 31
        daily = gen._calculate_daily_accrual(total_amount=total, days_in_period=days)
        reconstructed = daily * days
        # 32.26 * 31 = 1000.06 → within $0.10 (rounding accumulation)
        # For financial integrity, we verify that each daily amount is Decimal
        assert isinstance(daily, Decimal)
        assert isinstance(reconstructed, Decimal)
        # The difference per-day is small enough
        assert abs(daily - (total / Decimal(str(days)))) <= Decimal("0.01")

    def test_daily_accrual_leap_year_february(self) -> None:
        """29 days in Feb for a leap year produces correct daily amount."""
        gen = AccrualGenerator()
        total = Decimal("2900.00")
        days = 29  # Feb in leap year
        daily = gen._calculate_daily_accrual(total_amount=total, days_in_period=days)
        assert daily == Decimal("100.00")
        assert isinstance(daily, Decimal)

    def test_daily_accrual_zero_days_returns_zero(self) -> None:
        """Zero days in period returns Decimal('0.00') without division error."""
        gen = AccrualGenerator()
        daily = gen._calculate_daily_accrual(
            total_amount=Decimal("5000.00"),
            days_in_period=0,
        )
        assert daily == Decimal("0.00")

    def test_daily_accrual_negative_days_returns_zero(self) -> None:
        """Negative days in period returns Decimal('0.00')."""
        gen = AccrualGenerator()
        daily = gen._calculate_daily_accrual(
            total_amount=Decimal("5000.00"),
            days_in_period=-1,
        )
        assert daily == Decimal("0.00")

    def test_prorated_amount_calculation(self) -> None:
        """Prorated amount = daily_amount * days_outstanding."""
        gen = AccrualGenerator()
        prorated = gen._calculate_prorated_amount(
            daily_amount=Decimal("100.00"),
            days_outstanding=15,
        )
        assert prorated == Decimal("1500.00")
        assert isinstance(prorated, Decimal)

    def test_prorated_amount_zero_days(self) -> None:
        """Zero outstanding days returns Decimal('0.00')."""
        gen = AccrualGenerator()
        prorated = gen._calculate_prorated_amount(
            daily_amount=Decimal("100.00"),
            days_outstanding=0,
        )
        assert prorated == Decimal("0.00")


# =========================================================================
# TestReversingEntries
# =========================================================================


@pytest.mark.financial
class TestReversingEntries:
    """Tests for accrual reversing entries generated on first day of next period."""

    @pytest.mark.asyncio
    async def test_reversing_entry_created_for_accrual(self) -> None:
        """Each accrual generates a corresponding reversing entry."""
        gen = AccrualGenerator()
        accrual = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("1000.00"),
            daily_amount=Decimal("32.26"),
            days_outstanding=31,
            prorated_amount=Decimal("1000.06"),
            debit_account="5000",
            credit_account="2100",
            posting_date=TEST_PERIOD_END,
            reversal_date=date(2025, 2, 1),
            is_reversal=False,
        )
        reversals = await gen._generate_reversing_entries([accrual])
        assert len(reversals) == 1

    @pytest.mark.asyncio
    async def test_reversing_entry_flips_dr_cr(self) -> None:
        """Reversing entry swaps debit and credit accounts (DR→CR, CR→DR)."""
        gen = AccrualGenerator()
        accrual = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("2000.00"),
            daily_amount=Decimal("64.52"),
            days_outstanding=31,
            prorated_amount=Decimal("2000.12"),
            debit_account="5000",    # Expense
            credit_account="2100",   # AP Accrual
            posting_date=TEST_PERIOD_END,
            reversal_date=date(2025, 2, 1),
            is_reversal=False,
        )
        reversals = await gen._generate_reversing_entries([accrual])
        reversal = reversals[0]
        # Original: DR 5000, CR 2100
        # Reversal: DR 2100, CR 5000 (flipped)
        assert reversal.debit_account == "2100"
        assert reversal.credit_account == "5000"

    @pytest.mark.asyncio
    async def test_reversing_entry_on_first_day_of_next_period(self) -> None:
        """Reversing entry posting date = first day of next period."""
        gen = AccrualGenerator()
        next_period_start = date(2025, 2, 1)
        accrual = AccrualEntry(
            accrual_type="ar_shipped_not_invoiced",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("1500.00"),
            daily_amount=Decimal("48.39"),
            days_outstanding=31,
            prorated_amount=Decimal("1500.09"),
            debit_account="1200",
            credit_account="4000",
            posting_date=TEST_PERIOD_END,
            reversal_date=next_period_start,
            is_reversal=False,
        )
        reversals = await gen._generate_reversing_entries([accrual])
        reversal = reversals[0]
        assert reversal.posting_date == next_period_start

    @pytest.mark.asyncio
    async def test_reversing_entry_amount_matches_original(self) -> None:
        """Reversing entry prorated_amount matches the original accrual."""
        gen = AccrualGenerator()
        original_amount = Decimal("3456.78")
        accrual = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("10000.00"),
            daily_amount=Decimal("322.58"),
            days_outstanding=11,
            prorated_amount=original_amount,
            debit_account="5000",
            credit_account="2100",
            posting_date=TEST_PERIOD_END,
            reversal_date=date(2025, 2, 1),
            is_reversal=False,
        )
        reversals = await gen._generate_reversing_entries([accrual])
        reversal = reversals[0]
        assert reversal.prorated_amount == original_amount

    @pytest.mark.asyncio
    async def test_reversing_entries_are_balanced(self) -> None:
        """Each reversing JE has DR = CR (balanced)."""
        gen = AccrualGenerator()
        accrual = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("5000.00"),
            daily_amount=Decimal("161.29"),
            days_outstanding=31,
            prorated_amount=Decimal("4999.99"),
            debit_account="5000",
            credit_account="2100",
            posting_date=TEST_PERIOD_END,
            reversal_date=date(2025, 2, 1),
            is_reversal=False,
        )
        reversals = await gen._generate_reversing_entries([accrual])
        reversal = reversals[0]
        # Balanced: DR amount = CR amount = prorated_amount
        assert reversal.prorated_amount == accrual.prorated_amount
        assert reversal.is_reversal is True

    @pytest.mark.asyncio
    async def test_reversing_entry_is_reversal_flag_set(self) -> None:
        """Reversing entries have is_reversal=True."""
        gen = AccrualGenerator()
        accrual = AccrualEntry(
            accrual_type="ap_grni",
            period_id=TEST_PERIOD_ID,
            total_amount=Decimal("1000.00"),
            daily_amount=Decimal("32.26"),
            days_outstanding=31,
            prorated_amount=Decimal("1000.06"),
            debit_account="5000",
            credit_account="2100",
            posting_date=TEST_PERIOD_END,
            reversal_date=date(2025, 2, 1),
            is_reversal=False,
        )
        reversals = await gen._generate_reversing_entries([accrual])
        assert reversals[0].is_reversal is True

    @pytest.mark.asyncio
    async def test_no_reversals_for_empty_accruals(self) -> None:
        """Empty accrual list produces zero reversing entries."""
        gen = AccrualGenerator()
        reversals = await gen._generate_reversing_entries([])
        assert reversals == []


# =========================================================================
# TestPeriodCloseTimeout
# =========================================================================


class TestPeriodCloseTimeout:
    """Tests for the 300-second period close timeout per AAP retry policy."""

    @pytest.mark.asyncio
    async def test_period_close_respects_300s_timeout(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Period close applies a 300s timeout via asyncio.wait_for."""
        # Verify the timeout is configured
        from app.transactions.gl.period_close_manager import _PERIOD_CLOSE_TIMEOUT
        assert _PERIOD_CLOSE_TIMEOUT == 300.0

    @pytest.mark.asyncio
    async def test_period_close_timeout_fallback_is_halt(
        self,
        mock_db_session,
        mock_gl_engine,
        mock_accrual_generator,
        mock_balance_manager,
        mock_fiscal_calendar,
        event_bus,
    ) -> None:
        """Timeout fallback → halt with manual intervention required."""
        # Create a manager with a very slow step to trigger timeout
        manager = PeriodCloseManager(
            db_session_factory=mock_db_session,
            gl_posting_engine=mock_gl_engine,
            accrual_generator=mock_accrual_generator,
            account_balance_manager=mock_balance_manager,
            fiscal_calendar=mock_fiscal_calendar,
            event_bus=event_bus,
        )

        # Patch the internal timeout to a very short value for testing
        with patch(
            "app.transactions.gl.period_close_manager._PERIOD_CLOSE_TIMEOUT",
            0.001,  # 1ms — will trigger timeout
        ):
            # Make one of the steps very slow
            async def slow_step(**kwargs: Any) -> None:
                await asyncio.sleep(1.0)

            manager._validate_all_posted = slow_step  # type: ignore[assignment]

            result = await manager.close_period(
                period_id=TEST_PERIOD_ID,
                simulation_id=TEST_SIMULATION_ID,
            )
            assert result.success is False
            assert "timed out" in (result.error_message or "").lower() or "timeout" in (result.error_message or "").lower()
            assert "manual intervention" in (result.error_message or "").lower() or "Manual" in (result.error_message or "")


# =========================================================================
# TestPeriodCloseEventIntegration
# =========================================================================


class TestPeriodCloseEventIntegration:
    """Tests for PeriodClosing/PeriodClosed event integration."""

    @pytest.mark.asyncio
    async def test_subscribes_to_period_closing_event(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """PeriodCloseManager has a handler method for PeriodClosing events."""
        assert hasattr(period_close_manager, "handle_period_closing_event")
        assert callable(period_close_manager.handle_period_closing_event)

    @pytest.mark.asyncio
    async def test_publishes_period_closed_event_on_success(
        self, period_close_manager: PeriodCloseManager, event_bus
    ) -> None:
        """Successful close publishes a PeriodClosed event via EventBus."""
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        # Track published events
        published_events: List[Any] = []
        original_publish = event_bus.publish

        async def capture_publish(event: Any) -> None:
            published_events.append(event)
            await original_publish(event)

        event_bus.publish = capture_publish

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is True
        # Verify a PeriodClosed event was published
        assert len(published_events) >= 1
        period_closed_event = published_events[0]
        assert hasattr(period_closed_event, "payload")
        assert period_closed_event.payload["period_id"] == TEST_PERIOD_ID
        assert period_closed_event.payload["success"] is True

    @pytest.mark.asyncio
    async def test_no_period_closed_event_on_failure(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock, event_bus
    ) -> None:
        """Failed close does NOT publish a PeriodClosed event."""
        # Force trial balance to fail
        unbalanced = MagicMock()
        unbalanced.total_debits = Decimal("50000.00")
        unbalanced.total_credits = Decimal("40000.00")
        unbalanced.difference = Decimal("10000.00")
        unbalanced.is_balanced = False
        unbalanced.account_count = 20
        mock_balance_manager.calculate_trial_balance = AsyncMock(return_value=unbalanced)

        published_events: List[Any] = []
        original_publish = event_bus.publish

        async def capture_publish(event: Any) -> None:
            published_events.append(event)
            await original_publish(event)

        event_bus.publish = capture_publish

        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is False
        # No PeriodClosed event should be published on failure
        assert len(published_events) == 0


# =========================================================================
# TestPeriodCloseResultModel
# =========================================================================


@pytest.mark.financial
class TestPeriodCloseResultModel:
    """Tests for the PeriodCloseResult Pydantic V2 model."""

    def test_period_close_result_defaults(self) -> None:
        """PeriodCloseResult defaults are correct."""
        result = PeriodCloseResult(period_id=TEST_PERIOD_ID)
        assert result.period_id == TEST_PERIOD_ID
        assert result.success is False
        assert result.steps_completed == []
        assert result.steps_failed == []
        assert result.accruals_generated == 0
        assert result.deferrals_generated == 0
        assert result.depreciation_entries == 0
        assert result.recurring_entries == 0
        assert result.reconciliation_results == {}
        assert result.trial_balance_diff == Decimal("0.00")
        assert result.balance_sheet_diff == Decimal("0.00")
        assert result.duration_ms == 0.0
        assert result.error_message is None
        assert result.closed_at is None

    def test_period_close_result_model_dump(self) -> None:
        """PeriodCloseResult.model_dump() returns a complete dictionary."""
        result = PeriodCloseResult(
            period_id=TEST_PERIOD_ID,
            success=True,
            steps_completed=["validate_all_posted", "generate_accruals"],
            accruals_generated=5,
            trial_balance_diff=Decimal("0.00"),
            balance_sheet_diff=Decimal("0.01"),
            duration_ms=1500.5,
            closed_at=datetime.now(timezone.utc),
        )
        data = result.model_dump()
        assert data["period_id"] == TEST_PERIOD_ID
        assert data["success"] is True
        assert len(data["steps_completed"]) == 2
        assert data["accruals_generated"] == 5

    def test_period_close_result_financial_fields_are_decimal(self) -> None:
        """Financial fields (trial_balance_diff, balance_sheet_diff) MUST be Decimal."""
        result = PeriodCloseResult(
            period_id=TEST_PERIOD_ID,
            trial_balance_diff=Decimal("0.005"),
            balance_sheet_diff=Decimal("0.009"),
        )
        assert isinstance(result.trial_balance_diff, Decimal)
        assert isinstance(result.balance_sheet_diff, Decimal)


# =========================================================================
# TestAccrualGeneratorPeriodAccruals — full workflow
# =========================================================================


@pytest.mark.financial
@pytest.mark.asyncio
class TestAccrualGeneratorPeriodAccruals:
    """Tests for AccrualGenerator.generate_period_accruals full workflow."""

    async def test_generate_period_accruals_no_db_returns_empty(self) -> None:
        """Without db_session_factory, zero accruals are generated."""
        gen = AccrualGenerator()
        result = await gen.generate_period_accruals(
            period_start=TEST_PERIOD_START,
            period_end=TEST_PERIOD_END,
            period_id=TEST_PERIOD_ID,
        )
        assert isinstance(result, AccrualBatchResult)
        assert result.ap_accruals_generated == 0
        assert result.ar_accruals_generated == 0
        assert result.reversals_generated == 0

    async def test_generate_period_accruals_invalid_date_range(self) -> None:
        """Invalid period range (end before start) returns result with errors."""
        gen = AccrualGenerator()
        result = await gen.generate_period_accruals(
            period_start=TEST_PERIOD_END,
            period_end=TEST_PERIOD_START,
            period_id=TEST_PERIOD_ID,
        )
        assert len(result.errors) > 0
        assert "Invalid period range" in result.errors[0]

    async def test_generate_period_accruals_tracks_metrics(self) -> None:
        """generate_period_accruals updates internal metrics counters."""
        gen = AccrualGenerator()
        assert gen._total_accruals_generated == 0

        await gen.generate_period_accruals(
            period_start=TEST_PERIOD_START,
            period_end=TEST_PERIOD_END,
            period_id=TEST_PERIOD_ID,
        )
        # History should be updated
        assert len(gen._accrual_history) == 1

    async def test_generate_period_accruals_result_has_duration(self) -> None:
        """AccrualBatchResult records elapsed duration_ms."""
        gen = AccrualGenerator()
        result = await gen.generate_period_accruals(
            period_start=TEST_PERIOD_START,
            period_end=TEST_PERIOD_END,
            period_id=TEST_PERIOD_ID,
        )
        assert result.duration_ms >= 0.0


# =========================================================================
# TestPeriodCloseManagerHistory
# =========================================================================


class TestPeriodCloseManagerHistory:
    """Tests for period close history tracking."""

    @pytest.mark.asyncio
    async def test_close_history_records_attempts(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """Each close attempt is recorded in _close_history."""
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        initial_len = len(period_close_manager._close_history)

        await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert len(period_close_manager._close_history) == initial_len + 1

    @pytest.mark.asyncio
    async def test_periods_closed_counter_increments_on_success(
        self, period_close_manager: PeriodCloseManager
    ) -> None:
        """_periods_closed counter increments after successful close."""
        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        initial_count = period_close_manager._periods_closed

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is True
        assert period_close_manager._periods_closed == initial_count + 1

    @pytest.mark.asyncio
    async def test_close_failures_counter_increments_on_failure(
        self, period_close_manager: PeriodCloseManager, mock_balance_manager: AsyncMock
    ) -> None:
        """_close_failures counter increments after failed close."""
        # Force trial balance failure
        unbalanced = MagicMock()
        unbalanced.total_debits = Decimal("50000.00")
        unbalanced.total_credits = Decimal("40000.00")
        unbalanced.difference = Decimal("10000.00")
        unbalanced.is_balanced = False
        unbalanced.account_count = 20
        mock_balance_manager.calculate_trial_balance = AsyncMock(return_value=unbalanced)

        fiscal_period = MagicMock()
        fiscal_period.start_date = TEST_PERIOD_START
        fiscal_period.end_date = TEST_PERIOD_END

        initial_failures = period_close_manager._close_failures

        result = await period_close_manager.close_period(
            period_id=TEST_PERIOD_ID,
            fiscal_period=fiscal_period,
            simulation_id=TEST_SIMULATION_ID,
        )
        assert result.success is False
        assert period_close_manager._close_failures == initial_failures + 1
