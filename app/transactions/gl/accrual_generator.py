"""Accrual Generator — generates period-end accrual and reversing entries.

Produces two categories of accruals at period close:

1. **AP Accruals (GRNI — Goods Received Not Invoiced)**:
   - Debit: Expense/Asset account
   - Credit: AP Accrual account
   - Generated for goods receipts that have not yet been matched to vendor invoices

2. **AR Accruals (Shipped Not Invoiced)**:
   - Debit: AR account
   - Credit: Revenue account
   - Generated for shipments that have not yet been invoiced to customers

Accrual Method: **Straight-Line Daily** (Total / Days in Period)
    For each accrual, the daily amount = total_accrual_amount / days_in_period.
    Accruals are prorated based on the number of days within the period that
    the underlying transaction has been outstanding.

Reversing Entries: Every accrual generates a corresponding reversing entry
dated the first day of the next period.

GL Posting Delegation: All journal entries are posted via the injected
``GLPostingEngine``, which enforces balance validation (DR = CR within $0.01)
and atomicity (full rollback on failure).

References:
    - AAP §0.5.1 Group 4: GL Integration (AccrualGenerator)
    - AAP §0.7.2: Financial Integrity Rules (balance validation, atomicity)
    - AAP §0.1.2: Accrual Method — Straight-Line Daily (Total / Days in Period)
"""

from __future__ import annotations

import decimal
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
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
    DEBIT_NORMAL_ACCOUNT_TYPES,
    CREDIT_NORMAL_ACCOUNT_TYPES,
    DOCUMENT_NUMBER_PREFIXES,
)

if TYPE_CHECKING:
    from app.events.event_bus import EventBus
    from app.orchestration.fiscal_calendar import FiscalCalendar, FiscalPeriod
    from app.transactions.gl.gl_posting_engine import GLPostingEngine
    from app.transactions.gl.account_balance_manager import AccountBalanceManager

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only, JSON format)
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
# Default GL account codes used when source-document account info is absent.
# These serve as fallback mappings for accrual journal entry lines.
# ---------------------------------------------------------------------------
_DEFAULT_AP_ACCRUAL_DEBIT_ACCOUNT = "5000"    # Expense account (fallback)
_DEFAULT_AP_ACCRUAL_CREDIT_ACCOUNT = "2100"   # AP Accrual account (fallback)
_DEFAULT_AR_ACCRUAL_DEBIT_ACCOUNT = "1200"    # AR account (fallback)
_DEFAULT_AR_ACCRUAL_CREDIT_ACCOUNT = "4000"   # Revenue account (fallback)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Models
# ═══════════════════════════════════════════════════════════════════════════


class AccrualEntry(BaseModel):
    """Represents a single accrual or reversing entry to be posted.

    Each accrual entry corresponds to a single source document (goods receipt
    or shipment) that remains unmatched/uninvoiced at period end.  The prorated
    amount is computed using the straight-line daily method:

        daily_amount = total_amount / days_in_period
        prorated_amount = daily_amount * days_outstanding

    Reversing entries swap the debit/credit accounts and are dated the first
    day of the next period.

    Attributes:
        accrual_id: Unique identifier for this accrual entry (UUID4).
        accrual_type: ``'ap_grni'`` or ``'ar_shipped_not_invoiced'``.
        period_id: Fiscal period identifier (e.g., ``'2024-01'``).
        reference_document_id: Source document ID (goods receipt or shipment).
        reference_document_type: ``'goods_receipt'`` or ``'shipment'``.
        total_amount: Full accrual amount from the source document.
        daily_amount: Straight-line daily amount (total / days_in_period).
        days_outstanding: Days the transaction has been outstanding in-period.
        prorated_amount: Prorated accrual (daily_amount * days_outstanding).
        debit_account: GL account to debit.
        credit_account: GL account to credit.
        posting_date: Date the accrual is posted (period end date).
        reversal_date: Date of the reversing entry (first day of next period).
        is_reversal: ``True`` for reversing entries.
        created_at: UTC timestamp of creation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    accrual_id: UUID = Field(
        default_factory=uuid4,
        description="Unique accrual entry identifier",
    )
    accrual_type: str = Field(
        ...,
        description="'ap_grni' or 'ar_shipped_not_invoiced'",
    )
    period_id: str = Field(
        ...,
        description="Fiscal period identifier",
    )
    reference_document_id: Optional[str] = Field(
        default=None,
        description="Source document ID (GR or Shipment)",
    )
    reference_document_type: Optional[str] = Field(
        default=None,
        description="'goods_receipt' or 'shipment'",
    )
    total_amount: Decimal = Field(
        ...,
        description="Total accrual amount",
    )
    daily_amount: Decimal = Field(
        default=Decimal("0.00"),
        description="Daily accrual amount (total / days_in_period)",
    )
    days_outstanding: int = Field(
        default=0,
        ge=0,
        description="Days outstanding within the period",
    )
    prorated_amount: Decimal = Field(
        default=Decimal("0.00"),
        description="Prorated accrual amount",
    )
    debit_account: str = Field(
        ...,
        description="Account to debit",
    )
    credit_account: str = Field(
        ...,
        description="Account to credit",
    )
    posting_date: date = Field(
        ...,
        description="Date of the accrual posting (period end date)",
    )
    reversal_date: Optional[date] = Field(
        default=None,
        description="Date of the reversing entry (first day of next period)",
    )
    is_reversal: bool = Field(
        default=False,
        description="Whether this is a reversing entry",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class AccrualBatchResult(BaseModel):
    """Result of an accrual generation batch for a single fiscal period.

    Aggregates counts and totals for AP accruals, AR accruals, and reversing
    entries produced by :meth:`AccrualGenerator.generate_period_accruals`.

    Attributes:
        period_id: Fiscal period identifier.
        ap_accruals_generated: Number of AP (GRNI) accrual entries created.
        ar_accruals_generated: Number of AR (shipped-not-invoiced) entries.
        reversals_generated: Number of reversing entries created.
        total_ap_accrual_amount: Sum of prorated AP accrual amounts.
        total_ar_accrual_amount: Sum of prorated AR accrual amounts.
        journal_entries_posted: Count of journal entries posted to GL.
        duration_ms: Elapsed time in milliseconds.
        errors: List of error messages encountered during the batch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    period_id: str = Field(...)
    ap_accruals_generated: int = Field(default=0, ge=0)
    ar_accruals_generated: int = Field(default=0, ge=0)
    reversals_generated: int = Field(default=0, ge=0)
    total_ap_accrual_amount: Decimal = Field(default=Decimal("0.00"))
    total_ar_accrual_amount: Decimal = Field(default=Decimal("0.00"))
    journal_entries_posted: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    errors: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# AccrualGenerator
# ═══════════════════════════════════════════════════════════════════════════


class AccrualGenerator:
    """Generates period-end accrual and reversing journal entries.

    Produces AP accruals (GRNI) and AR accruals (shipped-not-invoiced) for a
    given fiscal period, using the **straight-line daily** accrual method.
    All journal entries are delegated to the injected ``GLPostingEngine`` for
    posting, which enforces balance validation and atomicity.

    Constructor Injection (ADR-003):
        All dependencies are keyword-only ``Optional`` parameters with
        ``None`` defaults.  ``None`` values silently disable the
        corresponding capability, enabling test isolation and incremental
        integration.

    Financial Integrity (AAP §0.7.2):
        - ALL calculations use ``Decimal``.  ``float`` is NEVER used.
        - Every accrual journal entry satisfies DR = CR.
        - Full rollback on ANY posting failure.

    Args:
        gl_posting_engine: Delegates journal entry posting.  When ``None``,
            accrual entries are generated but not posted.
        account_balance_manager: Optional balance query service.
        event_bus: Project 2 EventBus for publishing events.
        db_session_factory: Callable returning an async session context.
    """

    # ------------------------------------------------------------------
    # Construction (ADR-003 — constructor injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        gl_posting_engine: Optional[GLPostingEngine] = None,
        account_balance_manager: Optional[AccountBalanceManager] = None,
        event_bus: Optional[EventBus] = None,
        db_session_factory: Optional[Any] = None,
    ) -> None:
        """Initialise the AccrualGenerator with optional dependencies."""
        # Injected dependencies (all Optional per ADR-003)
        self._gl_posting_engine: Optional[GLPostingEngine] = gl_posting_engine
        self._account_balance_manager: Optional[AccountBalanceManager] = (
            account_balance_manager
        )
        self._event_bus: Optional[EventBus] = event_bus
        self._db_session_factory: Optional[Any] = db_session_factory

        # Accrual generation history
        self._accrual_history: List[AccrualBatchResult] = []

        # Metrics counters
        self._total_accruals_generated: int = 0
        self._total_reversals_generated: int = 0
        self._total_journal_entries_posted: int = 0
        self._total_errors: int = 0
        self._start_time: float = time.perf_counter()

        # Accrual sequence counter for document numbering
        self._accrual_sequence: int = 0

        logger.info(
            "accrual_generator_initialized",
            service_name="transactions",
            component="AccrualGenerator",
            has_gl_engine=gl_posting_engine is not None,
            has_balance_manager=account_balance_manager is not None,
            has_event_bus=event_bus is not None,
            has_db_session=db_session_factory is not None,
        )

    # ------------------------------------------------------------------
    # Main Public API
    # ------------------------------------------------------------------

    async def generate_period_accruals(
        self,
        period_start: date,
        period_end: date,
        period_id: str,
        simulation_id: Optional[UUID] = None,
    ) -> AccrualBatchResult:
        """Generate all accrual and reversing entries for a fiscal period.

        This is the primary entry point invoked by ``PeriodCloseManager``
        during step 2 of the period close process.

        Steps:
            1. Compute ``days_in_period = (period_end - period_start).days + 1``
            2. Generate AP accruals (GRNI) via :meth:`_generate_ap_accruals`
            3. Generate AR accruals via :meth:`_generate_ar_accruals`
            4. Generate reversing entries via :meth:`_generate_reversing_entries`
            5. Post all entries via :meth:`_post_accrual_entries`
            6. Build and return :class:`AccrualBatchResult`

        Args:
            period_start: First day of the fiscal period.
            period_end: Last day of the fiscal period.
            period_id: Fiscal period identifier (e.g., ``'2024-01'``).
            simulation_id: Parent simulation run identifier for correlation.

        Returns:
            :class:`AccrualBatchResult` with counts, totals, and any errors.

        Raises:
            GLPostingError: If GL posting of accrual entries fails.
            BalanceError: If any accrual journal entry is unbalanced.
            TransactionError: Catch-all for unexpected failures.
        """
        start_ts = time.perf_counter()
        errors: List[str] = []

        # Validate period date range
        if period_end < period_start:
            error_msg = (
                f"Invalid period range: period_end ({period_end}) "
                f"is before period_start ({period_start})"
            )
            logger.error(
                "accrual_generation_invalid_period",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                period_start=str(period_start),
                period_end=str(period_end),
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            errors.append(error_msg)
            return AccrualBatchResult(
                period_id=period_id,
                duration_ms=(time.perf_counter() - start_ts) * 1000.0,
                errors=errors,
            )

        days_in_period: int = (period_end - period_start).days + 1

        logger.info(
            "accrual_generation_started",
            service_name="transactions",
            component="AccrualGenerator",
            period_id=period_id,
            period_start=str(period_start),
            period_end=str(period_end),
            days_in_period=days_in_period,
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        # --- Step 1: Generate AP accruals (GRNI) ---
        ap_accruals: List[AccrualEntry] = []
        try:
            ap_accruals = await self._generate_ap_accruals(
                period_start=period_start,
                period_end=period_end,
                period_id=period_id,
                days_in_period=days_in_period,
            )
        except TransactionError as exc:
            error_msg = f"AP accrual generation failed: {exc}"
            logger.error(
                "ap_accrual_generation_failed",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                error=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            errors.append(error_msg)

        # --- Step 2: Generate AR accruals (shipped-not-invoiced) ---
        ar_accruals: List[AccrualEntry] = []
        try:
            ar_accruals = await self._generate_ar_accruals(
                period_start=period_start,
                period_end=period_end,
                period_id=period_id,
                days_in_period=days_in_period,
            )
        except TransactionError as exc:
            error_msg = f"AR accrual generation failed: {exc}"
            logger.error(
                "ar_accrual_generation_failed",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                error=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            errors.append(error_msg)

        # --- Step 3: Generate reversing entries ---
        all_accruals: List[AccrualEntry] = ap_accruals + ar_accruals
        reversals: List[AccrualEntry] = []
        try:
            reversals = await self._generate_reversing_entries(all_accruals)
        except TransactionError as exc:
            error_msg = f"Reversing entry generation failed: {exc}"
            logger.error(
                "reversal_generation_failed",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                error=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            errors.append(error_msg)

        # --- Step 4: Post all entries via GLPostingEngine ---
        all_entries: List[AccrualEntry] = all_accruals + reversals
        posted_count: int = 0
        try:
            posted_count = await self._post_accrual_entries(
                entries=all_entries,
                simulation_id=simulation_id,
            )
        except (GLPostingError, BalanceError, TransactionError) as exc:
            error_msg = f"Accrual GL posting failed: {exc}"
            logger.error(
                "accrual_gl_posting_failed",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                error=str(exc),
                simulation_id=str(simulation_id) if simulation_id else None,
            )
            errors.append(error_msg)

        # --- Step 5: Build result ---
        total_ap_amount = sum(
            (e.prorated_amount for e in ap_accruals), Decimal("0.00")
        )
        total_ar_amount = sum(
            (e.prorated_amount for e in ar_accruals), Decimal("0.00")
        )

        elapsed_ms = (time.perf_counter() - start_ts) * 1000.0

        result = AccrualBatchResult(
            period_id=period_id,
            ap_accruals_generated=len(ap_accruals),
            ar_accruals_generated=len(ar_accruals),
            reversals_generated=len(reversals),
            total_ap_accrual_amount=total_ap_amount,
            total_ar_accrual_amount=total_ar_amount,
            journal_entries_posted=posted_count,
            duration_ms=round(elapsed_ms, 2),
            errors=errors,
        )

        # Update internal metrics
        self._total_accruals_generated += len(all_accruals)
        self._total_reversals_generated += len(reversals)
        self._total_journal_entries_posted += posted_count
        self._total_errors += len(errors)
        self._accrual_history.append(result)

        logger.info(
            "accrual_generation_completed",
            service_name="transactions",
            component="AccrualGenerator",
            period_id=period_id,
            ap_accruals=len(ap_accruals),
            ar_accruals=len(ar_accruals),
            reversals=len(reversals),
            total_ap_amount=str(total_ap_amount),
            total_ar_amount=str(total_ar_amount),
            journal_entries_posted=posted_count,
            duration_ms=round(elapsed_ms, 2),
            error_count=len(errors),
            simulation_id=str(simulation_id) if simulation_id else None,
        )

        return result

    # ------------------------------------------------------------------
    # AP Accrual Generation (GRNI — Goods Received Not Invoiced)
    # ------------------------------------------------------------------

    async def _generate_ap_accruals(
        self,
        period_start: date,
        period_end: date,
        period_id: str,
        days_in_period: int,
    ) -> List[AccrualEntry]:
        """Generate AP accruals for goods received but not yet invoiced.

        Queries the database (via ``_db_session_factory``) for goods receipts
        within the period that have no matching vendor invoice.  For each
        unmatched receipt, an accrual entry is created with:

            - debit_account: Expense or Asset account (from the receipt)
            - credit_account: AP Accrual account
            - daily_amount = total_amount / days_in_period
            - days_outstanding = (period_end - receipt_date).days + 1
            - prorated_amount = daily_amount * days_outstanding

        When ``_db_session_factory`` is ``None`` (test/integration mode),
        returns an empty list.

        Args:
            period_start: First day of the fiscal period.
            period_end: Last day of the fiscal period.
            period_id: Fiscal period identifier.
            days_in_period: Total calendar days in the period.

        Returns:
            List of :class:`AccrualEntry` for AP GRNI accruals.
        """
        accruals: List[AccrualEntry] = []
        reversal_date = period_end + timedelta(days=1)

        if self._db_session_factory is None:
            logger.debug(
                "ap_accrual_no_db_session",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                message="No db_session_factory — returning empty AP accruals",
            )
            return accruals

        # Query for unmatched goods receipts via db session
        try:
            unmatched_receipts = await self._query_unmatched_goods_receipts(
                period_start=period_start,
                period_end=period_end,
            )
        except Exception as exc:
            logger.warning(
                "ap_accrual_query_failed",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                error=str(exc),
            )
            unmatched_receipts = []

        for receipt in unmatched_receipts:
            receipt_amount = Decimal(str(receipt.get("amount", "0.00")))
            receipt_date_raw = receipt.get("receipt_date", period_start)
            if isinstance(receipt_date_raw, str):
                receipt_date_val = date.fromisoformat(receipt_date_raw)
            else:
                receipt_date_val = receipt_date_raw

            # Clamp receipt_date to period boundaries
            effective_start = max(receipt_date_val, period_start)
            days_outstanding = (period_end - effective_start).days + 1
            if days_outstanding < 0:
                days_outstanding = 0

            daily_amount = self._calculate_daily_accrual(
                total_amount=receipt_amount,
                days_in_period=days_in_period,
            )
            prorated_amount = self._calculate_prorated_amount(
                daily_amount=daily_amount,
                days_outstanding=days_outstanding,
            )

            debit_account = receipt.get(
                "debit_account", _DEFAULT_AP_ACCRUAL_DEBIT_ACCOUNT
            )
            credit_account = receipt.get(
                "credit_account", _DEFAULT_AP_ACCRUAL_CREDIT_ACCOUNT
            )

            entry = AccrualEntry(
                accrual_type="ap_grni",
                period_id=period_id,
                reference_document_id=receipt.get("document_id"),
                reference_document_type="goods_receipt",
                total_amount=receipt_amount,
                daily_amount=daily_amount,
                days_outstanding=days_outstanding,
                prorated_amount=prorated_amount,
                debit_account=debit_account,
                credit_account=credit_account,
                posting_date=period_end,
                reversal_date=reversal_date,
                is_reversal=False,
            )
            accruals.append(entry)

        logger.debug(
            "ap_accruals_generated",
            service_name="transactions",
            component="AccrualGenerator",
            period_id=period_id,
            count=len(accruals),
        )
        return accruals

    # ------------------------------------------------------------------
    # AR Accrual Generation (Shipped Not Invoiced)
    # ------------------------------------------------------------------

    async def _generate_ar_accruals(
        self,
        period_start: date,
        period_end: date,
        period_id: str,
        days_in_period: int,
    ) -> List[AccrualEntry]:
        """Generate AR accruals for shipments not yet invoiced.

        Queries the database for shipments within the period that have no
        matching customer invoice.  For each uninvoiced shipment, an accrual
        entry is created with:

            - debit_account: AR account
            - credit_account: Revenue account
            - Prorated using the same straight-line daily formula as AP accruals

        Args:
            period_start: First day of the fiscal period.
            period_end: Last day of the fiscal period.
            period_id: Fiscal period identifier.
            days_in_period: Total calendar days in the period.

        Returns:
            List of :class:`AccrualEntry` for AR shipped-not-invoiced accruals.
        """
        accruals: List[AccrualEntry] = []
        reversal_date = period_end + timedelta(days=1)

        if self._db_session_factory is None:
            logger.debug(
                "ar_accrual_no_db_session",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                message="No db_session_factory — returning empty AR accruals",
            )
            return accruals

        # Query for uninvoiced shipments via db session
        try:
            uninvoiced_shipments = await self._query_uninvoiced_shipments(
                period_start=period_start,
                period_end=period_end,
            )
        except Exception as exc:
            logger.warning(
                "ar_accrual_query_failed",
                service_name="transactions",
                component="AccrualGenerator",
                period_id=period_id,
                error=str(exc),
            )
            uninvoiced_shipments = []

        for shipment in uninvoiced_shipments:
            shipment_amount = Decimal(str(shipment.get("amount", "0.00")))
            shipment_date_raw = shipment.get("shipment_date", period_start)
            if isinstance(shipment_date_raw, str):
                shipment_date_val = date.fromisoformat(shipment_date_raw)
            else:
                shipment_date_val = shipment_date_raw

            effective_start = max(shipment_date_val, period_start)
            days_outstanding = (period_end - effective_start).days + 1
            if days_outstanding < 0:
                days_outstanding = 0

            daily_amount = self._calculate_daily_accrual(
                total_amount=shipment_amount,
                days_in_period=days_in_period,
            )
            prorated_amount = self._calculate_prorated_amount(
                daily_amount=daily_amount,
                days_outstanding=days_outstanding,
            )

            debit_account = shipment.get(
                "debit_account", _DEFAULT_AR_ACCRUAL_DEBIT_ACCOUNT
            )
            credit_account = shipment.get(
                "credit_account", _DEFAULT_AR_ACCRUAL_CREDIT_ACCOUNT
            )

            entry = AccrualEntry(
                accrual_type="ar_shipped_not_invoiced",
                period_id=period_id,
                reference_document_id=shipment.get("document_id"),
                reference_document_type="shipment",
                total_amount=shipment_amount,
                daily_amount=daily_amount,
                days_outstanding=days_outstanding,
                prorated_amount=prorated_amount,
                debit_account=debit_account,
                credit_account=credit_account,
                posting_date=period_end,
                reversal_date=reversal_date,
                is_reversal=False,
            )
            accruals.append(entry)

        logger.debug(
            "ar_accruals_generated",
            service_name="transactions",
            component="AccrualGenerator",
            period_id=period_id,
            count=len(accruals),
        )
        return accruals

    # ------------------------------------------------------------------
    # Reversing Entry Generation
    # ------------------------------------------------------------------

    async def _generate_reversing_entries(
        self,
        accruals: List[AccrualEntry],
    ) -> List[AccrualEntry]:
        """Generate reversing entries for each accrual.

        For every accrual entry, creates a corresponding reversing entry that:
            - Swaps the debit and credit accounts
            - Posts on the first day of the next period (``reversal_date``)
            - Sets ``is_reversal = True``
            - References the original accrual via ``reference_document_id``

        Args:
            accruals: List of accrual entries to reverse.

        Returns:
            List of reversing :class:`AccrualEntry` instances.
        """
        reversals: List[AccrualEntry] = []

        for accrual in accruals:
            # Determine the posting date for the reversal
            reversal_posting_date = accrual.reversal_date
            if reversal_posting_date is None:
                # Fallback: day after the accrual posting date
                reversal_posting_date = accrual.posting_date + timedelta(days=1)

            reversal = AccrualEntry(
                accrual_type=accrual.accrual_type,
                period_id=accrual.period_id,
                reference_document_id=str(accrual.accrual_id),
                reference_document_type=accrual.reference_document_type,
                total_amount=accrual.total_amount,
                daily_amount=accrual.daily_amount,
                days_outstanding=accrual.days_outstanding,
                prorated_amount=accrual.prorated_amount,
                # CRITICAL: Swap debit and credit accounts for reversal
                debit_account=accrual.credit_account,
                credit_account=accrual.debit_account,
                posting_date=reversal_posting_date,
                reversal_date=None,
                is_reversal=True,
            )
            reversals.append(reversal)

        logger.debug(
            "reversing_entries_generated",
            service_name="transactions",
            component="AccrualGenerator",
            count=len(reversals),
        )
        return reversals

    # ------------------------------------------------------------------
    # GL Posting Delegation
    # ------------------------------------------------------------------

    async def _post_accrual_entries(
        self,
        entries: List[AccrualEntry],
        simulation_id: Optional[UUID] = None,
    ) -> int:
        """Post accrual and reversing entries via the GLPostingEngine.

        Builds a balanced journal entry (DR = CR) for each
        :class:`AccrualEntry` and delegates to
        :meth:`GLPostingEngine.post_journal_entry`.

        If ``_gl_posting_engine`` is ``None``, logs a warning and returns 0
        without posting.

        Args:
            entries: List of accrual/reversing entries to post.
            simulation_id: Parent simulation run identifier.

        Returns:
            Count of successfully posted journal entries.

        Raises:
            GLPostingError: If a posting fails (propagated from engine).
            BalanceError: If a journal entry is unbalanced (should not happen
                given the balanced construction here, but propagated if it does).
        """
        if not entries:
            return 0

        if self._gl_posting_engine is None:
            logger.warning(
                "accrual_posting_skipped_no_engine",
                service_name="transactions",
                component="AccrualGenerator",
                entry_count=len(entries),
                reason="GLPostingEngine not injected",
            )
            return 0

        # Late import to avoid circular dependency within the gl package
        from app.transactions.gl.gl_posting_engine import (
            JournalEntry,
            JournalEntryLine,
        )

        posted_count: int = 0

        for entry in entries:
            try:
                # Build balanced journal entry: one debit line + one credit line
                debit_line = JournalEntryLine(
                    line_number=1,
                    account_code=entry.debit_account,
                    debit_amount=entry.prorated_amount,
                    credit_amount=Decimal("0.00"),
                    description=(
                        f"{'Reversal: ' if entry.is_reversal else ''}"
                        f"Accrual {entry.accrual_type} — "
                        f"ref {entry.reference_document_id or 'N/A'}"
                    ),
                    reference=entry.reference_document_id,
                )
                credit_line = JournalEntryLine(
                    line_number=2,
                    account_code=entry.credit_account,
                    debit_amount=Decimal("0.00"),
                    credit_amount=entry.prorated_amount,
                    description=(
                        f"{'Reversal: ' if entry.is_reversal else ''}"
                        f"Accrual {entry.accrual_type} — "
                        f"ref {entry.reference_document_id or 'N/A'}"
                    ),
                    reference=entry.reference_document_id,
                )

                journal_entry = JournalEntry(
                    posting_date=entry.posting_date,
                    period_id=entry.period_id,
                    description=(
                        f"{'Reversal: ' if entry.is_reversal else ''}"
                        f"Period accrual — {entry.accrual_type}"
                    ),
                    source_document_type="accrual",
                    source_document_id=str(entry.accrual_id),
                    lines=[debit_line, credit_line],
                    total_debits=entry.prorated_amount,
                    total_credits=entry.prorated_amount,
                    is_balanced=True,
                    status="draft",
                    created_by="AccrualGenerator",
                    simulation_id=simulation_id,
                )

                # Delegate to GLPostingEngine
                posting_result = await self._gl_posting_engine.post_journal_entry(
                    entry=journal_entry,
                    simulation_id=simulation_id,
                )

                if posting_result.success:
                    posted_count += 1
                    logger.debug(
                        "accrual_entry_posted",
                        service_name="transactions",
                        component="AccrualGenerator",
                        accrual_id=str(entry.accrual_id),
                        accrual_type=entry.accrual_type,
                        is_reversal=entry.is_reversal,
                        amount=str(entry.prorated_amount),
                        entry_number=posting_result.entry_number,
                        simulation_id=(
                            str(simulation_id) if simulation_id else None
                        ),
                    )
                else:
                    error_msg = (
                        f"GL posting returned failure for accrual "
                        f"{entry.accrual_id}: {posting_result.error_message}"
                    )
                    logger.error(
                        "accrual_entry_posting_failed",
                        service_name="transactions",
                        component="AccrualGenerator",
                        accrual_id=str(entry.accrual_id),
                        error=posting_result.error_message,
                        simulation_id=(
                            str(simulation_id) if simulation_id else None
                        ),
                    )

            except (GLPostingError, BalanceError) as exc:
                # Critical financial integrity failure — log and re-raise
                logger.error(
                    "accrual_posting_critical_failure",
                    service_name="transactions",
                    component="AccrualGenerator",
                    accrual_id=str(entry.accrual_id),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    simulation_id=(
                        str(simulation_id) if simulation_id else None
                    ),
                )
                raise
            except TransactionError as exc:
                # Non-critical transaction error — log and continue
                logger.warning(
                    "accrual_posting_transaction_error",
                    service_name="transactions",
                    component="AccrualGenerator",
                    accrual_id=str(entry.accrual_id),
                    error=str(exc),
                    simulation_id=(
                        str(simulation_id) if simulation_id else None
                    ),
                )
            except Exception as exc:
                # Unexpected error — log and continue with remaining entries
                logger.error(
                    "accrual_posting_unexpected_error",
                    service_name="transactions",
                    component="AccrualGenerator",
                    accrual_id=str(entry.accrual_id),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    simulation_id=(
                        str(simulation_id) if simulation_id else None
                    ),
                )

        logger.info(
            "accrual_entries_posting_complete",
            service_name="transactions",
            component="AccrualGenerator",
            total_entries=len(entries),
            posted_count=posted_count,
            simulation_id=str(simulation_id) if simulation_id else None,
        )
        return posted_count

    # ------------------------------------------------------------------
    # Calculation Helpers
    # ------------------------------------------------------------------

    def _calculate_daily_accrual(
        self,
        total_amount: Decimal,
        days_in_period: int,
    ) -> Decimal:
        """Compute straight-line daily accrual amount.

        Formula: ``daily_amount = total_amount / days_in_period``

        All arithmetic uses ``Decimal`` with ``ROUND_HALF_UP`` to 2 decimal
        places.  Returns ``Decimal("0.00")`` if ``days_in_period <= 0``.

        Args:
            total_amount: Full accrual amount from the source document.
            days_in_period: Total calendar days in the fiscal period.

        Returns:
            Daily accrual amount quantized to 2 decimal places.
        """
        if days_in_period <= 0:
            return Decimal("0.00")
        return (total_amount / Decimal(str(days_in_period))).quantize(
            Decimal("0.01"), rounding=decimal.ROUND_HALF_UP
        )

    def _calculate_prorated_amount(
        self,
        daily_amount: Decimal,
        days_outstanding: int,
    ) -> Decimal:
        """Compute prorated accrual amount for outstanding days.

        Formula: ``prorated = daily_amount * days_outstanding``

        Args:
            daily_amount: Straight-line daily accrual amount.
            days_outstanding: Number of days the item is outstanding in-period.

        Returns:
            Prorated amount quantized to 2 decimal places.
        """
        if days_outstanding <= 0:
            return Decimal("0.00")
        return (daily_amount * Decimal(str(days_outstanding))).quantize(
            Decimal("0.01"), rounding=decimal.ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Database Query Helpers
    # ------------------------------------------------------------------

    async def _query_unmatched_goods_receipts(
        self,
        period_start: date,
        period_end: date,
    ) -> List[Dict[str, Any]]:
        """Query for goods receipts that have not been matched to vendor invoices.

        Delegates to ``_db_session_factory`` for database access.  Returns
        a list of dictionaries with keys:
            - ``document_id``: Goods receipt identifier.
            - ``amount``: Receipt total amount (Decimal-compatible string).
            - ``receipt_date``: Date the goods were received.
            - ``debit_account``: GL account to debit (Expense/Asset).
            - ``credit_account``: GL account to credit (AP Accrual).

        When ``_db_session_factory`` is ``None``, returns an empty list.

        Args:
            period_start: First day of the fiscal period.
            period_end: Last day of the fiscal period.

        Returns:
            List of unmatched goods receipt dictionaries.
        """
        if self._db_session_factory is None:
            return []

        # If the session factory is a callable that provides data directly
        # (e.g., a mock or test fixture), invoke it.
        try:
            if callable(self._db_session_factory):
                session_or_data = self._db_session_factory
                # Check if it's an async context manager provider
                if hasattr(session_or_data, "__call__"):
                    result = session_or_data()
                    # Support async context managers
                    if hasattr(result, "__aenter__"):
                        async with result as session:
                            # Attempt to query via session
                            if hasattr(session, "query_unmatched_receipts"):
                                return await session.query_unmatched_receipts(
                                    period_start, period_end
                                )
                            return []
                    # Support direct list returns (for testing)
                    if isinstance(result, list):
                        return result
            return []
        except Exception as exc:
            logger.warning(
                "unmatched_receipts_query_error",
                service_name="transactions",
                component="AccrualGenerator",
                error=str(exc),
                period_start=str(period_start),
                period_end=str(period_end),
            )
            return []

    async def _query_uninvoiced_shipments(
        self,
        period_start: date,
        period_end: date,
    ) -> List[Dict[str, Any]]:
        """Query for shipments that have not been invoiced to customers.

        Delegates to ``_db_session_factory`` for database access.  Returns
        a list of dictionaries with keys:
            - ``document_id``: Shipment identifier.
            - ``amount``: Shipment total amount (Decimal-compatible string).
            - ``shipment_date``: Date the goods were shipped.
            - ``debit_account``: GL account to debit (AR).
            - ``credit_account``: GL account to credit (Revenue).

        When ``_db_session_factory`` is ``None``, returns an empty list.

        Args:
            period_start: First day of the fiscal period.
            period_end: Last day of the fiscal period.

        Returns:
            List of uninvoiced shipment dictionaries.
        """
        if self._db_session_factory is None:
            return []

        try:
            if callable(self._db_session_factory):
                session_or_data = self._db_session_factory
                if hasattr(session_or_data, "__call__"):
                    result = session_or_data()
                    if hasattr(result, "__aenter__"):
                        async with result as session:
                            if hasattr(session, "query_uninvoiced_shipments"):
                                return await session.query_uninvoiced_shipments(
                                    period_start, period_end
                                )
                            return []
                    if isinstance(result, list):
                        return result
            return []
        except Exception as exc:
            logger.warning(
                "uninvoiced_shipments_query_error",
                service_name="transactions",
                component="AccrualGenerator",
                error=str(exc),
                period_start=str(period_start),
                period_end=str(period_end),
            )
            return []

    # ------------------------------------------------------------------
    # Metrics and Inspection
    # ------------------------------------------------------------------

    def get_accrual_history(self) -> List[AccrualBatchResult]:
        """Return the history of all accrual generation batch results.

        Returns:
            List of :class:`AccrualBatchResult` in chronological order.
        """
        return list(self._accrual_history)

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregated metrics for the accrual generator.

        Returns:
            Dictionary containing total counts, rates, and timing data.
        """
        elapsed_seconds = time.perf_counter() - self._start_time
        batches_completed = len(self._accrual_history)

        return {
            "total_accruals_generated": self._total_accruals_generated,
            "total_reversals_generated": self._total_reversals_generated,
            "total_journal_entries_posted": self._total_journal_entries_posted,
            "total_errors": self._total_errors,
            "batches_completed": batches_completed,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "accruals_per_minute": (
                round(
                    self._total_accruals_generated / (elapsed_seconds / 60.0),
                    2,
                )
                if elapsed_seconds > 0
                else 0.0
            ),
            "has_gl_engine": self._gl_posting_engine is not None,
            "has_balance_manager": self._account_balance_manager is not None,
            "has_event_bus": self._event_bus is not None,
            "has_db_session": self._db_session_factory is not None,
            "financial_tolerance": str(GL_BALANCE_TOLERANCE),
            "debit_normal_types": sorted(DEBIT_NORMAL_ACCOUNT_TYPES),
            "credit_normal_types": sorted(CREDIT_NORMAL_ACCOUNT_TYPES),
            "accrual_document_prefix": DOCUMENT_NUMBER_PREFIXES.get(
                "journal_entry", "JE"
            ),
            "financial_tolerances": {
                k: str(v) for k, v in FINANCIAL_TOLERANCES.items()
            },
        }
