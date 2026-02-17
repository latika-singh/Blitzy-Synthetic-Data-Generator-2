"""Bank Simulator — Simulates bank interactions with the ERP system.

This module implements the ``BankSimulator`` class responsible for generating
realistic bank statements for month-end reconciliation and simulating payment
processing (check clearing, ACH transfers, wire transfers).  Banks represent
the financial-institution layer that processes payments, generates statements,
and provides banking services to the simulated enterprise.

Key capabilities
----------------
- **Bank Statement Generation**: Monthly statements with debits, credits, fees,
  interest, running balances, and bank-style reference numbers.
- **Payment Processing Simulation**: Clearing-time modelling for checks (1–3
  business days), ACH transfers (1–2 business days), and wire transfers (same
  day), including configurable payment-failure rates.
- **Event Publishing**: Publishes ``DocumentGenerated`` events through the
  injected ``EventBus`` when bank statements are produced.

All data contracts use **Pydantic V2** ``BaseModel`` for subsystem-boundary
validation (AAP §0.7.1).  Logging uses **structlog** to stdout in structured
JSON format (AAP §0.7.6).  Reproducible randomness is achieved through
``numpy.random.RandomState`` seeding.

References
----------
- README.md lines 456–486: Bank interaction specification
- README.md lines 463–467: Entity pool definitions
- AAP §0.5.1 Group 7: BankSimulator deliverable
- AAP §0.7.1: Constructor injection for all dependencies
- AAP §0.7.3: Fully async with no blocking I/O
- AAP §0.7.6: Structured JSON logging via structlog
"""

from __future__ import annotations

import asyncio
import calendar
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import structlog
from pydantic import BaseModel, Field

from app.external_world.behavior_profiles import BehaviorProfile

if TYPE_CHECKING:
    from app.statistical.amount_distributions import AmountDistribution
    from app.events.event_bus import EventBus

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Failure reason pool — used when a simulated payment fails
# ---------------------------------------------------------------------------
_FAILURE_REASONS: List[str] = [
    "insufficient_funds",
    "account_closed",
    "invalid_account",
    "payment_stopped",
    "bank_error",
]


# ===========================================================================
# Pydantic V2 Configuration Model
# ===========================================================================


class BankSimulatorConfig(BaseModel):
    """Validated configuration for :class:`BankSimulator`.

    All fields carry sensible defaults so that a ``BankSimulator`` can be
    instantiated with zero arguments for development and unit testing.

    Attributes:
        check_clearing_days_min: Minimum business days for check clearing.
        check_clearing_days_max: Maximum business days for check clearing.
        ach_clearing_days_min: Minimum business days for ACH clearing.
        ach_clearing_days_max: Maximum business days for ACH clearing.
        wire_clearing_days: Business days for wire transfer (0 = same day).
        payment_failure_rate: Fraction of payments that randomly fail (0.0–1.0).
        monthly_service_fee: Monthly bank service fee in USD.
        nsf_fee: Insufficient-funds fee in USD.
        interest_rate_annual: Annual interest rate on positive balances.
        payment_methods: Supported payment method identifiers.
    """

    check_clearing_days_min: int = Field(
        default=1, ge=0, description="Minimum check clearing days"
    )
    check_clearing_days_max: int = Field(
        default=3, ge=1, description="Maximum check clearing days"
    )
    ach_clearing_days_min: int = Field(
        default=1, ge=0, description="Minimum ACH clearing days"
    )
    ach_clearing_days_max: int = Field(
        default=2, ge=1, description="Maximum ACH clearing days"
    )
    wire_clearing_days: int = Field(
        default=0, ge=0, description="Wire transfer clearing days (0 = same day)"
    )
    payment_failure_rate: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Fraction of payments that fail"
    )
    monthly_service_fee: float = Field(
        default=25.0, ge=0.0, description="Monthly bank service fee (USD)"
    )
    nsf_fee: float = Field(
        default=35.0, ge=0.0, description="Insufficient-funds fee (USD)"
    )
    interest_rate_annual: float = Field(
        default=0.01, ge=0.0, description="Annual interest rate on deposits"
    )
    payment_methods: List[str] = Field(
        default_factory=lambda: ["check", "ach", "wire"],
        description="Supported payment method identifiers",
    )


# ===========================================================================
# Bank Data Models — Pydantic V2 at subsystem boundaries (AAP §0.7.1)
# ===========================================================================


class BankStatementEntry(BaseModel):
    """A single line item on a bank statement.

    Amounts follow banking convention: positive for credits (incoming funds),
    negative for debits (outgoing payments and fees).

    Attributes:
        entry_id: Unique identifier for this entry (UUID string).
        date: Calendar date of the entry.
        description: Human-readable description of the entry.
        amount: Signed amount — positive for credits, negative for debits.
        entry_type: One of ``"debit"``, ``"credit"``, ``"fee"``, ``"interest"``.
        reference_number: Bank-style reference string.
        running_balance: Account balance after this entry is applied.
        payment_method: Optional payment method (check/ach/wire).
    """

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    date: date
    description: str
    amount: float
    entry_type: str
    reference_number: str
    running_balance: float
    payment_method: Optional[str] = None


class BankStatement(BaseModel):
    """Complete monthly bank statement for a single account.

    Attributes:
        statement_id: Unique identifier for the statement (UUID string).
        bank_id: Identifier of the issuing bank entity.
        account_number: Account number on the statement.
        statement_period_start: First day of the statement period.
        statement_period_end: Last day of the statement period.
        opening_balance: Account balance at the start of the period.
        closing_balance: Account balance at the end of the period.
        total_debits: Sum of all debit (negative) entries as a positive number.
        total_credits: Sum of all credit (positive) entries.
        total_fees: Sum of all fee entries as a positive number.
        total_interest: Sum of all interest entries.
        entries: Chronologically ordered list of statement entries.
        generated_date: Date the statement was generated.
    """

    statement_id: str = Field(default_factory=lambda: str(uuid4()))
    bank_id: str
    account_number: str
    statement_period_start: date
    statement_period_end: date
    opening_balance: float
    closing_balance: float
    total_debits: float
    total_credits: float
    total_fees: float
    total_interest: float
    entries: List[BankStatementEntry]
    generated_date: date


class PaymentProcessingResult(BaseModel):
    """Result of a simulated payment-processing operation.

    Tracks the full lifecycle of a payment from initiation through
    clearing or failure.

    Attributes:
        result_id: Unique identifier for this result (UUID string).
        payment_id: Identifier of the original payment request.
        bank_id: Identifier of the processing bank.
        payment_method: Payment method used (check/ach/wire).
        amount: Payment amount in USD.
        initiated_date: Date the payment was initiated.
        cleared_date: Date the payment cleared (``None`` if not yet cleared).
        is_successful: Whether the payment completed successfully.
        failure_reason: Explanation if the payment failed; ``None`` on success.
        status: Lifecycle status — ``"pending"``, ``"cleared"``, ``"failed"``,
            or ``"returned"``.
    """

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    payment_id: str
    bank_id: str
    payment_method: str
    amount: float
    initiated_date: date
    cleared_date: Optional[date] = None
    is_successful: bool = True
    failure_reason: Optional[str] = None
    status: str = "pending"


# ===========================================================================
# BankSimulator
# ===========================================================================


class BankSimulator:
    """Simulates bank interactions with the ERP system.

    Generates realistic monthly bank statements (with debits, credits, fees,
    interest, running balances) and processes payments with method-dependent
    clearing times and configurable failure rates.

    Constructor injection is used for **all** dependencies (AAP §0.7.1):

    * ``config`` — operational parameters (clearing days, fees, rates).
    * ``amount_distribution`` — optional statistical model for amount sampling.
    * ``event_bus`` — optional event bus for publishing ``DocumentGenerated``
      events when statements are produced.
    * ``seed`` — optional RNG seed for reproducible simulations.

    All public methods are ``async`` with zero blocking I/O (AAP §0.7.3).

    Args:
        config: Optional :class:`BankSimulatorConfig`.  Defaults apply when
            ``None`` is provided.
        amount_distribution: Optional :class:`AmountDistribution` instance for
            transaction-amount sampling during statement generation.
        event_bus: Optional :class:`EventBus` for publishing events.
        seed: Optional integer seed for ``numpy.random.RandomState``.
    """

    def __init__(
        self,
        config: Optional[BankSimulatorConfig] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        event_bus: Optional[EventBus] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config: BankSimulatorConfig = config or BankSimulatorConfig()
        self.amount_distribution: Optional[AmountDistribution] = amount_distribution
        self.event_bus: Optional[EventBus] = event_bus
        self.rng: np.random.RandomState = np.random.RandomState(seed)

        logger.info(
            "bank_simulator_initialized",
            payment_failure_rate=self.config.payment_failure_rate,
            monthly_service_fee=self.config.monthly_service_fee,
            interest_rate_annual=self.config.interest_rate_annual,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Bank Statement Generation
    # ------------------------------------------------------------------

    async def generate_bank_statement(
        self,
        bank: Dict[str, Any],
        transactions: List[Dict[str, Any]],
        statement_month: date,
        opening_balance: float = 0.0,
    ) -> BankStatement:
        """Generate a monthly bank statement for a single bank account.

        Builds chronologically-ordered entries from the supplied transactions,
        appends a monthly service-fee debit and an interest credit, and
        computes running balances and summary totals.

        When an ``event_bus`` is available a ``DocumentGenerated`` event is
        published for downstream consumers.

        Args:
            bank: Dictionary with at least ``"bank_id"`` and
                ``"account_number"`` keys describing the bank entity.
            transactions: List of transaction dictionaries.  Each must contain
                ``"date"`` (:class:`date`), ``"amount"`` (:class:`float`),
                ``"description"`` (:class:`str`), ``"type"`` (``"debit"`` or
                ``"credit"``), and optionally ``"payment_method"``.
            statement_month: Any date within the statement month (the
                first and last days of that month are derived automatically).
            opening_balance: Account balance carried forward from the
                previous period.  Defaults to ``0.0``.

        Returns:
            A fully-populated :class:`BankStatement` instance.
        """
        bank_id: str = str(bank.get("bank_id", "unknown"))
        account_number: str = str(bank.get("account_number", "0000000000"))

        # Derive statement period boundaries
        period_start = statement_month.replace(day=1)
        last_day = calendar.monthrange(statement_month.year, statement_month.month)[1]
        period_end = statement_month.replace(day=last_day)

        # -- Build transaction entries (sorted chronologically) ----------------
        raw_entries: List[Dict[str, Any]] = []
        for txn in transactions:
            txn_date = txn.get("date", period_start)
            if isinstance(txn_date, str):
                txn_date = date.fromisoformat(txn_date)
            txn_amount = float(txn.get("amount", 0.0))
            txn_type = str(txn.get("type", "debit")).lower()
            txn_desc = str(txn.get("description", "Transaction"))
            txn_method = txn.get("payment_method")

            # Banking convention: debits are negative, credits positive
            if txn_type == "debit":
                signed_amount = -abs(txn_amount)
                entry_type = "debit"
            else:
                signed_amount = abs(txn_amount)
                entry_type = "credit"

            raw_entries.append(
                {
                    "date": txn_date,
                    "description": txn_desc,
                    "amount": signed_amount,
                    "entry_type": entry_type,
                    "payment_method": txn_method,
                }
            )

        # -- Append monthly service fee (debit) --------------------------------
        fee_amount = self.config.monthly_service_fee
        if fee_amount > 0:
            raw_entries.append(
                {
                    "date": period_end,
                    "description": "Monthly Service Fee",
                    "amount": -fee_amount,
                    "entry_type": "fee",
                    "payment_method": None,
                }
            )

        # -- Calculate interest on average daily balance -----------------------
        #    Approximate average balance = opening + half the net movement
        net_movement = sum(e["amount"] for e in raw_entries)
        approx_avg_balance = max(opening_balance + net_movement / 2.0, 0.0)
        interest_amount = self._calculate_interest(approx_avg_balance)

        if interest_amount > 0:
            raw_entries.append(
                {
                    "date": period_end,
                    "description": "Interest Credit",
                    "amount": interest_amount,
                    "entry_type": "interest",
                    "payment_method": None,
                }
            )

        # -- Sort entries chronologically and compute running balances ----------
        raw_entries.sort(key=lambda e: e["date"])

        entries: List[BankStatementEntry] = []
        running_balance = opening_balance
        total_debits = 0.0
        total_credits = 0.0
        total_fees = 0.0
        total_interest = 0.0

        for raw in raw_entries:
            running_balance += raw["amount"]
            ref_number = self._generate_reference_number()

            entry = BankStatementEntry(
                date=raw["date"],
                description=raw["description"],
                amount=raw["amount"],
                entry_type=raw["entry_type"],
                reference_number=ref_number,
                running_balance=round(running_balance, 2),
                payment_method=raw.get("payment_method"),
            )
            entries.append(entry)

            # Accumulate totals
            if raw["entry_type"] == "debit":
                total_debits += abs(raw["amount"])
            elif raw["entry_type"] == "credit":
                total_credits += raw["amount"]
            elif raw["entry_type"] == "fee":
                total_fees += abs(raw["amount"])
            elif raw["entry_type"] == "interest":
                total_interest += raw["amount"]

        closing_balance = round(
            opening_balance + total_credits - total_debits - total_fees + total_interest,
            2,
        )

        statement = BankStatement(
            bank_id=bank_id,
            account_number=account_number,
            statement_period_start=period_start,
            statement_period_end=period_end,
            opening_balance=opening_balance,
            closing_balance=closing_balance,
            total_debits=round(total_debits, 2),
            total_credits=round(total_credits, 2),
            total_fees=round(total_fees, 2),
            total_interest=round(total_interest, 2),
            entries=entries,
            generated_date=period_end,
        )

        # -- Publish DocumentGenerated event if event_bus is available ---------
        if self.event_bus is not None:
            try:
                from app.events.event_types import DocumentGenerated

                event = DocumentGenerated(
                    payload={
                        "document_id": statement.statement_id,
                        "document_type": "bank_statement",
                        "bank_id": bank_id,
                        "account_number": account_number,
                        "statement_period_start": period_start.isoformat(),
                        "statement_period_end": period_end.isoformat(),
                        "closing_balance": closing_balance,
                        "entry_count": len(entries),
                    },
                )
                await self.event_bus.publish(event)
            except Exception as exc:
                logger.warning(
                    "bank_statement_event_publish_failed",
                    bank_id=bank_id,
                    error=str(exc),
                )

        logger.info(
            "bank_statement_generated",
            bank_id=bank_id,
            period=f"{period_start.isoformat()} to {period_end.isoformat()}",
            entry_count=len(entries),
            closing_balance=closing_balance,
        )

        return statement

    async def generate_monthly_statements(
        self,
        banks: List[Dict[str, Any]],
        all_transactions: Dict[str, List[Dict[str, Any]]],
        statement_month: date,
    ) -> List[BankStatement]:
        """Generate bank statements for multiple banks in a single period.

        For each bank in *banks*, the method retrieves that bank's transaction
        list from *all_transactions* (keyed by ``bank_id``) and delegates to
        :meth:`generate_bank_statement`.

        Args:
            banks: List of bank entity dictionaries.  Each must contain at
                least a ``"bank_id"`` key.
            all_transactions: Mapping of ``bank_id`` → list of transaction
                dictionaries for the statement period.
            statement_month: Any date within the target month.

        Returns:
            List of :class:`BankStatement` instances, one per bank.
        """
        statements: List[BankStatement] = []

        for bank in banks:
            bank_id = str(bank.get("bank_id", "unknown"))
            bank_txns = all_transactions.get(bank_id, [])
            opening = float(bank.get("opening_balance", 0.0))

            statement = await self.generate_bank_statement(
                bank=bank,
                transactions=bank_txns,
                statement_month=statement_month,
                opening_balance=opening,
            )
            statements.append(statement)

        return statements

    # ------------------------------------------------------------------
    # Payment Processing Simulation
    # ------------------------------------------------------------------

    async def process_payment(
        self,
        bank: Dict[str, Any],
        payment: Dict[str, Any],
        initiation_date: date,
    ) -> PaymentProcessingResult:
        """Simulate the processing of a single payment through a bank.

        Determines clearing time based on payment method, applies weekend
        adjustment, and randomly determines whether the payment fails
        according to the configured ``payment_failure_rate``.

        Args:
            bank: Bank entity dictionary (must contain ``"bank_id"``).
            payment: Payment dictionary with at least ``"payment_id"``,
                ``"amount"``, and optionally ``"payment_method"`` keys.
            initiation_date: Calendar date the payment was initiated.

        Returns:
            A :class:`PaymentProcessingResult` describing the outcome.
        """
        bank_id = str(bank.get("bank_id", "unknown"))
        payment_id = str(payment.get("payment_id", str(uuid4())))
        amount = float(payment.get("amount", 0.0))
        method = str(payment.get("payment_method", "check")).lower()

        # Ensure the method is recognised; default to check
        if method not in self.config.payment_methods:
            method = "check"

        # Calculate clearing days and adjusted clearing date
        clearing_days = self._calculate_clearing_days(method)
        raw_cleared = initiation_date + timedelta(days=clearing_days)
        cleared_date = self._adjust_for_weekends(raw_cleared)

        # Determine payment success / failure
        is_failed = float(self.rng.random()) < self.config.payment_failure_rate
        failure_reason: Optional[str] = None
        status: str

        if is_failed:
            failure_reason = _FAILURE_REASONS[
                int(self.rng.randint(0, len(_FAILURE_REASONS)))
            ]
            status = "failed"
            is_successful = False
            result_cleared_date: Optional[date] = None
        else:
            status = "cleared"
            is_successful = True
            result_cleared_date = cleared_date

        result = PaymentProcessingResult(
            payment_id=payment_id,
            bank_id=bank_id,
            payment_method=method,
            amount=amount,
            initiated_date=initiation_date,
            cleared_date=result_cleared_date,
            is_successful=is_successful,
            failure_reason=failure_reason,
            status=status,
        )

        logger.info(
            "payment_processed",
            bank_id=bank_id,
            method=method,
            amount=amount,
            clearing_days=clearing_days,
            success=is_successful,
        )

        return result

    async def process_payments_batch(
        self,
        bank: Dict[str, Any],
        payments: List[Dict[str, Any]],
        initiation_date: date,
    ) -> List[PaymentProcessingResult]:
        """Process a batch of payments through a single bank.

        Iterates over *payments* and delegates each to
        :meth:`process_payment`.  Target performance is < 2 seconds for
        entity-response latency (AAP criterion #18).

        Args:
            bank: Bank entity dictionary (must contain ``"bank_id"``).
            payments: List of payment dictionaries.
            initiation_date: Calendar date the batch was initiated.

        Returns:
            List of :class:`PaymentProcessingResult` instances.
        """
        results: List[PaymentProcessingResult] = []
        for payment in payments:
            result = await self.process_payment(
                bank=bank,
                payment=payment,
                initiation_date=initiation_date,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Utility / Private Methods
    # ------------------------------------------------------------------

    def _calculate_clearing_days(self, payment_method: str) -> int:
        """Return the number of clearing days for a given payment method.

        Uses the configured min/max ranges for check and ACH methods and
        a fixed value for wire transfers.

        Args:
            payment_method: One of ``"check"``, ``"ach"``, or ``"wire"``.

        Returns:
            Integer number of business days until the payment clears.
        """
        method = payment_method.lower()
        if method == "wire":
            return self.config.wire_clearing_days
        if method == "ach":
            return int(
                self.rng.randint(
                    self.config.ach_clearing_days_min,
                    self.config.ach_clearing_days_max + 1,
                )
            )
        # Default to check
        return int(
            self.rng.randint(
                self.config.check_clearing_days_min,
                self.config.check_clearing_days_max + 1,
            )
        )

    def _adjust_for_weekends(self, dt: date) -> date:
        """Shift a date forward to the next Monday if it falls on a weekend.

        Saturday (weekday 5) → Monday (+2 days).
        Sunday (weekday 6) → Monday (+1 day).

        Args:
            dt: The date to adjust.

        Returns:
            The adjusted date (unchanged if already a weekday).
        """
        weekday = dt.weekday()
        if weekday == 5:  # Saturday
            return dt + timedelta(days=2)
        if weekday == 6:  # Sunday
            return dt + timedelta(days=1)
        return dt

    def _calculate_interest(self, average_balance: float) -> float:
        """Calculate monthly interest on an average daily balance.

        Monthly interest = average_balance × (annual_rate / 12), rounded
        to two decimal places.  Returns ``0.0`` when the balance is zero or
        negative.

        Args:
            average_balance: Average daily account balance for the month.

        Returns:
            Monthly interest amount (always ≥ 0.0).
        """
        if average_balance <= 0:
            return 0.0
        monthly_rate = self.config.interest_rate_annual / 12.0
        return round(average_balance * monthly_rate, 2)

    def _generate_reference_number(self) -> str:
        """Generate a bank-style reference number.

        Format: ``"REF"`` followed by a 10-digit zero-padded random integer.
        The generator uses the instance-level ``numpy.random.RandomState``
        for reproducibility when seeded.

        Returns:
            A string like ``"REF0042837491"``.
        """
        digits = int(self.rng.randint(0, 10_000_000_000))
        return f"REF{digits:010d}"
