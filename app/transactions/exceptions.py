"""Custom exception hierarchy for the P3 Transaction Workflows & Discrepancies system.

Defines 10 exception classes organized in a single-root hierarchy under
:class:`TransactionError`. Each exception type corresponds to a specific
failure domain within the transaction generation pipeline, enabling targeted
catch/retry logic per AAP Section 0.7.4.

Exception Hierarchy::

    TransactionError (base)
    ├── TransactionGenerationError   — error during transaction generation
    ├── BalanceError                  — GL entry doesn't balance (DR ≠ CR)
    ├── ThreeWayMatchError            — three-way match validation failure
    ├── DiscrepancyInjectionError     — error during discrepancy injection
    ├── ReworkLoopError               — error during rework loop processing
    ├── PeriodClosedError             — attempt to post to a closed period
    ├── GLPostingError                — error during GL journal entry posting
    ├── PaymentAllocationError        — error during FIFO payment allocation
    └── ConcurrencyError              — SELECT FOR UPDATE conflict

Each exception carries an optional ``details`` dictionary for structured
context that can be included in structlog entries without credential leakage.

References:
    - AAP Section 0.1.2 (User example — Exception Hierarchy)
    - AAP Section 0.7.4 (Error Handling Conventions)
    - app/errors/error_handlers.py (LLMError, DatabaseError sentinel pattern)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "TransactionError",
    "TransactionGenerationError",
    "BalanceError",
    "ThreeWayMatchError",
    "DiscrepancyInjectionError",
    "ReworkLoopError",
    "PeriodClosedError",
    "GLPostingError",
    "PaymentAllocationError",
    "ConcurrencyError",
]


# ═══════════════════════════════════════════════════════════════════════════
# Base Exception
# ═══════════════════════════════════════════════════════════════════════════


class TransactionError(Exception):
    """Base exception for all transaction errors in the P3 system.

    All P3 exception types inherit from this class, enabling callers to
    catch ``TransactionError`` as a broad catch-all for any transaction-related
    failure while still allowing targeted handling of specific subtypes.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary carrying structured context for logging.
            Values in this dictionary are safe for inclusion in structlog
            entries — callers must ensure no credentials are passed.

    Example::

        try:
            await gl_engine.post_journal_entry(entry)
        except BalanceError as exc:
            logger.error("gl_balance_violation", **exc.details)
        except TransactionError as exc:
            logger.error("transaction_failed", error=str(exc))
    """

    def __init__(
        self,
        message: str = "Transaction error occurred",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message: str = message
        self.details: Dict[str, Any] = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return human-readable representation including details if present."""
        if self.details:
            return f"{self.message} | details={self.details}"
        return self.message


# ═══════════════════════════════════════════════════════════════════════════
# Transaction Generation Exceptions
# ═══════════════════════════════════════════════════════════════════════════


class TransactionGenerationError(TransactionError):
    """Error during transaction generation.

    Raised when a P2P or O2C generator fails to produce a valid transaction
    after exhausting retry attempts.  Common causes include missing master
    data (no active vendors/customers), invalid statistical model output,
    or database persistence failures during transaction creation.

    Retry Policy (AAP §0.1.2):
        - P2P cycle generation: 2 attempts, linear backoff (1s, 2s),
          60s timeout, fallback: skip transaction.
        - O2C cycle generation: 2 attempts, linear backoff (1s, 2s),
          60s timeout, fallback: skip transaction.

    Typical ``details`` keys:
        - ``transaction_type``: The type of transaction that failed (e.g.,
          "purchase_order", "sales_order").
        - ``generator_class``: Fully-qualified class name of the generator.
        - ``attempt_number``: Which retry attempt failed.
        - ``root_cause``: Stringified original exception, if any.
    """

    def __init__(
        self,
        message: str = "Transaction generation failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


# ═══════════════════════════════════════════════════════════════════════════
# GL / Financial Integrity Exceptions
# ═══════════════════════════════════════════════════════════════════════════


class BalanceError(TransactionError):
    """GL entry doesn't balance — SUM(debits) ≠ SUM(credits) within $0.01.

    Raised by ``GLPostingEngine`` when a journal entry fails the fundamental
    balance validation:
    ``abs(SUM(debits) - SUM(credits)) > Decimal("0.01")``.

    This triggers immediate rollback of the entire transaction per the
    atomicity requirement (AAP §0.7.2).  No partial postings are permitted.

    Typical ``details`` keys:
        - ``journal_entry_id``: ID of the failing journal entry.
        - ``total_debits``: Sum of debit amounts as string (Decimal repr).
        - ``total_credits``: Sum of credit amounts as string (Decimal repr).
        - ``imbalance``: Absolute difference as string (Decimal repr).
        - ``tolerance``: The configured tolerance (normally "0.01").
    """

    def __init__(
        self,
        message: str = "GL entry does not balance",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class GLPostingError(TransactionError):
    """Error during GL journal entry posting.

    Raised for non-balance-related GL posting failures such as account
    validation (account not in Chart of Accounts or ``is_posting`` is False),
    period validation (target period is not OPEN), or persistence errors
    during journal entry or entry line creation.

    Retry Policy (AAP §0.1.2):
        - 3 attempts, exponential backoff (1s, 2s, 4s), 30s timeout,
          fallback: rollback transaction.

    Circuit Breaker (AAP §0.1.2):
        - 10 consecutive failures → circuit OPEN, 60s recovery window.

    Typical ``details`` keys:
        - ``journal_entry_id``: ID of the failing journal entry.
        - ``account_number``: The account that failed validation.
        - ``fiscal_period``: The target fiscal period.
        - ``period_status``: Current status of the period (OPEN/CLOSING/CLOSED).
        - ``error_category``: "account_validation", "period_validation",
          or "persistence".
    """

    def __init__(
        self,
        message: str = "GL posting failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class PeriodClosedError(TransactionError):
    """Attempt to post to a closed fiscal period.

    Raised by ``GLPostingEngine`` when a journal entry targets a fiscal
    period with status ``CLOSED`` or ``CLOSING``.  Only periods with status
    ``OPEN`` accept new postings.

    This is a distinct error from ``GLPostingError`` because closed-period
    violations are never retryable — the caller must either change the
    posting date to an open period or request a period re-opening (which
    requires Controller/CFO approval).

    Typical ``details`` keys:
        - ``fiscal_period``: The target fiscal period identifier.
        - ``period_status``: Current status ("CLOSED" or "CLOSING").
        - ``posting_date``: The date of the attempted posting.
        - ``journal_entry_id``: ID of the rejected journal entry.
    """

    def __init__(
        self,
        message: str = "Cannot post to closed fiscal period",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


# ═══════════════════════════════════════════════════════════════════════════
# Three-Way Matching Exception
# ═══════════════════════════════════════════════════════════════════════════


class ThreeWayMatchError(TransactionError):
    """Three-way match validation failure.

    Raised when PO/Receipt/Invoice matching fails outside the configured
    tolerances:
        - Price tolerance: ±5% (PO unit price vs. Invoice unit price)
        - Quantity tolerance: ±2% (Receipt quantity vs. Invoice quantity)

    Retry Policy (AAP §0.1.2):
        - 2 attempts, linear backoff (500ms, 1s), 10s timeout,
          fallback: mark as exception.

    Typical ``details`` keys:
        - ``purchase_order_id``: The PO involved in the match.
        - ``goods_receipt_id``: The receipt involved in the match.
        - ``vendor_invoice_id``: The invoice involved in the match.
        - ``price_variance_pct``: Actual price variance percentage.
        - ``quantity_variance_pct``: Actual quantity variance percentage.
        - ``match_status``: Resulting match status (e.g., "PRICE_EXCEPTION",
          "QUANTITY_EXCEPTION", "BOTH_EXCEPTION").
        - ``price_tolerance``: Configured price tolerance (normally 0.05).
        - ``quantity_tolerance``: Configured quantity tolerance (normally 0.02).
    """

    def __init__(
        self,
        message: str = "Three-way match validation failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


# ═══════════════════════════════════════════════════════════════════════════
# Discrepancy Injection Exception
# ═══════════════════════════════════════════════════════════════════════════


class DiscrepancyInjectionError(TransactionError):
    """Error during discrepancy injection.

    Raised when the ``DiscrepancyInjector`` fails to inject a discrepancy
    into a transaction.  Common causes include parameter bounds violations,
    transaction type incompatibility with the selected discrepancy, or
    ground truth record creation failures.

    Retry Policy (AAP §0.1.2):
        - 1 attempt, no backoff, 5s timeout,
          fallback: skip discrepancy (transaction proceeds clean).

    Circuit Breaker (AAP §0.1.2):
        - 20 consecutive failures → circuit OPEN, 30s recovery window.

    Typical ``details`` keys:
        - ``discrepancy_type``: Code of the discrepancy (e.g., "P2P-001").
        - ``transaction_id``: ID of the target transaction.
        - ``category``: Discrepancy category ("P2P", "O2C", "GL", "Control").
        - ``difficulty``: Configured difficulty ("easy" or "medium").
        - ``parameter_bounds``: The configured parameter bounds that were
          violated (if applicable).
    """

    def __init__(
        self,
        message: str = "Discrepancy injection failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


# ═══════════════════════════════════════════════════════════════════════════
# Rework Loop Exception
# ═══════════════════════════════════════════════════════════════════════════


class ReworkLoopError(TransactionError):
    """Error during rework loop processing.

    Raised when the ``ReworkLoopEngine`` fails to correct a transaction
    after the maximum number of fix attempts (3).  The rework loop
    classifies validation failures, selects fix scenarios sorted by
    success rate, applies corrections, and re-validates.  If all 3
    attempts fail, the transaction is escalated for human review.

    Retry Policy (AAP §0.1.2):
        - 3 attempts per transaction (fix scenario applications),
          linear backoff (1s, 2s, 3s), 30s timeout,
          fallback: escalate to admin.

    Circuit Breaker (AAP §0.1.2):
        - 50 consecutive failures → circuit OPEN, 120s recovery window.

    Typical ``details`` keys:
        - ``transaction_id``: ID of the transaction under rework.
        - ``attempt_number``: Which rework attempt failed (1–3).
        - ``fix_scenario``: Name of the fix scenario that was applied.
        - ``failure_classification``: "planned_discrepancy" or "unplanned_error".
        - ``scenarios_tried``: List of all scenarios attempted so far.
        - ``escalation_reason``: Why escalation was triggered.
    """

    def __init__(
        self,
        message: str = "Rework loop processing failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


# ═══════════════════════════════════════════════════════════════════════════
# Payment Allocation Exception
# ═══════════════════════════════════════════════════════════════════════════


class PaymentAllocationError(TransactionError):
    """Error during FIFO payment allocation.

    Raised when payment allocation fails — e.g., no open invoices to
    allocate against, allocation calculation produces invalid results,
    or the allocation would create a negative invoice balance.

    FIFO (oldest invoice first) is the only supported allocation strategy
    per AAP §0.1.2.

    CRITICAL BUSINESS RULE:
        - Overpayment → create Unapplied Cash record.
        - NEVER allow a negative invoice balance.

    Retry Policy (AAP §0.1.2):
        - Balance update: 2 attempts, linear backoff (500ms, 1s),
          10s timeout, fallback: rollback transaction.

    Typical ``details`` keys:
        - ``payment_id``: ID of the payment being allocated.
        - ``customer_id`` / ``vendor_id``: Entity making/receiving payment.
        - ``payment_amount``: Total payment amount as string (Decimal repr).
        - ``invoices_targeted``: Number of invoices in the allocation.
        - ``allocated_amount``: Amount successfully allocated so far.
        - ``remaining_amount``: Unallocated remainder.
        - ``allocation_strategy``: Always "FIFO".
    """

    def __init__(
        self,
        message: str = "Payment allocation failed",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


# ═══════════════════════════════════════════════════════════════════════════
# Concurrency Exception
# ═══════════════════════════════════════════════════════════════════════════


class ConcurrencyError(TransactionError):
    """Concurrent access conflict during SELECT FOR UPDATE.

    Raised when a balance update encounters a lock conflict due to
    concurrent access.  The ``AccountBalanceManager`` uses
    ``SELECT FOR UPDATE`` to prevent race conditions on account balance
    rows.  When two transactions attempt to update the same account
    balance simultaneously, the loser receives this exception.

    Retry Policy (AAP §0.1.2):
        - Balance update: 2 attempts, linear backoff (500ms, 1s),
          10s timeout, fallback: rollback transaction.

    Typical ``details`` keys:
        - ``account_number``: The contested account.
        - ``lock_holder``: Transaction ID holding the lock (if known).
        - ``lock_requester``: Transaction ID that was blocked.
        - ``wait_time_ms``: How long the requester waited before giving up.
        - ``retry_eligible``: Whether this conflict is retryable.
    """

    def __init__(
        self,
        message: str = "Concurrent access conflict",
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
