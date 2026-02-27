"""Shared constants for the P3 Transaction Workflows & Discrepancies system.

Centralizes all numeric limits, tolerance thresholds, circuit breaker configurations,
timeout matrices, performance targets, sequential numbering prefixes, and financial
tolerance constants used across the P2P, O2C, GL, Discrepancy, and Rework subsystems.

These values are derived from the Project 3 specification and represent non-negotiable
constraints that govern transaction generation, financial integrity, and error handling.

References:
    - AAP Section 0.1.2 (Retry Policy Table, Circuit Breaker Configuration)
    - AAP Section 0.7.2 (Financial Integrity Rules)
    - AAP Section 0.7.3 (Performance Requirements)
    - AAP Section 0.7.4 (Error Handling Conventions)
    - AAP Section 0.7.5 (Discrepancy Injection Rules)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Final


# ---------------------------------------------------------------------------
# Transaction Processing Limits
# ---------------------------------------------------------------------------

TRANSACTION_BATCH_SIZE: Final[int] = 100
"""Default number of transactions per generation batch."""

GL_POSTING_BATCH_SIZE: Final[int] = 500
"""Maximum GL journal entries per posting batch."""

CONCURRENT_TRANSACTION_LIMIT: Final[int] = 100
"""Maximum concurrent transaction workflows (AAP §0.7.3)."""

MAX_TRANSACTIONS_PER_HOUR: Final[int] = 2_000
"""Sustained transaction throughput target per hour."""

TRANSACTION_PROCESSING_LIMITS: Dict[str, int] = {
    "batch_size": TRANSACTION_BATCH_SIZE,
    "gl_posting_batch_size": GL_POSTING_BATCH_SIZE,
    "concurrent_limit": CONCURRENT_TRANSACTION_LIMIT,
    "max_per_hour": MAX_TRANSACTIONS_PER_HOUR,
}


# ---------------------------------------------------------------------------
# Performance Thresholds (AAP §0.7.3)
# ---------------------------------------------------------------------------

PERFORMANCE_THRESHOLDS: Dict[str, Any] = {
    "p2p_cycles_per_minute": 50,
    "o2c_cycles_per_minute": 60,
    "gl_postings_per_minute": 200,
    "transaction_completion_rate": 0.995,
    "concurrent_transactions": 100,
    "transactions_per_hour": 2_000,
    "rework_loop_timeout_seconds": 30,
    "one_month_simulation_hours_min": 4,
    "one_month_simulation_hours_max": 8,
}


# ---------------------------------------------------------------------------
# Financial Tolerances (AAP §0.7.2)
# ---------------------------------------------------------------------------
# CRITICAL: ALL values MUST be Decimal type, NEVER float.

GL_BALANCE_TOLERANCE: Final[Decimal] = Decimal("0.01")
"""Maximum allowable imbalance for SUM(debits) - SUM(credits) per journal entry."""

TRIAL_BALANCE_TOLERANCE: Final[Decimal] = Decimal("0.01")
"""Maximum cumulative trial balance deviation."""

SUB_LEDGER_TOLERANCE: Final[Decimal] = Decimal("0.01")
"""Maximum sub-ledger to GL control account reconciliation difference."""

BALANCE_SHEET_TOLERANCE: Final[Decimal] = Decimal("0.01")
"""Maximum tolerance for Assets = Liabilities + Equity validation."""

FINANCIAL_TOLERANCES: Dict[str, Decimal] = {
    "gl_balance": GL_BALANCE_TOLERANCE,
    "trial_balance": TRIAL_BALANCE_TOLERANCE,
    "sub_ledger": SUB_LEDGER_TOLERANCE,
    "balance_sheet": BALANCE_SHEET_TOLERANCE,
}


# ---------------------------------------------------------------------------
# Three-Way Match Tolerances
# ---------------------------------------------------------------------------

THREE_WAY_MATCH_PRICE_TOLERANCE: Final[Decimal] = Decimal("0.05")
"""±5% price tolerance for PO vs Invoice matching."""

THREE_WAY_MATCH_QUANTITY_TOLERANCE: Final[Decimal] = Decimal("0.02")
"""±2% quantity tolerance for Receipt vs Invoice matching."""

THREE_WAY_MATCH_TOLERANCES: Dict[str, Decimal] = {
    "price_tolerance_percent": THREE_WAY_MATCH_PRICE_TOLERANCE,
    "quantity_tolerance_percent": THREE_WAY_MATCH_QUANTITY_TOLERANCE,
}


# ---------------------------------------------------------------------------
# Retry Policies (AAP §0.1.2 — Retry Policy Table)
# ---------------------------------------------------------------------------
# Each entry defines:
#   max_attempts      — total number of attempts (including the initial try)
#   backoff_strategy  — "linear", "exponential", or "none"
#   backoff_seconds   — list of wait durations between attempts
#   timeout_seconds   — per-operation timeout
#   fallback          — action to take when all attempts are exhausted

RETRY_POLICIES: Dict[str, Dict[str, Any]] = {
    "p2p_cycle_generation": {
        "max_attempts": 2,
        "backoff_strategy": "linear",
        "backoff_seconds": [1, 2],
        "timeout_seconds": 60,
        "fallback": "skip_transaction",
    },
    "o2c_cycle_generation": {
        "max_attempts": 2,
        "backoff_strategy": "linear",
        "backoff_seconds": [1, 2],
        "timeout_seconds": 60,
        "fallback": "skip_transaction",
    },
    "gl_posting": {
        "max_attempts": 3,
        "backoff_strategy": "exponential",
        "backoff_seconds": [1, 2, 4],
        "timeout_seconds": 30,
        "fallback": "rollback_transaction",
    },
    "three_way_matching": {
        "max_attempts": 2,
        "backoff_strategy": "linear",
        "backoff_seconds": [0.5, 1],
        "timeout_seconds": 10,
        "fallback": "mark_as_exception",
    },
    "discrepancy_injection": {
        "max_attempts": 1,
        "backoff_strategy": "none",
        "backoff_seconds": [],
        "timeout_seconds": 5,
        "fallback": "skip_discrepancy",
    },
    "rework_loop_fix": {
        "max_attempts": 3,
        "backoff_strategy": "linear",
        "backoff_seconds": [1, 2, 3],
        "timeout_seconds": 30,
        "fallback": "escalate_to_admin",
    },
    "period_close": {
        "max_attempts": 1,
        "backoff_strategy": "none",
        "backoff_seconds": [],
        "timeout_seconds": 300,
        "fallback": "halt_manual_intervention",
    },
    "balance_update": {
        "max_attempts": 2,
        "backoff_strategy": "linear",
        "backoff_seconds": [0.5, 1],
        "timeout_seconds": 10,
        "fallback": "rollback_transaction",
    },
}


# ---------------------------------------------------------------------------
# Circuit Breaker Configuration (AAP §0.1.2)
# ---------------------------------------------------------------------------
# Each entry defines:
#   failure_threshold         — number of consecutive failures before opening
#   recovery_timeout_seconds  — seconds before attempting half-open probe
#   fallback                  — action when circuit is open

CIRCUIT_BREAKER_CONFIG: Dict[str, Dict[str, Any]] = {
    "gl_posting": {
        "failure_threshold": 10,
        "recovery_timeout_seconds": 60,
        "fallback": "rollback_transaction",
    },
    "discrepancy_injection": {
        "failure_threshold": 20,
        "recovery_timeout_seconds": 30,
        "fallback": "skip_discrepancy",
    },
    "rework_loop": {
        "failure_threshold": 50,
        "recovery_timeout_seconds": 120,
        "fallback": "escalate_to_admin",
    },
}


# ---------------------------------------------------------------------------
# Memory Limits
# ---------------------------------------------------------------------------

MEMORY_LIMITS: Dict[str, int] = {
    "transaction_cache_max_items": 10_000,
    "gl_entry_buffer_max_items": 5_000,
    "discrepancy_buffer_max_items": 1_000,
    "rework_queue_max_items": 500,
}


# ---------------------------------------------------------------------------
# Batch Limits
# ---------------------------------------------------------------------------

BATCH_LIMITS: Dict[str, int] = {
    "transaction_batch_size": TRANSACTION_BATCH_SIZE,
    "gl_posting_batch_size": GL_POSTING_BATCH_SIZE,
}


# ---------------------------------------------------------------------------
# Sequential Numbering Prefixes
# ---------------------------------------------------------------------------
# Format: PREFIX-YYYY-NNNN  (e.g. PO-2024-0001)

DOCUMENT_NUMBER_PREFIXES: Dict[str, str] = {
    "purchase_order": "PO",
    "sales_order": "SO",
    "customer_invoice": "INV",
    "vendor_invoice": "VINV",
    "goods_receipt": "GR",
    "shipment": "SHP",
    "vendor_payment": "VPAY",
    "customer_payment": "CPAY",
    "journal_entry": "JE",
}


# ---------------------------------------------------------------------------
# Approval Thresholds (from config/workflows/approval_thresholds.yaml)
# ---------------------------------------------------------------------------
# Replicated here for fast programmatic access without YAML I/O.
# Each transaction type has an ordered list of tier dictionaries:
#   max_amount     — upper bound (Decimal) for the tier, or None for unbounded
#   required_role  — approver role string, or None when no approval is needed

APPROVAL_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "purchase_order": {
        "tiers": [
            {"max_amount": Decimal("5000"), "required_role": None},
            {"max_amount": Decimal("25000"), "required_role": "purchasing_manager"},
            {"max_amount": Decimal("100000"), "required_role": "controller"},
            {"max_amount": None, "required_role": "cfo"},
        ],
    },
    "vendor_invoice": {
        "tiers": [
            {"max_amount": Decimal("10000"), "required_role": None},
            {"max_amount": Decimal("50000"), "required_role": "ap_manager"},
            {"max_amount": Decimal("100000"), "required_role": "controller"},
            {"max_amount": None, "required_role": "cfo"},
        ],
    },
    "journal_entry": {
        "tiers": [
            {"max_amount": Decimal("50000"), "required_role": "senior_accountant"},
            {"max_amount": None, "required_role": "controller"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Discrepancy Injection Defaults (AAP §0.7.5)
# ---------------------------------------------------------------------------
# Rate control:  actual injection rate MUST be within ±1% of configured target.
# Difficulty:    Easy (70%) / Medium (30%) / Hard (0%) — ±5% tolerance.
# Auto-adjust:   when True, out-of-bounds parameters are clamped to nearest bound.

DISCREPANCY_DEFAULTS: Dict[str, Any] = {
    "injection_rate": Decimal("0.02"),
    "rate_tolerance": Decimal("0.01"),
    "difficulty_distribution": {
        "easy": Decimal("0.70"),
        "medium": Decimal("0.30"),
        "hard": Decimal("0.00"),
    },
    "difficulty_tolerance": Decimal("0.05"),
    "auto_adjust_to_bounds": True,
}


# ---------------------------------------------------------------------------
# Rework Loop Configuration
# ---------------------------------------------------------------------------

REWORK_LOOP_CONFIG: Dict[str, Any] = {
    "max_attempts_per_transaction": 3,
    "escalation_failure_rate_threshold": Decimal("0.05"),
    "timeout_seconds": 30,
    "fix_scenario_timeout_seconds": 10,
}


# ---------------------------------------------------------------------------
# Account Type Classifications (for normal balance direction)
# ---------------------------------------------------------------------------
# AAP §0.7.2: Asset and Expense accounts increase with debits;
# Liability, Equity, and Revenue accounts increase with credits.

DEBIT_NORMAL_ACCOUNT_TYPES: frozenset[str] = frozenset({"asset", "expense"})
"""Account types where debits increase the balance."""

CREDIT_NORMAL_ACCOUNT_TYPES: frozenset[str] = frozenset({"liability", "equity", "revenue"})
"""Account types where credits increase the balance."""


# ---------------------------------------------------------------------------
# Period Status Constants
# ---------------------------------------------------------------------------

PERIOD_STATUS_OPEN: Final[str] = "OPEN"
PERIOD_STATUS_CLOSING: Final[str] = "CLOSING"
PERIOD_STATUS_CLOSED: Final[str] = "CLOSED"
