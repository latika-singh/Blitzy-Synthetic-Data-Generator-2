"""
Discrepancy Injection System package (Project 3) — Configurable discrepancy injection
for the Synthetic ERP Data Generation Platform.

This package implements the complete discrepancy injection system for Project 3:
Transaction Workflows & Discrepancies. It provides a configurable injector that
introduces 35+ discrepancy types across four categories into generated transactions,
with full ground truth label generation for downstream audit detection validation.

Core Components:
    DiscrepancyInjector
        Rate-based injection trigger (default 2% ±1%), weighted type selection by
        category using seeded RNG, parameter validation against configured bounds,
        auto-adjust to bounds when enabled, and circuit breaker (20 failures → open,
        30s recovery). Publishes DiscrepancyDetected events via EventBus.

    GroundTruthGenerator
        Creates ground truth records with the full 16-field schema (discrepancy_id,
        type_code, category, difficulty, transaction_ids, affected_fields,
        original_values, modified_values, detection_method, detection_difficulty,
        financial_impact, description, injection_timestamp, simulation_id,
        ground_truth_label, metadata). Supports JSON and CSV output.

    DiscrepancyCatalog
        Registry mapping 35+ type codes (P2P-001 through CTL-005) to implementation
        classes. Loads configuration from config/discrepancies/*.yaml files. Provides
        lookup by code, category, and difficulty.

    BaseDiscrepancy
        Abstract base class defining the `inject(transaction, params, rng)` contract
        that returns `Tuple[ModifiedTransaction, GroundTruth]`. Each concrete
        discrepancy type overrides `inject()` with type-specific mutation logic.

Sub-packages:
    p2p
        15 Procure-to-Pay discrepancy types (P2P-001 through P2P-015):
        duplicate invoice, price mismatch, quantity variance, missing PO,
        PO not approved, invoice before receipt, round-dollar invoice,
        weekend processing, duplicate payment, payment before invoice,
        unapproved vendor, split PO, fictitious vendor, vendor concentration,
        ghost expense.

    o2c
        10 Order-to-Cash discrepancy types (O2C-001 through O2C-010):
        duplicate customer invoice, invoice without shipment, credit limit
        exceeded, short payment, overpayment not returned, revenue recognition
        timing, fictitious customer, round-tripping, channel stuffing,
        side agreements.

    gl
        5 General Ledger discrepancy types (GL-001 through GL-005):
        unbalanced journal, journal no approval, suspicious adjusting entry,
        unusual account combination, manual override.

    control
        5 Control discrepancy types (CTL-001 through CTL-005):
        SoD violation, self-approval, approval limit exceeded, backdated
        transaction, holiday transaction.

Discrepancy Injection Rules (AAP §0.7.5):
    - Rate Control: Actual injection rate within ±1% of configured target (default 2%)
    - Difficulty Distribution: Easy (70%) / Medium (30%) / Hard (0%) — ±5% tolerance
    - Parameter Bounds: ALL parameters within configured bounds
    - Auto-Adjust: When enabled, clamp out-of-bounds parameters to nearest bound
    - Ground Truth Coverage: 100% of injected discrepancies have ground truth records
    - Transaction Linkage: Every ground truth record references valid transaction IDs

Architecture:
    - Constructor injection (ADR-003) for all dependencies
    - Pydantic V2 for all data contracts at subsystem boundaries
    - EventBus (ADR-001) for DiscrepancyDetected event publication
    - structlog for JSON structured logging to stdout
    - Deterministic seeded random.Random for reproducibility
    - Decimal for all financial calculations (prec=28, ROUND_HALF_UP)
    - tenacity for retry with configurable backoff

References:
    - AAP Section 0.5.1 Group 5: Discrepancy Injection System
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.1: Constructor injection, Pydantic V2, EventBus
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core class re-exports — kept lightweight (no sub-package eager loading)
# ---------------------------------------------------------------------------

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.discrepancies.discrepancy_injector import DiscrepancyInjector
from app.discrepancies.ground_truth_generator import GroundTruthGenerator
from app.discrepancies.discrepancy_catalog import DiscrepancyCatalog

# ---------------------------------------------------------------------------
# Public API declaration
# ---------------------------------------------------------------------------

__all__: list[str] = [
    # Core classes
    "BaseDiscrepancy",
    "DiscrepancyInjector",
    "GroundTruthGenerator",
    "DiscrepancyCatalog",
    # Sub-packages (documented but not eagerly imported)
    "p2p",
    "o2c",
    "gl",
    "control",
]
"""Public API of the ``app.discrepancies`` package.

Sub-packages (``p2p``, ``o2c``, ``gl``, ``control``) are NOT eagerly imported
to keep ``import app.discrepancies`` lightweight. Consumers should import
specific discrepancy types directly::

    from app.discrepancies.p2p.duplicate_invoice import DuplicateInvoice
    from app.discrepancies.gl.unbalanced_journal import UnbalancedJournal
"""
