"""Transaction Generation Engine — core pipeline for Project 3 (Transaction
Workflows & Discrepancies).

This package implements the complete transaction generation, GL integration,
and financial integrity subsystem of the Synthetic ERP Data Generation
Platform.  It produces realistic end-to-end P2P, O2C, and GL workflows with
deterministic reproducibility, configurable discrepancy injection, and
continuous trial-balance validation.

Sub-packages
------------
p2p
    **Procure-to-Pay** generators — five-class pipeline producing complete
    P2P cycles: Purchase Order → Goods Receipt → Vendor Invoice (with
    three-way matching) → Vendor Payment, including artifacts and GL
    postings at every step.

o2c
    **Order-to-Cash** generators — four-class pipeline producing complete
    O2C cycles: Sales Order → Shipment → Customer Invoice → Customer
    Payment, with credit checks, FIFO payment allocation, and GL postings.

gl
    **General Ledger** integration — posting engine writing balanced
    journal entries (DR = CR within $0.01), real-time account balance
    maintenance with ``SELECT FOR UPDATE`` locking, accrual generation
    (GRNI, shipped-not-invoiced), and period-close orchestration.

Root-level modules
------------------
base_generator
    Abstract :class:`TransactionGenerator` base class defining the
    ``generate → validate → post`` contract for all generators, plus
    :class:`GenerationContext` (per-invocation Pydantic V2 context model)
    and :class:`TransactionResult` (generation cycle output model).

exceptions
    Custom exception hierarchy — 10 classes rooted at
    :class:`TransactionError`, enabling targeted catch/retry logic per
    AAP §0.7.4.

constants
    Shared constants: processing limits, performance thresholds, memory
    limits, financial tolerances, batch limits, and circuit-breaker
    configuration.

Key exports
-----------
TransactionGenerator
    Abstract base class (ABC) for all P2P, O2C, and GL generators.
GenerationContext
    Pydantic V2 model carrying simulation_id, current_date, fiscal_period,
    rng_seed, and discrepancy configuration per invocation.
TransactionResult
    Pydantic V2 model with transaction_id, artifacts, gl_entries, timing
    metrics, and discrepancy information.

Cross-cutting concerns
----------------------
*   **Constructor injection** (ADR-003) — every generator and subsystem
    receives dependencies through ``__init__`` keyword parameters; all
    parameters are ``Optional`` for partial composition and testing.
*   **Pydantic V2 data contracts** — every data model crossing subsystem
    boundaries is a Pydantic V2 ``BaseModel`` with explicit field types.
*   **Structured JSON logging** — all logging uses ``structlog`` to stdout
    with ``service_name``, ``component``, ``trace_id``, ``simulation_id``.
*   **EventBus notifications** (ADR-001) — cross-subsystem async events flow
    through the injected :class:`~app.events.event_bus.EventBus`.
*   **Decimal precision** — ``decimal.getcontext().prec = 28`` and
    ``rounding = ROUND_HALF_UP``; monetary amounts are *always* ``Decimal``.
*   **Deterministic reproducibility** — seeded ``random.Random`` instances
    guarantee identical transaction sequences for the same seed.

Performance targets (AAP §0.7.3)
---------------------------------
*   P2P cycle generation:  ≥ 50 / minute
*   O2C cycle generation:  ≥ 60 / minute
*   GL posting:            ≥ 200 / minute
*   Transaction throughput: ≥ 2,000 / hour sustained

Database pattern (Project 1 session management)::

    from synthetic_erp.db.session import get_session
    async with get_session() as session:
        ...

Usage examples::

    # Import core classes directly from the package
    from app.transactions import TransactionGenerator, GenerationContext
    from app.transactions import TransactionError, BalanceError

    # Import sub-package generators explicitly
    from app.transactions.p2p.purchase_order_generator import PurchaseOrderGenerator
    from app.transactions.o2c.sales_order_generator import SalesOrderGenerator
    from app.transactions.gl.gl_posting_engine import GLPostingEngine
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base classes — abstract generator contract and Pydantic V2 data models
# ---------------------------------------------------------------------------

from app.transactions.base_generator import (
    GenerationContext,
    TransactionGenerator,
    TransactionResult,
)

# ---------------------------------------------------------------------------
# Exception hierarchy — 10 domain-specific exception classes
# ---------------------------------------------------------------------------

from app.transactions.exceptions import (
    BalanceError,
    ConcurrencyError,
    DiscrepancyInjectionError,
    GLPostingError,
    PaymentAllocationError,
    PeriodClosedError,
    ReworkLoopError,
    ThreeWayMatchError,
    TransactionError,
    TransactionGenerationError,
)

# ---------------------------------------------------------------------------
# Public API — all symbols re-exported for ``from app.transactions import …``
# ---------------------------------------------------------------------------
#
# Sub-packages (``p2p``, ``o2c``, ``gl``) are listed in ``__all__`` for
# documentation purposes but are NOT eagerly imported.  To use a specific
# generator, import it explicitly from its sub-package:
#
#     from app.transactions.p2p.purchase_order_generator import PurchaseOrderGenerator
#     from app.transactions.gl.gl_posting_engine import GLPostingEngine
#     from app.transactions.o2c.customer_payment_processor import CustomerPaymentProcessor
#

__all__: list[str] = [
    # Base classes
    "TransactionGenerator",
    "GenerationContext",
    "TransactionResult",
    # Exception hierarchy
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
    # Sub-packages (documented but not imported — use explicit imports)
    "p2p",
    "o2c",
    "gl",
]
