"""Order-to-Cash (O2C) Transaction Engine — four-class pipeline for the
Transaction Workflows system (Project 3).

This package implements the complete Order-to-Cash cycle of the Synthetic
ERP Data Generation Platform.  It produces realistic end-to-end O2C
workflows following the natural business flow::

    Sales Order → Shipment → Customer Invoice → Customer Payment

Each generator extends
:class:`~app.transactions.base_generator.TransactionGenerator` and receives
all dependencies through constructor injection (ADR-003).

Generators
----------
SalesOrderGenerator (``sales_order_generator.py``)
    First step — revenue-weighted customer selection using the Pareto
    (80/20) distribution, mandatory credit limit check (blocks the order
    when ``credit_limit < current_ar_balance + order_total``), product
    selection with 70/30 repeat-to-new split, pricing with volume and
    customer-tier discounts, and sequential SO-YYYY-NNNN numbering.

ShipmentGenerator (``shipment_generator.py``)
    Second step — creates shipments against open (confirmed/approved)
    sales orders, selects carriers via weighted random (UPS, FedEx,
    USPS, DHL), generates tracking numbers (TRK-{carrier}-{YYYYMMDD}-
    {NNNN}), reduces on-hand inventory, and posts GL entries:

    * **DR** Cost of Goods Sold (COGS) — shipped qty × unit cost
    * **CR** Inventory — shipped qty × unit cost

CustomerInvoiceGenerator (``customer_invoice_generator.py``)
    Third step — creates invoices from **shipped** orders.  **CRITICAL**:
    invoice amounts are calculated from **shipment quantities**, NOT from
    order quantities.  If a sales order was partially shipped the invoice
    reflects only what was actually shipped.  Applies payment terms from
    customer master data (Net 30, Net 60, 2/10 Net 30), calculates due
    dates and discount due dates, and posts GL entries:

    * **DR** Accounts Receivable — invoice total
    * **CR** Revenue — invoice total

    Sequential numbering: INV-YYYY-NNNN.

CustomerPaymentProcessor (``customer_payment_processor.py``)
    Final step — processes customer payments using **FIFO (First In,
    First Out) allocation** (oldest invoice first).  FIFO is the ONLY
    supported allocation strategy and is **NOT configurable** (AAP
    §0.1.2).

    * **Short pay**: remaining balance is either written off (below
      threshold) or left open on the oldest unpaid invoice.
    * **Overpayment** (CRITICAL): excess is recorded as **Unapplied
      Cash**.  A negative invoice balance is **NEVER** permitted — this
      is a non-negotiable constraint (AAP §0.1.2).

    GL entries:

    * **DR** Cash — payment amount
    * **CR** Accounts Receivable — allocated amount
    * **DR** Discount Allowed — early payment discount (if any)
    * **DR** Bad Debt Expense — short pay write-off (if any)
    * **CR** Unapplied Cash — overpayment excess (if any)

    Sequential numbering: CPAY-YYYY-NNNN.

Critical Business Rules
-----------------------
*   **FIFO payment allocation ONLY** — not configurable.
*   **Overpayment → Unapplied Cash** — NEVER negative invoice balance.
*   **Credit check** blocks the sales order when
    ``credit_limit < current_ar_balance + order_total``.
*   **Invoice amounts** come from **shipment** quantities, NOT order
    quantities.

Cross-cutting Concerns
----------------------
*   **Constructor injection** (ADR-003) — every generator receives
    dependencies through ``__init__`` keyword parameters; all parameters
    are ``Optional`` for partial composition and test isolation.
*   **Pydantic V2 data contracts** — every data model crossing subsystem
    boundaries is a Pydantic V2 ``BaseModel``.
*   **Structured JSON logging** — ``structlog`` to stdout with
    ``service_name``, ``component``, ``trace_id``, ``simulation_id``.
*   **EventBus notifications** (ADR-001) — cross-subsystem async events
    flow through the injected
    :class:`~app.events.event_bus.EventBus`.
*   **Decimal precision** — ``decimal.getcontext().prec = 28`` and
    ``rounding = ROUND_HALF_UP``; monetary amounts are *always*
    ``Decimal``, never ``float``.
*   **Deterministic reproducibility** — seeded ``random.Random``
    instances guarantee identical transaction sequences for the same
    seed.

Performance target (AAP §0.7.3)
--------------------------------
*   O2C cycle generation: ≥ 60 cycles / minute

Database pattern (Project 1 session management)::

    from synthetic_erp.db.session import get_session
    async with get_session() as session:
        ...

Usage examples::

    from app.transactions.o2c import SalesOrderGenerator, ShipmentGenerator
    from app.transactions.o2c import CustomerInvoiceGenerator, CustomerPaymentProcessor

References:
    - AAP §0.5.1 Group 3: O2C Transaction Engine
    - AAP §0.1.2: FIFO allocation, overpayment → Unapplied Cash, credit checks
    - AAP §0.7.2: Financial Integrity Rules
    - AAP §0.7.3: Performance Requirements
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# O2C Generator re-exports
# ---------------------------------------------------------------------------

from app.transactions.o2c.sales_order_generator import SalesOrderGenerator
from app.transactions.o2c.shipment_generator import ShipmentGenerator
from app.transactions.o2c.customer_invoice_generator import CustomerInvoiceGenerator
from app.transactions.o2c.customer_payment_processor import CustomerPaymentProcessor

# ---------------------------------------------------------------------------
# Public API — all symbols re-exported for ``from app.transactions.o2c import …``
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "SalesOrderGenerator",
    "ShipmentGenerator",
    "CustomerInvoiceGenerator",
    "CustomerPaymentProcessor",
]
"""Public classes exposed by the ``app.transactions.o2c`` package.

Each generator extends :class:`~app.transactions.base_generator.TransactionGenerator`
and receives all dependencies via constructor injection (ADR-003)::

    from app.transactions.o2c import SalesOrderGenerator, ShipmentGenerator
    from app.transactions.o2c import CustomerInvoiceGenerator, CustomerPaymentProcessor
"""
