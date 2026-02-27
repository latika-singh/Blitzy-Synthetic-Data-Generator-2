"""Procure-to-Pay (P2P) Transaction Engine package.

Part of **Project 3 — Transaction Workflows & Discrepancies** for the
Synthetic ERP Data Generation Platform.

This package implements the complete Procure-to-Pay cycle, the five-stage
pipeline that drives procurement-side financial transaction generation:

    Purchase Order → Goods Receipt → Vendor Invoice (with three-way matching)
    → Vendor Payment

Exported Classes
----------------
PurchaseOrderGenerator
    Stage 1 — Vendor selection using Pareto 80/20 distribution,
    EOQ-based (Economic Order Quantity) line quantities, unit pricing from
    log-normal ``AmountDistribution``, approval routing per monetary
    thresholds ($5 K / $25 K / $100 K), PO header/lines creation, and
    sequential numbering (``PO-YYYY-NNNN``).  Extends
    :class:`~app.transactions.base_generator.TransactionGenerator`.

GoodsReceiptGenerator
    Stage 2 — Receipt against open approved POs, quantity variance handling
    (over-receipt / under-receipt / partial receipt), inventory balance
    update, and GL posting (DR Inventory, CR AP Accrual / GRNI).  Receipt
    date = PO date + lead time.  Extends ``TransactionGenerator``.

VendorInvoiceProcessor
    Stage 3 — Invoice creation from vendor submission, PO linkage,
    three-way match invocation (PO / Receipt / Invoice), AP clerk routing
    via ``WorkflowOrchestrator``, GL account coding, approval chain
    enforcement ($10 K / $50 K / $100 K), and GL posting (DR Expense /
    Asset, CR Accounts Payable).  Extends ``TransactionGenerator``.

ThreeWayMatcher
    Stateless matching engine — validates PO / Receipt / Invoice line items
    with configurable tolerances (±5 % price, ±2 % quantity).  Calculates
    variance at the **line-item level** (per line, then summed) as mandated
    by the specification.  Returns ``ThreeWayMatchResult`` with match
    status, per-line variances, and approval-requirement flag.  Consumed by
    ``VendorInvoiceProcessor`` via constructor injection.

VendorPaymentGenerator
    Stage 5 (final) — Selects approved unpaid invoices, groups by vendor
    and payment terms, calculates early payment discounts (e.g. 2/10 Net
    30), selects payment method (check < $5 K, ACH $5 K–$50 K, wire
    ≥ $50 K), creates payment allocations, and GL posting (DR AP, CR Cash;
    DR Purchase Discount if applicable).  Extends ``TransactionGenerator``.

Integration Points
-------------------
*   ``app.transactions.base_generator.TransactionGenerator`` — abstract base
    class defining the ``generate() → validate() → post()`` contract with
    shared discrepancy trigger checking, GL posting delegation, event
    publishing, and ``tenacity``-based retry/timeout wrappers.
*   ``app.transactions.gl.gl_posting_engine.GLPostingEngine`` — all P2P
    generators delegate journal entry creation and balance validation to the
    GL posting engine (DR = CR within $0.01).
*   ``app.orchestration.workflow_orchestrator.WorkflowOrchestrator`` —
    transaction routing via ``ROLE_MAPPING`` (``purchase_order``,
    ``goods_receipt``, ``vendor_invoice``, ``vendor_payment``).
*   ``app.orchestration.approval_system.ApprovalSystem`` — monetary
    threshold-based approval chains for POs and vendor invoices.
*   ``app.events.event_bus.EventBus`` — publishes ``TransactionCreated``,
    ``TransactionCompleted``, ``ApprovalRequired``, and
    ``DocumentGenerated`` events (ADR-001).
*   ``app.agents.agent_registry.AgentRegistry`` — agent assignment for
    decision-making within transaction workflows.

Design Decisions
----------------
*   **Constructor injection** (ADR-003): all generators receive
    dependencies (``EventBus``, ``GLPostingEngine``, ``AgentRegistry``,
    ``DiscrepancyInjector``, etc.) through constructor parameters.
*   **Pydantic V2** data contracts at all subsystem boundaries.
*   **structlog** JSON logging (stdout only) inside generators —
    *not* in this ``__init__.py``.
*   **EventBus** (ADR-001) for async cross-subsystem notifications.
*   **Decimal precision**: ``getcontext().prec = 28``,
    ``rounding = ROUND_HALF_UP`` — ``float`` is **never** used for
    monetary amounts.
*   **Deterministic reproducibility**: seeded ``random.Random`` instances
    passed via ``GenerationContext``.

Performance Target
------------------
P2P ≥ 50 complete cycles / minute (AAP §0.7.3).

Database Pattern
----------------
``from synthetic_erp.db.session import get_session`` with
``async with get_session() as session:`` for all database operations.
"""

from __future__ import annotations

from app.transactions.p2p.purchase_order_generator import PurchaseOrderGenerator
from app.transactions.p2p.goods_receipt_generator import GoodsReceiptGenerator
from app.transactions.p2p.vendor_invoice_processor import VendorInvoiceProcessor
from app.transactions.p2p.three_way_matcher import ThreeWayMatcher
from app.transactions.p2p.vendor_payment_generator import VendorPaymentGenerator

__all__: list[str] = [
    "PurchaseOrderGenerator",
    "GoodsReceiptGenerator",
    "VendorInvoiceProcessor",
    "ThreeWayMatcher",
    "VendorPaymentGenerator",
]
