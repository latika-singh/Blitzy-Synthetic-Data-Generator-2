"""Procure-to-Pay (P2P) discrepancy type implementations.

This sub-package contains 15 concrete ``BaseDiscrepancy`` subclasses
implementing P2P discrepancy types P2P-001 through P2P-015 for the
Synthetic ERP Data Generation Platform's Discrepancy Injection System.

Discrepancy Types by Difficulty:

**Easy (9 types)**:
    - P2P-001 ``DuplicateInvoice`` — Exact/near-duplicate invoice
    - P2P-002 ``PriceMismatch`` — Invoice/PO price variance outside ±5%
    - P2P-003 ``QuantityVariance`` — Invoice/Receipt quantity variance outside ±2%
    - P2P-004 ``MissingPO`` — Invoice without PO reference
    - P2P-005 ``PONotApproved`` — PO bypass or insufficient approval
    - P2P-006 ``InvoiceBeforeReceipt`` — Temporal sequence violation
    - P2P-008 ``WeekendProcessing`` — Unusual timing indicator
    - P2P-009 ``DuplicatePayment`` — Same vendor/amount/period
    - P2P-010 ``PaymentBeforeInvoice`` — Temporal anomaly

**Medium (6 types)**:
    - P2P-007 ``RoundDollarInvoice`` — Fraud indicator (exact round amounts)
    - P2P-011 ``UnapprovedVendor`` — Vendor not in approved list
    - P2P-012 ``SplitPO`` — Split PO to avoid approval threshold
    - P2P-013 ``FictitiousVendor`` — Address/phone matches employee
    - P2P-014 ``VendorConcentration`` — Disproportionate spend
    - P2P-015 ``GhostExpense`` — No supporting documentation

The convenience mapping ``P2P_DISCREPANCY_CLASSES`` maps each type code
to its implementation class, enabling the ``DiscrepancyCatalog`` to
instantiate discrepancy types dynamically by code.

Usage::

    from app.discrepancies.p2p import DuplicateInvoice, P2P_DISCREPANCY_CLASSES

    # Instantiate directly
    disc = DuplicateInvoice()

    # Or look up by type code
    cls = P2P_DISCREPANCY_CLASSES["P2P-001"]
    disc = cls()

References:
    - AAP Section 0.5.1 Group 5: P2P discrepancy types
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - app/agents/specialized/__init__.py: Pattern reference for package init
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# P2P-001 through P2P-005: Document and Approval Discrepancies
# ---------------------------------------------------------------------------
from app.discrepancies.p2p.duplicate_invoice import DuplicateInvoice
from app.discrepancies.p2p.price_mismatch import PriceMismatch
from app.discrepancies.p2p.quantity_variance import QuantityVariance
from app.discrepancies.p2p.missing_po import MissingPO
from app.discrepancies.p2p.po_not_approved import PONotApproved

# ---------------------------------------------------------------------------
# P2P-006 through P2P-010: Temporal and Payment Discrepancies
# ---------------------------------------------------------------------------
from app.discrepancies.p2p.invoice_before_receipt import InvoiceBeforeReceipt
from app.discrepancies.p2p.round_dollar_invoice import RoundDollarInvoice
from app.discrepancies.p2p.weekend_processing import WeekendProcessing
from app.discrepancies.p2p.duplicate_payment import DuplicatePayment
from app.discrepancies.p2p.payment_before_invoice import PaymentBeforeInvoice

# ---------------------------------------------------------------------------
# P2P-011 through P2P-015: Vendor and Expense Discrepancies
# ---------------------------------------------------------------------------
from app.discrepancies.p2p.unapproved_vendor import UnapprovedVendor
from app.discrepancies.p2p.split_po import SplitPO
from app.discrepancies.p2p.fictitious_vendor import FictitiousVendor
from app.discrepancies.p2p.vendor_concentration import VendorConcentration
from app.discrepancies.p2p.ghost_expense import GhostExpense

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # P2P-001 through P2P-005
    "DuplicateInvoice",
    "PriceMismatch",
    "QuantityVariance",
    "MissingPO",
    "PONotApproved",
    # P2P-006 through P2P-010
    "InvoiceBeforeReceipt",
    "RoundDollarInvoice",
    "WeekendProcessing",
    "DuplicatePayment",
    "PaymentBeforeInvoice",
    # P2P-011 through P2P-015
    "UnapprovedVendor",
    "SplitPO",
    "FictitiousVendor",
    "VendorConcentration",
    "GhostExpense",
    # Type-code-to-class mapping
    "P2P_DISCREPANCY_CLASSES",
]

# ---------------------------------------------------------------------------
# Type-code-to-class mapping
# ---------------------------------------------------------------------------
# Maps each P2P discrepancy type code to its class object. Keys match the
# ``type_code`` class attribute on each discrepancy, the catalog entries in
# ``DiscrepancyCatalog.register_defaults()``, and the type identifiers in
# ``config/discrepancies/p2p_discrepancies.yaml``.
# ---------------------------------------------------------------------------
P2P_DISCREPANCY_CLASSES: dict[str, type] = {
    "P2P-001": DuplicateInvoice,
    "P2P-002": PriceMismatch,
    "P2P-003": QuantityVariance,
    "P2P-004": MissingPO,
    "P2P-005": PONotApproved,
    "P2P-006": InvoiceBeforeReceipt,
    "P2P-007": RoundDollarInvoice,
    "P2P-008": WeekendProcessing,
    "P2P-009": DuplicatePayment,
    "P2P-010": PaymentBeforeInvoice,
    "P2P-011": UnapprovedVendor,
    "P2P-012": SplitPO,
    "P2P-013": FictitiousVendor,
    "P2P-014": VendorConcentration,
    "P2P-015": GhostExpense,
}
