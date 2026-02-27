"""Order-to-Cash (O2C) discrepancy type implementations.

This sub-package contains 10 discrepancy types that target the Order-to-Cash
business process flow: Sales Order → Shipment → Customer Invoice → Customer
Payment. Each class extends :class:`~app.discrepancies.base_discrepancy.BaseDiscrepancy`
and implements the ``inject()`` method with type-specific mutation logic.

O2C Discrepancy Types:

- **O2C-001** :class:`DuplicateCustomerInvoice` — Duplicate customer invoice
  (exact or near-duplicate within configurable date window)
- **O2C-002** :class:`InvoiceWithoutShipment` — Invoice without corresponding
  shipment record
- **O2C-003** :class:`CreditLimitExceeded` — Order processed exceeding customer
  credit limit
- **O2C-004** :class:`ShortPayment` — Unauthorized short payment without dispute
  documentation
- **O2C-005** :class:`OverpaymentNotReturned` — Overpayment not returned or
  recorded as Unapplied Cash
- **O2C-006** :class:`RevenueRecognitionTiming` — Premature revenue recognition
  in earlier fiscal period
- **O2C-007** :class:`FictitiousCustomer` — Transaction with fabricated or
  non-existent customer entity
- **O2C-008** :class:`RoundTripping` — Circular transactions inflating revenue
- **O2C-009** :class:`ChannelStuffing` — Excessive period-end shipments to
  inflate revenue
- **O2C-010** :class:`SideAgreements` — Undisclosed side agreements modifying
  effective pricing

Difficulty Distribution:
    Easy (O2C-001 through O2C-004): 4 types
    Medium (O2C-005 through O2C-010): 6 types

The convenience mapping :data:`O2C_DISCREPANCY_CLASSES` maps each type code
to its implementation class, enabling the
:class:`~app.discrepancies.discrepancy_catalog.DiscrepancyCatalog` to
instantiate discrepancy types dynamically by code.

Usage::

    from app.discrepancies.o2c import DuplicateCustomerInvoice, O2C_DISCREPANCY_CLASSES

    # Instantiate by class directly
    discrepancy = DuplicateCustomerInvoice()

    # Or look up by type code
    cls = O2C_DISCREPANCY_CLASSES["O2C-001"]
    discrepancy = cls()

References:
    - AAP Section 0.5.1 Group 5: O2C discrepancy types
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.1: Deterministic reproducibility, structlog, Decimal
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Easy difficulty (O2C-001 through O2C-004)
# ---------------------------------------------------------------------------
from app.discrepancies.o2c.duplicate_customer_invoice import DuplicateCustomerInvoice
from app.discrepancies.o2c.invoice_without_shipment import InvoiceWithoutShipment
from app.discrepancies.o2c.credit_limit_exceeded import CreditLimitExceeded
from app.discrepancies.o2c.short_payment import ShortPayment

# ---------------------------------------------------------------------------
# Medium difficulty (O2C-005 through O2C-010)
# ---------------------------------------------------------------------------
from app.discrepancies.o2c.overpayment_not_returned import OverpaymentNotReturned
from app.discrepancies.o2c.revenue_recognition_timing import RevenueRecognitionTiming
from app.discrepancies.o2c.fictitious_customer import FictitiousCustomer
from app.discrepancies.o2c.round_tripping import RoundTripping
from app.discrepancies.o2c.channel_stuffing import ChannelStuffing
from app.discrepancies.o2c.side_agreements import SideAgreements

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # Easy difficulty
    "DuplicateCustomerInvoice",
    "InvoiceWithoutShipment",
    "CreditLimitExceeded",
    "ShortPayment",
    # Medium difficulty
    "OverpaymentNotReturned",
    "RevenueRecognitionTiming",
    "FictitiousCustomer",
    "RoundTripping",
    "ChannelStuffing",
    "SideAgreements",
    # Registry
    "O2C_DISCREPANCY_CLASSES",
]

# ---------------------------------------------------------------------------
# Type-code-to-class mapping
# ---------------------------------------------------------------------------
# Maps each O2C discrepancy type code (str) to its implementation class.
# Keys match the ``type_code`` class attribute on each discrepancy, the
# type code identifiers in ``config/discrepancies/o2c_discrepancies.yaml``,
# and the codes used by the
# :class:`~app.discrepancies.discrepancy_catalog.DiscrepancyCatalog`.
#
# Used by :class:`~app.discrepancies.discrepancy_injector.DiscrepancyInjector`
# to instantiate discrepancy types dynamically by code.
# ---------------------------------------------------------------------------
O2C_DISCREPANCY_CLASSES: dict[str, type] = {
    "O2C-001": DuplicateCustomerInvoice,
    "O2C-002": InvoiceWithoutShipment,
    "O2C-003": CreditLimitExceeded,
    "O2C-004": ShortPayment,
    "O2C-005": OverpaymentNotReturned,
    "O2C-006": RevenueRecognitionTiming,
    "O2C-007": FictitiousCustomer,
    "O2C-008": RoundTripping,
    "O2C-009": ChannelStuffing,
    "O2C-010": SideAgreements,
}
