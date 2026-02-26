"""Discrepancy Catalog — Registry of 35+ discrepancy type implementations.

The ``DiscrepancyCatalog`` maps type codes (P2P-001 through CTL-005) to their
concrete ``BaseDiscrepancy`` implementation classes, parameter bounds, base
injection rates, categories, and difficulty levels.

Type Code Ranges:
    P2P-001 through P2P-015 — 15 Procure-to-Pay discrepancies
    O2C-001 through O2C-010 — 10 Order-to-Cash discrepancies
    GL-001 through GL-005   — 5 General Ledger discrepancies
    CTL-001 through CTL-005 — 5 Control discrepancies

Configuration:
    The catalog can be populated either programmatically via ``register()``
    or by loading YAML configuration from config/discrepancies/*.yaml files.

References:
    - AAP Section 0.5.1 Group 5: discrepancy_catalog.py
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.1: Constructor injection, Pydantic V2
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
)
import importlib

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.constants import DISCREPANCY_DEFAULTS

if TYPE_CHECKING:
    from app.discrepancies.base_discrepancy import BaseDiscrepancy

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# CatalogEntry — Pydantic V2 data model for a single catalog entry
# ---------------------------------------------------------------------------


class CatalogEntry(BaseModel):
    """A single entry in the discrepancy catalog.

    Each entry maps a unique type code (e.g. ``'P2P-001'``) to the metadata
    and implementation details needed by the :class:`DiscrepancyInjector` to
    instantiate and invoke the correct :class:`BaseDiscrepancy` subclass.

    Attributes:
        type_code: Unique type code (e.g. ``'P2P-001'``).
        name: Human-readable name (e.g. ``'Duplicate Invoice'``).
        category: Category — ``'p2p'``, ``'o2c'``, ``'gl'``, or ``'control'``.
        difficulty: Difficulty — ``'easy'`` or ``'medium'``.
        base_rate: Base injection rate weight for this type (Decimal, NEVER float).
        description: Description of the discrepancy.
        detection_method: Expected detection method (e.g. ``'duplicate_check'``).
        parameter_bounds: Parameter name → ``{min, max, type}`` bounds mapping.
        implementation_class: Dotted path to the ``BaseDiscrepancy`` subclass
            (e.g. ``'app.discrepancies.p2p.duplicate_invoice.DuplicateInvoice'``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type_code: str = Field(
        ..., description="Unique type code (e.g., 'P2P-001')"
    )
    name: str = Field(
        ..., description="Human-readable name (e.g., 'Duplicate Invoice')"
    )
    category: str = Field(
        ..., description="Category: 'p2p', 'o2c', 'gl', or 'control'"
    )
    difficulty: str = Field(
        ..., description="Difficulty: 'easy' or 'medium'"
    )
    base_rate: Decimal = Field(
        ..., description="Base injection rate weight for this type"
    )
    description: str = Field(
        default="", description="Description of the discrepancy"
    )
    detection_method: str = Field(
        default="", description="Expected detection method"
    )
    parameter_bounds: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameter name → {min, max, type} bounds",
    )
    implementation_class: Optional[str] = Field(
        default=None,
        description=(
            "Dotted path to BaseDiscrepancy subclass "
            "(e.g., 'app.discrepancies.p2p.duplicate_invoice.DuplicateInvoice')"
        ),
    )


# ---------------------------------------------------------------------------
# DiscrepancyCatalog — Central registry of all discrepancy types
# ---------------------------------------------------------------------------


class DiscrepancyCatalog:
    """Central registry of all discrepancy type implementations.

    Supports lookup by type_code, category, and difficulty level.
    Can be populated programmatically or from YAML configuration.

    The catalog stores :class:`CatalogEntry` objects keyed by type code and
    maintains an optional class reference cache for lazily-resolved
    :class:`BaseDiscrepancy` subclasses.

    Usage::

        catalog = DiscrepancyCatalog()
        catalog.register_defaults()  # registers all 35 types
        entry = catalog.get_entry("P2P-001")
        p2p_types = catalog.get_by_category("p2p")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialize an empty catalog with no registered entries."""
        self._entries: Dict[str, CatalogEntry] = {}
        self._class_cache: Dict[str, Type[BaseDiscrepancy]] = {}
        logger.info(
            "discrepancy_catalog_initialized",
            service_name="transactions",
            component="DiscrepancyCatalog",
        )

    # ------------------------------------------------------------------
    # Registration Methods
    # ------------------------------------------------------------------

    def register(
        self,
        entry: CatalogEntry,
        class_ref: Optional[Type[BaseDiscrepancy]] = None,
    ) -> None:
        """Register a discrepancy type in the catalog.

        Args:
            entry: CatalogEntry with type code, category, etc.
            class_ref: Optional direct class reference (avoids import-time
                resolution).

        Raises:
            ValueError: If type_code is already registered.
        """
        if entry.type_code in self._entries:
            raise ValueError(
                f"Discrepancy type '{entry.type_code}' is already registered "
                f"in the catalog"
            )
        self._entries[entry.type_code] = entry
        if class_ref is not None:
            self._class_cache[entry.type_code] = class_ref
        logger.debug(
            "discrepancy_type_registered",
            service_name="transactions",
            component="DiscrepancyCatalog",
            type_code=entry.type_code,
            name=entry.name,
            category=entry.category,
            difficulty=entry.difficulty,
        )

    def register_defaults(self) -> None:
        """Register all 35 default discrepancy types.

        Creates CatalogEntry objects for all P2P (15), O2C (10), GL (5),
        and Control (5) discrepancy types with their parameter bounds,
        base rates, and implementation class paths.

        Uses :data:`DISCREPANCY_DEFAULTS` from
        ``app.transactions.constants`` for the default ``base_rate`` value
        (``injection_rate``), difficulty distribution validation, and
        auto-adjust configuration.
        """
        default_rate = DISCREPANCY_DEFAULTS["injection_rate"]
        # Access remaining DISCREPANCY_DEFAULTS members for validation context
        difficulty_dist = DISCREPANCY_DEFAULTS["difficulty_distribution"]
        rate_tolerance = DISCREPANCY_DEFAULTS["rate_tolerance"]
        difficulty_tolerance = DISCREPANCY_DEFAULTS["difficulty_tolerance"]
        auto_adjust = DISCREPANCY_DEFAULTS["auto_adjust_to_bounds"]

        logger.debug(
            "registering_default_discrepancies",
            service_name="transactions",
            component="DiscrepancyCatalog",
            default_injection_rate=str(default_rate),
            rate_tolerance=str(rate_tolerance),
            difficulty_distribution={
                k: str(v) for k, v in difficulty_dist.items()
            },
            difficulty_tolerance=str(difficulty_tolerance),
            auto_adjust_to_bounds=auto_adjust,
        )

        # ---------------------------------------------------------------
        # P2P Discrepancies (15 types: P2P-001 through P2P-015)
        # ---------------------------------------------------------------
        self._register_p2p_types(default_rate)

        # ---------------------------------------------------------------
        # O2C Discrepancies (10 types: O2C-001 through O2C-010)
        # ---------------------------------------------------------------
        self._register_o2c_types(default_rate)

        # ---------------------------------------------------------------
        # GL Discrepancies (5 types: GL-001 through GL-005)
        # ---------------------------------------------------------------
        self._register_gl_types(default_rate)

        # ---------------------------------------------------------------
        # Control Discrepancies (5 types: CTL-001 through CTL-005)
        # ---------------------------------------------------------------
        self._register_control_types(default_rate)

        logger.info(
            "discrepancy_defaults_registered",
            service_name="transactions",
            component="DiscrepancyCatalog",
            total_types=len(self._entries),
            p2p_count=len(self.get_by_category("p2p")),
            o2c_count=len(self.get_by_category("o2c")),
            gl_count=len(self.get_by_category("gl")),
            control_count=len(self.get_by_category("control")),
        )

    # ------------------------------------------------------------------
    # Private: P2P Type Registration
    # ------------------------------------------------------------------

    def _register_p2p_types(self, default_rate: Decimal) -> None:
        """Register all 15 P2P discrepancy types."""
        # P2P-001: Duplicate Invoice
        self.register(CatalogEntry(
            type_code="P2P-001",
            name="Duplicate Invoice",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Exact or near-duplicate vendor invoice submitted for payment. "
                "May involve minor variations in invoice number, date, or amount "
                "to evade simple duplicate detection."
            ),
            detection_method="duplicate_check",
            parameter_bounds={
                "days_apart": {"min": 1, "max": 90, "type": "int"},
                "amount_variance_pct": {"min": 0, "max": 5, "type": "float"},
                "number_variation": {"min": 0, "max": 1, "type": "bool"},
            },
            implementation_class=(
                "app.discrepancies.p2p.duplicate_invoice.DuplicateInvoice"
            ),
        ))

        # P2P-002: Price Mismatch
        self.register(CatalogEntry(
            type_code="P2P-002",
            name="Invoice/PO Price Mismatch",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Invoice unit price differs from the purchase order price by "
                "more than the ±5% three-way match tolerance. Detected during "
                "PO-Receipt-Invoice matching."
            ),
            detection_method="three_way_match",
            parameter_bounds={
                "variance_percent": {"min": 1, "max": 50, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.p2p.price_mismatch.PriceMismatch"
            ),
        ))

        # P2P-003: Quantity Variance
        self.register(CatalogEntry(
            type_code="P2P-003",
            name="Invoice/Receipt Quantity Variance",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Invoice quantity differs from the goods receipt quantity by "
                "more than the ±2% three-way match tolerance. Detected during "
                "three-way matching."
            ),
            detection_method="three_way_match",
            parameter_bounds={
                "variance_percent": {"min": 1, "max": 30, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.p2p.quantity_variance.QuantityVariance"
            ),
        ))

        # P2P-004: Missing PO
        self.register(CatalogEntry(
            type_code="P2P-004",
            name="Missing Purchase Order",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Vendor invoice submitted without a corresponding purchase "
                "order reference. Indicates a bypass of the procurement process."
            ),
            detection_method="document_linkage",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.p2p.missing_po.MissingPO"
            ),
        ))

        # P2P-005: PO Not Approved
        self.register(CatalogEntry(
            type_code="P2P-005",
            name="PO Not Approved",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Purchase order was processed without required approval or with "
                "an insufficient approval level for the amount."
            ),
            detection_method="approval_check",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.p2p.po_not_approved.PONotApproved"
            ),
        ))

        # P2P-006: Invoice Before Receipt
        self.register(CatalogEntry(
            type_code="P2P-006",
            name="Invoice Before Receipt",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Vendor invoice date is earlier than the goods receipt date, "
                "violating the expected temporal sequence of procure-to-pay."
            ),
            detection_method="date_sequence",
            parameter_bounds={
                "days_before": {"min": 1, "max": 30, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.p2p.invoice_before_receipt.InvoiceBeforeReceipt"
            ),
        ))

        # P2P-007: Round-Dollar Invoice
        self.register(CatalogEntry(
            type_code="P2P-007",
            name="Round-Dollar Invoice",
            category="p2p",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Invoice total is a suspiciously round dollar amount (e.g. "
                "$1,000.00, $10,000.00), which is a known fraud indicator."
            ),
            detection_method="amount_pattern",
            parameter_bounds={
                "round_to": {"min": 100, "max": 10000, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.p2p.round_dollar_invoice.RoundDollarInvoice"
            ),
        ))

        # P2P-008: Weekend Processing
        self.register(CatalogEntry(
            type_code="P2P-008",
            name="Weekend Processing",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Transaction processed on a Saturday or Sunday, which is "
                "unusual for standard business operations and may indicate "
                "unauthorized activity."
            ),
            detection_method="date_check",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.p2p.weekend_processing.WeekendProcessing"
            ),
        ))

        # P2P-009: Duplicate Payment
        self.register(CatalogEntry(
            type_code="P2P-009",
            name="Duplicate Payment",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Same vendor paid the same amount within a short period, "
                "indicating a potential duplicate payment."
            ),
            detection_method="duplicate_check",
            parameter_bounds={
                "days_apart": {"min": 1, "max": 90, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.p2p.duplicate_payment.DuplicatePayment"
            ),
        ))

        # P2P-010: Payment Before Invoice
        self.register(CatalogEntry(
            type_code="P2P-010",
            name="Payment Before Invoice Date",
            category="p2p",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Vendor payment date precedes the invoice date, violating "
                "the expected temporal sequence."
            ),
            detection_method="date_sequence",
            parameter_bounds={
                "days_before": {"min": 1, "max": 30, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.p2p.payment_before_invoice.PaymentBeforeInvoice"
            ),
        ))

        # P2P-011: Unapproved Vendor
        self.register(CatalogEntry(
            type_code="P2P-011",
            name="Vendor Not in Approved List",
            category="p2p",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Transaction processed with a vendor that is not on the "
                "approved vendor list, bypassing vendor qualification controls."
            ),
            detection_method="vendor_validation",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.p2p.unapproved_vendor.UnapprovedVendor"
            ),
        ))

        # P2P-012: Split PO
        self.register(CatalogEntry(
            type_code="P2P-012",
            name="Split PO to Avoid Approval",
            category="p2p",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "A large purchase is split across multiple purchase orders "
                "to keep each below the approval threshold, circumventing "
                "approval controls."
            ),
            detection_method="pattern_analysis",
            parameter_bounds={
                "split_count": {"min": 2, "max": 5, "type": "int"},
                "threshold_amount": {
                    "min": Decimal("1000"),
                    "max": Decimal("100000"),
                    "type": "decimal",
                },
            },
            implementation_class=(
                "app.discrepancies.p2p.split_po.SplitPO"
            ),
        ))

        # P2P-013: Fictitious Vendor
        self.register(CatalogEntry(
            type_code="P2P-013",
            name="Fictitious Vendor",
            category="p2p",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Vendor address or phone number matches an employee record, "
                "indicating a potentially fictitious vendor created for fraud."
            ),
            detection_method="entity_validation",
            parameter_bounds={
                "match_type": {"min": 0, "max": 2, "type": "enum",
                               "values": ["address", "phone", "both"]},
            },
            implementation_class=(
                "app.discrepancies.p2p.fictitious_vendor.FictitiousVendor"
            ),
        ))

        # P2P-014: Vendor Concentration
        self.register(CatalogEntry(
            type_code="P2P-014",
            name="Unusual Vendor Concentration",
            category="p2p",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Disproportionate spend concentrated on a single vendor, "
                "indicating potential kickback or collusion risk."
            ),
            detection_method="statistical_analysis",
            parameter_bounds={
                "concentration_threshold": {
                    "min": Decimal("0.3"),
                    "max": Decimal("0.9"),
                    "type": "decimal",
                },
            },
            implementation_class=(
                "app.discrepancies.p2p.vendor_concentration.VendorConcentration"
            ),
        ))

        # P2P-015: Ghost Expense
        self.register(CatalogEntry(
            type_code="P2P-015",
            name="Ghost Expense",
            category="p2p",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Expense recorded without supporting documentation such as "
                "a receipt, invoice, or purchase order."
            ),
            detection_method="document_review",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.p2p.ghost_expense.GhostExpense"
            ),
        ))

    # ------------------------------------------------------------------
    # Private: O2C Type Registration
    # ------------------------------------------------------------------

    def _register_o2c_types(self, default_rate: Decimal) -> None:
        """Register all 10 O2C discrepancy types."""
        # O2C-001: Duplicate Customer Invoice
        self.register(CatalogEntry(
            type_code="O2C-001",
            name="Duplicate Customer Invoice",
            category="o2c",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Customer invoice created more than once for the same shipment "
                "or sales order, resulting in duplicate billing."
            ),
            detection_method="duplicate_check",
            parameter_bounds={
                "days_apart": {"min": 1, "max": 90, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.o2c.duplicate_customer_invoice.DuplicateCustomerInvoice"
            ),
        ))

        # O2C-002: Invoice Without Shipment
        self.register(CatalogEntry(
            type_code="O2C-002",
            name="Invoice Without Shipment",
            category="o2c",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Customer invoice created without a corresponding shipment "
                "record, indicating potential premature or fictitious billing."
            ),
            detection_method="document_linkage",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.o2c.invoice_without_shipment.InvoiceWithoutShipment"
            ),
        ))

        # O2C-003: Credit Limit Exceeded
        self.register(CatalogEntry(
            type_code="O2C-003",
            name="Credit Limit Exceeded",
            category="o2c",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Sales order accepted or invoice issued to a customer whose "
                "current AR balance plus the new order exceeds their credit limit."
            ),
            detection_method="credit_check",
            parameter_bounds={
                "excess_percent": {"min": 1, "max": 50, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.o2c.credit_limit_exceeded.CreditLimitExceeded"
            ),
        ))

        # O2C-004: Short Payment
        self.register(CatalogEntry(
            type_code="O2C-004",
            name="Short Payment",
            category="o2c",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Customer payment amount is less than the invoice amount "
                "without an authorized deduction or discount."
            ),
            detection_method="payment_matching",
            parameter_bounds={
                "short_percent": {"min": 1, "max": 20, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.o2c.short_payment.ShortPayment"
            ),
        ))

        # O2C-005: Overpayment Not Returned
        self.register(CatalogEntry(
            type_code="O2C-005",
            name="Overpayment Not Returned",
            category="o2c",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Customer overpayment recorded as unapplied cash but never "
                "returned or applied to future invoices."
            ),
            detection_method="payment_matching",
            parameter_bounds={
                "overpay_percent": {"min": 1, "max": 30, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.o2c.overpayment_not_returned.OverpaymentNotReturned"
            ),
        ))

        # O2C-006: Revenue Recognition Timing
        self.register(CatalogEntry(
            type_code="O2C-006",
            name="Revenue Recognition Timing Error",
            category="o2c",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Revenue recognized in the wrong fiscal period — booked too "
                "early relative to delivery or performance obligation completion."
            ),
            detection_method="period_analysis",
            parameter_bounds={
                "days_early": {"min": 1, "max": 60, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.o2c.revenue_recognition_timing.RevenueRecognitionTiming"
            ),
        ))

        # O2C-007: Fictitious Customer
        self.register(CatalogEntry(
            type_code="O2C-007",
            name="Fictitious Customer",
            category="o2c",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Sales transactions recorded for a fictitious or non-existent "
                "customer entity, indicating potential revenue manipulation."
            ),
            detection_method="entity_validation",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.o2c.fictitious_customer.FictitiousCustomer"
            ),
        ))

        # O2C-008: Round-Tripping
        self.register(CatalogEntry(
            type_code="O2C-008",
            name="Round-Tripping",
            category="o2c",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Circular transactions where goods or services are sold to a "
                "party and simultaneously purchased back, artificially inflating "
                "revenue."
            ),
            detection_method="pattern_analysis",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.o2c.round_tripping.RoundTripping"
            ),
        ))

        # O2C-009: Channel Stuffing
        self.register(CatalogEntry(
            type_code="O2C-009",
            name="Channel Stuffing",
            category="o2c",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Premature shipment of goods to customers or distribution "
                "channels to inflate current-period revenue."
            ),
            detection_method="pattern_analysis",
            parameter_bounds={
                "volume_multiplier": {
                    "min": Decimal("1.5"),
                    "max": Decimal("5.0"),
                    "type": "decimal",
                },
            },
            implementation_class=(
                "app.discrepancies.o2c.channel_stuffing.ChannelStuffing"
            ),
        ))

        # O2C-010: Side Agreements
        self.register(CatalogEntry(
            type_code="O2C-010",
            name="Side Agreements Not Disclosed",
            category="o2c",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Undisclosed side agreements granting customers unapproved "
                "discounts, return privileges, or extended payment terms."
            ),
            detection_method="document_review",
            parameter_bounds={
                "discount_percent": {"min": 5, "max": 30, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.o2c.side_agreements.SideAgreements"
            ),
        ))

    # ------------------------------------------------------------------
    # Private: GL Type Registration
    # ------------------------------------------------------------------

    def _register_gl_types(self, default_rate: Decimal) -> None:
        """Register all 5 GL discrepancy types."""
        # GL-001: Unbalanced Journal Entry
        self.register(CatalogEntry(
            type_code="GL-001",
            name="Unbalanced Journal Entry",
            category="gl",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Journal entry where SUM(debits) does not equal SUM(credits) "
                "within the $0.01 tolerance, violating double-entry bookkeeping."
            ),
            detection_method="balance_check",
            parameter_bounds={
                "imbalance_amount": {
                    "min": Decimal("0.02"),
                    "max": Decimal("1000.00"),
                    "type": "decimal",
                },
            },
            implementation_class=(
                "app.discrepancies.gl.unbalanced_journal.UnbalancedJournal"
            ),
        ))

        # GL-002: Journal Entry Without Approval
        self.register(CatalogEntry(
            type_code="GL-002",
            name="Journal Entry Without Approval",
            category="gl",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Journal entry posted without the required approval based "
                "on the monetary amount and approval threshold configuration."
            ),
            detection_method="approval_check",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.gl.journal_no_approval.JournalNoApproval"
            ),
        ))

        # GL-003: Suspicious Period-End Adjusting Entry
        self.register(CatalogEntry(
            type_code="GL-003",
            name="Suspicious Period-End Adjusting Entry",
            category="gl",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Adjusting journal entry posted in the final days before "
                "period close, which may indicate earnings management or "
                "fraudulent financial reporting."
            ),
            detection_method="period_analysis",
            parameter_bounds={
                "days_before_close": {"min": 1, "max": 5, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.gl.suspicious_adjusting.SuspiciousAdjusting"
            ),
        ))

        # GL-004: Unusual Account Combination
        self.register(CatalogEntry(
            type_code="GL-004",
            name="Unusual Account Combination",
            category="gl",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Journal entry uses a debit/credit account combination that "
                "does not match any standard posting rule, suggesting manual "
                "override or error."
            ),
            detection_method="pattern_analysis",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.gl.unusual_account_combo.UnusualAccountCombo"
            ),
        ))

        # GL-005: Manual Override
        self.register(CatalogEntry(
            type_code="GL-005",
            name="Manual Entry Overriding System Entry",
            category="gl",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Manual journal entry posted that overrides or reverses a "
                "system-generated entry without proper authorization."
            ),
            detection_method="system_log_review",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.gl.manual_override.ManualOverride"
            ),
        ))

    # ------------------------------------------------------------------
    # Private: Control Type Registration
    # ------------------------------------------------------------------

    def _register_control_types(self, default_rate: Decimal) -> None:
        """Register all 5 Control discrepancy types."""
        # CTL-001: Segregation of Duties Violation
        self.register(CatalogEntry(
            type_code="CTL-001",
            name="Segregation of Duties Violation",
            category="control",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "A single user performed multiple incompatible functions in "
                "the same transaction (e.g. created vendor and approved "
                "payment), violating segregation of duties controls."
            ),
            detection_method="access_review",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.control.sod_violation.SoDViolation"
            ),
        ))

        # CTL-002: Self-Approval
        self.register(CatalogEntry(
            type_code="CTL-002",
            name="Same User Created and Approved",
            category="control",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "The same user who created a transaction also approved it, "
                "bypassing the maker-checker control."
            ),
            detection_method="approval_check",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.control.self_approval.SelfApproval"
            ),
        ))

        # CTL-003: Approval Limit Exceeded
        self.register(CatalogEntry(
            type_code="CTL-003",
            name="Approval Limit Exceeded",
            category="control",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Approver approved a transaction that exceeds their "
                "authorized approval limit."
            ),
            detection_method="approval_check",
            parameter_bounds={
                "excess_percent": {"min": 1, "max": 100, "type": "float"},
            },
            implementation_class=(
                "app.discrepancies.control.approval_limit_exceeded.ApprovalLimitExceeded"
            ),
        ))

        # CTL-004: Backdated Transaction
        self.register(CatalogEntry(
            type_code="CTL-004",
            name="Backdated Transaction",
            category="control",
            difficulty="medium",
            base_rate=default_rate,
            description=(
                "Transaction date is set to a past date, potentially to "
                "manipulate period-end financial results."
            ),
            detection_method="date_check",
            parameter_bounds={
                "days_back": {"min": 1, "max": 90, "type": "int"},
            },
            implementation_class=(
                "app.discrepancies.control.backdated_transaction.BackdatedTransaction"
            ),
        ))

        # CTL-005: Holiday Transaction
        self.register(CatalogEntry(
            type_code="CTL-005",
            name="Transaction on Holiday/Weekend",
            category="control",
            difficulty="easy",
            base_rate=default_rate,
            description=(
                "Transaction processed on a recognized holiday or weekend, "
                "which is unusual for standard business operations."
            ),
            detection_method="date_check",
            parameter_bounds={},
            implementation_class=(
                "app.discrepancies.control.holiday_transaction.HolidayTransaction"
            ),
        ))

    # ------------------------------------------------------------------
    # Lookup Methods
    # ------------------------------------------------------------------

    def get_entry(self, type_code: str) -> Optional[CatalogEntry]:
        """Return catalog entry by type code or None if not found.

        Args:
            type_code: The discrepancy type code (e.g. ``'P2P-001'``).

        Returns:
            The :class:`CatalogEntry` if registered, otherwise ``None``.
        """
        return self._entries.get(type_code)

    def get_by_category(self, category: str) -> List[CatalogEntry]:
        """Return all entries in a category.

        Args:
            category: Category to filter by — ``'p2p'``, ``'o2c'``,
                ``'gl'``, or ``'control'``.

        Returns:
            List of matching :class:`CatalogEntry` objects (may be empty).
        """
        return [
            entry for entry in self._entries.values()
            if entry.category == category
        ]

    def get_by_difficulty(self, difficulty: str) -> List[CatalogEntry]:
        """Return all entries at a difficulty level.

        Args:
            difficulty: Difficulty to filter by — ``'easy'`` or ``'medium'``.

        Returns:
            List of matching :class:`CatalogEntry` objects (may be empty).
        """
        return [
            entry for entry in self._entries.values()
            if entry.difficulty == difficulty
        ]

    def get_by_category_and_difficulty(
        self,
        category: str,
        difficulty: str,
    ) -> List[CatalogEntry]:
        """Return entries matching both category and difficulty.

        Args:
            category: Category to filter by.
            difficulty: Difficulty to filter by.

        Returns:
            List of matching :class:`CatalogEntry` objects (may be empty).
        """
        return [
            entry for entry in self._entries.values()
            if entry.category == category and entry.difficulty == difficulty
        ]

    def get_implementation_class(
        self, type_code: str
    ) -> Optional[Type[BaseDiscrepancy]]:
        """Return the implementation class for a type code.

        Uses the class cache first.  If not cached, dynamically imports
        the class from the dotted ``implementation_class`` path stored in
        the :class:`CatalogEntry`.

        Args:
            type_code: The discrepancy type code (e.g. ``'P2P-001'``).

        Returns:
            The :class:`BaseDiscrepancy` subclass, or ``None`` if the type
            code is not registered or the implementation class path is not
            set.
        """
        # Check cache first
        if type_code in self._class_cache:
            return self._class_cache[type_code]

        entry = self._entries.get(type_code)
        if entry is None or entry.implementation_class is None:
            return None

        # Dynamic import: split dotted path into module and class name
        dotted_path = entry.implementation_class
        try:
            module_path, class_name = dotted_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._class_cache[type_code] = cls
            logger.debug(
                "discrepancy_class_loaded",
                service_name="transactions",
                component="DiscrepancyCatalog",
                type_code=type_code,
                class_path=dotted_path,
            )
            return cls
        except (ImportError, AttributeError, ValueError) as exc:
            logger.warning(
                "discrepancy_class_load_failed",
                service_name="transactions",
                component="DiscrepancyCatalog",
                type_code=type_code,
                class_path=dotted_path,
                error=str(exc),
            )
            return None

    def get_parameter_bounds(
        self, type_code: str
    ) -> Dict[str, Dict[str, Any]]:
        """Return parameter bounds for a type code.

        Args:
            type_code: The discrepancy type code (e.g. ``'P2P-001'``).

        Returns:
            The parameter bounds dictionary, or an empty dict if the type
            code is not registered.
        """
        entry = self._entries.get(type_code)
        if entry is None:
            return {}
        return entry.parameter_bounds

    def get_all_type_codes(self) -> List[str]:
        """Return all registered type codes.

        Returns:
            Sorted list of all registered type code strings.
        """
        return sorted(self._entries.keys())

    def get_metrics(self) -> Dict[str, Any]:
        """Return catalog metrics and diagnostic statistics.

        Returns:
            Dictionary with total entries, per-category counts,
            per-difficulty counts, and list of registered type codes.
        """
        category_counts: Dict[str, int] = {}
        difficulty_counts: Dict[str, int] = {}

        for entry in self._entries.values():
            category_counts[entry.category] = (
                category_counts.get(entry.category, 0) + 1
            )
            difficulty_counts[entry.difficulty] = (
                difficulty_counts.get(entry.difficulty, 0) + 1
            )

        metrics: Dict[str, Any] = {
            "total_entries": len(self._entries),
            "category_counts": category_counts,
            "difficulty_counts": difficulty_counts,
            "type_codes": self.get_all_type_codes(),
            "default_injection_rate": str(
                DISCREPANCY_DEFAULTS["injection_rate"]
            ),
            "rate_tolerance": str(DISCREPANCY_DEFAULTS["rate_tolerance"]),
            "difficulty_distribution": {
                k: str(v)
                for k, v in DISCREPANCY_DEFAULTS[
                    "difficulty_distribution"
                ].items()
            },
            "difficulty_tolerance": str(
                DISCREPANCY_DEFAULTS["difficulty_tolerance"]
            ),
            "auto_adjust_to_bounds": DISCREPANCY_DEFAULTS[
                "auto_adjust_to_bounds"
            ],
        }

        logger.debug(
            "catalog_metrics_retrieved",
            service_name="transactions",
            component="DiscrepancyCatalog",
            total_entries=metrics["total_entries"],
            category_counts=category_counts,
            difficulty_counts=difficulty_counts,
        )

        return metrics

    # ------------------------------------------------------------------
    # YAML Loading
    # ------------------------------------------------------------------

    def load_from_yaml(self, config_dir: str = "config/discrepancies") -> None:
        """Load discrepancy type definitions from YAML config files.

        Reads the following YAML files from the specified directory:

        - ``p2p_discrepancies.yaml``
        - ``o2c_discrepancies.yaml``
        - ``gl_discrepancies.yaml``
        - ``discrepancy_rates.yaml``

        Each YAML file is expected to contain a ``discrepancies`` key with
        a list of type definitions that map directly to :class:`CatalogEntry`
        fields.

        Args:
            config_dir: Path to the directory containing YAML config files.
                Defaults to ``'config/discrepancies'``.

        Raises:
            ImportError: If the ``yaml`` (PyYAML) library is not installed.
                The error is logged and the method returns without loading.
        """
        try:
            import yaml  # noqa: F811
        except ImportError:
            logger.warning(
                "yaml_library_not_available",
                service_name="transactions",
                component="DiscrepancyCatalog",
                message=(
                    "PyYAML is not installed. Cannot load YAML configuration. "
                    "Use register_defaults() for programmatic registration."
                ),
            )
            return

        config_path = Path(config_dir)
        yaml_files = [
            "p2p_discrepancies.yaml",
            "o2c_discrepancies.yaml",
            "gl_discrepancies.yaml",
            "discrepancy_rates.yaml",
        ]

        loaded_count = 0
        for yaml_file in yaml_files:
            file_path = config_path / yaml_file
            if not file_path.exists():
                logger.debug(
                    "yaml_config_file_not_found",
                    service_name="transactions",
                    component="DiscrepancyCatalog",
                    file_path=str(file_path),
                )
                continue

            try:
                raw_text = file_path.read_text(encoding="utf-8")
                data = yaml.safe_load(raw_text)
                if data is None:
                    continue

                # Handle discrepancy_rates.yaml (global config, not entries)
                if yaml_file == "discrepancy_rates.yaml":
                    logger.debug(
                        "yaml_rates_config_loaded",
                        service_name="transactions",
                        component="DiscrepancyCatalog",
                        file_path=str(file_path),
                    )
                    continue

                # Process discrepancy type definitions
                discrepancies = data.get("discrepancies", [])
                if isinstance(discrepancies, list):
                    for disc_def in discrepancies:
                        if not isinstance(disc_def, dict):
                            continue
                        type_code = disc_def.get("type_code", "")
                        if not type_code or type_code in self._entries:
                            continue

                        # Convert base_rate to Decimal if present
                        raw_rate = disc_def.get("base_rate")
                        base_rate = (
                            Decimal(str(raw_rate))
                            if raw_rate is not None
                            else DISCREPANCY_DEFAULTS["injection_rate"]
                        )

                        # Convert parameter bounds numeric values to proper types
                        param_bounds = disc_def.get("parameter_bounds", {})
                        converted_bounds: Dict[str, Dict[str, Any]] = {}
                        if isinstance(param_bounds, dict):
                            for param_name, bound_spec in param_bounds.items():
                                if isinstance(bound_spec, dict):
                                    converted_spec: Dict[str, Any] = {}
                                    for k, v in bound_spec.items():
                                        if k in ("min", "max") and isinstance(v, (int, float)):
                                            bound_type = bound_spec.get("type", "float")
                                            if bound_type == "decimal":
                                                converted_spec[k] = Decimal(str(v))
                                            else:
                                                converted_spec[k] = v
                                        else:
                                            converted_spec[k] = v
                                    converted_bounds[param_name] = converted_spec

                        entry = CatalogEntry(
                            type_code=type_code,
                            name=disc_def.get("name", type_code),
                            category=disc_def.get("category", ""),
                            difficulty=disc_def.get("difficulty", "easy"),
                            base_rate=base_rate,
                            description=disc_def.get("description", ""),
                            detection_method=disc_def.get("detection_method", ""),
                            parameter_bounds=converted_bounds,
                            implementation_class=disc_def.get(
                                "implementation_class"
                            ),
                        )
                        self.register(entry)
                        loaded_count += 1

            except yaml.YAMLError as exc:
                logger.error(
                    "yaml_parse_error",
                    service_name="transactions",
                    component="DiscrepancyCatalog",
                    file_path=str(file_path),
                    error=str(exc),
                )
            except Exception as exc:
                logger.error(
                    "yaml_load_error",
                    service_name="transactions",
                    component="DiscrepancyCatalog",
                    file_path=str(file_path),
                    error=str(exc),
                )

        logger.info(
            "yaml_config_loaded",
            service_name="transactions",
            component="DiscrepancyCatalog",
            loaded_count=loaded_count,
            total_entries=len(self._entries),
        )
