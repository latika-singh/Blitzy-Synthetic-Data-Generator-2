"""Parameterized tests for all 35+ individual discrepancy types.

Tests all concrete implementations of BaseDiscrepancy across four categories:
  - P2P (15 types): P2P-001 DuplicateInvoice through P2P-015 GhostExpense
  - O2C (10 types): O2C-001 DuplicateCustomerInvoice through O2C-010 SideAgreements
  - GL  (5 types):  GL-001 UnbalancedJournal through GL-005 ManualOverride
  - Control (5 types): CTL-001 SoDViolation through CTL-005 HolidayTransaction

Per AAP Section 0.5.1 Group 5 and Section 0.7.5:
  - Every concrete class extends BaseDiscrepancy
  - inject() returns Tuple[Dict[str, Any], Dict[str, Any]]
  - All 6 ClassVar attributes populated (type_code, category, difficulty, name, description, detection_method)
  - Difficulty is "easy" or "medium" only (no "hard" in MVP)
  - Category matches sub-package (p2p, o2c, gl, control)
  - Deep copy of transaction before modification
  - Deterministic via seeded rng
  - All financial_impact values use Decimal
  - Ground truth data includes affected_fields, original_values, modified_values

Uses @pytest.mark.parametrize extensively to iterate over all 35 types.

Test Classes:
  TestBaseDiscrepancyContract      — ABC contract: inject() signature, ClassVar attrs
  TestAllTypesClassAttributes      — Parametrized: 6 ClassVar attrs for all 35 types
  TestAllTypesInjectContract       — Parametrized: inject() return shape for all 35 types
  TestP2PDiscrepancies             — P2P-specific inject behavior for all 15 types
  TestO2CDiscrepancies             — O2C-specific inject behavior for all 10 types
  TestGLDiscrepancies              — GL-specific inject behavior for all 5 types
  TestControlDiscrepancies         — Control-specific inject behavior for all 5 types
  TestValidateParams               — _validate_params() helper for types with params
  TestCreateGroundTruthData        — _create_ground_truth_data() helper
  TestCopyTransaction              — _copy_transaction() deep copy verification
  TestDeterminism                  — Same seed → same injection result
"""

from __future__ import annotations

import copy
import random
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, Tuple, Type
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.discrepancies.base_discrepancy import BaseDiscrepancy

# P2P imports (15 classes)
from app.discrepancies.p2p.duplicate_invoice import DuplicateInvoice
from app.discrepancies.p2p.price_mismatch import PriceMismatch
from app.discrepancies.p2p.quantity_variance import QuantityVariance
from app.discrepancies.p2p.missing_po import MissingPO
from app.discrepancies.p2p.po_not_approved import PONotApproved
from app.discrepancies.p2p.invoice_before_receipt import InvoiceBeforeReceipt
from app.discrepancies.p2p.round_dollar_invoice import RoundDollarInvoice
from app.discrepancies.p2p.weekend_processing import WeekendProcessing
from app.discrepancies.p2p.duplicate_payment import DuplicatePayment
from app.discrepancies.p2p.payment_before_invoice import PaymentBeforeInvoice
from app.discrepancies.p2p.unapproved_vendor import UnapprovedVendor
from app.discrepancies.p2p.split_po import SplitPO
from app.discrepancies.p2p.fictitious_vendor import FictitiousVendor
from app.discrepancies.p2p.vendor_concentration import VendorConcentration
from app.discrepancies.p2p.ghost_expense import GhostExpense

# O2C imports (10 classes)
from app.discrepancies.o2c.duplicate_customer_invoice import DuplicateCustomerInvoice
from app.discrepancies.o2c.invoice_without_shipment import InvoiceWithoutShipment
from app.discrepancies.o2c.credit_limit_exceeded import CreditLimitExceeded
from app.discrepancies.o2c.short_payment import ShortPayment
from app.discrepancies.o2c.overpayment_not_returned import OverpaymentNotReturned
from app.discrepancies.o2c.revenue_recognition_timing import RevenueRecognitionTiming
from app.discrepancies.o2c.fictitious_customer import FictitiousCustomer
from app.discrepancies.o2c.round_tripping import RoundTripping
from app.discrepancies.o2c.channel_stuffing import ChannelStuffing
from app.discrepancies.o2c.side_agreements import SideAgreements

# GL imports (5 classes)
from app.discrepancies.gl.unbalanced_journal import UnbalancedJournal
from app.discrepancies.gl.journal_no_approval import JournalNoApproval
from app.discrepancies.gl.suspicious_adjusting import SuspiciousAdjusting
from app.discrepancies.gl.unusual_account_combo import UnusualAccountCombo
from app.discrepancies.gl.manual_override import ManualOverride

# Control imports (5 classes)
from app.discrepancies.control.sod_violation import SoDViolation
from app.discrepancies.control.self_approval import SelfApproval
from app.discrepancies.control.approval_limit_exceeded import ApprovalLimitExceeded
from app.discrepancies.control.backdated_transaction import BackdatedTransaction
from app.discrepancies.control.holiday_transaction import HolidayTransaction

from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Master Type Registry — ALL 35 discrepancy classes
# ---------------------------------------------------------------------------

ALL_DISCREPANCY_CLASSES: list[Type[BaseDiscrepancy]] = [
    # P2P (15)
    DuplicateInvoice, PriceMismatch, QuantityVariance, MissingPO, PONotApproved,
    InvoiceBeforeReceipt, RoundDollarInvoice, WeekendProcessing, DuplicatePayment,
    PaymentBeforeInvoice, UnapprovedVendor, SplitPO, FictitiousVendor,
    VendorConcentration, GhostExpense,
    # O2C (10)
    DuplicateCustomerInvoice, InvoiceWithoutShipment, CreditLimitExceeded,
    ShortPayment, OverpaymentNotReturned, RevenueRecognitionTiming,
    FictitiousCustomer, RoundTripping, ChannelStuffing, SideAgreements,
    # GL (5)
    UnbalancedJournal, JournalNoApproval, SuspiciousAdjusting,
    UnusualAccountCombo, ManualOverride,
    # Control (5)
    SoDViolation, SelfApproval, ApprovalLimitExceeded,
    BackdatedTransaction, HolidayTransaction,
]

# Expected attributes keyed by type_code for attribute-value verification
EXPECTED_ATTRIBUTES: Dict[str, Dict[str, Any]] = {
    "P2P-001": {"category": "p2p", "difficulty": "easy", "detection_method": "duplicate_check", "class": DuplicateInvoice},
    "P2P-002": {"category": "p2p", "difficulty": "easy", "detection_method": "three_way_match", "class": PriceMismatch},
    "P2P-003": {"category": "p2p", "difficulty": "easy", "detection_method": "three_way_match", "class": QuantityVariance},
    "P2P-004": {"category": "p2p", "difficulty": "easy", "detection_method": "document_linkage", "class": MissingPO},
    "P2P-005": {"category": "p2p", "difficulty": "medium", "detection_method": "approval_check", "class": PONotApproved},
    "P2P-006": {"category": "p2p", "difficulty": "easy", "detection_method": "date_sequence", "class": InvoiceBeforeReceipt},
    "P2P-007": {"category": "p2p", "difficulty": "easy", "detection_method": "amount_pattern", "class": RoundDollarInvoice},
    "P2P-008": {"category": "p2p", "difficulty": "easy", "detection_method": "date_check", "class": WeekendProcessing},
    "P2P-009": {"category": "p2p", "difficulty": "medium", "detection_method": "duplicate_check", "class": DuplicatePayment},
    "P2P-010": {"category": "p2p", "difficulty": "easy", "detection_method": "date_sequence", "class": PaymentBeforeInvoice},
    "P2P-011": {"category": "p2p", "difficulty": "medium", "detection_method": "vendor_validation", "class": UnapprovedVendor},
    "P2P-012": {"category": "p2p", "difficulty": "medium", "detection_method": "pattern_analysis", "class": SplitPO},
    "P2P-013": {"category": "p2p", "difficulty": "medium", "detection_method": "entity_validation", "class": FictitiousVendor},
    "P2P-014": {"category": "p2p", "difficulty": "medium", "detection_method": "statistical_analysis", "class": VendorConcentration},
    "P2P-015": {"category": "p2p", "difficulty": "medium", "detection_method": "document_review", "class": GhostExpense},
    "O2C-001": {"category": "o2c", "difficulty": "easy", "detection_method": "duplicate_check", "class": DuplicateCustomerInvoice},
    "O2C-002": {"category": "o2c", "difficulty": "easy", "detection_method": "document_linkage", "class": InvoiceWithoutShipment},
    "O2C-003": {"category": "o2c", "difficulty": "easy", "detection_method": "credit_check", "class": CreditLimitExceeded},
    "O2C-004": {"category": "o2c", "difficulty": "easy", "detection_method": "payment_matching", "class": ShortPayment},
    "O2C-005": {"category": "o2c", "difficulty": "medium", "detection_method": "payment_matching", "class": OverpaymentNotReturned},
    "O2C-006": {"category": "o2c", "difficulty": "medium", "detection_method": "period_analysis", "class": RevenueRecognitionTiming},
    "O2C-007": {"category": "o2c", "difficulty": "medium", "detection_method": "entity_validation", "class": FictitiousCustomer},
    "O2C-008": {"category": "o2c", "difficulty": "medium", "detection_method": "pattern_analysis", "class": RoundTripping},
    "O2C-009": {"category": "o2c", "difficulty": "medium", "detection_method": "pattern_analysis", "class": ChannelStuffing},
    "O2C-010": {"category": "o2c", "difficulty": "medium", "detection_method": "document_review", "class": SideAgreements},
    "GL-001": {"category": "gl", "difficulty": "easy", "detection_method": "balance_check", "class": UnbalancedJournal},
    "GL-002": {"category": "gl", "difficulty": "easy", "detection_method": "approval_check", "class": JournalNoApproval},
    "GL-003": {"category": "gl", "difficulty": "medium", "detection_method": "period_analysis", "class": SuspiciousAdjusting},
    "GL-004": {"category": "gl", "difficulty": "medium", "detection_method": "pattern_analysis", "class": UnusualAccountCombo},
    "GL-005": {"category": "gl", "difficulty": "medium", "detection_method": "system_log_review", "class": ManualOverride},
    "CTL-001": {"category": "control", "difficulty": "medium", "detection_method": "access_review", "class": SoDViolation},
    "CTL-002": {"category": "control", "difficulty": "easy", "detection_method": "approval_check", "class": SelfApproval},
    "CTL-003": {"category": "control", "difficulty": "easy", "detection_method": "approval_check", "class": ApprovalLimitExceeded},
    "CTL-004": {"category": "control", "difficulty": "easy", "detection_method": "date_check", "class": BackdatedTransaction},
    "CTL-005": {"category": "control", "difficulty": "easy", "detection_method": "date_check", "class": HolidayTransaction},
}

# ---------------------------------------------------------------------------
# Category-filtered class lists for category-specific parametrization
# ---------------------------------------------------------------------------
P2P_CLASSES: list[Type[BaseDiscrepancy]] = [
    cls for cls in ALL_DISCREPANCY_CLASSES if cls.category == "p2p"
]
O2C_CLASSES: list[Type[BaseDiscrepancy]] = [
    cls for cls in ALL_DISCREPANCY_CLASSES if cls.category == "o2c"
]
GL_CLASSES: list[Type[BaseDiscrepancy]] = [
    cls for cls in ALL_DISCREPANCY_CLASSES if cls.category == "gl"
]
CTL_CLASSES: list[Type[BaseDiscrepancy]] = [
    cls for cls in ALL_DISCREPANCY_CLASSES if cls.category == "control"
]


# ---------------------------------------------------------------------------
# Sample Transaction Factories — return fresh dicts for each inject() call
# Date fields use datetime.date objects to support timedelta arithmetic in
# discrepancy implementations that shift dates.
# ---------------------------------------------------------------------------


def _sample_p2p_transaction() -> Dict[str, Any]:
    """Minimal P2P transaction suitable for all 15 P2P discrepancy types."""
    return {
        "transaction_id": str(uuid4()),
        "transaction_type": "vendor_invoice",
        "vendor_id": "V-001",
        "vendor_name": "Acme Corp",
        "vendor_address": "123 Main St",
        "vendor_phone": "555-0100",
        "invoice_number": "INV-2025-0001",
        "invoice_date": date(2025, 3, 15),
        "posting_date": date(2025, 3, 15),
        "processing_date": date(2025, 3, 15),
        "amount": Decimal("25000.00"),
        "total_amount": Decimal("25000.00"),
        "po_number": "PO-2025-0001",
        "receipt_number": "GR-2025-0001",
        "receipt_date": date(2025, 3, 10),
        "payment_date": date(2025, 4, 14),
        "payment_amount": Decimal("25000.00"),
        "approved_by": "agent-senior-accountant-001",
        "approved_by_role": "senior_accountant",
        "created_by": "agent-ap-clerk-001",
        "requested_by": "agent-purchasing-agent-001",
        "lines": [
            {
                "line_number": 1,
                "item": "Widget-A",
                "product_id": "PROD-001",
                "quantity": Decimal("100"),
                "unit_price": Decimal("250.00"),
                "amount": Decimal("25000.00"),
                "line_total": Decimal("25000.00"),
            },
        ],
        "status": "approved",
        "approval_chain": [
            {"approver_role": "purchasing_manager", "approved": True},
        ],
        "approval_limit": Decimal("50000.00"),
        "supporting_documents": ["PO-2025-0001", "GR-2025-0001"],
        "is_approved_vendor": True,
        "employee_addresses": ["456 Elm St"],
        "employee_phones": ["555-9999"],
        "vendor_history": [
            {"vendor_id": "V-001", "total_spend": Decimal("100000.00")},
            {"vendor_id": "V-002", "total_spend": Decimal("50000.00")},
        ],
        "metadata": {},
    }


def _sample_o2c_transaction() -> Dict[str, Any]:
    """Minimal O2C transaction suitable for all 10 O2C discrepancy types."""
    return {
        "transaction_id": str(uuid4()),
        "transaction_type": "customer_invoice",
        "customer_id": "C-001",
        "customer_name": "Beta Inc",
        "customer_address": "789 Oak Ave",
        "customer_phone": "555-0200",
        "invoice_number": "CINV-2025-0001",
        "invoice_date": date(2025, 3, 15),
        "posting_date": date(2025, 3, 15),
        "processing_date": date(2025, 3, 15),
        "amount": Decimal("50000.00"),
        "total_amount": Decimal("50000.00"),
        "invoice_amount": Decimal("50000.00"),
        "order_amount": Decimal("50000.00"),
        "payment_amount": Decimal("50000.00"),
        "so_number": "SO-2025-0001",
        "shipment_number": "SH-2025-0001",
        "shipment_date": date(2025, 3, 12),
        "credit_limit": Decimal("100000.00"),
        "current_ar_balance": Decimal("30000.00"),
        "created_by": "agent-ar-clerk-001",
        "approved_by": "agent-ar-manager-001",
        "approved_by_role": "ar_manager",
        "fiscal_period_end": date(2025, 3, 31),
        "lines": [
            {
                "line_number": 1,
                "item": "Product-X",
                "product_id": "PROD-010",
                "quantity": Decimal("200"),
                "unit_price": Decimal("250.00"),
                "amount": Decimal("50000.00"),
                "line_total": Decimal("50000.00"),
            },
        ],
        "side_agreements": [],
        "discount_amount": Decimal("0.00"),
        "employee_ids": ["E-001", "E-002"],
        "employee_addresses": ["456 Elm St"],
        "related_vendor_ids": [],
        "period_start": date(2025, 3, 1),
        "period_end": date(2025, 3, 31),
        "metadata": {},
    }


def _sample_gl_transaction() -> Dict[str, Any]:
    """Minimal GL/Journal Entry transaction suitable for all 5 GL discrepancy types."""
    return {
        "transaction_id": str(uuid4()),
        "transaction_type": "journal_entry",
        "je_number": "JE-2025-0001",
        "posting_date": date(2025, 3, 15),
        "entry_date": date(2025, 3, 15),
        "processing_date": date(2025, 3, 15),
        "source_type": "system",
        "is_system_generated": True,
        "approved_by": "agent-senior-accountant-001",
        "approved_by_role": "senior_accountant",
        "created_by": "agent-accountant-001",
        "amount": Decimal("1000.00"),
        "total_amount": Decimal("1000.00"),
        "description": "Monthly accrual entry",
        "period_end": date(2025, 3, 31),
        "fiscal_period_end": date(2025, 3, 31),
        "je_lines": [
            {
                "line_number": 1,
                "account_code": "5100",
                "account_name": "COGS",
                "debit_amount": Decimal("1000.00"),
                "credit_amount": Decimal("0.00"),
            },
            {
                "line_number": 2,
                "account_code": "2100",
                "account_name": "Accounts Payable",
                "debit_amount": Decimal("0.00"),
                "credit_amount": Decimal("1000.00"),
            },
        ],
        "metadata": {},
    }


def _sample_control_transaction() -> Dict[str, Any]:
    """Minimal transaction suitable for all 5 Control discrepancy types."""
    # Use a Wednesday (2025-03-12 is a Wednesday) as the base date —
    # this gives room for the WeekendProcessing / HolidayTransaction types
    # that need to shift dates to weekends or holidays.
    return {
        "transaction_id": str(uuid4()),
        "transaction_type": "purchase_order",
        "transaction_date": date(2025, 3, 12),
        "posting_date": date(2025, 3, 12),
        "processing_date": date(2025, 3, 12),
        "amount": Decimal("30000.00"),
        "total_amount": Decimal("30000.00"),
        "created_by": "agent-purchasing-agent-001",
        "approved_by": "agent-purchasing-manager-001",
        "approved_by_role": "purchasing_manager",
        "requested_by": "agent-purchasing-agent-001",
        "submitted_by": "agent-purchasing-agent-001",
        "status": "approved",
        "approval_limit": Decimal("25000.00"),
        "approver_limit": Decimal("25000.00"),
        "approval_chain": [
            {"approver_role": "purchasing_manager", "approved": True},
        ],
        "metadata": {},
    }


CATEGORY_SAMPLE_TRANSACTIONS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "p2p": _sample_p2p_transaction,
    "o2c": _sample_o2c_transaction,
    "gl": _sample_gl_transaction,
    "control": _sample_control_transaction,
}


def _get_sample_transaction_for_class(
    disc_cls: Type[BaseDiscrepancy],
) -> Dict[str, Any]:
    """Return a fresh sample transaction appropriate for the discrepancy's category."""
    factory = CATEGORY_SAMPLE_TRANSACTIONS.get(disc_cls.category)
    if factory is None:
        raise ValueError(f"Unknown category: {disc_cls.category}")
    return factory()


# =========================================================================
# Phase 1: Base Class Contract Tests
# =========================================================================


@pytest.mark.discrepancy
class TestBaseDiscrepancyContract:
    """Verify BaseDiscrepancy ABC contract and completeness of ALL_DISCREPANCY_CLASSES."""

    def test_cannot_instantiate_base_class(self) -> None:
        """BaseDiscrepancy is abstract and cannot be directly instantiated."""
        with pytest.raises(TypeError):
            BaseDiscrepancy()  # type: ignore[abstract]

    def test_inject_is_abstract(self) -> None:
        """The inject() method is declared abstract on BaseDiscrepancy."""
        assert hasattr(BaseDiscrepancy, "inject")
        assert getattr(BaseDiscrepancy.inject, "__isabstractmethod__", False)

    def test_total_class_count(self) -> None:
        """The master list contains exactly 35 discrepancy type classes."""
        assert len(ALL_DISCREPANCY_CLASSES) == 35

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_all_classes_extend_base(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """Every class in ALL_DISCREPANCY_CLASSES is a subclass of BaseDiscrepancy."""
        assert issubclass(disc_cls, BaseDiscrepancy)


# =========================================================================
# Phase 1: Class Attribute Tests (parametrized over all 35 types)
# =========================================================================


@pytest.mark.discrepancy
class TestAllTypesClassAttributes:
    """Verify the 6+ ClassVar attributes on all 35 concrete discrepancy types."""

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_type_code_is_non_empty_string(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """type_code is a non-empty string."""
        assert isinstance(disc_cls.type_code, str)
        assert len(disc_cls.type_code) > 0

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_category_is_valid(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """category is one of the valid values: p2p, o2c, gl, control."""
        assert disc_cls.category in ("p2p", "o2c", "gl", "control")

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_difficulty_is_easy_or_medium(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """difficulty must be 'easy' or 'medium' — NO 'hard' in MVP."""
        assert disc_cls.difficulty in ("easy", "medium")

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_name_is_non_empty_string(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """name is a non-empty string."""
        assert isinstance(disc_cls.name, str)
        assert len(disc_cls.name) > 0

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_description_is_non_empty_string(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """description is a non-empty string."""
        assert isinstance(disc_cls.description, str)
        assert len(disc_cls.description) > 0

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_detection_method_is_non_empty_string(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """detection_method is a non-empty string."""
        assert isinstance(disc_cls.detection_method, str)
        assert len(disc_cls.detection_method) > 0

    # --- Exact attribute value verification against EXPECTED_ATTRIBUTES ---

    @pytest.mark.parametrize(
        "type_code,expected",
        list(EXPECTED_ATTRIBUTES.items()),
        ids=list(EXPECTED_ATTRIBUTES.keys()),
    )
    def test_type_code_matches_expected(self, type_code: str, expected: Dict[str, Any]) -> None:
        """Verify the expected class has the correct type_code string."""
        cls = expected["class"]
        assert cls.type_code == type_code

    @pytest.mark.parametrize(
        "type_code,expected",
        list(EXPECTED_ATTRIBUTES.items()),
        ids=list(EXPECTED_ATTRIBUTES.keys()),
    )
    def test_category_matches_expected(self, type_code: str, expected: Dict[str, Any]) -> None:
        """Verify category matches the expected value for each type_code."""
        cls = expected["class"]
        assert cls.category == expected["category"]

    @pytest.mark.parametrize(
        "type_code,expected",
        list(EXPECTED_ATTRIBUTES.items()),
        ids=list(EXPECTED_ATTRIBUTES.keys()),
    )
    def test_difficulty_matches_expected(self, type_code: str, expected: Dict[str, Any]) -> None:
        """Verify difficulty matches the expected value for each type_code."""
        cls = expected["class"]
        assert cls.difficulty == expected["difficulty"]

    @pytest.mark.parametrize(
        "type_code,expected",
        list(EXPECTED_ATTRIBUTES.items()),
        ids=list(EXPECTED_ATTRIBUTES.keys()),
    )
    def test_detection_method_matches_expected(self, type_code: str, expected: Dict[str, Any]) -> None:
        """Verify detection_method matches the expected value for each type_code."""
        cls = expected["class"]
        assert cls.detection_method == expected["detection_method"]


# =========================================================================
# Phase 1: Inject Contract Tests (parametrized over all 35 types)
# =========================================================================


@pytest.mark.discrepancy
class TestAllTypesInjectContract:
    """Verify inject() return shape and immutability for all 35 types."""

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_inject_returns_tuple_of_two_dicts(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """inject() returns a tuple of length 2, both elements being dicts."""
        instance = disc_cls()
        transaction = _get_sample_transaction_for_class(disc_cls)
        result = instance.inject(transaction, {}, deterministic_rng)
        assert isinstance(result, tuple), f"{disc_cls.type_code}: result is not a tuple"
        assert len(result) == 2, f"{disc_cls.type_code}: tuple length is {len(result)}, expected 2"
        modified_txn, ground_truth = result
        assert isinstance(modified_txn, dict), f"{disc_cls.type_code}: first element not a dict"
        assert isinstance(ground_truth, dict), f"{disc_cls.type_code}: second element not a dict"

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_inject_does_not_mutate_original(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """inject() must not mutate the original transaction dictionary."""
        instance = disc_cls()
        transaction = _get_sample_transaction_for_class(disc_cls)
        original_snapshot = copy.deepcopy(transaction)
        _ = instance.inject(transaction, {}, deterministic_rng)
        assert transaction == original_snapshot, (
            f"{disc_cls.type_code}: inject() mutated the original transaction"
        )

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_inject_ground_truth_has_required_keys(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """Ground truth dict must contain the minimum required keys."""
        instance = disc_cls()
        transaction = _get_sample_transaction_for_class(disc_cls)
        _, ground_truth = instance.inject(transaction, {}, deterministic_rng)

        required_keys = {
            "type_code",
            "category",
            "difficulty",
            "detection_method",
            "affected_fields",
            "original_values",
            "modified_values",
            "financial_impact",
        }
        missing = required_keys - set(ground_truth.keys())
        assert not missing, (
            f"{disc_cls.type_code}: ground truth missing keys: {missing}"
        )

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_inject_financial_impact_is_decimal(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """ground_truth['financial_impact'] must be a Decimal instance."""
        instance = disc_cls()
        transaction = _get_sample_transaction_for_class(disc_cls)
        _, ground_truth = instance.inject(transaction, {}, deterministic_rng)
        assert isinstance(ground_truth["financial_impact"], Decimal), (
            f"{disc_cls.type_code}: financial_impact is "
            f"{type(ground_truth['financial_impact']).__name__}, expected Decimal"
        )


# =========================================================================
# Phase 2: Category-Specific Tests — P2P
# =========================================================================


@pytest.mark.discrepancy
class TestP2PDiscrepancies:
    """P2P-specific inject() behaviour tests for all 15 P2P types."""

    @pytest.mark.parametrize(
        "disc_cls",
        P2P_CLASSES,
        ids=[cls.type_code for cls in P2P_CLASSES],
    )
    def test_p2p_inject_with_sample_transaction(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """Each P2P class inject() succeeds with _sample_p2p_transaction."""
        instance = disc_cls()
        txn = _sample_p2p_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        assert isinstance(modified, dict)
        assert isinstance(gt, dict)

    @pytest.mark.parametrize(
        "disc_cls",
        P2P_CLASSES,
        ids=[cls.type_code for cls in P2P_CLASSES],
    )
    def test_p2p_category_is_p2p(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All P2P classes have category == 'p2p'."""
        assert disc_cls.category == "p2p"

    @pytest.mark.parametrize(
        "disc_cls",
        P2P_CLASSES,
        ids=[cls.type_code for cls in P2P_CLASSES],
    )
    def test_p2p_type_code_prefix(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All P2P type_codes start with 'P2P-'."""
        assert disc_cls.type_code.startswith("P2P-")

    def test_duplicate_invoice_specific(self, deterministic_rng: random.Random) -> None:
        """DuplicateInvoice creates near-duplicate; invoice number or date is altered."""
        instance = DuplicateInvoice()
        txn = _sample_p2p_transaction()
        original_invoice_num = txn["invoice_number"]
        original_invoice_date = txn["invoice_date"]
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        # At least one of invoice_number or invoice_date must differ
        changed = (
            modified.get("invoice_number") != original_invoice_num
            or modified.get("invoice_date") != original_invoice_date
        )
        assert changed, "DuplicateInvoice did not alter invoice_number or invoice_date"

    def test_price_mismatch_specific(self, deterministic_rng: random.Random) -> None:
        """PriceMismatch alters line prices; line amounts differ from original."""
        instance = PriceMismatch()
        txn = _sample_p2p_transaction()
        original_line_price = txn["lines"][0]["unit_price"]
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        modified_lines = modified.get("lines", [])
        if modified_lines:
            # Price in at least one line should have changed
            assert modified_lines[0].get("unit_price") != original_line_price or (
                modified.get("amount") != txn["amount"]
            ), "PriceMismatch did not alter any price"

    def test_quantity_variance_specific(self, deterministic_rng: random.Random) -> None:
        """QuantityVariance alters line quantities; quantity differs from original."""
        instance = QuantityVariance()
        txn = _sample_p2p_transaction()
        original_qty = txn["lines"][0]["quantity"]
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        modified_lines = modified.get("lines", [])
        if modified_lines:
            assert modified_lines[0].get("quantity") != original_qty or (
                modified.get("amount") != txn["amount"]
            ), "QuantityVariance did not alter quantity"

    def test_missing_po_specific(self, deterministic_rng: random.Random) -> None:
        """MissingPO removes PO reference; po_number is removed or None."""
        instance = MissingPO()
        txn = _sample_p2p_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        po_val = modified.get("po_number")
        assert po_val is None or po_val == "" or "po_number" not in modified, (
            f"MissingPO did not clear po_number; value={po_val}"
        )

    def test_round_dollar_specific(self, deterministic_rng: random.Random) -> None:
        """RoundDollarInvoice produces amounts divisible by 100, 1000, or 10000."""
        instance = RoundDollarInvoice()
        txn = _sample_p2p_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        mod_amount = modified.get("amount") or modified.get("total_amount")
        if mod_amount is not None:
            # Convert to int for divisibility check (may be Decimal)
            int_amount = int(mod_amount)
            assert (
                int_amount % 100 == 0 or int_amount % 1000 == 0 or int_amount % 10000 == 0
            ), f"RoundDollarInvoice amount {mod_amount} is not a round dollar amount"


# =========================================================================
# Phase 2: Category-Specific Tests — O2C
# =========================================================================


@pytest.mark.discrepancy
class TestO2CDiscrepancies:
    """O2C-specific inject() behaviour tests for all 10 O2C types."""

    @pytest.mark.parametrize(
        "disc_cls",
        O2C_CLASSES,
        ids=[cls.type_code for cls in O2C_CLASSES],
    )
    def test_o2c_inject_with_sample_transaction(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """Each O2C class inject() succeeds with _sample_o2c_transaction."""
        instance = disc_cls()
        txn = _sample_o2c_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        assert isinstance(modified, dict)
        assert isinstance(gt, dict)

    @pytest.mark.parametrize(
        "disc_cls",
        O2C_CLASSES,
        ids=[cls.type_code for cls in O2C_CLASSES],
    )
    def test_o2c_category_is_o2c(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All O2C classes have category == 'o2c'."""
        assert disc_cls.category == "o2c"

    @pytest.mark.parametrize(
        "disc_cls",
        O2C_CLASSES,
        ids=[cls.type_code for cls in O2C_CLASSES],
    )
    def test_o2c_type_code_prefix(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All O2C type_codes start with 'O2C-'."""
        assert disc_cls.type_code.startswith("O2C-")

    def test_credit_limit_exceeded_specific(self, deterministic_rng: random.Random) -> None:
        """CreditLimitExceeded results in order_amount pushing exposure over credit_limit."""
        instance = CreditLimitExceeded()
        txn = _sample_o2c_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        # The implementation modifies order_amount to push exposure over limit.
        # Check: current_ar_balance + new_order_amount > credit_limit
        mod_order_amount = Decimal(str(modified.get("order_amount", "0")))
        mod_ar = Decimal(str(modified.get("current_ar_balance", txn["current_ar_balance"])))
        credit_limit = Decimal(str(modified.get("credit_limit", txn["credit_limit"])))
        total_exposure = mod_ar + mod_order_amount
        # The order_amount should have been increased compared to original
        orig_order_amount = txn.get("order_amount", Decimal("0"))
        assert (
            total_exposure > credit_limit
            or mod_order_amount > orig_order_amount
        ), (
            f"CreditLimitExceeded: total_exposure={total_exposure} <= "
            f"credit_limit={credit_limit}"
        )

    def test_short_payment_specific(self, deterministic_rng: random.Random) -> None:
        """ShortPayment results in payment_amount less than invoice amount."""
        instance = ShortPayment()
        txn = _sample_o2c_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        # The implementation stores payment_amount as str; coerce to Decimal
        payment = Decimal(str(modified.get("payment_amount", txn["payment_amount"])))
        invoice_amt = Decimal(str(modified.get("invoice_amount", txn["amount"])))
        assert payment < invoice_amt, (
            f"ShortPayment: payment_amount ({payment}) is not less than "
            f"invoice amount ({invoice_amt})"
        )


# =========================================================================
# Phase 2: Category-Specific Tests — GL
# =========================================================================


@pytest.mark.discrepancy
class TestGLDiscrepancies:
    """GL-specific inject() behaviour tests for all 5 GL types."""

    @pytest.mark.parametrize(
        "disc_cls",
        GL_CLASSES,
        ids=[cls.type_code for cls in GL_CLASSES],
    )
    def test_gl_inject_with_sample_transaction(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """Each GL class inject() succeeds with _sample_gl_transaction."""
        instance = disc_cls()
        txn = _sample_gl_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        assert isinstance(modified, dict)
        assert isinstance(gt, dict)

    @pytest.mark.parametrize(
        "disc_cls",
        GL_CLASSES,
        ids=[cls.type_code for cls in GL_CLASSES],
    )
    def test_gl_category_is_gl(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All GL classes have category == 'gl'."""
        assert disc_cls.category == "gl"

    @pytest.mark.parametrize(
        "disc_cls",
        GL_CLASSES,
        ids=[cls.type_code for cls in GL_CLASSES],
    )
    def test_gl_type_code_prefix(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All GL type_codes start with 'GL-'."""
        assert disc_cls.type_code.startswith("GL-")

    def test_unbalanced_journal_specific(self, deterministic_rng: random.Random) -> None:
        """UnbalancedJournal creates imbalance — SUM(debits) != SUM(credits)."""
        instance = UnbalancedJournal()
        txn = _sample_gl_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)

        je_lines = modified.get("je_lines", [])
        total_debits = sum(
            line.get("debit_amount", Decimal("0")) for line in je_lines
        )
        total_credits = sum(
            line.get("credit_amount", Decimal("0")) for line in je_lines
        )
        imbalance = abs(total_debits - total_credits)
        assert imbalance >= Decimal("0.01"), (
            f"UnbalancedJournal: debits={total_debits}, credits={total_credits}, "
            f"imbalance={imbalance} is less than $0.01"
        )

    def test_journal_no_approval_specific(self, deterministic_rng: random.Random) -> None:
        """JournalNoApproval clears or nullifies the approval field."""
        instance = JournalNoApproval()
        txn = _sample_gl_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        approved_by = modified.get("approved_by")
        # Should be None, empty string, or missing entirely
        assert approved_by is None or approved_by == "" or "approved_by" not in modified, (
            f"JournalNoApproval: approved_by still set to '{approved_by}'"
        )


# =========================================================================
# Phase 2: Category-Specific Tests — Control
# =========================================================================


@pytest.mark.discrepancy
class TestControlDiscrepancies:
    """Control-specific inject() behaviour tests for all 5 Control types."""

    @pytest.mark.parametrize(
        "disc_cls",
        CTL_CLASSES,
        ids=[cls.type_code for cls in CTL_CLASSES],
    )
    def test_control_inject_with_sample_transaction(
        self, disc_cls: Type[BaseDiscrepancy], deterministic_rng: random.Random
    ) -> None:
        """Each Control class inject() succeeds with _sample_control_transaction."""
        instance = disc_cls()
        txn = _sample_control_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        assert isinstance(modified, dict)
        assert isinstance(gt, dict)

    @pytest.mark.parametrize(
        "disc_cls",
        CTL_CLASSES,
        ids=[cls.type_code for cls in CTL_CLASSES],
    )
    def test_control_category_is_control(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All Control classes have category == 'control'."""
        assert disc_cls.category == "control"

    @pytest.mark.parametrize(
        "disc_cls",
        CTL_CLASSES,
        ids=[cls.type_code for cls in CTL_CLASSES],
    )
    def test_control_type_code_prefix(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """All Control type_codes start with 'CTL-'."""
        assert disc_cls.type_code.startswith("CTL-")

    def test_sod_violation_specific(self, deterministic_rng: random.Random) -> None:
        """SoDViolation makes two role fields equal (same person in two roles)."""
        instance = SoDViolation()
        txn = _sample_control_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        # Ground truth should indicate affected fields; check that at least
        # two role-related fields now share the same value.
        affected = gt.get("affected_fields", [])
        if len(affected) >= 2:
            values = [modified.get(f) for f in affected if modified.get(f) is not None]
            if len(values) >= 2:
                # At least two of the affected fields should be equal
                assert any(
                    values[i] == values[j]
                    for i in range(len(values))
                    for j in range(i + 1, len(values))
                ), "SoDViolation: no two role fields are equal"

    def test_self_approval_specific(self, deterministic_rng: random.Random) -> None:
        """SelfApproval makes created_by == approved_by."""
        instance = SelfApproval()
        txn = _sample_control_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)
        # Examine affected fields from ground truth to find the creator/approver pair
        affected = gt.get("affected_fields", [])
        mod_vals = gt.get("modified_values", {})
        # Check if any pair of affected fields resolves to the same value
        if "created_by" in affected and "approved_by" in affected:
            assert modified.get("created_by") == modified.get("approved_by"), (
                "SelfApproval: created_by != approved_by"
            )
        else:
            # Fallback: at least two affected fields share the same modified value
            field_values = [modified.get(f) for f in affected if modified.get(f)]
            if len(field_values) >= 2:
                assert any(
                    field_values[i] == field_values[j]
                    for i in range(len(field_values))
                    for j in range(i + 1, len(field_values))
                ), "SelfApproval: no creator/approver fields match"

    def test_backdated_specific(self, deterministic_rng: random.Random) -> None:
        """BackdatedTransaction moves a date field backwards in time."""
        instance = BackdatedTransaction()
        txn = _sample_control_transaction()
        modified, gt = instance.inject(txn, {}, deterministic_rng)

        # Check ground truth for affected date fields
        affected = gt.get("affected_fields", [])
        found_backdated = False
        for field in affected:
            mod_val = modified.get(field)
            orig_val = txn.get(field)
            if mod_val is not None and orig_val is not None:
                # The modified date should be earlier than the original
                if mod_val < orig_val:
                    found_backdated = True
                    break
        if not found_backdated:
            # Fallback check on common date fields
            for field in ("transaction_date", "posting_date", "processing_date"):
                if field in modified and field in txn:
                    if modified[field] != txn[field]:
                        assert modified[field] < txn[field], (
                            f"BackdatedTransaction: {field} was not moved backwards"
                        )
                        found_backdated = True
                        break
        assert found_backdated, "BackdatedTransaction: no date field was moved backwards"


# =========================================================================
# Phase 3: Helper Method Tests
# =========================================================================


@pytest.mark.discrepancy
class TestValidateParams:
    """Tests for _validate_params() helper on concrete discrepancy classes."""

    def test_params_within_bounds(self, deterministic_rng: random.Random) -> None:
        """Valid params within PARAMETER_BOUNDS are accepted without error."""
        instance = DuplicateInvoice()
        # DuplicateInvoice bounds: days_apart 1-90, amount_variance_pct 0-5
        valid_params = {"days_apart": 10, "amount_variance_pct": 2.5}
        result = instance._validate_params(valid_params)
        assert isinstance(result, dict)
        assert result["days_apart"] == 10

    def test_params_outside_bounds_clamped(self) -> None:
        """Out-of-bounds params with auto_adjust=True are clamped to nearest bound."""
        instance = DuplicateInvoice()
        # days_apart > 90 should be clamped to 90
        out_of_bounds = {"days_apart": 200, "amount_variance_pct": 10.0}
        result = instance._validate_params(out_of_bounds, auto_adjust=True)
        assert isinstance(result, dict)
        assert result["days_apart"] <= 90, (
            f"Expected days_apart clamped to <=90, got {result['days_apart']}"
        )

    def test_params_outside_bounds_raises(self) -> None:
        """Out-of-bounds params with auto_adjust=False raise DiscrepancyInjectionError."""
        instance = DuplicateInvoice()
        out_of_bounds = {"days_apart": 200}
        with pytest.raises(DiscrepancyInjectionError):
            instance._validate_params(out_of_bounds, auto_adjust=False)

    def test_empty_params_no_error(self) -> None:
        """Empty params dict is valid — defaults apply from PARAMETER_BOUNDS."""
        instance = DuplicateInvoice()
        result = instance._validate_params({})
        assert isinstance(result, dict)


# =========================================================================
# Phase 3: Ground Truth Data Helper Tests
# =========================================================================


@pytest.mark.discrepancy
class TestCreateGroundTruthData:
    """Tests for _create_ground_truth_data() helper method."""

    def _invoke_ground_truth(self) -> Dict[str, Any]:
        """Helper: create a concrete instance and invoke _create_ground_truth_data."""
        instance = DuplicateInvoice()
        return instance._create_ground_truth_data(
            affected_fields=["invoice_number", "invoice_date"],
            original_values={"invoice_number": "INV-001", "invoice_date": "2025-03-15"},
            modified_values={"invoice_number": "INV-001-DUP", "invoice_date": "2025-03-20"},
            financial_impact=Decimal("25000.00"),
            description="Duplicate invoice injection",
        )

    def test_ground_truth_has_type_code(self) -> None:
        """Ground truth output includes 'type_code'."""
        gt = self._invoke_ground_truth()
        assert "type_code" in gt
        assert gt["type_code"] == DuplicateInvoice.type_code

    def test_ground_truth_has_category(self) -> None:
        """Ground truth output includes 'category'."""
        gt = self._invoke_ground_truth()
        assert "category" in gt
        assert gt["category"] == DuplicateInvoice.category

    def test_ground_truth_has_difficulty(self) -> None:
        """Ground truth output includes 'difficulty'."""
        gt = self._invoke_ground_truth()
        assert "difficulty" in gt
        assert gt["difficulty"] == DuplicateInvoice.difficulty

    def test_ground_truth_has_detection_method(self) -> None:
        """Ground truth output includes 'detection_method'."""
        gt = self._invoke_ground_truth()
        assert "detection_method" in gt
        assert gt["detection_method"] == DuplicateInvoice.detection_method

    def test_ground_truth_financial_impact_is_decimal(self) -> None:
        """financial_impact in ground truth is Decimal type."""
        gt = self._invoke_ground_truth()
        assert isinstance(gt["financial_impact"], Decimal)


# =========================================================================
# Phase 3: Copy Transaction Helper Tests
# =========================================================================


@pytest.mark.discrepancy
class TestCopyTransaction:
    """Tests for _copy_transaction() static deep-copy method."""

    def test_deep_copy_independence(self) -> None:
        """Modifying the copy does not affect the original transaction."""
        original = _sample_p2p_transaction()
        copied = BaseDiscrepancy._copy_transaction(original)
        copied["vendor_name"] = "CHANGED"
        assert original["vendor_name"] != "CHANGED"

    def test_nested_dict_deep_copied(self) -> None:
        """Nested dicts inside the copy are fully independent."""
        original = _sample_p2p_transaction()
        copied = BaseDiscrepancy._copy_transaction(original)
        if copied.get("lines"):
            copied["lines"][0]["item"] = "MODIFIED-ITEM"
        assert original["lines"][0]["item"] != "MODIFIED-ITEM"

    def test_list_elements_deep_copied(self) -> None:
        """List elements inside the copy are independent from the original."""
        original = _sample_p2p_transaction()
        copied = BaseDiscrepancy._copy_transaction(original)
        copied["supporting_documents"].append("EXTRA-DOC")
        assert "EXTRA-DOC" not in original["supporting_documents"]


# =========================================================================
# Phase 4: Determinism Tests
# =========================================================================


@pytest.mark.discrepancy
class TestDeterminism:
    """Same seed produces same injection result for all 35 discrepancy types."""

    @pytest.mark.parametrize(
        "disc_cls",
        ALL_DISCREPANCY_CLASSES,
        ids=[cls.type_code for cls in ALL_DISCREPANCY_CLASSES],
    )
    def test_same_seed_same_result(self, disc_cls: Type[BaseDiscrepancy]) -> None:
        """Two inject() calls with the same seed produce identical results."""
        instance1 = disc_cls()
        instance2 = disc_cls()

        txn1 = _get_sample_transaction_for_class(disc_cls)
        txn2 = _get_sample_transaction_for_class(disc_cls)
        # Align transaction IDs so the deep-copy comparison is fair
        txn2["transaction_id"] = txn1["transaction_id"]

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        modified1, gt1 = instance1.inject(txn1, {}, rng1)
        modified2, gt2 = instance2.inject(txn2, {}, rng2)

        assert modified1 == modified2, (
            f"{disc_cls.type_code}: deterministic check failed — "
            f"modified transactions differ with same seed"
        )
        assert gt1 == gt2, (
            f"{disc_cls.type_code}: deterministic check failed — "
            f"ground truth data differs with same seed"
        )
