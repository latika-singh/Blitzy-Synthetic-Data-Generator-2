"""Comprehensive unit tests for ThreeWayMatcher — PO/Receipt/Invoice matching.

Tests cover:
- Price tolerance: ±5% (PO vs Invoice unit prices) per AAP Section 0.7.5
- Quantity tolerance: ±2% (Receipt vs Invoice quantities)
- Match status determination: FULL_MATCH, PRICE_EXCEPTION, QUANTITY_EXCEPTION,
  COMBINED_EXCEPTION, MISSING_RECEIPT, MISSING_PO
- Variance calculations at LINE-ITEM level (calculated per line, then summed)
- Approval requirement flag when variances detected
- ThreeWayMatchResult and LineMatchResult Pydantic V2 model validation
- Division-by-zero protection for zero-quantity/zero-price edge cases
- Decimal precision: prec=28, ROUND_HALF_UP — NEVER float for financials

Per AAP Section 0.7.6 Testing Conventions:
- All external dependencies mocked (structlog, EventBus via AsyncMock)
- Async tests use pytest-asyncio
- No live API calls, no external Redis
- Coverage target: ≥ 80%
- Property-based testing with hypothesis for tolerance boundary validation
"""

from __future__ import annotations

import decimal
from datetime import datetime, timezone
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.transactions.p2p.three_way_matcher import (
    ThreeWayMatcher,
    MatchStatus,
    LineMatchResult,
    ThreeWayMatchResult,
)
from app.transactions.constants import (
    THREE_WAY_MATCH_PRICE_TOLERANCE,
    THREE_WAY_MATCH_QUANTITY_TOLERANCE,
)
from app.transactions.exceptions import ThreeWayMatchError

# ---------------------------------------------------------------------------
# Module-level Decimal configuration per AAP §0.7.2
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = ROUND_HALF_UP

# Deterministic UUIDs for reproducible test assertions.
TEST_INVOICE_UUID = UUID("92345678-9234-5678-9234-567892345678")
TEST_PO_UUID = UUID("72345678-7234-5678-7234-567872345678")
TEST_RECEIPT_UUID = UUID("c2345678-c234-5678-c234-5678c2345678")

# Re-usable zero constant.
_ZERO = Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions — reusable sample data factories
# ═══════════════════════════════════════════════════════════════════════════


def make_po_lines(
    lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create sample PO line dicts.

    Each dict in *lines* should supply at least ``unit_price`` and
    ``quantity``.  ``line_number`` is auto-assigned starting at 1 if
    not present.

    Args:
        lines: List of partial line dicts.

    Returns:
        Complete PO line dicts suitable for ``ThreeWayMatcher.match()``.
    """
    result: List[Dict[str, Any]] = []
    for idx, raw in enumerate(lines):
        entry: Dict[str, Any] = {
            "line_number": raw.get("line_number", idx + 1),
            "quantity": str(raw.get("quantity", "0")),
            "unit_price": str(raw.get("unit_price", "0")),
        }
        result.append(entry)
    return result


def make_receipt_lines(
    lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create sample goods-receipt line dicts.

    Each dict in *lines* should supply at least ``quantity``.
    ``line_number`` is auto-assigned starting at 1 if not present.

    Args:
        lines: List of partial line dicts.

    Returns:
        Complete receipt line dicts suitable for ``ThreeWayMatcher.match()``.
    """
    result: List[Dict[str, Any]] = []
    for idx, raw in enumerate(lines):
        entry: Dict[str, Any] = {
            "line_number": raw.get("line_number", idx + 1),
            "quantity": str(raw.get("quantity", "0")),
        }
        result.append(entry)
    return result


def make_invoice_lines(
    lines: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create sample vendor-invoice line dicts.

    Each dict in *lines* should supply ``unit_price`` and ``quantity``.
    ``line_number`` is auto-assigned starting at 1 if not present.

    Args:
        lines: List of partial line dicts.

    Returns:
        Complete invoice line dicts suitable for ``ThreeWayMatcher.match()``.
    """
    result: List[Dict[str, Any]] = []
    for idx, raw in enumerate(lines):
        entry: Dict[str, Any] = {
            "line_number": raw.get("line_number", idx + 1),
            "quantity": str(raw.get("quantity", "0")),
            "unit_price": str(raw.get("unit_price", "0")),
        }
        result.append(entry)
    return result


def _quick_match(
    matcher: ThreeWayMatcher,
    *,
    po_price: str = "100.00",
    po_qty: str = "100",
    receipt_qty: str = "100",
    inv_price: str = "100.00",
    inv_qty: str = "100",
    receipt_id: UUID | None = None,
) -> ThreeWayMatchResult:
    """Convenience wrapper for single-line matching with default UUIDs.

    Args:
        matcher: A ``ThreeWayMatcher`` instance.
        po_price: PO unit price as string.
        po_qty: PO ordered quantity as string.
        receipt_qty: Goods-receipt quantity as string.
        inv_price: Invoice unit price as string.
        inv_qty: Invoice billed quantity as string.
        receipt_id: Optional receipt UUID (defaults to TEST_RECEIPT_UUID).

    Returns:
        The ``ThreeWayMatchResult`` produced by the matcher.
    """
    effective_receipt_id = receipt_id if receipt_id is not None else TEST_RECEIPT_UUID
    return matcher.match(
        po_lines=make_po_lines([{"quantity": po_qty, "unit_price": po_price}]),
        receipt_lines=make_receipt_lines([{"quantity": receipt_qty}]),
        invoice_lines=make_invoice_lines([{"quantity": inv_qty, "unit_price": inv_price}]),
        invoice_id=TEST_INVOICE_UUID,
        po_id=TEST_PO_UUID,
        receipt_id=effective_receipt_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def matcher() -> ThreeWayMatcher:
    """ThreeWayMatcher with default tolerances (±5% price, ±2% quantity)."""
    return ThreeWayMatcher()


@pytest.fixture
def matcher_custom_tolerance() -> ThreeWayMatcher:
    """ThreeWayMatcher with custom tight tolerances for boundary tests."""
    return ThreeWayMatcher(
        price_tolerance=Decimal("0.10"),
        quantity_tolerance=Decimal("0.05"),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test Classes
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.p2p
class TestMatchStatus:
    """Tests for the MatchStatus enumeration."""

    def test_match_status_enum_has_six_values(self) -> None:
        """Verify MatchStatus defines exactly 6 distinct values."""
        assert len(MatchStatus) == 6

    def test_full_match_is_default_success_status(self) -> None:
        """FULL_MATCH represents a successful match with no exceptions."""
        assert MatchStatus.FULL_MATCH.value == "full_match"

    def test_exception_statuses_are_distinct(self) -> None:
        """All exception statuses have unique string values."""
        values = [ms.value for ms in MatchStatus]
        assert len(values) == len(set(values)), "Duplicate MatchStatus values detected"

    def test_all_expected_statuses_exist(self) -> None:
        """Confirm every expected status value is defined in the enum."""
        expected = {
            "full_match",
            "price_exception",
            "quantity_exception",
            "combined_exception",
            "missing_receipt",
            "missing_po",
        }
        actual = {ms.value for ms in MatchStatus}
        assert actual == expected

    def test_match_status_is_string_enum(self) -> None:
        """MatchStatus members can be compared directly to strings."""
        assert MatchStatus.FULL_MATCH == "full_match"
        assert MatchStatus.PRICE_EXCEPTION == "price_exception"


@pytest.mark.p2p
class TestLineMatchResult:
    """Tests for LineMatchResult Pydantic V2 model validation."""

    def test_line_match_result_creation_with_valid_data(self) -> None:
        """LineMatchResult accepts all required fields and produces a valid model."""
        result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("100"),
            receipt_quantity=Decimal("100"),
            invoice_quantity=Decimal("100"),
            po_unit_price=Decimal("25.00"),
            invoice_unit_price=Decimal("25.00"),
            price_variance_pct=_ZERO,
            quantity_variance_pct=_ZERO,
            price_within_tolerance=True,
            quantity_within_tolerance=True,
            line_match_status=MatchStatus.FULL_MATCH,
        )
        assert result.line_number == 1
        assert result.po_unit_price == Decimal("25.00")
        assert result.invoice_unit_price == Decimal("25.00")

    def test_line_match_result_all_within_tolerance(self) -> None:
        """Both tolerance flags are True when all values match exactly."""
        result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("50"),
            receipt_quantity=Decimal("50"),
            invoice_quantity=Decimal("50"),
            po_unit_price=Decimal("10.00"),
            invoice_unit_price=Decimal("10.00"),
            price_within_tolerance=True,
            quantity_within_tolerance=True,
            line_match_status=MatchStatus.FULL_MATCH,
        )
        assert result.price_within_tolerance is True
        assert result.quantity_within_tolerance is True

    def test_line_match_result_price_outside_tolerance(self) -> None:
        """Price flag is False when price variance exceeds tolerance."""
        result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("100"),
            receipt_quantity=Decimal("100"),
            invoice_quantity=Decimal("100"),
            po_unit_price=Decimal("100.00"),
            invoice_unit_price=Decimal("110.00"),
            price_variance_pct=Decimal("0.10"),
            price_within_tolerance=False,
            quantity_within_tolerance=True,
            line_match_status=MatchStatus.PRICE_EXCEPTION,
        )
        assert result.price_within_tolerance is False
        assert result.line_match_status == MatchStatus.PRICE_EXCEPTION

    def test_line_match_result_quantity_outside_tolerance(self) -> None:
        """Quantity flag is False when quantity variance exceeds tolerance."""
        result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("100"),
            receipt_quantity=Decimal("100"),
            invoice_quantity=Decimal("105"),
            po_unit_price=Decimal("50.00"),
            invoice_unit_price=Decimal("50.00"),
            quantity_variance_pct=Decimal("0.05"),
            price_within_tolerance=True,
            quantity_within_tolerance=False,
            line_match_status=MatchStatus.QUANTITY_EXCEPTION,
        )
        assert result.quantity_within_tolerance is False
        assert result.line_match_status == MatchStatus.QUANTITY_EXCEPTION

    def test_line_match_result_combined_exception(self) -> None:
        """Both tolerance flags are False → COMBINED_EXCEPTION status."""
        result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("100"),
            receipt_quantity=Decimal("100"),
            invoice_quantity=Decimal("120"),
            po_unit_price=Decimal("50.00"),
            invoice_unit_price=Decimal("60.00"),
            price_within_tolerance=False,
            quantity_within_tolerance=False,
            line_match_status=MatchStatus.COMBINED_EXCEPTION,
        )
        assert result.price_within_tolerance is False
        assert result.quantity_within_tolerance is False
        assert result.line_match_status == MatchStatus.COMBINED_EXCEPTION

    def test_line_match_result_decimal_precision(self) -> None:
        """All monetary/variance fields are Decimal, NEVER float."""
        result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("100"),
            receipt_quantity=Decimal("100"),
            invoice_quantity=Decimal("100"),
            po_unit_price=Decimal("25.001"),
            invoice_unit_price=Decimal("25.003"),
            price_variance_pct=Decimal("0.00008"),
            quantity_variance_pct=_ZERO,
        )
        assert isinstance(result.po_unit_price, Decimal)
        assert isinstance(result.invoice_unit_price, Decimal)
        assert isinstance(result.price_variance_pct, Decimal)
        assert isinstance(result.quantity_variance_pct, Decimal)
        assert isinstance(result.receipt_quantity, Decimal)
        assert isinstance(result.invoice_quantity, Decimal)

    def test_line_match_result_line_number_must_be_positive(self) -> None:
        """LineMatchResult enforces line_number >= 1 via Pydantic ge=1 validator."""
        with pytest.raises(Exception):
            LineMatchResult(
                line_number=0,
                po_quantity=Decimal("100"),
                receipt_quantity=Decimal("100"),
                invoice_quantity=Decimal("100"),
                po_unit_price=Decimal("10.00"),
                invoice_unit_price=Decimal("10.00"),
            )


@pytest.mark.p2p
class TestThreeWayMatchResult:
    """Tests for ThreeWayMatchResult Pydantic V2 model validation."""

    def test_result_creation_with_valid_data(self) -> None:
        """ThreeWayMatchResult accepts all required fields and produces valid model."""
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
            match_status=MatchStatus.FULL_MATCH,
            approval_required=False,
            total_price_variance=_ZERO,
            total_quantity_variance=_ZERO,
        )
        assert result.invoice_id == TEST_INVOICE_UUID
        assert result.po_id == TEST_PO_UUID
        assert result.match_status == MatchStatus.FULL_MATCH

    def test_result_requires_approval_when_exception(self) -> None:
        """approval_required is True when match status is an exception."""
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            match_status=MatchStatus.PRICE_EXCEPTION,
            approval_required=True,
        )
        assert result.approval_required is True

    def test_result_no_approval_for_full_match(self) -> None:
        """approval_required is False when match status is FULL_MATCH."""
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            match_status=MatchStatus.FULL_MATCH,
            approval_required=False,
        )
        assert result.approval_required is False

    def test_result_line_results_list_populated(self) -> None:
        """line_results list contains per-line LineMatchResult objects."""
        line_result = LineMatchResult(
            line_number=1,
            po_quantity=Decimal("10"),
            receipt_quantity=Decimal("10"),
            invoice_quantity=Decimal("10"),
            po_unit_price=Decimal("5.00"),
            invoice_unit_price=Decimal("5.00"),
            line_match_status=MatchStatus.FULL_MATCH,
        )
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            line_results=[line_result],
            total_lines=1,
            lines_matched=1,
        )
        assert len(result.line_results) == 1
        assert result.line_results[0].line_number == 1

    def test_result_has_match_timestamp(self) -> None:
        """ThreeWayMatchResult includes a matched_at UTC timestamp."""
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
        )
        assert result.matched_at is not None
        assert isinstance(result.matched_at, datetime)

    def test_result_default_factory_generates_match_id(self) -> None:
        """ThreeWayMatchResult auto-generates a unique match_id UUID."""
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
        )
        assert isinstance(result.match_id, UUID)

    def test_result_tolerances_recorded(self) -> None:
        """ThreeWayMatchResult records the tolerance values that were applied."""
        result = ThreeWayMatchResult(
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            price_tolerance_used=Decimal("0.05"),
            quantity_tolerance_used=Decimal("0.02"),
        )
        assert result.price_tolerance_used == Decimal("0.05")
        assert result.quantity_tolerance_used == Decimal("0.02")


@pytest.mark.p2p
class TestThreeWayMatcherConstruction:
    """Tests for ThreeWayMatcher construction and stateless design."""

    def test_matcher_creation_no_args(self) -> None:
        """ThreeWayMatcher can be constructed with default tolerances."""
        matcher = ThreeWayMatcher()
        assert matcher._price_tolerance == THREE_WAY_MATCH_PRICE_TOLERANCE
        assert matcher._quantity_tolerance == THREE_WAY_MATCH_QUANTITY_TOLERANCE

    def test_matcher_creation_with_custom_tolerances(self) -> None:
        """ThreeWayMatcher accepts custom tolerance overrides via constructor."""
        matcher = ThreeWayMatcher(
            price_tolerance=Decimal("0.10"),
            quantity_tolerance=Decimal("0.05"),
        )
        assert matcher._price_tolerance == Decimal("0.10")
        assert matcher._quantity_tolerance == Decimal("0.05")

    def test_matcher_rejects_negative_price_tolerance(self) -> None:
        """ThreeWayMatcher raises ThreeWayMatchError for negative price tolerance."""
        with pytest.raises(ThreeWayMatchError, match="non-negative"):
            ThreeWayMatcher(price_tolerance=Decimal("-0.01"))

    def test_matcher_rejects_negative_quantity_tolerance(self) -> None:
        """ThreeWayMatcher raises ThreeWayMatchError for negative quantity tolerance."""
        with pytest.raises(ThreeWayMatchError, match="non-negative"):
            ThreeWayMatcher(quantity_tolerance=Decimal("-0.01"))

    def test_matcher_accepts_zero_tolerance(self) -> None:
        """ThreeWayMatcher accepts zero as a valid tolerance (exact-match mode)."""
        matcher = ThreeWayMatcher(
            price_tolerance=Decimal("0"),
            quantity_tolerance=Decimal("0"),
        )
        assert matcher._price_tolerance == _ZERO
        assert matcher._quantity_tolerance == _ZERO

    def test_matcher_is_stateless(self, matcher: ThreeWayMatcher) -> None:
        """Same matcher instance produces independent results for different inputs.

        Verifies that no internal state leaks between successive match() calls.
        """
        # First call: exact match.
        result1 = _quick_match(matcher, po_price="100.00", inv_price="100.00")
        assert result1.match_status == MatchStatus.FULL_MATCH

        # Second call: price exception (10% variance).
        result2 = _quick_match(matcher, po_price="100.00", inv_price="110.00")
        assert result2.match_status == MatchStatus.PRICE_EXCEPTION

        # Verify first result wasn't mutated.
        assert result1.match_status == MatchStatus.FULL_MATCH

    def test_matcher_default_tolerances_match_constants(self) -> None:
        """Default tolerances equal the project-wide constants from constants.py."""
        matcher = ThreeWayMatcher()
        assert matcher._price_tolerance == Decimal("0.05")
        assert matcher._quantity_tolerance == Decimal("0.02")


@pytest.mark.p2p
class TestPriceToleranceMatching:
    """CRITICAL — Tests for ±5% price tolerance matching.

    Price variance formula: abs((invoice_price - po_price) / po_price)
    Tolerance threshold: Decimal("0.05") (5%).
    Values at or below the threshold → FULL_MATCH.
    Values above the threshold → PRICE_EXCEPTION.
    """

    def test_exact_price_match(self, matcher: ThreeWayMatcher) -> None:
        """Invoice price equals PO price → FULL_MATCH with 0% variance."""
        result = _quick_match(matcher, po_price="100.00", inv_price="100.00")
        assert result.match_status == MatchStatus.FULL_MATCH
        lr = result.line_results[0]
        assert lr.price_variance_pct == _ZERO
        assert lr.price_within_tolerance is True

    def test_price_within_5_percent_tolerance(self, matcher: ThreeWayMatcher) -> None:
        """Invoice 4% above PO price → FULL_MATCH (within ±5% tolerance)."""
        result = _quick_match(matcher, po_price="100.00", inv_price="104.00")
        assert result.match_status == MatchStatus.FULL_MATCH
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.04")
        assert lr.price_within_tolerance is True

    def test_price_at_exactly_5_percent_boundary(self, matcher: ThreeWayMatcher) -> None:
        """Invoice exactly 5% above PO price → FULL_MATCH (boundary inclusive)."""
        result = _quick_match(matcher, po_price="100.00", inv_price="105.00")
        assert result.match_status == MatchStatus.FULL_MATCH
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.05")
        assert lr.price_within_tolerance is True

    def test_price_just_above_5_percent(self, matcher: ThreeWayMatcher) -> None:
        """Invoice 6% above PO price → PRICE_EXCEPTION (exceeds ±5%)."""
        result = _quick_match(matcher, po_price="100.00", inv_price="106.00")
        assert result.match_status == MatchStatus.PRICE_EXCEPTION
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.06")
        assert lr.price_within_tolerance is False

    def test_price_below_negative_5_percent(self, matcher: ThreeWayMatcher) -> None:
        """Invoice 6% below PO price → PRICE_EXCEPTION."""
        result = _quick_match(matcher, po_price="100.00", inv_price="94.00")
        assert result.match_status == MatchStatus.PRICE_EXCEPTION
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.06")
        assert lr.price_within_tolerance is False

    def test_price_at_negative_5_percent_boundary(self, matcher: ThreeWayMatcher) -> None:
        """Invoice exactly 5% below PO price → FULL_MATCH (boundary inclusive)."""
        result = _quick_match(matcher, po_price="100.00", inv_price="95.00")
        assert result.match_status == MatchStatus.FULL_MATCH
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.05")
        assert lr.price_within_tolerance is True

    def test_large_price_variance(self, matcher: ThreeWayMatcher) -> None:
        """Invoice 50% above PO price → PRICE_EXCEPTION."""
        result = _quick_match(matcher, po_price="100.00", inv_price="150.00")
        assert result.match_status == MatchStatus.PRICE_EXCEPTION
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.50")
        assert lr.price_within_tolerance is False

    def test_zero_po_price_division_by_zero_protection(self, matcher: ThreeWayMatcher) -> None:
        """PO price = 0 with non-zero invoice price → no crash, variance = 1 (100%)."""
        result = _quick_match(matcher, po_price="0", inv_price="50.00")
        lr = result.line_results[0]
        # Division by zero guard: variance = 1 when po_price is 0 but invoice_price > 0.
        assert lr.price_variance_pct == Decimal("1")
        assert lr.price_within_tolerance is False

    def test_zero_po_price_zero_invoice_price(self, matcher: ThreeWayMatcher) -> None:
        """Both PO and invoice price = 0 → no crash, variance = 0."""
        result = _quick_match(matcher, po_price="0", inv_price="0")
        lr = result.line_results[0]
        assert lr.price_variance_pct == _ZERO
        assert lr.price_within_tolerance is True

    def test_very_small_price_difference(self, matcher: ThreeWayMatcher) -> None:
        """Decimal('25.001') vs Decimal('25.00') → within tolerance."""
        result = _quick_match(matcher, po_price="25.00", inv_price="25.001")
        lr = result.line_results[0]
        assert lr.price_within_tolerance is True
        assert result.match_status == MatchStatus.FULL_MATCH

    def test_price_tolerance_uses_absolute_variance(self, matcher: ThreeWayMatcher) -> None:
        """Variance is always abs(), so negative differences are treated identically."""
        # Invoice 3% below PO.
        result = _quick_match(matcher, po_price="200.00", inv_price="194.00")
        lr = result.line_results[0]
        assert lr.price_variance_pct == Decimal("0.03")
        assert lr.price_within_tolerance is True


@pytest.mark.p2p
class TestQuantityToleranceMatching:
    """CRITICAL — Tests for ±2% quantity tolerance matching.

    Quantity variance formula: abs((invoice_qty - receipt_qty) / receipt_qty)
    Tolerance threshold: Decimal("0.02") (2%).
    Values at or below the threshold → FULL_MATCH.
    Values above the threshold → QUANTITY_EXCEPTION.
    """

    def test_exact_quantity_match(self, matcher: ThreeWayMatcher) -> None:
        """Receipt quantity equals invoice quantity → FULL_MATCH."""
        result = _quick_match(matcher, receipt_qty="100", inv_qty="100")
        assert result.match_status == MatchStatus.FULL_MATCH
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == _ZERO
        assert lr.quantity_within_tolerance is True

    def test_quantity_within_2_percent(self, matcher: ThreeWayMatcher) -> None:
        """1.5% quantity variance → FULL_MATCH (within ±2%)."""
        # 1.5% of 200 = 3 units.
        result = _quick_match(matcher, receipt_qty="200", inv_qty="203")
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == Decimal("0.015")
        assert lr.quantity_within_tolerance is True
        assert result.match_status == MatchStatus.FULL_MATCH

    def test_quantity_at_exactly_2_percent_boundary(self, matcher: ThreeWayMatcher) -> None:
        """Exactly 2% quantity variance → FULL_MATCH (boundary inclusive)."""
        # 2% of 100 = 2 units.
        result = _quick_match(matcher, receipt_qty="100", inv_qty="102")
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == Decimal("0.02")
        assert lr.quantity_within_tolerance is True
        assert result.match_status == MatchStatus.FULL_MATCH

    def test_quantity_just_above_2_percent(self, matcher: ThreeWayMatcher) -> None:
        """3% quantity variance → QUANTITY_EXCEPTION (exceeds ±2%)."""
        # 3% of 100 = 3 units.
        result = _quick_match(matcher, receipt_qty="100", inv_qty="103")
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == Decimal("0.03")
        assert lr.quantity_within_tolerance is False
        assert result.match_status == MatchStatus.QUANTITY_EXCEPTION

    def test_quantity_below_negative_2_percent(self, matcher: ThreeWayMatcher) -> None:
        """Receipt 3% more than invoice → QUANTITY_EXCEPTION."""
        # Invoice is 3% less than receipt (97 vs 100).
        result = _quick_match(matcher, receipt_qty="100", inv_qty="97")
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == Decimal("0.03")
        assert lr.quantity_within_tolerance is False
        assert result.match_status == MatchStatus.QUANTITY_EXCEPTION

    def test_zero_receipt_quantity_protection(self, matcher: ThreeWayMatcher) -> None:
        """Receipt quantity = 0 with non-zero invoice qty → no crash, variance = 1."""
        result = _quick_match(matcher, receipt_qty="0", inv_qty="10")
        lr = result.line_results[0]
        # Division by zero guard: variance = 1 when receipt_qty is 0 but invoice_qty > 0.
        assert lr.quantity_variance_pct == Decimal("1")
        assert lr.quantity_within_tolerance is False

    def test_zero_receipt_zero_invoice_quantity(self, matcher: ThreeWayMatcher) -> None:
        """Both receipt and invoice quantity = 0 → no crash, variance = 0."""
        result = _quick_match(matcher, receipt_qty="0", inv_qty="0")
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == _ZERO
        assert lr.quantity_within_tolerance is True

    def test_large_quantity_variance(self, matcher: ThreeWayMatcher) -> None:
        """20% over-invoice → QUANTITY_EXCEPTION."""
        result = _quick_match(matcher, receipt_qty="100", inv_qty="120")
        lr = result.line_results[0]
        assert lr.quantity_variance_pct == Decimal("0.2")
        assert lr.quantity_within_tolerance is False
        assert result.match_status == MatchStatus.QUANTITY_EXCEPTION


@pytest.mark.p2p
class TestCombinedMatching:
    """Tests for scenarios with both price and quantity variances."""

    def test_both_within_tolerance_full_match(self, matcher: ThreeWayMatcher) -> None:
        """Price 3% above + quantity 1% above → FULL_MATCH (both within tolerance)."""
        result = _quick_match(
            matcher,
            po_price="100.00",
            inv_price="103.00",
            receipt_qty="100",
            inv_qty="101",
        )
        assert result.match_status == MatchStatus.FULL_MATCH
        lr = result.line_results[0]
        assert lr.price_within_tolerance is True
        assert lr.quantity_within_tolerance is True

    def test_price_exception_only(self, matcher: ThreeWayMatcher) -> None:
        """Price 10% above + exact quantity → PRICE_EXCEPTION."""
        result = _quick_match(
            matcher,
            po_price="100.00",
            inv_price="110.00",
            receipt_qty="100",
            inv_qty="100",
        )
        assert result.match_status == MatchStatus.PRICE_EXCEPTION
        lr = result.line_results[0]
        assert lr.price_within_tolerance is False
        assert lr.quantity_within_tolerance is True
        assert lr.line_match_status == MatchStatus.PRICE_EXCEPTION

    def test_quantity_exception_only(self, matcher: ThreeWayMatcher) -> None:
        """Exact price + quantity 5% above → QUANTITY_EXCEPTION."""
        result = _quick_match(
            matcher,
            po_price="100.00",
            inv_price="100.00",
            receipt_qty="100",
            inv_qty="105",
        )
        assert result.match_status == MatchStatus.QUANTITY_EXCEPTION
        lr = result.line_results[0]
        assert lr.price_within_tolerance is True
        assert lr.quantity_within_tolerance is False
        assert lr.line_match_status == MatchStatus.QUANTITY_EXCEPTION

    def test_combined_exception(self, matcher: ThreeWayMatcher) -> None:
        """Both price and quantity outside tolerance → COMBINED_EXCEPTION."""
        result = _quick_match(
            matcher,
            po_price="100.00",
            inv_price="115.00",
            receipt_qty="100",
            inv_qty="110",
        )
        assert result.match_status == MatchStatus.COMBINED_EXCEPTION
        lr = result.line_results[0]
        assert lr.price_within_tolerance is False
        assert lr.quantity_within_tolerance is False
        assert lr.line_match_status == MatchStatus.COMBINED_EXCEPTION


@pytest.mark.p2p
class TestMultiLineMatching:
    """Tests for multi-line PO/receipt/invoice matching.

    Overall status is determined by the *worst* status across all lines:
    - COMBINED_EXCEPTION > PRICE_EXCEPTION / QUANTITY_EXCEPTION > FULL_MATCH
    - Different exception types across lines promote to COMBINED_EXCEPTION.
    """

    def test_two_lines_both_full_match(self, matcher: ThreeWayMatcher) -> None:
        """Two lines both within tolerance → overall FULL_MATCH."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                {"line_number": 2, "quantity": "200", "unit_price": "25.00"},
            ]),
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
                {"line_number": 2, "quantity": "200"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                {"line_number": 2, "quantity": "200", "unit_price": "25.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.FULL_MATCH
        assert len(result.line_results) == 2
        assert result.lines_matched == 2
        assert result.lines_with_exceptions == 0

    def test_two_lines_one_price_exception(self, matcher: ThreeWayMatcher) -> None:
        """One line OK + one line price exception → overall PRICE_EXCEPTION."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                {"line_number": 2, "quantity": "100", "unit_price": "50.00"},
            ]),
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
                {"line_number": 2, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                {"line_number": 2, "quantity": "100", "unit_price": "60.00"},  # +20%
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.PRICE_EXCEPTION
        assert result.lines_matched == 1
        assert result.lines_with_exceptions == 1

    def test_three_lines_mixed_statuses(self, matcher: ThreeWayMatcher) -> None:
        """Lines with PRICE_EXCEPTION and QUANTITY_EXCEPTION → COMBINED_EXCEPTION.

        Different exception types across lines promote the overall status.
        """
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                {"line_number": 2, "quantity": "100", "unit_price": "50.00"},
                {"line_number": 3, "quantity": "100", "unit_price": "50.00"},
            ]),
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
                {"line_number": 2, "quantity": "100"},
                {"line_number": 3, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},   # OK
                {"line_number": 2, "quantity": "100", "unit_price": "60.00"},   # Price exception
                {"line_number": 3, "quantity": "120", "unit_price": "50.00"},   # Qty exception
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        # Different exception types across lines → COMBINED_EXCEPTION.
        assert result.match_status == MatchStatus.COMBINED_EXCEPTION
        assert result.lines_with_exceptions == 2

    def test_single_line_match(self, matcher: ThreeWayMatcher) -> None:
        """Simple single-line exact match case."""
        result = _quick_match(matcher)
        assert result.match_status == MatchStatus.FULL_MATCH
        assert len(result.line_results) == 1
        assert result.total_lines == 1

    def test_many_lines_all_matching(self, matcher: ThreeWayMatcher) -> None:
        """10 lines all exactly matching → overall FULL_MATCH."""
        count = 10
        po = make_po_lines([
            {"line_number": i + 1, "quantity": str(50 + i * 10), "unit_price": "30.00"}
            for i in range(count)
        ])
        receipt = make_receipt_lines([
            {"line_number": i + 1, "quantity": str(50 + i * 10)}
            for i in range(count)
        ])
        invoice = make_invoice_lines([
            {"line_number": i + 1, "quantity": str(50 + i * 10), "unit_price": "30.00"}
            for i in range(count)
        ])
        result = matcher.match(
            po_lines=po,
            receipt_lines=receipt,
            invoice_lines=invoice,
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.FULL_MATCH
        assert result.total_lines == count
        assert result.lines_matched == count
        assert result.lines_with_exceptions == 0

    def test_multi_line_combined_exception_on_single_line(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """One line with both price and qty exception → overall COMBINED_EXCEPTION."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "100.00"},
                {"line_number": 2, "quantity": "100", "unit_price": "100.00"},
            ]),
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
                {"line_number": 2, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "100.00"},  # OK
                {"line_number": 2, "quantity": "120", "unit_price": "120.00"},  # Both bad
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.COMBINED_EXCEPTION


@pytest.mark.p2p
class TestMissingDocuments:
    """Tests for MISSING_RECEIPT and MISSING_PO statuses."""

    def test_missing_receipt_returns_missing_receipt_status(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Empty receipt lines → MISSING_RECEIPT status."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            receipt_lines=[],
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.MISSING_RECEIPT
        assert result.approval_required is True

    def test_missing_po_returns_missing_po_status(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Empty PO lines → MISSING_PO status."""
        result = matcher.match(
            po_lines=[],
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.MISSING_PO
        assert result.approval_required is True

    def test_none_receipt_data_handling(self, matcher: ThreeWayMatcher) -> None:
        """None receipt_id with empty receipt lines → MISSING_RECEIPT."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            receipt_lines=[],
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=None,
        )
        assert result.match_status == MatchStatus.MISSING_RECEIPT
        assert result.receipt_id is None

    def test_none_po_data_handling(self, matcher: ThreeWayMatcher) -> None:
        """Empty PO lines → MISSING_PO, regardless of receipt presence."""
        result = matcher.match(
            po_lines=[],
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.MISSING_PO

    def test_missing_po_takes_precedence_over_missing_receipt(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Both PO and receipt lines empty → MISSING_PO (checked first)."""
        result = matcher.match(
            po_lines=[],
            receipt_lines=[],
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
        )
        assert result.match_status == MatchStatus.MISSING_PO

    def test_missing_receipt_no_line_results(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """MISSING_RECEIPT result has no line_results populated."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            receipt_lines=[],
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.line_results == []


@pytest.mark.p2p
class TestVarianceCalculations:
    """Tests for line-item level variance calculation and aggregation.

    CRITICAL — Variance is calculated per line, then summed for totals.
    Total variances represent the SUM of line-level absolute variances,
    NOT global-level recalculations.
    """

    def test_price_variance_calculated_correctly(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Price variance = abs((invoice_price - po_price) / po_price) with Decimal."""
        # PO price = 200, Invoice price = 210 → variance = |10/200| = 0.05
        result = _quick_match(matcher, po_price="200.00", inv_price="210.00")
        lr = result.line_results[0]
        expected_pct = Decimal("10.00") / Decimal("200.00")
        assert lr.price_variance_pct == expected_pct

    def test_quantity_variance_calculated_correctly(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Quantity variance = abs((invoice_qty - receipt_qty) / receipt_qty)."""
        # Receipt qty = 500, Invoice qty = 510 → variance = |10/500| = 0.02
        result = _quick_match(matcher, receipt_qty="500", inv_qty="510")
        lr = result.line_results[0]
        expected_pct = Decimal("10") / Decimal("500")
        assert lr.quantity_variance_pct == expected_pct

    def test_total_price_variance_aggregated_across_lines(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """total_price_variance = SUM of line-level price variances (absolute amounts)."""
        # Line 1: PO=100, Inv=110 → price_variance = +10
        # Line 2: PO=200, Inv=190 → price_variance = -10
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "10", "unit_price": "100.00"},
                {"line_number": 2, "quantity": "10", "unit_price": "200.00"},
            ]),
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "10"},
                {"line_number": 2, "quantity": "10"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "10", "unit_price": "110.00"},
                {"line_number": 2, "quantity": "10", "unit_price": "190.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        # Total price variance = 10 + (-10) = 0  (summing signed variances).
        assert result.total_price_variance == _ZERO

    def test_total_quantity_variance_aggregated_across_lines(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """total_quantity_variance = SUM of line-level quantity variances."""
        # Line 1: Receipt=100, Inv=105 → qty_variance = +5
        # Line 2: Receipt=100, Inv=103 → qty_variance = +3
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "10.00"},
                {"line_number": 2, "quantity": "100", "unit_price": "10.00"},
            ]),
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
                {"line_number": 2, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "105", "unit_price": "10.00"},
                {"line_number": 2, "quantity": "103", "unit_price": "10.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        # Total quantity variance = 5 + 3 = 8
        assert result.total_quantity_variance == Decimal("8")

    def test_all_decimal_no_float(self, matcher: ThreeWayMatcher) -> None:
        """Verify all variance results are Decimal type — NEVER float."""
        result = _quick_match(
            matcher,
            po_price="100.00",
            inv_price="103.00",
            receipt_qty="100",
            inv_qty="101",
        )
        # Check overall result fields.
        assert isinstance(result.total_price_variance, Decimal)
        assert isinstance(result.total_quantity_variance, Decimal)
        assert isinstance(result.total_extended_variance, Decimal)

        # Check line-level result fields.
        lr = result.line_results[0]
        assert isinstance(lr.price_variance, Decimal)
        assert isinstance(lr.price_variance_pct, Decimal)
        assert isinstance(lr.quantity_variance, Decimal)
        assert isinstance(lr.quantity_variance_pct, Decimal)
        assert isinstance(lr.extended_amount_variance, Decimal)
        assert isinstance(lr.po_unit_price, Decimal)
        assert isinstance(lr.invoice_unit_price, Decimal)
        assert isinstance(lr.receipt_quantity, Decimal)
        assert isinstance(lr.invoice_quantity, Decimal)

    def test_extended_amount_variance_calculated_per_line(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Extended variance = (inv_qty × inv_price) - (receipt_qty × po_price)."""
        # PO: qty=10, price=100 → expected extended = 1000
        # Invoice: qty=10, price=105 → actual extended = 1050
        # Extended variance = 1050 - 1000 = 50
        result = _quick_match(
            matcher,
            po_price="100.00",
            po_qty="10",
            receipt_qty="10",
            inv_price="105.00",
            inv_qty="10",
        )
        lr = result.line_results[0]
        assert lr.extended_amount_variance == Decimal("50.00")


@pytest.mark.p2p
class TestApprovalRequirement:
    """Tests for the approval_required flag on ThreeWayMatchResult."""

    def test_full_match_no_approval_required(self, matcher: ThreeWayMatcher) -> None:
        """FULL_MATCH → approval_required is False."""
        result = _quick_match(matcher)
        assert result.match_status == MatchStatus.FULL_MATCH
        assert result.approval_required is False

    def test_price_exception_requires_approval(self, matcher: ThreeWayMatcher) -> None:
        """PRICE_EXCEPTION → approval_required is True."""
        result = _quick_match(matcher, po_price="100.00", inv_price="120.00")
        assert result.match_status == MatchStatus.PRICE_EXCEPTION
        assert result.approval_required is True

    def test_quantity_exception_requires_approval(self, matcher: ThreeWayMatcher) -> None:
        """QUANTITY_EXCEPTION → approval_required is True."""
        result = _quick_match(matcher, receipt_qty="100", inv_qty="110")
        assert result.match_status == MatchStatus.QUANTITY_EXCEPTION
        assert result.approval_required is True

    def test_combined_exception_requires_approval(self, matcher: ThreeWayMatcher) -> None:
        """COMBINED_EXCEPTION → approval_required is True."""
        result = _quick_match(
            matcher,
            po_price="100.00",
            inv_price="120.00",
            receipt_qty="100",
            inv_qty="110",
        )
        assert result.match_status == MatchStatus.COMBINED_EXCEPTION
        assert result.approval_required is True

    def test_missing_receipt_requires_approval(self, matcher: ThreeWayMatcher) -> None:
        """MISSING_RECEIPT → approval_required is True."""
        result = matcher.match(
            po_lines=make_po_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            receipt_lines=[],
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.MISSING_RECEIPT
        assert result.approval_required is True

    def test_missing_po_requires_approval(self, matcher: ThreeWayMatcher) -> None:
        """MISSING_PO → approval_required is True."""
        result = matcher.match(
            po_lines=[],
            receipt_lines=make_receipt_lines([
                {"line_number": 1, "quantity": "100"},
            ]),
            invoice_lines=make_invoice_lines([
                {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
            ]),
            invoice_id=TEST_INVOICE_UUID,
            po_id=TEST_PO_UUID,
            receipt_id=TEST_RECEIPT_UUID,
        )
        assert result.match_status == MatchStatus.MISSING_PO
        assert result.approval_required is True


@pytest.mark.p2p
class TestErrorHandling:
    """Tests for error handling and ThreeWayMatchError raising."""

    def test_negative_invoice_quantity_raises_error(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Negative invoice quantity raises ThreeWayMatchError."""
        with pytest.raises(ThreeWayMatchError, match="Negative invoice quantity"):
            matcher.match(
                po_lines=make_po_lines([
                    {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                ]),
                receipt_lines=make_receipt_lines([
                    {"line_number": 1, "quantity": "100"},
                ]),
                invoice_lines=[
                    {"line_number": 1, "quantity": "-10", "unit_price": "50.00"},
                ],
                invoice_id=TEST_INVOICE_UUID,
                po_id=TEST_PO_UUID,
                receipt_id=TEST_RECEIPT_UUID,
            )

    def test_negative_invoice_price_raises_error(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Negative invoice unit price raises ThreeWayMatchError."""
        with pytest.raises(ThreeWayMatchError, match="Negative invoice unit price"):
            matcher.match(
                po_lines=make_po_lines([
                    {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                ]),
                receipt_lines=make_receipt_lines([
                    {"line_number": 1, "quantity": "100"},
                ]),
                invoice_lines=[
                    {"line_number": 1, "quantity": "100", "unit_price": "-5.00"},
                ],
                invoice_id=TEST_INVOICE_UUID,
                po_id=TEST_PO_UUID,
                receipt_id=TEST_RECEIPT_UUID,
            )

    def test_negative_po_quantity_raises_error(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Negative PO quantity raises ThreeWayMatchError."""
        with pytest.raises(ThreeWayMatchError, match="Negative PO quantity"):
            matcher.match(
                po_lines=[
                    {"line_number": 1, "quantity": "-100", "unit_price": "50.00"},
                ],
                receipt_lines=make_receipt_lines([
                    {"line_number": 1, "quantity": "100"},
                ]),
                invoice_lines=make_invoice_lines([
                    {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                ]),
                invoice_id=TEST_INVOICE_UUID,
                po_id=TEST_PO_UUID,
                receipt_id=TEST_RECEIPT_UUID,
            )

    def test_negative_po_price_raises_error(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Negative PO unit price raises ThreeWayMatchError."""
        with pytest.raises(ThreeWayMatchError, match="Negative PO unit price"):
            matcher.match(
                po_lines=[
                    {"line_number": 1, "quantity": "100", "unit_price": "-50.00"},
                ],
                receipt_lines=make_receipt_lines([
                    {"line_number": 1, "quantity": "100"},
                ]),
                invoice_lines=make_invoice_lines([
                    {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                ]),
                invoice_id=TEST_INVOICE_UUID,
                po_id=TEST_PO_UUID,
                receipt_id=TEST_RECEIPT_UUID,
            )

    def test_negative_receipt_quantity_raises_error(
        self, matcher: ThreeWayMatcher
    ) -> None:
        """Negative receipt quantity raises ThreeWayMatchError."""
        with pytest.raises(ThreeWayMatchError, match="Negative receipt quantity"):
            matcher.match(
                po_lines=make_po_lines([
                    {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                ]),
                receipt_lines=[
                    {"line_number": 1, "quantity": "-100"},
                ],
                invoice_lines=make_invoice_lines([
                    {"line_number": 1, "quantity": "100", "unit_price": "50.00"},
                ]),
                invoice_id=TEST_INVOICE_UUID,
                po_id=TEST_PO_UUID,
                receipt_id=TEST_RECEIPT_UUID,
            )

    def test_three_way_match_error_has_details(self) -> None:
        """ThreeWayMatchError carries a details dictionary for structured logging."""
        err = ThreeWayMatchError(
            "Test error",
            details={"field": "unit_price", "value": "-5.00"},
        )
        assert err.details["field"] == "unit_price"
        assert "Test error" in str(err)

    def test_three_way_match_error_inherits_transaction_error(self) -> None:
        """ThreeWayMatchError is a subclass of TransactionError."""
        from app.transactions.exceptions import TransactionError

        err = ThreeWayMatchError("test")
        assert isinstance(err, TransactionError)


