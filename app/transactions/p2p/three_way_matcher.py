"""Three-way matching engine for PO / Receipt / Invoice validation.

Implements line-item level three-way matching for the Procure-to-Pay (P2P) cycle.
The matcher validates vendor invoices against their originating purchase orders
and goods receipts, applying configurable tolerance thresholds:

    - **Price tolerance**: ±5 % (PO unit price vs. Invoice unit price)
    - **Quantity tolerance**: ±2 % (Receipt quantity vs. Invoice quantity)

**CRITICAL DESIGN — Line-Item Level Variance Calculation**

    Variance is calculated at the LINE-ITEM level (per line), then summed to
    produce aggregate totals.  This is mandated by AAP §0.1.2:
    "Variance Calculation: Line-Item Level (calculated per line, then summed)".

Match Statuses:
    - ``FULL_MATCH``         — all lines within tolerance
    - ``PRICE_EXCEPTION``    — one or more lines outside price tolerance
    - ``QUANTITY_EXCEPTION`` — one or more lines outside quantity tolerance
    - ``COMBINED_EXCEPTION`` — lines have both price and quantity exceptions
    - ``MISSING_RECEIPT``    — no goods receipt found for the PO
    - ``MISSING_PO``         — no PO reference found

Integration:
    Consumed by :class:`VendorInvoiceProcessor` via constructor injection.
    Produces :class:`ThreeWayMatchResult` Pydantic V2 data contracts.

Retry Policy (AAP §0.1.2):
    2 attempts · linear backoff (500 ms, 1 s) · 10 s timeout
    Fallback: mark as exception

References:
    - AAP §0.1.2  — tolerance values, retry policy
    - AAP §0.5.1  — three-way matcher specification
    - AAP §0.7.1  — Pydantic V2 data contracts, structlog, Decimal precision
    - AAP §0.7.2  — financial integrity (Decimal only, ROUND_HALF_UP)
"""

from __future__ import annotations

import decimal
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.constants import (
    THREE_WAY_MATCH_PRICE_TOLERANCE,
    THREE_WAY_MATCH_QUANTITY_TOLERANCE,
    THREE_WAY_MATCH_TOLERANCES,
    RETRY_POLICIES,
)
from app.transactions.exceptions import ThreeWayMatchError

# ---------------------------------------------------------------------------
# Module-Level Configuration
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# Configure Decimal context globally for financial precision per AAP §0.7.2.
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# Cache the three-way matching retry policy from project-wide constants.
_THREE_WAY_RETRY_POLICY: Dict[str, Any] = RETRY_POLICIES.get(
    "three_way_matching",
    {
        "max_attempts": 2,
        "backoff_strategy": "linear",
        "backoff_seconds": [0.5, 1],
        "timeout_seconds": 10,
        "fallback": "mark_as_exception",
    },
)

# Pre-compute zero Decimal once to avoid repeated object creation.
_ZERO = Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════
# MatchStatus Enum
# ═══════════════════════════════════════════════════════════════════════════


class MatchStatus(str, Enum):
    """Match status outcomes for three-way matching.

    Each value represents the overall or per-line result of comparing a
    purchase order, goods receipt, and vendor invoice.

    Values:
        FULL_MATCH:         All lines within both price and quantity tolerances.
        PRICE_EXCEPTION:    One or more lines outside ±5 % price tolerance.
        QUANTITY_EXCEPTION: One or more lines outside ±2 % quantity tolerance.
        COMBINED_EXCEPTION: Lines outside *both* price and quantity tolerances.
        MISSING_RECEIPT:    No goods receipt exists for the referenced PO.
        MISSING_PO:         No purchase order reference found on the invoice.
    """

    FULL_MATCH = "full_match"
    PRICE_EXCEPTION = "price_exception"
    QUANTITY_EXCEPTION = "quantity_exception"
    COMBINED_EXCEPTION = "combined_exception"
    MISSING_RECEIPT = "missing_receipt"
    MISSING_PO = "missing_po"


# ═══════════════════════════════════════════════════════════════════════════
# LineMatchResult — per-line match data
# ═══════════════════════════════════════════════════════════════════════════


class LineMatchResult(BaseModel):
    """Match result for a single line item.

    Captures the PO, receipt, and invoice quantities/prices for one line,
    together with the computed variance percentages and tolerance checks.

    All monetary fields use :class:`~decimal.Decimal` — never ``float``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Identifiers -------------------------------------------------------
    line_number: int = Field(
        ...,
        ge=1,
        description="Line item number being matched (1-based).",
    )

    # --- Input quantities & prices -----------------------------------------
    po_quantity: Decimal = Field(
        ..., description="Ordered quantity from the purchase order."
    )
    receipt_quantity: Decimal = Field(
        ..., description="Received quantity from the goods receipt."
    )
    invoice_quantity: Decimal = Field(
        ..., description="Billed quantity from the vendor invoice."
    )
    po_unit_price: Decimal = Field(
        ..., description="Unit price on the purchase order."
    )
    invoice_unit_price: Decimal = Field(
        ..., description="Unit price on the vendor invoice."
    )

    # --- Calculated variances ----------------------------------------------
    quantity_variance: Decimal = Field(
        default=_ZERO,
        description="invoice_quantity − receipt_quantity.",
    )
    quantity_variance_pct: Decimal = Field(
        default=_ZERO,
        description="abs(quantity_variance) / receipt_quantity as a fraction.",
    )
    price_variance: Decimal = Field(
        default=_ZERO,
        description="invoice_unit_price − po_unit_price.",
    )
    price_variance_pct: Decimal = Field(
        default=_ZERO,
        description="abs(price_variance) / po_unit_price as a fraction.",
    )
    extended_amount_variance: Decimal = Field(
        default=_ZERO,
        description=(
            "Line-level extended amount variance: "
            "(invoice_qty × invoice_price) − (receipt_qty × po_price)."
        ),
    )

    # --- Tolerance checks --------------------------------------------------
    quantity_within_tolerance: bool = Field(
        default=True,
        description="True if quantity variance is within ±2 % tolerance.",
    )
    price_within_tolerance: bool = Field(
        default=True,
        description="True if price variance is within ±5 % tolerance.",
    )
    line_match_status: MatchStatus = Field(
        default=MatchStatus.FULL_MATCH,
        description="Per-line match status determination.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ThreeWayMatchResult — aggregate match result
# ═══════════════════════════════════════════════════════════════════════════


class ThreeWayMatchResult(BaseModel):
    """Complete three-way match result for a vendor invoice.

    Aggregates per-line :class:`LineMatchResult` objects into an overall
    match assessment with totals, status, and approval metadata.

    All monetary fields use :class:`~decimal.Decimal` — never ``float``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Identifiers -------------------------------------------------------
    match_id: UUID = Field(
        default_factory=uuid4,
        description="Unique match operation identifier.",
    )
    invoice_id: UUID = Field(
        ..., description="Vendor invoice being matched."
    )
    po_id: UUID = Field(
        ..., description="Purchase order reference."
    )
    receipt_id: Optional[UUID] = Field(
        default=None,
        description="Goods receipt reference (None → MISSING_RECEIPT).",
    )

    # --- Overall status ----------------------------------------------------
    match_status: MatchStatus = Field(
        default=MatchStatus.FULL_MATCH,
        description="Overall match outcome across all lines.",
    )
    approval_required: bool = Field(
        default=False,
        description="Whether match exceptions require manager approval.",
    )

    # --- Aggregate variances (line-item level, then summed — CRITICAL) -----
    total_price_variance: Decimal = Field(
        default=_ZERO,
        description="Sum of line-level price variances.",
    )
    total_quantity_variance: Decimal = Field(
        default=_ZERO,
        description="Sum of line-level quantity variances.",
    )
    total_extended_variance: Decimal = Field(
        default=_ZERO,
        description="Sum of line-level extended amount variances.",
    )
    price_tolerance_used: Decimal = Field(
        default=THREE_WAY_MATCH_PRICE_TOLERANCE,
        description="Price tolerance applied (±5 % default).",
    )
    quantity_tolerance_used: Decimal = Field(
        default=THREE_WAY_MATCH_QUANTITY_TOLERANCE,
        description="Quantity tolerance applied (±2 % default).",
    )

    # --- Line-level results ------------------------------------------------
    line_results: List[LineMatchResult] = Field(
        default_factory=list,
        description="Per-line match results.",
    )
    lines_matched: int = Field(
        default=0,
        description="Count of lines with FULL_MATCH status.",
    )
    lines_with_exceptions: int = Field(
        default=0,
        description="Count of lines with any exception status.",
    )
    total_lines: int = Field(
        default=0,
        description="Total number of lines compared.",
    )

    # --- Metadata ----------------------------------------------------------
    matched_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the match operation.",
    )
    matched_by: Optional[str] = Field(
        default="system",
        description="Identity of the matcher (user ID or 'system').",
    )
    notes: str = Field(
        default="",
        description="Free-form notes about the match result.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value metadata for extensibility.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# ThreeWayMatcher — stateless matching engine
# ═══════════════════════════════════════════════════════════════════════════


class ThreeWayMatcher:
    """Three-way matching engine for PO / Receipt / Invoice validation.

    Validates vendor invoices against their originating purchase orders and
    goods receipts at the **LINE-ITEM** level, applying configurable tolerance
    thresholds:

        - Price tolerance:    ±5 % (PO unit price vs. Invoice unit price)
        - Quantity tolerance:  ±2 % (Receipt quantity vs. Invoice quantity)

    **CRITICAL DESIGN** — Variance is calculated at the LINE-ITEM level
    (per line), then summed to produce aggregate totals.  This is per
    AAP §0.1.2: "Variance Calculation: Line-Item Level (calculated per
    line, then summed)".

    The matcher is a **stateless** service class — it does NOT extend
    ``TransactionGenerator``.  It receives no database session or event bus
    and produces no side effects.  It is consumed by
    :class:`VendorInvoiceProcessor` via constructor injection.

    Retry Policy (AAP §0.1.2):
        - 2 attempts, linear backoff (500 ms, 1 s), 10 s timeout
        - Fallback: mark as exception

    Attributes:
        _price_tolerance:    Decimal fraction for price variance comparison.
        _quantity_tolerance:  Decimal fraction for quantity variance comparison.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        *,
        price_tolerance: Optional[Decimal] = None,
        quantity_tolerance: Optional[Decimal] = None,
    ) -> None:
        """Initialise the three-way matcher with optional tolerance overrides.

        Args:
            price_tolerance:    Override for price tolerance (default ±5 %).
                                Must be a positive ``Decimal`` fraction (e.g.
                                ``Decimal("0.05")`` for 5 %).
            quantity_tolerance: Override for quantity tolerance (default ±2 %).
                                Must be a positive ``Decimal`` fraction (e.g.
                                ``Decimal("0.02")`` for 2 %).

        Raises:
            ThreeWayMatchError: If a provided tolerance is negative.
        """
        # Validate and assign price tolerance.
        if price_tolerance is not None:
            if price_tolerance < _ZERO:
                raise ThreeWayMatchError(
                    "Price tolerance must be non-negative",
                    details={"price_tolerance": str(price_tolerance)},
                )
            self._price_tolerance: Decimal = Decimal(str(price_tolerance))
        else:
            self._price_tolerance = THREE_WAY_MATCH_PRICE_TOLERANCE

        # Validate and assign quantity tolerance.
        if quantity_tolerance is not None:
            if quantity_tolerance < _ZERO:
                raise ThreeWayMatchError(
                    "Quantity tolerance must be non-negative",
                    details={"quantity_tolerance": str(quantity_tolerance)},
                )
            self._quantity_tolerance: Decimal = Decimal(str(quantity_tolerance))
        else:
            self._quantity_tolerance = THREE_WAY_MATCH_QUANTITY_TOLERANCE

        logger.info(
            "three_way_matcher_initialized",
            service_name="transactions",
            component="ThreeWayMatcher",
            price_tolerance=str(self._price_tolerance),
            quantity_tolerance=str(self._quantity_tolerance),
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def match(
        self,
        *,
        po_lines: Sequence[Dict[str, Any]],
        receipt_lines: Sequence[Dict[str, Any]],
        invoice_lines: Sequence[Dict[str, Any]],
        invoice_id: UUID,
        po_id: UUID,
        receipt_id: Optional[UUID] = None,
    ) -> ThreeWayMatchResult:
        """Execute three-way matching for a vendor invoice.

        Compares each invoice line against the corresponding PO and receipt
        lines by ``line_number``.  Variance is calculated at the line-item
        level, then aggregated to produce totals.

        Expected line dict schema (minimum fields)::

            {
                "line_number": int,           # 1-based line identifier
                "quantity":    Decimal | str,  # quantity (PO/receipt/invoice)
                "unit_price":  Decimal | str,  # unit price (PO/invoice only)
            }

        Args:
            po_lines:       Sequence of PO line dicts.
            receipt_lines:  Sequence of goods receipt line dicts.
            invoice_lines:  Sequence of vendor invoice line dicts.
            invoice_id:     UUID of the vendor invoice.
            po_id:          UUID of the purchase order.
            receipt_id:     UUID of the goods receipt (None → MISSING_RECEIPT).

        Returns:
            :class:`ThreeWayMatchResult` with per-line details and aggregates.

        Raises:
            ThreeWayMatchError: On unexpected matching errors (e.g. malformed
                line data).
        """
        try:
            return self._execute_match(
                po_lines=po_lines,
                receipt_lines=receipt_lines,
                invoice_lines=invoice_lines,
                invoice_id=invoice_id,
                po_id=po_id,
                receipt_id=receipt_id,
            )
        except ThreeWayMatchError:
            # Re-raise known match errors as-is.
            raise
        except Exception as exc:
            # Wrap unexpected exceptions for consistent error handling.
            logger.error(
                "three_way_match_unexpected_error",
                service_name="transactions",
                component="ThreeWayMatcher",
                invoice_id=str(invoice_id),
                po_id=str(po_id),
                error=str(exc),
            )
            raise ThreeWayMatchError(
                f"Unexpected error during three-way matching: {exc}",
                details={
                    "invoice_id": str(invoice_id),
                    "po_id": str(po_id),
                    "receipt_id": str(receipt_id) if receipt_id else None,
                    "error_type": type(exc).__name__,
                },
            ) from exc

    # -----------------------------------------------------------------------
    # Private — main execution
    # -----------------------------------------------------------------------

    def _execute_match(
        self,
        *,
        po_lines: Sequence[Dict[str, Any]],
        receipt_lines: Sequence[Dict[str, Any]],
        invoice_lines: Sequence[Dict[str, Any]],
        invoice_id: UUID,
        po_id: UUID,
        receipt_id: Optional[UUID],
    ) -> ThreeWayMatchResult:
        """Core matching logic separated from error boundary.

        This method handles:
            1. Missing document detection (MISSING_PO, MISSING_RECEIPT)
            2. Per-line matching via :meth:`_match_single_line`
            3. Aggregation via :meth:`_aggregate_variances`
            4. Overall status determination via :meth:`_determine_overall_status`
        """
        # --- 1. Missing document detection ---------------------------------

        if not po_lines:
            logger.warning(
                "three_way_match_missing_po",
                service_name="transactions",
                component="ThreeWayMatcher",
                invoice_id=str(invoice_id),
                po_id=str(po_id),
            )
            return ThreeWayMatchResult(
                invoice_id=invoice_id,
                po_id=po_id,
                receipt_id=receipt_id,
                match_status=MatchStatus.MISSING_PO,
                approval_required=True,
                price_tolerance_used=self._price_tolerance,
                quantity_tolerance_used=self._quantity_tolerance,
                notes="No purchase order lines found for matching.",
            )

        if not receipt_lines:
            logger.warning(
                "three_way_match_missing_receipt",
                service_name="transactions",
                component="ThreeWayMatcher",
                invoice_id=str(invoice_id),
                po_id=str(po_id),
            )
            return ThreeWayMatchResult(
                invoice_id=invoice_id,
                po_id=po_id,
                receipt_id=receipt_id,
                match_status=MatchStatus.MISSING_RECEIPT,
                approval_required=True,
                price_tolerance_used=self._price_tolerance,
                quantity_tolerance_used=self._quantity_tolerance,
                notes="No goods receipt lines found for matching.",
            )

        # --- 2. Build lookup indices by line_number ------------------------

        po_index: Dict[int, Dict[str, Any]] = {
            int(line.get("line_number", idx + 1)): line
            for idx, line in enumerate(po_lines)
        }
        receipt_index: Dict[int, Dict[str, Any]] = {
            int(line.get("line_number", idx + 1)): line
            for idx, line in enumerate(receipt_lines)
        }

        # --- 3. Per-line matching ------------------------------------------

        line_results: List[LineMatchResult] = []

        for idx, inv_line in enumerate(invoice_lines):
            line_number = int(inv_line.get("line_number", idx + 1))

            # Extract invoice values — coerce to Decimal for safety.
            invoice_quantity = Decimal(str(inv_line.get("quantity", "0")))
            invoice_unit_price = Decimal(str(inv_line.get("unit_price", "0")))

            # Input validation: reject negative quantities and prices (CWE-20).
            # Negative values would produce incorrect variance calculations
            # and erroneously flag legitimate matches as exceptions.
            if invoice_quantity < _ZERO:
                raise ThreeWayMatchError(
                    f"Negative invoice quantity ({invoice_quantity}) on "
                    f"line {line_number}",
                    details={
                        "line_number": line_number,
                        "field": "invoice_quantity",
                        "value": str(invoice_quantity),
                    },
                )
            if invoice_unit_price < _ZERO:
                raise ThreeWayMatchError(
                    f"Negative invoice unit price ({invoice_unit_price}) on "
                    f"line {line_number}",
                    details={
                        "line_number": line_number,
                        "field": "invoice_unit_price",
                        "value": str(invoice_unit_price),
                    },
                )

            # Lookup corresponding PO line.
            po_line = po_index.get(line_number, {})
            po_quantity = Decimal(str(po_line.get("quantity", "0")))
            po_unit_price = Decimal(str(po_line.get("unit_price", "0")))

            # Validate PO line values for non-negative quantities and prices.
            if po_quantity < _ZERO:
                raise ThreeWayMatchError(
                    f"Negative PO quantity ({po_quantity}) on "
                    f"line {line_number}",
                    details={
                        "line_number": line_number,
                        "field": "po_quantity",
                        "value": str(po_quantity),
                    },
                )
            if po_unit_price < _ZERO:
                raise ThreeWayMatchError(
                    f"Negative PO unit price ({po_unit_price}) on "
                    f"line {line_number}",
                    details={
                        "line_number": line_number,
                        "field": "po_unit_price",
                        "value": str(po_unit_price),
                    },
                )

            # Lookup corresponding receipt line.
            receipt_line = receipt_index.get(line_number, {})
            receipt_quantity = Decimal(str(receipt_line.get("quantity", "0")))

            # Validate receipt quantity for non-negative value.
            if receipt_quantity < _ZERO:
                raise ThreeWayMatchError(
                    f"Negative receipt quantity ({receipt_quantity}) on "
                    f"line {line_number}",
                    details={
                        "line_number": line_number,
                        "field": "receipt_quantity",
                        "value": str(receipt_quantity),
                    },
                )

            # Delegate per-line variance computation.
            line_result = self._match_single_line(
                line_number=line_number,
                po_quantity=po_quantity,
                po_unit_price=po_unit_price,
                receipt_quantity=receipt_quantity,
                invoice_quantity=invoice_quantity,
                invoice_unit_price=invoice_unit_price,
            )
            line_results.append(line_result)

        # --- 4. Aggregate from line-level (CRITICAL — NOT global calc) -----

        aggregates = self._aggregate_variances(line_results)

        # --- 5. Overall status determination -------------------------------

        overall_status = self._determine_overall_status(line_results)

        lines_matched = sum(
            1 for lr in line_results
            if lr.line_match_status == MatchStatus.FULL_MATCH
        )
        lines_with_exceptions = len(line_results) - lines_matched
        approval_required = overall_status != MatchStatus.FULL_MATCH

        # --- 6. Build result -----------------------------------------------

        result = ThreeWayMatchResult(
            invoice_id=invoice_id,
            po_id=po_id,
            receipt_id=receipt_id,
            match_status=overall_status,
            approval_required=approval_required,
            total_price_variance=aggregates["total_price_variance"],
            total_quantity_variance=aggregates["total_quantity_variance"],
            total_extended_variance=aggregates["total_extended_variance"],
            price_tolerance_used=self._price_tolerance,
            quantity_tolerance_used=self._quantity_tolerance,
            line_results=line_results,
            lines_matched=lines_matched,
            lines_with_exceptions=lines_with_exceptions,
            total_lines=len(line_results),
            notes=self._build_notes(
                overall_status, lines_matched, lines_with_exceptions,
            ),
            metadata={
                "retry_policy": _THREE_WAY_RETRY_POLICY,
                "tolerances": {
                    "price": str(self._price_tolerance),
                    "quantity": str(self._quantity_tolerance),
                },
            },
        )

        # --- 7. Log the result ---------------------------------------------

        logger.info(
            "three_way_match_completed",
            service_name="transactions",
            component="ThreeWayMatcher",
            invoice_id=str(invoice_id),
            po_id=str(po_id),
            receipt_id=str(receipt_id) if receipt_id else None,
            match_status=overall_status.value,
            total_lines=len(line_results),
            lines_matched=lines_matched,
            lines_with_exceptions=lines_with_exceptions,
            total_price_variance=str(aggregates["total_price_variance"]),
            total_quantity_variance=str(aggregates["total_quantity_variance"]),
            total_extended_variance=str(aggregates["total_extended_variance"]),
            approval_required=approval_required,
        )

        return result

    # -----------------------------------------------------------------------
    # Private — single-line matching
    # -----------------------------------------------------------------------

    def _match_single_line(
        self,
        *,
        line_number: int,
        po_quantity: Decimal,
        po_unit_price: Decimal,
        receipt_quantity: Decimal,
        invoice_quantity: Decimal,
        invoice_unit_price: Decimal,
    ) -> LineMatchResult:
        """Compute variance metrics for a single line item.

        All calculations use :class:`~decimal.Decimal` exclusively.
        Division-by-zero guards protect against zero denominators.

        Args:
            line_number:        1-based line identifier.
            po_quantity:        Ordered quantity from the PO.
            po_unit_price:      Unit price from the PO.
            receipt_quantity:    Received quantity from the goods receipt.
            invoice_quantity:    Billed quantity from the vendor invoice.
            invoice_unit_price: Unit price from the vendor invoice.

        Returns:
            Populated :class:`LineMatchResult` with variances and tolerances.
        """
        # --- Quantity variance (Receipt vs Invoice) ------------------------
        quantity_variance = invoice_quantity - receipt_quantity

        if receipt_quantity > _ZERO:
            quantity_variance_pct = abs(quantity_variance) / receipt_quantity
        else:
            # Guard: cannot compute percentage when receipt qty is zero.
            quantity_variance_pct = (
                _ZERO if invoice_quantity == _ZERO else Decimal("1")
            )

        quantity_within_tolerance = quantity_variance_pct <= self._quantity_tolerance

        # --- Price variance (PO vs Invoice) --------------------------------
        price_variance = invoice_unit_price - po_unit_price

        if po_unit_price > _ZERO:
            price_variance_pct = abs(price_variance) / po_unit_price
        else:
            # Guard: cannot compute percentage when PO unit price is zero.
            price_variance_pct = (
                _ZERO if invoice_unit_price == _ZERO else Decimal("1")
            )

        price_within_tolerance = price_variance_pct <= self._price_tolerance

        # --- Extended amount variance per line -----------------------------
        expected_extended = receipt_quantity * po_unit_price
        actual_extended = invoice_quantity * invoice_unit_price
        extended_amount_variance = actual_extended - expected_extended

        # --- Determine line match status -----------------------------------
        if price_within_tolerance and quantity_within_tolerance:
            line_status = MatchStatus.FULL_MATCH
        elif not price_within_tolerance and not quantity_within_tolerance:
            line_status = MatchStatus.COMBINED_EXCEPTION
        elif not price_within_tolerance:
            line_status = MatchStatus.PRICE_EXCEPTION
        else:
            line_status = MatchStatus.QUANTITY_EXCEPTION

        return LineMatchResult(
            line_number=line_number,
            po_quantity=po_quantity,
            receipt_quantity=receipt_quantity,
            invoice_quantity=invoice_quantity,
            po_unit_price=po_unit_price,
            invoice_unit_price=invoice_unit_price,
            quantity_variance=quantity_variance,
            quantity_variance_pct=quantity_variance_pct,
            price_variance=price_variance,
            price_variance_pct=price_variance_pct,
            extended_amount_variance=extended_amount_variance,
            quantity_within_tolerance=quantity_within_tolerance,
            price_within_tolerance=price_within_tolerance,
            line_match_status=line_status,
        )

    # -----------------------------------------------------------------------
    # Private — overall status determination
    # -----------------------------------------------------------------------

    def _determine_overall_status(
        self,
        line_results: List[LineMatchResult],
    ) -> MatchStatus:
        """Determine the overall match status from individual line results.

        Decision logic (evaluated in priority order):
            1. If *all* lines are ``FULL_MATCH`` → ``FULL_MATCH``.
            2. If *any* line is ``COMBINED_EXCEPTION`` → ``COMBINED_EXCEPTION``.
            3. If *both* ``PRICE_EXCEPTION`` and ``QUANTITY_EXCEPTION`` exist
               across different lines → ``COMBINED_EXCEPTION``.
            4. If only ``PRICE_EXCEPTION`` lines exist → ``PRICE_EXCEPTION``.
            5. If only ``QUANTITY_EXCEPTION`` lines exist → ``QUANTITY_EXCEPTION``.

        Args:
            line_results: Computed per-line match results.

        Returns:
            The single overall :class:`MatchStatus`.
        """
        if not line_results:
            # No lines to compare — treat as a degenerate full match.
            return MatchStatus.FULL_MATCH

        statuses = {lr.line_match_status for lr in line_results}

        # All lines matched.
        if statuses == {MatchStatus.FULL_MATCH}:
            return MatchStatus.FULL_MATCH

        # Explicit combined exception on any single line.
        if MatchStatus.COMBINED_EXCEPTION in statuses:
            return MatchStatus.COMBINED_EXCEPTION

        has_price_exception = MatchStatus.PRICE_EXCEPTION in statuses
        has_quantity_exception = MatchStatus.QUANTITY_EXCEPTION in statuses

        # Mixed exceptions across lines → combined.
        if has_price_exception and has_quantity_exception:
            return MatchStatus.COMBINED_EXCEPTION

        # Only price exceptions.
        if has_price_exception:
            return MatchStatus.PRICE_EXCEPTION

        # Only quantity exceptions.
        if has_quantity_exception:
            return MatchStatus.QUANTITY_EXCEPTION

        # Fallback — should not reach here, but safeguard.
        return MatchStatus.FULL_MATCH  # pragma: no cover

    # -----------------------------------------------------------------------
    # Private — variance aggregation
    # -----------------------------------------------------------------------

    def _aggregate_variances(
        self,
        line_results: List[LineMatchResult],
    ) -> Dict[str, Decimal]:
        """Sum line-level variances into aggregate totals.

        CRITICAL: Totals are the *sum* of per-line variances, NOT a global
        recalculation.  This follows AAP §0.1.2:
        "Variance Calculation: Line-Item Level (calculated per line, then
        summed)".

        Args:
            line_results: Computed per-line match results.

        Returns:
            Dictionary with keys ``total_price_variance``,
            ``total_quantity_variance``, ``total_extended_variance``.
        """
        total_price_variance = sum(
            (lr.price_variance for lr in line_results),
            _ZERO,
        )
        total_quantity_variance = sum(
            (lr.quantity_variance for lr in line_results),
            _ZERO,
        )
        total_extended_variance = sum(
            (lr.extended_amount_variance for lr in line_results),
            _ZERO,
        )
        return {
            "total_price_variance": total_price_variance,
            "total_quantity_variance": total_quantity_variance,
            "total_extended_variance": total_extended_variance,
        }

    # -----------------------------------------------------------------------
    # Private — utility helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_notes(
        status: MatchStatus,
        lines_matched: int,
        lines_with_exceptions: int,
    ) -> str:
        """Generate a human-readable note summarising the match result.

        Args:
            status:                Overall match status.
            lines_matched:         Number of lines within tolerance.
            lines_with_exceptions: Number of lines with exceptions.

        Returns:
            Summary string suitable for display and audit logging.
        """
        if status == MatchStatus.FULL_MATCH:
            return (
                f"All {lines_matched} line(s) matched within tolerance. "
                "No approval required."
            )
        return (
            f"{lines_with_exceptions} of "
            f"{lines_matched + lines_with_exceptions} "
            f"line(s) have exceptions (status: {status.value}). "
            "Approval required before payment processing."
        )
