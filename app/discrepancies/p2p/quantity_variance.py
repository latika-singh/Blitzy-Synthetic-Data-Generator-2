"""P2P-003: Invoice/Receipt Quantity Variance discrepancy type implementation.

Modifies the invoice quantity on one or more line items to create a variance
that exceeds the standard ±2% three-way match quantity tolerance when compared
to the goods receipt quantity.

Configurable Parameters:
    variance_percent (Decimal): Quantity variance percentage to apply.
        Bounds: 1–30%. Default: 5%.
        Direction (over/under) is randomly selected via seeded RNG.

Detection Method: three_way_match
    Detected when the three-way matching engine compares receipt quantity
    to invoice quantity and finds variance exceeding ±2% tolerance.

Difficulty: easy

Financial Impact:
    Calculated as: abs(modified_qty - original_qty) × unit_price per affected line.

References:
    - AAP Section 0.5.1 Group 5: P2P-003 Quantity Variance
    - Three-way match tolerance: ±2% quantity (AAP Section 0.5.1 Group 2)
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["QuantityVariance"]

# ---------------------------------------------------------------------------
# Constants for quantity field resolution
# ---------------------------------------------------------------------------
_QUANTITY_FIELD_NAMES: Tuple[str, ...] = (
    "quantity",
    "invoice_quantity",
    "invoiced_quantity",
    "qty",
    "invoice_qty",
    "billed_quantity",
)

_UNIT_PRICE_FIELD_NAMES: Tuple[str, ...] = (
    "unit_price",
    "price",
    "unit_cost",
    "price_per_unit",
)

_EXTENDED_AMOUNT_FIELD_NAMES: Tuple[str, ...] = (
    "extended_amount",
    "line_amount",
    "line_total",
    "amount",
    "total",
)

_LINES_FIELD_NAMES: Tuple[str, ...] = (
    "lines",
    "invoice_lines",
    "line_items",
    "items",
    "detail_lines",
)

_TOTAL_AMOUNT_FIELD_NAMES: Tuple[str, ...] = (
    "total_amount",
    "invoice_total",
    "invoice_amount",
    "amount",
    "grand_total",
)

# Cent precision for financial rounding
_CENTS = Decimal("0.01")

# Precision for quantity rounding (up to 4 decimal places)
_QTY_PRECISION = Decimal("0.0001")


class QuantityVariance(BaseDiscrepancy):
    """P2P-003: Invoice/Receipt Quantity Variance discrepancy.

    Modifies the invoice quantity on one or more line items to produce a
    variance that exceeds the standard ±2% three-way match quantity
    tolerance.  The direction of variance (over-invoice or under-invoice)
    and the specific line(s) affected are determined by the seeded RNG
    for deterministic reproducibility.

    Class Attributes:
        type_code: ``"P2P-003"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"Invoice/Receipt Quantity Variance"``
        description: Human-readable explanation of the discrepancy.
        detection_method: ``"three_way_match"``
        PARAMETER_BOUNDS: Configurable ``variance_percent`` (1–30%,
            default 5%).
    """

    # ------------------------------------------------------------------
    # ClassVar attributes — REQUIRED overrides from BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-003"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Invoice/Receipt Quantity Variance"
    description: ClassVar[str] = (
        "Invoice quantity differs from goods receipt quantity by more than "
        "the ±2% three-way match tolerance"
    )
    detection_method: ClassVar[str] = "three_way_match"

    # ------------------------------------------------------------------
    # Parameter bounds for variance_percent
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "variance_percent": {
            "min": Decimal("1"),
            "max": Decimal("30"),
            "type": "Decimal",
            "default": Decimal("5"),
        },
    }

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a quantity variance discrepancy into the transaction.

        Modifies the quantity on one or more invoice line items so that
        the variance exceeds the ±2% three-way match tolerance.  For
        each affected line the ``extended_amount`` is recalculated and
        the transaction ``total_amount`` is updated accordingly.

        Args:
            transaction: The original transaction data dictionary.  Must
                contain a ``lines`` (or equivalent) key with at least one
                line item that has a ``quantity`` and ``unit_price`` field.
            params: Injection parameters.  Recognised key:

                - ``variance_percent`` (Decimal | int | float): Percentage
                  by which to vary the quantity.  Validated against
                  :attr:`PARAMETER_BOUNDS` (1–30%).

            rng: A seeded :class:`random.Random` instance for
                deterministic behaviour.  Used for direction selection
                and line selection.

        Returns:
            A 2-tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction has no line
                items, if a line has no quantity field, or if the
                modified quantity would be ≤ 0.
        """
        # 1. Deep copy to preserve the original transaction data
        modified = self._copy_transaction(transaction)

        # 2. Validate params against bounds (auto-adjust / defaults)
        validated_params = self._validate_params(params)

        # 3. Extract the validated variance_percent as Decimal
        variance_percent = Decimal(
            str(validated_params.get("variance_percent", Decimal("5")))
        )

        # 4. Locate the lines collection in the transaction
        lines, lines_key = _resolve_lines(modified)
        if not lines:
            raise DiscrepancyInjectionError(
                "Transaction has no line items for quantity variance injection",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(modified.get("transaction_id", "unknown")),
                },
            )

        # 5. Filter to lines that have a usable quantity field
        eligible_indices: List[int] = []
        for idx, line in enumerate(lines):
            qty_field = _resolve_field(line, _QUANTITY_FIELD_NAMES)
            price_field = _resolve_field(line, _UNIT_PRICE_FIELD_NAMES)
            if qty_field is not None and price_field is not None:
                eligible_indices.append(idx)

        if not eligible_indices:
            raise DiscrepancyInjectionError(
                "No line items have both quantity and unit_price fields",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(modified.get("transaction_id", "unknown")),
                    "line_count": len(lines),
                },
            )

        # 6. Select line(s) to modify — at least one, at most all eligible
        num_to_modify = rng.randint(1, max(1, len(eligible_indices)))
        selected_indices = (
            rng.sample(eligible_indices, num_to_modify)
            if num_to_modify < len(eligible_indices)
            else list(eligible_indices)
        )

        # 7. Apply quantity variance to each selected line
        original_values: Dict[str, Any] = {}
        modified_values: Dict[str, Any] = {}
        total_financial_impact = Decimal("0")
        directions_applied: List[str] = []

        for idx in selected_indices:
            line = lines[idx]
            line_key_prefix = f"line_{idx}"

            # Resolve field names for this line
            qty_field_name = _resolve_field(line, _QUANTITY_FIELD_NAMES)
            price_field_name = _resolve_field(line, _UNIT_PRICE_FIELD_NAMES)
            ext_amt_field_name = _resolve_field(line, _EXTENDED_AMOUNT_FIELD_NAMES)

            # These are guaranteed non-None by the eligible_indices filter
            assert qty_field_name is not None
            assert price_field_name is not None

            original_qty = Decimal(str(line[qty_field_name]))
            unit_price = Decimal(str(line[price_field_name]))

            if original_qty <= Decimal("0"):
                # Skip lines with zero or negative quantity — cannot
                # apply meaningful variance
                continue

            # Determine direction: +1 = over-invoice, -1 = under-invoice
            direction = rng.choice([-1, 1])
            direction_label = "over" if direction == 1 else "under"
            directions_applied.append(direction_label)

            # Calculate modified quantity
            variance_factor = Decimal("1") + (
                Decimal(str(direction)) * variance_percent / Decimal("100")
            )
            raw_modified_qty = original_qty * variance_factor

            # Round quantity: integer for whole units, 4 dp for fractional
            if original_qty == original_qty.to_integral_value():
                # Original was a whole number — round to nearest integer
                # but keep as Decimal for consistency
                modified_qty = raw_modified_qty.quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
                # Ensure we actually have a difference; if rounding
                # collapsed the variance, nudge by at least 1 unit
                if modified_qty == original_qty:
                    modified_qty = original_qty + Decimal(str(direction))
            else:
                modified_qty = raw_modified_qty.quantize(
                    _QTY_PRECISION, rounding=ROUND_HALF_UP
                )

            # Ensure modified quantity > 0 (no negative or zero quantities)
            if modified_qty <= Decimal("0"):
                raise DiscrepancyInjectionError(
                    f"Modified quantity would be ≤ 0 ({modified_qty}) on line {idx}",
                    details={
                        "discrepancy_type": self.type_code,
                        "line_index": idx,
                        "original_qty": str(original_qty),
                        "variance_percent": str(variance_percent),
                        "direction": direction_label,
                        "calculated_qty": str(modified_qty),
                    },
                )

            # Update line quantity
            line[qty_field_name] = modified_qty

            # Recalculate extended amount if the field exists
            new_extended_amount = (modified_qty * unit_price).quantize(
                _CENTS, rounding=ROUND_HALF_UP
            )
            if ext_amt_field_name is not None:
                original_ext_amt = Decimal(str(line[ext_amt_field_name]))
                line[ext_amt_field_name] = new_extended_amount
                original_values[f"{line_key_prefix}_extended_amount"] = str(
                    original_ext_amt
                )
                modified_values[f"{line_key_prefix}_extended_amount"] = str(
                    new_extended_amount
                )

            # Track original / modified values for ground truth
            original_values[f"{line_key_prefix}_quantity"] = str(original_qty)
            modified_values[f"{line_key_prefix}_quantity"] = str(modified_qty)

            # Calculate per-line financial impact: |qty_diff| × unit_price
            qty_diff = abs(modified_qty - original_qty)
            line_impact = (qty_diff * unit_price).quantize(
                _CENTS, rounding=ROUND_HALF_UP
            )
            total_financial_impact += line_impact

        # 8. Recalculate the transaction total_amount from all lines
        original_total = _get_total_amount(transaction)
        new_total = _recalculate_total(lines, modified)
        total_field_name = _resolve_field(modified, _TOTAL_AMOUNT_FIELD_NAMES)
        if total_field_name is not None:
            original_values["total_amount"] = str(original_total)
            modified_values["total_amount"] = str(new_total)
            modified[total_field_name] = new_total

        # 9. Build the list of affected fields
        affected_fields: List[str] = ["quantity", "extended_amount", "total_amount"]

        # 10. Determine primary direction label for metadata
        primary_direction = (
            "over"
            if directions_applied.count("over") >= directions_applied.count("under")
            else "under"
        )

        # 11. Build human-readable description
        line_count = len(selected_indices)
        description = (
            f"Quantity variance of {variance_percent}% ({primary_direction}) "
            f"applied to {line_count} line(s), exceeding ±2% tolerance"
        )

        # 12. Create ground truth data via base class helper
        ground_truth = self._create_ground_truth_data(
            affected_fields=affected_fields,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=total_financial_impact,
            description=description,
            extra_metadata={
                "variance_percent": str(variance_percent),
                "direction": primary_direction,
                "directions_per_line": directions_applied,
                "affected_line_count": line_count,
                "affected_line_indices": selected_indices,
            },
        )

        # 13. Log the injection event
        transaction_id = str(
            modified.get("transaction_id", modified.get("invoice_id", "unknown"))
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=total_financial_impact,
            context={
                "simulation_id": modified.get("simulation_id"),
                "trace_id": modified.get("trace_id"),
            },
        )

        logger.debug(
            "quantity_variance_injected",
            service_name="transactions",
            component="QuantityVariance",
            type_code=self.type_code,
            transaction_id=transaction_id,
            variance_percent=str(variance_percent),
            direction=primary_direction,
            affected_lines=line_count,
            financial_impact=str(total_financial_impact),
        )

        return modified, ground_truth


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _resolve_field(
    data: Dict[str, Any], candidates: Tuple[str, ...]
) -> Optional[str]:
    """Return the first field name from *candidates* that exists in *data*.

    Args:
        data: The dictionary to search.
        candidates: Ordered tuple of candidate field names.

    Returns:
        The first matching field name, or ``None`` if none match.
    """
    for name in candidates:
        if name in data:
            return name
    return None


def _resolve_lines(
    transaction: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Locate the line-items collection within the transaction.

    Tries several common field names (``lines``, ``invoice_lines``,
    ``line_items``, ``items``, ``detail_lines``) and returns the first
    match along with the field key.

    Args:
        transaction: The transaction dictionary.

    Returns:
        A tuple of ``(lines_list, field_key)`` where *lines_list* is the
        list of line-item dicts and *field_key* is the key used.  If no
        lines field is found, returns ``([], None)``.
    """
    for name in _LINES_FIELD_NAMES:
        if name in transaction and isinstance(transaction[name], list):
            return transaction[name], name
    return [], None


def _get_total_amount(transaction: Dict[str, Any]) -> Decimal:
    """Extract the total amount from a transaction dictionary.

    Searches several common field names and returns the value as a
    :class:`Decimal`.  Defaults to ``Decimal("0")`` if no total field
    is found.

    Args:
        transaction: The transaction dictionary.

    Returns:
        The total amount as :class:`Decimal`.
    """
    field_name = _resolve_field(transaction, _TOTAL_AMOUNT_FIELD_NAMES)
    if field_name is not None:
        try:
            return Decimal(str(transaction[field_name]))
        except Exception:
            return Decimal("0")
    return Decimal("0")


def _recalculate_total(
    lines: List[Dict[str, Any]], transaction: Dict[str, Any]
) -> Decimal:
    """Recalculate the transaction total from line extended amounts.

    Sums the ``extended_amount`` (or equivalent) across all lines.  If a
    line does not have an extended amount field, it attempts to compute
    ``quantity × unit_price`` on the fly.  Falls back to returning the
    existing transaction total if computation fails.

    Args:
        lines: The list of line-item dictionaries.
        transaction: The full transaction dictionary (used for fallback).

    Returns:
        The recalculated total as :class:`Decimal`.
    """
    total = Decimal("0")
    for line in lines:
        ext_field = _resolve_field(line, _EXTENDED_AMOUNT_FIELD_NAMES)
        if ext_field is not None:
            try:
                total += Decimal(str(line[ext_field]))
            except Exception:
                # Fall back to qty × price if ext amount is not numeric
                total += _compute_line_amount(line)
        else:
            total += _compute_line_amount(line)
    return total.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _compute_line_amount(line: Dict[str, Any]) -> Decimal:
    """Compute a line amount from quantity × unit_price.

    Args:
        line: A single line-item dictionary.

    Returns:
        The computed amount as :class:`Decimal`, or ``Decimal("0")`` if
        the required fields are missing or non-numeric.
    """
    qty_field = _resolve_field(line, _QUANTITY_FIELD_NAMES)
    price_field = _resolve_field(line, _UNIT_PRICE_FIELD_NAMES)
    if qty_field is not None and price_field is not None:
        try:
            qty = Decimal(str(line[qty_field]))
            price = Decimal(str(line[price_field]))
            return (qty * price).quantize(_CENTS, rounding=ROUND_HALF_UP)
        except Exception:
            return Decimal("0")
    return Decimal("0")
