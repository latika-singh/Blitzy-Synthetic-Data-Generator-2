"""P2P-002: Invoice/PO Price Mismatch discrepancy type implementation.

Modifies the invoice unit price on one or more line items to create a
variance that exceeds the standard ±5% three-way match price tolerance.
This forces the three-way matcher to flag the invoice for review.

Configurable Parameters:
    variance_percent (Decimal): Price variance percentage to apply.
        Bounds: 1–50%. Default: 10%.
        The actual price is adjusted by this percentage (over or under the PO price).
        Direction (over/under) is randomly selected via seeded RNG.

Detection Method: three_way_match
    Detected when the three-way matching engine compares PO unit price
    to invoice unit price and finds variance exceeding ±5% tolerance.

Difficulty: easy
    Three-way matching algorithms are standard in all ERP systems.

Financial Impact:
    Calculated as: abs(modified_price - original_price) × quantity per affected line.

References:
    - AAP Section 0.5.1 Group 5: P2P-002 Price Mismatch
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - Three-way match tolerance: ±5% price (AAP Section 0.5.1 Group 2)
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
__all__ = ["PriceMismatch"]


class PriceMismatch(BaseDiscrepancy):
    """P2P-002: Invoice/PO Price Mismatch discrepancy.

    Injects a price variance into one or more invoice line items so that
    the unit price deviates from the PO unit price by a configurable
    percentage.  The direction of the deviation (over- or under-pricing)
    is selected deterministically via the seeded RNG provided to
    :meth:`inject`.

    The three-way match tolerance for price is ±5% (AAP Section 0.5.1
    Group 2).  A ``variance_percent`` greater than 5 guarantees detection
    by the three-way matcher; values between 1 and 5 are valid but may
    fall within tolerance — this is by design for edge-case testing.

    Variance is calculated at the **line-item level** (per AAP: "Variance
    Calculation: Line-Item Level") — each affected line is independently
    adjusted, and the financial impact is summed across all affected lines.

    Class Attributes:
        type_code: ``"P2P-002"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"Invoice/PO Price Mismatch"``
        description: Human-readable description.
        detection_method: ``"three_way_match"``
        PARAMETER_BOUNDS: ``variance_percent`` with min=1, max=50,
            default=10 (all ``Decimal``).
    """

    # ------------------------------------------------------------------
    # Class-level attributes (override BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-002"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Invoice/PO Price Mismatch"
    description: ClassVar[str] = (
        "Invoice unit price differs from PO unit price by more than "
        "the ±5% three-way match tolerance"
    )
    detection_method: ClassVar[str] = "three_way_match"

    # ------------------------------------------------------------------
    # Parameter bounds — used by _validate_params()
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "variance_percent": {
            "min": Decimal("1"),
            "max": Decimal("50"),
            "type": "Decimal",
            "default": Decimal("10"),
        },
    }

    # ------------------------------------------------------------------
    # Recognised line-list field names in transaction dictionaries
    # ------------------------------------------------------------------
    _LINE_FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "lines",
        "invoice_lines",
        "line_items",
        "items",
    )

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an Invoice/PO Price Mismatch into *transaction*.

        Modifies one or more line items so that their ``unit_price``
        deviates from the original by ``variance_percent``.  The total
        amount is recalculated and a full ground truth record is
        returned.

        Args:
            transaction: Invoice/PO transaction data dictionary.  Must
                contain a list of line items (under one of the keys
                ``lines``, ``invoice_lines``, ``line_items``, or
                ``items``) with each line having at least ``unit_price``
                and ``quantity`` fields.
            params: Injection parameters.  Recognised key:
                ``variance_percent`` (``Decimal``, 1–50, default 10).
            rng: A seeded :class:`random.Random` instance for
                deterministic direction and line selection.

        Returns:
            A ``(modified_transaction, ground_truth_data)`` tuple where
            *modified_transaction* carries the adjusted prices and
            *ground_truth_data* follows the ``GroundTruthGenerator``
            schema.

        Raises:
            DiscrepancyInjectionError: If the transaction has no line
                items or a selected line lacks a ``unit_price`` field.
        """
        try:
            # 1. Deep-copy the transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate and fill defaults for injection parameters
            validated_params: Dict[str, Any] = self._validate_params(params)

            # 3. Extract the variance percentage as Decimal
            variance_percent: Decimal = Decimal(
                str(validated_params["variance_percent"])
            )

            # 4. Locate line items in the transaction
            lines: Optional[List[Dict[str, Any]]] = None
            line_field_name: str = ""
            for field_name in self._LINE_FIELD_NAMES:
                candidate = modified.get(field_name)
                if isinstance(candidate, list) and len(candidate) > 0:
                    lines = candidate
                    line_field_name = field_name
                    break

            if lines is None or len(lines) == 0:
                raise DiscrepancyInjectionError(
                    "Transaction has no line items to modify for "
                    "P2P-002 price mismatch",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "checked_fields": list(self._LINE_FIELD_NAMES),
                    },
                )

            # 5. Determine how many lines to affect (at least 1)
            #    For simplicity: single-line invoices → affect the one line.
            #    Multi-line invoices → affect 1 line via rng.choice.
            affected_indices: List[int] = [rng.randint(0, len(lines) - 1)]

            # 6. Determine direction: +1 = over-pricing, -1 = under-pricing
            direction: int = rng.choice([-1, 1])
            direction_label: str = "over" if direction == 1 else "under"

            # ----------------------------------------------------------
            # 7. Apply price variance to each affected line
            # ----------------------------------------------------------
            total_financial_impact: Decimal = Decimal("0")
            original_values: Dict[str, Any] = {}
            modified_values: Dict[str, Any] = {}
            affected_fields: List[str] = []

            for idx in affected_indices:
                line = lines[idx]
                line_key = f"line_{idx}"

                # Validate that unit_price exists on the line
                raw_price = line.get("unit_price")
                if raw_price is None:
                    raise DiscrepancyInjectionError(
                        f"Line {idx} has no 'unit_price' field for "
                        f"P2P-002 price mismatch injection",
                        details={
                            "discrepancy_type": self.type_code,
                            "transaction_id": str(
                                transaction.get("transaction_id", "unknown")
                            ),
                            "line_index": idx,
                            "available_keys": list(line.keys()),
                        },
                    )

                # Convert original price and quantity to Decimal
                original_price: Decimal = Decimal(str(raw_price))

                # Quantity defaults to 1 if absent (single-item line)
                raw_qty = line.get("quantity", 1)
                quantity: Decimal = Decimal(str(raw_qty))

                # Calculate the modified price:
                #   modified = original × (1 + direction × variance% / 100)
                variance_factor: Decimal = (
                    Decimal("1")
                    + Decimal(str(direction)) * variance_percent / Decimal("100")
                )
                modified_price: Decimal = (
                    original_price * variance_factor
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                # Ensure modified price is non-negative (floor at $0.01)
                if modified_price < Decimal("0.01"):
                    modified_price = Decimal("0.01")

                # Calculate extended amount for this line
                extended_amount: Decimal = (
                    modified_price * quantity
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                # Calculate per-line financial impact
                line_impact: Decimal = (
                    abs(modified_price - original_price) * quantity
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                total_financial_impact += line_impact

                # Record original and modified values for ground truth
                original_values[f"{line_key}_unit_price"] = str(original_price)
                modified_values[f"{line_key}_unit_price"] = str(modified_price)
                original_values[f"{line_key}_extended_amount"] = str(
                    Decimal(str(line.get("extended_amount", original_price * quantity)))
                    .quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                )
                modified_values[f"{line_key}_extended_amount"] = str(extended_amount)

                # Update the line in the modified transaction
                line["unit_price"] = modified_price
                line["extended_amount"] = extended_amount

                # Track affected fields for this line
                affected_fields.extend([
                    f"{line_key}.unit_price",
                    f"{line_key}.extended_amount",
                ])

            # ----------------------------------------------------------
            # 8. Recalculate transaction total_amount from all lines
            # ----------------------------------------------------------
            original_total_str = str(
                transaction.get("total_amount", Decimal("0"))
            )
            original_total: Decimal = Decimal(original_total_str).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

            new_total: Decimal = Decimal("0")
            for line in lines:
                line_ext = line.get("extended_amount")
                if line_ext is not None:
                    new_total += Decimal(str(line_ext))
                else:
                    # Fallback: unit_price × quantity
                    lp = Decimal(str(line.get("unit_price", 0)))
                    lq = Decimal(str(line.get("quantity", 1)))
                    new_total += lp * lq

            new_total = new_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            modified["total_amount"] = new_total

            original_values["total_amount"] = str(original_total)
            modified_values["total_amount"] = str(new_total)
            affected_fields.append("total_amount")

            # ----------------------------------------------------------
            # 9. Build ground truth data via base class helper
            # ----------------------------------------------------------
            num_affected: int = len(affected_indices)
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=total_financial_impact,
                description=(
                    f"Price mismatch of {variance_percent}% "
                    f"({direction_label}) on {num_affected} line(s)"
                ),
                extra_metadata={
                    "variance_percent": str(variance_percent),
                    "direction": direction_label,
                    "affected_line_count": num_affected,
                    "affected_line_indices": affected_indices,
                    "line_field_name": line_field_name,
                },
            )

            # ----------------------------------------------------------
            # 10. Log the injection event
            # ----------------------------------------------------------
            transaction_id: str = str(
                modified.get(
                    "transaction_id",
                    modified.get("invoice_id", "unknown"),
                )
            )
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=total_financial_impact,
                context={
                    "variance_percent": str(variance_percent),
                    "direction": direction_label,
                    "affected_line_count": num_affected,
                    "simulation_id": str(
                        transaction.get("simulation_id", "")
                    ),
                    "trace_id": str(
                        transaction.get("trace_id", "")
                    ),
                },
            )

            logger.debug(
                "price_mismatch_details",
                service_name="transactions",
                component="PriceMismatch",
                type_code=self.type_code,
                transaction_id=transaction_id,
                variance_percent=str(variance_percent),
                direction=direction_label,
                affected_lines=num_affected,
                financial_impact=str(total_financial_impact),
                original_total=str(original_total),
                new_total=str(new_total),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Unexpected error during P2P-002 price mismatch injection: "
                f"{exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
