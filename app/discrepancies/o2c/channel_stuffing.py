"""O2C-009: Channel Stuffing discrepancy type.

Simulates channel stuffing — the practice of shipping excessively large
quantities of product to customers near period-end to artificially inflate
current-period revenue. Products are often shipped with implicit or explicit
return rights.

Channel stuffing indicators:
- Unusually large order volumes near period close dates
- Significant increase in shipments in the last week of a fiscal period
- Order quantities significantly exceeding historical averages for the customer
- High return rates in the subsequent period

Detection Method: ``pattern_analysis`` — analyzing order volumes and shipment
timing relative to period-end dates and comparing against historical baselines.

Difficulty: ``medium`` — requires temporal pattern analysis and historical
volume comparison.

Parameter Bounds:
    volume_multiplier: float, range [2.0, 5.0] — multiplier applied to the
        normal order quantity to simulate stuffed volume.

References:
    - AAP Section 0.5.1 Group 5: O2C-009 Channel Stuffing
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Financial Integrity Rules
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, ClassVar, Dict, List, Tuple

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Decimal constants for financial calculations (AAP §0.7.2)
# ---------------------------------------------------------------------------
_TWO_PLACES = Decimal("0.01")
_ZERO = Decimal("0")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["ChannelStuffing"]


class ChannelStuffing(BaseDiscrepancy):
    """O2C-009: Channel Stuffing discrepancy implementation.

    Simulates shipping excessively large quantities of product to customers
    near period-end to artificially inflate current-period revenue.  The
    ``inject()`` method inflates line-item quantities by a configurable
    ``volume_multiplier`` (default randomly chosen within [2.0, 5.0]),
    recalculates line-level and order-level totals using :class:`Decimal`
    arithmetic, and sets period-end indicator flags.

    The financial impact is computed as the excess revenue — the difference
    between the inflated order total and the original order total.

    Class Attributes:
        type_code: ``"O2C-009"``
        category: ``"o2c"``
        difficulty: ``"medium"``
        name: ``"Channel Stuffing"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"pattern_analysis"``
        PARAMETER_BOUNDS: ``{"volume_multiplier": {"min": 2.0, "max": 5.0, "type": "float"}}``
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-009"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Channel Stuffing"
    description: ClassVar[str] = (
        "Excessive product shipped to customers near period-end to inflate "
        "current-period revenue, often with implicit right-of-return."
    )
    detection_method: ClassVar[str] = "pattern_analysis"

    # ------------------------------------------------------------------
    # Parameter bounds (AAP §0.7.5 — auto-adjust to bounds if enabled)
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "volume_multiplier": {"min": 2.0, "max": 5.0, "type": "float"},
    }

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a channel stuffing discrepancy into the transaction.

        Inflates order/shipment line quantities by ``volume_multiplier``,
        recalculates all line-level amounts and the order total, sets
        period-end indicator flags, and produces a ground truth record
        documenting the injection.

        Args:
            transaction: Original transaction data dictionary.  Expected
                keys include ``"transaction_id"`` (or ``"order_id"``),
                ``"lines"`` (list of line-item dicts with ``"quantity"``
                and ``"unit_price"`` keys), and ``"total_amount"`` (or
                ``"order_total"``).
            params: Injection parameters.  Recognised key:

                - ``"volume_multiplier"`` (float) — multiplier for
                  line quantities.  If absent, a random value in
                  [2.0, 5.0] is generated via *rng*.

            rng: Seeded :class:`random.Random` instance for deterministic
                reproducibility.  CRITICAL: MUST use this RNG for all
                random operations — NEVER ``random.random()`` etc.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If injection fails due to missing
                required transaction fields or calculation errors.
        """
        try:
            # ----------------------------------------------------------
            # 1. Deep copy transaction to preserve original
            # ----------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ----------------------------------------------------------
            # 2. Validate params against PARAMETER_BOUNDS
            # ----------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # ----------------------------------------------------------
            # 3. Extract volume_multiplier (generate default if absent)
            # ----------------------------------------------------------
            raw_multiplier: float = validated_params.get(
                "volume_multiplier",
                rng.uniform(2.0, 5.0),
            )
            # Convert to Decimal for all financial calculations
            multiplier: Decimal = Decimal(str(raw_multiplier))

            # ----------------------------------------------------------
            # 4. Extract transaction identifier for logging
            # ----------------------------------------------------------
            transaction_id: str = str(
                modified.get("transaction_id")
                or modified.get("order_id")
                or modified.get("sales_order_id")
                or modified.get("shipment_id")
                or "UNKNOWN"
            )

            # ----------------------------------------------------------
            # 5. Get original quantities and amounts from lines
            # ----------------------------------------------------------
            lines: List[Dict[str, Any]] = modified.get("lines", [])
            if not lines:
                raise DiscrepancyInjectionError(
                    "Channel stuffing injection requires 'lines' in transaction data",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": transaction_id,
                        "reason": "missing_lines",
                    },
                )

            original_quantities: List[int] = []
            new_quantities: List[int] = []
            original_total: Decimal = _ZERO
            new_total: Decimal = _ZERO

            # ----------------------------------------------------------
            # 6–7. Inflate quantities and recalculate line amounts
            # ----------------------------------------------------------
            for line in lines:
                # Extract original quantity (int for physical items)
                orig_qty_raw = line.get("quantity", 0)
                orig_qty: int = int(orig_qty_raw) if orig_qty_raw else 0
                original_quantities.append(orig_qty)

                # Extract unit price as Decimal
                unit_price_raw = line.get("unit_price", "0")
                unit_price: Decimal = Decimal(str(unit_price_raw))

                # Calculate original line amount
                orig_line_amount: Decimal = (
                    Decimal(str(orig_qty)) * unit_price
                ).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
                original_total += orig_line_amount

                # Apply volume multiplier — round to int for physical items
                inflated_qty: int = max(
                    1,
                    int(
                        (Decimal(str(orig_qty)) * multiplier).quantize(
                            Decimal("1"), rounding=ROUND_HALF_UP
                        )
                    ),
                )
                new_quantities.append(inflated_qty)

                # Recalculate line amount with inflated quantity
                new_line_amount: Decimal = (
                    Decimal(str(inflated_qty)) * unit_price
                ).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
                new_total += new_line_amount

                # Update line in modified transaction
                line["quantity"] = inflated_qty
                line["line_amount"] = str(new_line_amount)
                line["original_quantity"] = orig_qty

            # Quantize totals to 2 decimal places
            original_total = original_total.quantize(
                _TWO_PLACES, rounding=ROUND_HALF_UP
            )
            new_total = new_total.quantize(
                _TWO_PLACES, rounding=ROUND_HALF_UP
            )

            # Update order-level total in modified transaction
            modified["total_amount"] = str(new_total)
            if "order_total" in modified:
                modified["order_total"] = str(new_total)

            # ----------------------------------------------------------
            # 8. Set period-end indicator flags
            # ----------------------------------------------------------
            modified["period_end_shipment"] = True
            modified["implicit_return_right"] = True
            modified["volume_unusual"] = True
            modified["channel_stuffing_indicator"] = True
            modified["volume_multiplier_applied"] = str(multiplier)

            # ----------------------------------------------------------
            # 9. Calculate financial impact (excess revenue)
            # ----------------------------------------------------------
            financial_impact: Decimal = (new_total - original_total).quantize(
                _TWO_PLACES, rounding=ROUND_HALF_UP
            )

            # ----------------------------------------------------------
            # 10. Build ground truth data via base class helper
            # ----------------------------------------------------------
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[
                    "quantity",
                    "total_amount",
                    "period_end_shipment",
                    "volume_unusual",
                ],
                original_values={
                    "quantity": original_quantities,
                    "total_amount": str(original_total),
                },
                modified_values={
                    "quantity": new_quantities,
                    "total_amount": str(new_total),
                    "volume_multiplier": str(multiplier),
                },
                financial_impact=financial_impact,
                description=(
                    f"Channel stuffing \u2014 volume inflated by "
                    f"{multiplier}x near period-end"
                ),
                extra_metadata={
                    "volume_multiplier": str(multiplier),
                    "original_total": str(original_total),
                    "inflated_total": str(new_total),
                    "line_count": len(lines),
                    "original_quantities": original_quantities,
                    "inflated_quantities": new_quantities,
                    "period_end_shipment": True,
                    "implicit_return_right": True,
                },
            )

            # ----------------------------------------------------------
            # 11. Log the injection event
            # ----------------------------------------------------------
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "simulation_id": modified.get("simulation_id"),
                    "trace_id": modified.get("trace_id"),
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors without wrapping
            raise
        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Channel stuffing injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id")
                        or transaction.get("order_id")
                        or "UNKNOWN"
                    ),
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                },
            ) from exc
