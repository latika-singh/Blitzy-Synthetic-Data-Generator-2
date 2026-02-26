"""P2P-007: Round-Dollar Invoice discrepancy type implementation.

Modifies the invoice total amount to an exact round dollar figure (e.g.,
$1,000.00, $5,000.00, $10,000.00). Round-dollar invoices are a well-known
fraud indicator because legitimate business transactions rarely result in
perfectly round amounts.

Configurable Parameters:
    round_to (int): The rounding level — one of 100, 1000, or 10000.
        Default: 1000.
        The invoice amount is rounded to the nearest multiple of this value.

Detection Method: amount_pattern
    Detected by pattern analysis checking for amounts that are exact
    multiples of 100, 1000, or 10000 with zero cents.

Difficulty: medium
    Requires pattern-based analysis rather than simple rule checking.

Financial Impact: abs(rounded_amount - original_amount)

References:
    - AAP Section 0.5.1 Group 5: P2P-007 Round-Dollar Invoice
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - AAP Section 0.7.5: Discrepancy Injection Rules
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
__all__ = ["RoundDollarInvoice"]

# ---------------------------------------------------------------------------
# Allowed rounding levels for the round_to parameter
# ---------------------------------------------------------------------------
_ALLOWED_ROUND_VALUES: Tuple[int, ...] = (100, 1000, 10000)


class RoundDollarInvoice(BaseDiscrepancy):
    """P2P-007: Round-Dollar Invoice discrepancy (fraud indicator).

    Modifies the invoice ``total_amount`` to an exact multiple of a
    configurable rounding factor (100, 1000, or 10000).  The resulting
    amount always has zero cents (e.g., ``$3,000.00``, ``$10,000.00``).

    When the transaction contains line items, the line-level amounts are
    proportionally adjusted so they sum to the new rounded total, keeping
    the internal document structure consistent.

    Attributes:
        type_code: ``"P2P-007"``
        category: ``"p2p"``
        difficulty: ``"medium"``
        name: ``"Round-Dollar Invoice"``
        description: Human-readable description of this discrepancy type.
        detection_method: ``"amount_pattern"``
        PARAMETER_BOUNDS: Defines ``round_to`` parameter with allowed
            values ``[100, 1000, 10000]`` and default ``1000``.
    """

    # ------------------------------------------------------------------
    # Class-level attributes (MUST be set by all BaseDiscrepancy subclasses)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-007"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Round-Dollar Invoice"
    description: ClassVar[str] = (
        "Invoice with suspiciously round dollar amount (fraud indicator)"
    )
    detection_method: ClassVar[str] = "amount_pattern"

    # ------------------------------------------------------------------
    # Parameter bounds — consumed by _validate_params()
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "round_to": {
            "min": 100,
            "max": 10000,
            "type": "int",
            "default": 1000,
            "allowed_values": [100, 1000, 10000],
        },
    }

    # ------------------------------------------------------------------
    # inject() — core contract method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a round-dollar invoice discrepancy into the transaction.

        Rounds the invoice ``total_amount`` to the nearest multiple of the
        configured ``round_to`` value (100, 1000, or 10000).  The rounded
        amount always has zero cents, which is a classic fraud indicator.

        If the transaction contains ``lines`` or ``invoice_lines``, the
        individual line ``extended_amount`` values are proportionally
        adjusted so they sum to the new rounded total.

        Args:
            transaction: The original transaction data dictionary.
                Must contain ``"total_amount"`` (numeric or Decimal).
                Optionally contains ``"lines"`` or ``"invoice_lines"``
                with per-line ``"extended_amount"`` values.
            params: Injection parameters.  Recognised keys:

                - ``"round_to"`` (int): Rounding level — must be one of
                  100, 1000, or 10000.  Default 1000.

            rng: A seeded :class:`random.Random` instance for deterministic
                behaviour.  Used here only as a contract parameter — the
                rounding itself is deterministic.

        Returns:
            A tuple of:

            - **modified_transaction** — Deep copy with the
              ``total_amount`` replaced by the rounded value.
            - **ground_truth_data** — Dictionary with full ground truth
              schema fields for ``GroundTruthGenerator``.

        Raises:
            DiscrepancyInjectionError: If ``total_amount`` is missing or
                invalid, or if the ``round_to`` value is not an allowed
                value after validation.
        """
        try:
            # -------------------------------------------------------
            # Step 1: Deep copy the transaction to preserve original
            # -------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # -------------------------------------------------------
            # Step 2: Validate params against PARAMETER_BOUNDS
            # -------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # -------------------------------------------------------
            # Step 3: Extract and validate round_to
            # -------------------------------------------------------
            round_to: int = int(validated_params.get("round_to", 1000))

            # Enforce allowed_values — _validate_params only handles
            # min/max numeric bounds, so explicit check is needed
            if round_to not in _ALLOWED_ROUND_VALUES:
                # Snap to nearest allowed value
                round_to = min(
                    _ALLOWED_ROUND_VALUES,
                    key=lambda v: abs(v - round_to),
                )
                logger.debug(
                    "round_to_snapped_to_allowed",
                    service_name="transactions",
                    component=self.__class__.__name__,
                    type_code=self.type_code,
                    snapped_value=round_to,
                )

            # -------------------------------------------------------
            # Step 4: Extract original total_amount as Decimal
            # -------------------------------------------------------
            raw_amount = modified.get("total_amount")
            if raw_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required field 'total_amount' "
                    "for round-dollar invoice discrepancy",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                    },
                )

            original_amount: Decimal = Decimal(str(raw_amount))

            if original_amount <= Decimal("0"):
                raise DiscrepancyInjectionError(
                    f"total_amount must be positive, got {original_amount}",
                    details={
                        "discrepancy_type": self.type_code,
                        "total_amount": str(original_amount),
                    },
                )

            # -------------------------------------------------------
            # Step 5: Calculate the rounded amount
            # -------------------------------------------------------
            round_factor: Decimal = Decimal(str(round_to))

            # Round to the nearest multiple of round_factor
            rounded_amount: Decimal = (
                (original_amount / round_factor).quantize(
                    Decimal("1"), rounding=ROUND_HALF_UP
                )
                * round_factor
            )

            # Guard: avoid zero amounts (e.g., if original was < 50
            # and round_to is 100)
            if rounded_amount == Decimal("0"):
                rounded_amount = round_factor

            # Ensure the result ends in .00 cents (exact round dollar)
            rounded_amount = rounded_amount.quantize(Decimal("0.01"))

            # -------------------------------------------------------
            # Step 6: Update total_amount on modified transaction
            # -------------------------------------------------------
            modified["total_amount"] = rounded_amount

            # -------------------------------------------------------
            # Step 7: Proportionally adjust line amounts if present
            # -------------------------------------------------------
            lines_key: Optional[str] = None
            if "lines" in modified and isinstance(modified["lines"], list):
                lines_key = "lines"
            elif "invoice_lines" in modified and isinstance(
                modified["invoice_lines"], list
            ):
                lines_key = "invoice_lines"

            if lines_key is not None and len(modified[lines_key]) > 0:
                self._adjust_line_amounts(
                    modified[lines_key], original_amount, rounded_amount
                )

            # -------------------------------------------------------
            # Step 8: Calculate financial impact
            # -------------------------------------------------------
            financial_impact: Decimal = abs(
                rounded_amount - original_amount
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # -------------------------------------------------------
            # Step 9: Create ground truth data
            # -------------------------------------------------------
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=["total_amount"],
                original_values={"total_amount": str(original_amount)},
                modified_values={"total_amount": str(rounded_amount)},
                financial_impact=financial_impact,
                description=(
                    f"Invoice amount rounded to nearest ${round_to} "
                    f"(${original_amount} \u2192 ${rounded_amount})"
                ),
                extra_metadata={
                    "round_to": round_to,
                    "original_amount": str(original_amount),
                    "rounded_amount": str(rounded_amount),
                    "round_factor": str(round_factor),
                },
            )

            # -------------------------------------------------------
            # Step 10: Log the injection event
            # -------------------------------------------------------
            transaction_id: str = str(
                modified.get(
                    "transaction_id",
                    modified.get("invoice_id", "unknown"),
                )
            )

            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "simulation_id": modified.get("simulation_id"),
                    "trace_id": modified.get("trace_id"),
                },
            )

            logger.debug(
                "round_dollar_invoice_injected",
                service_name="transactions",
                component=self.__class__.__name__,
                type_code=self.type_code,
                transaction_id=transaction_id,
                round_to=round_to,
                original_amount=str(original_amount),
                rounded_amount=str(rounded_amount),
                financial_impact=str(financial_impact),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise our own exceptions unmodified
            raise
        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Round-dollar invoice injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Private helper: _adjust_line_amounts
    # ------------------------------------------------------------------
    @staticmethod
    def _adjust_line_amounts(
        lines: List[Dict[str, Any]],
        original_total: Decimal,
        new_total: Decimal,
    ) -> None:
        """Proportionally adjust line-item amounts to match the new total.

        Each line's ``extended_amount`` is scaled by the ratio
        ``new_total / original_total`` so the individual lines sum to the
        rounded total.  A residual rounding difference is absorbed by the
        last line to guarantee the sum equals ``new_total`` exactly.

        If a line has no ``extended_amount`` key (or ``original_total`` is
        zero), the line is left unchanged.

        Args:
            lines: List of line-item dictionaries to modify in place.
            original_total: The original invoice total before rounding.
            new_total: The new (rounded) invoice total.
        """
        if original_total == Decimal("0") or not lines:
            return

        ratio: Decimal = new_total / original_total
        running_sum: Decimal = Decimal("0")
        amount_key: str = "extended_amount"

        # Scale all lines except the last
        for i, line in enumerate(lines):
            if amount_key not in line:
                continue

            line_amount: Decimal = Decimal(str(line[amount_key]))

            if i < len(lines) - 1:
                adjusted: Decimal = (line_amount * ratio).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                line[amount_key] = adjusted
                running_sum += adjusted
            else:
                # Last line absorbs residual to ensure exact sum
                line[amount_key] = (new_total - running_sum).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
