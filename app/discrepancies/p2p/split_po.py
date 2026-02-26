"""P2P-012: Split PO to Avoid Approval discrepancy type implementation.

Simulates splitting a purchase order into multiple smaller orders, each
below the approval threshold, to circumvent the approval process. This
is a common procurement fraud technique where a $50,000 purchase is split
into three POs of ~$16,667 each to stay below the $25,000 controller
approval threshold.

PO Approval Thresholds (from config/workflows/approval_thresholds.yaml):
    - < $5,000: No approval required
    - $5,000–$25,000: purchasing_manager required
    - $25,000–$100,000: controller required
    - > $100,000: cfo required

Configurable Parameters:
    split_count (int): Number of POs to split into.
        Bounds: 2–5. Default: 3.
    threshold_amount (Decimal): The approval threshold being avoided.
        Default: Decimal("25000") (controller threshold).
        The PO is split so each part falls below this threshold.

Detection Method: pattern_analysis
    Detected by analyzing multiple POs to the same vendor within a short
    period where the combined total exceeds an approval threshold.

Difficulty: medium

Financial Impact: Full original PO amount (the combined total that should
    have required higher-level approval).

References:
    - AAP Section 0.5.1 Group 5: P2P-012 Split PO to Avoid Approval
    - config/workflows/approval_thresholds.yaml
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
__all__ = ["SplitPO"]

# ---------------------------------------------------------------------------
# Approval threshold role lookup — maps threshold amount boundaries to the
# required approver role.  Used when populating ``threshold_role`` in the
# ground truth metadata.  Values sourced from
# ``config/workflows/approval_thresholds.yaml`` and
# ``app/orchestration/approval_system.APPROVAL_THRESHOLDS``.
# ---------------------------------------------------------------------------
_THRESHOLD_ROLE_LOOKUP: List[Tuple[Decimal, Optional[str]]] = [
    (Decimal("100000"), "cfo"),
    (Decimal("25000"), "controller"),
    (Decimal("5000"), "purchasing_manager"),
    (Decimal("0"), None),  # No approval required below $5,000
]


def _resolve_threshold_role(threshold_amount: Decimal) -> str:
    """Determine the approver role being circumvented by the split.

    Walks the ``_THRESHOLD_ROLE_LOOKUP`` list (sorted from highest to
    lowest) and returns the role associated with the first threshold
    whose minimum boundary is less than or equal to *threshold_amount*.
    Falls back to ``"purchasing_manager"`` if no match (defensive).

    Args:
        threshold_amount: The approval threshold the split is designed
            to stay below.

    Returns:
        The name of the approver role being avoided, e.g.
        ``"controller"`` for the $25,000 threshold.
    """
    for boundary, role in _THRESHOLD_ROLE_LOOKUP:
        if threshold_amount >= boundary and role is not None:
            return role
    return "purchasing_manager"


class SplitPO(BaseDiscrepancy):
    """P2P-012: Split PO to avoid approval threshold discrepancy.

    Injects a split-PO discrepancy by dividing a purchase order's total
    amount into *split_count* smaller portions, each designed to fall
    below the configured *threshold_amount*.  The modified transaction
    represents the **first** split order; metadata describing the
    remaining N-1 splits is attached so downstream systems can
    reconstruct the complete split pattern.

    The ground truth record captures the full original amount, every
    individual split amount, the threshold being circumvented, and the
    approver role that should have been required.

    Attributes:
        type_code: ``"P2P-012"``
        category: ``"p2p"``
        difficulty: ``"medium"``
        name: ``"Split PO to Avoid Approval"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"pattern_analysis"``
        PARAMETER_BOUNDS: Split count (2–5) and threshold amount
            ($1,000–$100,000) bounds.
    """

    # ------------------------------------------------------------------
    # ClassVar attributes — MUST match schema exports exactly
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-012"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Split PO to Avoid Approval"
    description: ClassVar[str] = (
        "Purchase order split into multiple smaller orders to circumvent "
        "approval thresholds"
    )
    detection_method: ClassVar[str] = "pattern_analysis"

    # ------------------------------------------------------------------
    # Parameter bounds
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "split_count": {
            "min": 2,
            "max": 5,
            "type": "int",
            "default": 3,
        },
        "threshold_amount": {
            "min": Decimal("1000"),
            "max": Decimal("100000"),
            "type": "Decimal",
            "default": Decimal("25000"),
        },
    }

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a split-PO discrepancy into the transaction.

        Creates a modified transaction whose ``total_amount`` is reduced
        to the first split portion.  Metadata fields are added to
        describe the overall split pattern (``split_indicator``,
        ``split_count``, ``original_total``, ``split_amounts``,
        ``other_split_amounts``).

        Args:
            transaction: Original purchase order transaction dictionary.
                Expected keys include ``total_amount`` (Decimal or
                numeric), ``po_number`` or ``transaction_id`` (str),
                and optionally ``vendor_id`` (str).
            params: Injection parameters — ``split_count`` (int) and
                ``threshold_amount`` (Decimal).  Missing parameters are
                filled from :attr:`PARAMETER_BOUNDS` defaults by
                :meth:`_validate_params`.
            rng: Seeded :class:`random.Random` instance for
                deterministic variation in split amounts.

        Returns:
            A ``(modified_transaction, ground_truth_data)`` tuple.

        Raises:
            DiscrepancyInjectionError: If the transaction lacks a
                ``total_amount`` field or the computed split amounts are
                invalid (e.g. all zero or negative).
        """
        try:
            # 1. Deep-copy transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate and extract parameters
            validated_params = self._validate_params(params)
            split_count: int = int(validated_params.get(
                "split_count",
                self.PARAMETER_BOUNDS["split_count"]["default"],
            ))
            threshold_amount: Decimal = Decimal(str(validated_params.get(
                "threshold_amount",
                self.PARAMETER_BOUNDS["threshold_amount"]["default"],
            )))

            # 3. Extract original amount — MUST be present
            raw_amount = transaction.get("total_amount")
            if raw_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required 'total_amount' field "
                    "for P2P-012 Split PO injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id")
                            or transaction.get("po_number", "unknown")
                        ),
                    },
                )
            original_amount: Decimal = Decimal(str(raw_amount))

            # Retrieve identifiers for logging and ground truth
            transaction_id: str = str(
                transaction.get("transaction_id")
                or transaction.get("po_number", "unknown")
            )
            vendor_id: str = str(transaction.get("vendor_id", "unknown"))
            po_number: str = str(transaction.get("po_number", "unknown"))

            # 4. Calculate split amounts with small random variation
            split_amounts: List[Decimal] = self._calculate_split_amounts(
                original_amount=original_amount,
                split_count=split_count,
                rng=rng,
            )

            # Verify all splits are below the threshold (best-effort; if
            # the original amount is too small to split meaningfully the
            # split still proceeds for ground-truth purposes).
            splits_below_threshold: bool = all(
                amt < threshold_amount for amt in split_amounts
            )

            # 5. Modify the transaction — this PO becomes split #1
            modified["total_amount"] = split_amounts[0]
            modified["split_indicator"] = True
            modified["split_count"] = split_count
            modified["original_total"] = original_amount
            modified["split_amounts"] = [str(a) for a in split_amounts]
            modified["other_split_amounts"] = [
                str(a) for a in split_amounts[1:]
            ]
            modified["threshold_amount"] = str(threshold_amount)

            # Compute average split for description text
            split_avg: Decimal = (
                original_amount / Decimal(str(split_count))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # 6. Financial impact — the full original amount that should
            #    have required higher-level approval
            financial_impact: Decimal = original_amount

            # 7. Resolve the role being circumvented
            threshold_role: str = _resolve_threshold_role(threshold_amount)

            # 8. Build ground truth data
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[
                    "total_amount",
                    "split_indicator",
                    "split_count",
                ],
                original_values={
                    "total_amount": str(original_amount),
                    "split_indicator": False,
                    "split_count": 1,
                },
                modified_values={
                    "total_amount": str(split_amounts[0]),
                    "split_indicator": True,
                    "split_count": split_count,
                },
                financial_impact=financial_impact,
                description=(
                    f"PO for ${original_amount} split into {split_count} "
                    f"orders (each ~${split_avg}) to avoid "
                    f"${threshold_amount} threshold"
                ),
                extra_metadata={
                    "split_count": split_count,
                    "split_amounts": [str(a) for a in split_amounts],
                    "threshold_amount": str(threshold_amount),
                    "threshold_role": threshold_role,
                    "vendor_id": vendor_id,
                    "po_number": po_number,
                    "original_total": str(original_amount),
                    "splits_below_threshold": splits_below_threshold,
                },
            )

            # 9. Log the injection event
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "simulation_id": transaction.get("simulation_id"),
                    "trace_id": transaction.get("trace_id"),
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise injection errors as-is
            raise
        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Unexpected error during P2P-012 Split PO injection: "
                f"{exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id")
                        or transaction.get("po_number", "unknown")
                    ),
                    "error": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Private helper: split amount calculation
    # ------------------------------------------------------------------
    def _calculate_split_amounts(
        self,
        original_amount: Decimal,
        split_count: int,
        rng: random.Random,
    ) -> List[Decimal]:
        """Calculate individual split amounts that sum to the original total.

        Each split starts as ``original_amount / split_count``, then
        receives a small random variation (±5% of the base amount) via
        the seeded *rng*.  All amounts are rounded to cents
        (``Decimal("0.01")``, ``ROUND_HALF_UP``).  The **last** split
        is adjusted so that the splits sum exactly to *original_amount*
        (no penny leakage).

        Negative or zero splits are prevented by clamping each
        intermediate value to a minimum of ``Decimal("0.01")``.

        Args:
            original_amount: The total PO amount to split.
            split_count: Number of portions to divide into (2–5).
            rng: Seeded random number generator.

        Returns:
            A list of *split_count* :class:`Decimal` values whose sum
            equals *original_amount*.

        Raises:
            DiscrepancyInjectionError: If the original amount is too
                small to produce meaningful splits (all amounts would
                be ≤ 0).
        """
        # Guard: if the original amount is zero or negative, bail out
        if original_amount <= Decimal("0"):
            raise DiscrepancyInjectionError(
                "Cannot split a PO with zero or negative total_amount",
                details={
                    "discrepancy_type": self.type_code,
                    "original_amount": str(original_amount),
                    "split_count": split_count,
                },
            )

        _PENNY = Decimal("0.01")
        base_split: Decimal = (
            original_amount / Decimal(str(split_count))
        ).quantize(_PENNY, rounding=ROUND_HALF_UP)

        splits: List[Decimal] = []
        running_total: Decimal = Decimal("0")

        for i in range(split_count - 1):
            # Apply ±5% random variation to the base split
            variation_factor: Decimal = Decimal(
                str(rng.uniform(-0.05, 0.05))
            )
            split_value: Decimal = (
                base_split * (Decimal("1") + variation_factor)
            ).quantize(_PENNY, rounding=ROUND_HALF_UP)

            # Clamp to at least one cent
            if split_value < _PENNY:
                split_value = _PENNY

            splits.append(split_value)
            running_total += split_value

        # Last split absorbs any rounding remainder
        last_split: Decimal = (original_amount - running_total).quantize(
            _PENNY, rounding=ROUND_HALF_UP,
        )
        if last_split < _PENNY:
            last_split = _PENNY
        splits.append(last_split)

        # Sanity check — verify the sum equals original_amount.  If there
        # is a rounding discrepancy of more than one cent (extremely
        # unlikely), adjust the last split.
        total_splits: Decimal = sum(splits)
        diff: Decimal = original_amount - total_splits
        if abs(diff) > Decimal("0"):
            splits[-1] = (splits[-1] + diff).quantize(
                _PENNY, rounding=ROUND_HALF_UP,
            )

        return splits
