"""P2P-014: Unusual Vendor Concentration discrepancy type implementation.

Modifies transaction metadata to indicate disproportionate spend concentration
to a single vendor. While Pareto distributions naturally create some vendor
concentration (top 20% receive ~80% of spend), this discrepancy creates
concentration beyond normal statistical patterns, which may indicate kickback
arrangements, conflicts of interest, or vendor favoritism.

Configurable Parameters:
    concentration_pct (Decimal): The spend concentration ratio to simulate.
        Bounds: 0.3–0.80 (30%–80% of total spend to one vendor).
        Default: 0.6 (60%).
        Normal healthy threshold is ~0.3 for top vendor.

Detection Method: statistical_analysis
    Detected by analyzing spend distribution across vendors and
    comparing to expected Pareto distribution patterns.

Difficulty: medium
    Requires statistical analysis of spend patterns across vendors.

Financial Impact: Transaction amount × excess concentration factor.

References:
    - AAP Section 0.5.1 Group 5: P2P-014 Unusual Vendor Concentration
    - app/statistical/selection_models.py: Pareto 80/20 distribution reference
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
__all__ = ["VendorConcentration"]

# ---------------------------------------------------------------------------
# Normal concentration threshold constant — the typical upper bound for the
# highest-spending vendor in a healthy Pareto 80/20 procurement distribution.
# Concentrations above this level are flagged as anomalous.
# ---------------------------------------------------------------------------
_NORMAL_CONCENTRATION_THRESHOLD = Decimal("0.3")


class VendorConcentration(BaseDiscrepancy):
    """P2P-014: Unusual Vendor Concentration discrepancy.

    Simulates disproportionate spend concentration to a single vendor that
    exceeds what a healthy Pareto (80/20) distribution would predict.  The
    injected discrepancy marks the transaction as part of a concentrated
    vendor pattern and optionally inflates the transaction amount to
    represent a larger share of category spend.

    This type of anomaly is commonly associated with:

    - **Kickback arrangements** — a buyer directs volume to a vendor in
      exchange for personal benefit.
    - **Conflicts of interest** — the vendor is connected to an insider.
    - **Vendor favoritism** — preferential treatment without competitive
      bidding.

    The ``concentration_pct`` parameter controls the simulated
    concentration ratio (default 60%).  Normal healthy top-vendor
    concentration is approximately 30%.

    Attributes:
        type_code: ``"P2P-014"``
        category: ``"p2p"``
        difficulty: ``"medium"``
        name: ``"Unusual Vendor Concentration"``
        description: Human-readable summary of this discrepancy type.
        detection_method: ``"statistical_analysis"``
        PARAMETER_BOUNDS: Validation bounds for ``concentration_pct``
            (min 0.3, max 0.80, default 0.6).
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-014"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Unusual Vendor Concentration"
    description: ClassVar[str] = (
        "Disproportionate spend concentration to a single vendor"
    )
    detection_method: ClassVar[str] = "statistical_analysis"

    # ------------------------------------------------------------------
    # Parameter bounds — consumed by _validate_params()
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "concentration_pct": {
            "min": Decimal("0.3"),
            "max": Decimal("0.80"),
            "type": "Decimal",
            "default": Decimal("0.6"),
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
        """Inject an unusual vendor concentration discrepancy.

        Marks the transaction as part of a concentrated vendor spending
        pattern, inflates the transaction amount to represent a larger
        share of category spend, and produces a ground truth record for
        the ``GroundTruthGenerator``.

        Algorithm:

        1. Deep-copy the transaction to preserve the original.
        2. Validate and extract the ``concentration_pct`` parameter.
        3. Retrieve the vendor identifier and original total amount.
        4. Compute an inflation factor using ``rng.uniform(0.2, 0.5)`` to
           deterministically inflate the transaction amount, simulating
           the outsized share of spend directed to this vendor.
        5. Set concentration metadata flags on the modified transaction.
        6. Calculate the financial impact as the excess above the normal
           concentration threshold (``original_amount × (threshold − 0.3)``).
        7. Build the ground truth record and log the injection event.

        Args:
            transaction: Original transaction data dictionary.  Expected
                keys include ``vendor_id`` (or ``vendor``), ``total_amount``
                (or ``amount``), and ``transaction_id`` (or ``id``).
            params: Injection parameters.  Recognised key:
                ``concentration_pct`` (Decimal in [0.3, 0.80]).
            rng: Seeded :class:`random.Random` instance for deterministic
                behaviour.  **CRITICAL**: Only this instance may be used
                for random operations — never the module-level RNG.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction lacks a
                ``vendor_id`` or ``total_amount`` field.
        """
        try:
            # ---------------------------------------------------------- #
            # Step 1: Deep copy the transaction                          #
            # ---------------------------------------------------------- #
            modified = self._copy_transaction(transaction)

            # ---------------------------------------------------------- #
            # Step 2: Validate parameters and extract threshold          #
            # ---------------------------------------------------------- #
            validated_params = self._validate_params(params)
            concentration_pct = Decimal(
                str(validated_params.get("concentration_pct", Decimal("0.6")))
            )

            # ---------------------------------------------------------- #
            # Step 3: Extract vendor identifier                          #
            # ---------------------------------------------------------- #
            vendor_id = (
                modified.get("vendor_id")
                or modified.get("vendor", {}).get("id")
                or modified.get("vendor", {}).get("vendor_id")
                or modified.get("vendor_name")
            )
            if not vendor_id:
                raise DiscrepancyInjectionError(
                    "Transaction missing vendor_id for P2P-014 injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_keys": list(modified.keys()),
                    },
                )

            # ---------------------------------------------------------- #
            # Step 4: Extract and validate the original total amount     #
            # ---------------------------------------------------------- #
            raw_amount = modified.get("total_amount") or modified.get("amount")
            if raw_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing total_amount for P2P-014 injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "vendor_id": str(vendor_id),
                        "transaction_keys": list(modified.keys()),
                    },
                )
            original_amount = Decimal(str(raw_amount))

            # ---------------------------------------------------------- #
            # Step 5: Compute inflation factor and inflated amount       #
            # ---------------------------------------------------------- #
            # The inflation factor simulates a larger-than-normal share
            # of spend being directed to this vendor.  The uniform draw
            # is bounded to [0.2, 0.5] for realistic inflation.
            inflation_factor = Decimal(str(rng.uniform(0.2, 0.5)))
            inflated_amount = (
                original_amount * (Decimal("1") + inflation_factor)
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Ensure the inflated amount is strictly positive
            if inflated_amount <= Decimal("0"):
                inflated_amount = original_amount  # Safeguard: no change

            # ---------------------------------------------------------- #
            # Step 6: Apply concentration metadata to the transaction    #
            # ---------------------------------------------------------- #
            modified["total_amount"] = inflated_amount
            modified["vendor_concentration_flag"] = True
            modified["vendor_concentration_ratio"] = str(concentration_pct)
            modified["vendor_concentration_category"] = "high_concentration"
            modified["vendor_concentration_normal_threshold"] = str(
                _NORMAL_CONCENTRATION_THRESHOLD
            )
            modified["vendor_concentration_inflation_factor"] = str(inflation_factor)

            # ---------------------------------------------------------- #
            # Step 7: Calculate financial impact                         #
            # ---------------------------------------------------------- #
            # The financial impact represents the excess amount beyond
            # what a normal concentration would produce.  We use the
            # threshold delta multiplied by the original amount as a
            # proxy, plus the absolute inflation difference.
            excess_concentration = concentration_pct - _NORMAL_CONCENTRATION_THRESHOLD
            # Ensure non-negative excess (threshold is always >= 0.3 per bounds)
            if excess_concentration < Decimal("0"):
                excess_concentration = Decimal("0")

            financial_impact = (
                original_amount * excess_concentration
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Fall back to the actual inflated difference if impact is zero
            if financial_impact == Decimal("0"):
                financial_impact = abs(inflated_amount - original_amount).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            # ---------------------------------------------------------- #
            # Step 8: Extract transaction identifier                     #
            # ---------------------------------------------------------- #
            transaction_id = str(
                modified.get("transaction_id")
                or modified.get("id")
                or modified.get("po_number")
                or modified.get("invoice_number")
                or "unknown"
            )

            # ---------------------------------------------------------- #
            # Step 9: Build ground truth record                          #
            # ---------------------------------------------------------- #
            concentration_display_pct = (
                concentration_pct * Decimal("100")
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

            ground_truth = self._create_ground_truth_data(
                affected_fields=[
                    "total_amount",
                    "vendor_concentration_flag",
                    "vendor_concentration_ratio",
                ],
                original_values={
                    "total_amount": str(original_amount),
                    "vendor_concentration_flag": False,
                    "vendor_concentration_ratio": None,
                },
                modified_values={
                    "total_amount": str(inflated_amount),
                    "vendor_concentration_flag": True,
                    "vendor_concentration_ratio": str(concentration_pct),
                },
                financial_impact=financial_impact,
                description=(
                    f"Vendor {vendor_id} shows {concentration_display_pct}% spend "
                    f"concentration (threshold: 30%)"
                ),
                extra_metadata={
                    "concentration_pct": str(concentration_pct),
                    "normal_threshold": str(_NORMAL_CONCENTRATION_THRESHOLD),
                    "vendor_id": str(vendor_id),
                    "inflation_factor": str(inflation_factor),
                    "original_amount": str(original_amount),
                    "inflated_amount": str(inflated_amount),
                },
            )

            # ---------------------------------------------------------- #
            # Step 10: Log the injection event                           #
            # ---------------------------------------------------------- #
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
            # Re-raise domain exceptions unmodified
            raise
        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            logger.error(
                "vendor_concentration_injection_failed",
                service_name="transactions",
                component="VendorConcentration",
                type_code=self.type_code,
                error=str(exc),
            )
            raise DiscrepancyInjectionError(
                f"P2P-014 injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "error": str(exc),
                },
            ) from exc
