"""O2C-008: Round-Tripping discrepancy type.

Simulates circular transactions (round-tripping) where goods or services
are sold to a party and simultaneously purchased back from the same or a
related party, creating artificial revenue and expense inflation.

Round-tripping artificially inflates both revenue and cost of goods sold
while maintaining the appearance of legitimate business activity. The
net economic benefit is zero, but financial statements show higher
top-line revenue.

Indicators of round-tripping:
- Same entity appears as both customer and vendor
- Sale and purchase of identical or similar amounts within a short period
- Lack of physical movement of goods
- Related-party transactions at non-market prices

Detection Method: ``pattern_analysis`` — identifying circular transaction
flows where revenue is matched by corresponding purchases to/from the
same or related entities.

Difficulty: ``medium`` — requires cross-referencing P2P and O2C transactions
to identify circular patterns.

Parameter Bounds: None — no configurable parameters.

References:
    - AAP Section 0.5.1 Group 5: O2C-008 Round-Tripping
    - AAP Section 0.7.5: Discrepancy Injection Rules
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Tuple
from uuid import uuid4

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
__all__ = ["RoundTripping"]

# ---------------------------------------------------------------------------
# Possible entity relationship types for the round-trip leg
# ---------------------------------------------------------------------------
_ENTITY_RELATIONSHIPS: List[str] = [
    "same_entity",
    "related_party",
]


class RoundTripping(BaseDiscrepancy):
    """O2C-008: Round-Tripping — circular transactions inflating revenue.

    Simulates a circular transaction where goods or services are sold to
    a party and simultaneously purchased back from the same or a related
    party.  The net economic benefit is zero, but financial statements
    show artificially higher top-line revenue.

    This discrepancy modifies a sales-side transaction to flag it as part
    of a round-trip cycle and generates a ``corresponding_purchase_id``
    referencing the buy-back leg.  No physical movement of goods occurs
    in a round-trip — the ``physical_goods_movement`` field is explicitly
    set to ``False``.

    Class-Level Attributes:
        type_code: ``"O2C-008"``
        category: ``"o2c"``
        difficulty: ``"medium"``
        name: ``"Round-Tripping"``
        description: Circular transaction description.
        detection_method: ``"pattern_analysis"``
        PARAMETER_BOUNDS: Empty dict — no configurable parameters.

    Inherits from :class:`BaseDiscrepancy`:
        - ``inject()`` abstract method implementation
        - ``_validate_params()`` parameter validation
        - ``_create_ground_truth_data()`` ground truth scaffolding
        - ``_log_injection()`` structured logging
        - ``_copy_transaction()`` deep-copy helper
    """

    # ------------------------------------------------------------------
    # Class-level attributes (REQUIRED by BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-008"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Round-Tripping"
    description: ClassVar[str] = (
        "Circular transaction where goods/services are sold and purchased back "
        "from the same or related party, artificially inflating revenue."
    )
    detection_method: ClassVar[str] = "pattern_analysis"

    # No configurable parameters for this discrepancy type.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection implementation
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a round-tripping indicator into the given transaction.

        Marks the sales-side transaction as part of a circular round-trip
        by setting indicator flags and generating a corresponding purchase
        identifier.  The entire sale amount is considered artificial
        revenue because the net economic benefit is zero.

        Steps:
            1. Deep-copy the transaction to preserve the original.
            2. Validate parameters (none expected).
            3. Determine entity relationship (same_entity or related_party).
            4. Generate a ``corresponding_purchase_id`` for the buy-back.
            5. Set round-trip indicator flags on the transaction.
            6. Calculate financial impact as the full order/sale amount.
            7. Build ground truth data for the GroundTruthGenerator.
            8. Log the injection event.

        Args:
            transaction: Sales order / customer invoice transaction data.
                Expected keys include ``transaction_id`` (str), and one of
                ``total_amount``, ``order_amount``, or ``invoice_amount``
                for the financial impact calculation.  ``customer_id`` is
                used to model the same-entity-as-vendor pattern.
            params: Injection parameters — empty dict expected for this
                type (no configurable parameters).
            rng: Seeded :class:`random.Random` instance for deterministic
                behavior.  MUST use this RNG exclusively.

        Returns:
            A 2-tuple of:
            - **modified_transaction** (Dict[str, Any]) — transaction with
              round-trip flags and metadata set.
            - **ground_truth_data** (Dict[str, Any]) — ground truth record
              for the GroundTruthGenerator.

        Raises:
            DiscrepancyInjectionError: If the transaction lacks a usable
                amount field or the injection logic fails for any reason.
        """
        try:
            # ---------------------------------------------------------------
            # Step 1: Deep-copy to preserve original transaction data
            # ---------------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ---------------------------------------------------------------
            # Step 2: Validate parameters (empty bounds — pass-through)
            # ---------------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # ---------------------------------------------------------------
            # Step 3: Determine entity relationship via seeded RNG
            # ---------------------------------------------------------------
            entity_relationship: str = rng.choice(_ENTITY_RELATIONSHIPS)

            # ---------------------------------------------------------------
            # Step 4: Generate corresponding purchase identifier
            # The corresponding_purchase_id represents the buy-back leg of
            # the round-trip that would appear in the P2P pipeline.
            # ---------------------------------------------------------------
            corresponding_purchase_id: str = str(uuid4())

            # ---------------------------------------------------------------
            # Step 5: Mark the transaction as a round-trip
            # ---------------------------------------------------------------
            # Preserve original customer_id as the entity that also acts
            # as vendor in the buy-back leg.
            customer_id: str = str(modified.get("customer_id", "unknown"))

            modified["round_trip_indicator"] = True
            modified["corresponding_purchase_id"] = corresponding_purchase_id
            modified["related_party_transaction"] = True
            modified["physical_goods_movement"] = False
            modified["vendor_reference"] = customer_id

            # Round-trip metadata — enriches the transaction record for
            # downstream pattern-analysis detection.
            modified["round_trip_metadata"] = {
                "matching_transaction_type": "purchase_order",
                "entity_relationship": entity_relationship,
                "corresponding_purchase_id": corresponding_purchase_id,
                "vendor_reference": customer_id,
            }

            # ---------------------------------------------------------------
            # Step 6: Calculate financial impact — full order amount
            # Revenue is entirely artificial in a round-trip.
            # ---------------------------------------------------------------
            raw_amount = (
                modified.get("total_amount")
                or modified.get("order_amount")
                or modified.get("invoice_amount")
                or modified.get("amount")
                or Decimal("0")
            )
            financial_impact: Decimal = Decimal(str(raw_amount))

            # Also record the matching_amount on the buy-back leg.
            # In a real round-trip the purchase amount closely mirrors
            # the sale amount — we use the same value here.
            matching_amount: Decimal = financial_impact
            modified["round_trip_metadata"]["matching_amount"] = str(
                matching_amount
            )

            # ---------------------------------------------------------------
            # Step 7: Build ground truth data
            # ---------------------------------------------------------------
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[
                    "round_trip_indicator",
                    "related_party_transaction",
                    "physical_goods_movement",
                ],
                original_values={
                    "round_trip_indicator": False,
                    "related_party_transaction": False,
                    "physical_goods_movement": True,
                },
                modified_values={
                    "round_trip_indicator": True,
                    "corresponding_purchase_id": corresponding_purchase_id,
                    "related_party_transaction": True,
                    "physical_goods_movement": False,
                    "vendor_reference": customer_id,
                },
                financial_impact=financial_impact,
                description=(
                    "Circular transaction \u2014 revenue artificially inflated "
                    "via round-tripping"
                ),
                extra_metadata={
                    "entity_relationship": entity_relationship,
                    "matching_amount": str(matching_amount),
                    "matching_transaction_type": "purchase_order",
                    "corresponding_purchase_id": corresponding_purchase_id,
                },
            )

            # ---------------------------------------------------------------
            # Step 8: Log the injection event
            # ---------------------------------------------------------------
            transaction_id: str = str(
                modified.get("transaction_id", "unknown")
            )
            self._log_injection(
                transaction_id=transaction_id,
                financial_impact=financial_impact,
                context={
                    "entity_relationship": entity_relationship,
                    "corresponding_purchase_id": corresponding_purchase_id,
                },
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in a DiscrepancyInjectionError so the
            # caller can apply the standard fallback (skip discrepancy).
            logger.error(
                "round_tripping_injection_failed",
                service_name="transactions",
                component="RoundTripping",
                type_code=self.type_code,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise DiscrepancyInjectionError(
                f"Failed to inject round-tripping discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            ) from exc
