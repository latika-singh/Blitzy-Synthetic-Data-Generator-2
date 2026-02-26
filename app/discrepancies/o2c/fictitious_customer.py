"""O2C-007: Fictitious Customer discrepancy type.

Simulates a transaction involving a fictitious (fabricated or non-existent)
customer entity. This is a fraud indicator where fake customers are created
to generate fictitious revenue or divert payments.

The discrepancy replaces the legitimate customer reference with indicators
of a fictitious entity — such as a customer ID that doesn't exist in the
master data, or attributes that match known fraud patterns (e.g., address
matching an employee's address, as seen in P2P-013 for vendors).

Detection Method: ``entity_validation`` — verifying customer exists in master
data, cross-referencing addresses with employee records, checking for
activity patterns consistent with real customers.

Difficulty: ``medium`` — requires master data cross-referencing and pattern
analysis beyond simple lookup.

Parameter Bounds: None — no configurable parameters.

References:
    - AAP Section 0.5.1 Group 5: O2C-007 Fictitious Customer
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - P2P-013 (Fictitious Vendor) as analogous pattern
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
# Module-level structured logger (AAP Section 0.7.7 — JSON to stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Fictitious customer name pools — used with seeded rng for deterministic
# name generation.  These pools combine common-sounding first and last name
# fragments to produce plausible-but-fabricated names that wouldn't appear
# in master data.  Kept as module-level tuples for fast random access.
# ---------------------------------------------------------------------------
_FICTITIOUS_FIRST_NAMES: Tuple[str, ...] = (
    "Apex",
    "Summit",
    "Zenith",
    "Pinnacle",
    "Crest",
    "Vertex",
    "Meridian",
    "Nexus",
    "Prism",
    "Forge",
    "Beacon",
    "Vanguard",
    "Nova",
    "Stratos",
    "Titan",
)

_FICTITIOUS_LAST_NAMES: Tuple[str, ...] = (
    "Holdings",
    "Enterprises",
    "Solutions",
    "Partners",
    "Associates",
    "Industries",
    "Ventures",
    "Trading Co",
    "Global",
    "Dynamics",
    "Systems",
    "Consulting",
    "Services",
    "Corporation",
    "Group",
)

_FICTITIOUS_STREET_ADDRESSES: Tuple[str, ...] = (
    "123 Main St",
    "456 Oak Ave",
    "789 Elm Blvd",
    "321 Pine Rd",
    "654 Maple Dr",
    "987 Cedar Ln",
    "111 Birch Way",
    "222 Walnut Ct",
    "333 Spruce Pl",
    "444 Willow Ter",
)

_FICTITIOUS_CITIES: Tuple[str, ...] = (
    "Springfield",
    "Riverdale",
    "Centerville",
    "Lakewood",
    "Fairview",
    "Georgetown",
    "Millville",
    "Greenfield",
    "Plainview",
    "Brookside",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["FictitiousCustomer"]


class FictitiousCustomer(BaseDiscrepancy):
    """O2C-007: Fictitious Customer discrepancy.

    Simulates a transaction involving a fabricated or non-existent customer
    entity.  The ``inject()`` method replaces the legitimate customer
    reference with a fictitious customer ID and name that do not exist
    in the master data, and flags the transaction as not validated.

    The financial impact is the full order/invoice amount because all
    revenue attributed to the fictitious customer is fraudulent — there
    is no real customer to fulfill the obligation.

    This is the O2C analogue of P2P-013 (Fictitious Vendor), which
    operates on the procurement side.

    Attributes:
        type_code: ``"O2C-007"``
        category: ``"o2c"``
        difficulty: ``"medium"``
        name: ``"Fictitious Customer"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"entity_validation"``
        PARAMETER_BOUNDS: Empty dict — no configurable parameters.
    """

    # ------------------------------------------------------------------
    # Class-level attributes (required by BaseDiscrepancy contract)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "O2C-007"
    category: ClassVar[str] = "o2c"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Fictitious Customer"
    description: ClassVar[str] = (
        "Transaction involving a fabricated or non-existent customer entity, "
        "a fraud indicator for fictitious revenue or payment diversion."
    )
    detection_method: ClassVar[str] = "entity_validation"

    # No configurable parameters for this discrepancy type.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a fictitious customer into the transaction.

        Replaces the legitimate customer reference in the transaction with
        a fabricated customer entity that does not exist in the master data.
        The entire order/invoice amount is treated as fictitious revenue.

        The method:

        1. Creates a deep copy of the original transaction data.
        2. Preserves original customer information for ground truth.
        3. Generates a fictitious customer ID (``CUST-FAKE-<hex>`` format),
           a plausible-sounding but fabricated company name, and an address
           that could match an employee record (fraud indicator).
        4. Replaces customer fields on the copied transaction.
        5. Calculates the financial impact as the full order/invoice amount.
        6. Builds ground truth data for the ``GroundTruthGenerator``.
        7. Logs the injection event via structured logging.

        Args:
            transaction: The original transaction data dictionary.  Expected
                keys include ``"customer_id"``, ``"customer_name"``,
                ``"total_amount"`` (or ``"order_amount"`` / ``"invoice_amount"``),
                and ``"transaction_id"``.  Missing keys are handled gracefully
                with safe defaults.
            params: Injection parameters.  This discrepancy type has no
                configurable parameters, so *params* is accepted but unused
                (validated against empty :attr:`PARAMETER_BOUNDS`).
            rng: A seeded :class:`random.Random` instance for deterministic
                reproducibility.  All random selections (name fragments,
                address components, customer IDs) MUST use this RNG.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)`` where:

            - **modified_transaction** has the customer replaced with a
              fictitious entity and ``customer_validated`` set to ``False``.
            - **ground_truth_data** contains all fields required by the
              ``GroundTruthGenerator``, including the original and modified
              customer identifiers and the full financial impact.

        Raises:
            DiscrepancyInjectionError: If injection fails due to missing
                critical transaction fields or unexpected data shapes.
        """
        try:
            # ---- Step 1: Deep copy to preserve original -----------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ---- Step 2: Validate params (empty bounds — pass-through) --
            self._validate_params(params, self.PARAMETER_BOUNDS)

            # ---- Step 3: Capture original customer data -----------------
            original_customer_id: str = str(
                transaction.get("customer_id", "unknown")
            )
            original_customer_name: str = str(
                transaction.get("customer_name", "Unknown Customer")
            )
            original_customer_address: str = str(
                transaction.get("customer_address", "")
            )
            original_customer_status: str = str(
                transaction.get("customer_status", "active")
            )

            # ---- Step 4: Generate fictitious customer data --------------
            # Use a UUID-based fake customer ID that clearly won't exist in
            # master data.  The hex suffix is derived from uuid4 for
            # uniqueness, but we use rng to seed a deterministic selection
            # so that the same seed produces the same fake ID.
            fake_id_suffix: str = format(rng.randint(0, 0xFFFFFFFF), "08x")
            fictitious_customer_id: str = f"CUST-FAKE-{fake_id_suffix}"

            # Generate a plausible-but-fabricated company name from the
            # name pools, deterministically via rng.
            first_name: str = rng.choice(_FICTITIOUS_FIRST_NAMES)
            last_name: str = rng.choice(_FICTITIOUS_LAST_NAMES)
            fictitious_customer_name: str = f"{first_name} {last_name}"

            # Generate a fictitious address that could match an employee
            # address (a common fraud indicator — P2P-013 analogue).
            fictitious_street: str = rng.choice(_FICTITIOUS_STREET_ADDRESSES)
            fictitious_city: str = rng.choice(_FICTITIOUS_CITIES)
            fictitious_address: str = f"{fictitious_street}, {fictitious_city}"

            # ---- Step 5: Apply fictitious customer to transaction -------
            modified["customer_id"] = fictitious_customer_id
            modified["customer_name"] = fictitious_customer_name
            modified["customer_address"] = fictitious_address
            modified["customer_validated"] = False
            modified["customer_status"] = "unverified"
            modified["customer_verified"] = False
            modified["fictitious_entity"] = True

            # ---- Step 6: Calculate financial impact ---------------------
            # The financial impact is the FULL order/invoice amount because
            # all revenue attributed to a fictitious customer is fraudulent.
            raw_amount = transaction.get(
                "total_amount",
                transaction.get(
                    "order_amount",
                    transaction.get("invoice_amount", 0),
                ),
            )
            financial_impact: Decimal = Decimal(str(raw_amount))

            # ---- Step 7: Extract transaction ID for logging -------------
            transaction_id: str = str(
                transaction.get(
                    "transaction_id",
                    transaction.get("invoice_id", "unknown"),
                )
            )

            # ---- Step 8: Build ground truth data ------------------------
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[
                    "customer_id",
                    "customer_name",
                    "customer_address",
                    "customer_validated",
                    "customer_status",
                    "customer_verified",
                ],
                original_values={
                    "customer_id": original_customer_id,
                    "customer_name": original_customer_name,
                    "customer_address": original_customer_address,
                    "customer_status": original_customer_status,
                    "customer_validated": True,
                    "customer_verified": True,
                },
                modified_values={
                    "customer_id": fictitious_customer_id,
                    "customer_name": fictitious_customer_name,
                    "customer_address": fictitious_address,
                    "customer_status": "unverified",
                    "customer_validated": False,
                    "customer_verified": False,
                },
                financial_impact=financial_impact,
                description=(
                    "Transaction routed to fictitious customer entity "
                    f"'{fictitious_customer_name}' ({fictitious_customer_id})"
                ),
                extra_metadata={
                    "original_customer_id": original_customer_id,
                    "fictitious_customer_id": fictitious_customer_id,
                    "fictitious_customer_name": fictitious_customer_name,
                    "fictitious_address": fictitious_address,
                    "fraud_indicator": "entity_does_not_exist_in_master_data",
                    "analogous_p2p_type": "P2P-013",
                },
            )

            # ---- Step 9: Log the injection event ------------------------
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
            # Re-raise known injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap all unexpected errors in DiscrepancyInjectionError per
            # AAP §0.7.4 (1 attempt, no backoff, 5s timeout, fallback:
            # skip discrepancy).
            raise DiscrepancyInjectionError(
                f"Failed to inject fictitious customer discrepancy "
                f"(O2C-007): {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "category": self.category,
                    "difficulty": self.difficulty,
                    "error": str(exc),
                },
            ) from exc
