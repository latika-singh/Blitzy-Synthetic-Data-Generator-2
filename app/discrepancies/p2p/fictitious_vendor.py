"""P2P-013: Fictitious Vendor discrepancy type implementation.

Modifies vendor contact information (address and/or phone) to match an
employee's personal information, simulating a fictitious vendor that is
actually a shell company controlled by an insider. This is a classic
procurement fraud scheme where an employee creates a vendor whose contact
details (address, phone number, or both) overlap with the employee's
personal information — payments to such a vendor are effectively
embezzlement routed through a sham entity.

Configurable Parameters:
    match_type (str): Which field(s) to match to the employee.
        Allowed values: "address", "phone", "both".
        Default: "address".

Detection Method: entity_validation
    Detected by cross-referencing vendor addresses/phones against
    employee master data.

Difficulty: medium
    Requires cross-entity data analysis (vendor vs. employee records).

Financial Impact: Full transaction amount (potential embezzlement).

References:
    - AAP Section 0.5.1 Group 5: P2P-013 Fictitious Vendor
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - AAP Section 0.7.1: Deterministic reproducibility (seeded RNG)
"""

from __future__ import annotations

import copy
import random
from decimal import Decimal
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
__all__ = ["FictitiousVendor"]

# ---------------------------------------------------------------------------
# Street name choices for deterministic fake address generation
# ---------------------------------------------------------------------------
_STREET_NAMES: Tuple[str, ...] = (
    "Main",
    "Oak",
    "Elm",
    "First",
    "Maple",
    "Pine",
    "Cedar",
    "Walnut",
    "Park",
    "Washington",
)

_STREET_SUFFIXES: Tuple[str, ...] = (
    "St",
    "Ave",
    "Blvd",
    "Dr",
    "Ln",
    "Ct",
)

_CITY_NAMES: Tuple[str, ...] = (
    "Springfield",
    "Riverside",
    "Fairview",
    "Greenville",
    "Franklin",
    "Madison",
    "Clinton",
    "Georgetown",
    "Salem",
    "Bristol",
)

_STATE_CODES: Tuple[str, ...] = (
    "IL",
    "CA",
    "TX",
    "NY",
    "FL",
    "OH",
    "PA",
    "GA",
    "NC",
    "MI",
)

# Allowed match_type values for parameter validation
_ALLOWED_MATCH_TYPES: Tuple[str, ...] = ("address", "phone", "both")


class FictitiousVendor(BaseDiscrepancy):
    """P2P-013: Fictitious Vendor discrepancy.

    Modifies vendor contact information so that the vendor's address or
    phone number matches employee personal data, indicating a potential
    shell company controlled by an insider.  This is one of the most
    common procurement fraud schemes.

    The ``inject()`` method:
    1. Deep-copies the transaction to preserve the original.
    2. Validates and extracts ``match_type`` from params.
    3. Records original vendor contact fields.
    4. Generates employee-like contact data (from transaction context or
       deterministic fake data via the seeded RNG).
    5. Overwrites the corresponding vendor fields based on ``match_type``.
    6. Computes financial impact as the full transaction amount.
    7. Creates the ground truth record for ``GroundTruthGenerator``.

    Class Attributes:
        type_code: ``"P2P-013"``
        category: ``"p2p"``
        difficulty: ``"medium"``
        name: ``"Fictitious Vendor"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"entity_validation"``
        PARAMETER_BOUNDS: Parameter specification for ``match_type``.
    """

    # ------------------------------------------------------------------
    # ClassVar attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-013"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Fictitious Vendor"
    description: ClassVar[str] = (
        "Vendor address or phone matches an employee "
        "(shell company indicator)"
    )
    detection_method: ClassVar[str] = "entity_validation"

    # ------------------------------------------------------------------
    # PARAMETER_BOUNDS — match_type is a string enum, not numeric
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "match_type": {
            "type": "str",
            "default": "address",
            "allowed_values": ["address", "phone", "both"],
        },
    }

    # ------------------------------------------------------------------
    # inject() — Core injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a fictitious vendor discrepancy into the transaction.

        Modifies vendor contact information (address and/or phone) to
        match employee-like data, simulating a shell company.

        Args:
            transaction: The original transaction data dictionary.  Expected
                keys include ``vendor_id``, ``vendor_name``, ``vendor_address``,
                ``vendor_phone``, ``total_amount``, and optionally
                ``employee_data`` or ``employee_address`` / ``employee_phone``
                in the transaction context.
            params: Injection parameters.  Recognised keys:

                - ``match_type`` (str): ``"address"``, ``"phone"``, or
                  ``"both"``.  Defaults to ``"address"`` if absent.

            rng: A seeded :class:`random.Random` instance.  MUST be used
                for ALL random operations — NEVER the module-level RNG.

        Returns:
            A 2-tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If the transaction lacks required
                fields (``vendor_id``, ``total_amount``) or if
                ``match_type`` is not a valid allowed value.
        """
        try:
            # ----------------------------------------------------------
            # Step 1: Deep copy the transaction
            # ----------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ----------------------------------------------------------
            # Step 2: Validate params and extract match_type
            # ----------------------------------------------------------
            # _validate_params expects "min"/"max" in every bound spec
            # but match_type is a string enum without numeric bounds.
            # We pass an empty bounds dict and handle match_type
            # validation (default + allowed_values) explicitly.
            validated_params: Dict[str, Any] = self._validate_params(
                params, bounds={}
            )
            match_type: str = str(
                validated_params.get(
                    "match_type",
                    self.PARAMETER_BOUNDS["match_type"]["default"],
                )
            )

            if match_type not in _ALLOWED_MATCH_TYPES:
                raise DiscrepancyInjectionError(
                    f"Invalid match_type '{match_type}' for {self.type_code}. "
                    f"Allowed values: {list(_ALLOWED_MATCH_TYPES)}",
                    details={
                        "discrepancy_type": self.type_code,
                        "param_name": "match_type",
                        "value": match_type,
                        "allowed_values": list(_ALLOWED_MATCH_TYPES),
                    },
                )

            # ----------------------------------------------------------
            # Step 3: Extract and validate required transaction fields
            # ----------------------------------------------------------
            vendor_id: str = str(
                modified.get("vendor_id", modified.get("vendor", {}).get("id", ""))
            )
            if not vendor_id:
                raise DiscrepancyInjectionError(
                    f"Transaction missing 'vendor_id' for {self.type_code}",
                    details={
                        "discrepancy_type": self.type_code,
                        "missing_field": "vendor_id",
                    },
                )

            # Financial amount — must be present for impact calculation
            raw_amount = modified.get(
                "total_amount",
                modified.get("amount", modified.get("invoice_total", Decimal("0"))),
            )
            total_amount: Decimal = (
                Decimal(str(raw_amount))
                if not isinstance(raw_amount, Decimal)
                else raw_amount
            )

            # ----------------------------------------------------------
            # Step 4: Store original vendor contact fields
            # ----------------------------------------------------------
            original_vendor_address: Optional[str] = modified.get("vendor_address")
            original_vendor_phone: Optional[str] = modified.get("vendor_phone")
            original_vendor_city: Optional[str] = modified.get("vendor_city")
            original_vendor_state: Optional[str] = modified.get("vendor_state")
            original_vendor_zip: Optional[str] = modified.get("vendor_zip")

            # ----------------------------------------------------------
            # Step 5: Generate employee-like contact data
            # ----------------------------------------------------------
            # Prefer employee data from the transaction context if present;
            # otherwise generate plausible fake data deterministically.
            employee_data: Dict[str, Any] = modified.get("employee_data", {})

            # --- Address data ---
            employee_address: str = str(
                employee_data.get(
                    "address",
                    modified.get("employee_address", ""),
                )
            )
            employee_city: str = str(
                employee_data.get(
                    "city",
                    modified.get("employee_city", ""),
                )
            )
            employee_state: str = str(
                employee_data.get(
                    "state",
                    modified.get("employee_state", ""),
                )
            )
            employee_zip: str = str(
                employee_data.get(
                    "zip",
                    modified.get("employee_zip", ""),
                )
            )

            # --- Phone data ---
            employee_phone: str = str(
                employee_data.get(
                    "phone",
                    modified.get("employee_phone", ""),
                )
            )

            # Generate fake employee-like data when context is missing
            if not employee_address:
                street_num = rng.randint(100, 9999)
                street_name = rng.choice(_STREET_NAMES)
                street_suffix = rng.choice(_STREET_SUFFIXES)
                employee_address = f"{street_num} {street_name} {street_suffix}"
            if not employee_city:
                employee_city = rng.choice(_CITY_NAMES)
            if not employee_state:
                employee_state = rng.choice(_STATE_CODES)
            if not employee_zip:
                employee_zip = f"{rng.randint(10000, 99999)}"
            if not employee_phone:
                area = rng.randint(100, 999)
                prefix = rng.randint(100, 999)
                line = rng.randint(1000, 9999)
                employee_phone = f"555-{area}-{prefix}{line}"

            # ----------------------------------------------------------
            # Step 6: Overwrite vendor fields based on match_type
            # ----------------------------------------------------------
            affected_fields: List[str] = []
            original_values: Dict[str, Any] = {}
            modified_values: Dict[str, Any] = {}

            if match_type in ("address", "both"):
                # Overwrite vendor address fields with employee data
                affected_fields.extend([
                    "vendor_address",
                    "vendor_city",
                    "vendor_state",
                    "vendor_zip",
                ])
                original_values["vendor_address"] = original_vendor_address
                original_values["vendor_city"] = original_vendor_city
                original_values["vendor_state"] = original_vendor_state
                original_values["vendor_zip"] = original_vendor_zip

                modified["vendor_address"] = employee_address
                modified["vendor_city"] = employee_city
                modified["vendor_state"] = employee_state
                modified["vendor_zip"] = employee_zip

                modified_values["vendor_address"] = employee_address
                modified_values["vendor_city"] = employee_city
                modified_values["vendor_state"] = employee_state
                modified_values["vendor_zip"] = employee_zip

            if match_type in ("phone", "both"):
                affected_fields.append("vendor_phone")
                original_values["vendor_phone"] = original_vendor_phone

                modified["vendor_phone"] = employee_phone

                modified_values["vendor_phone"] = employee_phone

            # Set fictitious vendor indicator metadata
            modified["vendor_is_fictitious"] = True
            modified["fictitious_match_type"] = match_type
            affected_fields.append("vendor_is_fictitious")
            original_values["vendor_is_fictitious"] = False
            modified_values["vendor_is_fictitious"] = True

            # ----------------------------------------------------------
            # Step 7: Calculate financial impact
            # ----------------------------------------------------------
            # Full transaction amount represents potential embezzlement
            financial_impact: Decimal = abs(total_amount)

            # ----------------------------------------------------------
            # Step 8: Build ground truth data
            # ----------------------------------------------------------
            # Construct the matched_fields list for metadata
            matched_fields: List[str] = []
            if match_type in ("address", "both"):
                matched_fields.append("address")
            if match_type in ("phone", "both"):
                matched_fields.append("phone")

            gt_description: str = (
                f"Vendor {vendor_id} {match_type} matches employee data "
                f"(fictitious vendor / shell company indicator)"
            )

            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=gt_description,
                extra_metadata={
                    "match_type": match_type,
                    "matched_fields": matched_fields,
                    "employee_match_indicator": True,
                    "vendor_id": vendor_id,
                    "vendor_name": str(modified.get("vendor_name", "")),
                },
            )

            # ----------------------------------------------------------
            # Step 9: Log and return
            # ----------------------------------------------------------
            self._log_injection(
                transaction_id=str(
                    modified.get(
                        "transaction_id",
                        modified.get("invoice_id", modified.get("id", "unknown")),
                    )
                ),
                financial_impact=financial_impact,
                context={
                    "simulation_id": modified.get("simulation_id"),
                    "trace_id": modified.get("trace_id"),
                    "match_type": match_type,
                    "vendor_id": vendor_id,
                },
            )

            logger.debug(
                "fictitious_vendor_injected",
                service_name="transactions",
                component="FictitiousVendor",
                type_code=self.type_code,
                vendor_id=vendor_id,
                match_type=match_type,
                matched_fields=matched_fields,
                financial_impact=str(financial_impact),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors as-is
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Unexpected error injecting {self.type_code}: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            ) from exc
