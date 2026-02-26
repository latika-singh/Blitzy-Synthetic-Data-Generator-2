"""P2P-009: Duplicate Payment discrepancy type implementation.

Creates a duplicate payment record to the same vendor for the same amount
within a configurable time period. Duplicate payments are a high-risk
procurement issue — they result in actual cash outflows and can be either
accidental (system/process failure) or intentional (fraud).

Configurable Parameters:
    days_apart (int): Days between original and duplicate payments.
        Bounds: 1–90 days. Default: 7 days.

Detection Method: duplicate_check
    Detected by matching payments to the same vendor with the same amount
    within the specified time window.

Difficulty: easy

Financial Impact: The full duplicate payment amount (double exposure).

References:
    - AAP Section 0.5.1 Group 5: P2P-009 Duplicate Payment
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - AAP Section 0.7.1: Deterministic reproducibility (seeded RNG)
"""

from __future__ import annotations

import copy
import random
from datetime import timedelta
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
__all__ = ["DuplicatePayment"]

# ---------------------------------------------------------------------------
# Possible suffixes appended to payment_number to create subtle variation
# in the duplicate payment record while preserving the core number for
# duplicate-detection algorithm validation.
# ---------------------------------------------------------------------------
_PAYMENT_NUMBER_SUFFIXES: List[str] = [
    "-A",
    "-R",
    "-DUP",
    "-2",
    "-REV",
    "-RP",
    "-B",
    "-X",
]


class DuplicatePayment(BaseDiscrepancy):
    """P2P-009: Duplicate Payment discrepancy.

    Injects a duplicate payment to the same vendor for the same amount
    within a configurable time window (``days_apart``).  The duplicate
    retains the same ``vendor_id`` and ``payment_amount`` as the original
    — the defining characteristics of a duplicate payment — while shifting
    the ``payment_date`` forward by ``days_apart`` days and optionally
    modifying the ``payment_number`` with a minor suffix.

    Duplicate payments are one of the highest-risk procurement issues
    because they result in *actual* cash outflows.  They can be either:
    - **Accidental**: System double-post, process failure, or timing issue
    - **Intentional**: Fraud scheme where an insider processes the same
      payment twice and diverts the duplicate

    The financial impact is the *full* duplicate payment amount because
    the entire second payment represents unnecessary cash exposure.

    Attributes:
        type_code: ``"P2P-009"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"Duplicate Payment"``
        description: Short human-readable description.
        detection_method: ``"duplicate_check"``
        PARAMETER_BOUNDS: Bounds for ``days_apart`` (1–90, default 7).
    """

    # ------------------------------------------------------------------
    # Class-level attributes — required by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-009"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Duplicate Payment"
    description: ClassVar[str] = (
        "Same payment to same vendor/amount within a short period"
    )
    detection_method: ClassVar[str] = "duplicate_check"

    # ------------------------------------------------------------------
    # Parameter bounds (consumed by _validate_params)
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "days_apart": {
            "min": 1,
            "max": 90,
            "type": "int",
            "default": 7,
        },
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
        """Inject a duplicate payment discrepancy into the transaction.

        Creates a duplicate payment record by shifting the payment date
        forward by ``days_apart`` days while keeping the vendor and amount
        identical.  Optionally modifies the payment number with a minor
        suffix to simulate a slightly-varied duplicate.

        The method follows these steps:

        1. Deep-copy the transaction to preserve the original.
        2. Validate and extract injection parameters.
        3. Record original payment fields for ground truth.
        4. Shift ``payment_date`` forward by ``days_apart`` days.
        5. Optionally modify ``payment_number`` with a suffix.
        6. Mark the transaction with ``is_duplicate_payment = True``.
        7. Calculate financial impact (full payment amount).
        8. Assemble ground truth data via ``_create_ground_truth_data()``.
        9. Log the injection event.
        10. Return ``(modified_transaction, ground_truth_data)``.

        Args:
            transaction: Original payment transaction dictionary.  Expected
                keys include ``payment_date``, ``payment_amount``,
                ``vendor_id``, and optionally ``payment_number``,
                ``payment_id``, ``check_number``, ``payment_method``,
                ``transaction_id``.
            params: Injection parameters.  Recognised keys:

                - ``days_apart`` (int): Days between original and duplicate.
                  Bounds: 1–90.  Default: 7.

            rng: Seeded :class:`random.Random` instance for deterministic
                behaviour.  Used for payment number suffix selection.

        Returns:
            A tuple of ``(modified_transaction, ground_truth_data)``.

        Raises:
            DiscrepancyInjectionError: If required fields (``payment_date``,
                ``payment_amount``, ``vendor_id``) are missing from the
                transaction, or if injection logic encounters an error.
        """
        try:
            # ----------------------------------------------------------
            # Step 1 — Deep copy to preserve original transaction
            # ----------------------------------------------------------
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # ----------------------------------------------------------
            # Step 2 — Validate parameters against PARAMETER_BOUNDS
            # ----------------------------------------------------------
            validated_params: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )
            days_apart: int = int(validated_params.get("days_apart", 7))

            # ----------------------------------------------------------
            # Step 3 — Extract and validate required payment fields
            # ----------------------------------------------------------
            payment_date = modified.get("payment_date")
            if payment_date is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required 'payment_date' field "
                    "for duplicate payment injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                    },
                )

            payment_amount = modified.get("payment_amount")
            if payment_amount is None:
                # Fall back to total_amount if payment_amount is absent
                payment_amount = modified.get("total_amount")
            if payment_amount is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing 'payment_amount' and 'total_amount' "
                    "fields for duplicate payment injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                    },
                )

            vendor_id = modified.get("vendor_id")
            if vendor_id is None:
                raise DiscrepancyInjectionError(
                    "Transaction missing required 'vendor_id' field "
                    "for duplicate payment injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            modified.get("transaction_id", "unknown")
                        ),
                    },
                )

            # Record original values before modification
            original_payment_date = payment_date
            original_payment_number: Optional[str] = modified.get(
                "payment_number"
            )
            original_payment_id: Optional[str] = modified.get("payment_id")
            original_check_number: Optional[str] = modified.get(
                "check_number"
            )

            # ----------------------------------------------------------
            # Step 4 — Shift payment_date forward by days_apart
            # ----------------------------------------------------------
            new_payment_date = payment_date + timedelta(days=days_apart)
            modified["payment_date"] = new_payment_date

            # ----------------------------------------------------------
            # Step 5 — Optionally modify payment_number with suffix
            # ----------------------------------------------------------
            # Use rng to decide whether to modify the payment number
            # (50/50 chance — exact duplicates are also realistic)
            modified_payment_number: Optional[str] = original_payment_number
            if original_payment_number is not None:
                should_modify_number: bool = rng.choice([True, False])
                if should_modify_number:
                    suffix: str = rng.choice(_PAYMENT_NUMBER_SUFFIXES)
                    modified_payment_number = (
                        f"{original_payment_number}{suffix}"
                    )
                    modified["payment_number"] = modified_payment_number

            # ----------------------------------------------------------
            # Step 6 — Mark as duplicate payment in metadata
            # ----------------------------------------------------------
            modified["is_duplicate_payment"] = True
            modified.setdefault("metadata", {})
            if isinstance(modified["metadata"], dict):
                modified["metadata"]["duplicate_source"] = str(
                    original_payment_id or original_payment_number or "unknown"
                )
                modified["metadata"]["days_apart"] = days_apart

            # vendor_id and payment_amount are deliberately UNCHANGED
            # — this is what makes it a duplicate payment

            # ----------------------------------------------------------
            # Step 7 — Calculate financial impact
            # ----------------------------------------------------------
            # Financial impact is the FULL duplicate payment amount
            # because the entire second payment is unnecessary exposure
            financial_impact: Decimal = Decimal(str(payment_amount))

            # ----------------------------------------------------------
            # Step 8 — Build affected fields and ground truth
            # ----------------------------------------------------------
            affected_fields: List[str] = [
                "payment_date",
                "is_duplicate_payment",
            ]
            original_values: Dict[str, Any] = {
                "payment_date": str(original_payment_date),
                "is_duplicate_payment": False,
            }
            modified_values: Dict[str, Any] = {
                "payment_date": str(new_payment_date),
                "is_duplicate_payment": True,
            }

            # Include payment_number in affected fields if it was modified
            if modified_payment_number != original_payment_number:
                affected_fields.append("payment_number")
                original_values["payment_number"] = str(
                    original_payment_number
                )
                modified_values["payment_number"] = str(
                    modified_payment_number
                )

            # Retrieve payment method for metadata (if available)
            payment_method: str = str(
                modified.get("payment_method", "unknown")
            )

            # Build description with financial details
            amount_str: str = str(payment_amount)
            gt_description: str = (
                f"Duplicate payment to vendor {vendor_id} "
                f"for ${amount_str}, {days_apart} days apart"
            )

            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=gt_description,
                extra_metadata={
                    "days_apart": days_apart,
                    "vendor_id": str(vendor_id),
                    "payment_method": payment_method,
                    "payment_amount": str(payment_amount),
                    "original_payment_number": str(
                        original_payment_number or ""
                    ),
                    "modified_payment_number": str(
                        modified_payment_number or ""
                    ),
                    "original_payment_id": str(
                        original_payment_id or ""
                    ),
                    "original_check_number": str(
                        original_check_number or ""
                    ),
                },
            )

            # ----------------------------------------------------------
            # Step 9 — Log the injection event
            # ----------------------------------------------------------
            transaction_id: str = str(
                modified.get("transaction_id")
                or modified.get("payment_id")
                or "unknown"
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
                "duplicate_payment_injected",
                service_name="transactions",
                component="DuplicatePayment",
                type_code=self.type_code,
                vendor_id=str(vendor_id),
                payment_amount=str(payment_amount),
                days_apart=days_apart,
                payment_number_modified=(
                    modified_payment_number != original_payment_number
                ),
                payment_method=payment_method,
            )

            # ----------------------------------------------------------
            # Step 10 — Return modified transaction and ground truth
            # ----------------------------------------------------------
            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors without wrapping
            raise

        except Exception as exc:
            # Wrap unexpected errors in DiscrepancyInjectionError
            raise DiscrepancyInjectionError(
                f"Unexpected error during {self.type_code} injection: "
                f"{exc!s}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error_type": type(exc).__name__,
                },
            ) from exc
