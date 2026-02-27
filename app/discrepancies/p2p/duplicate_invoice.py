"""P2P-001: Duplicate Invoice discrepancy type implementation.

Creates an exact or near-duplicate of an existing vendor invoice. This is one
of the most common procurement fraud indicators — a vendor (intentionally or
accidentally) submits the same invoice twice, potentially resulting in double
payment.

Configurable Parameters:
    days_apart (int): Days between original and duplicate invoice dates.
        Bounds: 1–90 days. Default: 5 days.
    amount_variance_pct (Decimal): Percentage variation in invoice amount.
        Bounds: 0–5%. Default: 0% (exact duplicate).
        When > 0, creates a "near-duplicate" with slightly different amount.
    number_variation (int): Degree of invoice number variation (0–3).
        Default: 0 (exact same number = obvious duplicate).
        Higher values apply more character-level mutations (suffix, swap,
        prefix) for progressively harder-to-detect near-duplicates.

Detection Method: duplicate_check
    Detectable via invoice number comparison, vendor+amount+date proximity
    matching, or fuzzy matching on invoice attributes.

Difficulty: easy
    Standard duplicate detection algorithms catch this reliably.

References:
    - AAP Section 0.5.1 Group 5: P2P-001 Duplicate Invoice
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Decimal precision (prec=28, ROUND_HALF_UP)
    - AAP Section 0.7.1: Deterministic reproducibility (seeded RNG)
"""

from __future__ import annotations

import copy
import random
from datetime import timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import structlog

from app.discrepancies.base_discrepancy import BaseDiscrepancy
from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — JSON to stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["DuplicateInvoice"]

# ---------------------------------------------------------------------------
# Invoice number variation suffixes used when ``number_variation`` > 0.
# Selected deterministically via the seeded ``rng`` parameter.
# ---------------------------------------------------------------------------
_NUMBER_SUFFIXES: Tuple[str, ...] = ("-A", "-DUP", "-R", "-2", "-COPY")


class DuplicateInvoice(BaseDiscrepancy):
    """P2P-001: Duplicate Invoice discrepancy.

    Injects a duplicate (exact or near-duplicate) vendor invoice into the
    transaction data.  Modifies the transaction to appear as a second
    submission of the same invoice, with configurable date offset, amount
    variance, and invoice number variation.

    The class can produce two variants:

    * **Exact duplicate** — same invoice number, same amount, different date.
      This is the most obvious form of duplicate and is readily caught by
      standard duplicate-detection algorithms.
    * **Near-duplicate** — slightly altered invoice number and/or amount,
      different date.  This requires fuzzy-matching or proximity-based
      detection to identify.

    All randomness uses the seeded ``random.Random`` instance passed to
    :meth:`inject` — module-level RNG is **never** used.
    """

    # ------------------------------------------------------------------
    # Required ClassVar attributes (from BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-001"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Duplicate Invoice"
    description: ClassVar[str] = (
        "Exact or near-duplicate vendor invoice submission, "
        "potentially resulting in double payment"
    )
    detection_method: ClassVar[str] = "duplicate_check"

    # ------------------------------------------------------------------
    # Parameter bounds for validation
    # ------------------------------------------------------------------
    # ``days_apart`` and ``amount_variance_pct`` are validated numerically
    # by ``_validate_params()``.  ``number_variation`` is an integer (0–3)
    # representing the degree of invoice number mutation applied.  A value
    # of 0 means no variation (exact duplicate number).
    # ------------------------------------------------------------------
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "days_apart": {
            "min": 1,
            "max": 90,
            "type": "int",
            "default": 5,
        },
        "amount_variance_pct": {
            "min": Decimal("0"),
            "max": Decimal("5"),
            "type": "Decimal",
            "default": Decimal("0"),
        },
        "number_variation": {
            "min": 0,
            "max": 3,
            "type": "int",
            "default": 0,
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
        """Inject a duplicate invoice discrepancy into *transaction*.

        Creates a modified copy of the transaction that represents a
        second (duplicate) submission of the same vendor invoice, with
        configurable temporal offset, amount variance, and invoice number
        variation.

        Args:
            transaction: Original transaction data dictionary.  Must
                contain at minimum ``"invoice_number"``, ``"invoice_date"``,
                and ``"total_amount"``.  An optional ``"transaction_id"``
                key is used for logging; if absent a placeholder is used.
            params: Injection parameters — ``"days_apart"`` (int),
                ``"amount_variance_pct"`` (Decimal), and
                ``"number_variation"`` (int, 0–3).  Missing keys receive
                default values from :attr:`PARAMETER_BOUNDS`.
            rng: A seeded :class:`random.Random` instance for
                deterministic behaviour.  **NEVER** use module-level
                ``random`` functions.

        Returns:
            A 2-tuple of:

            - **modified_transaction** — deep-copied transaction with the
              duplicate invoice modifications applied.
            - **ground_truth_data** — dictionary describing the injected
              discrepancy for the ``GroundTruthGenerator``.

        Raises:
            DiscrepancyInjectionError: If a required field
                (``invoice_number``, ``invoice_date``, ``total_amount``)
                is missing from *transaction*, or if injection logic
                encounters an unrecoverable error.
        """
        try:
            # 1. Deep-copy the transaction to preserve the original
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate & apply defaults for injection parameters
            validated: Dict[str, Any] = self._validate_params(
                params, self.PARAMETER_BOUNDS
            )

            # 3. Extract validated parameters
            days_apart: int = int(validated.get("days_apart", 5))
            amount_variance_pct: Decimal = Decimal(
                str(validated.get("amount_variance_pct", Decimal("0")))
            )
            number_variation: int = int(
                validated.get("number_variation", 0)
            )

            # 4. Validate that all required fields are present
            self._validate_required_fields(modified)

            # 5. Store original values for ground truth
            original_invoice_number: str = str(modified["invoice_number"])
            original_invoice_date = modified["invoice_date"]
            original_total_amount: Decimal = Decimal(
                str(modified["total_amount"])
            )

            # ----------------------------------------------------------
            # 6. Apply duplicate modifications
            # ----------------------------------------------------------

            # 6a. Date shift — move the invoice date forward by days_apart
            modified["invoice_date"] = original_invoice_date + timedelta(
                days=days_apart
            )

            # 6b. Amount variation — adjust amount within [-pct, +pct]
            modified_total_amount: Decimal = original_total_amount
            if amount_variance_pct > Decimal("0"):
                # Use seeded RNG for deterministic variance factor
                variance_factor: float = rng.uniform(
                    float(-amount_variance_pct), float(amount_variance_pct)
                )
                # Convert to Decimal for financial-grade arithmetic
                factor_decimal: Decimal = Decimal(str(variance_factor))
                modified_total_amount = (
                    original_total_amount
                    * (Decimal("1") + factor_decimal / Decimal("100"))
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                # Ensure the modified amount is never negative
                if modified_total_amount < Decimal("0"):
                    modified_total_amount = Decimal("0.01")

                modified["total_amount"] = modified_total_amount

            # 6c. Invoice number variation (graduated: 0=none, 1-3=degree)
            modified_invoice_number: str = original_invoice_number
            if number_variation > 0:
                modified_invoice_number = self._vary_invoice_number(
                    original_invoice_number, rng, degree=number_variation
                )
                modified["invoice_number"] = modified_invoice_number

            # 6d. Mark as duplicate in transaction metadata
            modified["is_duplicate"] = True
            modified.setdefault("metadata", {})
            if isinstance(modified["metadata"], dict):
                modified["metadata"]["is_duplicate"] = True
                modified["metadata"]["original_invoice_number"] = (
                    original_invoice_number
                )
                modified["metadata"]["duplicate_type"] = (
                    "near"
                    if (
                        amount_variance_pct > Decimal("0")
                        or number_variation > 0
                    )
                    else "exact"
                )

            # ----------------------------------------------------------
            # 7. Calculate financial impact
            # ----------------------------------------------------------
            if amount_variance_pct > Decimal("0"):
                # Near-duplicate: impact is the amount difference
                financial_impact: Decimal = abs(
                    modified_total_amount - original_total_amount
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                # Exact duplicate: full amount is the duplicate exposure
                financial_impact = original_total_amount

            # ----------------------------------------------------------
            # 8. Determine affected fields
            # ----------------------------------------------------------
            affected_fields: List[str] = ["invoice_date"]
            original_values: Dict[str, Any] = {
                "invoice_date": str(original_invoice_date),
            }
            modified_values: Dict[str, Any] = {
                "invoice_date": str(modified["invoice_date"]),
            }

            if amount_variance_pct > Decimal("0"):
                affected_fields.append("total_amount")
                original_values["total_amount"] = str(original_total_amount)
                modified_values["total_amount"] = str(modified_total_amount)

            if number_variation > 0:
                affected_fields.append("invoice_number")
                original_values["invoice_number"] = original_invoice_number
                modified_values["invoice_number"] = modified_invoice_number

            # ----------------------------------------------------------
            # 9. Create ground truth data
            # ----------------------------------------------------------
            duplicate_type: str = (
                "near"
                if (amount_variance_pct > Decimal("0") or number_variation > 0)
                else "exact"
            )

            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=affected_fields,
                original_values=original_values,
                modified_values=modified_values,
                financial_impact=financial_impact,
                description=(
                    f"Duplicate invoice {original_invoice_number} created "
                    f"{days_apart} days later"
                ),
                extra_metadata={
                    "days_apart": days_apart,
                    "amount_variance_pct": str(amount_variance_pct),
                    "number_variation": number_variation,
                    "duplicate_type": duplicate_type,
                },
            )

            # ----------------------------------------------------------
            # 10. Log the injection event
            # ----------------------------------------------------------
            transaction_id: str = str(
                modified.get("transaction_id", "unknown")
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
                "duplicate_invoice_injected",
                service_name="transactions",
                component="DuplicateInvoice",
                type_code=self.type_code,
                original_invoice_number=original_invoice_number,
                days_apart=days_apart,
                amount_variance_pct=str(amount_variance_pct),
                number_variation=number_variation,
                duplicate_type=duplicate_type,
                financial_impact=str(financial_impact),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise domain-specific errors as-is
            raise
        except Exception as exc:
            raise DiscrepancyInjectionError(
                f"Failed to inject duplicate invoice discrepancy: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "root_cause": str(exc),
                },
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required_fields(transaction: Dict[str, Any]) -> None:
        """Validate that *transaction* contains all fields required for P2P-001.

        Args:
            transaction: The transaction dictionary to validate.

        Raises:
            DiscrepancyInjectionError: If ``invoice_number``,
                ``invoice_date``, or ``total_amount`` is missing.
        """
        required_fields: Tuple[str, ...] = (
            "invoice_number",
            "invoice_date",
            "total_amount",
        )
        missing: List[str] = [
            field for field in required_fields if field not in transaction
        ]
        if missing:
            raise DiscrepancyInjectionError(
                f"Transaction missing required fields for P2P-001 "
                f"(Duplicate Invoice): {', '.join(missing)}",
                details={
                    "discrepancy_type": "P2P-001",
                    "missing_fields": missing,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                },
            )

    @staticmethod
    def _vary_invoice_number(
        original_number: str, rng: random.Random, *, degree: int = 1
    ) -> str:
        """Create a varied invoice number for a near-duplicate.

        Applies graduated mutation based on *degree* (number of characters
        that differ from the original):

        - **degree 1** — Single mutation: one of suffix append, last-char
          swap, or prefix alteration.
        - **degree 2** — Two mutations applied sequentially.
        - **degree 3** — Three mutations applied sequentially.

        Each mutation step is selected deterministically via *rng*.

        Available single-mutation strategies:

        1. **Suffix** — appends a suffix from ``_NUMBER_SUFFIXES``
           (e.g., ``"INV-1234"`` → ``"INV-1234-A"``).
        2. **Last-character swap** — replaces the last character with a
           different digit or letter (e.g., ``"INV-1234"`` → ``"INV-1235"``).
        3. **Prefix alteration** — prepends ``"DUP-"`` to the number
           (e.g., ``"INV-1234"`` → ``"DUP-INV-1234"``).

        Args:
            original_number: The original invoice number string.
            rng: Seeded :class:`random.Random` instance.
            degree: Number of character-level mutations to apply (1–3).
                Clamped to ``[1, 3]`` internally.

        Returns:
            A modified invoice number string.
        """
        degree = max(1, min(degree, 3))
        result: str = original_number

        for _ in range(degree):
            result = DuplicateInvoice._apply_single_mutation(result, rng)

        return result

    @staticmethod
    def _apply_single_mutation(number: str, rng: random.Random) -> str:
        """Apply a single mutation to an invoice number string.

        Selects one of three strategies deterministically via *rng*:
        suffix append, last-character swap, or prefix alteration.

        Args:
            number: Current invoice number string.
            rng: Seeded :class:`random.Random` instance.

        Returns:
            The mutated invoice number string.
        """
        strategy: int = rng.randint(0, 2)

        if strategy == 0:
            # Strategy 1: Append a suffix
            suffix: str = rng.choice(_NUMBER_SUFFIXES)
            return f"{number}{suffix}"

        elif strategy == 1:
            # Strategy 2: Swap last character
            if not number:
                return number + rng.choice(_NUMBER_SUFFIXES)

            last_char: str = number[-1]
            if last_char.isdigit():
                # Pick a different digit
                new_digit: str = str((int(last_char) + rng.randint(1, 9)) % 10)
                return number[:-1] + new_digit
            else:
                # Pick a different letter
                replacement: str = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                while replacement == last_char.upper():
                    replacement = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                return number[:-1] + replacement

        else:
            # Strategy 3: Prepend "DUP-"
            return f"DUP-{number}"
