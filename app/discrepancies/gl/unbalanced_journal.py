"""GL-001: Unbalanced Journal Entry — introduces debit/credit imbalance.

Implements the Unbalanced Journal Entry discrepancy type that modifies a journal
entry so that SUM(debits) != SUM(credits), violating the fundamental GL balance
invariant (AAP Section 0.7.2: every journal entry MUST satisfy SUM(debits) =
SUM(credits) within $0.01 tolerance).

The ``GLPostingEngine`` enforces the balance check before persistence, so this
discrepancy simulates the scenario where an unbalanced entry bypasses that check
(perhaps due to a manual override or system glitch).

The ``imbalance_amount`` parameter controls the magnitude of the imbalance
(Decimal 0.02-1000.00). The minimum of 0.02 ensures the imbalance exceeds the
$0.01 tolerance threshold, making it detectable.

Catalog Entry:
    Type Code: GL-001
    Category: gl
    Difficulty: easy
    Detection Method: balance_check
    Parameters: imbalance_amount (Decimal 0.02-1000.00)

Financial Integrity Context (AAP Section 0.7.2):
    - GL Balance Invariant: SUM(debits) = SUM(credits) within $0.01
    - Continuous Trial Balance: cumulative TB = 0 within $0.01
    - Decimal Precision: prec=28, ROUND_HALF_UP
    - This discrepancy intentionally VIOLATES the balance invariant

References:
    - AAP Section 0.5.1 Group 5: GL-001 Unbalanced Journal Entry
    - AAP Section 0.7.2: Financial Integrity Rules (balance invariant)
    - AAP Section 0.7.5: Discrepancy Injection Rules (parameter bounds)
    - app/orchestration/transaction_orchestrator.py: REQUIRED_ARTIFACTS for journal_entry
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
# Module-level structured logger (AAP Section 0.7.7 — stdout only, JSON format)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Journal Entry Field Name Constants
# ---------------------------------------------------------------------------

# Possible field names for journal entry line items in the transaction dict.
# The transaction_orchestrator.py REQUIRED_ARTIFACTS defines "je_lines" for
# journal_entry type; other names are supported for flexibility.
JE_LINES_FIELDS: List[str] = [
    "je_lines",
    "journal_lines",
    "lines",
    "entry_lines",
]

# Standard field names for debit and credit amounts on JE line items.
JE_DEBIT_FIELD: str = "debit_amount"
JE_CREDIT_FIELD: str = "credit_amount"

# Fallback field name when lines use a single "amount" field instead of
# separate debit_amount / credit_amount fields.
JE_AMOUNT_FIELD: str = "amount"


# ---------------------------------------------------------------------------
# Imbalance Injection Strategies
# ---------------------------------------------------------------------------
_STRATEGY_INCREASE_DEBIT: str = "increase_debit"
_STRATEGY_DECREASE_CREDIT: str = "decrease_credit"
_STRATEGY_ADD_UNMATCHED_LINE: str = "add_unmatched_line"

_ALL_STRATEGIES: List[str] = [
    _STRATEGY_INCREASE_DEBIT,
    _STRATEGY_DECREASE_CREDIT,
    _STRATEGY_ADD_UNMATCHED_LINE,
]


class UnbalancedJournal(BaseDiscrepancy):
    """GL-001: Unbalanced Journal Entry.

    Injects a debit/credit imbalance into a journal entry by modifying one
    or more line item amounts so that SUM(debits) != SUM(credits).

    This is an 'easy' difficulty discrepancy because the detection method
    is straightforward: sum all debit amounts, sum all credit amounts, and
    check whether |SUM(debits) - SUM(credits)| > $0.01.

    Three injection strategies are supported:
        1. **increase_debit** — pick a random debit line and increase its
           amount by the imbalance_amount.
        2. **decrease_credit** — pick a random credit line and decrease its
           amount by the imbalance_amount.
        3. **add_unmatched_line** — add a new debit-only line with no
           corresponding credit entry.

    Attributes:
        type_code: ``"GL-001"``
        category: ``"gl"``
        difficulty: ``"easy"``
        name: ``"Unbalanced Journal Entry"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"balance_check"``
    """

    # ------------------------------------------------------------------
    # Class-level attributes — REQUIRED by BaseDiscrepancy
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "GL-001"
    category: ClassVar[str] = "gl"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Unbalanced Journal Entry"
    description: ClassVar[str] = (
        "Journal entry where SUM(debits) != SUM(credits), "
        "violating the fundamental double-entry bookkeeping principle"
    )
    detection_method: ClassVar[str] = "balance_check"

    # Parameter bounds — moved from module-level to ClassVar per
    # BaseDiscrepancy pattern (imbalance_amount: $0.02 – $1,000)
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "imbalance_amount": {
            "min": Decimal("0.02"),
            "max": Decimal("1000.00"),
            "type": "Decimal",
        },
    }

    # ------------------------------------------------------------------
    # inject() — Core discrepancy injection method
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a debit/credit imbalance into a journal entry.

        Modifies one or more journal entry line items to create an imbalance
        where SUM(debits) != SUM(credits) by the specified imbalance_amount.

        Args:
            transaction: Journal entry transaction data containing journal
                lines under one of the recognised field names (``je_lines``,
                ``journal_lines``, ``lines``, or ``entry_lines``).
            params: Injection parameters.  Recognised keys:

                - ``imbalance_amount`` (:class:`Decimal`): Magnitude of the
                  imbalance to inject.  Range: 0.02-1000.00.  If not
                  provided, a random amount within bounds is generated
                  using *rng*.

            rng: A seeded :class:`random.Random` instance for deterministic
                behaviour.  CRITICAL: MUST use this RNG instance — NEVER
                the module-level ``random`` functions.

        Returns:
            A tuple of two dictionaries:

            - **modified_transaction** — The transaction with the injected
              debit/credit imbalance.
            - **ground_truth_data** — Dictionary containing metadata about
              the injection for the ``GroundTruthGenerator``.

        Raises:
            DiscrepancyInjectionError: If no journal entry lines are found
                in the transaction, or if the lines list is empty.
        """
        # Step 1: Deep copy to preserve original transaction
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # Step 2: Validate and normalise params against PARAMETER_BOUNDS
        validated_params: Dict[str, Any] = self._validate_params(
            params, self.PARAMETER_BOUNDS
        )

        # Step 3: Extract or generate imbalance_amount
        imbalance_amount: Decimal = self._resolve_imbalance_amount(
            validated_params, rng
        )

        # Step 4: Locate journal entry lines in the transaction dict
        lines_field, lines = self._find_lines(modified)

        if not lines:
            raise DiscrepancyInjectionError(
                "Journal entry has no line items",
                details={
                    "discrepancy_type": self.type_code,
                    "lines_field": lines_field,
                },
            )

        # Capture pre-modification sums for ground truth comparison
        original_sum_debits: Decimal = self._sum_amounts(lines, JE_DEBIT_FIELD)
        original_sum_credits: Decimal = self._sum_amounts(
            lines, JE_CREDIT_FIELD
        )

        # Step 5: Select imbalance strategy deterministically via rng
        debit_lines: List[Tuple[int, Dict[str, Any]]] = self._get_debit_lines(
            lines
        )
        credit_lines: List[Tuple[int, Dict[str, Any]]] = (
            self._get_credit_lines(lines)
        )

        # Build list of available strategies based on existing lines
        available_strategies: List[str] = []
        if debit_lines:
            available_strategies.append(_STRATEGY_INCREASE_DEBIT)
        if credit_lines:
            available_strategies.append(_STRATEGY_DECREASE_CREDIT)
        # add_unmatched_line is always available as a fallback
        available_strategies.append(_STRATEGY_ADD_UNMATCHED_LINE)

        strategy: str = rng.choice(available_strategies)

        # Step 6-7: Apply the selected strategy and track original values
        affected_fields: List[str] = []
        original_values: Dict[str, Any] = {}
        modified_values: Dict[str, Any] = {}
        line_index: int = -1

        if strategy == _STRATEGY_INCREASE_DEBIT:
            line_index, original_values, modified_values, affected_fields = (
                self._apply_increase_debit(
                    lines, debit_lines, imbalance_amount, lines_field, rng
                )
            )

        elif strategy == _STRATEGY_DECREASE_CREDIT:
            line_index, original_values, modified_values, affected_fields = (
                self._apply_decrease_credit(
                    lines, credit_lines, imbalance_amount, lines_field, rng
                )
            )

        elif strategy == _STRATEGY_ADD_UNMATCHED_LINE:
            line_index, original_values, modified_values, affected_fields = (
                self._apply_add_unmatched_line(
                    lines, imbalance_amount, lines_field
                )
            )

        # Step 8: Calculate actual post-modification sums
        sum_debits: Decimal = self._sum_amounts(lines, JE_DEBIT_FIELD)
        sum_credits: Decimal = self._sum_amounts(lines, JE_CREDIT_FIELD)
        actual_imbalance: Decimal = abs(sum_debits - sum_credits)

        # Step 9: Financial impact equals the imbalance amount
        financial_impact: Decimal = actual_imbalance

        # Step 10: Build ground truth data
        description: str = (
            f"Journal entry unbalanced by ${imbalance_amount}: "
            f"SUM(debits)=${sum_debits}, SUM(credits)=${sum_credits}"
        )

        ground_truth_data: Dict[str, Any] = self._create_ground_truth_data(
            affected_fields=affected_fields,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=financial_impact,
            description=description,
            extra_metadata={
                "imbalance_amount": str(imbalance_amount),
                "strategy": strategy,
                "sum_debits": str(sum_debits),
                "sum_credits": str(sum_credits),
                "original_sum_debits": str(original_sum_debits),
                "original_sum_credits": str(original_sum_credits),
                "line_index": line_index,
                "actual_imbalance": str(actual_imbalance),
            },
        )

        # Step 11: Log the injection event
        transaction_id: str = str(
            modified.get(
                "transaction_id",
                modified.get("je_id", modified.get("id", "unknown")),
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
            "unbalanced_journal_injected",
            service_name="transactions",
            component="UnbalancedJournal",
            type_code=self.type_code,
            category=self.category,
            difficulty=self.difficulty,
            transaction_id=transaction_id,
            imbalance_amount=str(imbalance_amount),
            strategy=strategy,
            sum_debits=str(sum_debits),
            sum_credits=str(sum_credits),
            line_index=line_index,
        )

        # Step 12: Return modified transaction and ground truth
        return modified, ground_truth_data

    # ------------------------------------------------------------------
    # Helper: _find_lines() — Locate JE lines in the transaction dict
    # ------------------------------------------------------------------
    @staticmethod
    def _find_lines(
        transaction: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Find journal entry lines in the transaction dictionary.

        Iterates through the recognised line-item field names
        (``je_lines``, ``journal_lines``, ``lines``, ``entry_lines``) and
        returns the first match along with its field name.

        Args:
            transaction: The transaction data dictionary to search.

        Returns:
            A tuple of ``(field_name, lines_list)`` where *field_name* is
            the key under which the lines were found and *lines_list* is
            the corresponding list of line-item dictionaries.

        Raises:
            DiscrepancyInjectionError: If none of the recognised field
                names exist in the transaction dictionary.
        """
        for field_name in JE_LINES_FIELDS:
            if field_name in transaction:
                lines = transaction[field_name]
                if isinstance(lines, list):
                    return field_name, lines

        raise DiscrepancyInjectionError(
            "No journal entry lines found in transaction",
            details={
                "discrepancy_type": "GL-001",
                "searched_fields": JE_LINES_FIELDS,
                "available_keys": list(transaction.keys()),
            },
        )

    # ------------------------------------------------------------------
    # Helper: _sum_amounts() — Sum a specific amount field across lines
    # ------------------------------------------------------------------
    @staticmethod
    def _sum_amounts(lines: List[Dict[str, Any]], field: str) -> Decimal:
        """Sum a specific amount field across all journal entry lines.

        Handles values that may be ``Decimal``, ``float``, ``int``, or
        string representations of numbers.  All values are converted to
        ``Decimal`` before summation to maintain financial-grade precision.

        CRITICAL: NEVER uses ``float`` arithmetic for summation.  All
        numeric values are coerced to ``Decimal`` via ``Decimal(str(v))``
        before accumulation.

        Args:
            lines: List of line-item dictionaries.
            field: The field name to sum (e.g. ``"debit_amount"``).

        Returns:
            The sum of the field values as a :class:`Decimal`.  Returns
            ``Decimal("0")`` if no lines have the specified field or if
            all values are zero/absent.
        """
        total: Decimal = Decimal("0")

        for line in lines:
            if not isinstance(line, dict):
                continue

            raw_value = line.get(field)
            if raw_value is None:
                continue

            try:
                if isinstance(raw_value, Decimal):
                    total += raw_value
                else:
                    # Convert float, int, or string to Decimal safely
                    total += Decimal(str(raw_value))
            except Exception:
                # Skip non-numeric values gracefully
                continue

        return total

    # ------------------------------------------------------------------
    # Helper: _get_debit_lines() — Filter lines with positive debits
    # ------------------------------------------------------------------
    @staticmethod
    def _get_debit_lines(
        lines: List[Dict[str, Any]],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Return (index, line) tuples for lines with debit amounts > 0.

        Checks for ``debit_amount`` first; if not present, checks for a
        positive ``amount`` field on lines that have a ``"debit"`` type
        indicator.

        Args:
            lines: List of line-item dictionaries.

        Returns:
            List of ``(index, line_dict)`` tuples for lines identified as
            having debit amounts greater than zero.
        """
        debit_lines: List[Tuple[int, Dict[str, Any]]] = []

        for idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue

            # Check explicit debit_amount field
            debit_val = line.get(JE_DEBIT_FIELD)
            if debit_val is not None:
                try:
                    if Decimal(str(debit_val)) > Decimal("0"):
                        debit_lines.append((idx, line))
                        continue
                except Exception:
                    pass

            # Fallback: check "amount" field with "debit" type indicator
            amount_val = line.get(JE_AMOUNT_FIELD)
            line_type = str(line.get("type", "")).lower()
            line_side = str(line.get("side", "")).lower()
            if amount_val is not None and (
                line_type == "debit" or line_side == "debit"
            ):
                try:
                    if Decimal(str(amount_val)) > Decimal("0"):
                        debit_lines.append((idx, line))
                except Exception:
                    pass

        return debit_lines

    # ------------------------------------------------------------------
    # Helper: _get_credit_lines() — Filter lines with positive credits
    # ------------------------------------------------------------------
    @staticmethod
    def _get_credit_lines(
        lines: List[Dict[str, Any]],
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """Return (index, line) tuples for lines with credit amounts > 0.

        Checks for ``credit_amount`` first; if not present, checks for a
        positive ``amount`` field on lines that have a ``"credit"`` type
        indicator.

        Args:
            lines: List of line-item dictionaries.

        Returns:
            List of ``(index, line_dict)`` tuples for lines identified as
            having credit amounts greater than zero.
        """
        credit_lines: List[Tuple[int, Dict[str, Any]]] = []

        for idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue

            # Check explicit credit_amount field
            credit_val = line.get(JE_CREDIT_FIELD)
            if credit_val is not None:
                try:
                    if Decimal(str(credit_val)) > Decimal("0"):
                        credit_lines.append((idx, line))
                        continue
                except Exception:
                    pass

            # Fallback: check "amount" field with "credit" type indicator
            amount_val = line.get(JE_AMOUNT_FIELD)
            line_type = str(line.get("type", "")).lower()
            line_side = str(line.get("side", "")).lower()
            if amount_val is not None and (
                line_type == "credit" or line_side == "credit"
            ):
                try:
                    if Decimal(str(amount_val)) > Decimal("0"):
                        credit_lines.append((idx, line))
                except Exception:
                    pass

        return credit_lines

    # ------------------------------------------------------------------
    # Private: _resolve_imbalance_amount()
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_imbalance_amount(
        validated_params: Dict[str, Any],
        rng: random.Random,
    ) -> Decimal:
        """Extract or generate the imbalance amount from validated params.

        If ``imbalance_amount`` is present in *validated_params*, ensures
        it is a :class:`Decimal`.  Otherwise generates a random amount
        within the configured bounds [0.02, 1000.00] using *rng*.

        Args:
            validated_params: Parameters already validated by
                ``_validate_params()``.
            rng: Seeded :class:`random.Random` for deterministic generation.

        Returns:
            The resolved imbalance amount as a :class:`Decimal`, quantized
            to two decimal places with ``ROUND_HALF_UP``.
        """
        raw_amount = validated_params.get("imbalance_amount")

        if raw_amount is not None:
            if isinstance(raw_amount, Decimal):
                return raw_amount.quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            return Decimal(str(raw_amount)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Generate random imbalance amount within bounds
        random_float: float = rng.uniform(0.02, 1000.00)
        return Decimal(str(random_float)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    # ------------------------------------------------------------------
    # Private: _apply_increase_debit()
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_increase_debit(
        lines: List[Dict[str, Any]],
        debit_lines: List[Tuple[int, Dict[str, Any]]],
        imbalance_amount: Decimal,
        lines_field: str,
        rng: random.Random,
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any], List[str]]:
        """Apply Strategy A: Increase a debit line amount.

        Selects a random debit line and increases its debit_amount by the
        specified imbalance_amount, causing SUM(debits) > SUM(credits).

        Args:
            lines: The mutable list of journal entry line dicts.
            debit_lines: Pre-filtered list of ``(index, line)`` tuples for
                lines with positive debit amounts.
            imbalance_amount: The :class:`Decimal` amount to add.
            lines_field: Name of the field containing the lines in the
                transaction (for ground truth field path construction).
            rng: Seeded :class:`random.Random` for line selection.

        Returns:
            A tuple of ``(line_index, original_values, modified_values,
            affected_fields)``.
        """
        idx, target_line = rng.choice(debit_lines)

        # Determine the debit field and current value
        field_name: str = JE_DEBIT_FIELD
        original_raw = target_line.get(JE_DEBIT_FIELD)

        # Handle the case where the line uses "amount" with type="debit"
        if original_raw is None:
            field_name = JE_AMOUNT_FIELD
            original_raw = target_line.get(JE_AMOUNT_FIELD, Decimal("0"))

        # Convert to Decimal for safe arithmetic
        original_amount: Decimal = (
            original_raw
            if isinstance(original_raw, Decimal)
            else Decimal(str(original_raw))
        )
        new_amount: Decimal = (original_amount + imbalance_amount).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Apply modification in-place (the list is already from the copied
        # transaction)
        target_line[field_name] = new_amount

        affected_field_path: str = f"{lines_field}[{idx}].{field_name}"

        return (
            idx,
            {affected_field_path: str(original_amount)},
            {affected_field_path: str(new_amount)},
            [affected_field_path],
        )

    # ------------------------------------------------------------------
    # Private: _apply_decrease_credit()
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_decrease_credit(
        lines: List[Dict[str, Any]],
        credit_lines: List[Tuple[int, Dict[str, Any]]],
        imbalance_amount: Decimal,
        lines_field: str,
        rng: random.Random,
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any], List[str]]:
        """Apply Strategy B: Decrease a credit line amount.

        Selects a random credit line and decreases its credit_amount by
        the specified imbalance_amount, causing SUM(debits) > SUM(credits).
        If the decrease would make the credit amount negative, the credit
        amount is set to ``Decimal("0.00")`` and the effective imbalance
        is the original credit amount.

        Args:
            lines: The mutable list of journal entry line dicts.
            credit_lines: Pre-filtered list of ``(index, line)`` tuples for
                lines with positive credit amounts.
            imbalance_amount: The :class:`Decimal` amount to subtract.
            lines_field: Name of the field containing the lines.
            rng: Seeded :class:`random.Random` for line selection.

        Returns:
            A tuple of ``(line_index, original_values, modified_values,
            affected_fields)``.
        """
        idx, target_line = rng.choice(credit_lines)

        # Determine the credit field and current value
        field_name: str = JE_CREDIT_FIELD
        original_raw = target_line.get(JE_CREDIT_FIELD)

        # Handle the case where the line uses "amount" with type="credit"
        if original_raw is None:
            field_name = JE_AMOUNT_FIELD
            original_raw = target_line.get(JE_AMOUNT_FIELD, Decimal("0"))

        original_amount: Decimal = (
            original_raw
            if isinstance(original_raw, Decimal)
            else Decimal(str(original_raw))
        )

        # Ensure the result does not go negative
        new_amount: Decimal = max(
            Decimal("0.00"),
            (original_amount - imbalance_amount).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
        )

        # Apply modification in-place
        target_line[field_name] = new_amount

        affected_field_path: str = f"{lines_field}[{idx}].{field_name}"

        return (
            idx,
            {affected_field_path: str(original_amount)},
            {affected_field_path: str(new_amount)},
            [affected_field_path],
        )

    # ------------------------------------------------------------------
    # Private: _apply_add_unmatched_line()
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_add_unmatched_line(
        lines: List[Dict[str, Any]],
        imbalance_amount: Decimal,
        lines_field: str,
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any], List[str]]:
        """Apply Strategy C: Add a new debit-only line.

        Appends a new line item to the journal entry with only a debit
        amount and no corresponding credit, creating an imbalance.

        Args:
            lines: The mutable list of journal entry line dicts.
            imbalance_amount: The :class:`Decimal` debit amount for the
                new line.
            lines_field: Name of the field containing the lines.

        Returns:
            A tuple of ``(new_line_index, original_values, modified_values,
            affected_fields)``.
        """
        new_line_index: int = len(lines)

        # Construct a minimal debit-only line item
        new_line: Dict[str, Any] = {
            JE_DEBIT_FIELD: imbalance_amount.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            JE_CREDIT_FIELD: Decimal("0.00"),
            "account_code": "9999",
            "description": "Unmatched debit entry",
            "line_number": new_line_index + 1,
        }

        lines.append(new_line)

        affected_field_path: str = f"{lines_field}[{new_line_index}]"

        return (
            new_line_index,
            {affected_field_path: "NOT_PRESENT"},
            {
                affected_field_path: {
                    JE_DEBIT_FIELD: str(imbalance_amount),
                    JE_CREDIT_FIELD: "0.00",
                }
            },
            [
                f"{affected_field_path}.{JE_DEBIT_FIELD}",
                f"{affected_field_path}.{JE_CREDIT_FIELD}",
            ],
        )
