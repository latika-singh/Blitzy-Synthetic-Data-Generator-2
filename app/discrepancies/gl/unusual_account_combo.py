"""GL-004: Unusual Account Combination — simulates illogical account pairings.

Implements the Unusual Account Combination discrepancy type that modifies
journal entry line items to use account combinations that are unusual,
unexpected, or suspicious within standard ERP accounting patterns.

Standard account categories (US GAAP):
    Assets (1xxx)     — Normal balance: DEBIT
    Liabilities (2xxx) — Normal balance: CREDIT
    Equity (3xxx)     — Normal balance: CREDIT
    Revenue (4xxx)    — Normal balance: CREDIT
    Expenses (5xxx)   — Normal balance: DEBIT

Normal Journal Entry Patterns:
    DR Expense    / CR Cash (payment)
    DR Inventory  / CR AP Accrual (receipt)
    DR AR         / CR Revenue (invoice)
    DR Cash       / CR AR (collection)
    DR AP         / CR Cash (vendor payment)
    DR COGS       / CR Inventory (shipment)

This discrepancy replaces normal account codes with unusual combinations
that violate expected accounting patterns — e.g., DR Revenue / CR Asset.

Catalog Entry:
    Type Code: GL-004
    Category: gl
    Difficulty: medium
    Detection Method: pattern_analysis
    Parameters: None (no configurable parameters)

References:
    - AAP Section 0.5.1 Group 5: GL-004 Unusual Account Combination
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.2: Normal Balance Direction enforcement
    - app/orchestration/transaction_orchestrator.py: journal_entry artifacts
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


# ═══════════════════════════════════════════════════════════════════════════
# Account Category and Combination Definitions
# ═══════════════════════════════════════════════════════════════════════════

# Account category prefixes (standard US GAAP chart of accounts).
# The first digit of an account code determines its category:
#   1xxx = Asset, 2xxx = Liability, 3xxx = Equity,
#   4xxx = Revenue, 5xxx/6xxx = Expense
ACCOUNT_CATEGORIES: Dict[str, str] = {
    "1": "asset",
    "2": "liability",
    "3": "equity",
    "4": "revenue",
    "5": "expense",
    "6": "expense",  # SGA expenses share the expense category
}

# Normal balance direction per account category (AAP §0.7.2).
# Asset and Expense accounts increase with debits.
# Liability, Equity, and Revenue accounts increase with credits.
NORMAL_BALANCE: Dict[str, str] = {
    "asset": "debit",
    "liability": "credit",
    "equity": "credit",
    "revenue": "credit",
    "expense": "debit",
}

# Unusual/suspicious account combinations: (debit_category, credit_category).
# These are pairings that rarely or never occur in normal business operations
# and represent audit red flags when found in journal entries.
UNUSUAL_COMBINATIONS: list[tuple[str, str]] = [
    ("revenue", "asset"),       # Revenue should be credited, not debited
    ("revenue", "expense"),     # Revenue offsetting expense (self-canceling)
    ("equity", "expense"),      # Equity entries rarely offset expense
    ("expense", "revenue"),     # Self-canceling (possible fraud)
    ("asset", "equity"),        # Unusual outside opening balances
    ("liability", "asset"),     # Unusual direct liability-asset offset
    ("revenue", "liability"),   # Revenue debited against liability
    ("equity", "asset"),        # Equity rarely debits against asset
]

# Sample unusual account codes per category for injection when specific
# replacement codes are needed. Each list contains representative accounts
# from the standard US GAAP chart of accounts.
SAMPLE_UNUSUAL_ACCOUNTS: Dict[str, list[str]] = {
    "revenue": ["4100", "4200", "4300"],     # Sales Revenue, Service Revenue
    "asset": ["1100", "1200", "1500"],       # Cash, AR, Fixed Assets
    "expense": ["5100", "5200", "6100"],     # COGS, Operating, SGA
    "liability": ["2100", "2200"],           # AP, AP Accrual
    "equity": ["3100", "3200"],              # Common Stock, Retained Earnings
}


# ═══════════════════════════════════════════════════════════════════════════
# Journal Entry Line Field Name Mappings
# ═══════════════════════════════════════════════════════════════════════════

# Possible field names for the journal entry lines list within a transaction.
# Multiple names are checked because different transaction generators may use
# different naming conventions.
_JE_LINES_FIELDS: list[str] = [
    "je_lines",
    "journal_lines",
    "lines",
    "entry_lines",
]

# Possible field names for account codes on individual JE line items.
_ACCOUNT_CODE_FIELDS: list[str] = [
    "account_code",
    "account_number",
    "account_id",
    "gl_account",
]

# Possible field names for account descriptions/names on JE line items.
_ACCOUNT_NAME_FIELDS: list[str] = [
    "account_name",
    "account_description",
]

# Friendly display names per category, used in account_name replacements
# when the original line has an account_name field.
_CATEGORY_DISPLAY_NAMES: Dict[str, list[str]] = {
    "revenue": ["Sales Revenue", "Service Revenue", "Other Revenue"],
    "asset": ["Cash and Equivalents", "Accounts Receivable", "Fixed Assets"],
    "expense": ["Cost of Goods Sold", "Operating Expenses", "SGA Expenses"],
    "liability": ["Accounts Payable", "Accrued Liabilities"],
    "equity": ["Common Stock", "Retained Earnings"],
}


# ═══════════════════════════════════════════════════════════════════════════
# UnusualAccountCombo Class
# ═══════════════════════════════════════════════════════════════════════════


class UnusualAccountCombo(BaseDiscrepancy):
    """GL-004: Unusual Account Combination.

    Injects an unusual account combination by replacing standard account
    codes in journal entry line items with accounts from unexpected
    categories that violate normal accounting patterns.

    This is a 'medium' difficulty discrepancy because detection requires:

    1. Understanding the Chart of Accounts category structure
    2. Knowing which account combinations are normal vs. unusual
    3. Pattern analysis across multiple journal entry lines

    The financial_impact is always ``Decimal("0")`` because this is a
    classification/coding issue — the monetary amounts on the lines remain
    unchanged; only the account codes are modified.

    Attributes:
        type_code: ``"GL-004"``
        category: ``"gl"``
        difficulty: ``"medium"``
        name: ``"Unusual Account Combination"``
        description: Human-readable description of the discrepancy.
        detection_method: ``"pattern_analysis"``
    """

    # ------------------------------------------------------------------
    # Class-level attributes (required by BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "GL-004"
    category: ClassVar[str] = "gl"
    difficulty: ClassVar[str] = "medium"
    name: ClassVar[str] = "Unusual Account Combination"
    description: ClassVar[str] = (
        "Journal entry uses account combinations that are unusual or "
        "illogical within standard ERP accounting patterns"
    )
    detection_method: ClassVar[str] = "pattern_analysis"

    # GL-004 has no configurable parameters — PARAMETER_BOUNDS stays empty
    # (inherited from BaseDiscrepancy as an empty dict).

    # ------------------------------------------------------------------
    # inject()
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject an unusual account combination into a journal entry.

        Replaces one or more account codes on journal entry line items with
        codes from unexpected categories to create illogical debit/credit
        pairings (e.g., DR Revenue / CR Asset).

        The method:
        1. Deep-copies the transaction to preserve the original.
        2. Locates journal entry lines and their account codes.
        3. Identifies debit lines and credit lines.
        4. Selects a random unusual combination from
           :data:`UNUSUAL_COMBINATIONS`.
        5. Replaces the chosen debit line's account code with one from the
           unusual *debit* category and the chosen credit line's account
           code with one from the unusual *credit* category.
        6. Constructs a ground truth record with all original/modified
           values, the chosen combination, and line indices.

        Args:
            transaction: Journal entry transaction data dictionary.
                Expected to contain a list of line items under one of the
                keys in :data:`_JE_LINES_FIELDS` (``"je_lines"``,
                ``"journal_lines"``, ``"lines"``, or ``"entry_lines"``).
            params: Injection parameters.  GL-004 has **no configurable
                parameters**; this dictionary is accepted but not used.
            rng: A seeded :class:`random.Random` instance for
                deterministic reproducibility.  MUST be the only source
                of randomness (never use module-level RNG).

        Returns:
            A tuple of two dictionaries:

            - **modified_transaction** — Deep copy of *transaction* with
              account codes replaced on the selected debit and credit
              lines.
            - **ground_truth_data** — Dictionary suitable for
              ``GroundTruthGenerator.create_record()`` with fields:
              ``type_code``, ``category``, ``difficulty``,
              ``affected_fields``, ``original_values``,
              ``modified_values``, ``detection_method``,
              ``financial_impact`` (always ``Decimal("0")``),
              ``description``, ``name``, ``metadata``.

        Raises:
            DiscrepancyInjectionError: If no journal entry lines are found
                in the transaction, if lines contain no recognisable
                account code fields, or if debit/credit lines cannot be
                identified.
        """
        # 1. Deep copy to preserve original transaction data
        modified: Dict[str, Any] = self._copy_transaction(transaction)

        # 2. Find journal entry lines
        lines_field_name, lines = self._find_je_lines(modified)

        # 3. Locate the account code field name used on these lines
        account_field = self._find_account_code_field(lines)

        # 4. Classify lines into debit and credit buckets
        debit_indices, credit_indices = self._classify_lines(lines)

        # If we have no debit OR no credit lines, we cannot create a
        # meaningful unusual combination — try a fallback: treat all lines
        # as eligible for modification.
        if not debit_indices and not credit_indices:
            raise DiscrepancyInjectionError(
                "No debit or credit lines found in journal entry",
                details={
                    "discrepancy_type": self.type_code,
                    "lines_field": lines_field_name,
                    "line_count": len(lines),
                },
            )

        # 5. Select an unusual combination
        unusual_debit_cat, unusual_credit_cat = rng.choice(
            UNUSUAL_COMBINATIONS
        )

        # Track original and modified values for ground truth
        affected_fields: List[str] = []
        original_values: Dict[str, Any] = {}
        modified_values: Dict[str, Any] = {}
        modified_line_indices: List[int] = []

        # 6a. Modify a debit line's account code
        original_debit_account: Optional[str] = None
        new_debit_account: Optional[str] = None
        if debit_indices:
            debit_idx = rng.choice(debit_indices)
            debit_line = lines[debit_idx]
            original_debit_account = str(debit_line.get(account_field, ""))
            new_debit_account = self._get_replacement_account(
                unusual_debit_cat, rng
            )
            debit_line[account_field] = new_debit_account

            field_key = f"{lines_field_name}[{debit_idx}].{account_field}"
            affected_fields.append(field_key)
            original_values[field_key] = original_debit_account
            modified_values[field_key] = new_debit_account
            modified_line_indices.append(debit_idx)

            # Update account name if present
            self._update_account_name(
                debit_line,
                unusual_debit_cat,
                rng,
                lines_field_name,
                debit_idx,
                affected_fields,
                original_values,
                modified_values,
            )
        else:
            # No debit lines — pick any line and treat it as the "debit"
            # target for the unusual combination.
            fallback_idx = rng.choice(credit_indices)
            fallback_line = lines[fallback_idx]
            original_debit_account = str(
                fallback_line.get(account_field, "")
            )
            new_debit_account = self._get_replacement_account(
                unusual_debit_cat, rng
            )
            fallback_line[account_field] = new_debit_account

            field_key = (
                f"{lines_field_name}[{fallback_idx}].{account_field}"
            )
            affected_fields.append(field_key)
            original_values[field_key] = original_debit_account
            modified_values[field_key] = new_debit_account
            modified_line_indices.append(fallback_idx)

            self._update_account_name(
                fallback_line,
                unusual_debit_cat,
                rng,
                lines_field_name,
                fallback_idx,
                affected_fields,
                original_values,
                modified_values,
            )

        # 6b. Modify a credit line's account code
        original_credit_account: Optional[str] = None
        new_credit_account: Optional[str] = None
        if credit_indices:
            credit_idx = rng.choice(credit_indices)
            credit_line = lines[credit_idx]
            original_credit_account = str(
                credit_line.get(account_field, "")
            )
            new_credit_account = self._get_replacement_account(
                unusual_credit_cat, rng
            )
            credit_line[account_field] = new_credit_account

            field_key = (
                f"{lines_field_name}[{credit_idx}].{account_field}"
            )
            affected_fields.append(field_key)
            original_values[field_key] = original_credit_account
            modified_values[field_key] = new_credit_account
            modified_line_indices.append(credit_idx)

            self._update_account_name(
                credit_line,
                unusual_credit_cat,
                rng,
                lines_field_name,
                credit_idx,
                affected_fields,
                original_values,
                modified_values,
            )
        else:
            # No credit lines — pick any line and treat it as the "credit"
            # target for the unusual combination.
            fallback_idx = rng.choice(debit_indices)
            fallback_line = lines[fallback_idx]
            original_credit_account = str(
                fallback_line.get(account_field, "")
            )
            new_credit_account = self._get_replacement_account(
                unusual_credit_cat, rng
            )
            fallback_line[account_field] = new_credit_account

            field_key = (
                f"{lines_field_name}[{fallback_idx}].{account_field}"
            )
            affected_fields.append(field_key)
            original_values[field_key] = original_credit_account
            modified_values[field_key] = new_credit_account
            modified_line_indices.append(fallback_idx)

            self._update_account_name(
                fallback_line,
                unusual_credit_cat,
                rng,
                lines_field_name,
                fallback_idx,
                affected_fields,
                original_values,
                modified_values,
            )

        # 7. Financial impact is Decimal("0") — account coding is a
        #    classification issue; the monetary amounts are unchanged.
        financial_impact = Decimal("0")

        # 8. Build the description string
        description = (
            f"Unusual account combination: "
            f"DR {unusual_debit_cat} / CR {unusual_credit_cat}"
        )

        # 9. Build ground truth data via BaseDiscrepancy helper
        ground_truth = self._create_ground_truth_data(
            affected_fields=affected_fields,
            original_values=original_values,
            modified_values=modified_values,
            financial_impact=financial_impact,
            description=description,
            extra_metadata={
                "unusual_combo": [unusual_debit_cat, unusual_credit_cat],
                "original_debit_account": (
                    original_debit_account if original_debit_account else ""
                ),
                "original_credit_account": (
                    original_credit_account
                    if original_credit_account
                    else ""
                ),
                "new_debit_account": (
                    new_debit_account if new_debit_account else ""
                ),
                "new_credit_account": (
                    new_credit_account if new_credit_account else ""
                ),
                "modified_line_indices": modified_line_indices,
                "account_code_field": account_field,
                "lines_field": lines_field_name,
            },
        )

        # 10. Log the injection event
        transaction_id = str(
            transaction.get(
                "transaction_id",
                transaction.get("je_id", transaction.get("id", "unknown")),
            )
        )
        self._log_injection(
            transaction_id=transaction_id,
            financial_impact=financial_impact,
            context={
                "unusual_combo": f"DR {unusual_debit_cat} / CR {unusual_credit_cat}",
                "modified_line_indices": modified_line_indices,
            },
        )

        return modified, ground_truth

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize_account(account_code: str) -> Optional[str]:
        """Determine the account category from the account code prefix.

        Uses the first digit of the account code string to look up the
        category in :data:`ACCOUNT_CATEGORIES`.

        Args:
            account_code: Account number string (e.g., ``"1100"``,
                ``"4200"``, ``"6100"``).

        Returns:
            Category string (``"asset"``, ``"liability"``, ``"equity"``,
            ``"revenue"``, ``"expense"``) or ``None`` if the first digit
            is not a recognised prefix.
        """
        if not account_code or not isinstance(account_code, str):
            return None
        first_digit = account_code[0]
        return ACCOUNT_CATEGORIES.get(first_digit)

    @staticmethod
    def _get_replacement_account(
        target_category: str,
        rng: random.Random,
    ) -> str:
        """Get a replacement account code from the target category.

        Selects a random account code from
        :data:`SAMPLE_UNUSUAL_ACCOUNTS` for the specified category.

        Args:
            target_category: The desired account category
                (``"revenue"``, ``"asset"``, ``"expense"``,
                ``"liability"``, ``"equity"``).
            rng: A seeded :class:`random.Random` instance.

        Returns:
            A representative account code string from the target
            category (e.g., ``"4100"`` for revenue).

        Raises:
            DiscrepancyInjectionError: If *target_category* is not found
                in :data:`SAMPLE_UNUSUAL_ACCOUNTS`.
        """
        account_list = SAMPLE_UNUSUAL_ACCOUNTS.get(target_category)
        if not account_list:
            raise DiscrepancyInjectionError(
                f"No sample accounts available for category "
                f"'{target_category}'",
                details={
                    "discrepancy_type": "GL-004",
                    "target_category": target_category,
                    "available_categories": list(
                        SAMPLE_UNUSUAL_ACCOUNTS.keys()
                    ),
                },
            )
        return rng.choice(account_list)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_je_lines(
        transaction: Dict[str, Any],
    ) -> Tuple[str, list]:
        """Locate journal entry lines in the transaction dictionary.

        Iterates :data:`_JE_LINES_FIELDS` and returns the first matching
        key with its associated list of line-item dictionaries.

        Args:
            transaction: The (already deep-copied) transaction dict.

        Returns:
            Tuple of ``(field_name, lines_list)`` where *field_name* is
            the key that was found and *lines_list* is the list value.

        Raises:
            DiscrepancyInjectionError: If no recognised JE lines field is
                found or the lines list is empty.
        """
        for field_name in _JE_LINES_FIELDS:
            lines = transaction.get(field_name)
            if lines is not None and isinstance(lines, list) and lines:
                return field_name, lines

        raise DiscrepancyInjectionError(
            "No journal entry lines found in transaction",
            details={
                "discrepancy_type": "GL-004",
                "checked_fields": _JE_LINES_FIELDS,
                "transaction_keys": list(transaction.keys()),
            },
        )

    @staticmethod
    def _find_account_code_field(lines: list) -> str:
        """Identify which account code field name is used on the lines.

        Scans the first non-empty line-item dictionary for any key in
        :data:`_ACCOUNT_CODE_FIELDS`.

        Args:
            lines: List of journal entry line-item dictionaries.

        Returns:
            The field name string (e.g., ``"account_code"``).

        Raises:
            DiscrepancyInjectionError: If no recognisable account code
                field is found on any line item.
        """
        for line in lines:
            if not isinstance(line, dict):
                continue
            for field_name in _ACCOUNT_CODE_FIELDS:
                if field_name in line:
                    return field_name

        raise DiscrepancyInjectionError(
            "No account codes found on journal entry lines",
            details={
                "discrepancy_type": "GL-004",
                "checked_fields": _ACCOUNT_CODE_FIELDS,
                "line_count": len(lines),
            },
        )

    @staticmethod
    def _classify_lines(lines: list) -> Tuple[List[int], List[int]]:
        """Classify JE lines into debit and credit buckets by index.

        A line is classified as a **debit line** if it has a positive
        ``debit_amount`` or ``amount`` > 0 with ``entry_type``/``type``
        == ``"debit"``.  A line is classified as a **credit line** if it
        has a positive ``credit_amount`` or ``amount`` > 0 with
        ``entry_type``/``type`` == ``"credit"``.

        Args:
            lines: List of journal entry line-item dictionaries.

        Returns:
            Tuple of ``(debit_indices, credit_indices)`` — lists of
            integer indices into *lines*.
        """
        debit_indices: List[int] = []
        credit_indices: List[int] = []

        for idx, line in enumerate(lines):
            if not isinstance(line, dict):
                continue

            # Check explicit debit_amount / credit_amount fields
            debit_amt = line.get("debit_amount", Decimal("0"))
            credit_amt = line.get("credit_amount", Decimal("0"))

            # Coerce to Decimal for comparison
            try:
                debit_amt = Decimal(str(debit_amt)) if debit_amt else Decimal("0")
            except Exception:
                debit_amt = Decimal("0")
            try:
                credit_amt = Decimal(str(credit_amt)) if credit_amt else Decimal("0")
            except Exception:
                credit_amt = Decimal("0")

            if debit_amt > Decimal("0"):
                debit_indices.append(idx)
            elif credit_amt > Decimal("0"):
                credit_indices.append(idx)
            else:
                # Fall back to entry_type / type field classification
                entry_type = str(
                    line.get("entry_type", line.get("type", ""))
                ).lower()
                if entry_type == "debit":
                    debit_indices.append(idx)
                elif entry_type == "credit":
                    credit_indices.append(idx)
                else:
                    # Check generic amount field with sign convention
                    generic_amount = line.get("amount")
                    if generic_amount is not None:
                        try:
                            amt = Decimal(str(generic_amount))
                            if amt > Decimal("0"):
                                debit_indices.append(idx)
                            elif amt < Decimal("0"):
                                credit_indices.append(idx)
                        except Exception:
                            pass

        return debit_indices, credit_indices

    @staticmethod
    def _update_account_name(
        line: Dict[str, Any],
        target_category: str,
        rng: random.Random,
        lines_field_name: str,
        line_idx: int,
        affected_fields: List[str],
        original_values: Dict[str, Any],
        modified_values: Dict[str, Any],
    ) -> None:
        """Update the account name on a line if an account_name field exists.

        If the line has a recognisable account name field (from
        :data:`_ACCOUNT_NAME_FIELDS`), replaces the name with a
        category-appropriate display name from
        :data:`_CATEGORY_DISPLAY_NAMES`.

        Args:
            line: The journal entry line-item dictionary (mutated in place).
            target_category: The new account category (e.g., ``"revenue"``).
            rng: Seeded random instance for name selection.
            lines_field_name: Parent field name for ground truth tracking.
            line_idx: Line index within the parent list.
            affected_fields: Accumulator list for ground truth.
            original_values: Accumulator dict for ground truth.
            modified_values: Accumulator dict for ground truth.
        """
        for name_field in _ACCOUNT_NAME_FIELDS:
            if name_field in line:
                original_name = str(line[name_field])
                display_names = _CATEGORY_DISPLAY_NAMES.get(
                    target_category, [target_category.title()]
                )
                new_name = rng.choice(display_names)
                line[name_field] = new_name

                field_key = (
                    f"{lines_field_name}[{line_idx}].{name_field}"
                )
                affected_fields.append(field_key)
                original_values[field_key] = original_name
                modified_values[field_key] = new_name
                break  # Only update the first matching name field
