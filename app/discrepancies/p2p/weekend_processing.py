"""P2P-008: Weekend Processing discrepancy type implementation.

Modifies the transaction processing date to fall on a Saturday or Sunday.
Normal business processing occurs Monday–Friday (08:00–17:00) per the
business calendar. Weekend processing is an unusual timing indicator that
may suggest unauthorized access or automated fraud.

Configurable Parameters:
    None — the nearest weekend date is automatically calculated.

Detection Method: date_check
    Detected by checking if the transaction processing date falls on a
    Saturday (weekday=5) or Sunday (weekday=6).

Difficulty: easy

Financial Impact: Full transaction amount (timing integrity violation).

References:
    - AAP Section 0.5.1 Group 5: P2P-008 Weekend Processing
    - app/orchestration/business_calendar.py: Working hours (08:00–17:00 Mon-Fri)
"""

from __future__ import annotations

import copy
import random
from datetime import date, timedelta
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
# Weekday name lookup for ground truth descriptions
# ---------------------------------------------------------------------------
_WEEKDAY_NAMES: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# ---------------------------------------------------------------------------
# Ordered date field candidates — first match wins
# ---------------------------------------------------------------------------
_DATE_FIELD_CANDIDATES: Tuple[str, ...] = (
    "processing_date",
    "transaction_date",
    "invoice_date",
    "created_date",
    "payment_date",
    "po_date",
    "receipt_date",
)


class WeekendProcessing(BaseDiscrepancy):
    """P2P-008: Weekend Processing discrepancy.

    Shifts a transaction's processing/transaction date so that it falls on
    a Saturday or Sunday.  The target weekend day (Saturday *or* Sunday)
    and direction (nearest preceding or following weekend) are selected
    deterministically via the seeded ``rng`` instance passed to
    :meth:`inject`.

    This discrepancy has **no configurable parameters** — the weekend date
    is calculated automatically relative to the original transaction date.

    Attributes:
        type_code: ``"P2P-008"``
        category: ``"p2p"``
        difficulty: ``"easy"``
        name: ``"Weekend Processing"``
        description: ``"Transaction processed on a Saturday or Sunday"``
        detection_method: ``"date_check"``
        PARAMETER_BOUNDS: Empty dict (no configurable parameters).
    """

    # ------------------------------------------------------------------
    # Class-level attributes (overrides from BaseDiscrepancy)
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = "P2P-008"
    category: ClassVar[str] = "p2p"
    difficulty: ClassVar[str] = "easy"
    name: ClassVar[str] = "Weekend Processing"
    description: ClassVar[str] = (
        "Transaction processed on a Saturday or Sunday"
    )
    detection_method: ClassVar[str] = "date_check"

    # No configurable parameters for this discrepancy type.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # inject() — core discrepancy injection logic
    # ------------------------------------------------------------------
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a weekend-processing discrepancy into the transaction.

        Shifts the first recognised date field to the nearest Saturday or
        Sunday.  The target weekend day and shift direction are chosen
        deterministically via the provided seeded *rng*.

        Args:
            transaction: Original transaction data dictionary.  Must contain
                at least one date field from the candidate list
                (``processing_date``, ``transaction_date``, ``invoice_date``,
                ``created_date``, ``payment_date``, ``po_date``,
                ``receipt_date``).
            params: Injection parameters — ignored for this discrepancy
                (no configurable parameters).
            rng: A seeded :class:`random.Random` instance for deterministic
                reproducibility.  MUST use this — never module-level RNG.

        Returns:
            A tuple of:
            - **modified_transaction** — Copy of *transaction* with the date
              field shifted to a weekend.
            - **ground_truth_data** — Dictionary describing the injection
              for the ``GroundTruthGenerator``.

        Raises:
            DiscrepancyInjectionError: If no recognised date field exists in
                the transaction or the date value is not a
                :class:`datetime.date`.
        """
        try:
            # 1. Deep-copy the transaction to preserve the original.
            modified: Dict[str, Any] = self._copy_transaction(transaction)

            # 2. Validate parameters (no-op for this type, but follows the
            #    contract pattern for consistency with all other discrepancies).
            self._validate_params(params, self.PARAMETER_BOUNDS)

            # 3. Locate the first available date field.
            date_field_name: Optional[str] = None
            original_date: Optional[date] = None

            for candidate in _DATE_FIELD_CANDIDATES:
                raw_value = modified.get(candidate)
                if raw_value is not None:
                    # Accept both datetime.date and ISO-format date strings.
                    resolved = _resolve_date(raw_value, candidate)
                    if resolved is not None:
                        date_field_name = candidate
                        original_date = resolved
                        break

            if date_field_name is None or original_date is None:
                raise DiscrepancyInjectionError(
                    "Transaction has no recognised date field for "
                    "weekend-processing injection",
                    details={
                        "discrepancy_type": self.type_code,
                        "transaction_id": str(
                            transaction.get("transaction_id", "unknown")
                        ),
                        "searched_fields": list(_DATE_FIELD_CANDIDATES),
                    },
                )

            # 4. Calculate modified weekend date.
            original_weekday: int = original_date.weekday()  # 0=Mon … 6=Sun

            if original_weekday >= 5:
                # Already a weekend — keep the date as-is (extremely rare for
                # legitimate transactions but fully handles the edge case).
                modified_date = original_date
            else:
                # Pick Saturday (weekday 5) or Sunday (weekday 6).
                target_weekday: int = rng.choice([5, 6])

                # Compute the next occurrence of the target weekday.
                days_forward: int = (target_weekday - original_weekday) % 7
                if days_forward == 0:
                    days_forward = 7  # pragma: no cover — defensive guard
                next_weekend: date = original_date + timedelta(
                    days=days_forward
                )

                # Compute the previous occurrence of the target weekday.
                days_backward: int = (original_weekday - target_weekday) % 7
                if days_backward == 0:
                    days_backward = 7  # pragma: no cover — defensive guard
                prev_weekend: date = original_date - timedelta(
                    days=days_backward
                )

                # Randomly pick forward or backward shift.
                modified_date = rng.choice([prev_weekend, next_weekend])

            # 5. Update the date field in the modified transaction.
            modified[date_field_name] = modified_date

            # 6. Determine financial impact — full transaction amount.
            financial_impact: Decimal = _extract_financial_impact(modified)

            # 7. Build human-readable descriptions.
            original_weekday_name: str = _WEEKDAY_NAMES[original_date.weekday()]
            modified_weekday_name: str = _WEEKDAY_NAMES[modified_date.weekday()]

            gt_description: str = (
                f"Transaction processed on {modified_weekday_name} "
                f"({modified_date.isoformat()})"
            )

            # 8. Create the ground truth data dictionary.
            ground_truth: Dict[str, Any] = self._create_ground_truth_data(
                affected_fields=[date_field_name],
                original_values={
                    date_field_name: original_date.isoformat(),
                },
                modified_values={
                    date_field_name: modified_date.isoformat(),
                },
                financial_impact=financial_impact,
                description=gt_description,
                extra_metadata={
                    "original_weekday": original_weekday_name,
                    "modified_weekday": modified_weekday_name,
                    "original_date": original_date.isoformat(),
                    "modified_date": modified_date.isoformat(),
                    "date_field_used": date_field_name,
                    "was_already_weekend": original_date.weekday() >= 5,
                },
            )

            # 9. Log the injection event.
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
                "weekend_processing_details",
                service_name="transactions",
                component="WeekendProcessing",
                type_code=self.type_code,
                date_field=date_field_name,
                original_date=original_date.isoformat(),
                modified_date=modified_date.isoformat(),
                original_weekday=original_weekday_name,
                modified_weekday=modified_weekday_name,
                financial_impact=str(financial_impact),
            )

            return modified, ground_truth

        except DiscrepancyInjectionError:
            # Re-raise known injection errors as-is.
            raise
        except Exception as exc:
            # Wrap unexpected errors so callers see a consistent exception.
            raise DiscrepancyInjectionError(
                f"Weekend processing injection failed: {exc}",
                details={
                    "discrepancy_type": self.type_code,
                    "transaction_id": str(
                        transaction.get("transaction_id", "unknown")
                    ),
                    "error": str(exc),
                },
            ) from exc


# ---------------------------------------------------------------------------
# Module-level helper functions (private)
# ---------------------------------------------------------------------------


def _resolve_date(value: Any, field_name: str) -> Optional[date]:
    """Resolve a date field value to a :class:`datetime.date`.

    Accepts:
    - ``datetime.date`` instances (returned as-is).
    - ISO-8601 date strings (``"YYYY-MM-DD"``), parsed via
      ``date.fromisoformat()``.

    Returns ``None`` for values that cannot be interpreted as dates
    (e.g., ``None``, numeric values, empty strings).
    """
    if isinstance(value, date):
        return value
    if isinstance(value, str) and len(value) >= 10:
        try:
            # Handle both "YYYY-MM-DD" and "YYYY-MM-DDTHH:MM:SS" forms.
            return date.fromisoformat(value[:10])
        except (ValueError, TypeError):
            logger.debug(
                "date_parse_failed",
                service_name="transactions",
                component="WeekendProcessing",
                field_name=field_name,
                raw_value=str(value)[:50],
            )
            return None
    return None


def _extract_financial_impact(transaction: Dict[str, Any]) -> Decimal:
    """Extract the financial impact from the transaction.

    Searches for amount fields in priority order and returns the first
    valid Decimal value found.  Falls back to ``Decimal("0")`` when no
    amount field is present, per the defensive convention used across all
    discrepancy types.

    All returned values are :class:`Decimal` — never ``float``.
    """
    amount_fields: Tuple[str, ...] = (
        "total_amount",
        "amount",
        "invoice_amount",
        "payment_amount",
        "po_amount",
        "extended_amount",
    )
    for field in amount_fields:
        raw_value = transaction.get(field)
        if raw_value is not None:
            try:
                return Decimal(str(raw_value))
            except Exception:
                continue
    return Decimal("0")
