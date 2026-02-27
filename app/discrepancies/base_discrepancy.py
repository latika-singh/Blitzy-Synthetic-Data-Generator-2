"""Abstract Base Discrepancy — Contract for all 35+ discrepancy type implementations.

The ``BaseDiscrepancy`` abstract base class defines the ``inject()`` contract
that every concrete discrepancy type must implement. It provides shared
infrastructure for:

- Parameter validation against configured bounds
- Structured logging via structlog
- Ground truth data scaffolding (affected_fields, original/modified values)
- Deterministic random behavior via seeded ``random.Random`` instances

Every concrete discrepancy class (P2P-001 through CTL-005) MUST:
1. Extend ``BaseDiscrepancy``
2. Implement ``inject()`` returning ``Tuple[Dict[str, Any], Dict[str, Any]]``
3. Set the class-level ``type_code``, ``category``, and ``difficulty`` attributes
4. Use ONLY the provided seeded ``rng`` parameter — NEVER module-level random

The ``inject()`` method receives a transaction dictionary, injection parameters,
and a seeded random.Random instance. It returns a tuple of:
- Modified transaction data (Dict[str, Any])
- Ground truth data (Dict[str, Any]) with fields for GroundTruthGenerator

References:
    - AAP Section 0.5.1 Group 5: base_discrepancy.py
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.1: Pydantic V2, structlog, deterministic reproducibility
    - app/agents/base_agent.py: ABC pattern reference
"""

from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
)

import structlog

from app.transactions.exceptions import DiscrepancyInjectionError

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = ["BaseDiscrepancy"]


# ---------------------------------------------------------------------------
# Valid values for category and difficulty validation
# ---------------------------------------------------------------------------
_VALID_CATEGORIES: Tuple[str, ...] = ("p2p", "o2c", "gl", "control")
_VALID_DIFFICULTIES: Tuple[str, ...] = ("easy", "medium")


class BaseDiscrepancy(ABC):
    """Abstract base class for all discrepancy type implementations.

    Every concrete discrepancy type (P2P-001 through CTL-005) MUST extend
    this class and implement the :meth:`inject` abstract method.  The class
    provides shared infrastructure for parameter validation, structured
    logging, ground truth data creation, and deep-copy transaction handling.

    Seven class-level ``ClassVar`` attributes MUST be overridden by every
    subclass before instantiation, plus a ``PARAMETER_BOUNDS`` dict mapping
    parameter names to ``(min, max)`` tuples:

    Attributes:
        type_code: Unique type code identifying this discrepancy.  Format is
            ``"<CATEGORY>-<NNN>"``, e.g. ``"P2P-001"``, ``"O2C-003"``,
            ``"GL-001"``, ``"CTL-002"``.  MUST be set by subclasses.
        category: Discrepancy category — one of ``"p2p"``, ``"o2c"``,
            ``"gl"``, or ``"control"``.  MUST be set by subclasses.
        difficulty: Difficulty level of detection — ``"easy"`` or
            ``"medium"``.  Hard difficulty is excluded from MVP
            (distribution = 0.00).  MUST be set by subclasses.
        name: Human-readable name for this discrepancy type, e.g.
            ``"Duplicate Invoice"``.  MUST be set by subclasses.
        description: Detailed description of what the discrepancy represents
            and how it manifests in real ERP data.  MUST be set by subclasses.
        detection_method: The expected method for detecting this discrepancy,
            e.g. ``"duplicate_check"``, ``"three_way_match"``,
            ``"date_sequence"``.  MUST be set by subclasses.
        detection_difficulty: The estimated difficulty of detecting this
            discrepancy, e.g. ``"easy"`` or ``"medium"``.  Typically mirrors
            ``difficulty`` but may differ for specific detection scenarios.
            MUST be set by subclasses.
        PARAMETER_BOUNDS: Dictionary mapping parameter names to bound
            specifications: ``{"param": {"min": N, "max": M}}``.
            Used by ``_validate_params()`` as the default bounds when no
            explicit ``bounds`` argument is provided.  MUST be overridden
            by subclasses that accept injection parameters.
    """

    # ------------------------------------------------------------------
    # Class-level attributes — MUST be overridden by every subclass
    # ------------------------------------------------------------------
    type_code: ClassVar[str] = ""
    category: ClassVar[str] = ""
    difficulty: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    detection_method: ClassVar[str] = ""
    detection_difficulty: ClassVar[str] = ""

    # Parameter bounds for this discrepancy type.  Maps parameter names to
    # ``{"min": <value>, "max": <value>, "default": <optional_value>}``
    # bound specification dicts.  Subclasses MUST override with their
    # specific parameter bounds.  The ``_validate_params()`` helper defaults
    # to this dict when the ``bounds`` argument is ``None``.
    PARAMETER_BOUNDS: ClassVar[Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize the base discrepancy and validate required attributes.

        Validates that the concrete subclass has set all seven required
        class-level attributes (``type_code``, ``category``, ``difficulty``,
        ``name``, ``description``, ``detection_method``,
        ``detection_difficulty``).  Raises :class:`ValueError` if any
        attribute is missing or invalid.

        If ``detection_difficulty`` is not explicitly set by the subclass,
        it defaults to the value of ``difficulty`` for convenience.

        Raises:
            ValueError: If ``type_code`` is empty.
            ValueError: If ``category`` is not one of the valid categories.
            ValueError: If ``difficulty`` is not ``"easy"`` or ``"medium"``.
            ValueError: If ``name`` is empty.
            ValueError: If ``description`` is empty.
            ValueError: If ``detection_method`` is empty.
        """
        cls_name = self.__class__.__name__

        if not self.type_code:
            raise ValueError(
                f"{cls_name} must set 'type_code' class attribute"
            )

        if self.category not in _VALID_CATEGORIES:
            raise ValueError(
                f"{cls_name} must set 'category' to one of: "
                f"{', '.join(_VALID_CATEGORIES)}"
            )

        if self.difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"{cls_name} must set 'difficulty' to 'easy' or 'medium'"
            )

        if not self.name:
            raise ValueError(
                f"{cls_name} must set 'name' class attribute"
            )

        if not self.description:
            raise ValueError(
                f"{cls_name} must set 'description' class attribute"
            )

        if not self.detection_method:
            raise ValueError(
                f"{cls_name} must set 'detection_method' class attribute"
            )

        # Auto-derive detection_difficulty from difficulty if not explicitly
        # set by the subclass (convenience default).
        if not self.detection_difficulty:
            # Set on the instance so that subclasses that don't explicitly
            # override detection_difficulty still have a valid value.
            object.__setattr__(self, "detection_difficulty", self.difficulty)

        logger.debug(
            "discrepancy_type_initialized",
            service_name="transactions",
            component=cls_name,
            type_code=self.type_code,
            category=self.category,
            difficulty=self.difficulty,
        )

    # ------------------------------------------------------------------
    # Abstract Method: inject()
    # ------------------------------------------------------------------
    @abstractmethod
    def inject(
        self,
        transaction: Dict[str, Any],
        params: Dict[str, Any],
        rng: random.Random,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject a discrepancy into the given transaction.

        This is the core contract method that every concrete discrepancy type
        MUST implement.  It receives a transaction data dictionary, modifies
        it to introduce the discrepancy, and returns both the modified
        transaction and a ground truth data dictionary.

        Args:
            transaction: The original transaction data dictionary to modify.
                The implementation SHOULD create a copy (via
                :meth:`_copy_transaction`) before modifying to preserve the
                original.
            params: Injection parameters (e.g., ``variance_percent``,
                ``days_apart``).  These have already been validated against
                parameter bounds by the ``DiscrepancyInjector``.
            rng: A seeded :class:`random.Random` instance for deterministic
                behavior.  CRITICAL: MUST use this RNG, NEVER
                ``random.random()`` or ``random.choice()`` on the
                module-level RNG.

        Returns:
            A tuple of two dictionaries:

            - **modified_transaction** — The transaction data with the
              discrepancy injected.
            - **ground_truth_data** — Dictionary with these keys for
              ``GroundTruthGenerator``:

              - ``"type_code"``  (str)
              - ``"category"``  (str)
              - ``"difficulty"``  (str)
              - ``"affected_fields"``  (List[str])
              - ``"original_values"``  (Dict[str, Any])
              - ``"modified_values"``  (Dict[str, Any])
              - ``"detection_method"``  (str)
              - ``"detection_difficulty"``  (str)
              - ``"financial_impact"``  (Decimal)
              - ``"description"``  (str)
              - ``"name"``  (str)
              - ``"metadata"``  (Dict[str, Any])

        Raises:
            DiscrepancyInjectionError: If injection fails for any reason
                (parameter validation failure, logic error, etc.).
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Helper: _validate_params()
    # ------------------------------------------------------------------
    def _validate_params(
        self,
        params: Dict[str, Any],
        bounds: Optional[Dict[str, Dict[str, Any]]] = None,
        auto_adjust: bool = True,
    ) -> Dict[str, Any]:
        """Validate injection parameters against configured bounds.

        For each parameter name defined in *bounds*, checks whether the
        corresponding value in *params* falls within ``[min, max]``.  When
        ``auto_adjust`` is ``True``, out-of-bounds values are silently
        clamped to the nearest bound.  When ``False``, an out-of-bounds
        value raises :class:`DiscrepancyInjectionError`.

        If a parameter has a ``"default"`` key in its bound specification
        and the parameter is absent from *params*, the default value is
        applied.

        When ``bounds`` is ``None``, the class-level :attr:`PARAMETER_BOUNDS`
        dictionary is used as the default bound specification.  This enables
        subclasses to declare their bounds once as a class attribute and have
        ``_validate_params`` pick them up automatically.

        All numeric comparisons use :class:`Decimal` to ensure financial-
        grade precision per AAP §0.7.2.

        Args:
            params: Parameters to validate.  This dictionary is **not**
                modified in place — a new copy is returned.
            bounds: Mapping of parameter name → bound specification.  Each
                specification is a dictionary with at least ``"min"`` and
                ``"max"`` keys (numeric or ``Decimal``).  An optional
                ``"default"`` key provides a fallback when the parameter
                is absent from *params*.  When ``None``, defaults to
                ``self.PARAMETER_BOUNDS``.
            auto_adjust: When ``True`` (default), silently clamp out-of-
                bounds values to the nearest bound.  When ``False``, raise
                ``DiscrepancyInjectionError`` on the first violation.

        Returns:
            A new dictionary containing validated (and possibly adjusted)
            parameter values.  Only parameters present in *bounds* are
            validated; extra keys in *params* pass through unchanged.

        Raises:
            DiscrepancyInjectionError: If ``auto_adjust`` is ``False`` and
                any parameter value lies outside its configured bounds.
        """
        # Default to class-level PARAMETER_BOUNDS when no explicit bounds given
        if bounds is None:
            bounds = self.PARAMETER_BOUNDS

        validated: Dict[str, Any] = dict(params)

        for param_name, bound_spec in bounds.items():
            # Determine bound limits, coercing to Decimal for comparison
            bound_min = Decimal(str(bound_spec["min"]))
            bound_max = Decimal(str(bound_spec["max"]))

            if param_name not in validated:
                # Apply default if one is configured
                if "default" in bound_spec:
                    validated[param_name] = bound_spec["default"]
                    logger.debug(
                        "param_default_applied",
                        service_name="transactions",
                        component=self.__class__.__name__,
                        type_code=self.type_code,
                        param_name=param_name,
                        default_value=str(bound_spec["default"]),
                    )
                continue

            # Coerce current value to Decimal for comparison
            raw_value = validated[param_name]
            try:
                current_value = Decimal(str(raw_value))
            except Exception:
                # Non-numeric parameter — skip numeric bounds checking
                continue

            # Check lower bound
            if current_value < bound_min:
                if auto_adjust:
                    logger.debug(
                        "param_clamped_to_min",
                        service_name="transactions",
                        component=self.__class__.__name__,
                        type_code=self.type_code,
                        param_name=param_name,
                        original_value=str(current_value),
                        clamped_value=str(bound_min),
                    )
                    validated[param_name] = _coerce_to_original_type(
                        raw_value, bound_min
                    )
                else:
                    raise DiscrepancyInjectionError(
                        f"Parameter '{param_name}' value {current_value} is "
                        f"below minimum {bound_min} for {self.type_code}",
                        details={
                            "discrepancy_type": self.type_code,
                            "param_name": param_name,
                            "value": str(current_value),
                            "min": str(bound_min),
                            "max": str(bound_max),
                        },
                    )

            # Check upper bound
            elif current_value > bound_max:
                if auto_adjust:
                    logger.debug(
                        "param_clamped_to_max",
                        service_name="transactions",
                        component=self.__class__.__name__,
                        type_code=self.type_code,
                        param_name=param_name,
                        original_value=str(current_value),
                        clamped_value=str(bound_max),
                    )
                    validated[param_name] = _coerce_to_original_type(
                        raw_value, bound_max
                    )
                else:
                    raise DiscrepancyInjectionError(
                        f"Parameter '{param_name}' value {current_value} "
                        f"exceeds maximum {bound_max} for {self.type_code}",
                        details={
                            "discrepancy_type": self.type_code,
                            "param_name": param_name,
                            "value": str(current_value),
                            "min": str(bound_min),
                            "max": str(bound_max),
                        },
                    )

        return validated

    # ------------------------------------------------------------------
    # Helper: _create_ground_truth_data()
    # ------------------------------------------------------------------
    def _create_ground_truth_data(
        self,
        *,
        affected_fields: List[str],
        original_values: Dict[str, Any],
        modified_values: Dict[str, Any],
        financial_impact: Decimal,
        description: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a ground truth data dictionary from injection results.

        Provides a consistent ground truth structure consumed by the
        ``GroundTruthGenerator`` to create the full 16-field ground truth
        record.  All class-level attributes (type_code, category,
        difficulty, detection_method, name) are automatically included.

        Args:
            affected_fields: List of field names that were modified during
                injection.  Must be non-empty.
            original_values: Original values of the affected fields before
                injection.  Keys should align with *affected_fields*.
            modified_values: New values of the affected fields after
                injection.  Keys should align with *affected_fields*.
            financial_impact: Dollar impact of the discrepancy as a
                :class:`Decimal`.  Must use Decimal — NEVER float.
            description: Human-readable description of the specific
                discrepancy instance (e.g. "Invoice total increased
                by 15% from PO amount").
            extra_metadata: Optional dictionary of additional context to
                include in the ground truth record (e.g. related entity
                IDs, computation details).

        Returns:
            A dictionary suitable for passing to
            ``GroundTruthGenerator.create_record()``.
        """
        return {
            "type_code": self.type_code,
            "category": self.category,
            "difficulty": self.difficulty,
            "affected_fields": affected_fields,
            "original_values": original_values,
            "modified_values": modified_values,
            "detection_method": self.detection_method,
            "detection_difficulty": self.difficulty,
            "financial_impact": financial_impact,
            "description": description,
            "name": self.name,
            "metadata": extra_metadata if extra_metadata is not None else {},
        }

    # ------------------------------------------------------------------
    # Helper: _log_injection()
    # ------------------------------------------------------------------
    def _log_injection(
        self,
        transaction_id: str,
        financial_impact: Decimal,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a successful discrepancy injection event.

        Uses :mod:`structlog` with all required fields per AAP §0.7.7.
        Includes ``service_name``, ``component``, ``type_code``,
        ``category``, ``difficulty``, ``transaction_id``, and
        ``financial_impact``.  If *context* is provided and contains
        ``"simulation_id"`` or ``"trace_id"`` keys, they are included
        in the log entry.

        Args:
            transaction_id: The unique identifier of the transaction
                that received the discrepancy injection.
            financial_impact: The dollar impact of the discrepancy as
                a :class:`Decimal`.
            context: Optional dictionary of additional context.
                Recognised keys:

                - ``"simulation_id"`` — included in the log entry
                - ``"trace_id"`` — included in the log entry

                Any other keys are ignored.
        """
        log_kwargs: Dict[str, Any] = {
            "service_name": "transactions",
            "component": self.__class__.__name__,
            "type_code": self.type_code,
            "category": self.category,
            "difficulty": self.difficulty,
            "transaction_id": transaction_id,
            "financial_impact": str(financial_impact),
        }

        if context is not None:
            if "simulation_id" in context:
                log_kwargs["simulation_id"] = str(context["simulation_id"])
            if "trace_id" in context:
                log_kwargs["trace_id"] = str(context["trace_id"])

        logger.info("discrepancy_injected", **log_kwargs)

    # ------------------------------------------------------------------
    # Helper: _copy_transaction()  (static)
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of transaction data to preserve the original.

        Uses :func:`copy.deepcopy` to handle nested structures such as
        line-item lists and embedded dictionaries.  Transaction data is
        expected to be JSON-serializable (``Dict[str, Any]``), so
        ``deepcopy`` is safe for all expected data shapes.

        Args:
            transaction: The original transaction data dictionary.

        Returns:
            A deep copy of *transaction* that can be safely mutated
            without affecting the original.
        """
        return copy.deepcopy(transaction)


# ---------------------------------------------------------------------------
# Module-level helper (private)
# ---------------------------------------------------------------------------


def _coerce_to_original_type(original_value: Any, decimal_value: Decimal) -> Any:
    """Coerce a clamped Decimal bound back to the original value's type.

    When auto-adjusting parameters, we perform comparison using Decimal
    but the caller may have provided an ``int`` or ``float``.  This helper
    converts the clamped Decimal back to the same Python type as the
    original value so downstream code receives consistent types.

    If the original value is already a :class:`Decimal`, the Decimal
    is returned as-is.

    Args:
        original_value: The original parameter value (before clamping),
            used only to determine the target type.
        decimal_value: The clamped bound value as a Decimal.

    Returns:
        The clamped value coerced to match the type of *original_value*.
    """
    if isinstance(original_value, int):
        return int(decimal_value)
    if isinstance(original_value, float):
        return float(decimal_value)
    # For Decimal or any other type, return as Decimal
    return decimal_value
