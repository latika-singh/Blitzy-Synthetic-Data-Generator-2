"""
Agent Configuration and State module for the Agent System (F-001).

This module defines the core configuration data structures and state management
for the AI-powered agents in the Synthetic ERP Data Generation Platform.

Provides:
    - AgentState: A string enum representing agent lifecycle states
      (IDLE, THINKING, ACTING, WAITING, ERROR) with serialization support.
    - AgentConfig: A Pydantic V2 model for comprehensive agent configuration
      including identity (agent_id, role, name), personality traits
      (thoroughness, risk_tolerance, efficiency, compliance in range 0.0–1.0),
      work schedule (start/end hours), and LLM settings (temperature, max_tokens).
    - VALID_TRAIT_NAMES: Set of the 4 required personality trait names.
    - DEFAULT_TRAITS: Default trait values per the specification.
    - VALID_AGENT_ROLES: Set of the 12 valid specialized agent role identifiers.

Design Decisions:
    - Uses Pydantic V2 BaseModel (not dataclass) per AAP Section 0.7.1 mandate
      that all data contracts at subsystem boundaries use Pydantic V2 models.
    - validate_assignment=True enables runtime trait updates with validation.
    - UUID fields use uuid4 factory for automatic default generation.
    - Lightweight initialization — no heavy computation in model creation
      to support the ≥50 agents/second creation rate requirement.
    - structlog for structured JSON logging to stdout per AAP Section 0.7.6.

Performance:
    - Agent creation rate target: ≥ 50 agents/second
    - No I/O or heavy computation during AgentConfig instantiation
"""

from enum import Enum
from uuid import UUID, uuid4
from typing import Any, Dict, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

import structlog

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# AgentState Enum — README.md lines 740-745
# ---------------------------------------------------------------------------
class AgentState(str, Enum):
    """Agent lifecycle state as a string enum for serialization-friendly state management.

    States follow the agent processing lifecycle:
        IDLE     -> Agent is available and waiting for work items.
        THINKING -> Agent is evaluating a decision (statistical or LLM).
        ACTING   -> Agent is executing an action via the ActionRegistry.
        WAITING  -> Agent is waiting for external input (approval, API response).
        ERROR    -> Agent encountered an error during processing.

    Typical happy-path transition: IDLE -> THINKING -> ACTING -> IDLE
    Error path transition:         IDLE -> THINKING -> ERROR -> IDLE (after recovery)
    """

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Trait Constants
# ---------------------------------------------------------------------------

VALID_TRAIT_NAMES: Set[str] = {
    "thoroughness",
    "risk_tolerance",
    "efficiency",
    "compliance",
}
"""The four required personality trait names that every agent must possess.

Each trait is a float in the range [0.0, 1.0]:
    - thoroughness:   How carefully the agent reviews data (higher = more careful).
    - risk_tolerance: Willingness to approve borderline items (higher = bolder).
    - efficiency:     Speed/quality trade-off preference (higher = faster).
    - compliance:     Adherence to rules and policies (higher = stricter).
"""

DEFAULT_TRAITS: Dict[str, float] = {
    "thoroughness": 0.7,
    "risk_tolerance": 0.5,
    "efficiency": 0.6,
    "compliance": 0.8,
}
"""Default personality trait values per README.md lines 757-762.

These defaults represent a moderately thorough, conservative, efficient,
and highly compliant agent profile suitable as a baseline for all roles.
"""


# ---------------------------------------------------------------------------
# Valid Agent Roles — 12 specialized types (README.md lines 68-82)
# ---------------------------------------------------------------------------

VALID_AGENT_ROLES: Set[str] = {
    "ap_clerk",
    "ap_manager",
    "ar_clerk",
    "ar_manager",
    "purchasing_agent",
    "purchasing_manager",
    "warehouse_clerk",
    "warehouse_manager",
    "accountant",
    "senior_accountant",
    "controller",
    "cfo",
}
"""The 12 valid agent role identifiers matching the specialized agent types.

Each role maps to a dedicated agent subclass:
    ap_clerk           -> APClerkAgent
    ap_manager         -> APManagerAgent
    ar_clerk           -> ARClerkAgent
    ar_manager         -> ARManagerAgent
    purchasing_agent   -> PurchasingAgent
    purchasing_manager -> PurchasingManagerAgent
    warehouse_clerk    -> WarehouseClerkAgent
    warehouse_manager  -> WarehouseManagerAgent
    accountant         -> AccountantAgent
    senior_accountant  -> SeniorAccountantAgent
    controller         -> ControllerAgent
    cfo                -> CFOAgent
"""


# ---------------------------------------------------------------------------
# AgentConfig — Pydantic V2 Model (README.md lines 747-771)
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Comprehensive configuration for an ERP simulation agent.

    This Pydantic V2 model captures all parameters needed to create and operate
    an agent, including identity, personality traits, work schedule, and LLM
    settings. It serves as the primary data contract at the boundary between
    the orchestration layer and the agent system.

    Attributes:
        agent_id:         Unique identifier for the agent (auto-generated UUID4).
        role:             Agent role from VALID_AGENT_ROLES (e.g., "ap_clerk").
        name:             Human-readable agent name (e.g., "Alice Johnson").
        employee_id:      Linked employee UUID in Project 1's HR system.
        company_id:       Company context UUID for multi-tenant operation.
        traits:           Personality trait dictionary mapping trait names to
                          float values in [0.0, 1.0]. Must contain all four
                          required traits: thoroughness, risk_tolerance,
                          efficiency, compliance.
        work_hours_start: Hour of day (0-23) when the agent starts work. Default 8 (8 AM).
        work_hours_end:   Hour of day (0-23) when the agent ends work. Default 17 (5 PM).
        temperature:      LLM sampling temperature [0.0, 2.0]. Default 0.7.
        max_tokens:       Maximum tokens per LLM completion [1, 4096]. Default 1000.

    Example::

        config = AgentConfig(
            role="ap_clerk",
            name="Alice Johnson",
            traits={
                "thoroughness": 0.8,
                "risk_tolerance": 0.3,
                "efficiency": 0.7,
                "compliance": 0.9,
            },
        )
    """

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "example": {
                "role": "ap_clerk",
                "name": "Alice Johnson",
                "traits": {
                    "thoroughness": 0.8,
                    "risk_tolerance": 0.3,
                    "efficiency": 0.7,
                    "compliance": 0.9,
                },
                "work_hours_start": 8,
                "work_hours_end": 17,
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        },
    )

    # --- Identity Fields ---
    agent_id: UUID = Field(
        default_factory=uuid4,
        description="Unique agent identifier (auto-generated UUID4).",
    )
    role: str = Field(
        default="",
        description=(
            "Agent role identifier. Should be one of VALID_AGENT_ROLES "
            "(e.g., 'ap_clerk', 'purchasing_agent', 'cfo')."
        ),
    )
    name: str = Field(
        default="",
        description="Human-readable agent name (e.g., 'Alice Johnson').",
    )
    employee_id: UUID = Field(
        default_factory=uuid4,
        description="Linked employee UUID in Project 1's HR system.",
    )
    company_id: UUID = Field(
        default_factory=uuid4,
        description="Company context UUID for multi-tenant operation.",
    )

    # --- Personality Traits ---
    traits: Dict[str, float] = Field(
        default_factory=lambda: DEFAULT_TRAITS.copy(),
        description=(
            "Personality trait dictionary. Values in [0.0, 1.0]. "
            "Must include: thoroughness, risk_tolerance, efficiency, compliance."
        ),
    )

    # --- Work Schedule ---
    work_hours_start: int = Field(
        default=8,
        ge=0,
        le=23,
        description="Work start hour (0-23). Default 8 (8 AM).",
    )
    work_hours_end: int = Field(
        default=17,
        ge=0,
        le=23,
        description="Work end hour (0-23). Default 17 (5 PM).",
    )

    # --- LLM Settings ---
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature [0.0, 2.0]. Default 0.7.",
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4096,
        description="Maximum tokens per LLM completion [1, 4096]. Default 1000.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("traits")
    @classmethod
    def validate_traits(cls, value: Dict[str, float]) -> Dict[str, float]:
        """Validate personality traits dictionary.

        Ensures:
            1. All four required trait names are present.
            2. Every trait value is within the [0.0, 1.0] range.
            3. Unknown trait names are allowed for extensibility but logged.

        Args:
            value: The raw traits dictionary to validate.

        Returns:
            The validated (and possibly clamped) traits dictionary.

        Raises:
            ValueError: If required traits are missing or values are outside
                        the valid range.
        """
        # Check for missing required traits
        missing_traits = VALID_TRAIT_NAMES - set(value.keys())
        if missing_traits:
            raise ValueError(
                f"Missing required personality traits: {sorted(missing_traits)}. "
                f"All of {sorted(VALID_TRAIT_NAMES)} must be present."
            )

        # Validate value ranges and clamp if necessary
        validated: Dict[str, float] = {}
        for trait_name, trait_value in value.items():
            if not isinstance(trait_value, (int, float)):
                raise ValueError(
                    f"Trait '{trait_name}' must be a numeric value, "
                    f"got {type(trait_value).__name__}."
                )

            float_value = float(trait_value)

            if float_value < 0.0 or float_value > 1.0:
                raise ValueError(
                    f"Trait '{trait_name}' value {float_value} is outside "
                    f"the valid range [0.0, 1.0]."
                )

            validated[trait_name] = float_value

        # Log warning for unknown traits (but allow them for extensibility)
        unknown_traits = set(value.keys()) - VALID_TRAIT_NAMES
        if unknown_traits:
            logger.warning(
                "unknown_traits_detected",
                unknown_traits=sorted(unknown_traits),
                message=(
                    "Unknown trait names detected. They are allowed for "
                    "extensibility but may not be used by the decision engine."
                ),
            )

        return validated

    @field_validator("work_hours_end")
    @classmethod
    def validate_work_hours_end(cls, value: int, info: Any) -> int:
        """Validate that work_hours_end is after work_hours_start.

        Args:
            value: The work_hours_end value to validate.
            info:  Pydantic V2 validation info providing access to other fields.

        Returns:
            The validated work_hours_end value.

        Raises:
            ValueError: If work_hours_end <= work_hours_start.
        """
        # Access work_hours_start from the data being validated
        work_hours_start = info.data.get("work_hours_start", 8)
        if value <= work_hours_start:
            raise ValueError(
                f"work_hours_end ({value}) must be greater than "
                f"work_hours_start ({work_hours_start})."
            )
        return value

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_trait(self, trait_name: str) -> float:
        """Retrieve the value of a named personality trait.

        Returns the trait value if present in the traits dictionary, or a
        neutral default of 0.5 if the trait name is not found. This safe
        accessor prevents KeyError in agent decision logic when optional
        or custom traits are queried.

        Args:
            trait_name: The name of the trait to retrieve (e.g., "thoroughness").

        Returns:
            The trait value as a float in [0.0, 1.0], or 0.5 as the default.
        """
        return self.traits.get(trait_name, 0.5)

    def is_within_work_hours(self, hour: int) -> bool:
        """Check whether the given hour falls within this agent's work schedule.

        The work schedule is defined by ``work_hours_start`` (inclusive) and
        ``work_hours_end`` (exclusive), mirroring a standard business day
        (default 8 AM to 5 PM).

        Args:
            hour: The hour of day to check (0-23).

        Returns:
            True if ``work_hours_start <= hour < work_hours_end``, False otherwise.
        """
        return self.work_hours_start <= hour < self.work_hours_end

    def get_llm_settings(self) -> Dict[str, Any]:
        """Return the LLM configuration settings for this agent.

        Provides a dictionary suitable for passing to the LLMClient when
        making completions on behalf of this agent.

        Returns:
            Dictionary with keys ``temperature`` and ``max_tokens``.
        """
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
