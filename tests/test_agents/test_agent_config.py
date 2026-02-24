"""Comprehensive tests for AgentConfig (Pydantic V2 model) and AgentState enum.

This module validates the foundational agent configuration data structures
used throughout the Agent System (F-001). It covers:

- AgentState enum: 5 states with correct string values and lookup behaviour
- AgentConfig defaults: UUID generation and default field values
- AgentConfig custom values: all fields accept overrides including 12 agent roles
- Trait validation: 0.0-1.0 boundary enforcement for all 4 required traits
- Serialization: model_dump, JSON round-trip, and model_validate_json
- Edge cases: work hours, temperature, max_tokens boundaries
- Factory fixtures: sample_agent_config and sample_agent_config_factory

Coverage target: >=80 %
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.agents.agent_config import (
    AgentConfig,
    AgentState,
    DEFAULT_TRAITS,
    VALID_AGENT_ROLES,
    VALID_TRAIT_NAMES,
)


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_traits() -> dict[str, float]:
    """Return a valid traits dictionary matching DEFAULT_TRAITS values."""
    return {
        "thoroughness": 0.7,
        "risk_tolerance": 0.5,
        "efficiency": 0.6,
        "compliance": 0.8,
    }


@pytest.fixture
def minimal_config_kwargs() -> dict:
    """Return the minimum keyword arguments for a meaningful AgentConfig."""
    return {"role": "ap_clerk", "name": "Test Agent"}


# =========================================================================
# Phase 3: AgentState Enum
# =========================================================================

class TestAgentState:
    """Validate AgentState enum values, count, and lookup behaviour."""

    def test_idle_state(self) -> None:
        assert AgentState.IDLE.value == "idle"

    def test_thinking_state(self) -> None:
        assert AgentState.THINKING.value == "thinking"

    def test_acting_state(self) -> None:
        assert AgentState.ACTING.value == "acting"

    def test_waiting_state(self) -> None:
        assert AgentState.WAITING.value == "waiting"

    def test_error_state(self) -> None:
        assert AgentState.ERROR.value == "error"

    def test_state_count(self) -> None:
        """Exactly 5 states must be defined."""
        assert len(AgentState) == 5

    def test_state_is_str_enum(self) -> None:
        """Every member must be a string (str, Enum)."""
        for state in AgentState:
            assert isinstance(state, str)
            assert isinstance(state.value, str)

    def test_state_string_conversion(self) -> None:
        """str() on a member should produce a usable string."""
        result = str(AgentState.IDLE)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_state_from_value(self) -> None:
        """String lookup must resolve to the correct member."""
        assert AgentState("idle") is AgentState.IDLE
        assert AgentState("thinking") is AgentState.THINKING
        assert AgentState("acting") is AgentState.ACTING
        assert AgentState("waiting") is AgentState.WAITING
        assert AgentState("error") is AgentState.ERROR

    def test_invalid_state_raises_error(self) -> None:
        """Looking up a non-existent value must raise ValueError."""
        with pytest.raises(ValueError):
            AgentState("invalid")

    def test_state_equality_with_string(self) -> None:
        """AgentState(str, Enum) members should compare equal to their value."""
        assert AgentState.IDLE == "idle"
        assert AgentState.ERROR == "error"

    def test_state_iteration_values(self) -> None:
        """Iterating should yield the 5 expected string values."""
        values = {s.value for s in AgentState}
        assert values == {"idle", "thinking", "acting", "waiting", "error"}


# =========================================================================
# Phase 4: AgentConfig Creation with Defaults
# =========================================================================

class TestAgentConfigDefaults:
    """Ensure AgentConfig supplies correct defaults for every field."""

    def test_default_agent_id_is_uuid(self) -> None:
        config = AgentConfig()
        assert isinstance(config.agent_id, UUID)

    def test_default_role_is_empty(self) -> None:
        config = AgentConfig()
        assert config.role == ""

    def test_default_name_is_empty(self) -> None:
        config = AgentConfig()
        assert config.name == ""

    def test_default_employee_id_is_uuid(self) -> None:
        config = AgentConfig()
        assert isinstance(config.employee_id, UUID)

    def test_default_company_id_is_uuid(self) -> None:
        config = AgentConfig()
        assert isinstance(config.company_id, UUID)

    def test_default_traits(self, valid_traits: dict[str, float]) -> None:
        config = AgentConfig()
        assert config.traits == valid_traits

    def test_default_traits_match_constant(self) -> None:
        config = AgentConfig()
        assert config.traits == DEFAULT_TRAITS

    def test_default_work_hours_start(self) -> None:
        config = AgentConfig()
        assert config.work_hours_start == 8

    def test_default_work_hours_end(self) -> None:
        config = AgentConfig()
        assert config.work_hours_end == 17

    def test_default_temperature(self) -> None:
        config = AgentConfig()
        assert config.temperature == 0.7

    def test_default_max_tokens(self) -> None:
        config = AgentConfig()
        assert config.max_tokens == 1000

    def test_agent_ids_are_unique(self) -> None:
        c1 = AgentConfig()
        c2 = AgentConfig()
        assert c1.agent_id != c2.agent_id

    def test_employee_ids_are_unique(self) -> None:
        c1 = AgentConfig()
        c2 = AgentConfig()
        assert c1.employee_id != c2.employee_id

    def test_company_ids_are_unique(self) -> None:
        c1 = AgentConfig()
        c2 = AgentConfig()
        assert c1.company_id != c2.company_id

    def test_default_traits_are_independent_copies(self) -> None:
        """Mutating one config's traits must not affect another."""
        c1 = AgentConfig()
        c2 = AgentConfig()
        c1.traits["thoroughness"] = 0.1
        assert c2.traits["thoroughness"] == 0.7


# =========================================================================
# Phase 5: AgentConfig Custom Values
# =========================================================================

class TestAgentConfigCustomValues:
    """Verify that every field can be set to a custom value."""

    def test_custom_role(self) -> None:
        config = AgentConfig(role="ap_clerk")
        assert config.role == "ap_clerk"

    def test_custom_name(self) -> None:
        config = AgentConfig(name="Alice Johnson")
        assert config.name == "Alice Johnson"

    def test_custom_agent_id(self) -> None:
        custom_id = uuid4()
        config = AgentConfig(agent_id=custom_id)
        assert config.agent_id == custom_id

    def test_custom_employee_id(self) -> None:
        custom_id = uuid4()
        config = AgentConfig(employee_id=custom_id)
        assert config.employee_id == custom_id

    def test_custom_company_id(self) -> None:
        custom_id = uuid4()
        config = AgentConfig(company_id=custom_id)
        assert config.company_id == custom_id

    def test_custom_traits(self) -> None:
        custom = {
            "thoroughness": 0.9,
            "risk_tolerance": 0.3,
            "efficiency": 0.8,
            "compliance": 0.95,
        }
        config = AgentConfig(traits=custom)
        assert config.traits == custom

    def test_custom_work_hours(self) -> None:
        config = AgentConfig(work_hours_start=9, work_hours_end=18)
        assert config.work_hours_start == 9
        assert config.work_hours_end == 18

    def test_custom_temperature(self) -> None:
        config = AgentConfig(temperature=0.5)
        assert config.temperature == 0.5

    def test_custom_max_tokens(self) -> None:
        config = AgentConfig(max_tokens=2000)
        assert config.max_tokens == 2000

    @pytest.mark.parametrize(
        "role",
        sorted(VALID_AGENT_ROLES),
    )
    def test_all_12_valid_roles(self, role: str) -> None:
        """Every role in VALID_AGENT_ROLES must be accepted."""
        config = AgentConfig(role=role)
        assert config.role == role

    def test_valid_agent_roles_has_12_entries(self) -> None:
        assert len(VALID_AGENT_ROLES) == 12

    def test_valid_trait_names_has_4_entries(self) -> None:
        assert len(VALID_TRAIT_NAMES) == 4
        assert VALID_TRAIT_NAMES == {
            "thoroughness",
            "risk_tolerance",
            "efficiency",
            "compliance",
        }

    def test_custom_uuid_strings_accepted(self) -> None:
        """UUIDs passed as strings should be coerced."""
        uid = uuid4()
        config = AgentConfig(agent_id=str(uid))
        assert config.agent_id == uid


# =========================================================================
# Phase 6: Trait Validation (CRITICAL — 0.0 to 1.0)
# =========================================================================

class TestTraitValidation:
    """Exhaustive tests for trait boundary enforcement."""

    # --- Missing required traits ---

    def test_traits_require_thoroughness(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "risk_tolerance": 0.5,
                    "efficiency": 0.6,
                    "compliance": 0.8,
                }
            )

    def test_traits_require_risk_tolerance(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": 0.7,
                    "efficiency": 0.6,
                    "compliance": 0.8,
                }
            )

    def test_traits_require_efficiency(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": 0.7,
                    "risk_tolerance": 0.5,
                    "compliance": 0.8,
                }
            )

    def test_traits_require_compliance(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": 0.7,
                    "risk_tolerance": 0.5,
                    "efficiency": 0.6,
                }
            )

    # --- Boundary values (valid) ---

    def test_trait_value_at_minimum_0(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 0.0,
                "risk_tolerance": 0.5,
                "efficiency": 0.6,
                "compliance": 0.8,
            }
        )
        assert config.traits["thoroughness"] == 0.0

    def test_trait_value_at_maximum_1(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 1.0,
                "risk_tolerance": 0.5,
                "efficiency": 0.6,
                "compliance": 0.8,
            }
        )
        assert config.traits["thoroughness"] == 1.0

    def test_all_traits_at_minimum(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 0.0,
                "risk_tolerance": 0.0,
                "efficiency": 0.0,
                "compliance": 0.0,
            }
        )
        assert all(v == 0.0 for v in config.traits.values())

    def test_all_traits_at_maximum(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 1.0,
                "risk_tolerance": 1.0,
                "efficiency": 1.0,
                "compliance": 1.0,
            }
        )
        assert all(v == 1.0 for v in config.traits.values())

    def test_trait_boundary_0_001_valid(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 0.001,
                "risk_tolerance": 0.5,
                "efficiency": 0.6,
                "compliance": 0.8,
            }
        )
        assert config.traits["thoroughness"] == pytest.approx(0.001)

    def test_trait_boundary_0_999_valid(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 0.999,
                "risk_tolerance": 0.5,
                "efficiency": 0.6,
                "compliance": 0.8,
            }
        )
        assert config.traits["thoroughness"] == pytest.approx(0.999)

    def test_trait_mid_value_accepted(self) -> None:
        config = AgentConfig(
            traits={
                "thoroughness": 0.5,
                "risk_tolerance": 0.5,
                "efficiency": 0.5,
                "compliance": 0.5,
            }
        )
        assert config.traits["thoroughness"] == 0.5

    # --- Invalid values (rejected) ---

    def test_trait_value_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": -0.1,
                    "risk_tolerance": 0.5,
                    "efficiency": 0.6,
                    "compliance": 0.8,
                }
            )

    def test_trait_value_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": 1.1,
                    "risk_tolerance": 0.5,
                    "efficiency": 0.6,
                    "compliance": 0.8,
                }
            )

    @pytest.mark.parametrize(
        "trait_name",
        ["thoroughness", "risk_tolerance", "efficiency", "compliance"],
    )
    def test_each_trait_below_0_rejected(self, trait_name: str) -> None:
        """Each trait individually must reject values < 0."""
        base = {
            "thoroughness": 0.7,
            "risk_tolerance": 0.5,
            "efficiency": 0.6,
            "compliance": 0.8,
        }
        base[trait_name] = -0.01
        with pytest.raises(ValidationError):
            AgentConfig(traits=base)

    @pytest.mark.parametrize(
        "trait_name",
        ["thoroughness", "risk_tolerance", "efficiency", "compliance"],
    )
    def test_each_trait_above_1_rejected(self, trait_name: str) -> None:
        """Each trait individually must reject values > 1."""
        base = {
            "thoroughness": 0.7,
            "risk_tolerance": 0.5,
            "efficiency": 0.6,
            "compliance": 0.8,
        }
        base[trait_name] = 1.01
        with pytest.raises(ValidationError):
            AgentConfig(traits=base)

    def test_extra_trait_accepted(self) -> None:
        """Extra trait keys are accepted for extensibility."""
        config = AgentConfig(
            traits={
                "thoroughness": 0.7,
                "risk_tolerance": 0.5,
                "efficiency": 0.6,
                "compliance": 0.8,
                "creativity": 0.5,
            }
        )
        assert config.traits["creativity"] == 0.5
        # All required traits still present
        for key in VALID_TRAIT_NAMES:
            assert key in config.traits

    def test_empty_traits_dict_rejected(self) -> None:
        """An empty traits dict should be rejected (all 4 required missing)."""
        with pytest.raises(ValidationError):
            AgentConfig(traits={})

    def test_trait_value_large_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": -100.0,
                    "risk_tolerance": 0.5,
                    "efficiency": 0.6,
                    "compliance": 0.8,
                }
            )

    def test_trait_value_large_positive_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(
                traits={
                    "thoroughness": 100.0,
                    "risk_tolerance": 0.5,
                    "efficiency": 0.6,
                    "compliance": 0.8,
                }
            )


# =========================================================================
# Phase 7: AgentConfig Serialization
# =========================================================================

class TestAgentConfigSerialization:
    """Test model_dump, model_dump_json, and round-trip deserialization."""

    def test_model_dump_returns_dict(self) -> None:
        config = AgentConfig()
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        expected_keys = {
            "agent_id",
            "role",
            "name",
            "employee_id",
            "company_id",
            "traits",
            "work_hours_start",
            "work_hours_end",
            "temperature",
            "max_tokens",
        }
        assert set(dumped.keys()) == expected_keys

    def test_serialized_traits_is_dict(self) -> None:
        dumped = AgentConfig().model_dump()
        assert isinstance(dumped["traits"], dict)
        assert len(dumped["traits"]) == 4

    def test_model_dump_json_returns_valid_json(self) -> None:
        config = AgentConfig()
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_deserialization_from_dict(self) -> None:
        original = AgentConfig(role="accountant", name="Bob")
        dumped = original.model_dump()
        restored = AgentConfig(**dumped)
        assert restored.role == original.role
        assert restored.name == original.name
        assert restored.agent_id == original.agent_id
        assert restored.traits == original.traits

    def test_round_trip_serialization(self) -> None:
        original = AgentConfig(
            role="controller",
            name="Carol",
            traits={
                "thoroughness": 0.9,
                "risk_tolerance": 0.2,
                "efficiency": 0.7,
                "compliance": 0.95,
            },
            temperature=0.4,
            max_tokens=1500,
        )
        dumped = original.model_dump()
        restored = AgentConfig(**dumped)
        assert restored.agent_id == original.agent_id
        assert restored.role == original.role
        assert restored.name == original.name
        assert restored.traits == original.traits
        assert restored.temperature == original.temperature
        assert restored.max_tokens == original.max_tokens
        assert restored.work_hours_start == original.work_hours_start
        assert restored.work_hours_end == original.work_hours_end

    def test_json_round_trip(self) -> None:
        original = AgentConfig(role="cfo", name="Diana")
        json_str = original.model_dump_json()
        restored = AgentConfig.model_validate_json(json_str)
        assert restored.agent_id == original.agent_id
        assert restored.role == original.role
        assert restored.name == original.name
        assert restored.traits == original.traits

    def test_json_dumps_produces_string(self) -> None:
        config = AgentConfig()
        dumped = config.model_dump()
        json_string = json.dumps(dumped, default=str)
        assert isinstance(json_string, str)
        parsed_back = json.loads(json_string)
        assert "traits" in parsed_back

    def test_serialized_uuid_fields(self) -> None:
        """UUIDs should be serializable in model_dump."""
        config = AgentConfig()
        dumped = config.model_dump()
        # UUIDs may come out as UUID objects from model_dump
        assert dumped["agent_id"] is not None
        assert dumped["employee_id"] is not None
        assert dumped["company_id"] is not None


# =========================================================================
# Phase 8: AgentConfig Methods and Edge Cases
# =========================================================================

class TestAgentConfigMethods:
    """Test get_trait, is_within_work_hours, and get_llm_settings."""

    def test_get_trait_known(self) -> None:
        config = AgentConfig()
        assert config.get_trait("thoroughness") == 0.7
        assert config.get_trait("risk_tolerance") == 0.5
        assert config.get_trait("efficiency") == 0.6
        assert config.get_trait("compliance") == 0.8

    def test_get_trait_unknown_returns_default(self) -> None:
        config = AgentConfig()
        assert config.get_trait("unknown_trait") == 0.5

    def test_is_within_work_hours_inside(self) -> None:
        config = AgentConfig()  # 8-17
        assert config.is_within_work_hours(8) is True
        assert config.is_within_work_hours(12) is True
        assert config.is_within_work_hours(16) is True

    def test_is_within_work_hours_at_end_exclusive(self) -> None:
        config = AgentConfig()  # 8-17
        assert config.is_within_work_hours(17) is False

    def test_is_within_work_hours_before_start(self) -> None:
        config = AgentConfig()  # 8-17
        assert config.is_within_work_hours(7) is False
        assert config.is_within_work_hours(0) is False

    def test_is_within_work_hours_after_end(self) -> None:
        config = AgentConfig()  # 8-17
        assert config.is_within_work_hours(18) is False
        assert config.is_within_work_hours(23) is False

    def test_is_within_work_hours_custom_schedule(self) -> None:
        config = AgentConfig(work_hours_start=9, work_hours_end=18)
        assert config.is_within_work_hours(8) is False
        assert config.is_within_work_hours(9) is True
        assert config.is_within_work_hours(17) is True
        assert config.is_within_work_hours(18) is False

    def test_get_llm_settings_returns_dict(self) -> None:
        config = AgentConfig()
        settings = config.get_llm_settings()
        assert isinstance(settings, dict)
        assert "temperature" in settings
        assert "max_tokens" in settings

    def test_get_llm_settings_values(self) -> None:
        config = AgentConfig(temperature=0.3, max_tokens=500)
        settings = config.get_llm_settings()
        assert settings["temperature"] == 0.3
        assert settings["max_tokens"] == 500


class TestAgentConfigEdgeCases:
    """Edge cases and boundary conditions for AgentConfig."""

    def test_empty_role_accepted(self) -> None:
        config = AgentConfig(role="")
        assert config.role == ""

    def test_empty_name_accepted(self) -> None:
        config = AgentConfig(name="")
        assert config.name == ""

    def test_work_hours_start_0_end_23(self) -> None:
        """Extreme but valid work schedule."""
        config = AgentConfig(work_hours_start=0, work_hours_end=23)
        assert config.work_hours_start == 0
        assert config.work_hours_end == 23

    def test_work_hours_start_after_end_rejected(self) -> None:
        """start=18, end=8 must be rejected (end must be > start)."""
        with pytest.raises(ValidationError):
            AgentConfig(work_hours_start=18, work_hours_end=8)

    def test_work_hours_equal_rejected(self) -> None:
        """start=8, end=8 must be rejected (end must be > start)."""
        with pytest.raises(ValidationError):
            AgentConfig(work_hours_start=8, work_hours_end=8)

    def test_work_hours_start_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(work_hours_start=-1)

    def test_work_hours_start_above_23_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(work_hours_start=24)

    def test_work_hours_end_above_23_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(work_hours_end=24)

    def test_temperature_zero_valid(self) -> None:
        config = AgentConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_max_valid(self) -> None:
        config = AgentConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_temperature_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)

    def test_temperature_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(temperature=2.1)

    def test_max_tokens_minimum_valid(self) -> None:
        config = AgentConfig(max_tokens=1)
        assert config.max_tokens == 1

    def test_max_tokens_maximum_valid(self) -> None:
        config = AgentConfig(max_tokens=4096)
        assert config.max_tokens == 4096

    def test_max_tokens_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(max_tokens=0)

    def test_max_tokens_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(max_tokens=-1)

    def test_max_tokens_above_4096_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig(max_tokens=4097)

    def test_validate_assignment_allows_mutable_update(self) -> None:
        """Model config has validate_assignment=True — mutations are validated."""
        config = AgentConfig()
        config.role = "controller"
        assert config.role == "controller"

    def test_validate_assignment_rejects_invalid_max_tokens(self) -> None:
        config = AgentConfig()
        with pytest.raises(ValidationError):
            config.max_tokens = 0

    def test_validate_assignment_rejects_invalid_temperature(self) -> None:
        config = AgentConfig()
        with pytest.raises(ValidationError):
            config.temperature = -1.0

    def test_work_hours_minimal_gap(self) -> None:
        """Start=0, end=1 should be valid (end > start)."""
        config = AgentConfig(work_hours_start=0, work_hours_end=1)
        assert config.work_hours_start == 0
        assert config.work_hours_end == 1

    def test_work_hours_max_gap(self) -> None:
        config = AgentConfig(work_hours_start=0, work_hours_end=23)
        assert config.is_within_work_hours(0) is True
        assert config.is_within_work_hours(22) is True
        assert config.is_within_work_hours(23) is False


# =========================================================================
# Phase 9: AgentConfig Factory Pattern (conftest fixtures)
# =========================================================================

class TestAgentConfigFactory:
    """Tests using conftest.py fixtures: sample_agent_config, sample_agent_config_factory."""

    def test_sample_fixture_is_valid_config(
        self,
        sample_agent_config: AgentConfig,
    ) -> None:
        assert isinstance(sample_agent_config, AgentConfig)
        assert isinstance(sample_agent_config.agent_id, UUID)
        assert sample_agent_config.role == "ap_clerk"

    def test_sample_fixture_has_expected_name(
        self,
        sample_agent_config: AgentConfig,
    ) -> None:
        assert sample_agent_config.name == "Test AP Clerk"

    def test_sample_fixture_traits_are_default(
        self,
        sample_agent_config: AgentConfig,
    ) -> None:
        assert sample_agent_config.traits == DEFAULT_TRAITS

    def test_factory_creates_valid_config(
        self,
        sample_agent_config_factory,
    ) -> None:
        config = sample_agent_config_factory()
        assert isinstance(config, AgentConfig)
        assert isinstance(config.agent_id, UUID)

    def test_factory_custom_role(
        self,
        sample_agent_config_factory,
    ) -> None:
        config = sample_agent_config_factory(role="controller")
        assert config.role == "controller"

    def test_factory_custom_name(
        self,
        sample_agent_config_factory,
    ) -> None:
        config = sample_agent_config_factory(name="Custom Agent")
        assert config.name == "Custom Agent"

    def test_factory_custom_traits(
        self,
        sample_agent_config_factory,
    ) -> None:
        custom = {
            "thoroughness": 0.1,
            "risk_tolerance": 0.9,
            "efficiency": 0.3,
            "compliance": 0.4,
        }
        config = sample_agent_config_factory(traits=custom)
        assert config.traits == custom

    def test_factory_generates_unique_ids(
        self,
        sample_agent_config_factory,
    ) -> None:
        c1 = sample_agent_config_factory()
        c2 = sample_agent_config_factory()
        assert c1.agent_id != c2.agent_id
