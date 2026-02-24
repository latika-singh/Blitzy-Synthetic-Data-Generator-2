"""LLM configuration models using Pydantic V2.

Defines LLMConfig (provider settings, model selection, API keys, budget limits,
rate limits) and related configuration types. This is the foundational module
for the LLM integration package — all other llm modules depend on this.

Exports:
    LLMProviderType: Enum of supported LLM providers (anthropic, openai, azure_openai).
    LLMConfig: Pydantic V2 BaseModel for all LLM configuration parameters.
    SUPPORTED_MODELS: Dict mapping provider names to lists of supported model identifiers.
    MODEL_PRICING: Dict mapping model identifiers to (input_price, output_price) per 1M tokens.
    DEFAULT_RATE_LIMITS: Dict mapping provider/model to requests-per-minute limits.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

# ---------------------------------------------------------------------------
# Module-level structured logger (stdout-only per AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider Enum
# Per README.md lines 178-181: Supported providers: Anthropic, Azure OpenAI, OpenAI
# ---------------------------------------------------------------------------


class LLMProviderType(str, Enum):
    """Supported LLM provider types.

    Each value corresponds to the environment variable string used in
    ``LLM_PROVIDER`` to select the active provider at runtime.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"


# ---------------------------------------------------------------------------
# Supported Models Constant
# Per README.md lines 178-181 and AAP Section 0.1.1
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: Dict[str, List[str]] = {
    "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-20250514"],
    "openai": ["gpt-4-turbo", "gpt-3.5-turbo"],
    "azure_openai": ["gpt-4-turbo", "gpt-3.5-turbo"],
}

# ---------------------------------------------------------------------------
# Model Pricing Constant
# Per README.md lines 1157-1162 and lines 263-265
# Tuple format: (input_price_per_1M_tokens_usd, output_price_per_1M_tokens_usd)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-20250514": (0.25, 1.25),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.5, 1.5),
}

# ---------------------------------------------------------------------------
# Default Rate Limits Constant
# Per README.md lines 1881-1889 and AAP Section 0.1.2
# Nested dict: provider → model_prefix → requests_per_minute
# ---------------------------------------------------------------------------

DEFAULT_RATE_LIMITS: Dict[str, Dict[str, int]] = {
    "anthropic": {
        "claude-sonnet-4": 50,
        "claude-haiku-4": 100,
    },
    "openai": {
        "gpt-4-turbo": 500,
        "gpt-3.5-turbo": 3500,
    },
}

# ---------------------------------------------------------------------------
# LLMConfig — Pydantic V2 configuration model
# Per AAP Section 0.5.1 Group 4 and README.md lines 183-189
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Pydantic V2 configuration model for all LLM integration components.

    This is the foundational configuration class consumed by ``LLMClient``,
    ``LLMQueue``, ``LLMMonitor``, and ``PromptManager``.  It validates
    provider selection, model compatibility, budget thresholds, retry
    parameters, and rate-limit overrides at construction time.

    The ``api_key`` field uses Pydantic's ``SecretStr`` type so that it is
    never accidentally serialised into logs or JSON output (AAP §0.7.4).

    Factory methods:
        * ``from_env()`` — load from environment variables.
        * ``from_yaml(path)`` — load from a YAML file with env-var overrides.
    """

    # -- Provider & model selection ----------------------------------------

    provider: LLMProviderType = Field(
        default=LLMProviderType.ANTHROPIC,
        description="LLM provider selection (anthropic | openai | azure_openai).",
    )

    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Primary model identifier.",
    )

    fallback_model: Optional[str] = Field(
        default="claude-haiku-4-20250514",
        description="Fallback model identifier used when the primary model fails persistently.",
    )

    # -- Authentication ----------------------------------------------------

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Provider API key — NEVER logged or serialised.",
    )

    # -- Completion parameters ---------------------------------------------

    max_tokens: int = Field(
        default=1000,
        ge=1,
        le=4096,
        description="Maximum tokens per completion (default 1000).",
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (default 0.7).",
    )

    # -- Budget controls ---------------------------------------------------
    # Per AAP Section 0.1.2: $100/month, warning at 80%, block at 100%

    monthly_budget_usd: float = Field(
        default=100.0,
        gt=0,
        description="Monthly LLM budget in USD (hard cap).",
    )

    budget_warning_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Fraction of monthly budget that triggers a warning (0.80 = 80%).",
    )

    budget_block_threshold: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Fraction of monthly budget that blocks all requests (1.0 = 100%).",
    )

    # -- Retry configuration -----------------------------------------------
    # Per README.md lines 1015-1017: 3 attempts, exponential 2s–10s

    request_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Timeout per LLM completion request in seconds.",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient LLM failures.",
    )

    retry_min_wait: float = Field(
        default=2.0,
        gt=0,
        description="Minimum retry backoff wait in seconds.",
    )

    retry_max_wait: float = Field(
        default=10.0,
        gt=0,
        description="Maximum retry backoff wait in seconds.",
    )

    # -- Infrastructure ----------------------------------------------------

    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for queue and budget storage.",
    )

    rate_limit_per_minute: Optional[int] = Field(
        default=None,
        ge=1,
        description="Manual override for the per-minute request rate limit.",
    )

    # -- Pydantic V2 model configuration -----------------------------------

    model_config = {
        "str_strip_whitespace": True,
        "validate_default": True,
        "json_schema_extra": {
            "examples": [
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "fallback_model": "claude-haiku-4-20250514",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "monthly_budget_usd": 100.0,
                }
            ]
        },
    }

    # ======================================================================
    # Field Validators (Pydantic V2 syntax)
    # ======================================================================

    @field_validator("provider", mode="before")
    @classmethod
    def _validate_provider(cls, value: Any) -> Any:
        """Coerce string values to ``LLMProviderType`` and validate."""
        if isinstance(value, str):
            normalised = value.strip().lower()
            try:
                return LLMProviderType(normalised)
            except ValueError:
                valid = [p.value for p in LLMProviderType]
                raise ValueError(
                    f"Unsupported LLM provider '{value}'. "
                    f"Supported providers: {valid}"
                )
        return value

    @field_validator("model", mode="after")
    @classmethod
    def _warn_unsupported_model(cls, value: str) -> str:
        """Log a warning if the model is not in the known supported list.

        Custom / new models are still accepted — this is advisory only.
        """
        all_known: List[str] = []
        for models in SUPPORTED_MODELS.values():
            all_known.extend(models)
        if value not in all_known:
            logger.warning(
                "unsupported_model_name",
                model=value,
                known_models=all_known,
            )
        return value

    @field_validator("retry_min_wait", "retry_max_wait", mode="after")
    @classmethod
    def _validate_retry_waits(cls, value: float) -> float:
        """Ensure retry wait values are positive."""
        if value <= 0:
            raise ValueError("Retry wait times must be positive.")
        return value

    # ======================================================================
    # Model Validators (cross-field, Pydantic V2 syntax)
    # ======================================================================

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> "LLMConfig":
        """Apply cross-field validation rules after all fields are set.

        Rules enforced:
        1. ``budget_warning_threshold`` must be less than ``budget_block_threshold``.
        2. ``fallback_model`` must differ from ``model``.
        3. ``retry_min_wait`` must not exceed ``retry_max_wait``.
        """
        # 1. Budget thresholds
        if self.budget_warning_threshold >= self.budget_block_threshold:
            raise ValueError(
                f"budget_warning_threshold ({self.budget_warning_threshold}) must be "
                f"less than budget_block_threshold ({self.budget_block_threshold})."
            )

        # 2. Fallback model must differ from primary
        if self.fallback_model is not None and self.fallback_model == self.model:
            raise ValueError(
                f"fallback_model ('{self.fallback_model}') must differ from "
                f"the primary model ('{self.model}')."
            )

        # 3. Warn if fallback model is not typically cheaper
        if self.fallback_model is not None:
            primary_pricing = MODEL_PRICING.get(self.model)
            fallback_pricing = MODEL_PRICING.get(self.fallback_model)
            if (
                primary_pricing is not None
                and fallback_pricing is not None
                and fallback_pricing[0] >= primary_pricing[0]
            ):
                logger.warning(
                    "fallback_model_not_cheaper",
                    primary_model=self.model,
                    fallback_model=self.fallback_model,
                    primary_input_price=primary_pricing[0],
                    fallback_input_price=fallback_pricing[0],
                )

        # 4. Retry wait ordering
        if self.retry_min_wait > self.retry_max_wait:
            raise ValueError(
                f"retry_min_wait ({self.retry_min_wait}) must not exceed "
                f"retry_max_wait ({self.retry_max_wait})."
            )

        return self

    # ======================================================================
    # Factory Methods
    # ======================================================================

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Construct an ``LLMConfig`` from environment variables.

        Mapped variables (per README.md lines 183-189):
            * ``LLM_PROVIDER``       → provider
            * ``LLM_MODEL``          → model
            * ``LLM_FALLBACK_MODEL`` → fallback_model
            * ``LLM_API_KEY``        → api_key
            * ``LLM_MAX_TOKENS``     → max_tokens
            * ``LLM_TEMPERATURE``    → temperature
            * ``REDIS_URL``          → redis_url

        Returns:
            A fully-validated ``LLMConfig`` instance.
        """
        logger.info("llm_config_loading_from_env")
        config = cls(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            fallback_model=os.getenv("LLM_FALLBACK_MODEL", "claude-haiku-4-20250514"),
            api_key=os.getenv("LLM_API_KEY", ""),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            redis_url=os.getenv("REDIS_URL"),
        )
        logger.info(
            "llm_config_loaded_from_env",
            provider=config.provider.value,
            model=config.model,
        )
        return config

    @classmethod
    def from_yaml(cls, config_path: str) -> "LLMConfig":
        """Construct an ``LLMConfig`` from a YAML configuration file.

        Environment variables take precedence over YAML values for any
        overlapping keys, allowing deployment-time overrides.

        Args:
            config_path: Filesystem path to the YAML configuration file
                         (typically ``config/llm/llm_config.yaml``).

        Returns:
            A fully-validated ``LLMConfig`` instance.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        logger.info("llm_config_loading_from_yaml", config_path=config_path)

        with open(config_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        # Environment variables override YAML values
        env_overrides: Dict[str, Any] = {}
        env_provider = os.getenv("LLM_PROVIDER")
        if env_provider is not None:
            env_overrides["provider"] = env_provider
        env_model = os.getenv("LLM_MODEL")
        if env_model is not None:
            env_overrides["model"] = env_model
        env_fallback = os.getenv("LLM_FALLBACK_MODEL")
        if env_fallback is not None:
            env_overrides["fallback_model"] = env_fallback
        env_api_key = os.getenv("LLM_API_KEY")
        if env_api_key is not None:
            env_overrides["api_key"] = env_api_key
        env_max_tokens = os.getenv("LLM_MAX_TOKENS")
        if env_max_tokens is not None:
            env_overrides["max_tokens"] = int(env_max_tokens)
        env_temperature = os.getenv("LLM_TEMPERATURE")
        if env_temperature is not None:
            env_overrides["temperature"] = float(env_temperature)
        env_redis = os.getenv("REDIS_URL")
        if env_redis is not None:
            env_overrides["redis_url"] = env_redis

        merged = {**raw, **env_overrides}

        config = cls(**merged)
        logger.info(
            "llm_config_loaded_from_yaml",
            config_path=config_path,
            provider=config.provider.value,
            model=config.model,
            overrides_applied=list(env_overrides.keys()),
        )
        return config

    # ======================================================================
    # Utility Methods
    # ======================================================================

    def get_rate_limit(self) -> int:
        """Return the effective per-minute rate limit for the active model.

        Resolution order:
        1. ``rate_limit_per_minute`` field (explicit override).
        2. ``DEFAULT_RATE_LIMITS`` lookup by provider and model prefix.
        3. Fallback default of **50** requests/minute.
        """
        if self.rate_limit_per_minute is not None:
            return self.rate_limit_per_minute

        provider_limits = DEFAULT_RATE_LIMITS.get(self.provider.value, {})
        # Match against model prefix keys (e.g. "claude-sonnet-4" matches
        # model "claude-sonnet-4-20250514")
        for model_prefix, limit in provider_limits.items():
            if self.model.startswith(model_prefix):
                return limit

        logger.warning(
            "rate_limit_not_found_using_default",
            provider=self.provider.value,
            model=self.model,
            default_limit=50,
        )
        return 50

    def get_model_pricing(self) -> tuple[float, float]:
        """Return the pricing tuple for the active model.

        Returns:
            ``(input_price_per_1M_tokens, output_price_per_1M_tokens)`` in USD.
            Falls back to ``(0.0, 0.0)`` with a logged warning if the model
            is not present in ``MODEL_PRICING``.
        """
        pricing = MODEL_PRICING.get(self.model)
        if pricing is not None:
            return pricing

        logger.warning(
            "model_pricing_not_found",
            model=self.model,
            available_models=list(MODEL_PRICING.keys()),
        )
        return (0.0, 0.0)

    # ======================================================================
    # Serialisation Safety (AAP §0.7.4)
    # ======================================================================

    def safe_dict(self) -> Dict[str, Any]:
        """Return a config dictionary with the API key redacted.

        Suitable for structured logging and diagnostics output where the
        ``api_key`` must **never** appear in clear text.
        """
        data = self.model_dump()
        data["api_key"] = "***REDACTED***"
        # Convert LLMProviderType enum to its string value for serialisation
        if isinstance(data.get("provider"), LLMProviderType):
            data["provider"] = data["provider"].value
        return data

    def __repr__(self) -> str:
        """Custom repr that excludes the api_key for safety."""
        return (
            f"LLMConfig("
            f"provider={self.provider.value!r}, "
            f"model={self.model!r}, "
            f"fallback_model={self.fallback_model!r}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}, "
            f"monthly_budget_usd={self.monthly_budget_usd}"
            f")"
        )
