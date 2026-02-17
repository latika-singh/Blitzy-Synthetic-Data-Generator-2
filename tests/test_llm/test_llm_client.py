"""Comprehensive unit tests for LLMClient — unified async LLM client.

Tests cover provider initialization, Anthropic & OpenAI completion flows,
tenacity retry logic (3 attempts, exponential backoff), automatic fallback
to cheaper model, cost calculation from MODEL_PRICING, metrics tracking,
structured logging, API key security, monitor integration, and async
context manager lifecycle.

CRITICAL (AAP §0.7.5):
- ALL LLM API calls are MOCKED — ZERO live API calls in this module.
- ALL async tests use pytest-asyncio (asyncio_mode=auto in pytest.ini).
- Coverage target: ≥80%.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from pydantic import ValidationError

from app.llm.llm_client import LLMClient, MODEL_PRICING
from app.llm.llm_config import LLMConfig, LLMProviderType
from app.llm.llm_monitor import LLMBudgetManager, LLMMonitor

# ---------------------------------------------------------------------------
# pytest.ini has asyncio_mode=auto, so async test functions are automatically
# detected.  No module-level pytestmark is needed (avoids spurious warnings
# on synchronous test functions).
# ---------------------------------------------------------------------------


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def anthropic_config() -> LLMConfig:
    """LLMConfig wired for Anthropic with a non-real test API key."""
    return LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        fallback_model="claude-haiku-4-20250514",
        api_key="test-key-not-real",
        max_tokens=1000,
        temperature=0.7,
        monthly_budget_usd=100.0,
    )


@pytest.fixture
def openai_config() -> LLMConfig:
    """LLMConfig wired for OpenAI with a non-real test API key."""
    return LLMConfig(
        provider="openai",
        model="gpt-4-turbo",
        fallback_model="gpt-3.5-turbo",
        api_key="test-key-not-real",
        max_tokens=1000,
        temperature=0.7,
        monthly_budget_usd=100.0,
    )


@pytest.fixture
def anthropic_config_no_fallback() -> LLMConfig:
    """LLMConfig for Anthropic WITHOUT a fallback model (clean retry tests)."""
    return LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        fallback_model=None,
        api_key="test-key-not-real",
        max_tokens=1000,
        temperature=0.7,
        monthly_budget_usd=100.0,
    )


@pytest.fixture
def mock_anthropic_response() -> MagicMock:
    """Simulated ``anthropic.Message`` returned by ``messages.create()``."""
    response = MagicMock()
    response.content = [
        MagicMock(text='{"decision": "approve", "confidence": 0.9}')
    ]
    response.usage = MagicMock(input_tokens=500, output_tokens=100)
    response.model = "claude-sonnet-4-20250514"
    return response


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Simulated ``openai.ChatCompletion`` returned by ``chat.completions.create()``."""
    response = MagicMock()
    response.choices = [
        MagicMock(message=MagicMock(content='{"decision": "approve"}'))
    ]
    response.usage = MagicMock(prompt_tokens=500, completion_tokens=100)
    response.model = "gpt-4-turbo"
    return response


# ============================================================
# Phase 3: Provider Initialization Tests
# ============================================================


class TestProviderInitialization:
    """Verify LLMClient constructs the correct SDK client per provider."""

    async def test_anthropic_client_initialization(self, anthropic_config: LLMConfig) -> None:
        """Anthropic provider creates ``AsyncAnthropic`` with the correct api_key."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_mod.AsyncAnthropic.return_value = MagicMock()
            client = LLMClient(config=anthropic_config)

            assert client._provider == "anthropic"
            mock_mod.AsyncAnthropic.assert_called_once_with(api_key="test-key-not-real")

            # Metrics start at zero
            metrics = client.get_metrics()
            assert metrics["total_requests"] == 0
            assert metrics["successful_requests"] == 0
            assert metrics["failed_requests"] == 0
            assert metrics["total_tokens"] == 0
            assert metrics["total_cost"] == 0.0

    async def test_openai_client_initialization(self, openai_config: LLMConfig) -> None:
        """OpenAI provider creates ``AsyncOpenAI`` with the correct api_key."""
        with patch("app.llm.llm_client.openai") as mock_mod:
            mock_mod.AsyncOpenAI.return_value = MagicMock()
            client = LLMClient(config=openai_config)

            assert client._provider == "openai"
            mock_mod.AsyncOpenAI.assert_called_once_with(api_key="test-key-not-real")

    async def test_unsupported_provider_config_raises_validation_error(self) -> None:
        """LLMConfig rejects unrecognised provider strings at construction."""
        with pytest.raises((ValueError, ValidationError)):
            LLMConfig(provider="unsupported_provider")

    async def test_unsupported_provider_in_client_raises_value_error(self) -> None:
        """LLMClient raises ``ValueError`` when the provider dispatch has no match.

        Bypasses the Pydantic enum validator by mocking the config object.
        """
        mock_config = MagicMock(spec=LLMConfig)
        mock_config.provider = MagicMock()
        mock_config.provider.value = "exotic_provider"
        mock_config.api_key = MagicMock()
        mock_config.api_key.get_secret_value.return_value = "test-key"
        mock_config.fallback_model = None
        mock_config.model = "some-model"
        mock_config.request_timeout_seconds = 30.0

        with pytest.raises(ValueError, match="Unsupported"):
            LLMClient(config=mock_config)

    async def test_azure_openai_client_initialization(self) -> None:
        """Azure OpenAI provider creates ``AsyncAzureOpenAI``."""
        config = LLMConfig(
            provider="azure_openai",
            model="gpt-4-turbo",
            fallback_model="gpt-3.5-turbo",
            api_key="test-key-not-real",
        )
        with patch("app.llm.llm_client.openai") as mock_mod:
            mock_mod.AsyncAzureOpenAI.return_value = MagicMock()
            client = LLMClient(config=config)

            assert client._provider == "azure_openai"
            mock_mod.AsyncAzureOpenAI.assert_called_once()


# ============================================================
# Phase 4: Completion Tests — Anthropic
# ============================================================


class TestAnthropicCompletion:
    """Verify Anthropic-specific completion flow via ``messages.create``."""

    async def test_complete_anthropic_success(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Successful Anthropic completion returns text and updates metrics."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config)
            result = await client.complete("test prompt", system="You are helpful")

            assert result == '{"decision": "approve", "confidence": 0.9}'

            # Verify the SDK call
            mock_sdk.messages.create.assert_called_once()
            call_kw = mock_sdk.messages.create.call_args[1]
            assert call_kw["model"] == "claude-sonnet-4-20250514"
            assert call_kw["max_tokens"] == 1000
            assert call_kw["temperature"] == 0.7
            assert call_kw["system"] == "You are helpful"
            assert call_kw["messages"] == [{"role": "user", "content": "test prompt"}]

            # Metrics
            metrics = client.get_metrics()
            assert metrics["successful_requests"] == 1
            assert metrics["total_tokens"] == 600  # 500 input + 100 output

    async def test_complete_anthropic_with_system_prompt(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """System prompt is forwarded to ``messages.create`` as ``system`` kwarg."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config)
            await client.complete("prompt", system="System context")

            call_kw = mock_sdk.messages.create.call_args[1]
            assert call_kw["system"] == "System context"

    async def test_complete_anthropic_without_system_prompt(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """When ``system`` is omitted, empty string is sent to the SDK."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config)
            await client.complete("prompt")

            call_kw = mock_sdk.messages.create.call_args[1]
            assert call_kw["system"] == ""

    async def test_complete_anthropic_custom_parameters(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Custom ``max_tokens`` and ``temperature`` are forwarded to the SDK."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config)
            await client.complete("prompt", max_tokens=2000, temperature=0.3)

            call_kw = mock_sdk.messages.create.call_args[1]
            assert call_kw["max_tokens"] == 2000
            assert call_kw["temperature"] == 0.3


# ============================================================
# Phase 5: Completion Tests — OpenAI
# ============================================================


class TestOpenAICompletion:
    """Verify OpenAI-specific completion flow via ``chat.completions.create``."""

    async def test_complete_openai_success(
        self,
        openai_config: LLMConfig,
        mock_openai_response: MagicMock,
    ) -> None:
        """Successful OpenAI completion returns content and updates metrics."""
        with patch("app.llm.llm_client.openai") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncOpenAI.return_value = mock_sdk
            mock_sdk.chat.completions.create = AsyncMock(return_value=mock_openai_response)

            client = LLMClient(config=openai_config)
            result = await client.complete("test prompt", system="You are helpful")

            assert result == '{"decision": "approve"}'

            # Verify the SDK call
            mock_sdk.chat.completions.create.assert_called_once()
            call_kw = mock_sdk.chat.completions.create.call_args[1]
            assert call_kw["model"] == "gpt-4-turbo"
            assert call_kw["max_tokens"] == 1000
            assert call_kw["temperature"] == 0.7

            # Messages array: system + user
            msgs = call_kw["messages"]
            assert msgs[0] == {"role": "system", "content": "You are helpful"}
            assert msgs[1] == {"role": "user", "content": "test prompt"}

            # Metrics
            metrics = client.get_metrics()
            assert metrics["successful_requests"] == 1
            assert metrics["total_tokens"] == 600  # 500 + 100

    async def test_complete_openai_with_system_prompt(
        self,
        openai_config: LLMConfig,
        mock_openai_response: MagicMock,
    ) -> None:
        """System prompt appears as the first element in the messages array."""
        with patch("app.llm.llm_client.openai") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncOpenAI.return_value = mock_sdk
            mock_sdk.chat.completions.create = AsyncMock(return_value=mock_openai_response)

            client = LLMClient(config=openai_config)
            await client.complete("prompt", system="System instructions")

            msgs = mock_sdk.chat.completions.create.call_args[1]["messages"]
            assert msgs[0] == {"role": "system", "content": "System instructions"}
            assert msgs[1] == {"role": "user", "content": "prompt"}

    async def test_complete_openai_without_system_prompt(
        self,
        openai_config: LLMConfig,
        mock_openai_response: MagicMock,
    ) -> None:
        """Without a system prompt only the user message is sent."""
        with patch("app.llm.llm_client.openai") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncOpenAI.return_value = mock_sdk
            mock_sdk.chat.completions.create = AsyncMock(return_value=mock_openai_response)

            client = LLMClient(config=openai_config)
            await client.complete("prompt")

            msgs = mock_sdk.chat.completions.create.call_args[1]["messages"]
            assert len(msgs) == 1
            assert msgs[0] == {"role": "user", "content": "prompt"}


# ============================================================
# Phase 6: Retry Logic Tests (tenacity)
# ============================================================


class TestRetryLogic:
    """Verify tenacity retry behaviour: 3 attempts, exponential backoff.

    Per README.md lines 1015-1019:
        stop_after_attempt(3), wait_exponential(multiplier=1, min=2, max=10)
    RuntimeError is excluded from retry (``retry_if_not_exception_type``).
    """

    async def test_retry_on_transient_failure(
        self,
        anthropic_config_no_fallback: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Succeed on the 3rd attempt after 2 transient failures."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(
                side_effect=[
                    Exception("Transient error 1"),
                    Exception("Transient error 2"),
                    mock_anthropic_response,
                ]
            )

            client = LLMClient(config=anthropic_config_no_fallback)
            result = await client.complete("test prompt")

            assert result == '{"decision": "approve", "confidence": 0.9}'
            assert mock_sdk.messages.create.call_count == 3
            # 2 failures + 1 success
            metrics = client.get_metrics()
            assert metrics["failed_requests"] == 2
            assert metrics["successful_requests"] == 1

    async def test_retry_exhaustion_raises(
        self,
        anthropic_config_no_fallback: LLMConfig,
    ) -> None:
        """All 3 attempts fail → exception is re-raised to the caller."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(
                side_effect=Exception("Persistent failure"),
            )

            client = LLMClient(config=anthropic_config_no_fallback)
            with pytest.raises(Exception, match="Persistent failure"):
                await client.complete("test prompt")

            assert client.metrics["failed_requests"] == 3
            assert mock_sdk.messages.create.call_count == 3

    async def test_runtime_error_is_not_retried(
        self,
        anthropic_config_no_fallback: LLMConfig,
    ) -> None:
        """``RuntimeError`` bypasses tenacity retry (budget exhaustion path)."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(
                side_effect=RuntimeError("Fatal — do not retry"),
            )

            client = LLMClient(config=anthropic_config_no_fallback)
            with pytest.raises(RuntimeError, match="Fatal"):
                await client.complete("test prompt")

            # Exactly 1 call — RuntimeError skips retry
            assert mock_sdk.messages.create.call_count == 1

    async def test_retry_is_configured_on_complete(
        self,
        anthropic_config: LLMConfig,
    ) -> None:
        """The ``complete`` method has the tenacity retry decorator."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_mod.AsyncAnthropic.return_value = MagicMock()
            client = LLMClient(config=anthropic_config)
            # Tenacity decorates the callable with a `.retry` attribute
            assert hasattr(client.complete, "retry")


# ============================================================
# Phase 7: Fallback Model Tests
# ============================================================


class TestFallbackLogic:
    """Verify automatic fallback to cheaper model on persistent errors.

    Per README.md lines 1056-1067: automatic fallback to config.fallback_model.
    """

    async def test_fallback_to_cheaper_model_on_failure(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Primary (Sonnet) fails → fallback (Haiku) succeeds → return content."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk

            # Primary model raises; fallback model returns the mock response
            def dispatch(**kwargs: Any) -> MagicMock:
                model = kwargs.get("model", "")
                if model == "claude-sonnet-4-20250514":
                    raise Exception("Primary model unavailable")
                return mock_anthropic_response

            mock_sdk.messages.create = AsyncMock(side_effect=dispatch)

            client = LLMClient(config=anthropic_config)
            result = await client.complete("test prompt")

            assert result == '{"decision": "approve", "confidence": 0.9}'
            # Primary attempted once + fallback attempted once
            assert mock_sdk.messages.create.call_count == 2

    async def test_no_fallback_when_already_using_fallback(
        self,
        anthropic_config: LLMConfig,
    ) -> None:
        """When ``_using_fallback`` is True, no recursive fallback is attempted."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(
                side_effect=Exception("all models fail"),
            )

            client = LLMClient(config=anthropic_config)
            client._using_fallback = True  # simulate being inside fallback

            with pytest.raises(Exception, match="all models fail"):
                await client.complete("test prompt")

            # Retried 3 times, but never tried fallback (flag was True)
            assert client.metrics["failed_requests"] == 3

    async def test_no_fallback_when_not_configured(
        self,
        anthropic_config_no_fallback: LLMConfig,
    ) -> None:
        """Without ``fallback_model``, failures re-raise directly."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(
                side_effect=Exception("Primary failure"),
            )

            client = LLMClient(config=anthropic_config_no_fallback)
            with pytest.raises(Exception, match="Primary failure"):
                await client.complete("test prompt")

            # All 3 tenacity retries hit the primary; no fallback configured
            assert mock_sdk.messages.create.call_count == 3


# ============================================================
# Phase 8: Cost Calculation Tests
# ============================================================


class TestCostCalculation:
    """Verify ``_calculate_cost`` against the ``MODEL_PRICING`` table."""

    def _make_client(self, config: LLMConfig) -> LLMClient:
        """Helper: create an LLMClient with the correct SDK mock."""
        provider = config.provider.value.lower()
        if provider == "anthropic":
            patch_target = "app.llm.llm_client.anthropic"
            sdk_class = "AsyncAnthropic"
        else:
            patch_target = "app.llm.llm_client.openai"
            sdk_class = "AsyncOpenAI"
        patcher = patch(patch_target)
        mock_mod = patcher.start()
        getattr(mock_mod, sdk_class).return_value = MagicMock()
        client = LLMClient(config=config)
        patcher.stop()
        return client

    def test_calculate_cost_anthropic_sonnet(self, anthropic_config: LLMConfig) -> None:
        """Claude Sonnet 4: $3/$15 per 1 M tokens → $0.0105 for 1 K/500."""
        client = self._make_client(anthropic_config)
        cost = client._calculate_cost(
            "claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500,
        )
        expected = (1000 / 1_000_000) * 3.0 + (500 / 1_000_000) * 15.0
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.0105) < 1e-9

    def test_calculate_cost_anthropic_haiku(self, anthropic_config: LLMConfig) -> None:
        """Claude Haiku 4: $0.25/$1.25 per 1 M tokens → $0.000875 for 1 K/500."""
        client = self._make_client(anthropic_config)
        cost = client._calculate_cost(
            "claude-haiku-4-20250514", input_tokens=1000, output_tokens=500,
        )
        expected = (1000 / 1_000_000) * 0.25 + (500 / 1_000_000) * 1.25
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.000875) < 1e-9

    def test_calculate_cost_openai_gpt4(self, openai_config: LLMConfig) -> None:
        """GPT-4 Turbo: $10/$30 per 1 M tokens → $0.025 for 1 K/500."""
        client = self._make_client(openai_config)
        cost = client._calculate_cost(
            "gpt-4-turbo", input_tokens=1000, output_tokens=500,
        )
        expected = (1000 / 1_000_000) * 10.0 + (500 / 1_000_000) * 30.0
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.025) < 1e-9

    def test_calculate_cost_unknown_model_returns_zero(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """Unrecognised model identifiers fall back to $0.00."""
        client = self._make_client(anthropic_config)
        cost = client._calculate_cost(
            "unknown-model", input_tokens=1000, output_tokens=500,
        )
        assert cost == 0.0

    def test_model_pricing_dict_contents(self) -> None:
        """``MODEL_PRICING`` contains all expected model entries."""
        expected_models = {
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250514",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        }
        assert expected_models.issubset(set(MODEL_PRICING.keys()))
        for model, (inp_price, out_price) in MODEL_PRICING.items():
            assert isinstance(inp_price, (int, float)), f"Bad input price for {model}"
            assert isinstance(out_price, (int, float)), f"Bad output price for {model}"
            assert inp_price >= 0, f"Negative input price for {model}"
            assert out_price >= 0, f"Negative output price for {model}"


# ============================================================
# Phase 9: Metrics Tracking Tests
# ============================================================


class TestMetricsTracking:
    """Verify metrics accumulation across multiple requests."""

    async def test_metrics_tracking_successful_requests(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Three successive successful completions accumulate correctly."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config)
            for _ in range(3):
                await client.complete("prompt")

            metrics = client.get_metrics()
            assert metrics["total_requests"] == 3
            assert metrics["successful_requests"] == 3
            assert metrics["failed_requests"] == 0
            assert metrics["total_tokens"] == 3 * 600
            assert metrics["total_cost"] > 0

    async def test_metrics_tracking_failed_request(
        self,
        anthropic_config_no_fallback: LLMConfig,
    ) -> None:
        """Failed requests increment ``failed_requests`` counter."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(side_effect=Exception("fail"))

            client = LLMClient(config=anthropic_config_no_fallback)
            with pytest.raises(Exception):
                await client.complete("prompt")

            metrics = client.get_metrics()
            assert metrics["failed_requests"] >= 1
            assert metrics["successful_requests"] == 0

    def test_get_metrics_returns_complete_dict(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``get_metrics()`` returns all required keys."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_mod.AsyncAnthropic.return_value = MagicMock()
            client = LLMClient(config=anthropic_config)
            metrics = client.get_metrics()

            required = {
                "total_requests",
                "successful_requests",
                "failed_requests",
                "total_tokens",
                "total_cost",
            }
            assert required.issubset(set(metrics.keys()))

    def test_get_metrics_returns_copy(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``get_metrics()`` returns a copy — mutations do not affect internals."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_mod.AsyncAnthropic.return_value = MagicMock()
            client = LLMClient(config=anthropic_config)
            metrics = client.get_metrics()
            metrics["total_requests"] = 999

            assert client.get_metrics()["total_requests"] == 0


# ============================================================
# Phase 10: Structured Logging Tests
# ============================================================


class TestStructuredLogging:
    """Verify structured log entries per AAP §0.7.6.

    "Every LLM request must produce request and response log entries including
    request_id, model, token counts, cost, duration, and success status."
    """

    async def test_record_success_logs_structured_data(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """A successful completion emits ``llm_request_success`` with all fields."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("app.llm.llm_client.logger") as mock_logger,
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config)
            await client.complete("test prompt")

            # Locate the "llm_request_success" call
            info_calls = mock_logger.info.call_args_list
            success_calls = [
                c for c in info_calls
                if c.args and c.args[0] == "llm_request_success"
            ]
            assert len(success_calls) >= 1, (
                "Expected at least one 'llm_request_success' log entry"
            )

            kwargs = success_calls[0].kwargs
            assert "model" in kwargs
            assert "input_tokens" in kwargs
            assert "output_tokens" in kwargs
            assert "cost" in kwargs
            assert "duration_seconds" in kwargs
            assert "request_id" in kwargs
            assert kwargs["success"] is True

    async def test_fallback_attempt_logs_warning(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Fallback triggers a ``llm_primary_failed_trying_fallback`` warning."""
        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("app.llm.llm_client.logger") as mock_logger,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk

            def dispatch(**kwargs: Any) -> MagicMock:
                if kwargs.get("model") == "claude-sonnet-4-20250514":
                    raise Exception("primary down")
                return mock_anthropic_response

            mock_sdk.messages.create = AsyncMock(side_effect=dispatch)

            client = LLMClient(config=anthropic_config)
            await client.complete("test")

            warn_calls = mock_logger.warning.call_args_list
            fallback_warnings = [
                c for c in warn_calls
                if c.args and c.args[0] == "llm_primary_failed_trying_fallback"
            ]
            assert len(fallback_warnings) >= 1


# ============================================================
# Phase 11: API Key Security Tests
# ============================================================


class TestAPIKeySecurity:
    """API keys must NEVER be logged or serialised (AAP §0.7.4)."""

    def test_api_key_not_in_client_repr(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """The raw API key string never appears in ``str`` or ``repr``."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_mod.AsyncAnthropic.return_value = MagicMock()
            client = LLMClient(config=anthropic_config)

            assert "test-key-not-real" not in str(client)
            assert "test-key-not-real" not in repr(client)

    def test_api_key_not_in_config_repr(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``LLMConfig`` SecretStr hides the key in ``repr``."""
        config_repr = repr(anthropic_config)
        assert "test-key-not-real" not in config_repr

    def test_api_key_not_in_config_str(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``LLMConfig`` SecretStr hides the key in ``str``."""
        config_str = str(anthropic_config)
        assert "test-key-not-real" not in config_str


# ============================================================
# Phase 12: Monitor Integration Tests
# ============================================================


class TestMonitorIntegration:
    """Verify LLMClient delegates metrics to an injected LLMMonitor."""

    async def test_monitor_records_successful_request(
        self,
        anthropic_config: LLMConfig,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """Monitor receives a success record with token counts and cost."""
        budget_manager = LLMBudgetManager(monthly_budget_usd=100.0)
        monitor = LLMMonitor(budget_manager=budget_manager)

        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(return_value=mock_anthropic_response)

            client = LLMClient(config=anthropic_config, monitor=monitor)
            await client.complete("test prompt")

            # Monitor recorded the successful request
            assert monitor.total_requests == 1
            assert monitor.successful_requests == 1
            assert monitor.total_tokens == 600  # 500 + 100
            assert monitor.total_cost > 0

            # Budget manager accumulated spend
            assert budget_manager.current_spend > 0
            assert budget_manager.get_remaining_budget() < 100.0

    async def test_monitor_budget_block_raises_runtime_error(
        self,
        anthropic_config: LLMConfig,
    ) -> None:
        """Budget exhaustion results in ``RuntimeError`` (not retried)."""
        budget_manager = LLMBudgetManager(monthly_budget_usd=0.001)
        budget_manager.record_spend(0.001)  # exhaust the budget
        monitor = LLMMonitor(budget_manager=budget_manager)

        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_mod.AsyncAnthropic.return_value = MagicMock()

            client = LLMClient(config=anthropic_config, monitor=monitor)
            with pytest.raises(RuntimeError, match="budget exhausted"):
                await client.complete("test prompt")

    async def test_monitor_records_failed_request(
        self,
        anthropic_config_no_fallback: LLMConfig,
    ) -> None:
        """Monitor receives failure records when completions fail."""
        budget_manager = LLMBudgetManager(monthly_budget_usd=100.0)
        monitor = LLMMonitor(budget_manager=budget_manager)

        with (
            patch("app.llm.llm_client.anthropic") as mock_mod,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_sdk = MagicMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk
            mock_sdk.messages.create = AsyncMock(side_effect=Exception("error"))

            client = LLMClient(
                config=anthropic_config_no_fallback, monitor=monitor,
            )
            with pytest.raises(Exception):
                await client.complete("test")

            assert monitor.failed_requests >= 1


# ============================================================
# Phase 13: Context Manager Tests
# ============================================================


class TestAsyncContextManager:
    """Verify ``async with LLMClient(...) as client:`` lifecycle."""

    async def test_async_context_manager_enter_returns_client(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``__aenter__`` returns the ``LLMClient`` instance."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_sdk.close = AsyncMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk

            async with LLMClient(config=anthropic_config) as client:
                assert isinstance(client, LLMClient)

    async def test_async_context_manager_calls_close(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``__aexit__`` calls ``close()`` on the underlying SDK client."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_sdk.close = AsyncMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk

            async with LLMClient(config=anthropic_config) as client:
                pass  # normal exit

            mock_sdk.close.assert_called_once()

    async def test_async_context_manager_close_on_exception(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """``close()`` is called even when the body raises."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_sdk.close = AsyncMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk

            with pytest.raises(ValueError, match="body error"):
                async with LLMClient(config=anthropic_config) as client:
                    raise ValueError("body error")

            mock_sdk.close.assert_called_once()

    async def test_close_idempotent(
        self, anthropic_config: LLMConfig,
    ) -> None:
        """Calling ``close()`` multiple times does not raise."""
        with patch("app.llm.llm_client.anthropic") as mock_mod:
            mock_sdk = MagicMock()
            mock_sdk.close = AsyncMock()
            mock_mod.AsyncAnthropic.return_value = mock_sdk

            client = LLMClient(config=anthropic_config)
            await client.close()
            await client.close()
            # No exception expected — the implementation handles this gracefully
