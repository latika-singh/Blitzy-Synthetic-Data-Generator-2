"""Unified async LLM client supporting Anthropic Claude and OpenAI/Azure OpenAI.

Provides retry with exponential backoff, automatic fallback to cheaper model,
cost tracking, and structured request/response logging.

Exports:
    LLMClient: Unified async LLM client with multi-provider support,
        tenacity retry (3 attempts, 2s/4s/10s exponential backoff),
        automatic fallback to cheaper model, per-request cost tracking,
        and structured logging of every request/response.
    MODEL_PRICING: Dict mapping model identifiers to
        (input_price_per_1M_tokens, output_price_per_1M_tokens) tuples.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

import structlog
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.llm.llm_config import LLMConfig

if TYPE_CHECKING:
    from app.llm.llm_monitor import LLMMonitor
    from app.llm.llm_queue import LLMQueue

# ---------------------------------------------------------------------------
# Provider SDK imports — graceful handling if not installed.
# Only the selected provider needs to be available at runtime.
# ---------------------------------------------------------------------------

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Module-level structured logger — stdout only per AAP Section 0.7.6
# ---------------------------------------------------------------------------

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Model Pricing Constant
# Per README.md lines 1157-1162 and AAP Section 0.5.1 Group 4
# Tuple format: (input_price_per_1M_tokens_usd, output_price_per_1M_tokens_usd)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-20250514": (0.25, 1.25),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-3.5-turbo": (0.5, 1.5),
}


# ---------------------------------------------------------------------------
# LLMClient
# Per README.md lines 984–1176 and AAP Section 0.5.1 Group 4
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified LLM client supporting multiple providers with retry and fallback.

    Wraps ``anthropic.AsyncAnthropic`` and ``openai.AsyncOpenAI`` behind a
    single ``complete()`` interface.  Features include:

    - **Tenacity retry**: 3 attempts with exponential back-off (2 s → 4 s → 10 s).
    - **Automatic fallback**: on persistent primary-model failure, transparently
      switches to the cheaper ``config.fallback_model``.
    - **Cost tracking**: per-request cost calculated from ``MODEL_PRICING`` and
      accumulated in ``self.metrics``.
    - **Structured logging**: every request/response logged via ``structlog``
      with model, token counts, cost, and duration (AAP §0.7.6).
    - **Budget awareness**: optional ``LLMMonitor`` integration for budget
      enforcement (AAP §0.1.2, $100/month cap).

    Constructor injection per AAP §0.7.1::

        client = LLMClient(config=llm_config, monitor=monitor, queue=queue)

    Supports ``async with`` for automatic resource cleanup::

        async with LLMClient(config) as client:
            text = await client.complete("Hello")

    Attributes:
        config: The ``LLMConfig`` instance driving provider and model selection.
        metrics: Running counters for requests, tokens, and cost.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: LLMConfig,
        monitor: Optional[LLMMonitor] = None,
        queue: Optional[LLMQueue] = None,
    ) -> None:
        """Initialise the LLM client with provider-specific async SDK.

        Args:
            config: Pydantic V2 configuration providing provider, model,
                fallback_model, api_key (SecretStr), timeouts, and retry params.
            monitor: Optional ``LLMMonitor`` for recording request metrics and
                enforcing the monthly budget cap.
            queue: Optional ``LLMQueue`` for Redis Streams-backed request
                queuing and circuit-breaker coordination.

        Raises:
            ValueError: If the provider specified in *config* is not one of
                ``anthropic``, ``openai``, or ``azure_openai``.
            RuntimeError: If the required provider SDK is not installed.
        """
        # Store injected dependencies (AAP §0.7.1: constructor injection)
        self.config: LLMConfig = config
        self._monitor: Optional[LLMMonitor] = monitor
        self._queue: Optional[LLMQueue] = queue

        # Normalise provider to lower-case string for dispatch
        self._provider: str = config.provider.value.lower()

        # --- Provider client initialisation --------------------------------
        # CRITICAL: api_key is SecretStr; extract raw value only for SDK init.
        # The raw key must NEVER be logged (AAP §0.7.4).
        raw_api_key: str = config.api_key.get_secret_value()

        if self._provider == "anthropic":
            if anthropic is None:
                raise RuntimeError(
                    "The 'anthropic' package is required for the Anthropic "
                    "provider but is not installed."
                )
            self._client: Any = anthropic.AsyncAnthropic(api_key=raw_api_key)

        elif self._provider == "openai":
            if openai is None:
                raise RuntimeError(
                    "The 'openai' package is required for the OpenAI provider "
                    "but is not installed."
                )
            self._client = openai.AsyncOpenAI(api_key=raw_api_key)

        elif self._provider == "azure_openai":
            if openai is None:
                raise RuntimeError(
                    "The 'openai' package is required for the Azure OpenAI "
                    "provider but is not installed."
                )
            # Azure-specific parameters sourced from environment variables
            # since LLMConfig does not carry them as fields.
            azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            azure_api_version: str = os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-02-01"
            )
            self._client = openai.AsyncAzureOpenAI(
                api_key=raw_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        # --- Fallback state ------------------------------------------------
        self._fallback_client: Optional[Any] = None  # lazy-created
        self._using_fallback: bool = False

        # --- Metrics -------------------------------------------------------
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

        # --- Structured log (NEVER include api_key) ------------------------
        logger.info(
            "llm_client_initialized",
            provider=self._provider,
            model=config.model,
            fallback_model=config.fallback_model,
            request_timeout_seconds=config.request_timeout_seconds,
        )

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        retry=retry_if_not_exception_type(RuntimeError),
    )
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Get a completion from the configured LLM provider.

        Automatically retries on transient failures with exponential back-off
        (3 attempts at ≈2 s, 4 s, 10 s intervals).  Falls back to the cheaper
        model on persistent errors when ``config.fallback_model`` is set.

        Args:
            prompt: The user/prompt message to send to the LLM.
            system: Optional system-level instruction prepended to the request.
            max_tokens: Maximum tokens for the completion (default 1000).
            temperature: Sampling temperature (default 0.7).

        Returns:
            The text content of the LLM response.

        Raises:
            Exception: Re-raised after all retry attempts and fallback are
                exhausted, or if the fallback itself fails.
        """
        # Generate a unique request ID for correlation logging
        request_id: str = uuid.uuid4().hex[:12]
        start_time: float = time.time()

        # --- Budget check (OUTSIDE try/except — must NOT trigger fallback) -
        # Budget exhaustion is not a transient provider error; falling back
        # to a cheaper model won't help when the monthly cap is hit.
        if self._monitor is not None:
            estimated_cost = self._calculate_cost(
                self.config.model,
                len(prompt) // 4,  # rough input token estimate
                max_tokens,
            )
            if not self._monitor.check_budget(estimated_cost):
                self.metrics["failed_requests"] += 1
                raise RuntimeError(
                    "LLM budget exhausted — request blocked by monitor."
                )

        try:
            # --- Dispatch to provider-specific implementation --------------
            if self._provider == "anthropic":
                response = await self._complete_anthropic(
                    prompt, system, max_tokens, temperature
                )
            elif self._provider in ("openai", "azure_openai"):
                response = await self._complete_openai(
                    prompt, system, max_tokens, temperature
                )
            else:
                raise ValueError(f"Unsupported provider: {self._provider}")

            # --- Record success metrics ------------------------------------
            duration: float = time.time() - start_time
            self._record_success(response, duration, request_id)

            return response["content"]

        except Exception as e:
            # --- Record failure --------------------------------------------
            self.metrics["failed_requests"] += 1
            duration = time.time() - start_time

            # Record failure in monitor if available
            if self._monitor is not None:
                self._monitor.record_request(
                    success=False,
                    latency=duration,
                    model=self.config.model,
                    request_id=request_id,
                )

            # --- Attempt fallback if configured and not already falling back
            if self.config.fallback_model and not self._is_using_fallback():
                logger.warning(
                    "llm_primary_failed_trying_fallback",
                    error=str(e),
                    primary_model=self.config.model,
                    fallback_model=self.config.fallback_model,
                    request_id=request_id,
                )
                return await self._complete_with_fallback(
                    prompt, system, max_tokens, temperature
                )

            # No fallback available — let tenacity handle retry / re-raise
            raise

    # ------------------------------------------------------------------
    # Provider-specific implementations
    # ------------------------------------------------------------------

    async def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        *,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete using the Anthropic Claude API.

        Calls ``client.messages.create()`` with the configured model and wraps
        the coroutine in ``asyncio.wait_for`` to enforce the request timeout
        (default 30 s per README.md line 1778).

        Args:
            prompt: User message text.
            system: Optional system instruction.
            max_tokens: Max completion tokens.
            temperature: Sampling temperature.
            model_override: If provided, overrides ``config.model`` (used for
                fallback completions).

        Returns:
            Dict with keys ``content``, ``input_tokens``, ``output_tokens``,
            and ``model``.
        """
        model: str = model_override or self.config.model
        timeout: float = self.config.request_timeout_seconds

        message = await asyncio.wait_for(
            self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout,
        )

        return {
            "content": message.content[0].text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "model": message.model,
        }

    async def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        *,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete using the OpenAI / Azure OpenAI API.

        Builds a messages array (optional system message + user message) and
        calls ``client.chat.completions.create()``.  The call is wrapped in
        ``asyncio.wait_for`` to enforce the request timeout.

        Args:
            prompt: User message text.
            system: Optional system instruction.
            max_tokens: Max completion tokens.
            temperature: Sampling temperature.
            model_override: If provided, overrides ``config.model`` (used for
                fallback completions).

        Returns:
            Dict with keys ``content``, ``input_tokens``, ``output_tokens``,
            and ``model``.
        """
        model: str = model_override or self.config.model
        timeout: float = self.config.request_timeout_seconds

        messages: list[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await asyncio.wait_for(
            self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=timeout,
        )

        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": response.model,
        }

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    async def _complete_with_fallback(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Attempt completion using the configured fallback model.

        Sets the internal ``_using_fallback`` flag for the duration of the
        call so that recursive fallback is prevented.  Uses the same
        underlying provider client but overrides the model identifier.

        Args:
            prompt: User message text.
            system: Optional system instruction.
            max_tokens: Max completion tokens.
            temperature: Sampling temperature.

        Returns:
            The text content of the fallback LLM response.

        Raises:
            Exception: If the fallback model also fails.
        """
        self._using_fallback = True
        request_id: str = uuid.uuid4().hex[:12]
        start_time: float = time.time()

        try:
            fallback_model: str = self.config.fallback_model or self.config.model

            if self._provider == "anthropic":
                response = await self._complete_anthropic(
                    prompt,
                    system,
                    max_tokens,
                    temperature,
                    model_override=fallback_model,
                )
            elif self._provider in ("openai", "azure_openai"):
                response = await self._complete_openai(
                    prompt,
                    system,
                    max_tokens,
                    temperature,
                    model_override=fallback_model,
                )
            else:
                raise ValueError(f"Unsupported provider: {self._provider}")

            # Record fallback success
            duration: float = time.time() - start_time
            self._record_success(response, duration, request_id)

            logger.info(
                "llm_fallback_success",
                fallback_model=fallback_model,
                duration_seconds=round(duration, 3),
                request_id=request_id,
            )

            return response["content"]

        except Exception:
            duration = time.time() - start_time
            self.metrics["failed_requests"] += 1

            if self._monitor is not None:
                self._monitor.record_request(
                    success=False,
                    latency=duration,
                    model=self.config.fallback_model or "",
                    request_id=request_id,
                )

            logger.error(
                "llm_fallback_failed",
                fallback_model=self.config.fallback_model,
                duration_seconds=round(duration, 3),
                request_id=request_id,
            )
            raise

        finally:
            self._using_fallback = False

    # ------------------------------------------------------------------
    # Metrics recording
    # ------------------------------------------------------------------

    def _record_success(
        self,
        response: Dict[str, Any],
        duration: float,
        request_id: str = "",
    ) -> None:
        """Record metrics for a successful LLM completion.

        Updates internal counters, calculates per-request cost, and emits a
        structured log entry.  If a ``LLMMonitor`` was injected, delegates
        metrics recording to it as well.

        Per AAP §0.7.6: "Every LLM request must produce request and response
        log entries including request_id, model, token counts, cost, duration,
        and success status."

        Args:
            response: Provider response dict with ``content``, ``input_tokens``,
                ``output_tokens``, and ``model`` keys.
            duration: Wall-clock duration of the request in seconds.
            request_id: Unique request identifier for log correlation.
        """
        input_tokens: int = response.get("input_tokens", 0)
        output_tokens: int = response.get("output_tokens", 0)
        model: str = response.get("model", "")

        # --- Update local metrics ------------------------------------------
        self.metrics["total_requests"] += 1
        self.metrics["successful_requests"] += 1
        self.metrics["total_tokens"] += input_tokens + output_tokens

        cost: float = self._calculate_cost(model, input_tokens, output_tokens)
        self.metrics["total_cost"] += cost

        # --- Structured log per AAP §0.7.6 ---------------------------------
        logger.info(
            "llm_request_success",
            request_id=request_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=round(cost, 6),
            duration_seconds=round(duration, 3),
            success=True,
        )

        # --- Delegate to injected monitor ----------------------------------
        if self._monitor is not None:
            self._monitor.record_request(
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency=duration,
                model=model,
                request_id=request_id,
            )

    # ------------------------------------------------------------------
    # Cost calculation
    # ------------------------------------------------------------------

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate the cost of an LLM request in USD.

        Uses the ``MODEL_PRICING`` module-level constant which maps model
        identifiers to ``(input_price, output_price)`` tuples expressed as
        USD per 1 million tokens.

        Args:
            model: The model identifier (e.g. ``"claude-sonnet-4-20250514"``).
            input_tokens: Number of prompt/input tokens consumed.
            output_tokens: Number of completion/output tokens generated.

        Returns:
            The calculated cost in USD, or ``0.0`` if the model is unknown.
        """
        if model not in MODEL_PRICING:
            logger.warning(
                "llm_unknown_model_for_pricing",
                model=model,
                known_models=list(MODEL_PRICING.keys()),
            )
            return 0.0

        input_price, output_price = MODEL_PRICING[model]

        cost: float = (
            (input_tokens / 1_000_000) * input_price
            + (output_tokens / 1_000_000) * output_price
        )
        return cost

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def _is_using_fallback(self) -> bool:
        """Return whether the client is currently operating in fallback mode.

        Returns:
            True if a fallback completion is in progress, False otherwise.
        """
        return self._using_fallback

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot copy of the client's accumulated metrics.

        Returns:
            Dictionary containing ``total_requests``, ``successful_requests``,
            ``failed_requests``, ``total_tokens``, and ``total_cost``.
        """
        return dict(self.metrics)

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying provider client connections.

        Attempts to close both the primary client and any lazily-created
        fallback client.  Logs a structured lifecycle event on completion.
        """
        # Close the primary client
        if hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception as exc:
                logger.debug(
                    "llm_client_close_error",
                    error=str(exc),
                )

        # Close the fallback client if it was created
        if self._fallback_client is not None and hasattr(
            self._fallback_client, "close"
        ):
            try:
                await self._fallback_client.close()
            except Exception as exc:
                logger.debug(
                    "llm_fallback_client_close_error",
                    error=str(exc),
                )

        logger.info("llm_client_closed", provider=self._provider)

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> LLMClient:
        """Enter the async context manager.

        Returns:
            The ``LLMClient`` instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the async context manager and release resources."""
        await self.close()
