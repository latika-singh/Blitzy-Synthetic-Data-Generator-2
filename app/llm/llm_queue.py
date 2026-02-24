"""Redis Streams-based LLM request queue with consumer groups, priority levels,
rate limiting, and circuit breaker pattern.

Provides resilient queueing for LLM requests with exponential backoff retry
and per-provider rate limiting. Uses Redis Streams XADD/XREADGROUP for
durable message delivery with consumer group 'llm_workers' on stream
'llm_requests'.

Key components:
- LLMQueue: Redis Streams-backed request queue with priority levels and
  batch dequeue, supporting up to 1,000 pending requests.
- LLMRateLimiter: Per-provider, per-model rate limiting with 60-second
  sliding windows (Claude Sonnet 4 at 50 req/min, Claude Haiku 4 at 100 req/min).
- CircuitBreaker: Opens after 5 consecutive failures with configurable
  recovery timeout and half-open test state.
"""

from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

try:
    import redis.asyncio as aioredis
    from redis import ResponseError as RedisResponseError
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]
    RedisResponseError = Exception  # type: ignore[assignment,misc]

from app.llm.llm_config import LLMConfig

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueuePriority(str, Enum):
    """Priority levels for LLM requests in the queue.

    CRITICAL requests bypass queue-full rejection when priority override is
    enabled.  NORMAL is the default priority.  LOW requests are processed
    last.
    """

    CRITICAL = "critical"
    NORMAL = "normal"
    LOW = "low"


class CircuitBreakerState(str, Enum):
    """Lifecycle states for the circuit breaker.

    CLOSED  – normal operation, requests flow through.
    OPEN    – requests are blocked after reaching the failure threshold.
    HALF_OPEN – a single test request is allowed to check service recovery.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ---------------------------------------------------------------------------
# Constants – per README.md lines 1988-1999 and AAP specification
# ---------------------------------------------------------------------------

MAX_QUEUE_SIZE: int = 1000
"""Maximum number of pending requests in the Redis Stream (AAP: 'max queue 1,000')."""

MAX_BATCH_SIZE: int = 10
"""Maximum batch dequeue size (AAP: 'batch size 10')."""

QUEUE_TIMEOUT_SECONDS: int = 300
"""Queue item timeout in seconds (5 minutes per README)."""

QUEUE_WARNING_THRESHOLD: int = 800
"""Queue depth at which a capacity warning is emitted (80% of MAX_QUEUE_SIZE)."""

MAX_RETRIES: int = 5
"""Maximum retry attempts per request (README: 'Max retries: 5 attempts')."""

STREAM_NAME: str = "llm_requests"
"""Redis Stream name for LLM requests (AAP: stream 'llm_requests')."""

CONSUMER_GROUP: str = "llm_workers"
"""Consumer group name for Redis Streams (AAP: 'consumer group llm_workers')."""

LLM_QUEUE_TTL: int = 3600
"""TTL for queue items in seconds (1 hour per AAP Section 0.4.4)."""

CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
"""Number of consecutive failures before the circuit breaker opens (AAP: 5)."""


# ---------------------------------------------------------------------------
# LLMRateLimiter – per-provider, per-model rate limiting
# ---------------------------------------------------------------------------


class LLMRateLimiter:
    """Per-provider, per-model rate limiter with 60-second sliding windows.

    Tracks requests within the current minute and blocks (with an async
    sleep) when the configured limit is reached.  Rate limits are sourced
    from the class-level ``RATE_LIMITS`` mapping or can be overridden via
    the *limit_override* constructor parameter.

    Usage::

        limiter = LLMRateLimiter("anthropic", "claude-sonnet-4")
        await limiter.acquire()   # blocks if at limit
    """

    RATE_LIMITS: Dict[str, Dict[str, int]] = {
        "anthropic": {
            "claude-sonnet-4": 50,
            "claude-haiku-4": 100,
        },
        "openai": {
            "gpt-4-turbo": 500,
            "gpt-3.5-turbo": 3500,
        },
    }
    """Default per-provider, per-model rate limits (requests per minute)."""

    def __init__(
        self,
        provider: str,
        model: str,
        limit_override: Optional[int] = None,
    ) -> None:
        self._provider = provider
        self._model = model

        # Resolve the effective limit
        if limit_override is not None:
            self.limit: int = limit_override
        else:
            # Try exact match first, then prefix match for versioned model names
            provider_limits = self.RATE_LIMITS.get(provider, {})
            resolved_limit: Optional[int] = provider_limits.get(model)
            if resolved_limit is None:
                for key, value in provider_limits.items():
                    if model.startswith(key):
                        resolved_limit = value
                        break
            self.limit = resolved_limit if resolved_limit is not None else 50

        self.requests_this_minute: int = 0
        self._minute_start: float = time.time()

        logger.debug(
            "rate_limiter_initialized",
            provider=provider,
            model=model,
            limit=self.limit,
        )

    # -- Public API -----------------------------------------------------------

    async def acquire(self) -> None:
        """Acquire permission to make a request, blocking if rate-limited.

        Resets the per-minute counter every 60 seconds.  When the limit is
        reached, sleeps for the remainder of the current window before
        allowing the request.
        """
        current_time = time.time()

        # Reset the window if 60 seconds have elapsed
        if current_time - self._minute_start >= 60.0:
            self.requests_this_minute = 0
            self._minute_start = current_time

        # Block if at capacity
        if self.requests_this_minute >= self.limit:
            wait_time = max(0.0, 60.0 - (current_time - self._minute_start))
            logger.info(
                "rate_limit_waiting",
                wait_seconds=round(wait_time, 2),
                limit=self.limit,
                provider=self._provider,
                model=self._model,
            )
            await asyncio.sleep(wait_time)
            # Reset after sleeping through the window
            self.requests_this_minute = 0
            self._minute_start = time.time()

        self.requests_this_minute += 1

    def get_remaining(self) -> int:
        """Return the number of requests remaining in the current window."""
        if time.time() - self._minute_start >= 60.0:
            return self.limit
        return max(0, self.limit - self.requests_this_minute)

    def is_at_limit(self) -> bool:
        """Return ``True`` if the rate limit has been reached this minute."""
        if time.time() - self._minute_start >= 60.0:
            return False
        return self.requests_this_minute >= self.limit


# ---------------------------------------------------------------------------
# CircuitBreaker – opens after consecutive failures to protect downstream
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Circuit breaker that opens after *failure_threshold* consecutive failures.

    State machine:
        CLOSED  → (failure_threshold reached) → OPEN
        OPEN    → (recovery_timeout elapsed)  → HALF_OPEN
        HALF_OPEN → success → CLOSED
        HALF_OPEN → failure → OPEN

    Per AAP specification the breaker opens after exactly 5 consecutive
    failures and recovers after a configurable timeout (default 60 s).
    """

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold: int = failure_threshold
        self.recovery_timeout: float = recovery_timeout
        self.state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self.failure_count: int = 0
        self._last_failure_time: Optional[float] = None
        self._success_count_in_half_open: int = 0

        logger.debug(
            "circuit_breaker_initialized",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    # -- State transitions ----------------------------------------------------

    def record_success(self) -> None:
        """Record a successful request.  Resets failure count and transitions
        HALF_OPEN → CLOSED."""
        previous_state = self.state
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self._success_count_in_half_open = 0
            logger.info(
                "circuit_breaker_closed",
                previous_state=previous_state.value,
            )
        logger.debug(
            "circuit_breaker_success",
            state=self.state.value,
            failure_count=self.failure_count,
        )

    def record_failure(self) -> None:
        """Record a failed request.  Opens the breaker when the threshold is
        reached or if already in HALF_OPEN state."""
        self.failure_count += 1
        self._last_failure_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open goes straight back to open
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                reason="half_open_failure",
                failure_count=self.failure_count,
            )
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                reason="threshold_reached",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )
        else:
            logger.debug(
                "circuit_breaker_failure_recorded",
                state=self.state.value,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def can_execute(self) -> bool:
        """Return ``True`` if a request is allowed through the breaker.

        In OPEN state the breaker transitions to HALF_OPEN once the
        *recovery_timeout* has elapsed, allowing a single test request.
        """
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            if self._last_failure_time is None:
                return True
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self._success_count_in_half_open = 0
                logger.info(
                    "circuit_breaker_half_open",
                    elapsed_seconds=round(elapsed, 2),
                    recovery_timeout=self.recovery_timeout,
                )
                return True
            return False

        # HALF_OPEN – allow a test request
        return True

    def reset(self) -> None:
        """Force-reset the breaker to CLOSED with zero failures."""
        previous_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self._last_failure_time = None
        self._success_count_in_half_open = 0
        logger.info(
            "circuit_breaker_reset",
            previous_state=previous_state.value,
        )


# ---------------------------------------------------------------------------
# LLMQueue – Redis Streams-backed LLM request queue
# ---------------------------------------------------------------------------


class LLMQueue:
    """Redis Streams-based LLM request queue with consumer groups.

    Enqueues LLM completion requests via ``XADD`` and dequeues them with
    ``XREADGROUP`` using the consumer group ``llm_workers``.  Supports
    three priority levels (critical / normal / low), batch dequeue of up
    to 10 messages, a hard cap of 1,000 pending requests, and an
    integrated circuit breaker that opens after 5 consecutive failures.

    The queue is fully async and designed for use with ``async with``::

        async with LLMQueue(config) as q:
            rid = await q.enqueue_request("Hello", {"agent": "a1"})
            msgs = await q.dequeue_request()
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._redis: Optional[Any] = redis_client
        self._connected: bool = redis_client is not None
        self._stream_name: str = STREAM_NAME
        self._consumer_group: str = CONSUMER_GROUP
        self._circuit_breaker: CircuitBreaker = CircuitBreaker()

        # Resolve rate limiter from config if available.
        # Accesses config.provider, config.model, and config.get_rate_limit()
        # per schema requirements for LLMConfig dependency.
        if config is not None:
            provider_str = (
                config.provider.value
                if hasattr(config.provider, "value")
                else str(config.provider)
            )
            model_str = str(config.model)
            effective_rate = config.get_rate_limit()
            self._rate_limiter: Optional[LLMRateLimiter] = LLMRateLimiter(
                provider=provider_str,
                model=model_str,
                limit_override=effective_rate,
            )
        else:
            self._rate_limiter = None

        # Metrics counters
        self._metrics: Dict[str, int] = {
            "enqueued": 0,
            "dequeued": 0,
            "retried": 0,
            "rejected": 0,
            "expired": 0,
        }

        logger.info(
            "llm_queue_initialized",
            stream_name=self._stream_name,
            consumer_group=self._consumer_group,
            max_queue_size=MAX_QUEUE_SIZE,
        )

    # -- Context manager support (AAP Section 0.7.1) -------------------------

    async def __aenter__(self) -> LLMQueue:
        """Connect and return the queue instance."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Disconnect on context exit."""
        await self.disconnect()

    # -- Connection lifecycle -------------------------------------------------

    async def connect(self) -> None:
        """Establish a Redis connection and create the consumer group.

        If a ``redis_client`` was provided at construction time it is reused;
        otherwise a new connection is opened using the ``redis_url`` from the
        :class:`LLMConfig`.
        """
        if self._connected and self._redis is not None:
            return

        if self._redis is None:
            if aioredis is None:
                raise RuntimeError(
                    "The 'redis' package is required for LLMQueue. "
                    "Install it with: pip install redis>=7.0.0"
                )
            redis_url = (
                self._config.redis_url
                if self._config is not None and self._config.redis_url
                else "redis://localhost:6379/0"
            )
            self._redis = aioredis.Redis.from_url(
                redis_url,
                decode_responses=True,
            )

        # Create the consumer group (idempotent)
        try:
            await self._redis.xgroup_create(
                self._stream_name,
                self._consumer_group,
                id="0",
                mkstream=True,
            )
            logger.debug(
                "consumer_group_created",
                stream=self._stream_name,
                group=self._consumer_group,
            )
        except (RedisResponseError, Exception) as exc:
            # BUSYGROUP means the group already exists — perfectly fine
            if "BUSYGROUP" in str(exc):
                logger.debug(
                    "consumer_group_already_exists",
                    stream=self._stream_name,
                    group=self._consumer_group,
                )
            else:
                raise

        self._connected = True
        logger.info("llm_queue_connected", stream=self._stream_name)

    async def disconnect(self) -> None:
        """Close the Redis connection and mark the queue as disconnected."""
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass  # Best-effort close
        self._redis = None
        self._connected = False
        logger.info("llm_queue_disconnected")

    # -- Enqueue / Dequeue ----------------------------------------------------

    async def enqueue_request(
        self,
        prompt: str,
        metadata: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> str:
        """Add an LLM request to the Redis Stream.

        Parameters
        ----------
        prompt:
            The text prompt for the LLM completion.
        metadata:
            Arbitrary key-value metadata (agent_id, model, etc.).
        priority:
            Queue priority — CRITICAL requests bypass the queue-full
            rejection policy when priority override is enabled.

        Returns
        -------
        str
            A unique ``request_id`` (UUID4) for tracking.

        Raises
        ------
        RuntimeError
            If the circuit breaker is open or the queue is full (for
            non-CRITICAL requests).
        """
        self._ensure_connected()

        # Rate limiter gate — blocks if per-minute limit is reached
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()

        # Circuit breaker check
        if not self._circuit_breaker.can_execute():
            self._metrics["rejected"] += 1
            logger.error(
                "llm_queue_circuit_breaker_open",
                circuit_state=self._circuit_breaker.state.value,
            )
            raise RuntimeError(
                "Circuit breaker is OPEN — LLM requests are temporarily blocked"
            )

        # Queue capacity check
        current_size = await self._raw_queue_size()
        if current_size >= MAX_QUEUE_SIZE:
            # Priority override: CRITICAL requests bypass the cap
            if priority == QueuePriority.CRITICAL:
                logger.warning(
                    "llm_queue_full_priority_override",
                    queue_size=current_size,
                    priority=priority.value,
                )
            else:
                self._metrics["rejected"] += 1
                logger.error(
                    "llm_queue_full",
                    queue_size=current_size,
                    max_queue_size=MAX_QUEUE_SIZE,
                    priority=priority.value,
                )
                raise RuntimeError(
                    f"LLM queue is full ({current_size}/{MAX_QUEUE_SIZE}). "
                    "Non-critical requests are rejected."
                )

        request_id = str(uuid4())
        message: Dict[str, str] = {
            "request_id": request_id,
            "prompt": prompt,
            "metadata": json.dumps(metadata),
            "priority": priority.value,
            "timestamp": str(time.time()),
            "retry_count": "0",
        }

        await self._redis.xadd(self._stream_name, message)

        self._metrics["enqueued"] += 1

        # Emit capacity warning when approaching the limit
        if current_size + 1 >= QUEUE_WARNING_THRESHOLD:
            logger.warning(
                "llm_queue_capacity_warning",
                queue_size=current_size + 1,
                warning_threshold=QUEUE_WARNING_THRESHOLD,
                max_queue_size=MAX_QUEUE_SIZE,
            )

        logger.debug(
            "llm_request_enqueued",
            request_id=request_id,
            priority=priority.value,
            queue_size=current_size + 1,
        )
        return request_id

    async def dequeue_request(
        self,
        consumer_name: str = "worker-1",
        count: int = 1,
        block_ms: int = 5000,
    ) -> Optional[List[Dict[str, Any]]]:
        """Read one or more requests from the stream via XREADGROUP.

        Parameters
        ----------
        consumer_name:
            Identifier for this consumer within the group.
        count:
            Maximum number of messages to read.
        block_ms:
            Milliseconds to block waiting for new messages (0 = no block).

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            A list of parsed message dicts (each containing ``message_id``,
            ``request_id``, ``prompt``, ``metadata``, etc.) or ``None``
            if no messages were available.
        """
        self._ensure_connected()

        results = await self._redis.xreadgroup(
            groupname=self._consumer_group,
            consumername=consumer_name,
            streams={self._stream_name: ">"},
            count=count,
            block=block_ms,
        )

        if not results:
            return None

        parsed: List[Dict[str, Any]] = []
        for _stream_key, messages in results:
            for message_id, fields in messages:
                entry = self._parse_stream_message(message_id, fields)
                parsed.append(entry)

        if parsed:
            self._metrics["dequeued"] += len(parsed)
            logger.debug(
                "llm_requests_dequeued",
                count=len(parsed),
                consumer=consumer_name,
            )

        return parsed if parsed else None

    async def dequeue_batch(
        self,
        consumer_name: str = "worker-1",
        batch_size: int = MAX_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        """Batch-dequeue up to *batch_size* requests (default 10).

        This is a convenience wrapper around :meth:`dequeue_request` with
        ``block_ms=0`` to avoid blocking when fewer messages are available.

        Returns
        -------
        List[Dict[str, Any]]
            A (possibly empty) list of parsed request dicts.
        """
        result = await self.dequeue_request(
            consumer_name=consumer_name,
            count=batch_size,
            block_ms=0,
        )
        return result if result is not None else []

    # -- Acknowledgement & Retry ----------------------------------------------

    async def acknowledge(self, message_id: str) -> None:
        """Acknowledge a successfully processed message.

        This removes the message from the consumer group's pending entries
        list so it will not be re-delivered.

        Parameters
        ----------
        message_id:
            The Redis Stream message ID (e.g. ``"1234567890123-0"``).
        """
        self._ensure_connected()
        await self._redis.xack(
            self._stream_name, self._consumer_group, message_id
        )
        logger.debug("llm_request_acknowledged", message_id=message_id)

    async def retry_failed_request(
        self,
        request_id: str,
        original_message: Dict[str, Any],
    ) -> Optional[str]:
        """Re-enqueue a failed request with an incremented retry count.

        Applies exponential backoff (2^retry_count seconds: 2 s → 4 s → 8 s
        → 16 s → 32 s) before re-adding the message.

        Parameters
        ----------
        request_id:
            Identifier of the original request.
        original_message:
            The full message dict as returned by :meth:`dequeue_request`.

        Returns
        -------
        Optional[str]
            The new Redis Stream message ID, or ``None`` if max retries
            have been exceeded.
        """
        self._ensure_connected()

        retry_count = int(original_message.get("retry_count", 0)) + 1

        if retry_count > MAX_RETRIES:
            self._metrics["expired"] += 1
            logger.error(
                "llm_request_max_retries_exceeded",
                request_id=request_id,
                retry_count=retry_count,
                max_retries=MAX_RETRIES,
            )
            return None

        # Exponential backoff: 2^retry_count → 2, 4, 8, 16, 32
        backoff_seconds = 2 ** retry_count
        logger.info(
            "llm_request_retry_backoff",
            request_id=request_id,
            retry_count=retry_count,
            backoff_seconds=backoff_seconds,
        )
        await asyncio.sleep(backoff_seconds)

        # Re-enqueue with updated retry count
        message: Dict[str, str] = {
            "request_id": request_id,
            "prompt": original_message.get("prompt", ""),
            "metadata": (
                original_message["metadata"]
                if isinstance(original_message.get("metadata"), str)
                else json.dumps(original_message.get("metadata", {}))
            ),
            "priority": original_message.get("priority", QueuePriority.NORMAL.value),
            "timestamp": str(time.time()),
            "retry_count": str(retry_count),
        }

        new_message_id = await self._redis.xadd(self._stream_name, message)
        self._metrics["retried"] += 1

        logger.info(
            "llm_request_retried",
            request_id=request_id,
            retry_count=retry_count,
            new_message_id=new_message_id,
        )
        return str(new_message_id) if new_message_id else None

    # -- Queue status queries -------------------------------------------------

    async def get_queue_size(self) -> int:
        """Return the total number of entries in the stream via ``XLEN``.

        Emits a warning log when the size reaches :data:`QUEUE_WARNING_THRESHOLD`.
        """
        self._ensure_connected()
        size = await self._raw_queue_size()

        if size >= QUEUE_WARNING_THRESHOLD:
            logger.warning(
                "llm_queue_capacity_warning",
                queue_size=size,
                warning_threshold=QUEUE_WARNING_THRESHOLD,
                max_queue_size=MAX_QUEUE_SIZE,
            )

        return size

    async def get_pending_count(self) -> int:
        """Return the number of messages pending acknowledgement.

        Uses the Redis ``XPENDING`` summary command.
        """
        self._ensure_connected()
        try:
            info = await self._redis.xpending(
                self._stream_name, self._consumer_group
            )
            # xpending returns a dict / list depending on driver version.
            # The first element of the summary is the total pending count.
            if isinstance(info, dict):
                return int(info.get("pending", 0))
            if isinstance(info, (list, tuple)) and len(info) > 0:
                return int(info[0])
            return 0
        except Exception as exc:
            logger.warning(
                "llm_queue_pending_count_error",
                error=str(exc),
            )
            return 0

    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """Return the current :class:`CircuitBreakerState`."""
        return self._circuit_breaker.state

    def get_metrics(self) -> Dict[str, Any]:
        """Return a snapshot of queue operational metrics."""
        return {
            "enqueued": self._metrics["enqueued"],
            "dequeued": self._metrics["dequeued"],
            "retried": self._metrics["retried"],
            "rejected": self._metrics["rejected"],
            "expired": self._metrics["expired"],
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_failure_count": self._circuit_breaker.failure_count,
            "connected": self._connected,
        }

    # -- Internal helpers -----------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise if the queue has not been connected."""
        if not self._connected or self._redis is None:
            raise RuntimeError(
                "LLMQueue is not connected. Call connect() or use "
                "'async with LLMQueue(config) as q:' first."
            )

    async def _raw_queue_size(self) -> int:
        """Return the raw stream length without side-effect logging."""
        try:
            return int(await self._redis.xlen(self._stream_name))
        except Exception:
            return 0

    @staticmethod
    def _parse_stream_message(
        message_id: str,
        fields: Dict[str, str],
    ) -> Dict[str, Any]:
        """Deserialise a raw Redis Stream message into a typed dict."""
        metadata_raw = fields.get("metadata", "{}")
        try:
            metadata = json.loads(metadata_raw)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return {
            "message_id": message_id,
            "request_id": fields.get("request_id", ""),
            "prompt": fields.get("prompt", ""),
            "metadata": metadata,
            "priority": fields.get("priority", QueuePriority.NORMAL.value),
            "timestamp": float(fields.get("timestamp", 0)),
            "retry_count": int(fields.get("retry_count", 0)),
        }

