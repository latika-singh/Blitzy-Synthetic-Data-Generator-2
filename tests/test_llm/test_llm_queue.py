"""Comprehensive unit tests for app.llm.llm_queue module.

Tests cover:
- LLMQueue: Redis Streams-based enqueue/dequeue (XADD/XREADGROUP), batch
  processing, acknowledgement, retry with exponential backoff, queue
  overflow rejection, circuit breaker integration, and async lifecycle.
- LLMRateLimiter: Per-provider rate limiting with 60-second windows,
  counter reset, and limit enforcement.
- CircuitBreaker: CLOSED → OPEN → HALF_OPEN state machine, threshold
  detection, recovery timeout, and forced reset.

All Redis interactions use fakeredis (AAP §0.7.5) — zero external Redis
connections.  Async tests are automatically discovered by pytest-asyncio
(``asyncio_mode = "auto"`` in pytest configuration).
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import fakeredis.aioredis
import pytest

from app.llm.llm_config import LLMConfig, LLMProviderType
from app.llm.llm_queue import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CONSUMER_GROUP,
    MAX_BATCH_SIZE,
    MAX_QUEUE_SIZE,
    STREAM_NAME,
    CircuitBreaker,
    CircuitBreakerState,
    LLMQueue,
    LLMRateLimiter,
    QueuePriority,
)

# ---------------------------------------------------------------------------
# pytest-asyncio auto-mode is enabled via pytest.ini / pyproject.toml
# (asyncio_mode = "auto"), so async test functions are automatically
# detected — no module-level ``pytestmark`` needed.
# ---------------------------------------------------------------------------


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
async def fake_redis():
    """Create an in-memory FakeRedis instance for queue tests.

    Yields the client, then flushes and closes it on teardown so that each
    test starts with a clean Redis state.
    """
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield redis_client
    await redis_client.flushall()
    await redis_client.aclose()


@pytest.fixture
def llm_config() -> LLMConfig:
    """Provide an ``LLMConfig`` with safe test values for queue tests.

    Uses the Anthropic provider and a fake API key — never a real credential.
    """
    return LLMConfig(
        provider=LLMProviderType.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        fallback_model="claude-haiku-4-20250514",
        api_key="test-key-not-real",
        redis_url="redis://fake:6379",
    )


@pytest.fixture
async def llm_queue(llm_config, fake_redis):
    """Create a connected ``LLMQueue`` backed by fakeredis.

    When ``redis_client`` is provided to the constructor, LLMQueue marks
    itself as already connected — causing ``connect()`` to skip consumer
    group creation.  We temporarily clear the flag so that ``connect()``
    runs its full initialisation path (creates the XREADGROUP consumer
    group) while still reusing the injected fakeredis instance.
    """
    queue = LLMQueue(config=llm_config, redis_client=fake_redis)
    # Force connect() to execute fully (consumer group creation)
    queue._connected = False
    await queue.connect()
    yield queue
    await queue.disconnect()


@pytest.fixture
def rate_limiter() -> LLMRateLimiter:
    """Create an ``LLMRateLimiter`` configured for Anthropic Claude Sonnet 4.

    Uses the default rate-limit lookup (50 req/min for claude-sonnet-4).
    """
    return LLMRateLimiter(provider="anthropic", model="claude-sonnet-4")


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Create a ``CircuitBreaker`` with the default 5-failure threshold."""
    return CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)


# =========================================================================
# Phase 3 — Queue Constants Tests
# =========================================================================


class TestQueueConstants:
    """Verify module-level constants match AAP specification values."""

    def test_max_queue_size_constant(self) -> None:
        """MAX_QUEUE_SIZE must be 1,000 (AAP: 'max queue 1,000')."""
        assert MAX_QUEUE_SIZE == 1000

    def test_max_batch_size_constant(self) -> None:
        """MAX_BATCH_SIZE must be 10 (AAP: 'batch size 10')."""
        assert MAX_BATCH_SIZE == 10

    def test_circuit_breaker_failure_threshold_constant(self) -> None:
        """CIRCUIT_BREAKER_FAILURE_THRESHOLD must be 5."""
        assert CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5

    def test_stream_name_constant(self) -> None:
        """STREAM_NAME must be 'llm_requests'."""
        assert STREAM_NAME == "llm_requests"

    def test_consumer_group_constant(self) -> None:
        """CONSUMER_GROUP must be 'llm_workers'."""
        assert CONSUMER_GROUP == "llm_workers"


# =========================================================================
# Phase 4 — Priority Enum Tests
# =========================================================================


class TestQueuePriority:
    """Verify the three priority levels exist with the correct string values."""

    def test_queue_priority_critical(self) -> None:
        assert QueuePriority.CRITICAL.value == "critical"

    def test_queue_priority_normal(self) -> None:
        assert QueuePriority.NORMAL.value == "normal"

    def test_queue_priority_low(self) -> None:
        assert QueuePriority.LOW.value == "low"

    def test_queue_priority_member_count(self) -> None:
        """Exactly three priority levels must exist."""
        assert len(QueuePriority) == 3


# =========================================================================
# Phase 5 — Enqueue Tests (XADD)
# =========================================================================


class TestEnqueue:
    """Tests for ``LLMQueue.enqueue_request``."""

    async def test_enqueue_request_returns_request_id(self, llm_queue) -> None:
        """enqueue_request must return a non-empty UUID string."""
        request_id = await llm_queue.enqueue_request(
            prompt="Test prompt",
            metadata={"key": "value"},
        )
        assert isinstance(request_id, str)
        assert len(request_id) > 0
        # UUID4 format check (8-4-4-4-12)
        parts = request_id.split("-")
        assert len(parts) == 5

    async def test_enqueue_request_returns_unique_ids(self, llm_queue) -> None:
        """Each enqueue must produce a distinct request_id."""
        ids = set()
        for i in range(5):
            rid = await llm_queue.enqueue_request(
                prompt=f"Prompt {i}", metadata={"i": i}
            )
            ids.add(rid)
        assert len(ids) == 5

    async def test_enqueue_request_with_critical_priority(self, llm_queue) -> None:
        """Enqueue with CRITICAL priority must succeed."""
        rid = await llm_queue.enqueue_request(
            prompt="Urgent",
            metadata={},
            priority=QueuePriority.CRITICAL,
        )
        assert isinstance(rid, str) and len(rid) > 0

    async def test_enqueue_request_with_normal_priority(self, llm_queue) -> None:
        """Enqueue with NORMAL priority (default) must succeed."""
        rid = await llm_queue.enqueue_request(
            prompt="Standard", metadata={}
        )
        assert isinstance(rid, str) and len(rid) > 0

    async def test_enqueue_request_with_low_priority(self, llm_queue) -> None:
        """Enqueue with LOW priority must succeed."""
        rid = await llm_queue.enqueue_request(
            prompt="Background",
            metadata={},
            priority=QueuePriority.LOW,
        )
        assert isinstance(rid, str) and len(rid) > 0

    async def test_enqueue_increments_metrics(self, llm_queue) -> None:
        """Each successful enqueue must increment the 'enqueued' metric."""
        for i in range(3):
            await llm_queue.enqueue_request(
                prompt=f"P{i}", metadata={"i": i}
            )
        metrics = llm_queue.get_metrics()
        assert metrics["enqueued"] == 3

    async def test_enqueue_stores_in_redis_stream(
        self, llm_queue, fake_redis
    ) -> None:
        """After enqueue, the Redis Stream must contain exactly one entry."""
        await llm_queue.enqueue_request(prompt="Hello", metadata={"a": 1})
        stream_len = await fake_redis.xlen(STREAM_NAME)
        assert stream_len == 1

    async def test_enqueue_preserves_metadata(self, llm_queue) -> None:
        """Metadata must round-trip correctly through enqueue → dequeue."""
        metadata = {"agent_id": "agent-42", "model": "claude-sonnet-4"}
        await llm_queue.enqueue_request(
            prompt="With metadata", metadata=metadata
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        assert len(messages) == 1
        assert messages[0]["metadata"] == metadata


# =========================================================================
# Phase 6 — Dequeue Tests (XREADGROUP)
# =========================================================================


class TestDequeue:
    """Tests for ``LLMQueue.dequeue_request`` and ``dequeue_batch``."""

    async def test_dequeue_request_returns_enqueued_message(
        self, llm_queue
    ) -> None:
        """Dequeue must return the prompt and metadata from the enqueued message."""
        await llm_queue.enqueue_request(
            prompt="Test dequeue", metadata={"key": "val"}
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        assert len(messages) >= 1
        msg = messages[0]
        assert msg["prompt"] == "Test dequeue"
        assert msg["metadata"] == {"key": "val"}
        assert "message_id" in msg
        assert "request_id" in msg

    async def test_dequeue_empty_queue_returns_none(self, llm_queue) -> None:
        """Dequeue on an empty queue with block_ms=0 must return None."""
        result = await llm_queue.dequeue_request(block_ms=0)
        assert result is None

    async def test_dequeue_batch_returns_multiple(self, llm_queue) -> None:
        """dequeue_batch(batch_size=3) must return exactly 3 messages."""
        for i in range(5):
            await llm_queue.enqueue_request(
                prompt=f"Batch {i}", metadata={"i": i}
            )
        batch = await llm_queue.dequeue_batch(batch_size=3)
        assert len(batch) == 3

    async def test_dequeue_batch_default_size_ten(self, llm_queue) -> None:
        """dequeue_batch() with default size must return at most MAX_BATCH_SIZE (10)."""
        for i in range(15):
            await llm_queue.enqueue_request(
                prompt=f"Big batch {i}", metadata={"i": i}
            )
        batch = await llm_queue.dequeue_batch()
        assert len(batch) == MAX_BATCH_SIZE

    async def test_dequeue_batch_empty_returns_empty_list(
        self, llm_queue
    ) -> None:
        """dequeue_batch on an empty queue must return an empty list (not None)."""
        batch = await llm_queue.dequeue_batch()
        assert batch == []

    async def test_dequeue_increments_metrics(self, llm_queue) -> None:
        """Dequeued messages must be reflected in the 'dequeued' metric."""
        for i in range(3):
            await llm_queue.enqueue_request(
                prompt=f"M{i}", metadata={}
            )
        await llm_queue.dequeue_batch(batch_size=3)
        metrics = llm_queue.get_metrics()
        assert metrics["dequeued"] == 3

    async def test_dequeue_returns_correct_priority(self, llm_queue) -> None:
        """Priority value must be preserved in the dequeued message."""
        await llm_queue.enqueue_request(
            prompt="Critical task",
            metadata={},
            priority=QueuePriority.CRITICAL,
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        assert messages[0]["priority"] == "critical"


# =========================================================================
# Phase 7 — Acknowledge Tests
# =========================================================================


class TestAcknowledge:
    """Tests for ``LLMQueue.acknowledge``."""

    async def test_acknowledge_message(self, llm_queue) -> None:
        """acknowledge() must succeed without error for a dequeued message."""
        await llm_queue.enqueue_request(
            prompt="Ack test", metadata={}
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        msg = messages[0]
        # Should not raise
        await llm_queue.acknowledge(msg["message_id"])

    async def test_acknowledge_returns_none(self, llm_queue) -> None:
        """acknowledge() is a void operation — no return value expected."""
        await llm_queue.enqueue_request(prompt="X", metadata={})
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        result = await llm_queue.acknowledge(messages[0]["message_id"])
        assert result is None


# =========================================================================
# Phase 8 — Queue Size and Overflow Tests
# =========================================================================


class TestQueueSizeAndOverflow:
    """Tests for queue capacity management and overflow rejection."""

    async def test_get_queue_size(self, llm_queue) -> None:
        """get_queue_size must reflect the number of enqueued messages."""
        for i in range(5):
            await llm_queue.enqueue_request(
                prompt=f"Size {i}", metadata={}
            )
        size = await llm_queue.get_queue_size()
        assert size == 5

    async def test_get_queue_size_empty(self, llm_queue) -> None:
        """An empty queue must report size 0."""
        size = await llm_queue.get_queue_size()
        assert size == 0

    async def test_queue_full_rejects_normal_requests(
        self, llm_queue
    ) -> None:
        """When the queue reaches MAX_QUEUE_SIZE, NORMAL requests must be rejected."""
        # Mock _raw_queue_size to simulate a full queue
        with patch.object(
            llm_queue,
            "_raw_queue_size",
            new_callable=AsyncMock,
            return_value=MAX_QUEUE_SIZE,
        ):
            with pytest.raises(RuntimeError, match="queue is full"):
                await llm_queue.enqueue_request(
                    prompt="Overflow",
                    metadata={},
                    priority=QueuePriority.NORMAL,
                )

    async def test_queue_full_rejects_low_priority(self, llm_queue) -> None:
        """When the queue is full, LOW priority requests must also be rejected."""
        with patch.object(
            llm_queue,
            "_raw_queue_size",
            new_callable=AsyncMock,
            return_value=MAX_QUEUE_SIZE,
        ):
            with pytest.raises(RuntimeError, match="queue is full"):
                await llm_queue.enqueue_request(
                    prompt="Low overflow",
                    metadata={},
                    priority=QueuePriority.LOW,
                )

    async def test_queue_full_allows_critical_priority(
        self, llm_queue
    ) -> None:
        """CRITICAL requests must bypass the queue-full cap (priority override)."""
        with patch.object(
            llm_queue,
            "_raw_queue_size",
            new_callable=AsyncMock,
            return_value=MAX_QUEUE_SIZE,
        ):
            # Must NOT raise — CRITICAL bypasses the cap
            rid = await llm_queue.enqueue_request(
                prompt="Critical override",
                metadata={},
                priority=QueuePriority.CRITICAL,
            )
            assert isinstance(rid, str) and len(rid) > 0

    async def test_queue_full_increments_rejected_metric(
        self, llm_queue
    ) -> None:
        """Rejected requests must increment the 'rejected' metric counter."""
        with patch.object(
            llm_queue,
            "_raw_queue_size",
            new_callable=AsyncMock,
            return_value=MAX_QUEUE_SIZE,
        ):
            with pytest.raises(RuntimeError):
                await llm_queue.enqueue_request(
                    prompt="Rejected",
                    metadata={},
                    priority=QueuePriority.NORMAL,
                )
        metrics = llm_queue.get_metrics()
        assert metrics["rejected"] >= 1


# =========================================================================
# Phase 9 — Retry Tests
# =========================================================================


class TestRetry:
    """Tests for ``LLMQueue.retry_failed_request``."""

    async def test_retry_failed_request(self, llm_queue) -> None:
        """retry_failed_request must re-enqueue the message and return a new ID."""
        await llm_queue.enqueue_request(
            prompt="Retry me", metadata={"task": "test"}
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        msg = messages[0]

        # Patch asyncio.sleep so the backoff doesn't actually wait
        with patch("asyncio.sleep", new_callable=AsyncMock):
            new_id = await llm_queue.retry_failed_request(
                request_id=msg["request_id"],
                original_message=msg,
            )
        assert new_id is not None
        # retried metric must have incremented
        metrics = llm_queue.get_metrics()
        assert metrics["retried"] >= 1

    async def test_retry_increments_retry_count(self, llm_queue) -> None:
        """After retry, the re-enqueued message must have retry_count == 1."""
        await llm_queue.enqueue_request(
            prompt="Retry count test", metadata={}
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        msg = messages[0]
        assert msg["retry_count"] == 0

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await llm_queue.retry_failed_request(
                request_id=msg["request_id"],
                original_message=msg,
            )

        # Dequeue the retried message
        retried_messages = await llm_queue.dequeue_request(block_ms=0)
        assert retried_messages is not None
        assert retried_messages[0]["retry_count"] == 1

    async def test_retry_max_retries_exceeded(self, llm_queue) -> None:
        """retry_failed_request must return None when max retries are exceeded."""
        # Create a message dict that has already exhausted retries
        fake_msg = {
            "request_id": "test-id",
            "prompt": "Exhausted",
            "metadata": "{}",
            "priority": "normal",
            "timestamp": str(time.time()),
            "retry_count": 5,  # MAX_RETRIES is 5 — count will become 6
        }
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await llm_queue.retry_failed_request(
                request_id="test-id",
                original_message=fake_msg,
            )
        assert result is None
        # expired metric must have incremented
        metrics = llm_queue.get_metrics()
        assert metrics["expired"] >= 1

    async def test_retry_preserves_prompt_and_metadata(
        self, llm_queue
    ) -> None:
        """Retried messages must preserve the original prompt and metadata."""
        original_metadata = {"agent": "ag-1", "model": "test"}
        await llm_queue.enqueue_request(
            prompt="Preserve me", metadata=original_metadata
        )
        messages = await llm_queue.dequeue_request(block_ms=0)
        assert messages is not None
        msg = messages[0]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await llm_queue.retry_failed_request(
                request_id=msg["request_id"],
                original_message=msg,
            )

        retried = await llm_queue.dequeue_request(block_ms=0)
        assert retried is not None
        assert retried[0]["prompt"] == "Preserve me"
        assert retried[0]["metadata"] == original_metadata


# =========================================================================
# Phase 10 — Circuit Breaker Tests
# =========================================================================


class TestCircuitBreaker:
    """Tests for the ``CircuitBreaker`` state machine.

    State transitions:
        CLOSED → (failure_threshold reached) → OPEN
        OPEN   → (recovery_timeout elapsed) → HALF_OPEN
        HALF_OPEN → success → CLOSED
        HALF_OPEN → failure → OPEN
    """

    def test_circuit_breaker_initial_state(self, circuit_breaker) -> None:
        """A fresh breaker must start CLOSED with zero failures."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.can_execute() is True

    def test_circuit_breaker_stays_closed_under_threshold(
        self, circuit_breaker
    ) -> None:
        """Recording fewer than 5 failures must keep the breaker CLOSED."""
        for _ in range(4):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.failure_count == 4

    def test_circuit_breaker_opens_at_threshold(
        self, circuit_breaker
    ) -> None:
        """Recording exactly 5 consecutive failures must OPEN the breaker."""
        for _ in range(5):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert circuit_breaker.can_execute() is False
        assert circuit_breaker.failure_count == 5

    def test_circuit_breaker_resets_on_success(
        self, circuit_breaker
    ) -> None:
        """A success after failures must reset failure_count to 0 and keep CLOSED."""
        for _ in range(3):
            circuit_breaker.record_failure()
        assert circuit_breaker.failure_count == 3
        circuit_breaker.record_success()
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_half_open_after_recovery_timeout(
        self, circuit_breaker
    ) -> None:
        """After OPEN + recovery_timeout, can_execute must transition to HALF_OPEN."""
        # Open the breaker
        for _ in range(5):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Simulate passage of time beyond recovery_timeout (60s)
        circuit_breaker._last_failure_time = time.time() - 61.0

        # Calling can_execute triggers transition to HALF_OPEN
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_closes_from_half_open_on_success(
        self, circuit_breaker
    ) -> None:
        """A success in HALF_OPEN state must transition the breaker to CLOSED."""
        # Open the breaker
        for _ in range(5):
            circuit_breaker.record_failure()
        # Transition to HALF_OPEN
        circuit_breaker._last_failure_time = time.time() - 61.0
        circuit_breaker.can_execute()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Record success → must close
        circuit_breaker.record_success()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_circuit_breaker_reopens_from_half_open_on_failure(
        self, circuit_breaker
    ) -> None:
        """A failure in HALF_OPEN state must re-open the breaker immediately."""
        # Open the breaker
        for _ in range(5):
            circuit_breaker.record_failure()
        # Transition to HALF_OPEN
        circuit_breaker._last_failure_time = time.time() - 61.0
        circuit_breaker.can_execute()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

        # Record failure → must re-open
        circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_reset(self, circuit_breaker) -> None:
        """reset() must force-close the breaker and zero out all counters."""
        # Open the breaker
        for _ in range(5):
            circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitBreakerState.OPEN

        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.failure_count == 0

    def test_circuit_breaker_state_enum_values(self) -> None:
        """CircuitBreakerState enum must have exactly 3 values with correct strings."""
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"
        assert len(CircuitBreakerState) == 3

    async def test_queue_respects_circuit_breaker(self, llm_queue) -> None:
        """When the queue's circuit breaker is OPEN, enqueue must raise."""
        # Open the internal circuit breaker on the queue
        cb = llm_queue._circuit_breaker
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            await llm_queue.enqueue_request(
                prompt="Blocked", metadata={}
            )
        metrics = llm_queue.get_metrics()
        assert metrics["rejected"] >= 1

    async def test_queue_circuit_breaker_state_accessor(
        self, llm_queue
    ) -> None:
        """get_circuit_breaker_state must return the current CB state."""
        assert llm_queue.get_circuit_breaker_state() == CircuitBreakerState.CLOSED

        cb = llm_queue._circuit_breaker
        for _ in range(5):
            cb.record_failure()
        assert llm_queue.get_circuit_breaker_state() == CircuitBreakerState.OPEN


# =========================================================================
# Phase 11 — LLMRateLimiter Tests
# =========================================================================


class TestLLMRateLimiter:
    """Tests for the per-provider, per-model rate limiter.

    Per AAP specification and README.md lines 1877–1918:
    - Claude Sonnet 4: 50 req/min
    - Claude Haiku 4: 100 req/min
    - GPT-4 Turbo: 500 req/min
    """

    def test_rate_limiter_initialization_anthropic_sonnet(self) -> None:
        """Anthropic Claude Sonnet 4 must default to 50 req/min."""
        rl = LLMRateLimiter(provider="anthropic", model="claude-sonnet-4")
        assert rl.limit == 50

    def test_rate_limiter_initialization_anthropic_haiku(self) -> None:
        """Anthropic Claude Haiku 4 must default to 100 req/min."""
        rl = LLMRateLimiter(provider="anthropic", model="claude-haiku-4")
        assert rl.limit == 100

    def test_rate_limiter_initialization_openai_gpt4(self) -> None:
        """OpenAI GPT-4 Turbo must default to 500 req/min."""
        rl = LLMRateLimiter(provider="openai", model="gpt-4-turbo")
        assert rl.limit == 500

    def test_rate_limiter_initialization_openai_gpt35(self) -> None:
        """OpenAI GPT-3.5 Turbo must default to 3500 req/min."""
        rl = LLMRateLimiter(provider="openai", model="gpt-3.5-turbo")
        assert rl.limit == 3500

    def test_rate_limiter_initialization_unknown_model(self) -> None:
        """An unknown model must fall back to the default limit of 50."""
        rl = LLMRateLimiter(provider="unknown", model="mystery-model")
        assert rl.limit == 50

    def test_rate_limiter_initialization_with_override(self) -> None:
        """limit_override must take precedence over lookup tables."""
        rl = LLMRateLimiter(
            provider="anthropic", model="claude-sonnet-4", limit_override=200
        )
        assert rl.limit == 200

    def test_rate_limiter_prefix_matching(self) -> None:
        """Versioned model names must match the prefix (e.g. claude-sonnet-4-20250514)."""
        rl = LLMRateLimiter(
            provider="anthropic", model="claude-sonnet-4-20250514"
        )
        assert rl.limit == 50

    async def test_rate_limiter_acquire_increments_counter(
        self, rate_limiter
    ) -> None:
        """acquire() must increment requests_this_minute by 1."""
        assert rate_limiter.requests_this_minute == 0
        await rate_limiter.acquire()
        assert rate_limiter.requests_this_minute == 1

    async def test_rate_limiter_allows_under_limit(
        self, rate_limiter
    ) -> None:
        """Acquiring up to (limit - 1) times must succeed without blocking."""
        for _ in range(49):  # limit is 50
            await rate_limiter.acquire()
        assert rate_limiter.requests_this_minute == 49
        assert rate_limiter.is_at_limit() is False

    async def test_rate_limiter_get_remaining(self, rate_limiter) -> None:
        """get_remaining must reflect limit - requests_this_minute."""
        for _ in range(10):
            await rate_limiter.acquire()
        assert rate_limiter.get_remaining() == 40

    async def test_rate_limiter_get_remaining_at_zero(
        self, rate_limiter
    ) -> None:
        """get_remaining must return 0 when requests equal the limit."""
        for _ in range(50):
            await rate_limiter.acquire()
        assert rate_limiter.get_remaining() == 0

    async def test_rate_limiter_is_at_limit(self, rate_limiter) -> None:
        """is_at_limit must return True when requests_this_minute >= limit."""
        for _ in range(50):
            await rate_limiter.acquire()
        assert rate_limiter.is_at_limit() is True

    async def test_rate_limiter_is_at_limit_false(
        self, rate_limiter
    ) -> None:
        """is_at_limit must return False when under the limit."""
        await rate_limiter.acquire()
        assert rate_limiter.is_at_limit() is False

    async def test_rate_limiter_resets_after_minute(
        self, rate_limiter
    ) -> None:
        """After a 60-second window expires, acquire must reset the counter."""
        # Acquire some requests
        for _ in range(10):
            await rate_limiter.acquire()
        assert rate_limiter.requests_this_minute == 10

        # Simulate the minute window expiring
        rate_limiter._minute_start = time.time() - 61.0

        await rate_limiter.acquire()
        # Counter should have been reset to 0, then incremented to 1
        assert rate_limiter.requests_this_minute == 1

    async def test_rate_limiter_get_remaining_resets_after_minute(
        self, rate_limiter
    ) -> None:
        """get_remaining must return the full limit after the window expires."""
        for _ in range(10):
            await rate_limiter.acquire()
        # Simulate the window expiring
        rate_limiter._minute_start = time.time() - 61.0
        assert rate_limiter.get_remaining() == rate_limiter.limit

    async def test_rate_limiter_is_at_limit_resets_after_minute(
        self, rate_limiter
    ) -> None:
        """is_at_limit must return False after the window expires."""
        for _ in range(50):
            await rate_limiter.acquire()
        assert rate_limiter.is_at_limit() is True
        # Simulate the window expiring
        rate_limiter._minute_start = time.time() - 61.0
        assert rate_limiter.is_at_limit() is False


# =========================================================================
# Phase 12 — Context Manager Tests
# =========================================================================


class TestQueueContextManager:
    """Tests for LLMQueue's ``async with`` lifecycle management."""

    async def test_queue_async_context_manager(
        self, llm_config, fake_redis
    ) -> None:
        """Using `async with LLMQueue(...)` must connect and disconnect."""
        async with LLMQueue(
            config=llm_config, redis_client=fake_redis
        ) as queue:
            # Should be connected inside the context
            assert queue._connected is True
            # Enqueue should work
            rid = await queue.enqueue_request(
                prompt="Context test", metadata={}
            )
            assert isinstance(rid, str) and len(rid) > 0

        # After exiting, queue should be disconnected
        assert queue._connected is False

    async def test_queue_not_connected_raises(
        self, llm_config, fake_redis
    ) -> None:
        """Operations on an unconnected queue must raise RuntimeError."""
        queue = LLMQueue(config=llm_config)
        # Don't call connect() — redis_client is None, so not connected
        with pytest.raises(RuntimeError, match="not connected"):
            await queue.enqueue_request(prompt="Fail", metadata={})


# =========================================================================
# Phase 13 — Metrics Tests
# =========================================================================


class TestQueueMetrics:
    """Tests for ``LLMQueue.get_metrics`` tracking."""

    async def test_queue_metrics_has_all_keys(self, llm_queue) -> None:
        """get_metrics must return a dict with all required metric keys."""
        metrics = llm_queue.get_metrics()
        expected_keys = {
            "enqueued",
            "dequeued",
            "retried",
            "rejected",
            "expired",
            "circuit_breaker_state",
            "circuit_breaker_failure_count",
            "connected",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    async def test_queue_metrics_initial_values(self, llm_queue) -> None:
        """A fresh queue must have all numeric metrics at zero."""
        metrics = llm_queue.get_metrics()
        assert metrics["enqueued"] == 0
        assert metrics["dequeued"] == 0
        assert metrics["retried"] == 0
        assert metrics["rejected"] == 0
        assert metrics["expired"] == 0
        assert metrics["circuit_breaker_state"] == "closed"
        assert metrics["circuit_breaker_failure_count"] == 0
        assert metrics["connected"] is True

    async def test_queue_metrics_after_operations(self, llm_queue) -> None:
        """Metrics must accurately reflect enqueue and dequeue operations."""
        # Enqueue 3
        for i in range(3):
            await llm_queue.enqueue_request(
                prompt=f"Op {i}", metadata={}
            )
        # Dequeue 2
        await llm_queue.dequeue_batch(batch_size=2)

        metrics = llm_queue.get_metrics()
        assert metrics["enqueued"] == 3
        assert metrics["dequeued"] == 2

    async def test_queue_metrics_rejected_tracking(self, llm_queue) -> None:
        """Rejected requests due to full queue or open CB must be counted."""
        # Trigger a rejection via full queue
        with patch.object(
            llm_queue,
            "_raw_queue_size",
            new_callable=AsyncMock,
            return_value=MAX_QUEUE_SIZE,
        ):
            with pytest.raises(RuntimeError):
                await llm_queue.enqueue_request(
                    prompt="Reject", metadata={}, priority=QueuePriority.NORMAL
                )
        # Trigger a rejection via circuit breaker
        cb = llm_queue._circuit_breaker
        for _ in range(5):
            cb.record_failure()
        with pytest.raises(RuntimeError):
            await llm_queue.enqueue_request(prompt="CB Reject", metadata={})

        metrics = llm_queue.get_metrics()
        assert metrics["rejected"] == 2

    async def test_queue_metrics_retry_and_expired_tracking(
        self, llm_queue
    ) -> None:
        """Retried and expired counters must be updated by retry_failed_request."""
        # Enqueue and dequeue a message
        await llm_queue.enqueue_request(prompt="Track", metadata={})
        msgs = await llm_queue.dequeue_request(block_ms=0)
        assert msgs is not None
        msg = msgs[0]

        # Successful retry
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await llm_queue.retry_failed_request(
                request_id=msg["request_id"],
                original_message=msg,
            )

        # Expired retry (max retries exceeded)
        expired_msg = {
            "request_id": "exp-id",
            "prompt": "Expired",
            "metadata": "{}",
            "priority": "normal",
            "timestamp": str(time.time()),
            "retry_count": 5,
        }
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await llm_queue.retry_failed_request(
                request_id="exp-id", original_message=expired_msg
            )
        assert result is None

        metrics = llm_queue.get_metrics()
        assert metrics["retried"] >= 1
        assert metrics["expired"] >= 1
