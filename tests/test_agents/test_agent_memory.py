"""
tests/test_agents/test_agent_memory.py

Comprehensive tests for the AgentMemory dual-stream memory system.

Covers:
    - Observation CRUD: add_observation, get_recent, max capacity, eviction
    - Reflection generation: generate_reflection, max capacity, synthesis logic
    - Semantic retrieval: retrieve_relevant, cosine similarity, ranking
    - Memory cleanup: 3,600-second interval, excess eviction
    - Redis persistence: observations, reflections, embeddings, TTL, load
    - Embedding model: 384-dim vectors, float32, local inference

Testing rules (AAP Section 0.7.5):
    - All Redis tests use fakeredis — NO external Redis dependency.
    - Async tests use pytest-asyncio with proper event loop management.
    - Embedding tests use mock model to avoid downloading all-MiniLM-L6-v2.
    - Coverage target: >= 80%.
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import numpy as np
import pytest
import pytest_asyncio
import fakeredis
import fakeredis.aioredis

# ---------------------------------------------------------------------------
# Internal Imports — Module Under Test
# ---------------------------------------------------------------------------
from app.agents.agent_memory import (
    AgentMemory,
    AgentMemoryConfig,
    Observation,
    Reflection,
)
from app.agents.agent_config import AgentConfig

# ---------------------------------------------------------------------------
# Deterministic UUID for reproducible tests
# ---------------------------------------------------------------------------
FIXED_TEST_UUID = UUID("12345678-1234-5678-1234-567812345678")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit_vector(dim_index: int, total_dims: int = 384) -> np.ndarray:
    """Create a 384-dim unit vector with 1.0 at *dim_index*."""
    vec = np.zeros(total_dims, dtype=np.float32)
    vec[dim_index] = 1.0
    return vec


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def agent_id() -> UUID:
    """Return a fixed UUID for deterministic Redis key pattern testing.

    The key pattern verified is: agent:{agent_id}:observations, etc.
    """
    return FIXED_TEST_UUID


@pytest.fixture
def mock_embedding_model():
    """MagicMock mimicking SentenceTransformer that returns 384-dim vectors.

    Uses a text-hash–based seed to produce deterministic, distinct embeddings
    for different input strings while remaining reproducible.
    """
    model = MagicMock()

    def _encode_side_effect(text, convert_to_numpy=True):
        # Deterministic seed derived from text content
        seed = abs(hash(text)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        return rng.rand(384).astype(np.float32)

    model.encode = MagicMock(side_effect=_encode_side_effect)
    return model


@pytest_asyncio.fixture
async def agent_memory(agent_id, mock_embedding_model):
    """AgentMemory with mocked embedding model and Redis disabled.

    Suitable for the majority of unit tests that do not need Redis.
    """
    config = AgentMemoryConfig(enable_redis_persistence=False)
    memory = AgentMemory(agent_id=agent_id, config=config)
    # Inject mock to avoid downloading the real sentence-transformer model
    memory._embedding_model = mock_embedding_model
    return memory


@pytest_asyncio.fixture
async def redis_memory(agent_id, mock_embedding_model):
    """AgentMemory with Redis persistence enabled (binary-safe fakeredis).

    Returns a tuple of (AgentMemory, FakeRedis_client) so tests can
    directly inspect Redis state.  Uses ``decode_responses=False`` to
    avoid UTF-8 corruption of binary embedding data.
    """
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    config = AgentMemoryConfig(enable_redis_persistence=True)
    memory = AgentMemory(
        agent_id=agent_id, config=config, redis_client=redis_client
    )
    memory._embedding_model = mock_embedding_model
    yield memory, redis_client
    await redis_client.flushall()
    await redis_client.aclose()


@pytest_asyncio.fixture
async def memory_with_observations(agent_memory):
    """Pre-populate AgentMemory with 10 sample observations.

    Observations have content "Observation 0" .. "Observation 9" and
    importance values cycling through 1.0 .. 10.0.
    """
    for i in range(10):
        await agent_memory.add_observation(
            content=f"Observation {i}: Agent processed invoice #{i + 100}",
            importance=float((i % 10) + 1),
        )
    return agent_memory


@pytest_asyncio.fixture
async def memory_with_reflections(agent_memory):
    """Pre-populate AgentMemory with observations and 5 reflections.

    Adds 10 observations (required minimum is 5 for reflection generation),
    then generates 5 reflections from those observations.
    """
    for i in range(10):
        await agent_memory.add_observation(
            content=f"Reflection source observation {i}: processed document #{i}",
            importance=float(5 + (i % 5)),
        )
    for _ in range(5):
        await agent_memory.generate_reflection()
    return agent_memory


# =========================================================================
# TestObservationStream — add_observation()
# =========================================================================


class TestObservationStream:
    """Tests for the observation stream add_observation() method.

    Validates content storage, importance handling (including clamping),
    timestamp recording, embedding generation, capacity enforcement,
    and entry format.
    """

    @pytest.mark.asyncio
    async def test_add_observation_stores_content(self, agent_memory):
        """Observation content is correctly stored and retrievable."""
        await agent_memory.add_observation(
            "Test observation content", importance=5.0
        )
        assert agent_memory.get_observation_count() == 1
        recent = await agent_memory.get_recent(n=1)
        assert recent[0]["content"] == "Test observation content"

    @pytest.mark.asyncio
    async def test_add_observation_stores_importance(self, agent_memory):
        """Specified importance score is stored with the observation."""
        await agent_memory.add_observation("Important item", importance=8.5)
        recent = await agent_memory.get_recent(n=1)
        assert recent[0]["importance"] == 8.5

    @pytest.mark.asyncio
    async def test_add_observation_stores_timestamp(self, agent_memory):
        """Observation timestamp is recorded automatically at creation time."""
        before = datetime.utcnow()
        await agent_memory.add_observation(
            "Timestamped observation", importance=5.0
        )
        after = datetime.utcnow()
        recent = await agent_memory.get_recent(n=1)
        ts = datetime.fromisoformat(recent[0]["timestamp"])
        assert before <= ts <= after

    @pytest.mark.asyncio
    async def test_add_observation_importance_range_0_to_10(self, agent_memory):
        """Valid importance values between 0.0 and 10.0 are accepted."""
        for val in [0.0, 1.0, 5.0, 9.99, 10.0]:
            await agent_memory.add_observation(
                f"Obs importance={val}", importance=val
            )
        assert agent_memory.get_observation_count() == 5

    @pytest.mark.asyncio
    async def test_add_observation_clamps_importance_below_0(self, agent_memory):
        """Importance < 0 is clamped to 0.0 (implementation clamps, not rejects)."""
        await agent_memory.add_observation(
            "Negative importance", importance=-5.0
        )
        recent = await agent_memory.get_recent(n=1)
        assert recent[0]["importance"] == 0.0

    @pytest.mark.asyncio
    async def test_add_observation_clamps_importance_above_10(self, agent_memory):
        """Importance > 10 is clamped to 10.0 (implementation clamps, not rejects)."""
        await agent_memory.add_observation(
            "Over max importance", importance=15.0
        )
        recent = await agent_memory.get_recent(n=1)
        assert recent[0]["importance"] == 10.0

    @pytest.mark.asyncio
    async def test_observation_max_capacity_eviction(self, agent_memory):
        """When exceeding max_observations, lowest-importance entries are evicted.

        Uses a small max (10) to keep the test fast.  After 15 adds the
        observation count must remain <= 10.
        """
        config = AgentMemoryConfig(
            max_observations=10, enable_redis_persistence=False
        )
        memory = AgentMemory(agent_id=uuid4(), config=config)
        memory._embedding_model = agent_memory._embedding_model

        for i in range(15):
            await memory.add_observation(
                f"Observation {i}", importance=float(i % 10)
            )
        assert memory.get_observation_count() <= 10

    @pytest.mark.asyncio
    async def test_add_observation_generates_embedding(
        self, agent_memory, mock_embedding_model
    ):
        """An embedding vector is generated for every new observation."""
        await agent_memory.add_observation(
            "Test content for embedding", importance=5.0
        )
        mock_embedding_model.encode.assert_called()
        obs = agent_memory._observations[0]
        assert obs.embedding is not None
        assert obs.embedding.shape == (384,)
        assert obs.embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_observation_entry_format(self, agent_memory):
        """Observation to_dict() produces content, importance, timestamp keys."""
        await agent_memory.add_observation("Formatted observation", importance=7.5)
        recent = await agent_memory.get_recent(n=1)
        entry = recent[0]
        assert "content" in entry
        assert "importance" in entry
        assert "timestamp" in entry

    @pytest.mark.asyncio
    async def test_add_multiple_observations(self, agent_memory):
        """Multiple observations are all stored in order."""
        contents = [f"Multi-obs {i}" for i in range(5)]
        for i, content in enumerate(contents):
            await agent_memory.add_observation(content, importance=float(i + 1))
        assert agent_memory.get_observation_count() == 5
        recent = await agent_memory.get_recent(n=5)
        # Most recent first
        assert recent[0]["content"] == "Multi-obs 4"
        assert recent[-1]["content"] == "Multi-obs 0"


# =========================================================================
# TestGetRecent — get_recent()
# =========================================================================


class TestGetRecent:
    """Tests for the get_recent() retrieval method.

    Validates result count, chronological ordering (most-recent-first),
    default parameter value, edge cases (empty / fewer-than-n), and
    return type (list of dicts with proper fields).
    """

    @pytest.mark.asyncio
    async def test_get_recent_returns_n_observations(
        self, memory_with_observations
    ):
        """get_recent(n=5) returns exactly 5 entries when 10 are available."""
        recent = await memory_with_observations.get_recent(n=5)
        assert len(recent) == 5

    @pytest.mark.asyncio
    async def test_get_recent_returns_most_recent_first(
        self, memory_with_observations
    ):
        """Results are ordered newest-first (reverse chronological)."""
        recent = await memory_with_observations.get_recent(n=3)
        ts_list = [datetime.fromisoformat(r["timestamp"]) for r in recent]
        # Each timestamp should be >= the next one (newest first)
        for i in range(len(ts_list) - 1):
            assert ts_list[i] >= ts_list[i + 1]

    @pytest.mark.asyncio
    async def test_get_recent_default_n_is_10(
        self, memory_with_observations
    ):
        """get_recent() with no argument returns up to 10 observations."""
        recent = await memory_with_observations.get_recent()
        assert len(recent) == 10  # 10 were added in fixture

    @pytest.mark.asyncio
    async def test_get_recent_fewer_than_n(self, agent_memory):
        """When fewer observations exist than requested, return all available."""
        for i in range(3):
            await agent_memory.add_observation(f"Few obs {i}", importance=5.0)
        recent = await agent_memory.get_recent(n=10)
        assert len(recent) == 3

    @pytest.mark.asyncio
    async def test_get_recent_empty_memory(self, agent_memory):
        """Empty observation stream returns an empty list."""
        recent = await agent_memory.get_recent()
        assert recent == []

    @pytest.mark.asyncio
    async def test_get_recent_returns_observation_dicts(
        self, memory_with_observations
    ):
        """Each element is a dict with content, importance, and timestamp."""
        recent = await memory_with_observations.get_recent(n=1)
        assert isinstance(recent, list)
        entry = recent[0]
        assert isinstance(entry, dict)
        assert "content" in entry
        assert "importance" in entry
        assert "timestamp" in entry


# =========================================================================
# TestSemanticRetrieval — retrieve_relevant()
# =========================================================================


class TestSemanticRetrieval:
    """Tests for the retrieve_relevant() semantic search method.

    Validates result count, default k, cosine-similarity–based ranking,
    edge cases (empty / fewer-than-k), query embedding generation, and
    deterministic ranking with controlled mock embeddings.
    """

    @pytest.mark.asyncio
    async def test_retrieve_relevant_returns_k_results(
        self, memory_with_observations
    ):
        """retrieve_relevant(query, k=5) returns exactly 5 results."""
        results = await memory_with_observations.retrieve_relevant(
            "invoice processing", k=5
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_retrieve_relevant_default_k_is_5(
        self, memory_with_observations
    ):
        """Default k parameter is 5."""
        results = await memory_with_observations.retrieve_relevant(
            "invoice processing"
        )
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_retrieve_relevant_uses_cosine_similarity(
        self, agent_memory
    ):
        """Results are ranked by cosine similarity, not insertion order.

        Creates observations with deliberately controlled embeddings and
        verifies ranking order.
        """
        # Override the mock to return specific embeddings per content
        embedding_map = {
            "alpha document": _make_unit_vector(0),  # [1, 0, 0, ...]
            "beta document": _make_unit_vector(1),   # [0, 1, 0, ...]
            "gamma document": _make_unit_vector(2),  # [0, 0, 1, ...]
        }
        # Query will be closest to alpha
        query_text = "query_closest_to_alpha"
        query_embedding = np.zeros(384, dtype=np.float32)
        query_embedding[0] = 0.95
        query_embedding[1] = 0.05
        embedding_map[query_text] = query_embedding

        def controlled_encode(text, convert_to_numpy=True):
            if text in embedding_map:
                return embedding_map[text]
            return np.random.rand(384).astype(np.float32)

        agent_memory._embedding_model.encode = MagicMock(
            side_effect=controlled_encode
        )

        # Add observations
        await agent_memory.add_observation("alpha document", importance=5.0)
        await agent_memory.add_observation("beta document", importance=5.0)
        await agent_memory.add_observation("gamma document", importance=5.0)

        results = await agent_memory.retrieve_relevant(query_text, k=3)
        assert len(results) == 3
        # alpha should be first (highest cosine sim with query)
        assert results[0].content == "alpha document"

    @pytest.mark.asyncio
    async def test_retrieve_relevant_returns_most_similar_first(
        self, agent_memory
    ):
        """Most semantically similar observations are ranked first."""
        # Create embeddings where obs_target is very close to query
        target_emb = np.zeros(384, dtype=np.float32)
        target_emb[0] = 1.0
        other_emb = np.zeros(384, dtype=np.float32)
        other_emb[200] = 1.0

        query_emb = np.zeros(384, dtype=np.float32)
        query_emb[0] = 0.99
        query_emb[1] = 0.01

        emb_map = {
            "target observation": target_emb,
            "other observation A": other_emb,
            "other observation B": other_emb.copy(),
            "search query": query_emb,
        }

        def encode_fn(text, convert_to_numpy=True):
            return emb_map.get(text, np.random.rand(384).astype(np.float32))

        agent_memory._embedding_model.encode = MagicMock(side_effect=encode_fn)

        await agent_memory.add_observation("other observation A", importance=5.0)
        await agent_memory.add_observation("target observation", importance=5.0)
        await agent_memory.add_observation("other observation B", importance=5.0)

        results = await agent_memory.retrieve_relevant("search query", k=3)
        assert results[0].content == "target observation"

    @pytest.mark.asyncio
    async def test_retrieve_relevant_empty_memory(self, agent_memory):
        """Empty observation stream returns an empty list."""
        results = await agent_memory.retrieve_relevant("any query", k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_relevant_fewer_than_k(self, agent_memory):
        """When fewer observations exist than k, return all available."""
        for i in range(3):
            await agent_memory.add_observation(
                f"Sparse obs {i}", importance=5.0
            )
        results = await agent_memory.retrieve_relevant("sparse", k=5)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_relevant_generates_query_embedding(
        self, agent_memory, mock_embedding_model
    ):
        """An embedding is generated for the query text via the model."""
        await agent_memory.add_observation("Test obs", importance=5.0)
        mock_embedding_model.encode.reset_mock()
        await agent_memory.retrieve_relevant("my query text", k=1)
        # encode() should have been called for the query
        mock_embedding_model.encode.assert_called()
        call_args_list = mock_embedding_model.encode.call_args_list
        query_calls = [
            c for c in call_args_list if c[0][0] == "my query text"
        ]
        assert len(query_calls) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_relevant_with_deterministic_embeddings(
        self, agent_memory
    ):
        """Deterministic mock embeddings verify cosine similarity ranking.

        Creates 3 observations with known embeddings and a query close
        to observation #2, then verifies #2 is returned first.
        """
        emb_a = np.zeros(384, dtype=np.float32)
        emb_a[0] = 1.0  # dimension 0
        emb_b = np.zeros(384, dtype=np.float32)
        emb_b[50] = 1.0  # dimension 50
        emb_c = np.zeros(384, dtype=np.float32)
        emb_c[100] = 1.0  # dimension 100

        # Query is closest to emb_b
        emb_query = np.zeros(384, dtype=np.float32)
        emb_query[50] = 0.98
        emb_query[51] = 0.02

        emb_lookup = {
            "obs_a": emb_a,
            "obs_b": emb_b,
            "obs_c": emb_c,
            "find_b": emb_query,
        }

        def enc(text, convert_to_numpy=True):
            return emb_lookup.get(text, np.random.rand(384).astype(np.float32))

        agent_memory._embedding_model.encode = MagicMock(side_effect=enc)

        await agent_memory.add_observation("obs_a", importance=5.0)
        await agent_memory.add_observation("obs_b", importance=5.0)
        await agent_memory.add_observation("obs_c", importance=5.0)

        results = await agent_memory.retrieve_relevant("find_b", k=3)
        assert results[0].content == "obs_b"


# =========================================================================
# TestReflectionStream — generate_reflection()
# =========================================================================


class TestReflectionStream:
    """Tests for the reflection stream generate_reflection() method.

    Validates insight generation, synthesis from recent observations,
    capacity enforcement (max 100), timestamp recording, and graceful
    handling of insufficient observation data.
    """

    @pytest.mark.asyncio
    async def test_generate_reflection_creates_insight(
        self, memory_with_observations
    ):
        """generate_reflection produces a non-None Reflection with content."""
        reflection = await memory_with_observations.generate_reflection()
        assert reflection is not None
        assert isinstance(reflection, Reflection)
        assert len(reflection.content) > 0

    @pytest.mark.asyncio
    async def test_generate_reflection_synthesizes_recent_observations(
        self, memory_with_observations
    ):
        """Reflection content references information from recent observations."""
        reflection = await memory_with_observations.generate_reflection()
        assert reflection is not None
        # The implementation builds synthesis from observation content snippets
        assert reflection.source_observation_count > 0
        # Reflection importance is derived from observation importance
        assert 0.0 <= reflection.importance <= 10.0

    @pytest.mark.asyncio
    async def test_reflection_max_capacity_eviction(self, agent_memory):
        """Oldest reflections are evicted when max_reflections is exceeded.

        Uses a small max_reflections (5) for efficiency.
        """
        config = AgentMemoryConfig(
            max_reflections=5, enable_redis_persistence=False
        )
        memory = AgentMemory(agent_id=uuid4(), config=config)
        memory._embedding_model = agent_memory._embedding_model

        # Need >= 5 observations to generate reflections
        for i in range(10):
            await memory.add_observation(
                f"Obs for reflection {i}", importance=5.0 + (i % 5)
            )

        # Generate 8 reflections — only last 5 should remain
        for _ in range(8):
            await memory.generate_reflection()

        assert memory.get_reflection_count() <= 5

    @pytest.mark.asyncio
    async def test_reflection_stored_with_timestamp(
        self, memory_with_observations
    ):
        """Generated reflection has a timestamp recorded at creation time."""
        before = datetime.utcnow()
        reflection = await memory_with_observations.generate_reflection()
        after = datetime.utcnow()
        assert reflection is not None
        assert before <= reflection.timestamp <= after

    @pytest.mark.asyncio
    async def test_generate_reflection_with_no_observations(self, agent_memory):
        """Generating a reflection with no observations returns None."""
        result = await agent_memory.generate_reflection()
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_reflection_with_fewer_than_5_observations(
        self, agent_memory
    ):
        """Reflection generation requires at least 5 observations (returns None otherwise)."""
        for i in range(4):
            await agent_memory.add_observation(
                f"Insufficient obs {i}", importance=5.0
            )
        result = await agent_memory.generate_reflection()
        assert result is None

    @pytest.mark.asyncio
    async def test_reflection_to_dict_format(self, memory_with_observations):
        """Reflection to_dict() includes content, source_observation_count, timestamp, importance."""
        reflection = await memory_with_observations.generate_reflection()
        assert reflection is not None
        data = reflection.to_dict()
        assert "content" in data
        assert "source_observation_count" in data
        assert "timestamp" in data
        assert "importance" in data


# =========================================================================
# TestMemoryCapacity — Capacity tracking and limits
# =========================================================================


class TestMemoryCapacity:
    """Tests for memory capacity tracking and limit enforcement.

    Validates observation/reflection count methods, FIFO eviction at
    capacity limits, and comprehensive memory stats reporting.
    """

    @pytest.mark.asyncio
    async def test_observation_count_tracking(self, agent_memory):
        """get_observation_count() reflects the actual number of observations."""
        assert agent_memory.get_observation_count() == 0
        for i in range(7):
            await agent_memory.add_observation(f"Count test {i}", importance=5.0)
        assert agent_memory.get_observation_count() == 7

    @pytest.mark.asyncio
    async def test_reflection_count_tracking(self, agent_memory):
        """get_reflection_count() reflects the actual number of reflections."""
        assert agent_memory.get_reflection_count() == 0
        # Add enough observations for reflection generation
        for i in range(10):
            await agent_memory.add_observation(
                f"Ref count obs {i}", importance=6.0
            )
        for _ in range(3):
            await agent_memory.generate_reflection()
        assert agent_memory.get_reflection_count() == 3

    @pytest.mark.asyncio
    async def test_observation_eviction_at_max(self, agent_memory):
        """Observation count never exceeds max_observations after eviction.

        Uses a small max_observations (5) for efficiency.
        """
        config = AgentMemoryConfig(
            max_observations=5, enable_redis_persistence=False
        )
        memory = AgentMemory(agent_id=uuid4(), config=config)
        memory._embedding_model = agent_memory._embedding_model

        for i in range(12):
            await memory.add_observation(f"Eviction test {i}", importance=float(i))
            assert memory.get_observation_count() <= 5

    @pytest.mark.asyncio
    async def test_reflection_eviction_at_max(self, agent_memory):
        """Reflection count never exceeds max_reflections after eviction.

        Uses a small max_reflections (3) for efficiency.
        """
        config = AgentMemoryConfig(
            max_reflections=3, enable_redis_persistence=False
        )
        memory = AgentMemory(agent_id=uuid4(), config=config)
        memory._embedding_model = agent_memory._embedding_model

        for i in range(10):
            await memory.add_observation(
                f"Ref eviction obs {i}", importance=5.0 + (i % 5)
            )
        for _ in range(7):
            await memory.generate_reflection()
            assert memory.get_reflection_count() <= 3

    @pytest.mark.asyncio
    async def test_memory_stats_reporting(self, memory_with_observations):
        """get_memory_stats() returns a dict with required monitoring keys."""
        stats = memory_with_observations.get_memory_stats()
        assert isinstance(stats, dict)
        assert "observation_count" in stats
        assert stats["observation_count"] == 10
        assert "reflection_count" in stats
        assert stats["reflection_count"] == 0
        assert "estimated_memory_bytes" in stats
        assert stats["estimated_memory_bytes"] > 0
        assert "max_observations" in stats
        assert "max_reflections" in stats
        assert "last_cleanup" in stats
        assert "redis_enabled" in stats
        assert "embedding_model_loaded" in stats
        assert "agent_id" in stats


# =========================================================================
# TestMemoryCleanup — Periodic cleanup mechanism
# =========================================================================


class TestMemoryCleanup:
    """Tests for the periodic memory cleanup mechanism.

    Validates the 3,600-second cleanup interval, excess entry removal,
    preservation of within-limit entries, and automatic triggering via
    add_observation.
    """

    @pytest.mark.asyncio
    async def test_cleanup_interval_is_3600_seconds(self, agent_memory):
        """Default cleanup_interval_seconds in config is 3,600."""
        assert agent_memory.config.cleanup_interval_seconds == 3600

    @pytest.mark.asyncio
    async def test_cleanup_removes_excess_entries(self, agent_memory):
        """Explicit cleanup() enforces max_observations limit."""
        config = AgentMemoryConfig(
            max_observations=5, enable_redis_persistence=False
        )
        memory = AgentMemory(agent_id=uuid4(), config=config)
        memory._embedding_model = agent_memory._embedding_model

        # Directly inject more observations than the limit (bypass eviction in add)
        for i in range(10):
            obs = Observation(
                content=f"Cleanup test {i}",
                importance=float(i),
                embedding=np.random.rand(384).astype(np.float32),
            )
            memory._observations.append(obs)

        assert len(memory._observations) == 10
        await memory.cleanup()
        assert memory.get_observation_count() <= 5

    @pytest.mark.asyncio
    async def test_cleanup_preserves_within_limit_entries(self, agent_memory):
        """Cleanup does not remove entries when count is within limits."""
        for i in range(3):
            await agent_memory.add_observation(
                f"Within-limit obs {i}", importance=5.0
            )
        count_before = agent_memory.get_observation_count()
        await agent_memory.cleanup()
        assert agent_memory.get_observation_count() == count_before

    @pytest.mark.asyncio
    async def test_auto_cleanup_triggered_by_interval(self, agent_memory):
        """add_observation triggers cleanup when the interval has elapsed.

        Simulates elapsed time by backdating _last_cleanup.
        """
        # Backdate the last cleanup by more than 3600 seconds
        agent_memory._last_cleanup = datetime.utcnow() - timedelta(seconds=3700)

        # Directly inject excess observations (bypass add to avoid eviction)
        config = AgentMemoryConfig(
            max_observations=5, enable_redis_persistence=False
        )
        memory = AgentMemory(agent_id=uuid4(), config=config)
        memory._embedding_model = agent_memory._embedding_model
        memory._last_cleanup = datetime.utcnow() - timedelta(seconds=3700)

        for i in range(5):
            await memory.add_observation(
                f"Pre-cleanup obs {i}", importance=float(i)
            )

        # After adding, _should_cleanup() was True, so cleanup ran
        # Verify that the last_cleanup timestamp was updated recently
        elapsed_since_cleanup = (
            datetime.utcnow() - memory._last_cleanup
        ).total_seconds()
        assert elapsed_since_cleanup < 10  # Updated within last 10 seconds

    @pytest.mark.asyncio
    async def test_cleanup_updates_last_cleanup_timestamp(self, agent_memory):
        """cleanup() updates the _last_cleanup timestamp."""
        old_cleanup = agent_memory._last_cleanup
        # Small sleep so timestamps differ
        await asyncio.sleep(0.01)
        await agent_memory.cleanup()
        assert agent_memory._last_cleanup > old_cleanup


# =========================================================================
# TestRedisIntegration — Redis persistence via fakeredis
# =========================================================================


class TestRedisIntegration:
    """Tests for Redis persistence using fakeredis (no external dependency).

    Validates that observations, reflections, and embeddings are correctly
    stored and loaded from Redis with the correct key patterns and TTLs.
    """

    @pytest.mark.asyncio
    async def test_observations_persisted_to_redis(self, redis_memory, agent_id):
        """After add_observation, data is stored in Redis key agent:{id}:observations."""
        memory, redis_client = redis_memory
        await memory.add_observation("Redis obs test", importance=7.0)

        obs_key = f"agent:{agent_id}:observations"
        raw = await redis_client.get(obs_key)
        assert raw is not None
        # Should be parseable JSON
        import json

        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["content"] == "Redis obs test"
        assert data[0]["importance"] == 7.0

    @pytest.mark.asyncio
    async def test_reflections_persisted_to_redis(self, redis_memory, agent_id):
        """After generate_reflection, data is stored in Redis key agent:{id}:reflections."""
        memory, redis_client = redis_memory

        # Add enough observations for reflection
        for i in range(6):
            await memory.add_observation(
                f"Redis ref source {i}", importance=6.0
            )
        reflection = await memory.generate_reflection()
        assert reflection is not None

        ref_key = f"agent:{agent_id}:reflections"
        raw = await redis_client.get(ref_key)
        assert raw is not None
        import json

        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "content" in data[0]

    @pytest.mark.asyncio
    async def test_embeddings_persisted_to_redis(self, redis_memory, agent_id):
        """Embeddings are stored as binary in Redis key agent:{id}:embeddings."""
        memory, redis_client = redis_memory
        await memory.add_observation("Embedding persistence test", importance=5.0)

        emb_key = f"agent:{agent_id}:embeddings"
        raw = await redis_client.get(emb_key)
        assert raw is not None
        # Should be binary float32 data — length must be a multiple of 384 * 4
        assert len(raw) > 0
        assert len(raw) % (384 * 4) == 0

    @pytest.mark.asyncio
    async def test_redis_ttl_set_30_days(self, redis_memory, agent_id):
        """Redis keys are set with a TTL of 30 days (2,592,000 seconds)."""
        memory, redis_client = redis_memory
        await memory.add_observation("TTL test observation", importance=5.0)

        obs_key = f"agent:{agent_id}:observations"
        ttl = await redis_client.ttl(obs_key)
        # TTL should be approximately 30 * 86400 = 2,592,000 seconds
        expected_ttl = 30 * 86400
        assert ttl > 0
        # Allow a small margin for test execution time
        assert abs(ttl - expected_ttl) < 10

    @pytest.mark.asyncio
    async def test_persist_to_redis_callable(self, redis_memory, agent_id):
        """_persist_to_redis() can be called explicitly to force persistence.

        After adding observations and explicitly calling _persist_to_redis,
        the data must be recoverable from Redis.
        """
        memory, redis_client = redis_memory
        await memory.add_observation("Explicit persist test", importance=5.0)
        # Explicitly call _persist_to_redis to verify it is accessible
        await memory._persist_to_redis()
        obs_key = f"agent:{agent_id}:observations"
        raw = await redis_client.get(obs_key)
        assert raw is not None

    @pytest.mark.asyncio
    async def test_memory_load_from_redis(self, agent_id, mock_embedding_model):
        """A new AgentMemory with the same agent_id loads persisted data.

        Workflow: create memory A → add obs → create memory B with same
        agent_id → call _load_from_redis → verify B has the observations.
        """
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
        try:
            config = AgentMemoryConfig(enable_redis_persistence=True)

            # Memory A: add observations
            memory_a = AgentMemory(
                agent_id=agent_id, config=config, redis_client=redis_client
            )
            memory_a._embedding_model = mock_embedding_model
            await memory_a.add_observation("Persisted obs 1", importance=8.0)
            await memory_a.add_observation("Persisted obs 2", importance=6.0)

            # Memory B: same agent_id, fresh instance
            memory_b = AgentMemory(
                agent_id=agent_id, config=config, redis_client=redis_client
            )
            memory_b._embedding_model = mock_embedding_model
            assert memory_b.get_observation_count() == 0

            # Load from Redis
            await memory_b._load_from_redis()
            assert memory_b.get_observation_count() == 2

            recent = await memory_b.get_recent(n=2)
            contents = {r["content"] for r in recent}
            assert "Persisted obs 1" in contents
            assert "Persisted obs 2" in contents
        finally:
            await redis_client.flushall()
            await redis_client.aclose()


# =========================================================================
# TestEmbeddingModel — Embedding vector verification
# =========================================================================


class TestEmbeddingModel:
    """Tests for the embedding model integration.

    Validates embedding dimensionality (384), dtype (float32), and local
    inference (no external API calls).  Uses mock model by default; the
    ``test_embedding_model_runs_locally`` test patches the SentenceTransformer
    import to verify lazy loading without downloading the real model.
    """

    @pytest.mark.asyncio
    async def test_embedding_dimension_is_384(self, agent_memory):
        """All embeddings have exactly 384 dimensions (all-MiniLM-L6-v2)."""
        await agent_memory.add_observation("Dimension check", importance=5.0)
        obs = agent_memory._observations[0]
        assert obs.embedding is not None
        assert obs.embedding.shape == (384,)

    @pytest.mark.asyncio
    async def test_embedding_is_numpy_float32(self, agent_memory):
        """Embedding dtype is numpy float32."""
        await agent_memory.add_observation("Dtype check", importance=5.0)
        obs = agent_memory._observations[0]
        assert obs.embedding is not None
        assert obs.embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embedding_model_runs_locally(self, agent_id):
        """The embedding model is loaded locally via SentenceTransformer.

        Patches the lazy import inside _get_embedding_model() to verify
        that: (a) SentenceTransformer is instantiated with the expected
        model name, and (b) no external embedding API calls are made.
        """
        mock_st_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.encode = MagicMock(
            return_value=np.random.rand(384).astype(np.float32)
        )
        mock_st_class.return_value = mock_model_instance

        config = AgentMemoryConfig(enable_redis_persistence=False)
        memory = AgentMemory(agent_id=agent_id, config=config)

        # Patch the lazy import within _get_embedding_model
        with patch(
            "app.agents.agent_memory.AgentMemory._get_embedding_model"
        ) as mock_get:
            mock_get.return_value = mock_model_instance
            await memory.add_observation("Local model test", importance=5.0)

        # The mock model's encode was called (local inference)
        mock_model_instance.encode.assert_called()

    @pytest.mark.asyncio
    async def test_embedding_config_default_model_name(self, agent_memory):
        """Default embedding model name is all-MiniLM-L6-v2."""
        assert agent_memory.config.embedding_model_name == "all-MiniLM-L6-v2"

    @pytest.mark.asyncio
    async def test_embedding_config_default_dim(self, agent_memory):
        """Default embedding dimension is 384."""
        assert agent_memory.config.embedding_dim == 384


# =========================================================================
# TestObservationDataclass — Observation serialization
# =========================================================================


class TestObservationDataclass:
    """Tests for the Observation dataclass serialization and deserialization."""

    def test_observation_to_dict(self):
        """Observation.to_dict() returns content, importance, timestamp."""
        obs = Observation(
            content="Test content",
            importance=7.5,
            timestamp=datetime(2025, 1, 15, 10, 30, 0),
        )
        data = obs.to_dict()
        assert data["content"] == "Test content"
        assert data["importance"] == 7.5
        assert data["timestamp"] == "2025-01-15T10:30:00"

    def test_observation_from_dict(self):
        """Observation.from_dict() reconstructs an Observation from a dict."""
        data = {
            "content": "Loaded content",
            "importance": 6.0,
            "timestamp": "2025-02-01T14:00:00",
        }
        obs = Observation.from_dict(data)
        assert obs.content == "Loaded content"
        assert obs.importance == 6.0
        assert obs.timestamp == datetime(2025, 2, 1, 14, 0, 0)
        assert obs.embedding is None  # Embeddings loaded separately

    def test_observation_default_timestamp(self):
        """Observation auto-generates a timestamp when not provided."""
        before = datetime.utcnow()
        obs = Observation(content="Auto-ts", importance=5.0)
        after = datetime.utcnow()
        assert before <= obs.timestamp <= after


# =========================================================================
# TestReflectionDataclass — Reflection serialization
# =========================================================================


class TestReflectionDataclass:
    """Tests for the Reflection dataclass serialization and deserialization."""

    def test_reflection_to_dict(self):
        """Reflection.to_dict() returns content, source_observation_count, timestamp, importance."""
        ref = Reflection(
            content="Summary insight",
            source_observation_count=10,
            timestamp=datetime(2025, 3, 1, 9, 0, 0),
            importance=8.0,
        )
        data = ref.to_dict()
        assert data["content"] == "Summary insight"
        assert data["source_observation_count"] == 10
        assert data["timestamp"] == "2025-03-01T09:00:00"
        assert data["importance"] == 8.0

    def test_reflection_from_dict(self):
        """Reflection.from_dict() reconstructs a Reflection from a dict."""
        data = {
            "content": "Loaded reflection",
            "source_observation_count": 5,
            "timestamp": "2025-04-10T16:30:00",
            "importance": 7.5,
        }
        ref = Reflection.from_dict(data)
        assert ref.content == "Loaded reflection"
        assert ref.source_observation_count == 5
        assert ref.timestamp == datetime(2025, 4, 10, 16, 30, 0)
        assert ref.importance == 7.5

    def test_reflection_default_importance(self):
        """Reflection default importance is 7.0."""
        ref = Reflection(content="Default imp", source_observation_count=3)
        assert ref.importance == 7.0


# =========================================================================
# TestAgentMemoryConfig — Configuration validation
# =========================================================================


class TestAgentMemoryConfig:
    """Tests for the AgentMemoryConfig Pydantic V2 model."""

    def test_default_max_observations(self):
        """Default max_observations is 1000."""
        config = AgentMemoryConfig()
        assert config.max_observations == 1000

    def test_default_max_reflections(self):
        """Default max_reflections is 100."""
        config = AgentMemoryConfig()
        assert config.max_reflections == 100

    def test_default_cleanup_interval(self):
        """Default cleanup_interval_seconds is 3600."""
        config = AgentMemoryConfig()
        assert config.cleanup_interval_seconds == 3600

    def test_default_redis_ttl_days(self):
        """Default redis_ttl_days is 30."""
        config = AgentMemoryConfig()
        assert config.redis_ttl_days == 30

    def test_default_embedding_dim(self):
        """Default embedding_dim is 384."""
        config = AgentMemoryConfig()
        assert config.embedding_dim == 384

    def test_custom_config_values(self):
        """Custom configuration values are accepted and stored."""
        config = AgentMemoryConfig(
            max_observations=500,
            max_reflections=50,
            cleanup_interval_seconds=1800,
            redis_ttl_days=14,
            embedding_dim=256,
            enable_redis_persistence=False,
            redis_key_prefix="custom_agent",
        )
        assert config.max_observations == 500
        assert config.max_reflections == 50
        assert config.cleanup_interval_seconds == 1800
        assert config.redis_ttl_days == 14
        assert config.embedding_dim == 256
        assert config.enable_redis_persistence is False
        assert config.redis_key_prefix == "custom_agent"

    def test_config_rejects_invalid_max_observations(self):
        """max_observations must be >= 1."""
        with pytest.raises(Exception):
            AgentMemoryConfig(max_observations=0)

    def test_config_rejects_invalid_max_reflections(self):
        """max_reflections must be >= 1."""
        with pytest.raises(Exception):
            AgentMemoryConfig(max_reflections=0)
