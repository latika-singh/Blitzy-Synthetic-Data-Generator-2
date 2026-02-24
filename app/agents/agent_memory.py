"""
Agent Memory System — Dual-stream memory with semantic retrieval.

This module implements the AgentMemory class providing a dual-stream memory
system for ERP simulation agents:

- **Observation Stream**: Up to 1,000 timestamped observations (~500 bytes each)
  with importance scoring (0.0–10.0) and 384-dimensional embeddings generated
  by the local all-MiniLM-L6-v2 sentence-transformer model.

- **Reflection Stream**: Up to 100 synthesized reflections (~2 KB each)
  periodically generated from recent observations to capture higher-level
  insights and patterns.

Semantic retrieval uses cosine similarity over locally generated embeddings
(NO external Vector DBs). The embedding model is lazily loaded on first use
to keep agent creation rate >= 50 agents/second.

Redis persistence is supported with 30-day TTL for all memory keys:
  - agent:{agent_id}:observations  (JSON list)
  - agent:{agent_id}:reflections   (JSON list)
  - agent:{agent_id}:embeddings    (binary float32 arrays)

Timeout constraints:
  - add_observation: 1 second max
  - retrieve_relevant: 2 seconds max
  - generate_reflection: 10 seconds max
  - cleanup interval: 3,600 seconds (1 hour)

Performance targets:
  - Agent creation: >= 50 agents/second (lazy model loading)
  - Max 50 agents x ~700 KB = ~35 MB total agent memory

References:
  - README.md lines 84–112 (Memory System Structure)
  - AAP Section 0.1.2 (Memory Constraints)
  - AAP Section 0.4.1 (Redis Persistence)
  - AAP Section 0.7.4 (No external embedding APIs)
"""

import asyncio
import inspect
import time
import json
import numpy as np
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

from pydantic import BaseModel, Field
import structlog

# Module-level structured logger (AAP Section 0.7.6)
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_OBSERVATION_TIMEOUT_SECONDS: float = 1.0
_RETRIEVAL_TIMEOUT_SECONDS: float = 2.0
_REFLECTION_TIMEOUT_SECONDS: float = 10.0
_DEFAULT_EMBEDDING_DIM: int = 384
_DEFAULT_MAX_OBSERVATIONS: int = 1000
_DEFAULT_MAX_REFLECTIONS: int = 100
_DEFAULT_CLEANUP_INTERVAL: int = 3600
_DEFAULT_REDIS_TTL_DAYS: int = 30
_REFLECTION_MIN_OBSERVATIONS: int = 5
_REFLECTION_WINDOW_SIZE: int = 30


# ---------------------------------------------------------------------------
# Observation Data Model
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """A single agent observation with importance scoring and optional embedding.

    Each observation represents a timestamped record of what the agent noticed,
    did, or experienced.  Observations are stored in the observation stream
    (max 1,000 per agent, ~500 bytes each excluding the embedding vector).

    Attributes:
        content: The observation text content.
        importance: Importance score in the range [0.0, 10.0].
        timestamp: UTC datetime when the observation was recorded.
        embedding: Optional 384-dimensional embedding vector (lazy computed).
    """

    content: str
    importance: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the observation to a dictionary for Redis/logging.

        The embedding vector is excluded from serialization — it is stored
        separately in binary format.

        Returns:
            Dictionary with content, importance, and ISO-formatted timestamp.
        """
        return {
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Deserialize an Observation from a dictionary.

        Args:
            data: Dictionary with content, importance, and timestamp keys.

        Returns:
            A new Observation instance.
        """
        timestamp_raw = data.get("timestamp")
        if isinstance(timestamp_raw, str):
            ts = datetime.fromisoformat(timestamp_raw)
        elif isinstance(timestamp_raw, datetime):
            ts = timestamp_raw
        else:
            ts = datetime.utcnow()

        return cls(
            content=str(data.get("content", "")),
            importance=float(data.get("importance", 5.0)),
            timestamp=ts,
            embedding=None,  # Embeddings are loaded separately
        )


# ---------------------------------------------------------------------------
# Reflection Data Model
# ---------------------------------------------------------------------------
@dataclass
class Reflection:
    """A synthesized reflection derived from recent observations.

    Reflections are higher-level insights generated periodically from the
    observation stream.  They are stored in the reflection stream (max 100
    per agent, ~2 KB each).

    Attributes:
        content: The synthesized reflection text.
        source_observation_count: Number of observations that contributed.
        timestamp: UTC datetime when the reflection was generated.
        importance: Importance score (default 7.0 — reflections are high-value).
    """

    content: str
    source_observation_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 7.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the reflection to a dictionary for Redis/logging.

        Returns:
            Dictionary with content, source count, importance, and timestamp.
        """
        return {
            "content": self.content,
            "source_observation_count": self.source_observation_count,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reflection":
        """Deserialize a Reflection from a dictionary.

        Args:
            data: Dictionary with content, source_observation_count, timestamp,
                  and importance keys.

        Returns:
            A new Reflection instance.
        """
        timestamp_raw = data.get("timestamp")
        if isinstance(timestamp_raw, str):
            ts = datetime.fromisoformat(timestamp_raw)
        elif isinstance(timestamp_raw, datetime):
            ts = timestamp_raw
        else:
            ts = datetime.utcnow()

        return cls(
            content=str(data.get("content", "")),
            source_observation_count=int(data.get("source_observation_count", 0)),
            timestamp=ts,
            importance=float(data.get("importance", 7.0)),
        )


# ---------------------------------------------------------------------------
# AgentMemoryConfig — Pydantic V2 (AAP Section 0.7.1)
# ---------------------------------------------------------------------------
class AgentMemoryConfig(BaseModel):
    """Configuration for the AgentMemory dual-stream memory system.

    All cross-module interfaces use Pydantic V2 models for validation
    (AAP Section 0.7.1).

    Attributes:
        max_observations: Maximum number of observations (default 1,000).
        max_reflections: Maximum number of reflections (default 100).
        embedding_model_name: Name of the sentence-transformer model.
        embedding_dim: Dimensionality of the embedding vectors.
        cleanup_interval_seconds: Interval between automatic cleanups (3,600s).
        redis_ttl_days: TTL for Redis-persisted keys (30 days).
        enable_redis_persistence: Whether to persist to Redis.
        redis_key_prefix: Prefix for Redis keys (default "agent").
    """

    max_observations: int = Field(
        default=_DEFAULT_MAX_OBSERVATIONS,
        ge=1,
        description="Maximum number of observations per agent (default 1,000).",
    )
    max_reflections: int = Field(
        default=_DEFAULT_MAX_REFLECTIONS,
        ge=1,
        description="Maximum number of reflections per agent (default 100).",
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for semantic embeddings.",
    )
    embedding_dim: int = Field(
        default=_DEFAULT_EMBEDDING_DIM,
        ge=1,
        description="Dimensionality of embedding vectors (384 for MiniLM).",
    )
    cleanup_interval_seconds: int = Field(
        default=_DEFAULT_CLEANUP_INTERVAL,
        ge=1,
        description="Interval between automatic cleanups in seconds (3,600).",
    )
    redis_ttl_days: int = Field(
        default=_DEFAULT_REDIS_TTL_DAYS,
        ge=1,
        description="TTL in days for Redis-persisted memory keys (30).",
    )
    enable_redis_persistence: bool = Field(
        default=True,
        description="Enable Redis persistence for agent memory.",
    )
    redis_key_prefix: str = Field(
        default="agent",
        description="Prefix for Redis key patterns.",
    )


# ---------------------------------------------------------------------------
# AgentMemory — Dual-stream memory with semantic retrieval
# ---------------------------------------------------------------------------
class AgentMemory:
    """Dual-stream agent memory with semantic retrieval via all-MiniLM-L6-v2.

    Implements the memory system specified in README.md lines 84–112:

    * **Observation stream** — up to 1,000 timestamped entries with importance
      scoring (0.0–10.0) and 384-dimensional embeddings.
    * **Reflection stream** — up to 100 synthesized reflections derived from
      recent observations.

    Semantic retrieval computes cosine similarity between a query embedding
    and all stored observation embeddings using local inference only (no
    external Vector DBs per AAP Section 0.7.4).

    The sentence-transformer model is **lazily loaded** on first embedding
    request to ensure agent creation rate >= 50 agents/second.

    Args:
        agent_id: Unique identifier for the owning agent.
        config: Optional memory configuration; defaults are per-spec.
        redis_client: Optional async Redis client for persistence.
    """

    __slots__ = (
        "agent_id",
        "config",
        "_observations",
        "_reflections",
        "_embedding_model",
        "_redis",
        "_last_cleanup",
    )

    def __init__(
        self,
        agent_id: UUID,
        config: Optional[AgentMemoryConfig] = None,
        redis_client: Optional[Any] = None,
    ) -> None:
        self.agent_id: UUID = agent_id
        self.config: AgentMemoryConfig = config or AgentMemoryConfig()

        # In-memory dual streams
        self._observations: List[Observation] = []
        self._reflections: List[Reflection] = []

        # Lazy-loaded embedding model (CRITICAL: NOT loaded here)
        self._embedding_model: Optional[Any] = None

        # Optional Redis client for persistence
        self._redis: Optional[Any] = redis_client

        # Cleanup tracking
        self._last_cleanup: datetime = datetime.utcnow()

        logger.info(
            "agent_memory_initialized",
            agent_id=str(self.agent_id),
            max_observations=self.config.max_observations,
            max_reflections=self.config.max_reflections,
            redis_enabled=self._redis is not None and self.config.enable_redis_persistence,
        )

    # ------------------------------------------------------------------
    # Embedding Model (Lazy Loading — AAP Section 0.7.3)
    # ------------------------------------------------------------------
    def _get_embedding_model(self) -> Any:
        """Lazily load the sentence-transformer embedding model.

        The model is imported and instantiated only on the first call to
        avoid heavyweight initialization during agent creation, keeping
        the creation rate >= 50 agents/second.

        Returns:
            A SentenceTransformer model instance.
        """
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.config.embedding_model_name
            )
            logger.info(
                "embedding_model_loaded",
                agent_id=str(self.agent_id),
                model_name=self.config.embedding_model_name,
            )
        return self._embedding_model

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute a 384-dimensional embedding for the given text.

        Uses the lazily loaded all-MiniLM-L6-v2 model for local inference.
        No external embedding APIs are called (AAP Section 0.7.4).

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (embedding_dim,) with dtype float32.
        """
        model = self._get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        # Ensure correct shape — model.encode may return (1, dim) or (dim,)
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        if embedding.shape[0] != self.config.embedding_dim:
            # Fallback: pad or truncate to expected dimensionality
            result = np.zeros(self.config.embedding_dim, dtype=np.float32)
            copy_len = min(embedding.shape[0], self.config.embedding_dim)
            result[:copy_len] = embedding[:copy_len]
            return result
        return embedding

    # ------------------------------------------------------------------
    # Embedding Matrix Helpers
    # ------------------------------------------------------------------
    def _build_embedding_matrix(self) -> Optional[np.ndarray]:
        """Build an (N x embedding_dim) matrix from observation embeddings.

        Only observations with non-None embeddings are included.  Returns
        None if no observations have embeddings.

        Returns:
            Numpy array of shape (N, embedding_dim) or None.
        """
        embeddings = [
            obs.embedding
            for obs in self._observations
            if obs.embedding is not None
        ]
        if not embeddings:
            return None
        return np.vstack(embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # add_observation()  (README.md lines 96–98)
    # Timeout constraint: 1 second max
    # ------------------------------------------------------------------
    async def add_observation(
        self, content: str, importance: float = 5.0
    ) -> None:
        """Store a new observation with importance scoring and embedding.

        The importance value is clamped to the [0.0, 10.0] range.  An
        embedding is computed for the observation content to enable
        subsequent semantic retrieval.

        If the observation stream exceeds ``max_observations``, the lowest-
        importance (then oldest) observations are evicted.

        Timeout constraint: MUST complete within 1 second.

        Args:
            content: The observation text.
            importance: Importance score in [0.0, 10.0] (default 5.0).
        """
        start = time.monotonic()

        # Clamp importance to valid range
        importance = max(0.0, min(10.0, float(importance)))

        # Create observation
        observation = Observation(content=content, importance=importance)

        # Compute embedding
        observation.embedding = self._compute_embedding(content)

        # Append to stream
        self._observations.append(observation)

        # Enforce capacity limit — evict lowest-importance (then oldest)
        if len(self._observations) > self.config.max_observations:
            self._evict_observations()

        # Persist to Redis if enabled
        if self._is_redis_enabled():
            try:
                await self._persist_to_redis()
            except Exception as exc:
                logger.warning(
                    "redis_persist_failed",
                    agent_id=str(self.agent_id),
                    error=str(exc),
                )

        # Trigger cleanup if interval elapsed
        if self._should_cleanup():
            try:
                await self.cleanup()
            except Exception as exc:
                logger.warning(
                    "cleanup_failed_during_add",
                    agent_id=str(self.agent_id),
                    error=str(exc),
                )

        elapsed = time.monotonic() - start
        logger.info(
            "observation_added",
            agent_id=str(self.agent_id),
            importance=importance,
            observation_count=len(self._observations),
            elapsed_seconds=round(elapsed, 4),
        )

    # ------------------------------------------------------------------
    # retrieve_relevant()  (README.md lines 99–105)
    # Timeout constraint: 2 seconds max
    # ------------------------------------------------------------------
    async def retrieve_relevant(
        self, query: str, k: int = 5
    ) -> List[Observation]:
        """Retrieve the *k* most relevant observations via cosine similarity.

        Implementation (per README.md):
        1. Generate embedding for query using local all-MiniLM-L6-v2.
        2. Calculate cosine similarity with all stored observation embeddings.
        3. Return top-k results sorted by descending similarity.

        Timeout constraint: MUST complete within 2 seconds.

        Args:
            query: The query text.
            k: Number of results to return (default 5).

        Returns:
            List of Observation instances ordered by relevance (highest first).
        """
        start = time.monotonic()

        if not self._observations:
            return []

        # Compute query embedding
        query_embedding = self._compute_embedding(query)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0.0:
            return []

        # Build embedding matrix from observations
        embedding_matrix = self._build_embedding_matrix()
        if embedding_matrix is None or embedding_matrix.shape[0] == 0:
            return []

        # Compute cosine similarity: (N x D) · (D,) → (N,)
        dot_products = np.dot(embedding_matrix, query_embedding)
        norms = np.linalg.norm(embedding_matrix, axis=1)

        # Avoid division by zero
        denom = norms * query_norm
        denom = np.where(denom == 0.0, 1e-10, denom)
        similarities = dot_products / denom

        # Select top-k indices (descending similarity)
        actual_k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:actual_k]

        # Map indices back to observations with embeddings
        obs_with_embeddings = [
            obs for obs in self._observations if obs.embedding is not None
        ]
        results = [
            obs_with_embeddings[idx]
            for idx in top_indices
            if idx < len(obs_with_embeddings)
        ]

        elapsed = time.monotonic() - start
        logger.info(
            "memory_retrieval",
            agent_id=str(self.agent_id),
            query_length=len(query),
            results_count=len(results),
            elapsed_seconds=round(elapsed, 4),
        )

        return results

    # ------------------------------------------------------------------
    # generate_reflection()  (README.md lines 107–108)
    # Timeout constraint: 10 seconds max
    # ------------------------------------------------------------------
    async def generate_reflection(self) -> Optional[Reflection]:
        """Synthesize recent observations into a higher-level reflection.

        Examines the most recent observations (up to the configured window
        size) and generates a structured summary.  If fewer than 5
        observations are available, no reflection is generated.

        The reflection is appended to the reflection stream and, if the
        stream exceeds ``max_reflections``, the oldest reflections are
        evicted.

        Timeout constraint: MUST complete within 10 seconds.

        Returns:
            The newly generated Reflection, or None if insufficient data.
        """
        start = time.monotonic()

        if len(self._observations) < _REFLECTION_MIN_OBSERVATIONS:
            logger.debug(
                "reflection_skipped_insufficient_observations",
                agent_id=str(self.agent_id),
                observation_count=len(self._observations),
                required=_REFLECTION_MIN_OBSERVATIONS,
            )
            return None

        # Select the most recent observations within the reflection window
        recent_obs = self._observations[-_REFLECTION_WINDOW_SIZE:]
        source_count = len(recent_obs)

        # Compute average importance of recent observations
        avg_importance = sum(o.importance for o in recent_obs) / source_count

        # Group observations by approximate importance tier
        high_importance = [o for o in recent_obs if o.importance >= 7.0]
        medium_importance = [o for o in recent_obs if 4.0 <= o.importance < 7.0]
        low_importance = [o for o in recent_obs if o.importance < 4.0]

        # Build structured synthesis text
        synthesis_parts: List[str] = []
        synthesis_parts.append(
            f"Reflection based on {source_count} recent observations "
            f"(avg importance: {avg_importance:.1f})."
        )

        if high_importance:
            high_summaries = "; ".join(
                o.content[:120] for o in high_importance[:5]
            )
            synthesis_parts.append(
                f"High-priority items ({len(high_importance)}): {high_summaries}"
            )

        if medium_importance:
            med_summaries = "; ".join(
                o.content[:80] for o in medium_importance[:5]
            )
            synthesis_parts.append(
                f"Standard items ({len(medium_importance)}): {med_summaries}"
            )

        if low_importance:
            synthesis_parts.append(
                f"Routine items: {len(low_importance)} low-priority observations."
            )

        # Compose the final reflection content
        synthesis = " | ".join(synthesis_parts)

        # Compute reflection importance based on observation importance stats
        reflection_importance = min(10.0, max(5.0, avg_importance + 2.0))

        reflection = Reflection(
            content=synthesis,
            source_observation_count=source_count,
            importance=reflection_importance,
        )

        # Append to reflection stream
        self._reflections.append(reflection)

        # Enforce capacity limit — evict oldest
        while len(self._reflections) > self.config.max_reflections:
            self._reflections.pop(0)

        # Persist to Redis if enabled
        if self._is_redis_enabled():
            try:
                await self._persist_to_redis()
            except Exception as exc:
                logger.warning(
                    "redis_persist_failed_reflection",
                    agent_id=str(self.agent_id),
                    error=str(exc),
                )

        elapsed = time.monotonic() - start
        logger.info(
            "reflection_generated",
            agent_id=str(self.agent_id),
            source_count=source_count,
            reflection_count=len(self._reflections),
            elapsed_seconds=round(elapsed, 4),
        )

        return reflection

    # ------------------------------------------------------------------
    # get_recent()  (README.md lines 110–111)
    # ------------------------------------------------------------------
    async def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the *n* most recent observations as dictionaries.

        Results are ordered most-recent-first.  No embedding computation
        is performed — this is a quick in-memory operation.

        Args:
            n: Number of recent observations to return (default 10).

        Returns:
            List of observation dicts (most recent first).
        """
        n = max(1, n)
        return [obs.to_dict() for obs in self._observations[-n:][::-1]]

    # ------------------------------------------------------------------
    # cleanup()
    # ------------------------------------------------------------------
    async def cleanup(self) -> None:
        """Perform periodic memory cleanup.

        * Enforces the ``max_observations`` limit by evicting lowest-
          importance (then oldest) observations.
        * Enforces the ``max_reflections`` limit by evicting oldest
          reflections.
        * Updates the last cleanup timestamp.

        Cleanup is triggered automatically by ``add_observation`` when
        the configured interval (default 3,600 seconds) has elapsed.
        """
        initial_obs_count = len(self._observations)
        initial_ref_count = len(self._reflections)

        # Enforce observation limit
        if len(self._observations) > self.config.max_observations:
            self._evict_observations()

        # Enforce reflection limit
        while len(self._reflections) > self.config.max_reflections:
            self._reflections.pop(0)

        observations_removed = initial_obs_count - len(self._observations)
        reflections_removed = initial_ref_count - len(self._reflections)

        # Update cleanup timestamp
        self._last_cleanup = datetime.utcnow()

        # Persist cleaned state to Redis
        if self._is_redis_enabled():
            try:
                await self._persist_to_redis()
            except Exception as exc:
                logger.warning(
                    "redis_persist_failed_cleanup",
                    agent_id=str(self.agent_id),
                    error=str(exc),
                )

        logger.info(
            "memory_cleanup",
            agent_id=str(self.agent_id),
            observations_removed=observations_removed,
            reflections_removed=reflections_removed,
            final_observation_count=len(self._observations),
            final_reflection_count=len(self._reflections),
        )

    # ------------------------------------------------------------------
    # Utility / Stats Methods
    # ------------------------------------------------------------------
    def get_observation_count(self) -> int:
        """Return the current number of observations in the stream."""
        return len(self._observations)

    def get_reflection_count(self) -> int:
        """Return the current number of reflections in the stream."""
        return len(self._reflections)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return comprehensive memory statistics for monitoring.

        Returns:
            Dictionary with observation_count, reflection_count,
            estimated_memory_bytes, last_cleanup, and redis_enabled.
        """
        obs_count = len(self._observations)
        ref_count = len(self._reflections)

        # Estimate memory footprint
        # Observations: ~500 bytes content + 384*4 bytes embedding = ~2,036 bytes
        # Reflections: ~2,000 bytes content
        obs_bytes = obs_count * (500 + self.config.embedding_dim * 4)
        ref_bytes = ref_count * 2000
        estimated_memory_bytes = obs_bytes + ref_bytes

        return {
            "agent_id": str(self.agent_id),
            "observation_count": obs_count,
            "reflection_count": ref_count,
            "estimated_memory_bytes": estimated_memory_bytes,
            "max_observations": self.config.max_observations,
            "max_reflections": self.config.max_reflections,
            "last_cleanup": self._last_cleanup.isoformat(),
            "redis_enabled": self._is_redis_enabled(),
            "embedding_model_loaded": self._embedding_model is not None,
        }

    # ------------------------------------------------------------------
    # Redis Persistence Methods  (AAP Section 0.4.1)
    # ------------------------------------------------------------------
    async def _persist_to_redis(self) -> None:
        """Persist the current memory state to Redis.

        Stores observations and reflections as JSON lists and embeddings
        as binary data.  All keys use the pattern:
            {prefix}:{agent_id}:{stream}
        with a TTL of ``redis_ttl_days`` (default 30 days).
        """
        if not self._is_redis_enabled():
            return

        redis = self._redis
        prefix = self.config.redis_key_prefix
        agent_str = str(self.agent_id)
        ttl_seconds = int(
            timedelta(days=self.config.redis_ttl_days).total_seconds()
        )

        # --- Observations ---
        obs_key = f"{prefix}:{agent_str}:observations"
        obs_data = json.dumps([obs.to_dict() for obs in self._observations])
        try:
            await self._maybe_await(redis.set(obs_key, obs_data, ex=ttl_seconds))
        except Exception as exc:
            logger.warning(
                "redis_set_observations_failed",
                agent_id=agent_str,
                error=str(exc),
            )

        # --- Reflections ---
        ref_key = f"{prefix}:{agent_str}:reflections"
        ref_data = json.dumps([ref.to_dict() for ref in self._reflections])
        try:
            await self._maybe_await(redis.set(ref_key, ref_data, ex=ttl_seconds))
        except Exception as exc:
            logger.warning(
                "redis_set_reflections_failed",
                agent_id=agent_str,
                error=str(exc),
            )

        # --- Embeddings (binary) ---
        emb_key = f"{prefix}:{agent_str}:embeddings"
        emb_matrix = self._build_embedding_matrix()
        if emb_matrix is not None:
            emb_bytes = emb_matrix.astype(np.float32).tobytes()
            try:
                await self._maybe_await(redis.set(emb_key, emb_bytes, ex=ttl_seconds))
            except Exception as exc:
                logger.warning(
                    "redis_set_embeddings_failed",
                    agent_id=agent_str,
                    error=str(exc),
                )

    async def _load_from_redis(self) -> None:
        """Load memory state from Redis, replacing the current in-memory data.

        Observations and reflections are deserialized from JSON.  Embeddings
        are loaded from binary and re-attached to the observations list.
        """
        if not self._is_redis_enabled():
            return

        redis = self._redis
        prefix = self.config.redis_key_prefix
        agent_str = str(self.agent_id)

        # --- Observations ---
        obs_key = f"{prefix}:{agent_str}:observations"
        try:
            obs_raw = await self._maybe_await(redis.get(obs_key))

            if obs_raw is not None:
                if isinstance(obs_raw, bytes):
                    obs_raw = obs_raw.decode("utf-8")
                obs_list = json.loads(obs_raw)
                self._observations = [
                    Observation.from_dict(d) for d in obs_list
                ]
        except Exception as exc:
            logger.warning(
                "redis_load_observations_failed",
                agent_id=agent_str,
                error=str(exc),
            )

        # --- Reflections ---
        ref_key = f"{prefix}:{agent_str}:reflections"
        try:
            ref_raw = await self._maybe_await(redis.get(ref_key))

            if ref_raw is not None:
                if isinstance(ref_raw, bytes):
                    ref_raw = ref_raw.decode("utf-8")
                ref_list = json.loads(ref_raw)
                self._reflections = [
                    Reflection.from_dict(d) for d in ref_list
                ]
        except Exception as exc:
            logger.warning(
                "redis_load_reflections_failed",
                agent_id=agent_str,
                error=str(exc),
            )

        # --- Embeddings (binary) ---
        emb_key = f"{prefix}:{agent_str}:embeddings"
        try:
            emb_raw = await self._maybe_await(redis.get(emb_key))

            if emb_raw is not None:
                if isinstance(emb_raw, str):
                    emb_raw = emb_raw.encode("latin-1")
                emb_array = np.frombuffer(emb_raw, dtype=np.float32)
                dim = self.config.embedding_dim
                num_embeddings = len(emb_array) // dim
                emb_matrix = emb_array[: num_embeddings * dim].reshape(
                    num_embeddings, dim
                )
                # Re-attach embeddings to observations
                for idx, obs in enumerate(self._observations):
                    if idx < num_embeddings:
                        obs.embedding = emb_matrix[idx].copy()
        except Exception as exc:
            logger.warning(
                "redis_load_embeddings_failed",
                agent_id=agent_str,
                error=str(exc),
            )

        logger.info(
            "memory_loaded_from_redis",
            agent_id=agent_str,
            observation_count=len(self._observations),
            reflection_count=len(self._reflections),
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------
    def _is_redis_enabled(self) -> bool:
        """Check whether Redis persistence is enabled and a client is set."""
        return self._redis is not None and self.config.enable_redis_persistence

    @staticmethod
    async def _maybe_await(result: Any) -> Any:
        """Await *result* if it is a coroutine or awaitable, else return it.

        This handles both truly ``async def`` Redis methods and wrapper
        methods (e.g. fakeredis) that return awaitables without being
        flagged by ``asyncio.iscoroutinefunction``.
        """
        if inspect.isawaitable(result):
            return await result
        return result

    def _should_cleanup(self) -> bool:
        """Determine whether the cleanup interval has elapsed."""
        elapsed = (datetime.utcnow() - self._last_cleanup).total_seconds()
        return elapsed > self.config.cleanup_interval_seconds

    def _evict_observations(self) -> None:
        """Evict lowest-importance observations to stay within max capacity.

        Sorts observations by (importance ASC, timestamp ASC) and removes
        the lowest entries until the count is within ``max_observations``.
        """
        if len(self._observations) <= self.config.max_observations:
            return

        # Sort by importance ascending, then by timestamp ascending (oldest first)
        self._observations.sort(
            key=lambda obs: (obs.importance, obs.timestamp.timestamp())
        )

        # Keep only the top max_observations entries (highest importance)
        excess = len(self._observations) - self.config.max_observations
        self._observations = self._observations[excess:]

        # Re-sort by timestamp to maintain chronological order
        self._observations.sort(key=lambda obs: obs.timestamp.timestamp())
