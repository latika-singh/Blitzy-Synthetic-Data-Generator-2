"""
Test sub-package for the Event System module (F-007, app/events/).

Contains tests for:
- test_event_bus.py: Tests for dual-mode EventBus pub/sub — asyncio.Queue for
  in-memory mode, optional Redis Pub/Sub, priority handling (URGENT/HIGH/NORMAL/LOW),
  async dispatch to handlers, subscribe/unsubscribe lifecycle, handler timeout
  enforcement (5s), publish timeout enforcement (1s), event persistence integration,
  and context manager support.
- test_event_store.py: Tests for EventStore JSONB persistence — Redis-backed storage
  with event:{uuid} key pattern, 7-day TTL (604800 seconds), UUID-indexed queries,
  query filters (simulation_id, event_type, start_date, end_date), event replay
  for state reconstruction, serialization round-trip for all 8 event types, and
  event count/metrics tracking.

Testing standards:
- Uses fakeredis for all Redis mock operations (no external Redis dependency)
- Uses pytest-asyncio for async event handling tests
- Target coverage: ≥80%
"""
