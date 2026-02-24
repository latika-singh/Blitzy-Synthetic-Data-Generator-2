"""
Test suite for the External World Simulation module (F-005).

Tests cover:
- test_behavior_profiles: BehaviorProfile Pydantic models, predefined templates,
  BehaviorProfileFactory, TierDistribution, and all sub-models (PaymentBehavior,
  OrderBehavior, InvoiceBehavior).
- test_external_entity_manager: ExternalWorldManager tiered entity pool management
  (Strategic 10%, Standard 30%, Transactional 60%), behavior profile assignment,
  interaction generation delegation, and optional Project 1 REST API integration.
- test_simulators: CustomerSimulator (order generation, payment simulation, dispute
  handling), VendorSimulator (invoice generation with 95% PO match, goods delivery,
  inquiry response), BankSimulator (statement generation, payment processing with
  check/ACH/wire clearing times).

Testing Standards:
- All external HTTP calls mocked (aioresponses or unittest.mock)
- No real Redis connections (fakeredis only)
- Async tests use pytest-asyncio
- Random seeds used for reproducible statistical tests (seed=42)
- Target coverage: ≥80%
"""
