"""Discrepancy Injection System test package — Project 3 Transaction Workflows.

Tests for the configurable discrepancy injector implementing 35+ discrepancy types
across four categories: P2P (15 types), O2C (10 types), GL (5 types), and
Control (5 types). Validates injection rate control, ground truth label generation,
parameter-bounded injection, Easy (70%) / Medium (30%) difficulty distribution,
catalog completeness, and individual discrepancy behaviors.

Test Modules:
    test_discrepancy_injection
        Injection orchestration: rate control (±1% accuracy), difficulty distribution
        (70/30/0), circuit breaker (20 failures → open, 30s recovery), parameter
        validation/auto-adjust, EventBus integration, 5s timeout, deterministic
        reproducibility, DiscrepancyConfig and InjectionResult Pydantic models.

    test_ground_truth
        Ground truth records: 16-field GroundTruthRecord schema validation, record
        creation, to_dict/from_dict serialization round-trip, JSON and CSV output,
        transaction linkage (valid UUIDs), 100% coverage guarantee, metrics tracking.

    test_individual_discrepancies
        Parametrized tests for all 35+ concrete discrepancy types. Validates
        BaseDiscrepancy ABC contract, 6 ClassVar attributes (type_code, category,
        difficulty, name, description, detection_method), inject() return contract,
        deep copy independence, parameter bounds, and deterministic behavior.
        Category-specific tests for P2P, O2C, GL, and Control types.

    test_discrepancy_catalog
        Catalog completeness: 35+ types registered (15 P2P + 10 O2C + 5 GL +
        5 Control), type code uniqueness, CatalogEntry Pydantic model, lookup by
        type code/category/difficulty, parameter bounds per type, implementation
        class resolution (lazy import), YAML config loading, metrics.

Testing Standards:
    - pytest markers: @pytest.mark.discrepancy, @pytest.mark.asyncio
    - Monetary values: Decimal only, NEVER float (prec=28, ROUND_HALF_UP)
    - Mocking: MagicMock for sync, AsyncMock for async dependencies
    - Fixtures: Uses conftest.py Section 8 P3 fixtures (mock_discrepancy_injector,
      sample_discrepancy_config, sample_discrepancy_record, deterministic_rng,
      sample_generation_context, sample_purchase_order, sample_vendor_invoice)
    - Reproducibility: All inject() calls use seeded random.Random(42)
    - Coverage target: ≥ 80% line coverage

References:
    - AAP Section 0.5.1 Group 5: Discrepancy Injection System
    - AAP Section 0.5.1 Group 8: Test files
    - AAP Section 0.7.5: Discrepancy Injection Rules
    - AAP Section 0.7.6: Testing Conventions
"""
