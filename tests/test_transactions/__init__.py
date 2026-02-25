"""Transaction Generation Engine test package — Project 3.

This package contains comprehensive tests for the core transaction generation system
of the Synthetic ERP Data Generation Platform (Project 3: Transaction Workflows &
Discrepancies). Tests exercise all three transaction engine sub-packages and their
integration with the GL posting layer.

Test Modules:
    test_base_generator      — TransactionGenerator ABC contract, GenerationContext
                               and TransactionResult Pydantic V2 models, constructor
                               injection (ADR-003), shared helpers (seeded RNG, event
                               publishing, GL delegation, sequential numbering)

    test_p2p_cycle           — Full Procure-to-Pay cycle integration:
                               PurchaseOrderGenerator → GoodsReceiptGenerator →
                               VendorInvoiceProcessor (ThreeWayMatcher) →
                               VendorPaymentGenerator. Approval chains, GL balance
                               verification, artifact completeness, event publication.

    test_o2c_cycle           — Full Order-to-Cash cycle integration:
                               SalesOrderGenerator → ShipmentGenerator →
                               CustomerInvoiceGenerator → CustomerPaymentProcessor.
                               Credit checks, FIFO allocation, overpayment handling,
                               short pay, GL balance verification.

    test_gl_posting          — GLPostingEngine and AccountBalanceManager: balance
                               validation (DR = CR within $0.01), trial balance,
                               account/period validation, concurrent posting safety,
                               rollback atomicity, circuit breaker, normal balance
                               direction. Contains 5 MANDATORY critical financial
                               test scenarios per AAP Section 0.7.6.

    test_three_way_matching  — ThreeWayMatcher standalone service: price tolerance
                               (±5%), quantity tolerance (±2%), match statuses,
                               line-item variance calculations, Pydantic V2 models.

    test_period_close        — PeriodCloseManager 10-step close process and
                               AccrualGenerator: AP accruals (GRNI), AR accruals,
                               straight-line daily method, reversing entries,
                               reconciliations, trial balance, balance sheet,
                               period state transitions.

Testing Conventions (AAP Section 0.7.6):
    - All monetary amounts use Python Decimal (prec=28, ROUND_HALF_UP)
    - AsyncMock for database sessions (Project 1 get_session() pattern)
    - fakeredis for Redis operations (event bus/store)
    - Deterministic seeding via random.Random(42) fixtures
    - Factory fixtures for parameterized test data generation
    - Property-based testing with hypothesis for financial invariants

Pytest Markers:
    @pytest.mark.p2p         — P2P cycle tests
    @pytest.mark.o2c         — O2C cycle tests
    @pytest.mark.financial   — GL balance assertion tests
    @pytest.mark.asyncio     — Async test methods
"""
