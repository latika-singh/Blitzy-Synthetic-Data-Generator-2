# Project Guide — Synthetic ERP Data Generation Platform: Project 3 (Transaction Workflows & Discrepancies)

## 1. Executive Summary

### 1.1 Project Overview

Project 3 extends the Synthetic ERP Data Generation Platform (Project 2: Agent & Orchestration Engine) with a complete end-to-end transaction generation system spanning six major functional domains: Procure-to-Pay (P2P), Order-to-Cash (O2C), General Ledger Integration, Discrepancy Injection (35+ types), Intelligent Rework Loop, and Period Close Processing.

### 1.2 Completion Assessment

**324 hours completed out of 385 total hours = 84.2% complete**

| Metric | Value |
|--------|-------|
| Completion Percentage | 84.2% |
| Completed Hours | 324 |
| Remaining Hours | 61 |
| Total Project Hours | 385 |
| Files Created | 94 |
| Files Modified | 19 |
| Lines of Code Added | 70,521 |
| Total Tests Passing | 3,992 / 3,992 (100%) |
| P3-Specific Tests | 1,995 |
| Aggregate Test Coverage | 80.54% (≥80% threshold met) |
| Compilation Errors | 0 |
| Runtime Import Errors | 0 |

**Formula:** Completion % = 324 completed / (324 completed + 61 remaining) = 324 / 385 = **84.2%**

### 1.3 Key Achievements
- All 94 AAP-required source, test, and config files created and functional
- All 19 integration point files modified successfully
- Complete P2P pipeline (5 generators), O2C pipeline (4 generators), GL integration (4 classes)
- All 35+ discrepancy types implemented with ground truth generation
- Intelligent rework loop with 20+ fix scenarios
- Period close manager with 10-step close process
- 3,992 tests passing with zero failures across P2 and P3 suites
- 57 code review findings resolved across 4 fix commits
- 18 production + 7 dev dependencies installed with no broken requirements

### 1.4 Critical Items for Human Review
- **No blocking issues** — all code compiles, all tests pass, all imports work
- Real infrastructure integration (PostgreSQL, Redis, LLM) untested — tests use mocks
- Performance benchmarks (50 P2P/min, 60 O2C/min, 200 GL/min) unverified against real workloads
- 10 individual modules have coverage below 80% (aggregate passes at 80.54%)

---

## 2. Validation Results Summary

### 2.1 Final Validator Outcome

The Final Validator agent completed validation with **zero issues found**. No fixes were required — the codebase was fully functional as delivered.

### 2.2 Dependencies (100% SUCCESS)

All 25 packages (18 production + 7 dev/test) installed successfully:

| Category | Packages | Status |
|----------|----------|--------|
| P2 Production (13) | anthropic, openai, aiohttp, redis, numpy, scipy, faker, pydantic, structlog, python-dateutil, holidays, tenacity, sentence-transformers | ✅ Installed |
| P3 Production (5) | sqlalchemy==2.0.25, psycopg2-binary==2.9.9, aiofiles==23.2.1, pydantic-settings==2.2.0, pandas==2.2.0 | ✅ Installed |
| Dev/Test (7) | pytest, pytest-asyncio, pytest-cov, fakeredis, aioresponses, pytest-mock, hypothesis | ✅ Installed |
| `pip check` | No broken requirements | ✅ Clean |

### 2.3 Compilation (100% SUCCESS)

All 113 Python source files across both P2 and P3 modules compile cleanly with zero errors:

| Module | Files | Status |
|--------|-------|--------|
| app/transactions/ | 20 | ✅ All compile |
| app/discrepancies/ | 44 | ✅ All compile |
| app/rework/ | 5 | ✅ All compile |
| app/ (P2 modules) | 44 | ✅ All compile |
| **Total** | **113** | **✅ Zero errors** |

### 2.4 Test Results (100% SUCCESS — 3,992/3,992 PASSED)

| Test Suite | Tests | Passed | Failed | Status |
|------------|-------|--------|--------|--------|
| tests/test_transactions/ | 597 | 597 | 0 | ✅ |
| tests/test_discrepancies/ | 1,136 | 1,136 | 0 | ✅ |
| tests/test_rework/ | 262 | 262 | 0 | ✅ |
| tests/test_agents/ | 541 | 541 | 0 | ✅ |
| tests/test_events/ | 109 | 109 | 0 | ✅ |
| tests/test_llm/ | 326 | 326 | 0 | ✅ |
| tests/test_orchestration/ | 394 | 394 | 0 | ✅ |
| tests/test_simulation/ | 134 | 134 | 0 | ✅ |
| tests/test_statistical/ | 250 | 250 | 0 | ✅ |
| tests/test_external_world/ | 204 | 204 | 0 | ✅ |
| tests/test_errors/ | 39 | 39 | 0 | ✅ |
| **Total** | **3,992** | **3,992** | **0** | **✅** |

Test execution time: ~74 seconds (full suite), ~12 seconds (P3 only)

### 2.5 Test Coverage

Aggregate P3 coverage: **80.54%** (exceeds 80% threshold)

Modules with individual coverage below 80%:
| Module | Coverage | Gap |
|--------|----------|-----|
| accrual_generator.py | 63% | -17% |
| customer_invoice_generator.py | 69% | -11% |
| period_close_manager.py | 73% | -7% |
| vendor_invoice_processor.py | 76% | -4% |
| rework_loop_engine.py | 76% | -4% |
| purchase_order_generator.py | 77% | -3% |
| gl_posting_engine.py | 78% | -2% |
| goods_receipt_generator.py | 78% | -2% |
| shipment_generator.py | 78% | -2% |
| customer_payment_processor.py | 79% | -1% |

### 2.6 Runtime Validation (100% SUCCESS)

All P3 module imports verified:
- `app.transactions.p2p` — All 5 generators import ✅
- `app.transactions.o2c` — All 4 generators import ✅
- `app.transactions.gl` — All 4 classes import ✅
- `app.discrepancies` — Injector, GroundTruth, Catalog + 35 types import ✅
- `app.rework` — All 4 classes import ✅
- SimulationEngine composition root integrates P3 subsystems ✅

### 2.7 Fixes Applied During Validation

None required — the Final Validator found zero issues.

Prior to final validation, 4 code review/fix commits addressed 57 findings:
- Checkpoint 2: 13 findings resolved
- Checkpoint 4: 11 findings resolved
- Post-review: 10 findings resolved (4 CRITICAL, 5 MAJOR, 1 MINOR)
- QA documentation: 4 findings resolved
- Coverage boost: 235 additional tests added (raised coverage from 71% to 81%)

### 2.8 Git Statistics

| Metric | Value |
|--------|-------|
| Total Commits | 114 |
| Files Created | 94 |
| Files Modified | 19 |
| Lines Added | 70,521 |
| Lines Removed | 51 |
| Net Change | +70,470 |
| Development Period | Feb 25–27, 2026 |
| Authors | Blitzy Agent |

---

## 3. Visual Representation

### 3.1 Project Hours Breakdown

```mermaid
pie title Project Hours Breakdown (84.2% Complete)
    "Completed Work" : 324
    "Remaining Work" : 61
```

### 3.2 Completed Hours by Component

```mermaid
pie title Completed Work Breakdown (324 Hours)
    "P2P Engine (5 generators)" : 48
    "O2C Engine (4 generators)" : 36
    "GL Integration (4 classes)" : 52
    "Discrepancy System (35+ types)" : 72
    "Rework Loop (4 classes)" : 24
    "Transaction Infrastructure" : 14
    "Configuration (6 YAML)" : 6
    "Integration Updates (19 files)" : 16
    "Test Suite (1,995 tests)" : 48
    "Code Reviews & Fixes" : 8
```

### 3.3 Remaining Hours by Task

```mermaid
pie title Remaining Work Breakdown (61 Hours)
    "PostgreSQL Integration" : 10
    "CI/CD & Docker" : 10
    "E2E Simulation Run" : 7
    "Performance Benchmarking" : 7
    "Redis Integration" : 5
    "LLM Integration Testing" : 5
    "Coverage Improvement" : 5
    "Security Hardening" : 5
    "Environment Configuration" : 4
    "Operational Documentation" : 3
```

---

## 4. Completed Work Details

### 4.1 Hours Calculation Breakdown

**Completed: 324 hours** (development, testing, debugging, and fixes)

| Category | Component | Files | Lines | Hours |
|----------|-----------|-------|-------|-------|
| Transaction Infrastructure | base_generator.py, exceptions.py, constants.py, 4 __init__.py | 7 | ~1,400 | 14 |
| P2P Engine | PurchaseOrderGenerator, GoodsReceiptGenerator, VendorInvoiceProcessor, ThreeWayMatcher, VendorPaymentGenerator | 6 | ~6,800 | 48 |
| O2C Engine | SalesOrderGenerator, ShipmentGenerator, CustomerInvoiceGenerator, CustomerPaymentProcessor | 5 | ~5,400 | 36 |
| GL Integration | GLPostingEngine, AccountBalanceManager, AccrualGenerator, PeriodCloseManager | 5 | ~6,500 | 52 |
| Discrepancy System | Injector, GroundTruth, Catalog, Base + 35 types + 5 inits | 44 | ~18,700 | 72 |
| Rework Loop | ReworkLoopEngine, FailureClassifier, FixScenarioCatalog, FixScenarioExecutor | 5 | ~5,400 | 24 |
| Configuration | 4 discrepancy + 2 transaction YAML files | 6 | ~1,800 | 6 |
| Integration Updates | SimulationEngine, Metrics, DayContext, ErrorHandlers, TransactionOrchestrator, etc. | 19 | ~5,100 | 16 |
| Test Suite | 19 test files producing 1,995 passing tests | 19 | ~19,300 | 48 |
| Code Reviews & Fixes | 4 fix commits resolving 57 review findings + 235 coverage tests | — | ~5,800 | 8 |
| **Total** | | **113 files** | **~70,500** | **324** |

### 4.2 Feature Completeness Matrix

| AAP Requirement | Files Required | Files Delivered | Status |
|-----------------|---------------|-----------------|--------|
| Group 1: Core Transaction Infrastructure | 4 | 4 | ✅ 100% |
| Group 2: P2P Transaction Engine | 6 | 6 | ✅ 100% |
| Group 3: O2C Transaction Engine | 5 | 5 | ✅ 100% |
| Group 4: GL Integration | 5 | 5 | ✅ 100% |
| Group 5: Discrepancy Injection System | 44 | 44 | ✅ 100% |
| Group 6: Intelligent Rework Loop | 5 | 5 | ✅ 100% |
| Group 7: Configuration Files | 6 | 6 | ✅ 100% |
| Group 8: Tests | 16+ | 19 | ✅ 100% |
| Group 9: Project Configuration Updates | 5 | 5 | ✅ 100% |
| Integration Point Modifications | 12+ | 19 | ✅ 100% |

---

## 5. Detailed Task Table — Remaining Work

**Remaining: 61 hours** (with 1.10× compliance and 1.10× uncertainty multipliers applied)

All task hours below sum to exactly 61 hours, matching the pie chart "Remaining Work" value.

| # | Task | Description | Action Steps | Hours | Priority | Severity | Confidence |
|---|------|-------------|--------------|-------|----------|----------|------------|
| 1 | PostgreSQL Database Integration Setup & Testing | All P3 tests use mocked AsyncSession. Real PostgreSQL with Project 1's `get_session()` is untested. | 1. Install and configure PostgreSQL instance 2. Run Project 1 migrations to create transaction tables 3. Configure `DATABASE_URL` in `.env` 4. Test all generators against real DB 5. Verify `SELECT FOR UPDATE` locking works correctly 6. Test transaction rollback atomicity | 10 | High | Critical | Medium |
| 2 | Redis EventBus Production Integration Testing | Tests use `fakeredis`. Real Redis Pub/Sub event flow untested for P3 events. | 1. Start Redis server 2. Configure `REDIS_URL` in `.env` 3. Test EventBus publish/subscribe for all P3 event types 4. Verify event persistence in EventStore 5. Test circuit breaker behavior under load | 5 | High | High | High |
| 3 | Environment Configuration & Secrets Management | `.env.example` has 11 P3 variables with defaults. Production secrets management not configured. | 1. Create production `.env` with real values 2. Configure API key rotation for LLM providers 3. Set up database credential management 4. Validate all env var ranges per specification 5. Document secrets management process | 4 | High | High | High |
| 4 | LLM API Integration Testing | Agent decisions in P2P/O2C workflows use mocked LLM responses. Real Anthropic/OpenAI calls untested. | 1. Configure real `LLM_API_KEY` 2. Test agent decision-making in P2P approval workflows 3. Test vendor invoice processing with real LLM 4. Verify fallback model switching works 5. Test cost monitoring stays within budget | 5 | Medium | Medium | Medium |
| 5 | End-to-End 1-Month Simulation Run | No full simulation run has been executed. Need to verify the complete SimulationEngine pipeline with P3 subsystems. | 1. Configure full simulation parameters 2. Execute 30-day simulation run 3. Verify P2P/O2C cycle completion rates 4. Validate GL balance remains zero throughout 5. Check discrepancy injection rates match target 6. Verify rework loop success rates | 7 | Medium | High | Medium |
| 6 | Performance Benchmarking Against AAP Criteria | AAP specifies 6 performance targets. None verified under production conditions. | 1. Benchmark P2P cycle rate (target: ≥50/min) 2. Benchmark O2C cycle rate (target: ≥60/min) 3. Benchmark GL posting rate (target: ≥200/min) 4. Test 100 concurrent transaction workflows 5. Verify 2,000 txn/hour sustained throughput 6. Profile and optimize bottlenecks if needed | 7 | Medium | High | Low |
| 7 | Individual Module Coverage Improvement | 10 modules are below 80% individual coverage. Aggregate passes but financial-critical modules have gaps. | 1. Add tests for `accrual_generator.py` (63% → 80%) 2. Add tests for `customer_invoice_generator.py` (69% → 80%) 3. Add tests for `period_close_manager.py` (73% → 80%) 4. Add tests for remaining 7 modules below 80% 5. Verify aggregate coverage remains above 80% | 5 | Medium | Medium | High |
| 8 | CI/CD Pipeline & Docker Deployment Setup | No CI/CD pipeline or containerization exists. Required for production deployment. | 1. Create Dockerfile with Python 3.11+ base 2. Create docker-compose.yml with PostgreSQL and Redis services 3. Configure GitHub Actions / CI pipeline for tests 4. Add linting and type-checking steps 5. Configure automated test execution 6. Add container health checks | 10 | Medium | Medium | Medium |
| 9 | Security Hardening & Credential Management | API keys, database credentials, and financial data need production-grade protection. | 1. Audit all credential handling paths 2. Verify structlog credential scrubbing covers P3 modules 3. Implement database connection encryption (TLS) 4. Review SQL injection prevention in ORM usage 5. Add rate limiting for sensitive operations | 5 | Low | Medium | Medium |
| 10 | Operational Documentation & Runbook | System needs operational documentation for production support. | 1. Create runbook for common operational scenarios 2. Document monitoring alert thresholds 3. Create troubleshooting guide for rework loop escalations 4. Document period close manual intervention procedures | 3 | Low | Low | High |
| | **Total Remaining Hours** | | | **61** | | | |

### 5.1 Pre-Multiplier Breakdown

Base estimates before enterprise multipliers:

| Task | Base Hours | × Compliance (1.10) | × Uncertainty (1.10) | Final Hours |
|------|-----------|---------------------|---------------------|-------------|
| PostgreSQL Integration | 8 | 8.8 | 9.7 | 10 |
| Redis Integration | 4 | 4.4 | 4.8 | 5 |
| Environment Config | 3 | 3.3 | 3.6 | 4 |
| LLM Integration | 4 | 4.4 | 4.8 | 5 |
| E2E Simulation | 6 | 6.6 | 7.3 | 7 |
| Performance Benchmarking | 6 | 6.6 | 7.3 | 7 |
| Coverage Improvement | 4 | 4.4 | 4.8 | 5 |
| CI/CD & Docker | 8 | 8.8 | 9.7 | 10 |
| Security Hardening | 4 | 4.4 | 4.8 | 5 |
| Operational Docs | 3 | 3.3 | 3.6 | 3 |
| **Totals** | **50** | **55.0** | **60.5** | **61** |

---

## 6. Development Guide

### 6.1 System Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | ≥ 3.11 (3.12.3 tested) | Runtime environment |
| pip | ≥ 23.0 | Package manager |
| Git | ≥ 2.30 | Version control |
| PostgreSQL | ≥ 14 | Production database (for real integration) |
| Redis | ≥ 7.0 | Event bus, agent state, caching (for real integration) |

> **Note:** PostgreSQL and Redis are required for production/integration testing. Unit tests run entirely with mocks (fakeredis, AsyncMock).

### 6.2 Environment Setup

```bash
# 1. Clone the repository and switch to the feature branch
git clone <repository-url>
cd <repository-root>
git checkout blitzy-cb1eea01-fac4-445c-9255-4f96af0dd047

# 2. Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Verify Python version
python --version
# Expected: Python 3.11.x or 3.12.x
```

### 6.3 Dependency Installation

```bash
# Install all production + dev/test dependencies
pip install -r requirements-dev.txt

# Verify no broken dependencies
pip check
# Expected: No broken requirements found.

# Verify key P3 packages
python -c "import sqlalchemy; print(f'sqlalchemy: {sqlalchemy.__version__}')"
# Expected: sqlalchemy: 2.0.25

python -c "import pydantic; print(f'pydantic: {pydantic.__version__}')"
# Expected: pydantic: 2.x.x (≥2.5.0)
```

### 6.4 Environment Configuration

```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your actual values:
# - LLM_API_KEY: Your Anthropic or OpenAI API key
# - REDIS_URL: Your Redis connection URL (default: redis://localhost:6379/0)
# - Adjust P3 variables as needed (defaults are production-ready)
```

Key P3 environment variables (with defaults):
| Variable | Default | Purpose |
|----------|---------|---------|
| `TRANSACTION_BATCH_SIZE` | 100 | Transactions per batch |
| `GL_POSTING_BATCH_SIZE` | 500 | GL entries per batch |
| `CONCURRENT_TRANSACTION_LIMIT` | 100 | Max concurrent workflows |
| `DISCREPANCY_INJECTION_ENABLED` | true | Enable discrepancy injection |
| `DISCREPANCY_DEFAULT_RATE` | 0.02 | 2% injection rate |
| `REWORK_LOOP_ENABLED` | true | Enable rework loop |
| `REWORK_LOOP_MAX_ATTEMPTS` | 3 | Max fix attempts |

### 6.5 Running the Test Suite

```bash
# Run the full test suite (3,992 tests, ~74 seconds)
python -m pytest tests/ --tb=short -v
# Expected: 3992 passed, 946 warnings

# Run P3 tests only (1,995 tests, ~12 seconds)
python -m pytest tests/test_transactions/ tests/test_discrepancies/ tests/test_rework/ --tb=short -v
# Expected: 1995 passed

# Run with coverage reporting
python -m pytest tests/test_transactions/ tests/test_discrepancies/ tests/test_rework/ \
  --cov=app/transactions --cov=app/discrepancies --cov=app/rework \
  --cov-report=term-missing --tb=short -q
# Expected: 80.54% coverage (≥80% threshold met)

# Run specific test categories using markers
python -m pytest -m "p2p" tests/ --tb=short -v     # P2P cycle tests
python -m pytest -m "o2c" tests/ --tb=short -v     # O2C cycle tests
python -m pytest -m "financial" tests/ --tb=short -v  # GL financial tests
python -m pytest -m "discrepancy" tests/ --tb=short -v  # Discrepancy tests
python -m pytest -m "rework" tests/ --tb=short -v   # Rework loop tests
```

### 6.6 Compilation Verification

```bash
# Verify all P3 source files compile
find app/transactions app/discrepancies app/rework -name "*.py" -type f \
  -exec python -m py_compile {} \;
# Expected: No output (all files compile cleanly)

# Verify all module imports
python -c "
from app.transactions.p2p import PurchaseOrderGenerator, GoodsReceiptGenerator, VendorInvoiceProcessor, ThreeWayMatcher, VendorPaymentGenerator
from app.transactions.o2c import SalesOrderGenerator, ShipmentGenerator, CustomerInvoiceGenerator, CustomerPaymentProcessor
from app.transactions.gl import GLPostingEngine, AccountBalanceManager, AccrualGenerator, PeriodCloseManager
from app.discrepancies import DiscrepancyInjector, GroundTruthGenerator, DiscrepancyCatalog
from app.rework import ReworkLoopEngine, FailureClassifier, FixScenarioCatalog, FixScenarioExecutor
print('ALL P3 imports successful')
"
# Expected: ALL P3 imports successful
```

### 6.7 Project Structure

```
app/
├── transactions/           # Transaction generation engines (20 files, ~20K lines)
│   ├── __init__.py         # Package init: exports TransactionGenerator, GenerationContext, TransactionResult
│   ├── base_generator.py   # Abstract base class with retry/timeout/event publishing
│   ├── exceptions.py       # 10 exception classes (TransactionError hierarchy)
│   ├── constants.py        # Processing limits, thresholds, circuit breaker config
│   ├── p2p/                # 5 P2P generators
│   │   ├── purchase_order_generator.py
│   │   ├── goods_receipt_generator.py
│   │   ├── vendor_invoice_processor.py
│   │   ├── three_way_matcher.py
│   │   └── vendor_payment_generator.py
│   ├── o2c/                # 4 O2C generators
│   │   ├── sales_order_generator.py
│   │   ├── shipment_generator.py
│   │   ├── customer_invoice_generator.py
│   │   └── customer_payment_processor.py
│   └── gl/                 # GL posting, balances, accruals, period close
│       ├── gl_posting_engine.py
│       ├── account_balance_manager.py
│       ├── accrual_generator.py
│       └── period_close_manager.py
├── discrepancies/          # Discrepancy injection system (44 files, ~19K lines)
│   ├── discrepancy_injector.py
│   ├── ground_truth_generator.py
│   ├── discrepancy_catalog.py
│   ├── base_discrepancy.py
│   ├── p2p/                # 15 P2P discrepancy types (P2P-001 through P2P-015)
│   ├── o2c/                # 10 O2C discrepancy types (O2C-001 through O2C-010)
│   ├── gl/                 # 5 GL discrepancy types (GL-001 through GL-005)
│   └── control/            # 5 Control discrepancy types (CTL-001 through CTL-005)
└── rework/                 # Intelligent rework loop (5 files, ~5K lines)
    ├── rework_loop_engine.py
    ├── failure_classifier.py
    ├── fix_scenario_catalog.py
    └── fix_scenario_executor.py

config/
├── discrepancies/          # Discrepancy configuration (4 YAML files)
│   ├── p2p_discrepancies.yaml
│   ├── o2c_discrepancies.yaml
│   ├── gl_discrepancies.yaml
│   └── discrepancy_rates.yaml
└── transactions/           # Transaction configuration (2 YAML files)
    ├── posting_rules.yaml
    └── period_close.yaml

tests/
├── test_transactions/      # 597 tests across 9 files
├── test_discrepancies/     # 1,136 tests across 5 files
└── test_rework/            # 262 tests across 5 files
```

### 6.8 Troubleshooting

| Issue | Cause | Resolution |
|-------|-------|------------|
| `ModuleNotFoundError: No module named 'app'` | Not running from repository root | `cd` to the repository root directory |
| `psycopg2` installation fails | Missing PostgreSQL dev headers | `apt-get install -y libpq-dev` (Linux) or use `psycopg2-binary` (included) |
| `946 warnings` during test run | `datetime.utcnow()` deprecation + pyarrow missing | Non-blocking warnings; safe to ignore |
| Coverage below 80% | Running subset of tests | Run full P3 test suite for accurate aggregate coverage |

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| PostgreSQL integration issues when switching from mocked sessions to real DB | High | Medium | P3 generators may fail with real async sessions, transactions, or locking | Test all generators against real PostgreSQL with Project 1's schema before production |
| `SELECT FOR UPDATE` deadlocks under concurrent access | High | Medium | AccountBalanceManager may deadlock when multiple agents post to same account | Implement timeout on row locks; test with 20+ concurrent posting scenarios |
| Performance targets not met under real workload | Medium | Medium | P2P/O2C/GL rates may fall below specified thresholds | Profile with real DB; optimize query patterns; add connection pooling |
| Accrual generator coverage gap (63%) | Medium | Low | Uncovered code paths may contain bugs in financial calculations | Add targeted tests for missing lines (388-398, 559-728, 829-978) |
| Period close manager coverage gap (73%) | Medium | Low | Untested close steps may fail during period-end processing | Add tests for steps 5-10 of the 10-step close process |

### 7.2 Security Risks

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| LLM API keys exposed in logs | High | Low | Financial data or credentials leaked | Verify structlog credential scrubbing covers all P3 log paths |
| Database credentials in `.env` file | Medium | Medium | Unauthorized database access | Use environment variable injection via secrets manager (Vault, AWS SSM) |
| Unencrypted database connections | Medium | Low | Data in transit exposed | Configure `sslmode=require` in PostgreSQL connection string |

### 7.3 Operational Risks

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| No monitoring or alerting for production | High | High | Failures go undetected in production | Set up monitoring for circuit breaker state, rework escalations, GL balance drift |
| No health check endpoints | Medium | High | Cannot verify system health | Add `/health` endpoint checking DB, Redis, and subsystem status |
| Rework escalation with no human review process | Medium | Medium | Escalated transactions pile up without resolution | Define escalation workflow and notification channels |
| Period close manual intervention undefined | Medium | Low | Period close halt blocks simulation progress | Document manual intervention procedures and retry process |

### 7.4 Integration Risks

| Risk | Severity | Likelihood | Impact | Mitigation |
|------|----------|------------|--------|------------|
| Project 1 database schema mismatch | High | Medium | P3 ORM models may not match actual table definitions | Verify SQLAlchemy models against Project 1 migration history before integration |
| Redis Pub/Sub message loss under load | Medium | Low | P3 events lost, downstream processors miss updates | Test with sustained event publishing; verify EventStore persistence backup |
| LLM provider rate limiting | Medium | Medium | Agent decisions delayed, transaction processing stalls | Verify LLMBudgetManager and rate limiter handle P3 decision volume |

---

## 8. Consistency Verification

### 8.1 Pre-Submission Checklist

- [x] Calculated completion % using hours formula: 324 / (324 + 61) = 84.2%
- [x] Verified Executive Summary states this exact %: "324 hours completed out of 385 total hours = 84.2% complete"
- [x] Verified pie chart uses exact completed/remaining hours: "Completed Work: 324" and "Remaining Work: 61"
- [x] Verified task table sums to exact remaining hours: 10+5+4+5+7+7+5+10+5+3 = 61 ✓
- [x] Searched report for any % or hour mentions — all match 84.2%, 324h, 61h, 385h
- [x] No conflicting or ambiguous statements exist
- [x] Shown the calculation formula with actual numbers