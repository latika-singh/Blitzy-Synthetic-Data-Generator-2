# SYNTHETIC ERP DATA GENERATION PLATFORM
## Project 2: Agent & Orchestration Engine | Project 3: Transaction Workflows & Discrepancies

---

## 🎯 PROJECT OBJECTIVES

### Project 2 — Agent & Orchestration Engine

**BUILD** an AI-powered agent and orchestration system that simulates realistic employee behavior using autonomous agents with memory, reflection, and LLM-powered decision-making, along with workflow orchestration, time management, and external world simulation components.

**DELIVERABLE:** A fully functional agent-based simulation engine that can:
1. Create and manage autonomous AI agents representing employees
2. Make realistic business decisions using hybrid statistical + LLM approach
3. Orchestrate workflows with approval chains and routing logic
4. Simulate external entities (customers, vendors, banks) with tiered behavior
5. Manage simulation time with business calendars and fiscal periods
6. Generate transaction triggers using statistical models
7. Coordinate all components through event-driven architecture

### Project 3 — Transaction Workflows & Discrepancies

**BUILD** a complete end-to-end transaction generation system that produces realistic Procure-to-Pay and Order-to-Cash cycles, posts balanced General Ledger entries, injects configurable discrepancies with ground truth labels, autonomously corrects unintentional errors via an intelligent rework loop, and manages fiscal period close processing.

**DELIVERABLE:** A fully functional transaction workflow engine that can:
1. Generate complete P2P cycles (Purchase Order → Goods Receipt → Vendor Invoice → Vendor Payment)
2. Generate complete O2C cycles (Sales Order → Shipment → Customer Invoice → Customer Payment)
3. Post balanced journal entries to the General Ledger (DR = CR within $0.01)
4. Inject 35+ discrepancy types with full ground truth label generation
5. Autonomously classify and correct validation failures (max 3 attempts per transaction)
6. Execute 10-step fiscal period close with accruals, deferrals, depreciation, and trial balance validation
7. Maintain continuous financial integrity (trial balance, sub-ledger reconciliation, balance sheet equation)

---

## 🚀 GETTING STARTED

### Prerequisites

- **Python 3.11+** (tested with 3.11.7 and 3.12.x)
- **Redis** (optional — tests use `fakeredis`; required for production event persistence and agent state caching)
- **PostgreSQL** (optional — tests use mocks/fixtures; required for production transaction persistence via Project 1's database layer)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# Install production dependencies
pip install -r requirements.txt

# Install development/test dependencies (includes production deps)
pip install -r requirements-dev.txt
```

### Configuration

```bash
# Copy the environment template and configure
cp .env.example .env

# Edit .env with your settings (LLM API keys, Redis URL, etc.)
# See the "New Environment Variables" section below for all Project 3 variables.
```

### Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v --tb=short

# Run only Project 3 transaction tests
python -m pytest tests/test_transactions/ -v

# Run only discrepancy injection tests
python -m pytest tests/test_discrepancies/ -v

# Run only rework loop tests
python -m pytest tests/test_rework/ -v

# Run with coverage report
python -m pytest tests/ --cov=app --cov-report=term-missing
```

### Compilation Check

```bash
# Verify all source files compile without errors
find app/ -name "*.py" -type f -exec python -m py_compile {} \;
```

---

## ✅ SUCCESS CRITERIA (20 Measurable Thresholds)

### Agent System Performance (5 criteria)
1. **Agent Creation Rate**: Generate and initialize ≥ 50 agents/second with full profiles
2. **Decision Response Time**: Agent LLM decisions complete within p95 < 5 seconds per decision
3. **Memory Operations**: Agent memory storage/retrieval operations < 100ms per operation
4. **Agent Concurrency**: Support ≥ 20 agents processing work items simultaneously
5. **Decision Success Rate**: ≥ 98% of agent decisions produce valid outputs (pass validation)

### LLM Integration (5 criteria)
6. **LLM Request Success Rate**: ≥ 99% of LLM API calls succeed (with retry logic)
7. **LLM Cost Per Month**: Stay within $100/month budget for 1 month of data (≤ 5,000 txns)
8. **LLM Latency**: Average LLM response time < 3 seconds (p95 < 5 seconds)
9. **Prompt Token Efficiency**: Average prompt size ≤ 1,500 tokens per decision
10. **Response Validation Rate**: 100% of LLM responses validated before use

### Orchestration Performance (5 criteria)
11. **Workflow Routing Speed**: Route work items to agents within < 500ms per item
12. **Approval Chain Execution**: Multi-level approvals complete within < 10 seconds
13. **Transaction Completion Rate**: ≥ 99.5% of transactions reach completed state
14. **Workflow Concurrency**: Handle ≥ 100 concurrent workflow instances
15. **Event Processing Throughput**: Process ≥ 500 events/second through event bus

### Time & External World (5 criteria)
16. **Time Advancement Rate**: Advance simulation by 1 business day within < 30 seconds
17. **Calendar Operations**: Date calculations and business day checks < 10ms
18. **External Entity Response**: External entities generate responses within < 2 seconds
19. **Statistical Model Performance**: Generate random samples ≥ 10,000/second
20. **Month Generation Time**: Complete 1 month of simulation (2,000 txns) in 4-8 hours

---

## ✅ PROJECT 3 SUCCESS CRITERIA (24 Measurable Thresholds)

### Transaction Generation Performance (6 criteria)
1. **P2P Transaction Rate**: Generate ≥ 50 complete P2P cycles per minute
2. **O2C Transaction Rate**: Generate ≥ 60 complete O2C cycles per minute
3. **GL Posting Rate**: Post ≥ 200 journal entries per minute to the General Ledger
4. **Transaction Completion Rate**: ≥ 99.5% of all transactions reach completed state
5. **Concurrent Transactions**: Support ≥ 100 concurrent transaction workflows
6. **Transaction Throughput**: Process ≥ 2,000 transactions/hour sustained

### Discrepancy Injection (5 criteria)
7. **Rate Control**: Actual injection rate within ±1% of the configured target rate (default 2%)
8. **Difficulty Distribution**: Easy 70% / Medium 30% / Hard 0% with ±5% tolerance
9. **Parameter Bounds**: All discrepancy parameters fall within configured bounds
10. **Ground Truth Coverage**: 100% of injected discrepancies have a corresponding ground truth record
11. **Transaction Linkage**: All ground truth records reference valid transaction IDs

### Financial Integrity (6 criteria)
12. **GL Balance**: SUM(debits) = SUM(credits) within $0.01 for every journal entry
13. **Trial Balance**: Cumulative trial balance equals zero within $0.01 after every batch
14. **Balance Sheet Equation**: Assets = Liabilities + Equity within $0.01 at all times
15. **Sub-ledger Reconciliation**: AR, AP, and Inventory sub-ledgers match GL control accounts within $0.01
16. **Atomicity**: Failed multi-step GL postings fully rollback — no partial postings permitted
17. **Concurrent Posting**: Multiple agents post to the same GL account simultaneously with correct final balances

### Workflow Integrity (7 criteria)
18. **P2P Sequence**: PO → Receipt → Invoice → Payment — no out-of-order processing
19. **O2C Sequence**: Order → Shipment → Invoice → Payment — strict ordering enforced
20. **Approval Chains**: All transactions above configured thresholds require and receive approval
21. **Three-Way Match**: Tolerances enforced — price ±5%, quantity ±2%
22. **FIFO Allocation**: Customer payments allocated oldest invoice first
23. **Rework Loop**: Maximum 3 fix attempts per transaction, 30-second timeout, then escalate
24. **Period Close**: 10-step close process completes with trial balance at zero

---

## 📋 IN SCOPE

### Core Components to Build

#### 1. Agent System

**Agent Base Architecture:**
- **BaseAgent abstract class** with core agent capabilities
- **AgentConfig** dataclass for agent configuration
- **AgentState** enum (idle, thinking, acting, waiting, error)
- **Agent traits system** (thoroughness, risk_tolerance, efficiency, compliance)
- **Work queue management** per agent
- **Agent lifecycle** management (creation, activation, deactivation)

**Specialized Agent Types:**
```python
# Implement 12 specialized agent types:
- APClerkAgent          # Accounts Payable clerk
- APManagerAgent        # AP manager with approval authority
- ARClerkAgent          # Accounts Receivable clerk
- ARManagerAgent        # AR manager
- AccountantAgent       # General accountant
- SeniorAccountantAgent # Senior accountant
- PurchasingAgent       # Buyer/purchasing agent
- PurchasingManagerAgent # Purchasing manager
- WarehouseClerkAgent   # Warehouse worker
- WarehouseManagerAgent # Warehouse manager
- ControllerAgent       # Controller (high-level approvals)
- CFOAgent             # CFO (final authority)
```

**Memory System:**
```python
class AgentMemory:
    """Agent memory stream with importance scoring."""
    
    # Components to implement:
    - observation_stream: List[Observation]
        # Each observation: (timestamp, content, importance_score)
    
    - reflection_stream: List[Reflection]
        # Periodic synthesis of observations
    
    - add_observation(content, importance)
        # Store new observation with score 0-10
    
    - retrieve_relevant(query, k=5)
        # Retrieve k most relevant memories using Semantic Search
        # Implementation:
        # 1. Generate embedding for query (using local 'all-MiniLM-L6-v2' model)
        # 2. Calculate cosine similarity with stored memory embeddings
        # 3. Return top k results
        # Note: Use lightweight local library (e.g. sentence-transformers), NO external Vector DBs.
    
    - generate_reflection()
        # Synthesize recent observations into insights
    
    - get_recent(n=10)
        # Get n most recent observations
```

**Action System:**
```python
class ActionRegistry:
    """Registry of all actions agents can perform."""
    
    # Action types to register:
    - create_purchase_order
    - approve_purchase_order
    - receive_goods
    - process_vendor_invoice
    - match_three_way
    - approve_invoice
    - schedule_payment
    - create_sales_order
    - ship_order
    - create_customer_invoice
    - apply_payment
    - create_journal_entry
    - reconcile_account
    - close_period
    
    # Each action includes:
    - action_id: str
    - required_inputs: Dict[str, Type]
    - validation_rules: List[ValidationRule]
    - execute(agent, inputs): ActionResult
```

**Decision-Making System:**
```python
class DecisionEngine:
    """Hybrid statistical + LLM decision engine."""
    
    # Decision flow:
    1. Statistical Layer:
       - Determine amounts (distributions)
       - Select entities (weighted random)
       - Calculate timing (payment terms + variance)
       - Check for discrepancy triggers
    
    2. LLM Layer (if needed):
       - Generate descriptions
       - Add processing notes
       - Handle edge cases
       - Make judgment calls
    
    3. Validation Layer:
       - Validate against schema
       - Check value ranges
       - Retry on validation failure (max 3 attempts)
    
    4. Deterministic Layer:
       - Apply business rules
       - Calculate GL postings
       - Update balances
```

#### 2. LLM Integration

**LLM Client:**
```python
class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    # Supported providers:
    - Anthropic (Claude Sonnet 4, Claude Haiku 4)
    - Azure OpenAI
    - OpenAI
    
    # Configuration from environment:
    - LLM_PROVIDER: str
    - LLM_MODEL: str
    - LLM_FALLBACK_MODEL: str
    - LLM_API_KEY: str
    - LLM_MAX_TOKENS: int
    - LLM_TEMPERATURE: float
    
    # Features to implement:
    - async complete(prompt, max_tokens, temperature)
    - retry logic with exponential backoff
    - automatic fallback to cheaper model on error
    - request/response logging
    - cost tracking
    - rate limiting compliance
```

**Prompt Management:**
```python
class PromptManager:
    """Manages prompt templates and context assembly."""
    
    # Prompt templates for each decision type:
    - process_invoice_prompt
    - approve_transaction_prompt
    - match_documents_prompt
    - handle_exception_prompt
    - generate_description_prompt
    - reconcile_account_prompt
    
    # Context assembly:
    - build_context(agent, transaction, history)
        # Assemble: agent info, company context, transaction details
        # Token budget: ~1,500 tokens max
    
    # Output parsing:
    - parse_llm_response(response, expected_schema)
        # Extract structured data from LLM text
```

**LLM Resilience:**
```python
class LLMQueue:
    """Queue-based resilience for LLM requests."""
    
    # Queue mechanism (using Redis Streams):
    - enqueue_request(prompt, metadata) -> request_id
        # Use XADD to add to stream 'llm_requests'
    - dequeue_request() -> Request
        # Use XREADGROUP to read from stream as consumer group 'llm_workers'
    - retry_failed_request(request_id)
        # Re-add to stream with incremented retry count
    
    # Resilience features:
    - Exponential backoff (2s, 4s, 8s, 16s, 32s)
    - Circuit breaker (open after 5 consecutive failures)
    - Request prioritization (critical > normal > low)
    - Max retries: 5 attempts
    
    # Cost controls:
    - Track spending per simulation
    - Warn at 80% of budget
    - Block at 100% of budget
```

**LLM Monitoring:**
```python
class LLMMonitor:
    """Monitor LLM usage, costs, and performance."""
    
    # Metrics to track:
    - total_requests: int
    - successful_requests: int
    - failed_requests: int
    - total_tokens: int (prompt + completion)
    - estimated_cost: Decimal
    - average_latency: float
    - p95_latency: float
    
    # Cost calculation:
    - Claude Sonnet 4: $3/$15 per 1M tokens (input/output)
    - Claude Haiku 4: $0.25/$1.25 per 1M tokens
    
    # Alerts:
    - Budget warning at 80%
    - Budget exceeded at 100%
    - High latency (> 10s)
    - High error rate (> 5%)
```

#### 3. Workflow Orchestration

**Workflow Orchestrator:**
```python
class WorkflowOrchestrator:
    """Routes work to agents and manages workflows."""
    
    # Work routing:
    - route_transaction(transaction, day_context)
        # Map transaction type → agent roles
        # Find available agent with capacity
        # Create workflow instance
        # Add to agent's work queue
    
    # Approval workflows:
    - check_approval_required(transaction)
        # Check amount vs. thresholds
        # Determine approval level needed
    
    - route_for_approval(transaction, approver_role)
        # Find approver
        # Add to approval queue
        # Track approval chain
    
    # Queue management:
    - work_queues: Dict[UUID, Queue]  # Per agent
    - approval_queues: Dict[str, Queue]  # Per role
    
    # Metrics:
    - total_routed: int
    - total_completed: int
    - average_queue_depth: float
    - average_processing_time: float
```

**Approval System:**
```python
class ApprovalSystem:
    """Manages approval chains and thresholds."""
    
    # Approval thresholds:
    THRESHOLDS = {
        "purchase_order": [
            (0, 5000, None),          # < $5K: no approval
            (5000, 25000, "purchasing_manager"),
            (25000, 100000, "controller"),
            (100000, None, "cfo")
        ],
        "vendor_invoice": [
            (0, 10000, None),
            (10000, 50000, "ap_manager"),
            (50000, 100000, "controller"),
            (100000, None, "cfo")
        ],
        "journal_entry": [
            (0, 50000, "senior_accountant"),
            (50000, None, "controller")
        ]
    }
    
    # Approval workflow:
    - get_required_approval(txn_type, amount) -> Optional[str]
    - create_approval_request(transaction, approver_role)
    - process_approval(request, decision, notes)
    - escalate_if_rejected(request)
```

**Transaction Orchestrator:**
```python
class TransactionOrchestrator:
    """Tracks transaction lifecycles and completeness."""
    
    # Transaction tracking:
    - transactions: Dict[UUID, TransactionState]
    
    # Required artifacts per transaction type:
    REQUIRED_ARTIFACTS = {
        "purchase_order": ["po_header", "po_lines", "approval"],
        "vendor_invoice": ["invoice_header", "invoice_lines", 
                          "three_way_match", "gl_entries"],
        "vendor_payment": ["payment_record", "payment_allocation", 
                          "gl_entries"],
    }
    
    # Lifecycle management:
    - register_transaction(transaction)
    - add_artifact(transaction_id, artifact_type, artifact)
    - check_completeness(transaction_id) -> bool
    - mark_complete(transaction_id)
    - get_transaction_chain(transaction_id) -> List[Transaction]
```

#### 4. Time Controller

**Time Controller:**
```python
class TimeController:
    """Manages simulation time progression."""
    
    # Time state:
    - current_date: date
    - current_time: datetime
    - business_calendar: BusinessCalendar
    - fiscal_calendar: FiscalCalendar
    
    # Time advancement:
    - advance_to_next_business_day()
        # Skip weekends and holidays
        # Trigger day-start events
    
    - advance_by_hours(hours: int)
        # For intra-day simulation
    
    # Business calendar operations:
    - is_business_day(date) -> bool
    - get_next_business_day(date) -> date
    - get_previous_business_day(date) -> date
    - count_business_days(start, end) -> int
    - add_business_days(date, days) -> date
    
    # Fiscal calendar operations:
    - get_fiscal_period(date) -> FiscalPeriod
    - is_period_end(date) -> bool
    - is_quarter_end(date) -> bool
    - is_year_end(date) -> bool
    - get_fiscal_year(date) -> int
    
    # Holiday management:
    - load_holiday_calendar(country="US")
    - add_custom_holiday(date, name)
```

**Business Calendar:**
```python
class BusinessCalendar:
    """Business days and holidays."""
    
    # Holiday types:
    - federal_holidays: Set[date]
    - company_holidays: Set[date]
    - half_days: Set[date]
    
    # US Federal holidays (2024-2026):
    - New Year's Day
    - Martin Luther King Jr. Day
    - Presidents' Day
    - Memorial Day
    - Independence Day
    - Labor Day
    - Thanksgiving
    - Christmas
    
    # Working hours:
    - standard_start: time(8, 0)
    - standard_end: time(17, 0)
    - lunch_break: (time(12, 0), time(13, 0))
```

**Fiscal Calendar:**
```python
class FiscalCalendar:
    """Fiscal periods and year-end."""
    
    # Configuration:
    - fiscal_year_start: int  # Month (1-12)
    - period_type: str  # "monthly" or "quarterly"
    
    # Period structure:
    - periods: List[FiscalPeriod]
        # Each period:
        - period_number: int
        - start_date: date
        - end_date: date
        - is_quarter_end: bool
        - is_year_end: bool
        - status: str  # "open", "closing", "closed"
    
    # Period operations:
    - open_period(period_number)
    - close_period(period_number)
    - get_current_period() -> FiscalPeriod
```

#### 5. External World Simulation

**External Entity Manager:**
```python
class ExternalWorldManager:
    """Manages external entities (customers, vendors, banks)."""
    
    # Entity pools:
    - customers: List[Customer]
    - vendors: List[Vendor]
    - banks: List[Bank]
    - carriers: List[Carrier]
    
    # Tiered entities:
    TIER_DISTRIBUTION = {
        "strategic": 0.10,      # 10% - high interaction
        "standard": 0.30,       # 30% - medium interaction
        "transactional": 0.60   # 60% - low interaction
    }
    
    # Entity behavior profiles:
    - assign_behavior_profile(entity)
        # Based on tier and entity type
        # Profiles: excellent, good, average, poor, problem
    
    # Entity interactions:
    - generate_vendor_invoice(vendor, po)
    - generate_customer_order(customer)
    - generate_bank_statement(month)
    - generate_carrier_tracking(shipment)
```

**Entity Behavior Profiles:**
```python
class BehaviorProfile:
    """Defines entity behavior patterns."""
    
    # Payment behavior (for customers):
    - payment_segment: str  # early, prompt, on_time, late, problem
    - payment_variance: float  # Days variance from terms
    - short_pay_rate: float  # % of invoices paid short
    - dispute_rate: float  # % of invoices disputed
    
    # Order behavior (for customers):
    - order_frequency: str  # daily, weekly, monthly
    - order_size_pattern: str  # consistent, variable
    - seasonality: Dict[int, float]  # Month multipliers
    
    # Invoice behavior (for vendors):
    - invoice_timing: str  # immediate, prompt, slow
    - invoice_accuracy: float  # % accurate invoices
    - response_time: int  # Days to respond to inquiries
```

#### 6. Statistical Models

**Amount Distribution Models:**
```python
class AmountDistribution:
    """Statistical models for realistic amounts."""
    
    # Transaction amount models:
    - purchase_orders:
        # Log-normal: mean=$5,000, std=$10,000, min=$100, max=$500K
        distribution = stats.lognorm(s=1.2, scale=5000)
    
    - vendor_invoices:
        # 95% match PO amount ± 5%
        # 5% have variances (quantity or price differences)
    
    - customer_orders:
        # Log-normal by customer size
        # Small: $500-$5K, Medium: $2K-$50K, Large: $10K-$500K
    
    # Rounding behavior:
    - round_to_nearest(amount, precision=100)
        # $4,847.32 → $4,800 (round to $100)
        # $124,389 → $125,000 (round to $1,000)
    
    # Discount application:
    - apply_discount(amount, discount_rate)
        # 2/10 Net 30: 2% if paid within 10 days
```

**Timing Distribution Models:**
```python
class TimingDistribution:
    """Statistical models for realistic timing."""
    
    # Payment timing (from STATISTICAL_MODELS_SPEC):
    - customer_payment_timing:
        # Mixture model with 5 segments
        # Early discount takers: Normal(μ=-8, σ=2) - 20%
        # Prompt payers: Normal(μ=-2, σ=3) - 30%
        # On-time payers: Normal(μ=2, σ=4) - 25%
        # Late payers: LogNorm(μ=10, σ=10) - 15%
        # Problem accounts: LogNorm(μ=30, σ=25) - 10%
    
    # Order frequency:
    - order_generation:
        # Poisson process with day-of-week effects
        # Monday: 0.85x, Tuesday: 1.0x, Wednesday: 1.0x
        # Thursday: 1.1x, Friday: 1.2x
    
    # Approval delays:
    - approval_processing_time:
        # Normal(μ=4 hours, σ=2 hours)
        # Escalated items: +1 day
```

**Selection Models:**
```python
class SelectionModel:
    """Entity selection using weighted distributions."""
    
    # Vendor selection:
    - vendor_selection(category):
        # Pareto distribution (80/20 rule)
        # Top 20% vendors get 80% of business
        # Weights by spend history
    
    # Customer selection:
    - customer_selection():
        # Revenue-weighted random
        # Large customers: more frequent orders
    
    # Product selection:
    - product_selection(customer):
        # Based on customer purchase history
        # 70% repeat purchases, 30% new products
```

#### 7. Event System

**Event Bus:**
```python
class EventBus:
    """Event-driven communication between components."""
    
    # Implementation Strategy: 
    # Primary: In-Memory Async Pub/Sub (using asyncio.Queue) for simplicity (<500 events/sec).
    # Optional: Redis Pub/Sub only if specific scaling thresholds are met (>5000/sec).
    
    # Event types:
    - TransactionCreated
    - TransactionCompleted
    - ApprovalRequired
    - ApprovalCompleted
    - DocumentGenerated
    - PeriodClosing
    - PeriodClosed
    - DiscrepancyDetected
    
    # Event handling:
    - publish(event: Event)
        # Publish event to all subscribers
    
    - subscribe(event_type: Type[Event], handler: Callable)
        # Register handler for event type
    
    - unsubscribe(event_type: Type[Event], handler: Callable)
        # Remove handler
    
    # Features:
    - Asynchronous event processing
    - Event persistence (for audit trail)
    - Event replay capability
    - Priority handling (urgent events first)
```

**Event Store:**
```python
class EventStore:
    """Persistent event storage."""
    
    # Event storage:
    - save_event(event)
        # Store event with metadata
    
    - get_events(simulation_id, event_type=None, start_date=None)
        # Query events with filters
    
    # Event sourcing:
    - replay_events(simulation_id, target_date)
        # Rebuild state by replaying events
    
    # Schema:
    - event_id: UUID
    - simulation_id: UUID
    - event_type: str
    - payload: JSONB
    - timestamp: datetime
    - agent_id: Optional[UUID]
```

---

## 🚫 OUT OF SCOPE (Boundary Declarations)

### Explicitly Excluded from Project 2

#### Transaction Table Population
- ❌ Actual writing to transaction tables (purchase_orders, invoices, etc.)
- ❌ Journal entry posting logic
- ❌ GL account balance updates
- ❌ Invoice matching logic implementation
- ❌ Payment allocation logic
- **Rationale**: Transaction generation is Project 3's responsibility. Project 2 creates the engine that will DRIVE transaction generation, but doesn't actually create the transactions.

#### Discrepancy Injection Logic
- ❌ Discrepancy detection algorithms
- ❌ Ground truth label generation
- ❌ Specific discrepancy type implementations (duplicate invoices, etc.)
- ❌ Discrepancy parameter tuning
- **Rationale**: Project 3 will implement discrepancy injection using the agent framework from Project 2.

#### Document Generation
- ❌ PDF document creation
- ❌ Document templates
- ❌ Document styling
- ❌ Template rendering engine
- **Rationale**: Project 4's responsibility.

#### Export Functionality
- ❌ CDM 3.0 export implementation
- ❌ CSV/JSON export logic
- ❌ Export job execution
- **Rationale**: Project 4's responsibility.

#### Admin UI
- ❌ Frontend components
- ❌ React/Vue application
- ❌ Dashboard visualizations
- ❌ User management UI
- **Rationale**: Project 4's responsibility.

#### Advanced Agent Features
- ❌ Complex multi-agent negotiations
- ❌ Agent learning/adaptation (beyond basic memory)
- ❌ Advanced planning (beyond daily/weekly)
- **Rationale**: MVP scope - can be added in future phases.

### Infrastructure & Operations (Explicitly Excluded)
- ❌ **Deployment Infrastructure**: No Docker, Kubernetes, Helm charts, or CI/CD pipelines.
- ❌ **Monitoring/Observability**: No Prometheus, Grafana, or APM integration (log to stdout only).
- ❌ **API Gateway**: No rate limiting, API versioning, or gateway infrastructure.
- ❌ **Backup/Disaster Recovery**: No backup systems or DR procedures.
- ❌ **Authentication/Authorization**: Use Project 1's existing auth system, do not modify.
- **Rationale**: Focus on the application logic layer only. Infrastructure is handled separately.

---

## 🏗️ TECHNICAL SPECIFICATIONS

### Technology Stack

#### Core Framework
- **Language**: Python 3.11+
- **Async Framework**: asyncio, aiohttp
- **Event Processing**: Redis Pub/Sub
- **LLM Integration**: 
  - anthropic (Claude API)
  - openai (OpenAI/Azure OpenAI API)
- **Scientific Computing**: numpy, scipy (statistical models)

#### Supporting Libraries
- **Data Generation**: Faker (enhanced with custom providers)
- **Date/Time**: python-dateutil, holidays (US calendar)
- **Statistical**: scipy.stats (distributions)
- **Caching**: Redis (LLM queue, agent state)
- **Configuration**: Pydantic (validation)
- **Logging**: structlog (structured logging)

### Agent Architecture Implementation

#### Agent Base Class
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    ERROR = "error"

@dataclass
class AgentConfig:
    """Agent configuration."""
    agent_id: UUID = field(default_factory=uuid4)
    role: str = ""
    name: str = ""
    employee_id: UUID = field(default_factory=uuid4)
    company_id: UUID = field(default_factory=uuid4)
    
    # Personality traits (0.0 to 1.0)
    traits: Dict[str, float] = field(default_factory=lambda: {
        "thoroughness": 0.7,
        "risk_tolerance": 0.5,
        "efficiency": 0.6,
        "compliance": 0.8,
    })
    
    # Work schedule
    work_hours_start: int = 8
    work_hours_end: int = 17
    
    # LLM settings
    temperature: float = 0.7
    max_tokens: int = 1000

class BaseAgent(ABC):
    """
    Base class for all agent types.
    
    Provides core agent capabilities:
    - Memory management
    - Decision-making
    - Action execution
    - Work queue processing
    """
    
    def __init__(
        self,
        config: AgentConfig,
        memory: AgentMemory,
        decision_engine: DecisionEngine,
        action_registry: ActionRegistry
    ):
        self.config = config
        self.memory = memory
        self.decision_engine = decision_engine
        self.action_registry = action_registry
        
        self.state = AgentState.IDLE
        self.work_queue: asyncio.Queue = asyncio.Queue()
        self.current_work_item: Optional[WorkItem] = None
        
        # Performance metrics
        self.metrics = {
            "items_processed": 0,
            "decisions_made": 0,
            "errors": 0,
            "average_processing_time": 0.0
        }
    
    @abstractmethod
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a work item. Implemented by subclasses."""
        pass
    
    async def run(self):
        """Main agent loop - process work items from queue."""
        while True:
            try:
                # Wait for work
                work_item = await self.work_queue.get()
                
                # Process work
                self.state = AgentState.THINKING
                self.current_work_item = work_item
                
                result = await self.process_work_item(work_item)
                
                # Record in memory
                await self.memory.add_observation(
                    content=f"Processed {work_item.type}: {work_item.description}",
                    importance=self._calculate_importance(work_item, result)
                )
                
                # Update metrics
                self.metrics["items_processed"] += 1
                
                self.state = AgentState.IDLE
                self.current_work_item = None
                
            except Exception as e:
                logger.error(
                    "agent_error",
                    agent_id=str(self.config.agent_id),
                    error=str(e),
                    work_item=work_item.dict() if work_item else None
                )
                self.state = AgentState.ERROR
                self.metrics["errors"] += 1
    
    async def make_decision(
        self,
        decision_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a decision using the decision engine."""
        self.state = AgentState.THINKING
        
        # Add agent context
        context["agent"] = {
            "role": self.config.role,
            "traits": self.config.traits,
            "recent_observations": await self.memory.get_recent(n=5)
        }
        
        # Make decision
        decision = await self.decision_engine.decide(
            decision_type=decision_type,
            context=context,
            agent_config=self.config
        )
        
        self.metrics["decisions_made"] += 1
        
        return decision
    
    def _calculate_importance(self, work_item: WorkItem, result: WorkResult) -> float:
        """Calculate importance score (0-10) for memory."""
        importance = 5.0  # Base importance
        
        # Increase importance for large amounts
        if hasattr(work_item, "amount") and work_item.amount > 10000:
            importance += 2.0
        
        # Increase importance for errors
        if not result.success:
            importance += 3.0
        
        # Increase importance for exceptions
        if result.has_exceptions:
            importance += 2.0
        
        return min(importance, 10.0)
```

#### Specialized Agent Example (AP Clerk)
```python
class APClerkAgent(BaseAgent):
    """Accounts Payable Clerk agent."""
    
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process AP work items."""
        
        if work_item.type == "vendor_invoice":
            return await self._process_vendor_invoice(work_item)
        elif work_item.type == "invoice_exception":
            return await self._handle_invoice_exception(work_item)
        else:
            raise ValueError(f"Unknown work item type: {work_item.type}")
    
    async def _process_vendor_invoice(self, work_item: WorkItem) -> WorkResult:
        """Process a vendor invoice."""
        
        # Extract invoice data
        invoice_data = work_item.data
        
        # Make decisions using LLM
        decision = await self.make_decision(
            decision_type="process_vendor_invoice",
            context={
                "invoice": invoice_data,
                "po": invoice_data.get("purchase_order"),
                "receipt": invoice_data.get("goods_receipt")
            }
        )
        
        # Execute actions
        # 1. Code invoice to GL accounts
        gl_coding = decision["gl_coding"]
        
        # 2. Perform 3-way match
        match_result = await self._perform_three_way_match(
            invoice_data,
            decision
        )
        
        # 3. Determine if approval needed
        approval_needed = match_result["variance_exceeds_tolerance"]
        
        # Record result
        result = WorkResult(
            success=True,
            actions_taken=[
                "gl_coding_assigned",
                "three_way_match_performed"
            ],
            approval_needed=approval_needed,
            data={
                "gl_coding": gl_coding,
                "match_result": match_result,
                "processing_notes": decision["processing_notes"]
            }
        )
        
        return result
    
    async def _perform_three_way_match(
        self,
        invoice_data: Dict,
        decision: Dict
    ) -> Dict:
        """Perform three-way matching logic."""
        # This is a simplified example - full implementation in Project 3
        
        po = invoice_data.get("purchase_order", {})
        receipt = invoice_data.get("goods_receipt", {})
        invoice = invoice_data
        
        # Compare quantities and prices
        quantity_match = receipt.get("quantity") == invoice.get("quantity")
        price_match = abs(
            po.get("unit_price", 0) - invoice.get("unit_price", 0)
        ) < 0.01
        
        match_status = "matched" if (quantity_match and price_match) else "variance"
        
        return {
            "match_status": match_status,
            "quantity_match": quantity_match,
            "price_match": price_match,
            "variance_exceeds_tolerance": not (quantity_match and price_match)
        }
```

### LLM Integration Implementation

#### LLM Client
```python
import anthropic
import openai
from typing import Optional, Dict, Any
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    """Unified LLM client supporting multiple providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider.lower()
        
        # Initialize provider clients
        if self.provider == "anthropic":
            self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        elif self.provider in ["openai", "azure_openai"]:
            self.client = openai.AsyncOpenAI(api_key=config.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Get completion from LLM.
        
        Automatically retries on failure with exponential backoff.
        Falls back to cheaper model on persistent errors.
        """
        start_time = time.time()
        
        try:
            if self.provider == "anthropic":
                response = await self._complete_anthropic(
                    prompt, system, max_tokens, temperature
                )
            elif self.provider in ["openai", "azure_openai"]:
                response = await self._complete_openai(
                    prompt, system, max_tokens, temperature
                )
            
            # Track metrics
            duration = time.time() - start_time
            self._record_success(response, duration)
            
            return response["content"]
            
        except Exception as e:
            # Track failure
            self.metrics["failed_requests"] += 1
            
            # Try fallback model if available
            if self.config.fallback_model and not self._using_fallback():
                logger.warning(
                    "llm_primary_failed_trying_fallback",
                    error=str(e),
                    primary_model=self.config.model,
                    fallback_model=self.config.fallback_model
                )
                return await self._complete_with_fallback(
                    prompt, system, max_tokens, temperature
                )
            
            raise
    
    async def _complete_anthropic(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Complete using Anthropic Claude API."""
        
        message = await self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return {
            "content": message.content[0].text,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
            "model": message.model
        }
    
    async def _complete_openai(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Complete using OpenAI API."""
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "model": response.model
        }
    
    def _record_success(self, response: Dict, duration: float):
        """Record successful request metrics."""
        self.metrics["total_requests"] += 1
        self.metrics["successful_requests"] += 1
        self.metrics["total_tokens"] += (
            response["input_tokens"] + response["output_tokens"]
        )
        
        # Calculate cost
        cost = self._calculate_cost(
            response["model"],
            response["input_tokens"],
            response["output_tokens"]
        )
        self.metrics["total_cost"] += cost
        
        logger.info(
            "llm_request_success",
            model=response["model"],
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            cost=cost,
            duration_seconds=duration
        )
    
    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost in USD."""
        
        # Pricing per 1M tokens (as of Jan 2025)
        PRICING = {
            "claude-sonnet-4-20250514": (3.0, 15.0),    # input, output
            "claude-haiku-4-20250514": (0.25, 1.25),
            "gpt-4-turbo": (10.0, 30.0),
            "gpt-3.5-turbo": (0.5, 1.5)
        }
        
        if model not in PRICING:
            logger.warning(f"Unknown model for pricing: {model}")
            return 0.0
        
        input_price, output_price = PRICING[model]
        
        cost = (
            (input_tokens / 1_000_000) * input_price +
            (output_tokens / 1_000_000) * output_price
        )
        
        return cost
```

#### Prompt Manager
```python
class PromptManager:
    """Manages prompt templates and context assembly."""
    
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.templates = self._load_templates()
    
    def build_prompt(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Build prompt from template and context.
        
        Returns: (system_prompt, user_prompt)
        """
        template = self.templates[template_name]
        
        # Build system prompt
        system_prompt = template["system"].format(**context)
        
        # Build user prompt
        user_prompt = template["user"].format(**context)
        
        # Validate token count
        estimated_tokens = self._estimate_tokens(system_prompt + user_prompt)
        if estimated_tokens > 2000:
            logger.warning(
                "prompt_too_long",
                template=template_name,
                estimated_tokens=estimated_tokens
            )
        
        return system_prompt, user_prompt
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
```

**Example Prompt Template:**
```yaml
# prompts/process_vendor_invoice.yaml

system: |
  You are an Accounts Payable clerk named {agent_name} working for {company_name}.
  
  Your traits:
  - Thoroughness: {thoroughness}/10
  - Risk tolerance: {risk_tolerance}/10
  - Efficiency: {efficiency}/10
  
  Process the vendor invoice according to company policies.

user: |
  VENDOR INVOICE DETAILS:
  Vendor: {vendor_name}
  Invoice Number: {invoice_number}
  Invoice Date: {invoice_date}
  Amount: ${amount:,.2f}
  
  PURCHASE ORDER:
  PO Number: {po_number}
  PO Amount: ${po_amount:,.2f}
  
  GOODS RECEIPT:
  Receipt Date: {receipt_date}
  Quantity Received: {quantity_received}
  
  TASKS:
  1. Verify 3-way match (PO, Receipt, Invoice)
  2. Assign GL account codes for each line item
  3. Add any processing notes
  4. Recommend approval/rejection
  
  Respond in JSON format:
  {{
    "match_status": "matched" | "variance",
    "gl_coding": [{{"line": 1, "account": "5000", "amount": 100.00}}],
    "processing_notes": "Your notes here",
    "recommendation": "approve" | "reject" | "escalate",
    "reasoning": "Brief explanation"
  }}
```

### Orchestration Implementation

#### Workflow Orchestrator
```python
class WorkflowOrchestrator:
    """Routes work to agents and manages workflows."""
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        approval_system: ApprovalSystem,
        config: WorkflowConfig
    ):
        self.agent_registry = agent_registry
        self.approval_system = approval_system
        self.config = config
        
        # Active workflows
        self.workflows: Dict[UUID, WorkflowInstance] = {}
        
        # Work queues per agent
        self.work_queues: Dict[UUID, asyncio.Queue] = {}
    
    async def route_transaction(
        self,
        transaction: Dict[str, Any],
        transaction_type: str
    ) -> UUID:
        """
        Route transaction to appropriate agent.
        
        Returns workflow_id for tracking.
        """
        # Create workflow instance
        workflow = WorkflowInstance(
            workflow_id=uuid4(),
            transaction_type=transaction_type,
            transaction_data=transaction,
            status="pending"
        )
        self.workflows[workflow.workflow_id] = workflow
        
        # Determine agent role needed
        agent_roles = self._get_required_roles(transaction_type)
        
        # Find available agent
        agent = await self._find_available_agent(agent_roles)
        if not agent:
            logger.warning(
                "no_agent_available",
                transaction_type=transaction_type,
                required_roles=agent_roles
            )
            # Queue for later
            await self._queue_for_later(workflow)
            return workflow.workflow_id
        
        # Create work item
        work_item = WorkItem(
            type=transaction_type,
            data=transaction,
            workflow_id=workflow.workflow_id
        )
        
        # Add to agent's queue
        await agent.work_queue.put(work_item)
        
        workflow.assigned_agent_id = agent.config.agent_id
        workflow.status = "assigned"
        
        logger.info(
            "transaction_routed",
            workflow_id=str(workflow.workflow_id),
            transaction_type=transaction_type,
            agent_id=str(agent.config.agent_id),
            agent_role=agent.config.role
        )
        
        return workflow.workflow_id
    
    def _get_required_roles(self, transaction_type: str) -> List[str]:
        """Get list of agent roles that can process this transaction."""
        ROLE_MAPPING = {
            "vendor_invoice": ["ap_clerk", "ap_manager"],
            "purchase_order": ["purchasing_agent", "purchasing_manager"],
            "sales_order": ["ar_clerk"],
            "customer_payment": ["ar_clerk"],
            "journal_entry": ["accountant", "senior_accountant"]
        }
        return ROLE_MAPPING.get(transaction_type, [])
    
    async def _find_available_agent(
        self,
        roles: List[str]
    ) -> Optional[BaseAgent]:
        """Find an available agent with one of the required roles."""
        
        # Get all agents with matching roles
        candidates = self.agent_registry.get_agents_by_roles(roles)
        
        # Filter to idle agents
        idle_agents = [
            agent for agent in candidates
            if agent.state == AgentState.IDLE
        ]
        
        if not idle_agents:
            return None
        
        # Select agent with smallest queue
        return min(idle_agents, key=lambda a: a.work_queue.qsize())
```

### Statistical Models Implementation

#### Payment Timing Model
```python
from scipy import stats
import numpy as np
from datetime import date, timedelta

class PaymentTimingModel:
    """Mixture model for realistic payment timing."""
    
    # Payment segments with distributions
    SEGMENTS = {
        "early_discount": {
            "weight": 0.20,
            "distribution": stats.norm(loc=-8, scale=2),
            "description": "Pay early to capture discount"
        },
        "prompt": {
            "weight": 0.30,
            "distribution": stats.norm(loc=-2, scale=3),
            "description": "Pay slightly before due"
        },
        "on_time": {
            "weight": 0.25,
            "distribution": stats.norm(loc=2, scale=4),
            "description": "Pay around due date"
        },
        "late": {
            "weight": 0.15,
            "distribution": stats.lognorm(s=1.0, loc=10, scale=5),
            "description": "Consistently late"
        },
        "problem": {
            "weight": 0.10,
            "distribution": stats.lognorm(s=1.2, loc=30, scale=15),
            "description": "Severely late"
        }
    }
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    def get_payment_date(
        self,
        invoice_date: date,
        payment_terms: str,
        customer_profile: str = "average"
    ) -> date:
        """
        Calculate payment date based on customer profile.
        
        Args:
            invoice_date: Invoice date
            payment_terms: e.g., "Net 30", "2/10 Net 30"
            customer_profile: excellent, good, average, poor, problem
        
        Returns:
            Expected payment date
        """
        # Calculate due date
        due_date = self._calculate_due_date(invoice_date, payment_terms)
        
        # Get customer segment
        segment = self._select_segment(customer_profile)
        
        # Sample days offset from due date
        days_offset = self._sample_offset(segment)
        
        # Apply day-of-week adjustment
        payment_date = due_date + timedelta(days=int(days_offset))
        payment_date = self._adjust_for_weekend(payment_date)
        
        return payment_date
    
    def _calculate_due_date(self, invoice_date: date, terms: str) -> date:
        """Calculate due date from terms."""
        if "Net 30" in terms:
            return invoice_date + timedelta(days=30)
        elif "Net 60" in terms:
            return invoice_date + timedelta(days=60)
        elif "Due on Receipt" in terms:
            return invoice_date + timedelta(days=0)
        else:
            return invoice_date + timedelta(days=30)
    
    def _select_segment(self, profile: str) -> str:
        """Select payment segment based on customer profile."""
        PROFILE_WEIGHTS = {
            "excellent": [0.60, 0.30, 0.08, 0.02, 0.00],
            "good": [0.20, 0.50, 0.20, 0.08, 0.02],
            "average": [0.10, 0.30, 0.35, 0.20, 0.05],
            "poor": [0.05, 0.10, 0.25, 0.40, 0.20],
            "problem": [0.00, 0.05, 0.15, 0.35, 0.45]
        }
        
        weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["average"])
        segments = list(self.SEGMENTS.keys())
        
        return self.rng.choice(segments, p=weights)
    
    def _sample_offset(self, segment: str) -> float:
        """Sample days offset from segment distribution."""
        dist = self.SEGMENTS[segment]["distribution"]
        return dist.rvs(random_state=self.rng)
    
    def _adjust_for_weekend(self, dt: date) -> date:
        """Move to next business day if weekend."""
        if dt.weekday() == 5:  # Saturday
            return dt + timedelta(days=2)
        elif dt.weekday() == 6:  # Sunday
            return dt + timedelta(days=1)
        return dt
```

---

## 🔧 ERROR HANDLING SPECIFICATION

### Error Handling Strategy

#### LLM Error Handling
```python
class LLMErrorHandler:
    """Handle LLM-specific errors."""
    
    async def handle_llm_error(
        self,
        error: Exception,
        request_context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle LLM errors with appropriate retry/fallback.
        
        Error types:
        - Rate limit errors → Wait and retry
        - Timeout errors → Retry with shorter max_tokens
        - Invalid response → Retry with clarified prompt
        - Authentication errors → Fatal, escalate
        """
        
        if isinstance(error, anthropic.RateLimitError):
            # Wait and retry
            await asyncio.sleep(error.retry_after or 60)
            return "retry"
        
        elif isinstance(error, anthropic.APITimeoutError):
            # Retry with reduced max_tokens
            if request_context.get("retry_count", 0) < 2:
                request_context["max_tokens"] = int(
                    request_context.get("max_tokens", 1000) * 0.7
                )
                return "retry"
            else:
                # Use fallback model
                return "fallback"
        
        elif isinstance(error, anthropic.APIError):
            # Generic API error - retry once
            if request_context.get("retry_count", 0) < 1:
                return "retry"
            else:
                return "fallback"
        
        else:
            # Unknown error - escalate
            logger.error(
                "llm_unknown_error",
                error_type=type(error).__name__,
                error_message=str(error),
                context=request_context
            )
            return "escalate"
```

#### Agent Error Handling
```python
class AgentErrorHandler:
    """Handle agent-specific errors."""
    
    async def handle_agent_error(
        self,
        agent: BaseAgent,
        work_item: WorkItem,
        error: Exception
    ) -> str:
        """
        Handle errors during agent work processing.
        
        Returns action: "retry", "skip", "escalate"
        """
        
        # Log error
        logger.error(
            "agent_error",
            agent_id=str(agent.config.agent_id),
            agent_role=agent.config.role,
            work_item_type=work_item.type,
            error=str(error),
            exc_info=True
        )
        
        # Update agent state
        agent.state = AgentState.ERROR
        agent.metrics["errors"] += 1
        
        # Determine action
        if isinstance(error, ValidationError):
            # Invalid input - skip work item
            return "skip"
        
        elif isinstance(error, LLMError):
            # LLM error - retry with fallback
            if work_item.retry_count < 3:
                work_item.retry_count += 1
                return "retry"
            else:
                return "skip"
        
        elif isinstance(error, DatabaseError):
            # Database error - escalate
            return "escalate"
        
        else:
            # Unknown error - retry once then skip
            if work_item.retry_count < 1:
                work_item.retry_count += 1
                return "retry"
            else:
                return "skip"
```

#### Workflow Error Handling
```python
class WorkflowErrorHandler:
    """Handle workflow-level errors."""
    
    async def handle_workflow_error(
        self,
        workflow: WorkflowInstance,
        error: Exception
    ):
        """Handle errors at workflow level."""
        
        workflow.status = "error"
        workflow.error_message = str(error)
        workflow.error_timestamp = datetime.utcnow()
        
        logger.error(
            "workflow_error",
            workflow_id=str(workflow.workflow_id),
            transaction_type=workflow.transaction_type,
            error=str(error),
            exc_info=True
        )
        
        # Attempt recovery
        if workflow.retry_count < 3:
            workflow.retry_count += 1
            workflow.status = "retrying"
            # Re-queue workflow
            await self.orchestrator.retry_workflow(workflow)
        else:
            # Mark as failed
            workflow.status = "failed"
            # Notify monitoring system
            await self.monitoring.alert_workflow_failed(workflow)
```

---

## 📊 LOGGING SPECIFICATION

### Logging Strategy

#### Agent Activity Logging
```python
# Agent decision logging
logger.info(
    "agent_decision",
    agent_id=str(agent.config.agent_id),
    agent_role=agent.config.role,
    decision_type="process_vendor_invoice",
    context_summary={
        "vendor": vendor_name,
        "amount": amount,
        "match_status": match_status
    },
    decision={
        "recommendation": recommendation,
        "reasoning": reasoning_summary
    },
    duration_seconds=decision_time
)

# Agent work item processing
logger.info(
    "agent_work_completed",
    agent_id=str(agent.config.agent_id),
    work_item_type=work_item.type,
    work_item_id=str(work_item.id),
    processing_time_seconds=processing_time,
    success=result.success
)
```

#### LLM Request Logging
```python
# LLM request logging
logger.info(
    "llm_request",
    request_id=request_id,
    model=model,
    prompt_tokens=len(prompt) // 4,  # Estimate
    max_tokens=max_tokens,
    temperature=temperature
)

# LLM response logging
logger.info(
    "llm_response",
    request_id=request_id,
    model=model,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    cost_usd=cost,
    duration_seconds=duration,
    success=True
)

# LLM error logging
logger.error(
    "llm_error",
    request_id=request_id,
    model=model,
    error_type=type(error).__name__,
    error_message=str(error),
    retry_count=retry_count,
    will_retry=will_retry
)
```

#### Workflow Logging
```python
# Workflow routing
logger.info(
    "workflow_routed",
    workflow_id=str(workflow.workflow_id),
    transaction_type=workflow.transaction_type,
    agent_id=str(agent.config.agent_id),
    agent_role=agent.config.role,
    queue_depth=agent.work_queue.qsize()
)

# Workflow completion
logger.info(
    "workflow_completed",
    workflow_id=str(workflow.workflow_id),
    transaction_type=workflow.transaction_type,
    total_duration_seconds=duration,
    agents_involved=[str(id) for id in workflow.agent_ids],
    approval_chain=workflow.approval_chain
)
```

#### Performance Logging
```python
# Daily summary
logger.info(
    "daily_summary",
    simulation_date=str(current_date),
    transactions_generated=transaction_count,
    agents_active=active_agent_count,
    average_agent_utilization=avg_utilization,
    llm_requests=llm_request_count,
    llm_cost_usd=total_llm_cost,
    duration_seconds=day_duration
)
```

---

## ⚙️ CONSTRAINT PARAMETERS

### Agent Operation Timeouts (Quantified)

**ALL agent operations MUST have explicit timeout values to prevent indefinite hangs:**

| Operation Type | Timeout | Behavior on Timeout |
|----------------|---------|---------------------|
| Agent work item (simple decision) | 10 seconds | Mark work item failed, skip, log warning |
| Agent work item (LLM decision) | 30 seconds | Fallback to statistical decision, retry once |
| Agent work item (complex workflow) | 60 seconds | Mark work item failed, escalate to manager agent |
| Agent work item (absolute max) | 120 seconds | Force terminate, log error, restart agent |
| Memory add observation | 1 second | Skip memory update, log warning, continue |
| Memory retrieve relevant | 2 seconds | Return empty results, log warning, continue |
| Memory generate reflection | 10 seconds | Skip reflection, log warning, continue |
| Event publish | 1 second | Queue for retry, log warning |
| Event handler execution | 5 seconds | Skip handler, log error, continue to next |
| LLM completion request | 30 seconds | Retry with backoff (see LLM section) |
| LLM batch completion | 120 seconds | Process partial batch, retry failures |

#### Agent Timeout Escalation Policy

```python
# Escalation policy for consecutive timeouts
AGENT_TIMEOUT_ESCALATION = {
    \"consecutive_timeouts_threshold\": 3,     # Escalate after 3 timeouts
    \"escalation_action\": \"notify_supervisor\",
    \"auto_restart_agent\": True,              # Restart agent after escalation
    \"cooldown_period_seconds\": 60            # Wait before restarting
}

class AgentTimeoutManager:
    \"\"\"Manages agent timeout escalation.\"\"\"
    
    def __init__(self, agent_id: UUID):
        self.agent_id = agent_id
        self.consecutive_timeouts = 0
        self.last_timeout = None
    
    async def handle_timeout(self, operation: str, duration: float):
        \"\"\"Handle agent timeout with escalation.\"\"\"
        self.consecutive_timeouts += 1
        self.last_timeout = datetime.now()
        
        logger.error(
            \"agent_operation_timeout\",
            agent_id=str(self.agent_id),
            operation=operation,
            duration_seconds=duration,
            consecutive_timeouts=self.consecutive_timeouts
        )
        
        # Escalate if threshold reached
        if self.consecutive_timeouts >= AGENT_TIMEOUT_ESCALATION[\"consecutive_timeouts_threshold\"]:
            await self._escalate()
    
    async def _escalate(self):
        \"\"\"Escalate consecutive timeout issue.\"\"\"
        logger.critical(
            \"agent_timeout_escalation\",
            agent_id=str(self.agent_id),
            consecutive_timeouts=self.consecutive_timeouts
        )
        
        # Restart agent after cooldown
        if AGENT_TIMEOUT_ESCALATION[\"auto_restart_agent\"]:
            await asyncio.sleep(AGENT_TIMEOUT_ESCALATION[\"cooldown_period_seconds\"])
            await self._restart_agent()
    
    def reset(self):
        \"\"\"Reset timeout counter on successful operation.\"\"\"
        self.consecutive_timeouts = 0
```

### LLM Usage Constraints

#### Budget Controls
```python
class LLMBudgetManager:
    """Manage LLM spending against budget."""
    
    def __init__(self, monthly_budget_usd: float = 100.0):
        self.monthly_budget = monthly_budget_usd
        self.current_spend = 0.0
        self.budget_warning_threshold = 0.80  # 80%
        
    def check_budget_before_request(self, estimated_cost: float) -> bool:
        """Check if request would exceed budget."""
        projected_spend = self.current_spend + estimated_cost
        
        if projected_spend > self.monthly_budget:
            logger.error(
                "llm_budget_exceeded",
                current_spend=self.current_spend,
                monthly_budget=self.monthly_budget,
                estimated_cost=estimated_cost
            )
            return False
        
        if projected_spend > (self.monthly_budget * self.budget_warning_threshold):
            logger.warning(
                "llm_budget_warning",
                current_spend=self.current_spend,
                monthly_budget=self.monthly_budget,
                percent_used=(projected_spend / self.monthly_budget * 100)
            )
        
        return True
    
    def record_spend(self, cost: float):
        """Record actual spending."""
        self.current_spend += cost
```

#### Rate Limiting
```python
class LLMRateLimiter:
    """Rate limit LLM requests to avoid API limits."""
    
    # Provider rate limits (requests per minute)
    RATE_LIMITS = {
        "anthropic": {
            "claude-sonnet-4": 50,
            "claude-haiku-4": 100
        },
        "openai": {
            "gpt-4-turbo": 500,
            "gpt-3.5-turbo": 3500
        }
    }
    
    def __init__(self, provider: str, model: str):
        self.limit = self.RATE_LIMITS[provider][model]
        self.requests_this_minute = 0
        self.minute_start = time.time()
    
    async def acquire(self):
        """Acquire permission to make request (rate limited)."""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.minute_start = current_time
        
        # Wait if at limit
        if self.requests_this_minute >= self.limit:
            wait_time = 60 - (current_time - self.minute_start)
            logger.info(
                "rate_limit_waiting",
                wait_seconds=wait_time,
                limit=self.limit
            )
            await asyncio.sleep(wait_time)
            self.requests_this_minute = 0
            self.minute_start = time.time()
        
        self.requests_this_minute += 1
```

### Memory Constraints

#### Agent Memory Limits
```python
# Memory limits per agent
MAX_OBSERVATIONS = 1000        # Keep last 1000 observations
MAX_REFLECTIONS = 100          # Keep last 100 reflections
MEMORY_CLEANUP_INTERVAL = 3600 # Clean up every hour

# Memory size estimates
BYTES_PER_OBSERVATION = 500    # ~500 bytes per observation
BYTES_PER_REFLECTION = 2000    # ~2KB per reflection

# Total memory per agent
MAX_MEMORY_PER_AGENT = (
    MAX_OBSERVATIONS * BYTES_PER_OBSERVATION +
    MAX_REFLECTIONS * BYTES_PER_REFLECTION
)  # ≈ 700KB per agent
```

#### Redis Memory Management (Enhanced)

**Explicit memory management prevents resource exhaustion:**

```python
REDIS_MEMORY_CONFIG = {
    "max_memory": "2gb",                     # Maximum Redis memory
    "eviction_policy": "allkeys-lru",        # Evict least recently used keys
    "max_memory_samples": 5,
    
    # Per-key TTL policies
    "ttl_policies": {
        "agent_state": 86400,                # 24 hours
        "llm_queue_item": 3600,              # 1 hour
        "event": 604800,                     # 7 days
        "memory_observation": 2592000        # 30 days
    },
    
    # Monitoring thresholds
    "memory_warning_threshold_mb": 1536,     # Warn at 1.5GB (75%)
    "memory_critical_threshold_mb": 1843,    # Critical at 1.8GB (90%)
    
    # Cleanup automation
    "auto_cleanup_enabled": True,
    "cleanup_interval_seconds": 3600,        # Every hour
    "cleanup_expired_keys": True,
    "cleanup_lru_when_full": True
}
```

#### Agent State Storage Limits
```python
AGENT_STATE_LIMITS = {
    "max_observations_per_agent": 1000,      # Limit observation history
    "max_reflections_per_agent": 100,        # Limit reflection history
    "max_work_queue_size_per_agent": 100,    # Prevent queue overflow
    "max_total_agents": 50,                  # Hard limit on agent count
    
    # Memory calculations
    "bytes_per_observation": 500,
    "bytes_per_reflection": 2000,
    "estimated_memory_per_agent_kb": 700,
    
    # Total memory estimate: 50 agents × 700KB = 35MB (well within 2GB Redis limit)
}
```

#### LLM Queue Limits
```python
LLM_QUEUE_LIMITS = {
    "max_queue_size": 1000,                  # Max pending LLM requests
    "queue_timeout_seconds": 300,            # Request expires after 5 minutes
    "max_batch_size": 10,                    # Batch up to 10 requests
    "queue_warning_threshold": 800,          # Warn at 80% capacity
    
    # Queue overflow behavior
    "on_queue_full": "reject_new_requests",  # Reject when full
    "priority_override": True                # Allow critical requests to bypass limit
}
```

### Concurrency Constraints

#### Agent Concurrency
```python
# Maximum concurrent agents
MAX_CONCURRENT_AGENTS = 20

# Work queue sizes
MAX_WORK_QUEUE_SIZE = 100  # Per agent

# Workflow concurrency
MAX_CONCURRENT_WORKFLOWS = 100
```

---

## 🔍 VALIDATION SEQUENCE FRAMEWORK

### Validation Pipeline

#### 1. Agent Output Validation
```
┌─────────────────────────────────────────────────────────────┐
│            AGENT OUTPUT VALIDATION SEQUENCE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Schema Validation                                  │
│  ├─ LLM output matches expected JSON schema                │
│  ├─ All required fields present                            │
│  ├─ Field types correct                                     │
│  └─ [FAIL] → Retry LLM request (max 3 attempts)           │
│                                                              │
│  Step 2: Value Range Validation                             │
│  ├─ Amounts within valid range                             │
│  ├─ Dates within simulation period                         │
│  ├─ IDs reference existing entities                        │
│  └─ [FAIL] → Retry with adjusted constraints               │
│                                                              │
│  Step 3: Business Rule Validation                           │
│  ├─ Approval decisions respect thresholds                  │
│  ├─ GL codings use valid accounts                          │
│  ├─ Recommendations are sensible                           │
│  └─ [FAIL] → Retry with clarified prompt                   │
│                                                              │
│  [PASS] → Accept agent output                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 2. Workflow State Validation
```
┌─────────────────────────────────────────────────────────────┐
│         WORKFLOW STATE VALIDATION SEQUENCE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: State Transition Validation                        │
│  ├─ Valid state transition (pending→assigned→completed)    │
│  ├─ Required fields for state present                      │
│  └─ [FAIL] → Block invalid transition                      │
│                                                              │
│  Step 2: Approval Chain Validation                          │
│  ├─ Approvals in correct order                             │
│  ├─ Approver has required authority                        │
│  ├─ No approval chain bypasses                             │
│  └─ [FAIL] → Escalate to controller                        │
│                                                              │
│  Step 3: Completeness Validation                            │
│  ├─ All required artifacts present                         │
│  ├─ Artifact relationships valid                           │
│  └─ [FAIL] → Mark incomplete, retry missing steps          │
│                                                              │
│  [PASS] → Workflow complete                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Integration Validation
```
┌─────────────────────────────────────────────────────────────┐
│        INTEGRATION VALIDATION SEQUENCE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Database Consistency                               │
│  ├─ Agent records link to employee records                 │
│  ├─ Work items reference valid transactions                │
│  ├─ Events have valid simulation_id                        │
│  └─ [FAIL] → Log error, skip transaction                   │
│                                                              │
│  Step 2: Event Flow Validation                              │
│  ├─ Events in logical sequence                             │
│  ├─ No orphaned events                                     │
│  └─ [FAIL] → Log warning, continue                         │
│                                                              │
│  Step 3: Performance Validation                             │
│  ├─ Response times within SLA                              │
│  ├─ Memory usage within limits                             │
│  ├─ LLM costs within budget                                │
│  └─ [FAIL] → Alert and throttle                            │
│                                                              │
│  [PASS] → System healthy                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏛️ ARCHITECTURE CONTEXT

### Integration with Project 1

```
┌────────────────────────────────────────────────────────────────────┐
│                    INTEGRATION WITH PROJECT 1                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PROJECT 2 (Agent Engine) USES FROM PROJECT 1:                     │
│  ├─ Database schema (all tables)                                  │
│  ├─ Master data (customers, vendors, products, employees)         │
│  ├─ REST API for querying data                                    │
│  ├─ Configuration system                                           │
│  └─ Validation framework                                           │
│                                                                     │
│  PROJECT 2 ADDS:                                                   │
│  ├─ Agent system (creates autonomous employees)                   │
│  ├─ LLM integration (makes decisions)                             │
│  ├─ Workflow orchestration (routes work)                          │
│  ├─ Time controller (advances simulation)                         │
│  ├─ External world (simulates customer/vendor behavior)           │
│  ├─ Statistical models (generates realistic patterns)             │
│  └─ Event bus (coordinates components)                            │
│                                                                     │
│  PROJECT 2 PROVIDES TO PROJECT 3:                                  │
│  ├─ Agent framework for decision-making                           │
│  ├─ Workflow system for transaction routing                       │
│  ├─ LLM client for text generation                                │
│  ├─ Statistical models for realistic distributions                │
│  └─ Event system for transaction triggers                         │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
synthetic_erp_platform/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py           # BaseAgent class
│   │   ├── agent_config.py         # AgentConfig, AgentState
│   │   ├── agent_memory.py         # Memory system
│   │   ├── agent_registry.py       # Agent management
│   │   ├── action_registry.py      # Action definitions
│   │   ├── decision_engine.py      # Hybrid decision logic
│   │   │
│   │   ├── specialized/
│   │   │   ├── __init__.py
│   │   │   ├── ap_clerk_agent.py
│   │   │   ├── ap_manager_agent.py
│   │   │   ├── ar_clerk_agent.py
│   │   │   ├── purchasing_agent.py
│   │   │   ├── warehouse_agent.py
│   │   │   ├── accountant_agent.py
│   │   │   └── controller_agent.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_client.py           # Unified LLM client
│   │   ├── llm_config.py           # LLM configuration
│   │   ├── llm_queue.py            # Queue & resilience
│   │   ├── llm_monitor.py          # Usage monitoring
│   │   ├── prompt_manager.py       # Prompt templates
│   │   └── response_parser.py      # Parse LLM outputs
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── workflow_orchestrator.py
│   │   ├── transaction_orchestrator.py
│   │   ├── approval_system.py
│   │   ├── time_controller.py
│   │   ├── business_calendar.py
│   │   └── fiscal_calendar.py
│   │
│   ├── external_world/
│   │   ├── __init__.py
│   │   ├── external_entity_manager.py
│   │   ├── behavior_profiles.py
│   │   ├── customer_simulator.py
│   │   ├── vendor_simulator.py
│   │   └── bank_simulator.py
│   │
│   ├── statistical/
│   │   ├── __init__.py
│   │   ├── amount_distributions.py
│   │   ├── timing_models.py
│   │   ├── payment_timing_model.py
│   │   ├── order_frequency_model.py
│   │   └── selection_models.py
│   │
│   ├── events/
│   │   ├── __init__.py
│   │   ├── event_bus.py            # Event publishing/subscription
│   │   ├── event_store.py          # Event persistence
│   │   ├── event_types.py          # Event definitions
│   │   └── event_handlers.py       # Event handlers
│   │
│   └── simulation/
│       ├── __init__.py
│       ├── simulation_engine.py    # Main simulation loop
│       ├── day_context.py          # Daily context
│       └── simulation_metrics.py   # Performance tracking
│
├── prompts/
│   ├── process_vendor_invoice.yaml
│   ├── approve_transaction.yaml
│   ├── handle_exception.yaml
│   ├── generate_description.yaml
│   └── reconcile_account.yaml
│
├── config/
│   ├── agents/
│   │   └── agent_roles.yaml        # Agent role definitions
│   ├── workflows/
│   │   └── approval_thresholds.yaml
│   ├── statistical/
│   │   ├── payment_timing.yaml
│   │   └── amount_distributions.yaml
│   └── llm/
│       └── llm_config.yaml
│
└── tests/
    ├── test_agents/
    │   ├── test_base_agent.py
    │   ├── test_agent_memory.py
    │   └── test_specialized_agents.py
    ├── test_llm/
    │   ├── test_llm_client.py
    │   ├── test_prompt_manager.py
    │   └── test_llm_queue.py
    ├── test_orchestration/
    │   ├── test_workflow_orchestrator.py
    │   └── test_time_controller.py
    └── test_statistical/
        └── test_payment_timing_model.py
```

---

## 📦 DELIVERABLES CHECKLIST

### Project 2 Code Deliverables
- [ ] Complete agent system with 12 specialized agent types
- [ ] Agent memory system with observations and reflections
- [ ] LLM client supporting Anthropic and OpenAI
- [ ] Prompt management system with templates
- [ ] LLM queue with resilience and retry logic
- [ ] Workflow orchestrator with work routing
- [ ] Approval system with threshold management
- [ ] Transaction orchestrator with lifecycle tracking
- [ ] Time controller with business/fiscal calendars
- [ ] External world manager with entity simulation
- [ ] Statistical models (payment timing, amounts, selection)
- [ ] Event bus with pub/sub
- [ ] Event store with persistence
- [ ] Decision engine (hybrid statistical + LLM)

### Documentation Deliverables
- [ ] Agent system architecture documentation
- [ ] LLM integration guide
- [ ] Prompt template documentation
- [ ] Workflow configuration guide
- [ ] Statistical models documentation
- [ ] API documentation for new endpoints

### Testing Deliverables
- [ ] Unit tests for all agent types (≥80% coverage)
- [ ] LLM integration tests (with mocking)
- [ ] Workflow orchestration tests
- [ ] Statistical model validation tests
- [ ] Integration tests with Project 1
- [ ] Performance tests (throughput, latency)

---

## 🎓 DEVELOPMENT GUIDELINES

### Agent Development
- Each specialized agent extends BaseAgent
- Implement process_work_item() for agent-specific logic
- Use decision_engine for LLM decisions
- Record observations in agent memory
- Handle errors gracefully with retry logic

### LLM Integration Best Practices
- Always validate LLM responses before use
- Keep prompts under 2,000 tokens
- Use structured output formats (JSON)
- Implement retry logic with exponential backoff
- Track costs and stay within budget
- Use fallback models for resilience

### Statistical Models
- Use scipy.stats for distributions
- Seed random generators for reproducibility
- Profile performance with large sample sizes
- Validate outputs match expected distributions

---

## ✅ ACCEPTANCE CRITERIA

### Functional Requirements
1. ✅ Agents make realistic decisions using LLM + statistical models
2. ✅ Workflows route to appropriate agents based on role
3. ✅ Approvals enforced at correct thresholds
4. ✅ Time advances properly with business calendar
5. ✅ External entities generate realistic responses

### Performance Requirements
1. ✅ Agent decisions complete within 5 seconds (p95)
2. ✅ Workflow routing completes within 500ms
3. ✅ 1 month simulation completes in 4-8 hours
4. ✅ LLM costs stay under $100/month budget
5. ✅ System handles 20 concurrent agents

### Integration Requirements
1. ✅ Successfully queries Project 1 database
2. ✅ Successfully uses Project 1 APIs
3. ✅ Event bus coordinates all components
4. ✅ All validation rules from Project 1 pass
5. ✅ No data corruption in master data

---

## 📚 REFERENCE SPECIFICATIONS

> **Note:** The following specification documents are external upstream design documents that informed the implementation of this project. They are **not included** in this repository — they reside in the project's design documentation system and are referenced here for traceability purposes only.

This project implements specifications from:

1. **AGENT_ARCHITECTURE.md** - Agent structure, roles, memory
2. **AGENT_DECISION_SPEC.md** - Decision logic, LLM prompts
3. **ORCHESTRATION_DESIGN.md** - Workflow orchestration
4. **STATISTICAL_MODELS_SPEC.md** - Payment timing, amounts
5. **EXTERNAL_ENTITIES_SPEC.md** - Customer/vendor simulation
6. **FINANCIAL_GROUNDING_SPEC.md** - Financial targets

---

**END OF PROJECT 2 SPECIFICATION**

---
---

# PROJECT 3: TRANSACTION WORKFLOWS & DISCREPANCIES
## Synthetic ERP Data Generation Platform - Phase 3

---

## 🏗️ PROJECT 3 ARCHITECTURE

Project 3 builds upon Project 2's agent and orchestration engine to generate realistic financial transactions with configurable discrepancies. It comprises six major subsystems, all integrated via constructor injection (ADR-003) and communicating through the existing EventBus (ADR-001).

### 1. Procure-to-Pay (P2P) Engine

A five-class pipeline generating complete P2P cycles:

**Purchase Order → Goods Receipt → Vendor Invoice (with 3-way match) → Vendor Payment**

| Generator Class | Responsibility |
|----------------|---------------|
| `PurchaseOrderGenerator` | Weighted vendor selection, EOQ-based quantity calculation, pricing, approval routing per thresholds (PO: $5K/$25K/$100K), sequential PO numbering (PO-YYYY-NNNN) |
| `GoodsReceiptGenerator` | Receipt against open POs, lead time calculation, quantity variance handling, inventory balance update, GL posting (DR Inventory, CR AP Accrual) |
| `VendorInvoiceProcessor` | Invoice creation, PO linkage, three-way match invocation, AP clerk routing via WorkflowOrchestrator, GL posting (DR Expense/Asset, CR AP) |
| `ThreeWayMatcher` | PO/Receipt/Invoice matching with ±5% price tolerance and ±2% quantity tolerance, match status determination, variance calculation |
| `VendorPaymentGenerator` | Payment grouping by vendor/terms, discount calculation, check/ACH/wire selection, GL posting (DR AP, CR Cash; DR Discount if applicable) |

### 2. Order-to-Cash (O2C) Engine

A four-class pipeline generating complete O2C cycles:

**Sales Order → Shipment → Customer Invoice → Customer Payment**

| Generator Class | Responsibility |
|----------------|---------------|
| `SalesOrderGenerator` | Revenue-weighted customer selection, credit limit check (credit_limit vs current_ar_balance), product selection, pricing with discounts, sequential SO numbering (SO-YYYY-NNNN) |
| `ShipmentGenerator` | Ships against open SOs, carrier selection, tracking number generation, inventory reduction, GL posting (DR COGS, CR Inventory) |
| `CustomerInvoiceGenerator` | Invoice from shipped order, payment terms from customer master, due date calculation, GL posting (DR AR, CR Revenue), sequential numbering (INV-YYYY-NNNN) |
| `CustomerPaymentProcessor` | FIFO payment allocation (oldest invoice first), short pay handling, overpay → Unapplied Cash (no negative invoice balance), GL posting (DR Cash, CR AR) |

### 3. General Ledger Integration

Four classes providing the foundational posting layer consumed by all transaction generators:

| Class | Responsibility |
|-------|---------------|
| `GLPostingEngine` | Journal entry creation, balance validation (DR = CR within $0.01), account validation against Chart of Accounts (is_posting=TRUE), period validation (OPEN only), sequential JE numbering, continuous trial balance check |
| `AccountBalanceManager` | Real-time balance maintenance with `SELECT FOR UPDATE` locking for concurrent access, normal balance direction enforcement (Asset/Expense: DR increases; Liability/Equity/Revenue: CR increases), period-based tracking |
| `AccrualGenerator` | AP accruals for GRNI (Goods Received Not Invoiced), AR accruals for shipped-not-invoiced, straight-line daily method (Total / Days in Period), reversing entries for next period |
| `PeriodCloseManager` | 10-step close process: validate all posted → generate accruals → generate deferrals → post depreciation → recurring JEs → account reconciliations → trial balance → validate financial statements → close period → open next period |

### 4. Discrepancy Injection System

A configurable injector implementing 35+ discrepancy types with ground truth label generation:

| Category | Count | Examples |
|----------|-------|---------|
| **P2P Discrepancies** | 15 | Duplicate Invoice (P2P-001), Price Mismatch (P2P-002), Quantity Variance (P2P-003), Missing PO (P2P-004), PO Not Approved (P2P-005), Invoice Before Receipt (P2P-006), Round-Dollar Invoice (P2P-007), Weekend Processing (P2P-008), Duplicate Payment (P2P-009), Payment Before Invoice (P2P-010), Unapproved Vendor (P2P-011), Split PO (P2P-012), Fictitious Vendor (P2P-013), Vendor Concentration (P2P-014), Ghost Expense (P2P-015) |
| **O2C Discrepancies** | 10 | Duplicate Customer Invoice (O2C-001), Invoice Without Shipment (O2C-002), Credit Limit Exceeded (O2C-003), Short Payment (O2C-004), Overpayment Not Returned (O2C-005), Revenue Recognition Timing (O2C-006), Fictitious Customer (O2C-007), Round-Tripping (O2C-008), Channel Stuffing (O2C-009), Side Agreements (O2C-010) |
| **GL Discrepancies** | 5 | Unbalanced Journal (GL-001), Journal Without Approval (GL-002), Suspicious Adjusting Entry (GL-003), Unusual Account Combo (GL-004), Manual Override (GL-005) |
| **Control Discrepancies** | 5 | SoD Violation (CTL-001), Self-Approval (CTL-002), Approval Limit Exceeded (CTL-003), Backdated Transaction (CTL-004), Holiday Transaction (CTL-005) |

**Injection Configuration:**
- Default injection rate: 2% of transactions
- Difficulty distribution: Easy (70%) / Medium (30%) / Hard (0% — MVP)
- All parameters bounded and configurable via YAML
- Auto-adjust to bounds when enabled
- 100% ground truth coverage with 16-field schema

### 5. Intelligent Rework Loop

An autonomous error correction system for validation failures:

| Class | Responsibility |
|-------|---------------|
| `ReworkLoopEngine` | Orchestrates the classify → select fix → apply → re-validate → escalate cycle; max 3 attempts per transaction; escalates if >5% failure rate |
| `FailureClassifier` | Classifies validation failures as planned discrepancy (within/outside parameters) or unplanned error |
| `FixScenarioCatalog` | 20+ fix scenarios (e.g., ADJUST_AMOUNT_TO_RANGE, FIX_DATE_SEQUENCE, CORRECT_ENTITY_REFERENCE, REGENERATE_GL_ENTRY, RECALCULATE_BALANCE) sorted by success rate |
| `FixScenarioExecutor` | Executes fix scenario steps against a transaction; handles per-scenario timeout (10s); logs fix attempt details |

### 6. Period Close Processing

Period-end management integrated with the TimeController's `PeriodClosing`/`PeriodClosed` events:

- **Accruals**: GRNI (Goods Received Not Invoiced), shipped-not-invoiced, straight-line daily method
- **Deferrals**: Prepaid expenses and unearned revenue adjustments
- **Depreciation**: Fixed asset depreciation entries
- **Recurring Journal Entries**: Template-based monthly entries
- **Account Reconciliations**: AR, AP, and Inventory sub-ledger to GL control account reconciliation
- **Trial Balance Validation**: Cumulative zero validation before period close
- **Period State Transitions**: OPEN → CLOSING → CLOSED with next period auto-open

---

## 🏗️ PROJECT 3 MODULE STRUCTURE

```
app/transactions/                          # Transaction generation engines
  ├── __init__.py                          # Exports TransactionGenerator, GenerationContext, TransactionResult
  ├── base_generator.py                    # Abstract base class with shared logic
  ├── exceptions.py                        # Custom exception hierarchy (TransactionError, BalanceError, etc.)
  ├── constants.py                         # Shared constants, limits, circuit breaker config
  ├── p2p/                                 # Procure-to-Pay generators (5 classes)
  │   ├── __init__.py
  │   ├── purchase_order_generator.py
  │   ├── goods_receipt_generator.py
  │   ├── vendor_invoice_processor.py
  │   ├── three_way_matcher.py
  │   └── vendor_payment_generator.py
  ├── o2c/                                 # Order-to-Cash generators (4 classes)
  │   ├── __init__.py
  │   ├── sales_order_generator.py
  │   ├── shipment_generator.py
  │   ├── customer_invoice_generator.py
  │   └── customer_payment_processor.py
  └── gl/                                  # General Ledger integration (4 classes)
      ├── __init__.py
      ├── gl_posting_engine.py
      ├── account_balance_manager.py
      ├── accrual_generator.py
      └── period_close_manager.py

app/discrepancies/                         # Discrepancy injection system
  ├── __init__.py                          # Exports injector, ground truth generator, catalog
  ├── discrepancy_injector.py              # Rate-based injection trigger, type selection
  ├── ground_truth_generator.py            # Ground truth record creation (16-field schema)
  ├── discrepancy_catalog.py               # Registry mapping 35+ type codes to implementations
  ├── base_discrepancy.py                  # Abstract BaseDiscrepancy class
  ├── p2p/                                 # 15 P2P discrepancy type implementations
  │   ├── __init__.py
  │   ├── duplicate_invoice.py             # P2P-001
  │   ├── price_mismatch.py               # P2P-002
  │   ├── quantity_variance.py             # P2P-003
  │   ├── missing_po.py                   # P2P-004
  │   ├── po_not_approved.py              # P2P-005
  │   ├── invoice_before_receipt.py       # P2P-006
  │   ├── round_dollar_invoice.py         # P2P-007
  │   ├── weekend_processing.py           # P2P-008
  │   ├── duplicate_payment.py            # P2P-009
  │   ├── payment_before_invoice.py       # P2P-010
  │   ├── unapproved_vendor.py            # P2P-011
  │   ├── split_po.py                     # P2P-012
  │   ├── fictitious_vendor.py            # P2P-013
  │   ├── vendor_concentration.py         # P2P-014
  │   └── ghost_expense.py               # P2P-015
  ├── o2c/                                 # 10 O2C discrepancy type implementations
  │   ├── __init__.py
  │   ├── duplicate_customer_invoice.py   # O2C-001
  │   ├── invoice_without_shipment.py     # O2C-002
  │   ├── credit_limit_exceeded.py        # O2C-003
  │   ├── short_payment.py               # O2C-004
  │   ├── overpayment_not_returned.py     # O2C-005
  │   ├── revenue_recognition_timing.py   # O2C-006
  │   ├── fictitious_customer.py          # O2C-007
  │   ├── round_tripping.py              # O2C-008
  │   ├── channel_stuffing.py            # O2C-009
  │   └── side_agreements.py             # O2C-010
  ├── gl/                                  # 5 GL discrepancy type implementations
  │   ├── __init__.py
  │   ├── unbalanced_journal.py           # GL-001
  │   ├── journal_no_approval.py          # GL-002
  │   ├── suspicious_adjusting.py         # GL-003
  │   ├── unusual_account_combo.py        # GL-004
  │   └── manual_override.py             # GL-005
  └── control/                             # 5 Control discrepancy type implementations
      ├── __init__.py
      ├── sod_violation.py                # CTL-001
      ├── self_approval.py                # CTL-002
      ├── approval_limit_exceeded.py      # CTL-003
      ├── backdated_transaction.py        # CTL-004
      └── holiday_transaction.py          # CTL-005

app/rework/                                # Intelligent rework loop
  ├── __init__.py                          # Exports rework engine, classifier, catalog, executor
  ├── rework_loop_engine.py                # Orchestrates classify → fix → re-validate → escalate
  ├── failure_classifier.py                # Classifies failures as planned discrepancy or unplanned error
  ├── fix_scenario_catalog.py              # 20+ fix scenarios with success rates and step definitions
  └── fix_scenario_executor.py             # Executes fix scenario steps against transactions
```

---

## ⚙️ PROJECT 3 CONFIGURATION

### New Configuration Files

| File Path | Purpose |
|-----------|---------|
| `config/discrepancies/p2p_discrepancies.yaml` | 15 P2P discrepancy type definitions with codes, categories, difficulties, base rates, and parameter bounds |
| `config/discrepancies/o2c_discrepancies.yaml` | 10 O2C discrepancy type definitions with codes, categories, difficulties, base rates, and parameter bounds |
| `config/discrepancies/gl_discrepancies.yaml` | 5 GL + 5 Control discrepancy type definitions |
| `config/discrepancies/discrepancy_rates.yaml` | Global injection rate (default 2%), difficulty distribution (easy: 0.70, medium: 0.30, hard: 0.00), auto-adjust settings |
| `config/transactions/posting_rules.yaml` | GL posting account mappings per transaction type (e.g., goods_receipt: DR Inventory, CR AP Accrual) |
| `config/transactions/period_close.yaml` | Period close configuration: accrual rules, depreciation method, recurring JE templates, reconciliation targets |

### New Environment Variables (`.env.example`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSACTION_BATCH_SIZE` | `100` | Number of transactions per generation batch |
| `GL_POSTING_BATCH_SIZE` | `500` | Number of GL entries per posting batch |
| `DISCREPANCY_INJECTION_ENABLED` | `true` | Enable/disable discrepancy injection |
| `DISCREPANCY_DEFAULT_RATE` | `0.02` | Default discrepancy injection rate (2%) |
| `DISCREPANCY_DIFFICULTY_DISTRIBUTION` | `0.70,0.30,0.00` | Difficulty distribution for injected discrepancies (easy,medium,hard) |
| `REWORK_LOOP_ENABLED` | `true` | Enable/disable the intelligent rework loop |
| `REWORK_LOOP_MAX_ATTEMPTS` | `3` | Maximum fix attempts per transaction |
| `REWORK_LOOP_TIMEOUT_SECONDS` | `30` | Timeout per rework attempt |
| `REWORK_ESCALATION_THRESHOLD` | `0.05` | Failure rate threshold for escalation (5%) |
| `CONCURRENT_TRANSACTION_LIMIT` | `100` | Maximum concurrent transaction workflows |
| `ENABLE_TRANSACTION_CACHING` | `true` | Enable caching of transaction generation results |

---

## 🔧 PROJECT 3 ERROR HANDLING

### Exception Hierarchy

```python
class TransactionError(Exception):
    """Base exception for all transaction errors."""

class TransactionGenerationError(TransactionError):
    """Error during transaction generation."""

class BalanceError(TransactionError):
    """GL entry doesn't balance (DR ≠ CR)."""

class ThreeWayMatchError(TransactionError):
    """Three-way match failure beyond tolerance."""

class DiscrepancyInjectionError(TransactionError):
    """Error during discrepancy injection."""

class ReworkLoopError(TransactionError):
    """Error during rework loop processing."""

class PeriodClosedError(TransactionError):
    """Attempt to post to a closed period."""

class GLPostingError(TransactionError):
    """Error during GL posting."""

class PaymentAllocationError(TransactionError):
    """Error during payment allocation."""

class ConcurrencyError(TransactionError):
    """Concurrent access conflict."""
```

### Retry Policies

| Operation | Retry Attempts | Backoff Strategy | Timeout | Fallback |
|-----------|---------------|-----------------|---------|----------|
| P2P cycle generation | 2 | Linear (1s, 2s) | 60s | Skip transaction |
| O2C cycle generation | 2 | Linear (1s, 2s) | 60s | Skip transaction |
| GL posting | 3 | Exponential (1s, 2s, 4s) | 30s | Rollback transaction |
| Three-way matching | 2 | Linear (500ms, 1s) | 10s | Mark as exception |
| Discrepancy injection | 1 | None | 5s | Skip discrepancy |
| Rework loop fix | 3 | Linear (1s, 2s, 3s) | 30s | Escalate to admin |
| Period close | 1 | None | 300s | Halt, require manual intervention |
| Balance update | 2 | Linear (500ms, 1s) | 10s | Rollback transaction |

### Circuit Breakers

| Component | Failure Threshold | Recovery Period | Fallback Action |
|-----------|------------------|----------------|-----------------|
| GL Posting | 10 consecutive failures | 60 seconds | Rollback and queue for retry |
| Discrepancy Injection | 20 consecutive failures | 30 seconds | Skip injection, log warning |
| Rework Loop | 50 consecutive failures | 120 seconds | Escalate all pending items |

---

## 🔬 PROJECT 3 TECHNOLOGY ADDITIONS

### New Production Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sqlalchemy` | `==2.0.25` | ORM and async database access for transaction persistence via Project 1's `get_session()` |
| `psycopg2-binary` | `==2.9.9` | PostgreSQL adapter for SQLAlchemy async sessions |
| `aiofiles` | `==23.2.1` | Async file I/O for ground truth JSON/CSV output files |
| `pydantic-settings` | `==2.2.0` | Environment variable loading for P3 configuration |
| `pandas` | `==2.2.0` | DataFrame operations for ground truth summary CSV generation and batch analytics |

### New Test/Dev Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest-mock` | `>=3.12.0` | Advanced mocking for transaction generator tests and GL posting stubs |
| `hypothesis` | `>=6.92.0` | Property-based testing for transaction generation validation (amounts, balances, sequences) |

### Key Architectural Constraints

- **Python 3.11.7**: Required for specific `Decimal` rounding behavior and `typing.Self` support
- **Decimal Precision**: `decimal.getcontext().prec = 28`, `rounding = ROUND_HALF_UP` — never use `float` for monetary amounts
- **Database Access**: All database operations via `from synthetic_erp.db.session import get_session` with `async with get_session() as session:`
- **Constructor Injection**: All subsystems receive dependencies through constructors (ADR-003) — no service locator, no global state
- **Pydantic V2**: All data contracts at subsystem boundaries use `pydantic.BaseModel`
- **EventBus**: Cross-subsystem async notifications via existing EventBus (ADR-001)
- **Structured Logging**: `structlog` JSON to stdout only — every log entry includes `timestamp`, `service_name`, `component`, `level`, `message`, `trace_id`, `simulation_id`
- **Deterministic Reproducibility**: All random operations use seeded `random.Random` instances — identical seeds produce identical transaction sequences
- **Single Company Focus**: No intercompany, consolidation, or multi-currency for MVP
- **No External Workflow Engines**: In-memory code only — no Airflow, Prefect, or similar

---

## 🔗 PROJECT 3 INTEGRATION WITH PROJECTS 1 & 2

### Dependencies on Project 1

- **Database Sessions**: `get_session()` from `synthetic_erp.db.session` for all database operations
- **Master Data**: Customers, vendors, products, employees, Chart of Accounts
- **Validation Framework**: P0/P1 validation rules consumed by the rework loop

### Dependencies on Project 2

- **Agent Registry**: 12 specialized agents serve as decision-makers within transaction workflows
- **Workflow Orchestrator**: Routes transactions to agents via `ROLE_MAPPING`
- **EventBus**: Publishes `TransactionCreated`, `TransactionCompleted`, `DiscrepancyDetected`, `ApprovalRequired`, `PeriodClosing`, `PeriodClosed`, `DocumentGenerated` events
- **Statistical Models**: Amount distributions, payment timing, order frequency, entity selection
- **Time Controller**: Subscribes to `PeriodClosing` events for period close processing
- **Approval System**: Enforces PO ($5K/$25K/$100K), vendor invoice ($10K/$50K/$100K), and journal entry ($50K) thresholds

### Provided to Project 4

- Transaction data and artifacts for document generation
- Ground truth labels for discrepancy detection training
- GL balances and financial statements for reporting

---

## 📦 PROJECT 3 DELIVERABLES CHECKLIST

### Code Deliverables
- [ ] Abstract `TransactionGenerator` base class with shared logic
- [ ] Custom exception hierarchy (`TransactionError` and 9 subclasses)
- [ ] 5 P2P transaction generators (PO, Receipt, Invoice, 3-Way Match, Payment)
- [ ] 4 O2C transaction generators (Order, Shipment, Invoice, Payment)
- [ ] GL Posting Engine with balance validation and trial balance checks
- [ ] Account Balance Manager with `SELECT FOR UPDATE` locking
- [ ] Accrual Generator (GRNI, shipped-not-invoiced, reversing entries)
- [ ] Period Close Manager (10-step close process)
- [ ] Discrepancy Injector with rate-based trigger and type selection
- [ ] Ground Truth Generator with 16-field schema
- [ ] Discrepancy Catalog mapping 35+ type codes to implementations
- [ ] 35 individual discrepancy type implementations (15 P2P, 10 O2C, 5 GL, 5 Control)
- [ ] Rework Loop Engine with classify → fix → re-validate → escalate cycle
- [ ] Failure Classifier (planned discrepancy vs. unplanned error)
- [ ] Fix Scenario Catalog with 20+ fix scenarios
- [ ] Fix Scenario Executor with per-scenario timeout
- [ ] Integration into SimulationEngine composition root

### Configuration Deliverables
- [ ] P2P, O2C, GL discrepancy configuration YAML files
- [ ] Global discrepancy injection rate configuration
- [ ] GL posting rules YAML
- [ ] Period close configuration YAML
- [ ] 11 new environment variables in `.env.example`

### Testing Deliverables
- [ ] Unit tests for all transaction generators (≥80% coverage)
- [ ] Full P2P and O2C cycle integration tests with GL balance assertions
- [ ] GL posting engine tests (balance validation, trial balance, concurrent posting, rollback atomicity)
- [ ] Three-way matching tolerance tests
- [ ] Period close tests (accruals, trial balance, state transitions)
- [ ] Discrepancy injection rate and distribution tests
- [ ] Ground truth completeness and schema validation tests
- [ ] Parameterized tests for all 35+ discrepancy types
- [ ] Rework loop end-to-end tests (3-attempt limit, escalation)
- [ ] Failure classification tests (planned vs. unplanned)
- [ ] Fix scenario selection and execution tests
- [ ] Property-based tests using Hypothesis for financial invariants

---

## ✅ PROJECT 3 ACCEPTANCE CRITERIA

### Financial Integrity
1. ✅ GL Balance Zero: `SUM(Debits) - SUM(Credits) == $0.00` after batch of 1,000 mixed transactions
2. ✅ Concurrent Posting: 20 agents post to 'Cash' account simultaneously; final balance equals sum of inputs
3. ✅ Period Close: Transactions dated Dec 31 are posted; Jan 1 transactions blocked until period opens
4. ✅ Overpayment: Payment of $1,100 on $1,000 invoice → $0 Invoice Balance + $100 Unapplied Cash (no negative balance)
5. ✅ Rollback: DB error during 'Post Line 2' causes 'Post Line 1' to disappear (Atomicity)

### Transaction Workflows
1. ✅ Complete P2P cycle produces all required artifacts (PO, receipt, invoice, 3-way match, payment, GL entries)
2. ✅ Complete O2C cycle produces all required artifacts (SO, shipment, invoice, payment, GL entries)
3. ✅ Three-way match enforces ±5% price and ±2% quantity tolerances
4. ✅ FIFO payment allocation processes oldest invoices first
5. ✅ Approval chains enforced for amounts exceeding configured thresholds

### Discrepancy System
1. ✅ Injection rate within ±1% of configured target
2. ✅ All 35+ discrepancy types produce valid ground truth records
3. ✅ Parameters respect configured bounds
4. ✅ Easy/Medium difficulty distribution within ±5% tolerance

### Rework Loop
1. ✅ Unplanned errors corrected within 3 attempts or escalated
2. ✅ Fix scenarios sorted by success rate and excluding previously failed scenarios
3. ✅ Rework completes within 30-second timeout per transaction

---

## 📚 REFERENCE SPECIFICATIONS

> **Note:** The following specification documents are external upstream design documents that informed the implementation of this project. They are **not included** in this repository — they reside in the project's design documentation system and are referenced here for traceability purposes only.

### Project 2 References
1. **AGENT_ARCHITECTURE.md** - Agent structure, roles, memory
2. **AGENT_DECISION_SPEC.md** - Decision logic, LLM prompts
3. **ORCHESTRATION_DESIGN.md** - Workflow orchestration
4. **STATISTICAL_MODELS_SPEC.md** - Payment timing, amounts
5. **EXTERNAL_ENTITIES_SPEC.md** - Customer/vendor simulation
6. **FINANCIAL_GROUNDING_SPEC.md** - Financial targets

### Project 3 References
7. **DISCREPANCY_CATALOG.md** - All 65+ discrepancy types (35+ for MVP)
8. **GROUND_TRUTH_SPEC.md** - Ground truth schemas
9. **REWORK_FLOW_SPEC.md** - Intelligent rework loop
10. **DATA_CONSISTENCY_SPEC.md** - Integrity constraints
11. **VALIDATION_FRAMEWORK_SPEC.md** - Validation rules

---

**END OF PROJECT 2 & PROJECT 3 SPECIFICATION**
