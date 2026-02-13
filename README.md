# PROJECT 2: AGENT & ORCHESTRATION ENGINE
## Synthetic ERP Data Generation Platform - Phase 2

---

## 🎯 PROJECT OBJECTIVE

**BUILD** an AI-powered agent and orchestration system that simulates realistic employee behavior using autonomous agents with memory, reflection, and LLM-powered decision-making, along with workflow orchestration, time management, and external world simulation components.

**DELIVERABLE:** A fully functional agent-based simulation engine that can:
1. Create and manage autonomous AI agents representing employees
2. Make realistic business decisions using hybrid statistical + LLM approach
3. Orchestrate workflows with approval chains and routing logic
4. Simulate external entities (customers, vendors, banks) with tiered behavior
5. Manage simulation time with business calendars and fiscal periods
6. Generate transaction triggers using statistical models
7. Coordinate all components through event-driven architecture

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

### Code Deliverables
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

This project implements specifications from:

1. **AGENT_ARCHITECTURE.md** - Agent structure, roles, memory
2. **AGENT_DECISION_SPEC.md** - Decision logic, LLM prompts
3. **ORCHESTRATION_DESIGN.md** - Workflow orchestration
4. **STATISTICAL_MODELS_SPEC.md** - Payment timing, amounts
5. **EXTERNAL_ENTITIES_SPEC.md** - Customer/vendor simulation
6. **FINANCIAL_GROUNDING_SPEC.md** - Financial targets

---

**END OF PROJECT 2 SPECIFICATION**
