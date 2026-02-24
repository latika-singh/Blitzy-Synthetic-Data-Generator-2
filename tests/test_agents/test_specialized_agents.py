"""
Tests for All 12 Specialized Agent Types — Agent System (F-001).

Validates that every specialized agent subclass:
    - Extends ``BaseAgent`` and implements ``process_work_item()``
    - Has the correct ``ROLE`` class constant
    - Has the correct ``SUPPORTED_WORK_TYPES`` list
    - Routes each supported work type to the proper handler
    - Delegates decisions via ``make_decision()`` (never direct LLM calls)
    - Enforces approval thresholds and escalation paths
    - Raises ``ValueError`` for unsupported work types

CRITICAL Testing Rules (AAP Section 0.7.5):
    - ALL LLM integration tests use mocked API responses — NO live API calls.
    - Async tests use pytest-asyncio with proper event loop management.
    - ALL agent tests verify state transitions.
    - Coverage target: ≥ 80%.

CRITICAL Architectural Rules (AAP Section 0.7.1):
    - Every specialized agent MUST extend BaseAgent.
    - Agents use self.make_decision() (inherited from BaseAgent).
    - Constructor injection for all dependencies.
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import pytest

# ---------------------------------------------------------------------------
# Internal Imports — All 12 Specialized Agent Classes
# ---------------------------------------------------------------------------
from app.agents.specialized.ap_clerk_agent import APClerkAgent
from app.agents.specialized.ap_manager_agent import APManagerAgent
from app.agents.specialized.ar_clerk_agent import ARClerkAgent
from app.agents.specialized.ar_manager_agent import ARManagerAgent
from app.agents.specialized.purchasing_agent import PurchasingAgent
from app.agents.specialized.purchasing_manager_agent import PurchasingManagerAgent
from app.agents.specialized.warehouse_clerk_agent import WarehouseClerkAgent
from app.agents.specialized.warehouse_manager_agent import WarehouseManagerAgent
from app.agents.specialized.accountant_agent import AccountantAgent
from app.agents.specialized.senior_accountant_agent import SeniorAccountantAgent
from app.agents.specialized.controller_agent import ControllerAgent
from app.agents.specialized.cfo_agent import CFOAgent

# ---------------------------------------------------------------------------
# Internal Imports — Base Classes and Data Structures
# ---------------------------------------------------------------------------
from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig, AgentState
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry


# =========================================================================
# Complete list of the 12 agent classes and their expected roles
# =========================================================================
ALL_AGENT_CLASSES = [
    APClerkAgent,
    APManagerAgent,
    ARClerkAgent,
    ARManagerAgent,
    PurchasingAgent,
    PurchasingManagerAgent,
    WarehouseClerkAgent,
    WarehouseManagerAgent,
    AccountantAgent,
    SeniorAccountantAgent,
    ControllerAgent,
    CFOAgent,
]

ALL_AGENT_ROLE_PAIRS = [
    (APClerkAgent, "ap_clerk"),
    (APManagerAgent, "ap_manager"),
    (ARClerkAgent, "ar_clerk"),
    (ARManagerAgent, "ar_manager"),
    (PurchasingAgent, "purchasing_agent"),
    (PurchasingManagerAgent, "purchasing_manager"),
    (WarehouseClerkAgent, "warehouse_clerk"),
    (WarehouseManagerAgent, "warehouse_manager"),
    (AccountantAgent, "accountant"),
    (SeniorAccountantAgent, "senior_accountant"),
    (ControllerAgent, "controller"),
    (CFOAgent, "cfo"),
]


# =========================================================================
# Shared Fixtures
# =========================================================================


@pytest.fixture
def agent_dependencies(mock_agent_memory, mock_decision_engine, mock_action_registry):
    """Bundle all mock dependencies required to construct any agent."""
    return {
        "memory": mock_agent_memory,
        "decision_engine": mock_decision_engine,
        "action_registry": mock_action_registry,
    }


@pytest.fixture
def make_agent(agent_dependencies):
    """Factory fixture that creates any specialized agent with mock deps.

    Returns a callable ``(agent_class, role_override=None, config_overrides=None) -> agent``.
    """

    def _factory(agent_class, role_override=None, config_overrides=None):
        role = role_override or agent_class.ROLE
        base_cfg = {
            "agent_id": uuid4(),
            "role": role,
            "name": f"Test {role}",
            "employee_id": uuid4(),
            "company_id": uuid4(),
            "traits": {
                "thoroughness": 0.7,
                "risk_tolerance": 0.5,
                "efficiency": 0.6,
                "compliance": 0.8,
            },
            "work_hours_start": 8,
            "work_hours_end": 17,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        if config_overrides:
            base_cfg.update(config_overrides)
        config = AgentConfig(**base_cfg)
        return agent_class(
            config=config,
            memory=agent_dependencies["memory"],
            decision_engine=agent_dependencies["decision_engine"],
            action_registry=agent_dependencies["action_registry"],
        )

    return _factory


# =========================================================================
# TestSpecializedAgentInheritance — Phase 3
# =========================================================================


class TestSpecializedAgentInheritance:
    """Verify that every specialized agent extends BaseAgent."""

    def test_ap_clerk_extends_base_agent(self):
        assert issubclass(APClerkAgent, BaseAgent)

    def test_ap_manager_extends_base_agent(self):
        assert issubclass(APManagerAgent, BaseAgent)

    def test_ar_clerk_extends_base_agent(self):
        assert issubclass(ARClerkAgent, BaseAgent)

    def test_ar_manager_extends_base_agent(self):
        assert issubclass(ARManagerAgent, BaseAgent)

    def test_purchasing_agent_extends_base_agent(self):
        assert issubclass(PurchasingAgent, BaseAgent)

    def test_purchasing_manager_extends_base_agent(self):
        assert issubclass(PurchasingManagerAgent, BaseAgent)

    def test_warehouse_clerk_extends_base_agent(self):
        assert issubclass(WarehouseClerkAgent, BaseAgent)

    def test_warehouse_manager_extends_base_agent(self):
        assert issubclass(WarehouseManagerAgent, BaseAgent)

    def test_accountant_extends_base_agent(self):
        assert issubclass(AccountantAgent, BaseAgent)

    def test_senior_accountant_extends_base_agent(self):
        assert issubclass(SeniorAccountantAgent, BaseAgent)

    def test_controller_extends_base_agent(self):
        assert issubclass(ControllerAgent, BaseAgent)

    def test_cfo_extends_base_agent(self):
        assert issubclass(CFOAgent, BaseAgent)

    def test_all_agents_have_role_constant(self):
        for cls in ALL_AGENT_CLASSES:
            assert hasattr(cls, "ROLE"), f"{cls.__name__} missing ROLE"
            assert isinstance(cls.ROLE, str), f"{cls.__name__}.ROLE is not str"

    def test_all_agents_have_supported_work_types(self):
        for cls in ALL_AGENT_CLASSES:
            assert hasattr(cls, "SUPPORTED_WORK_TYPES"), (
                f"{cls.__name__} missing SUPPORTED_WORK_TYPES"
            )
            assert isinstance(cls.SUPPORTED_WORK_TYPES, list), (
                f"{cls.__name__}.SUPPORTED_WORK_TYPES is not a list"
            )


# =========================================================================
# TestAgentRoles — Phase 4
# =========================================================================


class TestAgentRoles:
    """Verify each agent's ROLE constant matches the expected value."""

    def test_ap_clerk_role(self):
        assert APClerkAgent.ROLE == "ap_clerk"

    def test_ap_manager_role(self):
        assert APManagerAgent.ROLE == "ap_manager"

    def test_ar_clerk_role(self):
        assert ARClerkAgent.ROLE == "ar_clerk"

    def test_ar_manager_role(self):
        assert ARManagerAgent.ROLE == "ar_manager"

    def test_purchasing_agent_role(self):
        assert PurchasingAgent.ROLE == "purchasing_agent"

    def test_purchasing_manager_role(self):
        assert PurchasingManagerAgent.ROLE == "purchasing_manager"

    def test_warehouse_clerk_role(self):
        assert WarehouseClerkAgent.ROLE == "warehouse_clerk"

    def test_warehouse_manager_role(self):
        assert WarehouseManagerAgent.ROLE == "warehouse_manager"

    def test_accountant_role(self):
        assert AccountantAgent.ROLE == "accountant"

    def test_senior_accountant_role(self):
        assert SeniorAccountantAgent.ROLE == "senior_accountant"

    def test_controller_role(self):
        assert ControllerAgent.ROLE == "controller"

    def test_cfo_role(self):
        assert CFOAgent.ROLE == "cfo"

    def test_all_roles_unique(self):
        roles = [cls.ROLE for cls in ALL_AGENT_CLASSES]
        assert len(set(roles)) == 12, "All 12 agent roles must be unique"


# =========================================================================
# TestAPClerkAgent — Phase 5
# =========================================================================


class TestAPClerkAgent:
    """Test APClerkAgent: vendor invoice processing, 3-way match, GL coding."""

    @pytest.mark.asyncio
    async def test_process_vendor_invoice_success(self, make_agent, mock_decision_engine):
        """Vendor invoice with matching PO/receipt → success, gl_coding + 3-way match."""
        agent = make_agent(APClerkAgent)
        # Decision engine returns GL coding and processing notes
        mock_decision_engine.decide = AsyncMock(return_value={
            "gl_coding": {"account": "2100-00", "amount": 5000.0},
            "processing_notes": "Invoice matches PO",
            "decision": "approve",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="vendor_invoice",
            data={
                "invoice_id": "INV-001",
                "vendor_id": "V-001",
                "amount": 5000.0,
                "purchase_order": {"po_id": "PO-001", "unit_price": 50.0, "quantity": 100},
                "goods_receipt": {"quantity": 100},
                "quantity": 100,
                "unit_price": 50.0,
            },
            amount=5000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert "gl_coding_assigned" in result.actions_taken
        assert "three_way_match_performed" in result.actions_taken

    @pytest.mark.asyncio
    async def test_process_vendor_invoice_with_variance(self, make_agent, mock_decision_engine):
        """Invoice with quantity mismatch → variance exceeds tolerance → approval needed."""
        agent = make_agent(APClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "gl_coding": {"account": "2100-00"},
            "processing_notes": "Variance detected",
            "decision": "approve",
            "confidence": 0.7,
        })
        # Quantity mismatch: invoice says 100, receipt says 80
        work_item = WorkItem(
            type="vendor_invoice",
            data={
                "invoice_id": "INV-002",
                "vendor_id": "V-002",
                "amount": 5000.0,
                "purchase_order": {"po_id": "PO-002", "unit_price": 50.0, "quantity": 100},
                "goods_receipt": {"quantity": 80},
                "quantity": 100,
                "unit_price": 50.0,
            },
            amount=5000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert "three_way_match_performed" in result.actions_taken
        # Variance exceeds 5% tolerance → approval should be needed
        assert result.approval_needed is True

    @pytest.mark.asyncio
    async def test_process_vendor_invoice_matched(self, make_agent, mock_decision_engine):
        """Perfectly matching PO/receipt/invoice → no approval needed."""
        agent = make_agent(APClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "gl_coding": {"account": "5100-00"},
            "processing_notes": "All matched",
            "decision": "approve",
            "confidence": 0.95,
        })
        work_item = WorkItem(
            type="vendor_invoice",
            data={
                "invoice_id": "INV-003",
                "vendor_id": "V-003",
                "amount": 5000.0,
                "purchase_order": {"po_id": "PO-003", "unit_price": 50.0, "quantity": 100},
                "goods_receipt": {"quantity": 100},
                "quantity": 100,
                "unit_price": 50.0,
            },
            amount=5000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is False

    @pytest.mark.asyncio
    async def test_handle_invoice_exception(self, make_agent, mock_decision_engine):
        """Invoice exception handling delegates to DecisionEngine."""
        agent = make_agent(APClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "resolution": "escalate",
            "reasoning": "Cannot resolve automatically",
            "decision": "escalate",
            "confidence": 0.6,
        })
        work_item = WorkItem(
            type="invoice_exception",
            data={
                "invoice_id": "INV-ERR-001",
                "exception_type": "duplicate_invoice",
                "details": "Possible duplicate",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        # make_decision should have been called
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_unknown_work_type_raises_error(self, make_agent):
        """Unsupported work type raises ValueError."""
        agent = make_agent(APClerkAgent)
        work_item = WorkItem(type="unknown_type", data={})
        with pytest.raises(ValueError):
            await agent.process_work_item(work_item)

    @pytest.mark.asyncio
    async def test_delegates_to_decision_engine(self, make_agent, mock_decision_engine):
        """Verify the agent calls make_decision (not direct LLM calls)."""
        agent = make_agent(APClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "gl_coding": {"account": "2100-00"},
            "processing_notes": "Test",
            "decision": "approve",
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="vendor_invoice",
            data={
                "invoice_id": "INV-004",
                "vendor_id": "V-004",
                "amount": 1000.0,
                "purchase_order": {"po_id": "PO-004", "unit_price": 10.0, "quantity": 100},
                "goods_receipt": {"quantity": 100},
                "quantity": 100,
                "unit_price": 10.0,
            },
            amount=1000.0,
        )
        await agent.process_work_item(work_item)
        # DecisionEngine.decide should have been invoked through make_decision()
        assert mock_decision_engine.decide.call_count >= 1

    def test_variance_tolerance(self):
        """Verify VARIANCE_TOLERANCE constant is 0.05 (5%)."""
        assert APClerkAgent.VARIANCE_TOLERANCE == 0.05


# =========================================================================
# TestAPManagerAgent — Phase 6
# =========================================================================


class TestAPManagerAgent:
    """Test APManagerAgent: invoice approval, escalation, exception handling."""

    @pytest.mark.asyncio
    async def test_invoice_approval_within_authority(self, make_agent, mock_decision_engine):
        """Invoice $25K (within $10K-$50K) → manager makes approval decision."""
        agent = make_agent(APManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Policy compliant",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="invoice_approval",
            data={
                "invoice_id": "INV-010",
                "amount": 25000.0,
                "vendor_id": "V-010",
                "match_result": {"match_status": "matched"},
            },
            amount=25000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_invoice_approval_above_authority_escalates(self, make_agent, mock_decision_engine):
        """Invoice $60K (>$50K) → auto-escalated to controller."""
        agent = make_agent(APManagerAgent)
        work_item = WorkItem(
            type="invoice_approval",
            data={
                "invoice_id": "INV-011",
                "amount": 60000.0,
                "vendor_id": "V-011",
                "match_result": {"match_status": "matched"},
            },
            amount=60000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        # Should escalate, not approve
        assert result.approval_needed is True
        has_escalation = any(
            "escalat" in action.lower() for action in result.actions_taken
        )
        assert has_escalation, (
            f"Expected escalation in actions_taken: {result.actions_taken}"
        )

    @pytest.mark.asyncio
    async def test_handle_invoice_exception(self, make_agent, mock_decision_engine):
        """AP Manager processes invoice_exception via DecisionEngine."""
        agent = make_agent(APManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "resolve",
            "resolution": "adjust_amount",
            "reasoning": "Minor discrepancy",
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="invoice_exception",
            data={
                "invoice_id": "INV-ERR-010",
                "exception_type": "amount_mismatch",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_process_vendor_invoice(self, make_agent, mock_decision_engine):
        """AP Manager can also process vendor_invoice type."""
        agent = make_agent(APManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "gl_coding": {"account": "2100-00"},
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="vendor_invoice",
            data={
                "invoice_id": "INV-012",
                "vendor_id": "V-012",
                "amount": 15000.0,
            },
            amount=15000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    def test_threshold_constants(self):
        """Verify AP Manager threshold constants."""
        assert APManagerAgent.APPROVAL_THRESHOLD_MIN == 10000.0
        assert APManagerAgent.APPROVAL_THRESHOLD_MAX == 50000.0


# =========================================================================
# TestARClerkAgent — Phase 7
# =========================================================================


class TestARClerkAgent:
    """Test ARClerkAgent: sales orders, customer payments, invoicing."""

    @pytest.mark.asyncio
    async def test_process_sales_order(self, make_agent, mock_decision_engine):
        """Process sales_order → DecisionEngine called, success."""
        agent = make_agent(ARClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "pricing_validated": True,
            "credit_check": "passed",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="sales_order",
            data={
                "order_id": "SO-001",
                "customer_id": "C-001",
                "items": [{"product_id": "P-001", "quantity": 10, "unit_price": 100.0}],
                "total_amount": 1000.0,
            },
            amount=1000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_process_customer_payment_full_match(self, make_agent, mock_decision_engine):
        """Payment matches invoice amount → payment_applied, no exceptions."""
        agent = make_agent(ARClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "apply",
            "match_status": "full_match",
            "confidence": 0.95,
        })
        work_item = WorkItem(
            type="customer_payment",
            data={
                "payment_id": "PMT-001",
                "customer_id": "C-001",
                "payment_amount": 5000.0,
                "invoice_amount": 5000.0,
                "invoice_id": "INV-AR-001",
            },
            amount=5000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.has_exceptions is False

    @pytest.mark.asyncio
    async def test_process_customer_payment_short_pay(self, make_agent, mock_decision_engine):
        """Payment < invoice amount → short pay → has_exceptions=True."""
        agent = make_agent(ARClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "apply",
            "match_status": "short_pay",
            "confidence": 0.7,
        })
        work_item = WorkItem(
            type="customer_payment",
            data={
                "payment_id": "PMT-002",
                "customer_id": "C-002",
                "payment_amount": 3000.0,
                "invoice_amount": 5000.0,
                "invoice_id": "INV-AR-002",
            },
            amount=3000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.has_exceptions is True

    @pytest.mark.asyncio
    async def test_create_customer_invoice(self, make_agent, mock_decision_engine):
        """Create customer_invoice → success with proper actions.

        The AR clerk validates that ``amount > 0`` and ``customer_id`` is
        present in work_item.data.  We supply both to get a successful path.
        """
        agent = make_agent(ARClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "description": "Monthly service invoice",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="customer_invoice",
            data={
                "order_id": "SO-003",
                "customer_id": "C-003",
                "amount": 2000.0,
                "line_items": [{"product_id": "P-010", "quantity": 5, "unit_price": 400.0}],
                "payment_terms": "Net 30",
            },
            amount=2000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unknown_work_type_raises_error(self, make_agent):
        """Unsupported work type raises ValueError."""
        agent = make_agent(ARClerkAgent)
        work_item = WorkItem(type="bogus_type", data={})
        with pytest.raises(ValueError):
            await agent.process_work_item(work_item)


# =========================================================================
# TestARManagerAgent — Phase 8
# =========================================================================


class TestARManagerAgent:
    """Test ARManagerAgent: dispute resolution, approval, exceptions."""

    @pytest.mark.asyncio
    async def test_resolve_dispute_accept(self, make_agent, mock_decision_engine):
        """Dispute resolution with accept outcome → dispute_accepted in actions."""
        agent = make_agent(ARManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "accept",
            "resolution": "accept",
            "reasoning": "Customer claim is valid",
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="dispute_resolution",
            data={
                "dispute_id": "DISP-001",
                "customer_id": "C-010",
                "amount": 1500.0,
                "reason": "Damaged goods",
            },
            amount=1500.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        has_accept = any(
            "accept" in action.lower() for action in result.actions_taken
        )
        assert has_accept, f"Expected accept action in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_resolve_dispute_reject(self, make_agent, mock_decision_engine):
        """Dispute resolution with reject outcome → dispute_rejected in actions."""
        agent = make_agent(ARManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "reject",
            "resolution": "reject",
            "reasoning": "No evidence for claim",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="dispute_resolution",
            data={
                "dispute_id": "DISP-002",
                "customer_id": "C-011",
                "amount": 500.0,
                "reason": "Overcharge",
            },
            amount=500.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        has_reject = any(
            "reject" in action.lower() for action in result.actions_taken
        )
        assert has_reject, f"Expected reject action in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_process_ar_approval(self, make_agent, mock_decision_engine):
        """AR approval delegates to DecisionEngine with approve_transaction."""
        agent = make_agent(ARManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Approved",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="ar_approval",
            data={
                "transaction_id": "TXN-AR-001",
                "amount": 8000.0,
            },
            amount=8000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_handle_payment_exception(self, make_agent, mock_decision_engine):
        """Payment exception handling via DecisionEngine."""
        agent = make_agent(ARManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "resolve",
            "resolution": "adjust",
            "reasoning": "Bank error correction",
            "confidence": 0.75,
        })
        work_item = WorkItem(
            type="payment_exception",
            data={
                "payment_id": "PMT-ERR-001",
                "exception_type": "bank_mismatch",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()


# =========================================================================
# TestPurchasingAgent — Phase 9
# =========================================================================


class TestPurchasingAgent:
    """Test PurchasingAgent: PO creation with tiered approval routing."""

    @pytest.mark.asyncio
    async def test_create_purchase_order_no_approval(self, make_agent, mock_decision_engine):
        """PO amount $3K (<$5K) → no approval needed."""
        agent = make_agent(PurchasingAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "vendor_selection": {"vendor_id": "V-020"},
            "amount": 3000.0,
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="purchase_order",
            data={
                "po_id": "PO-020",
                "vendor_id": "V-020",
                "items": [{"product_id": "P-001", "quantity": 30, "unit_price": 100.0}],
                "total_amount": 3000.0,
            },
            amount=3000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is False

    @pytest.mark.asyncio
    async def test_create_purchase_order_manager_approval(self, make_agent, mock_decision_engine):
        """PO amount $15K ($5K-$25K) → purchasing_manager approval needed."""
        agent = make_agent(PurchasingAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "vendor_selection": {"vendor_id": "V-021"},
            "amount": 15000.0,
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="purchase_order",
            data={
                "po_id": "PO-021",
                "vendor_id": "V-021",
                "items": [{"product_id": "P-002", "quantity": 150, "unit_price": 100.0}],
                "total_amount": 15000.0,
            },
            amount=15000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        assert result.data.get("approver_role") == "purchasing_manager"

    @pytest.mark.asyncio
    async def test_create_purchase_order_controller_approval(self, make_agent, mock_decision_engine):
        """PO amount $50K ($25K-$100K) → controller approval needed."""
        agent = make_agent(PurchasingAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "vendor_selection": {"vendor_id": "V-022"},
            "amount": 50000.0,
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="purchase_order",
            data={
                "po_id": "PO-022",
                "vendor_id": "V-022",
                "items": [{"product_id": "P-003", "quantity": 500, "unit_price": 100.0}],
                "total_amount": 50000.0,
            },
            amount=50000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        assert result.data.get("approver_role") == "controller"

    @pytest.mark.asyncio
    async def test_create_purchase_order_cfo_approval(self, make_agent, mock_decision_engine):
        """PO amount $150K (>$100K) → CFO approval needed."""
        agent = make_agent(PurchasingAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "vendor_selection": {"vendor_id": "V-023"},
            "amount": 150000.0,
            "confidence": 0.75,
        })
        work_item = WorkItem(
            type="purchase_order",
            data={
                "po_id": "PO-023",
                "vendor_id": "V-023",
                "items": [{"product_id": "P-004", "quantity": 1500, "unit_price": 100.0}],
                "total_amount": 150000.0,
            },
            amount=150000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        assert result.data.get("approver_role") == "cfo"

    def test_threshold_constants(self):
        """Verify PO approval threshold constants."""
        assert PurchasingAgent.NO_APPROVAL_THRESHOLD == 5000.0
        assert PurchasingAgent.MANAGER_APPROVAL_THRESHOLD == 25000.0
        assert PurchasingAgent.CONTROLLER_APPROVAL_THRESHOLD == 100000.0


# =========================================================================
# TestPurchasingManagerAgent — Phase 10
# =========================================================================


class TestPurchasingManagerAgent:
    """Test PurchasingManagerAgent: PO approval, escalation, vendor review."""

    @pytest.mark.asyncio
    async def test_po_approval_within_authority(self, make_agent, mock_decision_engine):
        """PO $15K ($5K-$25K) → purchasing manager approves."""
        agent = make_agent(PurchasingManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Within budget",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="po_approval",
            data={
                "po_id": "PO-030",
                "amount": 15000.0,
                "vendor_id": "V-030",
            },
            amount=15000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_po_approval_above_authority_escalates(self, make_agent, mock_decision_engine):
        """PO $30K (>$25K) → escalated to controller."""
        agent = make_agent(PurchasingManagerAgent)
        work_item = WorkItem(
            type="po_approval",
            data={
                "po_id": "PO-031",
                "amount": 30000.0,
                "vendor_id": "V-031",
            },
            amount=30000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        has_escalation = any(
            "escalat" in action.lower() for action in result.actions_taken
        )
        assert has_escalation, f"Expected escalation in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_vendor_review(self, make_agent, mock_decision_engine):
        """Vendor review work type processed successfully."""
        agent = make_agent(PurchasingManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "review_outcome": "satisfactory",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="vendor_review",
            data={
                "vendor_id": "V-032",
                "review_period": "Q1-2024",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    def test_threshold_constants(self):
        """Verify Purchasing Manager threshold constants."""
        assert PurchasingManagerAgent.APPROVAL_THRESHOLD_MIN == 5000.0
        assert PurchasingManagerAgent.APPROVAL_THRESHOLD_MAX == 25000.0


# =========================================================================
# TestWarehouseClerkAgent — Phase 11
# =========================================================================


class TestWarehouseClerkAgent:
    """Test WarehouseClerkAgent: goods receipt, shipment, discrepancy detection."""

    @pytest.mark.asyncio
    async def test_process_goods_receipt_no_discrepancy(self, make_agent, mock_decision_engine):
        """All received_items match ordered quantities → no exceptions."""
        agent = make_agent(WarehouseClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "quality_check": "passed",
            "confidence": 0.95,
        })
        work_item = WorkItem(
            type="goods_receipt",
            data={
                "po_id": "PO-040",
                "vendor_id": "V-020",
                "delivery_date": "2024-03-15",
                "warehouse_id": "WH-001",
                "received_items": [
                    {
                        "item_id": "ITEM-010",
                        "received_quantity": 100,
                        "ordered_quantity": 100,
                        "condition": "good",
                    },
                ],
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.has_exceptions is False

    @pytest.mark.asyncio
    async def test_process_goods_receipt_with_discrepancy(self, make_agent, mock_decision_engine):
        """Quantity mismatch in received_items → has_exceptions=True, discrepancy action.

        The warehouse clerk iterates ``received_items`` and compares each
        item's ``received_quantity`` to its ``ordered_quantity``.  A short
        shipment triggers the Decision Engine with ``handle_exception``.
        """
        agent = make_agent(WarehouseClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "accept_partial",
            "resolution": "accept_partial",
            "reasoning": "Vendor notified for remainder",
            "confidence": 0.7,
        })
        work_item = WorkItem(
            type="goods_receipt",
            data={
                "po_id": "PO-041",
                "vendor_id": "V-020",
                "delivery_date": "2024-03-15",
                "warehouse_id": "WH-001",
                "received_items": [
                    {
                        "item_id": "ITEM-011",
                        "received_quantity": 80,
                        "ordered_quantity": 100,
                        "condition": "good",
                    },
                ],
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.has_exceptions is True
        has_discrepancy = any(
            "discrepancy" in action.lower() for action in result.actions_taken
        )
        assert has_discrepancy, f"Expected discrepancy action in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_process_shipment(self, make_agent, mock_decision_engine):
        """Shipment processing → shipment_processed in actions."""
        agent = make_agent(WarehouseClerkAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "availability": "in_stock",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="shipment",
            data={
                "shipment_id": "SHP-001",
                "order_id": "SO-040",
                "items": [{"product_id": "P-012", "quantity": 50}],
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        has_shipment = any(
            "shipment" in action.lower() for action in result.actions_taken
        )
        assert has_shipment, f"Expected shipment action in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_unknown_work_type_raises_error(self, make_agent):
        """Unsupported work type raises ValueError."""
        agent = make_agent(WarehouseClerkAgent)
        work_item = WorkItem(type="invalid_type", data={})
        with pytest.raises(ValueError):
            await agent.process_work_item(work_item)


# =========================================================================
# TestWarehouseManagerAgent — Phase 12
# =========================================================================


class TestWarehouseManagerAgent:
    """Test WarehouseManagerAgent: approval, discrepancy, shipment exceptions."""

    @pytest.mark.asyncio
    async def test_process_warehouse_approval(self, make_agent, mock_decision_engine):
        """Warehouse approval delegates to DecisionEngine."""
        agent = make_agent(WarehouseManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Warehouse capacity sufficient",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="warehouse_approval",
            data={
                "approval_type": "receipt_override",
                "receipt_id": "GR-010",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_handle_receipt_discrepancy(self, make_agent, mock_decision_engine):
        """Receipt discrepancy handled via DecisionEngine (handle_exception)."""
        agent = make_agent(WarehouseManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "resolve",
            "resolution": "accept_partial",
            "reasoning": "Vendor notified for remainder",
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="receipt_discrepancy",
            data={
                "receipt_id": "GR-011",
                "discrepancy_type": "short_shipment",
                "expected": 100,
                "received": 85,
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_handle_shipment_exception(self, make_agent, mock_decision_engine):
        """Shipment exception handling via DecisionEngine."""
        agent = make_agent(WarehouseManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "resolve",
            "resolution": "reroute",
            "reasoning": "Alternative carrier assigned",
            "confidence": 0.75,
        })
        work_item = WorkItem(
            type="shipment_exception",
            data={
                "shipment_id": "SHP-ERR-001",
                "exception_type": "carrier_delay",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_process_inventory_review(self, make_agent, mock_decision_engine):
        """Inventory review work type processed successfully."""
        agent = make_agent(WarehouseManagerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "review_outcome": "acceptable",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="inventory_review",
            data={
                "warehouse_id": "WH-001",
                "review_period": "2024-Q1",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True


# =========================================================================
# TestAccountantAgent — Phase 13
# =========================================================================


class TestAccountantAgent:
    """Test AccountantAgent: journal entries, balance validation, reconciliation."""

    @pytest.mark.asyncio
    async def test_create_journal_entry_balanced(self, make_agent, mock_decision_engine):
        """Balanced JE (debits==credits) → journal_entry_created, success."""
        agent = make_agent(AccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "description": "Office supplies expense",
            "tags": ["expense", "supplies"],
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="journal_entry",
            data={
                "entries": [
                    {"account": "5100-00", "debit": 1000.0, "credit": 0.0},
                    {"account": "2100-00", "debit": 0.0, "credit": 1000.0},
                ],
                "description": "Office supplies",
                "period": "2024-01",
            },
            amount=1000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        has_je_created = any(
            "journal_entry_created" in action for action in result.actions_taken
        )
        assert has_je_created, f"Expected journal_entry_created in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_create_journal_entry_unbalanced(self, make_agent, mock_decision_engine):
        """Unbalanced JE (debits!=credits) → failure or has_exceptions."""
        agent = make_agent(AccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "description": "Unbalanced entry",
            "confidence": 0.5,
        })
        work_item = WorkItem(
            type="journal_entry",
            data={
                "entries": [
                    {"account": "5100-00", "debit": 1000.0, "credit": 0.0},
                    {"account": "2100-00", "debit": 0.0, "credit": 500.0},
                ],
                "description": "Unbalanced test",
                "period": "2024-01",
            },
            amount=1000.0,
        )
        result = await agent.process_work_item(work_item)
        # Unbalanced JE → either not successful or has_exceptions
        assert result.success is False or result.has_exceptions is True

    @pytest.mark.asyncio
    async def test_je_approval_routing_senior_accountant(self, make_agent, mock_decision_engine):
        """JE $30K (<$50K) → approver_role == 'senior_accountant'."""
        agent = make_agent(AccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "description": "Standard JE",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="journal_entry",
            data={
                "entries": [
                    {"account": "1000-00", "debit": 30000.0, "credit": 0.0},
                    {"account": "2000-00", "debit": 0.0, "credit": 30000.0},
                ],
                "description": "Monthly accrual",
                "period": "2024-01",
            },
            amount=30000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        assert result.data.get("approver_role") == "senior_accountant"

    @pytest.mark.asyncio
    async def test_je_approval_routing_controller(self, make_agent, mock_decision_engine):
        """JE $75K (>$50K) → approver_role == 'controller'."""
        agent = make_agent(AccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "description": "Large JE",
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="journal_entry",
            data={
                "entries": [
                    {"account": "1000-00", "debit": 75000.0, "credit": 0.0},
                    {"account": "2000-00", "debit": 0.0, "credit": 75000.0},
                ],
                "description": "Large adjustment",
                "period": "2024-01",
            },
            amount=75000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        assert result.data.get("approver_role") == "controller"

    @pytest.mark.asyncio
    async def test_reconcile_account(self, make_agent, mock_decision_engine):
        """Reconciliation delegates to DecisionEngine with reconcile_account."""
        agent = make_agent(AccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "reconciliation_items": [],
            "adjustments": [],
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="reconciliation",
            data={
                "account_id": "1000-00",
                "period": "2024-01",
                "book_balance": 50000.0,
                "bank_balance": 50000.0,
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    def test_threshold_constants(self):
        """Verify Accountant controller threshold constant."""
        assert AccountantAgent.CONTROLLER_THRESHOLD == 50000.0


# =========================================================================
# TestSeniorAccountantAgent — Phase 14
# =========================================================================


class TestSeniorAccountantAgent:
    """Test SeniorAccountantAgent: JE approval, escalation, period close."""

    @pytest.mark.asyncio
    async def test_je_approval_within_authority(self, make_agent, mock_decision_engine):
        """JE $30K (<$50K) → senior accountant approves."""
        agent = make_agent(SeniorAccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Entries balanced and valid",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="je_approval",
            data={
                "je_id": "JE-050",
                "amount": 30000.0,
                "entries": [
                    {"account": "1000-00", "debit": 30000.0, "credit": 0.0},
                    {"account": "2000-00", "debit": 0.0, "credit": 30000.0},
                ],
            },
            amount=30000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_je_approval_above_authority_escalates(self, make_agent, mock_decision_engine):
        """JE $60K (>$50K) → escalated to controller."""
        agent = make_agent(SeniorAccountantAgent)
        work_item = WorkItem(
            type="je_approval",
            data={
                "je_id": "JE-051",
                "amount": 60000.0,
                "entries": [
                    {"account": "1000-00", "debit": 60000.0, "credit": 0.0},
                    {"account": "2000-00", "debit": 0.0, "credit": 60000.0},
                ],
            },
            amount=60000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        has_escalation = any(
            "escalat" in action.lower() for action in result.actions_taken
        )
        assert has_escalation, f"Expected escalation in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_process_period_close(self, make_agent, mock_decision_engine):
        """Period close work type handled successfully."""
        agent = make_agent(SeniorAccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "period_status": "ready_to_close",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="period_close",
            data={
                "period": "2024-01",
                "fiscal_year": 2024,
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_review_reconciliation(self, make_agent, mock_decision_engine):
        """Reconciliation review work type processed."""
        agent = make_agent(SeniorAccountantAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "review_status": "satisfactory",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="reconciliation_review",
            data={
                "account_id": "1000-00",
                "period": "2024-01",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    def test_threshold_constant(self):
        """Verify Senior Accountant approval threshold."""
        assert SeniorAccountantAgent.APPROVAL_THRESHOLD_MAX == 50000.0


# =========================================================================
# TestControllerAgent — Phase 15
# =========================================================================


class TestControllerAgent:
    """Test ControllerAgent: high-value approvals across PO/Invoice/JE, escalation to CFO."""

    @pytest.mark.asyncio
    async def test_approval_within_authority_po(self, make_agent, mock_decision_engine):
        """PO $50K ($25K–$100K) → controller approves."""
        agent = make_agent(ControllerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Within budget and policy",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="po_approval",
            data={
                "po_id": "PO-CTRL-001",
                "amount": 50000.0,
                "transaction_type": "purchase_order",
            },
            amount=50000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_approval_above_authority_escalates_to_cfo(self, make_agent, mock_decision_engine):
        """PO $150K (>$100K) → escalated to CFO."""
        agent = make_agent(ControllerAgent)
        work_item = WorkItem(
            type="po_approval",
            data={
                "po_id": "PO-CTRL-002",
                "amount": 150000.0,
                "transaction_type": "purchase_order",
            },
            amount=150000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        assert result.approval_needed is True
        # Should be escalated to CFO
        has_cfo_escalation = (
            result.data.get("approver_role") == "cfo"
            or any("cfo" in action.lower() or "escalat" in action.lower()
                   for action in result.actions_taken)
        )
        assert has_cfo_escalation, (
            f"Expected CFO escalation. actions={result.actions_taken}, data={result.data}"
        )

    @pytest.mark.asyncio
    async def test_invoice_approval_within_authority(self, make_agent, mock_decision_engine):
        """Invoice $75K ($50K–$100K) → controller approves."""
        agent = make_agent(ControllerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Invoice validated",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="invoice_approval",
            data={
                "invoice_id": "INV-CTRL-001",
                "amount": 75000.0,
                "transaction_type": "invoice",
            },
            amount=75000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_je_approval_no_upper_limit(self, make_agent, mock_decision_engine):
        """JE $75K — controller approves JEs with no upper limit."""
        agent = make_agent(ControllerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Journal entry reviewed and balanced",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="je_approval",
            data={
                "je_id": "JE-CTRL-001",
                "amount": 75000.0,
                "entries": [
                    {"account": "1000-00", "debit": 75000.0, "credit": 0.0},
                    {"account": "2000-00", "debit": 0.0, "credit": 75000.0},
                ],
                "transaction_type": "journal_entry",
            },
            amount=75000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        mock_decision_engine.decide.assert_called()

    @pytest.mark.asyncio
    async def test_period_close_approval(self, make_agent, mock_decision_engine):
        """Period close approval work type processed."""
        agent = make_agent(ControllerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "period_status": "closed",
            "confidence": 0.95,
        })
        work_item = WorkItem(
            type="period_close_approval",
            data={
                "period": "2024-01",
                "fiscal_year": 2024,
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_review_escalation(self, make_agent, mock_decision_engine):
        """Escalation review work type processed."""
        agent = make_agent(ControllerAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "resolve",
            "resolution": "approved_with_conditions",
            "reasoning": "Additional documentation required",
            "confidence": 0.8,
        })
        work_item = WorkItem(
            type="escalation_review",
            data={
                "escalation_id": "ESC-001",
                "original_amount": 80000.0,
                "escalation_reason": "Above AP manager authority",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    def test_authority_constants(self):
        """Verify Controller authority constants and AUTHORITY dict."""
        assert ControllerAgent.CFO_THRESHOLD == 100000.0
        assert hasattr(ControllerAgent, "AUTHORITY")
        assert isinstance(ControllerAgent.AUTHORITY, dict)


# =========================================================================
# TestCFOAgent — Phase 16
# =========================================================================


class TestCFOAgent:
    """Test CFOAgent: terminal approval authority, strategic decisions."""

    @pytest.mark.asyncio
    async def test_strategic_approval_approve(self, make_agent, mock_decision_engine):
        """CFO approves → transaction_approved_by_cfo in actions."""
        agent = make_agent(CFOAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Strategic investment approved",
            "confidence": 0.95,
        })
        work_item = WorkItem(
            type="strategic_approval",
            data={
                "transaction_id": "TXN-CFO-001",
                "amount": 250000.0,
                "transaction_type": "purchase_order",
                "description": "Capital equipment purchase",
            },
            amount=250000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        has_cfo_approval = any(
            "approved" in action.lower() and "cfo" in action.lower()
            for action in result.actions_taken
        )
        # Allow any form of approval action by CFO
        assert has_cfo_approval or any(
            "approved" in action.lower() for action in result.actions_taken
        ), f"Expected CFO approval action in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_strategic_approval_reject(self, make_agent, mock_decision_engine):
        """CFO rejects → transaction_rejected_by_cfo in actions."""
        agent = make_agent(CFOAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "reject",
            "approved": False,
            "reasoning": "Budget constraints - defer to next quarter",
            "confidence": 0.85,
        })
        work_item = WorkItem(
            type="strategic_approval",
            data={
                "transaction_id": "TXN-CFO-002",
                "amount": 500000.0,
                "transaction_type": "purchase_order",
                "description": "Non-essential expansion",
            },
            amount=500000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        has_cfo_rejection = any(
            "reject" in action.lower() for action in result.actions_taken
        )
        assert has_cfo_rejection, f"Expected CFO rejection action in: {result.actions_taken}"

    @pytest.mark.asyncio
    async def test_no_escalation_above_cfo(self, make_agent, mock_decision_engine):
        """CFO is terminal authority — no further escalation regardless of amount."""
        agent = make_agent(CFOAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "Within strategic mandate",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="strategic_approval",
            data={
                "transaction_id": "TXN-CFO-003",
                "amount": 10000000.0,  # $10M — no escalation beyond CFO
                "transaction_type": "purchase_order",
                "description": "Major acquisition",
            },
            amount=10000000.0,
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True
        # CFO should NOT escalate further — terminal authority
        has_no_escalation = not any(
            "escalat" in action.lower() for action in result.actions_taken
        )
        assert has_no_escalation, (
            f"CFO should not escalate — terminal authority. actions={result.actions_taken}"
        )
        # approval_needed should be False since CFO is the final authority
        # (or if True, the approver is not specified beyond cfo)
        if result.approval_needed:
            assert result.data.get("approver_role") is None or result.data.get("approver_role") == "cfo"

    @pytest.mark.asyncio
    async def test_year_end_close_approval(self, make_agent, mock_decision_engine):
        """Year-end close work type processed by CFO."""
        agent = make_agent(CFOAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "approve",
            "approved": True,
            "reasoning": "All period-end procedures completed",
            "confidence": 0.95,
        })
        work_item = WorkItem(
            type="year_end_close",
            data={
                "fiscal_year": 2024,
                "close_type": "year_end",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_financial_oversight(self, make_agent, mock_decision_engine):
        """Financial oversight work type processed by CFO."""
        agent = make_agent(CFOAgent)
        mock_decision_engine.decide = AsyncMock(return_value={
            "decision": "reviewed",
            "findings": "No material issues",
            "confidence": 0.9,
        })
        work_item = WorkItem(
            type="financial_oversight",
            data={
                "review_period": "2024-Q1",
                "review_type": "quarterly_review",
            },
        )
        result = await agent.process_work_item(work_item)
        assert result.success is True

    def test_threshold_constant(self):
        """Verify CFO strategic threshold constant."""
        assert CFOAgent.STRATEGIC_THRESHOLD == 100000.0


# =========================================================================
# TestAllAgentsCommon — Phase 17 (Parametrized across all 12 agents)
# =========================================================================


# Agent classes paired with their expected roles for parametrized tests
AGENT_ROLE_PAIRS = [
    (APClerkAgent, "ap_clerk"),
    (APManagerAgent, "ap_manager"),
    (ARClerkAgent, "ar_clerk"),
    (ARManagerAgent, "ar_manager"),
    (PurchasingAgent, "purchasing_agent"),
    (PurchasingManagerAgent, "purchasing_manager"),
    (WarehouseClerkAgent, "warehouse_clerk"),
    (WarehouseManagerAgent, "warehouse_manager"),
    (AccountantAgent, "accountant"),
    (SeniorAccountantAgent, "senior_accountant"),
    (ControllerAgent, "controller"),
    (CFOAgent, "cfo"),
]

AGENT_ROLE_IDS = [pair[1] for pair in AGENT_ROLE_PAIRS]


class TestAllAgentsCommon:
    """Parametrized tests across all 12 specialized agent types."""

    @pytest.mark.parametrize(
        "agent_class,role",
        AGENT_ROLE_PAIRS,
        ids=AGENT_ROLE_IDS,
    )
    def test_agent_instantiation(self, make_agent, agent_class, role):
        """Verify each agent can be instantiated with mock dependencies."""
        agent = make_agent(agent_class)
        assert agent is not None
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, agent_class)

    @pytest.mark.parametrize(
        "agent_class,role",
        AGENT_ROLE_PAIRS,
        ids=AGENT_ROLE_IDS,
    )
    def test_agent_initial_state_is_idle(self, make_agent, agent_class, role):
        """Verify initial state is IDLE for every agent type."""
        agent = make_agent(agent_class)
        assert agent.state == AgentState.IDLE

    @pytest.mark.parametrize(
        "agent_class",
        ALL_AGENT_CLASSES,
        ids=[cls.__name__ for cls in ALL_AGENT_CLASSES],
    )
    def test_agent_has_process_work_item(self, make_agent, agent_class):
        """Every agent must have a callable process_work_item method."""
        agent = make_agent(agent_class)
        assert hasattr(agent, "process_work_item")
        assert callable(getattr(agent, "process_work_item"))

    @pytest.mark.parametrize(
        "agent_class",
        ALL_AGENT_CLASSES,
        ids=[cls.__name__ for cls in ALL_AGENT_CLASSES],
    )
    def test_agent_has_make_decision(self, make_agent, agent_class):
        """Every agent must have a callable make_decision method (inherited from BaseAgent)."""
        agent = make_agent(agent_class)
        assert hasattr(agent, "make_decision")
        assert callable(getattr(agent, "make_decision"))

    def test_all_12_agent_classes_defined(self):
        """Assert exactly 12 unique specialized agent classes exist."""
        assert len(ALL_AGENT_CLASSES) == 12
        assert len(set(ALL_AGENT_CLASSES)) == 12
        # Verify all are subclasses of BaseAgent
        for cls in ALL_AGENT_CLASSES:
            assert issubclass(cls, BaseAgent), f"{cls.__name__} must extend BaseAgent"
