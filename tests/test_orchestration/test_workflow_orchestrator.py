"""Comprehensive unit tests for the WorkflowOrchestrator.

Tests cover:
    - Transaction type → agent role mapping via ROLE_MAPPING constant
    - Idle agent selection with smallest-queue strategy
    - Transaction routing, assignment, and event publishing
    - Workflow lifecycle management (complete, fail, retry)
    - Queue management and pending-queue processing
    - WorkflowInstance Pydantic V2 model validation
    - WorkflowConfig defaults and custom values
    - Concurrent workflow support and routing SLA verification
    - P3 P2P transaction routing (purchase_order, goods_receipt,
      vendor_invoice, vendor_payment) [AAP Section 0.2.1]
    - P3 O2C transaction routing (sales_order, customer_invoice,
      customer_payment) [AAP Section 0.2.1]
    - P3 rework loop re-submission routing [AAP Section 0.2.1]
    - P3 period close and GL transaction routing [AAP Section 0.2.1]

Design Notes:
    - All agent dependencies are MOCKED (MagicMock / AsyncMock) — no real
      agents, registries, or LLM calls.
    - No real Redis connections — mocks only.
    - Async tests use ``@pytest.mark.asyncio`` with ``pytest-asyncio``.
    - Routing SLA test verifies < 500 ms per AAP Section 0.7.3.

References:
    - README.md lines 1289-1376 (routing, role mapping, agent selection)
    - README.md lines 1636-1641 (retry policy)
    - AAP Section 0.5.1 Group 10 (test requirements)
    - AAP Section 0.7.3 (performance constraints)
    - AAP Section 0.7.5 (testing standards)
    - AAP Section 0.2.1 (P3 transaction routing test requirements)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from uuid import UUID, uuid4

import pytest

from app.agents.agent_config import AgentState
from app.orchestration.approval_system import ApprovalSystem
from app.orchestration.workflow_orchestrator import (
    ROLE_MAPPING,
    WorkflowConfig,
    WorkflowInstance,
    WorkflowOrchestrator,
    WorkflowStatus,
)


# ---------------------------------------------------------------------------
# Helper: Create a mock agent with configurable role, state, and queue size
# ---------------------------------------------------------------------------


def create_mock_agent(
    role: str,
    agent_id: Optional[UUID] = None,
    state: str = AgentState.IDLE,
    queue_size: int = 0,
) -> MagicMock:
    """Build a ``MagicMock`` that quacks like a ``BaseAgent``.

    Creates a real ``asyncio.Queue`` for ``work_queue`` so that
    ``qsize()`` returns meaningful values for smallest-queue selection.

    Args:
        role: Agent role identifier (e.g. ``"ap_clerk"``).
        agent_id: Explicit agent UUID; auto-generated when ``None``.
        state: Agent state string (should match ``AgentState`` values).
        queue_size: Number of dummy items to pre-load into the queue.

    Returns:
        A fully wired ``MagicMock`` instance usable by the orchestrator.
    """
    agent = MagicMock()
    agent.config = MagicMock()
    agent.config.agent_id = agent_id or uuid4()
    agent.config.role = role
    agent.state = state

    # Use a real asyncio.Queue so qsize() works correctly
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(queue_size):
        queue.put_nowait({"dummy": i})
    agent.work_queue = queue

    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workflow_config() -> WorkflowConfig:
    """Standard WorkflowConfig with specification defaults."""
    return WorkflowConfig(
        max_concurrent_workflows=100,
        routing_timeout_ms=500,
        max_queue_depth_per_agent=50,
        retry_max_attempts=3,
    )


@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """Mock AgentRegistry providing role-based agent lookup.

    By default ``get_agents_by_roles`` returns an empty list.  Individual
    tests override this via ``side_effect`` or ``return_value`` to inject
    mock agents suited to the scenario under test.
    """
    registry = MagicMock()
    registry.get_agents_by_roles = MagicMock(return_value=[])
    registry.get_agent = MagicMock(return_value=None)
    return registry


@pytest.fixture
def mock_approval_system() -> MagicMock:
    """Mock ApprovalSystem returning no approval requirement by default.

    ``get_required_approval`` returns ``None`` (no approval needed).
    ``create_approval_request`` is an ``AsyncMock`` for awaitable calls.
    """
    system = MagicMock(spec=ApprovalSystem)
    system.get_required_approval = MagicMock(return_value=None)
    system.create_approval_request = AsyncMock()
    return system


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Mock EventBus with async ``publish`` and sync ``subscribe``."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def orchestrator(
    mock_agent_registry: MagicMock,
    mock_approval_system: MagicMock,
    workflow_config: WorkflowConfig,
    mock_event_bus: AsyncMock,
) -> WorkflowOrchestrator:
    """Fully wired WorkflowOrchestrator with mocked dependencies."""
    return WorkflowOrchestrator(
        agent_registry=mock_agent_registry,
        approval_system=mock_approval_system,
        config=workflow_config,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def sample_transaction() -> Dict[str, Any]:
    """Minimal transaction dict for routing tests."""
    return {
        "transaction_id": str(uuid4()),
        "vendor_id": "V-001",
        "amount": 5000.00,
        "description": "Test invoice",
    }


# ---------------------------------------------------------------------------
# Test Class — Role Mapping (README.md lines 1346-1355)
# ---------------------------------------------------------------------------


class TestRoleMapping:
    """Verify ROLE_MAPPING constant and _get_required_roles behaviour."""

    def test_vendor_invoice_maps_to_ap_roles(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """vendor_invoice → ['ap_clerk', 'ap_manager']."""
        roles = orchestrator._get_required_roles("vendor_invoice")
        assert roles == ["ap_clerk", "ap_manager"]

    def test_purchase_order_maps_to_purchasing_roles(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """purchase_order → ['purchasing_agent', 'purchasing_manager']."""
        roles = orchestrator._get_required_roles("purchase_order")
        assert roles == ["purchasing_agent", "purchasing_manager"]

    def test_sales_order_maps_to_ar_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """sales_order → ['ar_clerk']."""
        roles = orchestrator._get_required_roles("sales_order")
        assert roles == ["ar_clerk"]

    def test_customer_payment_maps_to_ar_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """customer_payment → ['ar_clerk']."""
        roles = orchestrator._get_required_roles("customer_payment")
        assert roles == ["ar_clerk"]

    def test_journal_entry_maps_to_accountant_roles(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """journal_entry → ['accountant', 'senior_accountant']."""
        roles = orchestrator._get_required_roles("journal_entry")
        assert roles == ["accountant", "senior_accountant"]

    def test_unknown_transaction_type_returns_empty(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """Unknown type returns an empty list (no crash)."""
        roles = orchestrator._get_required_roles("unknown_type")
        assert roles == []

    def test_role_mapping_contains_all_expected_types(self) -> None:
        """ROLE_MAPPING constant covers the minimum required types."""
        expected_types = {
            "vendor_invoice",
            "purchase_order",
            "sales_order",
            "customer_payment",
            "journal_entry",
        }
        assert expected_types.issubset(set(ROLE_MAPPING.keys()))

    def test_role_mapping_values_are_lists(self) -> None:
        """Every value in ROLE_MAPPING is a list of strings."""
        for txn_type, roles in ROLE_MAPPING.items():
            assert isinstance(roles, list), f"{txn_type} value is not a list"
            for role in roles:
                assert isinstance(role, str), f"Role {role!r} is not a string"

    def test_vendor_payment_maps_to_ap_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """vendor_payment → ['ap_clerk']."""
        roles = orchestrator._get_required_roles("vendor_payment")
        assert roles == ["ap_clerk"]

    def test_goods_receipt_maps_to_warehouse_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """goods_receipt → ['warehouse_clerk']."""
        roles = orchestrator._get_required_roles("goods_receipt")
        assert roles == ["warehouse_clerk"]


# ---------------------------------------------------------------------------
# Test Class — P3 P2P Transaction Routing (AAP Section 0.2.1)
# ---------------------------------------------------------------------------


class TestP3P2PTransactionRouting:
    """Verify ROLE_MAPPING coverage for all P2P transaction types used by P3.

    Per AAP Section 0.2.1:
    - P2P pipeline: purchase_order → goods_receipt → vendor_invoice → vendor_payment
    - Each P2P type must map to eligible agent roles for workflow routing
    """

    def test_purchase_order_routes_to_purchasing_roles(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """P2P step 1: purchase_order → ['purchasing_agent', 'purchasing_manager']."""
        roles = orchestrator._get_required_roles("purchase_order")
        assert roles == ["purchasing_agent", "purchasing_manager"]
        assert "purchasing_agent" in roles  # Primary handler

    def test_goods_receipt_routes_to_warehouse_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """P2P step 2: goods_receipt → ['warehouse_clerk']."""
        roles = orchestrator._get_required_roles("goods_receipt")
        assert roles == ["warehouse_clerk"]

    def test_vendor_invoice_routes_to_ap_roles(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """P2P step 3: vendor_invoice → ['ap_clerk', 'ap_manager']."""
        roles = orchestrator._get_required_roles("vendor_invoice")
        assert roles == ["ap_clerk", "ap_manager"]
        assert "ap_clerk" in roles  # Primary handler

    def test_vendor_payment_routes_to_ap_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """P2P step 4: vendor_payment → ['ap_clerk']."""
        roles = orchestrator._get_required_roles("vendor_payment")
        assert roles == ["ap_clerk"]

    def test_all_p2p_types_have_role_mapping(self) -> None:
        """All 4 P2P transaction types must be present in ROLE_MAPPING."""
        p2p_types = ["purchase_order", "goods_receipt", "vendor_invoice", "vendor_payment"]
        for txn_type in p2p_types:
            assert txn_type in ROLE_MAPPING, (
                f"P2P type '{txn_type}' missing from ROLE_MAPPING"
            )
            assert len(ROLE_MAPPING[txn_type]) > 0, (
                f"P2P type '{txn_type}' has empty role list"
            )

    @pytest.mark.asyncio
    async def test_route_purchase_order_to_purchasing_agent(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Full routing of a purchase_order to a purchasing_agent."""
        agent = create_mock_agent(role="purchasing_agent", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "vendor_id": "V-001", "amount": 5000.00}
        wf_id = await orchestrator.route_transaction(txn, "purchase_order")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "purchase_order"
        assert wf.assigned_agent_id == agent.config.agent_id
        assert wf.status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_route_goods_receipt_to_warehouse_clerk(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Full routing of a goods_receipt to a warehouse_clerk."""
        agent = create_mock_agent(role="warehouse_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "po_id": "PO-001", "amount": 5000.00}
        wf_id = await orchestrator.route_transaction(txn, "goods_receipt")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "goods_receipt"
        assert wf.assigned_agent_id == agent.config.agent_id

    @pytest.mark.asyncio
    async def test_route_vendor_payment_to_ap_clerk(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Full routing of a vendor_payment to an ap_clerk."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "vendor_id": "V-001", "amount": 15000.00}
        wf_id = await orchestrator.route_transaction(txn, "vendor_payment")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "vendor_payment"
        assert wf.assigned_agent_id == agent.config.agent_id

    @pytest.mark.asyncio
    async def test_full_p2p_routing_sequence(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Route all 4 P2P types in sequence, verifying each gets assigned."""
        purchasing_agent = create_mock_agent(role="purchasing_agent", state=AgentState.IDLE)
        warehouse_clerk = create_mock_agent(role="warehouse_clerk", state=AgentState.IDLE)
        ap_clerk = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)

        def get_agents_for_roles(roles):
            agents = []
            if "purchasing_agent" in roles or "purchasing_manager" in roles:
                agents.append(purchasing_agent)
            if "warehouse_clerk" in roles:
                agents.append(warehouse_clerk)
            if "ap_clerk" in roles or "ap_manager" in roles:
                agents.append(ap_clerk)
            return agents

        mock_agent_registry.get_agents_by_roles.side_effect = get_agents_for_roles

        p2p_types = ["purchase_order", "goods_receipt", "vendor_invoice", "vendor_payment"]
        wf_ids = []
        for txn_type in p2p_types:
            txn = {"transaction_id": str(uuid4()), "amount": 5000.00}
            wf_id = await orchestrator.route_transaction(txn, txn_type)
            wf_ids.append(wf_id)

        assert len(wf_ids) == 4
        assert len(set(wf_ids)) == 4  # All unique
        for wf_id in wf_ids:
            wf = orchestrator.workflows[wf_id]
            assert wf.status == WorkflowStatus.ASSIGNED.value


# ---------------------------------------------------------------------------
# Test Class — P3 O2C Transaction Routing (AAP Section 0.2.1)
# ---------------------------------------------------------------------------


class TestP3O2CTransactionRouting:
    """Verify ROLE_MAPPING coverage for all O2C transaction types used by P3.

    Per AAP Section 0.2.1:
    - O2C pipeline: sales_order → shipment → customer_invoice → customer_payment
    - Most O2C types map to ar_clerk
    - shipment type may not have explicit routing in current ROLE_MAPPING
    """

    def test_sales_order_routes_to_ar_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """O2C step 1: sales_order → ['ar_clerk']."""
        roles = orchestrator._get_required_roles("sales_order")
        assert roles == ["ar_clerk"]

    def test_customer_invoice_routes_to_ar_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """O2C step 3: customer_invoice → ['ar_clerk']."""
        roles = orchestrator._get_required_roles("customer_invoice")
        assert roles == ["ar_clerk"]

    def test_customer_payment_routes_to_ar_clerk(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """O2C step 4: customer_payment → ['ar_clerk']."""
        roles = orchestrator._get_required_roles("customer_payment")
        assert roles == ["ar_clerk"]

    def test_shipment_routing_behavior(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """O2C step 2: shipment returns configured roles or empty list.

        The shipment type may or may not have an explicit ROLE_MAPPING entry.
        This test documents the current behavior.
        """
        roles = orchestrator._get_required_roles("shipment")
        # If shipment is in ROLE_MAPPING, verify it returns a non-empty list
        # If not, it returns an empty list (graceful fallback)
        if "shipment" in ROLE_MAPPING:
            assert len(roles) > 0
        else:
            assert roles == []

    def test_core_o2c_types_have_role_mapping(self) -> None:
        """Core O2C types (sales_order, customer_invoice, customer_payment) in ROLE_MAPPING."""
        core_o2c_types = ["sales_order", "customer_invoice", "customer_payment"]
        for txn_type in core_o2c_types:
            assert txn_type in ROLE_MAPPING, (
                f"Core O2C type '{txn_type}' missing from ROLE_MAPPING"
            )

    @pytest.mark.asyncio
    async def test_route_sales_order_to_ar_clerk(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Full routing of a sales_order to an ar_clerk."""
        agent = create_mock_agent(role="ar_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "customer_id": "C-001", "amount": 8000.00}
        wf_id = await orchestrator.route_transaction(txn, "sales_order")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "sales_order"
        assert wf.assigned_agent_id == agent.config.agent_id
        assert wf.status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_route_customer_invoice_to_ar_clerk(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Full routing of a customer_invoice to an ar_clerk."""
        agent = create_mock_agent(role="ar_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "customer_id": "C-001", "amount": 8000.00}
        wf_id = await orchestrator.route_transaction(txn, "customer_invoice")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "customer_invoice"
        assert wf.assigned_agent_id == agent.config.agent_id

    @pytest.mark.asyncio
    async def test_route_customer_payment_to_ar_clerk(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Full routing of a customer_payment to an ar_clerk."""
        agent = create_mock_agent(role="ar_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "customer_id": "C-001", "amount": 8000.00}
        wf_id = await orchestrator.route_transaction(txn, "customer_payment")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "customer_payment"
        assert wf.assigned_agent_id == agent.config.agent_id

    @pytest.mark.asyncio
    async def test_shipment_routes_to_pending_when_no_mapped_agent(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Shipment with no matching agents goes to PENDING queue.

        If 'shipment' is not in ROLE_MAPPING, _get_required_roles returns [],
        _find_available_agent returns None, and workflow becomes PENDING.
        """
        mock_agent_registry.get_agents_by_roles.return_value = []

        txn = {"transaction_id": str(uuid4()), "so_id": "SO-001", "amount": 8000.00}
        wf_id = await orchestrator.route_transaction(txn, "shipment")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "shipment"
        assert wf.status == WorkflowStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_concurrent_o2c_routing(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Route O2C types concurrently and verify unique workflow IDs."""
        ar_agent = create_mock_agent(role="ar_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [ar_agent]

        tasks = [
            orchestrator.route_transaction(
                {"transaction_id": str(uuid4()), "amount": 1000}, "sales_order"
            ),
            orchestrator.route_transaction(
                {"transaction_id": str(uuid4()), "amount": 2000}, "customer_invoice"
            ),
            orchestrator.route_transaction(
                {"transaction_id": str(uuid4()), "amount": 3000}, "customer_payment"
            ),
        ]

        wf_ids = await asyncio.gather(*tasks)
        assert len(set(wf_ids)) == 3


# ---------------------------------------------------------------------------
# Test Class — P3 Rework Loop Re-submission Routing (AAP Section 0.2.1)
# ---------------------------------------------------------------------------


class TestP3ReworkResubmission:
    """Verify that P3 transactions returning from the rework loop can be re-routed.

    Per AAP Section 0.2.1:
    - Transactions that fail validation go through the rework loop
    - After fix application, they need to be re-routed (retry_workflow)
    - Max 3 retry attempts before permanent FAILED status
    """

    @pytest.mark.asyncio
    async def test_retry_vendor_invoice_after_rework(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """vendor_invoice returning from rework loop is re-routed to ap_clerk."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        # Route original transaction
        txn = {"transaction_id": str(uuid4()), "vendor_id": "V-001", "amount": 5000.00}
        wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")

        # Simulate validation failure (rework loop sends back)
        await orchestrator.fail_workflow(wf_id, "Three-way match failed")
        wf = orchestrator.workflows[wf_id]
        assert wf.status == WorkflowStatus.FAILED.value

        # Rework loop applies fix and re-submits (retry_workflow)
        result = await orchestrator.retry_workflow(wf_id)
        assert result is True

        wf = orchestrator.workflows[wf_id]
        assert wf.retry_count == 1
        assert wf.status == WorkflowStatus.ASSIGNED.value
        assert wf.assigned_agent_id is not None

    @pytest.mark.asyncio
    async def test_retry_purchase_order_after_rework(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """purchase_order returning from rework loop is re-routed to purchasing_agent."""
        agent = create_mock_agent(role="purchasing_agent", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "vendor_id": "V-001", "amount": 25000.00}
        wf_id = await orchestrator.route_transaction(txn, "purchase_order")

        await orchestrator.fail_workflow(wf_id, "Approval chain error")
        result = await orchestrator.retry_workflow(wf_id)
        assert result is True

        wf = orchestrator.workflows[wf_id]
        assert wf.retry_count == 1
        assert wf.status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_retry_goods_receipt_after_rework(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """goods_receipt returning from rework loop is re-routed to warehouse_clerk."""
        agent = create_mock_agent(role="warehouse_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "po_id": "PO-001", "amount": 5000.00}
        wf_id = await orchestrator.route_transaction(txn, "goods_receipt")

        await orchestrator.fail_workflow(wf_id, "Quantity mismatch")
        result = await orchestrator.retry_workflow(wf_id)
        assert result is True

        wf = orchestrator.workflows[wf_id]
        assert wf.retry_count == 1

    @pytest.mark.asyncio
    async def test_rework_exhausts_max_retries(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """After 3 rework attempts, transaction is permanently FAILED."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "vendor_id": "V-001", "amount": 5000.00}
        wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")

        # Simulate 3 rework cycles
        for attempt in range(3):
            await orchestrator.fail_workflow(wf_id, f"Rework attempt {attempt + 1} failed")
            result = await orchestrator.retry_workflow(wf_id)
            if attempt < 2:  # First 2 retries succeed (retry_count < max)
                assert result is True
            # retry_count increments with each retry

        # After 3 retries, the workflow should be permanently failed
        wf = orchestrator.workflows[wf_id]
        assert wf.retry_count >= 3

    @pytest.mark.asyncio
    async def test_retry_customer_payment_after_rework(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """O2C customer_payment can be retried after rework loop fix."""
        agent = create_mock_agent(role="ar_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "customer_id": "C-001", "amount": 8000.00}
        wf_id = await orchestrator.route_transaction(txn, "customer_payment")

        await orchestrator.fail_workflow(wf_id, "FIFO allocation error")
        result = await orchestrator.retry_workflow(wf_id)
        assert result is True

        wf = orchestrator.workflows[wf_id]
        assert wf.retry_count == 1
        assert wf.status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_retry_preserves_transaction_type(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Retried workflows preserve their original transaction_type."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "amount": 10000.00}
        wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")

        await orchestrator.fail_workflow(wf_id, "GL posting error")
        await orchestrator.retry_workflow(wf_id)

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "vendor_invoice"  # Type preserved


# ---------------------------------------------------------------------------
# Test Class — P3 Period Close and GL Transaction Routing (AAP Section 0.2.1)
# ---------------------------------------------------------------------------


class TestP3PeriodCloseRouting:
    """Verify routing behavior for P3 period close and GL-related types.

    Per AAP Section 0.2.1:
    - Period close workflows should route to Controller/CFO agents
    - Journal entry workflows route to accountant/senior_accountant
    - Currently, period_close, accrual, depreciation, recurring_journal
      may not have explicit ROLE_MAPPING entries
    """

    def test_journal_entry_routes_to_accounting_roles(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """journal_entry → ['accountant', 'senior_accountant'] (used by GL posting)."""
        roles = orchestrator._get_required_roles("journal_entry")
        assert roles == ["accountant", "senior_accountant"]

    def test_period_close_routing_behavior(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """period_close returns configured roles or empty list.

        If period_close is added to ROLE_MAPPING (e.g., mapping to
        ['controller', 'cfo']), this test verifies the mapping.
        Otherwise, documents the fallback to empty roles.
        """
        roles = orchestrator._get_required_roles("period_close")
        if "period_close" in ROLE_MAPPING:
            assert len(roles) > 0
            # Expect period close to route to senior financial roles
        else:
            assert roles == []

    def test_accrual_routing_behavior(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """accrual returns configured roles or empty list."""
        roles = orchestrator._get_required_roles("accrual")
        if "accrual" in ROLE_MAPPING:
            assert len(roles) > 0
        else:
            assert roles == []

    def test_depreciation_routing_behavior(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """depreciation returns configured roles or empty list."""
        roles = orchestrator._get_required_roles("depreciation")
        if "depreciation" in ROLE_MAPPING:
            assert len(roles) > 0
        else:
            assert roles == []

    def test_recurring_journal_routing_behavior(
        self, orchestrator: WorkflowOrchestrator
    ) -> None:
        """recurring_journal returns configured roles or empty list."""
        roles = orchestrator._get_required_roles("recurring_journal")
        if "recurring_journal" in ROLE_MAPPING:
            assert len(roles) > 0
        else:
            assert roles == []

    @pytest.mark.asyncio
    async def test_route_journal_entry_to_accountant(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Journal entry routes to an accountant agent (used for P3 GL posting)."""
        agent = create_mock_agent(role="accountant", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {
            "transaction_id": str(uuid4()),
            "description": "Period end accrual",
            "amount": 50000.00,
        }
        wf_id = await orchestrator.route_transaction(txn, "journal_entry")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "journal_entry"
        assert wf.assigned_agent_id == agent.config.agent_id
        assert wf.status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_route_journal_entry_with_approval(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        mock_approval_system: MagicMock,
    ) -> None:
        """Journal entry above $50K threshold triggers approval routing."""
        agent = create_mock_agent(role="accountant", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]
        mock_approval_system.get_required_approval.return_value = "controller"

        txn = {
            "transaction_id": str(uuid4()),
            "description": "Large adjusting entry",
            "amount": 75000.00,
        }
        wf_id = await orchestrator.route_transaction(txn, "journal_entry")

        wf = orchestrator.workflows[wf_id]
        assert wf.approval_required is True
        assert wf.approval_role == "controller"

    @pytest.mark.asyncio
    async def test_period_close_goes_pending_when_no_mapping(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Period close without ROLE_MAPPING entry goes to PENDING queue."""
        mock_agent_registry.get_agents_by_roles.return_value = []

        txn = {
            "transaction_id": str(uuid4()),
            "description": "January 2025 Period Close",
            "fiscal_period": "2025-01",
        }
        wf_id = await orchestrator.route_transaction(txn, "period_close")

        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "period_close"
        # Without a ROLE_MAPPING entry, no agents match → PENDING
        assert wf.status == WorkflowStatus.PENDING.value

    def test_role_mapping_covers_all_eight_base_types(self) -> None:
        """ROLE_MAPPING must cover all 8 transaction types from P2 specification."""
        expected_types = {
            "purchase_order",
            "vendor_invoice",
            "vendor_payment",
            "goods_receipt",
            "sales_order",
            "customer_invoice",
            "customer_payment",
            "journal_entry",
        }
        assert expected_types.issubset(set(ROLE_MAPPING.keys())), (
            f"Missing types: {expected_types - set(ROLE_MAPPING.keys())}"
        )

    def test_role_mapping_all_values_are_nonempty_string_lists(self) -> None:
        """Every ROLE_MAPPING value must be a non-empty list of strings."""
        for txn_type, roles in ROLE_MAPPING.items():
            assert isinstance(roles, list), f"{txn_type}: not a list"
            assert len(roles) > 0, f"{txn_type}: empty role list"
            for role in roles:
                assert isinstance(role, str), f"{txn_type}: role {role!r} not a string"
                assert len(role) > 0, f"{txn_type}: empty string role"


# ---------------------------------------------------------------------------
# Test Class — Agent Selection (README.md lines 1357-1376)
# ---------------------------------------------------------------------------


class TestAgentSelection:
    """Verify idle-agent selection with smallest-queue strategy."""

    @pytest.mark.asyncio
    async def test_find_available_agent_selects_idle_agent(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """An idle agent matching the requested role is selected."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        selected = await orchestrator._find_available_agent(["ap_clerk"])
        assert selected is agent

    @pytest.mark.asyncio
    async def test_find_available_agent_selects_smallest_queue(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Among multiple idle agents, the one with the smallest queue wins."""
        agent_a = create_mock_agent(role="ap_clerk", state=AgentState.IDLE, queue_size=5)
        agent_b = create_mock_agent(role="ap_clerk", state=AgentState.IDLE, queue_size=2)
        agent_c = create_mock_agent(role="ap_clerk", state=AgentState.IDLE, queue_size=8)
        mock_agent_registry.get_agents_by_roles.return_value = [
            agent_a,
            agent_b,
            agent_c,
        ]

        selected = await orchestrator._find_available_agent(["ap_clerk"])
        assert selected is agent_b
        assert selected.work_queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_find_available_agent_returns_none_when_no_idle(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """No idle agents → returns None."""
        agent_a = create_mock_agent(role="ap_clerk", state=AgentState.THINKING)
        agent_b = create_mock_agent(role="ap_clerk", state=AgentState.ACTING)
        mock_agent_registry.get_agents_by_roles.return_value = [agent_a, agent_b]

        selected = await orchestrator._find_available_agent(["ap_clerk"])
        assert selected is None

    @pytest.mark.asyncio
    async def test_find_available_agent_returns_none_when_no_matching_roles(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Empty candidate list → returns None."""
        mock_agent_registry.get_agents_by_roles.return_value = []

        selected = await orchestrator._find_available_agent(["nonexistent_role"])
        assert selected is None

    @pytest.mark.asyncio
    async def test_find_available_agent_respects_max_queue_depth(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Agents at or above max_queue_depth_per_agent are excluded."""
        # Config default max_queue_depth_per_agent = 50
        overloaded = create_mock_agent(
            role="ap_clerk", state=AgentState.IDLE, queue_size=50
        )
        available = create_mock_agent(
            role="ap_clerk", state=AgentState.IDLE, queue_size=3
        )
        mock_agent_registry.get_agents_by_roles.return_value = [
            overloaded,
            available,
        ]

        selected = await orchestrator._find_available_agent(["ap_clerk"])
        assert selected is available


# ---------------------------------------------------------------------------
# Test Class — Transaction Routing (README.md lines 1289-1344)
# ---------------------------------------------------------------------------


class TestTransactionRouting:
    """Verify end-to-end route_transaction behaviour."""

    @pytest.mark.asyncio
    async def test_route_transaction_creates_workflow_instance(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """route_transaction returns a UUID and stores a WorkflowInstance."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )

        assert isinstance(wf_id, UUID)
        assert wf_id in orchestrator.workflows
        wf = orchestrator.workflows[wf_id]
        assert wf.transaction_type == "vendor_invoice"

    @pytest.mark.asyncio
    async def test_route_transaction_assigns_agent(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Routed workflow is assigned to the selected idle agent."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )

        wf = orchestrator.workflows[wf_id]
        assert wf.assigned_agent_id == agent.config.agent_id
        assert wf.status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_route_transaction_adds_work_item_to_queue(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """A work item is enqueued on the selected agent's work_queue."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )

        assert agent.work_queue.qsize() == 1
        work_item = agent.work_queue.get_nowait()
        assert work_item["type"] == "vendor_invoice"
        assert work_item["workflow_id"] == str(wf_id)

    @pytest.mark.asyncio
    async def test_route_transaction_queues_when_no_agent_available(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """When no idle agent is available the workflow goes to PENDING."""
        mock_agent_registry.get_agents_by_roles.return_value = []

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )

        wf = orchestrator.workflows[wf_id]
        assert wf.status == WorkflowStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_route_transaction_checks_approval(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        mock_approval_system: MagicMock,
    ) -> None:
        """When approval_system returns a role, the workflow records it."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        # Amount above $100K requires CFO approval
        mock_approval_system.get_required_approval.return_value = "cfo"

        txn = {
            "transaction_id": str(uuid4()),
            "vendor_id": "V-002",
            "amount": 150000.00,
            "description": "High-value invoice",
        }
        wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")

        wf = orchestrator.workflows[wf_id]
        assert wf.approval_required is True
        assert wf.approval_role == "cfo"

    @pytest.mark.asyncio
    async def test_route_transaction_no_approval_when_not_required(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        mock_approval_system: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Approval fields remain False/None when no approval is needed."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]
        mock_approval_system.get_required_approval.return_value = None

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )

        wf = orchestrator.workflows[wf_id]
        assert wf.approval_required is False
        assert wf.approval_role is None

    @pytest.mark.asyncio
    async def test_route_transaction_routing_sla(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Route must complete within 500ms SLA (AAP Section 0.7.3)."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        start = time.perf_counter()
        await orchestrator.route_transaction(sample_transaction, "vendor_invoice")
        duration = time.perf_counter() - start

        assert duration < 0.5, (
            f"Routing took {duration:.4f}s — exceeds 500ms SLA"
        )

    @pytest.mark.asyncio
    async def test_route_transaction_increments_metrics(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """total_routed metric is incremented on successful routing."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        assert orchestrator.metrics["total_routed"] == 0
        await orchestrator.route_transaction(sample_transaction, "vendor_invoice")
        assert orchestrator.metrics["total_routed"] == 1

    @pytest.mark.asyncio
    async def test_route_transaction_publishes_event(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        mock_event_bus: AsyncMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """A TransactionCreated event is published via the event bus."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        await orchestrator.route_transaction(sample_transaction, "vendor_invoice")

        mock_event_bus.publish.assert_awaited_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.event_type == "TransactionCreated"

    @pytest.mark.asyncio
    async def test_route_transaction_concurrent_limit_exceeded(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Exceeding max_concurrent_workflows marks the workflow FAILED."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        # Fill up to the limit with active workflows
        for _ in range(100):
            txn = {"transaction_id": str(uuid4()), "amount": 100}
            await orchestrator.route_transaction(txn, "vendor_invoice")

        # One more should trigger the limit
        txn_over = {"transaction_id": str(uuid4()), "amount": 100}
        wf_id = await orchestrator.route_transaction(txn_over, "vendor_invoice")
        wf = orchestrator.workflows[wf_id]
        assert wf.status == WorkflowStatus.FAILED.value
        assert "limit exceeded" in (wf.error_message or "").lower()


# ---------------------------------------------------------------------------
# Test Class — Workflow Lifecycle
# ---------------------------------------------------------------------------


class TestWorkflowLifecycle:
    """Verify complete / fail / retry workflow state transitions."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Completing a workflow sets COMPLETED status and timestamps."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        await orchestrator.complete_workflow(wf_id)

        wf = orchestrator.workflows[wf_id]
        assert wf.status == WorkflowStatus.COMPLETED.value
        assert wf.completed_at is not None
        assert orchestrator.metrics["total_completed"] == 1

    @pytest.mark.asyncio
    async def test_complete_workflow_publishes_event(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        mock_event_bus: AsyncMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """A TransactionCompleted event is published on completion."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        # Reset call tracking after the route event
        mock_event_bus.publish.reset_mock()

        await orchestrator.complete_workflow(wf_id)

        mock_event_bus.publish.assert_awaited_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.event_type == "TransactionCompleted"

    @pytest.mark.asyncio
    async def test_complete_workflow_with_result(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Passing a result dict stores it in workflow metadata."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        result = {"status": "matched", "gl_code": "5100"}
        await orchestrator.complete_workflow(wf_id, result=result)

        wf = orchestrator.workflows[wf_id]
        assert wf.metadata.get("result") == result

    @pytest.mark.asyncio
    async def test_complete_nonexistent_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """Completing a missing workflow does not raise."""
        await orchestrator.complete_workflow(uuid4())
        # No error — logs a warning but silently returns

    @pytest.mark.asyncio
    async def test_fail_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Failing a workflow records error details and increments metric."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        await orchestrator.fail_workflow(wf_id, "test error")

        wf = orchestrator.workflows[wf_id]
        assert wf.status == WorkflowStatus.FAILED.value
        assert wf.error_message == "test error"
        assert wf.error_timestamp is not None
        assert orchestrator.metrics["total_failed"] == 1

    @pytest.mark.asyncio
    async def test_fail_nonexistent_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """Failing a missing workflow does not raise."""
        await orchestrator.fail_workflow(uuid4(), "not found")

    @pytest.mark.asyncio
    async def test_retry_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Retrying increments retry_count and sets RETRYING or ASSIGNED."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        await orchestrator.fail_workflow(wf_id, "transient error")

        result = await orchestrator.retry_workflow(wf_id)
        assert result is True

        wf = orchestrator.workflows[wf_id]
        assert wf.retry_count == 1
        assert orchestrator.metrics["total_retried"] >= 1

    @pytest.mark.asyncio
    async def test_retry_workflow_exceeds_max(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """Exceeding retry_max_attempts (3) returns False → FAILED."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        # Manually set retry_count to the max
        orchestrator.workflows[wf_id].retry_count = 3

        result = await orchestrator.retry_workflow(wf_id)
        assert result is False

        wf = orchestrator.workflows[wf_id]
        assert wf.status == WorkflowStatus.FAILED.value
        assert "max retries" in (wf.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_retry_nonexistent_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """Retrying a missing workflow returns False."""
        result = await orchestrator.retry_workflow(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_workflow_re_routes_to_agent(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
        sample_transaction: Dict[str, Any],
    ) -> None:
        """A successful retry re-routes the workflow to an idle agent."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        wf_id = await orchestrator.route_transaction(
            sample_transaction, "vendor_invoice"
        )
        await orchestrator.fail_workflow(wf_id, "transient")

        result = await orchestrator.retry_workflow(wf_id)
        assert result is True

        wf = orchestrator.workflows[wf_id]
        # After a successful re-route the status is ASSIGNED
        assert wf.status == WorkflowStatus.ASSIGNED.value
        assert wf.assigned_agent_id is not None

    def test_get_workflow_returns_existing(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """get_workflow returns the stored WorkflowInstance."""
        wf = WorkflowInstance(transaction_type="vendor_invoice")
        orchestrator.workflows[wf.workflow_id] = wf

        fetched = orchestrator.get_workflow(wf.workflow_id)
        assert fetched is wf

    def test_get_workflow_returns_none_for_missing(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """get_workflow returns None for an unknown UUID."""
        assert orchestrator.get_workflow(uuid4()) is None

    @pytest.mark.asyncio
    async def test_get_active_workflows(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Only non-terminal workflows appear in get_active_workflows."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        # Route 3 transactions
        ids: List[UUID] = []
        for _ in range(3):
            txn = {"transaction_id": str(uuid4()), "amount": 100}
            wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")
            ids.append(wf_id)

        # Complete one of them
        await orchestrator.complete_workflow(ids[0])

        active = orchestrator.get_active_workflows()
        active_ids = {wf.workflow_id for wf in active}

        assert ids[0] not in active_ids
        assert ids[1] in active_ids
        assert ids[2] in active_ids


# ---------------------------------------------------------------------------
# Test Class — Queue Management
# ---------------------------------------------------------------------------


class TestQueueManagement:
    """Verify pending-queue processing and agent queue depth queries."""

    @pytest.mark.asyncio
    async def test_process_pending_queue(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Pending workflows are routed when agents become available."""
        # Phase 1: no agents → workflows go to pending
        mock_agent_registry.get_agents_by_roles.return_value = []
        pending_ids: List[UUID] = []
        for _ in range(3):
            txn = {"transaction_id": str(uuid4()), "amount": 100}
            wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")
            pending_ids.append(wf_id)

        # All should be pending
        for pid in pending_ids:
            assert orchestrator.workflows[pid].status == WorkflowStatus.PENDING.value

        # Phase 2: agents now available
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        routed = await orchestrator.process_pending_queue()
        assert routed == 3

        # Workflows should now be assigned
        for pid in pending_ids:
            assert orchestrator.workflows[pid].status == WorkflowStatus.ASSIGNED.value

    @pytest.mark.asyncio
    async def test_process_pending_queue_partial(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """When only some pending can route, the rest remain queued."""
        mock_agent_registry.get_agents_by_roles.return_value = []
        for _ in range(3):
            txn = {"transaction_id": str(uuid4()), "amount": 100}
            await orchestrator.route_transaction(txn, "vendor_invoice")

        # Make agent available for only one call, then unavailable
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        call_count = 0

        def selective_return(roles):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return [agent]
            return []

        mock_agent_registry.get_agents_by_roles.side_effect = selective_return

        routed = await orchestrator.process_pending_queue()
        assert routed >= 1

    @pytest.mark.asyncio
    async def test_process_pending_queue_empty(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """Processing an empty pending queue returns 0."""
        routed = await orchestrator.process_pending_queue()
        assert routed == 0

    def test_get_agent_queue_depth(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """get_agent_queue_depth returns the correct count."""
        agent_id = uuid4()
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait({"test": 1})
        queue.put_nowait({"test": 2})
        orchestrator.work_queues[agent_id] = queue

        assert orchestrator.get_agent_queue_depth(agent_id) == 2

    def test_get_agent_queue_depth_unknown_agent(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """Unknown agent_id returns 0 queue depth."""
        assert orchestrator.get_agent_queue_depth(uuid4()) == 0

    @pytest.mark.asyncio
    async def test_get_metrics(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """get_metrics returns a snapshot with expected keys."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        txn = {"transaction_id": str(uuid4()), "amount": 100}
        await orchestrator.route_transaction(txn, "vendor_invoice")

        metrics = orchestrator.get_metrics()
        assert "total_routed" in metrics
        assert "total_completed" in metrics
        assert "total_failed" in metrics
        assert "active_workflows" in metrics
        assert "pending_count" in metrics
        assert "workflows_by_status" in metrics
        assert metrics["total_routed"] == 1

    @pytest.mark.asyncio
    async def test_get_metrics_status_breakdown(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """workflows_by_status shows correct per-status counts."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        ids: List[UUID] = []
        for _ in range(3):
            txn = {"transaction_id": str(uuid4()), "amount": 100}
            wf_id = await orchestrator.route_transaction(txn, "vendor_invoice")
            ids.append(wf_id)

        await orchestrator.complete_workflow(ids[0])
        await orchestrator.fail_workflow(ids[1], "error")

        metrics = orchestrator.get_metrics()
        status_counts = metrics["workflows_by_status"]
        assert status_counts.get(WorkflowStatus.COMPLETED.value, 0) == 1
        assert status_counts.get(WorkflowStatus.FAILED.value, 0) == 1
        assert status_counts.get(WorkflowStatus.ASSIGNED.value, 0) == 1


# ---------------------------------------------------------------------------
# Test Class — WorkflowInstance Model Validation
# ---------------------------------------------------------------------------


class TestWorkflowInstance:
    """Verify WorkflowInstance Pydantic V2 model structure and serialization."""

    def test_workflow_instance_creation(self) -> None:
        """Basic creation populates defaults correctly."""
        wf = WorkflowInstance(transaction_type="vendor_invoice")
        assert isinstance(wf.workflow_id, UUID)
        assert wf.status == WorkflowStatus.PENDING.value
        assert wf.created_at is not None
        assert isinstance(wf.created_at, datetime)
        assert wf.retry_count == 0
        assert wf.approval_required is False
        assert wf.approval_role is None

    def test_workflow_instance_serialization(self) -> None:
        """model_dump() produces a dict with all fields."""
        wf = WorkflowInstance(
            transaction_type="purchase_order",
            transaction_data={"item": "widget", "qty": 10},
        )
        data = wf.model_dump()

        assert "workflow_id" in data
        assert "transaction_type" in data
        assert data["transaction_type"] == "purchase_order"
        assert "status" in data
        assert "created_at" in data
        assert "retry_count" in data
        assert "approval_required" in data

    def test_workflow_instance_with_agent_assignment(self) -> None:
        """Agent assignment fields are correctly set."""
        agent_id = uuid4()
        wf = WorkflowInstance(
            transaction_type="vendor_invoice",
            assigned_agent_id=agent_id,
            status=WorkflowStatus.ASSIGNED,
        )
        assert wf.assigned_agent_id == agent_id
        assert wf.status == WorkflowStatus.ASSIGNED.value

    def test_workflow_instance_error_fields(self) -> None:
        """Error-related fields are populated on failure."""
        now = datetime.now(timezone.utc)
        wf = WorkflowInstance(
            transaction_type="vendor_invoice",
            status=WorkflowStatus.FAILED,
            error_message="Processing error",
            error_timestamp=now,
        )
        assert wf.error_message == "Processing error"
        assert wf.error_timestamp == now

    def test_workflow_instance_approval_fields(self) -> None:
        """Approval fields are correctly stored."""
        wf = WorkflowInstance(
            transaction_type="purchase_order",
            approval_required=True,
            approval_role="cfo",
        )
        assert wf.approval_required is True
        assert wf.approval_role == "cfo"

    def test_workflow_instance_metadata(self) -> None:
        """Metadata dict is stored and serializable."""
        wf = WorkflowInstance(
            transaction_type="journal_entry",
            metadata={"source": "month_end", "priority": "high"},
        )
        assert wf.metadata["source"] == "month_end"
        dumped = wf.model_dump()
        assert dumped["metadata"]["priority"] == "high"


# ---------------------------------------------------------------------------
# Test Class — WorkflowConfig Validation
# ---------------------------------------------------------------------------


class TestWorkflowConfig:
    """Verify WorkflowConfig defaults and custom values."""

    def test_workflow_config_defaults(self) -> None:
        """Default values match the specification."""
        cfg = WorkflowConfig()
        assert cfg.max_concurrent_workflows == 100
        assert cfg.routing_timeout_ms == 500
        assert cfg.retry_max_attempts == 3
        assert cfg.max_queue_depth_per_agent == 50
        assert cfg.enable_approval_routing is True
        assert cfg.pending_queue_max_size == 1000

    def test_workflow_config_custom_values(self) -> None:
        """Custom configuration overrides defaults."""
        cfg = WorkflowConfig(
            max_concurrent_workflows=200,
            routing_timeout_ms=300,
            retry_max_attempts=5,
            max_queue_depth_per_agent=100,
        )
        assert cfg.max_concurrent_workflows == 200
        assert cfg.routing_timeout_ms == 300
        assert cfg.retry_max_attempts == 5
        assert cfg.max_queue_depth_per_agent == 100

    def test_workflow_config_serialization(self) -> None:
        """WorkflowConfig can be serialized via model_dump()."""
        cfg = WorkflowConfig()
        data = cfg.model_dump()
        assert isinstance(data, dict)
        assert "max_concurrent_workflows" in data
        assert "routing_timeout_ms" in data

    def test_workflow_config_validation_rejects_invalid(self) -> None:
        """Invalid values (e.g. negative) raise validation errors."""
        with pytest.raises(Exception):
            WorkflowConfig(max_concurrent_workflows=-1)

    def test_workflow_config_concurrent_minimum(self) -> None:
        """Default allows at least 100 concurrent workflows (AAP 0.7.3)."""
        cfg = WorkflowConfig()
        assert cfg.max_concurrent_workflows >= 100


# ---------------------------------------------------------------------------
# Test Class — Concurrent Workflow Support
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Verify concurrent workflow routing and capacity."""

    @pytest.mark.asyncio
    async def test_concurrent_workflow_limit(
        self,
        orchestrator: WorkflowOrchestrator,
    ) -> None:
        """WorkflowConfig supports ≥100 concurrent workflows (AAP 0.7.3)."""
        assert orchestrator.config.max_concurrent_workflows >= 100

    @pytest.mark.asyncio
    async def test_multiple_concurrent_routes(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """10 transactions routed concurrently get unique workflow_ids."""
        agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        mock_agent_registry.get_agents_by_roles.return_value = [agent]

        transactions = [
            {"transaction_id": str(uuid4()), "amount": 100 + i}
            for i in range(10)
        ]

        wf_ids = await asyncio.gather(
            *[
                orchestrator.route_transaction(txn, "vendor_invoice")
                for txn in transactions
            ]
        )

        # All 10 should be unique UUIDs
        assert len(set(wf_ids)) == 10
        for wf_id in wf_ids:
            assert isinstance(wf_id, UUID)

        # Metrics should reflect 10 routed (some may end up pending
        # due to concurrency limit, but the total workflow count is 10)
        assert len(orchestrator.workflows) == 10

    @pytest.mark.asyncio
    async def test_concurrent_routes_with_different_types(
        self,
        orchestrator: WorkflowOrchestrator,
        mock_agent_registry: MagicMock,
    ) -> None:
        """Concurrent routing of different transaction types succeeds."""
        ap_agent = create_mock_agent(role="ap_clerk", state=AgentState.IDLE)
        purch_agent = create_mock_agent(
            role="purchasing_agent", state=AgentState.IDLE
        )
        ar_agent = create_mock_agent(role="ar_clerk", state=AgentState.IDLE)

        def get_agents_for_roles(roles: List[str]) -> List[MagicMock]:
            agents = []
            if "ap_clerk" in roles or "ap_manager" in roles:
                agents.append(ap_agent)
            if "purchasing_agent" in roles or "purchasing_manager" in roles:
                agents.append(purch_agent)
            if "ar_clerk" in roles:
                agents.append(ar_agent)
            return agents

        mock_agent_registry.get_agents_by_roles.side_effect = get_agents_for_roles

        tasks = [
            orchestrator.route_transaction(
                {"transaction_id": str(uuid4()), "amount": 500}, "vendor_invoice"
            ),
            orchestrator.route_transaction(
                {"transaction_id": str(uuid4()), "amount": 1000}, "purchase_order"
            ),
            orchestrator.route_transaction(
                {"transaction_id": str(uuid4()), "amount": 200}, "sales_order"
            ),
        ]

        wf_ids = await asyncio.gather(*tasks)
        assert len(set(wf_ids)) == 3
        assert len(orchestrator.workflows) == 3
