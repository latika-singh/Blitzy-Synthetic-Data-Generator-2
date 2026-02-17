"""Comprehensive unit tests for the ApprovalSystem approval chain engine.

Tests cover:
    - Threshold enforcement for all 3 transaction types (purchase_order,
      vendor_invoice, journal_entry) with exact boundary values from
      README.md lines 314-331.
    - Approval request creation lifecycle and event publishing.
    - Approval decision processing (approved, rejected).
    - Rejection escalation chains with next-level promotion.
    - Query methods (pending, by-role, by-id, chain, metrics).
    - ApprovalRequest Pydantic V2 model validation and defaults.
    - Edge cases: unknown types, negative amounts, extreme values.

CRITICAL Testing Rules (AAP Section 0.7.5):
    - All agent dependencies are MOCKED (AgentRegistry, EventBus).
    - Async tests use pytest-asyncio.
    - No live API calls, no external Redis.
    - Unit test coverage target: >= 80%.

References:
    - README.md lines 310-338 (ApprovalSystem specification)
    - AAP Section 0.5.1 Group 10 (Test Files)
    - AAP Section 0.7.5 (Testing and Quality Standards)
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import pytest

# ---------------------------------------------------------------------------
# Internal Imports — Module Under Test
# ---------------------------------------------------------------------------
from app.orchestration.approval_system import (
    APPROVAL_THRESHOLDS,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalSystem,
)


# =========================================================================
# Helper Functions
# =========================================================================


def make_transaction(
    txn_type: str,
    amount: float,
    transaction_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a minimal transaction dict for test inputs.

    Args:
        txn_type: Transaction category (e.g. ``"purchase_order"``).
        amount: Monetary amount for threshold evaluation.
        transaction_id: Optional explicit UUID string.  Generates one
            when omitted.

    Returns:
        Dictionary with ``transaction_type``, ``amount``, and
        ``transaction_id`` keys.
    """
    return {
        "transaction_type": txn_type,
        "amount": amount,
        "transaction_id": transaction_id or str(uuid4()),
    }


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """Provide a MagicMock stand-in for :class:`AgentRegistry`."""
    registry = MagicMock()
    registry.get_agents_by_role = MagicMock(return_value=[])
    return registry


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Provide an AsyncMock stand-in for :class:`EventBus`.

    The ``publish`` method is an :class:`AsyncMock` so that ``await``
    calls inside the system under test resolve without errors.
    """
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def approval_system(
    mock_agent_registry: MagicMock,
    mock_event_bus: AsyncMock,
) -> ApprovalSystem:
    """Fully-wired :class:`ApprovalSystem` with mocked dependencies."""
    return ApprovalSystem(
        agent_registry=mock_agent_registry,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def approval_system_no_deps() -> ApprovalSystem:
    """Bare :class:`ApprovalSystem` without AgentRegistry or EventBus.

    Useful for pure threshold-lookup tests that don't require event
    publishing or agent discovery.
    """
    return ApprovalSystem()


# =========================================================================
# Test Class 1 — Purchase Order Thresholds
# README.md lines 315-320
# =========================================================================


class TestPurchaseOrderThresholds:
    """Verify purchase_order approval thresholds.

    Tier structure (README.md lines 316-319):
        (0, 5000, None)                     — < $5K: no approval
        (5000, 25000, "purchasing_manager")  — $5K-$25K
        (25000, 100000, "controller")        — $25K-$100K
        (100000, None, "cfo")                — > $100K
    """

    def test_po_below_5k_no_approval(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Amounts strictly below $5,000 require no approval."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 4999.99)
        assert result is None

    def test_po_at_5k_needs_purchasing_manager(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $5,000 triggers purchasing_manager tier (inclusive lower bound)."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 5000.00)
        assert result == "purchasing_manager"

    def test_po_between_5k_and_25k_needs_purchasing_manager(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """Mid-range amount in $5K-$25K band routes to purchasing_manager."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 15000.00)
        assert result == "purchasing_manager"

    def test_po_at_25k_needs_controller(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $25,000 triggers controller tier (inclusive lower bound)."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 25000.00)
        assert result == "controller"

    def test_po_between_25k_and_100k_needs_controller(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """Mid-range amount in $25K-$100K band routes to controller."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 75000.00)
        assert result == "controller"

    def test_po_at_100k_needs_cfo(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $100,000 triggers CFO tier (inclusive lower bound)."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 100000.00)
        assert result == "cfo"

    def test_po_above_100k_needs_cfo(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Large amounts above $100K always require CFO approval."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 500000.00)
        assert result == "cfo"

    def test_po_zero_amount_no_approval(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Zero-dollar PO falls in the first tier — no approval needed."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 0)
        assert result is None

    def test_po_exactly_at_boundary_4999_99(self, approval_system_no_deps: ApprovalSystem) -> None:
        """$4,999.99 is just below the $5K boundary — no approval."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 4999.99)
        assert result is None


# =========================================================================
# Test Class 2 — Vendor Invoice Thresholds
# README.md lines 321-326
# =========================================================================


class TestVendorInvoiceThresholds:
    """Verify vendor_invoice approval thresholds.

    Tier structure (README.md lines 322-325):
        (0, 10000, None)                    — < $10K: no approval
        (10000, 50000, "ap_manager")         — $10K-$50K
        (50000, 100000, "controller")        — $50K-$100K
        (100000, None, "cfo")                — > $100K
    """

    def test_invoice_below_10k_no_approval(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Invoices below $10,000 need no approval."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 9999.99)
        assert result is None

    def test_invoice_at_10k_needs_ap_manager(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $10,000 triggers ap_manager tier."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 10000.00)
        assert result == "ap_manager"

    def test_invoice_between_10k_and_50k_needs_ap_manager(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """Mid-range amount in $10K-$50K band routes to ap_manager."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 30000.00)
        assert result == "ap_manager"

    def test_invoice_at_50k_needs_controller(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $50,000 triggers controller tier."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 50000.00)
        assert result == "controller"

    def test_invoice_between_50k_and_100k_needs_controller(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """Mid-range amount in $50K-$100K band routes to controller."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 80000.00)
        assert result == "controller"

    def test_invoice_at_100k_needs_cfo(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $100,000 triggers CFO tier."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 100000.00)
        assert result == "cfo"

    def test_invoice_above_100k_needs_cfo(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Large invoices above $100K require CFO approval."""
        result = approval_system_no_deps.get_required_approval("vendor_invoice", 250000.00)
        assert result == "cfo"


# =========================================================================
# Test Class 3 — Journal Entry Thresholds
# README.md lines 327-331
# =========================================================================


class TestJournalEntryThresholds:
    """Verify journal_entry approval thresholds.

    CRITICAL: Journal entries ALWAYS need approval — there is no ``None``
    tier.  Even the smallest JE requires ``senior_accountant``.

    Tier structure (README.md lines 328-329):
        (0, 50000, "senior_accountant")     — < $50K
        (50000, None, "controller")          — >= $50K
    """

    def test_je_below_50k_needs_senior_accountant(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """$10,000 JE requires senior_accountant approval."""
        result = approval_system_no_deps.get_required_approval("journal_entry", 10000.00)
        assert result == "senior_accountant"

    def test_je_at_zero_needs_senior_accountant(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """Even a minimal JE ($0.01) always requires senior_accountant."""
        result = approval_system_no_deps.get_required_approval("journal_entry", 0.01)
        assert result == "senior_accountant"

    def test_je_at_50k_needs_controller(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Exactly $50,000 JE triggers controller tier."""
        result = approval_system_no_deps.get_required_approval("journal_entry", 50000.00)
        assert result == "controller"

    def test_je_above_50k_needs_controller(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Large JE above $50K requires controller approval."""
        result = approval_system_no_deps.get_required_approval("journal_entry", 200000.00)
        assert result == "controller"


# =========================================================================
# Test Class 4 — Edge-Case Thresholds
# =========================================================================


class TestEdgeCaseThresholds:
    """Verify graceful handling of unknown types, negatives, and extremes."""

    def test_unknown_transaction_type_returns_none(
        self, approval_system_no_deps: ApprovalSystem
    ) -> None:
        """An unrecognised transaction type should return None."""
        result = approval_system_no_deps.get_required_approval("unknown_type", 50000.00)
        assert result is None

    def test_negative_amount_handling(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Negative amounts should not cause crashes — return None or first-tier result."""
        result = approval_system_no_deps.get_required_approval("purchase_order", -100.00)
        # Negative amount doesn't match any tier (min_amount=0 requires amount >= 0)
        # so the result is None — acceptable graceful handling
        assert result is None

    def test_very_large_amount(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Billion-dollar PO should still route to CFO (unbounded upper tier)."""
        result = approval_system_no_deps.get_required_approval("purchase_order", 1_000_000_000.00)
        assert result == "cfo"

    def test_thresholds_constant_matches_readme(self) -> None:
        """CRITICAL: Verify APPROVAL_THRESHOLDS dict structure matches README exactly.

        README.md lines 314-331 define:
          - purchase_order: 4 tiers
          - vendor_invoice: 4 tiers
          - journal_entry: 2 tiers
        """
        assert "purchase_order" in APPROVAL_THRESHOLDS
        assert "vendor_invoice" in APPROVAL_THRESHOLDS
        assert "journal_entry" in APPROVAL_THRESHOLDS

        assert len(APPROVAL_THRESHOLDS["purchase_order"]) == 4
        assert len(APPROVAL_THRESHOLDS["vendor_invoice"]) == 4
        assert len(APPROVAL_THRESHOLDS["journal_entry"]) == 2

    def test_po_thresholds_exact_values(self) -> None:
        """Verify each PO threshold tuple matches the specification."""
        po = APPROVAL_THRESHOLDS["purchase_order"]
        assert po[0] == (0, 5000, None)
        assert po[1] == (5000, 25000, "purchasing_manager")
        assert po[2] == (25000, 100000, "controller")
        assert po[3] == (100000, None, "cfo")

    def test_invoice_thresholds_exact_values(self) -> None:
        """Verify each vendor_invoice threshold tuple matches the specification."""
        inv = APPROVAL_THRESHOLDS["vendor_invoice"]
        assert inv[0] == (0, 10000, None)
        assert inv[1] == (10000, 50000, "ap_manager")
        assert inv[2] == (50000, 100000, "controller")
        assert inv[3] == (100000, None, "cfo")

    def test_je_thresholds_exact_values(self) -> None:
        """Verify each journal_entry threshold tuple matches the specification."""
        je = APPROVAL_THRESHOLDS["journal_entry"]
        assert je[0] == (0, 50000, "senior_accountant")
        assert je[1] == (50000, None, "controller")


# =========================================================================
# Test Class 5 — Approval Request Creation
# README.md line 335: create_approval_request(transaction, approver_role)
# =========================================================================


class TestApprovalRequestCreation:
    """Verify approval request creation, storage, event publishing, and metrics."""

    @pytest.mark.asyncio
    async def test_create_approval_request(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Created request should carry correct transaction metadata."""
        txn = make_transaction("purchase_order", 50000.00)
        request = await approval_system.create_approval_request(txn, "controller")

        assert isinstance(request, ApprovalRequest)
        assert isinstance(request.request_id, UUID)
        assert request.transaction_type == "purchase_order"
        assert request.amount == 50000.00
        assert request.approver_role == "controller"
        # With model_config use_enum_values=True, decision is stored as string
        assert request.decision in ("pending", ApprovalDecision.PENDING.value)

    @pytest.mark.asyncio
    async def test_create_approval_request_publishes_event(
        self,
        approval_system: ApprovalSystem,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Creating a request should publish an event via the EventBus."""
        txn = make_transaction("purchase_order", 50000.00)
        await approval_system.create_approval_request(txn, "controller")

        # The system publishes ApprovalRequired via event_bus.publish.
        # The call may succeed or silently fail (try/except inside),
        # but when the import works the mock should be invoked.
        # We check that publish was called at least once.
        assert mock_event_bus.publish.call_count >= 1

    @pytest.mark.asyncio
    async def test_create_approval_request_adds_to_pending(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Newly created request should appear in pending requests."""
        txn = make_transaction("purchase_order", 50000.00)
        request = await approval_system.create_approval_request(txn, "controller")

        pending = approval_system.get_pending_requests()
        assert len(pending) == 1
        assert pending[0].request_id == request.request_id

    @pytest.mark.asyncio
    async def test_create_approval_request_increments_metrics(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Each created request should increment total_requests counter."""
        txn1 = make_transaction("purchase_order", 10000.00)
        txn2 = make_transaction("vendor_invoice", 20000.00)

        await approval_system.create_approval_request(txn1, "purchasing_manager")
        await approval_system.create_approval_request(txn2, "ap_manager")

        assert approval_system.metrics["total_requests"] >= 2

    @pytest.mark.asyncio
    async def test_create_approval_request_sets_created_at(
        self, approval_system: ApprovalSystem
    ) -> None:
        """The created_at timestamp should be a UTC-aware datetime."""
        before = datetime.now(timezone.utc)
        txn = make_transaction("journal_entry", 5000.00)
        request = await approval_system.create_approval_request(txn, "senior_accountant")
        after = datetime.now(timezone.utc)

        assert isinstance(request.created_at, datetime)
        assert before <= request.created_at <= after

    @pytest.mark.asyncio
    async def test_create_approval_request_stores_transaction_id(
        self, approval_system: ApprovalSystem
    ) -> None:
        """The transaction_id from the dict should be stored as UUID."""
        fixed_id = str(uuid4())
        txn = make_transaction("purchase_order", 50000.00, transaction_id=fixed_id)
        request = await approval_system.create_approval_request(txn, "controller")

        assert request.transaction_id == UUID(fixed_id)


# =========================================================================
# Test Class 6 — Approval Decision Processing
# README.md line 336: process_approval(request, decision, notes)
# =========================================================================


class TestApprovalProcessing:
    """Verify approval processing: decision recording, metrics, events."""

    @pytest.mark.asyncio
    async def test_process_approval_approved(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Processing with APPROVED should update decision fields."""
        txn = make_transaction("purchase_order", 50000.00)
        request = await approval_system.create_approval_request(txn, "controller")

        updated = await approval_system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "Looks good"
        )

        assert updated.decision in ("approved", ApprovalDecision.APPROVED.value)
        assert updated.decision_notes == "Looks good"
        assert updated.decision_timestamp is not None
        assert isinstance(updated.decision_timestamp, datetime)

    @pytest.mark.asyncio
    async def test_process_approval_approved_increments_metrics(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Approved decision should increment total_approved counter."""
        txn = make_transaction("vendor_invoice", 30000.00)
        request = await approval_system.create_approval_request(txn, "ap_manager")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "OK"
        )

        assert approval_system.metrics["total_approved"] >= 1

    @pytest.mark.asyncio
    async def test_process_approval_rejected(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Processing with REJECTED should update decision and increment metrics."""
        txn = make_transaction("purchase_order", 80000.00)
        request = await approval_system.create_approval_request(txn, "controller")

        updated = await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Budget exceeded"
        )

        assert updated.decision in ("rejected", ApprovalDecision.REJECTED.value)
        assert updated.decision_notes == "Budget exceeded"
        assert approval_system.metrics["total_rejected"] >= 1

    @pytest.mark.asyncio
    async def test_process_approval_publishes_event(
        self,
        approval_system: ApprovalSystem,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Processing a decision should publish ApprovalCompleted via EventBus."""
        txn = make_transaction("purchase_order", 50000.00)
        request = await approval_system.create_approval_request(txn, "controller")
        # Reset publish call count after request creation event
        initial_calls = mock_event_bus.publish.call_count

        await approval_system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "Approved"
        )

        # Publish should have been called at least once more for ApprovalCompleted
        assert mock_event_bus.publish.call_count > initial_calls

    @pytest.mark.asyncio
    async def test_process_approval_moves_from_pending_to_completed(
        self, approval_system: ApprovalSystem
    ) -> None:
        """After processing, the request should move from pending to completed."""
        txn = make_transaction("purchase_order", 50000.00)
        request = await approval_system.create_approval_request(txn, "controller")

        # Before processing — request is in pending
        assert len(approval_system.get_pending_requests()) == 1

        await approval_system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "Fine"
        )

        # After processing — pending is empty, get_request still finds it
        assert len(approval_system.get_pending_requests()) == 0
        found = approval_system.get_request(request.request_id)
        assert found is not None
        assert found.decision in ("approved", ApprovalDecision.APPROVED.value)

    @pytest.mark.asyncio
    async def test_process_approval_nonexistent_request_raises(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Processing a non-existent request should raise ValueError."""
        fake_id = uuid4()
        with pytest.raises(ValueError, match="not found"):
            await approval_system.process_approval(
                fake_id, ApprovalDecision.APPROVED, "Should fail"
            )


# =========================================================================
# Test Class 7 — Rejection Escalation
# README.md line 337: escalate_if_rejected(request)
# =========================================================================


class TestEscalation:
    """Verify rejection escalation promotes requests up the approval chain."""

    @pytest.mark.asyncio
    async def test_escalate_rejected_po_to_next_level(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Rejected PO at purchasing_manager ($15K) should escalate to controller.

        Chain: purchasing_manager → controller → cfo.
        """
        txn = make_transaction("purchase_order", 15000.00)
        request = await approval_system.create_approval_request(txn, "purchasing_manager")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Needs higher approval"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is not None
        assert escalated.approver_role == "controller"
        assert escalated.transaction_type == "purchase_order"
        assert escalated.amount == 15000.00

    @pytest.mark.asyncio
    async def test_escalate_rejected_invoice_to_next_level(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Rejected invoice at ap_manager ($20K) should escalate to controller."""
        txn = make_transaction("vendor_invoice", 20000.00)
        request = await approval_system.create_approval_request(txn, "ap_manager")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Needs escalation"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is not None
        assert escalated.approver_role == "controller"

    @pytest.mark.asyncio
    async def test_escalate_at_highest_level_returns_none(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Rejected PO at CFO level ($150K) has no higher authority — returns None."""
        txn = make_transaction("purchase_order", 150000.00)
        request = await approval_system.create_approval_request(txn, "cfo")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Cannot approve"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is None

    @pytest.mark.asyncio
    async def test_escalate_non_rejected_returns_none(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Escalating an approved request should return None (only rejected escalate)."""
        txn = make_transaction("purchase_order", 15000.00)
        request = await approval_system.create_approval_request(txn, "purchasing_manager")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "All good"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is None

    @pytest.mark.asyncio
    async def test_escalation_links_to_original_request(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Escalated request should have escalated_from pointing to the original."""
        txn = make_transaction("vendor_invoice", 30000.00)
        request = await approval_system.create_approval_request(txn, "ap_manager")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Denied"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is not None
        assert escalated.escalated_from == request.request_id

    @pytest.mark.asyncio
    async def test_escalation_increments_metrics(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Escalation should increment total_escalated counter."""
        txn = make_transaction("purchase_order", 15000.00)
        request = await approval_system.create_approval_request(txn, "purchasing_manager")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "No"
        )

        before_escalated = approval_system.metrics["total_escalated"]
        await approval_system.escalate_if_rejected(request.request_id)

        assert approval_system.metrics["total_escalated"] == before_escalated + 1

    @pytest.mark.asyncio
    async def test_escalation_controller_to_cfo(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Escalation from controller should promote to CFO."""
        txn = make_transaction("purchase_order", 60000.00)
        request = await approval_system.create_approval_request(txn, "controller")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Too expensive"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is not None
        assert escalated.approver_role == "cfo"

    @pytest.mark.asyncio
    async def test_escalation_je_senior_accountant_to_controller(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Rejected JE at senior_accountant should escalate to controller."""
        txn = make_transaction("journal_entry", 20000.00)
        request = await approval_system.create_approval_request(txn, "senior_accountant")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Wrong account"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is not None
        assert escalated.approver_role == "controller"

    @pytest.mark.asyncio
    async def test_escalation_je_controller_no_higher(
        self, approval_system: ApprovalSystem
    ) -> None:
        """JE rejected at controller level — no higher authority, returns None."""
        txn = make_transaction("journal_entry", 80000.00)
        request = await approval_system.create_approval_request(txn, "controller")
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "Cannot approve"
        )

        escalated = await approval_system.escalate_if_rejected(request.request_id)

        assert escalated is None


# =========================================================================
# Test Class 8 — Query Methods
# =========================================================================


class TestQueryMethods:
    """Verify query methods: pending, by-role, by-id, chain, metrics."""

    @pytest.mark.asyncio
    async def test_get_pending_requests(self, approval_system: ApprovalSystem) -> None:
        """get_pending_requests should track pending count accurately."""
        txn1 = make_transaction("purchase_order", 10000.00)
        txn2 = make_transaction("vendor_invoice", 20000.00)
        txn3 = make_transaction("journal_entry", 30000.00)

        r1 = await approval_system.create_approval_request(txn1, "purchasing_manager")
        r2 = await approval_system.create_approval_request(txn2, "ap_manager")
        r3 = await approval_system.create_approval_request(txn3, "senior_accountant")

        assert len(approval_system.get_pending_requests()) == 3

        # Process one — count should drop
        await approval_system.process_approval(
            r1.request_id, ApprovalDecision.APPROVED, "OK"
        )

        assert len(approval_system.get_pending_requests()) == 2

    @pytest.mark.asyncio
    async def test_get_pending_requests_by_role(
        self, approval_system: ApprovalSystem
    ) -> None:
        """get_pending_requests with approver_role should filter correctly."""
        txn1 = make_transaction("purchase_order", 50000.00)
        txn2 = make_transaction("purchase_order", 60000.00)
        txn3 = make_transaction("vendor_invoice", 30000.00)

        await approval_system.create_approval_request(txn1, "controller")
        await approval_system.create_approval_request(txn2, "controller")
        await approval_system.create_approval_request(txn3, "ap_manager")

        controller_pending = approval_system.get_pending_requests(approver_role="controller")
        assert len(controller_pending) == 2

        ap_pending = approval_system.get_pending_requests(approver_role="ap_manager")
        assert len(ap_pending) == 1

        cfo_pending = approval_system.get_pending_requests(approver_role="cfo")
        assert len(cfo_pending) == 0

    @pytest.mark.asyncio
    async def test_get_request_by_id(self, approval_system: ApprovalSystem) -> None:
        """get_request should find requests by UUID in both stores."""
        txn = make_transaction("purchase_order", 50000.00)
        request = await approval_system.create_approval_request(txn, "controller")

        # Found in pending
        found = approval_system.get_request(request.request_id)
        assert found is not None
        assert found.request_id == request.request_id

        # Process and verify still found (in completed)
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "Done"
        )
        found_completed = approval_system.get_request(request.request_id)
        assert found_completed is not None
        assert found_completed.request_id == request.request_id

    @pytest.mark.asyncio
    async def test_get_request_by_id_returns_none_for_unknown(
        self, approval_system: ApprovalSystem
    ) -> None:
        """get_request should return None for an unknown UUID."""
        unknown = uuid4()
        result = approval_system.get_request(unknown)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_approval_chain(self, approval_system: ApprovalSystem) -> None:
        """get_approval_chain should return all requests for a transaction, ordered."""
        fixed_txn_id = str(uuid4())
        txn = make_transaction("purchase_order", 15000.00, transaction_id=fixed_txn_id)

        # Create initial request
        request = await approval_system.create_approval_request(txn, "purchasing_manager")
        # Reject it
        await approval_system.process_approval(
            request.request_id, ApprovalDecision.REJECTED, "No"
        )
        # Escalate
        escalated = await approval_system.escalate_if_rejected(request.request_id)
        assert escalated is not None

        # The chain should have both the original and escalated request
        chain = approval_system.get_approval_chain(UUID(fixed_txn_id))
        assert len(chain) >= 2

        # Chain should be sorted by created_at (oldest first)
        for i in range(len(chain) - 1):
            assert chain[i].created_at <= chain[i + 1].created_at

    def test_get_metrics(self, approval_system_no_deps: ApprovalSystem) -> None:
        """get_metrics should return dict with expected keys."""
        metrics = approval_system_no_deps.get_metrics()

        assert isinstance(metrics, dict)
        expected_keys = {
            "total_requests",
            "total_approved",
            "total_rejected",
            "total_escalated",
            "average_approval_time",
            "pending_count",
            "completed_count",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_get_metrics_initial_values(self, approval_system_no_deps: ApprovalSystem) -> None:
        """Initial metrics should all be zero."""
        metrics = approval_system_no_deps.get_metrics()

        assert metrics["total_requests"] == 0
        assert metrics["total_approved"] == 0
        assert metrics["total_rejected"] == 0
        assert metrics["total_escalated"] == 0
        assert metrics["pending_count"] == 0
        assert metrics["completed_count"] == 0


# =========================================================================
# Test Class 9 — ApprovalRequest Model
# =========================================================================


class TestApprovalRequestModel:
    """Verify ApprovalRequest Pydantic V2 model defaults and enum values."""

    def test_approval_request_defaults(self) -> None:
        """ApprovalRequest with minimal fields should have sensible defaults."""
        request = ApprovalRequest(
            transaction_type="purchase_order",
            amount=5000.00,
            approver_role="purchasing_manager",
        )

        assert isinstance(request.request_id, UUID)
        # Default decision is PENDING (stored as string with use_enum_values)
        assert request.decision in ("pending", ApprovalDecision.PENDING, ApprovalDecision.PENDING.value)
        assert isinstance(request.created_at, datetime)
        assert request.decision_notes is None
        assert request.decision_timestamp is None
        assert request.escalated_from is None
        assert request.metadata == {}

    def test_approval_decision_enum_values(self) -> None:
        """ApprovalDecision must have exactly APPROVED, REJECTED, ESCALATED, PENDING."""
        expected_members = {"APPROVED", "REJECTED", "ESCALATED", "PENDING"}
        actual_members = {member.name for member in ApprovalDecision}
        assert actual_members == expected_members

        # Verify string values match expected serialisation
        assert ApprovalDecision.APPROVED.value == "approved"
        assert ApprovalDecision.REJECTED.value == "rejected"
        assert ApprovalDecision.ESCALATED.value == "escalated"
        assert ApprovalDecision.PENDING.value == "pending"

    def test_approval_request_transaction_id_stored_as_uuid(self) -> None:
        """transaction_id should accept both UUID and None."""
        # With UUID
        uid = uuid4()
        req = ApprovalRequest(
            transaction_type="vendor_invoice",
            amount=10000.00,
            approver_role="ap_manager",
            transaction_id=uid,
        )
        assert req.transaction_id == uid

        # Without UUID (None by default)
        req2 = ApprovalRequest(
            transaction_type="vendor_invoice",
            amount=10000.00,
            approver_role="ap_manager",
        )
        assert req2.transaction_id is None

    def test_approval_request_with_escalated_from(self) -> None:
        """escalated_from field should store the originating request UUID."""
        original_id = uuid4()
        req = ApprovalRequest(
            transaction_type="purchase_order",
            amount=50000.00,
            approver_role="controller",
            escalated_from=original_id,
        )
        assert req.escalated_from == original_id


# =========================================================================
# Test Class 10 — Integration Scenarios (Multi-step Flows)
# =========================================================================


class TestIntegrationScenarios:
    """End-to-end multi-step scenarios combining creation, processing, escalation."""

    @pytest.mark.asyncio
    async def test_full_escalation_chain_po(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Full PO escalation: purchasing_manager → controller → cfo (rejected at each)."""
        fixed_txn_id = str(uuid4())
        txn = make_transaction("purchase_order", 15000.00, transaction_id=fixed_txn_id)

        # Step 1: Create at purchasing_manager
        req1 = await approval_system.create_approval_request(txn, "purchasing_manager")
        assert req1.approver_role == "purchasing_manager"

        # Step 2: Reject → escalate to controller
        await approval_system.process_approval(
            req1.request_id, ApprovalDecision.REJECTED, "Denied"
        )
        req2 = await approval_system.escalate_if_rejected(req1.request_id)
        assert req2 is not None
        assert req2.approver_role == "controller"

        # Step 3: Reject → escalate to cfo
        await approval_system.process_approval(
            req2.request_id, ApprovalDecision.REJECTED, "Still denied"
        )
        req3 = await approval_system.escalate_if_rejected(req2.request_id)
        assert req3 is not None
        assert req3.approver_role == "cfo"

        # Step 4: Reject at cfo → no further escalation
        await approval_system.process_approval(
            req3.request_id, ApprovalDecision.REJECTED, "Final rejection"
        )
        req4 = await approval_system.escalate_if_rejected(req3.request_id)
        assert req4 is None

        # Verify chain
        chain = approval_system.get_approval_chain(UUID(fixed_txn_id))
        assert len(chain) == 3  # Three requests in the chain

    @pytest.mark.asyncio
    async def test_metrics_after_multiple_operations(
        self, approval_system: ApprovalSystem
    ) -> None:
        """Metrics should accurately reflect a mix of approvals and rejections."""
        # Create and approve one
        txn1 = make_transaction("purchase_order", 50000.00)
        r1 = await approval_system.create_approval_request(txn1, "controller")
        await approval_system.process_approval(
            r1.request_id, ApprovalDecision.APPROVED, "OK"
        )

        # Create and reject another, then escalate
        txn2 = make_transaction("vendor_invoice", 30000.00)
        r2 = await approval_system.create_approval_request(txn2, "ap_manager")
        await approval_system.process_approval(
            r2.request_id, ApprovalDecision.REJECTED, "No"
        )
        r3 = await approval_system.escalate_if_rejected(r2.request_id)
        assert r3 is not None

        metrics = approval_system.get_metrics()
        # 2 direct creates + 1 escalation create = 3 total
        assert metrics["total_requests"] == 3
        assert metrics["total_approved"] == 1
        assert metrics["total_rejected"] == 1
        assert metrics["total_escalated"] == 1
        # r3 is still pending
        assert metrics["pending_count"] == 1
        # r1 (approved) + r2 (rejected) = 2 completed
        assert metrics["completed_count"] == 2

    @pytest.mark.asyncio
    async def test_no_event_bus_still_works(self) -> None:
        """ApprovalSystem without EventBus should function without errors."""
        system = ApprovalSystem()
        txn = make_transaction("purchase_order", 50000.00)
        request = await system.create_approval_request(txn, "controller")

        assert request is not None
        assert request.approver_role == "controller"

        updated = await system.process_approval(
            request.request_id, ApprovalDecision.APPROVED, "Fine"
        )
        assert updated.decision in ("approved", ApprovalDecision.APPROVED.value)


# =========================================================================
# Test Class 11 — Approval System Constructor
# =========================================================================


class TestApprovalSystemConstructor:
    """Verify constructor behaviour and custom thresholds."""

    def test_default_thresholds(self) -> None:
        """Without explicit thresholds, the system uses APPROVAL_THRESHOLDS."""
        system = ApprovalSystem()
        # Verify it uses the module-level thresholds
        result = system.get_required_approval("purchase_order", 5000.00)
        assert result == "purchasing_manager"

    def test_custom_thresholds(self) -> None:
        """Custom thresholds override the defaults."""
        custom = {
            "custom_type": [
                (0, 1000, None),
                (1000, None, "custom_approver"),
            ]
        }
        system = ApprovalSystem(thresholds=custom)

        assert system.get_required_approval("custom_type", 500) is None
        assert system.get_required_approval("custom_type", 1000) == "custom_approver"
        # Original types not available with custom-only thresholds
        assert system.get_required_approval("purchase_order", 5000) is None

    def test_metrics_initialized_to_zero(self) -> None:
        """Fresh system should have all-zero metrics."""
        system = ApprovalSystem()
        assert system.metrics["total_requests"] == 0
        assert system.metrics["total_approved"] == 0
        assert system.metrics["total_rejected"] == 0
        assert system.metrics["total_escalated"] == 0
