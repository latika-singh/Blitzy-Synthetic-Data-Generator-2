"""Comprehensive unit tests for TransactionOrchestrator.

Tests cover:
- Required artifacts constant validation (README.md lines 349-355)
- P3 extended REQUIRED_ARTIFACTS validation (accrual, period_close,
  depreciation, recurring_journal, extended vendor_invoice, goods_receipt)
- Transaction registration and state creation (P2 and P3 types)
- Artifact addition and status transitions
- Completeness validation (including P3 types)
- Mark complete (async) with event publishing
- Transaction chaining (parent-child relationships, including P3 chains)
- Failure and cancellation workflows
- Query methods (get_transaction, get_incomplete, get_by_type, metrics)
- TransactionState and TransactionStatus Pydantic V2 models
- Full lifecycle integration tests (P2 and P3 types)

Per AAP Section 0.7.5:
- All external dependencies are mocked (EventBus via AsyncMock)
- Async tests use pytest-asyncio
- No live API calls, no external Redis
- Coverage target: >= 80%
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.orchestration.transaction_orchestrator import (
    REQUIRED_ARTIFACTS,
    TransactionArtifact,
    TransactionOrchestrator,
    TransactionState,
    TransactionStatus,
)


# ---------------------------------------------------------------------------
# Helper functions — reusable sample transaction data
# ---------------------------------------------------------------------------


def sample_po_transaction() -> Dict[str, Any]:
    """Return sample purchase order transaction data for tests."""
    return {
        "description": "Test PO",
        "vendor_id": "V-001",
        "amount": 25000.00,
        "lines": [{"item": "Widget", "qty": 100}],
    }


def sample_invoice_transaction() -> Dict[str, Any]:
    """Return sample vendor invoice transaction data for tests."""
    return {
        "description": "Test Invoice",
        "vendor_id": "V-001",
        "invoice_number": "INV-001",
        "amount": 15000.00,
    }


def sample_payment_transaction() -> Dict[str, Any]:
    """Return sample vendor payment transaction data for tests."""
    return {
        "description": "Test Payment",
        "vendor_id": "V-001",
        "amount": 15000.00,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_event_bus() -> AsyncMock:
    """Create a mocked EventBus with an async publish method."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def orchestrator(mock_event_bus: AsyncMock) -> TransactionOrchestrator:
    """TransactionOrchestrator wired with a mocked EventBus."""
    return TransactionOrchestrator(event_bus=mock_event_bus)


@pytest.fixture
def orchestrator_no_events() -> TransactionOrchestrator:
    """TransactionOrchestrator without an EventBus — for simpler tests."""
    return TransactionOrchestrator()


# =========================================================================
# Test Class — Required Artifacts Constant Validation
# =========================================================================


class TestRequiredArtifacts:
    """Verify the REQUIRED_ARTIFACTS constant matches the README spec exactly."""

    def test_purchase_order_requires_three_artifacts(self) -> None:
        """PO requires po_header, po_lines, approval (README.md line 350)."""
        assert REQUIRED_ARTIFACTS["purchase_order"] == [
            "po_header",
            "po_lines",
            "approval",
        ]

    def test_vendor_invoice_requires_five_artifacts(self) -> None:
        """Vendor invoice requires header, lines, 3-way match, match result, GL entries."""
        assert REQUIRED_ARTIFACTS["vendor_invoice"] == [
            "invoice_header",
            "invoice_lines",
            "three_way_match",
            "three_way_match_result",  # P3: ThreeWayMatcher structured result
            "gl_entries",
        ]

    def test_vendor_payment_requires_three_artifacts(self) -> None:
        """Vendor payment requires payment_record, allocation, GL entries."""
        assert REQUIRED_ARTIFACTS["vendor_payment"] == [
            "payment_record",
            "payment_allocation",
            "gl_entries",
        ]

    def test_all_required_types_present(self) -> None:
        """All three primary transaction types are defined in the mapping."""
        assert "purchase_order" in REQUIRED_ARTIFACTS
        assert "vendor_invoice" in REQUIRED_ARTIFACTS
        assert "vendor_payment" in REQUIRED_ARTIFACTS

    def test_required_artifacts_is_dict(self) -> None:
        """REQUIRED_ARTIFACTS should be a dict of str -> list[str]."""
        assert isinstance(REQUIRED_ARTIFACTS, dict)
        for key, value in REQUIRED_ARTIFACTS.items():
            assert isinstance(key, str)
            assert isinstance(value, list)
            for item in value:
                assert isinstance(item, str)


# =========================================================================
# Test Class — P3 Required Artifacts Validation (AAP Section 0.5.1)
# =========================================================================


class TestP3RequiredArtifacts:
    """Verify REQUIRED_ARTIFACTS includes P3 transaction types and extensions.

    Per AAP Section 0.5.1:
    - vendor_invoice now includes three_way_match_result
    - goods_receipt now includes gl_entries
    - New types: accrual, period_close, depreciation, recurring_journal
    """

    def test_p3_transaction_types_present(self) -> None:
        """All P3 transaction types must be defined in REQUIRED_ARTIFACTS."""
        p3_types = {"accrual", "period_close", "depreciation", "recurring_journal"}
        for txn_type in p3_types:
            assert txn_type in REQUIRED_ARTIFACTS, (
                f"P3 transaction type '{txn_type}' missing from REQUIRED_ARTIFACTS"
            )

    def test_accrual_requires_four_artifacts(self) -> None:
        """Accrual requires accrual_entry, je_header, je_lines, gl_entries."""
        assert REQUIRED_ARTIFACTS["accrual"] == [
            "accrual_entry",
            "je_header",
            "je_lines",
            "gl_entries",
        ]

    def test_period_close_requires_three_artifacts(self) -> None:
        """Period close requires period_close_summary, trial_balance, gl_entries."""
        assert REQUIRED_ARTIFACTS["period_close"] == [
            "period_close_summary",
            "trial_balance",
            "gl_entries",
        ]

    def test_depreciation_requires_three_artifacts(self) -> None:
        """Depreciation requires je_header, je_lines, gl_entries."""
        assert REQUIRED_ARTIFACTS["depreciation"] == [
            "je_header",
            "je_lines",
            "gl_entries",
        ]

    def test_recurring_journal_requires_three_artifacts(self) -> None:
        """Recurring journal requires je_header, je_lines, gl_entries."""
        assert REQUIRED_ARTIFACTS["recurring_journal"] == [
            "je_header",
            "je_lines",
            "gl_entries",
        ]

    def test_goods_receipt_now_requires_gl_entries(self) -> None:
        """P3 extended goods_receipt includes gl_entries for inventory posting."""
        assert "gl_entries" in REQUIRED_ARTIFACTS["goods_receipt"]
        assert REQUIRED_ARTIFACTS["goods_receipt"] == [
            "receipt_header",
            "receipt_lines",
            "gl_entries",
        ]

    def test_vendor_invoice_includes_three_way_match_result(self) -> None:
        """P3 extended vendor_invoice includes three_way_match_result artifact."""
        assert "three_way_match_result" in REQUIRED_ARTIFACTS["vendor_invoice"]
        assert "three_way_match" in REQUIRED_ARTIFACTS["vendor_invoice"]  # Original still present

    def test_p3_artifacts_include_expected_p3_specific_keys(self) -> None:
        """P3-specific artifact keys exist across the REQUIRED_ARTIFACTS mapping."""
        all_artifacts = []
        for artifacts_list in REQUIRED_ARTIFACTS.values():
            all_artifacts.extend(artifacts_list)
        # P3-specific artifact types must appear somewhere
        assert "three_way_match_result" in all_artifacts
        assert "accrual_entry" in all_artifacts
        assert "period_close_summary" in all_artifacts
        assert "trial_balance" in all_artifacts

    def test_total_transaction_types_count(self) -> None:
        """REQUIRED_ARTIFACTS should have 12 transaction types (8 original + 4 P3 new)."""
        assert len(REQUIRED_ARTIFACTS) == 12

    def test_original_types_unchanged(self) -> None:
        """All 8 original P2 transaction types still present with correct artifacts."""
        # purchase_order unchanged
        assert REQUIRED_ARTIFACTS["purchase_order"] == ["po_header", "po_lines", "approval"]
        # vendor_payment unchanged
        assert REQUIRED_ARTIFACTS["vendor_payment"] == ["payment_record", "payment_allocation", "gl_entries"]
        # sales_order unchanged
        assert REQUIRED_ARTIFACTS["sales_order"] == ["order_header", "order_lines"]
        # customer_invoice unchanged
        assert REQUIRED_ARTIFACTS["customer_invoice"] == ["invoice_header", "invoice_lines", "gl_entries"]
        # customer_payment unchanged
        assert REQUIRED_ARTIFACTS["customer_payment"] == ["payment_record", "payment_allocation", "gl_entries"]
        # journal_entry unchanged
        assert REQUIRED_ARTIFACTS["journal_entry"] == ["je_header", "je_lines", "approval"]

    def test_all_artifact_values_are_non_empty_lists(self) -> None:
        """Every REQUIRED_ARTIFACTS entry must have at least one artifact."""
        for txn_type, artifacts in REQUIRED_ARTIFACTS.items():
            assert len(artifacts) > 0, f"Transaction type '{txn_type}' has empty artifact list"


# =========================================================================
# Test Class — Transaction Registration
# =========================================================================


class TestTransactionRegistration:
    """Tests for register_transaction: UUID generation, state creation, linking."""

    def test_register_transaction_returns_uuid(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Registration must return a valid UUID4 identifier."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        assert isinstance(txn_id, UUID)

    def test_register_transaction_creates_state(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """A TransactionState must exist after registration."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_type == "purchase_order"
        assert state.status == TransactionStatus.REGISTERED.value

    def test_register_transaction_sets_required_artifacts(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Required artifacts must be auto-resolved from REQUIRED_ARTIFACTS."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.required_artifacts == ["po_header", "po_lines", "approval"]

    def test_register_vendor_invoice_sets_correct_artifacts(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Vendor invoice required artifacts match README specification (extended for P3)."""
        txn_id = orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.required_artifacts == [
            "invoice_header",
            "invoice_lines",
            "three_way_match",
            "three_way_match_result",  # P3: ThreeWayMatcher structured result
            "gl_entries",
        ]

    def test_register_unknown_type_empty_artifacts(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Unknown transaction types should get an empty required artifacts list."""
        txn_id = orchestrator.register_transaction(
            {"desc": "test"}, "unknown_type"
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.required_artifacts == []

    def test_register_increments_metrics(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Each registration must bump total_registered counter."""
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        assert orchestrator.metrics["total_registered"] >= 1
        orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        assert orchestrator.metrics["total_registered"] >= 2

    def test_register_with_parent_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Parent-child linking must be bidirectional."""
        parent_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        child_id = orchestrator.register_transaction(
            sample_invoice_transaction(),
            "vendor_invoice",
            parent_transaction_id=parent_id,
        )
        child_state = orchestrator.get_transaction(child_id)
        assert child_state is not None
        assert child_state.parent_transaction_id == parent_id

        parent_state = orchestrator.get_transaction(parent_id)
        assert parent_state is not None
        assert child_id in parent_state.child_transaction_ids

    def test_register_with_invalid_parent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Referencing a non-existent parent must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.register_transaction(
                sample_po_transaction(),
                "purchase_order",
                parent_transaction_id=uuid4(),
            )

    def test_register_stores_transaction_data(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Original transaction payload should be stored in state."""
        data = sample_po_transaction()
        txn_id = orchestrator.register_transaction(data, "purchase_order")
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_data == data

    def test_register_sets_created_at_timestamp(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Registration must set a UTC created_at timestamp."""
        before = datetime.now(timezone.utc)
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        after = datetime.now(timezone.utc)
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert isinstance(state.created_at, datetime)
        assert before <= state.created_at <= after

    def test_register_multiple_transactions_unique_ids(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Each registration must produce a unique UUID."""
        ids = [
            orchestrator.register_transaction(
                sample_po_transaction(), "purchase_order"
            )
            for _ in range(10)
        ]
        assert len(set(ids)) == 10


# =========================================================================
# Test Class — P3 Transaction Registration (AAP Section 0.5.1)
# =========================================================================


class TestP3TransactionRegistration:
    """Tests for registering P3-specific transaction types."""

    def test_register_accrual_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Accrual transaction registers with correct required artifacts."""
        txn_id = orchestrator.register_transaction(
            {"description": "GRNI Accrual", "amount": 5000.00, "fiscal_period": "2025-01"},
            "accrual",
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_type == "accrual"
        assert state.required_artifacts == [
            "accrual_entry", "je_header", "je_lines", "gl_entries"
        ]

    def test_register_period_close_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Period close transaction registers with correct required artifacts."""
        txn_id = orchestrator.register_transaction(
            {"description": "January 2025 Period Close", "fiscal_period": "2025-01"},
            "period_close",
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_type == "period_close"
        assert state.required_artifacts == [
            "period_close_summary", "trial_balance", "gl_entries"
        ]

    def test_register_depreciation_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Depreciation transaction registers with correct required artifacts."""
        txn_id = orchestrator.register_transaction(
            {"description": "Monthly depreciation", "amount": 1500.00},
            "depreciation",
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_type == "depreciation"
        assert state.required_artifacts == ["je_header", "je_lines", "gl_entries"]

    def test_register_recurring_journal_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Recurring journal transaction registers with correct required artifacts."""
        txn_id = orchestrator.register_transaction(
            {"description": "Monthly rent expense", "amount": 3000.00},
            "recurring_journal",
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_type == "recurring_journal"
        assert state.required_artifacts == ["je_header", "je_lines", "gl_entries"]

    def test_register_goods_receipt_with_gl_entries(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """P3-extended goods_receipt now requires gl_entries alongside receipt artifacts."""
        txn_id = orchestrator.register_transaction(
            {"description": "Receipt for PO-001", "po_id": "PO-001"},
            "goods_receipt",
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert "gl_entries" in state.required_artifacts
        assert "receipt_header" in state.required_artifacts
        assert "receipt_lines" in state.required_artifacts


# =========================================================================
# Test Class — Artifact Addition
# =========================================================================


class TestArtifactAddition:
    """Tests for add_artifact: storage, status transitions, metrics."""

    def test_add_artifact_succeeds(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """add_artifact should return True on success."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        result = orchestrator.add_artifact(
            txn_id, "po_header", {"po_number": "PO-001", "date": "2024-01-15"}
        )
        assert result is True

    def test_add_artifact_stored_in_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Artifact must be retrievable from the transaction state."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"po_number": "PO-001"})
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert "po_header" in state.artifacts
        artifact = state.artifacts["po_header"]
        assert artifact.artifact_data["po_number"] == "PO-001"
        assert artifact.artifact_type == "po_header"

    def test_add_artifact_updates_status_to_in_progress(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """First artifact should move status from REGISTERED to IN_PROGRESS."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.REGISTERED.value

        orchestrator.add_artifact(txn_id, "po_header", {"po_number": "PO-001"})
        # Re-fetch to see updated status
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.IN_PROGRESS.value

    def test_add_artifact_with_agent_id(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Agent ID should be recorded on the artifact."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        agent_id = uuid4()
        orchestrator.add_artifact(
            txn_id, "po_header", {"data": "test"}, agent_id=agent_id
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.artifacts["po_header"].created_by_agent_id == agent_id

    def test_add_artifact_nonexistent_transaction_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Adding to a non-existent transaction must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.add_artifact(uuid4(), "po_header", {"data": "test"})

    def test_add_artifact_increments_metrics(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Each artifact addition bumps the artifacts_added counter."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        assert orchestrator.metrics["artifacts_added"] >= 1
        orchestrator.add_artifact(txn_id, "po_lines", {"data": "test"})
        assert orchestrator.metrics["artifacts_added"] >= 2

    def test_add_multiple_artifacts(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Multiple artifacts are stored independently."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"po_number": "PO-001"})
        orchestrator.add_artifact(
            txn_id, "po_lines", {"lines": [{"item": "Widget", "qty": 100}]}
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert len(state.artifacts) == 2
        assert "po_header" in state.artifacts
        assert "po_lines" in state.artifacts

    def test_add_artifact_overwrites_existing(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Re-adding the same artifact type should overwrite the previous one."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"version": 1})
        orchestrator.add_artifact(txn_id, "po_header", {"version": 2})
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.artifacts["po_header"].artifact_data["version"] == 2

    def test_add_artifact_sets_created_at(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Each artifact should carry a UTC timestamp."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        before = datetime.now(timezone.utc)
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        after = datetime.now(timezone.utc)
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        artifact = state.artifacts["po_header"]
        assert isinstance(artifact.created_at, datetime)
        assert before <= artifact.created_at <= after

    def test_add_artifact_without_agent_id(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Agent ID defaults to None when not provided."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.artifacts["po_header"].created_by_agent_id is None


# =========================================================================
# Test Class — Completeness Validation
# =========================================================================


class TestCompletenessValidation:
    """Tests for check_completeness and get_missing_artifacts."""

    def test_incomplete_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Transaction with partial artifacts is incomplete."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is False

    def test_complete_purchase_order(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """PO with all 3 required artifacts is complete."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        orchestrator.add_artifact(txn_id, "po_lines", {"data": "test"})
        orchestrator.add_artifact(txn_id, "approval", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_complete_vendor_invoice(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Vendor invoice with all 5 required artifacts is complete (extended for P3)."""
        txn_id = orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        for artifact_type in [
            "invoice_header",
            "invoice_lines",
            "three_way_match",
            "three_way_match_result",  # P3: ThreeWayMatcher structured result
            "gl_entries",
        ]:
            orchestrator.add_artifact(txn_id, artifact_type, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_complete_vendor_payment(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Vendor payment with all 3 required artifacts is complete."""
        txn_id = orchestrator.register_transaction(
            sample_payment_transaction(), "vendor_payment"
        )
        for artifact_type in ["payment_record", "payment_allocation", "gl_entries"]:
            orchestrator.add_artifact(txn_id, artifact_type, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_incomplete_missing_one_artifact(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Missing a single required artifact should still be incomplete."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        orchestrator.add_artifact(txn_id, "po_lines", {"data": "test"})
        # Missing "approval"
        assert orchestrator.check_completeness(txn_id) is False

    def test_check_completeness_nonexistent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Checking a non-existent transaction must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.check_completeness(uuid4())

    def test_get_missing_artifacts(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """get_missing_artifacts returns only artifacts not yet provided."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        missing = orchestrator.get_missing_artifacts(txn_id)
        assert "po_lines" in missing
        assert "approval" in missing
        assert "po_header" not in missing

    def test_get_missing_artifacts_all_present(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """When all artifacts are present, missing list should be empty."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        missing = orchestrator.get_missing_artifacts(txn_id)
        assert missing == []

    def test_get_missing_artifacts_nonexistent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """get_missing_artifacts for a non-existent transaction raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.get_missing_artifacts(uuid4())

    def test_no_required_artifacts_always_complete(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Unknown type with no required artifacts is immediately complete."""
        txn_id = orchestrator.register_transaction(
            {"desc": "test"}, "unknown_type"
        )
        assert orchestrator.check_completeness(txn_id) is True

    def test_extra_artifacts_do_not_break_completeness(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Extra artifacts beyond required should not affect completeness."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        # Add extra non-required artifact
        orchestrator.add_artifact(txn_id, "extra_notes", {"data": "bonus"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_get_missing_artifacts_returns_sorted(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Missing artifacts list should be sorted alphabetically."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        # No artifacts added — all 3 are missing
        missing = orchestrator.get_missing_artifacts(txn_id)
        assert missing == sorted(missing)


# =========================================================================
# Test Class — P3 Completeness Validation (AAP Section 0.5.1)
# =========================================================================


class TestP3CompletenessValidation:
    """Tests for P3 transaction type artifact completeness checking."""

    def test_complete_accrual_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Accrual with all 4 required artifacts is complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "GRNI Accrual"}, "accrual"
        )
        for art in ["accrual_entry", "je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_incomplete_accrual_missing_gl_entries(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Accrual missing gl_entries is incomplete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Accrual"}, "accrual"
        )
        for art in ["accrual_entry", "je_header", "je_lines"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is False
        missing = orchestrator.get_missing_artifacts(txn_id)
        assert "gl_entries" in missing

    def test_complete_period_close_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Period close with all 3 required artifacts is complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Period Close"}, "period_close"
        )
        for art in ["period_close_summary", "trial_balance", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_incomplete_period_close_missing_trial_balance(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Period close missing trial_balance is incomplete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Period Close"}, "period_close"
        )
        orchestrator.add_artifact(txn_id, "period_close_summary", {"data": "test"})
        orchestrator.add_artifact(txn_id, "gl_entries", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is False
        missing = orchestrator.get_missing_artifacts(txn_id)
        assert "trial_balance" in missing

    def test_complete_depreciation_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Depreciation with all 3 required artifacts is complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Depreciation"}, "depreciation"
        )
        for art in ["je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_complete_recurring_journal_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Recurring journal with all 3 required artifacts is complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Recurring JE"}, "recurring_journal"
        )
        for art in ["je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_complete_goods_receipt_with_gl_entries(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """P3-extended goods_receipt requires gl_entries to be complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "GR"}, "goods_receipt"
        )
        # Only receipt_header and receipt_lines — missing gl_entries
        orchestrator.add_artifact(txn_id, "receipt_header", {"data": "test"})
        orchestrator.add_artifact(txn_id, "receipt_lines", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is False
        # Add gl_entries to complete
        orchestrator.add_artifact(txn_id, "gl_entries", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True

    def test_p3_types_with_extra_artifacts_still_complete(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """P3 transactions with extra non-required artifacts remain complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Accrual"}, "accrual"
        )
        for art in ["accrual_entry", "je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})
        # Add extra artifact
        orchestrator.add_artifact(txn_id, "ground_truth", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True


# =========================================================================
# Test Class — Mark Complete (async)
# =========================================================================


class TestMarkComplete:
    """Tests for the async mark_complete method: validation, events, metrics."""

    @pytest.mark.asyncio
    async def test_mark_complete_success(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Fully-complete transaction should be markable as complete."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})

        result = await orchestrator.mark_complete(txn_id)
        assert result is True

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.COMPLETE.value
        assert state.completed_at is not None
        assert isinstance(state.completed_at, datetime)

    @pytest.mark.asyncio
    async def test_mark_complete_fails_when_incomplete(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Incomplete transaction must fail mark_complete (return False)."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})
        # Missing po_lines and approval

        result = await orchestrator.mark_complete(txn_id)
        assert result is False

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status != TransactionStatus.COMPLETE.value

    @pytest.mark.asyncio
    async def test_mark_complete_increments_metrics(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Successful completion must bump total_completed counter."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})

        await orchestrator.mark_complete(txn_id)
        assert orchestrator.metrics["total_completed"] >= 1

    @pytest.mark.asyncio
    async def test_mark_complete_publishes_event(
        self,
        orchestrator: TransactionOrchestrator,
        mock_event_bus: AsyncMock,
    ) -> None:
        """mark_complete should publish a TransactionCompleted event."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})

        await orchestrator.mark_complete(txn_id)
        mock_event_bus.publish.assert_awaited_once()
        # Verify the published event type
        published_event = mock_event_bus.publish.call_args[0][0]
        from app.events.event_types import TransactionCompleted as TC

        assert isinstance(published_event, TC)

    @pytest.mark.asyncio
    async def test_mark_complete_nonexistent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Marking a non-existent transaction must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.mark_complete(uuid4())

    @pytest.mark.asyncio
    async def test_mark_complete_sets_completed_at_utc(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """completed_at must be a timezone-aware UTC datetime."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(txn_id, art, {"data": "test"})

        before = datetime.now(timezone.utc)
        await orchestrator.mark_complete(txn_id)
        after = datetime.now(timezone.utc)

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.completed_at is not None
        assert before <= state.completed_at <= after

    @pytest.mark.asyncio
    async def test_mark_complete_no_event_bus(
        self, orchestrator_no_events: TransactionOrchestrator
    ) -> None:
        """mark_complete without event_bus should succeed silently."""
        txn_id = orchestrator_no_events.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator_no_events.add_artifact(txn_id, art, {"data": "test"})

        result = await orchestrator_no_events.mark_complete(txn_id)
        assert result is True

        state = orchestrator_no_events.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.COMPLETE.value

    @pytest.mark.asyncio
    async def test_mark_complete_failed_does_not_increment_metrics(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Failed mark_complete should not bump total_completed."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        # Incomplete — only 1 of 3 artifacts
        orchestrator.add_artifact(txn_id, "po_header", {"data": "test"})

        initial_completed = orchestrator.metrics["total_completed"]
        await orchestrator.mark_complete(txn_id)
        assert orchestrator.metrics["total_completed"] == initial_completed


# =========================================================================
# Test Class — Transaction Chain
# =========================================================================


class TestTransactionChain:
    """Tests for get_transaction_chain: parent-child traversal."""

    def test_get_transaction_chain_single(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """A lone transaction produces a single-element chain."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        chain = orchestrator.get_transaction_chain(txn_id)
        assert len(chain) == 1
        assert chain[0].transaction_id == txn_id

    def test_get_transaction_chain_parent_child(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Two-level chain: parent first, then child."""
        parent_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        child_id = orchestrator.register_transaction(
            sample_invoice_transaction(),
            "vendor_invoice",
            parent_transaction_id=parent_id,
        )
        chain = orchestrator.get_transaction_chain(parent_id)
        assert len(chain) == 2
        assert chain[0].transaction_id == parent_id
        assert chain[1].transaction_id == child_id

    def test_get_transaction_chain_from_child(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Querying from child should still find the root and full chain."""
        parent_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        child_id = orchestrator.register_transaction(
            sample_invoice_transaction(),
            "vendor_invoice",
            parent_transaction_id=parent_id,
        )
        chain = orchestrator.get_transaction_chain(child_id)
        assert len(chain) == 2
        # Root should appear first
        assert chain[0].transaction_id == parent_id
        assert chain[1].transaction_id == child_id

    def test_get_transaction_chain_multi_level(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Three-level chain: root → mid → leaf."""
        root_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        mid_id = orchestrator.register_transaction(
            sample_invoice_transaction(),
            "vendor_invoice",
            parent_transaction_id=root_id,
        )
        leaf_id = orchestrator.register_transaction(
            sample_payment_transaction(),
            "vendor_payment",
            parent_transaction_id=mid_id,
        )
        chain = orchestrator.get_transaction_chain(root_id)
        assert len(chain) == 3
        chain_ids = [s.transaction_id for s in chain]
        assert chain_ids[0] == root_id
        assert mid_id in chain_ids
        assert leaf_id in chain_ids

    def test_get_transaction_chain_nonexistent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Querying chain for non-existent ID must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.get_transaction_chain(uuid4())

    def test_get_transaction_chain_multiple_children(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Parent with multiple children should return all in chain."""
        parent_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        child1_id = orchestrator.register_transaction(
            sample_invoice_transaction(),
            "vendor_invoice",
            parent_transaction_id=parent_id,
        )
        child2_id = orchestrator.register_transaction(
            sample_payment_transaction(),
            "vendor_payment",
            parent_transaction_id=parent_id,
        )
        chain = orchestrator.get_transaction_chain(parent_id)
        assert len(chain) == 3
        chain_ids = [s.transaction_id for s in chain]
        assert parent_id in chain_ids
        assert child1_id in chain_ids
        assert child2_id in chain_ids


# =========================================================================
# Test Class — Failure and Cancellation
# =========================================================================


class TestFailureAndCancellation:
    """Tests for mark_failed and cancel_transaction."""

    def test_mark_failed(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """mark_failed should set status to FAILED and bump metrics."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.mark_failed(txn_id, "Validation error")
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.FAILED.value
        assert orchestrator.metrics["total_failed"] >= 1

    def test_mark_failed_stores_reason(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Failure reason should be stored in metadata."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.mark_failed(txn_id, "Budget exceeded")
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.metadata.get("failure_reason") == "Budget exceeded"

    def test_mark_failed_nonexistent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Failing a non-existent transaction must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.mark_failed(uuid4(), "reason")

    def test_cancel_transaction(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """cancel_transaction should set status to CANCELLED."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.cancel_transaction(txn_id, "User requested cancellation")
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.CANCELLED.value

    def test_cancel_transaction_stores_reason(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Cancellation reason should be stored in metadata."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.cancel_transaction(txn_id, "Duplicate")
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.metadata.get("cancellation_reason") == "Duplicate"

    def test_cancel_increments_cancelled_metrics(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Cancellation must bump total_cancelled counter."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.cancel_transaction(txn_id, "reason")
        assert orchestrator.metrics["total_cancelled"] >= 1

    def test_cancel_nonexistent_raises(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Cancelling a non-existent transaction must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator.cancel_transaction(uuid4(), "reason")


# =========================================================================
# Test Class — Query Methods
# =========================================================================


class TestQueryMethods:
    """Tests for get_transaction, get_incomplete, get_by_type, get_metrics."""

    def test_get_transaction_existing(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Retrieve an existing transaction by UUID."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.transaction_id == txn_id

    def test_get_transaction_nonexistent(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Non-existent UUID should return None (no exception)."""
        assert orchestrator.get_transaction(uuid4()) is None

    def test_get_incomplete_transactions(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """get_incomplete_transactions returns all non-terminal transactions."""
        id1 = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        id2 = orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        incomplete = orchestrator.get_incomplete_transactions()
        incomplete_ids = [t.transaction_id for t in incomplete]
        assert id1 in incomplete_ids
        assert id2 in incomplete_ids

    def test_get_incomplete_excludes_terminal(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """FAILED and CANCELLED transactions should not appear in incomplete."""
        id1 = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        id2 = orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        orchestrator.mark_failed(id1, "test failure")
        orchestrator.cancel_transaction(id2, "cancelled")
        incomplete = orchestrator.get_incomplete_transactions()
        incomplete_ids = [t.transaction_id for t in incomplete]
        assert id1 not in incomplete_ids
        assert id2 not in incomplete_ids

    def test_get_transactions_by_type(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Filter transactions by type."""
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        pos = orchestrator.get_transactions_by_type("purchase_order")
        assert len(pos) == 2

    def test_get_transactions_by_type_empty(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """No transactions of a given type should return empty list."""
        result = orchestrator.get_transactions_by_type("nonexistent_type")
        assert result == []

    def test_get_metrics_contains_required_keys(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """get_metrics must include all baseline counters."""
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        metrics = orchestrator.get_metrics()
        assert "total_registered" in metrics
        assert "total_completed" in metrics
        assert "total_failed" in metrics
        assert "total_cancelled" in metrics
        assert "artifacts_added" in metrics
        assert "active_transactions" in metrics
        assert "by_type" in metrics
        assert "by_status" in metrics
        assert "average_artifacts_per_transaction" in metrics

    def test_get_metrics_registration_count(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Registration count should match number of registered transactions."""
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        orchestrator.register_transaction(
            sample_payment_transaction(), "vendor_payment"
        )
        metrics = orchestrator.get_metrics()
        assert metrics["total_registered"] == 3

    def test_get_metrics_by_type_breakdown(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """by_type should count each transaction type."""
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        metrics = orchestrator.get_metrics()
        assert metrics["by_type"]["purchase_order"] == 2
        assert metrics["by_type"]["vendor_invoice"] == 1

    def test_get_metrics_empty_orchestrator(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Empty orchestrator should return zero counters."""
        metrics = orchestrator.get_metrics()
        assert metrics["total_registered"] == 0
        assert metrics["total_completed"] == 0
        assert metrics["active_transactions"] == 0
        assert metrics["average_artifacts_per_transaction"] == 0.0


# =========================================================================
# Test Class — TransactionState Model
# =========================================================================


class TestTransactionStateModel:
    """Tests for the TransactionState and TransactionStatus Pydantic models."""

    def test_transaction_state_creation(self) -> None:
        """TransactionState auto-generates UUID and timestamp."""
        state = TransactionState(transaction_type="purchase_order")
        assert isinstance(state.transaction_id, UUID)
        assert state.status == TransactionStatus.REGISTERED.value
        assert isinstance(state.created_at, datetime)
        assert state.completed_at is None

    def test_transaction_state_default_empty_artifacts(self) -> None:
        """Default artifacts dict should be empty."""
        state = TransactionState(transaction_type="purchase_order")
        assert state.artifacts == {}

    def test_transaction_state_default_empty_required(self) -> None:
        """Default required_artifacts should be empty list."""
        state = TransactionState(transaction_type="purchase_order")
        assert state.required_artifacts == []

    def test_transaction_state_default_no_parent(self) -> None:
        """Default parent_transaction_id should be None."""
        state = TransactionState(transaction_type="test")
        assert state.parent_transaction_id is None

    def test_transaction_state_default_empty_children(self) -> None:
        """Default child_transaction_ids should be empty list."""
        state = TransactionState(transaction_type="test")
        assert state.child_transaction_ids == []

    def test_transaction_status_enum_values(self) -> None:
        """Verify all expected status enum string values."""
        assert TransactionStatus.REGISTERED.value == "registered"
        assert TransactionStatus.IN_PROGRESS.value == "in_progress"
        assert TransactionStatus.COMPLETE.value == "complete"
        assert TransactionStatus.FAILED.value == "failed"
        assert TransactionStatus.CANCELLED.value == "cancelled"

    def test_transaction_status_awaiting_artifacts(self) -> None:
        """AWAITING_ARTIFACTS status exists in the enum."""
        assert TransactionStatus.AWAITING_ARTIFACTS.value == "awaiting_artifacts"

    def test_transaction_artifact_creation(self) -> None:
        """TransactionArtifact stores type, data, and timestamps."""
        artifact = TransactionArtifact(
            artifact_type="po_header",
            artifact_data={"po_number": "PO-001"},
        )
        assert artifact.artifact_type == "po_header"
        assert artifact.artifact_data["po_number"] == "PO-001"
        assert isinstance(artifact.created_at, datetime)
        assert artifact.created_by_agent_id is None

    def test_transaction_artifact_with_agent_id(self) -> None:
        """TransactionArtifact can carry a creating agent UUID."""
        agent_id = uuid4()
        artifact = TransactionArtifact(
            artifact_type="approval",
            artifact_data={"approved": True},
            created_by_agent_id=agent_id,
        )
        assert artifact.created_by_agent_id == agent_id

    def test_transaction_state_use_enum_values(self) -> None:
        """With use_enum_values=True, status is stored as string value."""
        state = TransactionState(
            transaction_type="test",
            status=TransactionStatus.IN_PROGRESS,
        )
        # use_enum_values=True converts to the string value
        assert state.status == "in_progress"
        assert isinstance(state.status, str)


# =========================================================================
# Test Class — Full Lifecycle Integration
# =========================================================================


class TestFullLifecycle:
    """End-to-end tests: register → add artifacts → check → mark complete."""

    @pytest.mark.asyncio
    async def test_full_po_lifecycle(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Complete PO lifecycle: register → add all → complete."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        assert orchestrator.check_completeness(txn_id) is False

        orchestrator.add_artifact(txn_id, "po_header", {"po_number": "PO-001"})
        assert orchestrator.check_completeness(txn_id) is False

        orchestrator.add_artifact(txn_id, "po_lines", {"lines": []})
        assert orchestrator.check_completeness(txn_id) is False

        orchestrator.add_artifact(txn_id, "approval", {"approved": True})
        assert orchestrator.check_completeness(txn_id) is True

        result = await orchestrator.mark_complete(txn_id)
        assert result is True

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.COMPLETE.value
        assert state.completed_at is not None

    @pytest.mark.asyncio
    async def test_full_vendor_invoice_lifecycle(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Complete vendor invoice lifecycle with all 5 artifacts (extended for P3)."""
        txn_id = orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        assert orchestrator.check_completeness(txn_id) is False

        for artifact_type in [
            "invoice_header",
            "invoice_lines",
            "three_way_match",
            "three_way_match_result",  # P3: ThreeWayMatcher structured result
            "gl_entries",
        ]:
            orchestrator.add_artifact(txn_id, artifact_type, {"data": artifact_type})

        assert orchestrator.check_completeness(txn_id) is True
        result = await orchestrator.mark_complete(txn_id)
        assert result is True

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.COMPLETE.value
        assert len(state.artifacts) == 5

    @pytest.mark.asyncio
    async def test_full_chained_lifecycle(
        self,
        orchestrator: TransactionOrchestrator,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Lifecycle for a PO → Invoice → Payment chain."""
        # Register PO
        po_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(po_id, art, {"data": "test"})
        await orchestrator.mark_complete(po_id)

        # Register Invoice linked to PO
        inv_id = orchestrator.register_transaction(
            sample_invoice_transaction(),
            "vendor_invoice",
            parent_transaction_id=po_id,
        )
        for art in ["invoice_header", "invoice_lines", "three_way_match", "three_way_match_result", "gl_entries"]:
            orchestrator.add_artifact(inv_id, art, {"data": "test"})
        await orchestrator.mark_complete(inv_id)

        # Register Payment linked to Invoice
        pay_id = orchestrator.register_transaction(
            sample_payment_transaction(),
            "vendor_payment",
            parent_transaction_id=inv_id,
        )
        for art in ["payment_record", "payment_allocation", "gl_entries"]:
            orchestrator.add_artifact(pay_id, art, {"data": "test"})
        await orchestrator.mark_complete(pay_id)

        # Verify full chain
        chain = orchestrator.get_transaction_chain(pay_id)
        assert len(chain) == 3

        # All should be complete
        for txn_state in chain:
            assert txn_state.status == TransactionStatus.COMPLETE.value

        # Metrics should reflect 3 completions
        assert orchestrator.metrics["total_completed"] == 3
        assert orchestrator.metrics["total_registered"] == 3

        # Event bus should have been called 3 times (once per mark_complete)
        assert mock_event_bus.publish.await_count == 3

    @pytest.mark.asyncio
    async def test_lifecycle_with_failure(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Lifecycle where a transaction fails mid-way."""
        txn_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        orchestrator.add_artifact(txn_id, "po_header", {"po_number": "PO-001"})

        # Simulate failure before completion
        orchestrator.mark_failed(txn_id, "Vendor not approved")

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.FAILED.value
        assert state.completed_at is None  # Never completed

        # Should not appear in incomplete list (FAILED is terminal)
        incomplete = orchestrator.get_incomplete_transactions()
        assert txn_id not in [t.transaction_id for t in incomplete]

    @pytest.mark.asyncio
    async def test_lifecycle_mixed_terminal_states(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Multiple transactions ending in different terminal states."""
        po_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        inv_id = orchestrator.register_transaction(
            sample_invoice_transaction(), "vendor_invoice"
        )
        pay_id = orchestrator.register_transaction(
            sample_payment_transaction(), "vendor_payment"
        )

        # Complete the PO
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(po_id, art, {"data": "test"})
        await orchestrator.mark_complete(po_id)

        # Fail the invoice
        orchestrator.mark_failed(inv_id, "Mismatch")

        # Cancel the payment
        orchestrator.cancel_transaction(pay_id, "No longer needed")

        # All should be terminal — no incomplete transactions
        incomplete = orchestrator.get_incomplete_transactions()
        assert len(incomplete) == 0

        # Verify metrics
        metrics = orchestrator.get_metrics()
        assert metrics["total_registered"] == 3
        assert metrics["total_completed"] == 1
        assert metrics["total_failed"] == 1
        assert metrics["total_cancelled"] == 1
        assert metrics["active_transactions"] == 0


# =========================================================================
# Test Class — P3 Full Lifecycle Integration (AAP Section 0.5.1)
# =========================================================================


class TestP3FullLifecycle:
    """End-to-end lifecycle tests for P3 transaction types."""

    @pytest.mark.asyncio
    async def test_full_accrual_lifecycle(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Complete accrual lifecycle: register → add all → complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "GRNI Accrual", "fiscal_period": "2025-01"},
            "accrual",
        )
        assert orchestrator.check_completeness(txn_id) is False

        for art in ["accrual_entry", "je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": art})

        assert orchestrator.check_completeness(txn_id) is True
        result = await orchestrator.mark_complete(txn_id)
        assert result is True

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.COMPLETE.value
        assert state.completed_at is not None
        assert len(state.artifacts) == 4

    @pytest.mark.asyncio
    async def test_full_period_close_lifecycle(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Complete period close lifecycle: register → add all → complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Jan 2025 Close", "fiscal_period": "2025-01"},
            "period_close",
        )
        assert orchestrator.check_completeness(txn_id) is False

        for art in ["period_close_summary", "trial_balance", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": art})

        assert orchestrator.check_completeness(txn_id) is True
        result = await orchestrator.mark_complete(txn_id)
        assert result is True

        state = orchestrator.get_transaction(txn_id)
        assert state is not None
        assert state.status == TransactionStatus.COMPLETE.value
        assert len(state.artifacts) == 3

    @pytest.mark.asyncio
    async def test_full_depreciation_lifecycle(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Complete depreciation lifecycle: register → add all → complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Monthly depreciation"}, "depreciation"
        )
        for art in ["je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": art})
        assert orchestrator.check_completeness(txn_id) is True
        result = await orchestrator.mark_complete(txn_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_full_recurring_journal_lifecycle(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """Complete recurring journal lifecycle: register → add all → complete."""
        txn_id = orchestrator.register_transaction(
            {"description": "Monthly rent"}, "recurring_journal"
        )
        for art in ["je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(txn_id, art, {"data": art})
        assert orchestrator.check_completeness(txn_id) is True
        result = await orchestrator.mark_complete(txn_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_p3_chained_period_close_lifecycle(
        self,
        orchestrator: TransactionOrchestrator,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Period close chain: accrual → depreciation → period_close."""
        # Register and complete accrual
        accrual_id = orchestrator.register_transaction(
            {"description": "GRNI Accrual"}, "accrual"
        )
        for art in ["accrual_entry", "je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(accrual_id, art, {"data": "test"})
        await orchestrator.mark_complete(accrual_id)

        # Register depreciation linked to accrual
        depr_id = orchestrator.register_transaction(
            {"description": "Depreciation"},
            "depreciation",
            parent_transaction_id=accrual_id,
        )
        for art in ["je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(depr_id, art, {"data": "test"})
        await orchestrator.mark_complete(depr_id)

        # Register period close linked to depreciation
        close_id = orchestrator.register_transaction(
            {"description": "Period Close"},
            "period_close",
            parent_transaction_id=depr_id,
        )
        for art in ["period_close_summary", "trial_balance", "gl_entries"]:
            orchestrator.add_artifact(close_id, art, {"data": "test"})
        await orchestrator.mark_complete(close_id)

        # Verify full chain
        chain = orchestrator.get_transaction_chain(close_id)
        assert len(chain) == 3
        for txn_state in chain:
            assert txn_state.status == TransactionStatus.COMPLETE.value

        # Metrics
        assert orchestrator.metrics["total_completed"] == 3
        assert mock_event_bus.publish.await_count == 3

    @pytest.mark.asyncio
    async def test_p3_goods_receipt_lifecycle_with_gl_entries(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """P3-extended goods_receipt lifecycle requires gl_entries for completion."""
        txn_id = orchestrator.register_transaction(
            {"description": "GR against PO-001"}, "goods_receipt"
        )
        # Incomplete without gl_entries
        orchestrator.add_artifact(txn_id, "receipt_header", {"data": "test"})
        orchestrator.add_artifact(txn_id, "receipt_lines", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is False

        # Complete with gl_entries
        orchestrator.add_artifact(txn_id, "gl_entries", {"data": "test"})
        assert orchestrator.check_completeness(txn_id) is True
        result = await orchestrator.mark_complete(txn_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_p3_mixed_p2_p3_lifecycle(
        self,
        orchestrator: TransactionOrchestrator,
        mock_event_bus: AsyncMock,
    ) -> None:
        """Lifecycle mixing P2 (PO) and P3 (accrual) transaction types."""
        # P2 PO lifecycle
        po_id = orchestrator.register_transaction(
            sample_po_transaction(), "purchase_order"
        )
        for art in ["po_header", "po_lines", "approval"]:
            orchestrator.add_artifact(po_id, art, {"data": "test"})
        await orchestrator.mark_complete(po_id)

        # P3 accrual lifecycle linked to PO
        accrual_id = orchestrator.register_transaction(
            {"description": "PO accrual"},
            "accrual",
            parent_transaction_id=po_id,
        )
        for art in ["accrual_entry", "je_header", "je_lines", "gl_entries"]:
            orchestrator.add_artifact(accrual_id, art, {"data": "test"})
        await orchestrator.mark_complete(accrual_id)

        # Both should be complete
        chain = orchestrator.get_transaction_chain(accrual_id)
        assert len(chain) == 2
        for txn in chain:
            assert txn.status == TransactionStatus.COMPLETE.value

        # Metrics
        metrics = orchestrator.get_metrics()
        assert metrics["total_registered"] == 2
        assert metrics["total_completed"] == 2
        assert metrics["by_type"]["purchase_order"] == 1
        assert metrics["by_type"]["accrual"] == 1

    def test_p3_metrics_by_type_breakdown(
        self, orchestrator: TransactionOrchestrator
    ) -> None:
        """P3 types appear correctly in get_metrics by_type breakdown."""
        orchestrator.register_transaction({"desc": "a"}, "accrual")
        orchestrator.register_transaction({"desc": "d"}, "depreciation")
        orchestrator.register_transaction({"desc": "p"}, "period_close")
        orchestrator.register_transaction({"desc": "r"}, "recurring_journal")
        orchestrator.register_transaction({"desc": "g"}, "goods_receipt")

        metrics = orchestrator.get_metrics()
        assert metrics["by_type"]["accrual"] == 1
        assert metrics["by_type"]["depreciation"] == 1
        assert metrics["by_type"]["period_close"] == 1
        assert metrics["by_type"]["recurring_journal"] == 1
        assert metrics["by_type"]["goods_receipt"] == 1
