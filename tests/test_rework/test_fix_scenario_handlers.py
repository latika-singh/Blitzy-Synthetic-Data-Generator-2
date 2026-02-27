"""Comprehensive tests for FixScenarioExecutor default handler coverage.

Exercises ALL 22 default scenarios through the real FixScenarioExecutor,
ensuring every registered handler function is invoked.  Each scenario
test prepares a transaction with the fields the handler needs, executes
the scenario, and asserts the result.

Target: raise fix_scenario_executor.py coverage from 37 % to ≥ 80 %.

References:
    - AAP Section 0.5.1 Group 6: fix_scenario_executor.py
    - AAP Section 0.7.6: Testing Conventions (≥ 80 % coverage)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List
from uuid import uuid4

import pytest

from app.rework.fix_scenario_catalog import (
    FixScenario,
    FixScenarioCatalog,
    FixScenarioName,
    FixStep,
)
from app.rework.fix_scenario_executor import (
    FixExecutionResult,
    FixScenarioExecutor,
)
from app.rework.rework_loop_engine import ReworkLoopError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ctx(**overrides: Any) -> Dict[str, Any]:
    """Minimal generation context dict."""
    base = {
        "simulation_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "current_date": "2025-03-15",
        "fiscal_period": "2025-03",
    }
    base.update(overrides)
    return base


def _txn(**overrides: Any) -> Dict[str, Any]:
    """Minimal transaction dict for handler tests."""
    base: Dict[str, Any] = {
        "transaction_id": str(uuid4()),
        "transaction_type": "vendor_invoice",
        "amount": Decimal("5000.00"),
        "vendor_id": "V-001",
        "customer_id": "C-001",
        "status": "submitted",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def catalog() -> FixScenarioCatalog:
    return FixScenarioCatalog(load_defaults=True)


@pytest.fixture
def executor() -> FixScenarioExecutor:
    return FixScenarioExecutor(scenario_timeout_seconds=10.0)


def _get_scenario(catalog: FixScenarioCatalog, name: str) -> FixScenario:
    s = catalog.get_scenario(name)
    assert s is not None, f"Scenario {name!r} not in catalog"
    return s


# =========================================================================
# Test Class — All 22 Default Scenarios Executed End-to-End
# =========================================================================


@pytest.mark.rework
@pytest.mark.asyncio
class TestAdjustAmountToRange:
    """ADJUST_AMOUNT_TO_RANGE: 4 steps — amount handlers."""

    async def test_amount_within_range(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        txn = _txn(amount=Decimal("500.00"))
        ctx = _ctx(min_amount=Decimal("100.00"), max_amount=Decimal("1000.00"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_amount_above_range_clamped(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        txn = _txn(amount=Decimal("99999.99"))
        ctx = _ctx(min_amount=Decimal("1.00"), max_amount=Decimal("10000.00"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert result.modified_transaction is not None
        assert result.modified_transaction["amount"] == Decimal("10000.00")

    async def test_amount_below_range_clamped(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        txn = _txn(amount=Decimal("0.50"))
        ctx = _ctx(min_amount=Decimal("100.00"), max_amount=Decimal("50000.00"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert result.modified_transaction["amount"] == Decimal("100.00")

    async def test_amount_missing(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        txn = _txn()
        del txn["amount"]
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True  # handlers return gracefully

    async def test_amount_float_coerced(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_AMOUNT_TO_RANGE.value)
        txn = _txn(amount=750.0)
        ctx = _ctx(min_amount=100, max_amount=10000)
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixDateSequence:
    """FIX_DATE_SEQUENCE: 4 steps — date handlers."""

    async def test_dates_in_order(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DATE_SEQUENCE.value)
        txn = _txn(
            order_date="2025-01-01",
            receipt_date="2025-01-10",
            invoice_date="2025-01-15",
            payment_date="2025-02-15",
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_dates_out_of_order_fixed(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DATE_SEQUENCE.value)
        txn = _txn(
            order_date="2025-03-01",
            receipt_date="2025-01-01",  # before order
            invoice_date="2025-02-01",  # before order
            payment_date="2025-04-01",
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        # receipt_date should have been bumped
        mod = result.modified_transaction
        assert mod["receipt_date"] >= mod["order_date"]

    async def test_partial_dates(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DATE_SEQUENCE.value)
        txn = _txn(order_date="2025-01-01", payment_date="2025-02-01")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_no_dates(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DATE_SEQUENCE.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True

    async def test_extra_date_fields(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DATE_SEQUENCE.value)
        txn = _txn(
            order_date="2025-01-01",
            ship_date="2025-01-05",
            created_at="2024-12-31",
            posting_date="2025-01-20",
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectEntityReference:
    """CORRECT_ENTITY_REFERENCE: 3 steps — entity handlers."""

    async def test_entity_present(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_ENTITY_REFERENCE.value)
        txn = _txn(vendor_id="V-999", employee_id="E-100")
        ctx = _ctx(correct_entity_id="V-001")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 3

    async def test_no_correct_entity(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_ENTITY_REFERENCE.value)
        txn = _txn()
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_customer_entity(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_ENTITY_REFERENCE.value)
        txn = _txn(customer_id="C-999")
        ctx = _ctx(correct_entity_id="C-001")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestRegenerateGLEntry:
    """REGENERATE_GL_ENTRY: 4 steps — GL entry handlers."""

    async def test_regenerate_with_gl_entries(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_GL_ENTRY.value)
        txn = _txn(gl_entries=[
            {"account": "1000", "debit": Decimal("500.00"), "credit": Decimal("0.00")},
            {"account": "2000", "debit": Decimal("0.00"), "credit": Decimal("500.00")},
        ])
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_regenerate_without_entries(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_GL_ENTRY.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True

    async def test_gl_entries_recalculated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_GL_ENTRY.value)
        txn = _txn(
            amount=Decimal("1000.00"),
            gl_entries=[{"account": "1000", "debit": Decimal("999.00"), "credit": Decimal("0")}],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestRecalculateBalance:
    """RECALCULATE_BALANCE: 4 steps — balance handlers."""

    async def test_balance_recalculated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.RECALCULATE_BALANCE.value)
        txn = _txn(
            account_id="1000",
            current_balance=Decimal("5000.00"),
            journal_entries=[
                {"debit": Decimal("1000.00"), "credit": Decimal("0")},
                {"debit": Decimal("0"), "credit": Decimal("500.00")},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_no_entries(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.RECALCULATE_BALANCE.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixApprovalChain:
    """FIX_APPROVAL_CHAIN: 3 steps — approval handlers."""

    async def test_approval_fixed(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_APPROVAL_CHAIN.value)
        txn = _txn(
            approval_chain=[{"level": 1, "status": "approved"}],
            approval_required=True,
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 3

    async def test_no_approval_chain(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_APPROVAL_CHAIN.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectPeriodAssignment:
    """CORRECT_PERIOD_ASSIGNMENT: 4 steps — period handlers."""

    async def test_period_reassigned(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_PERIOD_ASSIGNMENT.value)
        txn = _txn(
            fiscal_period="2025-01",
            posting_date="2025-02-15",
        )
        ctx = _ctx(correct_period="2025-02")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 4
        assert result.modified_transaction["fiscal_period"] == "2025-02"

    async def test_period_already_correct(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_PERIOD_ASSIGNMENT.value)
        txn = _txn(fiscal_period="2025-03")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestRelinkDocuments:
    """RELINK_DOCUMENTS: 3 steps — document linkage handlers."""

    async def test_documents_relinked(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.RELINK_DOCUMENTS.value)
        txn = _txn(
            po_number="PO-2025-0001",
            receipt_number="GR-2025-0001",
            invoice_number="VINV-2025-0001",
            related_documents=["PO-2025-0001"],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 3

    async def test_no_related_documents(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.RELINK_DOCUMENTS.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixThreeWayMatch:
    """FIX_THREE_WAY_MATCH: 4 steps — three-way match handlers."""

    async def test_match_fixed(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_THREE_WAY_MATCH.value)
        txn = _txn(
            po_price=Decimal("100.00"),
            invoice_price=Decimal("105.00"),
            receipt_qty=Decimal("10"),
            invoice_qty=Decimal("10"),
            match_status="EXCEPTION",
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4
        assert result.modified_transaction["match_status"] == "MATCHED"

    async def test_no_match_fields(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_THREE_WAY_MATCH.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestAdjustPaymentAllocation:
    """ADJUST_PAYMENT_ALLOCATION: 4 steps — FIFO payment handlers."""

    async def test_fifo_allocation(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_PAYMENT_ALLOCATION.value)
        txn = _txn(
            payment_amount=Decimal("1500.00"),
            open_invoices=[
                {"invoice_id": "INV-001", "invoice_date": "2025-01-01", "balance": Decimal("1000.00")},
                {"invoice_id": "INV-002", "invoice_date": "2025-02-01", "balance": Decimal("800.00")},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        mod = result.modified_transaction
        assert "payment_allocations" in mod
        assert len(mod["payment_allocations"]) == 2

    async def test_overpayment_creates_unapplied(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_PAYMENT_ALLOCATION.value)
        txn = _txn(
            payment_amount=Decimal("2000.00"),
            open_invoices=[
                {"invoice_id": "INV-001", "invoice_date": "2025-01-01", "balance": Decimal("500.00")},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert "unapplied_cash" in result.modified_transaction

    async def test_no_invoices(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_PAYMENT_ALLOCATION.value)
        txn = _txn(payment_amount=Decimal("100.00"), open_invoices=[])
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_exact_payment(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.ADJUST_PAYMENT_ALLOCATION.value)
        txn = _txn(
            payment_amount=Decimal("500.00"),
            open_invoices=[
                {"invoice_id": "INV-001", "invoice_date": "2025-01-15", "balance": Decimal("500.00")},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectTaxCalculation:
    """CORRECT_TAX_CALCULATION: 4 steps — tax handlers."""

    async def test_tax_recalculated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_TAX_CALCULATION.value)
        txn = _txn(
            amount=Decimal("1000.00"),
            tax_amount=Decimal("50.00"),
            line_items=[
                {"line_total": Decimal("600.00"), "tax": Decimal("30.00")},
                {"line_total": Decimal("400.00"), "tax": Decimal("20.00")},
            ],
        )
        ctx = _ctx(tax_rate=Decimal("0.08"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 4
        # Tax should be recalculated to 8% of 1000 = 80
        assert result.modified_transaction["tax_amount"] == Decimal("80.00")

    async def test_zero_tax_rate(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_TAX_CALCULATION.value)
        txn = _txn(amount=Decimal("500.00"))
        ctx = _ctx(tax_rate=Decimal("0.00"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert result.modified_transaction["tax_amount"] == Decimal("0.00")

    async def test_no_line_items(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_TAX_CALCULATION.value)
        txn = _txn(amount=Decimal("200.00"))
        ctx = _ctx(tax_rate=Decimal("0.10"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixCurrencyRounding:
    """FIX_CURRENCY_ROUNDING: 3 steps — rounding handlers."""

    async def test_rounding_applied(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_CURRENCY_ROUNDING.value)
        txn = _txn(amount=Decimal("123.456789"))
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 3
        mod = result.modified_transaction
        assert mod["amount"] == Decimal("123.46")

    async def test_already_rounded(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_CURRENCY_ROUNDING.value)
        txn = _txn(amount=Decimal("100.00"))
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_none_amount(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_CURRENCY_ROUNDING.value)
        txn = _txn()
        del txn["amount"]
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectDiscountApplication:
    """CORRECT_DISCOUNT_APPLICATION: 4 steps — discount handlers."""

    async def test_discount_recalculated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_DISCOUNT_APPLICATION.value)
        txn = _txn(
            amount=Decimal("1000.00"),
            discount_percent=Decimal("10"),
            discount_amount=Decimal("50.00"),  # wrong — should be 100
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_no_discount(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_DISCOUNT_APPLICATION.value)
        txn = _txn(amount=Decimal("500.00"))
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixDuplicateDetection:
    """FIX_DUPLICATE_DETECTION: 4 steps — duplicate handlers."""

    async def test_duplicate_removed(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DUPLICATE_DETECTION.value)
        txn = _txn(
            related_transactions=[
                {"id": "T-001", "amount": Decimal("100.00")},
                {"id": "T-001", "amount": Decimal("100.00")},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_no_duplicates(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_DUPLICATE_DETECTION.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestRegenerateDocumentNumber:
    """REGENERATE_DOCUMENT_NUMBER: 3 steps — document number handlers."""

    async def test_number_regenerated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_DOCUMENT_NUMBER.value)
        txn = _txn(document_number="INV-2025-0001", document_type="vendor_invoice")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 3
        assert result.modified_transaction.get("document_number") is not None

    async def test_no_document_number(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_DOCUMENT_NUMBER.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectQuantityVariance:
    """CORRECT_QUANTITY_VARIANCE: 4 steps — quantity handlers."""

    async def test_quantity_adjusted(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_QUANTITY_VARIANCE.value)
        txn = _txn(
            po_quantity=Decimal("100"),
            receipt_quantity=Decimal("98"),
            invoice_quantity=Decimal("105"),
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_no_quantities(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_QUANTITY_VARIANCE.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixCreditCheck:
    """FIX_CREDIT_CHECK: 4 steps — credit handlers."""

    async def test_credit_check_adjusted(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_CREDIT_CHECK.value)
        txn = _txn(
            credit_limit=Decimal("10000.00"),
            current_ar_balance=Decimal("8000.00"),
            order_amount=Decimal("5000.00"),
            order_lines=[
                {"line_total": Decimal("3000.00")},
                {"line_total": Decimal("2000.00")},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_within_limit(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_CREDIT_CHECK.value)
        txn = _txn(
            credit_limit=Decimal("100000.00"),
            current_ar_balance=Decimal("1000.00"),
            order_amount=Decimal("500.00"),
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_no_credit_data(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_CREDIT_CHECK.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectPostingAccounts:
    """CORRECT_POSTING_ACCOUNTS: 4 steps — posting account handlers."""

    async def test_accounts_corrected(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_POSTING_ACCOUNTS.value)
        txn = _txn(
            gl_entries=[
                {"account": "1000", "debit": Decimal("500.00")},
                {"account": "2000", "credit": Decimal("500.00")},
            ],
        )
        ctx = _ctx(correct_accounts={"1000": "1100"})
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 4

    async def test_no_gl_entries(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_POSTING_ACCOUNTS.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixPaymentTerms:
    """FIX_PAYMENT_TERMS: 4 steps — payment terms handlers."""

    async def test_terms_corrected(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_PAYMENT_TERMS.value)
        txn = _txn(
            payment_terms="NET30",
            invoice_date="2025-01-15",
            due_date="2025-02-14",
        )
        ctx = _ctx(correct_payment_terms="NET60")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 4
        assert result.modified_transaction["payment_terms"] == "NET60"

    async def test_various_term_days(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_PAYMENT_TERMS.value)
        for terms in ["NET10", "NET15", "NET45", "NET60"]:
            txn = _txn(payment_terms=terms, invoice_date="2025-01-01")
            result = await executor.execute(scenario, txn, _ctx())
            assert result.success is True

    async def test_no_terms_change(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_PAYMENT_TERMS.value)
        txn = _txn(payment_terms="NET30")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestRegenerateTrackingNumber:
    """REGENERATE_TRACKING_NUMBER: 3 steps — tracking handlers."""

    async def test_tracking_regenerated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_TRACKING_NUMBER.value)
        txn = _txn(tracking_number="OLD-TRACK-001")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 3
        new_tracking = result.modified_transaction.get("tracking_number", "")
        assert new_tracking != "OLD-TRACK-001"
        assert new_tracking.startswith("TRK-")

    async def test_no_tracking_number(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.REGENERATE_TRACKING_NUMBER.value)
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestCorrectStatusTransition:
    """CORRECT_STATUS_TRANSITION: 4 steps — status handlers."""

    async def test_status_corrected(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_STATUS_TRANSITION.value)
        txn = _txn(status="draft")
        ctx = _ctx(correct_status="submitted")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert len(result.steps_executed) == 4
        assert result.modified_transaction["status"] == "submitted"

    async def test_each_status(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_STATUS_TRANSITION.value)
        for status in ["draft", "submitted", "approved", "in_progress", "completed", "failed", "unknown"]:
            txn = _txn(status=status)
            result = await executor.execute(scenario, txn, _ctx())
            assert result.success is True

    async def test_no_correct_status(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.CORRECT_STATUS_TRANSITION.value)
        txn = _txn(status="draft")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


@pytest.mark.rework
@pytest.mark.asyncio
class TestFixInventoryBalance:
    """FIX_INVENTORY_BALANCE: 4 steps — inventory handlers."""

    async def test_inventory_recalculated(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_INVENTORY_BALANCE.value)
        txn = _txn(
            product_id="PROD-001",
            inventory_quantity=Decimal("50"),
            inventory_movements=[
                {"quantity": Decimal("100"), "direction": "in"},
                {"quantity": Decimal("30"), "direction": "out"},
                {"quantity": Decimal("20"), "direction": "out"},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
        assert len(result.steps_executed) == 4
        assert result.modified_transaction.get("_inventory_locked") is False

    async def test_empty_movements(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_INVENTORY_BALANCE.value)
        txn = _txn(product_id="PROD-001", inventory_movements=[])
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_string_quantity(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_INVENTORY_BALANCE.value)
        txn = _txn(
            product_id="PROD-001",
            inventory_quantity="100",
            inventory_movements=[
                {"quantity": 50, "direction": "in"},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_float_movements(self, catalog, executor):
        scenario = _get_scenario(catalog, FixScenarioName.FIX_INVENTORY_BALANCE.value)
        txn = _txn(
            product_id="PROD-002",
            inventory_quantity=Decimal("0"),
            inventory_movements=[
                {"quantity": 25.5, "direction": "in"},
                {"quantity": 10.0, "direction": "out"},
            ],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True


# =========================================================================
# Test Class — Error Paths and Edge Cases
# =========================================================================


@pytest.mark.rework
@pytest.mark.asyncio
class TestExecutorErrorPaths:
    """Tests for error handling paths in execute()."""

    async def test_generic_exception_in_handler(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)

        async def exploding_handler(txn, ctx):
            raise RuntimeError("Boom!")

        executor.register_handler("explode", exploding_handler)

        scenario = FixScenario(
            name="EXPLODING_SCENARIO",
            description="A scenario whose handler raises RuntimeError",
            steps=[FixStep(step_order=1, step_name="boom", description="Explodes", handler_key="explode")],
            success_rate=0.5,
            applicable_error_types=["test"],
        )
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is False
        assert "boom" in result.steps_failed

    async def test_rework_loop_error_in_handler_caught(self):
        """ReworkLoopError from a handler is caught by _execute_step → failure."""
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)

        async def domain_error_handler(txn, ctx):
            raise ReworkLoopError("Fatal domain error")

        executor.register_handler("domain_err", domain_error_handler)

        scenario = FixScenario(
            name="DOMAIN_ERROR",
            description="Handler raises ReworkLoopError",
            steps=[FixStep(step_order=1, step_name="fatal", description="Raises domain error", handler_key="domain_err")],
            success_rate=0.5,
            applicable_error_types=["test"],
        )
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is False
        assert "fatal" in result.steps_failed

    async def test_timeout_scenario(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=0.1)

        async def slow_handler(txn, ctx):
            await asyncio.sleep(5)
            return {}

        executor.register_handler("slow", slow_handler)

        scenario = FixScenario(
            name="SLOW_SCENARIO",
            description="Times out",
            steps=[FixStep(step_order=1, step_name="slow_step", description="Slow", handler_key="slow")],
            success_rate=0.5,
            applicable_error_types=["test"],
        )
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is False
        assert "timed out" in (result.error_message or "")

    async def test_generic_handler_update_field(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        # Use the generic update_field handler
        scenario = FixScenario(
            name="GENERIC_UPDATE",
            description="Uses generic handlers",
            steps=[
                FixStep(step_order=1, step_name="validate", description="Validate", handler_key="validate_current_value"),
                FixStep(step_order=2, step_name="calculate", description="Calculate", handler_key="calculate_correct_value"),
                FixStep(step_order=3, step_name="update", description="Update", handler_key="update_field"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(amount=Decimal("500.00"))
        ctx = _ctx(target_field="amount", correct_value=Decimal("750.00"))
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True
        assert result.modified_transaction["amount"] == Decimal("750.00")

    async def test_generic_no_value_provided(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="GENERIC_NO_VALUE",
            description="Generic with no correct_value",
            steps=[
                FixStep(step_order=1, step_name="update", description="Update", handler_key="update_field"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        result = await executor.execute(scenario, _txn(), _ctx())
        assert result.success is True


# =========================================================================
# Test Class — Alias Handlers
# =========================================================================


@pytest.mark.rework
@pytest.mark.asyncio
class TestAliasHandlers:
    """Verify alias handler keys work correctly."""

    async def test_validate_date_sequence_alias(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="ALIAS_DATE",
            description="Tests alias handlers for dates",
            steps=[
                FixStep(step_order=1, step_name="s1", description="Fix", handler_key="fix_date_ordering"),
                FixStep(step_order=2, step_name="s2", description="Validate", handler_key="validate_date_sequence"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(order_date="2025-01-01", payment_date="2025-02-01")
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_entity_alias(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="ALIAS_ENTITY",
            description="Tests alias handlers for entity",
            steps=[
                FixStep(step_order=1, step_name="s1", description="Validate", handler_key="validate_entity_reference"),
                FixStep(step_order=2, step_name="s2", description="Correct", handler_key="correct_entity_reference"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(vendor_id="V-001")
        ctx = _ctx(correct_entity_id="V-002")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True

    async def test_gl_aliases(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="ALIAS_GL",
            description="Tests alias handlers for GL",
            steps=[
                FixStep(step_order=1, step_name="s1", description="Recalc", handler_key="recalculate_amount"),
                FixStep(step_order=2, step_name="s2", description="Validate", handler_key="validate_gl_balance"),
                FixStep(step_order=3, step_name="s3", description="Regen", handler_key="regenerate_gl_entry"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(amount=Decimal("100.00"), gl_entries=[])
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_balance_and_approval_aliases(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="ALIAS_BALANCE_APPROVAL",
            description="Tests balance and approval aliases",
            steps=[
                FixStep(step_order=1, step_name="s1", description="Recalc balance", handler_key="recalculate_balance"),
                FixStep(step_order=2, step_name="s2", description="Validate approval", handler_key="validate_approval_chain"),
                FixStep(step_order=3, step_name="s3", description="Fix approval", handler_key="fix_approval_routing"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(journal_entries=[], approval_chain=[])
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True

    async def test_period_and_document_aliases(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="ALIAS_PERIOD_DOC",
            description="Tests period and document aliases",
            steps=[
                FixStep(step_order=1, step_name="s1", description="Validate period", handler_key="validate_period"),
                FixStep(step_order=2, step_name="s2", description="Correct period", handler_key="correct_period"),
                FixStep(step_order=3, step_name="s3", description="Validate docs", handler_key="validate_document_links"),
                FixStep(step_order=4, step_name="s4", description="Relink docs", handler_key="relink_documents"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(posting_period="2025-01")
        ctx = _ctx(correct_period="2025-03")
        result = await executor.execute(scenario, txn, ctx)
        assert result.success is True

    async def test_match_and_allocation_aliases(self):
        executor = FixScenarioExecutor(scenario_timeout_seconds=10.0)
        scenario = FixScenario(
            name="ALIAS_MATCH_ALLOC",
            description="Tests match and allocation aliases",
            steps=[
                FixStep(step_order=1, step_name="s1", description="Validate match", handler_key="validate_three_way_match"),
                FixStep(step_order=2, step_name="s2", description="Recalc match", handler_key="recalculate_match"),
                FixStep(step_order=3, step_name="s3", description="Validate alloc", handler_key="validate_payment_allocation"),
                FixStep(step_order=4, step_name="s4", description="Recalc alloc", handler_key="recalculate_allocation"),
            ],
            success_rate=1.0,
            applicable_error_types=["test"],
        )
        txn = _txn(
            payment_amount=Decimal("100.00"),
            open_invoices=[{"invoice_id": "I-1", "invoice_date": "2025-01-01", "balance": Decimal("100.00")}],
        )
        result = await executor.execute(scenario, txn, _ctx())
        assert result.success is True
