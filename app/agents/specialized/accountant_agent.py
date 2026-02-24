"""
AccountantAgent — General Accountant agent for the Agent System (F-001).

The Accountant agent is the front-line accounting role responsible for:
  - Creating journal entries with debit/credit balance validation
  - Performing LLM-assisted account reconciliation
  - Routing all journal entries for approval based on amount thresholds

Approval Thresholds (from README.md lines 327-330):
  - All journal entries: senior_accountant approval required
  - Journal Entries > $50K: escalated to controller approval

Exports:
    AccountantAgent — Specialized agent subclass extending BaseAgent with:
        - ROLE:                         "accountant"
        - SUPPORTED_WORK_TYPES:         ["journal_entry", "reconciliation"]
        - SENIOR_ACCOUNTANT_THRESHOLD:  $0.0 (all JEs need senior review)
        - CONTROLLER_THRESHOLD:         $50,000.0
        - process_work_item():          Main routing dispatcher
        - _create_journal_entry():      Journal entry creation handler
        - _reconcile_account():         Account reconciliation handler

Design Decisions:
  - Extends BaseAgent via constructor injection (config, memory,
    decision_engine, action_registry) per AAP Section 0.7.1.  No __init__
    override — all dependencies are injected by the base class.
  - Uses self.make_decision() (inherited from BaseAgent) to invoke the
    4-layer Decision Engine for LLM-powered journal entry description
    generation and account reconciliation decisions.
  - Financial calculations (GL postings, balance updates, debit/credit
    totals) are EXCLUSIVELY computed in the Deterministic Layer of the
    Decision Engine — the Accountant agent never computes them directly.
    Debit/credit validation is a read-only check, not a financial
    calculation (AAP Section 0.7.1).
  - Lightweight class: no __init__ override, no I/O at import time.
    Supports ≥50 agents/second creation rate (AAP Section 0.7.3).
  - All logging via structlog in structured JSON format to stdout
    (AAP Section 0.7.6).

Timeout Constraints (AAP Section 0.1.2):
  - Simple decision:       10 seconds
  - LLM decision:          30 seconds
  - Complex workflow:       60 seconds
  - Absolute max:          120 seconds

Performance Targets (AAP Section 0.7.3):
  - Agent creation:     ≥ 50 agents/second
  - Decision latency:   p95 < 5 seconds
  - Agent concurrency:  ≥ 20 simultaneous agents

References:
  - README.md line 80: "AccountantAgent  # Accountant"
  - README.md line 131: create_journal_entry action
  - README.md line 132: reconcile_account action
  - README.md line 211: reconcile_account is an LLM decision type
  - README.md lines 327-330: JE approval thresholds
  - README.md line 1353: ROLE_MAPPING "journal_entry":
    ["accountant", "senior_accountant"]
  - AAP Section 0.5.1 Group 5: "AccountantAgent: journal entry creation,
    reconciliation"
"""

from typing import Dict, Any, Optional, List

import structlog

from app.agents.base_agent import BaseAgent, WorkItem, WorkResult
from app.agents.agent_config import AgentConfig
from app.agents.agent_memory import AgentMemory
from app.agents.decision_engine import DecisionEngine
from app.agents.action_registry import ActionRegistry

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP Section 0.7.6)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


class AccountantAgent(BaseAgent):
    """General Accountant agent — front-line accounting specialist.

    Responsible for:
      - Creating journal entries with full debit/credit balance validation
      - Performing LLM-assisted account reconciliation
      - Routing JEs for approval based on monetary thresholds
      - Ensuring every journal entry is balanced (debits == credits)

    Role: ``"accountant"``

    Transaction types handled:
      - ``"journal_entry"``    — Journal entry creation with description
                                  generation via LLM and debit/credit
                                  balance validation
      - ``"reconciliation"``   — Account reconciliation using LLM-assisted
                                  matching and adjustment detection

    Approval Rules (README.md lines 327-330):
      - All journal entries require senior_accountant approval.
      - Journal entries > $50,000 require controller approval.

    Hierarchy Position:
      - Reports to: Senior Accountant
      - Escalation: All JEs → Senior Accountant → Controller (if > $50K)

    Actions used from ActionRegistry:
      - ``create_journal_entry``  — For journal entry creation
      - ``reconcile_account``     — For account reconciliation operations

    Inherits from :class:`BaseAgent`:
      - Constructor injection (config, memory, decision_engine,
        action_registry)
      - Async ``run()`` loop for work queue processing
      - ``make_decision()`` for 4-layer Decision Engine invocation
      - ``_calculate_importance()`` for memory observation scoring
      - Performance metrics tracking

    IMPORTANT:
        Financial calculations (GL postings, balance updates) are computed
        EXCLUSIVELY by the Deterministic Layer of the Decision Engine.
        This agent performs read-only validation (e.g., debit/credit balance
        checks) but never generates GL postings or balance updates directly.
    """

    # ------------------------------------------------------------------
    # Class-level Constants
    # ------------------------------------------------------------------
    ROLE: str = "accountant"
    """Agent role identifier matching the specification role mapping.

    Maps to AccountantAgent in VALID_AGENT_ROLES and ROLE_MAPPING:
        "journal_entry": ["accountant", "senior_accountant"]
    """

    SUPPORTED_WORK_TYPES: List[str] = [
        "journal_entry",
        "reconciliation",
    ]
    """Transaction types this agent is capable of processing.

    - journal_entry:   Create journal entries with LLM-generated descriptions,
                       validate debit/credit balance, route for approval.
    - reconciliation:  Perform LLM-assisted account reconciliation to detect
                       adjustments and unmatched items.
    """

    SENIOR_ACCOUNTANT_THRESHOLD: float = 0.0
    """Monetary threshold at which senior_accountant approval is required.

    Set to $0.0 because ALL journal entries require senior_accountant
    approval regardless of amount (README.md line 328).
    """

    CONTROLLER_THRESHOLD: float = 50_000.0
    """Monetary threshold above which controller approval is required.

    Journal entries exceeding $50,000 are escalated to the controller
    role for approval instead of the senior_accountant (README.md line 329).
    """

    # ------------------------------------------------------------------
    # process_work_item — Main routing dispatcher
    # ------------------------------------------------------------------
    async def process_work_item(self, work_item: WorkItem) -> WorkResult:
        """Process a work item according to accountant specialization.

        Routes the incoming work item to the appropriate handler method
        based on ``work_item.type``.  Supported types map to dedicated
        private handler methods:

          - ``"journal_entry"``    → :meth:`_create_journal_entry`
          - ``"reconciliation"``   → :meth:`_reconcile_account`

        Any unrecognized type raises ``ValueError`` which is caught by the
        :meth:`BaseAgent.run` loop and recorded as an error.

        Args:
            work_item: The :class:`WorkItem` to process.

        Returns:
            :class:`WorkResult` with the processing outcome.

        Raises:
            ValueError: If ``work_item.type`` is not in
                :attr:`SUPPORTED_WORK_TYPES`.
        """
        work_type: str = work_item.type

        logger.info(
            "accountant_processing",
            agent_id=str(self.config.agent_id),
            work_item_type=work_type,
            work_item_id=str(work_item.workflow_id),
        )

        try:
            if work_type == "journal_entry":
                return await self._create_journal_entry(work_item)

            if work_type == "reconciliation":
                return await self._reconcile_account(work_item)

            # Unsupported type — raise so run() captures it in ERROR state
            raise ValueError(
                f"AccountantAgent does not support work item type "
                f"'{work_type}'. Supported types: {self.SUPPORTED_WORK_TYPES}"
            )

        except ValueError:
            # Re-raise ValueError for unsupported types so BaseAgent.run()
            # records them in ERROR state with full structured logging.
            raise

        except Exception as exc:
            # Catch-all for unexpected errors within handler methods.
            # This prevents a single malformed work item from crashing the
            # agent's run() loop while still surfacing the error in the
            # WorkResult for upstream consumers.
            logger.error(
                "accountant_processing_error",
                agent_id=str(self.config.agent_id),
                work_item_type=work_type,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return WorkResult(
                success=False,
                actions_taken=["accountant_processing_failed"],
                has_exceptions=True,
                data={"error": str(exc), "work_item_type": work_type},
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Journal Entry Creation
    # ------------------------------------------------------------------
    async def _create_journal_entry(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Create a journal entry with LLM-generated description.

        Processing flow:
          1. Extract journal entry data and line items from the work item.
          2. Validate debit/credit balance — total debits must equal total
             credits within 1 cent tolerance.  Unbalanced entries are
             immediately rejected with ``has_exceptions=True``.
          3. Invoke ``self.make_decision()`` with
             ``decision_type="generate_description"`` for LLM-powered
             journal entry description generation.
          4. Determine the approval routing based on monetary thresholds:
             - All JEs → ``senior_accountant`` approval
             - JEs > $50K → ``controller`` approval
          5. Build and return the :class:`WorkResult` with JE details.

        CRITICAL:
            The debit/credit validation in step 2 is a **read-only balance
            check**, not a financial calculation.  Actual GL postings and
            balance updates are performed exclusively by the Deterministic
            Layer of the Decision Engine.

        Args:
            work_item: The work item containing journal entry data.
                Expected ``work_item.data`` keys:
                  - ``entries`` or ``line_items``: List of dicts with
                    ``debit`` and ``credit`` float values.
                  - ``description`` (optional): Pre-existing description.
                  - ``account_id`` (optional): Target account identifier.
                  - ``period`` (optional): Fiscal period identifier.
                  - ``department`` (optional): Originating department.

        Returns:
            :class:`WorkResult` with:
              - ``success=True`` and journal entry details on balanced JE.
              - ``success=False`` and ``has_exceptions=True`` on unbalanced JE.
              - ``approval_needed=True`` always (all JEs require approval).
              - ``data["approver_role"]``: Role for approval routing.
        """
        transaction_data: Dict[str, Any] = work_item.data

        # Extract line items — support both "entries" and "line_items" keys
        # for compatibility with different upstream producers.
        entries: List[Dict[str, Any]] = transaction_data.get(
            "entries", transaction_data.get("line_items", [])
        )
        existing_description: str = transaction_data.get("description", "")
        account_id: str = transaction_data.get("account_id", "")
        period: str = transaction_data.get("period", "")
        department: str = transaction_data.get("department", "")

        # -----------------------------------------------------------------
        # Step 1: Validate Debit/Credit Balance (read-only check)
        # -----------------------------------------------------------------
        total_debit: float = sum(
            float(entry.get("debit", 0.0)) for entry in entries
        )
        total_credit: float = sum(
            float(entry.get("credit", 0.0)) for entry in entries
        )
        balanced: bool = abs(total_debit - total_credit) < 0.01

        if not balanced:
            variance: float = round(abs(total_debit - total_credit), 2)
            logger.warning(
                "journal_entry_unbalanced",
                agent_id=str(self.config.agent_id),
                total_debit=round(total_debit, 2),
                total_credit=round(total_credit, 2),
                variance=variance,
            )
            return WorkResult(
                success=False,
                actions_taken=["journal_entry_balance_check_failed"],
                approval_needed=False,
                has_exceptions=True,
                data={
                    "entries": entries,
                    "total_debit": round(total_debit, 2),
                    "total_credit": round(total_credit, 2),
                    "variance": variance,
                    "balanced": False,
                    "rejection_reason": (
                        f"Journal entry is not balanced. "
                        f"Total debits (${total_debit:,.2f}) do not equal "
                        f"total credits (${total_credit:,.2f}). "
                        f"Variance: ${variance:,.2f}."
                    ),
                },
                error=(
                    f"Journal entry not balanced: debits=${total_debit:,.2f}, "
                    f"credits=${total_credit:,.2f}, variance=${variance:,.2f}"
                ),
            )

        # -----------------------------------------------------------------
        # Step 2: LLM-Powered Description Generation
        # -----------------------------------------------------------------
        # The generate_description decision type uses the LLM Layer (Layer 2)
        # of the Decision Engine to produce a contextual description for the
        # journal entry.  GL postings are NOT affected by this call — the
        # Deterministic Layer handles them separately.
        decision_context: Dict[str, Any] = {
            "journal_entry": transaction_data,
            "entries": entries,
            "total_debit": round(total_debit, 2),
            "total_credit": round(total_credit, 2),
            "account_id": account_id,
            "period": period,
            "department": department,
            "existing_description": existing_description,
            "entry_count": len(entries),
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="generate_description",
            context=decision_context,
        )

        # Extract the LLM-generated description from the decision result.
        # Falls back to existing description or a sensible default.
        generated_description: str = decision.get(
            "description",
            decision.get(
                "text",
                existing_description or "Journal entry",
            ),
        )
        generated_tags: List[str] = decision.get("tags", [])

        # -----------------------------------------------------------------
        # Step 3: Determine Approval Routing
        # -----------------------------------------------------------------
        # Per README.md lines 327-330:
        #   - All JEs: senior_accountant approval
        #   - >$50K: controller approval
        amount: float = max(total_debit, total_credit)

        if amount >= self.CONTROLLER_THRESHOLD:
            approver_role: str = "controller"
        else:
            approver_role = "senior_accountant"

        # All journal entries require approval regardless of amount
        approval_needed: bool = True

        # -----------------------------------------------------------------
        # Step 4: Build Result
        # -----------------------------------------------------------------
        actions_taken: List[str] = ["journal_entry_created"]

        result_data: Dict[str, Any] = {
            "entries": entries,
            "total_debit": round(total_debit, 2),
            "total_credit": round(total_credit, 2),
            "balanced": True,
            "description": generated_description,
            "tags": generated_tags,
            "approver_role": approver_role,
            "amount": round(amount, 2),
            "account_id": account_id,
            "period": period,
            "department": department,
            "entry_count": len(entries),
        }

        logger.info(
            "journal_entry_created",
            agent_id=str(self.config.agent_id),
            amount=round(amount, 2),
            balanced=True,
            approval_needed=approval_needed,
            approver_role=approver_role,
            entry_count=len(entries),
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=approval_needed,
            has_exceptions=False,
            data=result_data,
        )

    # ------------------------------------------------------------------
    # Account Reconciliation
    # ------------------------------------------------------------------
    async def _reconcile_account(
        self, work_item: WorkItem
    ) -> WorkResult:
        """Perform LLM-assisted account reconciliation.

        Account reconciliation is an LLM decision type (README.md line 211,
        prompts/reconcile_account.yaml) that uses the Decision Engine to
        analyze transactions against account records and identify:
          - Matched transactions (reconciliation_items)
          - Required adjustments (journal entries to correct balances)
          - Unmatched items needing investigation

        Processing flow:
          1. Extract reconciliation context (account, period, transactions).
          2. Invoke ``self.make_decision()`` with
             ``decision_type="reconcile_account"`` which triggers the
             LLM Layer for intelligent matching and adjustment detection.
          3. Parse the reconciliation results from the decision output.
          4. If adjustments are needed, flag ``has_exceptions=True`` so
             the Senior Accountant reviews the proposed adjustments.
          5. Build and return the :class:`WorkResult` with reconciliation
             details.

        CRITICAL:
            The LLM provides reconciliation *recommendations*, but all
            resulting GL postings and balance updates are computed by the
            Deterministic Layer of the Decision Engine — never generated
            directly by the LLM.

        Args:
            work_item: The work item containing reconciliation data.
                Expected ``work_item.data`` keys:
                  - ``account`` or ``account_id``: Account dict or ID.
                  - ``period``: Fiscal period for reconciliation.
                  - ``transactions``: List of transaction dicts to
                    reconcile.
                  - ``expected_balance`` (optional): Expected ending balance.
                  - ``book_balance`` (optional): Current book balance.

        Returns:
            :class:`WorkResult` with:
              - ``success=True`` and reconciliation details on completion.
              - ``has_exceptions=True`` if adjustments are needed.
              - ``data["reconciliation_items"]``: Matched transactions.
              - ``data["adjustments"]``: Proposed adjustments.
              - ``data["unmatched_items"]``: Items needing investigation.
        """
        transaction_data: Dict[str, Any] = work_item.data

        # Extract reconciliation parameters
        account: Dict[str, Any] = transaction_data.get("account", {})
        account_id: str = transaction_data.get(
            "account_id",
            account.get("account_id", account.get("id", "")),
        )
        period: str = transaction_data.get("period", "")
        transactions: List[Dict[str, Any]] = transaction_data.get(
            "transactions", []
        )
        expected_balance: Optional[float] = transaction_data.get(
            "expected_balance"
        )
        book_balance: Optional[float] = transaction_data.get(
            "book_balance"
        )

        # -----------------------------------------------------------------
        # LLM-Assisted Reconciliation Decision
        # -----------------------------------------------------------------
        # The reconcile_account decision type invokes the LLM Layer (Layer 2)
        # of the Decision Engine for intelligent transaction matching and
        # adjustment detection.  The Deterministic Layer (Layer 4) ensures
        # any resulting GL postings are calculated correctly.
        decision_context: Dict[str, Any] = {
            "account": account,
            "account_id": account_id,
            "period": period,
            "transactions": transactions,
            "transaction_count": len(transactions),
            "expected_balance": expected_balance,
            "book_balance": book_balance,
        }

        decision: Dict[str, Any] = await self.make_decision(
            decision_type="reconcile_account",
            context=decision_context,
        )

        # -----------------------------------------------------------------
        # Extract Reconciliation Results
        # -----------------------------------------------------------------
        reconciliation_items: List[Dict[str, Any]] = decision.get(
            "reconciliation_items", []
        )
        adjustments: List[Dict[str, Any]] = decision.get(
            "adjustments", []
        )
        unmatched_items: List[Dict[str, Any]] = decision.get(
            "unmatched_items", []
        )
        reconciled_balance: Optional[float] = decision.get(
            "reconciled_balance"
        )
        notes: str = decision.get("notes", decision.get("reasoning", ""))

        # Determine exception status — adjustments or unmatched items
        # indicate reconciliation exceptions that need senior review.
        adjustment_count: int = len(adjustments)
        unmatched_count: int = len(unmatched_items)
        has_exceptions: bool = adjustment_count > 0 or unmatched_count > 0

        # Build result data
        actions_taken: List[str] = ["account_reconciled"]

        result_data: Dict[str, Any] = {
            "account_id": account_id,
            "period": period,
            "reconciliation_items": reconciliation_items,
            "reconciliation_items_count": len(reconciliation_items),
            "adjustments": adjustments,
            "adjustment_count": adjustment_count,
            "unmatched_items": unmatched_items,
            "unmatched_count": unmatched_count,
            "reconciled_balance": reconciled_balance,
            "expected_balance": expected_balance,
            "book_balance": book_balance,
            "transaction_count": len(transactions),
            "notes": notes,
            "needs_senior_review": has_exceptions,
        }

        logger.info(
            "account_reconciled",
            agent_id=str(self.config.agent_id),
            account_id=account_id,
            period=period,
            adjustment_count=adjustment_count,
            unmatched_count=unmatched_count,
            has_exceptions=has_exceptions,
            transaction_count=len(transactions),
        )

        return WorkResult(
            success=True,
            actions_taken=actions_taken,
            approval_needed=False,
            has_exceptions=has_exceptions,
            data=result_data,
        )
