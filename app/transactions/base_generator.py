"""Abstract base class for all P3 transaction generators.

Defines the central contract (``generate → validate → post``) that every
concrete P2P, O2C, and GL generator must implement, plus shared
infrastructure consumed by all subclasses:

*   **Pydantic V2 data contracts** — :class:`GenerationContext` (per-invocation
    context carrying simulation identity, business date, fiscal period, RNG
    seed, and discrepancy configuration) and :class:`TransactionResult` (output
    of a single generation cycle with artifacts, GL entries, discrepancy info,
    and timing metrics).
*   **Constructor injection** (ADR-003) — every dependency is received via
    ``__init__`` as an ``Optional`` keyword argument with ``None`` default,
    enabling partial composition for testing and incremental integration.
*   **Event publishing** (ADR-001) — cross-subsystem notifications flow through
    the injected :class:`EventBus` via the :meth:`_publish_event` helper.
*   **Structured JSON logging** (AAP §0.7.7) — all logging uses ``structlog``
    to stdout with ``service_name``, ``component``, ``trace_id``, and
    ``simulation_id`` fields.
*   **Deterministic reproducibility** — seeded ``random.Random`` instances
    created via :meth:`_create_seeded_rng` guarantee identical transaction
    sequences for the same seed.
*   **Financial precision** — module-level ``decimal`` context configured to
    ``prec=28`` and ``ROUND_HALF_UP``; *all* monetary amounts use ``Decimal``.

Performance targets (AAP §0.7.3):
    - P2P cycle generation:  ≥ 50 / minute
    - O2C cycle generation:  ≥ 60 / minute
    - GL posting:            ≥ 200 / minute

Database pattern:
    All database operations use Project 1's session management::

        from synthetic_erp.db.session import get_session
        async with get_session() as session:
            ...

Retry policy reference (AAP §0.1.2):

    +--------------------------+--------+-------------+------+---------------------+
    | Operation                | Retry  | Backoff     | T/O  | Fallback            |
    +==========================+========+=============+======+=====================+
    | P2P cycle generation     | 2      | Linear 1–2s | 60s  | Skip transaction    |
    | O2C cycle generation     | 2      | Linear 1–2s | 60s  | Skip transaction    |
    | GL posting               | 3      | Exp 1-2-4s  | 30s  | Rollback            |
    | Three-way matching       | 2      | Lin 0.5–1s  | 10s  | Mark as exception   |
    | Discrepancy injection    | 1      | None        | 5s   | Skip discrepancy    |
    | Rework loop fix          | 3      | Linear 1–3s | 30s  | Escalate to admin   |
    | Period close             | 1      | None        | 300s | Halt (manual)       |
    | Balance update           | 2      | Lin 0.5–1s  | 10s  | Rollback            |
    +--------------------------+--------+-------------+------+---------------------+

Exports:
    :class:`GenerationContext`   — Pydantic V2 per-invocation context model
    :class:`TransactionResult`   — Pydantic V2 generation cycle result model
    :class:`TransactionGenerator` — Abstract base class (ABC) for all generators
"""

from __future__ import annotations

import asyncio
import decimal
import random
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.exceptions import (
    TransactionError,
    TransactionGenerationError,
)
from app.transactions.constants import (
    CIRCUIT_BREAKER_CONFIG,
    FINANCIAL_TOLERANCES,
    PERFORMANCE_THRESHOLDS,
    TRANSACTION_PROCESSING_LIMITS,
)

if TYPE_CHECKING:
    from app.agents.agent_registry import AgentRegistry
    from app.events.event_bus import EventBus
    from app.orchestration.workflow_orchestrator import WorkflowOrchestrator
    from app.orchestration.approval_system import ApprovalSystem

# ---------------------------------------------------------------------------
# Decimal precision configuration — AAP §0.7.2 (MUST be at module level)
# ---------------------------------------------------------------------------
decimal.getcontext().prec = 28
decimal.getcontext().rounding = decimal.ROUND_HALF_UP

# ---------------------------------------------------------------------------
# Module-level structured logger — AAP §0.7.7 (JSON to stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic V2 Data Contracts
# ═══════════════════════════════════════════════════════════════════════════════


class GenerationContext(BaseModel):
    """Per-generation invocation context passed to every TransactionGenerator.

    Carries the simulation identity, current business date, fiscal period
    state, discrepancy configuration, and a deterministic RNG seed.  Instances
    are created by the ``SimulationEngine`` (composition root) at the start of
    each daily processing cycle and threaded through every generator call.

    All fields have sensible defaults except ``current_date`` which is required
    so that every context is unambiguously tied to a business day.

    Usage::

        from datetime import date
        ctx = GenerationContext(current_date=date(2024, 3, 15))
        data = ctx.model_dump()          # serialise to dict
        ctx2 = GenerationContext.model_validate(data)  # deserialise
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation_id: UUID = Field(
        default_factory=uuid4,
        description="Simulation run identifier.",
    )
    current_date: date = Field(
        ...,
        description="Current simulation business date (required).",
    )
    fiscal_period: str = Field(
        default="",
        description="Current fiscal period identifier, e.g. '2024-01'.",
    )
    fiscal_period_status: str = Field(
        default="OPEN",
        description="Fiscal period lifecycle status: OPEN, CLOSING, or CLOSED.",
    )
    discrepancy_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Discrepancy injection configuration overrides.",
    )
    rng_seed: int = Field(
        default=42,
        description="Seed for deterministic random generation.",
    )
    trace_id: UUID = Field(
        default_factory=uuid4,
        description="Distributed trace identifier for log correlation.",
    )
    day_number: int = Field(
        default=1,
        ge=1,
        description="Day number within the simulation run.",
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Maximum number of transactions per generation batch.",
    )


class TransactionResult(BaseModel):
    """Result of a single transaction generation cycle.

    Produced by every concrete :class:`TransactionGenerator` subclass and
    consumed by the orchestration layer, rework loop, and metrics collector.

    The ``status`` field follows the three-state model:
    ``"completed"`` (default) | ``"failed"`` | ``"skipped"``.

    Financial amounts MUST use :class:`~decimal.Decimal` — never ``float``.

    Usage::

        result = TransactionResult(transaction_type="purchase_order")
        result.artifacts["po_header"] = {...}
        payload = result.model_dump()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transaction_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this transaction.",
    )
    transaction_type: str = Field(
        ...,
        description="Transaction type, e.g. 'purchase_order', 'vendor_invoice'.",
    )
    status: str = Field(
        default="completed",
        description="Result status: 'completed', 'failed', or 'skipped'.",
    )
    artifacts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Artifacts keyed by artifact type name.",
    )
    gl_entries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Journal entry references produced by this generation cycle.",
    )
    events_published: List[str] = Field(
        default_factory=list,
        description="Event type strings published during generation.",
    )
    has_discrepancy: bool = Field(
        default=False,
        description="Whether a discrepancy was injected into this transaction.",
    )
    discrepancy_type: Optional[str] = Field(
        default=None,
        description="Discrepancy type code (e.g. 'P2P-001') if injected.",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing duration in milliseconds.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message when status is 'failed'.",
    )
    amount: Optional[Decimal] = Field(
        default=None,
        description="Transaction total monetary amount (Decimal, never float).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of when this result was created.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract Base Class
# ═══════════════════════════════════════════════════════════════════════════════


class TransactionGenerator(ABC):
    """Abstract base class for all P2P, O2C, and GL transaction generators.

    Defines the ``generate → validate → post`` contract that every concrete
    generator must implement.  Provides shared infrastructure for:

    * Discrepancy trigger checking — delegates to injected DiscrepancyInjector
    * GL posting delegation — delegates to injected GLPostingEngine
    * Event publishing — delegates to injected EventBus
    * Retry / timeout wrappers using ``tenacity``
    * Structured JSON logging via ``structlog``

    **Constructor Injection (ADR-003)**:
        All dependencies are received through ``__init__`` keyword parameters.
        Every parameter is ``Optional`` with a ``None`` default so that the
        generator can be instantiated with zero dependencies for unit testing
        or with a partial subset for incremental integration.

    Subclasses MUST implement:
        - :meth:`generate` — produce artifacts and GL entries
        - :meth:`validate` — check business rules and integrity
        - :meth:`post` — persist to General Ledger via GLPostingEngine
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        db_session_factory: Optional[Any] = None,
        agent_registry: Optional[AgentRegistry] = None,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
        event_bus: Optional[EventBus] = None,
        approval_system: Optional[ApprovalSystem] = None,
        discrepancy_injector: Optional[Any] = None,
        gl_posting_engine: Optional[Any] = None,
        statistical_models: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise a TransactionGenerator with injected dependencies.

        All parameters are keyword-only and ``Optional`` with ``None``
        defaults, following ADR-003 (constructor injection).  When a
        dependency is ``None``, any method that requires it will either
        skip its operation gracefully or log a warning — the generator
        will never crash due to a missing optional dependency.

        Args:
            db_session_factory: Callable returning an async context manager
                that yields a database session (e.g., ``get_session``).
            agent_registry: Project 2 :class:`AgentRegistry` for agent
                lookup during transaction processing.
            workflow_orchestrator: Project 2 :class:`WorkflowOrchestrator`
                for routing transactions to agents.
            event_bus: Project 2 :class:`EventBus` for cross-subsystem
                event publication (ADR-001).
            approval_system: Project 2 :class:`ApprovalSystem` for
                threshold-based approval chain determination.
            discrepancy_injector: P3 ``DiscrepancyInjector`` instance for
                rate-based discrepancy injection.
            gl_posting_engine: P3 ``GLPostingEngine`` instance for
                journal entry creation and balance validation.
            statistical_models: Dictionary of statistical model instances
                (amount distributions, payment timing, etc.).
        """
        self._db_session_factory = db_session_factory
        self._agent_registry = agent_registry
        self._workflow_orchestrator = workflow_orchestrator
        self._event_bus = event_bus
        self._approval_system = approval_system
        self._discrepancy_injector = discrepancy_injector
        self._gl_posting_engine = gl_posting_engine
        self._statistical_models: Dict[str, Any] = statistical_models or {}

        # Internal sequence counter for document numbering per generator instance
        self._sequence_counter: int = 0

        # Cache performance thresholds and processing limits for quick access
        self._performance_thresholds: Dict[str, Any] = PERFORMANCE_THRESHOLDS
        self._processing_limits: Dict[str, int] = TRANSACTION_PROCESSING_LIMITS
        self._financial_tolerances: Dict[str, Decimal] = FINANCIAL_TOLERANCES
        self._circuit_breaker_config: Dict[str, Dict[str, Any]] = CIRCUIT_BREAKER_CONFIG

        logger.info(
            "transaction_generator_initialized",
            generator_type=self.__class__.__name__,
            has_db_session=db_session_factory is not None,
            has_agent_registry=agent_registry is not None,
            has_event_bus=event_bus is not None,
            has_gl_engine=gl_posting_engine is not None,
            has_workflow_orchestrator=workflow_orchestrator is not None,
            has_approval_system=approval_system is not None,
            has_discrepancy_injector=discrepancy_injector is not None,
            service_name="transactions",
            component=self.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Abstract Methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    async def generate(self, context: GenerationContext) -> TransactionResult:
        """Generate one or more transactions for the current context.

        Concrete generators produce transaction artifacts (headers, lines,
        document numbers) and GL journal entry specifications according to
        their business-domain rules.  The returned :class:`TransactionResult`
        carries all produced artifacts, GL entries, discrepancy information,
        and timing metrics.

        Args:
            context: Generation parameters including business date, fiscal
                period, RNG seed, and discrepancy configuration.

        Returns:
            A :class:`TransactionResult` with generated artifacts and GL
            entries.

        Raises:
            TransactionGenerationError: If generation fails after all
                configured retry attempts have been exhausted.
        """
        ...

    @abstractmethod
    async def validate(self, result: TransactionResult) -> bool:
        """Validate a generated transaction result against business rules.

        Checks include artifact completeness, GL entry balance verification,
        referential integrity, and business-rule compliance.  Returns a
        boolean indicating validity — callers use this to gate the
        subsequent ``post()`` call.

        Args:
            result: The :class:`TransactionResult` to validate.

        Returns:
            ``True`` if the result passes all validation checks,
            ``False`` otherwise.
        """
        ...

    @abstractmethod
    async def post(self, result: TransactionResult) -> None:
        """Post a validated transaction to the General Ledger.

        Delegates to the injected :class:`GLPostingEngine` for journal
        entry creation and balance validation.  If any step fails, the
        entire transaction MUST be rolled back (atomicity requirement
        per AAP §0.7.2).

        Args:
            result: The validated :class:`TransactionResult` to post.

        Raises:
            GLPostingError: If GL posting fails (account validation,
                period validation, persistence errors).
            BalanceError: If journal entries do not balance within the
                $0.01 tolerance.
        """
        ...

    # ------------------------------------------------------------------
    # Shared Helper Methods (non-abstract)
    # ------------------------------------------------------------------

    def _create_seeded_rng(self, context: GenerationContext) -> random.Random:
        """Create a deterministic Random instance seeded from the context.

        CRITICAL: Subclasses MUST use this method to obtain an RNG for
        *all* random operations (vendor selection, amount generation,
        discrepancy triggering, etc.).  Direct calls to ``random.random()``
        or ``random.choice()`` on the module-level RNG are FORBIDDEN
        because they break deterministic reproducibility (AAP §0.7.1).

        The seed is derived from the context's ``rng_seed`` so that
        identical seeds produce identical transaction sequences.

        Args:
            context: The current :class:`GenerationContext` carrying the
                ``rng_seed`` value.

        Returns:
            A ``random.Random`` instance seeded with ``context.rng_seed``.
        """
        return random.Random(context.rng_seed)

    async def _publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        context: GenerationContext,
    ) -> None:
        """Publish a cross-subsystem event via the injected EventBus.

        Creates an :class:`~app.events.event_types.Event` instance with the
        supplied type discriminator and payload, then delegates to the
        EventBus's ``publish()`` method.  If no ``EventBus`` was injected,
        a debug-level log is emitted and the call returns immediately.

        The event's ``simulation_id`` is populated from the generation
        context to ensure correct simulation scoping.

        Args:
            event_type: String discriminator identifying the event type
                (e.g., ``"TransactionCreated"``).
            payload: JSONB-compatible dictionary carrying event-specific data.
            context: The current :class:`GenerationContext` for simulation_id
                and trace_id extraction.
        """
        if self._event_bus is None:
            logger.debug(
                "event_publish_skipped_no_bus",
                event_type=event_type,
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            return

        # Lazy import to avoid circular dependency — Event is a lightweight
        # dataclass defined in app.events.event_types
        from app.events.event_types import Event

        event = Event(
            simulation_id=context.simulation_id,
            event_type=event_type,
            payload=payload,
        )

        try:
            await self._event_bus.publish(event)
            logger.debug(
                "event_published",
                event_type=event_type,
                event_id=str(event.event_id),
                simulation_id=str(context.simulation_id),
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "event_publish_timeout",
                event_type=event_type,
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
        except Exception as exc:
            logger.error(
                "event_publish_failed",
                event_type=event_type,
                error=str(exc),
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )

    async def _check_discrepancy_trigger(
        self,
        context: GenerationContext,
        transaction_type: str,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Check whether a discrepancy should be injected for this transaction.

        Delegates to the injected ``DiscrepancyInjector`` to perform a
        rate-based trigger check against the configured discrepancy rate
        and type distribution.  If no injector is available, returns a
        "no injection" triple.

        Args:
            context: The current :class:`GenerationContext` carrying
                discrepancy configuration and RNG seed.
            transaction_type: The transaction type string (e.g.,
                ``"purchase_order"``, ``"sales_order"``).

        Returns:
            A 3-tuple of ``(should_inject, discrepancy_type_code,
            discrepancy_params)`` where:

            - ``should_inject`` is ``True`` when a discrepancy should be
              injected into this transaction.
            - ``discrepancy_type_code`` is the type code string (e.g.,
              ``"P2P-001"``) or ``None``.
            - ``discrepancy_params`` is a parameter dictionary for the
              chosen discrepancy type, or ``None``.
        """
        if self._discrepancy_injector is None:
            return (False, None, None)

        try:
            # DiscrepancyInjector.check_trigger is expected to be an async
            # method returning (bool, Optional[str], Optional[Dict])
            should_inject, disc_type, disc_params = (
                await self._discrepancy_injector.check_trigger(
                    context=context,
                    transaction_type=transaction_type,
                )
            )
            if should_inject:
                logger.debug(
                    "discrepancy_trigger_positive",
                    transaction_type=transaction_type,
                    discrepancy_type=disc_type,
                    trace_id=str(context.trace_id),
                    simulation_id=str(context.simulation_id),
                    service_name="transactions",
                    component=self.__class__.__name__,
                )
            return (should_inject, disc_type, disc_params)
        except Exception as exc:
            # Per AAP §0.1.2 — discrepancy injection fallback is
            # "skip discrepancy" on failure
            logger.warning(
                "discrepancy_trigger_check_failed",
                transaction_type=transaction_type,
                error=str(exc),
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            return (False, None, None)

    async def _delegate_gl_posting(
        self,
        journal_entries: List[Dict[str, Any]],
        context: GenerationContext,
    ) -> None:
        """Delegate GL posting to the injected GLPostingEngine.

        CRITICAL ATOMICITY: If any step in the posting process fails, the
        entire transaction MUST be fully rolled back — no partial postings
        are permitted (AAP §0.7.2).  This method wraps the posting call
        in a ``try / except`` block and re-raises ``TransactionError``
        subclasses after logging the failure.

        If no ``GLPostingEngine`` was injected, a warning is logged and
        the method returns without error.

        Args:
            journal_entries: List of journal entry specification dicts to
                post.  Each dict should contain at minimum ``account``,
                ``debit``, and ``credit`` keys.
            context: The current :class:`GenerationContext` for simulation
                scoping and trace correlation.

        Raises:
            TransactionError: Re-raised from the posting engine if any
                step fails (e.g., BalanceError, GLPostingError,
                PeriodClosedError).
        """
        if self._gl_posting_engine is None:
            logger.warning(
                "gl_posting_skipped_no_engine",
                entry_count=len(journal_entries),
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            return

        if not journal_entries:
            logger.debug(
                "gl_posting_skipped_empty_entries",
                trace_id=str(context.trace_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            return

        try:
            # GLPostingEngine.post_entries is expected to be an async method
            await self._gl_posting_engine.post_entries(
                journal_entries=journal_entries,
                context=context,
            )
            logger.debug(
                "gl_posting_delegated",
                entry_count=len(journal_entries),
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
        except TransactionError:
            # Re-raise known transaction errors (BalanceError, GLPostingError,
            # PeriodClosedError) — caller handles rollback at the session level
            logger.error(
                "gl_posting_failed_transaction_error",
                entry_count=len(journal_entries),
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            raise
        except Exception as exc:
            # Unexpected error — wrap in TransactionError for uniform handling
            logger.error(
                "gl_posting_failed_unexpected",
                entry_count=len(journal_entries),
                error=str(exc),
                error_type=type(exc).__name__,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            raise TransactionError(
                f"GL posting failed unexpectedly: {exc}",
                details={
                    "entry_count": len(journal_entries),
                    "error_type": type(exc).__name__,
                    "trace_id": str(context.trace_id),
                },
            ) from exc

    def _generate_sequential_number(
        self,
        prefix: str,
        year: int,
        sequence: int,
    ) -> str:
        """Generate a sequential document number.

        Produces document identifiers following the AAP numbering pattern:
        ``PREFIX-YYYY-NNNN`` with a zero-padded 4-digit sequence number.

        Examples::

            _generate_sequential_number("PO", 2024, 1)    → "PO-2024-0001"
            _generate_sequential_number("INV", 2024, 42)   → "INV-2024-0042"
            _generate_sequential_number("SO", 2025, 1234)  → "SO-2025-1234"

        Args:
            prefix: Document type prefix (e.g., ``"PO"``, ``"SO"``,
                ``"INV"``, ``"VINV"``).
            year: Four-digit year component.
            sequence: Positive integer sequence number (1-based).

        Returns:
            Formatted document number string.
        """
        return f"{prefix}-{year}-{sequence:04d}"

    def _log_transaction(
        self,
        result: TransactionResult,
        context: GenerationContext,
    ) -> None:
        """Log a completed transaction per AAP §0.7.7.

        Emits a structured JSON log entry containing all fields required by
        the transaction-specific logging specification: ``transaction_type``,
        ``transaction_id``, ``amount``, ``has_discrepancy``,
        ``discrepancy_type``, ``gl_entries`` count, ``duration_ms``,
        ``status``, ``trace_id``, ``simulation_id``, ``service_name``,
        and ``component``.

        Args:
            result: The :class:`TransactionResult` to log.
            context: The :class:`GenerationContext` for trace and simulation
                identifiers.
        """
        logger.info(
            "transaction_generated",
            transaction_type=result.transaction_type,
            transaction_id=str(result.transaction_id),
            amount=str(result.amount) if result.amount is not None else None,
            has_discrepancy=result.has_discrepancy,
            discrepancy_type=result.discrepancy_type,
            gl_entries=len(result.gl_entries),
            duration_ms=result.duration_ms,
            status=result.status,
            trace_id=str(context.trace_id),
            simulation_id=str(context.simulation_id),
            service_name="transactions",
            component=self.__class__.__name__,
        )

    async def _execute_with_timing(
        self,
        context: GenerationContext,
    ) -> TransactionResult:
        """Execute the generate cycle with high-resolution timing.

        Wraps :meth:`generate` with ``time.perf_counter()``-based duration
        measurement and populates ``result.duration_ms``.  On success the
        result is also logged via :meth:`_log_transaction`.

        On :class:`TransactionGenerationError`, the exception is caught and
        a failed :class:`TransactionResult` is returned with ``status="failed"``
        and ``error_message`` populated.

        On any other :class:`TransactionError`, the same pattern applies to
        ensure that the caller always receives a ``TransactionResult`` and
        can decide on retry / escalation.

        Args:
            context: The current :class:`GenerationContext`.

        Returns:
            A :class:`TransactionResult` — either the successful output of
            :meth:`generate` with ``duration_ms`` set, or a failed result
            carrying the error message.
        """
        start = time.perf_counter()
        try:
            result = await self.generate(context)
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            result.duration_ms = elapsed_ms
            self._log_transaction(result, context)
            return result

        except TransactionGenerationError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            logger.error(
                "transaction_generation_failed",
                error=exc.message,
                details=exc.details,
                duration_ms=elapsed_ms,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            return TransactionResult(
                transaction_type=self.__class__.__name__,
                status="failed",
                error_message=exc.message,
                duration_ms=elapsed_ms,
            )

        except TransactionError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            logger.error(
                "transaction_error",
                error=exc.message,
                details=exc.details,
                duration_ms=elapsed_ms,
                trace_id=str(context.trace_id),
                simulation_id=str(context.simulation_id),
                service_name="transactions",
                component=self.__class__.__name__,
            )
            return TransactionResult(
                transaction_type=self.__class__.__name__,
                status="failed",
                error_message=exc.message,
                duration_ms=elapsed_ms,
            )
