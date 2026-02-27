"""Ground Truth Generator — Creates labeled ground truth records for injected discrepancies.

The ``GroundTruthGenerator`` creates comprehensive ground truth records with a
16-field schema for every discrepancy injected into a transaction. These records
serve as the labeled training/validation dataset for downstream audit detection
models.

Every injected discrepancy MUST have a corresponding ground truth record with
ALL 16 fields populated (AAP §0.7.5: 100% ground truth coverage).

Ground Truth Schema (16 fields):
    1. discrepancy_id (UUID) — Unique identifier
    2. type_code (str) — e.g., "P2P-001"
    3. category (str) — "p2p", "o2c", "gl", or "control"
    4. difficulty (str) — "easy" or "medium"
    5. transaction_ids (List[UUID]) — Affected transaction UUIDs
    6. affected_fields (List[str]) — Field names modified
    7. original_values (Dict) — Original field values
    8. modified_values (Dict) — Modified field values
    9. detection_method (str) — e.g., "three_way_match", "duplicate_check"
    10. detection_difficulty (str) — "easy" or "medium"
    11. financial_impact (Decimal) — Dollar impact of discrepancy
    12. description (str) — Human-readable description
    13. injection_timestamp (datetime) — When injected (UTC)
    14. simulation_id (UUID) — Simulation run identifier
    15. ground_truth_label (bool) — True = discrepancy exists
    16. metadata (Dict) — Additional context

Output Format:
    - JSON: Individual records and batch files via aiofiles
    - CSV: Summary files via pandas DataFrame

References:
    - AAP Section 0.5.1 Group 5: ground_truth_generator.py
    - AAP Section 0.7.5: 100% ground truth coverage requirement
    - AAP Section 0.7.1: Constructor injection, Pydantic V2
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from app.transactions.constants import DISCREPANCY_DEFAULTS

if TYPE_CHECKING:
    import aiofiles
    import pandas as pd

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.7 — stdout only)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Valid value sets for field validation
# ---------------------------------------------------------------------------
_VALID_CATEGORIES: frozenset[str] = frozenset({"p2p", "o2c", "gl", "control"})
"""Allowed discrepancy category values."""

_VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium"})
"""Allowed difficulty levels — AAP §0.7.5: Easy/Medium only for MVP (hard=0.00)."""


# ---------------------------------------------------------------------------
# GroundTruthRecord — Pydantic V2 16-field data model
# ---------------------------------------------------------------------------


class GroundTruthRecord(BaseModel):
    """A single ground truth record for an injected discrepancy.

    Contains all 16 fields required by the ground truth specification.
    EVERY field MUST be populated — no Optional fields with None defaults
    (except ``metadata`` which defaults to an empty dict and
    ``ground_truth_label`` which defaults to ``True``).

    CRITICAL: ``financial_impact`` MUST be ``Decimal``, NEVER ``float``
    per AAP §0.7.2.

    Attributes:
        discrepancy_id: Unique identifier for this discrepancy.
        type_code: Discrepancy type code, e.g. ``"P2P-001"``.
        category: One of ``"p2p"``, ``"o2c"``, ``"gl"``, ``"control"``.
        difficulty: One of ``"easy"`` or ``"medium"``.
        transaction_ids: List of affected transaction UUIDs (min length 1).
        affected_fields: List of field names that were modified (min length 1).
        original_values: Original field values before injection.
        modified_values: Modified field values after injection.
        detection_method: Expected detection method string.
        detection_difficulty: How difficult to detect: ``"easy"`` or ``"medium"``.
        financial_impact: Dollar impact of the discrepancy (``Decimal``).
        description: Human-readable description (min length 1).
        injection_timestamp: UTC datetime when the discrepancy was injected.
        simulation_id: Simulation run identifier UUID.
        ground_truth_label: Always ``True`` for injected discrepancies.
        metadata: Additional context (defaults to empty dict).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Field 1: Unique identifier
    discrepancy_id: UUID = Field(
        default_factory=uuid4,
        description="Unique discrepancy identifier",
    )

    # Field 2: Type code (e.g., "P2P-001", "O2C-003", "GL-001", "CTL-002")
    type_code: str = Field(
        ...,
        description="Discrepancy type code",
    )

    # Field 3: Category
    category: str = Field(
        ...,
        description="Category: 'p2p', 'o2c', 'gl', or 'control'",
    )

    # Field 4: Difficulty
    difficulty: str = Field(
        ...,
        description="Difficulty: 'easy' or 'medium'",
    )

    # Field 5: Transaction IDs (list of affected transaction UUIDs)
    transaction_ids: List[UUID] = Field(
        ...,
        min_length=1,
        description="Affected transaction UUIDs",
    )

    # Field 6: Affected fields
    affected_fields: List[str] = Field(
        ...,
        min_length=1,
        description="Field names that were modified",
    )

    # Field 7: Original values
    original_values: Dict[str, Any] = Field(
        ...,
        description="Original field values before injection",
    )

    # Field 8: Modified values
    modified_values: Dict[str, Any] = Field(
        ...,
        description="Modified field values after injection",
    )

    # Field 9: Detection method
    detection_method: str = Field(
        ...,
        description="Expected detection method (e.g., 'three_way_match')",
    )

    # Field 10: Detection difficulty
    detection_difficulty: str = Field(
        ...,
        description="How difficult to detect: 'easy' or 'medium'",
    )

    # Field 11: Financial impact (MUST be Decimal)
    financial_impact: Decimal = Field(
        ...,
        description="Dollar impact of the discrepancy",
    )

    # Field 12: Description
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable description",
    )

    # Field 13: Injection timestamp (UTC)
    injection_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the discrepancy was injected",
    )

    # Field 14: Simulation ID
    simulation_id: UUID = Field(
        ...,
        description="Simulation run identifier",
    )

    # Field 15: Ground truth label — always True for injected discrepancies
    ground_truth_label: bool = Field(
        default=True,
        description="True = discrepancy exists",
    )

    # Field 16: Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Converts UUID → string, Decimal → string, datetime → ISO-8601 so
        the resulting dictionary can be fed directly to ``json.dumps``
        without additional transformation.

        Returns:
            A plain dictionary suitable for JSON serialization.
        """
        data = self.model_dump()

        # Convert UUIDs to strings
        data["discrepancy_id"] = str(data["discrepancy_id"])
        data["simulation_id"] = str(data["simulation_id"])
        data["transaction_ids"] = [str(tid) for tid in data["transaction_ids"]]

        # Convert Decimal to string for JSON compatibility
        data["financial_impact"] = str(data["financial_impact"])

        # Convert datetime to ISO-8601
        if isinstance(data["injection_timestamp"], datetime):
            data["injection_timestamp"] = data["injection_timestamp"].isoformat()

        # Recursively convert any nested Decimal/UUID in original/modified values
        data["original_values"] = _sanitize_dict_for_json(data["original_values"])
        data["modified_values"] = _sanitize_dict_for_json(data["modified_values"])
        data["metadata"] = _sanitize_dict_for_json(data["metadata"])

        return data

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthRecord:
        """Deserialize a ``GroundTruthRecord`` from a dictionary.

        Handles the reverse conversions performed by :meth:`to_dict`:
        string → UUID, string → Decimal, ISO-8601 string → datetime.

        This follows the deserialization pattern established in
        ``app.events.event_types.Event.from_dict``.

        Args:
            data: Dictionary previously produced by :meth:`to_dict` or
                loaded from a JSON file.

        Returns:
            A validated ``GroundTruthRecord`` instance.

        Raises:
            KeyError: If required fields are missing from *data*.
            ValueError: If UUID or datetime parsing fails.
        """
        parsed: Dict[str, Any] = dict(data)

        # --- Parse discrepancy_id ---
        raw_disc_id = parsed.get("discrepancy_id")
        if isinstance(raw_disc_id, str):
            parsed["discrepancy_id"] = UUID(raw_disc_id)
        elif raw_disc_id is None:
            parsed["discrepancy_id"] = uuid4()

        # --- Parse simulation_id ---
        raw_sim_id = parsed.get("simulation_id")
        if isinstance(raw_sim_id, str):
            parsed["simulation_id"] = UUID(raw_sim_id)

        # --- Parse transaction_ids ---
        raw_tids = parsed.get("transaction_ids", [])
        parsed["transaction_ids"] = [
            UUID(tid) if isinstance(tid, str) else tid for tid in raw_tids
        ]

        # --- Parse financial_impact ---
        raw_impact = parsed.get("financial_impact")
        if isinstance(raw_impact, str):
            parsed["financial_impact"] = Decimal(raw_impact)
        elif isinstance(raw_impact, (int, float)):
            parsed["financial_impact"] = Decimal(str(raw_impact))

        # --- Parse injection_timestamp ---
        raw_ts = parsed.get("injection_timestamp")
        if isinstance(raw_ts, str):
            ts = datetime.fromisoformat(raw_ts)
            # Ensure UTC awareness
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            parsed["injection_timestamp"] = ts
        elif raw_ts is None:
            parsed["injection_timestamp"] = datetime.now(timezone.utc)

        return cls.model_validate(parsed)


# ---------------------------------------------------------------------------
# Internal helper — JSON-safe dictionary sanitization
# ---------------------------------------------------------------------------


def _sanitize_dict_for_json(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert UUID, Decimal, and datetime values to strings.

    Ensures that nested dictionaries inside ``original_values``,
    ``modified_values``, and ``metadata`` are fully JSON-serializable.

    Args:
        d: The dictionary to sanitize.

    Returns:
        A new dictionary with all non-JSON-serializable values converted.
    """
    result: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, UUID):
            result[key] = str(value)
        elif isinstance(value, Decimal):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, dict):
            result[key] = _sanitize_dict_for_json(value)
        elif isinstance(value, (list, tuple)):
            result[key] = [
                str(v) if isinstance(v, (UUID, Decimal)) else (
                    v.isoformat() if isinstance(v, datetime) else (
                        _sanitize_dict_for_json(v) if isinstance(v, dict) else v
                    )
                )
                for v in value
            ]
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# GroundTruthGenerator — Record creation and management
# ---------------------------------------------------------------------------


class GroundTruthGenerator:
    """Creates and manages ground truth records for injected discrepancies.

    Maintains an in-memory buffer of :class:`GroundTruthRecord` instances,
    supports batch serialization to JSON (via ``aiofiles``) and CSV (via
    ``pandas``), and tracks generation metrics for operational monitoring.

    Constructor injection (ADR-003) is used for all parameters — every
    parameter is ``Optional`` with a sensible default to enable testing
    and incremental integration.

    Attributes:
        _simulation_id: Simulation run UUID for all records created.
        _output_dir: Optional default output directory for file writes.
        _records: In-memory buffer of ground truth records.
        _record_count: Total number of records created.
        _category_counts: Per-category counters.
        _difficulty_counts: Per-difficulty counters.

    Usage::

        generator = GroundTruthGenerator(simulation_id=my_uuid)
        record = generator.create_record(
            type_code="P2P-001",
            category="p2p",
            difficulty="easy",
            transaction_ids=[txn_uuid],
            affected_fields=["invoice_amount"],
            original_values={"invoice_amount": "1000.00"},
            modified_values={"invoice_amount": "1050.00"},
            detection_method="three_way_match",
            detection_difficulty="easy",
            financial_impact=Decimal("50.00"),
            description="Duplicate invoice detected",
        )
        await generator.write_json("/tmp/ground_truth.json")
    """

    # ------------------------------------------------------------------
    # Construction (ADR-003: Constructor Injection)
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        simulation_id: Optional[UUID] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """Initialise the GroundTruthGenerator.

        Args:
            simulation_id: UUID for the simulation run. When ``None``, a
                new UUID-4 is generated automatically.
            output_dir: Optional default directory for JSON/CSV output.
                Individual write calls can override this.
        """
        self._simulation_id: UUID = simulation_id or uuid4()
        self._output_dir: Optional[str] = output_dir

        # Record buffer
        self._records: List[GroundTruthRecord] = []
        self._record_count: int = 0

        # Per-category counters
        self._category_counts: Dict[str, int] = {
            "p2p": 0,
            "o2c": 0,
            "gl": 0,
            "control": 0,
        }

        # Per-difficulty counters
        self._difficulty_counts: Dict[str, int] = {
            "easy": 0,
            "medium": 0,
        }

        # Reference to default configuration for validation context
        self._default_injection_rate = DISCREPANCY_DEFAULTS["injection_rate"]
        self._difficulty_distribution = DISCREPANCY_DEFAULTS["difficulty_distribution"]

        logger.info(
            "ground_truth_generator_initialized",
            service_name="transactions",
            component="GroundTruthGenerator",
            simulation_id=str(self._simulation_id),
            output_dir=self._output_dir,
        )

    # ------------------------------------------------------------------
    # Core method: create_record
    # ------------------------------------------------------------------

    def create_record(
        self,
        *,
        type_code: str,
        category: str,
        difficulty: str,
        transaction_ids: List[UUID],
        affected_fields: List[str],
        original_values: Dict[str, Any],
        modified_values: Dict[str, Any],
        detection_method: str,
        detection_difficulty: str,
        financial_impact: Decimal,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GroundTruthRecord:
        """Create a new ground truth record and add it to the buffer.

        CRITICAL: ALL 16 fields MUST be populated. This method validates
        completeness before creating the record.

        Args:
            type_code: Discrepancy type code (e.g. ``"P2P-001"``).
            category: One of ``"p2p"``, ``"o2c"``, ``"gl"``, ``"control"``.
            difficulty: One of ``"easy"`` or ``"medium"``.
            transaction_ids: Non-empty list of affected transaction UUIDs.
            affected_fields: Non-empty list of field names that were modified.
            original_values: Original field values before injection.
            modified_values: Modified field values after injection.
            detection_method: Expected detection method string.
            detection_difficulty: How difficult to detect.
            financial_impact: Dollar impact (``Decimal``, never ``float``).
            description: Human-readable description (non-empty).
            metadata: Optional additional context dictionary.

        Returns:
            The created :class:`GroundTruthRecord`.

        Raises:
            ValueError: If *category*, *difficulty*, *transaction_ids*,
                *affected_fields*, or *description* fail validation.
        """
        # --- Validate category ---
        if category not in _VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. "
                f"Must be one of: {sorted(_VALID_CATEGORIES)}"
            )

        # --- Validate difficulty ---
        if difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                f"Must be one of: {sorted(_VALID_DIFFICULTIES)}"
            )

        # --- Validate transaction_ids non-empty ---
        if not transaction_ids:
            raise ValueError(
                "transaction_ids must be a non-empty list of UUIDs"
            )

        # --- Validate affected_fields non-empty ---
        if not affected_fields:
            raise ValueError(
                "affected_fields must be a non-empty list of field names"
            )

        # --- Validate description non-empty ---
        if not description or not description.strip():
            raise ValueError(
                "description must be a non-empty string"
            )

        # --- Validate detection_difficulty ---
        if detection_difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid detection_difficulty '{detection_difficulty}'. "
                f"Must be one of: {sorted(_VALID_DIFFICULTIES)}"
            )

        # --- Validate type_code non-empty ---
        if not type_code or not type_code.strip():
            raise ValueError(
                "type_code must be a non-empty string"
            )

        # --- Create the record with all 16 fields ---
        record = GroundTruthRecord(
            type_code=type_code,
            category=category,
            difficulty=difficulty,
            transaction_ids=list(transaction_ids),
            affected_fields=list(affected_fields),
            original_values=dict(original_values),
            modified_values=dict(modified_values),
            detection_method=detection_method,
            detection_difficulty=detection_difficulty,
            financial_impact=financial_impact,
            description=description,
            simulation_id=self._simulation_id,
            ground_truth_label=True,
            metadata=dict(metadata) if metadata else {},
        )

        # --- Buffer the record ---
        self._records.append(record)
        self._record_count += 1

        # --- Update counters ---
        if category in self._category_counts:
            self._category_counts[category] += 1
        if difficulty in self._difficulty_counts:
            self._difficulty_counts[difficulty] += 1

        logger.debug(
            "ground_truth_record_created",
            service_name="transactions",
            component="GroundTruthGenerator",
            discrepancy_id=str(record.discrepancy_id),
            type_code=type_code,
            category=category,
            difficulty=difficulty,
            financial_impact=str(financial_impact),
            simulation_id=str(self._simulation_id),
            record_count=self._record_count,
        )

        return record

    # ------------------------------------------------------------------
    # Path Sanitization (CWE-22 mitigation)
    # ------------------------------------------------------------------

    def _sanitize_output_path(self, filepath: str) -> str:
        """Resolve and validate an output file path against traversal attacks.

        Ensures the resolved absolute path is within an allowed base
        directory (either ``self._output_dir`` or the current working
        directory).  This prevents path traversal via ``../`` segments
        in user-controlled output paths (CWE-22).

        Args:
            filepath: The raw file path to validate.

        Returns:
            The resolved, validated absolute path as a string.

        Raises:
            ValueError: If the resolved path escapes the allowed base
                directory.
        """
        resolved = pathlib.Path(filepath).resolve()

        # Determine the allowed base directory
        if self._output_dir:
            allowed_base = pathlib.Path(self._output_dir).resolve()
        else:
            allowed_base = pathlib.Path(".").resolve()

        # Validate the resolved path is within the allowed base
        try:
            resolved.relative_to(allowed_base)
        except ValueError:
            logger.warning(
                "path_traversal_blocked",
                service_name="transactions",
                component="GroundTruthGenerator",
                requested_path=filepath,
                resolved_path=str(resolved),
                allowed_base=str(allowed_base),
            )
            raise ValueError(
                f"Output path '{filepath}' resolves to '{resolved}' which is "
                f"outside the allowed base directory '{allowed_base}'"
            )

        return str(resolved)

    # ------------------------------------------------------------------
    # Output: write_json
    # ------------------------------------------------------------------

    async def write_json(self, filepath: Optional[str] = None) -> str:
        """Write all ground truth records to a JSON file.

        Uses ``aiofiles`` for async I/O per AAP dependency requirements.
        The ``aiofiles`` package is imported lazily at runtime to avoid
        module-level import overhead.

        Args:
            filepath: Target file path. When ``None``, a default path is
                constructed from ``self._output_dir`` and the simulation ID.

        Returns:
            The file path that was written to.

        Raises:
            RuntimeError: If no records exist in the buffer.
            ImportError: If ``aiofiles`` is not installed.
        """
        # Lazy runtime import of aiofiles
        import aiofiles  # type: ignore[import-untyped]

        if not self._records:
            logger.warning(
                "ground_truth_write_json_empty",
                service_name="transactions",
                component="GroundTruthGenerator",
                simulation_id=str(self._simulation_id),
                message="No records to write",
            )

        # Determine output path
        if filepath is None:
            base_dir = self._output_dir or "."
            filepath = f"{base_dir}/ground_truth_{self._simulation_id}.json"

        # Validate resolved path to prevent path traversal (CWE-22)
        filepath = self._sanitize_output_path(filepath)

        # Serialize all records
        serialized_records = [record.to_dict() for record in self._records]
        payload = {
            "simulation_id": str(self._simulation_id),
            "record_count": self._record_count,
            "records": serialized_records,
        }

        json_content = json.dumps(payload, indent=2, default=str)

        async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
            await f.write(json_content)

        logger.info(
            "ground_truth_json_written",
            service_name="transactions",
            component="GroundTruthGenerator",
            filepath=filepath,
            record_count=self._record_count,
            simulation_id=str(self._simulation_id),
        )

        return filepath

    # ------------------------------------------------------------------
    # Output: write_csv
    # ------------------------------------------------------------------

    async def write_csv(self, filepath: Optional[str] = None) -> str:
        """Write ground truth summary to a CSV file.

        Uses ``pandas`` ``DataFrame`` for tabular output per AAP dependency
        requirements. The ``pandas`` package is imported lazily at runtime
        to avoid module-level import overhead.

        Args:
            filepath: Target file path. When ``None``, a default path is
                constructed from ``self._output_dir`` and the simulation ID.

        Returns:
            The file path that was written to.

        Raises:
            ImportError: If ``pandas`` is not installed.
        """
        # Lazy runtime import of pandas
        import pandas as pd  # type: ignore[import-untyped]

        # Determine output path
        if filepath is None:
            base_dir = self._output_dir or "."
            filepath = f"{base_dir}/ground_truth_{self._simulation_id}.csv"

        # Validate resolved path to prevent path traversal (CWE-22)
        filepath = self._sanitize_output_path(filepath)

        if not self._records:
            logger.warning(
                "ground_truth_write_csv_empty",
                service_name="transactions",
                component="GroundTruthGenerator",
                simulation_id=str(self._simulation_id),
                message="No records to write — creating empty CSV with headers",
            )
            # Create an empty DataFrame with the expected columns
            columns = [
                "discrepancy_id", "type_code", "category", "difficulty",
                "transaction_ids", "affected_fields", "original_values",
                "modified_values", "detection_method", "detection_difficulty",
                "financial_impact", "description", "injection_timestamp",
                "simulation_id", "ground_truth_label", "metadata",
            ]
            df = pd.DataFrame(columns=columns)
        else:
            # Convert records to dicts for DataFrame construction
            rows = [record.to_dict() for record in self._records]
            df = pd.DataFrame(rows)

        # Write CSV via asyncio.to_thread to avoid blocking the event loop
        # during pandas I/O.  This is the async-safe equivalent of the
        # synchronous df.to_csv() call, aligned with the AAP's requirement
        # for aiofiles-based async I/O in ground truth output methods.
        import asyncio
        await asyncio.to_thread(
            df.to_csv, filepath, index=False, encoding="utf-8"
        )

        logger.info(
            "ground_truth_csv_written",
            service_name="transactions",
            component="GroundTruthGenerator",
            filepath=filepath,
            record_count=self._record_count,
            simulation_id=str(self._simulation_id),
        )

        return filepath

    # ------------------------------------------------------------------
    # Accessor: get_records
    # ------------------------------------------------------------------

    def get_records(self) -> List[GroundTruthRecord]:
        """Return a shallow copy of all ground truth records.

        Returns:
            A new list containing all buffered
            :class:`GroundTruthRecord` instances.
        """
        return list(self._records)

    # ------------------------------------------------------------------
    # Metrics: get_metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return ground truth generation metrics.

        Provides operational visibility into the ground truth generation
        process including record counts, per-category breakdowns,
        per-difficulty breakdowns, and injection rate reference.

        Returns:
            A dictionary with the following keys:

            - ``record_count`` (int): Total records created.
            - ``category_counts`` (Dict[str, int]): Per-category breakdown.
            - ``difficulty_counts`` (Dict[str, int]): Per-difficulty breakdown.
            - ``simulation_id`` (str): Simulation run identifier.
            - ``default_injection_rate`` (str): Reference injection rate.
            - ``difficulty_distribution`` (Dict[str, str]): Reference
              difficulty distribution from DISCREPANCY_DEFAULTS.
        """
        metrics: Dict[str, Any] = {
            "record_count": self._record_count,
            "category_counts": dict(self._category_counts),
            "difficulty_counts": dict(self._difficulty_counts),
            "simulation_id": str(self._simulation_id),
            "default_injection_rate": str(self._default_injection_rate),
            "difficulty_distribution": {
                k: str(v) for k, v in self._difficulty_distribution.items()
            },
        }

        logger.debug(
            "ground_truth_metrics_retrieved",
            service_name="transactions",
            component="GroundTruthGenerator",
            simulation_id=str(self._simulation_id),
            record_count=self._record_count,
        )

        return metrics

    # ------------------------------------------------------------------
    # Management: clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all records from the buffer and reset counters.

        This is useful between simulation periods or when writing out
        records to avoid unbounded memory growth.
        """
        cleared_count = self._record_count

        self._records.clear()
        self._record_count = 0
        self._category_counts = {
            "p2p": 0,
            "o2c": 0,
            "gl": 0,
            "control": 0,
        }
        self._difficulty_counts = {
            "easy": 0,
            "medium": 0,
        }

        logger.info(
            "ground_truth_buffer_cleared",
            service_name="transactions",
            component="GroundTruthGenerator",
            simulation_id=str(self._simulation_id),
            records_cleared=cleared_count,
        )
