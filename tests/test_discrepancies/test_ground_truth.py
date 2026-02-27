"""Comprehensive tests for the GroundTruthGenerator (app/discrepancies/ground_truth_generator.py).

Validates the GroundTruthRecord Pydantic V2 model (16 fields), the GroundTruthGenerator
class, and all associated operations including record creation, serialization,
file output (JSON/CSV), transaction linkage, and metrics tracking.

Per AAP Section 0.7.5:
  - 100% of injected discrepancies MUST have a ground truth record
  - Ground truth schema: 16 fields (discrepancy_id, type_code, category, difficulty,
    transaction_ids, affected_fields, original_values, modified_values,
    detection_method, detection_difficulty, financial_impact, description,
    injection_timestamp, simulation_id, ground_truth_label, metadata)
  - All monetary values use Decimal (prec=28, ROUND_HALF_UP)

Test Classes:
  TestGroundTruthRecord            — Pydantic V2 model: 16 fields, defaults, validation
  TestGroundTruthRecordSerialization — to_dict/from_dict round-trip, UUID→str, Decimal→str
  TestGroundTruthRecordValidation   — min_length constraints, required fields, type coercion
  TestGroundTruthGeneratorConstruction — Constructor, optional params, defaults
  TestCreateRecord                  — create_record() with full and partial kwargs
  TestTransactionLinkage            — transaction_ids list, min_length=1, valid UUIDs
  TestMetrics                       — Record count, category/difficulty counters
  TestWriteJson                     — Async JSON output generation
  TestWriteCsv                      — Async CSV output generation
  TestCoverage                      — 100% coverage: every discrepancy → ground truth record
  TestClear                         — clear() resets records and counters
"""

from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, mock_open, patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.discrepancies.ground_truth_generator import (
    GroundTruthGenerator,
    GroundTruthRecord,
)


# ---------------------------------------------------------------------------
# Deterministic Test UUIDs
# ---------------------------------------------------------------------------
TEST_SIMULATION_ID = UUID("52345678-5234-5678-5234-567852345678")
TEST_DISCREPANCY_ID = UUID("b2345678-b234-5678-b234-5678b2345678")
TEST_TXN_ID_1 = UUID("c2345678-c234-5678-c234-5678c2345678")
TEST_TXN_ID_2 = UUID("d2345678-d234-5678-d234-5678d2345678")
TEST_TXN_ID_3 = UUID("e2345678-e234-5678-e234-5678e2345678")


# ---------------------------------------------------------------------------
# Helper: Build full 16-field kwargs for GroundTruthRecord construction
# ---------------------------------------------------------------------------
def _make_record_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Return a complete dictionary of keyword arguments for GroundTruthRecord.

    All 16 fields are populated with valid defaults. Callers can override any
    field via keyword arguments to test specific scenarios.

    Returns:
        Dict with all 16 GroundTruthRecord fields.
    """
    base: Dict[str, Any] = {
        "discrepancy_id": TEST_DISCREPANCY_ID,
        "type_code": "P2P-001",
        "category": "p2p",
        "difficulty": "easy",
        "transaction_ids": [TEST_TXN_ID_1],
        "affected_fields": ["invoice_amount"],
        "original_values": {"invoice_amount": "1000.00"},
        "modified_values": {"invoice_amount": "1050.00"},
        "detection_method": "duplicate_check",
        "detection_difficulty": "easy",
        "financial_impact": Decimal("25000.00"),
        "description": "Duplicate invoice detected for vendor V-001",
        "injection_timestamp": datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        "simulation_id": TEST_SIMULATION_ID,
        "ground_truth_label": True,
        "metadata": {"source": "test"},
    }
    base.update(overrides)
    return base


def _make_create_record_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Return keyword arguments suitable for ``GroundTruthGenerator.create_record()``.

    Unlike ``_make_record_kwargs``, this omits fields that ``create_record()`` does
    not accept as parameters (discrepancy_id, injection_timestamp, simulation_id,
    ground_truth_label).

    Returns:
        Dict with all required ``create_record()`` keyword arguments.
    """
    base: Dict[str, Any] = {
        "type_code": "P2P-001",
        "category": "p2p",
        "difficulty": "easy",
        "transaction_ids": [TEST_TXN_ID_1],
        "affected_fields": ["invoice_amount"],
        "original_values": {"invoice_amount": "1000.00"},
        "modified_values": {"invoice_amount": "1050.00"},
        "detection_method": "duplicate_check",
        "detection_difficulty": "easy",
        "financial_impact": Decimal("25000.00"),
        "description": "Duplicate invoice detected for vendor V-001",
    }
    base.update(overrides)
    return base


# =========================================================================
# Phase 1: GroundTruthRecord — Pydantic V2 Model Tests (16 Fields)
# =========================================================================


@pytest.mark.discrepancy
class TestGroundTruthRecord:
    """Validate the 16-field Pydantic V2 GroundTruthRecord model.

    Covers field existence, types, default factories, and the total field count.
    """

    def test_all_16_fields_present(self) -> None:
        """Create a GroundTruthRecord with all 16 fields and verify types."""
        kwargs = _make_record_kwargs()
        record = GroundTruthRecord(**kwargs)

        assert isinstance(record.discrepancy_id, UUID)
        assert isinstance(record.type_code, str)
        assert isinstance(record.category, str)
        assert isinstance(record.difficulty, str)
        assert isinstance(record.transaction_ids, list)
        assert isinstance(record.affected_fields, list)
        assert isinstance(record.original_values, dict)
        assert isinstance(record.modified_values, dict)
        assert isinstance(record.detection_method, str)
        assert isinstance(record.detection_difficulty, str)
        assert isinstance(record.financial_impact, Decimal)
        assert isinstance(record.description, str)
        assert isinstance(record.injection_timestamp, datetime)
        assert isinstance(record.simulation_id, UUID)
        assert isinstance(record.ground_truth_label, bool)
        assert isinstance(record.metadata, dict)

    def test_default_discrepancy_id(self) -> None:
        """If discrepancy_id not provided, a UUID is auto-generated."""
        kwargs = _make_record_kwargs()
        del kwargs["discrepancy_id"]
        record = GroundTruthRecord(**kwargs)

        assert isinstance(record.discrepancy_id, UUID)
        # Should be a freshly-generated UUID, not the test constant
        assert record.discrepancy_id != UUID("00000000-0000-0000-0000-000000000000")

    def test_default_injection_timestamp(self) -> None:
        """If injection_timestamp not provided, it defaults to approximately now (UTC)."""
        kwargs = _make_record_kwargs()
        del kwargs["injection_timestamp"]
        before = datetime.now(timezone.utc)
        record = GroundTruthRecord(**kwargs)
        after = datetime.now(timezone.utc)

        assert before <= record.injection_timestamp <= after

    def test_default_ground_truth_label_is_true(self) -> None:
        """Default ground_truth_label is True."""
        kwargs = _make_record_kwargs()
        del kwargs["ground_truth_label"]
        record = GroundTruthRecord(**kwargs)

        assert record.ground_truth_label is True

    def test_default_metadata_is_empty_dict(self) -> None:
        """Default metadata is an empty dictionary."""
        kwargs = _make_record_kwargs()
        del kwargs["metadata"]
        record = GroundTruthRecord(**kwargs)

        assert record.metadata == {}

    def test_financial_impact_is_decimal(self) -> None:
        """financial_impact field must be a Decimal instance — NEVER float."""
        record = GroundTruthRecord(**_make_record_kwargs())

        assert isinstance(record.financial_impact, Decimal)
        assert record.financial_impact == Decimal("25000.00")

    def test_transaction_ids_is_list_of_uuids(self) -> None:
        """Every element in transaction_ids must be a UUID object."""
        kwargs = _make_record_kwargs(transaction_ids=[TEST_TXN_ID_1, TEST_TXN_ID_2])
        record = GroundTruthRecord(**kwargs)

        assert len(record.transaction_ids) == 2
        for tid in record.transaction_ids:
            assert isinstance(tid, UUID)

    def test_affected_fields_is_list_of_strings(self) -> None:
        """Every element in affected_fields must be a string."""
        kwargs = _make_record_kwargs(affected_fields=["amount", "invoice_number"])
        record = GroundTruthRecord(**kwargs)

        for field_name in record.affected_fields:
            assert isinstance(field_name, str)

    def test_field_count(self) -> None:
        """GroundTruthRecord.model_fields must have exactly 16 entries."""
        assert len(GroundTruthRecord.model_fields) == 16


# =========================================================================
# Phase 1 (cont.): Serialization Tests — to_dict() / from_dict()
# =========================================================================


@pytest.mark.discrepancy
class TestGroundTruthRecordSerialization:
    """Validate to_dict() and from_dict() serialization round-trips.

    to_dict() converts UUID→str, Decimal→str, datetime→ISO-8601.
    from_dict() reverses those conversions.
    """

    def _build_record(self) -> GroundTruthRecord:
        """Utility to build a fully-populated record."""
        return GroundTruthRecord(**_make_record_kwargs())

    def test_to_dict_converts_uuids_to_strings(self) -> None:
        """to_dict() converts discrepancy_id, simulation_id, and transaction_ids to strings."""
        record = self._build_record()
        data = record.to_dict()

        assert isinstance(data["discrepancy_id"], str)
        assert isinstance(data["simulation_id"], str)
        for tid in data["transaction_ids"]:
            assert isinstance(tid, str)

    def test_to_dict_converts_decimal_to_string(self) -> None:
        """to_dict() converts financial_impact from Decimal to string."""
        record = self._build_record()
        data = record.to_dict()

        assert isinstance(data["financial_impact"], str)
        assert data["financial_impact"] == "25000.00"

    def test_to_dict_converts_datetime_to_iso(self) -> None:
        """to_dict() converts injection_timestamp to ISO-8601 format string."""
        record = self._build_record()
        data = record.to_dict()

        assert isinstance(data["injection_timestamp"], str)
        # Verify the ISO string is parseable
        parsed = datetime.fromisoformat(data["injection_timestamp"])
        assert parsed.year == 2025
        assert parsed.month == 1

    def test_to_dict_all_keys_present(self) -> None:
        """to_dict() output contains all 16 expected keys."""
        record = self._build_record()
        data = record.to_dict()

        expected_keys = {
            "discrepancy_id", "type_code", "category", "difficulty",
            "transaction_ids", "affected_fields", "original_values",
            "modified_values", "detection_method", "detection_difficulty",
            "financial_impact", "description", "injection_timestamp",
            "simulation_id", "ground_truth_label", "metadata",
        }
        assert set(data.keys()) == expected_keys

    def test_from_dict_roundtrip(self) -> None:
        """Create record → to_dict() → from_dict() produces equivalent record."""
        original = self._build_record()
        serialized = original.to_dict()
        restored = GroundTruthRecord.from_dict(serialized)

        assert restored.discrepancy_id == original.discrepancy_id
        assert restored.type_code == original.type_code
        assert restored.category == original.category
        assert restored.difficulty == original.difficulty
        assert restored.transaction_ids == original.transaction_ids
        assert restored.affected_fields == original.affected_fields
        assert restored.detection_method == original.detection_method
        assert restored.detection_difficulty == original.detection_difficulty
        assert restored.financial_impact == original.financial_impact
        assert restored.description == original.description
        assert restored.simulation_id == original.simulation_id
        assert restored.ground_truth_label == original.ground_truth_label

    def test_from_dict_parses_uuid_strings(self) -> None:
        """from_dict() converts UUID strings back to UUID objects."""
        record = self._build_record()
        serialized = record.to_dict()
        # Verify the serialized form has strings
        assert isinstance(serialized["discrepancy_id"], str)
        assert isinstance(serialized["simulation_id"], str)

        restored = GroundTruthRecord.from_dict(serialized)
        assert isinstance(restored.discrepancy_id, UUID)
        assert isinstance(restored.simulation_id, UUID)
        for tid in restored.transaction_ids:
            assert isinstance(tid, UUID)

    def test_from_dict_parses_decimal_strings(self) -> None:
        """from_dict() converts string '25000.00' back to Decimal('25000.00')."""
        record = self._build_record()
        serialized = record.to_dict()
        assert isinstance(serialized["financial_impact"], str)

        restored = GroundTruthRecord.from_dict(serialized)
        assert isinstance(restored.financial_impact, Decimal)
        assert restored.financial_impact == Decimal("25000.00")

    def test_from_dict_parses_datetime_strings(self) -> None:
        """from_dict() converts ISO string back to datetime object."""
        record = self._build_record()
        serialized = record.to_dict()
        assert isinstance(serialized["injection_timestamp"], str)

        restored = GroundTruthRecord.from_dict(serialized)
        assert isinstance(restored.injection_timestamp, datetime)
        assert restored.injection_timestamp.year == 2025


# =========================================================================
# Phase 1 (cont.): Validation Tests — min_length, required, type coercion
# =========================================================================


@pytest.mark.discrepancy
class TestGroundTruthRecordValidation:
    """Validate Pydantic V2 constraints on GroundTruthRecord fields.

    Tests that empty lists, empty strings, and missing required fields
    trigger ValidationError appropriately.
    """

    def test_transaction_ids_min_length_one(self) -> None:
        """Empty transaction_ids list raises ValidationError (min_length=1)."""
        kwargs = _make_record_kwargs(transaction_ids=[])
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)

    def test_affected_fields_min_length_one(self) -> None:
        """Empty affected_fields list raises ValidationError (min_length=1)."""
        kwargs = _make_record_kwargs(affected_fields=[])
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)

    def test_description_min_length_one(self) -> None:
        """Empty description string raises ValidationError (min_length=1)."""
        kwargs = _make_record_kwargs(description="")
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)

    def test_type_code_required(self) -> None:
        """Missing type_code raises ValidationError."""
        kwargs = _make_record_kwargs()
        del kwargs["type_code"]
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)

    def test_category_required(self) -> None:
        """Missing category raises ValidationError."""
        kwargs = _make_record_kwargs()
        del kwargs["category"]
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)

    def test_simulation_id_required(self) -> None:
        """Missing simulation_id raises ValidationError."""
        kwargs = _make_record_kwargs()
        del kwargs["simulation_id"]
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)

    def test_invalid_uuid_raises(self) -> None:
        """Non-UUID string for discrepancy_id raises ValidationError."""
        kwargs = _make_record_kwargs(discrepancy_id="not-a-uuid")
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)


# =========================================================================
# Phase 2: GroundTruthGenerator Constructor Tests
# =========================================================================


@pytest.mark.discrepancy
class TestGroundTruthGeneratorConstruction:
    """Validate GroundTruthGenerator constructor and default state.

    Tests constructor injection pattern (ADR-003) with optional parameters.
    """

    def test_default_construction(self) -> None:
        """Create with no args; verify auto-generated simulation_id and None output_dir."""
        gen = GroundTruthGenerator()

        assert isinstance(gen._simulation_id, UUID)
        assert gen._output_dir is None

    def test_custom_simulation_id(self) -> None:
        """Pass explicit simulation_id; verify it is stored."""
        gen = GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)

        assert gen._simulation_id == TEST_SIMULATION_ID

    def test_custom_output_dir(self) -> None:
        """Pass output_dir path; verify it is stored."""
        gen = GroundTruthGenerator(output_dir="/tmp/output")

        assert gen._output_dir == "/tmp/output"

    def test_empty_records_on_init(self) -> None:
        """get_records() returns empty list after initialization."""
        gen = GroundTruthGenerator()

        assert gen.get_records() == []

    def test_initial_metrics_zero(self) -> None:
        """get_metrics() returns zeroed counters after initialization."""
        gen = GroundTruthGenerator()
        metrics = gen.get_metrics()

        assert metrics["record_count"] == 0
        assert all(v == 0 for v in metrics["category_counts"].values())
        assert all(v == 0 for v in metrics["difficulty_counts"].values())


# =========================================================================
# Phase 3: create_record() Tests
# =========================================================================


@pytest.mark.discrepancy
class TestCreateRecord:
    """Validate GroundTruthGenerator.create_record() behavior.

    Tests full and minimal kwargs, return type, count increments, and
    category/difficulty counter updates.
    """

    def _make_generator(self, **kwargs: Any) -> GroundTruthGenerator:
        """Build a GroundTruthGenerator with a known simulation_id."""
        return GroundTruthGenerator(
            simulation_id=kwargs.pop("simulation_id", TEST_SIMULATION_ID),
            **kwargs,
        )

    def test_create_record_with_all_fields(self) -> None:
        """Pass all accepted fields via kwargs; verify record is created and added."""
        gen = self._make_generator()
        kwargs = _make_create_record_kwargs(metadata={"source": "test"})
        record = gen.create_record(**kwargs)

        assert isinstance(record, GroundTruthRecord)
        assert record.type_code == "P2P-001"
        assert record.category == "p2p"
        assert record.difficulty == "easy"
        assert record.financial_impact == Decimal("25000.00")
        assert record.metadata == {"source": "test"}
        assert len(gen.get_records()) == 1

    def test_create_record_with_minimal_fields(self) -> None:
        """Pass only required fields; verify defaults are applied for optional fields."""
        gen = self._make_generator()
        kwargs = _make_create_record_kwargs()
        record = gen.create_record(**kwargs)

        assert isinstance(record, GroundTruthRecord)
        # ground_truth_label defaults to True
        assert record.ground_truth_label is True
        # metadata defaults to empty dict when not provided
        assert record.metadata == {}
        # simulation_id comes from the generator
        assert record.simulation_id == TEST_SIMULATION_ID

    def test_create_record_returns_record(self) -> None:
        """create_record() returns a GroundTruthRecord instance."""
        gen = self._make_generator()
        result = gen.create_record(**_make_create_record_kwargs())

        assert isinstance(result, GroundTruthRecord)

    def test_create_record_increments_count(self) -> None:
        """After create_record(), get_records() has one additional record."""
        gen = self._make_generator()
        assert len(gen.get_records()) == 0

        gen.create_record(**_make_create_record_kwargs())
        assert len(gen.get_records()) == 1

        gen.create_record(**_make_create_record_kwargs(type_code="P2P-002"))
        assert len(gen.get_records()) == 2

    def test_create_record_updates_category_counts(self) -> None:
        """Category counter is incremented after create_record()."""
        gen = self._make_generator()
        gen.create_record(**_make_create_record_kwargs(category="p2p"))

        metrics = gen.get_metrics()
        assert metrics["category_counts"]["p2p"] == 1
        assert metrics["category_counts"]["o2c"] == 0

    def test_create_record_updates_difficulty_counts(self) -> None:
        """Difficulty counter is incremented after create_record()."""
        gen = self._make_generator()
        gen.create_record(**_make_create_record_kwargs(difficulty="medium",
                                                       detection_difficulty="medium"))

        metrics = gen.get_metrics()
        assert metrics["difficulty_counts"]["medium"] == 1
        assert metrics["difficulty_counts"]["easy"] == 0

    def test_multiple_records(self) -> None:
        """Create 5 records and verify all 5 are in get_records()."""
        gen = self._make_generator()
        type_codes = ["P2P-001", "P2P-002", "O2C-001", "GL-001", "CTL-001"]
        categories = ["p2p", "p2p", "o2c", "gl", "control"]
        for tc, cat in zip(type_codes, categories):
            gen.create_record(**_make_create_record_kwargs(type_code=tc, category=cat))

        records = gen.get_records()
        assert len(records) == 5

    def test_create_record_uses_generator_simulation_id(self) -> None:
        """Records inherit the generator's simulation_id."""
        custom_sim_id = uuid4()
        gen = self._make_generator(simulation_id=custom_sim_id)
        record = gen.create_record(**_make_create_record_kwargs())

        assert record.simulation_id == custom_sim_id


# =========================================================================
# Phase 4: Transaction Linkage Tests
# =========================================================================


@pytest.mark.discrepancy
class TestTransactionLinkage:
    """Validate transaction_ids field for ground truth record linkage.

    Per AAP §0.7.5: Every ground truth record MUST reference valid
    transaction IDs in the transaction_ids list.
    """

    def test_single_transaction_id(self) -> None:
        """Record with a single transaction_id is valid."""
        kwargs = _make_record_kwargs(transaction_ids=[TEST_TXN_ID_1])
        record = GroundTruthRecord(**kwargs)

        assert len(record.transaction_ids) == 1
        assert record.transaction_ids[0] == TEST_TXN_ID_1

    def test_multiple_transaction_ids(self) -> None:
        """Record linked to 3 transactions (e.g., PO + Receipt + Invoice)."""
        ids = [TEST_TXN_ID_1, TEST_TXN_ID_2, TEST_TXN_ID_3]
        kwargs = _make_record_kwargs(transaction_ids=ids)
        record = GroundTruthRecord(**kwargs)

        assert len(record.transaction_ids) == 3
        assert record.transaction_ids == ids

    def test_transaction_ids_are_valid_uuids(self) -> None:
        """All IDs in transaction_ids are valid UUID objects."""
        ids = [uuid4() for _ in range(4)]
        kwargs = _make_record_kwargs(transaction_ids=ids)
        record = GroundTruthRecord(**kwargs)

        for tid in record.transaction_ids:
            assert isinstance(tid, UUID)

    def test_empty_transaction_ids_rejected(self) -> None:
        """Empty transaction_ids list raises ValidationError (min_length=1)."""
        kwargs = _make_record_kwargs(transaction_ids=[])
        with pytest.raises(ValidationError):
            GroundTruthRecord(**kwargs)


# =========================================================================
# Phase 5: Metrics Tests
# =========================================================================


@pytest.mark.discrepancy
class TestMetrics:
    """Validate GroundTruthGenerator.get_metrics() structure and counters."""

    def _make_generator(self) -> GroundTruthGenerator:
        """Build a GroundTruthGenerator with a known simulation_id."""
        return GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)

    def test_metrics_structure(self) -> None:
        """get_metrics() returns dict with required keys."""
        gen = self._make_generator()
        metrics = gen.get_metrics()

        assert "record_count" in metrics
        assert "category_counts" in metrics
        assert "difficulty_counts" in metrics
        assert "simulation_id" in metrics

    def test_metrics_total_records(self) -> None:
        """After N creates, record_count equals N."""
        gen = self._make_generator()
        for i in range(7):
            gen.create_record(**_make_create_record_kwargs(
                type_code=f"P2P-{i:03d}",
            ))

        metrics = gen.get_metrics()
        assert metrics["record_count"] == 7

    def test_metrics_category_counts(self) -> None:
        """After creating 3 p2p, 2 o2c, 1 gl records: counts match."""
        gen = self._make_generator()
        for _ in range(3):
            gen.create_record(**_make_create_record_kwargs(category="p2p"))
        for _ in range(2):
            gen.create_record(**_make_create_record_kwargs(category="o2c"))
        gen.create_record(**_make_create_record_kwargs(category="gl"))

        metrics = gen.get_metrics()
        assert metrics["category_counts"]["p2p"] == 3
        assert metrics["category_counts"]["o2c"] == 2
        assert metrics["category_counts"]["gl"] == 1
        assert metrics["category_counts"]["control"] == 0

    def test_metrics_difficulty_counts(self) -> None:
        """After creating 4 easy + 2 medium records: counts match."""
        gen = self._make_generator()
        for _ in range(4):
            gen.create_record(**_make_create_record_kwargs(
                difficulty="easy", detection_difficulty="easy",
            ))
        for _ in range(2):
            gen.create_record(**_make_create_record_kwargs(
                difficulty="medium", detection_difficulty="medium",
            ))

        metrics = gen.get_metrics()
        assert metrics["difficulty_counts"]["easy"] == 4
        assert metrics["difficulty_counts"]["medium"] == 2


# =========================================================================
# Phase 6: JSON Output Tests (async)
# =========================================================================


@pytest.mark.discrepancy
class TestWriteJson:
    """Validate GroundTruthGenerator.write_json() async JSON file output.

    Uses tmp_path for real file I/O since aiofiles is installed.
    """

    def _populated_generator(
        self, output_dir: str, count: int = 3,
    ) -> GroundTruthGenerator:
        """Create a generator with ``count`` records and ``output_dir`` set."""
        gen = GroundTruthGenerator(
            simulation_id=TEST_SIMULATION_ID,
            output_dir=output_dir,
        )
        for i in range(count):
            gen.create_record(**_make_create_record_kwargs(
                type_code=f"P2P-{i + 1:03d}",
            ))
        return gen

    @pytest.mark.asyncio
    async def test_write_json_creates_file(self, tmp_path: Path) -> None:
        """write_json() creates a file at the specified path."""
        gen = self._populated_generator(str(tmp_path))
        filepath = str(tmp_path / "test_output.json")
        result = await gen.write_json(filepath)

        assert Path(result).exists()
        assert Path(result).is_file()

    @pytest.mark.asyncio
    async def test_write_json_content_is_valid_json(self, tmp_path: Path) -> None:
        """Written content is valid JSON containing a list of record dicts."""
        gen = self._populated_generator(str(tmp_path))
        filepath = str(tmp_path / "test_output.json")
        await gen.write_json(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = json.loads(f.read())

        assert isinstance(content, dict)
        assert "records" in content
        assert isinstance(content["records"], list)
        for record_dict in content["records"]:
            assert isinstance(record_dict, dict)

    @pytest.mark.asyncio
    async def test_write_json_record_count(self, tmp_path: Path) -> None:
        """Written JSON records list length matches number of created records."""
        gen = self._populated_generator(str(tmp_path), count=5)
        filepath = str(tmp_path / "test_output.json")
        await gen.write_json(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = json.loads(f.read())

        assert content["record_count"] == 5
        assert len(content["records"]) == 5

    @pytest.mark.asyncio
    async def test_write_json_empty_records(self, tmp_path: Path) -> None:
        """With no records, write_json() writes a payload with empty records list."""
        gen = GroundTruthGenerator(
            simulation_id=TEST_SIMULATION_ID,
            output_dir=str(tmp_path),
        )
        filepath = str(tmp_path / "empty_output.json")
        await gen.write_json(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = json.loads(f.read())

        assert content["record_count"] == 0
        assert content["records"] == []

    @pytest.mark.asyncio
    async def test_write_json_uses_to_dict(self, tmp_path: Path) -> None:
        """Each record in the JSON output matches the to_dict() serialization format."""
        gen = self._populated_generator(str(tmp_path), count=1)
        filepath = str(tmp_path / "test_output.json")
        await gen.write_json(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            content = json.loads(f.read())

        json_record = content["records"][0]
        original_record = gen.get_records()[0]
        expected = original_record.to_dict()

        # Verify key overlap — all 16 keys present in both
        assert set(json_record.keys()) == set(expected.keys())
        # Verify UUIDs are serialized as strings
        assert isinstance(json_record["discrepancy_id"], str)
        assert isinstance(json_record["simulation_id"], str)
        assert isinstance(json_record["financial_impact"], str)


# =========================================================================
# Phase 6 (cont.): CSV Output Tests (async)
# =========================================================================


@pytest.mark.discrepancy
class TestWriteCsv:
    """Validate GroundTruthGenerator.write_csv() async CSV file output.

    Uses tmp_path for real file I/O since pandas is installed.
    """

    def _populated_generator(
        self, output_dir: str, count: int = 3,
    ) -> GroundTruthGenerator:
        """Create a generator with ``count`` records."""
        gen = GroundTruthGenerator(
            simulation_id=TEST_SIMULATION_ID,
            output_dir=output_dir,
        )
        for i in range(count):
            gen.create_record(**_make_create_record_kwargs(
                type_code=f"P2P-{i + 1:03d}",
            ))
        return gen

    @pytest.mark.asyncio
    async def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        """write_csv() creates a CSV file at the specified path."""
        gen = self._populated_generator(str(tmp_path))
        filepath = str(tmp_path / "test_output.csv")
        result = await gen.write_csv(filepath)

        assert Path(result).exists()
        assert Path(result).is_file()

    @pytest.mark.asyncio
    async def test_write_csv_column_count(self, tmp_path: Path) -> None:
        """CSV file has 16 columns (one per GroundTruthRecord field)."""
        gen = self._populated_generator(str(tmp_path))
        filepath = str(tmp_path / "test_output.csv")
        await gen.write_csv(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

        assert len(header) == 16

    @pytest.mark.asyncio
    async def test_write_csv_row_count(self, tmp_path: Path) -> None:
        """CSV row count (excluding header) matches record count."""
        gen = self._populated_generator(str(tmp_path), count=4)
        filepath = str(tmp_path / "test_output.csv")
        await gen.write_csv(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            rows = list(reader)

        assert len(rows) == 4

    @pytest.mark.asyncio
    async def test_write_csv_empty_records(self, tmp_path: Path) -> None:
        """With no records, CSV has header row but no data rows."""
        gen = GroundTruthGenerator(
            simulation_id=TEST_SIMULATION_ID,
            output_dir=str(tmp_path),
        )
        filepath = str(tmp_path / "empty_output.csv")
        await gen.write_csv(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert len(header) == 16
        assert len(rows) == 0


# =========================================================================
# Phase 7: 100% Coverage Guarantee Test
# =========================================================================


@pytest.mark.discrepancy
class TestCoverage:
    """Validate the 100% ground truth coverage guarantee.

    Per AAP §0.7.5: 100% of injected discrepancies MUST have a corresponding
    ground truth record with all 16 schema fields populated.
    """

    def test_every_injected_discrepancy_has_ground_truth(self) -> None:
        """Simulate N injections via create_record(); verify len(get_records()) == N."""
        gen = GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)
        injection_count = 20

        discrepancy_types = [
            ("P2P-001", "p2p", "easy"),
            ("P2P-002", "p2p", "medium"),
            ("O2C-001", "o2c", "easy"),
            ("GL-001", "gl", "easy"),
            ("CTL-001", "control", "medium"),
        ]

        for i in range(injection_count):
            tc, cat, diff = discrepancy_types[i % len(discrepancy_types)]
            gen.create_record(**_make_create_record_kwargs(
                type_code=tc,
                category=cat,
                difficulty=diff,
                detection_difficulty=diff,
                transaction_ids=[uuid4()],
            ))

        records = gen.get_records()
        # 100% coverage: one ground truth record per injection
        assert len(records) == injection_count

        # Verify each record has all 16 fields populated
        for record in records:
            assert len(GroundTruthRecord.model_fields) == 16
            data = record.to_dict()
            for key in GroundTruthRecord.model_fields:
                assert key in data, f"Missing key: {key}"

    def test_ground_truth_matches_injection(self) -> None:
        """After injection, ground truth type_code matches the injected discrepancy type."""
        gen = GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)
        injected_types = ["P2P-001", "O2C-003", "GL-002", "CTL-005"]

        categories = {"P2P": "p2p", "O2C": "o2c", "GL": "gl", "CTL": "control"}

        for tc in injected_types:
            prefix = tc.split("-")[0]
            cat = categories[prefix]
            gen.create_record(**_make_create_record_kwargs(
                type_code=tc,
                category=cat,
            ))

        records = gen.get_records()
        for i, record in enumerate(records):
            assert record.type_code == injected_types[i]


# =========================================================================
# Phase 8: Clear / Reset Tests
# =========================================================================


@pytest.mark.discrepancy
class TestClear:
    """Validate GroundTruthGenerator.clear() resets records and counters.

    clear() removes all buffered records and zeroes all metric counters
    but preserves the generator's simulation_id.
    """

    def test_clear_removes_all_records(self) -> None:
        """After clear(), get_records() returns an empty list."""
        gen = GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)
        for _ in range(5):
            gen.create_record(**_make_create_record_kwargs())
        assert len(gen.get_records()) == 5

        gen.clear()
        assert len(gen.get_records()) == 0

    def test_clear_resets_metrics(self) -> None:
        """After clear(), all metric counters are zero."""
        gen = GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)
        gen.create_record(**_make_create_record_kwargs(category="p2p", difficulty="easy"))
        gen.create_record(**_make_create_record_kwargs(
            category="o2c", difficulty="medium", detection_difficulty="medium",
        ))

        gen.clear()
        metrics = gen.get_metrics()

        assert metrics["record_count"] == 0
        assert all(v == 0 for v in metrics["category_counts"].values())
        assert all(v == 0 for v in metrics["difficulty_counts"].values())

    def test_clear_preserves_simulation_id(self) -> None:
        """clear() does NOT reset the generator's simulation_id."""
        gen = GroundTruthGenerator(simulation_id=TEST_SIMULATION_ID)
        gen.create_record(**_make_create_record_kwargs())

        gen.clear()

        assert gen._simulation_id == TEST_SIMULATION_ID
