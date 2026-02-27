"""Comprehensive tests for the DiscrepancyCatalog (app/discrepancies/discrepancy_catalog.py).

Validates catalog completeness (35+ types), type code uniqueness, CatalogEntry
Pydantic V2 model, registration/lookup operations, parameter bounds retrieval,
implementation class resolution (lazy import), YAML config loading, and metrics.

Per AAP Section 0.5.1 Group 5 and Section 0.7.5:
  - 35+ types registered: 15 P2P, 10 O2C, 5 GL, 5 Control
  - Type codes: P2P-001..P2P-015, O2C-001..O2C-010, GL-001..GL-005, CTL-001..CTL-005
  - Categories: p2p, o2c, gl, control
  - Difficulties: easy and medium only (hard=0 in MVP)
  - Parameter bounds per type (e.g., days_apart: 1-90 for P2P-001)
  - Detection methods vary per type
  - Base rates are Decimal values

Test Classes:
  TestCatalogEntry                 — Pydantic V2 model: fields, validation, defaults
  TestCatalogConstruction          — Constructor, initial state, register_defaults()
  TestRegistration                 — register(), duplicate rejection, class_ref
  TestRegisterDefaults             — All 35 types registered, counts, type codes
  TestLookupByTypeCode             — get_entry(), get_parameter_bounds(), get_implementation_class()
  TestLookupByCategory             — get_by_category("p2p") returns 15, etc.
  TestLookupByDifficulty           — get_by_difficulty("easy"/"medium")
  TestLookupByCategoryAndDifficulty — Combined filter
  TestTypeCodeUniqueness           — No duplicate type codes allowed
  TestParameterBounds              — Per-type parameter bounds verification
  TestImplementationClassLookup    — Lazy import resolution for all 35 types
  TestYAMLLoading                  — load_from_yaml() with mock config files
  TestMetrics                      — Total registered, by category, by difficulty
  TestEdgeCases                    — Missing type codes, invalid categories, empty catalog
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.discrepancies.discrepancy_catalog import (
    CatalogEntry,
    DiscrepancyCatalog,
)
from app.discrepancies.base_discrepancy import BaseDiscrepancy


# ---------------------------------------------------------------------------
# Constants — Expected Type Code Registry
# ---------------------------------------------------------------------------

# All 35 expected type codes after register_defaults()
ALL_TYPE_CODES: list[str] = [
    # P2P (15)
    "P2P-001", "P2P-002", "P2P-003", "P2P-004", "P2P-005",
    "P2P-006", "P2P-007", "P2P-008", "P2P-009", "P2P-010",
    "P2P-011", "P2P-012", "P2P-013", "P2P-014", "P2P-015",
    # O2C (10)
    "O2C-001", "O2C-002", "O2C-003", "O2C-004", "O2C-005",
    "O2C-006", "O2C-007", "O2C-008", "O2C-009", "O2C-010",
    # GL (5)
    "GL-001", "GL-002", "GL-003", "GL-004", "GL-005",
    # Control (5)
    "CTL-001", "CTL-002", "CTL-003", "CTL-004", "CTL-005",
]

# Expected counts per category
EXPECTED_CATEGORY_COUNTS: Dict[str, int] = {
    "p2p": 15,
    "o2c": 10,
    "gl": 5,
    "control": 5,
}

# Types with configurable parameters and their expected bounds (subset for validation).
# Matches the ACTUAL parameter_bounds registered in discrepancy_catalog.py.
TYPES_WITH_PARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "P2P-001": {
        "days_apart": {"min": 1, "max": 90},
        "amount_variance_pct": {"min": 0, "max": 5},
    },
    "P2P-002": {"variance_percent": {"min": 1, "max": 50}},
    "P2P-003": {"variance_percent": {"min": 1, "max": 30}},
    "P2P-006": {"days_before": {"min": 1, "max": 30}},
    "P2P-007": {"round_to": {"min": 100, "max": 10000}},
    "P2P-009": {"days_apart": {"min": 1, "max": 90}},
    "P2P-010": {"days_before": {"min": 1, "max": 30}},
    "P2P-012": {"split_count": {"min": 2, "max": 5}},
    "P2P-013": {"match_type": {"min": 0, "max": 2}},
    "P2P-014": {"concentration_threshold": {"min": Decimal("0.3"), "max": Decimal("0.9")}},
    "O2C-001": {"days_apart": {"min": 1, "max": 60}},
    "O2C-003": {"excess_percent": {"min": 1, "max": 50}},
    "O2C-004": {"short_percent": {"min": 1, "max": 20}},
    "O2C-005": {"overpay_percent": {"min": 1, "max": 30}},
    "O2C-006": {"days_early": {"min": 1, "max": 60}},
    "O2C-009": {"volume_multiplier": {"min": Decimal("1.5"), "max": Decimal("5.0")}},
    "O2C-010": {"discount_percent": {"min": 5, "max": 30}},
    "GL-001": {"imbalance_amount": {"min": Decimal("0.02"), "max": Decimal("1000")}},
    "GL-003": {"days_before_close": {"min": 1, "max": 5}},
    "CTL-003": {"excess_percent": {"min": 1, "max": 100}},
    "CTL-004": {"days_back": {"min": 1, "max": 90}},
}

# Types with NO configurable parameters (empty parameter_bounds)
TYPES_WITHOUT_PARAMS: list[str] = [
    "P2P-004", "P2P-005", "P2P-008", "P2P-011", "P2P-015",
    "O2C-002", "O2C-007", "O2C-008",
    "GL-002", "GL-004", "GL-005",
    "CTL-001", "CTL-002", "CTL-005",
]

# Mapping of type code prefix to expected category string
_PREFIX_TO_CATEGORY: Dict[str, str] = {
    "P2P": "p2p",
    "O2C": "o2c",
    "GL": "gl",
    "CTL": "control",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog_with_defaults() -> DiscrepancyCatalog:
    """Create and return a DiscrepancyCatalog with all 35 defaults registered."""
    catalog = DiscrepancyCatalog()
    catalog.register_defaults()
    return catalog


def _make_entry(**overrides: Any) -> CatalogEntry:
    """Create a CatalogEntry with sensible defaults, allowing overrides."""
    fields: Dict[str, Any] = {
        "type_code": "TEST-001",
        "name": "Test Discrepancy",
        "category": "p2p",
        "difficulty": "easy",
        "base_rate": Decimal("0.02"),
        "description": "A test discrepancy for unit tests.",
        "detection_method": "test_check",
        "parameter_bounds": {},
        "implementation_class": None,
    }
    fields.update(overrides)
    return CatalogEntry(**fields)


# =========================================================================
# Phase 1: CatalogEntry Pydantic Model Tests
# =========================================================================


@pytest.mark.discrepancy
class TestCatalogEntry:
    """Tests for the CatalogEntry Pydantic V2 model — fields, types, validation."""

    def test_create_entry_with_all_fields(self) -> None:
        """CatalogEntry creation with all fields populated should succeed."""
        entry = CatalogEntry(
            type_code="P2P-001",
            name="Duplicate Invoice",
            category="p2p",
            difficulty="easy",
            base_rate=Decimal("0.02"),
            description="Exact or near-duplicate vendor invoice.",
            detection_method="duplicate_check",
            parameter_bounds={"days_apart": {"min": 1, "max": 90}},
            implementation_class="app.discrepancies.p2p.duplicate_invoice.DuplicateInvoice",
        )
        assert entry.type_code == "P2P-001"
        assert entry.name == "Duplicate Invoice"
        assert entry.category == "p2p"
        assert entry.difficulty == "easy"
        assert entry.base_rate == Decimal("0.02")
        assert entry.description == "Exact or near-duplicate vendor invoice."
        assert entry.detection_method == "duplicate_check"
        assert "days_apart" in entry.parameter_bounds
        assert entry.implementation_class is not None

    def test_base_rate_is_decimal(self) -> None:
        """base_rate field must be Decimal type, never float."""
        entry = _make_entry(base_rate=Decimal("0.05"))
        assert isinstance(entry.base_rate, Decimal)
        assert entry.base_rate == Decimal("0.05")

    def test_parameter_bounds_is_dict(self) -> None:
        """parameter_bounds must be a Dict[str, Dict[str, Any]]."""
        bounds = {"variance_percent": {"min": 1, "max": 50, "type": "float"}}
        entry = _make_entry(parameter_bounds=bounds)
        assert isinstance(entry.parameter_bounds, dict)
        assert "variance_percent" in entry.parameter_bounds
        assert entry.parameter_bounds["variance_percent"]["min"] == 1
        assert entry.parameter_bounds["variance_percent"]["max"] == 50

    def test_implementation_class_optional(self) -> None:
        """implementation_class can be None (default)."""
        entry = _make_entry(implementation_class=None)
        assert entry.implementation_class is None

    def test_implementation_class_accepts_string(self) -> None:
        """implementation_class accepts a dotted path string."""
        path = "app.discrepancies.p2p.duplicate_invoice.DuplicateInvoice"
        entry = _make_entry(implementation_class=path)
        assert entry.implementation_class == path

    def test_model_dump_returns_dict(self) -> None:
        """CatalogEntry.model_dump() should produce a serializable dictionary."""
        entry = _make_entry()
        dumped = entry.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["type_code"] == "TEST-001"
        assert dumped["category"] == "p2p"

    def test_model_validate_round_trip(self) -> None:
        """model_dump/model_validate round-trip preserves all fields."""
        original = _make_entry(
            type_code="RT-001",
            name="Round Trip Test",
            base_rate=Decimal("0.03"),
            parameter_bounds={"x": {"min": 1, "max": 10}},
        )
        dumped = original.model_dump()
        restored = CatalogEntry.model_validate(dumped)
        assert restored.type_code == original.type_code
        assert restored.name == original.name
        assert restored.base_rate == original.base_rate
        assert restored.parameter_bounds == original.parameter_bounds

    def test_defaults_for_optional_fields(self) -> None:
        """CatalogEntry defaults: description='', detection_method='', parameter_bounds={}, implementation_class=None."""
        entry = CatalogEntry(
            type_code="MIN-001",
            name="Minimal Entry",
            category="p2p",
            difficulty="easy",
            base_rate=Decimal("0.01"),
        )
        assert entry.description == ""
        assert entry.detection_method == ""
        assert entry.parameter_bounds == {}
        assert entry.implementation_class is None


# =========================================================================
# Phase 2: Catalog Construction Tests
# =========================================================================


@pytest.mark.discrepancy
class TestCatalogConstruction:
    """Tests for DiscrepancyCatalog constructor and initial empty state."""

    def test_empty_catalog_on_init(self) -> None:
        """A newly constructed catalog should have zero entries."""
        catalog = DiscrepancyCatalog()
        assert len(catalog.get_all_type_codes()) == 0

    def test_get_all_type_codes_empty(self) -> None:
        """get_all_type_codes() returns an empty list on a fresh catalog."""
        catalog = DiscrepancyCatalog()
        codes = catalog.get_all_type_codes()
        assert isinstance(codes, list)
        assert len(codes) == 0

    def test_get_metrics_empty(self) -> None:
        """get_metrics() returns zero counts on a fresh empty catalog."""
        catalog = DiscrepancyCatalog()
        metrics = catalog.get_metrics()
        assert metrics["total_entries"] == 0
        assert metrics["category_counts"] == {}
        assert metrics["difficulty_counts"] == {}


# =========================================================================
# Phase 3: Registration Tests
# =========================================================================


@pytest.mark.discrepancy
class TestRegistration:
    """Tests for register(), duplicate rejection, class_ref handling."""

    def test_register_single_entry(self) -> None:
        """Register one CatalogEntry and retrieve it via get_entry()."""
        catalog = DiscrepancyCatalog()
        entry = _make_entry(type_code="REG-001")
        catalog.register(entry)
        result = catalog.get_entry("REG-001")
        assert result is not None
        assert result.type_code == "REG-001"

    def test_register_with_class_ref(self) -> None:
        """Register an entry with class_ref; get_implementation_class should return it."""
        catalog = DiscrepancyCatalog()
        entry = _make_entry(type_code="REG-002")
        mock_cls = MagicMock()
        catalog.register(entry, class_ref=mock_cls)
        # The class reference should be cached and returned
        result_cls = catalog.get_implementation_class("REG-002")
        assert result_cls is mock_cls

    def test_register_duplicate_type_code_raises(self) -> None:
        """Registering the same type_code twice must raise ValueError."""
        catalog = DiscrepancyCatalog()
        entry1 = _make_entry(type_code="DUP-001")
        entry2 = _make_entry(type_code="DUP-001", name="Duplicate Attempt")
        catalog.register(entry1)
        with pytest.raises(ValueError, match="already registered"):
            catalog.register(entry2)

    def test_register_preserves_order(self) -> None:
        """get_all_type_codes() returns codes in sorted order (per implementation)."""
        catalog = DiscrepancyCatalog()
        for code in ["Z-001", "A-001", "M-001"]:
            catalog.register(_make_entry(type_code=code))
        codes = catalog.get_all_type_codes()
        assert codes == sorted(["Z-001", "A-001", "M-001"])


# =========================================================================
# Phase 4: register_defaults() Tests — THE CRITICAL TEST
# =========================================================================


@pytest.mark.discrepancy
class TestRegisterDefaults:
    """Tests for register_defaults() — validates all 35 types are correctly registered."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_total_registered_count(self) -> None:
        """After register_defaults(), at least 35 types must be registered."""
        count = len(self.catalog.get_all_type_codes())
        assert count >= 35, f"Expected >= 35 types, got {count}"

    def test_all_expected_type_codes_present(self) -> None:
        """Every code in ALL_TYPE_CODES must be present in the catalog."""
        registered = set(self.catalog.get_all_type_codes())
        for code in ALL_TYPE_CODES:
            assert code in registered, f"Expected type code {code} not found"

    def test_p2p_count(self) -> None:
        """get_by_category('p2p') must return exactly 15 entries."""
        entries = self.catalog.get_by_category("p2p")
        assert len(entries) == 15

    def test_o2c_count(self) -> None:
        """get_by_category('o2c') must return exactly 10 entries."""
        entries = self.catalog.get_by_category("o2c")
        assert len(entries) == 10

    def test_gl_count(self) -> None:
        """get_by_category('gl') must return exactly 5 entries."""
        entries = self.catalog.get_by_category("gl")
        assert len(entries) == 5

    def test_control_count(self) -> None:
        """get_by_category('control') must return exactly 5 entries."""
        entries = self.catalog.get_by_category("control")
        assert len(entries) == 5

    def test_no_hard_difficulty(self) -> None:
        """get_by_difficulty('hard') must return 0 entries (hard=0 in MVP)."""
        entries = self.catalog.get_by_difficulty("hard")
        assert len(entries) == 0, f"Expected 0 hard entries, got {len(entries)}"

    def test_easy_and_medium_only(self) -> None:
        """All entries must have difficulty in ('easy', 'medium')."""
        for code in self.catalog.get_all_type_codes():
            entry = self.catalog.get_entry(code)
            assert entry is not None
            assert entry.difficulty in ("easy", "medium"), (
                f"{code} has unexpected difficulty '{entry.difficulty}'"
            )

    def test_each_entry_has_base_rate(self) -> None:
        """Every entry must have a base_rate that is a Decimal > 0."""
        for code in self.catalog.get_all_type_codes():
            entry = self.catalog.get_entry(code)
            assert entry is not None
            assert isinstance(entry.base_rate, Decimal), (
                f"{code} base_rate is {type(entry.base_rate)}, expected Decimal"
            )
            assert entry.base_rate > Decimal("0"), (
                f"{code} base_rate must be positive, got {entry.base_rate}"
            )

    def test_each_entry_has_detection_method(self) -> None:
        """Every entry must have a non-empty detection_method."""
        for code in self.catalog.get_all_type_codes():
            entry = self.catalog.get_entry(code)
            assert entry is not None
            assert entry.detection_method, f"{code} has empty detection_method"

    def test_each_entry_has_name(self) -> None:
        """Every entry must have a non-empty name."""
        for code in self.catalog.get_all_type_codes():
            entry = self.catalog.get_entry(code)
            assert entry is not None
            assert entry.name, f"{code} has empty name"

    def test_each_entry_has_description(self) -> None:
        """Every entry must have a non-empty description."""
        for code in self.catalog.get_all_type_codes():
            entry = self.catalog.get_entry(code)
            assert entry is not None
            assert entry.description, f"{code} has empty description"

    @pytest.mark.parametrize("type_code", ALL_TYPE_CODES, ids=ALL_TYPE_CODES)
    def test_individual_type_exists(self, type_code: str) -> None:
        """get_entry(type_code) must return non-None for each expected type."""
        entry = self.catalog.get_entry(type_code)
        assert entry is not None, f"Expected entry for {type_code} but got None"

    @pytest.mark.parametrize("type_code", ALL_TYPE_CODES, ids=ALL_TYPE_CODES)
    def test_individual_type_category(self, type_code: str) -> None:
        """Verify type code prefix matches category (P2P-* → 'p2p', etc.)."""
        entry = self.catalog.get_entry(type_code)
        assert entry is not None
        prefix = type_code.split("-")[0]
        expected_category = _PREFIX_TO_CATEGORY.get(prefix)
        assert expected_category is not None, f"Unknown prefix {prefix}"
        assert entry.category == expected_category, (
            f"{type_code} has category '{entry.category}', "
            f"expected '{expected_category}'"
        )


# =========================================================================
# Phase 5: Lookup Tests
# =========================================================================


@pytest.mark.discrepancy
class TestLookupByTypeCode:
    """Tests for get_entry(), get_parameter_bounds(), get_implementation_class()."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_get_entry_returns_catalog_entry(self) -> None:
        """get_entry('P2P-001') returns a CatalogEntry instance."""
        entry = self.catalog.get_entry("P2P-001")
        assert isinstance(entry, CatalogEntry)
        assert entry.type_code == "P2P-001"

    def test_get_entry_nonexistent_returns_none(self) -> None:
        """get_entry('NONEXISTENT') returns None."""
        result = self.catalog.get_entry("NONEXISTENT")
        assert result is None

    def test_get_parameter_bounds_with_params(self) -> None:
        """For P2P-001: get_parameter_bounds() returns a dict with expected keys."""
        bounds = self.catalog.get_parameter_bounds("P2P-001")
        assert isinstance(bounds, dict)
        assert "days_apart" in bounds
        assert "amount_variance_pct" in bounds

    def test_get_parameter_bounds_without_params(self) -> None:
        """For P2P-004 (no params): returns empty dict."""
        bounds = self.catalog.get_parameter_bounds("P2P-004")
        assert isinstance(bounds, dict)
        assert len(bounds) == 0

    def test_get_implementation_class_returns_class(self) -> None:
        """For P2P-001: get_implementation_class() returns DuplicateInvoice class."""
        cls = self.catalog.get_implementation_class("P2P-001")
        assert cls is not None
        assert cls.__name__ == "DuplicateInvoice"

    def test_get_implementation_class_nonexistent_returns_none(self) -> None:
        """get_implementation_class for a nonexistent type returns None."""
        result = self.catalog.get_implementation_class("NONEXISTENT")
        assert result is None


@pytest.mark.discrepancy
class TestLookupByCategory:
    """Tests for get_by_category() — filtering entries by category string."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_get_by_p2p(self) -> None:
        """get_by_category('p2p') returns a list of exactly 15 entries."""
        entries = self.catalog.get_by_category("p2p")
        assert len(entries) == 15
        assert all(e.category == "p2p" for e in entries)

    def test_get_by_o2c(self) -> None:
        """get_by_category('o2c') returns a list of exactly 10 entries."""
        entries = self.catalog.get_by_category("o2c")
        assert len(entries) == 10
        assert all(e.category == "o2c" for e in entries)

    def test_get_by_gl(self) -> None:
        """get_by_category('gl') returns a list of exactly 5 entries."""
        entries = self.catalog.get_by_category("gl")
        assert len(entries) == 5
        assert all(e.category == "gl" for e in entries)

    def test_get_by_control(self) -> None:
        """get_by_category('control') returns a list of exactly 5 entries."""
        entries = self.catalog.get_by_category("control")
        assert len(entries) == 5
        assert all(e.category == "control" for e in entries)

    def test_get_by_invalid_category(self) -> None:
        """get_by_category('invalid') returns an empty list."""
        entries = self.catalog.get_by_category("invalid")
        assert isinstance(entries, list)
        assert len(entries) == 0


@pytest.mark.discrepancy
class TestLookupByDifficulty:
    """Tests for get_by_difficulty() — filtering entries by difficulty level."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_get_by_easy(self) -> None:
        """get_by_difficulty('easy') returns entries with difficulty='easy'."""
        entries = self.catalog.get_by_difficulty("easy")
        assert len(entries) > 0
        assert all(e.difficulty == "easy" for e in entries)

    def test_get_by_medium(self) -> None:
        """get_by_difficulty('medium') returns entries with difficulty='medium'."""
        entries = self.catalog.get_by_difficulty("medium")
        assert len(entries) > 0
        assert all(e.difficulty == "medium" for e in entries)

    def test_get_by_hard_empty(self) -> None:
        """get_by_difficulty('hard') returns empty list (MVP)."""
        entries = self.catalog.get_by_difficulty("hard")
        assert len(entries) == 0

    def test_easy_plus_medium_equals_total(self) -> None:
        """Count of easy + medium entries equals total registered count."""
        easy = len(self.catalog.get_by_difficulty("easy"))
        medium = len(self.catalog.get_by_difficulty("medium"))
        total = len(self.catalog.get_all_type_codes())
        assert easy + medium == total, (
            f"easy ({easy}) + medium ({medium}) != total ({total})"
        )


@pytest.mark.discrepancy
class TestLookupByCategoryAndDifficulty:
    """Tests for get_by_category_and_difficulty() — combined filtering."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_p2p_easy(self) -> None:
        """get_by_category_and_difficulty('p2p', 'easy') returns the correct subset."""
        entries = self.catalog.get_by_category_and_difficulty("p2p", "easy")
        assert len(entries) > 0
        for e in entries:
            assert e.category == "p2p"
            assert e.difficulty == "easy"

    def test_p2p_medium(self) -> None:
        """get_by_category_and_difficulty('p2p', 'medium') returns the correct subset."""
        entries = self.catalog.get_by_category_and_difficulty("p2p", "medium")
        assert len(entries) > 0
        for e in entries:
            assert e.category == "p2p"
            assert e.difficulty == "medium"

    def test_p2p_easy_and_medium_equals_p2p_total(self) -> None:
        """P2P easy + P2P medium counts must equal total P2P count."""
        easy = self.catalog.get_by_category_and_difficulty("p2p", "easy")
        medium = self.catalog.get_by_category_and_difficulty("p2p", "medium")
        total_p2p = self.catalog.get_by_category("p2p")
        assert len(easy) + len(medium) == len(total_p2p)

    def test_gl_easy(self) -> None:
        """GL easy types: GL-001 and GL-002 should be easy."""
        entries = self.catalog.get_by_category_and_difficulty("gl", "easy")
        type_codes = {e.type_code for e in entries}
        assert "GL-001" in type_codes
        assert "GL-002" in type_codes

    def test_gl_medium(self) -> None:
        """GL medium types: GL-003, GL-004, GL-005 should be medium."""
        entries = self.catalog.get_by_category_and_difficulty("gl", "medium")
        type_codes = {e.type_code for e in entries}
        assert "GL-003" in type_codes
        assert "GL-004" in type_codes
        assert "GL-005" in type_codes

    def test_combined_filter_invalid_returns_empty(self) -> None:
        """Combined filter with invalid category or difficulty returns empty list."""
        result = self.catalog.get_by_category_and_difficulty("invalid", "easy")
        assert len(result) == 0
        result2 = self.catalog.get_by_category_and_difficulty("p2p", "hard")
        assert len(result2) == 0


# =========================================================================
# Phase 6: Type Code Uniqueness Tests
# =========================================================================


@pytest.mark.discrepancy
class TestTypeCodeUniqueness:
    """Tests ensuring all type codes are globally unique and non-overlapping."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_all_type_codes_unique(self) -> None:
        """After register_defaults(), all type codes must be unique (no duplicates)."""
        codes = self.catalog.get_all_type_codes()
        assert len(codes) == len(set(codes)), "Duplicate type codes detected"

    def test_no_overlapping_codes_across_categories(self) -> None:
        """P2P, O2C, GL, and CTL code sets must not overlap."""
        p2p_codes = {e.type_code for e in self.catalog.get_by_category("p2p")}
        o2c_codes = {e.type_code for e in self.catalog.get_by_category("o2c")}
        gl_codes = {e.type_code for e in self.catalog.get_by_category("gl")}
        ctl_codes = {e.type_code for e in self.catalog.get_by_category("control")}

        all_sets = [p2p_codes, o2c_codes, gl_codes, ctl_codes]
        for i, set_a in enumerate(all_sets):
            for j, set_b in enumerate(all_sets):
                if i < j:
                    overlap = set_a & set_b
                    assert len(overlap) == 0, f"Overlapping codes: {overlap}"


# =========================================================================
# Phase 7: Parameter Bounds Tests
# =========================================================================


@pytest.mark.discrepancy
class TestParameterBounds:
    """Tests for parameter bounds validation per type code."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    @pytest.mark.parametrize(
        "type_code, expected_params",
        list(TYPES_WITH_PARAMS.items()),
        ids=list(TYPES_WITH_PARAMS.keys()),
    )
    def test_parameter_bounds_present(
        self, type_code: str, expected_params: Dict[str, Dict[str, Any]]
    ) -> None:
        """get_parameter_bounds(type_code) is a non-empty dict for types with params."""
        bounds = self.catalog.get_parameter_bounds(type_code)
        assert isinstance(bounds, dict)
        assert len(bounds) > 0, f"{type_code} should have parameter bounds"

    @pytest.mark.parametrize(
        "type_code, expected_params",
        list(TYPES_WITH_PARAMS.items()),
        ids=list(TYPES_WITH_PARAMS.keys()),
    )
    def test_parameter_bounds_keys(
        self, type_code: str, expected_params: Dict[str, Dict[str, Any]]
    ) -> None:
        """Bounds dict keys must include all expected param names."""
        bounds = self.catalog.get_parameter_bounds(type_code)
        for param_name in expected_params:
            assert param_name in bounds, (
                f"{type_code}: missing expected param '{param_name}' in bounds. "
                f"Got: {list(bounds.keys())}"
            )

    @pytest.mark.parametrize(
        "type_code, expected_params",
        list(TYPES_WITH_PARAMS.items()),
        ids=list(TYPES_WITH_PARAMS.keys()),
    )
    def test_parameter_bounds_min_max(
        self, type_code: str, expected_params: Dict[str, Dict[str, Any]]
    ) -> None:
        """Each param must have 'min' and 'max' with min <= max."""
        bounds = self.catalog.get_parameter_bounds(type_code)
        for param_name, expected_spec in expected_params.items():
            actual_spec = bounds[param_name]
            assert "min" in actual_spec, f"{type_code}.{param_name}: missing 'min'"
            assert "max" in actual_spec, f"{type_code}.{param_name}: missing 'max'"
            actual_min = Decimal(str(actual_spec["min"]))
            actual_max = Decimal(str(actual_spec["max"]))
            assert actual_min <= actual_max, (
                f"{type_code}.{param_name}: min ({actual_min}) > max ({actual_max})"
            )

    def test_gl001_imbalance_min_is_002(self) -> None:
        """GL-001 imbalance_amount min must be Decimal('0.02') — exceeds $0.01 tolerance."""
        bounds = self.catalog.get_parameter_bounds("GL-001")
        assert "imbalance_amount" in bounds
        actual_min = Decimal(str(bounds["imbalance_amount"]["min"]))
        assert actual_min == Decimal("0.02"), (
            f"GL-001 imbalance_amount min should be 0.02, got {actual_min}"
        )

    @pytest.mark.parametrize(
        "type_code", TYPES_WITHOUT_PARAMS, ids=TYPES_WITHOUT_PARAMS
    )
    def test_no_parameter_bounds(self, type_code: str) -> None:
        """get_parameter_bounds(type_code) returns empty dict for types without params."""
        bounds = self.catalog.get_parameter_bounds(type_code)
        assert isinstance(bounds, dict)
        assert len(bounds) == 0, (
            f"{type_code} should have no parameter bounds, got: {list(bounds.keys())}"
        )


# =========================================================================
# Phase 8: Implementation Class Lookup Tests
# =========================================================================


@pytest.mark.discrepancy
class TestImplementationClassLookup:
    """Tests for lazy import resolution of implementation classes."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    @pytest.mark.parametrize("type_code", ALL_TYPE_CODES, ids=ALL_TYPE_CODES)
    def test_implementation_class_resolves(self, type_code: str) -> None:
        """get_implementation_class(type_code) returns a class (not None)."""
        cls = self.catalog.get_implementation_class(type_code)
        assert cls is not None, (
            f"get_implementation_class('{type_code}') returned None "
            f"— the implementation module may be missing"
        )

    @pytest.mark.parametrize("type_code", ALL_TYPE_CODES, ids=ALL_TYPE_CODES)
    def test_implementation_class_is_base_subclass(self, type_code: str) -> None:
        """Resolved class must be a subclass of BaseDiscrepancy."""
        cls = self.catalog.get_implementation_class(type_code)
        assert cls is not None
        assert issubclass(cls, BaseDiscrepancy), (
            f"{type_code}: {cls.__name__} is not a subclass of BaseDiscrepancy"
        )

    @pytest.mark.parametrize("type_code", ALL_TYPE_CODES, ids=ALL_TYPE_CODES)
    def test_implementation_class_type_code_matches(self, type_code: str) -> None:
        """Resolved class's type_code ClassVar must match the lookup key."""
        cls = self.catalog.get_implementation_class(type_code)
        assert cls is not None
        assert cls.type_code == type_code, (
            f"Class {cls.__name__}.type_code = '{cls.type_code}', "
            f"expected '{type_code}'"
        )


# =========================================================================
# Phase 9: YAML Loading Tests
# =========================================================================


@pytest.mark.discrepancy
class TestYAMLLoading:
    """Tests for load_from_yaml() — reading config/discrepancies/*.yaml files."""

    def test_load_from_yaml_reads_files(self, tmp_path: Path) -> None:
        """load_from_yaml() reads YAML files from the specified directory."""
        # Create a minimal YAML config file
        yaml_content = """
discrepancies:
  - type_code: "YAML-001"
    name: "YAML Test Entry"
    category: "p2p"
    difficulty: "easy"
    base_rate: 0.02
    description: "A test entry loaded from YAML."
    detection_method: "yaml_test"
    parameter_bounds: {}
"""
        yaml_file = tmp_path / "p2p_discrepancies.yaml"
        yaml_file.write_text(yaml_content)

        catalog = DiscrepancyCatalog()
        catalog.load_from_yaml(str(tmp_path))

        entry = catalog.get_entry("YAML-001")
        assert entry is not None
        assert entry.name == "YAML Test Entry"
        assert entry.category == "p2p"

    def test_load_from_yaml_adds_entries(self, tmp_path: Path) -> None:
        """Entries from YAML are correctly added to the catalog."""
        yaml_content = """
discrepancies:
  - type_code: "YAML-002"
    name: "Second YAML"
    category: "o2c"
    difficulty: "medium"
    base_rate: 0.03
    description: "Another YAML test."
    detection_method: "yaml_check"
  - type_code: "YAML-003"
    name: "Third YAML"
    category: "gl"
    difficulty: "easy"
    base_rate: 0.01
    description: "GL YAML test."
    detection_method: "balance_check"
"""
        yaml_file = tmp_path / "o2c_discrepancies.yaml"
        yaml_file.write_text(yaml_content)

        catalog = DiscrepancyCatalog()
        catalog.load_from_yaml(str(tmp_path))

        assert catalog.get_entry("YAML-002") is not None
        assert catalog.get_entry("YAML-003") is not None
        assert len(catalog.get_all_type_codes()) == 2

    def test_load_from_yaml_with_missing_dir(self) -> None:
        """load_from_yaml with a non-existent dir should not raise (no files found)."""
        catalog = DiscrepancyCatalog()
        # Should handle gracefully — the files simply don't exist
        catalog.load_from_yaml("/nonexistent/path/that/does/not/exist")
        assert len(catalog.get_all_type_codes()) == 0

    def test_load_from_yaml_skips_duplicate_type_codes(self, tmp_path: Path) -> None:
        """YAML loading should skip type codes that are already registered."""
        catalog = DiscrepancyCatalog()
        # Pre-register an entry
        catalog.register(_make_entry(type_code="YAML-DUP"))

        yaml_content = """
discrepancies:
  - type_code: "YAML-DUP"
    name: "Duplicate from YAML"
    category: "p2p"
    difficulty: "easy"
    base_rate: 0.02
    description: "Should be skipped."
    detection_method: "dup_check"
"""
        yaml_file = tmp_path / "p2p_discrepancies.yaml"
        yaml_file.write_text(yaml_content)

        catalog.load_from_yaml(str(tmp_path))
        # Entry should still be the original, not the YAML one
        entry = catalog.get_entry("YAML-DUP")
        assert entry is not None
        assert entry.name == "Test Discrepancy"  # from _make_entry default

    def test_load_from_yaml_handles_rates_yaml(self, tmp_path: Path) -> None:
        """discrepancy_rates.yaml is treated as config, not entries."""
        rates_content = """
injection_rate: 0.02
difficulty_distribution:
  easy: 0.70
  medium: 0.30
  hard: 0.00
"""
        rates_file = tmp_path / "discrepancy_rates.yaml"
        rates_file.write_text(rates_content)

        catalog = DiscrepancyCatalog()
        catalog.load_from_yaml(str(tmp_path))
        # Should not add any entries from rates file
        assert len(catalog.get_all_type_codes()) == 0


# =========================================================================
# Phase 10: Metrics Tests
# =========================================================================


@pytest.mark.discrepancy
class TestMetrics:
    """Tests for get_metrics() — total, by-category, and by-difficulty counts."""

    @pytest.fixture(autouse=True)
    def _setup_catalog(self) -> None:
        """Create a catalog with all defaults before each test."""
        self.catalog = _make_catalog_with_defaults()

    def test_metrics_total(self) -> None:
        """After register_defaults(), total_entries >= 35."""
        metrics = self.catalog.get_metrics()
        assert metrics["total_entries"] >= 35

    def test_metrics_by_category(self) -> None:
        """Category counts must match EXPECTED_CATEGORY_COUNTS."""
        metrics = self.catalog.get_metrics()
        category_counts = metrics["category_counts"]
        for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
            actual = category_counts.get(category, 0)
            assert actual == expected_count, (
                f"Category '{category}': expected {expected_count}, got {actual}"
            )

    def test_metrics_by_difficulty_sums_to_total(self) -> None:
        """Easy + Medium = total entries."""
        metrics = self.catalog.get_metrics()
        difficulty_counts = metrics["difficulty_counts"]
        easy = difficulty_counts.get("easy", 0)
        medium = difficulty_counts.get("medium", 0)
        total = metrics["total_entries"]
        assert easy + medium == total

    def test_metrics_contains_type_codes(self) -> None:
        """Metrics must include a 'type_codes' list with all registered codes."""
        metrics = self.catalog.get_metrics()
        assert "type_codes" in metrics
        assert len(metrics["type_codes"]) >= 35

    def test_metrics_contains_injection_rate(self) -> None:
        """Metrics must include 'default_injection_rate' from DISCREPANCY_DEFAULTS."""
        metrics = self.catalog.get_metrics()
        assert "default_injection_rate" in metrics
        # The value is stringified in get_metrics()
        assert metrics["default_injection_rate"] == "0.02"

    def test_metrics_contains_difficulty_distribution(self) -> None:
        """Metrics must include 'difficulty_distribution' with easy/medium/hard."""
        metrics = self.catalog.get_metrics()
        assert "difficulty_distribution" in metrics
        dist = metrics["difficulty_distribution"]
        assert "easy" in dist
        assert "medium" in dist
        assert "hard" in dist


# =========================================================================
# Phase 11: Edge Cases
# =========================================================================


@pytest.mark.discrepancy
class TestEdgeCases:
    """Tests for edge cases — empty catalog, invalid lookups, boundary conditions."""

    def test_empty_catalog_get_entry_returns_none(self) -> None:
        """Lookups on an empty catalog return None."""
        catalog = DiscrepancyCatalog()
        assert catalog.get_entry("P2P-001") is None

    def test_empty_catalog_get_by_category_returns_empty(self) -> None:
        """Category lookup on empty catalog returns empty list."""
        catalog = DiscrepancyCatalog()
        assert catalog.get_by_category("p2p") == []

    def test_empty_catalog_get_by_difficulty_returns_empty(self) -> None:
        """Difficulty lookup on empty catalog returns empty list."""
        catalog = DiscrepancyCatalog()
        assert catalog.get_by_difficulty("easy") == []

    def test_empty_catalog_get_parameter_bounds_returns_empty(self) -> None:
        """get_parameter_bounds on empty catalog returns empty dict."""
        catalog = DiscrepancyCatalog()
        assert catalog.get_parameter_bounds("P2P-001") == {}

    def test_empty_catalog_get_implementation_class_returns_none(self) -> None:
        """get_implementation_class on empty catalog returns None."""
        catalog = DiscrepancyCatalog()
        assert catalog.get_implementation_class("P2P-001") is None

    def test_get_entry_empty_string(self) -> None:
        """get_entry('') should return None gracefully."""
        catalog = _make_catalog_with_defaults()
        assert catalog.get_entry("") is None

    def test_category_case_sensitivity(self) -> None:
        """Categories are case-sensitive: 'P2P' should not match 'p2p'."""
        catalog = _make_catalog_with_defaults()
        uppercase = catalog.get_by_category("P2P")
        lowercase = catalog.get_by_category("p2p")
        assert len(uppercase) == 0, "Uppercase 'P2P' should not match entries"
        assert len(lowercase) == 15, "Lowercase 'p2p' should match 15 entries"

    def test_register_defaults_is_idempotent_raises(self) -> None:
        """Calling register_defaults() twice should raise ValueError (duplicate codes)."""
        catalog = DiscrepancyCatalog()
        catalog.register_defaults()
        with pytest.raises(ValueError, match="already registered"):
            catalog.register_defaults()

    def test_get_parameter_bounds_for_nonexistent_returns_empty(self) -> None:
        """get_parameter_bounds for a non-existent type code returns {}."""
        catalog = _make_catalog_with_defaults()
        result = catalog.get_parameter_bounds("FAKE-999")
        assert result == {}

    def test_get_by_category_and_difficulty_both_invalid(self) -> None:
        """Combined filter with both invalid values returns empty list."""
        catalog = _make_catalog_with_defaults()
        result = catalog.get_by_category_and_difficulty("nope", "nope")
        assert result == []
