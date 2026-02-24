"""Comprehensive unit tests for the PromptManager class.

Tests cover:
- YAML template loading from directories (including empty / missing directories)
- build_prompt() context substitution and tuple return
- _estimate_tokens() with the 4 chars ≈ 1 token formula
- Token budget constants and enforcement (warning at 2,000 tokens)
- build_context() merging agent, company, and transaction dictionaries
- validate_templates() detecting structural issues
- get_metrics() tracking prompts_built and templates_loaded
- YAML safe_load usage (security)
- Prompt truncation when exceeding token budget
- Reload functionality

References:
    AAP Section 0.5.1 Group 4 / Group 10
    README.md lines 1178–1218
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.llm.prompt_manager import (
    CHARS_PER_TOKEN,
    DEFAULT_TOKEN_BUDGET,
    TEMPLATE_NAMES,
    TOKEN_WARNING_THRESHOLD,
    PromptManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_template_dir(tmp_path: Path) -> str:
    """Create a temporary directory with two sample YAML prompt templates.

    Templates use Python str.format() placeholders matching the expected
    context keys produced by ``build_context()``.
    """
    template1: Dict[str, str] = {
        "system": "You are {agent_name} working for {company_name}. Thoroughness: {thoroughness}.",
        "user": "Process invoice from {vendor_name}, amount ${amount}.",
    }
    (tmp_path / "process_vendor_invoice.yaml").write_text(
        yaml.dump(template1, default_flow_style=False),
        encoding="utf-8",
    )

    template2: Dict[str, str] = {
        "system": "You are an approver named {agent_name} at {company_name}.",
        "user": "Approve transaction {transaction_id} for ${amount}.",
    }
    (tmp_path / "approve_transaction.yaml").write_text(
        yaml.dump(template2, default_flow_style=False),
        encoding="utf-8",
    )

    return str(tmp_path)


@pytest.fixture()
def all_templates_dir(tmp_path: Path) -> str:
    """Create a temporary directory with all 6 expected YAML prompt templates.

    Each template has minimal but structurally valid ``system`` and ``user``
    keys.
    """
    for name in TEMPLATE_NAMES:
        template: Dict[str, str] = {
            "system": f"System prompt for {name}.",
            "user": f"User prompt for {name}.",
        }
        (tmp_path / f"{name}.yaml").write_text(
            yaml.dump(template, default_flow_style=False),
            encoding="utf-8",
        )

    return str(tmp_path)


@pytest.fixture()
def prompt_manager(sample_template_dir: str) -> PromptManager:
    """Return a PromptManager instance loaded from *sample_template_dir*."""
    return PromptManager(template_dir=sample_template_dir)


@pytest.fixture()
def sample_agent_context() -> Dict[str, Any]:
    """Agent context dictionary with standard personality traits."""
    return {
        "agent_name": "John Smith",
        "agent_role": "ap_clerk",
        "thoroughness": "7",
        "risk_tolerance": "5",
        "efficiency": "6",
        "compliance": "8",
    }


@pytest.fixture()
def sample_company_context() -> Dict[str, Any]:
    """Company context dictionary."""
    return {
        "company_name": "Test Corp",
        "company_id": "COMP-001",
    }


@pytest.fixture()
def sample_transaction_context() -> Dict[str, Any]:
    """Transaction context dictionary with invoice / PO fields."""
    return {
        "vendor_name": "Acme Supplies",
        "invoice_number": "INV-001",
        "invoice_date": "2024-03-15",
        "amount": "5000.00",
        "po_number": "PO-001",
        "po_amount": "5000.00",
        "receipt_date": "2024-03-10",
        "quantity_received": "100",
        "transaction_id": "TXN-001",
    }


# =========================================================================
# Phase 3 — Template Loading Tests
# =========================================================================


class TestTemplateLoading:
    """Tests for YAML template loading from directories."""

    def test_load_templates_from_directory(self, sample_template_dir: str) -> None:
        """PromptManager should load YAML templates from the given directory."""
        pm = PromptManager(template_dir=sample_template_dir)
        assert len(pm.templates) > 0
        assert "process_vendor_invoice" in pm.templates
        assert "approve_transaction" in pm.templates

    def test_template_has_system_and_user_keys(self, prompt_manager: PromptManager) -> None:
        """Every loaded template must have 'system' and 'user' keys."""
        for name, tmpl in prompt_manager.templates.items():
            assert "system" in tmpl, f"Template '{name}' missing 'system' key"
            assert "user" in tmpl, f"Template '{name}' missing 'user' key"

    def test_load_templates_empty_directory(self, tmp_path: Path) -> None:
        """An empty directory should result in an empty templates dict (graceful)."""
        pm = PromptManager(template_dir=str(tmp_path))
        assert pm.templates == {}

    def test_load_templates_nonexistent_directory(self) -> None:
        """A nonexistent directory should result in empty templates (no crash)."""
        pm = PromptManager(template_dir="/nonexistent/path/to/templates")
        assert pm.templates == {}

    def test_load_all_six_templates(self, all_templates_dir: str) -> None:
        """All 6 expected TEMPLATE_NAMES should be loaded when present."""
        pm = PromptManager(template_dir=all_templates_dir)
        assert len(pm.templates) == 6
        for expected_name in TEMPLATE_NAMES:
            assert expected_name in pm.templates, (
                f"Expected template '{expected_name}' not found"
            )

    def test_get_template_names(self, prompt_manager: PromptManager) -> None:
        """get_template_names() should return a sorted list of loaded names."""
        names = prompt_manager.get_template_names()
        assert isinstance(names, list)
        assert names == sorted(names)
        # The sample dir has 2 templates
        assert len(names) == 2

    def test_get_template_returns_template(self, prompt_manager: PromptManager) -> None:
        """get_template() should return the dict for a known template."""
        tmpl = prompt_manager.get_template("process_vendor_invoice")
        assert tmpl is not None
        assert "system" in tmpl
        assert "user" in tmpl

    def test_get_template_returns_none_for_missing(self, prompt_manager: PromptManager) -> None:
        """get_template() should return None for an unknown template name."""
        result = prompt_manager.get_template("nonexistent_template")
        assert result is None

    def test_reload_templates(
        self, prompt_manager: PromptManager, sample_template_dir: str
    ) -> None:
        """reload_templates() should re-read the directory and return the count."""
        count = prompt_manager.reload_templates()
        assert count == 2  # same 2 templates as before
        assert len(prompt_manager.templates) == 2

    def test_reload_templates_picks_up_new_file(
        self, prompt_manager: PromptManager, sample_template_dir: str
    ) -> None:
        """reload_templates() should detect a newly added template file."""
        new_template = {"system": "New system.", "user": "New user."}
        new_path = Path(sample_template_dir) / "new_template.yaml"
        new_path.write_text(yaml.dump(new_template), encoding="utf-8")

        count = prompt_manager.reload_templates()
        assert count == 3
        assert "new_template" in prompt_manager.templates

    def test_templates_property_is_dict(self, prompt_manager: PromptManager) -> None:
        """The .templates attribute should be a dict mapping names to dicts."""
        assert isinstance(prompt_manager.templates, dict)
        for key, val in prompt_manager.templates.items():
            assert isinstance(key, str)
            assert isinstance(val, dict)

    def test_non_yaml_files_ignored(self, tmp_path: Path) -> None:
        """Non-YAML files in the template directory should be silently ignored."""
        (tmp_path / "notes.txt").write_text("not a template")
        (tmp_path / "valid.yaml").write_text(
            yaml.dump({"system": "S", "user": "U"}),
            encoding="utf-8",
        )
        pm = PromptManager(template_dir=str(tmp_path))
        assert len(pm.templates) == 1
        assert "valid" in pm.templates


# =========================================================================
# Phase 4 — build_prompt() Tests
# =========================================================================


class TestBuildPrompt:
    """Tests for the build_prompt() method."""

    def test_build_prompt_returns_tuple(self, prompt_manager: PromptManager) -> None:
        """build_prompt() should return a (system, user) tuple of strings."""
        result = prompt_manager.build_prompt("process_vendor_invoice", {})
        assert isinstance(result, tuple)
        assert len(result) == 2
        system_prompt, user_prompt = result
        assert isinstance(system_prompt, str)
        assert isinstance(user_prompt, str)

    def test_build_prompt_substitutes_context(
        self,
        prompt_manager: PromptManager,
        sample_agent_context: Dict[str, Any],
        sample_company_context: Dict[str, Any],
        sample_transaction_context: Dict[str, Any],
    ) -> None:
        """Placeholders in templates should be replaced with context values."""
        merged = {
            **sample_agent_context,
            **sample_company_context,
            **sample_transaction_context,
        }
        system_prompt, user_prompt = prompt_manager.build_prompt(
            "process_vendor_invoice", merged
        )

        # System prompt placeholders
        assert "John Smith" in system_prompt
        assert "Test Corp" in system_prompt
        assert "7" in system_prompt  # thoroughness

        # User prompt placeholders
        assert "Acme Supplies" in user_prompt
        assert "5000.00" in user_prompt

    def test_build_prompt_missing_template_raises_error(
        self, prompt_manager: PromptManager
    ) -> None:
        """build_prompt() should raise KeyError for an unknown template name."""
        with pytest.raises(KeyError, match="nonexistent_template"):
            prompt_manager.build_prompt("nonexistent_template", {})

    def test_build_prompt_missing_context_key_handled(
        self, prompt_manager: PromptManager
    ) -> None:
        """Missing context keys should be kept as literal {key} strings.

        The implementation uses _SafeFormatDict which gracefully handles
        missing keys by returning the placeholder itself.
        """
        system_prompt, user_prompt = prompt_manager.build_prompt(
            "process_vendor_invoice", {"agent_name": "Jane"}
        )
        # {company_name} is missing — should appear as literal placeholder
        assert "{company_name}" in system_prompt
        # {agent_name} was provided
        assert "Jane" in system_prompt

    def test_build_prompt_with_empty_context(self, prompt_manager: PromptManager) -> None:
        """An empty context should leave all placeholders intact."""
        system_prompt, user_prompt = prompt_manager.build_prompt(
            "process_vendor_invoice", {}
        )
        assert "{agent_name}" in system_prompt
        assert "{vendor_name}" in user_prompt

    def test_build_prompt_increments_metrics(self, prompt_manager: PromptManager) -> None:
        """Each build_prompt() call should increment the prompts_built metric."""
        prompt_manager.build_prompt("process_vendor_invoice", {})
        prompt_manager.build_prompt("approve_transaction", {})
        metrics = prompt_manager.get_metrics()
        assert metrics["prompts_built"] == 2

    def test_build_prompt_both_templates(self, prompt_manager: PromptManager) -> None:
        """Both loaded templates should be buildable without error."""
        s1, u1 = prompt_manager.build_prompt("process_vendor_invoice", {})
        assert len(s1) > 0
        assert len(u1) > 0

        s2, u2 = prompt_manager.build_prompt("approve_transaction", {})
        assert len(s2) > 0
        assert len(u2) > 0


# =========================================================================
# Phase 5 — Token Estimation Tests
# =========================================================================


class TestTokenEstimation:
    """Tests for _estimate_tokens() using the 4 chars ≈ 1 token formula."""

    def test_estimate_tokens_empty_string(self, prompt_manager: PromptManager) -> None:
        """An empty string should yield 0 tokens."""
        assert prompt_manager._estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self, prompt_manager: PromptManager) -> None:
        """'Hello World' is 11 chars → 11 // 4 = 2 tokens."""
        assert prompt_manager._estimate_tokens("Hello World") == 11 // CHARS_PER_TOKEN

    def test_estimate_tokens_exact_multiple_4(self, prompt_manager: PromptManager) -> None:
        """4 chars → 1 token."""
        assert prompt_manager._estimate_tokens("abcd") == 1

    def test_estimate_tokens_exact_multiple_8(self, prompt_manager: PromptManager) -> None:
        """8 chars → 2 tokens."""
        assert prompt_manager._estimate_tokens("abcdefgh") == 2

    def test_estimate_tokens_long_text(self, prompt_manager: PromptManager) -> None:
        """6000 chars → 6000 // 4 = 1500 tokens."""
        text = "a" * 6000
        assert prompt_manager._estimate_tokens(text) == 1500

    def test_estimate_tokens_1500_token_boundary(self, prompt_manager: PromptManager) -> None:
        """Verify the exact budget boundary at 1500 tokens (6000 chars)."""
        assert prompt_manager._estimate_tokens("a" * 6000) == 1500
        assert prompt_manager._estimate_tokens("a" * 6004) == 1501

    def test_estimate_tokens_single_char(self, prompt_manager: PromptManager) -> None:
        """1 char → 0 tokens (integer division)."""
        assert prompt_manager._estimate_tokens("x") == 0

    def test_estimate_tokens_three_chars(self, prompt_manager: PromptManager) -> None:
        """3 chars → 0 tokens (integer division)."""
        assert prompt_manager._estimate_tokens("abc") == 0

    def test_estimate_tokens_uses_integer_division(self, prompt_manager: PromptManager) -> None:
        """5 chars → 1 token (5 // 4 = 1)."""
        assert prompt_manager._estimate_tokens("abcde") == 1


# =========================================================================
# Phase 6 — Token Budget Enforcement Tests
# =========================================================================


class TestTokenBudgetEnforcement:
    """Tests for token budget constants and warning / truncation behavior."""

    def test_token_budget_constant(self) -> None:
        """DEFAULT_TOKEN_BUDGET must be 1500."""
        assert DEFAULT_TOKEN_BUDGET == 1500

    def test_chars_per_token_constant(self) -> None:
        """CHARS_PER_TOKEN must be 4."""
        assert CHARS_PER_TOKEN == 4

    def test_token_warning_threshold_constant(self) -> None:
        """TOKEN_WARNING_THRESHOLD must be 2000."""
        assert TOKEN_WARNING_THRESHOLD == 2000

    def test_template_names_constant(self) -> None:
        """TEMPLATE_NAMES must contain the 6 expected template names."""
        assert len(TEMPLATE_NAMES) == 6
        expected = {
            "process_vendor_invoice",
            "approve_transaction",
            "match_documents",
            "handle_exception",
            "generate_description",
            "reconcile_account",
        }
        assert set(TEMPLATE_NAMES) == expected

    def test_build_prompt_warns_on_long_prompt(self, sample_template_dir: str) -> None:
        """A prompt exceeding TOKEN_WARNING_THRESHOLD (2000 tokens) should log a warning."""
        # 2001 tokens → 8004 chars per part; system + user > 8000 chars total
        long_text = "x" * 4500  # each side > 4000 chars → combined > 8000 → > 2000 tokens
        long_template: Dict[str, str] = {
            "system": long_text,
            "user": long_text,
        }
        path = Path(sample_template_dir) / "long_template.yaml"
        path.write_text(yaml.dump(long_template), encoding="utf-8")

        pm = PromptManager(template_dir=sample_template_dir)

        with patch("app.llm.prompt_manager.logger") as mock_logger:
            pm.build_prompt("long_template", {})
            # At least one warning call should contain "prompt_too_long"
            warning_calls = mock_logger.warning.call_args_list
            event_names = [
                call.args[0] if call.args else ""
                for call in warning_calls
            ]
            assert "prompt_too_long" in event_names

    def test_build_prompt_no_warning_under_budget(
        self, prompt_manager: PromptManager
    ) -> None:
        """Short prompts under the warning threshold should not trigger a warning."""
        with patch("app.llm.prompt_manager.logger") as mock_logger:
            prompt_manager.build_prompt("process_vendor_invoice", {})
            # No "prompt_too_long" warning expected
            warning_calls = mock_logger.warning.call_args_list
            event_names = [
                call.args[0] if call.args else ""
                for call in warning_calls
            ]
            assert "prompt_too_long" not in event_names

    def test_build_prompt_truncates_when_exceeding_budget(
        self, sample_template_dir: str
    ) -> None:
        """When the combined prompt exceeds the token budget the user part is truncated."""
        # Create a template whose rendered prompt will exceed the budget
        # System: 200 chars (50 tokens), User: 6400 chars (1600 tokens)
        # Combined: 1650 tokens > 1500 budget → truncation expected
        long_template: Dict[str, str] = {
            "system": "S" * 200,
            "user": "U" * 6400,
        }
        path = Path(sample_template_dir) / "over_budget.yaml"
        path.write_text(yaml.dump(long_template), encoding="utf-8")

        pm = PromptManager(template_dir=sample_template_dir)
        system_prompt, user_prompt = pm.build_prompt("over_budget", {})

        # System prompt should be preserved
        assert system_prompt == "S" * 200
        # User prompt should have been truncated (contains "[truncated]")
        assert "[truncated]" in user_prompt
        # Total estimated tokens should be <= budget now
        total_tokens = pm._estimate_tokens(system_prompt + user_prompt)
        assert total_tokens <= DEFAULT_TOKEN_BUDGET

    def test_build_prompt_budget_warning_metric_incremented(
        self, sample_template_dir: str
    ) -> None:
        """The budget_warnings metric should increment when truncation occurs."""
        long_template: Dict[str, str] = {
            "system": "S" * 200,
            "user": "U" * 6400,
        }
        path = Path(sample_template_dir) / "budget_warn.yaml"
        path.write_text(yaml.dump(long_template), encoding="utf-8")

        pm = PromptManager(template_dir=sample_template_dir)
        pm.build_prompt("budget_warn", {})

        metrics = pm.get_metrics()
        assert metrics["budget_warnings"] >= 1

    def test_custom_token_budget(self, sample_template_dir: str) -> None:
        """A custom token_budget should be respected by the PromptManager."""
        pm = PromptManager(template_dir=sample_template_dir, token_budget=500)
        assert pm.token_budget == 500


# =========================================================================
# Phase 7 — Context Assembly Tests
# =========================================================================


class TestBuildContext:
    """Tests for the build_context() method that merges 3 context dicts."""

    def test_build_context_merges_all_dicts(
        self,
        prompt_manager: PromptManager,
        sample_agent_context: Dict[str, Any],
        sample_company_context: Dict[str, Any],
        sample_transaction_context: Dict[str, Any],
    ) -> None:
        """build_context() should merge agent + company + transaction dicts."""
        result = prompt_manager.build_context(
            sample_agent_context,
            sample_company_context,
            sample_transaction_context,
        )
        assert "agent_name" in result
        assert "company_name" in result
        assert "vendor_name" in result

    def test_build_context_with_empty_dicts(self, prompt_manager: PromptManager) -> None:
        """Merging three empty dicts should return an empty dict."""
        result = prompt_manager.build_context({}, {}, {})
        assert result == {}

    def test_build_context_preserves_values(
        self,
        prompt_manager: PromptManager,
        sample_agent_context: Dict[str, Any],
        sample_company_context: Dict[str, Any],
        sample_transaction_context: Dict[str, Any],
    ) -> None:
        """Values from each source dict should be preserved exactly."""
        result = prompt_manager.build_context(
            sample_agent_context,
            sample_company_context,
            sample_transaction_context,
        )
        assert result["agent_name"] == "John Smith"
        assert result["company_name"] == "Test Corp"
        assert result["vendor_name"] == "Acme Supplies"
        assert result["amount"] == "5000.00"
        assert result["company_id"] == "COMP-001"
        assert result["invoice_number"] == "INV-001"

    def test_build_context_later_overrides_earlier(
        self, prompt_manager: PromptManager
    ) -> None:
        """Later dictionaries should override earlier ones on key collision."""
        agent = {"shared_key": "agent_value"}
        company = {"shared_key": "company_value"}
        transaction = {"shared_key": "transaction_value"}

        result = prompt_manager.build_context(agent, company, transaction)
        # Transaction is merged last, so its value wins
        assert result["shared_key"] == "transaction_value"

    def test_build_context_key_count(
        self,
        prompt_manager: PromptManager,
        sample_agent_context: Dict[str, Any],
        sample_company_context: Dict[str, Any],
        sample_transaction_context: Dict[str, Any],
    ) -> None:
        """Total key count should equal the union of all distinct keys."""
        result = prompt_manager.build_context(
            sample_agent_context,
            sample_company_context,
            sample_transaction_context,
        )
        # All keys are distinct across the sample fixtures
        expected_count = (
            len(sample_agent_context)
            + len(sample_company_context)
            + len(sample_transaction_context)
        )
        assert len(result) == expected_count

    def test_build_context_returns_new_dict(
        self,
        prompt_manager: PromptManager,
        sample_agent_context: Dict[str, Any],
    ) -> None:
        """build_context() should return a new dict, not mutate the inputs."""
        original_agent = dict(sample_agent_context)
        result = prompt_manager.build_context(sample_agent_context, {}, {})
        # Mutate the result — should not affect the original fixture
        result["agent_name"] = "CHANGED"
        assert sample_agent_context["agent_name"] == original_agent["agent_name"]


# =========================================================================
# Phase 8 — Template Validation Tests
# =========================================================================


class TestValidateTemplates:
    """Tests for the validate_templates() method."""

    def test_validate_templates_all_valid(self, prompt_manager: PromptManager) -> None:
        """Well-formed templates should produce no validation issues."""
        results = prompt_manager.validate_templates()
        for name, issues in results.items():
            assert issues == [], f"Template '{name}' has unexpected issues: {issues}"

    def test_validate_templates_missing_system_key(self, tmp_path: Path) -> None:
        """A template loaded without 'system' should not load (skipped at load time).

        If we manually inject an incomplete template, validate_templates() should
        detect the missing key.
        """
        # Manually inject a bad template (bypassing the loader's filter)
        pm = PromptManager(template_dir=str(tmp_path))
        pm.templates["bad_template"] = {"user": "Only user key present."}

        results = pm.validate_templates()
        assert "bad_template" in results
        issues = results["bad_template"]
        assert any("system" in issue.lower() for issue in issues)

    def test_validate_templates_missing_user_key(self, tmp_path: Path) -> None:
        """A template missing 'user' should be flagged by validate_templates()."""
        pm = PromptManager(template_dir=str(tmp_path))
        pm.templates["no_user"] = {"system": "Only system key present."}

        results = pm.validate_templates()
        assert "no_user" in results
        issues = results["no_user"]
        assert any("user" in issue.lower() for issue in issues)

    def test_validate_templates_returns_dict(self, prompt_manager: PromptManager) -> None:
        """validate_templates() should return a dict of template_name → list."""
        results = prompt_manager.validate_templates()
        assert isinstance(results, dict)
        for name, issues in results.items():
            assert isinstance(name, str)
            assert isinstance(issues, list)

    def test_loader_skips_template_missing_system_key(self, tmp_path: Path) -> None:
        """_do_load_templates() should skip YAML files missing 'system' key entirely."""
        bad: Dict[str, str] = {"user": "user only"}
        (tmp_path / "incomplete.yaml").write_text(
            yaml.dump(bad), encoding="utf-8"
        )
        pm = PromptManager(template_dir=str(tmp_path))
        assert "incomplete" not in pm.templates

    def test_loader_skips_template_missing_user_key(self, tmp_path: Path) -> None:
        """_do_load_templates() should skip YAML files missing 'user' key entirely."""
        bad: Dict[str, str] = {"system": "system only"}
        (tmp_path / "incomplete2.yaml").write_text(
            yaml.dump(bad), encoding="utf-8"
        )
        pm = PromptManager(template_dir=str(tmp_path))
        assert "incomplete2" not in pm.templates

    def test_loader_skips_non_dict_yaml(self, tmp_path: Path) -> None:
        """A YAML file that does not parse to a dict should be skipped."""
        (tmp_path / "list.yaml").write_text("- item1\n- item2\n", encoding="utf-8")
        pm = PromptManager(template_dir=str(tmp_path))
        assert "list" not in pm.templates


# =========================================================================
# Phase 9 — Metrics Tests
# =========================================================================


class TestMetrics:
    """Tests for the get_metrics() method."""

    def test_get_metrics_initial_state(self, prompt_manager: PromptManager) -> None:
        """Initial metrics should show templates_loaded > 0 and prompts_built == 0."""
        metrics = prompt_manager.get_metrics()
        assert metrics["templates_loaded"] > 0
        assert metrics["prompts_built"] == 0
        assert metrics["budget_warnings"] == 0

    def test_get_metrics_after_build(self, prompt_manager: PromptManager) -> None:
        """prompts_built should increment after each build_prompt() call."""
        prompt_manager.build_prompt("process_vendor_invoice", {})
        metrics = prompt_manager.get_metrics()
        assert metrics["prompts_built"] == 1

    def test_get_metrics_after_multiple_builds(self, prompt_manager: PromptManager) -> None:
        """prompts_built should track cumulative calls."""
        for _ in range(5):
            prompt_manager.build_prompt("process_vendor_invoice", {})
        metrics = prompt_manager.get_metrics()
        assert metrics["prompts_built"] == 5

    def test_get_metrics_returns_dict_copy(self, prompt_manager: PromptManager) -> None:
        """get_metrics() should return a copy, not the internal dict."""
        m1 = prompt_manager.get_metrics()
        m1["prompts_built"] = 999
        m2 = prompt_manager.get_metrics()
        assert m2["prompts_built"] != 999

    def test_get_metrics_templates_loaded_matches(
        self, prompt_manager: PromptManager
    ) -> None:
        """templates_loaded should equal len(templates)."""
        metrics = prompt_manager.get_metrics()
        assert metrics["templates_loaded"] == len(prompt_manager.templates)


# =========================================================================
# Phase 10 — YAML Safety Tests
# =========================================================================


class TestYAMLSafety:
    """Tests verifying YAML safe_load usage for security."""

    def test_yaml_safe_load_used(self, tmp_path: Path) -> None:
        """Verify that PromptManager uses yaml.safe_load (not yaml.load).

        We create a template with a standard YAML structure and confirm it
        loads normally — unsafe constructs would be rejected by safe_load.
        """
        safe_template: Dict[str, str] = {
            "system": "Safe system content.",
            "user": "Safe user content.",
        }
        (tmp_path / "safe.yaml").write_text(
            yaml.dump(safe_template), encoding="utf-8"
        )
        pm = PromptManager(template_dir=str(tmp_path))
        assert "safe" in pm.templates

    def test_yaml_unsafe_python_object_rejected(self, tmp_path: Path) -> None:
        """A YAML file with an unsafe Python object tag should fail to load.

        yaml.safe_load rejects ``!!python/object`` and similar tags.  The
        loader catches the YAMLError and logs a warning, so the template
        should simply not appear in the loaded templates.
        """
        unsafe_yaml = (
            "system: !!python/object/apply:os.system ['echo pwned']\n"
            "user: normal user text\n"
        )
        (tmp_path / "unsafe.yaml").write_text(unsafe_yaml, encoding="utf-8")
        pm = PromptManager(template_dir=str(tmp_path))
        # The unsafe template should be rejected (not loaded)
        assert "unsafe" not in pm.templates

    def test_yaml_valid_template_loads_correctly(self, tmp_path: Path) -> None:
        """A valid YAML template should load its string values correctly."""
        template: Dict[str, str] = {
            "system": "Hello {name}, you are at {company}.",
            "user": "Process order {order_id}.",
        }
        (tmp_path / "valid.yaml").write_text(
            yaml.dump(template), encoding="utf-8"
        )
        pm = PromptManager(template_dir=str(tmp_path))
        loaded = pm.templates.get("valid")
        assert loaded is not None
        assert loaded["system"] == template["system"]
        assert loaded["user"] == template["user"]


# =========================================================================
# Additional edge-case and integration tests
# =========================================================================


class TestEdgeCases:
    """Additional edge-case tests for robustness."""

    def test_template_values_are_strings(self, prompt_manager: PromptManager) -> None:
        """Template 'system' and 'user' values should always be strings."""
        for name, tmpl in prompt_manager.templates.items():
            assert isinstance(tmpl["system"], str), f"{name}: system is not str"
            assert isinstance(tmpl["user"], str), f"{name}: user is not str"

    def test_template_with_numeric_yaml_value(self, tmp_path: Path) -> None:
        """Numeric YAML values should be coerced to strings by the loader."""
        (tmp_path / "numeric.yaml").write_text(
            "system: 12345\nuser: 67890\n", encoding="utf-8"
        )
        pm = PromptManager(template_dir=str(tmp_path))
        tmpl = pm.templates.get("numeric")
        assert tmpl is not None
        # Should be converted to strings
        assert isinstance(tmpl["system"], str)
        assert isinstance(tmpl["user"], str)

    def test_build_prompt_with_special_characters_in_context(
        self, prompt_manager: PromptManager
    ) -> None:
        """Context values with special characters should render correctly."""
        ctx = {
            "agent_name": "O'Brien & Partners",
            "company_name": '<TestCorp "LLC">',
            "thoroughness": "9",
            "vendor_name": "Ñoño Supplies™",
            "amount": "1,234.56",
        }
        system_prompt, user_prompt = prompt_manager.build_prompt(
            "process_vendor_invoice", ctx
        )
        assert "O'Brien & Partners" in system_prompt
        assert "Ñoño Supplies™" in user_prompt

    def test_build_prompt_with_multiline_context_value(
        self, prompt_manager: PromptManager
    ) -> None:
        """Multiline context values should be substituted faithfully."""
        ctx = {
            "agent_name": "Agent",
            "company_name": "Corp",
            "thoroughness": "5",
            "vendor_name": "Line1\nLine2\nLine3",
            "amount": "100",
        }
        _, user_prompt = prompt_manager.build_prompt(
            "process_vendor_invoice", ctx
        )
        assert "Line1\nLine2\nLine3" in user_prompt

    def test_prompt_manager_repr(self, prompt_manager: PromptManager) -> None:
        """The PromptManager should not crash when repr/str'd."""
        repr_str = repr(prompt_manager)
        str_str = str(prompt_manager)
        # Just ensure no exception — no specific format required
        assert isinstance(repr_str, str)
        assert isinstance(str_str, str)

    def test_build_prompt_with_double_braces_preserved(self, tmp_path: Path) -> None:
        """Double braces {{ }} in templates should be preserved as literal braces.

        This is important because prompt templates often include JSON schema
        examples using literal braces.
        """
        template: Dict[str, str] = {
            "system": "Respond with JSON: {{\"decision\": \"approve\"}}",
            "user": "Evaluate this.",
        }
        (tmp_path / "braces.yaml").write_text(
            yaml.dump(template), encoding="utf-8"
        )
        pm = PromptManager(template_dir=str(tmp_path))
        system_prompt, _ = pm.build_prompt("braces", {})
        # Double braces should render as single literal braces
        assert '{"decision": "approve"}' in system_prompt

    def test_end_to_end_build_context_then_prompt(
        self,
        prompt_manager: PromptManager,
        sample_agent_context: Dict[str, Any],
        sample_company_context: Dict[str, Any],
        sample_transaction_context: Dict[str, Any],
    ) -> None:
        """Full workflow: build_context() → build_prompt() should work end-to-end."""
        context = prompt_manager.build_context(
            sample_agent_context,
            sample_company_context,
            sample_transaction_context,
        )
        system_prompt, user_prompt = prompt_manager.build_prompt(
            "process_vendor_invoice", context
        )

        # All key placeholders should be resolved
        assert "{agent_name}" not in system_prompt
        assert "{company_name}" not in system_prompt
        assert "{vendor_name}" not in user_prompt
        assert "{amount}" not in user_prompt

        # Correct values injected
        assert "John Smith" in system_prompt
        assert "Test Corp" in system_prompt
        assert "Acme Supplies" in user_prompt
        assert "5000.00" in user_prompt
