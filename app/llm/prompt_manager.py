"""PromptManager for YAML template loading, context assembly, and token budget enforcement.

Loads 6 prompt templates from the prompts/ directory, assembles agent + company +
transaction context, and enforces the ~1,500 token budget per decision.

Template Format:
    Each YAML template file must contain 'system' and 'user' keys with Python
    str.format()-compatible placeholders (e.g., {agent_name}, {company_name}).
    Literal braces in JSON response schemas use double braces {{ }}.

Supported Templates:
    - process_vendor_invoice: Invoice processing with 3-way match
    - approve_transaction: Approval decisions with reasoning
    - match_documents: Document matching and reconciliation
    - handle_exception: Edge-case and exception handling
    - generate_description: Natural-language description generation
    - reconcile_account: Account reconciliation decisions
"""

from __future__ import annotations

import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
import yaml

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET: int = 1500
"""Default token budget per prompt (~1,500 tokens per decision)."""

TOKEN_WARNING_THRESHOLD: int = 2000
"""Token count that triggers a warning log (per README.md line 1207)."""

CHARS_PER_TOKEN: int = 4
"""Approximate character-to-token ratio (4 chars ≈ 1 token)."""

TEMPLATE_NAMES: List[str] = [
    "process_vendor_invoice",
    "approve_transaction",
    "match_documents",
    "handle_exception",
    "generate_description",
    "reconcile_account",
]
"""The 6 expected prompt template names matching YAML files in prompts/."""

# Truncation suffix appended when user prompts are trimmed to fit the budget.
_TRUNCATION_SUFFIX: str = "...[truncated]"


# ---------------------------------------------------------------------------
# Safe format helper
# ---------------------------------------------------------------------------

class _SafeFormatDict(defaultdict):
    """A defaultdict that returns the placeholder key itself for missing keys.

    This allows ``str.format_map`` to succeed even when the context is
    incomplete, replacing missing placeholders with ``{key}`` instead of
    raising a ``KeyError``.
    """

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """Manages LLM prompt templates and context assembly.

    Responsible for:
    * Loading YAML prompt templates from a configurable directory.
    * Assembling a flat context dictionary from agent, company, and
      transaction contexts.
    * Building (system_prompt, user_prompt) pairs by rendering templates
      with the assembled context.
    * Estimating token counts and enforcing a configurable budget.

    Parameters
    ----------
    template_dir : str
        Path to the directory containing YAML prompt templates.
        Defaults to ``"prompts"``.
    token_budget : int
        Maximum allowed token count for a single prompt.  Defaults to
        :data:`DEFAULT_TOKEN_BUDGET` (1,500).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        template_dir: str = "prompts",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self.template_dir: Path = Path(template_dir)
        self.token_budget: int = token_budget

        # Template storage: template_name → {"system": ..., "user": ...}
        self.templates: Dict[str, Dict[str, str]] = {}

        # Internal metrics counters
        self._metrics: Dict[str, int] = {
            "templates_loaded": 0,
            "prompts_built": 0,
            "budget_warnings": 0,
        }

        # Eagerly load templates from disk
        self._do_load_templates()

        logger.info(
            "prompt_manager_initialized",
            template_dir=str(self.template_dir),
            templates_loaded=self._metrics["templates_loaded"],
        )

    # ------------------------------------------------------------------
    # Public API — prompt building
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        template_name: str,
        context: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Build a prompt from a named template and a context dictionary.

        Parameters
        ----------
        template_name : str
            Key identifying the prompt template (e.g. ``"process_vendor_invoice"``).
        context : Dict[str, Any]
            Flat dictionary of placeholder values.  Missing keys are preserved
            as literal ``{key}`` strings rather than raising an error.

        Returns
        -------
        Tuple[str, str]
            ``(system_prompt, user_prompt)``

        Raises
        ------
        KeyError
            If *template_name* does not correspond to a loaded template.
        """

        if template_name not in self.templates:
            available = sorted(self.templates.keys())
            raise KeyError(
                f"Template '{template_name}' not found. "
                f"Available templates: {available}"
            )

        template = self.templates[template_name]

        # Build safe-format context that gracefully handles missing keys
        safe_ctx = _SafeFormatDict(str, context)

        system_prompt = template["system"].format_map(safe_ctx)
        user_prompt = template["user"].format_map(safe_ctx)

        estimated_tokens = self._estimate_tokens(system_prompt + user_prompt)

        # Warn at TOKEN_WARNING_THRESHOLD (2,000 tokens)
        if estimated_tokens > TOKEN_WARNING_THRESHOLD:
            logger.warning(
                "prompt_too_long",
                template=template_name,
                estimated_tokens=estimated_tokens,
            )

        # Enforce token budget — truncate if necessary
        if estimated_tokens > self.token_budget:
            self._metrics["budget_warnings"] += 1
            logger.warning(
                "prompt_exceeds_budget",
                template=template_name,
                estimated_tokens=estimated_tokens,
                budget=self.token_budget,
            )
            system_prompt, user_prompt = self._truncate_to_budget(
                system_prompt, user_prompt, self.token_budget
            )

        self._metrics["prompts_built"] += 1

        logger.debug(
            "prompt_built",
            template=template_name,
            estimated_tokens=estimated_tokens,
        )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Public API — context assembly
    # ------------------------------------------------------------------

    def build_context(
        self,
        agent_context: Dict[str, Any],
        company_context: Dict[str, Any],
        transaction_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble a flat context dict from agent, company, and transaction sources.

        The three dictionaries are merged in order (agent → company → transaction)
        so that later keys override earlier ones when there are conflicts.

        Expected keys per source (non-exhaustive):
        * **agent_context**: agent_name, agent_role, thoroughness,
          risk_tolerance, efficiency, compliance
        * **company_context**: company_name, company_id
        * **transaction_context**: vendor_name, invoice_number, amount, etc.

        Parameters
        ----------
        agent_context : Dict[str, Any]
        company_context : Dict[str, Any]
        transaction_context : Dict[str, Any]

        Returns
        -------
        Dict[str, Any]
            Merged context dictionary.
        """

        merged: Dict[str, Any] = {}
        merged.update(agent_context)
        merged.update(company_context)
        merged.update(transaction_context)

        logger.debug(
            "context_assembled",
            key_count=len(merged),
        )

        return merged

    # ------------------------------------------------------------------
    # Public API — template queries
    # ------------------------------------------------------------------

    def get_template(self, template_name: str) -> Optional[Dict[str, str]]:
        """Return the raw template dict for *template_name*, or ``None``."""
        return self.templates.get(template_name)

    def get_template_names(self) -> List[str]:
        """Return a sorted list of all currently loaded template names."""
        return sorted(self.templates.keys())

    # ------------------------------------------------------------------
    # Public API — lifecycle helpers
    # ------------------------------------------------------------------

    def reload_templates(self) -> int:
        """Re-load all templates from disk.

        Useful for hot-reloading during development or after template
        files have been updated on disk.

        Returns
        -------
        int
            The number of templates successfully loaded.
        """
        self.templates = {}
        self._do_load_templates()
        return self._metrics["templates_loaded"]

    def validate_templates(self) -> Dict[str, List[str]]:
        """Validate all currently loaded templates.

        Checks that each template has the required ``"system"`` and
        ``"user"`` keys and that format placeholders are syntactically
        valid Python format strings.

        Returns
        -------
        Dict[str, List[str]]
            Mapping of ``template_name → list_of_issues``.  An empty list
            means the template is valid.
        """

        results: Dict[str, List[str]] = {}
        formatter = string.Formatter()

        for name, tmpl in self.templates.items():
            issues: List[str] = []

            # Check required keys
            if "system" not in tmpl:
                issues.append("Missing required key 'system'")
            if "user" not in tmpl:
                issues.append("Missing required key 'user'")

            # Validate format strings are parseable
            for key in ("system", "user"):
                content = tmpl.get(key)
                if content is not None:
                    try:
                        list(formatter.parse(content))
                    except (ValueError, KeyError) as exc:
                        issues.append(
                            f"Invalid format string in '{key}': {exc}"
                        )

            results[name] = issues

        return results

    # ------------------------------------------------------------------
    # Public API — metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the prompt manager.

        Returns
        -------
        Dict[str, Any]
            Keys: ``templates_loaded``, ``prompts_built``, ``budget_warnings``.
        """
        return dict(self._metrics)

    # ------------------------------------------------------------------
    # Private — token estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        Uses the rough heuristic of 4 characters per token (per README.md
        lines 1216-1218).
        """
        return len(text) // CHARS_PER_TOKEN

    # ------------------------------------------------------------------
    # Private — template loading
    # ------------------------------------------------------------------

    def _do_load_templates(self) -> None:
        """Scan *template_dir* for ``*.yaml`` files and load them.

        Populates ``self.templates`` and updates the ``templates_loaded``
        metric.  Logs a warning if the directory is missing or empty but
        does **not** raise — this allows graceful initialisation when
        templates are not yet available.
        """

        loaded: Dict[str, Dict[str, str]] = {}

        if not self.template_dir.exists() or not self.template_dir.is_dir():
            logger.warning(
                "no_templates_found",
                template_dir=str(self.template_dir),
                reason="directory_missing",
            )
            self.templates = loaded
            self._metrics["templates_loaded"] = 0
            return

        yaml_files = sorted(self.template_dir.glob("*.yaml"))

        if not yaml_files:
            logger.warning(
                "no_templates_found",
                template_dir=str(self.template_dir),
                reason="no_yaml_files",
            )
            self.templates = loaded
            self._metrics["templates_loaded"] = 0
            return

        for yaml_path in yaml_files:
            template_name = yaml_path.stem
            try:
                raw_content = yaml_path.read_text(encoding="utf-8")
                parsed = yaml.safe_load(raw_content)

                if not isinstance(parsed, dict):
                    logger.warning(
                        "template_invalid_format",
                        template=template_name,
                        reason="root_not_dict",
                        path=str(yaml_path),
                    )
                    continue

                # Validate required keys — both "system" and "user" must exist
                missing_keys: List[str] = []
                if "system" not in parsed:
                    missing_keys.append("system")
                if "user" not in parsed:
                    missing_keys.append("user")

                if missing_keys:
                    logger.warning(
                        "template_missing_keys",
                        template=template_name,
                        missing_keys=missing_keys,
                        path=str(yaml_path),
                    )
                    continue

                # Ensure the values are strings
                loaded[template_name] = {
                    "system": str(parsed["system"]),
                    "user": str(parsed["user"]),
                }

            except yaml.YAMLError as exc:
                logger.error(
                    "template_parse_error",
                    template=template_name,
                    error=str(exc),
                    path=str(yaml_path),
                )
            except OSError as exc:
                logger.error(
                    "template_read_error",
                    template=template_name,
                    error=str(exc),
                    path=str(yaml_path),
                )

        self.templates = loaded
        self._metrics["templates_loaded"] = len(loaded)

        logger.info(
            "templates_loaded",
            count=len(loaded),
            names=sorted(loaded.keys()),
        )

    # ------------------------------------------------------------------
    # Private — budget truncation
    # ------------------------------------------------------------------

    def _truncate_to_budget(
        self,
        system_prompt: str,
        user_prompt: str,
        budget_tokens: int,
    ) -> Tuple[str, str]:
        """Truncate prompts to fit within *budget_tokens*.

        Strategy: keep the system prompt intact (it contains the agent
        persona and is typically shorter) and truncate the user prompt
        to fit within the remaining budget.  A ``...[truncated]`` suffix
        is appended to indicate truncation.

        Parameters
        ----------
        system_prompt : str
        user_prompt : str
        budget_tokens : int

        Returns
        -------
        Tuple[str, str]
            ``(system_prompt, user_prompt)`` — possibly truncated.
        """

        system_tokens = self._estimate_tokens(system_prompt)
        combined_tokens = self._estimate_tokens(system_prompt + user_prompt)

        if combined_tokens <= budget_tokens:
            return system_prompt, user_prompt

        # Calculate the character budget remaining for the user prompt
        remaining_token_budget = budget_tokens - system_tokens
        if remaining_token_budget <= 0:
            # System prompt alone exceeds the budget — nothing we can do
            # except return as-is.  The caller has already been warned.
            return system_prompt, user_prompt

        suffix_len = len(_TRUNCATION_SUFFIX)
        max_user_chars = (remaining_token_budget * CHARS_PER_TOKEN) - suffix_len

        if max_user_chars <= 0:
            user_prompt = _TRUNCATION_SUFFIX
        else:
            user_prompt = user_prompt[:max_user_chars] + _TRUNCATION_SUFFIX

        return system_prompt, user_prompt
