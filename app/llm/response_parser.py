"""ResponseParser for extracting structured JSON from LLM text output.

Validates against Pydantic V2 schemas with a 3-retry loop on parse failure.
Supports extracting JSON from mixed text/markdown output, handling code fences,
and providing detailed error diagnostics.

This module implements the structured JSON extraction and validation pipeline
described in AAP Section 0.5.1 Group 4 and README.md lines 218-221.  Every LLM
response is passed through the ``ResponseParser`` before being consumed by the
rest of the system, ensuring 100 % of LLM responses are validated before use
(AAP Section 0.1.1 Success Criterion #10).

The extraction logic handles common LLM output quirks:
- Clean JSON responses (direct ``json.loads``)
- Markdown code fences (````json … ```` or ```` … ````)
- JSON embedded in prose text (greedy brace/bracket matching)
- Trailing commas, single-quoted strings, and other malformed JSON artifacts

Validation uses Pydantic V2's ``model_validate()`` method (not the deprecated
``parse_obj``), producing detailed field-level error reports on failure.

Typical usage::

    from pydantic import BaseModel
    from app.llm.response_parser import ResponseParser

    class InvoiceDecision(BaseModel):
        match_status: str
        gl_coding: str

    parser = ResponseParser(max_retries=3)
    result = parser.parse(llm_output, InvoiceDecision)
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import structlog
from pydantic import BaseModel, ValidationError

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Generic type variable bound to Pydantic BaseModel — used as the schema
# parameter in parse(), parse_with_retry(), parse_list(), etc.
# ---------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Maximum length of raw text included in error messages / log entries to
# prevent log bloat from extremely large LLM outputs.
# ---------------------------------------------------------------------------
_MAX_RAW_TEXT_LOG_LENGTH: int = 500


# ============================================================================
# Custom Exception Classes
# ============================================================================


class ParseError(Exception):
    """Raised when JSON extraction or initial parsing of LLM output fails.

    Attributes:
        message:  Human-readable description of the failure.
        raw_text: The original (possibly truncated) LLM output that caused
                  the error.
        attempt:  1-based attempt number at which the error occurred.
        errors:   Optional list of structured error dictionaries providing
                  additional diagnostic detail (e.g. per-strategy failure
                  reasons during JSON extraction).
    """

    def __init__(
        self,
        message: str,
        raw_text: str = "",
        attempt: int = 1,
        errors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.raw_text: str = raw_text
        self.attempt: int = attempt
        self.errors: Optional[List[Dict[str, Any]]] = errors


class ValidationFailedError(ParseError):
    """Raised when extracted JSON fails Pydantic V2 schema validation.

    Extends :class:`ParseError` with a ``validation_errors`` attribute that
    contains the raw Pydantic error list so callers can inspect individual
    field-level failures.

    Attributes:
        validation_errors: List of Pydantic error dictionaries (as returned
                           by ``ValidationError.errors()``).
    """

    def __init__(
        self,
        message: str,
        raw_text: str = "",
        attempt: int = 1,
        errors: Optional[List[Dict[str, Any]]] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            message=message,
            raw_text=raw_text,
            attempt=attempt,
            errors=errors,
        )
        self.validation_errors: Optional[List[Dict[str, Any]]] = (
            validation_errors or []
        )


# ============================================================================
# ResponseParser — Main Public API
# ============================================================================


class ResponseParser:
    """Extract and validate structured JSON from raw LLM text output.

    The parser implements a multi-strategy JSON extraction pipeline followed
    by Pydantic V2 schema validation.  It supports both synchronous and
    asynchronous retry loops for seamless integration with the async agent
    and decision-engine subsystems.

    Parameters:
        max_retries: Maximum number of parse attempts before giving up.
                     Default is ``3`` per AAP ("3-retry loop on parse
                     failure").
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries: int = max_retries

        # Metrics counters
        self._total_parses: int = 0
        self._successful_parses: int = 0
        self._failed_parses: int = 0
        self._retry_count: int = 0

        logger.debug("response_parser_initialized", max_retries=max_retries)

    # ------------------------------------------------------------------
    # Public: parse (single attempt)
    # ------------------------------------------------------------------

    def parse(self, raw_text: str, schema: Type[T], strict: bool = True) -> T:
        """Parse *raw_text* and validate against a Pydantic V2 *schema*.

        This is the primary single-attempt parse method.  It extracts JSON
        from the raw LLM output and validates it using
        ``schema.model_validate()``.

        Args:
            raw_text: Raw LLM text output (may contain prose, code fences,
                      etc.).
            schema:   A Pydantic V2 ``BaseModel`` subclass used for
                      validation.
            strict:   When ``True`` (default) uses strict validation mode;
                      when ``False`` allows coercion of compatible types.

        Returns:
            A validated instance of *schema*.

        Raises:
            ParseError: If JSON extraction fails.
            ValidationFailedError: If extracted JSON fails schema validation.
        """
        self._total_parses += 1

        try:
            extracted = self._extract_json(raw_text)
        except ParseError:
            self._failed_parses += 1
            raise

        # Extracted data must be a dict for single-model validation
        if not isinstance(extracted, dict):
            self._failed_parses += 1
            raise ParseError(
                message="Expected a JSON object but extracted a JSON array",
                raw_text=raw_text[:_MAX_RAW_TEXT_LOG_LENGTH],
                attempt=1,
                errors=[{"detail": "Extracted JSON is a list, not an object"}],
            )

        try:
            result = self._validate_against_schema(extracted, schema, strict=strict)
        except ValidationFailedError:
            self._failed_parses += 1
            raise

        self._successful_parses += 1
        return result

    # ------------------------------------------------------------------
    # Public: parse_with_retry (synchronous retry loop)
    # ------------------------------------------------------------------

    def parse_with_retry(
        self,
        raw_text: str,
        schema: Type[T],
        retry_callback: Optional[Callable[..., str]] = None,
    ) -> T:
        """Parse with up to ``max_retries`` attempts, optionally re-prompting.

        On each ``ParseError`` the parser:
        1. Logs a WARNING with the attempt number and error detail.
        2. If *retry_callback* is provided, calls it to obtain fresh
           ``raw_text`` (e.g. by re-prompting the LLM with clarified
           instructions).
        3. Retries until ``max_retries`` is exhausted.

        Args:
            raw_text:       Initial raw LLM text.
            schema:         Pydantic V2 ``BaseModel`` subclass for validation.
            retry_callback: Optional callable returning a new ``raw_text``
                            string for the next attempt.  The callable
                            receives no arguments; callers should use
                            closures or ``functools.partial`` to bind
                            context.

        Returns:
            A validated instance of *schema*.

        Raises:
            ParseError: If all retry attempts are exhausted.
        """
        last_error: Optional[ParseError] = None
        accumulated_errors: List[Dict[str, Any]] = []
        current_text = raw_text

        for attempt in range(1, self.max_retries + 1):
            try:
                return self.parse(current_text, schema)
            except (ParseError, ValidationFailedError) as exc:
                last_error = exc
                self._retry_count += 1
                accumulated_errors.append(
                    {"attempt": attempt, "error": str(exc)}
                )

                logger.warning(
                    "response_parse_retry",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(exc),
                    raw_text_preview=current_text[:200],
                )

                # Obtain fresh text from the caller if possible
                if retry_callback is not None and attempt < self.max_retries:
                    try:
                        current_text = retry_callback()
                    except Exception as cb_err:  # noqa: BLE001
                        logger.warning(
                            "response_parse_retry_callback_failed",
                            attempt=attempt,
                            error=str(cb_err),
                        )

        # All retries exhausted
        self._failed_parses += 1
        truncated = raw_text[:_MAX_RAW_TEXT_LOG_LENGTH]
        logger.error(
            "response_parse_failed",
            raw_text_preview=truncated,
            total_attempts=self.max_retries,
            errors=accumulated_errors,
        )
        raise ParseError(
            message=(
                f"Failed to parse LLM response after {self.max_retries} "
                f"attempts: {last_error}"
            ),
            raw_text=truncated,
            attempt=self.max_retries,
            errors=accumulated_errors,
        )

    # ------------------------------------------------------------------
    # Public: async_parse_with_retry (async retry loop)
    # ------------------------------------------------------------------

    async def async_parse_with_retry(
        self,
        raw_text: str,
        schema: Type[T],
        retry_callback: Optional[Callable[..., Any]] = None,
    ) -> T:
        """Async version of :meth:`parse_with_retry`.

        Identical retry semantics, but ``retry_callback`` may be an async
        coroutine function.  If it is, the parser will ``await`` it;
        otherwise it is called synchronously.

        Args:
            raw_text:       Initial raw LLM text.
            schema:         Pydantic V2 ``BaseModel`` subclass.
            retry_callback: Optional sync or async callable returning a new
                            ``raw_text`` string.

        Returns:
            A validated instance of *schema*.

        Raises:
            ParseError: If all retry attempts are exhausted.
        """
        last_error: Optional[ParseError] = None
        accumulated_errors: List[Dict[str, Any]] = []
        current_text = raw_text

        for attempt in range(1, self.max_retries + 1):
            try:
                return self.parse(current_text, schema)
            except (ParseError, ValidationFailedError) as exc:
                last_error = exc
                self._retry_count += 1
                accumulated_errors.append(
                    {"attempt": attempt, "error": str(exc)}
                )

                logger.warning(
                    "response_parse_retry",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(exc),
                    raw_text_preview=current_text[:200],
                )

                # Obtain fresh text — await if the callback is async
                if retry_callback is not None and attempt < self.max_retries:
                    try:
                        if asyncio.iscoroutinefunction(retry_callback):
                            current_text = await retry_callback()
                        else:
                            current_text = retry_callback()
                    except Exception as cb_err:  # noqa: BLE001
                        logger.warning(
                            "response_parse_retry_callback_failed",
                            attempt=attempt,
                            error=str(cb_err),
                        )

        # All retries exhausted
        self._failed_parses += 1
        truncated = raw_text[:_MAX_RAW_TEXT_LOG_LENGTH]
        logger.error(
            "response_parse_failed",
            raw_text_preview=truncated,
            total_attempts=self.max_retries,
            errors=accumulated_errors,
        )
        raise ParseError(
            message=(
                f"Failed to parse LLM response after {self.max_retries} "
                f"attempts: {last_error}"
            ),
            raw_text=truncated,
            attempt=self.max_retries,
            errors=accumulated_errors,
        )

    # ------------------------------------------------------------------
    # Public: parse_json (untyped extraction)
    # ------------------------------------------------------------------

    def parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Extract a JSON dict from *raw_text* **without** schema validation.

        Useful when the caller needs flexibility or when no Pydantic schema
        is available.

        Args:
            raw_text: Raw LLM text output.

        Returns:
            A ``dict`` parsed from the extracted JSON.

        Raises:
            ParseError: If no valid JSON object can be extracted.
        """
        self._total_parses += 1
        try:
            extracted = self._extract_json(raw_text)
        except ParseError:
            self._failed_parses += 1
            raise

        if not isinstance(extracted, dict):
            self._failed_parses += 1
            raise ParseError(
                message="Expected a JSON object but extracted a JSON array",
                raw_text=raw_text[:_MAX_RAW_TEXT_LOG_LENGTH],
                attempt=1,
            )

        self._successful_parses += 1
        return extracted

    # ------------------------------------------------------------------
    # Public: parse_list (array extraction + per-element validation)
    # ------------------------------------------------------------------

    def parse_list(self, raw_text: str, schema: Type[T]) -> List[T]:
        """Parse a JSON array and validate each element against *schema*.

        Args:
            raw_text: Raw LLM text containing a JSON array.
            schema:   Pydantic V2 ``BaseModel`` subclass applied to every
                      element.

        Returns:
            A list of validated *schema* instances.

        Raises:
            ParseError: If JSON extraction fails or the extracted value is
                        not a list.
            ValidationFailedError: If any element fails schema validation.
        """
        self._total_parses += 1

        try:
            extracted = self._extract_json(raw_text)
        except ParseError:
            self._failed_parses += 1
            raise

        if not isinstance(extracted, list):
            # Wrap a single dict in a list as a convenience
            if isinstance(extracted, dict):
                extracted = [extracted]
            else:
                self._failed_parses += 1
                raise ParseError(
                    message="Expected a JSON array",
                    raw_text=raw_text[:_MAX_RAW_TEXT_LOG_LENGTH],
                    attempt=1,
                )

        validated: List[T] = []
        element_errors: List[Dict[str, Any]] = []
        for idx, item in enumerate(extracted):
            if not isinstance(item, dict):
                element_errors.append(
                    {"index": idx, "error": "Element is not a JSON object"}
                )
                continue
            try:
                validated.append(self._validate_against_schema(item, schema))
            except ValidationFailedError as exc:
                element_errors.append(
                    {
                        "index": idx,
                        "error": str(exc),
                        "validation_errors": exc.validation_errors,
                    }
                )

        if element_errors:
            self._failed_parses += 1
            raise ValidationFailedError(
                message=(
                    f"Validation failed for {len(element_errors)} of "
                    f"{len(extracted)} elements"
                ),
                raw_text=raw_text[:_MAX_RAW_TEXT_LOG_LENGTH],
                attempt=1,
                errors=element_errors,
                validation_errors=element_errors,
            )

        self._successful_parses += 1
        return validated

    # ------------------------------------------------------------------
    # Public: metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return parse operation metrics.

        Returns:
            Dictionary containing ``total_parses``, ``successful_parses``,
            ``failed_parses``, ``retry_count``, and ``success_rate``.
        """
        total = self._total_parses
        return {
            "total_parses": total,
            "successful_parses": self._successful_parses,
            "failed_parses": self._failed_parses,
            "retry_count": self._retry_count,
            "success_rate": (
                self._successful_parses / total if total > 0 else 0.0
            ),
        }

    def reset_metrics(self) -> None:
        """Reset all metric counters to zero."""
        self._total_parses = 0
        self._successful_parses = 0
        self._failed_parses = 0
        self._retry_count = 0

    # ==================================================================
    # Private: JSON extraction pipeline
    # ==================================================================

    def _extract_json(self, text: str) -> Union[Dict[str, Any], List[Any]]:
        """Try multiple strategies to extract JSON from *text*.

        Strategies are attempted in order of reliability:
        1. Direct ``json.loads`` on the stripped text.
        2. Extract from markdown code fences (````json … ````).
        3. Greedy brace matching (first ``{`` … last ``}``).
        4. Greedy bracket matching (first ``[`` … last ``]``).
        5. Clean common LLM artifacts then retry ``json.loads``.

        Args:
            text: Raw LLM output string.

        Returns:
            A parsed Python ``dict`` or ``list``.

        Raises:
            ParseError: If no strategy succeeds.
        """
        if not text or not text.strip():
            raise ParseError(
                message="Empty or whitespace-only LLM response",
                raw_text="",
                attempt=1,
            )

        stripped = text.strip()

        # Strategy 1 — direct parse
        try:
            result = json.loads(stripped)
            if isinstance(result, (dict, list)):
                logger.debug("json_extracted", strategy="direct")
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2 — markdown code fences
        fenced = self._extract_from_code_fence(text)
        if fenced is not None:
            try:
                result = json.loads(fenced)
                if isinstance(result, (dict, list)):
                    logger.debug("json_extracted", strategy="code_fence")
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 3 — greedy braces (JSON object)
        braced = self._extract_between_braces(text)
        if braced is not None:
            try:
                result = json.loads(braced)
                if isinstance(result, (dict, list)):
                    logger.debug("json_extracted", strategy="braces")
                    return result
            except json.JSONDecodeError:
                # Try cleaning before giving up on this strategy
                try:
                    cleaned = self._clean_json_string(braced)
                    result = json.loads(cleaned)
                    if isinstance(result, (dict, list)):
                        logger.debug(
                            "json_extracted", strategy="braces_cleaned"
                        )
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategy 4 — greedy brackets (JSON array)
        bracketed = self._extract_between_brackets(text)
        if bracketed is not None:
            try:
                result = json.loads(bracketed)
                if isinstance(result, (dict, list)):
                    logger.debug("json_extracted", strategy="brackets")
                    return result
            except json.JSONDecodeError:
                try:
                    cleaned = self._clean_json_string(bracketed)
                    result = json.loads(cleaned)
                    if isinstance(result, (dict, list)):
                        logger.debug(
                            "json_extracted", strategy="brackets_cleaned"
                        )
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass

        # Strategy 5 — full clean + retry
        try:
            cleaned_full = self._clean_json_string(stripped)
            result = json.loads(cleaned_full)
            if isinstance(result, (dict, list)):
                logger.debug("json_extracted", strategy="full_clean")
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # All strategies exhausted
        raise ParseError(
            message="Could not extract valid JSON from LLM response",
            raw_text=text[:_MAX_RAW_TEXT_LOG_LENGTH],
            attempt=1,
            errors=[
                {
                    "detail": (
                        "Tried 5 extraction strategies (direct, code_fence, "
                        "braces, brackets, full_clean) — all failed"
                    )
                }
            ],
        )

    # ------------------------------------------------------------------

    def _extract_from_code_fence(self, text: str) -> Optional[str]:
        """Extract JSON content from a markdown code fence.

        Matches patterns like:
        - ````json\\n{...}\\n````
        - ````\\n{...}\\n````

        Returns:
            The inner content of the code fence, or ``None`` if no match.
        """
        pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _extract_between_braces(self, text: str) -> Optional[str]:
        """Return substring from the first ``{`` to the last ``}``.

        Uses simple index-based extraction; caller must still validate
        with ``json.loads``.
        """
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]
        return None

    def _extract_between_brackets(self, text: str) -> Optional[str]:
        """Return substring from the first ``[`` to the last ``]``."""
        first = text.find("[")
        last = text.rfind("]")
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]
        return None

    def _clean_json_string(self, text: str) -> str:
        """Remove common LLM response artifacts that break JSON parsing.

        Handles:
        - Leading prose prefixes ("Here is the JSON:", etc.)
        - Trailing explanation text after the JSON body
        - Trailing commas before closing brackets/braces
        - Single-quoted strings (naïve replacement)
        - Escaped newlines inside string values
        """
        cleaned = text

        # Remove common LLM prose prefixes
        prefix_patterns = [
            r"^(?:Here\s+is\s+(?:the\s+)?(?:JSON|response|output)\s*:?\s*)",
            r"^(?:Sure[,!]?\s*(?:here(?:'s|\s+is)\s+)?(?:the\s+)?(?:JSON|response|output)\s*:?\s*)",
            r"^(?:The\s+(?:JSON|response|output)\s+(?:is|would\s+be)\s*:?\s*)",
            r"^(?:```json\s*)",
            r"^(?:```\s*)",
        ]
        for pat in prefix_patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove trailing prose after the JSON body — find the last } or ]
        # and strip everything after it.
        last_brace = cleaned.rfind("}")
        last_bracket = cleaned.rfind("]")
        last_close = max(last_brace, last_bracket)
        if last_close != -1 and last_close < len(cleaned) - 1:
            remainder = cleaned[last_close + 1 :].strip()
            # Only strip if the remainder looks like prose (starts with a
            # letter or common punctuation, not another JSON token).
            if remainder and not remainder[0] in ("{", "[", '"', "'"):
                cleaned = cleaned[: last_close + 1]

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

        # Naïve single-quote → double-quote for simple cases
        # Only replace when quotes are used as JSON string delimiters
        if cleaned and cleaned[0] == "'" and cleaned[-1] == "'":
            cleaned = '"' + cleaned[1:-1] + '"'
        cleaned = re.sub(
            r"(?<=[\[{,:])\s*'([^']*?)'\s*(?=[,\]}:])",
            r' "\1"',
            cleaned,
        )

        # Remove trailing code fence markers
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        return cleaned.strip()

    # ==================================================================
    # Private: Pydantic V2 validation
    # ==================================================================

    def _validate_against_schema(
        self,
        data: Dict[str, Any],
        schema: Type[T],
        strict: bool = True,
    ) -> T:
        """Validate *data* against a Pydantic V2 *schema*.

        Uses ``schema.model_validate(data)`` — the Pydantic V2 API.

        Args:
            data:   Parsed JSON dictionary.
            schema: Pydantic V2 ``BaseModel`` subclass.
            strict: Pass through to Pydantic's ``strict`` parameter.

        Returns:
            Validated model instance.

        Raises:
            ValidationFailedError: With detailed field-level error info.
        """
        try:
            return schema.model_validate(data, strict=strict)
        except ValidationError as exc:
            error_list = exc.errors()
            formatted = self._format_validation_errors(error_list)
            logger.debug(
                "schema_validation_failed",
                schema=schema.__name__,
                error_count=len(error_list),
                errors_formatted=formatted,
            )
            raise ValidationFailedError(
                message=(
                    f"Pydantic validation failed for {schema.__name__}: "
                    f"{formatted}"
                ),
                raw_text=json.dumps(data)[:_MAX_RAW_TEXT_LOG_LENGTH],
                attempt=1,
                errors=[{"validation": formatted}],
                validation_errors=error_list,  # type: ignore[arg-type]
            ) from exc

    @staticmethod
    def _format_validation_errors(errors: List[Dict[str, Any]]) -> str:
        """Format Pydantic validation errors into a readable string.

        Each error is rendered as ``field → type: message``.

        Args:
            errors: List of error dicts from ``ValidationError.errors()``.

        Returns:
            A semicolon-separated summary of all validation errors.
        """
        parts: List[str] = []
        for err in errors:
            loc = " → ".join(str(loc_part) for loc_part in err.get("loc", []))
            err_type = err.get("type", "unknown")
            msg = err.get("msg", "")
            parts.append(f"{loc} ({err_type}): {msg}")
        return "; ".join(parts) if parts else "No error details available"
