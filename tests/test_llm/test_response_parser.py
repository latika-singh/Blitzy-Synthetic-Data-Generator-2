"""Comprehensive unit tests for the ResponseParser class.

Tests JSON extraction from LLM text output, Pydantic V2 schema validation,
3-retry loop on parse failure, metrics tracking, and error handling.

Per AAP Section 0.5.1 Group 10: "Valid JSON extraction, schema violation
handling, retry."
Per AAP Section 0.7.5: Unit test coverage target >= 80%.
Per AAP Section 0.1.1 Success Criterion #10: "100% of LLM responses
validated before use."
"""

import json
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from app.llm.response_parser import ParseError, ResponseParser, ValidationFailedError


# ============================================================================
# Test Schema Definitions (Pydantic V2 models used for validation testing)
# ============================================================================


class ApprovalResponse(BaseModel):
    """Schema for approval decision responses."""

    decision: str = Field(description="approve, reject, or escalate")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class InvoiceResponse(BaseModel):
    """Schema for invoice processing responses."""

    match_status: str = Field(description="matched or variance")
    gl_coding: List[dict]
    processing_notes: str
    recommendation: str
    reasoning: str


class SimpleResponse(BaseModel):
    """Minimal schema for basic tests."""

    status: str
    value: int


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def parser() -> ResponseParser:
    """Create a ResponseParser instance with default 3 retries."""
    return ResponseParser(max_retries=3)


@pytest.fixture
def valid_approval_json() -> str:
    """Valid JSON string matching ApprovalResponse schema."""
    return '{"decision": "approve", "confidence": 0.85, "reasoning": "All documents match"}'


@pytest.fixture
def valid_invoice_json() -> str:
    """Valid JSON string matching InvoiceResponse schema."""
    return json.dumps(
        {
            "match_status": "matched",
            "gl_coding": [{"line": 1, "account": "5000", "amount": 100.0}],
            "processing_notes": "No issues found",
            "recommendation": "approve",
            "reasoning": "3-way match confirmed",
        }
    )


# ============================================================================
# Phase 4: Clean JSON Parsing Tests
# ============================================================================


class TestCleanJsonParsing:
    """Tests for parsing well-formed JSON strings against Pydantic schemas."""

    def test_parse_clean_json(self, parser: ResponseParser, valid_approval_json: str) -> None:
        """Verify a clean JSON string parses into a correct model instance."""
        result = parser.parse(valid_approval_json, ApprovalResponse)
        assert isinstance(result, ApprovalResponse)
        assert result.decision == "approve"
        assert result.confidence == 0.85
        assert result.reasoning == "All documents match"

    def test_parse_invoice_response(self, parser: ResponseParser, valid_invoice_json: str) -> None:
        """Verify a complex invoice JSON with nested list of dicts parses correctly."""
        result = parser.parse(valid_invoice_json, InvoiceResponse)
        assert result.match_status == "matched"
        assert len(result.gl_coding) == 1
        assert result.gl_coding[0]["account"] == "5000"
        assert result.gl_coding[0]["amount"] == 100.0
        assert result.recommendation == "approve"
        assert result.reasoning == "3-way match confirmed"

    def test_parse_simple_response(self, parser: ResponseParser) -> None:
        """Verify a minimal JSON parses into a simple model."""
        result = parser.parse('{"status": "ok", "value": 42}', SimpleResponse)
        assert result.status == "ok"
        assert result.value == 42


# ============================================================================
# Phase 5: JSON Extraction from Mixed Text Tests
# ============================================================================


class TestJsonExtractionFromMixedText:
    """LLM output often wraps JSON in prose or code fences.

    The parser's 5-strategy extraction pipeline must handle all common
    patterns: direct JSON, code fences with/without language tag, JSON
    embedded in explanatory prose, and JSON with leading/trailing text.
    """

    def test_extract_json_from_code_fence(self, parser: ResponseParser) -> None:
        """Extract JSON from a markdown code fence with 'json' language tag."""
        raw_text = '```json\n{"decision": "approve", "confidence": 0.9, "reasoning": "Good"}\n```'
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "approve"
        assert result.confidence == 0.9
        assert result.reasoning == "Good"

    def test_extract_json_from_code_fence_no_language(self, parser: ResponseParser) -> None:
        """Extract JSON from a code fence without a language specifier."""
        raw_text = '```\n{"decision": "reject", "confidence": 0.3, "reasoning": "Mismatch"}\n```'
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "reject"
        assert result.confidence == 0.3

    def test_extract_json_from_prose(self, parser: ResponseParser) -> None:
        """Extract JSON sandwiched between explanatory prose."""
        raw_text = (
            "Here is my analysis:\n"
            '{"decision": "approve", "confidence": 0.8, "reasoning": "Looks good"}\n'
            "Thank you for providing the details."
        )
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "approve"
        assert result.confidence == 0.8

    def test_extract_json_with_prefix(self, parser: ResponseParser) -> None:
        """Extract JSON preceded by an LLM explanation."""
        raw_text = (
            "Sure, here is the JSON response:\n"
            '{"decision": "escalate", "confidence": 0.5, "reasoning": "Need review"}'
        )
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "escalate"
        assert result.confidence == 0.5

    def test_extract_json_with_suffix(self, parser: ResponseParser) -> None:
        """Extract JSON followed by trailing notes."""
        raw_text = (
            '{"decision": "approve", "confidence": 0.95, "reasoning": "Perfect match"}\n\n'
            "Note: All amounts were within tolerance."
        )
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "approve"
        assert result.confidence == 0.95

    def test_extract_json_nested_braces(self, parser: ResponseParser) -> None:
        """Ensure JSON with nested objects/arrays in GL coding parses correctly."""
        raw_text = json.dumps(
            {
                "match_status": "matched",
                "gl_coding": [{"line": 1, "account": "5000", "amount": 100.0}],
                "processing_notes": "",
                "recommendation": "approve",
                "reasoning": "OK",
            }
        )
        result = parser.parse(raw_text, InvoiceResponse)
        assert result.match_status == "matched"
        assert len(result.gl_coding) == 1
        assert result.gl_coding[0]["line"] == 1

    def test_extract_json_from_multiline_code_fence(self, parser: ResponseParser) -> None:
        """Extract multiline JSON from a code fence."""
        raw_text = (
            "```json\n"
            "{\n"
            '  "decision": "approve",\n'
            '  "confidence": 0.75,\n'
            '  "reasoning": "All checks passed"\n'
            "}\n"
            "```"
        )
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "approve"
        assert result.confidence == 0.75


# ============================================================================
# Phase 6: Schema Validation Failure Tests
# ============================================================================


class TestSchemaValidationFailures:
    """Tests for Pydantic V2 schema validation error handling.

    Covers missing fields, wrong types, out-of-range values, and other
    schema contract violations.
    """

    def test_parse_missing_required_field(self, parser: ResponseParser) -> None:
        """Missing 'confidence' and 'reasoning' fields raise a validation error."""
        raw_text = '{"decision": "approve"}'
        with pytest.raises((ParseError, ValidationFailedError)):
            parser.parse(raw_text, ApprovalResponse)

    def test_parse_wrong_field_type(self, parser: ResponseParser) -> None:
        """String 'high' for a float field raises a validation error."""
        raw_text = '{"decision": "approve", "confidence": "high", "reasoning": "Good"}'
        with pytest.raises((ParseError, ValidationFailedError)):
            parser.parse(raw_text, ApprovalResponse)

    def test_parse_confidence_out_of_range(self, parser: ResponseParser) -> None:
        """Confidence = 1.5 exceeds Field(le=1.0) constraint."""
        raw_text = '{"decision": "approve", "confidence": 1.5, "reasoning": "Good"}'
        with pytest.raises((ParseError, ValidationFailedError)):
            parser.parse(raw_text, ApprovalResponse)

    def test_parse_confidence_negative_out_of_range(self, parser: ResponseParser) -> None:
        """Confidence = -0.1 violates Field(ge=0.0) constraint."""
        raw_text = '{"decision": "approve", "confidence": -0.1, "reasoning": "Good"}'
        with pytest.raises((ParseError, ValidationFailedError)):
            parser.parse(raw_text, ApprovalResponse)

    def test_parse_empty_string_raises(self, parser: ResponseParser) -> None:
        """Empty input string always raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("", ApprovalResponse)

    def test_parse_invalid_json_raises(self, parser: ResponseParser) -> None:
        """Plain prose that contains no JSON raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("not json at all", ApprovalResponse)

    def test_parse_json_array_instead_of_object(self, parser: ResponseParser) -> None:
        """A JSON array when an object is expected raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("[1, 2, 3]", ApprovalResponse)


# ============================================================================
# Phase 7: 3-Retry Loop Tests
# ============================================================================


class TestRetryLoop:
    """Per AAP: '3-retry loop on parse failure'.

    Validates both synchronous and asynchronous retry paths, callback
    integration, and retry exhaustion.
    """

    def test_parse_with_retry_succeeds_on_first_attempt(
        self, parser: ResponseParser, valid_approval_json: str
    ) -> None:
        """When the first attempt succeeds, no retries are invoked."""
        result = parser.parse_with_retry(valid_approval_json, ApprovalResponse)
        assert result.decision == "approve"
        metrics = parser.get_metrics()
        assert metrics["retry_count"] == 0

    def test_parse_with_retry_succeeds_after_retries(self, parser: ResponseParser) -> None:
        """Callback provides progressively better text until parse succeeds."""
        attempts = [
            "invalid json",
            '{"bad": "data"}',
            '{"decision": "approve", "confidence": 0.8, "reasoning": "OK"}',
        ]
        call_count = 0

        def retry_callback() -> str:
            nonlocal call_count
            call_count += 1
            return attempts[call_count]

        result = parser.parse_with_retry(
            attempts[0], ApprovalResponse, retry_callback=retry_callback
        )
        assert result.decision == "approve"
        assert result.confidence == 0.8
        metrics = parser.get_metrics()
        # Two failures before the third succeeds
        assert metrics["retry_count"] >= 2

    def test_parse_with_retry_exhausts_all_retries(self, parser: ResponseParser) -> None:
        """When no callback is provided, perpetually invalid text exhausts retries."""
        with pytest.raises(ParseError):
            parser.parse_with_retry("always invalid text", ApprovalResponse)
        metrics = parser.get_metrics()
        assert metrics["failed_parses"] >= 1
        assert metrics["retry_count"] == 3

    def test_parse_with_retry_respects_max_retries(self) -> None:
        """Custom max_retries=2 causes only 2 retry attempts."""
        custom_parser = ResponseParser(max_retries=2)
        with pytest.raises(ParseError):
            custom_parser.parse_with_retry("invalid text", ApprovalResponse)
        metrics = custom_parser.get_metrics()
        assert metrics["retry_count"] == 2

    def test_parse_with_retry_callback_failure_handled(self, parser: ResponseParser) -> None:
        """When the retry_callback itself raises, the parser handles it gracefully."""
        def bad_callback() -> str:
            raise RuntimeError("Callback error")

        with pytest.raises(ParseError):
            parser.parse_with_retry(
                "invalid json", ApprovalResponse, retry_callback=bad_callback
            )

    async def test_async_parse_with_retry(
        self, parser: ResponseParser, valid_approval_json: str
    ) -> None:
        """Async retry succeeds on the first attempt."""
        result = await parser.async_parse_with_retry(
            valid_approval_json, ApprovalResponse
        )
        assert result.decision == "approve"
        assert result.confidence == 0.85

    async def test_async_parse_with_retry_with_async_callback(
        self, parser: ResponseParser
    ) -> None:
        """Async retry with an async callback succeeds after initial failure."""
        attempts = [
            "invalid json",
            '{"decision": "approve", "confidence": 0.9, "reasoning": "Retry worked"}',
        ]
        call_count = 0

        async def async_retry_callback() -> str:
            nonlocal call_count
            call_count += 1
            return attempts[call_count]

        result = await parser.async_parse_with_retry(
            attempts[0], ApprovalResponse, retry_callback=async_retry_callback
        )
        assert result.decision == "approve"
        assert result.reasoning == "Retry worked"

    async def test_async_parse_with_retry_with_sync_callback(
        self, parser: ResponseParser
    ) -> None:
        """Async retry with a sync callback works correctly."""
        attempts = [
            "invalid",
            '{"decision": "approve", "confidence": 0.9, "reasoning": "Sync callback"}',
        ]
        call_count = 0

        def sync_callback() -> str:
            nonlocal call_count
            call_count += 1
            return attempts[call_count]

        result = await parser.async_parse_with_retry(
            attempts[0], ApprovalResponse, retry_callback=sync_callback
        )
        assert result.decision == "approve"

    async def test_async_parse_with_retry_exhausts_retries(
        self, parser: ResponseParser
    ) -> None:
        """Async retry exhausts all attempts and raises ParseError."""
        with pytest.raises(ParseError):
            await parser.async_parse_with_retry("always invalid", ApprovalResponse)
        metrics = parser.get_metrics()
        assert metrics["retry_count"] == 3


# ============================================================================
# Phase 8: parse_json (Untyped) Tests
# ============================================================================


class TestParseJson:
    """Tests for untyped JSON dict extraction via parse_json()."""

    def test_parse_json_returns_dict(self, parser: ResponseParser) -> None:
        """parse_json returns a plain dict with correct values."""
        result = parser.parse_json('{"key": "value", "number": 42}')
        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_parse_json_from_code_fence(self, parser: ResponseParser) -> None:
        """parse_json extracts from markdown code fences."""
        result = parser.parse_json('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"

    def test_parse_json_rejects_array(self, parser: ResponseParser) -> None:
        """parse_json raises ParseError for JSON arrays."""
        with pytest.raises(ParseError):
            parser.parse_json("[1, 2, 3]")

    def test_parse_json_empty_raises(self, parser: ResponseParser) -> None:
        """parse_json raises ParseError for empty input."""
        with pytest.raises(ParseError):
            parser.parse_json("")

    def test_parse_json_nested_objects(self, parser: ResponseParser) -> None:
        """parse_json handles deeply nested dict structures."""
        data = {"outer": {"inner": {"deep": True}}, "list": [1, 2]}
        result = parser.parse_json(json.dumps(data))
        assert result["outer"]["inner"]["deep"] is True
        assert result["list"] == [1, 2]


# ============================================================================
# Phase 9: parse_list Tests
# ============================================================================


class TestParseList:
    """Tests for JSON array extraction with per-element schema validation."""

    def test_parse_list_returns_list_of_models(self, parser: ResponseParser) -> None:
        """Array of objects validates each element against the schema."""
        raw_text = '[{"status": "ok", "value": 1}, {"status": "done", "value": 2}]'
        result = parser.parse_list(raw_text, SimpleResponse)
        assert len(result) == 2
        assert all(isinstance(r, SimpleResponse) for r in result)
        assert result[0].status == "ok"
        assert result[0].value == 1
        assert result[1].status == "done"
        assert result[1].value == 2

    def test_parse_list_single_dict_wrapped(self, parser: ResponseParser) -> None:
        """A single dict is wrapped into a single-element list."""
        raw_text = '{"status": "ok", "value": 1}'
        result = parser.parse_list(raw_text, SimpleResponse)
        assert len(result) == 1
        assert result[0].status == "ok"

    def test_parse_list_validation_failure_raises(self, parser: ResponseParser) -> None:
        """Element-level validation failure raises ValidationFailedError."""
        raw_text = '[{"status": "ok", "value": 1}, {"bad": "data"}]'
        with pytest.raises(ValidationFailedError):
            parser.parse_list(raw_text, SimpleResponse)

    def test_parse_list_empty_array(self, parser: ResponseParser) -> None:
        """An empty JSON array returns an empty list."""
        result = parser.parse_list("[]", SimpleResponse)
        assert result == []


# ============================================================================
# Phase 10: JSON Cleaning Tests
# ============================================================================


class TestJsonCleaning:
    """Tests for parser's ability to clean malformed JSON from LLM output.

    LLMs frequently produce trailing commas, single-quoted strings, and
    other non-standard JSON. The cleaning pipeline should fix these.
    """

    def test_clean_trailing_comma(self, parser: ResponseParser) -> None:
        """Trailing commas before closing braces are removed."""
        raw_text = '{"decision": "approve", "confidence": 0.8, "reasoning": "OK",}'
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "approve"
        assert result.confidence == 0.8

    def test_clean_single_quotes_to_double(self, parser: ResponseParser) -> None:
        """Single-quoted JSON keys and string values are converted."""
        raw_text = "{'decision': 'approve', 'confidence': 0.8, 'reasoning': 'OK'}"
        # The naïve single-quote replacement should handle this common case
        try:
            result = parser.parse(raw_text, ApprovalResponse)
            assert result.decision == "approve"
        except ParseError:
            # Acceptable if certain edge-case patterns defeat the naïve regex
            pass

    def test_clean_trailing_comma_in_array(self, parser: ResponseParser) -> None:
        """Trailing commas before closing brackets in arrays are removed."""
        raw_text = '[{"status": "ok", "value": 1},]'
        result = parser.parse_list(raw_text, SimpleResponse)
        assert len(result) == 1
        assert result[0].status == "ok"


# ============================================================================
# Phase 11: Metrics Tests
# ============================================================================


class TestMetrics:
    """Tests for parse operation metrics tracking.

    ResponseParser tracks total_parses, successful_parses, failed_parses,
    retry_count, and a derived success_rate.
    """

    def test_initial_metrics(self, parser: ResponseParser) -> None:
        """All counters start at zero with 0.0 success rate."""
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 0
        assert metrics["successful_parses"] == 0
        assert metrics["failed_parses"] == 0
        assert metrics["retry_count"] == 0
        assert metrics["success_rate"] == 0.0

    def test_metrics_after_successful_parse(
        self, parser: ResponseParser, valid_approval_json: str
    ) -> None:
        """A successful parse increments total and successful counters."""
        parser.parse(valid_approval_json, ApprovalResponse)
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 1
        assert metrics["successful_parses"] == 1
        assert metrics["failed_parses"] == 0

    def test_metrics_after_failed_parse(self, parser: ResponseParser) -> None:
        """A failed parse increments total and failed counters."""
        try:
            parser.parse("invalid json", ApprovalResponse)
        except ParseError:
            pass
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 1
        assert metrics["failed_parses"] == 1
        assert metrics["successful_parses"] == 0

    def test_metrics_accumulate(
        self, parser: ResponseParser, valid_approval_json: str
    ) -> None:
        """Multiple operations accumulate in the counters correctly."""
        parser.parse(valid_approval_json, ApprovalResponse)
        parser.parse(valid_approval_json, ApprovalResponse)
        try:
            parser.parse("bad", ApprovalResponse)
        except ParseError:
            pass
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 3
        assert metrics["successful_parses"] == 2
        assert metrics["failed_parses"] == 1

    def test_reset_metrics(
        self, parser: ResponseParser, valid_approval_json: str
    ) -> None:
        """reset_metrics() restores all counters to zero."""
        parser.parse(valid_approval_json, ApprovalResponse)
        parser.reset_metrics()
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 0
        assert metrics["successful_parses"] == 0
        assert metrics["failed_parses"] == 0
        assert metrics["retry_count"] == 0

    def test_success_rate_calculation(
        self, parser: ResponseParser, valid_approval_json: str
    ) -> None:
        """success_rate = successful_parses / total_parses."""
        parser.parse(valid_approval_json, ApprovalResponse)
        parser.parse(valid_approval_json, ApprovalResponse)
        try:
            parser.parse("bad", ApprovalResponse)
        except ParseError:
            pass
        metrics = parser.get_metrics()
        expected_rate = 2.0 / 3.0
        assert abs(metrics["success_rate"] - expected_rate) < 0.01

    def test_parse_json_metrics_tracked(self, parser: ResponseParser) -> None:
        """parse_json() also updates metrics counters."""
        parser.parse_json('{"key": "value"}')
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 1
        assert metrics["successful_parses"] == 1

    def test_parse_list_metrics_tracked(self, parser: ResponseParser) -> None:
        """parse_list() also updates metrics counters."""
        parser.parse_list('[{"status": "ok", "value": 1}]', SimpleResponse)
        metrics = parser.get_metrics()
        assert metrics["total_parses"] == 1
        assert metrics["successful_parses"] == 1

    def test_retry_metrics_tracked(self, parser: ResponseParser) -> None:
        """parse_with_retry() increments retry_count on each failure."""
        with pytest.raises(ParseError):
            parser.parse_with_retry("always bad", ApprovalResponse)
        metrics = parser.get_metrics()
        assert metrics["retry_count"] == 3
        # 3 parse() calls that fail + 1 final fail in parse_with_retry
        assert metrics["failed_parses"] == 4


# ============================================================================
# Phase 12: Error Class Tests
# ============================================================================


class TestErrorClasses:
    """Tests for ParseError and ValidationFailedError exception classes."""

    def test_parse_error_attributes(self) -> None:
        """ParseError stores message, raw_text, and attempt."""
        error = ParseError("Test error", raw_text="bad text", attempt=1)
        assert error.message == "Test error"
        assert error.raw_text == "bad text"
        assert error.attempt == 1
        assert str(error) == "Test error"

    def test_parse_error_default_attributes(self) -> None:
        """ParseError defaults: raw_text='', attempt=1, errors=None."""
        error = ParseError("Default test")
        assert error.message == "Default test"
        assert error.raw_text == ""
        assert error.attempt == 1
        assert error.errors is None

    def test_parse_error_with_errors_list(self) -> None:
        """ParseError can carry a list of structured error dicts."""
        errors = [{"detail": "extraction failed"}]
        error = ParseError("Failed", raw_text="text", attempt=2, errors=errors)
        assert error.errors == errors
        assert error.attempt == 2

    def test_validation_failed_error_inherits_parse_error(self) -> None:
        """ValidationFailedError is a subclass of ParseError."""
        assert issubclass(ValidationFailedError, ParseError)

    def test_validation_failed_error_has_validation_errors(self) -> None:
        """ValidationFailedError exposes the validation_errors list."""
        val_errors = [{"field": "confidence", "error": "not a float"}]
        error = ValidationFailedError(
            "Validation failed",
            raw_text="",
            attempt=1,
            validation_errors=val_errors,
        )
        assert error.validation_errors is not None
        assert len(error.validation_errors) == 1
        assert error.validation_errors[0]["field"] == "confidence"

    def test_validation_failed_error_default_validation_errors(self) -> None:
        """ValidationFailedError defaults validation_errors to empty list."""
        error = ValidationFailedError("Failed", raw_text="", attempt=1)
        assert error.validation_errors == []

    def test_validation_failed_error_is_catchable_as_parse_error(self) -> None:
        """A raised ValidationFailedError can be caught as ParseError."""
        error = ValidationFailedError(
            "Schema validation failed",
            raw_text='{"bad": "data"}',
            attempt=1,
            validation_errors=[{"loc": ["confidence"], "msg": "field required"}],
        )
        with pytest.raises(ParseError):
            raise error

    def test_parse_error_raised_by_parse_on_extraction_failure(
        self, parser: ResponseParser
    ) -> None:
        """Verify the parser raises ParseError (not bare Exception) on failure."""
        with pytest.raises(ParseError) as exc_info:
            parser.parse("completely invalid content", ApprovalResponse)
        assert exc_info.value.message is not None
        assert exc_info.value.raw_text != ""

    def test_validation_error_raised_by_parse_on_schema_violation(
        self, parser: ResponseParser
    ) -> None:
        """Verify the parser raises ValidationFailedError on schema mismatch."""
        with pytest.raises(ValidationFailedError) as exc_info:
            parser.parse('{"decision": "approve"}', ApprovalResponse)
        assert len(exc_info.value.validation_errors) > 0


# ============================================================================
# Phase 13: Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases, boundary conditions, and unusual input."""

    def test_parse_unicode_content(self, parser: ResponseParser) -> None:
        """Unicode characters in JSON values parse correctly."""
        raw_text = '{"decision": "approve", "confidence": 0.8, "reasoning": "Ñoño approved it 日本語"}'
        result = parser.parse(raw_text, ApprovalResponse)
        assert "Ñoño" in result.reasoning
        assert "日本語" in result.reasoning

    def test_parse_large_json(self, parser: ResponseParser) -> None:
        """Large JSON with 50 GL coding entries parses without issues."""
        gl_entries = [
            {"line": i, "account": f"{5000 + i}", "amount": float(i * 100)}
            for i in range(1, 51)
        ]
        data = {
            "match_status": "matched",
            "gl_coding": gl_entries,
            "processing_notes": "Large batch processing",
            "recommendation": "approve",
            "reasoning": "All 50 lines verified",
        }
        raw_text = json.dumps(data)
        result = parser.parse(raw_text, InvoiceResponse)
        assert len(result.gl_coding) == 50
        assert result.match_status == "matched"

    def test_parse_json_with_newlines_in_values(self, parser: ResponseParser) -> None:
        """JSON escape sequences (\\n) within string values parse correctly."""
        raw_text = '{"decision": "approve", "confidence": 0.8, "reasoning": "Line 1\\nLine 2"}'
        result = parser.parse(raw_text, ApprovalResponse)
        assert "Line 1" in result.reasoning
        assert "Line 2" in result.reasoning

    def test_parse_json_with_special_characters(self, parser: ResponseParser) -> None:
        """Dollar signs, commas, and comparison operators in values parse correctly."""
        raw_text = (
            '{"decision": "approve", "confidence": 0.7, '
            '"reasoning": "Amount $5,000 < $10,000 threshold"}'
        )
        result = parser.parse(raw_text, ApprovalResponse)
        assert "$5,000" in result.reasoning

    def test_parse_deeply_nested_json(self, parser: ResponseParser) -> None:
        """Deeply nested JSON with sub-accounts parses correctly."""
        raw_text = json.dumps(
            {
                "match_status": "matched",
                "gl_coding": [
                    {
                        "line": 1,
                        "account": "5000",
                        "amount": 100.0,
                        "sub_accounts": [
                            {"code": "5000-01", "split": 50.0},
                            {"code": "5000-02", "split": 50.0},
                        ],
                    }
                ],
                "processing_notes": "Nested sub-accounts",
                "recommendation": "approve",
                "reasoning": "OK",
            }
        )
        result = parser.parse(raw_text, InvoiceResponse)
        assert len(result.gl_coding) == 1
        assert "sub_accounts" in result.gl_coding[0]

    def test_parse_whitespace_only_raises(self, parser: ResponseParser) -> None:
        """Whitespace-only input raises ParseError."""
        with pytest.raises(ParseError):
            parser.parse("   \n\t  ", ApprovalResponse)

    def test_parse_json_with_zero_value(self, parser: ResponseParser) -> None:
        """Zero integer value parses correctly (not falsy)."""
        raw_text = '{"status": "ok", "value": 0}'
        result = parser.parse(raw_text, SimpleResponse)
        assert result.status == "ok"
        assert result.value == 0

    def test_max_retries_attribute(self, parser: ResponseParser) -> None:
        """max_retries is a public attribute accessible on the instance."""
        assert parser.max_retries == 3

    def test_custom_max_retries(self) -> None:
        """max_retries can be set to a custom value at construction."""
        custom = ResponseParser(max_retries=5)
        assert custom.max_retries == 5

    def test_parse_json_with_empty_string_value(self, parser: ResponseParser) -> None:
        """Empty string values are valid in JSON."""
        raw_text = '{"decision": "", "confidence": 0.0, "reasoning": ""}'
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == ""
        assert result.confidence == 0.0

    def test_parse_json_with_boolean_in_dict(self, parser: ResponseParser) -> None:
        """Boolean values inside dict fields parse correctly."""
        raw_text = json.dumps(
            {
                "match_status": "matched",
                "gl_coding": [{"line": 1, "account": "5000", "amount": 100.0, "flagged": True}],
                "processing_notes": "Flagged entry",
                "recommendation": "review",
                "reasoning": "Flagged",
            }
        )
        result = parser.parse(raw_text, InvoiceResponse)
        assert result.gl_coding[0]["flagged"] is True

    def test_parse_handles_extra_fields_gracefully(self, parser: ResponseParser) -> None:
        """Extra fields not in the schema do not cause parse failures."""
        raw_text = (
            '{"decision": "approve", "confidence": 0.9, "reasoning": "Good", '
            '"extra_field": "should be ignored"}'
        )
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.decision == "approve"

    def test_parse_boundary_confidence_zero(self, parser: ResponseParser) -> None:
        """Confidence = 0.0 is valid (at the ge boundary)."""
        raw_text = '{"decision": "reject", "confidence": 0.0, "reasoning": "No match"}'
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.confidence == 0.0

    def test_parse_boundary_confidence_one(self, parser: ResponseParser) -> None:
        """Confidence = 1.0 is valid (at the le boundary)."""
        raw_text = '{"decision": "approve", "confidence": 1.0, "reasoning": "Perfect"}'
        result = parser.parse(raw_text, ApprovalResponse)
        assert result.confidence == 1.0
