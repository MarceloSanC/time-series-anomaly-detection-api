from __future__ import annotations

import json
import logging

import pytest

from app.utils.logging import JsonFormatter, RequestIdFilter, reset_request_id, set_request_id, setup_logging

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_record(
    message: str = "test message",
    name: str = "test.logger",
    level: int = logging.INFO,
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(name, level, __file__, 1, message, (), None)
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    RequestIdFilter().filter(record)  # inject request_id attribute
    return record


# ── existing behaviour (text mode) ────────────────────────────────────────────


def test_setup_logging_adds_handler_and_request_filter_when_root_has_no_handlers() -> None:
    """Create stream handler + RequestIdFilter when root logger starts empty."""
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    root_logger.handlers.clear()
    try:
        setup_logging("INFO")
        assert len(root_logger.handlers) == 1
        handler = root_logger.handlers[0]
        assert any(isinstance(filter_obj, RequestIdFilter) for filter_obj in handler.filters)
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)


def test_set_request_id_accepts_explicit_value_and_can_be_reset() -> None:
    """Set explicit request id and restore previous context value on reset."""
    token = set_request_id("req-123")
    try:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "msg", (), None)
        assert RequestIdFilter().filter(record) is True
        assert getattr(record, "request_id") == "req-123"
    finally:
        reset_request_id(token)


def test_setup_logging_text_mode_uses_text_formatter() -> None:
    """text mode installs a plain Formatter, not JsonFormatter."""
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    root_logger.handlers.clear()
    try:
        setup_logging("INFO", "text")
        assert len(root_logger.handlers) == 1
        handler = root_logger.handlers[0]
        assert not isinstance(handler.formatter, JsonFormatter)
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)


# ── JSON mode ─────────────────────────────────────────────────────────────────


def test_setup_logging_json_mode_installs_json_formatter() -> None:
    """json mode installs JsonFormatter on the root handler."""
    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    root_logger.handlers.clear()
    try:
        setup_logging("INFO", "json")
        assert len(root_logger.handlers) == 1
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)
    finally:
        root_logger.handlers.clear()
        root_logger.handlers.extend(previous_handlers)
        root_logger.setLevel(previous_level)


@pytest.mark.parametrize("required_key", ["timestamp", "level", "logger", "message", "request_id"])
def test_json_formatter_always_includes_required_fields(required_key: str) -> None:
    """JsonFormatter output always contains the five mandatory top-level keys."""
    record = _make_record()
    output = JsonFormatter().format(record)
    payload = json.loads(output)
    assert required_key in payload


def test_json_formatter_output_is_valid_json() -> None:
    """JsonFormatter produces a single parseable JSON object."""
    record = _make_record(message="hello world")
    output = JsonFormatter().format(record)
    payload = json.loads(output)
    assert isinstance(payload, dict)


def test_json_formatter_message_matches_log_record_message() -> None:
    """JsonFormatter 'message' field equals the formatted log message."""
    record = _make_record(message="something happened")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "something happened"


def test_json_formatter_level_matches_log_record_level() -> None:
    """JsonFormatter 'level' field reflects the record's level name."""
    record = _make_record(level=logging.WARNING)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "WARNING"


def test_json_formatter_includes_request_id_from_context() -> None:
    """JsonFormatter captures request_id set via set_request_id()."""
    token = set_request_id("test-req-abc")
    try:
        record = _make_record()
        payload = json.loads(JsonFormatter().format(record))
        assert payload["request_id"] == "test-req-abc"
    finally:
        reset_request_id(token)


def test_json_formatter_merges_extra_fields() -> None:
    """Extra fields passed to logger.info(..., extra={}) appear in JSON output."""
    record = _make_record(extra={"event": "model_trained", "series_id": "sensor_X", "n_samples": 100})
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "model_trained"
    assert payload["series_id"] == "sensor_X"
    assert payload["n_samples"] == 100


def test_json_formatter_does_not_leak_stdlib_internals() -> None:
    """Standard LogRecord internals (levelno, lineno, etc.) are not in JSON output."""
    record = _make_record()
    payload = json.loads(JsonFormatter().format(record))
    for key in ("levelno", "lineno", "pathname", "thread", "processName"):
        assert key not in payload
