from __future__ import annotations

import logging

from app.utils.logging import RequestIdFilter, reset_request_id, set_request_id, setup_logging


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
