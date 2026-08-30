"""Unit tests for logging setup utilities."""

import logging

from opengate_gate_tree import logging_setup


def test_configure_logging_is_idempotent_and_updates_level() -> None:
    """Repeated calls should not duplicate handlers and should update level."""
    logger = logging.getLogger(logging_setup.LOGGER_NAME)
    original_handlers = list(logger.handlers)
    original_level = logger.level

    try:
        logger.handlers.clear()

        logging_setup.configure_logging(logging.INFO)
        assert len(logger.handlers) == 1
        assert logger.level == logging.INFO

        logging_setup.configure_logging(logging.DEBUG)
        assert len(logger.handlers) == 1
        assert logger.level == logging.DEBUG
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)


def test_get_logger_returns_logger_with_requested_name() -> None:
    """get_logger should return a logger for the provided name."""
    logger = logging_setup.get_logger("opengate_gate_tree.tests")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "opengate_gate_tree.tests"
