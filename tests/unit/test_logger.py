"""Unit tests for logger convenience module."""

import logging

from opengate_gate_tree.logger import log
from opengate_gate_tree.logging_setup import LOGGER_NAME


def test_log_returns_package_logger() -> None:
    """log should return the logger configured for the package name."""
    logger = log()
    assert isinstance(logger, logging.Logger)
    assert logger.name == LOGGER_NAME


def test_log_returns_same_logger_instance() -> None:
    """Repeated log calls should return the same logger singleton by name."""
    assert log() is logging.getLogger(LOGGER_NAME)
