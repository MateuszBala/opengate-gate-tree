"""Convenience access to the package logger."""

import logging

from opengate_gate_tree.logging_setup import LOGGER_NAME


def log() -> logging.Logger:
    """Return the package logger configured under the package logger name."""
    return logging.getLogger(LOGGER_NAME)
