"""Logging configuration for the tool.

The module configures the package logger (level, format, stream) so diagnostics
and user messages use the ``logging`` module instead of ``print``.

Public functions:

configure_logging(level: int = logging.INFO) -> None
    Configure the package logger (idempotent).
get_logger(name: str) -> logging.Logger
    Return a logger with the given name.
"""

import logging
from typing import Final

# Main package logger name.
LOGGER_NAME: Final[str] = "opengate_gate_tree"

# Log message format.
LOG_FORMAT: Final[str] = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the package logger.

    This function is idempotent: calling it repeatedly does not add duplicate
    handlers, it only updates the logging level.

    Parameters
    ----------
    level : int
        Logging level (for example ``logging.INFO``).
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__`` of the calling module).

    Returns
    -------
    logging.Logger
        Logger with the given name.
    """
    return logging.getLogger(name)
