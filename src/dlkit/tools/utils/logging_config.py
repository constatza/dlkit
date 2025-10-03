"""Centralized logging configuration for DLKit using loguru.

This module provides structured logging configuration with environment variable
control for debug logging and appropriate criticality levels for different components.
"""

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def _suppress_third_party_logging() -> None:
    """Suppress noisy third-party library logs using loguru interception.

    Intercepts standard library logging calls from third-party libraries
    (Alembic migrations, SQLAlchemy queries, Werkzeug HTTP logs) and filters
    them to WARNING level to reduce noise.
    """
    import logging

    # Intercept standard library logging and route through loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Get corresponding loguru level
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    # Set up interception for noisy third-party loggers
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for logger_name in ["alembic", "sqlalchemy", "werkzeug", "urllib3", "mlflow"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def configure_logging(
    *,
    level: str | None = None,
    debug_enabled: bool | None = None,
    format_type: str = "structured",
) -> None:
    """Configure loguru logger with appropriate levels and formatting.

    Args:
        level: Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        debug_enabled: Whether to enable debug logging (overrides env vars)
        format_type: Format type ('structured' for JSON-like, 'simple' for human-readable)
    """
    # Remove default handler
    logger.remove()

    # Determine log level
    if level is None:
        level = os.getenv("DLKIT_LOG_LEVEL", "INFO").upper()

    # Determine debug mode
    if debug_enabled is None:
        debug_enabled = _is_debug_enabled()

    # Override level if debug is enabled
    if debug_enabled and level == "INFO":
        level = "DEBUG"

    # Configure format based on type
    if format_type == "structured":
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level> | "
            "{extra}"
        )
    else:
        format_str = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
        filter=_debug_filter,
    )

    # Add file handler for errors if not in debug mode
    if not debug_enabled:
        default_log_file = _get_default_log_file_path()
        log_file = os.getenv("DLKIT_LOG_FILE", str(default_log_file))
        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message} | "
                "{extra}"
            ),
            level="WARNING",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    # Suppress noisy third-party library logs
    _suppress_third_party_logging()

    # Configure logger with minimal default context
    # Note: "service" tag removed as it's redundant in single-service applications
    logger.configure(extra={})


def _get_default_log_file_path() -> Path:
    """Get default log file path using environment settings.

    Returns:
        Path to default log file in DLKit internal directory
    """
    from dlkit.tools.config.environment import DLKitEnvironment

    env = DLKitEnvironment()
    return env.get_log_file_path()


def _is_debug_enabled() -> bool:
    """Check if debug logging is enabled via environment variables.

    Debug mode is enabled when:
    - DLKIT_LOG_LEVEL is set to "DEBUG"
    - Running in pytest (PYTEST_CURRENT_TEST is set)
    """
    # Auto-enable debug in tests
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True

    # Check explicit log level
    return os.getenv("DLKIT_LOG_LEVEL", "").upper() == "DEBUG"


def _debug_filter(record: Any) -> bool:
    """Filter debug messages based on debug mode and module origin."""
    # Always show non-debug messages
    if record["level"].name != "DEBUG":
        return True

    # Check if debug mode is enabled
    if not _is_debug_enabled():
        # If debug is not enabled, suppress all debug messages
        return False

    # In debug mode, only show debug messages from dlkit modules
    module_name = record.get("name", "")
    return module_name.startswith("dlkit") or module_name == "__main__"


def get_logger(name: str, component: str | None = None) -> Any:
    """Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)
        component: Optional component name for better log categorization

    Returns:
        Configured logger instance with appropriate bindings
    """
    bindings = {"module": name}
    if component:
        bindings["component"] = component
    return logger.bind(**bindings)


# Initialize logging on module import with appropriate defaults
# Always configure to ensure proper settings (loguru has default DEBUG handler)
_module_initialized = False
if not _module_initialized:
    configure_logging()
    _module_initialized = True
