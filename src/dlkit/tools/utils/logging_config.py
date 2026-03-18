"""Centralized logging configuration for DLKit using loguru only."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

_VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_ENV_ENABLED = "1"
_ENV_DISABLED = "0"
_LOG_ROTATION_SIZE = "10 MB"
_LOG_RETENTION_PERIOD = "7 days"
_LOG_COMPRESSION_FORMAT = "gz"


def _backtrace_enabled() -> bool:
    return os.getenv("DLKIT_LOG_BACKTRACE", _ENV_DISABLED) == _ENV_ENABLED


def _diagnose_enabled() -> bool:
    return os.getenv("DLKIT_LOG_DIAGNOSE", _ENV_DISABLED) == _ENV_ENABLED
_CURRENT_LOG_LEVEL = "INFO"
_MLFLOW_SQLITE_BOOTSTRAP_LOGGERS = (
    "alembic.runtime.migration",
    "alembic.runtime.plugins",
    "mlflow.store.db.utils",
)


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
    global _CURRENT_LOG_LEVEL

    logger.remove()
    resolved_level = get_effective_log_level(level=level, debug_enabled=debug_enabled)
    _CURRENT_LOG_LEVEL = resolved_level

    # Configure format based on type
    if format_type == "structured":
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
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
        level=resolved_level,
        colorize=True,
        backtrace=_backtrace_enabled(),
        diagnose=_diagnose_enabled(),
        filter=_debug_filter,
    )

    from dlkit.tools.config.environment import DLKitEnvironment

    env = DLKitEnvironment()
    default_log_file = _get_default_log_file_path()
    log_file = env.log_file or str(default_log_file)
    logger.add(
        log_file,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        level=resolved_level,
        rotation=_LOG_ROTATION_SIZE,
        retention=_LOG_RETENTION_PERIOD,
        compression=_LOG_COMPRESSION_FORMAT,
        colorize=False,
        backtrace=_backtrace_enabled(),
        diagnose=_diagnose_enabled(),
    )

    logger.configure(extra={})


def normalize_log_level(level: str | None) -> str:
    """Normalize a log level string to a valid Loguru level name."""
    if level is None:
        from dlkit.tools.config.environment import DLKitEnvironment

        candidate = DLKitEnvironment().log_level
    else:
        candidate = level

    normalized = str(candidate).strip().upper()
    if normalized not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level '{candidate}'. Expected one of: {', '.join(_VALID_LEVELS)}."
        )
    return normalized


def get_effective_log_level(
    *,
    level: str | None = None,
    debug_enabled: bool | None = None,
) -> str:
    """Resolve the effective log level after env and debug overrides."""
    resolved_level = normalize_log_level(level)
    debug_mode = debug_enabled if debug_enabled is not None else _is_debug_enabled(resolved_level)
    return "DEBUG" if debug_mode else resolved_level


def _get_default_log_file_path() -> Path:
    """Get default log file path using environment settings.

    Returns:
        Path to default log file in DLKit internal directory
    """
    from dlkit.tools.config.environment import DLKitEnvironment

    env = DLKitEnvironment()
    return env.get_log_file_path()


def _is_debug_enabled(resolved_level: str | None = None) -> bool:
    """Check if debug logging is enabled via effective level or test context."""
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True

    return normalize_log_level(resolved_level) == "DEBUG"


def should_enable_progress_bar(
    *,
    level: str | None = None,
    debug_enabled: bool | None = None,
) -> bool:
    """Return whether end-user training progress should be visible."""
    return get_effective_log_level(level=level, debug_enabled=debug_enabled) in {"DEBUG", "INFO"}


@contextmanager
def suppress_third_party_loggers(
    logger_names: tuple[str, ...],
    *,
    level: int = logging.WARNING,
):
    """Temporarily raise third-party stdlib logger levels.

    This is intentionally narrow and exists only for muting third-party
    bootstrap chatter that does not respect DLKit's Loguru configuration.
    """
    original_levels: list[tuple[logging.Logger, int]] = []
    try:
        for logger_name in logger_names:
            target_logger = logging.getLogger(logger_name)
            original_levels.append((target_logger, target_logger.level))
            target_logger.setLevel(level)
        yield
    finally:
        for target_logger, original_level in reversed(original_levels):
            target_logger.setLevel(original_level)


@contextmanager
def suppress_mlflow_sqlite_bootstrap_logs():
    """Suppress Alembic/MLflow DB bootstrap chatter for local SQLite setup."""
    with suppress_third_party_loggers(_MLFLOW_SQLITE_BOOTSTRAP_LOGGERS):
        yield


def _debug_filter(record: Any) -> bool:
    """Filter debug messages based on debug mode and module origin."""
    if record["level"].name != "DEBUG":
        return True

    if _CURRENT_LOG_LEVEL != "DEBUG":
        return False

    module_name = record["extra"].get("module") or record.get("name", "")
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
