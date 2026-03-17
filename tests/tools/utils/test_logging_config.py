from __future__ import annotations

import logging
from pathlib import Path

from loguru import logger

from dlkit.tools.utils.logging_config import (
    configure_logging,
    get_effective_log_level,
    suppress_third_party_loggers,
    should_enable_progress_bar,
)


def test_get_effective_log_level_reads_env(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DLKIT_LOG_LEVEL", "warning")
    assert get_effective_log_level(level=None, debug_enabled=False) == "WARNING"


def test_cli_debug_override_forces_debug(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DLKIT_LOG_LEVEL", "ERROR")
    assert get_effective_log_level(level=None, debug_enabled=True) == "DEBUG"


def test_should_enable_progress_bar_depends_on_effective_level() -> None:
    assert should_enable_progress_bar(level="INFO", debug_enabled=False) is True
    assert should_enable_progress_bar(level="DEBUG", debug_enabled=False) is True
    assert should_enable_progress_bar(level="WARNING", debug_enabled=False) is False
    assert should_enable_progress_bar(level="ERROR", debug_enabled=False) is False


def test_configure_logging_uses_env_log_file_and_effective_level(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    log_file = tmp_path / "dlkit-info.log"
    monkeypatch.setenv("DLKIT_LOG_LEVEL", "INFO")
    monkeypatch.setenv("DLKIT_LOG_FILE", str(log_file))

    configure_logging(level=None, debug_enabled=False, format_type="simple")

    test_logger = logger.bind(module="dlkit.tests.logging")
    test_logger.info("info message written to file")
    test_logger.debug("debug message hidden from info file")

    contents = log_file.read_text()
    assert "info message written to file" in contents
    assert "debug message hidden from info file" not in contents


def test_configure_logging_does_not_render_bound_extra_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    log_file = tmp_path / "dlkit-clean.log"
    monkeypatch.setenv("DLKIT_LOG_LEVEL", "INFO")
    monkeypatch.setenv("DLKIT_LOG_FILE", str(log_file))

    configure_logging(level=None, debug_enabled=False, format_type="structured")

    test_logger = logger.bind(module="dlkit.tests.logging", component="orchestrator")
    test_logger.info("clean output only")

    contents = log_file.read_text()
    assert "clean output only" in contents
    assert "component" not in contents
    assert "module" not in contents


def test_configure_logging_cli_level_overrides_env_for_file_sink(
    monkeypatch,
    tmp_path: Path,
) -> None:  # noqa: ANN001
    log_file = tmp_path / "dlkit-warning.log"
    monkeypatch.setenv("DLKIT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DLKIT_LOG_FILE", str(log_file))

    configure_logging(level="WARNING", debug_enabled=False, format_type="simple")

    test_logger = logger.bind(module="dlkit.tests.logging")
    test_logger.info("info should not be written")
    test_logger.warning("warning should be written")

    contents = log_file.read_text()
    assert "warning should be written" in contents
    assert "info should not be written" not in contents


def test_suppress_third_party_loggers_restores_original_levels() -> None:
    logger_a = logging.getLogger("alembic.runtime.migration")
    logger_b = logging.getLogger("mlflow.store.db.utils")
    unrelated = logging.getLogger("dlkit.tests.unrelated")

    original_a = logging.ERROR
    original_b = logging.NOTSET
    original_unrelated = logging.INFO
    logger_a.setLevel(original_a)
    logger_b.setLevel(original_b)
    unrelated.setLevel(original_unrelated)

    with suppress_third_party_loggers(("alembic.runtime.migration", "mlflow.store.db.utils")):
        assert logger_a.level == logging.WARNING
        assert logger_b.level == logging.WARNING
        assert unrelated.level == original_unrelated

    assert logger_a.level == original_a
    assert logger_b.level == original_b
    assert unrelated.level == original_unrelated
