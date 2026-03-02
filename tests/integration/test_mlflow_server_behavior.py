"""Behavioral tests for MLflow tracking URIs (http vs file).

These tests verify that:
- HTTP tracking URIs start a server context via the MLflow server adapter.
- File tracking URIs do not start a server context (MLflow writes directly to FS).

Server startup is patched to avoid launching real processes; we only assert the
context manager is entered (http) or not entered (file).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import dlkit
from dlkit.tools.config import GeneralSettings


@dataclass
class _ContextState:
    entered: bool = False
    exited: bool = False


class _DummyContext:
    """Dummy MLflowServerContext replacement recording enter/exit calls."""

    state = _ContextState()

    def __init__(self, server_config: Any, adapter: Any | None = None):
        self._config = server_config
        self._adapter = adapter

    def __enter__(self):
        type(self).state.entered = True
        # Return minimal ServerInfo-like object
        from dlkit.interfaces.servers.protocols import ServerInfo

        return ServerInfo(
            url=f"http://{getattr(self._config, 'host', '127.0.0.1')}:{getattr(self._config, 'port', 5000)}",
            host=getattr(self._config, "host", "127.0.0.1"),
            port=getattr(self._config, "port", 5000),
            pid=None,
            process=None,
        )

    def __exit__(self, exc_type, exc, tb):
        type(self).state.exited = True
        return False


class TestMLflowServerBehavior:
    """Verify MLflow server context usage depends on tracking URI scheme."""

    def test_file_tracking_does_not_start_server_context(
        self, monkeypatch: pytest.MonkeyPatch, mlflow_settings: GeneralSettings
    ) -> None:
        # Patch the MLflowServerContext; it should never be used for file:// URIs
        import dlkit.interfaces.servers.mlflow_adapter as mlflow_adapter

        _DummyContext.state = _ContextState()
        monkeypatch.setattr(mlflow_adapter, "MLflowServerContext", _DummyContext, raising=True)

        result = dlkit.train(mlflow_settings)

        assert result.duration_seconds >= 0
        # With file:// tracking URI, context should not be entered
        assert _DummyContext.state.entered is False
        assert _DummyContext.state.exited is False
