"""Regression tests for MLflow HTTP retry-budget resilience.

Reproduces the reported "too many 500 error responses" failure that aborted
optimization runs: with too tight a retry budget, a short burst of transient
500s from a lightly-loaded local ``mlflow server`` exhausts the client's
retries and kills the whole run instead of recovering.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import threading
import time
from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds, http_request

import dlkit.infrastructure.config.environment  # noqa: F401  # ensures env defaults are set
from dlkit.engine.tracking.backend import LocalServerBackend
from dlkit.engine.tracking.mlflow_client_factory import MLflowClientFactory
from dlkit.engine.tracking.mlflow_resource_manager import MLflowResourceManager
from dlkit.infrastructure.config.environment import best_effort_retry_budget
from dlkit.infrastructure.config.tracking_settings import TrackingSettings

# Chosen so it sits strictly between the pre-fix attempt budget (1 initial +
# 2 retries = 3 attempts) and the post-fix attempt budget (1 initial + 5
# retries = 6 attempts): too many transient 500s for the old budget to
# survive, comfortably within the new one.
_TRANSIENT_FAILURE_BURST = 4


@contextlib.contextmanager
def _flaky_server(fail_count: int) -> Iterator[str]:
    """Serve `fail_count` transient 500s, then 200s, on 127.0.0.1."""
    request_count = 0

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal request_count
            request_count += 1
            self.send_response(500 if request_count <= fail_count else 200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _get_env_default(url: str) -> None:
    """Issue a request with no explicit retry kwargs, so MLflow resolves
    max_retries/backoff_factor/timeout from the current env vars — this is
    what production best_effort-wrapped calls do."""
    http_request(MlflowHostCreds(host=url), "/ping", "GET")


def _get(url: str, max_retries: int) -> None:
    http_request(
        MlflowHostCreds(host=url),
        "/ping",
        "GET",
        max_retries=max_retries,
        backoff_factor=0.01,  # tiny so the test is fast; independent of retry *count*
        timeout=5,
    )


def test_configured_retry_budget_survives_transient_500_burst() -> None:
    """The process-wide MLFLOW_HTTP_REQUEST_MAX_RETRIES default must survive a
    short burst of transient 500s — this is the actual reported failure mode.
    """
    max_retries = int(os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"])
    with _flaky_server(fail_count=_TRANSIENT_FAILURE_BURST) as url:
        _get(url, max_retries=max_retries)  # must not raise


def test_tight_retry_budget_of_two_still_exhausts_on_the_same_burst() -> None:
    """A budget of 2 retries genuinely cannot survive this burst — proves the
    harness discriminates rather than always passing, and documents why 2 was
    the wrong choice regardless of what the process default is set to later.
    """
    with _flaky_server(fail_count=_TRANSIENT_FAILURE_BURST) as url:
        with pytest.raises(MlflowException, match="too many 500 error responses"):
            _get(url, max_retries=2)


def test_tracking_settings_max_retries_is_wired_through_to_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TrackingSettings.max_retries must reach MLFLOW_HTTP_REQUEST_MAX_RETRIES,
    giving users a real per-run override instead of only a pre-import env var.
    """
    # set_mlflow_max_retries() writes directly to os.environ (not via monkeypatch),
    # so register the current value for restore or it leaks into later tests.
    monkeypatch.setenv(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES", os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
    )

    settings = TrackingSettings(backend="mlflow", max_retries=9)
    manager = MLflowResourceManager(settings, LocalServerBackend())

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: Mock())
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _client: True)

    with manager:
        pass

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"


def test_explicit_max_retries_override_survives_a_more_hostile_500_burst(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit TrackingSettings.max_retries override must survive a burst
    longer than the process-wide default alone could.
    """
    # set_mlflow_max_retries() writes directly to os.environ (not via monkeypatch),
    # so register the current value for restore or it leaks into later tests.
    monkeypatch.setenv(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES", os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
    )

    settings = TrackingSettings(backend="mlflow", max_retries=9)
    manager = MLflowResourceManager(settings, LocalServerBackend())

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: Mock())
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _client: True)

    with manager:
        max_retries = int(os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"])
        assert max_retries == 9
        with _flaky_server(fail_count=7) as url:
            _get(url, max_retries=max_retries)  # must not raise


def test_best_effort_retry_budget_fails_fast_against_persistent_500s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistently-failing endpoint (not a recoverable burst) must be given
    up on quickly under the scoped best-effort budget, not after the ~62s of
    backoff sleep the raised process-wide budget would otherwise burn.
    """
    monkeypatch.setenv(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES", os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
    )
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"])
    monkeypatch.setenv(
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"]
    )

    with _flaky_server(fail_count=10_000) as url:
        start = time.monotonic()
        with pytest.raises(MlflowException, match="too many 500 error responses"):
            with best_effort_retry_budget():
                _get_env_default(url)
        elapsed = time.monotonic() - start

    assert elapsed < 15


def test_best_effort_retry_budget_restores_configured_max_retries_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TrackingSettings.max_retries override (not just the process default)
    must survive being temporarily shadowed by a best-effort call.
    """
    monkeypatch.setenv(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES", os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
    )

    settings = TrackingSettings(backend="mlflow", max_retries=9)
    manager = MLflowResourceManager(settings, LocalServerBackend())

    monkeypatch.setattr(MLflowClientFactory, "create_client", lambda **_kwargs: Mock())
    monkeypatch.setattr(MLflowClientFactory, "validate_client_connectivity", lambda _client: True)

    with manager:
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"
        with best_effort_retry_budget():
            assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"
        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"
