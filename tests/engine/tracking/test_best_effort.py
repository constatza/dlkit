"""Tests for the best_effort decorator — failure swallowing and retry scoping.

best_effort-wrapped calls must never raise into a training/optimization run,
and must run under a fail-fast MLflow retry budget rather than the wider one
sized for calls that must not silently fail (see best_effort_retry_budget in
dlkit.infrastructure.config.environment).
"""

from __future__ import annotations

import os

import pytest

from dlkit.engine.tracking.best_effort import best_effort


def test_exception_is_caught_and_logged_not_raised() -> None:
    """A failure inside the wrapped function must not propagate."""

    @best_effort("do a thing")
    def _always_fails() -> None:
        raise RuntimeError("boom")

    _always_fails()  # must not raise


def test_wrapped_call_runs_under_the_fail_fast_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside the wrapped function, the scoped fail-fast env vars are active."""
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "9")
    observed: dict[str, str] = {}

    @best_effort("observe env")
    def _observe() -> None:
        observed["max_retries"] = os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]

    _observe()

    assert observed["max_retries"] == "2"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"


@pytest.mark.timeout(5)
def test_nested_best_effort_calls_do_not_deadlock() -> None:
    """An outer best_effort function calling an inner one (the real shape of
    log_best_trial_result -> log_trial_artifacts) must not hang.
    """
    calls: list[str] = []

    @best_effort("inner")
    def _inner() -> None:
        calls.append("inner")

    @best_effort("outer")
    def _outer() -> None:
        calls.append("outer")
        _inner()

    _outer()

    assert calls == ["outer", "inner"]


def test_nested_best_effort_inner_failure_does_not_escape_outer() -> None:
    """The outer call still completes and swallows a failure raised by the inner one."""
    calls: list[str] = []

    @best_effort("inner")
    def _inner() -> None:
        raise RuntimeError("boom")

    @best_effort("outer")
    def _outer() -> None:
        _inner()
        calls.append("outer finished")

    _outer()  # must not raise

    assert calls == ["outer finished"]
