"""Tests for the optimize() entrypoint's shared ``EntrypointContext.run()`` plumbing.

``optimize()`` is one of five entrypoints (``fit``, ``training``,
``optimization``, ``convergence``, ``evaluate``) that route their workflow
execution through ``EntrypointContext.run()``: a raw exception from the
optimization strategy is wrapped as ``WorkflowError``, while a
``WorkflowError`` raised by the strategy itself propagates unchanged. These
tests stub ``OptimizationServiceFactory`` instead of wiring a real
Optuna/MLflow-backed optimization pipeline, mirroring ``test_fit.py``'s
pattern of standing in for the heavy collaborator.
"""

from __future__ import annotations

import pytest

from dlkit.common import OptimizationResult
from dlkit.common.errors import WorkflowError
from dlkit.engine.workflows.entrypoints import optimization as optimization_module
from dlkit.infrastructure.config.job_config import SearchJobConfig


class _StubOptimizationStrategy:
    """Stands in for OptimizationStrategy: returns a canned result or raises,
    instead of running a real Optuna study."""

    def __init__(self, result: OptimizationResult | None, exception: Exception | None) -> None:
        self._result = result
        self._exception = exception

    def execute_optimization(self, settings: SearchJobConfig) -> OptimizationResult:
        del settings
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result


class _StubOptimizationServiceFactory:
    """Stands in for OptimizationServiceFactory: skips real Optuna/MLflow
    service wiring, which is unrelated to what this change touches."""

    strategy: _StubOptimizationStrategy | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def create_study_tracker(self, settings: SearchJobConfig, hooks: object = None) -> None:
        del settings, hooks
        return None

    def create_optimization_strategy(self, settings: SearchJobConfig) -> _StubOptimizationStrategy:
        del settings
        assert _StubOptimizationServiceFactory.strategy is not None
        return _StubOptimizationServiceFactory.strategy


@pytest.fixture(autouse=True)
def stub_optimization_service_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        optimization_module, "OptimizationServiceFactory", _StubOptimizationServiceFactory
    )
    _StubOptimizationServiceFactory.strategy = None


def test_optimize_returns_result_built_from_strategy_execution(
    search_job_config: SearchJobConfig,
) -> None:
    _StubOptimizationServiceFactory.strategy = _StubOptimizationStrategy(
        result=OptimizationResult(
            best_trial=None, training_result=None, study_summary={"n": 1}, duration_seconds=2.0
        ),
        exception=None,
    )

    result = optimization_module.optimize(search_job_config)

    assert result.study_summary == {"n": 1}


def test_optimize_wraps_unexpected_exception_as_workflow_error(
    search_job_config: SearchJobConfig,
) -> None:
    _StubOptimizationServiceFactory.strategy = _StubOptimizationStrategy(
        result=None, exception=RuntimeError("boom")
    )

    with pytest.raises(WorkflowError, match="Optimization execution failed"):
        optimization_module.optimize(search_job_config)


def test_optimize_reraises_workflow_error_unchanged(search_job_config: SearchJobConfig) -> None:
    original = WorkflowError("strategy-specific failure", {"stage": "trial"})
    _StubOptimizationServiceFactory.strategy = _StubOptimizationStrategy(
        result=None, exception=original
    )

    with pytest.raises(WorkflowError) as exc_info:
        optimization_module.optimize(search_job_config)

    assert exc_info.value is original
