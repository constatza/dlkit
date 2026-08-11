"""Tests for optimization lifecycle ownership boundaries."""

from __future__ import annotations

import ast
import inspect
from textwrap import dedent

from dlkit.engine.workflows.entrypoints.optimization import optimize
from dlkit.engine.workflows.optimization.factory import OptimizationServiceFactory
from dlkit.engine.workflows.optimization.infrastructure import (
    InMemoryStudyRepository,
    NullOptimizationBackendSession,
    OptunaOptimizationBackendSession,
    OptunaStudyRepository,
)
from dlkit.engine.workflows.optimization.services import OptimizationOrchestrator
from dlkit.infrastructure.config.job_config import SearchJobConfig


def _parse_source(source: str) -> ast.AST:
    return ast.parse(dedent(source))


def _attribute_chain(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _attribute_chain(node.func)
    return ""


def _with_context_targets(source: str) -> list[str]:
    tree = _parse_source(source)
    contexts: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                contexts.append(_attribute_chain(item.context_expr))

    return contexts


def _call_targets(source: str) -> list[str]:
    tree = _parse_source(source)
    return [_attribute_chain(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]


def test_optimization_orchestrator_accepts_backend_session_dependency() -> None:
    signature = inspect.signature(OptimizationOrchestrator.__init__)

    assert "optimization_backend_session" in signature.parameters, (
        "OptimizationOrchestrator should accept an IOptimizationBackendSession "
        "dependency so backend lifecycle ownership stays at the orchestrator."
    )


def test_optimization_orchestrator_enters_backend_session_context() -> None:
    source = inspect.getsource(OptimizationOrchestrator.execute_optimization)
    context_targets = _with_context_targets(source)

    assert any("optimization_backend_session" in target for target in context_targets), (
        "OptimizationOrchestrator.execute_optimization should enter the backend "
        "session context directly."
    )


def test_optimization_orchestrator_no_longer_reaches_into_optuna_repository() -> None:
    source = inspect.getsource(OptimizationOrchestrator)

    assert "get_optuna_study" not in source, (
        "OptimizationOrchestrator should use IOptimizationBackendSession instead of "
        "reaching through IStudyRepository for Optuna-specific access."
    )


def test_backend_session_does_not_reach_through_repository_private_study_access() -> None:
    source = inspect.getsource(OptunaOptimizationBackendSession)

    assert "_require_optuna_study" not in source, (
        "OptunaOptimizationBackendSession should resolve backend studies through "
        "its dedicated infrastructure collaborator, not repository private state."
    )


def test_factory_creates_backend_session_for_orchestrator() -> None:
    source = inspect.getsource(OptimizationServiceFactory.create_optimization_orchestrator)
    call_targets = _call_targets(source)

    assert any(target.endswith("create_optimization_backend_session") for target in call_targets), (
        "OptimizationServiceFactory.create_optimization_orchestrator should build "
        "the backend session dependency explicitly."
    )


def test_factory_does_not_enter_backend_session_context() -> None:
    create_orchestrator_source = inspect.getsource(
        OptimizationServiceFactory.create_optimization_orchestrator
    )
    create_backend_session_source = inspect.getsource(
        OptimizationServiceFactory.create_optimization_backend_session
    )

    assert not any(
        "optimization_backend_session" in target
        for target in _with_context_targets(create_orchestrator_source)
    ), (
        "Factory orchestration should assemble dependencies without entering "
        "backend session contexts."
    )
    assert not any(
        "optimization_backend_session" in target
        for target in _with_context_targets(create_backend_session_source)
    ), "Factory backend-session creation should return an unentered context manager."


def test_runtime_entrypoint_does_not_own_tracker_or_backend_session_context() -> None:
    source = inspect.getsource(optimize)
    context_targets = _with_context_targets(source)

    assert not context_targets, (
        "The runtime optimization entrypoint should not enter any context "
        "manager itself — tracker and backend-session lifecycles both belong "
        "to the optimization orchestrator now, mirroring "
        f"MultiRunOrchestrator.run_sweep. Found with-targets: {context_targets!r}"
    )


def test_factory_orchestrator_forwards_hooks_to_study_tracker() -> None:
    source = inspect.getsource(OptimizationServiceFactory.create_optimization_orchestrator)
    tree = _parse_source(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _attribute_chain(node.func).endswith(
            "create_study_tracker"
        ):
            keyword_names = {kw.arg for kw in node.keywords}
            assert "hooks" in keyword_names, (
                "create_optimization_orchestrator must forward self._hooks to "
                "create_study_tracker; otherwise LifecycleHooks passed to "
                "OptimizationServiceFactory(hooks=...) silently never reach the "
                "tracker the orchestrator builds internally."
            )
            return

    raise AssertionError(
        "create_study_tracker call not found in create_optimization_orchestrator source"
    )


def test_factory_only_wires_optuna_infrastructure_when_enabled() -> None:
    factory = OptimizationServiceFactory()

    enabled_settings = SearchJobConfig.model_validate(
        {
            "run": {"type": "search", "seed": 42},
            "experiment": {"name": "test-search"},
            "model": {"class": "DummyModel", "module_path": "dlkit.domain.nn"},
            "data": {
                "batch_size": 8,
                "num_workers": 0,
            },
            "training": {
                "loss": "mse",
                "trainer": {"max_epochs": 1, "accelerator": "cpu"},
                "optimizer": {"name": "AdamW", "lr": 1e-3},
            },
            "search": {
                "space": {
                    "model.hidden_size": {
                        "type": "categorical",
                        "choices": [2, 4],
                    }
                }
            },
        }
    )

    enabled_repository = factory.create_study_repository(enabled_settings)
    disabled_repository = InMemoryStudyRepository()

    assert isinstance(enabled_repository, OptunaStudyRepository)

    disabled_session = factory.create_optimization_backend_session(
        enabled_settings,
        disabled_repository,
    )
    enabled_session = factory.create_optimization_backend_session(
        enabled_settings,
        enabled_repository,
    )

    assert isinstance(disabled_session, NullOptimizationBackendSession)
    assert isinstance(enabled_session, OptunaOptimizationBackendSession)
