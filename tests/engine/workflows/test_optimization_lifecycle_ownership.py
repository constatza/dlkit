"""Tests for optimization lifecycle ownership boundaries."""

from __future__ import annotations

import ast
import inspect
from textwrap import dedent
from types import SimpleNamespace
from typing import cast

from dlkit.engine.workflows.entrypoints.optimization import optimize
from dlkit.engine.workflows.optimization.factory import OptimizationServiceFactory
from dlkit.engine.workflows.optimization.infrastructure import (
    InMemoryStudyRepository,
    NullOptimizationBackendSession,
    OptunaOptimizationBackendSession,
    OptunaStudyRepository,
)
from dlkit.engine.workflows.optimization.services import OptimizationOrchestrator
from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig


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


def test_runtime_entrypoint_owns_tracker_but_not_backend_session_context() -> None:
    source = inspect.getsource(optimize)
    context_targets = _with_context_targets(source)

    assert any("experiment_tracker" in target for target in context_targets), (
        "The runtime optimization entrypoint should continue to own experiment "
        "tracker setup and cleanup."
    )
    assert not any("backend_session" in target for target in context_targets), (
        "The runtime optimization entrypoint should not enter backend sessions; "
        "that lifecycle belongs to the optimization orchestrator."
    )


def test_factory_only_wires_optuna_infrastructure_when_enabled() -> None:
    factory = OptimizationServiceFactory()

    disabled_settings = cast(
        OptimizationWorkflowConfig,
        SimpleNamespace(OPTUNA=SimpleNamespace(enabled=False)),
    )
    enabled_settings = cast(
        OptimizationWorkflowConfig,
        SimpleNamespace(OPTUNA=SimpleNamespace(enabled=True)),
    )

    disabled_repository = factory.create_study_repository(disabled_settings)
    enabled_repository = factory.create_study_repository(enabled_settings)

    assert isinstance(disabled_repository, InMemoryStudyRepository)
    assert isinstance(enabled_repository, OptunaStudyRepository)

    disabled_session = factory.create_optimization_backend_session(
        disabled_settings,
        disabled_repository,
    )
    enabled_session = factory.create_optimization_backend_session(
        enabled_settings,
        enabled_repository,
    )

    assert isinstance(disabled_session, NullOptimizationBackendSession)
    assert isinstance(enabled_session, OptunaOptimizationBackendSession)
