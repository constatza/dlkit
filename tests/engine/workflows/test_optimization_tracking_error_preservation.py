"""Regression test: OptimizationOrchestrator preserves a typed DLKitError.

`OptimizationOrchestrator.execute_optimization` used to hand-roll
`except Exception as e: raise WorkflowError(f"...: {e}", {...}) from e` with
no guard, so a `TrackingError` raised entering the study tracker (e.g. an
unreachable MLflow backend) got silently flattened into a generic
`WorkflowError` before it ever reached the CLI's per-type suggestion
dispatch. It now delegates to `raise_error`, which preserves any `DLKitError`
unchanged.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from dlkit.common.errors import TrackingError
from dlkit.engine.workflows.optimization.services import OptimizationOrchestrator, TrialExecutor
from dlkit.engine.workflows.optimization.value_objects import OptimizationDirection

from .test_hpo_correctness import _make_search_job, _make_study_manager, _RecordingBackendSession


def test_execute_optimization_preserves_tracking_error_unchanged() -> None:
    original = TrackingError("MLflow tracking backend unreachable")
    tracker = MagicMock()
    tracker.__enter__.side_effect = original
    orchestrator = OptimizationOrchestrator(
        study_manager=_make_study_manager(),
        trial_executor=cast(TrialExecutor, MagicMock()),
        optimization_backend_session=cast(Any, _RecordingBackendSession(sampled={})),
        study_tracker=tracker,
    )

    with pytest.raises(TrackingError) as exc_info:
        orchestrator.execute_optimization(
            study_name="unreachable-tracker",
            base_settings=_make_search_job(),
            n_trials=1,
            direction=OptimizationDirection.MINIMIZE,
        )

    assert exc_info.value is original
