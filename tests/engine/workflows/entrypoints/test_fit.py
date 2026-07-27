"""Tests for the fit() entrypoint's per-call checkpoint isolation.

FitJobConfig has no ``training`` section, so unlike trainer-backed jobs it
has no ``training.trainer.default_root_dir`` knob to give concurrent runs
distinct output locations. ``FitOverrides.checkpoints_dir`` is the
alternative: it threads through ``path_override_context`` so callers running
many fit jobs from the same cwd (e.g. a batch of assignments) don't collide
on OneShotFitExecutor's fixed ``checkpoints/fitted.ckpt`` write path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common import TrainingResult
from dlkit.engine.workflows.entrypoints import fit as fit_module
from dlkit.engine.workflows.entrypoints._override_types import FitOverrides
from dlkit.infrastructure.config.job_config import FitJobConfig
from dlkit.infrastructure.io.path_context_state import get_current_path_context


class _PathContextCapturingOrchestrator:
    """Stands in for Orchestrator: records the active checkpoints_dir override
    instead of running a real build+execute pipeline, since that pipeline is
    unrelated to what this change touches."""

    captured_checkpoints_dir: Path | None = None

    def execute_training(self, settings: FitJobConfig, hooks: object = None) -> TrainingResult:
        context = get_current_path_context()
        _PathContextCapturingOrchestrator.captured_checkpoints_dir = (
            context.checkpoints_dir if context else None
        )
        return TrainingResult(
            model_state=None,
            metrics={},
            artifacts={},
            duration_seconds=0.0,
        )


@pytest.fixture
def fit_settings() -> FitJobConfig:
    return FitJobConfig.model_validate(
        {
            "run": {"type": "fit", "seed": 0},
            "model": {"class": "Dummy", "module_path": "dlkit.domain.nn"},
            "data": {"batch_size": 4, "num_workers": 0},
        }
    )


@pytest.fixture(autouse=True)
def stub_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fit_module, "Orchestrator", _PathContextCapturingOrchestrator)


def test_fit_isolates_checkpoint_path_per_call_via_checkpoints_dir_override(
    fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    run_a = tmp_path / "pod2g-cg1" / "checkpoints"
    run_b = tmp_path / "pod2g-cg10" / "checkpoints"

    fit_module.fit(fit_settings, FitOverrides(checkpoints_dir=run_a))
    assert _PathContextCapturingOrchestrator.captured_checkpoints_dir == run_a.resolve()

    fit_module.fit(fit_settings, FitOverrides(checkpoints_dir=run_b))
    assert _PathContextCapturingOrchestrator.captured_checkpoints_dir == run_b.resolve()


def test_fit_leaves_path_context_untouched_when_no_checkpoints_dir_override_given(
    fit_settings: FitJobConfig,
) -> None:
    fit_module.fit(fit_settings, None)

    assert _PathContextCapturingOrchestrator.captured_checkpoints_dir is None


def test_fit_restores_previous_path_context_after_call(
    fit_settings: FitJobConfig, tmp_path: Path
) -> None:
    fit_module.fit(fit_settings, FitOverrides(checkpoints_dir=tmp_path / "run"))

    assert get_current_path_context() is None
