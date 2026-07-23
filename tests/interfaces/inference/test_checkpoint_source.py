"""Unit tests for `evaluate._resolve_checkpoint_path`.

Pure guard-clause / resolution logic tests: no real MLflow backend, no real
training. `find_latest_run_id`/`download_checkpoint_artifact` are
monkeypatched at their point of use inside
`dlkit.engine.workflows.entrypoints.evaluate` (not at their definition
module), so these tests only verify `_resolve_checkpoint_path`'s own
branching and forwarding — the real, network-touching behavior of those two
functions is covered by `tests/engine/tracking/test_run_queries.py` and
`tests/engine/tracking/test_checkpoint_recovery.py`.

`checkpoint_path`/`run_checkpoint` are no longer two mutually exclusive
kwargs — `settings.model.checkpoint` is a single `Path | str |
CheckpointSource` field, so "conflicting checkpoint sources" is now
structurally unreachable rather than a runtime validation error (see
`tests/integration/test_evaluate_integration.py`'s
`test_evaluate_checkpoint_field_accepts_either_a_path_or_a_run_checkpoint`).
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint
from dlkit.infrastructure.config.job_config import InferenceJobConfig

# `dlkit.engine.workflows.entrypoints`'s __init__ re-exports the `evaluate`
# *function* under the `evaluate` attribute, shadowing the submodule on the
# package object — `import dlkit.engine.workflows.entrypoints.evaluate as
# ...` would bind that function, not the module. `import_module` looks the
# submodule up in `sys.modules` directly, sidestepping the shadowing, so
# monkeypatching module-level names (`find_latest_run_id`,
# `download_checkpoint_artifact`) at their point of use inside the module
# actually works.
evaluate_module = import_module("dlkit.engine.workflows.entrypoints.evaluate")
_resolve_checkpoint_path = evaluate_module._resolve_checkpoint_path


def _inference_settings(checkpoint: object, *, experiment_name: str | None) -> InferenceJobConfig:
    """Build a minimal InferenceJobConfig with the given checkpoint/experiment."""
    payload: dict[str, object] = {
        "run": {"type": "predict"},
        "model": {
            "class": "FFNN",
            "module_path": "dlkit.domain.nn",
            "checkpoint": checkpoint if isinstance(checkpoint, str) else "placeholder.ckpt",
        },
    }
    if experiment_name is not None:
        payload["experiment"] = {"name": experiment_name}
    settings = InferenceJobConfig.model_validate(payload)
    if not isinstance(checkpoint, str):
        settings = settings.patch({"model": {"checkpoint": checkpoint}})
    return settings


@pytest.fixture
def downloaded_checkpoint_path(tmp_path: Path) -> Path:
    """Stand-in path returned by a monkeypatched `download_checkpoint_artifact`."""
    return tmp_path / "downloaded.ckpt"


def test_passes_literal_checkpoint_path_through_unchanged() -> None:
    settings = _inference_settings("explicit.ckpt", experiment_name="unit-test-experiment")

    result = _resolve_checkpoint_path(settings)

    assert result == "explicit.ckpt"


def test_run_checkpoint_forwards_run_id_to_download_checkpoint_artifact(
    downloaded_checkpoint_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_download(run_id: str, destination: Path, *, tracking_uri: str | None = None) -> Path:
        captured["run_id"] = run_id
        captured["destination"] = destination
        captured["tracking_uri"] = tracking_uri
        return downloaded_checkpoint_path

    monkeypatch.setattr(evaluate_module, "download_checkpoint_artifact", fake_download)

    settings = _inference_settings(
        RunCheckpoint(run_id="run-42"), experiment_name="unit-test-experiment"
    )
    result = _resolve_checkpoint_path(settings)

    assert captured["run_id"] == "run-42"
    assert result == downloaded_checkpoint_path


def test_latest_run_checkpoint_resolves_via_find_latest_run_id_using_settings_experiment_name(
    downloaded_checkpoint_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_find_latest_run_id(*, experiment_name: str, tracking_uri: str | None = None) -> str:
        captured["experiment_name"] = experiment_name
        return "resolved-run-id"

    def fake_download(run_id: str, destination: Path, *, tracking_uri: str | None = None) -> Path:
        captured["download_run_id"] = run_id
        return downloaded_checkpoint_path

    monkeypatch.setattr(evaluate_module, "find_latest_run_id", fake_find_latest_run_id)
    monkeypatch.setattr(evaluate_module, "download_checkpoint_artifact", fake_download)

    settings = _inference_settings(LatestRunCheckpoint(), experiment_name="unit-test-experiment")
    result = _resolve_checkpoint_path(settings)

    assert captured["experiment_name"] == "unit-test-experiment"
    assert captured["download_run_id"] == "resolved-run-id"
    assert result == downloaded_checkpoint_path


def test_latest_run_checkpoint_explicit_experiment_name_overrides_settings(
    downloaded_checkpoint_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_find_latest_run_id(*, experiment_name: str, tracking_uri: str | None = None) -> str:
        captured["experiment_name"] = experiment_name
        return "resolved-run-id"

    monkeypatch.setattr(evaluate_module, "find_latest_run_id", fake_find_latest_run_id)
    monkeypatch.setattr(
        evaluate_module,
        "download_checkpoint_artifact",
        lambda *args, **kwargs: downloaded_checkpoint_path,
    )

    settings = _inference_settings(
        LatestRunCheckpoint(experiment_name="explicit-experiment"),
        experiment_name="unit-test-experiment",
    )
    _resolve_checkpoint_path(settings)

    assert captured["experiment_name"] == "explicit-experiment"


def test_latest_run_checkpoint_falls_back_to_dlkit_evaluate_when_no_experiment_configured(
    downloaded_checkpoint_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_find_latest_run_id(*, experiment_name: str, tracking_uri: str | None = None) -> str:
        captured["experiment_name"] = experiment_name
        return "resolved-run-id"

    monkeypatch.setattr(evaluate_module, "find_latest_run_id", fake_find_latest_run_id)
    monkeypatch.setattr(
        evaluate_module,
        "download_checkpoint_artifact",
        lambda *args, **kwargs: downloaded_checkpoint_path,
    )

    settings = _inference_settings(LatestRunCheckpoint(), experiment_name=None)
    _resolve_checkpoint_path(settings)

    assert captured["experiment_name"] == "dlkit-evaluate"
