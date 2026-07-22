"""Unit tests for `evaluate._resolve_checkpoint_path`.

Pure guard-clause / resolution logic tests: no real MLflow backend, no real
training. `find_latest_run_id`/`download_checkpoint_artifact` are
monkeypatched at their point of use inside `dlkit.interfaces.inference.evaluate`
(not at their definition module), so these tests only verify
`_resolve_checkpoint_path`'s own branching and forwarding — the real,
network-touching behavior of those two functions is covered by
`tests/engine/tracking/test_run_queries.py` and
`tests/engine/tracking/test_checkpoint_recovery.py`.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from dlkit.common import ConfigurationError
from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint
from dlkit.infrastructure.config.job_config import InferenceJobConfig
from dlkit.interfaces.inference.evaluate import _resolve_checkpoint_path

# `dlkit.interfaces.inference`'s __init__ re-exports the `evaluate` *function*
# under the `evaluate` attribute, shadowing the submodule on the package
# object — `import dlkit.interfaces.inference.evaluate as ...` would bind
# that function, not the module. `import_module` looks the submodule up in
# `sys.modules` directly, sidestepping the shadowing, so monkeypatching
# module-level names (`find_latest_run_id`, `download_checkpoint_artifact`)
# at their point of use inside the module actually works.
evaluate_module = import_module("dlkit.interfaces.inference.evaluate")


@pytest.fixture
def inference_settings() -> InferenceJobConfig:
    """Minimal InferenceJobConfig with an experiment name, no tracking URI."""
    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "unit-test-experiment"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
                # Placeholder: InferenceJobConfig requires a non-None checkpoint
                # at construction time even though run_checkpoint resolution
                # overrides it before it is ever read.
                "checkpoint": "placeholder.ckpt",
            },
        }
    )


@pytest.fixture
def inference_settings_without_experiment() -> InferenceJobConfig:
    """InferenceJobConfig with no experiment configured (exercises the fallback)."""
    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
                "checkpoint": "placeholder.ckpt",
            },
        }
    )


@pytest.fixture
def downloaded_checkpoint_path(tmp_path: Path) -> Path:
    """Stand-in path returned by a monkeypatched `download_checkpoint_artifact`."""
    return tmp_path / "downloaded.ckpt"


def test_rejects_checkpoint_path_and_run_checkpoint_together(
    inference_settings: InferenceJobConfig,
) -> None:
    with pytest.raises(ConfigurationError, match="not both"):
        _resolve_checkpoint_path(
            checkpoint_path="explicit.ckpt",
            run_checkpoint=RunCheckpoint(run_id="run-1"),
            settings=inference_settings,
        )


def test_passes_checkpoint_path_through_unchanged_when_run_checkpoint_is_none(
    inference_settings: InferenceJobConfig,
) -> None:
    result = _resolve_checkpoint_path(
        checkpoint_path="explicit.ckpt",
        run_checkpoint=None,
        settings=inference_settings,
    )

    assert result == "explicit.ckpt"


def test_returns_none_when_neither_source_is_set(
    inference_settings: InferenceJobConfig,
) -> None:
    result = _resolve_checkpoint_path(
        checkpoint_path=None,
        run_checkpoint=None,
        settings=inference_settings,
    )

    assert result is None


def test_run_checkpoint_forwards_run_id_to_download_checkpoint_artifact(
    inference_settings: InferenceJobConfig,
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

    result = _resolve_checkpoint_path(
        checkpoint_path=None,
        run_checkpoint=RunCheckpoint(run_id="run-42"),
        settings=inference_settings,
    )

    assert captured["run_id"] == "run-42"
    assert result == downloaded_checkpoint_path


def test_latest_run_checkpoint_resolves_via_find_latest_run_id_using_settings_experiment_name(
    inference_settings: InferenceJobConfig,
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

    result = _resolve_checkpoint_path(
        checkpoint_path=None,
        run_checkpoint=LatestRunCheckpoint(),
        settings=inference_settings,
    )

    assert captured["experiment_name"] == "unit-test-experiment"
    assert captured["download_run_id"] == "resolved-run-id"
    assert result == downloaded_checkpoint_path


def test_latest_run_checkpoint_explicit_experiment_name_overrides_settings(
    inference_settings: InferenceJobConfig,
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

    _resolve_checkpoint_path(
        checkpoint_path=None,
        run_checkpoint=LatestRunCheckpoint(experiment_name="explicit-experiment"),
        settings=inference_settings,
    )

    assert captured["experiment_name"] == "explicit-experiment"


def test_latest_run_checkpoint_falls_back_to_dlkit_evaluate_when_no_experiment_configured(
    inference_settings_without_experiment: InferenceJobConfig,
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

    _resolve_checkpoint_path(
        checkpoint_path=None,
        run_checkpoint=LatestRunCheckpoint(),
        settings=inference_settings_without_experiment,
    )

    assert captured["experiment_name"] == "dlkit-evaluate"
