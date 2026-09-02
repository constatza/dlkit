"""Every generated config template must actually validate against real schema.

Takes each `build_*_template_dict()` output, substitutes real file paths for
the illustrative placeholders, and validates it against the matching
`JobConfig` subtype. This is the concrete evidence that template generation
produces working configs, not just TOML that renders without exception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.infrastructure.config._template_helpers import (
    build_convergence_template_dict,
    build_fit_template_dict,
    build_inference_template_dict,
    build_mlflow_template_dict,
    build_multirun_template_dict,
    build_search_template_dict,
    build_training_template_dict,
    get_template_dict,
)
from dlkit.infrastructure.config.job_config import (
    ConvergenceJobConfig,
    FitJobConfig,
    InferenceJobConfig,
    MultiRunJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

_REAL_MODEL = {"class": "FFNN", "module_path": "dlkit.domain.nn", "hidden_size": 8, "num_layers": 1}


@pytest.fixture
def dataset_paths(tmp_path: Path) -> dict[str, Path]:
    """Minimal real feature/target files for schema round-trip validation."""
    features = tmp_path / "features.npy"
    targets = tmp_path / "targets.npy"
    np.save(features, np.random.randn(10, 3).astype("float32"))
    np.save(targets, np.random.randn(10, 1).astype("float32"))
    return {"features": features, "targets": targets}


@pytest.fixture
def model_checkpoint(tmp_path: Path) -> Path:
    """Minimal checkpoint file - just needs to exist for path validation."""
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {}}, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def deeponet_dataset_paths(tmp_path: Path) -> dict[str, Path]:
    """Minimal real branch/trunk/target files matching FFNNDeepONet's InputSpec."""
    rng = np.random.default_rng(0)
    n = 20
    branch = tmp_path / "branch.npy"
    trunk = tmp_path / "trunk.npy"
    y = tmp_path / "y.npy"
    np.save(branch, rng.normal(size=(n, 10)).astype("float32"))
    np.save(trunk, rng.uniform(0, 1, size=(n, 2)).astype("float32"))
    np.save(y, rng.normal(size=(n, 1)).astype("float32"))
    return {"branch": branch, "trunk": trunk, "y": y}


def _fill_data_placeholders(
    template: dict, *, features: Path, targets: Path | None = None, checkpoint: Path | None = None
) -> dict:
    """Substitute real file paths for a template's illustrative placeholders."""
    template["model"] = {**_REAL_MODEL, **({"checkpoint": str(checkpoint)} if checkpoint else {})}
    template["data"].pop("root", None)
    template["data"]["features"][0]["path"] = str(features)
    if targets is not None and "targets" in template["data"]:
        template["data"]["targets"][0]["path"] = str(targets)
    return template


def test_training_template_round_trip(dataset_paths: dict[str, Path]) -> None:
    template = _fill_data_placeholders(build_training_template_dict(), **dataset_paths)
    TrainingJobConfig.model_validate(template)


def test_inference_template_round_trip(
    dataset_paths: dict[str, Path], model_checkpoint: Path
) -> None:
    template = _fill_data_placeholders(
        build_inference_template_dict(),
        features=dataset_paths["features"],
        checkpoint=model_checkpoint,
    )
    InferenceJobConfig.model_validate(template)


def test_mlflow_template_round_trip(dataset_paths: dict[str, Path]) -> None:
    template = _fill_data_placeholders(build_mlflow_template_dict(), **dataset_paths)
    TrainingJobConfig.model_validate(template)


def test_search_template_round_trip(dataset_paths: dict[str, Path]) -> None:
    template = _fill_data_placeholders(build_search_template_dict(), **dataset_paths)
    SearchJobConfig.model_validate(template)


def test_fit_template_round_trip(dataset_paths: dict[str, Path]) -> None:
    template = _fill_data_placeholders(build_fit_template_dict(), **dataset_paths)
    FitJobConfig.model_validate(template)


def test_convergence_template_round_trip(dataset_paths: dict[str, Path]) -> None:
    template = _fill_data_placeholders(build_convergence_template_dict(), **dataset_paths)
    ConvergenceJobConfig.model_validate(template)


def test_multirun_template_round_trip() -> None:
    MultiRunJobConfig.model_validate(build_multirun_template_dict())


def test_deeponet_model_introspection_round_trip(deeponet_dataset_paths: dict[str, Path]) -> None:
    """The generic --model mechanism, exercised against a real model+data shapes."""
    branch_path = deeponet_dataset_paths["branch"]
    trunk_path = deeponet_dataset_paths["trunk"]
    y_path = deeponet_dataset_paths["y"]

    template = get_template_dict("training", model="FFNNDeepONet", module_path="dlkit.domain.nn")
    template["data"].pop("root", None)
    template["model"].update({"basis_dim": 4, "branch_hidden_size": 8, "trunk_hidden_size": 8})
    for feature in template["data"]["features"]:
        if feature["name"] == "branch":
            feature["path"] = str(branch_path)
        elif feature["name"] == "trunk":
            feature["path"] = str(trunk_path)
            feature["transforms"] = [{"name": "Unsqueeze", "dim": 1}]
    template["data"]["targets"][0]["path"] = str(y_path)
    template["data"]["targets"][0]["transforms"] = [{"name": "Unsqueeze", "dim": 1}]

    TrainingJobConfig.model_validate(template)
