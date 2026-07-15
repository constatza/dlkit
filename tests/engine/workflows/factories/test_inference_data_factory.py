"""Regression tests for build_inference_datamodule() (DataModuleSelector resolution)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lightning.pytorch import LightningDataModule

from dlkit.engine.workflows.factories.inference_data_factory import build_inference_datamodule
from dlkit.infrastructure.config.job_config import InferenceJobConfig


def _make_minimal_dataset(tmp_path: Path) -> dict[str, Path]:
    """Write tiny feature/target .npy files for a fast datamodule build.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Mapping of "features"/"targets" to their written .npy paths.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    features = data_dir / "features.npy"
    targets = data_dir / "targets.npy"
    np.save(features, np.random.randn(20, 4).astype(np.float32))
    np.save(targets, np.random.randn(20, 1).astype(np.float32))
    return {"features": features, "targets": targets}


def _make_inference_job_config(
    dataset_paths: dict[str, Path],
    *,
    module_override: dict[str, str] | None = None,
) -> InferenceJobConfig:
    """Build a minimal InferenceJobConfig with a real data section.

    Args:
        dataset_paths: Feature/target file paths, e.g. from _make_minimal_dataset.
        module_override: Optional explicit data.module override
            (e.g. {"class": "ArrayDataModule"}); omit to exercise the default
            DataModuleSelector.

    Returns:
        A validated InferenceJobConfig.
    """
    data: dict[str, object] = {
        "batch_size": 4,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "persistent_workers": False,
        "features": [{"name": "x", "path": str(dataset_paths["features"])}],
        "targets": [{"name": "y", "path": str(dataset_paths["targets"])}],
    }
    if module_override is not None:
        data["module"] = module_override

    return InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "inference_data_factory_test"},
            "model": {"class": "FFNN", "checkpoint": "unused.ckpt"},
            "data": data,
        }
    )


def test_build_inference_datamodule_resolves_default_selector(tmp_path: Path) -> None:
    """Default DataModuleSelector must build without error.

    Regression test: previously raised AttributeError because DataModuleSelector
    (a plain settings selector, not a ComponentSettings) was routed through
    FactoryProvider.create_component(), which requires get_init_kwargs().
    """
    dataset_paths = _make_minimal_dataset(tmp_path)
    settings = _make_inference_job_config(dataset_paths)

    datamodule = build_inference_datamodule(settings)

    assert isinstance(datamodule, LightningDataModule)


def test_build_inference_datamodule_resolves_explicit_selector(tmp_path: Path) -> None:
    """Explicit data.module override (class=ArrayDataModule) must also build."""
    dataset_paths = _make_minimal_dataset(tmp_path)
    settings = _make_inference_job_config(
        dataset_paths,
        module_override={
            "class": "ArrayDataModule",
            "module_path": "dlkit.engine.adapters.lightning.datamodules",
        },
    )

    datamodule = build_inference_datamodule(settings)

    assert isinstance(datamodule, LightningDataModule)
