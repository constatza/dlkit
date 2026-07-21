"""Regression tests for build_inference_datamodule() (DataModuleSelector resolution)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from lightning.pytorch import LightningDataModule

from dlkit.common.errors import ConfigurationError
from dlkit.engine.workflows.factories.inference_data_factory import build_inference_datamodule
from dlkit.infrastructure.config.job_config import InferenceJobConfig

DATASET_NUM_SAMPLES: int = 20


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
    np.save(features, np.random.randn(DATASET_NUM_SAMPLES, 4).astype(np.float32))
    np.save(targets, np.random.randn(DATASET_NUM_SAMPLES, 1).astype(np.float32))
    return {"features": features, "targets": targets}


def _make_explicit_split_file(tmp_path: Path) -> Path:
    """Write a valid split file covering DATASET_NUM_SAMPLES indices.

    Evaluation-mode split resolution requires a resolvable split file (either
    explicit or colocated with the checkpoint) — this fixture supplies the
    explicit path so build_inference_datamodule() can reload it.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to the written split JSON file.
    """
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps(
            {
                "train": list(range(14)),
                "validation": list(range(14, 17)),
                "test": list(range(17, 20)),
            }
        )
    )
    return split_file


def _make_inference_job_config(
    dataset_paths: dict[str, Path],
    *,
    split_filepath: Path,
    module_override: dict[str, str] | None = None,
) -> InferenceJobConfig:
    """Build a minimal InferenceJobConfig with a real data section.

    Args:
        dataset_paths: Feature/target file paths, e.g. from _make_minimal_dataset.
        split_filepath: Path to an explicit split file covering the dataset.
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
        "splits": {"filepath": str(split_filepath)},
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
    split_filepath = _make_explicit_split_file(tmp_path)
    settings = _make_inference_job_config(dataset_paths, split_filepath=split_filepath)

    datamodule = build_inference_datamodule(settings)

    assert isinstance(datamodule, LightningDataModule)


def test_build_inference_datamodule_raises_when_no_split_resolvable(tmp_path: Path) -> None:
    """No explicit filepath and no colocated splits/ dir must fail loudly.

    This is the deliberate breaking-change guardrail from the split-resolution
    redesign: evaluation must never silently regenerate an unrelated random
    split when none can be resolved.
    """
    dataset_paths = _make_minimal_dataset(tmp_path)
    checkpoint_path = tmp_path / "run" / "checkpoints" / "best.ckpt"
    data: dict[str, object] = {
        "batch_size": 4,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "persistent_workers": False,
        "features": [{"name": "x", "path": str(dataset_paths["features"])}],
        "targets": [{"name": "y", "path": str(dataset_paths["targets"])}],
    }
    settings = InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "inference_data_factory_test"},
            "model": {"class": "FFNN", "checkpoint": str(checkpoint_path)},
            "data": data,
        }
    )

    with pytest.raises(ConfigurationError, match="Set data.splits.filepath explicitly"):
        build_inference_datamodule(settings)


def test_build_inference_datamodule_raises_when_ratios_overridden(tmp_path: Path) -> None:
    """test_ratio/val_ratio overrides are dead config in evaluation mode.

    The split is always reloaded verbatim from a file, so a ratio override
    would otherwise be silently ignored — this must fail loud instead.
    """
    dataset_paths = _make_minimal_dataset(tmp_path)
    split_filepath = _make_explicit_split_file(tmp_path)
    data: dict[str, object] = {
        "batch_size": 4,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "persistent_workers": False,
        "features": [{"name": "x", "path": str(dataset_paths["features"])}],
        "targets": [{"name": "y", "path": str(dataset_paths["targets"])}],
        "splits": {"filepath": str(split_filepath), "test": 0.3},
    }
    settings = InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "inference_data_factory_test"},
            "model": {"class": "FFNN", "checkpoint": "unused.ckpt"},
            "data": data,
        }
    )

    with pytest.raises(ConfigurationError, match="ignored during evaluation"):
        build_inference_datamodule(settings)


def test_build_inference_datamodule_resolves_split_via_checkpoint_override(
    tmp_path: Path,
) -> None:
    """A checkpoint override that differs from ``settings.model.checkpoint``
    must win for split resolution too, matching its established precedence
    for weight loading.

    Regression test for a real bug found during review: ``InferenceJobConfig``
    always requires ``model.checkpoint`` to be set (see
    ``JobConfig._checkpoint_required``), but the CLI ``evaluate``/``predict``
    commands and the ``evaluate()`` API accept a separate ``checkpoint``/
    ``checkpoint_path`` override that is documented to take precedence when
    both are supplied ("the CLI argument will take precedence when
    provided" — see ``interfaces/cli/commands/predict.py``'s module
    docstring). Before this fix, split auto-colocation only ever consulted
    ``settings.model.checkpoint`` and ignored the override entirely, so
    evaluating against a *different* checkpoint than the one named in the
    config would silently resolve the wrong run's (or no) split.
    """
    dataset_paths = _make_minimal_dataset(tmp_path)
    run_root = tmp_path / "run"
    checkpoint_path = run_root / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"unused")
    splits_dir = run_root / "splits"
    splits_dir.mkdir()
    split_file = splits_dir / "session_split.json"
    split_file.write_text(
        json.dumps(
            {
                "train": list(range(14)),
                "validation": list(range(14, 17)),
                "test": list(range(17, 20)),
            }
        )
    )
    data: dict[str, object] = {
        "batch_size": 4,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
        "persistent_workers": False,
        "features": [{"name": "x", "path": str(dataset_paths["features"])}],
        "targets": [{"name": "y", "path": str(dataset_paths["targets"])}],
    }
    settings = InferenceJobConfig.model_validate(
        {
            "run": {"type": "predict"},
            "experiment": {"name": "inference_data_factory_test"},
            # Stale/unrelated checkpoint value baked into the config — the
            # override below points at the run actually being evaluated.
            "model": {"class": "FFNN", "checkpoint": "some/other/unrelated.ckpt"},
            "data": data,
        }
    )

    datamodule = build_inference_datamodule(settings, checkpoint_override=checkpoint_path)

    assert isinstance(datamodule, LightningDataModule)


def test_build_inference_datamodule_resolves_explicit_selector(tmp_path: Path) -> None:
    """Explicit data.module override (class=ArrayDataModule) must also build."""
    dataset_paths = _make_minimal_dataset(tmp_path)
    split_filepath = _make_explicit_split_file(tmp_path)
    settings = _make_inference_job_config(
        dataset_paths,
        split_filepath=split_filepath,
        module_override={
            "class": "ArrayDataModule",
            "module_path": "dlkit.engine.adapters.lightning.datamodules",
        },
    )

    datamodule = build_inference_datamodule(settings)

    assert isinstance(datamodule, LightningDataModule)
