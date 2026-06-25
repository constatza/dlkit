from __future__ import annotations

import pytest

from dlkit.infrastructure.config.job_config import (
    SearchJobConfig,
    TrainingJobConfig,
)
from dlkit.interfaces.api import execute, optimize, train


@pytest.fixture
def training_config() -> TrainingJobConfig:
    """Minimal valid TrainingJobConfig for override type rejection tests.

    Returns:
        TrainingJobConfig with minimal required sections.
    """
    return TrainingJobConfig.model_validate(
        {
            "run": {"type": "train"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
            },
            "data": {
                "class": "FlexibleDataset",
                "batch_size": 32,
                "features": [],
                "targets": [],
            },
            "training": {
                "trainer": {"max_epochs": 1},
                "optimizer": {"name": "AdamW", "lr": 0.001},
            },
        }
    )


@pytest.fixture
def optimization_config() -> SearchJobConfig:
    """Minimal valid SearchJobConfig for override type rejection tests.

    Returns:
        SearchJobConfig with minimal required sections including a search space.
    """
    return SearchJobConfig.model_validate(
        {
            "run": {"type": "search"},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
            },
            "data": {
                "class": "FlexibleDataset",
                "batch_size": 32,
                "features": [],
                "targets": [],
            },
            "training": {
                "trainer": {"max_epochs": 1},
                "optimizer": {"name": "AdamW", "lr": 0.001},
            },
            "search": {
                "n_trials": 2,
                "space": {
                    "model__params__hidden_size": {
                        "type": "categorical",
                        "choices": [2, 4],
                    }
                },
            },
        }
    )


def test_train_rejects_dict_overrides(training_config: TrainingJobConfig) -> None:
    """Passing a dict as overrides to train() must raise TypeError with clear message."""
    with pytest.raises(
        TypeError, match="overrides must be provided as TrainingOverrides, got dict"
    ):
        train(training_config, overrides={"epochs": 10})  # type: ignore[arg-type]


def test_optimize_rejects_dict_overrides(optimization_config: SearchJobConfig) -> None:
    """Passing a dict as overrides to optimize() must raise TypeError with clear message."""
    with pytest.raises(
        TypeError, match="overrides must be provided as OptimizationOverrides, got dict"
    ):
        optimize(optimization_config, overrides={"trials": 10})  # type: ignore[arg-type]


def test_execute_rejects_dict_overrides(training_config: TrainingJobConfig) -> None:
    """Passing a dict as overrides to execute() must raise TypeError with clear message."""
    with pytest.raises(
        TypeError, match="overrides must be provided as ExecutionOverrides, got dict"
    ):
        execute(training_config, overrides={"epochs": 10})  # type: ignore[arg-type]
