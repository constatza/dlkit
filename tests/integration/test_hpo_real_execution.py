"""Real end-to-end HPO integration test.

No autouse stub — TrialExecutor.execute_trial runs for real.
Verifies that a full Optuna study completes at least one successful trial
and that the best trial carries a float objective value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.interfaces.api import optimize as api_optimize

_FEATURE_SIZE: int = 4
_TARGET_SIZE: int = 2
_NUM_SAMPLES: int = 20
_BATCH_SIZE: int = 4


@pytest.fixture
def hpo_dataset(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny supervised dataset for real HPO trial execution.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Dictionary with ``features`` and ``targets`` npy paths.
    """
    np.random.seed(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    features_path = data_dir / "features.npy"
    targets_path = data_dir / "targets.npy"
    np.save(features_path, np.random.randn(_NUM_SAMPLES, _FEATURE_SIZE).astype(np.float32))
    np.save(targets_path, np.random.randn(_NUM_SAMPLES, _TARGET_SIZE).astype(np.float32))
    return {"features": features_path, "targets": targets_path}


@pytest.fixture
def real_hpo_config(hpo_dataset: dict[str, Path]) -> SearchJobConfig:
    """Build a SearchJobConfig that runs real trial execution.

    Uses in-memory Optuna storage (storage=None), fast_dev_run=True,
    cpu-only accelerator, and n_trials=2 so the test finishes in seconds.

    Args:
        hpo_dataset: Fixture providing tiny dataset paths.

    Returns:
        SearchJobConfig ready for real optimize() execution.
    """
    return SearchJobConfig.model_validate(
        {
            "run": {"type": "search", "seed": 0},
            "model": {
                "class": "FFNN",
                "module_path": "dlkit.domain.nn",
                "hidden_size": 4,
                "num_layers": 0,
            },
            "data": {
                "class": "FlexibleDataset",
                "batch_size": _BATCH_SIZE,
                "num_workers": 0,
                "shuffle": True,
                "pin_memory": False,
                "persistent_workers": False,
                "features": [{"name": "x", "path": str(hpo_dataset["features"]), "format": "npy"}],
                "targets": [{"name": "y", "path": str(hpo_dataset["targets"]), "format": "npy"}],
            },
            "training": {
                "loss": "mse",
                "trainer": {
                    "fast_dev_run": True,
                    "enable_checkpointing": False,
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                },
                "optimizer": {"name": "AdamW", "lr": 1e-3},
                "metrics": [{"name": "MeanSquaredError", "module_path": "dlkit.domain.metrics"}],
            },
            "search": {
                "n_trials": 2,
                "direction": "minimize",
                "study_name": "real_hpo_test",
                "storage": None,
                "space": {
                    "model.hidden_size": {"type": "categorical", "choices": [2, 4]},
                },
            },
        }
    )


@pytest.mark.timeout(60)
def test_real_hpo_completes_with_successful_trial(real_hpo_config: SearchJobConfig) -> None:
    """Real HPO run should complete and yield at least one successful trial.

    Verifies:
    - optimize() returns a result without raising.
    - At least one trial succeeded.
    - best_trial.value is a float.

    Args:
        real_hpo_config: Fixture providing a fully-wired SearchJobConfig.
    """
    result = api_optimize(real_hpo_config)

    assert result is not None
    assert result.best_trial is not None
    assert isinstance(result.best_trial.value, float), (
        f"Expected best_trial.value to be float; got {type(result.best_trial.value)}"
    )
