"""Component settings test fixtures."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def model_component_data() -> dict[str, Any]:
    """Sample dataflow for ModelComponentSettings testing.

    Returns:
        Dict[str, Any]: Model component configuration
    """
    return {
        "name": "TestModel",
        "module_path": "dlkit.domain.nn.ffnn",
        "heads": 8,
        "num_layers": 6,
        "latent_size": 256,
        "kernel_size": 3,
        "in_channels": 3,
        "out_channels": 10,
        "num_heads": 4,  # Different from heads for testing
    }


@pytest.fixture
def model_component_with_checkpoint_data(tmp_path) -> dict[str, Any]:
    """Sample dataflow for ModelComponentSettings with checkpoint.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Dict[str, Any]: Model component configuration with checkpoint
    """
    # Create a temporary checkpoint file for validation
    checkpoint_file = tmp_path / "model.ckpt"
    checkpoint_file.write_text("fake checkpoint")

    return {
        "name": "CheckpointModel",
        "module_path": "dlkit.domain.nn.ffnn",
        "checkpoint": str(checkpoint_file),
        "latent_size": 128,
    }


@pytest.fixture
def hyperparameter_model_data() -> dict[str, Any]:
    """Sample dataflow for ModelComponentSettings with hyperparameters.

    Returns:
        Dict[str, Any]: Model component configuration with hyperparameters
    """
    return {
        "name": "HyperModel",
        "module_path": "dlkit.domain.nn.ffnn",
        "heads": {"low": 4, "high": 16, "step": 4},
        "latent_size": {"low": 64, "high": 512, "step": 64},  # Use low/high/step instead of choices
        "num_layers": {"low": 2, "high": 8, "step": 1},
    }


@pytest.fixture
def metric_component_data() -> dict[str, Any]:
    """Sample dataflow for MetricComponentSettings testing.

    Returns:
        Dict[str, Any]: Metric component configuration
    """
    return {
        "name": "Accuracy",
        "module_path": "torchmetrics.classification",
        "task": "multiclass",
        "num_classes": 10,
    }


@pytest.fixture
def loss_component_data() -> dict[str, Any]:
    """Sample dataflow for LossComponentSettings testing.

    Returns:
        Dict[str, Any]: Loss component configuration
    """
    return {
        "name": "CrossEntropyLoss",
        "module_path": "torch.nn",
        "weight": None,
        "ignore_index": -100,
    }


@pytest.fixture
def wrapper_component_data() -> dict[str, Any]:
    """Sample dataflow for WrapperComponentSettings testing.

    Returns:
        Dict[str, Any]: Wrapper component configuration
    """
    return {
        "name": "StandardWrapper",
        "module_path": "test.wrappers",
        "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 0.01},
        "scheduler": {"name": "StepLR", "step_size": 10, "gamma": 0.9},
        "train": True,
        "test": True,
        "predict": False,
        "loss_function": {"name": "mse_loss", "module_path": "torch.nn.functional"},
        "metrics": [
            {"name": "MeanSquaredError", "module_path": "torchmetrics.regression"},
            {"name": "R2Score", "module_path": "torchmetrics.regression"},
        ],
        "is_autoencoder": False,
    }


@pytest.fixture
def complex_wrapper_data() -> dict[str, Any]:
    """Complex wrapper configuration for advanced testing.

    Returns:
        Dict[str, Any]: Complex wrapper configuration
    """
    return {
        "name": "ComplexWrapper",
        "module_path": "test.wrappers.complex",
        "optimizer": {
            "name": "AdamW",
            "lr": 0.001,  # Use plain value instead of hyperparameter dict
            "weight_decay": 0.01,  # Use plain value instead of hyperparameter dict
            "betas": (0.9, 0.999),
        },
        "scheduler": {"name": "CosineAnnealingLR", "T_max": 100, "eta_min": 1e-6},
        "loss_function": {
            "name": "FocalLoss",
            "module_path": "dlkit.domain.losses",
            "alpha": 0.25,
            "gamma": 2.0,
        },
        "metrics": [
            {
                "name": "F1Score",
                "module_path": "torchmetrics.classification",
                "task": "multiclass",
                "num_classes": 5,
            }
        ],
        "is_autoencoder": True,
    }
