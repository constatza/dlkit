import pytest

from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.interfaces.api import execute, optimize, train


@pytest.fixture
def training_config():
    return TrainingWorkflowConfig.model_validate(
        {
            "SESSION": {"workflow": "train"},
            "TRAINING": {
                "epochs": 1,
                "trainer": {"max_epochs": 1},
                "optimizer": {"default_optimizer": {"name": "AdamW", "lr": 0.001}},
            },
            "DATAMODULE": {"dataloader": {"batch_size": 32}},
        }
    )


@pytest.fixture
def optimization_config():
    return OptimizationWorkflowConfig.model_validate(
        {
            "SESSION": {"workflow": "optimize"},
            "TRAINING": {
                "epochs": 1,
                "trainer": {"max_epochs": 1},
                "optimizer": {"default_optimizer": {"name": "AdamW", "lr": 0.001}},
            },
            "DATAMODULE": {"dataloader": {"batch_size": 32}},
            "OPTUNA": {"enabled": True, "storage": "sqlite:///test.db", "study_name": "test"},
        }
    )


def test_train_rejects_dict_overrides(training_config):
    with pytest.raises(
        TypeError, match="overrides must be provided as TrainingOverrides, got dict"
    ):
        train(training_config, overrides={"epochs": 10})


def test_optimize_rejects_dict_overrides(optimization_config):
    with pytest.raises(
        TypeError, match="overrides must be provided as OptimizationOverrides, got dict"
    ):
        optimize(optimization_config, overrides={"trials": 10})


def test_execute_rejects_dict_overrides(training_config):
    with pytest.raises(
        TypeError, match="overrides must be provided as ExecutionOverrides, got dict"
    ):
        execute(training_config, overrides={"epochs": 10})
