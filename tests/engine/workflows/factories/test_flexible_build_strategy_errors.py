from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dlkit.engine.workflows.factories.flexible_build_strategy import FlexibleBuildStrategy
from dlkit.infrastructure.config.data_entries import ValueEntry


class FailingDataset:
    def __getitem__(self, idx):
        raise RuntimeError("Simulated network share failure")


def test_flexible_build_strategy_propagates_dataset_loading_errors():
    """Verify that dataset loading errors are not swallowed during contract inference."""

    # Mock settings required to reach the dataset[0] call
    settings = SimpleNamespace(
        SESSION=SimpleNamespace(seed=42, workflow="train"),
        MODEL=SimpleNamespace(
            name="VarWidthDeepONet", module_path="dlkit.domain.nn", model_dump=lambda: {}
        ),
        TRAINING=SimpleNamespace(
            optimizer={"name": "Adam"}, loss_function={"name": "mse"}, metrics=[], scheduler=None
        ),
        DATASET=SimpleNamespace(
            type="flexible",
            features=[ValueEntry(name="feature1", value=torch.tensor([1.0]))],
            targets=[ValueEntry(name="target1", value=torch.tensor([1.0]))],
        ),
        DATAMODULE=SimpleNamespace(name="InMemoryModule"),
    )

    strategy = FlexibleBuildStrategy()

    # Mock the datamodule building so it doesn't crash before reaching contract inference
    strategy._dataset_builder = Mock()
    strategy._dataset_builder.build_context.return_value = {}
    strategy._dataset_builder.build_flexible_dataset.return_value = FailingDataset()

    import dlkit.engine.workflows.factories.flexible_build_strategy as fbs

    original_build_datamodule = fbs._build_datamodule
    fbs._build_datamodule = Mock(return_value=(Mock(), Mock()))

    try:
        with pytest.raises(RuntimeError, match="Simulated network share failure"):
            strategy._build_core(settings)
    finally:
        fbs._build_datamodule = original_build_datamodule
