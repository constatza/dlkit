"""Integration tests ensuring graph precision remains consistent across workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from fsspec.implementations.local import LocalFileSystem
from loguru import logger

from dlkit.common import TrainingResult
from dlkit.engine.workflows.factories.dataset_builder import DatasetBuilder
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.training_settings import TrainingSettings
from dlkit.infrastructure.precision import PrecisionStrategy
from dlkit.interfaces.api import train as api_train

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="MPS backend on macOS lacks float64 support",
)


class TestGraphPrecisionLRTuning:
    """Test GNN models with float64 precision and LR tuning."""

    @staticmethod
    def _build_graph_settings(
        graph_settings: TrainingJobConfig,
        *,
        precision: PrecisionStrategy = PrecisionStrategy.FULL_64,
        enable_lr_tuning: bool = True,
    ) -> TrainingJobConfig:
        """Create modified TrainingJobConfig for graph precision scenarios.

        Args:
            graph_settings: Base graph training configuration.
            precision: Floating-point precision strategy.
            enable_lr_tuning: Whether to enable LR tuning.

        Returns:
            Modified TrainingJobConfig with precision and optional LR tuning.
        """
        assert graph_settings.training is not None
        graph_model = graph_settings.model
        assert graph_model is not None

        updated_model = graph_model.model_copy(
            update={
                "name": "ScaledGATv2Projection",
                "module_path": "dlkit.domain.nn.graph.scaled_projection_networks",
                "params": graph_model.params.model_copy(
                    update={"hidden_size": 4, "num_layers": 1, "heads": 1}
                ),
            }
        )

        graph_training = graph_settings.training

        if enable_lr_tuning:
            updated_training = TrainingSettings(
                lr_tuner=LRTunerSettings(
                    min_lr=1e-6,
                    max_lr=0.1,
                    num_training=2,
                ),
                trainer=TrainerSettings.model_validate(
                    {
                        "fast_dev_run": False,
                        "enable_checkpointing": False,
                        "max_epochs": 1,
                        "limit_train_batches": 2,
                        "enable_progress_bar": False,
                    }
                ),
                metrics=graph_training.metrics,
                loss=graph_training.loss,
                optimizer=graph_training.optimizer,
            )
        else:
            updated_training = graph_training

        updated_run = graph_settings.run.model_copy(update={"precision": precision})

        return graph_settings.model_copy(
            update={
                "run": updated_run,
                "model": updated_model,
                "training": updated_training,
            }
        )

    def _load_dataset(self, graph_settings: TrainingJobConfig) -> Any:
        """Load graph dataset with the precision strategy from config.

        Args:
            graph_settings: Training job configuration with graph data settings.

        Returns:
            Loaded GraphDataset instance.
        """
        from dlkit.engine.data.datasets.graph import GraphDataset
        from dlkit.infrastructure.precision import precision_override

        data_cfg = graph_settings.data
        assert data_cfg is not None
        root = data_cfg.root
        assert root is not None

        # Extract graph-specific paths from features/targets entries
        x_path: Path | None = None
        edge_index_path: Path | None = None
        y_path: Path | None = None

        for entry in data_cfg.features:
            if entry.name == "x":
                x_path = getattr(entry, "path", None)
            elif entry.name == "edge_index":
                edge_index_path = getattr(entry, "path", None)
        for entry in data_cfg.targets:
            if entry.name == "y":
                y_path = getattr(entry, "path", None)

        assert x_path is not None
        assert edge_index_path is not None
        assert y_path is not None

        run_cfg = graph_settings.run
        precision: PrecisionStrategy | None = run_cfg.precision if run_cfg else None

        if precision is not None:
            with precision_override(precision):
                return GraphDataset(
                    root=root,
                    x=x_path,
                    edge_index=edge_index_path,
                    y=y_path,
                )
        return GraphDataset(
            root=root,
            x=x_path,
            edge_index=edge_index_path,
            y=y_path,
        )

    def test_graph_dataset_factory_uses_configured_root_on_windows(
        self,
        graph_settings: TrainingJobConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Graph dataset factory should not fall back to PyG's ``???`` root placeholder."""
        builder = DatasetBuilder()
        context = builder.build_context(graph_settings)
        created_paths: list[str] = []
        original_makedirs = LocalFileSystem.makedirs

        def _windows_guard(self, path: str, exist_ok: bool = False) -> None:
            created_paths.append(path)
            if "???" in path:
                raise OSError(
                    22,
                    "The filename, directory name, or volume label syntax is incorrect",
                    path,
                )
            original_makedirs(self, path, exist_ok=exist_ok)

        monkeypatch.setattr(LocalFileSystem, "makedirs", _windows_guard)

        dataset = builder.build_dataset_with_tensor_entries(graph_settings, context)
        data_cfg = graph_settings.data
        assert data_cfg is not None
        configured_root = data_cfg.root
        assert configured_root is not None
        dataset_root = getattr(dataset, "root", None)
        assert isinstance(dataset_root, str | Path)
        assert Path(dataset_root).resolve() == configured_root.resolve()
        assert created_paths
        assert all("???" not in path for path in created_paths)

    def test_graph_model_float64_lr_tuning_integration(
        self,
        graph_settings: TrainingJobConfig,
    ) -> None:
        """Graph + float64 + LR tuning should train without dtype drift."""
        modified_settings = self._build_graph_settings(graph_settings, enable_lr_tuning=True)
        result = api_train(modified_settings)
        assert isinstance(result, TrainingResult)

        dataset = self._load_dataset(modified_settings)
        sample = dataset[0]
        assert sample.x.dtype == torch.float64
        logger.info("Graph sample dtype after float64 run: {}", sample.x.dtype)

    def test_graph_model_float64_without_lr_tuning_baseline(
        self,
        graph_settings: TrainingJobConfig,
    ) -> None:
        """Baseline: Graph + float64 without LR tuning should still succeed."""
        modified_settings = self._build_graph_settings(
            graph_settings,
            enable_lr_tuning=False,
        )
        result = api_train(modified_settings)
        assert isinstance(result, TrainingResult)

        dataset = self._load_dataset(modified_settings)
        sample = dataset[0]
        assert sample.x.dtype == torch.float64
