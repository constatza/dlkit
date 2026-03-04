"""Integration tests ensuring graph precision remains consistent across workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger

import dlkit
from dlkit.interfaces.api.domain import TrainingResult
from dlkit.interfaces.api.overrides.manager import BasicOverrideManager
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.lr_tuner_settings import LRTunerSettings


pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="MPS backend on macOS lacks float64 support",
)


class TestGraphPrecisionLRTuning:
    """Test GNN models with float64 precision and LR tuning."""

    @staticmethod
    def _build_graph_settings(
        graph_settings: GeneralSettings,
        *,
        precision: str = "float64",
        enable_lr_tuning: bool = True,
    ) -> GeneralSettings:
        """Create modified GeneralSettings for graph precision scenarios."""
        from dlkit.tools.config.components.model_components import ModelComponentSettings
        from dlkit.tools.config.session_settings import SessionSettings
        from dlkit.tools.config.trainer_settings import TrainerSettings
        from dlkit.tools.config.training_settings import TrainingSettings

        session = SessionSettings(seed=42, precision=precision)

        model = ModelComponentSettings(
            name="ScaledGATv2Projection",
            module_path="dlkit.core.models.nn.graph.scaled_projection_networks",
            hidden_size=4,
            num_layers=1,
            heads=1,
            unified_shape=graph_settings.MODEL.unified_shape,
        )

        if enable_lr_tuning:
            training = TrainingSettings(
                epochs=2,
                lr_tuner=LRTunerSettings(
                    min_lr=1e-6,
                    max_lr=0.1,
                    num_training=10,
                ),
                trainer=TrainerSettings(
                    fast_dev_run=False,
                    enable_checkpointing=True,
                    max_epochs=2,
                    enable_progress_bar=False,
                ),
                metrics=graph_settings.TRAINING.metrics,
                loss_function=graph_settings.TRAINING.loss_function,
                optimizer=graph_settings.TRAINING.optimizer,
            )
        else:
            training = graph_settings.TRAINING

        return GeneralSettings(
            SESSION=session,
            DATASET=graph_settings.DATASET,
            DATAMODULE=graph_settings.DATAMODULE,
            MODEL=model,
            TRAINING=training,
        )

    def _load_dataset(self, graph_settings: GeneralSettings) -> Any:
        from dlkit.core.datasets.graph import GraphDataset
        from dlkit.interfaces.api.domain.precision import precision_override

        dataset_cfg = graph_settings.DATASET
        root: Path = dataset_cfg.root
        precision = graph_settings.SESSION.get_precision_strategy()
        with precision_override(precision):
            return GraphDataset(
                root=root,
                x=dataset_cfg.x,
                edge_index=dataset_cfg.edge_index,
                y=dataset_cfg.y,
            )

    def test_graph_model_float64_lr_tuning_integration(
        self,
        graph_settings: GeneralSettings,
    ) -> None:
        """Graph + float64 + LR tuning should train without dtype drift."""
        modified_settings = self._build_graph_settings(graph_settings, enable_lr_tuning=True)
        result = dlkit.train(modified_settings)
        assert isinstance(result, TrainingResult)

        dataset = self._load_dataset(modified_settings)
        sample = dataset[0]
        assert sample.x.dtype == torch.float64
        logger.info("Graph sample dtype after float64 run: {}", sample.x.dtype)

    def test_graph_model_float64_without_lr_tuning_baseline(
        self,
        graph_settings: GeneralSettings,
    ) -> None:
        """Baseline: Graph + float64 without LR tuning should still succeed."""
        modified_settings = self._build_graph_settings(
            graph_settings,
            enable_lr_tuning=False,
        )
        result = dlkit.train(modified_settings)
        assert isinstance(result, TrainingResult)

        dataset = self._load_dataset(modified_settings)
        sample = dataset[0]
        assert sample.x.dtype == torch.float64

    def test_graph_model_float64_lr_tuning_with_mlflow_tracking(
        self,
        graph_settings: GeneralSettings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MLflow tracking must not reintroduce dtype mismatches."""
        # Warm-up in default precision (float32) to populate cache
        base_run = dlkit.train(graph_settings)
        assert isinstance(base_run, TrainingResult)

        base_settings = self._build_graph_settings(graph_settings, enable_lr_tuning=True)
        manager = BasicOverrideManager()
        mlruns_dir = tmp_path / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(mlruns_dir / 'mlflow.db').as_posix()}")

        tracked_settings = manager.apply_overrides(
            base_settings,
            enable_mlflow=True,
            experiment_name=f"graph_precision_mlflow_{tmp_path.name}",
        )

        tracked_result = dlkit.train(tracked_settings)
        assert isinstance(tracked_result, TrainingResult)

        dataset = self._load_dataset(tracked_settings)
        sample = dataset[0]
        assert sample.x.dtype == torch.float64
        logger.info("Graph sample dtype after MLflow float64 run: {}", sample.x.dtype)
