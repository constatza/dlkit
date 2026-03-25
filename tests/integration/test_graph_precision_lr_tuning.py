"""Integration tests ensuring graph precision remains consistent across workflows."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
from loguru import logger

import dlkit
from dlkit.interfaces.api.domain import TrainingResult
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.lr_tuner_settings import LRTunerSettings
from dlkit.tools.config.precision import PrecisionStrategy

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
        precision: PrecisionStrategy = PrecisionStrategy.FULL_64,
        enable_lr_tuning: bool = True,
    ) -> GeneralSettings:
        """Create modified GeneralSettings for graph precision scenarios."""
        from dlkit.tools.config.session_settings import SessionSettings
        from dlkit.tools.config.trainer_settings import TrainerSettings
        from dlkit.tools.config.training_settings import TrainingSettings

        session = SessionSettings(seed=42, precision=precision)
        graph_model = graph_settings.MODEL
        assert graph_model is not None

        model = graph_model.model_copy(
            update={
                "name": "ScaledGATv2Projection",
                "module_path": "dlkit.core.models.nn.graph.scaled_projection_networks",
                "hidden_size": 4,
                "num_layers": 1,
                "heads": 1,
            }
        )

        graph_training = graph_settings.TRAINING
        assert graph_training is not None

        if enable_lr_tuning:
            training = TrainingSettings(
                epochs=1,
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
                loss_function=graph_training.loss_function,
                optimizer=graph_training.optimizer,
            )
        else:
            training = graph_training

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
        assert dataset_cfg is not None
        root = dataset_cfg.root
        assert root is not None
        x_path = getattr(dataset_cfg, "x", None)
        edge_index_path = getattr(dataset_cfg, "edge_index", None)
        y_path = getattr(dataset_cfg, "y", None)
        assert x_path is not None
        assert edge_index_path is not None
        assert y_path is not None
        precision = graph_settings.SESSION.get_precision_strategy()
        with precision_override(precision):
            return GraphDataset(
                root=root,
                x=x_path,
                edge_index=edge_index_path,
                y=y_path,
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
