"""Tests for NumpyWriter callback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch

from dlkit.engine.adapters.lightning.callbacks import NumpyWriter
from dlkit.engine.tracking.artifacts import InMemoryArtifactCollector


@pytest.fixture
def mock_trainer() -> Mock:
    """Create mock trainer."""
    trainer = Mock()
    trainer.default_root_dir = "/tmp/dlkit-lightning"
    return trainer


@pytest.fixture
def mock_module() -> Mock:
    """Create mock lightning module."""
    return Mock()


@pytest.fixture
def sample_dict_output() -> dict[str, torch.Tensor]:
    """Sample dictionary output."""
    return {"predictions": torch.tensor([1.0, 2.0, 3.0]), "logits": torch.tensor([0.1, 0.2, 0.3])}


@pytest.fixture
def sample_tensor_output() -> torch.Tensor:
    """Sample tensor output."""
    return torch.tensor([1.0, 2.0, 3.0])


class TestNumpyWriterBasic:
    """Basic tests for NumpyWriter callback."""

    def test_init_with_output_dir(self, tmp_path: Path) -> None:
        """Test initialization with custom output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        writer = NumpyWriter(output_dir=output_dir)
        assert writer.output_dir == output_dir

    def test_init_without_output_dir(
        self, tmp_path: Path, mock_trainer: Mock, mock_module: Mock
    ) -> None:
        """Test initialization without output directory uses lazy resolution on predict start."""
        writer = NumpyWriter()
        # output_dir is None until on_predict_start resolves it lazily
        assert writer.output_dir is None
        writer.on_predict_start(mock_trainer, mock_module)
        assert "predictions" in str(writer.output_dir)

    def test_dict_batch_end(
        self,
        tmp_path: Path,
        mock_trainer: Mock,
        mock_module: Mock,
        sample_dict_output: dict[str, torch.Tensor],
    ) -> None:
        """Test batch end with dictionary output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        writer = NumpyWriter(output_dir=output_dir)
        writer.on_predict_batch_end(mock_trainer, mock_module, sample_dict_output, (), 0, 0)

        # Check predictions are stored
        assert len(writer._predictions) > 0

    def test_tensor_batch_end(
        self,
        tmp_path: Path,
        mock_trainer: Mock,
        mock_module: Mock,
        sample_tensor_output: torch.Tensor,
    ) -> None:
        """Test batch end with tensor output."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        writer = NumpyWriter(output_dir=output_dir)
        writer.on_predict_batch_end(
            mock_trainer,
            mock_module,
            cast(Any, sample_tensor_output),
            (),
            0,
            0,
        )

        # Check predictions are stored
        assert len(writer._predictions) > 0

    def test_epoch_end_saves_files(
        self,
        tmp_path: Path,
        mock_trainer: Mock,
        mock_module: Mock,
        sample_dict_output: dict[str, torch.Tensor],
    ) -> None:
        """Test that epoch end saves files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        writer = NumpyWriter(output_dir=output_dir)

        # Add some predictions
        writer.on_predict_batch_end(mock_trainer, mock_module, sample_dict_output, (), 0, 0)

        # Trigger epoch end
        writer.on_predict_epoch_end(mock_trainer, mock_module)

        # Check files are created
        output_files = list(output_dir.glob("*.npy"))
        assert len(output_files) > 0

    def test_epoch_end_records_prediction_artifacts(
        self,
        tmp_path: Path,
        mock_trainer: Mock,
        mock_module: Mock,
        sample_tensor_output: torch.Tensor,
    ) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        collector = InMemoryArtifactCollector()
        writer = NumpyWriter(output_dir=output_dir, artifact_collector=collector)

        writer.on_predict_batch_end(
            mock_trainer,
            mock_module,
            cast(Any, sample_tensor_output),
            (),
            0,
            0,
        )
        writer.on_predict_epoch_end(mock_trainer, mock_module)

        artifacts = collector.snapshot()
        assert len(artifacts) == 1
        assert artifacts[0].artifact_path == "predictions/predictions.npy"

    def test_invalid_output_type(
        self, tmp_path: Path, mock_trainer: Mock, mock_module: Mock
    ) -> None:
        """Test handling of invalid output types."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        writer = NumpyWriter(output_dir=output_dir)

        # Should not crash with invalid output
        writer.on_predict_batch_end(
            mock_trainer,
            mock_module,
            "invalid",  # ty: ignore[invalid-argument-type]
            (),
            0,
            0,
        )

        # Should have no predictions stored
        assert len(writer._predictions) == 0


def test_callbacks_module_has_no_mlflow_dependency() -> None:
    source = Path("src/dlkit/engine/adapters/lightning/callbacks.py").read_text()
    assert "import mlflow" not in source
    assert "mlflow.active_run" not in source
