"""Tests for LossCurvePlotCallback and PredictionPlotCallback.

Covers:
- _extract_scalar: plain float, torch.Tensor, absent key
- LossCurvePlotCallback.on_fit_end: no-op when empty; calls log_artifact when populated
- LossCurvePlotCallback: sanity_checking guard (epochs excluded)
- PredictionPlotCallback.on_predict_batch_end: silent skip on missing keys
- PredictionPlotCallback.on_predict_epoch_end: no-op when empty; calls log_artifact
  once per generator when populated
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from dlkit.engine.adapters.lightning.plot_callbacks import (
    LossCurvePlotCallback,
    PredictionPlotCallback,
)
from dlkit.engine.artifacts import IArtifactLogger
from dlkit.infrastructure.config.plot_settings import PlotSettings

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
TRAIN_LOSS_KEY: str = "train_loss"
VAL_LOSS_KEY: str = "val_loss"
BATCH_SIZE: int = 8
NUM_OUTPUTS: int = 3

# ---------------------------------------------------------------------------
# Fixtures — shared infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture
def plot_settings() -> PlotSettings:
    """Default PlotSettings used across all callback tests.

    Returns:
        PlotSettings with dpi=72 and artifact_dir="test_plots".
    """
    return PlotSettings(dpi=72, artifact_dir="test_plots")


@pytest.fixture
def mock_run_context() -> MagicMock:
    """MagicMock satisfying IArtifactLogger for all callback tests.

    Returns:
        MagicMock spec'd to IArtifactLogger with log_artifact stub.
    """
    ctx = MagicMock(spec=IArtifactLogger)
    ctx.log_artifact.return_value = None
    return ctx


@pytest.fixture
def mock_trainer() -> SimpleNamespace:
    """Minimal trainer stand-in with callback_metrics and sanity_checking=False.

    Returns:
        SimpleNamespace mimicking the Trainer attributes used by callbacks.
    """
    return SimpleNamespace(
        callback_metrics={},
        sanity_checking=False,
        current_epoch=0,
    )


@pytest.fixture
def pl_module() -> MagicMock:
    """Bare MagicMock standing in for a LightningModule.

    Returns:
        MagicMock with no spec constraints.
    """
    return MagicMock()


# ---------------------------------------------------------------------------
# Fixtures — LossCurvePlotCallback data
# ---------------------------------------------------------------------------


@pytest.fixture
def loss_curve_callback(
    mock_run_context: MagicMock, plot_settings: PlotSettings
) -> LossCurvePlotCallback:
    """Fresh LossCurvePlotCallback with empty history.

    Args:
        mock_run_context: Stubbed IArtifactLogger.
        plot_settings: Default PlotSettings.

    Returns:
        LossCurvePlotCallback instance with no accumulated losses.
    """
    return LossCurvePlotCallback(
        run_context=mock_run_context,
        settings=plot_settings,
    )


@pytest.fixture
def train_loss_metrics() -> dict[str, float]:
    """Callback_metrics dict containing a plain float train_loss.

    Returns:
        Dict with TRAIN_LOSS_KEY mapped to a plain float.
    """
    return {TRAIN_LOSS_KEY: 0.42}


@pytest.fixture
def train_loss_tensor_metrics() -> dict[str, torch.Tensor]:
    """Callback_metrics dict containing a torch.Tensor train_loss.

    Returns:
        Dict with TRAIN_LOSS_KEY mapped to a scalar torch.Tensor.
    """
    return {TRAIN_LOSS_KEY: torch.tensor(0.42)}


@pytest.fixture
def val_loss_metrics() -> dict[str, float]:
    """Callback_metrics dict containing a plain float val_loss.

    Returns:
        Dict with VAL_LOSS_KEY mapped to a plain float.
    """
    return {VAL_LOSS_KEY: 0.55}


# ---------------------------------------------------------------------------
# Fixtures — PredictionPlotCallback data
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_generator() -> MagicMock:
    """Fake IFigureGenerator that returns a trivial Figure.

    Returns:
        MagicMock with .name and .generate() stubbed to return a Figure.
    """
    gen = MagicMock()
    gen.name = "fake_plot"

    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    gen.generate.return_value = fig
    return gen


@pytest.fixture
def prediction_callback(
    mock_run_context: MagicMock,
    fake_generator: MagicMock,
    plot_settings: PlotSettings,
) -> PredictionPlotCallback:
    """PredictionPlotCallback pre-wired with one fake generator.

    Args:
        mock_run_context: Stubbed IArtifactLogger.
        fake_generator: Fake IFigureGenerator.
        plot_settings: Default PlotSettings.

    Returns:
        PredictionPlotCallback with one generator and empty batch history.
    """
    return PredictionPlotCallback(
        run_context=mock_run_context,
        generators=[fake_generator],
        settings=plot_settings,
    )


@pytest.fixture
def prediction_batch_output() -> dict[str, torch.Tensor]:
    """Valid predict_step output with 'predictions' and 'targets' keys.

    Returns:
        Dict mapping 'predictions' and 'targets' to matching Tensors.
    """
    return {
        "predictions": torch.randn(BATCH_SIZE, NUM_OUTPUTS),
        "targets": torch.randn(BATCH_SIZE, NUM_OUTPUTS),
    }


@pytest.fixture
def missing_keys_output() -> dict[str, torch.Tensor]:
    """predict_step output that lacks both 'predictions' and 'targets'.

    Returns:
        Dict with an unrelated key only.
    """
    return {"latents": torch.randn(BATCH_SIZE, 4)}


# ---------------------------------------------------------------------------
# Tests — LossCurvePlotCallback._extract_scalar
# ---------------------------------------------------------------------------


class TestExtractScalar:
    """Unit tests for LossCurvePlotCallback._extract_scalar."""

    def test_plain_float_returned(self) -> None:
        """_extract_scalar converts a plain float value correctly."""
        result = LossCurvePlotCallback._extract_scalar({TRAIN_LOSS_KEY: 0.25}, (TRAIN_LOSS_KEY,))
        assert result == pytest.approx(0.25)

    def test_torch_tensor_returned(self) -> None:
        """_extract_scalar converts a scalar torch.Tensor via item()."""
        result = LossCurvePlotCallback._extract_scalar(
            {TRAIN_LOSS_KEY: torch.tensor(0.33)}, (TRAIN_LOSS_KEY,)
        )
        assert result == pytest.approx(0.33)

    def test_absent_key_returns_none(self) -> None:
        """_extract_scalar returns None when no candidate key is present."""
        result = LossCurvePlotCallback._extract_scalar({"other_key": 1.0}, (TRAIN_LOSS_KEY,))
        assert result is None

    def test_first_matching_candidate_wins(self) -> None:
        """_extract_scalar uses the first matching candidate key in order."""
        metrics = {
            "train/loss": 0.10,
            TRAIN_LOSS_KEY: 0.99,
        }
        result = LossCurvePlotCallback._extract_scalar(metrics, (TRAIN_LOSS_KEY, "train/loss"))
        assert result == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# Tests — LossCurvePlotCallback lifecycle
# ---------------------------------------------------------------------------


class TestLossCurvePlotCallbackFitEnd:
    """Tests for LossCurvePlotCallback.on_fit_end behavior."""

    def test_no_op_when_no_losses_accumulated(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_run_context: MagicMock,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
    ) -> None:
        """on_fit_end does not call run_context when no losses were accumulated.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_run_context: Stubbed IArtifactLogger.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
        """
        loss_curve_callback.on_fit_end(mock_trainer, pl_module)
        mock_run_context.log_artifact.assert_not_called()

    def test_log_artifact_called_when_losses_present(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_run_context: MagicMock,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        train_loss_metrics: dict[str, float],
    ) -> None:
        """on_fit_end calls run_context.log_artifact exactly once when losses exist.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_run_context: Stubbed IArtifactLogger.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            train_loss_metrics: Metrics dict with a plain float train_loss.
        """
        mock_trainer.callback_metrics = train_loss_metrics
        loss_curve_callback.on_train_epoch_end(mock_trainer, pl_module)

        with patch(
            "dlkit.engine.adapters.lightning.plot_callbacks._plot_and_log"
        ) as mock_plot_and_log:
            loss_curve_callback.on_fit_end(mock_trainer, pl_module)
            mock_plot_and_log.assert_called_once()

    def test_log_artifact_called_with_tensor_loss(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_run_context: MagicMock,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        train_loss_tensor_metrics: dict[str, torch.Tensor],
    ) -> None:
        """on_fit_end calls _plot_and_log when loss is a torch.Tensor.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_run_context: Stubbed IArtifactLogger.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            train_loss_tensor_metrics: Metrics dict with a tensor train_loss.
        """
        mock_trainer.callback_metrics = train_loss_tensor_metrics
        loss_curve_callback.on_train_epoch_end(mock_trainer, pl_module)

        with patch(
            "dlkit.engine.adapters.lightning.plot_callbacks._plot_and_log"
        ) as mock_plot_and_log:
            loss_curve_callback.on_fit_end(mock_trainer, pl_module)
            mock_plot_and_log.assert_called_once()


class TestLossCurvePlotCallbackSanityCheck:
    """Tests for sanity_checking guard in LossCurvePlotCallback."""

    def test_sanity_check_epoch_not_accumulated(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        train_loss_metrics: dict[str, float],
    ) -> None:
        """on_train_epoch_end skips accumulation during sanity check.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            train_loss_metrics: Metrics dict with a plain float train_loss.
        """
        mock_trainer.callback_metrics = train_loss_metrics
        mock_trainer.sanity_checking = True
        loss_curve_callback.on_train_epoch_end(mock_trainer, pl_module)
        # pylint: disable=protected-access
        assert loss_curve_callback._train_losses == []

    def test_normal_epoch_is_accumulated(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        train_loss_metrics: dict[str, float],
    ) -> None:
        """on_train_epoch_end accumulates loss when not sanity checking.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_trainer: Minimal trainer stand-in with sanity_checking=False.
            pl_module: Bare MagicMock module.
            train_loss_metrics: Metrics dict with a plain float train_loss.
        """
        mock_trainer.callback_metrics = train_loss_metrics
        loss_curve_callback.on_train_epoch_end(mock_trainer, pl_module)
        # pylint: disable=protected-access
        assert len(loss_curve_callback._train_losses) == 1

    def test_val_sanity_check_epoch_not_accumulated(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        val_loss_metrics: dict[str, float],
    ) -> None:
        """on_validation_epoch_end skips accumulation during sanity check.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            val_loss_metrics: Metrics dict with a plain float val_loss.
        """
        mock_trainer.callback_metrics = val_loss_metrics
        mock_trainer.sanity_checking = True
        loss_curve_callback.on_validation_epoch_end(mock_trainer, pl_module)
        # pylint: disable=protected-access
        assert loss_curve_callback._val_losses == []

    def test_val_normal_epoch_is_accumulated(
        self,
        loss_curve_callback: LossCurvePlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        val_loss_metrics: dict[str, float],
    ) -> None:
        """on_validation_epoch_end accumulates val loss when not sanity checking.

        Args:
            loss_curve_callback: Callback with empty loss history.
            mock_trainer: Minimal trainer stand-in with sanity_checking=False.
            pl_module: Bare MagicMock module.
            val_loss_metrics: Metrics dict with a plain float val_loss.
        """
        mock_trainer.callback_metrics = val_loss_metrics
        loss_curve_callback.on_validation_epoch_end(mock_trainer, pl_module)
        # pylint: disable=protected-access
        assert len(loss_curve_callback._val_losses) == 1


# ---------------------------------------------------------------------------
# Tests — PredictionPlotCallback.on_predict_batch_end
# ---------------------------------------------------------------------------


class TestPredictionBatchEnd:
    """Tests for PredictionPlotCallback.on_predict_batch_end."""

    def test_valid_batch_accumulated(
        self,
        prediction_callback: PredictionPlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        prediction_batch_output: dict[str, torch.Tensor],
    ) -> None:
        """A valid batch with predictions/targets keys is accumulated.

        Args:
            prediction_callback: Callback with one generator and empty history.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            prediction_batch_output: Valid output with predictions and targets.
        """
        prediction_callback.on_predict_batch_end(
            mock_trainer, pl_module, prediction_batch_output, batch=None, batch_idx=0
        )
        # pylint: disable=protected-access
        assert len(prediction_callback._pred_batches) == 1
        assert len(prediction_callback._target_batches) == 1

    def test_missing_predictions_key_skipped(
        self,
        prediction_callback: PredictionPlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        missing_keys_output: dict[str, torch.Tensor],
    ) -> None:
        """Batch without 'predictions'/'targets' keys is silently skipped.

        Args:
            prediction_callback: Callback with one generator and empty history.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            missing_keys_output: Output dict with unrelated keys only.
        """
        prediction_callback.on_predict_batch_end(
            mock_trainer, pl_module, missing_keys_output, batch=None, batch_idx=0
        )
        # pylint: disable=protected-access
        assert prediction_callback._pred_batches == []
        assert prediction_callback._target_batches == []

    def test_non_mapping_output_skipped(
        self,
        prediction_callback: PredictionPlotCallback,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
    ) -> None:
        """Non-Mapping output (e.g. a plain tensor) is silently skipped.

        Args:
            prediction_callback: Callback with one generator and empty history.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
        """
        prediction_callback.on_predict_batch_end(
            mock_trainer, pl_module, torch.randn(4, 3), batch=None, batch_idx=0
        )
        # pylint: disable=protected-access
        assert prediction_callback._pred_batches == []


# ---------------------------------------------------------------------------
# Tests — PredictionPlotCallback.on_predict_epoch_end
# ---------------------------------------------------------------------------


class TestPredictionEpochEnd:
    """Tests for PredictionPlotCallback.on_predict_epoch_end."""

    def test_no_op_when_no_batches_accumulated(
        self,
        prediction_callback: PredictionPlotCallback,
        mock_run_context: MagicMock,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
    ) -> None:
        """on_predict_epoch_end does not call generators or log_artifact when empty.

        Args:
            prediction_callback: Callback with one generator and empty history.
            mock_run_context: Stubbed IArtifactLogger.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
        """
        prediction_callback.on_predict_epoch_end(mock_trainer, pl_module)
        mock_run_context.log_artifact.assert_not_called()

    def test_generator_called_once_per_generator(
        self,
        prediction_callback: PredictionPlotCallback,
        fake_generator: MagicMock,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        prediction_batch_output: dict[str, torch.Tensor],
    ) -> None:
        """on_predict_epoch_end calls each generator's generate() exactly once.

        Args:
            prediction_callback: Callback with one fake generator.
            fake_generator: Fake IFigureGenerator with .generate() stub.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            prediction_batch_output: Valid output with predictions and targets.
        """
        prediction_callback.on_predict_batch_end(
            mock_trainer, pl_module, prediction_batch_output, batch=None, batch_idx=0
        )
        with patch(
            "dlkit.engine.adapters.lightning.plot_callbacks._plot_and_log"
        ) as mock_plot_and_log:
            prediction_callback.on_predict_epoch_end(mock_trainer, pl_module)
            fake_generator.generate.assert_called_once()
            mock_plot_and_log.assert_called_once()

    def test_log_artifact_called_once_per_generator_with_two_generators(
        self,
        mock_run_context: MagicMock,
        plot_settings: PlotSettings,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        prediction_batch_output: dict[str, torch.Tensor],
    ) -> None:
        """on_predict_epoch_end calls _plot_and_log once per generator.

        With two generators, _plot_and_log is called twice.

        Args:
            mock_run_context: Stubbed IArtifactLogger.
            plot_settings: Default PlotSettings.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            prediction_batch_output: Valid output with predictions and targets.
        """
        import matplotlib.pyplot as mpl_plt

        gen_a = MagicMock()
        gen_a.name = "plot_a"
        fig_a, _ = mpl_plt.subplots()
        gen_a.generate.return_value = fig_a

        gen_b = MagicMock()
        gen_b.name = "plot_b"
        fig_b, _ = mpl_plt.subplots()
        gen_b.generate.return_value = fig_b

        cb = PredictionPlotCallback(
            run_context=mock_run_context,
            generators=[gen_a, gen_b],
            settings=plot_settings,
        )
        cb.on_predict_batch_end(
            mock_trainer, pl_module, prediction_batch_output, batch=None, batch_idx=0
        )
        with patch(
            "dlkit.engine.adapters.lightning.plot_callbacks._plot_and_log"
        ) as mock_plot_and_log:
            cb.on_predict_epoch_end(mock_trainer, pl_module)
            assert mock_plot_and_log.call_count == 2

    def test_predictions_flattened_to_1d_before_generate(
        self,
        prediction_callback: PredictionPlotCallback,
        fake_generator: MagicMock,
        mock_trainer: SimpleNamespace,
        pl_module: MagicMock,
        prediction_batch_output: dict[str, torch.Tensor],
    ) -> None:
        """Tensors are concatenated and flattened to 1-D before being passed to generators.

        Args:
            prediction_callback: Callback with one fake generator.
            fake_generator: Fake IFigureGenerator with .generate() stub.
            mock_trainer: Minimal trainer stand-in.
            pl_module: Bare MagicMock module.
            prediction_batch_output: Valid output batch (BATCH_SIZE x NUM_OUTPUTS).
        """
        prediction_callback.on_predict_batch_end(
            mock_trainer, pl_module, prediction_batch_output, batch=None, batch_idx=0
        )
        with patch("dlkit.engine.adapters.lightning.plot_callbacks._plot_and_log"):
            prediction_callback.on_predict_epoch_end(mock_trainer, pl_module)

        # Inspect the numpy arrays passed to generate
        args, _ = fake_generator.generate.call_args
        preds_arg: np.ndarray = args[0]
        targets_arg: np.ndarray = args[1]
        assert preds_arg.ndim == 1
        assert targets_arg.ndim == 1
        expected_length = BATCH_SIZE * NUM_OUTPUTS
        assert len(preds_arg) == expected_length
        assert len(targets_arg) == expected_length
