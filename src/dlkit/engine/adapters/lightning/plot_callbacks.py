"""Lightning callbacks for automatic plot artifact generation.

Provides two callbacks:
- ``LossCurvePlotCallback``: accumulates epoch losses and logs a loss-curve PNG at fit_end.
- ``PredictionPlotCallback``: accumulates predict_step outputs and logs configured plots.

Both callbacks depend on ``IArtifactLogger`` for artifact upload and ``PlotSettings`` for
configuration. All matplotlib errors are caught and logged as warnings — callbacks never
raise on plot failure.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import torch
from lightning import Callback
from matplotlib.figure import Figure

from dlkit.domain.analysis.figures import loss_curve_figure
from dlkit.domain.analysis.figures._backend import plt
from dlkit.domain.analysis.protocols import IFigureGenerator
from dlkit.engine.adapters.lightning.tensor_extraction import as_flat_tensor
from dlkit.engine.artifacts import IArtifactLogger
from dlkit.infrastructure.config.plot_settings import PlotSettings
from dlkit.infrastructure.utils.logging_config import get_logger

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

_log = get_logger(__name__)


def _plot_and_log(
    fig: Figure,
    name: str,
    run_context: IArtifactLogger,
    settings: PlotSettings,
) -> None:
    """Encode fig to PNG bytes in-memory, log via run_context, then close it.

    Args:
        fig: matplotlib Figure to save.
        name: Artifact filename stem (no extension).
        run_context: Active MLflow run context.
        settings: Plot configuration for dpi and artifact_dir.
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format=settings.format, dpi=settings.dpi, bbox_inches="tight")
        artifact_file = f"{settings.artifact_dir}/{name}.{settings.format}"
        run_context.log_artifact_content(buf.getvalue(), artifact_file)
    except Exception as exc:
        _log.warning("_plot_and_log: failed to save/upload '{}' — {}", name, exc)
    finally:
        plt.close(fig)


class LossCurvePlotCallback(Callback):
    """Accumulate epoch losses and log a loss-curve PNG at fit_end.

    Reads ``train_loss`` and ``val_loss`` from ``trainer.callback_metrics``
    each epoch. Generates and uploads a PNG via IArtifactLogger on fit_end.

    Args:
        run_context: Active tracking run context for artifact upload.
        settings: Plot configuration (dpi, artifact_dir).
    """

    def __init__(self, run_context: IArtifactLogger, settings: PlotSettings) -> None:
        super().__init__()
        self._run_context = run_context
        self._settings = settings
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._lr_values: list[float] = []

    @staticmethod
    def _extract_scalar(
        metrics: Mapping[str, object],
        candidates: tuple[str, ...],
    ) -> float | None:
        """Try each candidate key; handle tensor .detach().cpu().item() chain.

        Args:
            metrics: trainer.callback_metrics dict.
            candidates: Ordered key names to try.

        Returns:
            Float value or None if not found/convertible.
        """
        for key in candidates:
            if key not in metrics:
                continue
            candidate: object = metrics[key]
            try:
                # Walk the detach → cpu → item chain safely without isinstance checks,
                # mirroring the _call_zero_arg_method pattern used in callbacks.py.
                for method in ("detach", "cpu", "item"):
                    result = getattr(candidate, method, None)
                    if callable(result):
                        candidate = result()
                return float(cast("float", candidate))
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_lr(trainer: Trainer) -> float | None:
        """Read the current LR from the first optimizer's first param group.

        Args:
            trainer: Lightning Trainer instance.

        Returns:
            Learning rate as float, or None if not available.
        """
        try:
            optimizers = trainer.optimizers
            opt = optimizers[0] if isinstance(optimizers, list) else optimizers
            return float(opt.param_groups[0]["lr"])
        except Exception:
            return None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Accumulate training loss and learning rate for the completed epoch.

        Skips during sanity check runs.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module (unused).
        """
        if getattr(trainer, "sanity_checking", False):
            return
        scalar = self._extract_scalar(
            trainer.callback_metrics or {},
            ("train_loss", "train/loss", "loss"),
        )
        if scalar is not None:
            self._train_losses.append(scalar)
        lr = self._extract_lr(trainer)
        if lr is not None:
            self._lr_values.append(lr)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Accumulate validation loss for the completed epoch.

        Skips during sanity check runs.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module (unused).
        """
        if getattr(trainer, "sanity_checking", False):
            return
        scalar = self._extract_scalar(
            trainer.callback_metrics or {},
            ("val_loss", "val/loss", "validation_loss"),
        )
        if scalar is not None:
            self._val_losses.append(scalar)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate and upload the loss-curve PNG.

        No-ops if no training losses were accumulated.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module (unused).
        """
        if not self._train_losses:
            _log.debug("LossCurvePlotCallback: no training losses accumulated; skipping.")
            return
        fig = loss_curve_figure(
            self._train_losses,
            self._val_losses or None,
            lr_values=self._lr_values or None,
        )
        _plot_and_log(fig, "loss_curve", self._run_context, self._settings)


class PredictionPlotCallback(Callback):
    """Accumulate predict_step TensorDict outputs and log configured plots.

    Mirrors NumpyWriter's accumulation pattern:
    ``on_predict_batch_end`` stores per-batch tensors,
    ``on_predict_epoch_end`` concatenates them, flattens to 1-D, and calls
    each injected generator.

    Args:
        run_context: Active tracking run context for artifact upload.
        generators: Ordered sequence of figure generators to apply.
        settings: Plot configuration (dpi, artifact_dir, max_scatter_points).
    """

    def __init__(
        self,
        run_context: IArtifactLogger,
        generators: Sequence[IFigureGenerator],
        settings: PlotSettings,
    ) -> None:
        super().__init__()
        self._run_context = run_context
        self._generators = generators
        self._settings = settings
        self._pred_batches: list[torch.Tensor] = []
        self._target_batches: list[torch.Tensor] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate per-batch prediction and target tensors.

        Expects ``outputs`` to be a TensorDict (or Mapping) with
        ``"predictions"`` and ``"targets"`` keys. Each value may be a bare
        tensor or a single-key TensorDict (the real predict_step contract for
        single-target models — see ``tensor_extraction.as_flat_tensor``).
        Skips, with a warning, batches where these can't be resolved.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module (unused).
            outputs: predict_step output, expected TensorDict.
            batch: Raw batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index (unused).
        """
        try:
            if not isinstance(outputs, Mapping):
                raise TypeError(f"expected Mapping, got {type(outputs).__name__}")
            outputs_map = cast("Mapping[str, object]", outputs)
            pred = as_flat_tensor(outputs_map["predictions"])
            tgt = as_flat_tensor(outputs_map["targets"])
            if pred is None or tgt is None:
                raise TypeError("predictions/targets must resolve to a single tensor")
            self._pred_batches.append(pred.detach().cpu())
            self._target_batches.append(tgt.detach().cpu())
        except (KeyError, AttributeError, TypeError) as exc:
            _log.warning("PredictionPlotCallback: skipping batch — {}", exc)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Concatenate batches, flatten to 1-D, and generate all configured plots.

        No-ops if no predictions were accumulated.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: Lightning module (unused).
        """
        if not self._pred_batches:
            _log.debug("PredictionPlotCallback: no predictions accumulated; skipping.")
            return
        preds_nd = torch.cat(self._pred_batches, dim=0).numpy()
        tgts_nd = torch.cat(self._target_batches, dim=0).numpy()
        # Flatten ALL dimensions to 1-D regardless of shape
        preds_flat = preds_nd.reshape(-1)
        tgts_flat = tgts_nd.reshape(-1)
        for gen in self._generators:
            _plot_and_log(
                gen.generate(preds_flat, tgts_flat),
                gen.name,
                self._run_context,
                self._settings,
            )


def build_plot_callbacks(
    run_context: IArtifactLogger,
    settings: PlotSettings,
) -> list[Callback]:
    """Construct plot callbacks based on PlotSettings flags.

    Instantiates domain generators inside this function so that
    ``engine.tracking`` (which cannot import ``domain``) can delegate
    callback construction here via a single import.

    Args:
        run_context: Active tracking run context for artifact upload.
        settings: Plot configuration controlling which callbacks are built.

    Returns:
        List of Lightning Callbacks to append to the trainer. May be empty.
    """
    from dlkit.domain.analysis.generators import (
        ErrorHistogramGenerator,
        ParityGenerator,
        ResidualGenerator,
        ResidualVsIndexGenerator,
    )

    callbacks: list[Callback] = []

    if settings.loss_curve:
        callbacks.append(LossCurvePlotCallback(run_context, settings))

    generators: list[IFigureGenerator] = []
    if settings.parity:
        generators.append(ParityGenerator(max_points=settings.max_scatter_points))
    if settings.residual:
        generators.append(ResidualGenerator(max_points=settings.max_scatter_points))
    if settings.error_histogram:
        generators.append(ErrorHistogramGenerator())
    if settings.residual_vs_index:
        generators.append(ResidualVsIndexGenerator(max_points=settings.max_scatter_points))

    if generators:
        callbacks.append(PredictionPlotCallback(run_context, generators, settings))

    return callbacks
