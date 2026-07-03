"""Plot artifact settings — opt-in visualization configuration."""

from __future__ import annotations

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class PlotSettings(BasicSettings):
    """Configuration for automatic plot artifact generation.

    All plot types are disabled by default (opt-in).
    Only active during workflows that produce a Lightning Trainer
    (training, search, convergence). Silently ignored for inference workflows.

    Args:
        enabled: Master switch. All other flags are ignored when False.
        loss_curve: Log a loss-vs-epoch curve PNG at fit_end.
        parity: Log a parity (predicted vs actual) PNG after predict.
        residual: Log a residuals-vs-predicted PNG after predict.
        error_histogram: Log an error distribution histogram PNG after predict.
        residual_vs_index: Log a residuals-vs-index PNG after predict.
        dpi: Resolution of generated PNG files.
        artifact_dir: MLflow artifact subdirectory for all plot PNGs.
        max_scatter_points: Random subsample cap for scatter-based plots.
            Prevents large datasets from stalling plot rendering.
    """

    enabled: bool = False
    loss_curve: bool = False
    parity: bool = False
    residual: bool = False
    error_histogram: bool = False
    residual_vs_index: bool = False
    dpi: int = 150
    artifact_dir: str = "plots"
    max_scatter_points: int = 5_000
