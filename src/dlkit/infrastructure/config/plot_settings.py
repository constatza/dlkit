"""Plot artifact settings — opt-in visualization configuration."""

from __future__ import annotations

from typing import Literal

from dlkit.infrastructure.config.core.base_settings import BasicSettings

ImageFormat = Literal["png", "svg", "pdf"]
HistogramBins = int | str
PercentileRange = tuple[float, float]
HistogramDisplayPercentiles = PercentileRange | Literal["full"]


class PlotSettings(BasicSettings):
    """Configuration for automatic plot artifact generation.

    All plot types are disabled by default (opt-in).
    Only active during workflows that produce a Lightning Trainer
    (training, search, convergence). Silently ignored for inference workflows.

    Args:
        enabled: Master switch. All other flags are ignored when False.
        loss_curve: Log a loss-vs-epoch curve at fit_end.
        parity: Log a parity (predicted vs actual) plot after predict.
        residual: Log a residuals-vs-predicted plot after predict.
        error_histogram: Log an error distribution histogram after predict.
        residual_vs_index: Log a residuals-vs-index plot after predict.
        format: Image format for all plots. ``"svg"`` gives vector graphics;
            ``"png"`` rasters at ``dpi``; ``"pdf"`` embeds as vector PDF.
        dpi: Resolution for raster formats (png). Ignored for svg/pdf.
        artifact_dir: MLflow artifact subdirectory for all plots.
        max_scatter_points: Random subsample cap for scatter-based plots.
            Prevents large datasets from stalling plot rendering.
        error_histogram_bins: Histogram bin strategy. ``"auto"`` delegates to
            NumPy adaptive bin selection; an integer fixes the bin count.
        error_histogram_display_percentiles: ``(lower, upper)`` percentile
            window for the error histogram x-axis, or ``"full"`` for the raw
            min/max range.
    """

    enabled: bool = False
    loss_curve: bool = False
    parity: bool = False
    residual: bool = False
    error_histogram: bool = False
    residual_vs_index: bool = False
    format: ImageFormat = "png"
    dpi: int = 300
    artifact_dir: str = "plots"
    max_scatter_points: int = 5_000
    error_histogram_bins: HistogramBins = "auto"
    error_histogram_display_percentiles: HistogramDisplayPercentiles = (0.5, 99.5)
