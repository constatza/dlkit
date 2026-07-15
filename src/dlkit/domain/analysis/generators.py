"""Concrete IFigureGenerator implementations for standard regression plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from dlkit.domain.analysis.figures import (
    error_histogram_figure,
    parity_figure,
    residual_figure,
    residual_vs_index_figure,
)


@dataclass(frozen=True)
class ParityGenerator:
    """Generates a parity (predicted vs actual) scatter plot.

    Args:
        name: Artifact filename stem.
        max_points: Random subsample cap to keep the plot responsive.
    """

    name: str = "parity_plot"
    max_points: int = 5_000

    def generate(self, predictions: np.ndarray, targets: np.ndarray) -> Figure:
        """Generate parity figure.

        Args:
            predictions: Flattened predictions array, shape ``(N,)``.
            targets: Flattened targets array, shape ``(N,)``.

        Returns:
            matplotlib Figure.
        """
        return parity_figure(predictions, targets, max_points=self.max_points)


@dataclass(frozen=True)
class ResidualGenerator:
    """Generates a residuals-vs-predicted scatter plot.

    Args:
        name: Artifact filename stem.
        max_points: Random subsample cap.
    """

    name: str = "residual_plot"
    max_points: int = 5_000

    def generate(self, predictions: np.ndarray, targets: np.ndarray) -> Figure:
        """Generate residual figure.

        Args:
            predictions: Flattened predictions array, shape ``(N,)``.
            targets: Flattened targets array, shape ``(N,)``.

        Returns:
            matplotlib Figure.
        """
        return residual_figure(predictions, targets, max_points=self.max_points)


@dataclass(frozen=True)
class ErrorHistogramGenerator:
    """Generates an error distribution histogram with normal and KDE overlays.

    Args:
        name: Artifact filename stem.
        bins: Histogram bin strategy. ``"auto"`` uses NumPy adaptive binning.
        display_percentiles: Optional ``(lower, upper)`` percentile window for
            the displayed x-axis. ``None`` uses the full raw min/max range.
    """

    name: str = "error_histogram"
    bins: int | str = "auto"
    display_percentiles: tuple[float, float] | None = None

    def generate(self, predictions: np.ndarray, targets: np.ndarray) -> Figure:
        """Generate error histogram figure.

        Args:
            predictions: Flattened predictions array, shape ``(N,)``.
            targets: Flattened targets array, shape ``(N,)``.

        Returns:
            matplotlib Figure.
        """
        return error_histogram_figure(
            predictions,
            targets,
            bins=self.bins,
            display_percentiles=self.display_percentiles,
        )


@dataclass(frozen=True)
class ResidualVsIndexGenerator:
    """Generates a residuals-vs-sample-index scatter plot.

    Useful for detecting trends and autocorrelation in prediction errors.

    Args:
        name: Artifact filename stem.
        max_points: Random subsample cap.
    """

    name: str = "residual_vs_index"
    max_points: int = 5_000

    def generate(self, predictions: np.ndarray, targets: np.ndarray) -> Figure:
        """Generate residual-vs-index figure.

        Args:
            predictions: Flattened predictions array, shape ``(N,)``.
            targets: Flattened targets array, shape ``(N,)``.

        Returns:
            matplotlib Figure.
        """
        return residual_vs_index_figure(predictions, targets, max_points=self.max_points)
