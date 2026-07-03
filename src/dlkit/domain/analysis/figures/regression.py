"""Regression analysis figure generators."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from dlkit.domain.analysis.figures._backend import plt  # noqa: F401


def _flatten_and_validate(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both arrays to 1-D and raise ValueError if lengths differ.

    Args:
        predictions: Predictions array of any shape.
        targets: Targets array of any shape.

    Returns:
        Tuple of (preds_flat, tgts_flat) both 1-D with matching lengths.

    Raises:
        ValueError: If flattened lengths differ.
    """
    preds_flat = np.asarray(predictions).reshape(-1)
    tgts_flat = np.asarray(targets).reshape(-1)
    if len(preds_flat) != len(tgts_flat):
        raise ValueError(
            f"predictions and targets must have the same total number of elements; "
            f"got {len(preds_flat)} vs {len(tgts_flat)}"
        )
    return preds_flat, tgts_flat


def _subsample(
    preds: np.ndarray,
    targets: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Random subsample when N > max_points (reproducible: seed=0).

    Args:
        preds: 1-D predictions array.
        targets: 1-D targets array.
        max_points: Maximum number of points to keep.

    Returns:
        Tuple of (preds, targets) with at most max_points elements.
    """
    n = len(preds)
    if n <= max_points:
        return preds, targets
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_points, replace=False)
    return preds[idx], targets[idx]


def parity_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    title: str = "Parity Plot",
    xlabel: str = "Predicted",
    ylabel: str = "Actual",
    max_points: int = 5_000,
) -> Figure:
    """Predicted vs actual scatter with identity line and R² annotation.

    Flattens all dimensions to 1-D before plotting. A (B, D) output
    produces B×D scatter points.

    Args:
        predictions: Model predictions, any shape; flattened to 1-D.
        targets: Ground-truth targets, any shape; must have same total elements.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        max_points: Random subsample cap for large datasets.

    Returns:
        matplotlib Figure. Caller must close it.

    Raises:
        ValueError: If flattened shapes differ.
    """
    preds_flat, tgts_flat = _flatten_and_validate(predictions, targets)
    preds_flat, tgts_flat = _subsample(preds_flat, tgts_flat, max_points)

    r2 = float(np.corrcoef(preds_flat, tgts_flat)[0, 1] ** 2)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(preds_flat, tgts_flat, alpha=0.5, s=10, label=f"R²={r2:.4f}")
    lo = float(min(preds_flat.min(), tgts_flat.min()))
    hi = float(max(preds_flat.max(), tgts_flat.max()))
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def residual_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    title: str = "Residuals vs Predicted",
    xlabel: str = "Predicted",
    ylabel: str = "Residual",
    max_points: int = 5_000,
) -> Figure:
    """(targets − predictions) vs predicted scatter with y=0 reference.

    Flattens all dimensions to 1-D before plotting.

    Args:
        predictions: Model predictions, any shape; flattened to 1-D.
        targets: Ground-truth targets, any shape; must have same total elements.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        max_points: Random subsample cap for large datasets.

    Returns:
        matplotlib Figure. Caller must close it.

    Raises:
        ValueError: If flattened shapes differ.
    """
    preds_flat, tgts_flat = _flatten_and_validate(predictions, targets)
    preds_flat, tgts_flat = _subsample(preds_flat, tgts_flat, max_points)

    residuals = tgts_flat - preds_flat

    fig, ax = plt.subplots(1, 1)
    ax.scatter(preds_flat, residuals, alpha=0.5, s=10)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def error_histogram_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    title: str = "Error Histogram",
    bins: int = 50,
) -> Figure:
    """Histogram of prediction errors (targets − predictions) with normal overlay.

    Flattens all dimensions to 1-D. Normal distribution overlay uses
    the sample mean and std of the errors.

    Args:
        predictions: Model predictions, any shape; flattened to 1-D.
        targets: Ground-truth targets, any shape; must have same total elements.
        title: Figure title.
        bins: Number of histogram bins.

    Returns:
        matplotlib Figure. Caller must close it.

    Raises:
        ValueError: If flattened shapes differ.
    """
    preds_flat, tgts_flat = _flatten_and_validate(predictions, targets)

    errors = tgts_flat - preds_flat
    mu = float(errors.mean())
    sigma = float(errors.std())

    fig, ax = plt.subplots(1, 1)
    ax.hist(errors, bins=bins, density=True)

    x = np.linspace(errors.min(), errors.max(), 300)
    if sigma > 0:
        normal_curve = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, normal_curve, "r-")

    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    fig.tight_layout()
    return fig


def residual_vs_index_figure(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    title: str = "Residuals vs Index",
    max_points: int = 5_000,
) -> Figure:
    """Residuals vs sample index scatter — detects trends and autocorrelation.

    Flattens all dimensions to 1-D. Index = position in the flattened array.

    Args:
        predictions: Model predictions, any shape; flattened to 1-D.
        targets: Ground-truth targets, any shape; must have same total elements.
        title: Figure title.
        max_points: Random subsample cap for large datasets.

    Returns:
        matplotlib Figure. Caller must close it.

    Raises:
        ValueError: If flattened shapes differ.
    """
    preds_flat, tgts_flat = _flatten_and_validate(predictions, targets)
    preds_flat, tgts_flat = _subsample(preds_flat, tgts_flat, max_points)

    residuals = tgts_flat - preds_flat
    indices = np.arange(len(residuals))

    fig, ax = plt.subplots(1, 1)
    ax.scatter(indices, residuals, alpha=0.5, s=10)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    return fig
