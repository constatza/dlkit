"""Regression analysis figure generators."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from dlkit.domain.analysis.figures._backend import plt  # noqa: F401

_KDE_POINTS = 300
_KDE_MAX_POINTS = 10_000
_TUKEY_K = 3.0


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


def _finite_values(values: np.ndarray) -> np.ndarray:
    """Return finite values only, preserving the raw numerical scale."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        raise ValueError("plot values must contain at least one finite value")
    return finite


def _padded_range(lo: float, hi: float) -> tuple[float, float]:
    """Return a non-zero plotting range around lo/hi."""
    if lo != hi:
        return lo, hi
    pad = max(abs(lo) * 0.05, 1.0)
    return lo - pad, hi + pad


def _error_display_range(
    errors: np.ndarray,
    display_percentiles: tuple[float, float] | None,
) -> tuple[float, float]:
    """Use an explicit percentile display window when configured.

    ``display_percentiles=None`` means "show the full raw range" and is
    left untouched. When a percentile window is requested, it is further
    clamped by a Tukey IQR fence (``[Q1 - k*IQR, Q3 + k*IQR]``), whichever
    bound is tighter: the percentile window alone degrades for
    heavy-tailed error distributions, since the 0.5th/99.5th percentiles
    of a Cauchy-shaped distribution can still span 100+ units even though
    the bulk of the mass sits within a few units of the center.
    """
    full_lo = float(errors.min())
    full_hi = float(errors.max())
    if display_percentiles is not None:
        lower_percentile, upper_percentile = display_percentiles
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError(
                "display_percentiles must satisfy "
                "0 <= lower < upper <= 100; "
                f"got {display_percentiles!r}"
            )
        display_lo, display_hi = np.percentile(
            errors,
            [lower_percentile, upper_percentile],
        )
        if full_lo < display_lo or display_hi < full_hi:
            lo, hi = float(display_lo), float(display_hi)
            q1, q3 = np.percentile(errors, [25, 75])
            iqr = float(q3 - q1)
            if iqr > 0:
                fence_lo, fence_hi = q1 - _TUKEY_K * iqr, q3 + _TUKEY_K * iqr
                lo, hi = max(lo, fence_lo), min(hi, fence_hi)
            return _padded_range(lo, hi)
    return _padded_range(full_lo, full_hi)


def _displayed_values(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return values inside the displayed x-axis window."""
    displayed = values[(values >= lo) & (values <= hi)]
    if len(displayed) == 0:
        return values
    return displayed


def _format_range(lo: float, hi: float) -> str:
    """Format a compact numeric range for plot annotations."""
    return f"[{lo:.3g}, {hi:.3g}]"


def _kde_density(values: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Gaussian KDE using Silverman's bandwidth rule, without scipy."""
    if len(values) > _KDE_MAX_POINTS:
        rng = np.random.default_rng(0)
        values = values[rng.choice(len(values), size=_KDE_MAX_POINTS, replace=False)]

    n = len(values)
    if n < 2:
        return None
    std = float(values.std(ddof=1))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    scale = min(std, iqr / 1.349) if iqr > 0 else std
    if scale <= 0:
        return None
    bandwidth = 0.9 * scale * n ** (-1 / 5)
    if bandwidth <= 0:
        return None

    z = (x[:, np.newaxis] - values[np.newaxis, :]) / bandwidth
    kernel = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return kernel.mean(axis=1) / bandwidth


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
    bins: int | str = "auto",
    display_percentiles: tuple[float, float] | None = None,
) -> Figure:
    """Histogram of raw prediction errors with adaptive bins and density overlays.

    Flattens all dimensions to 1-D. Errors are raw ``targets - predictions``
    values in target units; they are not scaled or normalized. Normal and KDE
    overlays use the displayed error distribution. The y-axis uses a log
    density scale: regression error distributions are commonly leptokurtic
    (a tall near-zero spike plus a much wider spread), which on a linear
    scale flattens everything but the spike to invisible.

    Args:
        predictions: Model predictions, any shape; flattened to 1-D.
        targets: Ground-truth targets, any shape; must have same total elements.
        title: Figure title.
        bins: Histogram bin strategy. ``"auto"`` delegates adaptive bin selection
            to NumPy; an integer fixes the bin count.
        display_percentiles: Optional ``(lower, upper)`` percentile window for
            the displayed x-axis. ``None`` uses the full raw min/max range.
            Always additionally clamped by a Tukey IQR fence
            (``[Q1 - 3*IQR, Q3 + 3*IQR]``) so heavy-tailed error
            distributions don't blow out the x-axis.

    Returns:
        matplotlib Figure. Caller must close it.

    Raises:
        ValueError: If flattened shapes differ.
    """
    preds_flat, tgts_flat = _flatten_and_validate(predictions, targets)

    errors = _finite_values(tgts_flat - preds_flat)
    full_lo = float(errors.min())
    full_hi = float(errors.max())
    x_lo, x_hi = _error_display_range(errors, display_percentiles)
    displayed_errors = _displayed_values(errors, x_lo, x_hi)
    mu = float(displayed_errors.mean())
    sigma = float(displayed_errors.std())

    fig, ax = plt.subplots(1, 1)
    ax.hist(displayed_errors, bins=bins, density=True, log=True, label="Errors")

    x = np.linspace(x_lo, x_hi, _KDE_POINTS)
    if sigma > 0:
        normal_curve = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, normal_curve, "r-", label="Normal fit")
    kde_curve = _kde_density(displayed_errors, x)
    if kde_curve is not None:
        ax.plot(x, kde_curve, color="black", linestyle="-", label="KDE")

    clipped_count = len(errors) - len(displayed_errors)
    range_subtitle = (
        f"full: {_format_range(full_lo, full_hi)}; view: {_format_range(x_lo, x_hi)}"
        if clipped_count
        else f"range: {_format_range(full_lo, full_hi)}"
    )
    if clipped_count:
        ax.text(
            0.99,
            0.98,
            (
                f"{clipped_count} tail values outside view\n"
                f"view: {_format_range(x_lo, x_hi)}\n"
                f"full: {_format_range(full_lo, full_hi)}"
            ),
            ha="right",
            va="top",
            transform=ax.transAxes,
        )

    ax.set_title(f"{title}\n{range_subtitle}")
    ax.set_xlabel("Error")
    ax.set_ylabel("Density (log scale)")
    ax.set_xlim(x_lo, x_hi)
    ax.legend()
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
