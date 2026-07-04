"""Tests for domain.analysis.figures — training and regression figure generators."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from dlkit.domain.analysis.figures import (
    error_histogram_figure,
    loss_curve_figure,
    parity_figure,
    residual_figure,
    residual_vs_index_figure,
)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
SMALL_N: int = 20
LARGE_N: int = 10_000
DEFAULT_MAX_POINTS: int = 5_000
FEW_EPOCHS: int = 5
MISMATCH_EXTRA: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _legend_texts(fig: Figure) -> list[str]:
    """Collect all legend label strings from every axes in *fig*.

    Args:
        fig: The matplotlib Figure to inspect.

    Returns:
        Flat list of label strings from all legend entries in the figure.
    """
    texts: list[str] = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            texts.extend(t.get_text() for t in legend.get_texts())
    return texts


def _line_count(fig: Figure) -> int:
    """Count the total number of Line2D objects across all axes in *fig*.

    Args:
        fig: The matplotlib Figure to inspect.

    Returns:
        Total number of Line2D instances.
    """
    return sum(len(ax.lines) for ax in fig.axes)


# ---------------------------------------------------------------------------
# Fixtures — data
# ---------------------------------------------------------------------------
@pytest.fixture
def train_losses_only() -> list[float]:
    """Five-epoch training loss sequence without validation losses.

    Returns:
        List of per-epoch training loss floats.
    """
    return [1.0, 0.8, 0.6, 0.4, 0.3]


@pytest.fixture
def train_and_val_losses() -> tuple[list[float], list[float]]:
    """Matching training and validation loss sequences.

    Returns:
        Tuple of (train_losses, val_losses) of equal length.
    """
    train = [1.0, 0.8, 0.6, 0.4, 0.3]
    val = [1.1, 0.9, 0.7, 0.5, 0.4]
    return train, val


@pytest.fixture
def mismatched_val_losses() -> tuple[list[float], list[float]]:
    """Training losses paired with a val list of different length.

    Returns:
        Tuple of (train_losses, val_losses) where lengths differ.
    """
    train = [1.0, 0.8, 0.6]
    val = [1.1, 0.9]  # one element short
    return train, val


@pytest.fixture
def small_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Small matched predictions/targets pair suitable for most regression figures.

    Returns:
        Tuple of (predictions, targets) both 1-D with SMALL_N elements.
    """
    rng = np.random.default_rng(42)
    preds = rng.standard_normal(SMALL_N).astype(np.float32)
    targets = preds + rng.standard_normal(SMALL_N).astype(np.float32) * 0.1
    return preds, targets


@pytest.fixture
def large_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Large matched predictions/targets pair that triggers subsampling.

    N exceeds DEFAULT_MAX_POINTS so subsampling logic is exercised.

    Returns:
        Tuple of (predictions, targets) both 1-D with LARGE_N elements.
    """
    rng = np.random.default_rng(0)
    preds = rng.standard_normal(LARGE_N).astype(np.float32)
    targets = preds + rng.standard_normal(LARGE_N).astype(np.float32) * 0.05
    return preds, targets


@pytest.fixture
def shape_mismatched_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Predictions and targets with incompatible total element counts.

    Returns:
        Tuple of arrays with different flattened lengths.
    """
    preds = np.ones(10, dtype=np.float32)
    targets = np.ones(10 + MISMATCH_EXTRA, dtype=np.float32)
    return preds, targets


@pytest.fixture
def multidim_arrays() -> tuple[np.ndarray, np.ndarray]:
    """2-D predictions and targets that must be flattened by regression figures.

    Noise is large enough to give the error histogram a non-trivial range
    when 50 bins are requested.

    Returns:
        Tuple of 2-D arrays (4, 5) — same total elements — for flattening tests.
    """
    rng = np.random.default_rng(7)
    preds = rng.standard_normal((4, 5)).astype(np.float32)
    targets = preds + rng.standard_normal((4, 5)).astype(np.float32) * 0.5
    return preds, targets


# ---------------------------------------------------------------------------
# loss_curve_figure
# ---------------------------------------------------------------------------
class TestLossCurveFigure:
    """Tests for loss_curve_figure."""

    def test_returns_figure_train_only(self, train_losses_only: list[float]) -> None:
        """loss_curve_figure returns a Figure when only train losses are given.

        Args:
            train_losses_only: Per-epoch train loss list fixture.
        """
        fig = loss_curve_figure(train_losses_only)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_one_line_without_val(self, train_losses_only: list[float]) -> None:
        """Exactly one line is plotted when val_losses is None.

        Args:
            train_losses_only: Per-epoch train loss list fixture.
        """
        fig = loss_curve_figure(train_losses_only)
        try:
            assert _line_count(fig) == 1
        finally:
            plt.close("all")

    def test_two_lines_with_val(
        self, train_and_val_losses: tuple[list[float], list[float]]
    ) -> None:
        """Exactly two lines are plotted when val_losses are provided.

        Args:
            train_and_val_losses: Matched (train, val) loss list fixture.
        """
        train, val = train_and_val_losses
        fig = loss_curve_figure(train, val)
        try:
            assert _line_count(fig) == 2
        finally:
            plt.close("all")

    def test_returns_figure_with_val(
        self, train_and_val_losses: tuple[list[float], list[float]]
    ) -> None:
        """loss_curve_figure returns a Figure when val losses are provided.

        Args:
            train_and_val_losses: Matched (train, val) loss list fixture.
        """
        train, val = train_and_val_losses
        fig = loss_curve_figure(train, val)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_empty_train_losses_raises(self) -> None:
        """Empty train_losses raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            loss_curve_figure([])

    def test_mismatched_val_losses_raises(
        self, mismatched_val_losses: tuple[list[float], list[float]]
    ) -> None:
        """val_losses length mismatch raises ValueError.

        Args:
            mismatched_val_losses: (train, val) lists of different lengths.
        """
        train, val = mismatched_val_losses
        with pytest.raises(ValueError, match="length"):
            loss_curve_figure(train, val)

    def test_custom_labels_applied(self, train_losses_only: list[float]) -> None:
        """Custom title/xlabel/ylabel kwargs are forwarded to the axes.

        Args:
            train_losses_only: Per-epoch train loss list fixture.
        """
        fig = loss_curve_figure(
            train_losses_only,
            title="My Title",
            xlabel="My X",
            ylabel="My Y",
        )
        try:
            ax = fig.axes[0]
            assert ax.get_title() == "My Title"
            assert ax.get_xlabel() == "My X"
            assert ax.get_ylabel() == "My Y"
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# parity_figure
# ---------------------------------------------------------------------------
class TestParityFigure:
    """Tests for parity_figure."""

    def test_returns_figure(self, small_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """parity_figure returns a matplotlib Figure.

        Args:
            small_arrays: Matched (predictions, targets) fixture.
        """
        preds, targets = small_arrays
        fig = parity_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_r2_annotation_in_legend(self, small_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """The legend contains an R² annotation string.

        Args:
            small_arrays: Matched (predictions, targets) fixture.
        """
        preds, targets = small_arrays
        fig = parity_figure(preds, targets)
        try:
            texts = _legend_texts(fig)
            assert any("R²" in t for t in texts), f"No R² found in legend texts: {texts}"
        finally:
            plt.close("all")

    def test_shape_mismatch_raises(
        self, shape_mismatched_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shape mismatch raises ValueError.

        Args:
            shape_mismatched_arrays: Arrays with differing flattened lengths.
        """
        preds, targets = shape_mismatched_arrays
        with pytest.raises(ValueError, match="same total number of elements"):
            parity_figure(preds, targets)

    def test_large_input_subsampled(self, large_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """parity_figure does not raise with large N (subsampling path exercised).

        Args:
            large_arrays: Arrays with LARGE_N elements, exceeding max_points.
        """
        preds, targets = large_arrays
        fig = parity_figure(preds, targets, max_points=DEFAULT_MAX_POINTS)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_multidim_arrays_flattened(
        self, multidim_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """2-D arrays are accepted and flattened without error.

        Args:
            multidim_arrays: 2-D arrays that share the same total element count.
        """
        preds, targets = multidim_arrays
        fig = parity_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# residual_figure
# ---------------------------------------------------------------------------
class TestResidualFigure:
    """Tests for residual_figure."""

    def test_returns_figure(self, small_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """residual_figure returns a matplotlib Figure.

        Args:
            small_arrays: Matched (predictions, targets) fixture.
        """
        preds, targets = small_arrays
        fig = residual_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_shape_mismatch_raises(
        self, shape_mismatched_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shape mismatch raises ValueError.

        Args:
            shape_mismatched_arrays: Arrays with differing flattened lengths.
        """
        preds, targets = shape_mismatched_arrays
        with pytest.raises(ValueError, match="same total number of elements"):
            residual_figure(preds, targets)

    def test_large_input_subsampled(self, large_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """residual_figure does not raise with large N (subsampling path).

        Args:
            large_arrays: Arrays with LARGE_N elements, exceeding max_points.
        """
        preds, targets = large_arrays
        fig = residual_figure(preds, targets, max_points=DEFAULT_MAX_POINTS)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_multidim_arrays_flattened(
        self, multidim_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """2-D arrays are accepted and flattened without error.

        Args:
            multidim_arrays: 2-D arrays that share the same total element count.
        """
        preds, targets = multidim_arrays
        fig = residual_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# error_histogram_figure
# ---------------------------------------------------------------------------
class TestErrorHistogramFigure:
    """Tests for error_histogram_figure."""

    def test_returns_figure(self, small_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """error_histogram_figure returns a matplotlib Figure.

        Args:
            small_arrays: Matched (predictions, targets) fixture.
        """
        preds, targets = small_arrays
        fig = error_histogram_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_shape_mismatch_raises(
        self, shape_mismatched_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shape mismatch raises ValueError.

        Args:
            shape_mismatched_arrays: Arrays with differing flattened lengths.
        """
        preds, targets = shape_mismatched_arrays
        with pytest.raises(ValueError, match="same total number of elements"):
            error_histogram_figure(preds, targets)

    def test_custom_bins_accepted(self, small_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """Custom bins kwarg is accepted without error.

        Args:
            small_arrays: Matched (predictions, targets) fixture.
        """
        preds, targets = small_arrays
        fig = error_histogram_figure(preds, targets, bins=10)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_multidim_arrays_flattened(
        self, multidim_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """2-D arrays are accepted and flattened without error.

        Args:
            multidim_arrays: 2-D arrays that share the same total element count.
        """
        preds, targets = multidim_arrays
        fig = error_histogram_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# residual_vs_index_figure
# ---------------------------------------------------------------------------
class TestResidualVsIndexFigure:
    """Tests for residual_vs_index_figure."""

    def test_returns_figure(self, small_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """residual_vs_index_figure returns a matplotlib Figure.

        Args:
            small_arrays: Matched (predictions, targets) fixture.
        """
        preds, targets = small_arrays
        fig = residual_vs_index_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_shape_mismatch_raises(
        self, shape_mismatched_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shape mismatch raises ValueError.

        Args:
            shape_mismatched_arrays: Arrays with differing flattened lengths.
        """
        preds, targets = shape_mismatched_arrays
        with pytest.raises(ValueError, match="same total number of elements"):
            residual_vs_index_figure(preds, targets)

    def test_large_input_subsampled(self, large_arrays: tuple[np.ndarray, np.ndarray]) -> None:
        """residual_vs_index_figure does not raise with large N (subsampling path).

        Args:
            large_arrays: Arrays with LARGE_N elements, exceeding max_points.
        """
        preds, targets = large_arrays
        fig = residual_vs_index_figure(preds, targets, max_points=DEFAULT_MAX_POINTS)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")

    def test_multidim_arrays_flattened(
        self, multidim_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """2-D arrays are accepted and flattened without error.

        Args:
            multidim_arrays: 2-D arrays that share the same total element count.
        """
        preds, targets = multidim_arrays
        fig = residual_vs_index_figure(preds, targets)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close("all")
