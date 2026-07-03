"""Training-phase figure generators."""

from __future__ import annotations

from matplotlib.figure import Figure

from dlkit.domain.analysis.figures._backend import plt  # noqa: F401


def loss_curve_figure(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    *,
    title: str = "Loss Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> Figure:
    """Line plot of train/val loss vs epoch.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses; must match length if provided.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        matplotlib Figure. Caller must close it.

    Raises:
        ValueError: If train_losses is empty or val_losses length mismatches.
    """
    if not train_losses:
        raise ValueError("train_losses must not be empty")
    if val_losses is not None and len(val_losses) != len(train_losses):
        raise ValueError(
            f"val_losses length ({len(val_losses)}) must match "
            f"train_losses length ({len(train_losses)})"
        )

    fig, ax = plt.subplots(1, 1)
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="blue", label="Train")
    if val_losses is not None:
        ax.plot(epochs, val_losses, color="orange", label="Val")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
