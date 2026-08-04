"""Training-phase figure generators."""

from __future__ import annotations

from matplotlib.figure import Figure

from dlkit.domain.analysis.figures._backend import plt  # noqa: F401


def loss_curve_figure(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    *,
    lr_values: list[float] | None = None,
    title: str = "Loss Curve",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    log_y: bool = True,
) -> Figure:
    """Line plot of train/val loss vs epoch with optional LR subplot below.

    When ``lr_values`` is provided a second axes is added below the loss axes
    sharing the same x-axis, showing learning rate vs epoch on a log scale.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses; must match length if provided.
        lr_values: Per-epoch learning rates; plotted on a second axes when given.
        title: Figure title (applied to the top axes).
        xlabel: X-axis label (applied to the bottom-most axes).
        ylabel: Y-axis label for the loss axes.
        log_y: Use log scale on loss y-axis. Defaults to True.

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

    epochs = range(1, len(train_losses) + 1)
    show_lr = lr_values is not None and len(lr_values) > 0

    if show_lr:
        fig, (ax_loss, ax_lr) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_loss = plt.subplots(1, 1)
        ax_lr = None

    ax_loss.plot(epochs, train_losses, color="blue", label="Train")
    if val_losses is not None:
        ax_loss.plot(epochs, val_losses, color="orange", label="Val")
        ax_loss.legend()
    if log_y:
        ax_loss.set_yscale("log")
    ax_loss.set_title(title)
    ax_loss.set_ylabel(ylabel)

    if show_lr and ax_lr is not None:
        lr_epochs = range(1, len(lr_values) + 1)
        ax_lr.plot(lr_epochs, lr_values, color="green")
        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("LR")
        ax_lr.set_xlabel(xlabel)
    else:
        ax_loss.set_xlabel(xlabel)

    fig.tight_layout()
    return fig
