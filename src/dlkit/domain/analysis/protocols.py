"""Protocols for the domain.analysis sub-package."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from matplotlib.figure import Figure


@runtime_checkable
class IFigureGenerator(Protocol):
    """Generates a matplotlib Figure from predictions and targets arrays.

    Implement this protocol to add a custom plot type without modifying
    any engine callback code (Open-Closed Principle).

    Attributes:
        name: Artifact filename stem for the uploaded PNG,
            e.g. ``"parity_plot"`` → ``parity_plot.png``.
    """

    name: str

    def generate(self, predictions: np.ndarray, targets: np.ndarray) -> Figure:
        """Generate a visualization figure.

        Both ``predictions`` and ``targets`` will be 1-D numpy arrays
        (already flattened by the caller).

        Args:
            predictions: Model predictions, shape ``(N,)``.
            targets: Ground-truth targets, shape ``(N,)``.

        Returns:
            matplotlib Figure ready for saving. Caller is responsible for closing it.
        """
        ...
