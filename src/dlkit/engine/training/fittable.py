"""One-shot, non-gradient model-fitting contract.

A ``Fittable`` model's entire "training" is a single deterministic call —
e.g. a thin-SVD into a ``register_buffer`` — with no epochs, optimizer, or
loss. This mirrors the shape of ``IFittableTransformer``
(``engine.training.transform_fitting``), but is a distinct protocol: that one
fits an input-side batch transformer from inside the build phase, while this
one fits a top-level model from inside :class:`OneShotFitExecutor`, deferred
until after ``RuntimeComponents`` has been assembled.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Fittable(Protocol):
    """Structural protocol for models fit via one deterministic, non-gradient call."""

    def fit(self, dataloader: Iterable[Any]) -> None:
        """Fit the model from a single pass (or as many as needed) over a dataloader.

        Args:
            dataloader: Training dataloader to iterate for fitting.
        """
        ...

    def is_fitted(self) -> bool:
        """Check whether fitting has already happened.

        Returns:
            True if the model has already been fit.
        """
        ...
