"""Transform-fitting precondition shared by training-tuning and Lightning adapters.

Defined in the training layer (not adapters) because dlkit's curated package
graph requires ``engine.training`` to never depend on ``engine.adapters``;
``engine.adapters`` already depends on ``engine.training`` (e.g. for
optimization controllers), so adapters imports this module, not the reverse.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IFittableTransformer(Protocol):
    """Minimal fit lifecycle a batch transformer must expose to be fitted here."""

    def fit(self, dataloader: Iterable[Any], device: Any | None = None) -> None:
        """Fit using training data.

        Args:
            dataloader: Training dataloader to iterate for fitting.
            device: Optional target device for the fitted buffers.
        """
        ...

    def is_fitted(self) -> bool:
        """Check whether fitting has already happened.

        Returns:
            True if already fitted (or nothing needs fitting).
        """
        ...


@runtime_checkable
class IHasBatchTransformer(Protocol):
    """Single responsibility: expose the batch transformer for callers outside the
    Lightning callback lifecycle (e.g. the LR tuner) that must ensure transforms
    are fitted before invoking Lightning APIs that may not run dlkit's callbacks.
    """

    @property
    def batch_transformer(self) -> IFittableTransformer:
        """The model's batch transformer.

        Returns:
            The configured fittable batch transformer.
        """
        ...


def fit_if_needed(
    transformer: IFittableTransformer, dataloader: Iterable[Any], device: Any | None = None
) -> None:
    """Fit a batch transformer from a dataloader if it is fittable and not yet fitted.

    Idempotent precondition check shared by any caller that needs fitted
    transforms before the first forward pass — both Lightning's normal
    ``on_fit_start`` callback path and callers that bypass Lightning's callback
    lifecycle entirely (e.g. the LR tuner, whose scan loop strips callbacks).

    Args:
        transformer: Candidate batch transformer; a no-op if not fittable.
        dataloader: Training dataloader to iterate for fitting.
        device: Optional target device for the fitted buffers.
    """
    if not isinstance(transformer, IFittableTransformer):
        return
    if transformer.is_fitted():
        return
    transformer.fit(dataloader, device=device)
