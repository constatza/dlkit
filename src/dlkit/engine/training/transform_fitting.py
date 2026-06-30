"""Transform-fitting precondition for the build phase.

Fitting runs once, deterministically, before any Lightning ``Trainer``/``Tuner``
object exists — see ``IBuildStrategy.build()`` in
``engine.workflows.factories.build_strategy``. This removes the dependency on
Lightning's callback lifecycle entirely (Lightning's ``Tuner.lr_find()`` strips
``trainer.callbacks`` before scanning, so a callback-based fit trigger cannot be
relied upon).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IFittableTransformer(Protocol):
    """Minimal fit lifecycle a batch transformer must expose to be fitted here."""

    def fit(self, dataloader: Iterable[Any]) -> None:
        """Fit using training data.

        Args:
            dataloader: Training dataloader to iterate for fitting.
        """
        ...

    def is_fitted(self) -> bool:
        """Check whether fitting has already happened.

        Returns:
            True if already fitted (or nothing needs fitting).
        """
        ...


@runtime_checkable
class IHasTrainDataloader(Protocol):
    """Single responsibility: expose a train dataloader for build-phase fitting."""

    def train_dataloader(self) -> Iterable[Any]:
        """Return the training dataloader.

        Returns:
            Iterable yielding training batches.
        """
        ...


def fit_if_needed(transformer: IFittableTransformer, dataloader: Iterable[Any]) -> None:
    """Fit a batch transformer from a dataloader if it is fittable and not yet fitted.

    Idempotent precondition check: a no-op when the transformer isn't fittable
    or is already fitted.

    Args:
        transformer: Candidate batch transformer; a no-op if not fittable.
        dataloader: Training dataloader to iterate for fitting.
    """
    if not isinstance(transformer, IFittableTransformer):
        return
    if transformer.is_fitted():
        return
    transformer.fit(dataloader)


def fit_transforms_if_needed(model: object, datamodule: object) -> None:
    """Fit a model's batch transformer from its datamodule's train split, if needed.

    The single fit-trigger call site for the build phase — called from
    ``IBuildStrategy.build()`` after ``RuntimeComponents`` is assembled, before
    any ``Trainer``/``Tuner`` object exists. Fitting on whatever device the raw
    dataset tensors are naturally on (typically CPU); Lightning's normal
    ``model.to(device)`` submodule traversal handles accelerator placement
    afterward, since the batch transformer is a registered ``nn.Module``
    submodule — the same idiom as BatchNorm running stats.

    A no-op for models/datamodules that don't expose a fittable batch
    transformer or a train dataloader (e.g. ``GraphBuildStrategy``, whose
    wrapper has no transform pipeline at all).

    Deliberately checks ``batch_transformer`` via ``hasattr`` rather than
    ``isinstance(model, IHasBatchTransformer)``: Python's ``runtime_checkable``
    Protocol isinstance check resolves property members with
    ``inspect.getattr_static``, which finds an inherited property *descriptor*
    on the class without calling its getter — so it would report a wrapper as
    "has batch_transformer" even when the getter raises ``AttributeError`` on
    that particular instance (e.g. a wrapper base class defines the property,
    but a subclass never sets the backing attribute).

    Args:
        model: Candidate model; a no-op unless it actually has a working
            ``batch_transformer`` attribute.
        datamodule: Candidate datamodule; a no-op unless it implements
            ``IHasTrainDataloader``.
    """
    if not hasattr(model, "batch_transformer"):
        return
    if not isinstance(datamodule, IHasTrainDataloader):
        return
    transformer = model.batch_transformer
    if not isinstance(transformer, IFittableTransformer):
        return
    fit_if_needed(transformer, datamodule.train_dataloader())
