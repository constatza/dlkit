"""Transform pipeline: per-entry named batch transformation for Lightning wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast, runtime_checkable

import torch
from loguru import logger
from tensordict import TensorDict, TensorDictBase
from torch import Tensor, nn

from dlkit.domain.transforms.base import (
    FittableTransform,
    IncrementalFittableTransform,
    InvertibleTransform,
)

from .batch_namespace import IBatchNamespaceSpec, StandardBatchNamespace


@runtime_checkable
class _FittableFromDataloader(Protocol):
    """Transform chain that supports dataloader-based fitting."""

    def fit_from_dataloader(
        self,
        dataloader: Any,
        extractor: Callable[..., Any],
    ) -> None: ...


class NamedBatchTransformer(nn.Module):
    """Applies named transform chains per entry key.

    Replaces positional ModuleList with named ModuleDict, eliminating
    fragile position-alignment requirements.

    State dict keys: `_feature_chains.<entry_name>.*` (named, stable).

    Attributes:
        _feature_chains: ModuleDict mapping feature entry names to transform chains.
        _target_chains: ModuleDict mapping target entry names to transform chains.
        _ns: Batch namespace specification (defaults to StandardBatchNamespace).
    """

    def __init__(
        self,
        feature_chains: dict[str, nn.Module],
        target_chains: dict[str, nn.Module],
        namespace_spec: IBatchNamespaceSpec | None = None,
    ) -> None:
        """Initialize with named transform chain dicts.

        Args:
            feature_chains: Dict mapping feature entry names to transform chains.
            target_chains: Dict mapping target entry names to transform chains.
            namespace_spec: Batch namespace specification; defaults to StandardBatchNamespace.
        """
        super().__init__()
        self._feature_chains = nn.ModuleDict(feature_chains)
        self._target_chains = nn.ModuleDict(target_chains)
        self._ns: IBatchNamespaceSpec = namespace_spec or StandardBatchNamespace()

    def transform(self, batch: Any) -> Any:
        """Apply transforms to all feature and target entries in the batch.

        Iterates registered chains (authoritative), not batch keys.
        Keys with no registered chain are passed through unchanged.

        Args:
            batch: Input TensorDict.

        Returns:
            Transformed TensorDict with same structure.

        Raises:
            ValueError: If a registered chain's entry is missing from batch.
        """
        fn = self._ns.feature_namespace
        tn = self._ns.target_namespace
        batch_feature_keys = set(batch[fn].keys())
        new_features: dict[str, Tensor] = {}

        for k in self._feature_chains:
            if k not in batch_feature_keys:
                raise ValueError(
                    f"Feature '{k}' required by transform chain is missing from batch. "
                    f"Available: {sorted(batch_feature_keys)}"
                )
            new_features[k] = self._feature_chains[k](batch[fn, k])

        for k in batch_feature_keys:
            if k not in new_features:
                new_features[k] = batch[fn, k]

        batch_target_keys = set(batch[tn].keys())
        new_targets: dict[str, Tensor] = {}

        for k in self._target_chains:
            if k not in batch_target_keys:
                raise ValueError(
                    f"Target '{k}' required by transform chain is missing from batch. "
                    f"Available: {sorted(batch_target_keys)}"
                )
            new_targets[k] = self._target_chains[k](batch[tn, k])

        for k in batch_target_keys:
            if k not in new_targets:
                new_targets[k] = batch[tn, k]

        return TensorDict(
            {
                fn: TensorDict(cast(Any, new_features), batch_size=batch.batch_size),
                tn: TensorDict(cast(Any, new_targets), batch_size=batch.batch_size),
            },
            batch_size=batch.batch_size,
        )

    def inverse_transform_predictions(
        self, predictions: Tensor | TensorDict, target_key: str
    ) -> Tensor | TensorDict:
        """Apply inverse target transform to predictions.

        Single-head (Tensor): looks up *target_key* in ``_target_chains``.
        Multi-head (TensorDict): applies per-key chain lookup; *target_key* ignored.

        Args:
            predictions: Normalized predictions — Tensor or TensorDict.
            target_key: Target entry name, used only for single-head Tensor case.

        Returns:
            Inverse-transformed predictions, same type as input.
        """
        match predictions:
            case torch.Tensor():
                if target_key not in self._target_chains:
                    return predictions
                chain = self._target_chains[target_key]
                if isinstance(chain, InvertibleTransform):
                    return chain.inverse_transform(predictions)
                return predictions
            case TensorDict():
                result: dict[str, Tensor | TensorDictBase] = {}
                for k, v in predictions.items():
                    match v:
                        case torch.Tensor() if k in self._target_chains:
                            chain = self._target_chains[k]
                            result[k] = (
                                chain.inverse_transform(v)
                                if isinstance(chain, InvertibleTransform)
                                else v
                            )
                        case _:
                            result[k] = cast(Tensor | TensorDictBase, v)
                return TensorDict(cast(Any, result), batch_size=predictions.batch_size)

    def fit(self, dataloader: Any) -> None:
        """Fit all fittable transforms using training data.

        Args:
            dataloader: Training DataLoader to iterate for fitting.
        """
        for namespace, chains in (
            ("features", self._feature_chains),
            ("targets", self._target_chains),
        ):
            for entry_name, chain in chains.items():
                if not isinstance(chain, FittableTransform):
                    continue

                logger.info(
                    "Fitting transform chain for {}.{} ({})",
                    namespace,
                    entry_name,
                    chain.__class__.__name__,
                )

                if isinstance(chain, _FittableFromDataloader):
                    cast(_FittableFromDataloader, chain).fit_from_dataloader(
                        dataloader,
                        lambda batch, ns=namespace, key=entry_name: batch[ns, key],
                    )
                    continue

                if isinstance(chain, IncrementalFittableTransform):
                    seen = False
                    chain.reset_fit_state()
                    for batch in dataloader:
                        chain.update_fit(batch[namespace, entry_name])
                        seen = True
                    if not seen:
                        raise ValueError("Cannot fit transforms on an empty dataloader.")
                    chain.finalize_fit()
                    continue

                if getattr(chain, "fitted", False):
                    continue

                raise TypeError(
                    f"Incremental fitting for '{chain.__class__.__name__}' is not implemented. "
                    "Remove this transform from online fit path. TODO: incremental PCA."
                )

    def is_fitted(self) -> bool:
        """Check if all fittable transforms are fitted.

        Returns:
            True if all transforms are fitted or no fittable transforms exist.
        """
        for chain in [*self._feature_chains.values(), *self._target_chains.values()]:
            if isinstance(chain, FittableTransform) and not chain.fitted:
                return False
        return True
