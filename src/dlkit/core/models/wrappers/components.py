"""Protocol implementations for Lightning wrapper components.

Provides concrete implementations of the SOLID protocols defined in protocols.py:
- StandardModelInvoker: extracts model_input features from TensorDict in config order
- RoutedLossComputer: routes batch keys to loss function kwargs
- MetricRoute: value object for per-metric routing config
- RoutedMetricsUpdater: routes each metric to its configured target/extra inputs
- NamedBatchTransformer: applies named transform chains per entry key
- WrapperCheckpointMetadata: serialisation-only wrapper metadata value object
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import Metric

from dlkit.core.training.transforms.base import FittableTransform, InvertibleTransform
from dlkit.tools.config.components.model_components import LossInputRef, MetricInputRef


def _parse_key(key: str) -> tuple[str, str]:
    """Parse 'namespace.entry_name' key into (namespace, entry_name).

    Args:
        key: Key string in 'features.name' or 'targets.name' format.

    Returns:
        Tuple of (namespace, entry_name).

    Raises:
        ValueError: If key format is invalid.
    """
    parts = key.split(".", 1)
    if len(parts) != 2 or parts[0] not in ("features", "targets"):
        raise ValueError(
            f"key must be 'features.<entry_name>' or 'targets.<entry_name>', got '{key}'"
        )
    return parts[0], parts[1]


class StandardModelInvoker:
    """Extracts model-input features from TensorDict in config-insertion order.

    Models receive features positionally in config-insertion order. Entry names
    are independent of the model's forward() parameter names. The user controls
    invocation order by the order of Feature entries in config.

    Attributes:
        _feature_keys: Ordered tuple of model_input=True feature names.
    """

    def __init__(self, feature_keys: tuple[str, ...]) -> None:
        """Initialize invoker with ordered feature keys.

        Args:
            feature_keys: Names of model_input=True features in config-insertion order.
        """
        self._feature_keys = feature_keys

    def invoke(self, model: nn.Module, batch: Any) -> Tensor:
        """Invoke model with positionally ordered feature tensors.

        Args:
            model: PyTorch model to invoke.
            batch: TensorDict with 'features' namespace.

        Returns:
            Model output tensor.

        Raises:
            ValueError: If no model-input features configured or batch missing keys.
        """
        if not self._feature_keys:
            raise ValueError("No model-input features configured")

        batch_feature_keys = set(batch["features"].keys())
        missing = [k for k in self._feature_keys if k not in batch_feature_keys]
        if missing:
            raise ValueError(
                f"StandardModelInvoker: batch missing expected feature keys {missing}. "
                f"Available: {sorted(batch_feature_keys)}"
            )

        tensors = tuple(batch["features", k] for k in self._feature_keys)
        match len(tensors):
            case 0:
                raise ValueError("No model-input features configured")
            case 1:
                return model(tensors[0])
            case _:
                return model(*tensors)


class RoutedLossComputer:
    """Routes batch keys to loss function kwargs per LossComponentSettings.

    Computes loss by extracting the configured target from the batch and
    passing any extra inputs as keyword arguments.

    Attributes:
        _loss_fn: The loss function callable.
        _target_ns: Target namespace ('targets').
        _target_name: Target entry name.
        _extra: Extra input refs for additional kwargs.
    """

    def __init__(
        self,
        loss_fn: Callable,
        target_key: str | None,
        default_target_key: str,
        extra_inputs: tuple[LossInputRef, ...] = (),
    ) -> None:
        """Initialize with routing configuration.

        Args:
            loss_fn: Loss function callable.
            target_key: Batch key in 'namespace.entry_name' format, or None.
            default_target_key: Name of first target entry (used when target_key is None).
            extra_inputs: Extra kwargs to route from batch.
        """
        self._loss_fn = loss_fn
        effective_key = target_key or f"targets.{default_target_key}"
        self._target_ns, self._target_name = _parse_key(effective_key)
        self._extra = extra_inputs

    def compute(self, predictions: Tensor, batch: Any) -> Tensor:
        """Compute loss from predictions and named batch.

        Args:
            predictions: Model output tensor.
            batch: TensorDict containing features and targets.

        Returns:
            Scalar loss tensor.
        """
        target = batch[self._target_ns, self._target_name].to(dtype=predictions.dtype)
        extra_kwargs = {
            ref.arg: batch[_parse_key(ref.key)]
            for ref in self._extra
        }
        return self._loss_fn(predictions, target, **extra_kwargs)


@dataclass(frozen=True)
class MetricRoute:
    """Value object carrying per-metric routing configuration.

    Attributes:
        metric: The torchmetrics Metric to update.
        target_ns: Target namespace ('targets').
        target_name: Target entry name.
        extra_inputs: Extra input refs for additional kwargs.
    """

    metric: Metric
    target_ns: str
    target_name: str
    extra_inputs: tuple[MetricInputRef, ...]


class RoutedMetricsUpdater:
    """Routes each metric to its configured target key and extra inputs.

    Does NOT use MetricCollection.update() because that broadcasts the same
    target to all metrics — per-metric target routing requires individual calls.

    Attributes:
        _routes: Dict mapping stage ('val', 'test') to list of MetricRoute.
    """

    def __init__(
        self,
        val_routes: list[MetricRoute],
        test_routes: list[MetricRoute],
    ) -> None:
        """Initialize with per-stage metric routes.

        Args:
            val_routes: Metric routes for validation stage.
            test_routes: Metric routes for test stage.
        """
        self._routes: dict[str, list[MetricRoute]] = {
            "val": val_routes,
            "test": test_routes,
        }

    def update(self, predictions: Tensor, batch: Any, stage: str) -> None:
        """Update metrics for the given stage.

        Args:
            predictions: Model output tensor.
            batch: TensorDict containing features and targets.
            stage: Stage identifier ('val' or 'test').
        """
        for route in self._routes.get(stage, []):
            target = batch[route.target_ns, route.target_name].to(dtype=predictions.dtype)
            extra = {ref.arg: batch[_parse_key(ref.key)] for ref in route.extra_inputs}
            route.metric.update(predictions, target, **extra)

    def compute(self, stage: str) -> dict[str, Any]:
        """Compute accumulated metric values for the given stage.

        Args:
            stage: Stage identifier ('val' or 'test').

        Returns:
            Dictionary mapping metric class names to computed values.
        """
        return {
            type(r.metric).__name__: r.metric.compute()
            for r in self._routes.get(stage, [])
        }

    def reset(self, stage: str) -> None:
        """Reset metric state for the given stage.

        Args:
            stage: Stage identifier ('val' or 'test').
        """
        for route in self._routes.get(stage, []):
            route.metric.reset()


class NamedBatchTransformer(nn.Module):
    """Applies named transform chains per entry key.

    Replaces positional ModuleList with named ModuleDict, eliminating
    fragile position-alignment requirements.

    State dict keys: `_feature_chains.<entry_name>.*` (named, stable).

    Attributes:
        _feature_chains: ModuleDict mapping feature entry names to transform chains.
        _target_chains: ModuleDict mapping target entry names to transform chains.
    """

    def __init__(
        self,
        feature_chains: dict[str, nn.Module],
        target_chains: dict[str, nn.Module],
    ) -> None:
        """Initialize with named transform chain dicts.

        Args:
            feature_chains: Dict mapping feature entry names to transform chains.
            target_chains: Dict mapping target entry names to transform chains.
        """
        super().__init__()
        self._feature_chains = nn.ModuleDict(feature_chains)
        self._target_chains = nn.ModuleDict(target_chains)

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
        from tensordict import TensorDict

        batch_feature_keys = set(batch["features"].keys())
        new_features: dict[str, Tensor] = {}

        # Apply registered chains (authoritative order)
        for k in self._feature_chains:
            if k not in batch_feature_keys:
                raise ValueError(
                    f"Feature '{k}' required by transform chain is missing from batch. "
                    f"Available: {sorted(batch_feature_keys)}"
                )
            new_features[k] = self._feature_chains[k](batch["features", k])

        # Pass through keys with no chain (context features etc.)
        for k in batch_feature_keys:
            if k not in new_features:
                new_features[k] = batch["features", k]

        batch_target_keys = set(batch["targets"].keys())
        new_targets: dict[str, Tensor] = {}

        for k in self._target_chains:
            if k not in batch_target_keys:
                raise ValueError(
                    f"Target '{k}' required by transform chain is missing from batch. "
                    f"Available: {sorted(batch_target_keys)}"
                )
            new_targets[k] = self._target_chains[k](batch["targets", k])

        for k in batch_target_keys:
            if k not in new_targets:
                new_targets[k] = batch["targets", k]

        return TensorDict(
            {
                "features": TensorDict(new_features, batch_size=batch.batch_size),
                "targets": TensorDict(new_targets, batch_size=batch.batch_size),
            },
            batch_size=batch.batch_size,
        )

    def inverse_transform_predictions(self, predictions: Tensor, target_key: str) -> Tensor:
        """Apply inverse target transform to predictions.

        Used in predict_step to convert predictions back to original data space.

        Args:
            predictions: Model output tensor in transformed space.
            target_key: Name of the target entry whose chain to invert.

        Returns:
            Predictions in original (untransformed) space.
        """
        if target_key not in self._target_chains:
            return predictions
        chain = self._target_chains[target_key]
        if isinstance(chain, InvertibleTransform):
            return chain.inverse_transform(predictions)
        return predictions

    def fit(self, dataloader: Any) -> None:
        """Fit all fittable transforms using training data.

        Accumulates full training data per entry name, then fits once.

        Args:
            dataloader: Training DataLoader to iterate for fitting.

        Note:
            Known limitation: full-data accumulation in memory. OOM risk for
            large datasets. Future: IIncrementalFittableTransform for streaming.
        """
        feat_buffers: dict[str, list[Tensor]] = {k: [] for k in self._feature_chains}
        tgt_buffers: dict[str, list[Tensor]] = {k: [] for k in self._target_chains}

        for batch in dataloader:
            for k in feat_buffers:
                feat_buffers[k].append(batch["features", k])
            for k in tgt_buffers:
                tgt_buffers[k].append(batch["targets", k])

        for k, chain in self._feature_chains.items():
            if isinstance(chain, FittableTransform) and feat_buffers.get(k):
                chain.fit(torch.cat(feat_buffers[k], dim=0))

        for k, chain in self._target_chains.items():
            if isinstance(chain, FittableTransform) and tgt_buffers.get(k):
                chain.fit(torch.cat(tgt_buffers[k], dim=0))

    def is_fitted(self) -> bool:
        """Check if all fittable transforms are fitted.

        Returns:
            True if all transforms are fitted or no fittable transforms exist.
        """
        for chain in [*self._feature_chains.values(), *self._target_chains.values()]:
            if isinstance(chain, FittableTransform) and not chain.fitted:
                return False
        return True


class _NullModelInvoker:
    """No-op model invoker for wrappers that override all step methods.

    Used by wrappers (e.g., GraphLightningWrapper) that pass their own
    batch format to the model and never delegate to the base step methods.
    """

    def invoke(self, model: nn.Module, batch: Any) -> Tensor:
        """Raise if accidentally called.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "_NullModelInvoker.invoke() was called unexpectedly. "
            "Subclass must override all step methods when using null invoker."
        )


class _NullLossComputer:
    """No-op loss computer for wrappers that override all step methods."""

    def compute(self, predictions: Tensor, batch: Any) -> Tensor:
        """Raise if accidentally called.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "_NullLossComputer.compute() was called unexpectedly. "
            "Subclass must override all step methods when using null computer."
        )


class _NullMetricsUpdater:
    """No-op metrics updater for wrappers that handle metrics directly."""

    def update(self, predictions: Tensor, batch: Any, stage: str) -> None:
        """No-op update."""

    def compute(self, stage: str) -> dict[str, Any]:
        """Return empty metrics dict.

        Args:
            stage: Stage identifier.

        Returns:
            Empty dict.
        """
        return {}

    def reset(self, stage: str) -> None:
        """No-op reset."""


@dataclass(frozen=True)
class WrapperCheckpointMetadata:
    """Value object carrying serialisation-only metadata for checkpoint persistence.

    Keeps the wrapper __init__ clean by separating checkpoint metadata
    from the operational constructor arguments.

    Attributes:
        model_settings: Model configuration for checkpoint reconstruction.
        wrapper_settings: Wrapper configuration for checkpoint reconstruction.
        entry_configs: Data entry configurations in config order.
        feature_names: Ordered feature names for inference restore.
        predict_target_key: Name of target whose chain is inverted at predict time.
        shape_summary: Shape summary from dataset inference, or None.
    """

    model_settings: Any
    wrapper_settings: Any
    entry_configs: tuple[Any, ...]
    feature_names: tuple[str, ...]
    predict_target_key: str
    shape_summary: Any | None = None
