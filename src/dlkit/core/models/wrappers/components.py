"""Protocol implementations for Lightning wrapper components.

Provides concrete implementations of the SOLID protocols defined in protocols.py:
- ModelOutputSpec: declares model output key paths for TensorDictModule routing
- TensorDictModelInvoker: invokes model via cached TensorDictModule with dispatch wrapper
- _build_invoker_from_entries: factory resolving model_input semantics into invoker
- RoutedLossComputer: routes batch keys to loss function kwargs
- MetricRoute: value object for per-metric routing config
- RoutedMetricsUpdater: routes each metric to its configured target/extra inputs
- NamedBatchTransformer: applies named transform chains per entry key
- WrapperCheckpointMetadata: serialisation-only wrapper metadata value object
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

import torch
import torch.nn as nn
from loguru import logger
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import Tensor
from torchmetrics import Metric

from dlkit.core.training.transforms.base import (
    FittableTransform,
    IncrementalFittableTransform,
    InvertibleTransform,
)
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


# ---------------------------------------------------------------------------
# ModelOutputSpec — declares named output key paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelOutputSpec:
    """Declares model output key paths for TensorDictModule routing.

    Controls how model forward() outputs are written into the batch TensorDict.
    Each position in the model's return tuple maps to a declared key path.

    Attributes:
        prediction_key: Top-level key written for the first (prediction) output.
        latent_keys: Ordered tuple of key paths for additional model outputs.
            Use ``("latents", "mu")`` to write ``batch["latents"]["mu"]``.
            When empty the model must return a single Tensor (no latents).

    Example:
        Standard model (single output)::

            ModelOutputSpec()  # writes batch["predictions"]

        VAE model (reconstruction + latent parameters)::

            ModelOutputSpec(latent_keys=(("latents", "mu"), ("latents", "logvar")))
            # model returns (recon, mu, logvar)
            # → batch["predictions"] = recon
            # → batch["latents"]["mu"] = mu
            # → batch["latents"]["logvar"] = logvar
    """

    prediction_key: str = "predictions"
    latent_keys: tuple[str | tuple[str, str], ...] = ()

    def all_out_keys(self) -> list[str | tuple[str, str]]:
        """Return all TensorDictModule out_keys in positional order.

        Returns:
            List starting with ``prediction_key`` followed by ``latent_keys``.
        """
        return [self.prediction_key, *self.latent_keys]


# ---------------------------------------------------------------------------
# TensorDictModelInvoker — cached TensorDictModule with dispatch wrapper
# ---------------------------------------------------------------------------


class TensorDictModelInvoker:
    """Invokes a model via a cached TensorDictModule and enriches the batch with outputs.

    Supports positional args, keyword args, and mixed dispatch in a single call.
    One ``TensorDictModule`` is built at construction time and reused across all
    batches.  The actual model is threaded through a mutable cell so the closure
    (and therefore the ``TensorDictModule``) never needs to be rebuilt.

    Dispatch mapping (built by ``_build_invoker_from_entries``):

    - *positional* (``in_keys``): tensors passed as positional args — model receives
      them in the declared order as ``model(*pos_tensors)``.
    - *keyword* (``kwarg_in_keys``): tensors passed as named kwargs — model receives
      them as ``model(kwarg_name=tensor, ...)``.
    - Mixed: positional args come first, then keyword args.

    After ``invoke()``, callers read ``batch["predictions"]`` (and any latent keys
    from ``output_spec.latent_keys``).

    Attributes:
        _output_spec: Output key specification.
        _out_keys: TensorDictModule-format output keys.
        _td_module: Cached ``TensorDictModule`` built once at construction.
        _model_cell: Mutable ``[model | None]`` list updated before each invoke.

    Args:
        in_keys: Ordered positional key paths, e.g.
            ``[("features", "x"), ("features", "z")]``.
        output_spec: Declares prediction and latent output key paths.
            Defaults to ``ModelOutputSpec()`` (single ``"predictions"`` key).
        kwarg_in_keys: Ordered mapping ``kwarg_name → batch_key`` for named
            dispatch, e.g. ``{"edge_attr": ("features", "edge_attr")}``.
    """

    def __init__(
        self,
        in_keys: list[str | tuple[str, str]],
        output_spec: ModelOutputSpec | None = None,
        kwarg_in_keys: dict[str, str | tuple[str, str]] | None = None,
    ) -> None:
        self._output_spec = output_spec or ModelOutputSpec()
        self._out_keys = self._output_spec.all_out_keys()
        n_pos: int = len(in_keys)
        kwarg_names: list[str] = list((kwarg_in_keys or {}).keys())
        all_in_keys: list[str | tuple[str, str]] = in_keys + list((kwarg_in_keys or {}).values())
        self._model_cell: list[nn.Module | None] = [None]

        cell = self._model_cell

        def _dispatch(*args: Any) -> Any:
            model = cell[0]
            assert model is not None, "model_cell must be set before dispatch"
            return model(*args[:n_pos], **dict(zip(kwarg_names, args[n_pos:])))

        self._td_module = TensorDictModule(_dispatch, in_keys=all_in_keys, out_keys=self._out_keys)

    @property
    def _in_keys(self) -> list[str | tuple[str, str]]:
        """All input keys passed to TensorDictModule (positional + kwarg values).

        Returns:
            List of nested key paths in the order ``TensorDictModule`` extracts them.
        """
        return self._td_module.in_keys  # type: ignore[return-value]

    def invoke(self, model: nn.Module, batch: TensorDict) -> TensorDict:
        """Invoke *model* and return *batch* enriched with output keys.

        Sets the model reference in the mutable cell, then calls the cached
        ``TensorDictModule`` which extracts inputs, dispatches to *model*, and
        writes outputs back into the batch.

        Args:
            model: PyTorch model to invoke.
            batch: TensorDict with a ``"features"`` namespace.

        Returns:
            Enriched TensorDict with ``"predictions"`` key (and any configured
            latent keys) added in-place.
        """
        self._model_cell[0] = model
        return self._td_module(batch)


# ---------------------------------------------------------------------------
# _build_invoker_from_entries — factory resolving model_input ordering
# ---------------------------------------------------------------------------


def _classify_feature_entries(
    feature_entries: list[Any],
) -> tuple[list[tuple[float, str]], dict[str, str]]:
    """Classify feature entries into positional and kwarg dispatch groups.

    Resolves the ``model_input`` field on each entry:

    - ``model_input=True``: kwarg using entry name as key.
    - ``model_input=int`` / ``"0"/"1"/...``: positional at that index.
    - ``model_input="name"`` (non-digit): kwarg with custom name.
    - ``model_input=False`` / ``None``: excluded.

    Args:
        feature_entries: Feature DataEntry objects in config-insertion order.

    Returns:
        Tuple of ``(positional, kwarg_map)`` where:

        - *positional*: ``[(sort_key, entry_name), ...]`` **unsorted** list.
        - *kwarg_map*: ``{kwarg_name: entry_name}`` dict.
    """
    positional: list[tuple[float, str]] = []
    kwarg_map: dict[str, str] = {}  # kwarg_name → entry_name

    for entry in feature_entries:
        mi = getattr(entry, "model_input", True)
        name: str | None = getattr(entry, "name", None)
        if name is None or mi is False or mi is None:
            continue
        elif mi is True:
            kwarg_map[name] = name  # entry name as kwarg key
        elif isinstance(mi, int) and not isinstance(mi, bool):
            positional.append((float(mi), name))  # explicit int positional
        elif isinstance(mi, str) and mi.isdigit():
            positional.append((float(mi), name))  # digit-string positional
        elif isinstance(mi, str):
            kwarg_map[mi] = name  # custom kwarg name

    return positional, kwarg_map


def _build_invoker_from_entries(
    feature_entries: list[Any],
    output_spec: ModelOutputSpec | None = None,
) -> TensorDictModelInvoker:
    """Build a TensorDictModelInvoker from feature entry configurations.

    Resolves the ``model_input`` field on each entry to determine how each
    feature tensor is dispatched to ``model.forward()``:

    - ``model_input=True`` (default): passed as **kwarg** using the entry name
      as the kwarg key — ``model(entry_name=tensor)``.
    - ``model_input=int`` or ``model_input="0"/"1"/...``: **positional** arg at
      the given index. Features are sorted by index before building the invoker.
    - ``model_input="name"`` (non-digit identifier): passed as **kwarg** with
      the given name — ``model(name=tensor)``. Decouples kwarg name from entry name.
    - ``model_input=False`` / ``None``: **excluded** from the model call entirely.

    Mixed dispatch (some positional, some kwarg) is supported — positional args
    come before kwarg args in the ``TensorDictModule`` extraction order.

    Args:
        feature_entries: Feature DataEntry objects in config-insertion order.
        output_spec: Output key spec for the invoker.  Defaults to
            ``ModelOutputSpec()`` (single ``"predictions"`` output).

    Returns:
        Configured ``TensorDictModelInvoker`` ready for use in a wrapper.

    Raises:
        ValueError: If no features are configured as model inputs.
    """
    positional, kwarg_map = _classify_feature_entries(feature_entries)
    positional.sort(key=lambda x: x[0])
    positional_in_keys: list[str | tuple[str, str]] = [("features", name) for _, name in positional]
    kwarg_in_keys: dict[str, str | tuple[str, str]] = {
        kw: ("features", entry_name) for kw, entry_name in kwarg_map.items()
    }

    if not positional_in_keys and not kwarg_in_keys:
        raise ValueError(
            "No model-input features found. Configure at least one Feature with "
            "model_input=True (kwarg by entry name), an int/digit-string positional "
            "index, or a kwarg name string."
        )
    return TensorDictModelInvoker(
        in_keys=positional_in_keys,
        output_spec=output_spec,
        kwarg_in_keys=kwarg_in_keys or None,
    )


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
        extra_kwargs = {ref.arg: batch[_parse_key(ref.key)] for ref in self._extra}
        return self._loss_fn(predictions, target, **extra_kwargs)


@dataclass(frozen=True, slots=True, kw_only=True)
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
        return {type(r.metric).__name__: r.metric.compute() for r in self._routes.get(stage, [])}

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
                result: dict[str, Tensor | TensorDict] = {}
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
                            result[k] = v
                return TensorDict(result, batch_size=predictions.batch_size)

    def fit(self, dataloader: Any) -> None:
        """Fit all fittable transforms using training data.

        Args:
            dataloader: Training DataLoader to iterate for fitting.
        """
        for namespace, chains in (("features", self._feature_chains), ("targets", self._target_chains)):
            for entry_name, chain in chains.items():
                if not isinstance(chain, FittableTransform):
                    continue

                logger.info(
                    "Fitting transform chain for {}.{} ({})",
                    namespace,
                    entry_name,
                    chain.__class__.__name__,
                )

                if hasattr(chain, "fit_from_dataloader"):
                    chain.fit_from_dataloader(
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


class _NullModelInvoker:
    """No-op model invoker for wrappers that override all step methods.

    Used by wrappers (e.g., GraphLightningWrapper) that pass their own
    batch format to the model and never delegate to the base step methods.
    """

    def invoke(self, model: nn.Module, batch: TensorDict) -> TensorDict:
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


@dataclass(frozen=True, slots=True, kw_only=True)
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
        output_spec: Model output key spec for checkpoint-driven invoker rebuild.
    """

    model_settings: Any
    wrapper_settings: Any
    entry_configs: tuple[Any, ...]
    feature_names: tuple[str, ...]
    predict_target_key: str
    shape_summary: Any | None = None
    output_spec: ModelOutputSpec = dataclasses.field(default_factory=ModelOutputSpec)
