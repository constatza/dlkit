"""Model invocation via cached TensorDictModule with dispatch routing."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from tensordict import NestedKey, TensorDict
from tensordict.nn import TensorDictModule
from torch import nn


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
    latent_keys: tuple[str | tuple[str, ...], ...] = ()

    def all_out_keys(self) -> list[NestedKey]:
        """Return all TensorDictModule out_keys in positional order.

        Returns:
            List starting with ``prediction_key`` followed by ``latent_keys``.
        """
        return cast(list[NestedKey], [self.prediction_key, *self.latent_keys])


class TensorDictModelInvoker:
    """Invokes a model via a cached TensorDictModule and enriches the batch with outputs.

    Supports positional args, keyword args, and mixed dispatch in a single call.
    One ``TensorDictModule`` is built at construction time and reused across all
    batches. The actual model is threaded through a mutable cell so the closure
    (and therefore the ``TensorDictModule``) never needs to be rebuilt.

    Dispatch mapping:

    - *positional* (``in_keys``): tensors passed as positional args — model receives
      them in the declared order as ``model(*pos_tensors)``.
    - *keyword* (``kwarg_in_keys``): tensors passed as named kwargs — model receives
      them as ``model(kwarg_name=tensor, ...)``.
    - Mixed: positional args come first, then keyword args.

    After ``invoke()``, callers read ``batch["predictions"]`` (and any latent keys
    from ``output_spec.latent_keys``).

    Note:
        The standard build path via ``_build_invoker_from_entries()`` uses kwarg
        dispatch for all named features: each entry's ``name`` is the forward arg name.
        Unnamed model-input entries use positional dispatch.

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
        in_keys: list[NestedKey],
        output_spec: ModelOutputSpec | None = None,
        kwarg_in_keys: dict[str, NestedKey] | None = None,
    ) -> None:
        self._output_spec = output_spec or ModelOutputSpec()
        self._out_keys = self._output_spec.all_out_keys()
        n_pos: int = len(in_keys)
        kwarg_names: list[str] = list((kwarg_in_keys or {}).keys())
        all_in_keys: list[NestedKey] = cast(
            list[NestedKey], in_keys + list((kwarg_in_keys or {}).values())
        )
        self._model_cell: list[nn.Module | None] = [None]

        cell = self._model_cell

        def _dispatch(*args: Any) -> Any:
            model = cell[0]
            if model is None:
                raise RuntimeError("model_cell must be set before dispatch")
            return model(*args[:n_pos], **dict(zip(kwarg_names, args[n_pos:], strict=True)))

        self._td_module = TensorDictModule(_dispatch, in_keys=all_in_keys, out_keys=self._out_keys)

    @property
    def _in_keys(self) -> list[NestedKey]:
        """All input keys passed to TensorDictModule (positional + kwarg values).

        Returns:
            List of nested key paths in the order ``TensorDictModule`` extracts them.
        """
        return self._td_module.in_keys

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


@dataclass(frozen=True, slots=True)
class InvokerBuildResult:
    """Value object returned by ``_build_invoker_from_entries``.

    Carries the configured invoker, a build-time model validator, and the
    forward-arg map used for checkpoint persistence. Keeps
    ``StandardLightningWrapper`` decoupled from the validation implementation.

    Attributes:
        invoker: Configured ``TensorDictModelInvoker`` ready for use.
        validator: Callable that validates ``model.forward()`` against the
            configured dispatch. No-op lambda for positional-dispatch mode.
        forward_arg_map: Mapping ``{kwarg_name: feature_name}`` used for
            checkpoint metadata. Empty dict for positional mode.
    """

    invoker: TensorDictModelInvoker
    validator: Callable[[nn.Module], None]
    forward_arg_map: dict[str, str]


def _validate_forward_signature(model: nn.Module, kwarg_names: frozenset[str]) -> None:
    """Validate that model.forward() is safe for named kwarg dispatch.

    Inspects the signature via ``inspect.signature`` only — never decorates or
    wraps the model, which would corrupt Lightning checkpoint serialization.
    Private — accessible only via ``InvokerBuildResult.validator``.

    Args:
        model: PyTorch module whose ``forward`` signature is inspected.
        kwarg_names: Set of kwarg names that will be passed to ``forward()``.

    Raises:
        TypeError: If ``forward()`` declares ``*args`` or ``**kwargs`` (unsafe
            for named dispatch — ambiguous routing).
        ValueError: If any name in ``kwarg_names`` is absent from the signature.
    """
    sig = inspect.signature(model.forward)
    params = sig.parameters

    for param in params.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(
                f"{type(model).__name__}.forward declares *{param.name} which is unsafe "
                "for named dispatch. Remove *args or use positional dispatch."
            )
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError(
                f"{type(model).__name__}.forward declares **{param.name} which is unsafe "
                "for named dispatch. Remove **kwargs or use positional dispatch."
            )

    allowed = {
        name
        for name, p in params.items()
        if p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    unknown = kwarg_names - allowed
    if unknown:
        raise ValueError(
            f"Features {sorted(unknown)} have no matching parameter in "
            f"{type(model).__name__}.forward. "
            f"Rename the feature(s) to match a forward() parameter name, or set "
            f"model_input=False to exclude them from dispatch. "
            f"Available parameters: {sorted(allowed)}"
        )


def _build_invoker_from_entries(
    feature_entries: list[Any],
    output_spec: ModelOutputSpec | None = None,
) -> InvokerBuildResult:
    """Build a TensorDictModelInvoker from feature entry configurations.

    Named features (``entry.name is not None``) with ``model_input=True`` are
    dispatched as keyword arguments — ``model(name=tensor, ...)``. Each entry's
    ``name`` is used directly as the ``forward()`` parameter name.

    Unnamed features (``entry.name is None``) with ``model_input=True`` use
    positional dispatch. Mixing named and unnamed model-input entries
    in the same invoker is an error.

    Args:
        feature_entries: Feature DataEntry objects in config-insertion order.
        output_spec: Output key spec for the invoker. Defaults to
            ``ModelOutputSpec()`` (single ``"predictions"`` output).

    Returns:
        ``InvokerBuildResult`` with invoker, build-time validator, and forward-arg map.

    Raises:
        ValueError: If no features are configured as model inputs, or if named
            and unnamed model-input entries are mixed.
    """
    resolved_spec = output_spec or ModelOutputSpec()

    named = [e for e in feature_entries if e.model_input and e.name is not None]
    unnamed = [e for e in feature_entries if e.model_input and e.name is None]

    if not named and not unnamed:
        raise ValueError(
            "No model-input features found. Configure at least one Feature with model_input=True."
        )

    if named and unnamed:
        raise ValueError(
            "Mixed dispatch: all model-input features must either have a name (kwarg dispatch) "
            "or all be unnamed (positional dispatch). "
            f"Named: {[e.name for e in named]}, Unnamed count: {len(unnamed)}"
        )

    if named:
        kwarg_in_keys: dict[str, NestedKey] = {e.name: ("features", e.name) for e in named}
        forward_arg_map: dict[str, str] = {e.name: e.name for e in named}
        kwarg_names = frozenset(forward_arg_map)
        invoker = TensorDictModelInvoker(
            in_keys=[], kwarg_in_keys=kwarg_in_keys, output_spec=resolved_spec
        )
        return InvokerBuildResult(
            invoker=invoker,
            validator=lambda model: _validate_forward_signature(model, kwarg_names),
            forward_arg_map=forward_arg_map,
        )

    # Positional-dispatch path (unnamed entries only)
    in_keys: list[NestedKey] = [("features", e.name) for e in unnamed]
    return InvokerBuildResult(
        invoker=TensorDictModelInvoker(in_keys=in_keys, output_spec=resolved_spec),
        validator=lambda _model: None,
        forward_arg_map={},
    )


def _ordered_model_input_names(feature_entries: list[Any]) -> tuple[str, ...]:
    """Return feature entry names in invoker dispatch order (config-list order).

    Used to store ``feature_names`` in checkpoint metadata so inference can
    map positional tensor args to the correct transform chain.

    Args:
        feature_entries: Feature DataEntry objects in config-insertion order.

    Returns:
        Tuple of entry names for entries with ``model_input=True``, in the
        same order the invoker dispatches tensors to ``model.forward()``.
    """
    return tuple(
        entry.name for entry in feature_entries if entry.model_input and entry.name is not None
    )
