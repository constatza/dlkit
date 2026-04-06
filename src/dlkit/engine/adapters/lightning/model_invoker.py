"""Model invocation via cached TensorDictModule with dispatch routing."""

from __future__ import annotations

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

    Dispatch mapping (built by ``_build_invoker_from_entries``):

    - *positional* (``in_keys``): tensors passed as positional args — model receives
      them in the declared order as ``model(*pos_tensors)``.
    - *keyword* (``kwarg_in_keys``): tensors passed as named kwargs — model receives
      them as ``model(kwarg_name=tensor, ...)``.
    - Mixed: positional args come first, then keyword args.

    After ``invoke()``, callers read ``batch["predictions"]`` (and any latent keys
    from ``output_spec.latent_keys``).

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
            assert model is not None, "model_cell must be set before dispatch"
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
    kwarg_map: dict[str, str] = {}

    for entry in feature_entries:
        mi = getattr(entry, "model_input", True)
        name: str | None = getattr(entry, "name", None)
        if name is None or mi is False or mi is None:
            continue
        if mi is True:
            kwarg_map[name] = name
        elif isinstance(mi, int) and not isinstance(mi, bool):
            positional.append((float(mi), name))
        elif isinstance(mi, str) and mi.isdigit():
            positional.append((float(mi), name))
        elif isinstance(mi, str):
            kwarg_map[mi] = name

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
        output_spec: Output key spec for the invoker. Defaults to
            ``ModelOutputSpec()`` (single ``"predictions"`` output).

    Returns:
        Configured ``TensorDictModelInvoker`` ready for use in a wrapper.

    Raises:
        ValueError: If no features are configured as model inputs.
    """
    positional, kwarg_map = _classify_feature_entries(feature_entries)
    positional.sort(key=lambda x: x[0])
    positional_in_keys: list[NestedKey] = [("features", name) for _, name in positional]
    kwarg_in_keys: dict[str, NestedKey] = {
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
