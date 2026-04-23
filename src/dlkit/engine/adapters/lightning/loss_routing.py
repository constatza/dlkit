"""Loss routing: maps batch keys to loss function arguments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tensordict import TensorDict
from torch import Tensor

from dlkit.infrastructure.config.data_entries import DataEntry, is_feature_entry
from dlkit.infrastructure.config.model_components import LossInputRef

from .batch_namespace import _parse_key


@dataclass(frozen=True, slots=True, kw_only=True)
class LossRoute:
    """Pre-parsed, typed routing command for a loss function invocation.

    All string parsing happens once at construction time; ``compute()`` uses
    only direct TensorDict key lookups — no per-batch string manipulation.

    Attributes:
        loss_fn: The loss function callable.
        target_route: Pre-parsed ``(namespace, entry_name)`` tuple for the main target.
        extra_routes: Ordered tuple of ``(kwarg_name, (namespace, entry_name))`` for
            additional loss function keyword arguments.
    """

    loss_fn: Callable
    target_route: tuple[str, str]
    extra_routes: tuple[tuple[str, tuple[str, str]], ...] = ()


class RoutedLossComputer:
    """Routes batch keys to loss function kwargs per LossComponentSettings.

    Computes loss by extracting the configured target from the batch and
    passing any extra inputs as keyword arguments. All key parsing is
    performed once at construction time via ``LossRoute``.

    Attributes:
        _route: Pre-parsed loss routing command.
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
        effective_key = target_key or f"targets.{default_target_key}"
        target_route = _parse_key(effective_key)
        extra_routes: tuple[tuple[str, tuple[str, str]], ...] = tuple(
            (ref.arg, _parse_key(ref.key)) for ref in extra_inputs
        )
        self._route = LossRoute(
            loss_fn=loss_fn,
            target_route=target_route,
            extra_routes=extra_routes,
        )

    @property
    def loss_fn(self) -> Callable:
        """Expose the routed loss callable without leaking route internals."""
        return self._route.loss_fn

    def compute(self, predictions: Tensor, batch: TensorDict) -> Tensor:
        """Compute loss from predictions and named batch.

        Args:
            predictions: Model output tensor.
            batch: TensorDict containing features and targets.

        Returns:
            Scalar loss tensor.
        """
        target = batch[self._route.target_route].to(dtype=predictions.dtype)
        extra_kwargs = {kwarg: batch[route] for kwarg, route in self._route.extra_routes}
        return self._route.loss_fn(predictions, target, **extra_kwargs)


def build_auto_extra_inputs(
    entry_configs: tuple[DataEntry, ...],
) -> dict[str, LossInputRef]:
    """Derive LossInputRef entries from DataEntry objects that declare ``loss_input``.

    Each entry with a non-None ``loss_input`` value is auto-routed as a loss
    function kwarg.  The kwarg name is the ``loss_input`` string; the batch key
    is derived from the entry's namespace and name.

    Args:
        entry_configs: DataEntry objects in config-insertion order.

    Returns:
        Mapping from kwarg name to LossInputRef, ready to merge with explicit routes.

    Raises:
        ValueError: If two entries declare the same ``loss_input`` kwarg name.
    """
    result: dict[str, LossInputRef] = {}
    for e in entry_configs:
        kwarg = getattr(e, "loss_input", None)
        if kwarg is None or e.name is None:
            continue
        if kwarg in result:
            raise ValueError(
                f"Duplicate loss_input kwarg '{kwarg}' declared on multiple entries. "
                "Each kwarg name must appear on exactly one entry."
            )
        namespace = "features" if is_feature_entry(e) else "targets"
        result[kwarg] = LossInputRef(arg=kwarg, key=f"{namespace}.{e.name}")
    return result


def merge_extra_inputs(
    auto: dict[str, LossInputRef],
    explicit: tuple[LossInputRef, ...],
) -> tuple[LossInputRef, ...]:
    """Merge auto-derived and explicit LossInputRef collections.

    Any overlap between auto-derived (from ``DataEntry.loss_input``) and explicit
    (from ``LossComponentSettings.extra_inputs``) routes is a configuration error —
    no silent overrides.

    Args:
        auto: Auto-derived routes keyed by kwarg name.
        explicit: Explicitly configured routes from LossComponentSettings.

    Returns:
        Merged tuple with no duplicate arg names.

    Raises:
        ValueError: If the same kwarg name appears in both sources.
    """
    explicit_by_arg = {r.arg: r for r in explicit}
    overlap = set(auto) & set(explicit_by_arg)
    if overlap:
        raise ValueError(
            f"Loss kwarg(s) {sorted(overlap)} declared on both DataEntry.loss_input and "
            "LossComponentSettings.extra_inputs. Remove one declaration — no silent overrides."
        )
    return tuple({**auto, **explicit_by_arg}.values())
