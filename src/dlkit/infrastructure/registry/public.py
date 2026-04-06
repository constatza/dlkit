from __future__ import annotations

from collections.abc import Callable
from difflib import get_close_matches
from typing import Any

from dlkit.infrastructure.registry.base import LockedRegistry

# One registry instance per component kind
_MODELS: LockedRegistry[Any] = LockedRegistry()
_DATASETS: LockedRegistry[Any] = LockedRegistry()
_LOSSES: LockedRegistry[Any] = LockedRegistry()
_METRICS: LockedRegistry[Any] = LockedRegistry()
_DATAMODULES: LockedRegistry[Any] = LockedRegistry()

_REGISTRIES: dict[str, LockedRegistry[Any]] = {
    "model": _MODELS,
    "dataset": _DATASETS,
    "loss": _LOSSES,
    "metric": _METRICS,
    "datamodule": _DATAMODULES,
}


def _make_register(kind: str):
    registry = _REGISTRIES[kind]

    def register(
        obj: Callable | type | None = None,
        name: str | None = None,
        *,
        aliases: list[str] | None = None,
        overwrite: bool = False,
        use: bool = False,
    ):
        """Decorator to register classes/callables for a given kind.

        Usage:
            @register_model()
            class MyNet(...): ...

            @register_loss(name="mae", aliases=["l1"], use=True)
            def mae_loss(...): ...
        """

        def _apply(target: Any):
            key = name or getattr(target, "__name__", None)
            if not key:
                raise ValueError(
                    "Cannot infer registry key: provide 'name' or ensure object has __name__"
                )
            registry.register(key, target, overwrite=overwrite)
            if aliases:
                for alias in aliases:
                    registry.add_alias(alias, key, overwrite=overwrite)
            if use:
                registry.set_forced(key)
            return target

        # Support bare decorator usage: @register_*(...)
        if obj is not None:
            return _apply(obj)
        return _apply

    return register


# Public decorator helpers (only API we export)
register_model = _make_register("model")
register_dataset = _make_register("dataset")
register_loss = _make_register("loss")
register_metric = _make_register("metric")
register_datamodule = _make_register("datamodule")


# Resolver used by factories
def resolve_from_registry(kind: str, name: str) -> Any:
    try:
        return _REGISTRIES[kind].get(name)
    except KeyError as e:
        # Improve error with suggestions
        keys = sorted(_REGISTRIES[kind]._all_keys())  # internal for suggestions only
        suggestions = get_close_matches(name, keys, n=3, cutoff=0.6)
        sug = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"No registered {kind} named '{name}'.{sug}") from e


def get_forced(kind: str) -> Any | None:
    return _REGISTRIES[kind].get_forced()


# Test-only reset hook (not exported via __all__)
def _reset_for_tests() -> None:
    for r in _REGISTRIES.values():
        r._reset_for_tests()


__all__ = [
    "register_datamodule",
    "register_dataset",
    "register_loss",
    "register_metric",
    "register_model",
]
