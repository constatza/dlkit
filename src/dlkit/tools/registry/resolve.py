from __future__ import annotations

from typing import Any

from dlkit.tools.registry.public import get_forced, resolve_from_registry
from dlkit.tools.utils.general import import_object


def resolve_component(kind: str, name: str | None, module_path: str | None = None) -> Any:
    """Resolve a component by kind using forced selection, registry, or import.

    Order:
      1) Forced selection (set during registration with use=True)
      2) Registered object by name/alias
      3) Import using dotted path or fallback module_path
    """
    forced = get_forced(kind)
    if forced is not None:
        return forced

    if name:
        # Try registry first
        try:
            return resolve_from_registry(kind, name)
        except KeyError:
            # Not registered; fall back to import for built-in/third-party
            return import_object(name, fallback_module=module_path or "")

    raise ValueError(
        f"No '{kind}' specified and no forced selection is set. "
        f"Either register with use=True or set a name in the config."
    )


__all__ = ["resolve_component"]
