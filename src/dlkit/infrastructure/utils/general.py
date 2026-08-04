from collections.abc import Callable
from importlib import import_module
from inspect import isclass, signature
from typing import Any, Literal


def get_mro_keys(cls: type) -> set[str] | list[str]:
    """Convert a dictionary to a format compatible with the constructor of a given class or function signature.

    Args:
        cls: The class or function to check against.

    Returns:
        dict[str, Any]: A dictionary with keys and values compatible with the constructor of the given class or function signature.
    """
    if isclass(cls):
        mro_keys = {
            name
            for base in cls.mro()
            for name in signature(base.__init__).parameters.keys()
            if name != "self"
        }
    else:
        mro_keys = [name for name, param in signature(cls).parameters.items() if name != "self"]
    return mro_keys


def kwargs_compatible_with(
    cls: type, which: Literal["compatible", "incompatible"] = "compatible", **kwargs
) -> dict[str, Any]:
    mro_keys = get_mro_keys(cls)
    if which == "incompatible":
        incompatible = {k: v for k, v in kwargs.items() if k not in mro_keys}
        return incompatible
    if which == "compatible":
        compatible = {k: v for k, v in kwargs.items() if k in mro_keys}
        return compatible
    raise ValueError(f"Invalid value for which: {which}")


def import_object(module_path: str, fallback_module: str = "") -> Callable:
    """Dynamically import an object given a path.
    Supports:
    - "module.Path:ClassName"
    - "module.Path" (module only)
    """
    _mod, obj_name = split_module_path(module_path)
    # use default module as fallback
    resolved_module = _mod if _mod is not None else fallback_module
    module = import_module(resolved_module)
    obj = getattr(module, obj_name)
    if obj is None:
        raise ImportError(f"Could not find {obj_name} in {module_path}")
    return obj


def split_module_path(path: str) -> tuple[str | None, str]:
    """Split a path into a module path and an object name."""
    if ":" not in path:
        # assume class or function only
        return None, path
    path, obj_name = path.split(":", 1)
    return path, obj_name
