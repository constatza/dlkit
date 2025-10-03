from typing import Literal
from importlib import import_module
from inspect import signature, isclass
from typing import Any
from collections.abc import Callable
from types import FunctionType


def get_mro_keys(cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
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


def slice_to_list(idx: slice, length: int):
    """
    Convert a slice to a list of indices of length `length`.

    When the slice is a reverse slice (i.e. `start` and `stop` are both zero and `step` is negative),
    the resulting list will be a reversed list of indices of length `length`.

    Args:
        idx: The slice to convert.
        length: The length of the resulting list.

    Returns:
        list[int]: A list of indices as defined by the slice.
    """
    start = (idx.start or 0) % (length + 1)
    stop = (idx.stop or length) % (length + 1)
    step = idx.step or 1
    if start == stop == 0 and step < 0:
        start, stop = length - 1, 0
    return list(range(start, stop, step))


def get_name(obj: object) -> str:
    """
    Return the name of a function or class. If `obj` is an instance,
    return the class name of that instance. If it’s a function or class,
    return its __name__. Otherwise, return the type’s name.
    """
    if isinstance(obj, FunctionType):
        # It’s a function object
        return obj.__name__
    elif isclass(obj):
        # It’s a class object
        return obj.__name__
    else:
        # Assume it’s an instance; return its class’s name
        return obj.__class__.__name__


def filter_dict(
    data: dict, predicate: Callable, *, which: Literal["key", "value"] = "value"
) -> dict:
    """Return a new dict by filtering items using filter().

    Args:
        data: The original dict of key→value pairs.
        predicate: A callable that takes one argument (key or value) and
                   returns True to keep that item.
        which:    "key" to apply predicate to the dict’s keys,
                  "value" to apply predicate to its values (default).

    Returns:
        A fresh dict containing only those (k, v) for which
        predicate(k) is True (when which=="key") or
        predicate(v) is True (when which=="value").

    Raises:
        ValueError: If `which` isn’t "key" or "value".
    """
    idx = 1
    if which == "key":
        idx = 0

    return dict(filter(lambda item: predicate(item[idx]), data.items()))


def get_signature_names(func: Callable) -> list[str]:
    return list(signature(func).parameters.keys())


def import_object(module_path: str, fallback_module: str = "") -> Callable:
    """
    Dynamically import an object given a path.
    Supports:
    - "module.Path:ClassName"
    - "module.Path" (module only)
    """
    module_path, obj_name = split_module_path(module_path)
    # use default module as fallback
    if module_path is None:
        module_path = fallback_module
    module = import_module(module_path)
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
