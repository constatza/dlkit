from typing import Literal
from inspect import signature, Parameter, isclass
from typing import get_type_hints, Any
from collections.abc import Callable
from types import FunctionType
from collections.abc import Sequence

from pathlib import Path


def kwargs_compatible_with(cls: type, exclude: Sequence[str] = (), **kwargs) -> dict[str, Any]:
    """Convert a dictionary to a format compatible with the constructor of a given class or function signature.

    Args:
        cls: The class or function to check against.
        exclude (Sequence[str]): A sequence of keys to exclude from the final dictionary.
        **kwargs: Additional keyword arguments to include in the final dictionary.

    Returns:
        dict[str, Any]: A dictionary with keys and values compatible with the constructor of the given class or function signature.
    """

    mro_keys = [
        name
        for base in cls.mro()
        for name, param in signature(base.__init__).parameters.items()
        if name != "self"
    ]

    return {
        field: value
        for field, value in kwargs.items()
        if field in mro_keys and field not in exclude
    }


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


def get_param_types(obj: Callable[..., object]) -> dict[str, type | None]:
    """Extract the type annotation for each parameter of a function or class.

    - For classes, inspects its __init__ and drops the implicit `self`.
    - Uses typing.get_type_hints so forward refs and string annotations are resolved.

    Args:
        obj: A function or class whose signature you want to inspect.

    Returns:
        A dict mapping each parameter name to its annotated type, or None.
    """
    # Choose the right call target
    target = obj if not isinstance(obj, type) else obj.__init__

    # Resolve all type hints (evaluates forward refs, etc.)
    hints = get_type_hints(target, include_extras=False)

    sig = signature(target)
    # Filter to only “real” params (drop *args/**kwargs)
    params = {
        name: hints.get(name, None)
        for name, param in sig.parameters.items()
        if param.kind
        in (
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        )
    }

    # If this was a class, drop the first “self” param
    if isinstance(obj, type) and params:
        params.pop("self", None)

    return params


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


def filter_paths_from_dict[T](data: dict[T, Path]) -> dict[T, Path]:
    return filter_dict(data, lambda x: isinstance(x, Path))
