import inspect
from collections.abc import Sequence
from typing import Any


def kwargs_compatible_with(cls: type, exclude: Sequence[str] = (), **kwargs) -> dict[str, Any]:
    """Convert a dictionary to a format compatible with the constructor of a given class or function signature.

    Args:
        cls: The class or function to check against.
        exclude (Sequence[str]): A sequence of keys to exclude from the final dictionary.
        **kwargs: Additional keyword arguments to include in the final dictionary.

    Returns:
        dict[str, Any]: A dictionary with keys and values compatible with the constructor of the given class or function signature.
    """
    signature = inspect.signature(cls)
    return {
        field: value
        for field, value in kwargs.items()
        if field in signature.parameters.keys() and field not in exclude
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
