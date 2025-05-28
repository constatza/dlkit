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
