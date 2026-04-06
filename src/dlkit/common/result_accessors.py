"""Non-mutating accessors for workflow result computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from dlkit.common.results import TrainingResult

NestedKey = str | tuple[str, ...]


def _select_nested_value(payload: Any, key: NestedKey) -> Any:
    """Select a value from a nested dict/TensorDict using a key path.

    Args:
        payload: The nested structure to query.
        key: A single string key or tuple of keys for nested access.

    Returns:
        The selected value.
    """
    if isinstance(key, tuple):
        value = payload
        for part in key:
            value = value[part]
        return value
    return payload[key]


def _assign_nested_value(target: dict[str, Any], key: NestedKey, value: Any) -> None:
    """Assign a value to a nested dict using a key path.

    Creates intermediate dicts as needed.

    Args:
        target: The target dict to modify.
        key: A single string key or tuple of keys for nested assignment.
        value: The value to assign.
    """
    if isinstance(key, str):
        target[key] = value
        return

    cursor = target
    for part in key[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[key[-1]] = value


def _torch_tree_to_numpy(payload: Any) -> Any:
    """Convert all torch tensors in a nested structure to numpy arrays.

    Args:
        payload: A nested structure of tensors, TensorDicts, dicts, etc.

    Returns:
        The same structure with all torch.Tensor converted to numpy.
    """
    if isinstance(payload, TensorDict):
        return {key: _torch_tree_to_numpy(payload[key]) for key in payload.keys()}
    if isinstance(payload, dict):
        return {key: _torch_tree_to_numpy(value) for key, value in payload.items()}
    if isinstance(payload, tuple):
        return tuple(_torch_tree_to_numpy(value) for value in payload)
    if isinstance(payload, list):
        return [_torch_tree_to_numpy(value) for value in payload]
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().numpy()
    return payload


class TrainingResultAccessor:
    """Computes and caches derived views of a TrainingResult.

    Provides lazy stacking and numpy conversion without mutating the result.
    """

    def __init__(self, result: TrainingResult) -> None:
        """Initialize the accessor with a TrainingResult.

        Args:
            result: The training result to wrap.
        """
        self._result = result
        self._stacked_cache: TensorDict | None = None
        self._stacked_computed = False

    @property
    def stacked(self) -> TensorDict | None:
        """All stacked prediction results, computed lazily and cached.

        Returns:
            A TensorDict of stacked predictions, or None if no predictions.
        """
        if not self._stacked_computed:
            self._stacked_cache = self._compute_stacked_results()
            self._stacked_computed = True
        return self._stacked_cache

    def to_numpy(self, *keys: NestedKey) -> dict[str, Any] | Any | None:
        """Convert stacked results to CPU numpy arrays.

        Args:
            *keys: Optional keys to extract. If empty, converts entire stacked result.

        Returns:
            A dict/array of numpy results, or None if no predictions.
        """
        if self.stacked is None:
            return None

        if not keys:
            return _torch_tree_to_numpy(self.stacked)

        selected: dict[str, Any] = {}
        for key in keys:
            _assign_nested_value(selected, key, _select_nested_value(self.stacked, key))
        return _torch_tree_to_numpy(selected)

    def _compute_stacked_results(self) -> TensorDict | None:
        """Concatenate per-batch prediction TensorDicts along dim 0.

        Returns:
            A single TensorDict with all predictions stacked, or None.
        """
        if not self._result.predictions:
            return None
        return cast(TensorDict, torch.cat(self._result.predictions, dim=0))
