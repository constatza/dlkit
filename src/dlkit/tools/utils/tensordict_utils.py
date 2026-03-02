"""TensorDict utilities for dlkit."""

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

# TensorDict's native nested key type: a flat string or a tuple path of strings.
# Example flat key:   "predictions"
# Example nested key: ("targets", "y")
NestedKey = str | tuple[str, ...]


def tensordict_to_numpy(td: TensorDict, *keys: NestedKey) -> dict[str, Any] | Any:
    """Convert a TensorDict (or a subset of its keys) to CPU numpy arrays.

    Delegates to :meth:`TensorDict.numpy` after optionally narrowing to
    *keys* via :meth:`TensorDict.select`.  Nested TensorDicts become nested
    dicts; all leaf Tensors are moved to CPU before conversion.

    Supports nested key paths: pass ``("targets", "y")`` to select a specific
    leaf, omitting sibling keys under ``"targets"``.

    Args:
        td: A TensorDict with Tensor leaves, possibly containing nested
            TensorDicts.
        *keys: Optional key names (``str``) or nested key paths
               (``tuple[str, ...]``) to select before converting.  When
               omitted all keys are converted.

    Returns:
        A nested ``dict[str, Any]`` whose leaves are ``np.ndarray`` objects.
        When *keys* is provided only those keys (and their sub-trees) appear
        in the result.

    Example::

        # Convert everything
        arrays = tensordict_to_numpy(stacked)
        preds = arrays["predictions"]  # np.ndarray  (N, out_dim)
        y_arr = arrays["targets"]["y"]  # np.ndarray  (N, 1) nested

        # Flat key — convert one top-level field
        preds_only = tensordict_to_numpy(stacked, "predictions")

        # Nested key path — convert a single leaf, drop siblings
        y_only = tensordict_to_numpy(stacked, ("targets", "y"))

        # Mix of flat and nested keys
        subset = tensordict_to_numpy(stacked, "predictions", ("targets", "y"))
    """
    view = td.select(*keys) if keys else td
    return view.detach().cpu().numpy()


def sequence_to_tensordict(tensors: Sequence[Tensor | TensorDict]) -> TensorDict:
    """Wrap a positional sequence in a TensorDict with keys ``"0"``, ``"1"``, ...

    Batch size is inferred from the first element. Used by ``_normalize_sequence``
    for list/tuple values within named dict outputs or 3+-element latent tuples.

    Args:
        tensors: Non-empty sequence of Tensors or TensorDicts.

    Returns:
        TensorDict with ``batch_size=[batch_size]`` and integer string keys.

    Raises:
        ValueError: If *tensors* is empty.
    """
    if not tensors:
        raise ValueError("Cannot create TensorDict from empty sequence")
    first = tensors[0]
    batch_size = first.shape[0] if isinstance(first, torch.Tensor) else int(first.batch_size[0])
    return TensorDict({str(i): t for i, t in enumerate(tensors)}, batch_size=[batch_size])
