"""Pure tensor-shape normalization for prediction-plot callback inputs.

predict_step legitimately wraps 'predictions'/'targets' in a nested TensorDict
keyed by target name (see prediction_strategies.py), even for single-target
regression models. This module extracts the flat tensor these plots need
without assuming the underlying multi-head-capable contract is wrong.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def as_flat_tensor(value: object) -> torch.Tensor | None:
    """Return a bare tensor from a Tensor or single-key TensorDict/Mapping, else None.

    Args:
        value: A torch.Tensor, or a Mapping (e.g. TensorDict) wrapping exactly
            one tensor leaf, as produced by predict_step for single-target models.

    Returns:
        The underlying tensor, or None if value doesn't resolve to exactly one tensor.
    """
    match value:
        case torch.Tensor():
            return value
        case Mapping():
            leaves = list(value.values())
            if len(leaves) != 1:
                return None
            leaf = leaves[0]
            return leaf if isinstance(leaf, torch.Tensor) else None
        case _:
            return None
