"""Simple shape inference from TensorDict dataset samples.

This module provides the ShapeSummary dataclass and infer_shapes_from_dataset
pure function that replaces the complex IShapeSpec subsystem for model
construction purposes.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeSummary:
    """Minimal shape info extracted from one dataset sample.

    Args:
        in_shapes: Shapes of feature tensors, one per Feature entry in config order.
        out_shapes: Shapes of target tensors, one per Target entry in config order.
    """

    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]

    @property
    def in_features(self) -> int:
        """Primary input feature size (first dim of first feature tensor).

        Returns:
            First dimension of the first feature tensor.
        """
        return self.in_shapes[0][0]

    @property
    def out_features(self) -> int:
        """Primary output feature size (first dim of first target tensor).

        Returns:
            First dimension of the first target tensor.
        """
        return self.out_shapes[0][0]

    @property
    def in_channels(self) -> int:
        """Input channels for conv models (alias for first dim of first feature).

        Returns:
            First dimension of the first feature tensor.
        """
        return self.in_shapes[0][0]

    @property
    def in_length(self) -> int:
        """Sequence length for conv/timeseries models (second dim of first feature).

        Returns:
            Second dimension of the first feature tensor.
        """
        return self.in_shapes[0][1]


def infer_shapes_from_dataset(dataset: object) -> ShapeSummary:
    """Sample index 0 from dataset and extract shapes from a nested TensorDict sample.

    Args:
        dataset: Any dataset object whose __getitem__ returns a nested TensorDict.

    Returns:
        ShapeSummary with in_shapes and out_shapes extracted from the sample.

    Raises:
        ValueError: If dataset[0] does not return a nested TensorDict sample.
    """
    sample = dataset[0]

    try:
        from tensordict import TensorDictBase

        if isinstance(sample, TensorDictBase):
            feat_td = sample["features"]
            targ_td = sample["targets"]
            in_shapes = tuple(tuple(int(d) for d in feat_td[k].shape) for k in feat_td.keys())
            out_shapes = tuple(tuple(int(d) for d in targ_td[k].shape) for k in targ_td.keys())
            return ShapeSummary(in_shapes=in_shapes, out_shapes=out_shapes)
    except ImportError:
        pass

    raise ValueError(
        f"Expected dataset[0] to return a nested TensorDict with 'features' and 'targets', "
        f"got {type(sample).__name__}. Update your dataset's __getitem__ accordingly."
    )
