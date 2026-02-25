"""Simple shape inference from positional Batch samples.

This module provides the ShapeSummary dataclass and infer_shapes_from_dataset
pure function that replaces the complex IShapeSpec subsystem for model
construction purposes.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
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
    """Sample index 0 from dataset and extract shapes from the returned Batch or TensorDict.

    Args:
        dataset: Any dataset object whose __getitem__ returns a Batch or TensorDict.

    Returns:
        ShapeSummary with in_shapes and out_shapes extracted from the sample.

    Raises:
        ValueError: If dataset[0] does not return a Batch or TensorDict instance.
    """
    from dlkit.core.datatypes.batch import Batch

    sample = dataset[0]

    # Handle TensorDict format (new)
    try:
        from tensordict import TensorDict

        if isinstance(sample, TensorDict):
            feat_td = sample["features"]
            targ_td = sample["targets"]
            in_shapes = tuple(
                tuple(int(d) for d in feat_td[k].shape)
                for k in feat_td.keys()
            )
            out_shapes = tuple(
                tuple(int(d) for d in targ_td[k].shape)
                for k in targ_td.keys()
            )
            return ShapeSummary(in_shapes=in_shapes, out_shapes=out_shapes)
    except ImportError:
        pass

    # Handle legacy Batch format (backward compat)
    if isinstance(sample, Batch):
        return ShapeSummary(
            in_shapes=tuple(tuple(int(d) for d in t.shape) for t in sample.features),
            out_shapes=tuple(tuple(int(d) for d in t.shape) for t in sample.targets),
        )

    raise ValueError(
        f"Expected dataset[0] to return a Batch or TensorDict instance, got {type(sample).__name__}. "
        "Update your dataset's __getitem__ to return Batch or TensorDict."
    )
