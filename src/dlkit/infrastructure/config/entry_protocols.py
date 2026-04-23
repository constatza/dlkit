"""Capability marker interfaces for DataEntry types.

These ABCs and pure-marker classes express *what a DataEntry can do*,
independent of any concrete hierarchy.  Downstream code uses
``isinstance(entry, IPathBased)`` rather than checking concrete types.

Interfaces:
    IPathBased - entry loads data from a file path
    IValueBased - entry holds an in-memory tensor/array
    IWritable - entry can be saved during inference
    IRuntimeGenerated - entry is produced by the model at run-time
    IFeatureReference - entry references another feature (e.g. AutoencoderTarget)
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class IPathBased(ABC):
    """Marks entries that load data from file paths.

    Example:
        >>> entry = PathFeature(name="x", path="data.npy")
        >>> isinstance(entry, IPathBased)
        True
    """

    @abstractmethod
    def get_path(self) -> Path | None:
        """Return the file path, or None in placeholder mode.

        Returns:
            Path if set, else None.
        """


class IValueBased(ABC):
    """Marks entries that contain in-memory tensor/array values.

    Example:
        >>> entry = ValueFeature(name="x", value=np.ones((10, 5)))
        >>> isinstance(entry, IValueBased)
        True
    """

    @abstractmethod
    def get_value(self) -> torch.Tensor | np.ndarray | None:
        """Return the in-memory value, or None in placeholder mode.

        Returns:
            Tensor or array if set, else None.
        """


class IWritable:
    """Marks entries whose data can be persisted during inference.

    Implementations must expose a ``write: bool`` attribute.

    Example:
        >>> target = PathTarget(name="y", path="targets.npy", write=True)
        >>> isinstance(target, IWritable)
        True
    """


class IRuntimeGenerated:
    """Marks entries created by the model at run-time (Latent, Prediction).

    Pure marker — no methods required.

    Example:
        >>> latent = Latent(name="z", write=True)
        >>> isinstance(latent, IRuntimeGenerated)
        True
    """


class IFeatureReference:
    """Marks entries that reference another feature entry.

    Implementations must expose a ``feature_ref: str`` attribute.

    Example:
        >>> target = AutoencoderTarget(name="y", feature_ref="x")
        >>> isinstance(target, IFeatureReference)
        True
    """
