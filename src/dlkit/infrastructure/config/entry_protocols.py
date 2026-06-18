"""Capability marker interfaces for DataEntry types.

These ABCs and pure-marker classes express *what a DataEntry can do*,
independent of any concrete hierarchy.  Downstream code uses
``isinstance(entry, IPathBased)`` rather than checking concrete types.

Interfaces:
    IPathBased - entry loads data from a file path
    IValueBased - entry holds an in-memory tensor/array
    IRuntimeGenerated - entry is produced by the model at run-time
    IFeatureReference - entry references another feature (e.g. AutoencoderTarget)
    PathBasedEntry - structural protocol for entries that resolve to a file path
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from dlkit.common.sources import OpenReaderResult


class IPathBased(ABC):
    """Marks entries that load data from file paths.

    Example:
        >>> entry = NpyEntry(name="x", path="data.npy", data_role=DataRole.FEATURE)
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
        >>> entry = ValueEntry(name="x", value=np.ones((10, 5)), data_role=DataRole.FEATURE)
        >>> isinstance(entry, IValueBased)
        True
    """

    @abstractmethod
    def get_value(self) -> torch.Tensor | np.ndarray | None:
        """Return the in-memory value, or None in placeholder mode.

        Returns:
            Tensor or array if set, else None.
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


@runtime_checkable
class PathBasedEntry(Protocol):
    """Structural protocol for entries that resolve to a file path via open_reader().

    This protocol formalises the attributes consumed by ``source_from_entry()``
    in the ``case Path()`` branch, replacing ad-hoc ``getattr()`` duck-typing
    with a typed structural check.

    Implementations:
        All concrete ``PathBasedEntry`` subclasses in ``entry_types`` satisfy
        this protocol — ``NpyEntry``, ``NpzEntry``, ``CsvEntry``, ``ParquetEntry``,
        ``Hdf5Entry``, and ``AutoencoderTarget``.

    Example:
        >>> from dlkit.infrastructure.config.entry_types import NpyEntry
        >>> entry = NpyEntry(name="x", path=some_path, data_role=DataRole.FEATURE)
        >>> isinstance(entry, PathBasedEntry)
        True
    """

    @property
    def name(self) -> str | None:
        """Entry name, used as identifier in the data pipeline.

        Returns:
            Name string or None in placeholder mode.
        """
        ...

    def open_reader(self) -> OpenReaderResult:
        """Return the IO source for this entry.

        Returns:
            ``ArraySource`` for lazy formats or ``Path`` for eager formats.
        """
        ...

    @property
    def array_key(self) -> str:
        """Array key used when loading multi-array sources (e.g. .npz).

        Returns:
            Key string identifying which array to load.
        """
        ...

    @property
    def is_multi_array(self) -> bool:
        """Whether this entry's source contains multiple named arrays.

        Returns:
            True for .npz archives; False for single-array formats.
        """
        ...

    @property
    def dtype(self) -> torch.dtype | None:
        """Optional dtype override for loaded tensors.

        Returns:
            ``torch.dtype`` when explicitly set; None defers to precision service.
        """
        ...

    @property
    def load_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments forwarded to ``load_array()``.

        Returns:
            Dict of kwargs (e.g. ``{"mmap_mode": "r"}`` for memory-mapped npy).
        """
        ...
