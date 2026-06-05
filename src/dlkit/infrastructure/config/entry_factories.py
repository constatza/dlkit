"""Type aliases and predicate helpers for DataEntry objects.

Type aliases:
    AnyPathEntry = ZarrEntry | NpyEntry | NpzEntry | CsvEntry | ParquetEntry | Hdf5Entry
    AnyEntry     = AnyPathEntry | ValueEntry

Type guards (all accept DataEntry, return bool):
    is_feature, is_target, is_path_based, is_value_based
"""

from typing import Annotated

from pydantic import Field as PydanticField

from .data_roles import DataRole
from .entry_base import DataEntry
from .entry_protocols import IPathBased, IValueBased
from .entry_types import (
    AutoencoderTarget,
    CsvEntry,
    Hdf5Entry,
    NpyEntry,
    NpzEntry,
    ParquetEntry,
    ValueEntry,
    ZarrEntry,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AnyPathEntry = Annotated[
    ZarrEntry | NpyEntry | NpzEntry | CsvEntry | ParquetEntry | Hdf5Entry,
    PydanticField(discriminator="format"),
]
AnyEntry = AnyPathEntry | ValueEntry | AutoencoderTarget

# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------


def is_path_based(entry: DataEntry) -> bool:
    """Return True if ``entry`` loads data from a file path.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IPathBased``.
    """
    return isinstance(entry, IPathBased)


def is_value_based(entry: DataEntry) -> bool:
    """Return True if ``entry`` holds an in-memory value.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IValueBased``.
    """
    return isinstance(entry, IValueBased)


def is_feature(entry: DataEntry) -> bool:
    """Return True if entry lives in the features partition of the batch.

    Args:
        entry: The entry to inspect.

    Returns:
        True when ``entry.data_role == DataRole.FEATURE``.
    """
    return entry.data_role == DataRole.FEATURE


def is_target(entry: DataEntry) -> bool:
    """Return True if entry lives in the targets partition of the batch.

    Args:
        entry: The entry to inspect.

    Returns:
        True when ``entry.data_role == DataRole.TARGET``.
    """
    return entry.data_role == DataRole.TARGET


def is_latent(entry: DataEntry) -> bool:
    """Return True if entry is a latent representation.

    Args:
        entry: The entry to inspect.

    Returns:
        True when ``entry.data_role == DataRole.LATENT``.
    """
    return entry.data_role == DataRole.LATENT


def is_prediction(entry: DataEntry) -> bool:
    """Return True if entry is a model prediction.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` is a ``Prediction`` instance.
    """
    from .entry_types import Prediction

    return isinstance(entry, Prediction)


__all__ = [
    "AnyPathEntry",
    "AnyEntry",
    "AutoencoderTarget",
    "is_path_based",
    "is_value_based",
    "is_feature",
    "is_target",
    "is_latent",
    "is_prediction",
]
