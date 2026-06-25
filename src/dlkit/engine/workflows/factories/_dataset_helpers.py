"""Dataset override helpers — one per family, extracted from DatasetBuilder for SRP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dlkit.infrastructure.config.data_entries import DataEntry
    from dlkit.infrastructure.config.data_settings import DataSettings


def graph_dataset_overrides(data_settings: DataSettings) -> dict[str, Any]:
    """Build path-keyed overrides for a graph dataset from DataSettings entries.

    Extracts named file paths from feature/target PathBasedEntry objects.
    The resulting dict is passed directly to the graph dataset constructor.

    Args:
        data_settings: The DataSettings instance containing feature/target entries.

    Returns:
        Dict mapping entry names (and optionally ``"root"``) to their paths.
    """
    from dlkit.infrastructure.config.entry_types import PathBasedEntry

    overrides: dict[str, Any] = {}
    for entry in data_settings.features or ():
        if isinstance(entry, PathBasedEntry) and entry.name and entry.path:
            overrides[entry.name] = entry.path
    for entry in data_settings.targets or ():
        if isinstance(entry, PathBasedEntry) and entry.name and entry.path:
            overrides[entry.name] = entry.path
    if data_settings.root is not None:
        overrides["root"] = data_settings.root
    return overrides


def flexible_dataset_overrides(
    features: tuple[DataEntry, ...],
    targets: tuple[DataEntry, ...],
) -> dict[str, Any]:
    """Pack raw DataEntry objects into overrides for FlexibleDataset.

    FlexibleDataset accepts path-based or value-based DataEntry objects directly;
    it resolves them lazily at access time. No tensor conversion is performed here.

    Args:
        features: Tuple of feature DataEntry objects to include in the dataset.
        targets: Tuple of target DataEntry objects to include in the dataset.

    Returns:
        Dict with ``"entries"`` key containing all feature and target entries.
    """
    return {"entries": (*features, *targets)}
