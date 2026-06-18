"""Internal helper functions for data sources."""

from __future__ import annotations

from dlkit.infrastructure.config.entry_factories import AnyEntry


def _require_name(entry: AnyEntry) -> str:
    """Return ``entry.name``, raising ``ValueError`` if it is ``None``.

    Args:
        entry: A ``DataEntry`` whose ``name`` field is checked.

    Returns:
        The non-``None`` entry name.

    Raises:
        ValueError: If ``entry.name`` is ``None``.
    """
    if entry.name is None:
        raise ValueError(
            f"Entry of type '{type(entry).__name__}' has no name; "
            "set the 'name' field before building a RoleSourceMap."
        )
    return entry.name
