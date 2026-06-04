"""Format registry for dense array pack I/O (Open-Closed Principle).

Adding a new format (e.g. HDF5) requires only:

1. Creating ``_hdf5_dense.py`` with a reader and writer.
2. Calling ``register_format(ArrayPackFormat.HDF5_DENSE, reader_factory,
   writer_factory=_open_hdf5_pack_writer)`` in ``__init__.py``.

Zero changes to ``_factory.py`` or ``_protocols.py`` are needed.
"""

from __future__ import annotations

from collections.abc import Callable

from ._protocols import AbstractArrayPackReader, ArrayPackFormat, IArrayPackWriter

_reader_factories: dict[ArrayPackFormat, Callable[..., AbstractArrayPackReader]] = {}
_writer_factories: dict[ArrayPackFormat, Callable[..., IArrayPackWriter]] = {}


def register_format(
    fmt: ArrayPackFormat,
    reader_factory: Callable[..., AbstractArrayPackReader],
    *,
    writer_factory: Callable[..., IArrayPackWriter] | None = None,
) -> None:
    """Register reader (and optionally writer) factories for a pack format.

    Args:
        fmt: The array pack format identifier.
        reader_factory: Callable that returns an ``AbstractArrayPackReader``
            when given ``(path, **kwargs)``.
        writer_factory: Optional callable that returns an ``IArrayPackWriter``
            when given ``(path, size, *, dtype, chunk_size)``.  Required for
            ``write_array_pack`` to work with this format.
    """
    _reader_factories[fmt] = reader_factory
    if writer_factory is not None:
        _writer_factories[fmt] = writer_factory


def get_reader_factory(fmt: ArrayPackFormat) -> Callable[..., AbstractArrayPackReader]:
    """Get the registered reader factory for a pack format.

    Args:
        fmt: The array pack format identifier.

    Returns:
        A callable that constructs an ``AbstractArrayPackReader``.

    Raises:
        ValueError: If the format has not been registered.
    """
    try:
        return _reader_factories[fmt]
    except KeyError:
        known = ", ".join(f.value for f in _reader_factories) or "<none>"
        raise ValueError(f"Unregistered array pack format '{fmt.value}'. Known: {known}") from None


def get_writer_factory(fmt: ArrayPackFormat) -> Callable[..., IArrayPackWriter]:
    """Get the registered writer factory for a pack format.

    Args:
        fmt: The array pack format identifier.

    Returns:
        A callable that constructs an ``IArrayPackWriter``.

    Raises:
        ValueError: If no writer factory has been registered for the format.
    """
    try:
        return _writer_factories[fmt]
    except KeyError:
        known = ", ".join(f.value for f in _writer_factories) or "<none>"
        raise ValueError(
            f"No writer registered for '{fmt.value}'. Formats with writers: {known}"
        ) from None
