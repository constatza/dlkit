"""Format registry for sparse pack I/O (Open-Closed Principle).

Adding a new format (e.g. CSR) requires only:
1. Creating ``_csr_pack.py`` with a codec and reader.
2. Calling ``register_format(SparseFormat.CSR, CsrPackCodec(), CsrPackReader)``.

Zero changes to ``_factory.py``, ``_validation.py``, or ``_protocols.py``.
"""

from __future__ import annotations

from collections.abc import Callable

from ._protocols import AbstractSparsePackReader, SparseCodec, SparseFormat

_codecs: dict[SparseFormat, SparseCodec] = {}
_reader_factories: dict[SparseFormat, Callable[..., AbstractSparsePackReader]] = {}


def register_format(
    fmt: SparseFormat,
    codec: SparseCodec,
    reader_factory: Callable[..., AbstractSparsePackReader],
) -> None:
    """Register a codec and reader factory for a sparse format.

    Args:
        fmt: The sparse format identifier.
        codec: Codec instance implementing both read and write.
        reader_factory: Callable that returns an ``AbstractSparsePackReader``
            when given ``(path, manifest, *, files, matrix_size, dtype, ...)``.
    """
    _codecs[fmt] = codec
    _reader_factories[fmt] = reader_factory


def get_codec(fmt: SparseFormat) -> SparseCodec:
    """Get the registered codec for a sparse format.

    Args:
        fmt: The sparse format identifier.

    Returns:
        The registered ``SparseCodec`` for the format.

    Raises:
        ValueError: If the format has not been registered.
    """
    try:
        return _codecs[fmt]
    except KeyError:
        known = ", ".join(f.value for f in _codecs)
        raise ValueError(
            f"Unregistered sparse format '{fmt.value}'. Known: {known}"
        ) from None


def get_reader_factory(fmt: SparseFormat) -> Callable[..., AbstractSparsePackReader]:
    """Get the registered reader factory for a sparse format.

    Args:
        fmt: The sparse format identifier.

    Returns:
        A callable that creates an ``AbstractSparsePackReader`` for the format.

    Raises:
        ValueError: If the format has not been registered.
    """
    try:
        return _reader_factories[fmt]
    except KeyError:
        known = ", ".join(f.value for f in _reader_factories)
        raise ValueError(
            f"Unregistered sparse format '{fmt.value}'. Known: {known}"
        ) from None
