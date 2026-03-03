"""Public sparse pack I/O API."""

# Import _coo_pack first to trigger COO format registration before any factory call.
from ._coo_pack import CooPackCodec, CooPackReader
from ._factory import is_sparse_pack_dir, open_sparse_pack, save_sparse_pack
from ._manifest import COO_PACK_SCHEMA, PackFiles, PackManifest, register_manifest_schema
from ._protocols import (
    AbstractSparsePackReader,
    SparseCodec,
    SparseFormat,
    SparseLoader,
    SparsePackReader,
    SparseWriter,
)
from ._registry import register_format
from ._validation import validate_sparse_pack

# Register built-in formats. Adding a new format (e.g. CSR) requires only
# creating _csr_pack.py and calling register_format here — zero other changes.
register_format(SparseFormat.COO, CooPackCodec(), CooPackReader)

__all__ = [
    "AbstractSparsePackReader",
    "COO_PACK_SCHEMA",
    "CooPackCodec",
    "CooPackReader",
    "PackFiles",
    "PackManifest",
    "SparseCodec",
    "SparseFormat",
    "SparseLoader",
    "SparsePackReader",
    "SparseWriter",
    "is_sparse_pack_dir",
    "open_sparse_pack",
    "register_format",
    "register_manifest_schema",
    "save_sparse_pack",
    "validate_sparse_pack",
]
