# Sparse Pack I/O

COO sparse matrix pack storage and retrieval for DLKit, with an OCP format registry
that makes adding new formats (e.g. CSR) a zero-touch change to existing code.

## Architecture

```
_protocols.py   ISP-split protocols + AbstractSparsePackReader ABC (LSP enforcement)
_manifest.py    Manifest dataclasses, _manifest_from_arrays, _normalize_value_scale
_registry.py    OCP format registry — register_format / get_codec / get_reader_factory
_coo_pack.py    CooPackCodec (SparseCodec) + CooPackReader (AbstractSparsePackReader)
_validation.py  validate_sparse_pack — single load, pure _validate_coo_pack, registry dispatch
_factory.py     Public factory functions — registry dispatch, no format switch statements
__init__.py     Public exports + COO registration
```

### Protocol hierarchy (ISP)

```
SparseWriter  — save()
SparseLoader  — load_arrays(), load_value_scale()
SparseCodec   — SparseWriter + SparseLoader (full codec)

SparsePackReader       — structural typing (runtime_checkable Protocol)
AbstractSparsePackReader — ABC enforcing LSP on concrete readers
```

`CooPackCodec` implements `SparseCodec`.
`CooPackReader` inherits `AbstractSparsePackReader` — missing any method is a class-definition error.

### OCP: adding a new format

1. Create `_csr_pack.py` with `CsrPackCodec(SparseCodec)` and `CsrPackReader(AbstractSparsePackReader)`.
2. In `__init__.py`: `register_format(SparseFormat.CSR, CsrPackCodec(), CsrPackReader)`.

Zero changes to `_factory.py`, `_validation.py`, or `_protocols.py`.

## Public API

```python
from dlkit.infrastructure.io.sparse import (
    save_sparse_pack,
    open_sparse_pack,
    validate_sparse_pack,
    is_sparse_pack_dir,
    SparseFormat,  # COO, CSR
    SparseCodec,  # full codec protocol
    SparseLoader,  # read-only protocol (ISP)
    SparseWriter,  # write-only protocol (ISP)
    SparsePackReader,  # structural reader protocol
    AbstractSparsePackReader,  # ABC for concrete readers
    PackFiles,  # payload filename contract
    PackManifest,  # typed manifest dataclass
    CooPackCodec,  # COO codec (read + write)
    CooPackReader,  # COO reader
    register_format,  # OCP extension point
    register_manifest_schema,
    COO_PACK_SCHEMA,
)
```

> `SparseFeature` is a config-layer entity — import it from
> `dlkit.infrastructure.config.data_entries`, not from this package.

## On-disk layout

```
<pack_dir>/
  indices.npy        # int64 (2, total_nnz)
  values.npy         # float dtype
  nnz_ptr.npy        # int64 (n_samples + 1)
  values_scale.npy   # float64 scalar  (optional — defaults to 1.0 if absent)
  manifest.json      # optional JSON contract
```

File names are a typed contract via `PackFiles` — rename without changing library code.

## Save order guarantee

All validation (array shapes, pointer consistency, manifest cross-checks) runs
**before** any `np.save` call. A validation failure never leaves a partially-written
pack on disk.

## Value scale

- Stored values are in stored-space: `A_original = A_stored * value_scale`.
- `value_scale` defaults to `1.0`.
- Denormalization is opt-in: `build_torch_sparse(..., denormalize=True)`.

## Manifest vs. payload precedence

Passing `manifest=PackManifest(...)` makes that dataclass the authoritative contract
(`value_scale`, filenames, `n_samples`, `total_nnz`). No JSON sidecar is required at
runtime. When no manifest is provided, one is inferred from the payload arrays.

## Broadcast (shared matrix)

A pack with `n_samples == 1` broadcasts the single stored matrix to any sample index.
Useful for static graph adjacency matrices shared across all dataset rows.

## Examples

### Basic save and read

```python
from pathlib import Path
import numpy as np
from dlkit.infrastructure.io.sparse import save_sparse_pack, open_sparse_pack, validate_sparse_pack

pack_path = Path("matrix_pack")
indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
values = np.array([2.0, 3.0], dtype=np.float64)
nnz_ptr = np.array([0, 2], dtype=np.int64)

save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(2, 2))
validate_sparse_pack(pack_path)

reader = open_sparse_pack(pack_path)
A0 = reader.build_torch_sparse(0)
```

### Custom filenames

```python
from dlkit.infrastructure.io.sparse import PackFiles, save_sparse_pack, open_sparse_pack

files = PackFiles(
    indices="row_index.npy",
    values="entries.npy",
    nnz_ptr="offsets.npy",
    values_scale="scale.npy",
)

save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(512, 512), files=files)
reader = open_sparse_pack(pack_path, files=files)
```

### Value scale / denormalization

```python
save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(512, 512), value_scale=10.0)

reader = open_sparse_pack(pack_path)
A_stored = reader.build_torch_sparse(0)  # stored-space
A_original = reader.build_torch_sparse(0, denormalize=True)  # × 10.0
```

### Injecting a stub loader (testing)

```python
from dlkit.infrastructure.io.sparse._coo_pack import CooPackReader


class StubLoader:
    def load_arrays(self, path, files=None):
        return indices, values, nnz_ptr

    def load_value_scale(self, path, files=None):
        return 1.0


reader = CooPackReader(pack_path, loader=StubLoader())
```

## Dataset integration

`FlexibleDataset` accepts sparse packs as feature entries:

- **Explicit**: `SparseFeature(name="matrix", path=<pack_dir>, denormalize=False)`
- **Auto-detected**: `Feature(name="matrix", path=<pack_dir>)` when `path` contains
  sparse payload files.

`SparseFeature` is imported from `dlkit.infrastructure.config.data_entries`.
