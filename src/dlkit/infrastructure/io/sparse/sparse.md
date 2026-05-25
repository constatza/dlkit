# Sparse Pack I/O

COO sparse matrix pack storage and retrieval for DLKit, with an OCP format registry
that makes adding new formats (e.g. CSR) a zero-touch change to existing code.

## Architecture

```
_protocols.py   ISP-split protocols + AbstractSparsePackReader ABC (LSP enforcement)
_manifest.py    Manifest dataclasses, _manifest_from_arrays, schema registry
_registry.py    OCP format registry — register_format / get_codec / get_reader_factory
_coo_pack.py    CooPackCodec (SparseCodec) + CooPackReader (AbstractSparsePackReader)
_validation.py  validate_sparse_pack — single load, pure _validate_coo_pack, registry dispatch
_factory.py     Public factory functions — registry dispatch, no format switch statements
__init__.py     Public exports + COO registration
```

### Protocol hierarchy (ISP)

```
SparseWriter  — save()
SparseLoader  — load_arrays(), load_size()
SparseCodec   — SparseWriter + SparseLoader (full codec)

SparsePackReader         — structural typing (runtime_checkable Protocol)
AbstractSparsePackReader — ABC enforcing LSP on concrete readers
```

`CooPackCodec` implements `SparseCodec`.
`CooPackReader` inherits `AbstractSparsePackReader` — missing any method is a class-definition error.

### OCP: adding a new format

1. Create `_csr_pack.py` with `CsrPackCodec(SparseCodec)` and `CsrPackReader(AbstractSparsePackReader)`.
2. In `__init__.py`:
   ```python
   def _open_csr_pack(path, manifest, *, files=None):
       if manifest is not None:
           return CsrPackReader.from_manifest(path, manifest)
       return CsrPackReader.from_directory(path, files=files)

   register_format(SparseFormat.CSR, CsrPackCodec(), _open_csr_pack)
   ```

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
  indices.npy      # int64  (2, total_nnz)   — row/col indices for all samples
  values.npy       # float  (total_nnz,)     — non-zero values for all samples
  nnz_ptr.npy      # int64  (n_samples + 1,) — CSR-style slice pointers
  size.npy         # int64  (2,)             — shared matrix dimensions (rows, cols)
  manifest.json    # optional JSON contract
```

File names are a typed contract via `PackFiles` — rename without changing library code.

## Batch COO format

COO is a 2D matrix format (`row`, `col`, `value` triples for non-zero elements). A
"batch" of matrices is stored as a **flat concatenation with a pointer array**:

- All samples' indices and values are concatenated in order.
- `nnz_ptr[i]` and `nnz_ptr[i+1]` delimit sample `i`'s slice in the flat arrays.
- `size.npy` holds one shared `(rows, cols)` that applies to every sample.

**Reading sample i:** slice `indices[:, nnz_ptr[i]:nnz_ptr[i+1]]` and
`values[nnz_ptr[i]:nnz_ptr[i+1]]`, then build a 2D sparse COO tensor of shape
`(rows, cols)`.

**`build_torch_sparse_stacked([i0, i1, ...])`:** assembles a single 3D sparse COO
tensor of shape `(B, rows, cols)` by prepending a batch-dimension coordinate to
each non-zero element. A non-zero at `(row, col)` in sample `i` becomes
`(b, row, col)` in the 3D tensor, where `b` is the element's position in the
requested index list. The resulting tensor is `is_coalesced=True` because each
sample's segment is already sorted at write time via `_coalesce_pack_payload`.

## Save order guarantee

All validation (array shapes, pointer consistency, manifest cross-checks) runs
**before** any `np.save` call. A validation failure never leaves a partially-written
pack on disk.

## Manifest vs. payload precedence

Passing `manifest=PackManifest(...)` makes that dataclass the authoritative contract
(filenames, `n_samples`, `total_nnz`, `matrix_size`, `dtype`). No JSON sidecar is
required at runtime. When no manifest is provided, one is inferred from the payload
arrays.

## Broadcast (shared matrix)

A pack with `n_samples == 1` broadcasts the single stored matrix to any sample index.
Useful for static graph adjacency matrices shared across all dataset rows.

## Coalescing at write time

`_coalesce_pack_payload` deduplicates COO coordinates once per sample before writing:
duplicate `(row, col)` entries are sum-reduced and coordinates are sorted in
row-major order. Readers can therefore set `is_coalesced=True` unconditionally, which
avoids an in-place sort by PyTorch at read time.

## Examples

### Basic save and read

```python
from pathlib import Path
import numpy as np
from dlkit.infrastructure.io.sparse import save_sparse_pack, open_sparse_pack, validate_sparse_pack

pack_path = Path("matrix_pack")

# Two 2×2 sparse matrices packed together
indices = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.int64)  # (2, 3 total nnz)
values = np.array([2.0, 3.0, 5.0], dtype=np.float64)
nnz_ptr = np.array([0, 2, 3], dtype=np.int64)  # sample 0: [0,2), sample 1: [2,3)

save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(2, 2))
validate_sparse_pack(pack_path)

reader = open_sparse_pack(pack_path)
A0 = reader.build_torch_sparse(0)          # shape (2, 2), sparse
A1 = reader.build_torch_sparse(1)          # shape (2, 2), sparse
batch = reader.build_torch_sparse_stacked([0, 1])  # shape (2, 2, 2), sparse
```

### Custom filenames

```python
from dlkit.infrastructure.io.sparse import PackFiles, save_sparse_pack, open_sparse_pack

files = PackFiles(
    indices="row_index.npy",
    values="entries.npy",
    nnz_ptr="offsets.npy",
    size="dims.npy",
)

save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(512, 512), files=files)
reader = open_sparse_pack(pack_path, files=files)
```

### Injecting a stub loader (testing)

```python
from dlkit.infrastructure.io.sparse._coo_pack import CooPackReader


class StubLoader:
    def load_arrays(self, path, files=None):
        return indices, values, nnz_ptr

    def load_size(self, path, files=None):
        return rows, cols


reader = CooPackReader.from_directory(pack_path, loader=StubLoader())
```

## Dataset integration

`FlexibleDataset` accepts sparse packs as feature entries:

- **Explicit**: `SparseFeature(name="matrix", path=<pack_dir>)`
- **Auto-detected**: `Feature(name="matrix", path=<pack_dir>)` when `path` contains
  sparse payload files.

`SparseFeature` is imported from `dlkit.infrastructure.config.data_entries`.
