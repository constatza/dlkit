# Array Pack Module (`infrastructure/io/packs/`)

## Purpose

Out-of-memory matrix storage backed by zarr v3 dense arrays with BloscCodec(zstd + bitshuffle).
Replaces the old COO sparse pack module (`infrastructure/io/sparse/`).

Each *array pack* is a zarr group directory (`*.zarr/`) containing one dataset per sample,
stored as a 2-D dense array.  Random access by sample index is O(1) and bounded in RAM to
one chunk regardless of dataset size.

## Module Structure

| File | Responsibility |
|------|---------------|
| `_protocols.py` | `IArrayPackReader` protocol — `__getitem__`, `n_samples`, `matrix_size` |
| `_manifest.py` | `ArrayPackManifest` Pydantic model — zarr group metadata |
| `_registry.py` | `_FormatRegistry` — maps `ArrayPackFormat` → `IArrayPackReader` factory |
| `_zarr_dense.py` | `ZarrDensePackReader` / `ZarrDensePackWriter` — zarr v3 implementation |
| `_factory.py` | `open_array_pack`, `write_array_pack`, `save_array_pack` — public factory functions |
| `__init__.py` | Public re-exports and default format registration |

## Key Types

- `IArrayPackReader` — protocol satisfied by `ZarrDensePackReader`.
  - `reader[i]` → `Tensor[rows, cols]`
  - `reader[list[int]]` → `Tensor[B, rows, cols]`
  - `reader.n_samples: int`
  - `reader.matrix_size: tuple[int, int]`

- `ZarrDensePackWriter` — context-manager writer; call `write_sample(data)` per sample.

- `ArrayPackFormat` — enum of registered backends (`ZARR_DENSE` is the default).

- `ArrayPackManifest` — frozen Pydantic model stored as zarr group attributes.

## Public API

```python
from dlkit.infrastructure.io.packs import (
    open_array_pack,     # open_array_pack(path) -> IArrayPackReader
    write_array_pack,    # write_array_pack(path, size) -> ZarrDensePackWriter
    save_array_pack,     # save_array_pack(path, samples)  (eager, list-based)
    ArrayPackFormat,
    register_format,
)
```

Top-level re-exports via `dlkit.io`:
```python
from dlkit.io import open_array_pack, save_array_pack, write_array_pack, ArrayPackFormat
```

## Extending with New Backends

Register a new reader factory without modifying library code:

```python
from dlkit.infrastructure.io.packs import register_format, ArrayPackFormat

class MyPackReader:
    ...

register_format(ArrayPackFormat.MY_FORMAT, lambda path: MyPackReader(path))
```

## Breaking Changes from `infrastructure/io/sparse/`

- `CooPackReader`, `CooPackWriter`, `ZarrPackReader`, `ZarrPackWriter` — removed.
- `SparseFormat`, `PackManifest`, `PackFiles` — removed.
- `open_sparse_pack`, `save_sparse_pack`, `write_coo_pack`, `write_zarr_pack` — removed.
- `SparseFeature` is now a type alias for `MatrixFeature`; the `files` field is gone.
- `FlexibleDataset` no longer has a `_sparse_bindings` attribute.
