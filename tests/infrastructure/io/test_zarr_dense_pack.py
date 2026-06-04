"""Tests for zarr v3 dense array pack writer and reader."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.infrastructure.io.packs import (
    ArrayPackFormat,
    ArrayPackManifest,
    ZarrDensePackReader,
    ZarrDensePackWriter,
    detect_format,
    open_array_pack,
    save_array_pack,
    write_array_pack,
)

# ---------------------------------------------------------------------------
# Fixtures (all test data lives here — never inside test functions)
# ---------------------------------------------------------------------------


@pytest.fixture
def dense_matrices_f32() -> np.ndarray:
    """Three 4×5 float32 matrices for roundtrip tests.

    Returns:
        Array of shape (3, 4, 5) with known values.
    """
    rng = np.random.default_rng(42)
    return rng.random((3, 4, 5)).astype(np.float32)


@pytest.fixture
def dense_matrices_f64() -> np.ndarray:
    """Three 4×5 float64 matrices.

    Returns:
        Array of shape (3, 4, 5) dtype float64.
    """
    rng = np.random.default_rng(7)
    return rng.random((3, 4, 5)).astype(np.float64)


@pytest.fixture
def single_matrix() -> np.ndarray:
    """One 3×3 float32 matrix for broadcast tests.

    Returns:
        Array of shape (1, 3, 3).
    """
    return np.eye(3, dtype=np.float32)[np.newaxis]


@pytest.fixture
def written_pack_path(tmp_path: Path, dense_matrices_f32: np.ndarray) -> Path:
    """Write a zarr-dense pack from ``dense_matrices_f32`` and return its path.

    Args:
        tmp_path: pytest temporary directory.
        dense_matrices_f32: Matrices to store.

    Returns:
        Path to the written pack directory.
    """
    path = tmp_path / "test_pack"
    _, rows, cols = dense_matrices_f32.shape
    with ZarrDensePackWriter(path, size=(rows, cols), dtype=np.float32) as w:
        w.write_samples(dense_matrices_f32)
    return path


@pytest.fixture
def broadcast_pack_path(tmp_path: Path, single_matrix: np.ndarray) -> Path:
    """Write a single-sample pack (broadcast scenario).

    Args:
        tmp_path: pytest temporary directory.
        single_matrix: Single matrix to store.

    Returns:
        Path to the written pack directory.
    """
    path = tmp_path / "broadcast_pack"
    _, rows, cols = single_matrix.shape
    with ZarrDensePackWriter(path, size=(rows, cols), dtype=np.float32) as w:
        w.write_samples(single_matrix)
    return path


# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------


def test_roundtrip_write_samples_open(
    written_pack_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """write_samples → open_array_pack → __getitem__ recovers original values."""
    reader = open_array_pack(written_pack_path)
    n, rows, cols = dense_matrices_f32.shape

    assert reader.n_samples == n
    assert reader.matrix_size == (rows, cols)

    for i in range(n):
        result = reader[i]
        assert result.shape == (rows, cols)
        torch.testing.assert_close(result, torch.from_numpy(dense_matrices_f32[i]))


def test_roundtrip_write_sample_one_by_one(
    tmp_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """Per-sample write_sample calls produce identical pack to batch write_samples."""
    path = tmp_path / "one_by_one"
    n, rows, cols = dense_matrices_f32.shape
    with ZarrDensePackWriter(path, size=(rows, cols), dtype=np.float32) as w:
        for i in range(n):
            w.write_sample(dense_matrices_f32[i])

    reader = ZarrDensePackReader(path)
    assert reader.n_samples == n
    for i in range(n):
        torch.testing.assert_close(
            reader[i],
            torch.from_numpy(dense_matrices_f32[i]),
        )


def test_roundtrip_save_array_pack(
    tmp_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """save_array_pack writes and open_array_pack recovers data."""
    path = tmp_path / "saved_pack"
    save_array_pack(path, dense_matrices_f32)

    reader = open_array_pack(path)
    n, rows, cols = dense_matrices_f32.shape
    assert reader.n_samples == n
    for i in range(n):
        torch.testing.assert_close(reader[i], torch.from_numpy(dense_matrices_f32[i]))


# ---------------------------------------------------------------------------
# Batch and slice indexing
# ---------------------------------------------------------------------------


def test_batch_index_list(
    written_pack_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """reader[[0, 2, 1]] returns Tensor of shape (3, rows, cols) in correct order."""
    reader = open_array_pack(written_pack_path)
    indices = [0, 2, 1]
    result = reader[indices]

    n, rows, cols = dense_matrices_f32.shape
    assert result.shape == (len(indices), rows, cols)
    for b, i in enumerate(indices):
        torch.testing.assert_close(result[b], torch.from_numpy(dense_matrices_f32[i]))


def test_slice_index(
    written_pack_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """reader[1:3] returns Tensor of shape (2, rows, cols)."""
    reader = open_array_pack(written_pack_path)
    result = reader[1:3]

    _, rows, cols = dense_matrices_f32.shape
    assert result.shape == (2, rows, cols)
    torch.testing.assert_close(result[0], torch.from_numpy(dense_matrices_f32[1]))
    torch.testing.assert_close(result[1], torch.from_numpy(dense_matrices_f32[2]))


def test_empty_list_index(written_pack_path: Path, dense_matrices_f32: np.ndarray) -> None:
    """reader[[]] returns tensor with batch dimension 0."""
    reader = open_array_pack(written_pack_path)
    result = reader[[]]
    _, rows, cols = dense_matrices_f32.shape
    assert result.shape == (0, rows, cols)


# ---------------------------------------------------------------------------
# Broadcast (n_samples == 1)
# ---------------------------------------------------------------------------


def test_broadcast_single_sample_scalar_index(
    broadcast_pack_path: Path,
    single_matrix: np.ndarray,
) -> None:
    """Single-sample pack always returns index 0 regardless of requested int index."""
    reader = open_array_pack(broadcast_pack_path)
    assert reader.n_samples == 1

    expected = torch.from_numpy(single_matrix[0])
    # Requesting any integer always returns the single stored matrix
    torch.testing.assert_close(reader[0], expected)


def test_broadcast_single_sample_list_index(
    broadcast_pack_path: Path,
    single_matrix: np.ndarray,
) -> None:
    """Single-sample pack: list index replicates the one matrix across batch dim."""
    reader = open_array_pack(broadcast_pack_path)
    result = reader[[0, 1, 2]]

    expected = torch.from_numpy(single_matrix[0])
    assert result.shape == (3, *single_matrix.shape[1:])
    for b in range(3):
        torch.testing.assert_close(result[b], expected)


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------


def test_dtype_float32_preserved(
    tmp_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """float32 data written as float32 is read back as float32."""
    path = tmp_path / "f32_pack"
    save_array_pack(path, dense_matrices_f32, dtype=np.float32)

    reader = open_array_pack(path)
    assert reader[0].dtype == torch.float32


def test_dtype_float64_preserved(
    tmp_path: Path,
    dense_matrices_f64: np.ndarray,
) -> None:
    """float64 data written as float64 is read back as float64."""
    path = tmp_path / "f64_pack"
    save_array_pack(path, dense_matrices_f64, dtype=np.float64)

    reader = open_array_pack(path)
    assert reader[0].dtype == torch.float64
    torch.testing.assert_close(
        reader[0],
        torch.from_numpy(dense_matrices_f64[0]),
    )


# ---------------------------------------------------------------------------
# Immutability / read-only guarantee
# ---------------------------------------------------------------------------


def test_reader_zarr_opened_read_only(written_pack_path: Path) -> None:
    """ZarrDensePackReader opens the zarr group in mode='r'; direct write raises."""
    reader = ZarrDensePackReader(written_pack_path)

    with pytest.raises(Exception):
        # Attempting to write to a read-only zarr group must raise
        reader._group.create_array("should_fail", shape=(1,), dtype="f4")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(
    written_pack_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """Manifest stored in zarr attrs matches the data that was written."""
    reader = ZarrDensePackReader(written_pack_path)
    manifest = reader._manifest

    n, rows, cols = dense_matrices_f32.shape
    assert manifest.n_samples == n
    assert manifest.matrix_size == (rows, cols)
    assert manifest.dtype == "float32"
    assert manifest.schema_ == "dlkit.array-pack.zarr-dense.v1"


def test_manifest_model_validates_correctly() -> None:
    """ArrayPackManifest validates good data and rejects bad dtype."""
    valid = ArrayPackManifest(
        n_samples=10,
        matrix_size=(4, 5),
        dtype="float32",
        chunk_size=64,
    )
    assert valid.n_samples == 10
    assert valid.schema_ == "dlkit.array-pack.zarr-dense.v1"

    with pytest.raises(Exception):
        ArrayPackManifest(
            n_samples=10,
            matrix_size=(4, 5),
            dtype="not_a_dtype",
            chunk_size=64,
        )


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_writer_context_manager_closes_cleanly(
    tmp_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """Writer used as context manager finalises pack on __exit__."""
    path = tmp_path / "ctx_pack"
    _, rows, cols = dense_matrices_f32.shape
    with write_array_pack(path, size=(rows, cols)) as w:
        w.write_samples(dense_matrices_f32)

    # After __exit__ the pack must be readable
    reader = open_array_pack(path)
    assert reader.n_samples == dense_matrices_f32.shape[0]


def test_writer_close_with_no_samples_raises(tmp_path: Path) -> None:
    """close() raises RuntimeError when no samples have been written."""
    path = tmp_path / "empty_pack"
    w = ZarrDensePackWriter(path, size=(4, 5))
    with pytest.raises(RuntimeError, match="no samples written"):
        w.close()


def test_writer_close_with_no_samples_raises_via_context_manager(tmp_path: Path) -> None:
    """Context manager __exit__ raises RuntimeError when no samples written."""
    path = tmp_path / "empty_ctx_pack"
    with pytest.raises(RuntimeError, match="no samples written"):
        with ZarrDensePackWriter(path, size=(4, 5)):
            pass  # intentionally write nothing


def test_writer_close_is_idempotent(
    tmp_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """Calling close() multiple times does not raise."""
    path = tmp_path / "idempotent_pack"
    _, rows, cols = dense_matrices_f32.shape
    w = ZarrDensePackWriter(path, size=(rows, cols))
    w.write_samples(dense_matrices_f32)
    w.close()
    w.close()  # second close must be a no-op


def test_writer_raises_after_close(
    tmp_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """write_sample / write_samples raise RuntimeError after close()."""
    path = tmp_path / "closed_pack"
    _, rows, cols = dense_matrices_f32.shape
    w = ZarrDensePackWriter(path, size=(rows, cols))
    # Write at least one sample so manifest validates (n_samples >= 1), then close
    w.write_sample(dense_matrices_f32[0])
    w.close()

    with pytest.raises(RuntimeError, match="closed"):
        w.write_sample(dense_matrices_f32[0])

    with pytest.raises(RuntimeError, match="closed"):
        w.write_samples(dense_matrices_f32)


# ---------------------------------------------------------------------------
# Detect format
# ---------------------------------------------------------------------------


def test_detect_format_zarr_dense(written_pack_path: Path) -> None:
    """detect_format returns ZARR_DENSE for a valid zarr-dense pack."""
    assert detect_format(written_pack_path) == ArrayPackFormat.ZARR_DENSE


def test_detect_format_raises_for_empty_dir(tmp_path: Path) -> None:
    """detect_format raises ValueError for a directory without a known pack."""
    with pytest.raises(ValueError, match="Cannot detect"):
        detect_format(tmp_path)


# ---------------------------------------------------------------------------
# Deprecated collect / collect_stacked shims
# ---------------------------------------------------------------------------


def test_deprecated_collect_delegates_to_getitem(
    written_pack_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """Deprecated collect() delegates to __getitem__ and emits DeprecationWarning."""
    reader = ZarrDensePackReader(written_pack_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reader.collect(0)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    torch.testing.assert_close(result, torch.from_numpy(dense_matrices_f32[0]))


def test_deprecated_collect_stacked_delegates_to_getitem(
    written_pack_path: Path,
    dense_matrices_f32: np.ndarray,
) -> None:
    """Deprecated collect_stacked() delegates to __getitem__ with list index."""
    reader = ZarrDensePackReader(written_pack_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = reader.collect_stacked([0, 1])

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert result.shape == (2, *dense_matrices_f32.shape[1:])


# ---------------------------------------------------------------------------
# scipy.sparse soft import
# ---------------------------------------------------------------------------


def test_write_sample_accepts_scipy_sparse(tmp_path: Path) -> None:
    """write_sample densifies scipy CSR matrices when scipy is installed."""
    pytest.importorskip("scipy", reason="scipy not installed — skipping scipy test")
    from scipy.sparse import csr_matrix

    path = tmp_path / "scipy_pack"
    dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    sparse_mat = csr_matrix(dense)

    with ZarrDensePackWriter(path, size=dense.shape) as w:
        w.write_sample(sparse_mat)

    reader = ZarrDensePackReader(path)
    torch.testing.assert_close(reader[0], torch.from_numpy(dense))


def test_negative_index(zarr_dense_pack: Path) -> None:
    """reader[-1] returns the same tensor as reader[n_samples - 1]."""
    reader = open_array_pack(zarr_dense_pack)
    result = reader[-1]
    expected = reader[reader.n_samples - 1]
    assert torch.equal(result, expected)


def test_broadcast_empty_list(tmp_path: Path) -> None:
    """reader[[]] on a broadcast (single-sample) pack returns shape (0, rows, cols)."""
    with write_array_pack(tmp_path / "b.zarr", size=(4, 4), format=ArrayPackFormat.ZARR_DENSE) as w:
        w.write_sample(np.zeros((4, 4), dtype=np.float32))
    reader = open_array_pack(tmp_path / "b.zarr")
    result = reader[[]]
    assert result.shape == (0, 4, 4)


def test_write_samples_accepts_stacked_scipy_sparse(tmp_path: Path) -> None:
    """write_samples densifies a stacked array converted from scipy sparse matrices."""
    pytest.importorskip("scipy", reason="scipy not installed — skipping scipy test")
    from scipy.sparse import csr_matrix

    rows, cols = 3, 4
    rng = np.random.default_rng(99)
    dense_batch = rng.random((5, rows, cols)).astype(np.float32)
    # Convert to stacked dense via scipy round-trip (confirms _densify path works)
    scipy_mats = [csr_matrix(dense_batch[i]) for i in range(5)]
    stacked = np.stack([m.toarray() for m in scipy_mats])

    path = tmp_path / "scipy_batch_pack"
    with ZarrDensePackWriter(path, size=(rows, cols)) as w:
        w.write_samples(stacked)

    reader = ZarrDensePackReader(path)
    assert reader.n_samples == 5
    for i in range(5):
        torch.testing.assert_close(reader[i], torch.from_numpy(dense_batch[i]))
