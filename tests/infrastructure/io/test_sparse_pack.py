"""Tests for sparse pack I/O and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.infrastructure.io.sparse import (
    PackFiles,
    PackManifest,
    open_sparse_pack,
    save_sparse_pack,
    validate_sparse_pack,
)
from dlkit.infrastructure.io.sparse._coo_pack import CooPackReader


def test_coo_codec_roundtrip(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    assert (saved_sparse_pack / "size.npy").exists()
    assert not (saved_sparse_pack / "values_scale.npy").exists()
    assert not (saved_sparse_pack / "manifest.json").exists()

    reader = open_sparse_pack(saved_sparse_pack)
    for i, matrix in enumerate(dense_matrices):
        sparse = reader.build_torch_sparse(i)
        assert torch.allclose(sparse.to_dense(), torch.from_numpy(matrix).to(sparse.dtype))


def test_size_npy_stores_correct_matrix_dimensions(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size)

    stored_size = np.load(pack_path / "size.npy", allow_pickle=False)
    assert stored_size.shape == (2,)
    assert int(stored_size[0]) == size[0]
    assert int(stored_size[1]) == size[1]


def test_size_inferred_from_data_when_not_explicit(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
    dense_matrices: list[np.ndarray],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size)

    reader = open_sparse_pack(pack_path)
    assert reader.matrix_size == size


def test_size_override_accepted_when_larger_than_inferred(
    tmp_path: Path,
) -> None:
    # 2x2 matrix but we declare it as 4x4 (boundary rows/cols are empty)
    indices = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    nnz_ptr = np.asarray([0, 2], dtype=np.int64)

    pack_path = tmp_path / "pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(4, 4))
    reader = open_sparse_pack(pack_path)
    assert reader.matrix_size == (4, 4)


def test_size_override_rejected_when_smaller_than_inferred(
    tmp_path: Path,
) -> None:
    indices = np.asarray([[0, 2], [0, 2]], dtype=np.int64)
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    nnz_ptr = np.asarray([0, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="smaller than the inferred size"):
        save_sparse_pack(tmp_path / "pack", indices, values, nnz_ptr, size=(2, 2))


def test_custom_payload_filenames_roundtrip(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
    dense_matrices: list[np.ndarray],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "matrix_pack"
    files = PackFiles(
        indices="row_index.npy",
        values="entries.npy",
        nnz_ptr="offsets.npy",
        size="dims.npy",
    )

    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, files=files)
    reader = open_sparse_pack(pack_path, files=files)

    assert (pack_path / "row_index.npy").exists()
    assert (pack_path / "entries.npy").exists()
    assert (pack_path / "offsets.npy").exists()
    assert (pack_path / "dims.npy").exists()
    assert not (pack_path / "indices.npy").exists()
    assert torch.allclose(
        reader.build_torch_sparse(1).to_dense(),
        torch.from_numpy(dense_matrices[1]).to(dtype=reader.build_torch_sparse(1).dtype),
    )


def test_open_sparse_pack_requires_size_file(saved_sparse_pack: Path) -> None:
    (saved_sparse_pack / "size.npy").unlink()

    with pytest.raises(FileNotFoundError):
        open_sparse_pack(saved_sparse_pack)


def test_per_sample_builder_matches_dense(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    reader = open_sparse_pack(saved_sparse_pack)

    for i, dense in enumerate(dense_matrices):
        sparse = reader.build_torch_sparse(i)
        expected = torch.from_numpy(dense).to(dtype=sparse.dtype)
        assert torch.allclose(sparse.to_dense(), expected)


def test_batch_builder_returns_correct_sparse_list(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    # build_torch_sparse_batch is a concrete convenience method on CooPackReader (not abstract)
    reader = CooPackReader.from_directory(saved_sparse_pack)
    batch = reader.build_torch_sparse_batch([0, 2])

    assert len(batch) == 2
    assert batch[0].is_sparse
    assert batch[1].is_sparse
    assert torch.allclose(
        batch[0].to_dense(), torch.from_numpy(dense_matrices[0]).to(batch[0].dtype)
    )
    assert torch.allclose(
        batch[1].to_dense(), torch.from_numpy(dense_matrices[2]).to(batch[1].dtype)
    )


def test_stacked_builder_returns_correct_sparse_tensor(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    reader = open_sparse_pack(saved_sparse_pack)
    stacked = reader.build_torch_sparse_stacked([0, 2])

    assert stacked.is_sparse
    assert stacked.shape[0] == 2
    assert torch.allclose(
        stacked[0].to_dense(), torch.from_numpy(dense_matrices[0]).to(stacked.dtype)
    )
    assert torch.allclose(
        stacked[1].to_dense(), torch.from_numpy(dense_matrices[2]).to(stacked.dtype)
    )


def test_save_sparse_pack_coalesces_duplicate_coordinates_on_write(tmp_path: Path) -> None:
    """Duplicate COO coordinates are canonicalized once at save time."""
    pack_path = tmp_path / "dupe_pack"
    indices = np.asarray(
        [
            [0, 0, 1, 1],
            [1, 1, 2, 2],
        ],
        dtype=np.int64,
    )
    values = np.asarray([1.0, 3.0, 2.0, 5.0], dtype=np.float32)
    nnz_ptr = np.asarray([0, 2, 4], dtype=np.int64)

    save_sparse_pack(pack_path, indices, values, nnz_ptr, size=(2, 3))
    reader = open_sparse_pack(pack_path)

    sample0 = reader.build_torch_sparse(0)
    sample1 = reader.build_torch_sparse(1)

    assert sample0._nnz() == 1
    assert sample1._nnz() == 1

    expected0 = torch.zeros(sample0.shape, dtype=sample0.dtype)
    expected0[0, 1] = 4.0
    expected1 = torch.zeros(sample1.shape, dtype=sample1.dtype)
    expected1[1, 2] = 7.0
    torch.testing.assert_close(sample0.to_dense(), expected0)
    torch.testing.assert_close(sample1.to_dense(), expected1)


def test_validate_sparse_pack_requires_size_file(saved_sparse_pack: Path) -> None:
    (saved_sparse_pack / "size.npy").unlink()

    with pytest.raises(FileNotFoundError):
        validate_sparse_pack(saved_sparse_pack)


def test_open_sparse_pack_with_custom_files_requires_size_file(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "custom_pack"
    files = PackFiles(indices="i.npy", values="v.npy", nnz_ptr="p.npy", size="d.npy")
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, files=files)
    (pack_path / "d.npy").unlink()

    with pytest.raises(FileNotFoundError):
        open_sparse_pack(pack_path, files=files)


def test_manifest_validates_n_samples_and_total_nnz(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size)

    wrong_n_samples = PackManifest(
        n_samples=99,
        matrix_size=size,
        dtype="float64",
        total_nnz=int(values.size),
    )
    with pytest.raises(ValueError, match="n_samples"):
        open_sparse_pack(pack_path, manifest=wrong_n_samples)


def test_validate_sparse_pack_detects_mismatched_nnz_ptr(
    saved_sparse_pack: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
) -> None:
    _, _, nnz_ptr, _ = coo_pack_arrays
    broken_ptr = nnz_ptr.copy()
    broken_ptr[-1] = broken_ptr[-1] + 1
    np.save(saved_sparse_pack / "nnz_ptr.npy", broken_ptr)

    with pytest.raises(ValueError, match="nnz_ptr last value"):
        validate_sparse_pack(saved_sparse_pack)


# ---------------------------------------------------------------------------
# Lossless reconstruction tests
# ---------------------------------------------------------------------------


def test_per_sample_reconstruction_lossless(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    """Every sample reconstructs losslessly from the pack via build_torch_sparse."""
    reader = open_sparse_pack(saved_sparse_pack)

    for i, matrix in enumerate(dense_matrices):
        recovered = reader.build_torch_sparse(i).to_dense()
        expected = torch.from_numpy(matrix).to(dtype=recovered.dtype)
        torch.testing.assert_close(recovered, expected)


def test_stacked_reconstruction_all_samples(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    """build_torch_sparse_stacked reconstructs all samples losslessly."""
    reader = open_sparse_pack(saved_sparse_pack)
    all_indices = list(range(len(dense_matrices)))
    stacked = reader.build_torch_sparse_stacked(all_indices)

    rows, cols = dense_matrices[0].shape
    assert stacked.shape == (len(dense_matrices), rows, cols)
    assert stacked.is_sparse

    for batch_pos, matrix in enumerate(dense_matrices):
        recovered = stacked[batch_pos].to_dense()
        expected = torch.from_numpy(matrix).to(dtype=stacked.dtype)
        torch.testing.assert_close(recovered, expected)


def test_stacked_reconstruction_float32(tmp_path: Path) -> None:
    """float32 pack reconstructs losslessly through the stacked batch builder."""
    matrices = [
        np.array([[1.5, 0.0, 0.0], [0.0, 2.5, 0.0]], dtype=np.float32),
        np.array([[0.0, 3.5, 0.0], [0.0, 0.0, 4.5]], dtype=np.float32),
    ]
    row_parts, col_parts, val_parts, nnz_ptr = [], [], [], [0]
    for m in matrices:
        rows, cols = np.nonzero(m)
        row_parts.append(rows.astype(np.int64))
        col_parts.append(cols.astype(np.int64))
        val_parts.append(m[rows, cols])
        nnz_ptr.append(nnz_ptr[-1] + int(rows.size))

    indices = np.vstack([np.concatenate(row_parts), np.concatenate(col_parts)])
    values = np.concatenate(val_parts).astype(np.float32)
    ptr = np.asarray(nnz_ptr, dtype=np.int64)
    size = matrices[0].shape

    pack_path = tmp_path / "f32_pack"
    save_sparse_pack(pack_path, indices, values, ptr, size)

    reader = open_sparse_pack(pack_path)
    stacked = reader.build_torch_sparse_stacked([0, 1])

    assert stacked.dtype == torch.float32
    for b, matrix in enumerate(matrices):
        recovered = stacked[b].to_dense()
        torch.testing.assert_close(recovered, torch.from_numpy(matrix))


def test_broadcast_pack_stacked_reconstruction(tmp_path: Path) -> None:
    """Single-sample broadcast pack: stacked builder replicates the matrix to any B."""
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float64)
    rows, cols = np.nonzero(matrix)
    indices = np.vstack([rows.astype(np.int64), cols.astype(np.int64)])
    values = matrix[rows, cols]
    nnz_ptr = np.array([0, len(values)], dtype=np.int64)

    pack_path = tmp_path / "broadcast_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, matrix.shape)

    reader = open_sparse_pack(pack_path)
    assert reader.n_samples == 1

    stacked = reader.build_torch_sparse_stacked([0, 1, 2])
    assert stacked.shape == (3, *matrix.shape)
    expected = torch.from_numpy(matrix).to(dtype=stacked.dtype)
    for b in range(3):
        torch.testing.assert_close(stacked[b].to_dense(), expected)


def test_stacked_reconstruction_empty_batch(saved_sparse_pack: Path) -> None:
    """Empty sample list produces a valid zero-batch sparse tensor."""
    reader = open_sparse_pack(saved_sparse_pack)
    stacked = reader.build_torch_sparse_stacked([])

    rows, cols = reader.matrix_size
    assert stacked.shape == (0, rows, cols)
    assert stacked.is_sparse
