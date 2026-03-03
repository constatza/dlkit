"""Tests for sparse pack I/O and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.tools.io.sparse import (
    PackFiles,
    PackManifest,
    SparsePackReader,
    open_sparse_pack,
    save_sparse_pack,
    validate_sparse_pack,
)
from dlkit.tools.io.sparse._coo_pack import CooPackReader


def test_coo_codec_roundtrip(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    assert (saved_sparse_pack / "values_scale.npy").exists()
    assert not (saved_sparse_pack / "manifest.json").exists()

    reader = open_sparse_pack(saved_sparse_pack)
    for i, matrix in enumerate(dense_matrices):
        sparse = reader.build_torch_sparse(i)
        assert torch.allclose(sparse.to_dense(), torch.from_numpy(matrix).to(sparse.dtype))


def test_pack_manifest_dataclass_exposes_scale_contract() -> None:
    manifest = PackManifest()
    assert manifest.value_scale == 1.0
    assert manifest.files.values_scale.endswith(".npy")


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
        values_scale="scale.npy",
    )

    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, files=files)
    reader = open_sparse_pack(pack_path, files=files)

    assert (pack_path / "row_index.npy").exists()
    assert (pack_path / "entries.npy").exists()
    assert (pack_path / "offsets.npy").exists()
    assert (pack_path / "scale.npy").exists()
    assert not (pack_path / "indices.npy").exists()
    assert torch.allclose(
        reader.build_torch_sparse(1).to_dense(),
        torch.from_numpy(dense_matrices[1]).to(dtype=reader.build_torch_sparse(1).dtype),
    )


def test_open_sparse_pack_reads_legacy_directory_without_scale_file(
    saved_sparse_pack: Path,
    dense_matrices: list[np.ndarray],
) -> None:
    (saved_sparse_pack / "values_scale.npy").unlink()

    reader = open_sparse_pack(saved_sparse_pack)

    assert isinstance(reader, CooPackReader)
    assert isinstance(reader, SparsePackReader)
    assert reader.value_scale == 1.0
    assert torch.allclose(
        reader.build_torch_sparse(0).to_dense(),
        torch.from_numpy(dense_matrices[0]).to(dtype=reader.build_torch_sparse(0).dtype),
    )


def test_manifest_contract_precedence_for_value_scale(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "matrix_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, value_scale=2.0)

    contract = PackManifest(
        n_samples=3,
        matrix_size=size,
        dtype="float64",
        total_nnz=int(values.size),
        value_scale=5.0,
        files=PackFiles(),
    )
    reader = open_sparse_pack(pack_path, manifest=contract)

    stored = reader.build_torch_sparse(0).to_dense()
    denorm = reader.build_torch_sparse(0, denormalize=True).to_dense()
    assert torch.allclose(denorm, stored * 5.0)


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
    reader = open_sparse_pack(saved_sparse_pack)
    batch = reader.build_torch_sparse_batch([0, 2])

    assert len(batch) == 2
    assert batch[0].is_sparse
    assert batch[1].is_sparse
    assert torch.allclose(batch[0].to_dense(), torch.from_numpy(dense_matrices[0]).to(batch[0].dtype))
    assert torch.allclose(batch[1].to_dense(), torch.from_numpy(dense_matrices[2]).to(batch[1].dtype))


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


def test_denormalize_applies_value_scale(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "scaled_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, value_scale=4.0)

    reader = open_sparse_pack(pack_path)
    stored = reader.build_torch_sparse(1, denormalize=False).to_dense()
    denorm = reader.build_torch_sparse(1, denormalize=True).to_dense()
    assert torch.allclose(denorm, stored * 4.0)


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

    sample0 = reader.build_torch_sparse(0, coalesce=False)
    sample1 = reader.build_torch_sparse(1, coalesce=False)

    assert sample0._nnz() == 1
    assert sample1._nnz() == 1

    expected0 = torch.zeros(sample0.shape, dtype=sample0.dtype)
    expected0[0, 1] = 4.0
    expected1 = torch.zeros(sample1.shape, dtype=sample1.dtype)
    expected1[1, 2] = 7.0
    torch.testing.assert_close(sample0.to_dense(), expected0)
    torch.testing.assert_close(sample1.to_dense(), expected1)


def test_validate_sparse_pack_accepts_legacy_without_scale_file(
    saved_sparse_pack: Path,
) -> None:
    (saved_sparse_pack / "values_scale.npy").unlink()
    validate_sparse_pack(saved_sparse_pack)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, np.nan, np.inf])
def test_save_sparse_pack_rejects_invalid_value_scale(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
    bad_scale: float,
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "matrix_pack"

    with pytest.raises(ValueError, match="value_scale"):
        save_sparse_pack(pack_path, indices, values, nnz_ptr, size, value_scale=float(bad_scale))


def test_validate_sparse_pack_rejects_invalid_values_scale_payload(
    saved_sparse_pack: Path,
) -> None:
    np.save(saved_sparse_pack / "values_scale.npy", np.asarray(0.0, dtype=np.float64))

    with pytest.raises(ValueError, match="value_scale"):
        validate_sparse_pack(saved_sparse_pack)


def test_open_sparse_pack_with_custom_files_without_scale_file(
    tmp_path: Path,
    coo_pack_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]],
    dense_matrices: list[np.ndarray],
) -> None:
    indices, values, nnz_ptr, size = coo_pack_arrays
    pack_path = tmp_path / "custom_pack"
    files = PackFiles(indices="i.npy", values="v.npy", nnz_ptr="p.npy", values_scale="s.npy")
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, files=files)
    (pack_path / "s.npy").unlink()

    reader = open_sparse_pack(pack_path, files=files)

    assert torch.allclose(
        reader.build_torch_sparse(2).to_dense(),
        torch.from_numpy(dense_matrices[2]).to(dtype=reader.build_torch_sparse(2).dtype),
    )


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
