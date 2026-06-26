"""Integration tests for FlexibleDataset with HDF5 files.

Tests cover:
- Lazy loading (Hdf5LazyReader) — default
- Eager loading (EagerFileSource) — lazy=False
- Group/key navigation
- __getitem__ and __getitems__ correctness
- Mixed HDF5 lazy + other source types in one dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from dlkit.engine.data.datasets.flexible import FlexibleDataset
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import Hdf5Entry, NpyEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hdf5_file(tmp_path: Path) -> dict[str, Any]:
    """HDF5 file with features (10, 4) and targets (10, 2) under 'arrays' group.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Dict with ``path``, ``x`` (feature array), ``y`` (target array).
    """
    rng = np.random.default_rng(42)
    x = rng.random((10, 4)).astype(np.float32)
    y = rng.random((10, 2)).astype(np.float32)
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("arrays")
        grp.create_dataset("x", data=x)
        grp.create_dataset("y", data=y)
    return {"path": path, "x": x, "y": y}


# ---------------------------------------------------------------------------
# Lazy loading (default)
# ---------------------------------------------------------------------------


class TestFlexibleDatasetHdf5Lazy:
    """Tests for FlexibleDataset with Hdf5Entry in lazy mode (default)."""

    def test_len(self, hdf5_file: dict) -> None:
        """Dataset length matches the HDF5 dataset axis-0 size."""
        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    data_role=DataRole.FEATURE,
                ),
            ]
        )
        assert len(dataset) == 10

    def test_getitem_returns_correct_tensordict(self, hdf5_file: dict) -> None:
        """__getitem__ returns a TensorDict with the correct sample values."""
        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    data_role=DataRole.FEATURE,
                ),
                Hdf5Entry(
                    name="y",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="y",
                    data_role=DataRole.TARGET,
                ),
            ]
        )

        sample = dataset[3]
        assert isinstance(sample, TensorDict)
        assert sample["features", "x"].shape == torch.Size([4])
        assert sample["targets", "y"].shape == torch.Size([2])
        np.testing.assert_allclose(
            cast(torch.Tensor, sample["features", "x"]).numpy(), hdf5_file["x"][3], rtol=1e-5
        )
        np.testing.assert_allclose(
            cast(torch.Tensor, sample["targets", "y"]).numpy(), hdf5_file["y"][3], rtol=1e-5
        )

    def test_getitems_returns_batched_tensordict(self, hdf5_file: dict) -> None:
        """__getitems__ returns pre-batched TensorDict from a single HDF5 read per source."""
        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    data_role=DataRole.FEATURE,
                ),
                Hdf5Entry(
                    name="y",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="y",
                    data_role=DataRole.TARGET,
                ),
            ]
        )

        indices = [0, 5, 3]
        batch = dataset.__getitems__(indices)
        assert isinstance(batch, TensorDict)
        assert batch["features", "x"].shape == torch.Size([3, 4])
        assert batch["targets", "y"].shape == torch.Size([3, 2])
        np.testing.assert_allclose(
            cast(torch.Tensor, batch["features", "x"]).numpy(), hdf5_file["x"][indices], rtol=1e-5
        )

    def test_getitems_unsorted_indices_correct_values(self, hdf5_file: dict) -> None:
        """__getitems__ with unsorted indices returns data in the requested order."""
        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    data_role=DataRole.FEATURE,
                ),
            ]
        )

        indices = [9, 1, 5]
        batch = dataset.__getitems__(indices)
        expected = torch.from_numpy(hdf5_file["x"][indices])
        assert torch.allclose(batch["features", "x"].float(), expected.float())


# ---------------------------------------------------------------------------
# Eager loading
# ---------------------------------------------------------------------------


class TestFlexibleDatasetHdf5Eager:
    """Tests for FlexibleDataset with Hdf5Entry in eager mode (lazy=False)."""

    def test_getitem_eager_returns_correct_values(self, hdf5_file: dict) -> None:
        """Eager path loads the full array at construction; __getitem__ slices it."""
        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    lazy=False,
                    data_role=DataRole.FEATURE,
                ),
            ]
        )

        sample = dataset[7]
        np.testing.assert_allclose(
            cast(torch.Tensor, sample["features", "x"]).numpy(), hdf5_file["x"][7], rtol=1e-5
        )

    def test_getitems_eager_returns_batched_values(self, hdf5_file: dict) -> None:
        """Eager __getitems__ returns the same values as lazy mode."""
        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    lazy=False,
                    data_role=DataRole.FEATURE,
                ),
            ]
        )

        indices = [2, 4, 6]
        batch = dataset.__getitems__(indices)
        np.testing.assert_allclose(
            cast(torch.Tensor, batch["features", "x"]).numpy(), hdf5_file["x"][indices], rtol=1e-5
        )


# ---------------------------------------------------------------------------
# Mixed sources
# ---------------------------------------------------------------------------


class TestFlexibleDatasetHdf5Mixed:
    """Tests combining HDF5 lazy entries with other source types."""

    def test_hdf5_feature_npy_target(self, hdf5_file: dict, tmp_path: Path) -> None:
        """HDF5 feature and NPY target can coexist in one FlexibleDataset."""
        y_data = np.random.default_rng(7).random((10, 2)).astype(np.float32)
        npy_path = tmp_path / "y.npy"
        np.save(npy_path, y_data)

        dataset = FlexibleDataset(
            entries=[
                Hdf5Entry(
                    name="x",
                    path=hdf5_file["path"],
                    group="arrays",
                    key="x",
                    data_role=DataRole.FEATURE,
                ),
                NpyEntry(name="y", path=npy_path, data_role=DataRole.TARGET),
            ]
        )

        assert len(dataset) == 10
        sample = dataset[0]
        assert sample["features", "x"].shape == torch.Size([4])
        assert sample["targets", "y"].shape == torch.Size([2])
