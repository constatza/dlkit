"""Integration tests for FlexibleDataset with NPZ files.

Tests cover:
- Loading features and targets from NPZ files
- Using entry name as array_key
- Mixed NPY and NPZ files in same dataset
- Precision handling with NPZ files
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dlkit.core.datasets.flexible import BatchComplianceError, FlexibleDataset
from dlkit.interfaces.api.domain.precision import precision_override, PrecisionStrategy
from dlkit.tools.config.data_entries import Feature, Target


class TestFlexibleDatasetWithNpz:
    """Test FlexibleDataset loading from NPZ files."""

    def test_load_single_feature_from_single_array_npz(self, npz_single_array: dict) -> None:
        """Test loading a feature from single-array NPZ (auto-detection)."""
        features = [Feature(name="data", path=npz_single_array["path"])]
        dataset = FlexibleDataset(features=features)

        assert len(dataset) == 10
        assert len(dataset._feature_names) == 1
        assert dataset._dataset_td["features", "data"].shape == (10, 5)

    def test_load_multiple_arrays_from_multi_npz(self, npz_multi_array: dict) -> None:
        """Test loading features and targets from same multi-array NPZ."""
        features = [Feature(name="features", path=npz_multi_array["path"])]
        targets = [Target(name="targets", path=npz_multi_array["path"])]

        dataset = FlexibleDataset(features=features, targets=targets)

        assert len(dataset) == 10
        assert len(dataset._feature_names) == 1
        assert len(dataset._target_names) == 1
        assert dataset._dataset_td["features", "features"].shape == (10, 5)
        assert dataset._dataset_td["targets", "targets"].shape == (10, 1)

    def test_load_multiple_features_from_same_npz(self, npz_multi_array: dict) -> None:
        """Test loading multiple features from same NPZ using different names."""
        features = [
            Feature(name="features", path=npz_multi_array["path"]),
            Feature(name="latent", path=npz_multi_array["path"]),
        ]

        dataset = FlexibleDataset(features=features)

        assert len(dataset) == 10
        assert len(dataset._feature_names) == 2
        assert dataset._dataset_td["features", "features"].shape == (10, 5)
        assert dataset._dataset_td["features", "latent"].shape == (10, 3)

    def test_mixed_npy_and_npz_files(self, npz_multi_array: dict, tmp_path) -> None:
        """Test dataset with mix of NPY and NPZ files."""
        # Create a regular NPY file
        npy_data = np.random.randn(10, 4).astype(np.float32)
        npy_path = tmp_path / "extra.npy"
        np.save(npy_path, npy_data)

        features = [
            Feature(name="features", path=npz_multi_array["path"]),  # From NPZ
            Feature(name="extra", path=npy_path),  # From NPY
        ]

        dataset = FlexibleDataset(features=features)

        assert len(dataset) == 10
        assert len(dataset._feature_names) == 2
        assert dataset._dataset_td["features", "features"].shape == (10, 5)
        assert dataset._dataset_td["features", "extra"].shape == (10, 4)

    def test_getitem_returns_correct_data(self, npz_multi_array: dict) -> None:
        """Test that __getitem__ returns correct slices from NPZ data."""
        from tensordict import TensorDict

        features = [Feature(name="features", path=npz_multi_array["path"])]
        targets = [Target(name="targets", path=npz_multi_array["path"])]

        dataset = FlexibleDataset(features=features, targets=targets)

        # Get first sample
        sample = dataset[0]
        assert isinstance(sample, TensorDict)
        assert len(sample["features"].keys()) == 1
        assert len(sample["targets"].keys()) == 1
        assert sample["features", "features"].shape == (5,)
        assert sample["targets", "targets"].shape == (1,)

        # Verify data matches original
        np.testing.assert_allclose(
            sample["features", "features"].numpy(),
            npz_multi_array["features"][0],
            rtol=1e-5,
            atol=1e-7,
        )

    def test_npz_with_precision_context(self, npz_single_array: dict) -> None:
        """Test that NPZ loading respects precision context in dataset."""
        features = [Feature(name="data", path=npz_single_array["path"])]

        # Load with float64 precision
        with precision_override(PrecisionStrategy.FULL_64):
            dataset = FlexibleDataset(features=features)
            assert dataset._dataset_td["features", "data"].dtype == torch.float64

        # Load with float32 precision
        with precision_override(PrecisionStrategy.FULL_32):
            dataset = FlexibleDataset(features=features)
            assert dataset._dataset_td["features", "data"].dtype == torch.float32

    def test_npz_data_integrity(self, npz_multi_array: dict) -> None:
        """Test that data is preserved correctly through dataset loading."""
        features = [Feature(name="features", path=npz_multi_array["path"])]
        dataset = FlexibleDataset(features=features)

        # Compare with original data
        loaded_data = dataset._dataset_td["features", "features"].numpy()
        original_data = npz_multi_array["features"]

        np.testing.assert_allclose(loaded_data, original_data, rtol=1e-5, atol=1e-7)

    def test_entry_name_used_as_array_key(self, npz_multi_array: dict) -> None:
        """Test that entry name is correctly used as array_key for NPZ files."""
        # Use "features" as entry name - should load "features" array from NPZ
        features = [Feature(name="features", path=npz_multi_array["path"])]
        dataset = FlexibleDataset(features=features)

        assert dataset._dataset_td["features", "features"].shape == (10, 5)
        np.testing.assert_allclose(
            dataset._dataset_td["features", "features"].numpy(),
            npz_multi_array["features"],
            rtol=1e-5,
            atol=1e-7,
        )

        # Use "latent" as entry name - should load "latent" array from NPZ
        features2 = [Feature(name="latent", path=npz_multi_array["path"])]
        dataset2 = FlexibleDataset(features=features2)

        assert dataset2._dataset_td["features", "latent"].shape == (10, 3)
        np.testing.assert_allclose(
            dataset2._dataset_td["features", "latent"].numpy(),
            npz_multi_array["latent"],
            rtol=1e-5,
            atol=1e-7,
        )


class TestFlexibleDatasetNpzErrors:
    """Test error handling for NPZ files in FlexibleDataset."""

    def test_wrong_entry_name_raises(self, npz_multi_array: dict) -> None:
        """Test that using wrong entry name raises clear error."""
        # Entry name "missing" doesn't exist in NPZ
        features = [Feature(name="missing", path=npz_multi_array["path"])]

        with pytest.raises(ValueError, match="Array key 'missing' not found"):
            FlexibleDataset(features=features)

    def test_single_array_npz_with_matching_name(self, npz_single_array: dict) -> None:
        """Test that single-array NPZ requires matching entry name."""
        # Entry name must match the array key in NPZ
        features = [Feature(name="data", path=npz_single_array["path"])]
        dataset = FlexibleDataset(features=features)

        # Should work when name matches
        assert len(dataset) == 10
        assert dataset._dataset_td["features", "data"].shape == (10, 5)

    def test_single_array_npz_with_wrong_name_raises(self, npz_single_array: dict) -> None:
        """Test that single-array NPZ with wrong name raises error."""
        # Entry name "wrong_name" doesn't match "data" in NPZ
        features = [Feature(name="wrong_name", path=npz_single_array["path"])]

        with pytest.raises(ValueError, match="Array key 'wrong_name' not found"):
            FlexibleDataset(features=features)

    def test_inconsistent_array_lengths_raises(self, tmp_path) -> None:
        """Test that NPZ arrays with inconsistent lengths raise error."""
        # Create NPZ with arrays of different lengths
        arr1 = np.ones((10, 5), dtype=np.float32)
        arr2 = np.ones((15, 3), dtype=np.float32)  # Different length!

        path = tmp_path / "inconsistent.npz"
        np.savez(path, features=arr1, targets=arr2)

        features = [Feature(name="features", path=path)]
        targets = [Target(name="targets", path=path)]

        with pytest.raises(BatchComplianceError, match="same first dimension N"):
            FlexibleDataset(features=features, targets=targets)


class TestFlexibleDatasetNpzPerformance:
    """Test performance characteristics of NPZ loading."""

    def test_multiple_loads_from_same_npz(self, npz_multi_array: dict) -> None:
        """Test that loading multiple arrays from same NPZ works efficiently."""
        # This tests that we can load from the same NPZ file multiple times
        # Each load should work independently
        features = [
            Feature(name="features", path=npz_multi_array["path"]),
            Feature(name="latent", path=npz_multi_array["path"]),
        ]
        targets = [Target(name="targets", path=npz_multi_array["path"])]

        dataset = FlexibleDataset(features=features, targets=targets)

        assert len(dataset._feature_names) == 2
        assert len(dataset._target_names) == 1
        assert dataset._dataset_td["features", "features"].shape == (10, 5)
        assert dataset._dataset_td["features", "latent"].shape == (10, 3)
        assert dataset._dataset_td["targets", "targets"].shape == (10, 1)

    def test_lazy_loading_behavior(self, npz_multi_array: dict) -> None:
        """Test that only requested arrays are loaded from NPZ."""
        # Create dataset with only features (not targets)
        features = [Feature(name="features", path=npz_multi_array["path"])]
        dataset = FlexibleDataset(features=features)

        # Only features should be loaded, not other arrays in the NPZ
        assert len(dataset._feature_names) == 1
        assert len(dataset._target_names) == 0
