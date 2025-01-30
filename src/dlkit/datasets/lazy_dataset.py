import json
from pydantic import FilePath
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from dlkit.transforms.chaining import TransformationChain
from dlkit.datasets.split import split_indices


# -------------------------------------------------------------------------
# TransformSet - A view of the dataset that applies a transform
# -------------------------------------------------------------------------
class TransformSet(Dataset):
    """
    A subset of a LazyDataset that applies a given transform chain when retrieving items.

    This subset is typically constructed by LazyDataset's .train / .validate / .test properties
    or by external logic that slices indices.
    """

    def __init__(
        self,
        parent_dataset: "LazyDataset",
        subset_indices: List[int],
        transforms: Optional[TransformationChain] = None,
        no_targets: bool = False,
    ) -> None:
        """
        Args:
            parent_dataset (LazyDataset): The original LazyDataset instance.
            subset_indices (List[int]): Indices defining this subset.
            transforms (Optional[TransformationChain]): A fitted transform chain.
            train (bool): Whether this is a training subset (if needed in logic).
            no_targets (bool): If True, dataset has no targets.
        """
        self.parent_dataset = parent_dataset
        self.indices = subset_indices
        self.transforms = transforms
        self.no_targets = no_targets

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_index = self.indices[idx]
        feature, target = self.parent_dataset[real_index]

        # Apply transforms only if we have a transform chain
        if self.transforms is not None:
            with torch.no_grad():
                feature = self.transforms(feature)
        if self.no_targets:
            # if no target file is provided
            return feature, feature.clone()

        return feature, target


# -------------------------------------------------------------------------
# Main LazyDataset
# -------------------------------------------------------------------------
class LazyDataset(Dataset):
    """
    Lazy-loading Dataset for loading data from memory-mapped files.

    This class also handles the train/val/test index splitting and
    can optionally load precomputed indices from a JSON file.

    NOTE: The actual .fit() of the transform chain should happen externally,
    after referencing the train subset. Once fitted, you can assign the same
    transform chain to all subsets to ensure consistent transforms.
    """

    def __init__(
        self,
        features_path: FilePath,
        targets_path: Optional[FilePath] = None,
        indices_path: Optional[FilePath] = None,
        test_size: float = 0.3,
        val_size: float = 0.5,
        transforms: Optional[TransformationChain] = None,
    ) -> None:
        """
        Args:
            features_path (Path): Path to the features file (.npy).
            targets_path (Optional[Path]): Path to the targets file (.npy). Defaults to None.
            indices_path (Optional[Path]): Path to a JSON containing pre-split indices. Defaults to None.
            test_size (float): Fraction of entire dataset to use for testing. Defaults to 0.3.
            val_size (float): Fraction of the test portion to use as validation. Defaults to 0.5.
            transforms (Optional[TransformationChain]): An already-fitted transform pipeline.
        """
        super().__init__()
        self.no_targets = targets_path is None
        self.features_path = features_path
        self.targets_path = targets_path or features_path
        self.indices_path = indices_path
        self.features = np.load(features_path, mmap_mode="r")
        self.targets = (
            np.load(targets_path, mmap_mode="r") if targets_path else self.features
        )

        # If an indices file is provided, load it. Otherwise, split automatically.
        if indices_path:
            with open(indices_path, "r", encoding="utf-8") as f:
                saved_indices: Dict[str, List[int]] = json.load(f)
            self._train_idx = saved_indices["train"]
            self._val_idx = saved_indices["val"]
            self._test_idx = saved_indices["test"]
            self.indices = saved_indices
        else:
            all_ids = list(range(len(self.features)))
            self._train_idx, self._val_idx, self._test_idx = split_indices(
                all_ids, test_size=test_size, val_size=val_size
            )
            self.indices = {
                "train": self._train_idx,
                "val": self._val_idx,
                "test": self._test_idx,
            }

        self.test_size = test_size
        self.val_size = val_size
        self.transforms = transforms

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Returns the shapes of features and targets arrays."""
        return self.features.shape, self.targets.shape

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_np = np.copy(self.features[idx]).astype(np.float32)
        feature_torch = torch.from_numpy(feature_np)
        if self.no_targets:
            return feature_torch, feature_torch

        target_np = np.copy(self.targets[idx]).astype(np.float32)
        target_torch = torch.from_numpy(target_np)
        return feature_torch, target_torch

    @property
    def train(self) -> TransformSet:
        """
        Returns the training subset of this dataset.
        """
        return TransformSet(
            parent_dataset=self,
            subset_indices=self._train_idx,
            transforms=self.transforms,
            no_targets=self.no_targets,
        )

    @property
    def validate(self) -> TransformSet:
        """
        Returns the validation subset of this dataset.
        """
        return TransformSet(
            parent_dataset=self,
            subset_indices=self._val_idx,
            transforms=self.transforms,
            no_targets=self.no_targets,
        )

    @property
    def test(self) -> TransformSet:
        """
        Returns the test subset of this dataset.
        """
        return TransformSet(
            parent_dataset=self,
            subset_indices=self._test_idx,
            transforms=self.transforms,
            no_targets=self.no_targets,
        )

    @property
    def predict(self) -> TransformSet:
        """
        Returns the entire dataset for prediction.
        """
        return TransformSet(
            parent_dataset=self,
            subset_indices=list(range(len(self))),
            transforms=self.transforms,
            no_targets=self.no_targets,
        )


# -------------------------------------------------------------------------
# A helper function to produce (train, val, test) after fitting transforms
# -------------------------------------------------------------------------
def prepare_datasets(
    features_path: FilePath,
    targets_path: Optional[FilePath],
    transform_chain: Optional[TransformationChain],
    indices_path: Optional[FilePath] = None,
    test_size: float = 0.3,
    val_size: float = 0.5,
) -> Tuple[TransformSet, TransformSet, TransformSet, TransformSet]:
    """
    Create a LazyDataset, split into (train, val, test), then fit the transform_chain on train
    and set it back to each subset for consistent transforms.

    Args:
        features_path (FilePath): Path to the features .npy.
        targets_path (Optional[FilePath]): Path to the targets .npy.
        transform_chain (Optional[TransformationChain]): Transformation pipeline to be fitted on train.
        indices_path (Optional[FilePath]): Path to saved indices file, if any.
        test_size (float): Fraction for test set.
        val_size (float): Fraction of test portion for val set.

    Returns:
        Tuple[TransformSet, TransformSet, TransformSet, TransformSet]: train_set, val_set, test_set, pred_set
    """
    # Step 1: Create one dataset that knows about the splits
    dataset = LazyDataset(
        features_path=features_path,
        targets_path=targets_path,
        indices_path=indices_path,
        test_size=test_size,
        val_size=val_size,
        transforms=None,  # We'll set this after we fit
    )

    # Step 2: Get subsets
    train_dataloader = DataLoader(
        dataset.train,
        batch_size=len(dataset.train),
        shuffle=False,
    )
    train_subset = dataset.train
    val_subset = dataset.validate
    test_subset = dataset.test
    pred_subset = dataset.predict

    # Step 3: Fit the transform_chain on the training subset if provided
    if transform_chain is not None:
        # Collect all features from train subset
        # (In large-scale scenarios, you'd iterate in mini-batches or partial_fit.)
        all_train_features = []
        for x, y in train_dataloader:
            all_train_features.append(x)

        transform_chain.fit(torch.cat(all_train_features, dim=0))

        # Step 4: Assign the fitted transform chain to all subsets
        train_subset.transforms = transform_chain
        val_subset.transforms = transform_chain
        test_subset.transforms = transform_chain
        pred_subset.transforms = transform_chain

    return train_subset, val_subset, test_subset, pred_subset
