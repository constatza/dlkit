import json
import torch

from pathlib import Path
from lightning import LightningDataModule
from loguru import logger
from pydantic import FilePath, validate_call, DirectoryPath
from torch.utils.data import TensorDataset, DataLoader, Subset

from dlkit.datasets.numpy_dataset import load_dataset, split_or_load_indices
from dlkit.transforms.chaining import TransformationChain


class NumpyModule(LightningDataModule):
    """
    LightningDataModule for handling datasets with train/val/test splits and lazy loading.

    Args:
        features_path (FilePath): Path to features file (e.g. .npy).
        targets_path (FilePath | None): Path to targets file (e.g. .npy). Defaults to None.
        idx_split_path (FilePath | None): Path to saved index splits for train/val/test. Defaults to None.
        dataloader_config (dict, optional): Configuration for DataLoader. Defaults to None.
        transform_chain (TransformationChain | None): Transformation chain for dataset. Defaults to None.
        test_size (float, optional): Fraction for test split (0 to 1). Defaults to 0.3.
        val_size (float, optional): Fraction for val split (0 to 1). Defaults to 0.5.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        features_path: FilePath,
        targets_path: FilePath = None,
        idx_split_path: FilePath | None = None,
        dataloader_config: dict | None = None,
        transform_chain: TransformationChain | None = None,
        test_size: float = 0.3,
        val_size: float = 0.5,
    ):
        super().__init__()
        self.features_path = features_path
        self.targets_path = targets_path
        self.idx_split_path: Path = idx_split_path
        self.dataloader_config = dataloader_config or {}
        self.transform_chain = transform_chain

        if self.idx_split_path is None:
            logger.warning("No index path provided, saving to current directory.")

        self.dataset: TensorDataset | None = None
        self.idx_split = {}
        self.test_size = test_size
        self.val_size = val_size
        self.train_set: Subset | None = None
        self.val_set: Subset | None = None
        self.test_set: Subset | None = None
        self.predict_set: Subset | None = None

        # Tensors
        self.features: torch.Tensor | None = None
        self.transformed_features: torch.Tensor | None = None
        self.targets: torch.Tensor | None = None
        self.transformed_targets: torch.Tensor | None = None
        self.shapes: tuple | None = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self, stage: str | None = None) -> None:
        """
        Set up datasets for different stages.

        Args:
            stage (str | None, optional): Stage ('fit', 'test', or 'predict'). Defaults to None.
        """
        # -------------------------
        # FIT STAGE
        # -------------------------
        if stage in ["fit", None]:
            # Load features/targets into memory (CPU by default)
            self.features, self.targets = load_dataset(
                self.features_path, self.targets_path
            )
            self.shapes = (self.features.shape, self.targets.shape)

            # Generate or load train/val/test indices
            self.idx_split = split_or_load_indices(
                self.idx_split_path,
                len(self.features),
                test_size=self.test_size,
                val_size=self.val_size,
            )
            if self.idx_split_path is None:
                self.idx_split_path = self.features_path.parent / "idx_split.json"
                logger.warning(
                    f"No index split path provided, generating and saving to {self.idx_split_path}"
                )
                with open(self.idx_split_path, "w") as f:
                    self.idx_split["idx_path"] = str(self.idx_split_path)
                    json.dump(self.idx_split, f)
            else:
                logger.info(f"Loaded indices from {self.idx_split_path}")

            # If transform_chain is provided, move data to GPU, apply transforms
            # Then delete the old (untransformed) copy
            if self.transform_chain is not None:
                with torch.no_grad():
                    self.features = self.features.to(self.device)
                    self.transform_chain.to(self.device)
                    self.transform_chain.fit(self.features[self.idx_split["train"]])
                    self.transformed_features = self.transform_chain(self.features)

                # Free GPU memory from the original features, since they're no longer needed
                # If no target file was provided, the transformed features act as targets
                if not self.targets_path:
                    self.transformed_targets = self.transformed_features
                else:
                    # Future targets transformation
                    pass

            else:
                # If no transform, just store them as-is (may still want to put them on GPU depending on your pipeline)
                self.transformed_features = self.features

            self.features = self.features.cpu().numpy()
            self.targets = self.targets.cpu().numpy()
            self.transform_chain = self.transform_chain.cpu()
            self.transformed_targets = self.transformed_targets.cpu()
            self.transformed_features = self.transformed_features.cpu()
            # Build the main dataset
            self.dataset = TensorDataset(
                self.transformed_features, self.transformed_targets
            )

            # Create subsets for train/val
            self.train_set = Subset(self.dataset, self.idx_split["train"])
            if self.val_size > 0:
                self.val_set = Subset(self.dataset, self.idx_split["val"])

        # -------------------------
        # TEST STAGE
        # -------------------------
        if stage in ["test", None]:
            if self.dataset is None:
                # Edge case: if someone calls test without 'fit' first
                # handle logic or raise an error
                # but typically, you'll want to call setup("fit") first
                raise RuntimeError(
                    "Dataset is not initialized. Call setup('fit') first."
                )
            self.test_set = Subset(self.dataset, self.idx_split["test"])

        # -------------------------
        # PREDICT STAGE
        # -------------------------
        if stage in ["predict", None]:
            if self.dataset is None:
                raise RuntimeError(
                    "Dataset is not initialized. Call setup('fit') first."
                )
            self.predict_set = self.dataset

    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training set."""
        return DataLoader(self.train_set, shuffle=False, **self.dataloader_config)

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation set."""
        return DataLoader(self.val_set, shuffle=False, **self.dataloader_config)

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for test set."""
        return DataLoader(self.test_set, shuffle=False, **self.dataloader_config)

    def predict_dataloader(self) -> DataLoader:
        """Create DataLoader for predict set."""
        return DataLoader(self.predict_set, shuffle=False, **self.dataloader_config)
