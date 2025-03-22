import json
import torch

from lightning import LightningDataModule
from loguru import logger
from torch.utils.data import TensorDataset, DataLoader, Subset

from dlkit.datasets.numpy_dataset import load_dataset, split_or_load_indices
from dlkit.settings import DatamoduleSettings, Paths
from dlkit.settings.types import Shape
from dlkit.transforms.chaining import TransformationChain


class NumpyModule(LightningDataModule):
    """
    LightningDataModule for handling datasets with train/val/test splits and lazy loading.

    Args:

    """

    settings: DatamoduleSettings
    paths: Paths
    transform_chain: TransformationChain

    dataset: TensorDataset | None
    train_set: Subset | None
    val_set: Subset | None
    test_set: Subset | None
    predict_set: Subset | None

    idx_split: dict[str, tuple[int]]

    features: torch.Tensor | None
    targets: torch.Tensor | None
    transformed_features: torch.Tensor | None
    transformed_targets: torch.Tensor | None

    shapes: Shape

    def __init__(
        self,
        settings: DatamoduleSettings,
        paths: Paths,
        transform_chain: TransformationChain | None = None,
    ):
        super().__init__()
        self.settings = settings
        self.paths = paths
        self.transform_chain = transform_chain

        if self.paths.idx_split is None:
            logger.warning("No index path provided, saving to current directory.")

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
        if stage in ["fit"]:
            # Load features/targets into memory (CPU by default)
            self.features, self.targets = load_dataset(
                self.paths.features, self.paths.targets
            )
            self.shapes = Shape(
                features=self.features.shape[1:], targets=self.targets.shape[1:]
            )

            # Generate or load train/val/test indices
            self.idx_split = split_or_load_indices(
                self.paths.idx_split,
                len(self.features),
                test_size=self.settings.test_size,
                val_size=self.settings.val_size,
            )
            if self.paths.idx_split is None:
                self.paths = self.paths.model_copy(
                    update={"idx_split": self.paths.input / "idx_split.json"}
                )
                with open(self.paths.idx_split, "w") as f:
                    self.idx_split["idx_path"] = str(self.paths.idx_split)
                    json.dump(self.idx_split, f)
                logger.warning(
                    f"No index split path provided, generating and saving to {self.paths.idx_split}"
                )
            else:
                logger.info(f"Loaded indices from {self.paths.idx_split}")

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
                if self.settings.autoencoder_dataset:
                    self.transformed_targets = self.transformed_features
                else:
                    # TODO: Future targets transformation
                    logger.warning("No transformation applied to targets. ")

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
            if self.settings.val_size > 0:
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
        return DataLoader(
            self.train_set,
            **self.settings.dataloader.model_dump(),
        )

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation set."""
        return DataLoader(self.val_set, **self.settings.dataloader.model_dump())

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for test set."""
        return DataLoader(self.test_set, **self.settings.dataloader.model_dump())

    def predict_dataloader(self) -> DataLoader:
        """Create DataLoader for predict set."""
        # don't shuffle predict set
        return DataLoader(
            self.predict_set,
            shuffle=False,
            **self.settings.dataloader.to_dict_compatible_with(
                DataLoader, exclude=("shuffle",)
            ),
        )
