from typing import Literal

import torch

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset, ConcatDataset

from dlkit.settings.datamodule_settings import DatamoduleSettings
from dlkit.settings.paths_settings import PathSettings
from dlkit.transforms.chaining import Pipeline
from dlkit.datatypes.dataset import SplitDataset
from dlkit.settings.datamodule_settings import SplitIndices
from dlkit.datatypes import Shape
from .utils import index_split


# --- Pure function for generating splits ---


class InMemoryModule(LightningDataModule):
    """
    DataModule using NumpyDataset to load raw data,
    create splits in 'fit', apply transforms via helper methods,
    and provide DataLoaders.
    """

    idx_split: SplitIndices | None
    paths: PathSettings
    device: torch.device
    features_pipeline: Pipeline
    targets_pipeline: Pipeline
    dataset: SplitDataset
    shape: Shape

    def __init__(
        self,
        dataset: Dataset,
        settings: DatamoduleSettings,
        paths: PathSettings,
        features_pipeline: Pipeline = Pipeline(()),
        targets_pipeline: Pipeline = Pipeline(()),
        device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        # Raw dataset bridge
        self.settings = settings
        self.paths = paths
        self.dataset = SplitDataset(raw=dataset)
        self.device = torch.device(device)
        self.features_pipeline = features_pipeline.to(self.device)
        self.targets_pipeline = targets_pipeline.to(self.device)
        self.fitted = False
        self.idx_split = None

    def setup(self, stage: str | None = None) -> None:
        # FIT: generate splits, build processed_dataset
        if stage in ("fit", None) and not self.fitted:
            self.idx_split, split_path = index_split(
                idx_split_path=self.paths.idx_split,
                save_dir=self.paths.input_dir,
                n=len(self.dataset.raw),
                test_size=self.settings.test_size,
                val_size=self.settings.val_size,
            )
            self.paths = self.paths.model_copy(update={"idx_split": split_path})
            raw_train = Subset(self.dataset.raw, self.idx_split.train)
            raw_validation = Subset(self.dataset.raw, self.idx_split.validation)
            self.dataset.train = self.fit_transform_dataset(raw_train)
            self.dataset.validation = self.transform_dataset(raw_validation)
            self.fitted = True
            self.shape = Shape(
                features=self.features_pipeline.output_shape,
                targets=self.targets_pipeline.output_shape,
            )

        # TEST: build test split
        if stage in ("test", None):
            if self.dataset is None:
                raise RuntimeError("Call setup('fit') before setup('test')")
            raw_test = Subset(self.dataset.raw, self.idx_split.test)
            self.dataset.test = self.transform_dataset(raw_test)

        # PREDICT: build predict split
        if stage in ("predict", None):
            if self.dataset is None:
                raise RuntimeError("Call setup('fit') before setup('predict')")
            self.dataset.predict = ConcatDataset(
                [self.dataset.train, self.dataset.validation, self.dataset.test]
            )

    def train_dataloader(self):
        return DataLoader(self.dataset.train, **self.settings.dataloader.model_dump())

    def val_dataloader(self):
        return DataLoader(
            self.dataset.validation, **self.settings.dataloader.model_dump()
        )

    def test_dataloader(self):
        return DataLoader(self.dataset.test, **self.settings.dataloader.model_dump())

    def predict_dataloader(self):
        return DataLoader(
            self.dataset.predict,
            shuffle=False,
            **self.settings.dataloader.to_dict_compatible_with(
                DataLoader, exclude=("shuffle",)
            ),
        )

    def fit_dataset(self, dataset: Dataset):
        """
        Apply fit & transformation to raw features and targets and return a TensorDataset.
        Parameters:
            dataset (Dataset): The dataset to transform.
        """
        raw_loader = DataLoader(dataset, batch_size=len(dataset))
        features, targets = next(iter(raw_loader))
        # Apply input transform
        self.features_pipeline.fit(features.to(self.device))
        if targets is not None:
            self.targets_pipeline.fit(targets.to(self.device))

    def transform_dataset(self, dataset: Dataset) -> TensorDataset:
        """
        Apply transformation to raw features and targets and return a TensorDataset.
        Parameters:
            dataset (Dataset): The dataset to transform.
        """
        raw_loader = DataLoader(dataset, batch_size=len(dataset))
        features, targets = next(iter(raw_loader))
        # Apply input transform
        features = self.features_pipeline.transform(features.to(self.device))
        if self.settings.is_autoencoder or targets is None:
            return TensorDataset(features, features)

        targets = self.targets_pipeline.transform(features.to(self.device))
        return TensorDataset(features, targets)

    def fit_transform_dataset(self, dataset: Dataset):
        """
        Apply fit & transformation to raw features and targets and return a TensorDataset.
        Parameters:
            dataset (Dataset): The dataset to transform.
        """
        self.fit_dataset(dataset)
        return self.transform_dataset(dataset)
