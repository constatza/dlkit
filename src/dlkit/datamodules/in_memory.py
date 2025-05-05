from typing import Literal

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset, ConcatDataset

from dlkit.settings.data_settings import DataSettings
from dlkit.settings.paths_settings import PathSettings
from dlkit.transforms.chaining import Pipeline
from dlkit.datatypes.dataset import SplitDataset, SplitIndices
from dlkit.datatypes.dataset import Shape
from .base import BaseDataModule


class InMemoryModule(BaseDataModule):
    """
    This datamodule is designed to handle in-memory datasets efficiently.
    """

    paths: PathSettings
    device: torch.device
    features_pipeline: Pipeline
    targets_pipeline: Pipeline
    dataset: SplitDataset
    shape: Shape
    idx_split: SplitIndices

    def __init__(
        self,
        dataset: Dataset,
        settings: DataSettings,
        idx_split: SplitIndices,
        features_pipeline: Pipeline = Pipeline(()),
        targets_pipeline: Pipeline = Pipeline(()),
        device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ) -> None:
        super().__init__(
            dataset,
            settings=settings,
            idx_split=idx_split,
            device=device,
        )
        self.fitted = False
        self.features_pipeline = features_pipeline.to(device)
        self.targets_pipeline = targets_pipeline.to(device)

    def setup(self, stage: str | None = None) -> None:
        # FIT: generate splits, build processed_dataset
        if stage in ("fit", None) and not self.fitted:
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
        if self.settings.targets_exist or targets is None:
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
