from pathlib import Path
from typing import Optional, Callable, Iterable

import numpy as np
from lightning import LightningDataModule
from pydantic import FilePath, validate_call
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from dlkit.io.logging import get_logger
from dlkit.io.writers import savez_asarray
from dlkit.mytypes import TupleLike, CreateIfNotExistsDir
from dlkit.transforms import NumpyToTensor
from dlkit.pipeline import Pipeline

logger = get_logger(__name__)


class FileDataModule(LightningDataModule):
    """Class to load data from files and prepare it for training, validation, and testing.

    Args:
        features_path (FilePath): Path to the features file.
        targets_path (Optional[FilePath], optional): Path to the targets file. Defaults to None.
        save_dir (Optional[CreateIfNotExistsDir], optional): Directory to save the indices. Defaults to ".".
        features_pipeline (Pipeline, optional): Pipeline to transform the features. Defaults to Pipeline(NumpyToTensor()).
        targets_pipeline (Pipeline, optional): Pipeline to transform the targets. Defaults to Pipeline(NumpyToTensor()).
        feature_preprocessors (TupleLike[Callable], optional): Preprocessors for the features. Defaults to (lambda x: x,).
        target_preprocessors (TupleLike[Callable], optional): Preprocessors for the targets. Defaults to (lambda x: x,).
        test_size (float, optional): Percentage of the whole dataset to use for testing. Defaults to 0.2.
        val_size (float, optional): Percentage of the test dataset to use for validation. Defaults to 0.5.
        batch_size (int, optional): Batch size for the dataloaders. Defaults to 64.
        indices_path (Optional[FilePath], optional): Path to the indices file. Defaults to None.
        dataloader_config (Optional[dict], optional): Configuration for the dataloaders. Defaults to None.
    """

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        features_path: FilePath,
        targets_path: Optional[FilePath] = None,
        save_dir: Optional[CreateIfNotExistsDir] = ".",
        features_pipeline: Pipeline = Pipeline(NumpyToTensor()),
        targets_pipeline: Pipeline = Pipeline(NumpyToTensor()),
        feature_preprocessors: TupleLike[Callable, ...] = (lambda x: x,),
        target_preprocessors: TupleLike[Callable, ...] = (lambda x: x,),
        test_size: float = 0.2,  # percentage of the whole dataset
        val_size: float = 0.5,  # percentage of the test dataset
        batch_size: int = 64,
        indices_path: Optional[FilePath] = None,
        dataloader_config: Optional[dict] = None,
    ):

        super().__init__()
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if not targets_path:
            logger.warn(
                "No targets path provided. Assuming targets are the same as features."
            )
            self.targets_exist = False
            targets_path = features_path

        self.features_path = features_path
        self.targets_path = targets_path
        self._features = None
        self._targets = None

        self.indices_path = indices_path
        self._indices = None
        self._shapes = None
        self._num_samples = None

        self.features_shape = None
        self.targets_shape = None

        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        self.feature_transforms = features_pipeline
        self.target_transforms = targets_pipeline
        self.feature_preprocessors = feature_preprocessors
        self.target_preprocessors = target_preprocessors

        self.is_prepared = False
        self.dataloader_config = dataloader_config

    def local_path_to(self, filename: str, suffix: str = ".npy") -> Path:
        return (self.save_dir / filename).with_suffix(suffix)

    def prepare_data(self):
        if self.is_prepared:
            return
        else:
            self.is_prepared = True

        if self.indices_path:
            # if external path given
            logger.info(
                f"Loading train/validation/test indices from {self.indices_path}."
            )

        elif not self.local_path_to("indices").exists():
            logger.info("Generating new train/validation/test indices.")
            self.save_indices(self.num_samples)
            self.indices_path = self.local_path_to("indices", suffix=".npz")

    def get_train_val_test_indices(self, num_samples: int = None):
        # Save indices for train, val, test
        train_idx, val_idx, test_idx = train_val_test_split(
            indices=np.arange(num_samples),
            test_size=self.test_size,
            val_size=self.val_size,
        )
        return {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "num_samples": num_samples,
        }

    def save_indices(self, num_samples: int):
        """Save metadata to a file.
        Train/Val/Test indices are generated if not provided in indices_path.
        """
        indices = self.get_train_val_test_indices(num_samples)
        savez_asarray(self.local_path_to("indices"), **indices)

    @property
    def num_samples(self):
        if not self._num_samples:
            self._num_samples = self.features.shape[0]
        return self._num_samples

    @property
    def indices(self):
        if self._indices is None and self.indices_path is not None:
            self._indices = np.load(self.indices_path)
        else:
            self._indices = self.get_train_val_test_indices(self.num_samples)
        return self._indices

    @property
    def shapes(self):
        if self._shapes is None:
            self._shapes = {
                "features": self.features.shape,
                "targets": self.targets.shape,
            }
        return self._shapes

    @property
    def features(self):
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r")
        return self._features

    @property
    def targets(self):
        if self._targets is None and self.targets_exist:
            self._targets = np.load(self.targets_path, mmap_mode="r")
        else:
            self._targets = self._features
        return self._targets

    def setup(self, stage: str | None = None):
        # To be implemented by child classes

        if stage == "fit":
            train_idx = self.indices["train_idx"]
            val_idx = self.indices["val_idx"]

            self.x_train = self.feature_transforms.fit_transform(
                self.features[train_idx]
            )
            self.x_val = self.feature_transforms.transform(self.features[val_idx])

            if self.targets_exist:
                self.y_train = self.target_transforms.fit_transform(
                    self.targets[train_idx]
                )
                self.y_val = self.target_transforms.transform(self.targets[val_idx])
            else:
                self.y_train = self.x_train
                self.y_val = self.x_val

            self.train_dataset = TensorDataset(self.x_train, self.y_train)
            self.val_dataset = TensorDataset(self.x_val, self.y_val)

        if stage == "test":
            test_idx = self.indices["test_idx"]
            x_test = self.features[test_idx]
            self.x_test = self.feature_transforms.transform(x_test)
            if self.targets_exist:
                y_test = self.targets[test_idx]
                self.y_test = self.target_transforms.transform(y_test)
            else:
                self.y_test = self.x_test

            self.test_dataset = TensorDataset(self.x_test, self.y_test)

        if stage == "predict":
            features = self.feature_transforms.transform(self.features)
            if self.targets_exist:
                targets = self.target_transforms.transform(self.targets)
            else:
                targets = self.features
            self.predict_dataset = TensorDataset(features, targets)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.dataloader_config,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.dataloader_config,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **self.dataloader_config,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=len(self.predict_dataset),
            shuffle=False,
            **self.dataloader_config,
        )


def train_val_test_split(indices=None, test_size=0.2, val_size=0.5):
    train, test_plus_val = train_test_split(
        indices,
        test_size=test_size,
        shuffle=True,
    )

    val, test = train_test_split(
        test_plus_val,
        test_size=val_size,
        shuffle=True,
    )

    return train, val, test
