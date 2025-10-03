from torch import randperm
from pydantic import BaseModel


class IndexSplit(BaseModel):
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    predict: tuple[int, ...] | None


class Splitter:
    """Create immutable train/val/test index splits for a dataset of size num_samples."""

    def __init__(self, *, num_samples: int, test_ratio: float, val_ratio: float) -> None:
        """Create immutable train/val/test index splits for a dataset of size num_samples.

        Args:
            num_samples (int): Number of samples in the dataset.
            test_ratio (float): Fraction of dataflow used for testing.
            val_ratio (float): Fraction of dataflow used for validation.
        """
        if num_samples < 0:
            msg = "num_samples must be a non-negative integer"
            raise ValueError(msg)
        self.num_samples = num_samples
        self.test_count = int(num_samples * test_ratio)
        self.val_count = int(num_samples * val_ratio)
        self.train_count = num_samples - self.test_count - self.val_count

    def split(self) -> IndexSplit:
        perm = randperm(self.num_samples)
        train_count = self.train_count
        val_count = self.val_count
        test_count = self.test_count
        return IndexSplit(
            train=tuple(perm[:train_count].tolist()),
            validation=tuple(perm[train_count : train_count + val_count].tolist()),
            test=tuple(perm[train_count + val_count : train_count + val_count + test_count].tolist()),
            predict=tuple(perm[train_count + val_count + test_count :].tolist()),
        )

    def __len__(self) -> int:
        return self.num_samples
