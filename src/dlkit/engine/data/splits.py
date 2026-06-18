from typing import cast

from tensordict.base import TensorDictBase
from torch.utils.data import Dataset

from dlkit.infrastructure.types.split import IndexSplit


class SplitDataset[T: Dataset]:
    """A split of a dataset based on indices."""

    raw: T
    _train: T | None
    _validation: T | None
    _test: T | None
    _predict: T | None

    def __init__(self, dataset: T, split: IndexSplit) -> None:
        self.raw = dataset
        self.split = split
        self._train = None
        self._validation = None
        self._test = None
        self._predict = None

    @property
    def train(self) -> T:
        """Get train split - use custom if set, otherwise compute from indices."""
        if self._train is not None:
            return self._train
        return cast(T, _SubsetDataset(self.raw, self._normalize_indices(self.split.train)))

    @train.setter
    def train(self, value: T) -> None:
        """Set custom train dataset."""
        self._train = value

    @property
    def validation(self) -> T:
        """Get validation split - use custom if set, otherwise compute from indices."""
        if self._validation is not None:
            return self._validation
        return cast(T, _SubsetDataset(self.raw, self._normalize_indices(self.split.validation)))

    @validation.setter
    def validation(self, value: T) -> None:
        """Set custom validation dataset."""
        self._validation = value

    @property
    def test(self) -> T:
        """Get test split - use custom if set, otherwise compute from indices."""
        if self._test is not None:
            return self._test
        return cast(T, _SubsetDataset(self.raw, self._normalize_indices(self.split.test)))

    @test.setter
    def test(self, value: T) -> None:
        """Set custom test dataset."""
        self._test = value

    @property
    def predict(self) -> T:
        """Get predict split - use custom if set, otherwise compute from indices or use raw."""
        if self._predict is not None:
            return self._predict
        if self.split.predict is None:
            return self.raw
        return cast(T, _SubsetDataset(self.raw, self._normalize_indices(self.split.predict)))

    @predict.setter
    def predict(self, value: T) -> None:
        """Set custom predict dataset."""
        self._predict = value

    @staticmethod
    def _normalize_indices(indices: tuple[int, ...] | None) -> list[int] | None:
        """Normalize IndexSplit tuples into list[int] for tensor advanced indexing."""
        if indices is None:
            return None
        return list(indices)


class _SubsetDataset:
    """Simple subset view over an indexable dataset.

    Provides PyTorch Dataset-like interface by remapping indices
    to a base dataset using a precomputed index list.
    """

    def __init__(self, base_dataset: Dataset, indices: list[int] | None) -> None:
        self._base = base_dataset
        self._indices: list[int] = list(indices) if indices is not None else []

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> object:
        return self._base[self._indices[i]]

    def __getitems__(self, indices: list[int]) -> TensorDictBase | list[object]:
        """Batch-retrieve items, remapping split indices to base dataset indices.

        Args:
            indices: List of indices into this split view.

        Returns:
            Batch TensorDict from the base dataset if it supports ``__getitems__``,
            otherwise a list of individual items.
        """
        remapped = [self._indices[i] for i in indices]
        getitems = getattr(self._base, "__getitems__", None)
        if getitems is not None:
            return getitems(remapped)
        return [self._base[i] for i in remapped]
