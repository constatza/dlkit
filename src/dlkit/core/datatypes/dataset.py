from typing import cast

from dlkit.core.datatypes.split import IndexSplit


class SplitDataset[T]:
    """A split of a dataset based on indices."""

    raw: T
    _train: T | None
    _validation: T | None
    _test: T | None
    _predict: T | None

    def __init__(self, dataset: T, split: IndexSplit) -> None:
        self.raw = dataset
        self.split = split
        # Initialize private attributes
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

    def __init__(self, base_dataset: object, indices: list[int] | None) -> None:
        self._base = base_dataset
        # Convert to a plain list of ints for robustness
        self._indices: list[int] = list(indices) if indices is not None else []

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> object:
        base_idx = self._indices[i]
        return self._base[base_idx]  # type: ignore[index]  # base is typed as object
