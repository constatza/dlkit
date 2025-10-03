import torch
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
        return _SubsetDataset(self.raw, self._normalize_indices(self.split.train))  # type: ignore[index,return-value]

    @train.setter
    def train(self, value: T) -> None:
        """Set custom train dataset."""
        self._train = value

    @property
    def validation(self) -> T:
        """Get validation split - use custom if set, otherwise compute from indices."""
        if self._validation is not None:
            return self._validation
        return _SubsetDataset(self.raw, self._normalize_indices(self.split.validation))  # type: ignore[index,return-value]

    @validation.setter
    def validation(self, value: T) -> None:
        """Set custom validation dataset."""
        self._validation = value

    @property
    def test(self) -> T:
        """Get test split - use custom if set, otherwise compute from indices."""
        if self._test is not None:
            return self._test
        return _SubsetDataset(self.raw, self._normalize_indices(self.split.test))  # type: ignore[index,return-value]

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
        return _SubsetDataset(self.raw, self._normalize_indices(self.split.predict))  # type: ignore[index,return-value]

    @predict.setter
    def predict(self, value: T) -> None:
        """Set custom predict dataset."""
        self._predict = value

    @staticmethod
    def _normalize_indices(indices):
        """Normalize IndexSplit tuples into list[int] for tensor advanced indexing.

        Torch randperm-based tuples may contain scalar tensors; convert them to
        plain Python ints to avoid multi-dimensional indexing semantics.
        """
        try:
            import torch  # type: ignore
        except Exception:
            torch = None  # type: ignore

        if indices is None:
            return indices
        # Convert tuple to list to avoid multi-d indexing semantics
        if isinstance(indices, tuple):
            if torch is not None and all(hasattr(i, "item") for i in indices):
                return [int(i.item()) for i in indices]
            return list(indices)
        return indices


class _SubsetDataset:
    """Simple subset view over an indexable dataset.

    Provides PyTorch Dataset-like interface by remapping indices
    to a base dataset using a precomputed index list.
    """

    def __init__(self, base_dataset, indices):
        self._base = base_dataset
        # Convert to a plain list of ints for robustness
        self._indices = list(indices) if not isinstance(indices, list) else indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int):
        base_idx = self._indices[i]
        return self._base[base_idx]
