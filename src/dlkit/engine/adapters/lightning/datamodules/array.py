from torch.utils.data import DataLoader

from .base import BaseDataModule


class ArrayDataModule[Dataset_T](BaseDataModule):
    """LightningDataModule for in-memory array/TensorDict datasets.

    Delegates collation to ``dataset.collate_fn``; if the dataset returns
    ``None`` PyTorch's default collate is used.
    """

    fitted: bool

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    def setup(self, stage: str | None = None) -> None:
        self.fitted = True

    def train_dataloader(self) -> DataLoader:
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=True)
        return DataLoader(self.split_dataset.train, collate_fn=self.dataset.collate_fn, **kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return DataLoader(
            self.split_dataset.validation, collate_fn=self.dataset.collate_fn, **kwargs
        )

    def test_dataloader(self) -> DataLoader:
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return DataLoader(self.split_dataset.test, collate_fn=self.dataset.collate_fn, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        # Use the test split for prediction by default to ensure non-empty predictions
        # and consistent behavior across datamodules. The BaseDataModule already
        # defaults to test_dataloader; mirror that here.
        return self.test_dataloader()
