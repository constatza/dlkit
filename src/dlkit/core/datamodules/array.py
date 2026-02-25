from torch.utils.data import DataLoader

from dlkit.core.datasets.flexible import collate_tensordict
from .base import BaseDataModule


class InMemoryModule[Dataset_T](BaseDataModule):
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
        return DataLoader(self.split_dataset.train, collate_fn=collate_tensordict, **kwargs)

    def val_dataloader(self) -> DataLoader:
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return DataLoader(self.split_dataset.validation, collate_fn=collate_tensordict, **kwargs)

    def test_dataloader(self) -> DataLoader:
        kwargs = self._get_dataloader_kwargs(DataLoader, shuffle=False)
        return DataLoader(self.split_dataset.test, collate_fn=collate_tensordict, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        # Use the test split for prediction by default to ensure non-empty predictions
        # and consistent behavior across datamodules. The BaseDataModule already
        # defaults to test_dataloader; mirror that here.
        return self.test_dataloader()
