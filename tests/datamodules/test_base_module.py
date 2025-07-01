import pytest
import torch
from torch.utils.data import TensorDataset

from dlkit.datatypes.dataset import SplitIndices
from dlkit.settings import DataloaderSettings
from dlkit.datamodules import InMemoryModule  # Replace with actual module path
from dlkit.utils.split import generate_split


@pytest.fixture
def ordered_dataset() -> TensorDataset:
    """Dataset with x [[0], [1], ..., [99]] and dummy targets."""
    data = torch.arange(100).unsqueeze(1)  # Shape: (100, 1)
    targets = torch.zeros_like(data)  # Shape: (100, 1)
    return TensorDataset(data, targets)


@pytest.fixture
def split_indices() -> SplitIndices:
    """Static deterministic split for all phases."""
    return generate_split(n=100, test_size=0.2, val_size=0.2)


@pytest.fixture
def dataloader_settings() -> DataloaderSettings:
    """Settings with shuffle=True (will be overridden in predict)."""
    return DataloaderSettings(batch_size=10, num_workers=1, shuffle=True)


def test_predict_dataloader_preserves_order(
    ordered_dataset: TensorDataset,
    split_indices: SplitIndices,
    dataloader_settings: DataloaderSettings,
):
    """Ensure predict_dataloader yields samples in original order."""
    dm = InMemoryModule(
        dataset=ordered_dataset, idx_split=split_indices, dataloader_settings=dataloader_settings
    )

    predicted_features = torch.cat([batch[0] for batch in dm.predict_dataloader()], dim=0)

    assert torch.equal(predicted_features, ordered_dataset[:][0]), (
        "Predict dataloader shuffled the data!"
    )
