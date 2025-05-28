from abc import abstractmethod
from pathlib import Path
from torch.utils.data import Dataset
from dlkit.datatypes.dataset import Shape


class BaseDataset(Dataset):
    """Base class for datasets in the DLKit.

    This class provides a common interface for datasets, including methods
    for getting the length of the dataset and retrieving items by index.
    """

    def __init__(
        self,
        features: Path,
        targets: Path | None = None,
    ) -> None:
        """Initializes a new instance of the BaseDataset class.

        Args:
            features (Path): The path to the features file.
            targets (Path | None, optional): The path to the targets file.
        """
        super().__init__()
        self.features_path = features
        self.targets_path = targets

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        raise NotImplementedError("Subclasses must implement the shape property.")
