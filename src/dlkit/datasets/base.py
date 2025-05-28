from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from dlkit.datatypes.dataset import Shape


class BaseDataset(Dataset, ABC):
    """Base class for datasets in the DLKit.

    This class provides a common interface for datasets, including methods
    for getting the length of the dataset and retrieving items by index.
    """

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        raise NotImplementedError("Subclasses must implement the shape property.")
