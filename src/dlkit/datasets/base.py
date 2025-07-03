from abc import abstractmethod, ABC
from torch.utils.data import Dataset
from dlkit.datatypes.dataset import Shape

from dlkit.register import Registry


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


_dataset_registry = Registry[BaseDataset]()

register_dataset = _dataset_registry.register
get_dataset = _dataset_registry.get
