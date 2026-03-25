from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from dlkit.tools.registry.registry import Registry


class BaseDataset[T](Dataset[T], ABC):
    """Base class for datasets in the DLKit.

    This class provides a common interface for datasets, including methods
    for getting the length of the dataset and retrieving items by index.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset."""
        raise NotImplementedError


_dataset_registry = Registry[type[BaseDataset[Any]]]()

register_dataset = _dataset_registry.register
get_dataset = _dataset_registry.get
