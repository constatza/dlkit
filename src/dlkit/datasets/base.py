from abc import abstractmethod
from torch.utils.data import TensorDataset
from pydantic import validate_call, FilePath
from dlkit.datatypes.dataset import Shape
from dlkit.io import load_array


class BaseDataset(TensorDataset):
    """Base class for datasets in the DLKit.

    This class provides a common interface for datasets, including methods
    for getting the length of the dataset and retrieving items by index.
    """

    @validate_call
    def __init__(
        self,
        *file_paths: FilePath,
    ) -> None:
        """Initializes a new instance of the BaseDataset class.

        Args:
            features (Path): The path to the x file.
            targets (Path | None, optional): The path to the targets file.
        """
        tensors = []
        names = []
        for file_path in file_paths:
            tensors.append(load_array(file_path))
            names.append(file_path.with_suffix("").name)

        super().__init__(*tensors)
        self.names = names

    @property
    @abstractmethod
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        raise NotImplementedError("Subclasses must implement the shape property.")
