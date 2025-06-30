from pydantic import Field

from torch.utils.data import Dataset
from ..base_settings import ClassSettings


class DatasetSettings(ClassSettings[Dataset]):
    """Settings for the pytorch dataset."""

    name: str = Field("NumpyDataset", description="Dataset name.")
    module_path: str = Field(
        default="dlkit.datasets",
        description="Module path where the dataset class is located.",
    )
