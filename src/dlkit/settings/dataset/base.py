from pydantic import Field

from torch.utils.data import Dataset
from ..base_settings import ClassSettings
from pydantic import FilePath, model_validator, DirectoryPath


class DatasetSettings(ClassSettings[Dataset]):
    """Settings for the pytorch dataset."""

    name: str = Field(default="SupervisedArrayDataset", description="Dataset name.")
    module_path: str = Field(
        default="dlkit.datasets",
        description="Module path where the dataset class is located.",
    )
    root: DirectoryPath | None = Field(
        default=None, description="Root directory of the dataset.", alias="root_dir"
    )
    x: FilePath = Field(..., description="Features file path.", alias="features")
    y: FilePath | None = Field(None, description="Targets file path.", alias="targets")
    edge_index: FilePath | None = Field(
        default=None, description="Edge index file path.", alias="adjacency"
    )
    edge_attr: FilePath | None = Field(
        default=None, description="Edge x file path.", alias="edge_features"
    )

    @model_validator(mode="after")
    def populate_targets(self):
        if self.y is None:
            self.y = self.x
        return self
