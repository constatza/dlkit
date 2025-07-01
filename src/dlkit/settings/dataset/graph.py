from pydantic import Field
from pydantic import FilePath, DirectoryPath

from .base import DatasetSettings


class GraphDatasetSettings(DatasetSettings):
    name: str = Field(default="GraphDataset", description="Dataset name.")
    root: DirectoryPath | None = Field(
        default=None, description="Root directory of the dataset.", alias="root_dir"
    )
    edge_index: FilePath | None = Field(default=None, description="Edge index file path.")
    edge_attr: FilePath | None = Field(default=None, description="Edge x file path.")
