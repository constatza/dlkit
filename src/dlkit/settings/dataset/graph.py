from pydantic import Field
from pydantic import FilePath


class GraphDatasetSettings:
    features: FilePath = Field(..., description="Node features file path.", alias="x")
    targets: FilePath | None = Field(
        None, description="Node targets file path.", alias="y"
    )
    edge_index: FilePath | None = Field(None, description="Edge index file path.")
    edge_attr: FilePath | None = Field(None, description="Edge features file path.")
