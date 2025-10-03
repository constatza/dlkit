from pydantic import Field
from .core.base_settings import ComponentSettings


class TransformSettings(ComponentSettings):
    name: str = Field(..., description="Name of the transform.")
    module_path: str = Field(
        default="dlkit.core.training.transforms", description="Module path to the transform."
    )
    dim: tuple[int, ...] | int = Field(
        default=0,
        description="List of dimensions to apply the transform on.",
    )
