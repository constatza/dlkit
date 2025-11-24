from pydantic import Field
from .core.base_settings import ComponentSettings


class TransformSettings(ComponentSettings):
    """Typed configuration for a single transform in a ``TransformChain``.

    The settings mirror the arguments passed to the transform component. Use them inside
    dataset entries (features/targets) or anywhere a chain is constructed manually.

    Example:
        ```python
        from dlkit.tools.config.transform_settings import TransformSettings

        minmax = TransformSettings(
            name="MinMaxScaler",
            module_path="dlkit.core.training.transforms.minmax",
            dim=0,
        )
        ```
    """

    name: str = Field(..., description="Name of the transform class or registry alias.")
    module_path: str = Field(
        default="dlkit.core.training.transforms",
        description="Python module path containing the transform implementation.",
    )
    dim: tuple[int, ...] | int = Field(
        default=0,
        description="Dimension(s) over which the transform computes its statistics.",
    )
