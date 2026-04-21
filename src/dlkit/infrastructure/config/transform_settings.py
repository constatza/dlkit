from pydantic import Field

from .core.base_settings import StringNamedComponentSettings


class TransformSettings(StringNamedComponentSettings):
    """Typed configuration for a single transform in a ``TransformChain``.

    The settings mirror the arguments passed to the transform component. Use them inside
    dataset entries (features/targets) or anywhere a chain is constructed manually.

    Example:
        ```python
        from dlkit.infrastructure.config.transform_settings import TransformSettings

        minmax = TransformSettings(name="MinMaxScaler", dim=0)
        ```
    """

    module_path: str | None = Field(
        default=None,
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Optional Python module path containing the transform implementation.",
    )
    dim: tuple[int, ...] | int = Field(
        default=0,
        description="Dimension(s) over which the transform computes its statistics.",
    )
