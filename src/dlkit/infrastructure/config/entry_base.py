"""Abstract base class for all DataEntry configuration objects.

DataEntry is the root of the entry hierarchy; it carries the fields that
every entry type shares (name, dtype, transform chain, routing flags) and
enforces cross-field invariants via Pydantic validators.

Concrete entry types live in ``entry_types``; factory functions in
``entry_factories``.
"""

from abc import ABC

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from dlkit.common.geometry import FieldRole, GeometryKind

from .core.base_settings import BasicSettings
from .data_roles import DataRole
from .transform_settings import TransformSettings


class DataEntry(BasicSettings, ABC):
    """Base abstraction for dataflow configuration.

    Attributes:
        name: Entry name; defaults to the dict key when stored in a mapping.
        dtype: PyTorch tensor dtype.  None resolves via the session precision
            strategy at load time.
        transforms: Transform chain applied to this entry's data.
        model_input: When True (default), the entry is forwarded to
            ``model.forward()``. Named features are dispatched as keyword
            arguments using ``name`` as the forward-parameter name; unnamed
            features use positional dispatch. When False, the entry is excluded
            from model dispatch.
        loss_input: When set, routes this entry as a keyword argument to the
            loss function using the given name.
        field_role: Physics-domain role of this field (feature, coordinates,
            etc.).  Excluded from serialization.
        geometry_kind: Spatial structure of the field data.  Excluded from
            serialization.
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    name: str | None = Field(default=None, description="Entry name (defaults to dict key)")
    dtype: torch.dtype | None = Field(
        default=None,
        description="PyTorch dtype.  None → resolved from session precision strategy.",
    )
    transforms: list[TransformSettings] = Field(
        default_factory=list, description="Transform chain for this entry"
    )
    model_input: bool = Field(
        default=True,
        strict=True,
        description=(
            "Controls whether this feature is passed to model.forward(). "
            "True (default): include in model dispatch. Named features bind by "
            "keyword using the entry name as the forward() parameter name; "
            "unnamed features use positional dispatch. "
            "False: excluded from model dispatch entirely."
        ),
    )
    loss_input: str | None = Field(
        default=None,
        description=(
            "If set, this entry is routed to the loss function as a kwarg with this name. "
            "Combine with model_input=False to create context tensors that feed the loss "
            "but are not passed to model.forward()."
        ),
    )
    data_role: DataRole = Field(
        default=DataRole.FEATURE,
        description="ML pipeline role: where this entry lives in the batch TensorDict.",
    )
    write: bool = Field(
        default=False,
        description="When True, save predictions/latents for this entry during inference.",
    )
    field_role: FieldRole = Field(
        default=FieldRole.FEATURE,
        exclude=True,
        description="Physics-domain role of this field. Excluded from model_dump() output.",
    )
    geometry_kind: GeometryKind = Field(
        default=GeometryKind.TABULAR,
        exclude=True,
        description="Spatial structure of the field data. Excluded from model_dump() output.",
    )

    @field_validator("name")
    @classmethod
    def _no_dots_in_name(cls, v: str | None) -> str | None:
        """Reject names that contain dots (dots are batch-key separators).

        Args:
            v: The name to validate.

        Returns:
            The validated name unchanged.

        Raises:
            ValueError: If the name contains a dot.
        """
        if v and "." in v:
            raise ValueError(f"Entry name must not contain '.', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_name_when_data_present(self) -> DataEntry:
        """Require a name whenever a concrete data source is specified.

        Catches the production bug where a TOML entry has a path or value but
        no name field, which would otherwise fail at dataset instantiation time.

        Returns:
            The validated DataEntry instance.

        Raises:
            ValueError: If the entry has a data source but no name.
        """
        if self.is_placeholder():
            return self

        has_data_source = self.has_path() or self.has_value()
        if has_data_source and not self.name:
            entry_type = self.__class__.__name__

            if self.has_path():
                source_type = "path"
                try:
                    path_val = getattr(self, "path", None)
                    source_value = str(path_val) if path_val else "unknown"
                except Exception:
                    source_value = "unknown"
            else:
                source_type = "value"
                try:
                    value_val = getattr(self, "value", None)
                    source_value = type(value_val).__name__ if value_val is not None else "unknown"
                except Exception:
                    source_value = "unknown"

            raise ValueError(
                f"{entry_type} requires 'name' field when '{source_type}' is specified.\n"
                f"Found: {source_type}={source_value}, name=None.\n"
                f"\n"
                f"Fix: Add 'name' field to your TOML config:\n"
                f"  [[DATASET.features]]\n"
                f'  name = "your_feature_name"\n'
                f'  {source_type} = "..."\n'
                f"\n"
                f"Or remove '{source_type}' field for placeholder mode (programmatic injection)."
            )

        return self

    def has_value(self) -> bool:
        """Return True if this entry holds an in-memory value.

        Returns:
            False by default; overridden by ValueBasedEntry.
        """
        return False

    def has_path(self) -> bool:
        """Return True if this entry has a file path.

        Returns:
            False by default; overridden by PathBasedEntry.
        """
        return False

    def is_placeholder(self) -> bool:
        """Return True if this entry is awaiting value injection.

        Returns:
            False by default; overridden by PathBasedEntry and ValueBasedEntry.
        """
        return False
