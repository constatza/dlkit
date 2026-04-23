"""Abstract base class for all DataEntry configuration objects.

DataEntry is the root of the entry hierarchy; it carries the fields that
every entry type shares (name, dtype, transform chain, routing flags) and
enforces cross-field invariants via Pydantic validators.

Concrete entry types live in ``entry_types``; factory functions in
``entry_factories``.
"""

from abc import ABC, abstractmethod
from enum import StrEnum

import torch
from pydantic import Field, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from .core.base_settings import BasicSettings
from .transform_settings import TransformSettings


class EntryRole(StrEnum):
    """Semantic role of a data entry in the pipeline."""

    FEATURE = "feature"
    TARGET = "target"
    LATENT = "latent"
    PREDICTION = "prediction"
    AUTOENCODER_TARGET = "autoencoder_target"


class DataEntry(BasicSettings, ABC):
    """Base abstraction for dataflow configuration.

    Attributes:
        name: Entry name; defaults to the dict key when stored in a mapping.
        dtype: PyTorch tensor dtype.  None resolves via the session precision
            strategy at load time.
        transforms: Transform chain applied to this entry's data.
        model_input: Controls how (or whether) the entry is forwarded to
            ``model.forward()``.  See field description for full semantics.
        loss_input: When set, routes this entry as a keyword argument to the
            loss function using the given name.
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
    model_input: int | str | bool | None = Field(
        default=True,
        description=(
            "Controls whether and how this feature is passed to model.forward(). "
            "False/None: excluded. "
            "True: kwarg using the entry name — model(entry_name=tensor). "
            "int: explicit positional index (0 = first arg). "
            "str digit ('0','1',...): explicit positional index. "
            "str identifier ('name'): kwarg with this name — model(name=tensor)."
        ),
    )

    @field_validator("model_input")
    @classmethod
    def _validate_model_input(cls, v: int | str | bool | None) -> int | str | bool | None:
        """Reject empty or non-identifier/non-digit string values.

        Args:
            v: The model_input value to validate.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If ``v`` is a string that is neither a digit string nor
                a valid Python identifier.
        """
        if isinstance(v, str):
            if not v:
                raise ValueError("model_input must be non-empty. Use False/None to exclude.")
            if not v.isdigit() and not v.isidentifier():
                raise ValueError(
                    f"model_input '{v}' must be a digit string ('0','1',...) "
                    "or a valid Python identifier (kwarg name)."
                )
        return v

    loss_input: str | None = Field(
        default=None,
        description=(
            "If set, this entry is routed to the loss function as a kwarg with this name. "
            "Combine with model_input=False to create context tensors that feed the loss "
            "but are not passed to model.forward()."
        ),
    )

    entry_role: EntryRole = Field(exclude=True, description="Semantic role of the entry.")

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

    @abstractmethod
    def has_value(self) -> bool:
        """Return True if this entry holds an in-memory value.

        Returns:
            True if value is present, False otherwise.
        """

    @abstractmethod
    def has_path(self) -> bool:
        """Return True if this entry has a file path.

        Returns:
            True if path is set, False otherwise.
        """

    @abstractmethod
    def is_placeholder(self) -> bool:
        """Return True if this entry is awaiting value injection.

        Returns:
            True if the entry needs a value before it can be used.
        """

    def get_effective_dtype(self, precision_provider=None) -> torch.dtype:
        """Resolve the effective dtype for this entry.

        Args:
            precision_provider: Optional precision provider for strategy resolution.

        Returns:
            The resolved ``torch.dtype``.
        """
        if self.dtype is not None:
            return self.dtype

        from dlkit.infrastructure.precision.service import get_precision_service

        return get_precision_service().get_torch_dtype(precision_provider)

    def resolve_dtype_with_fallback(
        self, fallback_dtype: torch.dtype = torch.float32
    ) -> torch.dtype:
        """Resolve dtype with an explicit fallback when precision service is unavailable.

        Args:
            fallback_dtype: Dtype to use if resolution fails.

        Returns:
            The resolved ``torch.dtype``.
        """
        if self.dtype is not None:
            return self.dtype

        try:
            from dlkit.infrastructure.precision.service import get_precision_service

            return get_precision_service().get_torch_dtype()
        except Exception:
            return fallback_dtype
