"""Data entry abstractions for FlexibleDataset configuration.

This module defines the core dataflow entry types used for configuring
flexible datasets. These are pure configuration objects with no
processing logic, following the single responsibility principle.

Architecture:
    DataEntry (ABC) - base with name, dtype, transforms, model_input, loss_input
        ├── PathBasedEntry (ABC) - file-based entries with path validation
        │   ├── PathFeature - feature loaded from file
        │   └── PathTarget - target loaded from file
        ├── ValueBasedEntry (ABC) - in-memory entries with tensor/array value
        │   ├── ValueFeature - feature from memory
        │   └── ValueTarget - target from memory
        ├── Latent - model-generated intermediate (no path/value)
        ├── Prediction - model output (runtime only)
        └── AutoencoderTarget - references a feature for transform inversion

Factory Functions:
    Feature() - creates PathFeature or ValueFeature based on arguments
    Target() - creates PathTarget or ValueTarget based on arguments

Capability Interfaces (ABC Mixins):
    IPathBased - marks entries that load data from file paths
    IValueBased - marks entries with in-memory tensor/array values
    IWritable - marks entries that can be saved during inference
    IRuntimeGenerated - marks entries created during model execution
    IFeatureReference - marks entries that reference other features
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
import torch
from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_settings import SettingsConfigDict

from .core.base_settings import BasicSettings
from .transform_settings import TransformSettings

# =============================================================================
# Base Classes
# =============================================================================


class DataEntry(BasicSettings, ABC):
    """Base abstraction for dataflow configuration.

    This class defines the common interface for all dataflow entries
    without any processing logic. Following the Open/Closed principle,
    it can be extended with new dataflow entry types without modification.

    Attributes:
        name: Entry name (optional - defaults to dict key when stored in dict)
        dtype: PyTorch dataflow type for the tensor dataflow
        transforms: List of transforms to apply to this dataflow entry
        model_input: Whether this entry is passed to the model forward()
        loss_input: If set, route this entry as a loss function kwarg with this name
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    name: str | None = Field(default=None, description="Entry name (defaults to dict key)")
    dtype: torch.dtype | None = Field(
        default=None, description="PyTorch dataflow type. If None, uses session precision strategy."
    )
    transforms: list[TransformSettings] = Field(
        default_factory=list, description="Transform chain for this dataflow entry"
    )
    model_input: int | str | bool | None = Field(
        default=True,
        description=(
            "Controls whether and how this feature is passed to model.forward(). "
            "False/None: excluded (context tensor only, not passed to model). "
            "True (default): passed as kwarg using the entry name — model(entry_name=tensor). "
            "int: explicit positional index — 0 = first arg, 1 = second, etc. "
            "    Features are sorted by this index before building the invoker. "
            "str digit ('0','1',...): explicit positional index (useful in TOML configs). "
            "str identifier ('name'): keyword argument with this name — model(name=tensor). "
            "    Decouples the kwarg name from the entry name."
        ),
    )

    @field_validator("model_input")
    @classmethod
    def _validate_model_input(cls, v: int | str | bool | None) -> int | str | bool | None:
        """Validate string model_input values.

        Args:
            v: The model_input value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If v is a non-empty string that is neither a digit string
                nor a valid Python identifier.
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
            "If set, this entry is automatically routed to the loss function as a keyword "
            "argument with the given name. E.g. loss_input='K' passes the tensor as "
            "loss_fn(..., K=tensor). Entry name and kwarg name are explicitly decoupled. "
            "Use model_input=False together with loss_input to create context tensors "
            "that feed custom loss functions but are not passed to the model forward()."
        ),
    )

    @field_validator("name")
    @classmethod
    def _no_dots_in_name(cls, v: str | None) -> str | None:
        """Validate that entry names do not contain dots.

        Dots are reserved as separators in batch key format (namespace.entry_name).

        Args:
            v: The name to validate

        Returns:
            The validated name

        Raises:
            ValueError: If name contains a dot
        """
        if v and "." in v:
            raise ValueError(f"Entry name must not contain '.', got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_name_when_data_present(self) -> DataEntry:
        """Enforce name requirement when data source exists.

        This validator implements fail-fast validation for malformed configs.
        It catches the production bug where TOML entries have path/value but
        no name field, which would otherwise fail at runtime during dataset
        instantiation.

        Valid cases:
            - Placeholder: name=None, path=None, value=None (for later injection)
            - Path-based: name="x", path="/data.npy" (config file usage)
            - Value-based: name="x", value=<array> (programmatic usage)

        Invalid cases (caught at config load time):
            - Missing name: name=None, path="/data.npy"
            - Missing name: name=None, value=<array>

        Returns:
            The validated DataEntry instance.

        Raises:
            ValueError: If entry has data source (path or value) but no name.
        """
        # Skip validation for placeholders (per-module laziness support)
        if self.is_placeholder():
            return self

        # Enforce name when data source exists
        has_data_source = self.has_path() or self.has_value()
        if has_data_source and not self.name:
            entry_type = self.__class__.__name__

            # Determine which field is present for better error message
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
        """Check if this entry has an in-memory value.

        Returns:
            True if value is present, False otherwise
        """
        ...

    @abstractmethod
    def has_path(self) -> bool:
        """Check if this entry has a file path.

        Returns:
            True if path attribute exists and is set, False otherwise
        """
        ...

    @abstractmethod
    def is_placeholder(self) -> bool:
        """Check if this entry is a placeholder awaiting value injection.

        Returns:
            True if entry needs value to be injected before use, False otherwise
        """
        ...

    def get_effective_dtype(self, precision_provider=None) -> torch.dtype:
        """Get the effective dtype for this dataflow entry.

        Args:
            precision_provider: Optional precision provider for strategy resolution.

        Returns:
            torch.dtype to use for this dataflow entry, resolved from explicit dtype
            or session precision strategy.
        """
        if self.dtype is not None:
            return self.dtype

        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        return precision_service.get_torch_dtype(precision_provider)

    def resolve_dtype_with_fallback(
        self, fallback_dtype: torch.dtype = torch.float32
    ) -> torch.dtype:
        """Resolve dtype with explicit fallback.

        Args:
            fallback_dtype: Fallback dtype if no precision can be resolved.

        Returns:
            torch.dtype to use for this dataflow entry.
        """
        if self.dtype is not None:
            return self.dtype

        try:
            from dlkit.interfaces.api.services.precision_service import get_precision_service

            precision_service = get_precision_service()
            return precision_service.get_torch_dtype()
        except Exception:
            return fallback_dtype


# =============================================================================
# Capability ABC Interfaces (Mixin Pattern)
# =============================================================================


class IPathBased(ABC):
    """Capability interface for file path-based data entries.

    Marks entries that load data from file paths.
    Following the Mixin ABCs pattern from transform architecture.

    This interface enables runtime capability checking with isinstance()
    and enforces explicit contracts for path-based data access.

    Example:
        >>> entry = PathFeature(name="x", path="data.npy")
        >>> isinstance(entry, IPathBased)  # True
        >>> is_path_based(entry)  # True
    """

    @abstractmethod
    def get_path(self) -> Path | None:
        """Get the file path for this entry.

        Returns:
            Path object if available, None if placeholder mode.
        """


class IValueBased(ABC):
    """Capability interface for in-memory data entries.

    Marks entries that contain in-memory tensor/array values.
    Following the Mixin ABCs pattern from transform architecture.

    This interface enables runtime capability checking with isinstance()
    and enforces explicit contracts for value-based data access.

    Example:
        >>> entry = ValueFeature(name="x", value=np.ones((10, 5)))
        >>> isinstance(entry, IValueBased)  # True
        >>> is_value_based(entry)  # True
    """

    @abstractmethod
    def get_value(self) -> torch.Tensor | np.ndarray | None:
        """Get the in-memory value for this entry.

        Returns:
            Tensor or array if available, None if placeholder mode.
        """


class IWritable:
    """Capability interface for writable entries.

    Marks entries that can save their data during inference.
    Targets, Predictions, and Latents can be writable.

    This interface enables runtime capability checking with isinstance()
    and defines the write capability contract. Implementations must
    provide a 'write: bool' attribute.

    Example:
        >>> target = PathTarget(name="y", path="targets.npy", write=True)
        >>> isinstance(target, IWritable)  # True
        >>> is_writable(target)  # True
        >>> target.write  # True
    """

    # Pure marker interface - implementations provide 'write' attribute


class IRuntimeGenerated:
    """Capability interface for runtime-generated entries.

    Marks entries created during model execution (Latent, Prediction),
    not loaded from configuration or dataset.

    This is a pure marker interface with no methods. It enables
    runtime capability checking with isinstance().

    Example:
        >>> latent = Latent(name="z", write=True)
        >>> isinstance(latent, IRuntimeGenerated)  # True
        >>> is_runtime_generated(latent)  # True
    """

    # Pure marker interface


class IFeatureReference:
    """Capability interface for entries that reference features.

    Used by AutoencoderTarget to reference a feature for transform inversion.

    This interface enables runtime capability checking with isinstance()
    and defines the feature reference contract. Implementations must
    provide a 'feature_ref: str' attribute.

    Example:
        >>> target = AutoencoderTarget(name="y", feature_ref="x")
        >>> isinstance(target, IFeatureReference)  # True
        >>> has_feature_reference(target)  # True
        >>> target.feature_ref  # "x"
    """

    # Pure marker interface - implementations provide 'feature_ref' attribute


# =============================================================================
# Base Classes with Capability Interfaces
# =============================================================================


class PathBasedEntry(DataEntry, IPathBased, ABC):
    """Base class for file-based data entries.

    PathBasedEntry provides shared path validation and access for entries
    that load data from files. Supports placeholder mode where path
    can be None if value will be injected programmatically.

    Attributes:
        path: File path to the data file (None for placeholder mode)
    """

    path: Path | None = Field(
        default=None, description="Path to the data file (None for placeholder mode)"
    )

    def get_path(self) -> Path | None:
        """Get the file path for this entry (IPathBased implementation).

        Returns:
            Path object if available, None if placeholder mode.
        """
        return self.path

    def has_value(self) -> bool:
        """PathBasedEntry does not have in-memory value.

        Returns:
            Always False for path-based entries
        """
        return False

    def has_path(self) -> bool:
        """Check if this entry has a valid file path.

        Returns:
            True if path is set, False for placeholder mode
        """
        return self.path is not None

    def is_placeholder(self) -> bool:
        """Check if this entry is in placeholder mode.

        Returns:
            True if path is None (awaiting value injection), False otherwise
        """
        return self.path is None

    @model_validator(mode="after")
    def validate_path_existence(self, info: ValidationInfo) -> PathBasedEntry:
        """Validate path existence with eager validation.

        Validates path immediately when not None, supporting fail-fast
        error detection during configuration loading.

        Args:
            info: Pydantic validation info (unused, kept for compatibility).

        Returns:
            The validated PathBasedEntry instance

        Raises:
            ValueError: If path is specified but does not exist.
        """
        if self.path is not None and not self.path.exists():
            raise ValueError(f"Path does not exist: {self.path}")
        return self


class ValueBasedEntry(DataEntry, IValueBased, ABC):
    """Base class for in-memory data entries.

    ValueBasedEntry provides shared value handling for entries that
    receive data programmatically (e.g., for testing or API use).
    Supports placeholder mode where value can be None if it will
    be injected programmatically.

    Attributes:
        value: In-memory tensor or numpy array (None for placeholder mode)
    """

    value: torch.Tensor | np.ndarray | None = Field(
        default=None,
        description="In-memory tensor/array value (None for placeholder mode)",
        exclude=True,  # exclude from serialization; retain for runtime use
    )

    def get_value(self) -> torch.Tensor | np.ndarray | None:
        """Get the in-memory value for this entry (IValueBased implementation).

        Returns:
            Tensor or array if available, None if placeholder mode.
        """
        return self.value

    def has_value(self) -> bool:
        """Check if this entry has an in-memory value.

        Returns:
            True if value is set, False for placeholder mode
        """
        return self.value is not None

    def has_path(self) -> bool:
        """Value-based entries never have file paths."""
        return False

    def is_placeholder(self) -> bool:
        """Placeholder when no value has been provided."""
        return self.value is None


# =============================================================================
# Runtime tensor entries (non-Pydantic)
# =============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorDataEntry:
    """Runtime entry that holds a resolved tensor for model consumption."""

    name: str
    tensor: torch.Tensor
    write: bool = False
    transforms: tuple[TransformSettings, ...] = ()


def _coerce_tensor(value: torch.Tensor | np.ndarray, dtype: torch.dtype | None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def to_tensor_entry(entry: DataEntry) -> TensorDataEntry:
    """Convert a config entry (path or value) into a concrete tensor entry."""
    name = entry.name or "anonymous"
    dtype = getattr(entry, "dtype", None)
    write = getattr(entry, "write", False)
    transforms = tuple(getattr(entry, "transforms", ()) or ())

    if isinstance(entry, PathBasedEntry):
        if not entry.has_path() or entry.path is None:
            raise ValueError(f"Entry '{name}' is a placeholder without a path or value")
        from dlkit.tools.io.arrays import load_array  # Local import to avoid import cycles

        tensor = load_array(entry.path, dtype=dtype)
        return TensorDataEntry(name=name, tensor=tensor, write=write, transforms=transforms)

    if isinstance(entry, ValueBasedEntry):
        if not entry.has_value() or entry.value is None:
            raise ValueError(f"Entry '{name}' is a placeholder without a value")
        tensor = _coerce_tensor(entry.value, dtype=dtype)
        return TensorDataEntry(name=name, tensor=tensor, write=write, transforms=transforms)

    raise TypeError(f"Unsupported entry type for tensor conversion: {type(entry)}")


def convert_to_tensor_entries(entries: Iterable[DataEntry]) -> tuple[TensorDataEntry, ...]:
    """Convert a sequence of config entries to concrete tensor entries."""
    return tuple(to_tensor_entry(entry) for entry in entries)


# =============================================================================
# Path-Based Concrete Classes
# =============================================================================


class PathFeature(PathBasedEntry):
    """Feature entry loaded from a file path.

    PathFeature represents input data for model features loaded from disk.
    Supports placeholder mode where path can be None if value will be
    injected programmatically before use.

    Attributes:
        path: File path to the feature data (None for placeholder mode)

    Notes:
        Autoencoder behaviour:
        - When the wrapper is configured with ``is_autoencoder = True`` and no explicit
          targets are provided in the dataset configuration, the processing pipeline
          (LossPairingStep) will automatically treat the feature entries as targets.
    """


def _validate_sparse_filename(name: str, field_name: str) -> None:
    """Validate sparse payload filename configuration."""
    if not name:
        raise ValueError(f"{field_name} filename must be non-empty")
    if "/" in name or "\\" in name:
        raise ValueError(f"{field_name} filename must be a local basename, got '{name}'")
    if not name.endswith(".npy"):
        raise ValueError(f"{field_name} filename must end with '.npy', got '{name}'")


@pydantic_dataclass(config=ConfigDict(frozen=True))
class SparseFilesConfig:
    """Pydantic contract for sparse payload filenames in data entries."""

    indices: str = "indices.npy"
    values: str = "values.npy"
    nnz_ptr: str = "nnz_ptr.npy"
    values_scale: str = "values_scale.npy"

    def __post_init__(self) -> None:
        _validate_sparse_filename(self.indices, "indices")
        _validate_sparse_filename(self.values, "values")
        _validate_sparse_filename(self.nnz_ptr, "nnz_ptr")
        _validate_sparse_filename(self.values_scale, "values_scale")


class SparseFeature(PathBasedEntry):
    """Feature entry loaded from a sparse pack directory.

    The `path` points to a sparse pack directory containing sparse payload arrays.
    Runtime loading does not require a manifest file.
    """

    files: SparseFilesConfig = Field(
        default_factory=SparseFilesConfig,
        description="Sparse payload file naming",
    )
    denormalize: bool = Field(
        default=False,
        description=(
            "If true, apply values_scale during read: A_original = A_stored * values_scale."
        ),
    )

    @model_validator(mode="after")
    def validate_is_sparse_pack(self) -> SparseFeature:
        """Validate that `path` is a sparse pack directory."""
        if self.path is None:
            return self
        if not self.path.is_dir():
            raise ValueError(f"SparseFeature path must be a directory, got: {self.path}")
        required = (self.files.indices, self.files.values, self.files.nnz_ptr)
        if not all((self.path / name).exists() for name in required):
            raise ValueError(
                f"Not a sparse pack directory: {self.path}. Expected payload files: {self.files}"
            )
        scale_path = self.path / self.files.values_scale
        if scale_path.exists():
            raw_scale = np.load(scale_path, allow_pickle=False)
            if raw_scale.ndim == 0:
                value_scale = float(raw_scale)
            elif raw_scale.ndim == 1 and raw_scale.size == 1:
                value_scale = float(raw_scale[0])
            else:
                raise ValueError(
                    f"SparseFeature values_scale must be scalar or shape (1,), got {raw_scale.shape}"
                )
            if not np.isfinite(value_scale) or value_scale <= 0.0:
                raise ValueError(
                    f"SparseFeature values_scale must be finite and > 0, got {value_scale}"
                )
        return self


class PathTarget(PathBasedEntry, IWritable):
    """Target entry loaded from a file path.

    PathTarget represents ground truth data loaded from disk.
    Supports placeholder mode where path can be None if value will be
    injected programmatically before use.

    Attributes:
        path: File path to the target data (None for placeholder mode)
        write: Whether to save predictions for this target during inference

    Notes:
        Strict pairing:
        - During training/validation/testing the processing pipeline enforces a strict
          1:1 mapping between target names and prediction names.
    """

    write: bool = Field(
        default=False, description="Save the prediction related to this data during inference"
    )


# =============================================================================
# Value-Based Concrete Classes
# =============================================================================


class ValueFeature(ValueBasedEntry):
    """Feature entry with in-memory value.

    ValueFeature represents input data for model features provided
    programmatically (e.g., numpy arrays or tensors for testing).

    Attributes:
        value: In-memory tensor/array for the feature data

    Notes:
        Autoencoder behaviour:
        - When the wrapper is configured with ``is_autoencoder = True`` and no explicit
          targets are provided in the dataset configuration, the processing pipeline
          (LossPairingStep) will automatically treat the feature entries as targets.
    """


class ValueTarget(ValueBasedEntry, IWritable):
    """Target entry with in-memory value.

    ValueTarget represents ground truth data provided programmatically
    (e.g., numpy arrays or tensors for testing).

    Attributes:
        value: In-memory tensor/array for the target data
        write: Whether to save predictions for this target during inference

    Notes:
        Strict pairing:
        - During training/validation/testing the processing pipeline enforces a strict
          1:1 mapping between target names and prediction names.
    """

    write: bool = Field(
        default=False, description="Save the prediction related to this data during inference"
    )


# =============================================================================
# Factory Functions (Backwards Compatibility)
# =============================================================================

# Type aliases for backwards compatibility and type hints
FeatureType = PathFeature | ValueFeature | SparseFeature
TargetType = PathTarget | ValueTarget


@overload
def Feature(
    name: str | None = None,
    *,
    value: torch.Tensor | np.ndarray,
    dtype: torch.dtype | None = None,
    model_input: int | str | bool | None = True,
    loss_input: str | None = None,
    transforms: list[TransformSettings] | None = None,
) -> ValueFeature: ...


@overload
def Feature(
    name: str | None = None,
    *,
    path: Path | str | None = None,
    dtype: torch.dtype | None = None,
    model_input: int | str | bool | None = True,
    loss_input: str | None = None,
    transforms: list[TransformSettings] | None = None,
) -> PathFeature: ...


def Feature(
    name: str | None = None,
    *,
    path: Path | str | None = None,
    value: torch.Tensor | np.ndarray | None = None,
    dtype: torch.dtype | None = None,
    model_input: int | str | bool | None = True,
    loss_input: str | None = None,
    transforms: list[TransformSettings] | None = None,
) -> FeatureType:
    """Factory function to create the appropriate Feature type.

    Creates a PathFeature or ValueFeature based on provided arguments.

    Args:
        name: Entry name (optional - defaults to dict key when stored in dict)
        path: File path to feature data (creates PathFeature)
        value: In-memory tensor/array (creates ValueFeature)
        dtype: PyTorch dtype override
        model_input: If False, tensor is in batch but not passed to model forward()
        loss_input: If set, route as loss function kwarg with this name
        transforms: Transform chain for this entry

    Returns:
        PathFeature if path provided or placeholder mode, ValueFeature if value provided

    Raises:
        ValueError: If both path and value are provided

    Examples:
        >>> # Path-based feature (config file style)
        >>> f1 = Feature(name="x", path="data/features.npy")
        >>> isinstance(f1, PathFeature)
        True

        >>> # Value-based feature (programmatic style)
        >>> f2 = Feature(name="x", value=np.ones((100, 5)))
        >>> isinstance(f2, ValueFeature)
        True

        >>> # Context feature for loss (not passed to model)
        >>> f3 = Feature(name="K", value=K_matrix, model_input=False, loss_input="K")
        >>> f3.model_input
        False
    """
    if value is not None and path is not None:
        name_str = name or "unknown"
        raise ValueError(
            f"Feature '{name_str}' cannot have both 'path' and 'value' specified (use one or the other)"
        )

    transform_list = transforms or []

    if value is not None:
        return ValueFeature(
            name=name,
            value=value,
            dtype=dtype,
            model_input=model_input,
            loss_input=loss_input,
            transforms=transform_list,
        )

    # Path-based or placeholder mode
    resolved_path = Path(path) if path is not None else None
    return PathFeature(
        name=name,
        path=resolved_path,
        dtype=dtype,
        model_input=model_input,
        loss_input=loss_input,
        transforms=transform_list,
    )


@overload
def Target(
    name: str | None = None,
    *,
    value: torch.Tensor | np.ndarray,
    dtype: torch.dtype | None = None,
    write: bool = False,
    loss_input: str | None = None,
    transforms: list[TransformSettings] | None = None,
) -> ValueTarget: ...


@overload
def Target(
    name: str | None = None,
    *,
    path: Path | str | None = None,
    dtype: torch.dtype | None = None,
    write: bool = False,
    loss_input: str | None = None,
    transforms: list[TransformSettings] | None = None,
) -> PathTarget: ...


def Target(
    name: str | None = None,
    *,
    path: Path | str | None = None,
    value: torch.Tensor | np.ndarray | None = None,
    dtype: torch.dtype | None = None,
    write: bool = False,
    loss_input: str | None = None,
    transforms: list[TransformSettings] | None = None,
) -> TargetType:
    """Factory function to create the appropriate Target type.

    Creates a PathTarget or ValueTarget based on provided arguments.

    Args:
        name: Entry name (optional - defaults to dict key when stored in dict)
        path: File path to target data (creates PathTarget)
        value: In-memory tensor/array (creates ValueTarget)
        dtype: PyTorch dtype override
        write: Save predictions during inference (default False)
        loss_input: If set, route as loss function kwarg with this name
        transforms: Transform chain for this entry

    Returns:
        PathTarget if path provided or placeholder mode, ValueTarget if value provided

    Raises:
        ValueError: If both path and value are provided

    Examples:
        >>> # Path-based target (config file style)
        >>> t1 = Target(name="y", path="data/targets.npy")
        >>> isinstance(t1, PathTarget)
        True

        >>> # Value-based target (programmatic style)
        >>> t2 = Target(name="y", value=np.zeros((100, 1)))
        >>> isinstance(t2, ValueTarget)
        True
    """
    if value is not None and path is not None:
        name_str = name or "unknown"
        raise ValueError(
            f"Target '{name_str}' cannot have both 'path' and 'value' specified (use one or the other)"
        )

    transform_list = transforms or []

    if value is not None:
        return ValueTarget(
            name=name,
            value=value,
            dtype=dtype,
            write=write,
            loss_input=loss_input,
            transforms=transform_list,
        )

    # Path-based or placeholder mode
    resolved_path = Path(path) if path is not None else None
    return PathTarget(
        name=name,
        path=resolved_path,
        dtype=dtype,
        write=write,
        loss_input=loss_input,
        transforms=transform_list,
    )


def ContextFeature(
    name: str | None = None,
    *,
    path: Path | str | None = None,
    value: torch.Tensor | np.ndarray | None = None,
    dtype: torch.dtype | None = None,
    transforms: list[TransformSettings] | None = None,
) -> FeatureType:
    """Factory function to create a context feature (not passed to model).

    Context features are loaded into the batch and available for loss/metric
    computation, but are NOT passed as arguments to model.forward().
    Use for tensors like stiffness matrices needed only by custom loss functions.

    Args:
        name: Entry name
        path: File path to feature data
        value: In-memory tensor/array
        dtype: PyTorch dtype override
        transforms: Transform chain for this entry

    Returns:
        PathFeature or ValueFeature with model_input=False

    Raises:
        ValueError: If both path and value are specified
    """
    if value is not None and path is not None:
        name_str = name or "unknown"
        raise ValueError(
            f"ContextFeature '{name_str}' cannot have both 'path' and 'value' specified"
        )

    transform_list = transforms or []

    if value is not None:
        return ValueFeature(
            name=name,
            value=value,
            dtype=dtype,
            model_input=False,
            transforms=transform_list,
        )

    resolved_path = Path(path) if path is not None else None
    return PathFeature(
        name=name,
        path=resolved_path,
        dtype=dtype,
        model_input=False,
        transforms=transform_list,
    )


# =============================================================================
# Special Entry Types
# =============================================================================


class Latent(DataEntry, IRuntimeGenerated, IWritable):
    """Intermediate dataflow configuration (model-generated).

    Latents represent intermediate representations generated by the model
    during inference. They have no file path and are created dynamically.

    Note: Latents are NOT handled by the FlexibleDataset - they are managed
    by the processing pipeline during model inference.

    Attributes:
        write: Whether to save this latent dataflow during inference
    """

    write: bool = Field(default=False, description="Save this latent during inference")

    def has_value(self) -> bool:
        """Latent does not have an in-memory value.

        Returns:
            Always False (latents are generated at runtime)
        """
        return False

    def has_path(self) -> bool:
        """Latent does not have a file path.

        Returns:
            Always False (latents are generated at runtime)
        """
        return False

    def is_placeholder(self) -> bool:
        """Latent is never a placeholder.

        Returns:
            Always False (latents are generated at runtime)
        """
        return False


class AutoencoderTarget(PathBasedEntry, IWritable, IFeatureReference):
    """Autoencoder reconstruction target that references a feature entry.

    AutoencoderTarget provides automatic transform inversion for autoencoder architectures.
    It references a Feature entry and automatically creates an inverted transform chain
    that applies inverse_transform() in reverse order.

    This is essential for transforms like SampleNormL2 that store per-sample state
    during forward() and require the same state for inverse_transform().

    Attributes:
        feature_ref: Name of the Feature entry to reference
        path: Copied from referenced feature at runtime (do not specify manually)
        write: Whether to save reconstruction data during inference

    Validation:
        - feature_ref must be specified
        - Referenced feature must exist in entry_configs
        - Referenced entry must be a Feature instance
        - Transforms should not be manually specified (derived from feature)

    Example:
        ```python
        x = Feature(
            name="x",
            path="input.npy",
            transforms=[
                TransformSettings(name="StandardScaler"),
                TransformSettings(name="SampleNormL2"),
            ],
        )

        y = AutoencoderTarget(
            name="y",
            feature_ref="x",  # Automatically inverts x's transforms
        )

        # Runtime: x transforms applied forward, y transforms applied inverse
        # SampleNormL2._last_norms preserved from forward to inverse
        ```
    """

    feature_ref: str = Field(
        ..., description="Name of the Feature entry to reference for transform inversion"
    )
    write: bool = Field(default=False, description="Save the reconstruction data during inference")

    # Internal value storage for when feature_ref is resolved
    _resolved_value: torch.Tensor | np.ndarray | None = None

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_feature_ref(self) -> AutoencoderTarget:
        """Validate feature_ref and warn about manual transforms.

        Returns:
            The validated AutoencoderTarget instance

        Raises:
            ValueError: If feature_ref is not provided
        """
        if not self.feature_ref:
            raise ValueError("AutoencoderTarget must have 'feature_ref' specified")

        if self.transforms:
            from dlkit.tools.utils.logging_config import get_logger

            get_logger(__name__).warning(
                "AutoencoderTarget '{}' ignores manual transforms because feature_ref='{}' "
                "derives them automatically.",
                self.name,
                self.feature_ref,
            )

        return self

    def has_value(self) -> bool:
        """Check if this entry has an in-memory value from resolved feature.

        Returns:
            True if value was resolved from feature_ref, False otherwise
        """
        return self._resolved_value is not None

    def is_placeholder(self) -> bool:
        """AutoencoderTarget is a placeholder until feature_ref is resolved.

        Returns:
            True if neither path nor value is available, False otherwise
        """
        return self.path is None and self._resolved_value is None


class Prediction(DataEntry, IRuntimeGenerated, IWritable):
    """Model prediction configuration.

    Predictions represent model outputs that correspond to specific targets.
    They are generated during model inference and can be compared to targets.

    Attributes:
        target_name: Name of the target this prediction corresponds to
        write: Whether to save prediction dataflow during inference
    """

    target_name: str = Field(..., description="Corresponding target name")
    write: bool = Field(default=True, description="Save predictions during inference")

    def has_value(self) -> bool:
        """Prediction does not have an in-memory value.

        Returns:
            Always False (predictions are generated at runtime)
        """
        return False

    def has_path(self) -> bool:
        """Prediction does not have a file path.

        Returns:
            Always False (predictions are generated at runtime)
        """
        return False

    def is_placeholder(self) -> bool:
        """Prediction is never a placeholder.

        Returns:
            Always False (predictions are generated at runtime)
        """
        return False


# =============================================================================
# Type Guards and Utilities
# =============================================================================


def is_feature_entry(entry: DataEntry) -> bool:
    """Check if entry is any type of feature.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry is PathFeature, ValueFeature, or created via Feature()
    """
    return isinstance(entry, (PathFeature, ValueFeature, SparseFeature))


def is_target_entry(entry: DataEntry) -> bool:
    """Check if entry is any type of target.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry is PathTarget, ValueTarget, or AutoencoderTarget
    """
    return isinstance(entry, (PathTarget, ValueTarget, AutoencoderTarget))


def is_path_based(entry: DataEntry) -> bool:
    """Check if entry is path-based.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry loads data from a file path
    """
    return isinstance(entry, IPathBased)


def is_value_based(entry: DataEntry) -> bool:
    """Check if entry has in-memory value.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry contains in-memory tensor/array
    """
    return isinstance(entry, IValueBased)


def is_writable(entry: DataEntry) -> bool:
    """Check if entry can be written during inference.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry can be saved during inference
    """
    return isinstance(entry, IWritable)


def is_runtime_generated(entry: DataEntry) -> bool:
    """Check if entry is generated at runtime.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry is created during model execution
    """
    return isinstance(entry, IRuntimeGenerated)


def has_feature_reference(entry: DataEntry) -> bool:
    """Check if entry references another feature.

    Args:
        entry: DataEntry to check

    Returns:
        True if entry references a feature (e.g., AutoencoderTarget)
    """
    return isinstance(entry, IFeatureReference)
