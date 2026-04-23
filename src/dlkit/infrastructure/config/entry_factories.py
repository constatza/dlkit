"""Factory functions and type-guard helpers for DataEntry objects.

Public API:
    Feature() - create PathFeature or ValueFeature
    Target()  - create PathTarget or ValueTarget
    ContextFeature() - create a feature excluded from model.forward()

Type aliases:
    FeatureType = PathFeature | ValueFeature | SparseFeature
    TargetType  = PathTarget  | ValueTarget

Type guards (all accept DataEntry, return bool):
    is_feature_entry, is_target_entry, is_path_based, is_value_based,
    is_writable, is_runtime_generated, has_feature_reference
"""

from pathlib import Path
from typing import overload

import numpy as np
import torch

from .entry_base import DataEntry
from .entry_protocols import (
    IFeatureReference,
    IPathBased,
    IRuntimeGenerated,
    IValueBased,
    IWritable,
)
from .entry_types import (
    AutoencoderTarget,
    PathFeature,
    PathTarget,
    SparseFeature,
    ValueFeature,
    ValueTarget,
)
from .transform_settings import TransformSettings

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FeatureType = PathFeature | ValueFeature | SparseFeature
TargetType = PathTarget | ValueTarget

# ---------------------------------------------------------------------------
# Feature factory
# ---------------------------------------------------------------------------


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
    """Create a ``PathFeature`` or ``ValueFeature`` based on the supplied arguments.

    Args:
        name: Entry name; defaults to the dict key when stored in a mapping.
        path: File path — produces a ``PathFeature`` (or placeholder if None).
        value: In-memory array — produces a ``ValueFeature``.
        dtype: PyTorch dtype override.
        model_input: Forwarding semantics for ``model.forward()``.
        loss_input: Route this entry as a loss kwarg with this name.
        transforms: Transform chain for this entry.

    Returns:
        ``ValueFeature`` when ``value`` is provided; ``PathFeature`` otherwise.

    Raises:
        ValueError: If both ``path`` and ``value`` are provided.

    Examples:
        >>> f1 = Feature(name="x", path="data/features.npy")
        >>> isinstance(f1, PathFeature)
        True
        >>> f2 = Feature(name="x", value=np.ones((100, 5)))
        >>> isinstance(f2, ValueFeature)
        True
    """
    if value is not None and path is not None:
        raise ValueError(
            f"Feature '{name or 'unknown'}' cannot have both 'path' and 'value' (use one)."
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

    resolved_path = Path(path) if path is not None else None
    return PathFeature(
        name=name,
        path=resolved_path,
        dtype=dtype,
        model_input=model_input,
        loss_input=loss_input,
        transforms=transform_list,
    )


# ---------------------------------------------------------------------------
# Target factory
# ---------------------------------------------------------------------------


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
    """Create a ``PathTarget`` or ``ValueTarget`` based on the supplied arguments.

    Args:
        name: Entry name; defaults to the dict key when stored in a mapping.
        path: File path — produces a ``PathTarget`` (or placeholder if None).
        value: In-memory array — produces a ``ValueTarget``.
        dtype: PyTorch dtype override.
        write: Save the corresponding prediction during inference.
        loss_input: Route this entry as a loss kwarg with this name.
        transforms: Transform chain for this entry.

    Returns:
        ``ValueTarget`` when ``value`` is provided; ``PathTarget`` otherwise.

    Raises:
        ValueError: If both ``path`` and ``value`` are provided.

    Examples:
        >>> t1 = Target(name="y", path="data/targets.npy")
        >>> isinstance(t1, PathTarget)
        True
        >>> t2 = Target(name="y", value=np.zeros((100, 1)))
        >>> isinstance(t2, ValueTarget)
        True
    """
    if value is not None and path is not None:
        raise ValueError(
            f"Target '{name or 'unknown'}' cannot have both 'path' and 'value' (use one)."
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

    resolved_path = Path(path) if path is not None else None
    return PathTarget(
        name=name,
        path=resolved_path,
        dtype=dtype,
        write=write,
        loss_input=loss_input,
        transforms=transform_list,
    )


# ---------------------------------------------------------------------------
# ContextFeature convenience factory
# ---------------------------------------------------------------------------


def ContextFeature(
    name: str | None = None,
    *,
    path: Path | str | None = None,
    value: torch.Tensor | np.ndarray | None = None,
    dtype: torch.dtype | None = None,
    transforms: list[TransformSettings] | None = None,
) -> FeatureType:
    """Create a feature that is available in the batch but not forwarded to the model.

    Context features are useful for tensors (e.g. stiffness matrices) that a
    custom loss function needs but that should not appear as model inputs.

    Args:
        name: Entry name.
        path: File path for the feature data.
        value: In-memory tensor or array.
        dtype: PyTorch dtype override.
        transforms: Transform chain for this entry.

    Returns:
        ``PathFeature`` or ``ValueFeature`` with ``model_input=False``.

    Raises:
        ValueError: If both ``path`` and ``value`` are provided.
    """
    if value is not None and path is not None:
        raise ValueError(
            f"ContextFeature '{name or 'unknown'}' cannot have both 'path' and 'value'."
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


# ---------------------------------------------------------------------------
# Type guards
# ---------------------------------------------------------------------------


def is_feature_entry(entry: DataEntry) -> bool:
    """Return True if ``entry`` is any kind of feature.

    Args:
        entry: The entry to inspect.

    Returns:
        True for ``PathFeature``, ``ValueFeature``, or ``SparseFeature``.
    """
    return isinstance(entry, (PathFeature, ValueFeature, SparseFeature))


def is_target_entry(entry: DataEntry) -> bool:
    """Return True if ``entry`` is any kind of target.

    Args:
        entry: The entry to inspect.

    Returns:
        True for ``PathTarget``, ``ValueTarget``, or ``AutoencoderTarget``.
    """
    return isinstance(entry, (PathTarget, ValueTarget, AutoencoderTarget))


def is_path_based(entry: DataEntry) -> bool:
    """Return True if ``entry`` loads data from a file path.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IPathBased``.
    """
    return isinstance(entry, IPathBased)


def is_value_based(entry: DataEntry) -> bool:
    """Return True if ``entry`` holds an in-memory value.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IValueBased``.
    """
    return isinstance(entry, IValueBased)


def is_writable(entry: DataEntry) -> bool:
    """Return True if ``entry`` can be persisted during inference.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IWritable``.
    """
    return isinstance(entry, IWritable)


def is_runtime_generated(entry: DataEntry) -> bool:
    """Return True if ``entry`` is created by the model at run-time.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IRuntimeGenerated``.
    """
    return isinstance(entry, IRuntimeGenerated)


def has_feature_reference(entry: DataEntry) -> bool:
    """Return True if ``entry`` references another feature entry.

    Args:
        entry: The entry to inspect.

    Returns:
        True if ``entry`` implements ``IFeatureReference``.
    """
    return isinstance(entry, IFeatureReference)


__all__ = [
    "Feature",
    "FeatureType",
    "Target",
    "TargetType",
    "ContextFeature",
    "is_feature_entry",
    "is_target_entry",
    "is_path_based",
    "is_value_based",
    "is_writable",
    "is_runtime_generated",
    "has_feature_reference",
]
