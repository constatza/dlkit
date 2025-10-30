import numpy as np
import torch
from pathlib import Path
from pydantic import FilePath, validate_call, ConfigDict
from torch import Tensor
from collections.abc import Mapping
from types import MappingProxyType
from collections.abc import Callable

from dlkit.interfaces.api.services.precision_service import get_precision_service

# ──────────────────────────────────────────────────────────────────────────────
# Frozen, immutable loader map, typed as a Mapping
_LOADER_MAP: Mapping[str, Callable[..., object]] = MappingProxyType({
    ".npy": np.load,
    ".txt": np.loadtxt,
    ".csv": np.loadtxt,
    ".pt": torch.load,
    ".pth": torch.load,
})
# ──────────────────────────────────────────────────────────────────────────────


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_array(
    path: FilePath,
    dtype: torch.dtype | None = None,
    **kwargs,
) -> Tensor:
    """Load an array or tensor from disk with precision-aware dtype resolution.

    This function loads data from various file formats and applies consistent
    precision handling based on the precision context or explicit overrides.

    Precision is resolved automatically from the global precision service,
    which checks the precision context (set via precision_override()) and
    falls back to the global default (FULL_32).

    Args:
        path: A FilePath pointing to .npy, .txt/.csv, or .pt/.pth file.
        dtype: Explicit torch.dtype to convert the loaded data to.
               If None, uses precision service to resolve from context.
        **kwargs: Forwarded to the underlying loader (e.g. np.load or torch.load).

    Returns:
        A torch.Tensor containing the loaded data with appropriate precision.

    Raises:
        ValueError: If `path.suffix` is not one of the supported extensions.
        TypeError: If the loader returns a type other than np.ndarray or Tensor.

    Examples:
        # Use precision from context (recommended)
        with precision_override(PrecisionStrategy.FULL_64):
            tensor = load_array("data.npy")  # Automatically float64

        # Override with explicit dtype (rare cases)
        tensor = load_array("data.npy", dtype=torch.float16)
    """
    suffix = Path(path).suffix.lower()
    loader = _LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix!r}")

    # Resolve target dtype using precision service if not explicitly provided
    if dtype is None:
        precision_service = get_precision_service()
        dtype = precision_service.get_torch_dtype()

    data = loader(path, **kwargs)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype)
    if isinstance(data, Tensor):
        return data.to(dtype=dtype)

    raise TypeError(f"Loader for {suffix!r} returned unexpected type: {type(data).__name__}")


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def load_array_with_session_precision(path: FilePath, **kwargs) -> Tensor:
    """Convenience function to load array using session precision.

    This is a convenience wrapper around load_array that explicitly uses
    the session precision without allowing dtype overrides.

    Args:
        path: A FilePath pointing to .npy, .txt/.csv, or .pt/.pth file.
        **kwargs: Forwarded to the underlying loader (e.g. np.load or torch.load).

    Returns:
        A torch.Tensor loaded with session-consistent precision.
    """
    return load_array(path, dtype=None, **kwargs)
