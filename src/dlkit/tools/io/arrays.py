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
    precision_provider: object | None = None,
    **kwargs,
) -> Tensor:
    """Load an array or tensor from disk with precision-aware dtype resolution.

    This function loads dataflow from various file formats and applies consistent
    precision handling based on the session configuration or explicit overrides.

    Args:
        path: A FilePath pointing to .npy, .txt/.csv, or .pt/.pth file.
        dtype: Explicit torch.dtype to convert the loaded dataflow to.
               If None, uses precision service to resolve from session/context.
        precision_provider: Optional provider for precision strategy.
                           If None, uses global precision service.
        **kwargs: Forwarded to the underlying loader (e.g. np.load or torch.load).

    Returns:
        A torch.Tensor containing the loaded dataflow with appropriate precision.

    Raises:
        ValueError: If `path.suffix` is not one of the supported extensions.
        TypeError: If the loader returns a type other than np.ndarray or Tensor.

    Examples:
        # Use session precision (default)
        tensor = load_array("npy")

        # Override with explicit dtype
        tensor = load_array("npy", dtype=torch.float16)

        # Use custom precision provider
        tensor = load_array("npy", precision_provider=my_provider)
    """
    suffix = Path(path).suffix.lower()
    loader = _LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix!r}")

    # Resolve target dtype using precision service if not explicitly provided
    if dtype is None:
        precision_service = get_precision_service()
        # Cast to the protocol type for the service call
        from typing import cast
        from dlkit.interfaces.api.domain.precision import PrecisionProvider

        provider = cast(PrecisionProvider, precision_provider) if precision_provider else None
        dtype = precision_service.get_torch_dtype(provider)

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
