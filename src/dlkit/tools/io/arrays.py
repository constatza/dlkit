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


def _load_npz(path: Path | str, array_key: str | None = None, **kwargs) -> np.ndarray:
    """Load a single array from an NPZ file.

    NPZ files can contain multiple named arrays. This function extracts a single
    array either by explicit key or through auto-detection for single-array files.

    Args:
        path: Path to the .npz file.
        array_key: Name of the array to extract. If None, auto-detects for
                   single-array files.
        **kwargs: Forwarded to np.load (e.g., mmap_mode='r').

    Returns:
        The numpy array extracted from the NPZ file.

    Raises:
        ValueError: If array_key is None and file contains multiple arrays,
                   or if the specified array_key is not found in the file.

    Examples:
        # Auto-detection for single-array npz
        arr = _load_npz("data.npz")

        # Explicit key for multi-array npz
        features = _load_npz("data.npz", array_key="features")
        targets = _load_npz("data.npz", array_key="targets")
    """
    npz = np.load(path, **kwargs)
    keys = list(npz.keys())

    if array_key is None:
        # Auto-detect for single array
        if len(keys) == 1:
            return npz[keys[0]]
        raise ValueError(
            f"NPZ file '{path}' contains multiple arrays {keys}. Specify array_key to select one."
        )

    # Use explicit key
    if array_key not in keys:
        raise ValueError(
            f"Array key '{array_key}' not found in NPZ file '{path}'. Available keys: {keys}"
        )
    return npz[array_key]


def _load_torch_array(path: Path | str, **kwargs) -> object:
    """Load a torch-backed array payload with explicit checkpoint semantics."""
    kwargs.setdefault("weights_only", False)
    return torch.load(path, **kwargs)


# Frozen, immutable loader map, typed as a Mapping
_LOADER_MAP: Mapping[str, Callable[..., object]] = MappingProxyType({
    ".npz": _load_npz,
    ".npy": np.load,
    ".txt": np.loadtxt,
    ".csv": np.loadtxt,
    ".pt": _load_torch_array,
    ".pth": _load_torch_array,
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
        path: A FilePath pointing to .npy, .npz, .txt/.csv, or .pt/.pth file.
        dtype: Explicit torch.dtype to convert the loaded data to.
               If None, uses precision service to resolve from context.
        **kwargs: Forwarded to the underlying loader (e.g. np.load or torch.load).
                 For .npz files, pass array_key to select a specific array.

    Returns:
        A torch.Tensor containing the loaded data with appropriate precision.

    Raises:
        ValueError: If `path.suffix` is not one of the supported extensions,
                   or if .npz file requires array_key but none provided.
        TypeError: If the loader returns a type other than np.ndarray or Tensor.

    Examples:
        # Use precision from context (recommended)
        with precision_override(PrecisionStrategy.FULL_64):
            tensor = load_array("data.npy")  # Automatically float64

        # Override with explicit dtype (rare cases)
        tensor = load_array("data.npy", dtype=torch.float16)

        # Load from NPZ file with explicit array key
        features = load_array("data.npz", array_key="features")
        targets = load_array("data.npz", array_key="targets")
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
        path: A FilePath pointing to .npy, .npz, .txt/.csv, or .pt/.pth file.
        **kwargs: Forwarded to the underlying loader (e.g. np.load or torch.load).
                 For .npz files, pass array_key to select a specific array.

    Returns:
        A torch.Tensor loaded with session-consistent precision.
    """
    return load_array(path, dtype=None, **kwargs)
