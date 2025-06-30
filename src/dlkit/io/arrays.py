import numpy as np
import torch
from pathlib import Path
from pydantic import FilePath, validate_call, ConfigDict
from torch import Tensor
from collections.abc import Mapping
from types import MappingProxyType
from collections.abc import Callable

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
def load_array(path: FilePath, dtype: torch.dtype = torch.float32, **kwargs) -> Tensor:
    """Load an array or tensor from disk, inferring loader by file suffix.

    Args:
        path: A FilePath pointing to .npy, .txt/.csv, or .pt/.pth file.
        dtype: A torch.dtype to convert the loaded data to.
        **kwargs: Forwarded to the underlying loader (e.g. np.load or torch.load).

    Returns:
        A torch.Tensor containing the loaded data.

    Raises:
        ValueError: If `path.suffix` is not one of the supported extensions.
        TypeError: If the loader returns a type other than np.ndarray or Tensor.
    """
    suffix = Path(path).suffix.lower()
    loader = _LOADER_MAP.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file format: {suffix!r}")

    data = loader(path, **kwargs)
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype)
    if isinstance(data, Tensor):
        return data.to(dtype=dtype)

    raise TypeError(f"Loader for {suffix!r} returned unexpected type: {type(data).__name__}")
