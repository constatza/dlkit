import json
from pydantic import FilePath
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import torch
from dlkit.datasets.split import split_indices


def load_dataset(
    features_path: FilePath, targets_path: Optional[FilePath] = None
) -> [torch.Tensor, torch.Tensor]:
    features = np.load(features_path)
    features_tensor = torch.from_numpy(features).float()
    if targets_path:
        targets = np.load(targets_path)
        targets_tensor = torch.from_numpy(targets).float()
    else:
        targets_tensor = features_tensor

    return features_tensor, targets_tensor


def split_or_load_indices(
    indices_path: Optional[FilePath] = None,
    size: Optional[int] = None,
    test_size: float = 0.3,
    val_size: float = 0.5,
):
    if indices_path:
        return get_idx_split_from_file(indices_path)
    if size:
        return generate_idx_split_dict(size, test_size, val_size)

    raise ValueError("indices_path or size must be provided")


def get_idx_split_from_file(indices_path: FilePath) -> Dict[str, List[int]]:
    with open(indices_path, "r", encoding="utf-8") as f:
        saved_indices: Dict[str, List[int]] = json.load(f)
    return saved_indices


def generate_idx_split_dict(
    size: int, test_size: float, val_size: float
) -> Dict[str, List[int]]:
    all_ids = list(range(size))
    train_idx, val_idx, test_idx = split_indices(
        all_ids, test_size=test_size, val_size=val_size
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}
