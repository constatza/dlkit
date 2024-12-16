from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import validate_call

from tomlkit import dumps


@validate_call(config={"arbitrary_types_allowed": True})
def save_asarray(path: Path, array: np.ndarray | torch.Tensor):
    array = array.numpy().cpu() if isinstance(array, torch.Tensor) else array
    np.save(path.with_suffix(".npy"), array)


@validate_call(config={"arbitrary_types_allowed": True})
def savez_asarray(path: Path, **kwargs: Any):
    kwargs = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
    }
    np.savez(path.with_suffix(".npz"), **kwargs)


@validate_call
def write_toml(config: dict, path: str | Path):
    path = str(path) if isinstance(path, Path) else path
    config["paths"] = {k: str(v) for k, v in config["paths"].items()}

    # Remove hyperparameter optimization keys
    config["optuna"] = {}
    config["pruner"] = {}
    config["sampler"] = {}
    with open(path, "w") as file:
        file.write(dumps(config))
