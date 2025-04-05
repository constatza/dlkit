from pathlib import Path
from pydantic import validate_call
import numpy as np
import tables


@validate_call
def load_hdf5(path: str, group_name: str = "data"):
    with tables.open_file(path, "r") as f:
        return f.get_node(group_name)


@validate_call
def save_hdf5(path: str, data: np.ndarray, group_name: str = "data"):
    with tables.open_file(path, "w") as f:
        f.create_array(f.root, group_name, data)
