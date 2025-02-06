import numpy as np
from pydantic import validate_call, ConfigDict


@validate_call
def linear_interpolation_int(sequence: list[int | float], num_points: int, dtype: type = int) -> np.ndarray:
    x = np.arange(len(sequence))
    x_new = np.linspace(0, len(sequence) - 1, num_points)
    y_new = np.interp(x_new, x, sequence)
    return y_new.astype(dtype)
