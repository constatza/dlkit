import numpy as np
from pydantic import validate_call


@validate_call
def linear_interpolation_int(
    sequence: list[int | float], num_points: int, dtype: type = int
) -> np.ndarray:
    x = np.arange(len(sequence))
    x_new = np.linspace(0, len(sequence) - 1, num_points)
    y_new = np.interp(x_new, x, sequence)
    return y_new.astype(dtype)


def interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Perform piecewise linear interpolation for values within the data range,
    and linear extrapolation for values outside the data range.

    For x values below xp[0], extrapolation is done using the slope between xp[0] and xp[1].
    For x values above xp[-1], extrapolation is done using the slope between xp[-2] and xp[-1].
    For x values within [xp[0], xp[-1]], standard linear interpolation is performed.

    Args:
        x (np.ndarray): 1-D array of points at which to evaluate the function.
        xp (np.ndarray): 1-D array of known x-values (must be in ascending order).
        fp (np.ndarray): 1-D array of known function values corresponding to xp.

    Returns:
        np.ndarray: Array of interpolated or extrapolated values at points x.
    """
    # Start with standard interpolation for in-range values.
    y = np.interp(x, xp, fp)

    # Extrapolate for values below the range.
    mask_lower = x < xp[0]
    if np.any(mask_lower):
        slope_lower = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[mask_lower] = fp[0] + slope_lower * (x[mask_lower] - xp[0])

    # Extrapolate for values above the range.
    mask_upper = x > xp[-1]
    if np.any(mask_upper):
        slope_upper = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[mask_upper] = fp[-1] + slope_upper * (x[mask_upper] - xp[-1])

    return y
