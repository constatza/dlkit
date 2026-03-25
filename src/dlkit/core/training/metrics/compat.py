"""Compatibility layer delegating standard metrics to external torchmetrics library.

This module provides aliases to standard torchmetrics implementations for
common metrics like MSE, MAE, RMSE, etc. We delegate to the external library
instead of reimplementing to:
    - Avoid code duplication
    - Leverage well-tested, optimized implementations
    - Stay compatible with torchmetrics ecosystem
    - Focus our custom implementations on specialized metrics

For custom/exotic metrics (e.g., normalized vector norm error, temporal derivatives),
see torchmetrics_wrappers.py instead.
"""

# Standard regression metrics from torchmetrics
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
)

# Can add more as needed
# from torchmetrics.regression import ...

__all__ = [
    # Regression metrics
    "MeanSquaredError",
    "MeanAbsoluteError",
    "MeanSquaredLogError",
    "MeanAbsolutePercentageError",
    "R2Score",
]
