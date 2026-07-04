"""DLKit domain analysis sub-package.

Provides pure figure-generation functions and IFigureGenerator protocol
for opt-in ML training visualizations.
"""

from dlkit.domain.analysis.figures import (
    error_histogram_figure,
    loss_curve_figure,
    parity_figure,
    residual_figure,
    residual_vs_index_figure,
)
from dlkit.domain.analysis.generators import (
    ErrorHistogramGenerator,
    ParityGenerator,
    ResidualGenerator,
    ResidualVsIndexGenerator,
)
from dlkit.domain.analysis.protocols import IFigureGenerator

__all__ = [
    "ErrorHistogramGenerator",
    "IFigureGenerator",
    "ParityGenerator",
    "ResidualGenerator",
    "ResidualVsIndexGenerator",
    "error_histogram_figure",
    "loss_curve_figure",
    "parity_figure",
    "residual_figure",
    "residual_vs_index_figure",
]
