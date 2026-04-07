"""User-facing result and state namespace.

Thin re-exports from ``dlkit.common`` so users can write::

    from dlkit.results import TrainingResult, ModelState

instead of the internal paths::

    from dlkit.common.results import TrainingResult
    from dlkit.common.state import ModelState
"""

from dlkit.common.results import InferenceResult, OptimizationResult, TrainingResult
from dlkit.common.state import ModelState

__all__ = [
    "TrainingResult",
    "InferenceResult",
    "OptimizationResult",
    "ModelState",
]
