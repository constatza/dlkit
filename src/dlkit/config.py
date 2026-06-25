"""User-facing configuration namespace.

Thin re-exports from the internal config and IO layers so users can write::

    from dlkit.config import TrainingJobConfig, load_job

instead of mixing internal module paths.
"""

from dlkit.infrastructure.config import *  # noqa: F403
from dlkit.infrastructure.config import __all__ as _CONFIG_ALL
from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.io.config_loader import (
    load_inference_config_eager as load_inference_config,
)
from dlkit.infrastructure.io.config_loader import (
    load_optimization_config_eager as load_optimization_config,
)
from dlkit.infrastructure.io.config_loader import (
    load_training_config_eager as load_training_config,
)

__all__ = [
    *_CONFIG_ALL,
    "load_job",
    "load_training_config",
    "load_inference_config",
    "load_optimization_config",
]
