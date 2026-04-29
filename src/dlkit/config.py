"""User-facing configuration namespace.

Thin re-exports from the internal config and IO layers so users can write::

    from dlkit.config import GeneralSettings, load_training_config

instead of mixing internal module paths.
"""

from dlkit.infrastructure.config import *  # noqa: F403
from dlkit.infrastructure.config import __all__ as _CONFIG_ALL
from dlkit.infrastructure.io.config import (
    load_inference_config_eager,
    load_optimization_config_eager,
    load_training_config_eager,
)

load_training_config = load_training_config_eager
load_inference_config = load_inference_config_eager
load_optimization_config = load_optimization_config_eager

__all__ = [
    *_CONFIG_ALL,
    "load_training_config",
    "load_inference_config",
    "load_optimization_config",
]
