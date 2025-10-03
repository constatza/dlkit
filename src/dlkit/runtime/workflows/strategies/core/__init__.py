"""Core training execution interfaces and implementations."""

from .interfaces import ITrainingExecutor
from .vanilla_executor import VanillaExecutor

__all__ = [
    "ITrainingExecutor",
    "VanillaExecutor",
]
