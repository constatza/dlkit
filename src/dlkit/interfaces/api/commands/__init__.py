"""Command pattern implementation for DLKit API.

This module provides the Command pattern implementation to break down
the monolithic API into focused, single-responsibility command handlers.

Each command encapsulates a specific workflow operation (train, infer, etc.)
with proper dependency injection and error handling.
"""

from .base import BaseCommand
from .dispatcher import CommandDispatcher, get_dispatcher
from .train_command import TrainCommand, TrainCommandInput
from .inference_command import InferenceCommand, InferenceCommandInput
from .optimization_command import OptimizationCommand, OptimizationCommandInput
from .validation_command import ValidationCommand, ValidationCommandInput
from .configuration_command import (
    GenerateTemplateCommand,
    GenerateTemplateCommandInput,
    GenerateTemplateCommandOutput,
    ValidateTemplateCommand,
    ValidateTemplateCommandInput,
    ValidateTemplateCommandOutput,
)

__all__ = [
    "BaseCommand",
    "CommandDispatcher",
    "get_dispatcher",
    "TrainCommand",
    "TrainCommandInput",
    "InferenceCommand",
    "InferenceCommandInput",
    "OptimizationCommand",
    "OptimizationCommandInput",
    "ValidationCommand",
    "ValidationCommandInput",
    "GenerateTemplateCommand",
    "GenerateTemplateCommandInput",
    "GenerateTemplateCommandOutput",
    "ValidateTemplateCommand",
    "ValidateTemplateCommandInput",
    "ValidateTemplateCommandOutput",
]
