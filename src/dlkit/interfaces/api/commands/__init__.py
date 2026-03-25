"""Command pattern implementation for DLKit API.

This module provides the Command pattern implementation to break down
the monolithic API into focused, single-responsibility command handlers.

Each command encapsulates a specific workflow operation (train, infer, etc.)
with proper dependency injection and error handling.
"""

from .base import BaseCommand
from .configuration_command import (
    GenerateTemplateCommand,
    GenerateTemplateCommandInput,
    GenerateTemplateCommandOutput,
    ValidateTemplateCommand,
    ValidateTemplateCommandInput,
    ValidateTemplateCommandOutput,
)
from .dispatcher import CommandDispatcher, get_dispatcher
from .inference_command import InferenceCommand, InferenceCommandInput
from .optimization_command import OptimizationCommand, OptimizationCommandInput
from .train_command import TrainCommand, TrainCommandInput
from .validation_command import ValidationCommand, ValidationCommandInput

_dispatcher = get_dispatcher()
_dispatcher.register_command("train", TrainCommand)
_dispatcher.register_command("optimize", OptimizationCommand)
_dispatcher.register_command("validate_config", ValidationCommand)
_dispatcher.register_command("generate_template", GenerateTemplateCommand)
_dispatcher.register_command("validate_template", ValidateTemplateCommand)

__all__ = [
    "BaseCommand",
    "CommandDispatcher",
    "GenerateTemplateCommand",
    "GenerateTemplateCommandInput",
    "GenerateTemplateCommandOutput",
    "InferenceCommand",
    "InferenceCommandInput",
    "OptimizationCommand",
    "OptimizationCommandInput",
    "TrainCommand",
    "TrainCommandInput",
    "ValidateTemplateCommand",
    "ValidateTemplateCommandInput",
    "ValidateTemplateCommandOutput",
    "ValidationCommand",
    "ValidationCommandInput",
    "get_dispatcher",
]
