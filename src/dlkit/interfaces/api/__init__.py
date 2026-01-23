"""DLKit API package - Public interface for training, optimization, and inference."""

from __future__ import annotations

# Import command infrastructure
from .commands import (
    get_dispatcher,
    TrainCommand,
    InferenceCommand,
    OptimizationCommand,
    ValidationCommand,
    GenerateTemplateCommand,
    ValidateTemplateCommand,
    TrainCommandInput,
    InferenceCommandInput,
    OptimizationCommandInput,
    ValidationCommandInput,
    GenerateTemplateCommandInput,
    ValidateTemplateCommandInput,
)

# Import domain models and errors
from .domain import (
    ConfigurationError,
    InferenceResult,
    OptimizationResult,
    StrategyError,
    TrainingResult,
    WorkflowError,
)

# Import configuration settings
from dlkit.tools.config import GeneralSettings

# Import API functions from dedicated modules
from .functions import (
    # Core workflow functions
    train,
    optimize,
    # Configuration functions
    validate_config,
    generate_template,
    validate_template,
    # Unified execution function
    execute,
)

# NEW INFERENCE API: Stateful predictors
from dlkit.interfaces.inference import (
    load_model,
    validate_checkpoint,
    get_checkpoint_info,
)

# Initialize command dispatcher with all available commands
_dispatcher = get_dispatcher()
_dispatcher.register_command("train", TrainCommand)
_dispatcher.register_command("optimize", OptimizationCommand)
_dispatcher.register_command("validate_config", ValidationCommand)
_dispatcher.register_command("generate_template", GenerateTemplateCommand)
_dispatcher.register_command("validate_template", ValidateTemplateCommand)

# Export all public interfaces
__all__ = [
    # Main workflow functions
    "execute",  # Unified intelligent workflow function
    "train",
    "load_model",  # NEW: Primary inference API
    "optimize",
    "validate_config",
    # Inference utilities
    "validate_checkpoint",
    "get_checkpoint_info",
    # Configuration functions
    "generate_template",
    "validate_template",
    # Domain models
    "TrainingResult",
    "InferenceResult",
    "OptimizationResult",
    # Error types
    "WorkflowError",
    "ConfigurationError",
    "StrategyError",
    "GenerateTemplateCommandInput",
    "OptimizationCommandInput",
    "TrainCommandInput",
    "ValidateTemplateCommandInput",
    "ValidationCommandInput",
    "GeneralSettings",
    "InferenceCommand",
    "InferenceCommandInput",
]
