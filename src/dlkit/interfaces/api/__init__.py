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
    infer,  # BREAKING CHANGE: Now inference only
    predict_with_config,  # NEW: Lightning-based simple prediction
    optimize,
    # Configuration functions
    validate_config,
    generate_template,
    validate_template,
    # Unified execution function
    execute,
)

# BREAKING CHANGE: Import new inference API
from dlkit.interfaces.inference import (
    infer,
    predict,
    InferenceInput,
    InferenceConfig,
)

# Initialize command dispatcher with all available commands
_dispatcher = get_dispatcher()
_dispatcher.register_command("train", TrainCommand)
_dispatcher.register_command("infer", InferenceCommand)
_dispatcher.register_command("optimize", OptimizationCommand)
_dispatcher.register_command("validate_config", ValidationCommand)
_dispatcher.register_command("generate_template", GenerateTemplateCommand)
_dispatcher.register_command("validate_template", ValidateTemplateCommand)

# Export all public interfaces
__all__ = [
    # Main workflow functions
    "execute",  # Unified intelligent workflow function
    "train",
    "infer",  # BREAKING CHANGE: Now inference only
    "predict_with_config",  # NEW: Lightning-based simple prediction
    "optimize",
    "validate_config",
    # NEW: Inference API
    "predict",
    "InferenceInput",
    "InferenceConfig",
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
    "InferenceCommandInput",  # Legacy - may be removed
    "OptimizationCommandInput",
    "TrainCommandInput",
    "ValidateTemplateCommandInput",
    "ValidationCommandInput",
    "GeneralSettings",
]
