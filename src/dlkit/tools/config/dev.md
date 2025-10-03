# Configuration System Module

## Overview
The configuration module provides a comprehensive SOLID-compliant settings system for DLKit with typed TOML configuration management, factory patterns for object construction, precision strategies, hyperparameter sampling, and environment-aware path resolution. It implements separation of concerns between configuration storage (settings classes) and object construction (factories), with strong Pydantic-based validation and type safety throughout.

## Architecture & Design Patterns
- **Factory Pattern**: Separates configuration from object construction through `ComponentFactory` and `FactoryProvider`
- **Builder Pattern**: `BuildContext` provides clean dependency injection for component construction
- **Strategy Pattern**: `PrecisionStrategy` encapsulates precision modes with Lightning compatibility
- **Single Responsibility Principle (SRP)**: Settings classes only hold/validate config; factories handle construction; samplers handle hyperparameter sampling
- **Interface Segregation Principle (ISP)**: Focused protocols like `ISettingsSampler` with minimal contracts
- **Dependency Inversion Principle (DIP)**: Components depend on abstract `ComponentSettings` and `BaseWorkflowSettings`
- **Null Object Pattern**: `NullSettingsSampler` eliminates conditional logic when optimization is disabled
- **Partial Loading**: Efficient section-based config loading to minimize parsing overhead

Key architectural decisions:
- Flattened configuration hierarchy reduces nesting complexity
- Pydantic v2 for validation and serialization with frozen models
- Dynaconf compatibility for migration paths
- Top-level CAPITALS follow dynaconf convention for easy migration
- Settings are immutable (frozen=True) ensuring thread safety

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `GeneralSettings` | Class | Top-level settings with flattened architecture | N/A |
| `SessionSettings` | Class | Session mode control (training/inference) | N/A |
| `TrainingWorkflowSettings` | Class | Training-specific workflow settings | N/A |
| `load_settings()` | Function | Unified settings loading with multiple strategies | `BaseWorkflowSettings` |
| `load_training_settings()` | Function | Load training-optimized settings | `TrainingWorkflowSettings` |
| `load_sections()` | Function | Load arbitrary configuration sections | `BaseWorkflowSettings` |
| `PrecisionStrategy` | Enum | Precision modes with Lightning compatibility | N/A |
| `ComponentFactory` | Abstract Class | Base factory for component construction | N/A |
| `FactoryProvider` | Class | Global factory registry and access point | N/A |
| `BuildContext` | Class | Dependency injection context | N/A |
| `ISettingsSampler` | Protocol | Hyperparameter sampling interface | N/A |
| `OptunaSettingsSampler` | Class | Optuna-specific sampler implementation | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `PartialSettingsLoader` | Class | Factory for efficient partial config loading | Various settings types |
| `DefaultComponentFactory` | Class | Default factory implementation | Component instances |
| `ComponentRegistry` | Class | Registry managing component factories | N/A |
| `DLKitEnvironment` | Class | Environment variable management | N/A |
| `NullSettingsSampler` | Class | No-op sampler when optimization disabled | N/A |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `ISettingsSampler` | `sample()` | Hyperparameter sampling interface following ISP |

### Sub-modules
| Sub-module | Purpose | Key Classes |
|------------|---------|-------------|
| `core/` | Core infrastructure (base classes, factories, context) | `BasicSettings`, `ComponentSettings`, `ComponentFactory`, `BuildContext` |
| `components/` | Component-specific settings (model, loss, metric, wrapper) | `ModelComponentSettings`, `LossComponentSettings`, `MetricComponentSettings` |
| `precision/` | Precision strategy enumeration | `PrecisionStrategy` |
| `samplers/` | Hyperparameter sampling implementations | `ISettingsSampler`, `OptunaSettingsSampler` |

## Dependencies

### Internal Dependencies
- `dlkit.tools.io.config`: TOML configuration loading (`load_config`, `load_sections_config`, `write_config`)
- `dlkit.tools.io.locations`: Standard location resolution
- `dlkit.tools.io.resolution`: Path resolution with context
- `dlkit.tools.utils.general`: Utilities (`kwargs_compatible_with`, `import_object`)
- `dlkit.tools.utils.logging_config`: Logger configuration (`get_logger`)
- `dlkit.tools.registry.resolve`: Component resolution (`resolve_component`)
- `dlkit.core.datatypes.secure_uris`: Secure path handling (`SecurePath`)

### External Dependencies
- `pydantic`: V2 settings and validation (`BaseModel`, `BaseSettings`, `Field`)
- `pydantic_settings`: Environment variable management
- `dynaconf`: Legacy configuration support (`LazySettings`)
- `torch`: PyTorch dtypes for precision strategies
- `pathlib`: Path manipulation

## Key Components

### Component 1: `GeneralSettings`

**Purpose**: Top-level flattened configuration settings for DLKit with SOLID principles and mode separation.

**Parameters**:
- `SESSION: SessionSettings` - Session mode control (training/inference)
- `MODEL: ModelComponentSettings | None` - Model configuration (preferred at top-level)
- `MLFLOW: MLflowSettings | None` - MLflow experiment tracking configuration
- `OPTUNA: OptunaSettings | None` - Optuna hyperparameter optimization configuration
- `DATAMODULE: DataModuleSettings | None` - Data loading and processing configuration
- `DATASET: DatasetSettings | None` - Dataset-specific configuration
- `TRAINING: TrainingSettings | None` - Core training configuration with nested library settings
- `PATHS: PathsSettings | None` - Optional standardized paths with automatic resolution
- `EXTRAS: ExtrasSettings | None` - Optional free-form user-defined helper settings

**Returns**: N/A (Pydantic model)

**Raises**:
- `ValueError`: If inference mode lacks checkpoint path in MODEL section

**Example**:
```python
from dlkit.tools.config import GeneralSettings

# Load from TOML file
settings = GeneralSettings.from_toml_file("config.toml")

# Access flattened sections
if settings.mlflow_enabled:
    print(f"MLflow tracking to: {settings.MLFLOW.tracking_uri}")

if settings.optuna_enabled:
    print(f"Running {settings.OPTUNA.n_trials} trials")

# Check execution mode
if settings.is_training:
    training_config = settings.get_training_config()
    print(f"Max epochs: {training_config.trainer.max_epochs}")
```

**Implementation Notes**:
- Inherits from `BasicSettings` which provides immutable frozen models
- Validates inference mode requires checkpoint path
- Provides convenience properties (`mlflow_enabled`, `optuna_enabled`, `has_training_config`)
- Supports Dynaconf migration via `dynaconf_to_settings()` classmethod
- All settings sections are optional except SESSION (has defaults)
- Top-level MODEL preferred over nested model configurations

---

### Component 2: `load_settings()`

**Purpose**: Unified settings loading function with multiple loading strategies for optimal performance.

**Parameters**:
- `config_path: Path | str` - Path to TOML configuration file
- `inference: bool = False` - If True, load for inference mode; if False, load for training mode
- `sections: list[str] | None = None` - Specific sections to load (overrides inference if provided)
- `strict: bool = False` - If True with sections, all specified sections must exist

**Returns**: `BaseWorkflowSettings` - Appropriate settings instance for the request

**Raises**:
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If inference mode is requested (removed feature)
- `ConfigSectionError`: If required sections are missing
- `ConfigValidationError`: If validation fails

**Example**:
```python
from dlkit.tools.config import load_settings

# Strategy 1: Mode-optimized loading (recommended)
settings = load_settings("config.toml", inference=False)  # training

# Strategy 2: Custom section loading (flexible)
settings = load_settings("config.toml", sections=["MODEL", "DATASET"])

# Strategy 3: Strict loading (all sections must exist)
settings = load_settings(
    "config.toml",
    sections=["MODEL", "DATASET", "TRAINING"],
    strict=True
)
```

**Implementation Notes**:
- Delegates to `default_settings_loader` (global PartialSettingsLoader instance)
- Inference mode removed - use `dlkit.interfaces.inference.infer` instead
- Custom section loading provides maximum flexibility
- Training mode is the default behavior

---

### Component 3: `PrecisionStrategy`

**Purpose**: Precision strategies with direct PyTorch Lightning Trainer compatibility and torch dtype conversion.

**Enum Values**:
- `FULL_64 = "64"` - Double precision (64-bit) for maximum numerical accuracy
- `FULL_32 = "32"` - Full precision (32-bit) - default for safety
- `MIXED_16 = "16-mixed"` - Mixed precision (16-bit/32-bit) for memory efficiency
- `TRUE_16 = "16"` - True half precision (16-bit) for maximum memory savings
- `MIXED_BF16 = "bf16-mixed"` - Mixed bfloat16 precision for improved gradient stability
- `TRUE_BF16 = "bf16"` - True bfloat16 precision for memory efficiency

**Key Methods**:
- `to_lightning_precision() -> str | int` - Convert to Lightning Trainer precision parameter
- `to_torch_dtype() -> torch.dtype` - Convert to torch.dtype for model weights
- `get_compute_dtype() -> torch.dtype` - Get dtype used for computation and gradients
- `supports_autocast() -> bool` - Check if strategy uses automatic mixed precision
- `is_reduced_precision() -> bool` - Check if strategy uses < 32-bit precision
- `get_memory_factor() -> float` - Estimate memory usage relative to FULL_32
- `from_lightning_precision(precision) -> PrecisionStrategy` - Create from Lightning precision value
- `get_default() -> PrecisionStrategy` - Get default strategy (FULL_32)

**Example**:
```python
from dlkit.tools.config.precision import PrecisionStrategy

# Use in configuration
strategy = PrecisionStrategy.MIXED_16

# Convert to Lightning format
lightning_precision = strategy.to_lightning_precision()  # "16-mixed"

# Get torch dtype
model_dtype = strategy.to_torch_dtype()  # torch.float16

# Check capabilities
if strategy.supports_autocast():
    print("Using automatic mixed precision")

# Memory estimation
memory_factor = strategy.get_memory_factor()  # 0.7 (30% savings)

# Create from Lightning precision
strategy = PrecisionStrategy.from_lightning_precision("bf16-mixed")
```

**Implementation Notes**:
- Inherits from `StrEnum` for JSON/TOML serialization compatibility
- Provides bidirectional conversion with PyTorch Lightning
- Mixed precision strategies return lower precision dtype but compute in float32
- Memory factors are approximations for planning purposes

---

### Component 4: `ComponentFactory` and `FactoryProvider`

**Purpose**: Abstract factory pattern for creating components from settings, separating configuration from construction logic.

**ComponentFactory (Abstract Base)**:
- `create(settings: ComponentSettings[T], context: BuildContext) -> T` - Create component from settings

**DefaultComponentFactory**:
- Handles common case: resolve class reference and construct with compatible kwargs
- Supports string class references, direct class/callable references, and pre-constructed instances
- Filters kwargs for constructor compatibility using `kwargs_compatible_with`
- Integrates with component registry for custom resolution

**FactoryProvider (Singleton)**:
- `get_registry() -> ComponentRegistry` - Get global component registry
- `create_component(settings, context) -> Any` - Create component using global registry
- `register_factory(settings_type, factory)` - Register custom factory

**Example**:
```python
from dlkit.tools.config import FactoryProvider, BuildContext
from dlkit.tools.config.components import ModelComponentSettings

# Create build context
context = BuildContext(
    mode="training",
    device="cuda",
    random_seed=42
)

# Create component using global factory
model_settings = ModelComponentSettings(
    name="dlkit.core.models.nn.ffnn.SimpleFFNN",
    hidden_size=128,
    num_layers=3
)

model = FactoryProvider.create_component(model_settings, context)

# Register custom factory for specialized construction
from dlkit.tools.config.core import ComponentFactory

class CustomModelFactory(ComponentFactory):
    def create(self, settings, context):
        # Custom construction logic
        pass

FactoryProvider.register_factory(ModelComponentSettings, CustomModelFactory())
```

**Implementation Notes**:
- Follows Open/Closed Principle - extensible via registration
- `DefaultComponentFactory` handles 90% of use cases
- Supports dynamic import from module paths
- Integrates with user registry for custom component resolution
- Filters kwargs to match constructor signature
- Handles both classes and callable factories

---

### Component 5: `BuildContext`

**Purpose**: Context object for passing dependencies during object construction via dependency injection.

**Parameters**:
- `mode: str` - Execution mode (training, inference, etc.)
- `device: str = "auto"` - Target device for computation
- `random_seed: int | None = None` - Random seed for reproducibility
- `working_directory: Path` - Current working directory (defaults to CWD)
- `checkpoint_path: Path | None = None` - Path to model checkpoint if needed
- `overrides: dict[str, Any]` - Additional keyword arguments for construction

**Key Methods**:
- `with_overrides(**kwargs) -> BuildContext` - Create new context with additional overrides
- `get_override(key, default=None) -> Any` - Get an override value by key

**Example**:
```python
from dlkit.tools.config import BuildContext
from pathlib import Path

# Create base context
context = BuildContext(
    mode="training",
    device="cuda:0",
    random_seed=42,
    working_directory=Path("/path/to/project")
)

# Add overrides for component construction
model_context = context.with_overrides(
    input_dim=784,
    output_dim=10
)

# Access override values
input_dim = model_context.get_override("input_dim", default=512)
```

**Implementation Notes**:
- Immutable pattern - `with_overrides` creates new instance
- Pydantic model with `arbitrary_types_allowed` for Path objects
- Replaces complex parameter passing in old build() methods
- Enables clean dependency injection throughout construction pipeline

---

### Component 6: `ISettingsSampler` and `OptunaSettingsSampler`

**Purpose**: Hyperparameter sampling following Interface Segregation Principle. Converts trial suggestions into complete GeneralSettings.

**ISettingsSampler Protocol**:
- `sample(trial: Any, base_settings: GeneralSettings) -> GeneralSettings` - Sample hyperparameters

**OptunaSettingsSampler**:
- `__init__(optuna_settings: OptunaSettings)` - Initialize with Optuna configuration
- `sample(trial, base_settings) -> GeneralSettings` - Sample from OPTUNA.model ranges

**NullSettingsSampler**:
- Returns settings unchanged when optimization is disabled
- Follows Null Object Pattern

**Example**:
```python
from dlkit.tools.config.samplers import OptunaSettingsSampler, create_settings_sampler
from dlkit.tools.config import GeneralSettings
import optuna

# Load base settings
base_settings = GeneralSettings.from_toml_file("config.toml")

# Create sampler
sampler = create_settings_sampler(base_settings.OPTUNA)

# In Optuna objective function
def objective(trial):
    # Sample hyperparameters
    sampled_settings = sampler.sample(trial, base_settings)

    # Use sampled settings for training
    result = train_model(sampled_settings)
    return result.loss

# Null sampler when optimization disabled
null_sampler = create_settings_sampler(None)
unchanged = null_sampler.sample(trial, base_settings)  # Returns base_settings
```

**Implementation Notes**:
- Follows Single Responsibility Principle - only handles sampling
- Settings classes no longer have `sample()` method (SRP separation)
- Supports integer, float, and categorical hyperparameter ranges
- Deep merges sampled parameters into MODEL section
- Gracefully handles missing or invalid configurations
- Factory function `create_settings_sampler()` returns appropriate implementation

---

### Component 7: `PartialSettingsLoader`

**Purpose**: Factory class for efficient partial config loading with minimal parsing overhead.

**Key Methods**:
- `load_training_settings(config_path) -> TrainingWorkflowSettings` - Load training-optimized settings
- `load_sections(config_path, sections, strict=False) -> BaseWorkflowSettings` - Load arbitrary sections
- `load_custom_settings(config_path, settings_class, required_sections, optional_sections) -> T` - Load into custom settings class
- `create_settings_for_workflow(config_path, workflow_type) -> BaseWorkflowSettings` - Factory method by workflow type

**Example**:
```python
from dlkit.tools.config.factories import PartialSettingsLoader

loader = PartialSettingsLoader()

# Training-optimized loading (fastest)
training_settings = loader.load_training_settings("config.toml")
# Loads: SESSION, DATAMODULE, DATASET, TRAINING (required)
#        MODEL, MLFLOW, OPTUNA, PATHS, EXTRAS (optional)

# Flexible section loading
eval_settings = loader.load_sections(
    "config.toml",
    sections=["MODEL", "DATASET"],
    strict=False  # ignore missing sections
)

# Custom settings class
from dlkit.tools.config.workflow_settings import BaseWorkflowSettings

class CustomSettings(BaseWorkflowSettings):
    pass

custom = loader.load_custom_settings(
    "config.toml",
    CustomSettings,
    required_sections=["MODEL", "DATASET"],
    optional_sections=["PATHS"]
)
```

**Implementation Notes**:
- Implements Factory Method pattern for workflow-specific loading
- Only parses requested sections, minimizing overhead
- Strict mode enforces section existence checking
- Validates custom settings classes inherit from `BaseWorkflowSettings`
- Chooses appropriate settings class based on loaded sections

---

### Component 8: `DLKitEnvironment`

**Purpose**: Environment-aware root configuration following SRP. Manages only environment-level configuration affecting the entire DLKit system.

**Parameters**:
- `root_dir: SecurePath | None = None` - Root directory for relative path resolution
- `internal_dir: str = ".dlkit"` - Directory for DLKit internal artifacts
- `log_filename: str = "dlkit.log"` - Default log file name
- `server_tracking_file: str = "servers.json"` - Server tracking file name

**Key Methods**:
- `get_root_path() -> Path` - Get effective root directory (CWD if root_dir is None)
- `create_resolver_context() -> ResolverContext` - Create path resolver context
- `get_internal_dir_path() -> Path` - Get path to DLKit internal directory
- `get_log_file_path() -> Path` - Get path to default log file
- `get_server_tracking_path() -> Path` - Get path to server tracking file (in user home)

**Example**:
```python
from dlkit.tools.config.environment import DLKitEnvironment
import os

# Override via environment variable
os.environ["DLKIT_ROOT_DIR"] = "/custom/root"

env = DLKitEnvironment()

# Get effective root
root = env.get_root_path()  # Path("/custom/root")

# Get internal paths
internal_dir = env.get_internal_dir_path()  # /custom/root/.dlkit/
log_file = env.get_log_file_path()  # /custom/root/.dlkit/dlkit.log
server_tracking = env.get_server_tracking_path()  # ~/.dlkit/servers.json

# Create resolver context
resolver = env.create_resolver_context()
```

**Implementation Notes**:
- Reads from environment variables with `DLKIT_` prefix
- Supports `.env` file loading
- SecurePath handles tilde expansion and security checks
- Server tracking always in user home for global access
- Creates directories as needed when getting paths

## Usage Patterns

### Common Use Case 1: Loading Settings for Training
```python
from dlkit.tools.config import load_settings

# Load complete training configuration
settings = load_settings("config.toml")

# Access flattened configuration
print(f"Model: {settings.MODEL.name}")
print(f"Max epochs: {settings.TRAINING.trainer.max_epochs}")
print(f"Batch size: {settings.DATAMODULE.dataloader.batch_size}")

# Check optional features
if settings.mlflow_enabled:
    print(f"Tracking to: {settings.MLFLOW.tracking_uri}")

if settings.optuna_enabled:
    print(f"Running {settings.OPTUNA.n_trials} optimization trials")
```

### Common Use Case 2: Partial Section Loading
```python
from dlkit.tools.config import load_sections

# Load only what you need for evaluation
eval_settings = load_sections("config.toml", ["MODEL", "DATASET"])

# Model and dataset are available
model_config = eval_settings.MODEL
dataset_config = eval_settings.DATASET

# Other sections are None
assert eval_settings.TRAINING is None
assert eval_settings.MLFLOW is None
```

### Common Use Case 3: Component Construction with Factory
```python
from dlkit.tools.config import FactoryProvider, BuildContext
from dlkit.tools.config.components import ModelComponentSettings

# Define model configuration
model_settings = ModelComponentSettings(
    name="dlkit.core.models.nn.ffnn.SimpleFFNN",
    hidden_size=256,
    num_layers=4,
    activation="relu"
)

# Create build context
context = BuildContext(
    mode="training",
    device="cuda",
    random_seed=42
)

# Construct model using factory
model = FactoryProvider.create_component(model_settings, context)

# Model is ready to use
output = model(input_tensor)
```

### Common Use Case 4: Hyperparameter Optimization with Optuna
```python
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.samplers import create_settings_sampler
import optuna

# Load base configuration
base_settings = GeneralSettings.from_toml_file("config.toml")

# Create sampler from OPTUNA section
sampler = create_settings_sampler(base_settings.OPTUNA)

# Define Optuna objective
def objective(trial):
    # Sample hyperparameters
    trial_settings = sampler.sample(trial, base_settings)

    # Train with sampled settings
    result = train_model(trial_settings)

    return result.val_loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=base_settings.OPTUNA.n_trials)
```

### Common Use Case 5: Precision Strategy Management
```python
from dlkit.tools.config.precision import PrecisionStrategy
import torch

# Select precision strategy
strategy = PrecisionStrategy.MIXED_16

# Configure PyTorch Lightning Trainer
trainer_kwargs = {
    "precision": strategy.to_lightning_precision(),  # "16-mixed"
    "max_epochs": 100
}

# Get torch dtype for manual operations
model_dtype = strategy.to_torch_dtype()  # torch.float16
model = model.to(dtype=model_dtype)

# Check memory savings
memory_savings = (1.0 - strategy.get_memory_factor()) * 100
print(f"Estimated memory savings: {memory_savings}%")  # ~30%
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: Invalid configuration values or missing required sections
- `FileNotFoundError`: Config file doesn't exist
- `ConfigSectionError`: Required sections missing from config file
- `ConfigValidationError`: Pydantic validation fails
- `TypeError`: Incorrect types during component construction

**Error Handling Pattern**:
```python
from dlkit.tools.config import load_settings

try:
    settings = load_settings("config.toml")
except FileNotFoundError:
    print("Config file not found")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

### Test Coverage
- Unit tests: `tests/tools/config/test_*.py`
- Precision tests: `tests/tools/config/precision/test_*.py`
- Integration tests: `tests/tools/config/precision/test_integration.py`
- Edge case tests: `tests/tools/config/precision/test_edge_cases.py`

### Key Test Scenarios
1. **Settings loading**: TOML parsing, validation, section loading
2. **Factory pattern**: Component construction, kwargs filtering, registry
3. **Precision strategies**: Lightning conversion, dtype mapping, memory factors
4. **Hyperparameter sampling**: Optuna integration, range specifications, deep merge
5. **Environment management**: Path resolution, environment variable override
6. **Partial loading**: Efficient section-based loading, strict mode

### Fixtures Used
- `tmp_path` (pytest built-in): Temporary paths for config files
- Custom fixtures in `tests/tools/config/precision/conftest.py`

## Performance Considerations
- Partial loading minimizes TOML parsing overhead
- Frozen Pydantic models enable caching and thread safety
- Lazy factory resolution defers imports until construction
- Section-based loading avoids parsing unused configuration
- ComponentRegistry uses dict lookup for O(1) factory retrieval

## Future Improvements / TODOs
- [ ] YAML configuration file support
- [ ] JSON schema generation from settings classes
- [ ] Configuration diff/merge utilities
- [ ] Hot reload for development workflows
- [ ] Configuration templates/presets
- [ ] Validation of cross-section dependencies
- [ ] Environment-specific overlays (dev/staging/prod)

## Related Modules
- `dlkit.tools.io.config`: Low-level TOML loading and writing
- `dlkit.tools.io.resolution`: Path resolution with context
- `dlkit.tools.registry`: Component registry for custom resolution
- `dlkit.runtime.workflows.factories`: Uses settings to build workflow components
- `dlkit.interfaces.api`: API layer uses settings for workflow execution

## Change Log
- **2025-10-03**: Comprehensive documentation created
- **2024-10-02**: Flattened architecture migration completed
- **2024-09-30**: Factory pattern implementation (removed build() methods)
- **2024-09-28**: Precision strategy enum with Lightning compatibility
- **2024-09-24**: Hyperparameter sampling separated into sampler classes (SRP)
