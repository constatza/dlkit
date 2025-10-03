# Build Factory Module

## Overview
The build factory module provides component construction for DLKit, implementing a strategy pattern to build models, datamodules, and trainers from configuration. It supports multiple dataset families (flexible, graph, timeseries) with automatic detection, shape inference, and wrapper creation. The module replaces legacy build flows with a SOLID-compliant factory system.

## Architecture & Design Patterns
- **Strategy Pattern**: Pluggable build strategies for different dataset families
- **Factory Pattern**: Component creation abstracted behind factory interface
- **Chain of Responsibility**: Model type detection via detector chain
- **Dependency Inversion**: Depends on settings abstractions, not concrete implementations
- **Template Method**: Common build flow with family-specific customizations
- **Builder Pattern**: Incremental component construction with overrides
- **Single Responsibility**: Each strategy handles one dataset family

Key architectural decisions:
- ABC-based model detection replaces isinstance checks
- Shape inference integrated for shape-aware models
- Entry registry provides user access to data entries
- Split caching via split provider for reproducibility
- Wrapper factory creates family-appropriate wrappers
- Trainer root directory defaults to standard location
- Checkpoint saving disabled for optimization trials

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `BuildFactory` | Class | Main factory for component construction | N/A |
| `BuildComponents` | Dataclass | Container for built components | N/A |
| `IBuildStrategy` | Protocol | Build strategy interface | N/A |
| `FlexibleBuildStrategy` | Class | Strategy for flexible/array datasets | N/A |
| `GraphBuildStrategy` | Class | Strategy for graph datasets | N/A |
| `TimeSeriesBuildStrategy` | Class | Strategy for timeseries datasets | N/A |

### Model Detection
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `ModelType` | Enum | Model type classifications | N/A |
| `IModelTypeDetector` | Protocol | Model type detector interface | N/A |
| `ABCModelTypeDetector` | Class | ABC-based model detection | N/A |
| `ModelTypeDetectionChain` | Class | Chain of responsibility for detection | N/A |
| `detect_model_type` | Function | Detect model type from settings | `ModelType` |
| `requires_shape_spec` | Function | Check if model needs shape spec | `bool` |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_get_model_class` | Method | Extract model class from settings | `Type \| None` |

## Dependencies

### Internal Dependencies
- `dlkit.tools.config`: Settings management (`GeneralSettings`, `BuildContext`)
- `dlkit.tools.config.core.factories`: Factory provider
- `dlkit.core.datatypes.split`: Split management (`IndexSplit`)
- `dlkit.core.shape_specs`: Shape inference system
- `dlkit.core.models.wrappers.factories`: Wrapper factory
- `dlkit.runtime.workflows.selectors.defaults`: Family defaults
- `dlkit.runtime.workflows.entry_registry`: Data entry registry
- `dlkit.tools.io.split_provider`: Split caching
- `dlkit.tools.io.locations`: Standard paths

### External Dependencies
- `lightning.pytorch`: PyTorch Lightning (`LightningModule`, `LightningDataModule`, `Trainer`)
- `dataclasses`: Data containers
- `pathlib`: Path handling
- `enum`: Enumerations

## Key Components

### Component 1: `BuildFactory`

**Purpose**: Main factory that selects appropriate build strategy based on dataset type and constructs components.

**Constructor Parameters**:
- `strategies: list[IBuildStrategy] | None` - Custom strategies (default: Graph, Timeseries, Flexible)

**Key Methods**:
- `build_components(settings: GeneralSettings) -> BuildComponents` - Build all runtime components

**Returns**: `BuildComponents` with model, datamodule, trainer, shape_spec, and metadata

**Example**:
```python
from dlkit.runtime.workflows.factories import BuildFactory
from dlkit.tools.config import GeneralSettings

# Load configuration
settings = GeneralSettings.from_toml("config.toml")

# Create factory
factory = BuildFactory()

# Build all components
components = factory.build_components(settings)

# Access components
model = components.model  # LightningModule
datamodule = components.datamodule  # LightningDataModule
trainer = components.trainer  # Trainer or None
shape_spec = components.shape_spec  # IShapeSpec or None
meta = components.meta  # {"dataset_type": "flexible"}
```

**Implementation Notes**:
- Strategies tried in order: Graph → Timeseries → Flexible (fallback)
- First matching strategy builds components
- Flexible strategy always returns True (fallback)
- Extensible via custom strategies

---

### Component 2: `BuildComponents`

**Purpose**: Immutable container for built runtime components with metadata.

**Properties**:
- `model: LightningModule` - Wrapped or unwrapped model
- `datamodule: LightningDataModule` - Data module with dataset and split
- `trainer: Trainer | None` - Trainer (None for inference mode)
- `shape_spec: IShapeSpec | None` - Shape specification (None for shape-agnostic)
- `meta: dict[str, Any]` - Metadata (dataset_type, etc.)

**Example**:
```python
from dlkit.runtime.workflows.factories import BuildComponents

# Components are frozen dataclasses
components = BuildComponents(
    model=model,
    datamodule=datamodule,
    trainer=trainer,
    shape_spec=shape_spec,
    meta={"dataset_type": "flexible"},
)

# Immutable - cannot modify
# components.model = other_model  # Raises error
```

---

### Component 3: `FlexibleBuildStrategy`

**Purpose**: Default build strategy for flexible array-like datasets (numpy, torch tensors, etc.).

**Key Methods**:
- `can_handle(settings: GeneralSettings) -> bool` - Check if dataset is flexible type
- `build(settings: GeneralSettings) -> BuildComponents` - Build components for flexible dataset

**Returns**: `BuildComponents` with flexible dataset family

**Example**:
```python
from dlkit.runtime.workflows.factories import FlexibleBuildStrategy

strategy = FlexibleBuildStrategy()

# Can handle flexible or unspecified datasets
assert strategy.can_handle(settings)  # True for flexible datasets

# Build components
components = strategy.build(settings)
assert components.meta["dataset_type"] == "flexible"
```

**Implementation Notes**:
- Handles legacy `SupervisedArrayDataset` x/y parameters
- Translates x/y to flexible features/targets
- Uses `FlexibleDataset` for entries-based datasets
- Registers entry configs for transform-aware pipelines
- Infers shapes for shape-aware models
- Validates shape inference results (raises for NullShapeSpec)
- Uses `WrapperFactory.create_standard_wrapper()`
- Sets trainer default_root_dir to standard location

---

### Component 4: `GraphBuildStrategy`

**Purpose**: Build strategy for graph (PyG) datasets and models.

**Key Methods**:
- `can_handle(settings: GeneralSettings) -> bool` - Check if dataset/model is graph type
- `build(settings: GeneralSettings) -> BuildComponents` - Build components for graph dataset

**Returns**: `BuildComponents` with graph dataset family

**Example**:
```python
from dlkit.runtime.workflows.factories import GraphBuildStrategy

strategy = GraphBuildStrategy()

# Detects graph datasets by type or name keywords
assert strategy.can_handle(graph_settings)  # True for graph datasets

# Build components
components = strategy.build(graph_settings)
assert components.meta["dataset_type"] == "graph"
```

**Implementation Notes**:
- Detects via explicit `DATASET.type = "graph"` or name keywords (graph, pyg, geometric)
- Allows `features_path` auxiliary key for simple file paths
- Defaults to graph-aware datamodule when not specified
- Uses graph shape inference engine
- Uses `WrapperFactory.create_graph_wrapper()`
- Entry configs not used by default for graphs

---

### Component 5: `TimeSeriesBuildStrategy`

**Purpose**: Build strategy for time series / forecasting datasets and models (PyTorch Forecasting, etc.).

**Key Methods**:
- `can_handle(settings: GeneralSettings) -> bool` - Check if dataset/model is timeseries type
- `build(settings: GeneralSettings) -> BuildComponents` - Build components for timeseries dataset

**Returns**: `BuildComponents` with timeseries dataset family

**Example**:
```python
from dlkit.runtime.workflows.factories import TimeSeriesBuildStrategy

strategy = TimeSeriesBuildStrategy()

# Detects timeseries datasets by type or name keywords
assert strategy.can_handle(timeseries_settings)  # True for timeseries

# Build components
components = strategy.build(timeseries_settings)
assert components.meta["dataset_type"] == "timeseries"
```

**Implementation Notes**:
- Detects via explicit `DATASET.type = "timeseries"` or name keywords (timeseries, forecast)
- Allows `features_path` auxiliary key for simple file paths
- Defaults to timeseries-aware datamodule when not specified
- Skips wrapper for PyTorch Forecasting models (inherit LightningModule)
- Uses `WrapperFactory.create_timeseries_wrapper()` for DLKit models
- Detects model type using ABC-based detection
- Entry configs not used by default for timeseries

---

### Component 6: `ModelType` and Detection

**Purpose**: ABC-based model detection system replacing hardcoded isinstance checks.

**ModelType Enum**:
- `SHAPE_AWARE_DLKIT` - DLKit models requiring shape specs
- `SHAPE_AGNOSTIC_EXTERNAL` - External models without shape requirements
- `GRAPH` - Graph neural network models
- `TIMESERIES` - Time series / forecasting models

**Detection Functions**:
- `detect_model_type(model_settings, settings) -> ModelType` - Detect model type
- `requires_shape_spec(model_type: ModelType) -> bool` - Check if shape spec needed

**Example**:
```python
from dlkit.runtime.workflows.factories.model_detection import (
    detect_model_type,
    requires_shape_spec,
    ModelType,
)

# Detect model type
model_type = detect_model_type(settings.MODEL, settings)

# Check if shape spec required
if requires_shape_spec(model_type):
    shape_spec = inference_engine.infer_from_dataset(dataset, settings.MODEL)
else:
    shape_spec = None

# Model type specific logic
if model_type == ModelType.GRAPH:
    wrapper = WrapperFactory.create_graph_wrapper(...)
elif model_type == ModelType.SHAPE_AWARE_DLKIT:
    wrapper = WrapperFactory.create_standard_wrapper(...)
```

**Implementation Notes**:
- Uses ABC inheritance checks (`ShapeAwareModel`, `ShapeAgnosticModel`, `BaseGraphNetwork`)
- Falls back to `LightningModule` check for external models
- Chain of responsibility pattern for extensibility
- Default detector always returns result (no failures)
- Shape spec required for: SHAPE_AWARE_DLKIT, GRAPH, TIMESERIES

---

### Component 7: `ABCModelTypeDetector`

**Purpose**: Concrete detector using ABC inheritance to classify models.

**Key Methods**:
- `can_detect(model_settings, settings) -> bool` - Always True (default detector)
- `detect_type(model_settings, settings) -> ModelType` - Detect via ABC checks
- `_get_model_class(model_settings) -> Type | None` - Extract model class

**Example**:
```python
from dlkit.runtime.workflows.factories.model_detection import ABCModelTypeDetector

detector = ABCModelTypeDetector()

# Always can detect
assert detector.can_detect(settings.MODEL, settings)

# Detect type via ABC inheritance
model_type = detector.detect_type(settings.MODEL, settings)
```

**Implementation Notes**:
- Checks `BaseGraphNetwork` → GRAPH
- Checks `ShapeAwareModel` → SHAPE_AWARE_DLKIT
- Checks `ShapeAgnosticModel` → SHAPE_AGNOSTIC_EXTERNAL
- Checks `LightningModule` → SHAPE_AGNOSTIC_EXTERNAL
- Default: SHAPE_AGNOSTIC_EXTERNAL
- Handles string model names via `import_object()`
- Handles type model names directly

## Usage Patterns

### Common Use Case 1: Basic Component Building
```python
from dlkit.runtime.workflows.factories import BuildFactory
from dlkit.tools.config import GeneralSettings

# Load configuration
settings = GeneralSettings.from_toml("training_config.toml")

# Create factory and build components
factory = BuildFactory()
components = factory.build_components(settings)

# Use components for training
trainer = components.trainer
trainer.fit(components.model, datamodule=components.datamodule)

# Access shape spec if needed
if components.shape_spec:
    print(f"Input shape: {components.shape_spec.input_shape}")
    print(f"Output shape: {components.shape_spec.output_shape}")
```

### Common Use Case 2: Custom Build Strategy
```python
from dlkit.runtime.workflows.factories import BuildFactory, IBuildStrategy

class CustomBuildStrategy(IBuildStrategy):
    def can_handle(self, settings):
        # Custom detection logic
        return settings.DATASET.type == "custom"

    def build(self, settings):
        # Custom building logic
        model = build_custom_model(settings)
        datamodule = build_custom_datamodule(settings)
        return BuildComponents(
            model=model,
            datamodule=datamodule,
            trainer=None,
            shape_spec=None,
            meta={"dataset_type": "custom"},
        )

# Use custom strategy
factory = BuildFactory(strategies=[CustomBuildStrategy(), FlexibleBuildStrategy()])
components = factory.build_components(settings)
```

### Common Use Case 3: Inference Mode Building
```python
# Configuration with inference mode
settings = GeneralSettings.from_toml("inference_config.toml")
settings.SESSION.inference = True

# Build components (no trainer in inference mode)
factory = BuildFactory()
components = factory.build_components(settings)

assert components.trainer is None  # No trainer for inference
assert components.model is not None
assert components.datamodule is not None

# Use for predictions
predictions = components.model.predict(datamodule=components.datamodule)
```

### Common Use Case 4: Model Type Detection
```python
from dlkit.runtime.workflows.factories.model_detection import (
    detect_model_type,
    requires_shape_spec,
    ModelType,
)

# Detect model type from settings
model_type = detect_model_type(settings.MODEL, settings)

# Conditional logic based on model type
if model_type == ModelType.GRAPH:
    print("Using graph build strategy")
    strategy = GraphBuildStrategy()
elif model_type == ModelType.SHAPE_AWARE_DLKIT:
    print("Using flexible strategy with shape inference")
    strategy = FlexibleBuildStrategy()
else:
    print("Using external model without shape inference")
    strategy = FlexibleBuildStrategy()

components = strategy.build(settings)
```

### Common Use Case 5: Legacy x/y Parameter Support
```python
# Legacy configuration with x/y
legacy_config = """
[DATASET]
name = "SupervisedArrayDataset"
x = "data/features.npy"
y = "data/targets.npy"

[DATASET.split]
test_ratio = 0.2
val_ratio = 0.1
"""

settings = GeneralSettings.from_toml_string(legacy_config)

# FlexibleBuildStrategy handles legacy x/y
factory = BuildFactory()
components = factory.build_components(settings)

# Automatically translated to flexible features/targets
# FlexibleDataset(features={"x": "data/features.npy"}, targets={"y": "data/targets.npy"})
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: Shape inference failed for shape-aware model
- `ImportError`: Model class cannot be imported
- Various exceptions from FactoryProvider during component creation

**Error Handling Pattern**:
```python
from dlkit.runtime.workflows.factories import BuildFactory

try:
    factory = BuildFactory()
    components = factory.build_components(settings)
except ValueError as e:
    if "Shape inference failed" in str(e):
        logger.error("Shape-aware model requires shape information")
        # Provide shape manually or fix dataset
    else:
        raise
except ImportError as e:
    logger.error(f"Failed to import model class: {e}")
    # Check model name and module_path in settings
except Exception as e:
    logger.error(f"Component building failed: {e}")
    # Handle general failures
```

**Fail-Safe Design**:
- Flexible strategy always succeeds (fallback)
- Best-effort detection for model types
- Graceful fallback on missing configurations
- Null shape spec for models not requiring shapes

## Testing

### Test Coverage
- Unit tests: `tests/runtime/workflows/test_build_factory_basic.py`
- Integration tests: `tests/integration/test_basic_integration.py`
- Model detection tests: `tests/runtime/workflows/test_model_detection.py`

### Key Test Scenarios
1. **Strategy selection**: Correct strategy chosen for dataset type
2. **Component construction**: All components built correctly
3. **Shape inference**: Shapes inferred for shape-aware models
4. **Legacy support**: x/y parameters translated to flexible entries
5. **Model type detection**: ABC-based detection works correctly
6. **Inference mode**: Trainer not built in inference mode
7. **Entry registry**: Entry configs registered for user access
8. **Split caching**: Splits cached and reused correctly

### Fixtures Used
- `general_settings` (from `conftest.py`): Complete configuration
- `tmp_path` (pytest built-in): Temporary paths for datasets
- `monkeypatch` (pytest built-in): Mock component creation

## Performance Considerations
- Shape inference runs once per build (cached in shape_spec)
- Split caching via split provider (filesystem cache)
- Entry registry singleton (no redundant registrations)
- Lazy strategy evaluation (first match wins)
- Model class import cached by Python import system

## Future Improvements / TODOs
- [ ] Support for multi-modal datasets
- [ ] Parallel component building
- [ ] Component validation before returning
- [ ] Build caching based on settings hash
- [ ] More sophisticated model type detection
- [ ] Support for custom shape inference engines
- [ ] Build event hooks for plugins
- [ ] Automatic hyperparameter tuning during build

## Related Modules
- `dlkit.runtime.workflows.strategies.core`: Uses BuildComponents for execution
- `dlkit.runtime.workflows.optimization`: Uses BuildFactory for trial execution
- `dlkit.core.shape_specs`: Shape inference system
- `dlkit.core.models.wrappers.factories`: Wrapper creation
- `dlkit.tools.config`: Settings management
- `dlkit.tools.io.split_provider`: Split caching

## Change Log
- **2024-10-03**: Added ABC-based model detection system
- **2024-10-02**: Implemented strategy pattern for dataset families
- **2024-10-01**: Added shape inference integration
- **2024-09-30**: Created BuildComponents dataclass
- **2024-09-28**: Migrated from legacy build_model_state
- **2024-09-25**: Added entry registry integration
