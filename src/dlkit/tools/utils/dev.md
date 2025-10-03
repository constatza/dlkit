# Utilities Module

## Overview
The utilities module provides essential helper functions and system-level utilities for DLKit, including dynamic import/reflection utilities, centralized logging configuration with loguru, system utilities for path/worker management, and PyTorch-specific tensor manipulation helpers. It implements clean functional programming principles with pure helper functions and proper separation of concerns.

## Architecture & Design Patterns
- **Pure Functions**: Most utilities are pure functions with no side effects
- **Single Responsibility Principle (SRP)**: Each utility has one focused purpose
- **Functional Programming**: Emphasis on composable, reusable helper functions
- **Structured Logging**: Loguru-based logging with environment control and third-party suppression
- **Introspection**: Dynamic class/function introspection for reflection-based operations
- **Type Safety**: Strong type hints throughout with Pydantic validation where appropriate

Key architectural decisions:
- No global state mutation (except logger configuration)
- Minimal dependencies - utilities are leaf nodes in dependency graph
- Cross-platform compatibility (Path-based, OS-aware logic)
- Test environment detection for debug logging auto-enable

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `kwargs_compatible_with()` | Function | Filter kwargs for constructor compatibility | `dict[str, Any]` |
| `import_object()` | Function | Dynamically import class/function from path | `Callable` |
| `get_name()` | Function | Get name of function/class/instance | `str` |
| `slice_to_list()` | Function | Convert slice to list of indices | `list[int]` |
| `filter_dict()` | Function | Filter dict by predicate on keys/values | `dict` |
| `get_signature_names()` | Function | Get parameter names from callable | `list[str]` |
| `get_logger()` | Function | Get configured logger instance | `Logger` |
| `configure_logging()` | Function | Configure loguru logging system | `None` |
| `mkdir_for_local()` | Function | Create directory for local file URI | `None` |
| `recommended_uvicorn_workers()` | Function | Calculate optimal worker count | `int` |
| `dataloader_to_xy()` | Function | Convert DataLoader to (X, y) tensors | `tuple[Tensor, Tensor]` |
| `xy_from_batch()` | Function | Extract (x, y) from batch dict/sequence | `tuple` |
| `ensure2d()` | Function | Ensure tensor is at least 2D | `Tensor` |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `get_mro_keys()` | Function | Extract parameter names from class MRO | `tuple[dict, dict]` |
| `split_module_path()` | Function | Split module:obj path notation | `tuple[str | None, str]` |
| `_is_debug_enabled()` | Function | Check if debug logging enabled | `bool` |
| `_debug_filter()` | Function | Filter debug messages by module origin | `bool` |
| `_suppress_third_party_logging()` | Function | Suppress noisy third-party logs | `None` |
| `_get_default_log_file_path()` | Function | Get default log file path | `Path` |
| `split_first_from_sequence()` | Function | Split sequence into first and rest | `tuple` |

### Protocols/Interfaces
None - pure utility functions

### Sub-modules
| Sub-module | Purpose | Key Functions |
|------------|---------|---------------|
| `general.py` | General utilities (import, reflection, filtering) | `import_object`, `kwargs_compatible_with`, `filter_dict` |
| `logging_config.py` | Centralized loguru logging configuration | `configure_logging`, `get_logger` |
| `system_utils.py` | System-level utilities (paths, workers) | `mkdir_for_local`, `recommended_uvicorn_workers` |
| `torch_utils.py` | PyTorch tensor utilities | `dataloader_to_xy`, `xy_from_batch`, `ensure2d` |

## Dependencies

### Internal Dependencies
- `dlkit.tools.config.environment`: Environment settings (`DLKitEnvironment`)

### External Dependencies
- `loguru`: Structured logging (`logger`)
- `torch`: Tensor operations and DataLoader
- `pydantic`: URL validation (`AnyUrl`)
- `inspect`: Introspection (`signature`, `isclass`)
- `importlib`: Dynamic imports (`import_module`)
- `pathlib`: Path manipulation

## Key Components

### Component 1: `kwargs_compatible_with()`

**Purpose**: Filter keyword arguments to only those compatible with a given class or function signature. Essential for dynamic component construction.

**Parameters**:
- `cls: type` - Class or function to check against
- `which: Literal["compatible", "incompatible"] = "compatible"` - Which kwargs to return
- `**kwargs` - Keyword arguments to filter

**Returns**: `dict[str, Any]` - Filtered kwargs

**Raises**:
- `ValueError`: If `which` parameter is invalid

**Example**:
```python
from dlkit.tools.utils.general import kwargs_compatible_with

class MyModel:
    def __init__(self, hidden_size: int, num_layers: int):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

# Filter kwargs to match constructor
all_kwargs = {
    "hidden_size": 128,
    "num_layers": 3,
    "learning_rate": 0.001,  # Not in constructor
    "batch_size": 32  # Not in constructor
}

compatible = kwargs_compatible_with(MyModel, **all_kwargs)
# Result: {"hidden_size": 128, "num_layers": 3}

incompatible = kwargs_compatible_with(MyModel, which="incompatible", **all_kwargs)
# Result: {"learning_rate": 0.001, "batch_size": 32}

# Use filtered kwargs for construction
model = MyModel(**compatible)
```

**Implementation Notes**:
- Inspects Method Resolution Order (MRO) for classes
- Handles both classes and callable functions
- Excludes "self" parameter from checks
- Used extensively by factory pattern for constructor filtering

---

### Component 2: `import_object()`

**Purpose**: Dynamically import and return a class or function from a module path string. Supports both `module.path:ClassName` and `ClassName` (with fallback module) notation.

**Parameters**:
- `module_path: str` - Import path (e.g., "torch.nn:Linear" or "MyClass")
- `fallback_module: str = ""` - Module to use if no module specified in path

**Returns**: `Callable` - The imported class or function

**Raises**:
- `ImportError`: If module or object cannot be found

**Example**:
```python
from dlkit.tools.utils.general import import_object

# Import with full path
Linear = import_object("torch.nn:Linear")
model = Linear(10, 5)

# Import with fallback module
MyModel = import_object("MyModel", fallback_module="dlkit.core.models.nn.ffnn")

# Import function
relu = import_object("torch.nn.functional:relu")
```

**Implementation Notes**:
- Splits on ":" to separate module from object name
- Uses `importlib.import_module` for module loading
- Falls back to fallback_module if no module path specified
- Essential for configuration-based component instantiation

---

### Component 3: `get_logger()`

**Purpose**: Get a configured loguru logger instance with appropriate bindings for the given module and optional component.

**Parameters**:
- `name: str` - Module name (typically `__name__`)
- `component: str | None = None` - Optional component name for categorization

**Returns**: Configured logger instance with bindings

**Example**:
```python
from dlkit.tools.utils.logging_config import get_logger

# Basic logger
logger = get_logger(__name__)
logger.info("Processing started")

# Logger with component binding
logger = get_logger(__name__, component="training")
logger.debug("Epoch completed", epoch=10, loss=0.5)

# Structured logging with extra context
logger.info("Model saved", path="/path/to/model.ckpt", size_mb=150)
```

**Implementation Notes**:
- Returns loguru logger with module and optional component bindings
- Structured logging support with extra fields in log context
- Level filtering based on debug mode and module origin
- Auto-configured on module import

---

### Component 4: `configure_logging()`

**Purpose**: Configure loguru logger with appropriate levels, formatting, and third-party log suppression.

**Parameters**:
- `level: str | None = None` - Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `debug_enabled: bool | None = None` - Whether to enable debug logging
- `format_type: str = "structured"` - Format type ('structured' or 'simple')

**Returns**: `None` (configures global logger)

**Example**:
```python
from dlkit.tools.utils.logging_config import configure_logging

# Use defaults (INFO level, structured format)
configure_logging()

# Enable debug logging
configure_logging(level="DEBUG")

# Simple format for development
configure_logging(format_type="simple")

# Explicit debug enable
configure_logging(debug_enabled=True)

# Environment variables also work:
# DLKIT_LOG_LEVEL=DEBUG
# DLKIT_LOG_FILE=/custom/path/app.log
```

**Implementation Notes**:
- Removes default loguru handler and configures custom handlers
- Console handler with color and backtrace support
- File handler for warnings/errors with rotation (10MB, 7 days retention)
- Suppresses third-party logs (alembic, sqlalchemy, werkzeug, urllib3, mlflow) to WARNING
- Auto-enables debug in pytest environment
- Debug messages filtered to dlkit modules only

---

### Component 5: `mkdir_for_local()`

**Purpose**: Ensure the local directory for a given URI or file path exists, handling various URI schemes (file, sqlite, plain paths).

**Parameters**:
- `uri: AnyUrl | str` - File URI, sqlite URI, or plain file path

**Returns**: `None` (creates directories as side effect)

**Example**:
```python
from dlkit.tools.utils.system_utils import mkdir_for_local

# Plain file path
mkdir_for_local("/path/to/output/results.csv")

# File URI
mkdir_for_local("file:///path/to/output/data.db")

# SQLite URI
mkdir_for_local("sqlite:////absolute/path/database.db")

# Windows paths handled correctly
mkdir_for_local("C:\\Users\\data\\file.txt")

# Non-local schemes ignored (no-op)
mkdir_for_local("http://example.com/file.csv")  # Does nothing
```

**Implementation Notes**:
- Handles file://, sqlite:// schemes and plain paths
- Ignores non-local schemes (http, https, s3, etc.)
- Cross-platform path handling (Windows vs Unix)
- Creates parent directories recursively
- Idempotent - safe to call multiple times

---

### Component 6: `recommended_uvicorn_workers()`

**Purpose**: Calculate optimal worker count for Uvicorn/Gunicorn servers using standard (2 * cores) + 1 heuristic.

**Returns**: `int` - Recommended worker count

**Example**:
```python
from dlkit.tools.utils.system_utils import recommended_uvicorn_workers

# Get recommended worker count
workers = recommended_uvicorn_workers()

# Use with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000, workers=workers)

# On 8-core machine: returns (2 * 8) + 1 = 17 workers
# On 4-core machine: returns (2 * 4) + 1 = 9 workers
```

**Implementation Notes**:
- Uses Gunicorn's recommended formula: (2 * cores) + 1
- Falls back to 8 cores if `os.cpu_count()` returns None
- Balances CPU and I/O bound operations

---

### Component 7: `dataloader_to_xy()`

**Purpose**: Convert entire PyTorch DataLoader to (X, y) tensors by iterating through all batches.

**Parameters**:
- `dataloader: DataLoader` - PyTorch DataLoader to convert

**Returns**: `tuple[torch.Tensor, torch.Tensor]` - Concatenated features and targets

**Example**:
```python
from dlkit.tools.utils.torch_utils import dataloader_to_xy
from torch.utils.data import DataLoader, TensorDataset
import torch

# Create sample dataloader
dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
dataloader = DataLoader(dataset, batch_size=16)

# Convert to tensors
X, y = dataloader_to_xy(dataloader)

print(X.shape)  # torch.Size([100, 10])
print(y.shape)  # torch.Size([100, 1])
```

**Implementation Notes**:
- Iterates through all batches and concatenates
- Uses `xy_from_batch()` to handle dict/sequence batch formats
- Memory-intensive for large datasets - consider alternatives
- Useful for small datasets or evaluation

---

### Component 8: `xy_from_batch()`

**Purpose**: Extract features (x) and targets (y) from a batch in dict or sequence format.

**Parameters**:
- `batch` - Batch data (dict with "features"/"x" and "targets"/"y" keys, or sequence)

**Returns**: `tuple` - (x, y) extracted from batch

**Example**:
```python
from dlkit.tools.utils.torch_utils import xy_from_batch
import torch

# Dict format
batch_dict = {"features": torch.randn(32, 10), "targets": torch.randn(32, 1)}
x, y = xy_from_batch(batch_dict)

# Alternative dict keys
batch_dict2 = {"x": torch.randn(32, 10), "y": torch.randn(32, 1)}
x, y = xy_from_batch(batch_dict2)

# Sequence format (tuple/list)
batch_seq = (torch.randn(32, 10), torch.randn(32, 1))
x, y = xy_from_batch(batch_seq)

# Single tensor (returns as-is)
single = torch.randn(32, 10)
result = xy_from_batch(single)  # Returns single tensor
```

**Implementation Notes**:
- Supports dict format with flexible key names (features/x, targets/y)
- Supports sequence format (tuple, list)
- Returns single tensor unchanged
- Essential for flexible batch handling across different data formats

---

### Component 9: `ensure2d()`

**Purpose**: Ensure tensor is at least 2D by adding dimension if needed. Useful for node/graph-level features.

**Parameters**:
- `tensor: torch.Tensor` - Input tensor to ensure is 2D

**Returns**: `torch.Tensor` - Tensor with at least 2 dimensions

**Example**:
```python
from dlkit.tools.utils.torch_utils import ensure2d
import torch

# 1D tensor becomes 2D
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
tensor_2d = ensure2d(tensor_1d)
print(tensor_2d.shape)  # torch.Size([5, 1])

# 2D tensor unchanged
tensor_already_2d = torch.randn(10, 5)
result = ensure2d(tensor_already_2d)
print(result.shape)  # torch.Size([10, 5])

# 3D tensor unchanged
tensor_3d = torch.randn(10, 5, 3)
result = ensure2d(tensor_3d)
print(result.shape)  # torch.Size([10, 5, 3])
```

**Implementation Notes**:
- Uses `unsqueeze(1)` to add dimension to 1D tensors
- Leaves tensors with 2+ dimensions unchanged
- Common for graph neural network node features

---

### Component 10: Third-Party Log Suppression

**Purpose**: Suppress noisy third-party library logs using loguru interception and filtering.

**Affected Libraries**:
- `alembic`: Database migration logs
- `sqlalchemy`: SQL query logs
- `werkzeug`: HTTP request logs (Flask/MLflow)
- `urllib3`: HTTP connection logs
- `mlflow`: MLflow internal logs

**Example**:
```python
# Automatically configured on module import
from dlkit.tools.utils.logging_config import get_logger

# Third-party logs are suppressed to WARNING
# Only dlkit logs shown at configured level

logger = get_logger(__name__)
logger.debug("This debug message shows")  # Visible

# SQLAlchemy debug logs are suppressed
# Werkzeug HTTP logs are suppressed
# Only warnings/errors from third-party libs show
```

**Implementation Notes**:
- Uses loguru's `InterceptHandler` to capture standard library logging
- Sets third-party loggers to WARNING level
- Preserves important error messages while reducing noise
- Automatic on module import - no explicit configuration needed

## Usage Patterns

### Common Use Case 1: Dynamic Component Construction
```python
from dlkit.tools.utils.general import import_object, kwargs_compatible_with

# Import class dynamically
model_class = import_object("torch.nn:Linear")

# Filter kwargs for compatibility
all_config = {
    "in_features": 784,
    "out_features": 10,
    "bias": True,
    "learning_rate": 0.001  # Not in Linear constructor
}

compatible_kwargs = kwargs_compatible_with(model_class, **all_config)

# Construct with compatible kwargs only
model = model_class(**compatible_kwargs)
```

### Common Use Case 2: Structured Logging
```python
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__, component="training")

# Structured logging with context
logger.info("Training started", model="ResNet50", dataset="ImageNet")

# Error logging with exception
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", error=str(e), exc_info=True)

# Debug logging (auto-filtered by module)
logger.debug("Batch processed", batch_size=32, loss=0.5)
```

### Common Use Case 3: DataLoader Utilities
```python
from dlkit.tools.utils.torch_utils import dataloader_to_xy, ensure2d
from torch.utils.data import DataLoader

# Convert dataloader to tensors
X, y = dataloader_to_xy(train_loader)

# Ensure features are 2D for matrix operations
X = ensure2d(X)

# Now use for evaluation or visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.numpy())
```

### Common Use Case 4: Path/Directory Management
```python
from dlkit.tools.utils.system_utils import mkdir_for_local

# Ensure output directory exists before writing
output_path = "sqlite:////path/to/database.db"
mkdir_for_local(output_path)

# Now safe to create database
import sqlite3
conn = sqlite3.connect(output_path.replace("sqlite:///", ""))
```

### Common Use Case 5: Dictionary Filtering
```python
from dlkit.tools.utils.general import filter_dict

config = {
    "model_path": "/path/to/model.ckpt",
    "data_path": "/path/to/data",
    "learning_rate": 0.001,
    "batch_size": 32,
    "debug": True
}

# Filter by value type
paths = filter_dict(config, lambda v: isinstance(v, str))
# Result: {"model_path": ..., "data_path": ...}

# Filter by key name
hyperparams = filter_dict(
    config,
    lambda k: k in ("learning_rate", "batch_size"),
    which="key"
)
# Result: {"learning_rate": 0.001, "batch_size": 32}
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: Invalid parameters (e.g., wrong `which` value in `filter_dict`)
- `ImportError`: Module or object not found in `import_object`
- `KeyError`: Missing registry entry

**Error Handling Pattern**:
```python
from dlkit.tools.utils.general import import_object

try:
    cls = import_object("some.module:ClassName")
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    # Fallback to default
    cls = DefaultClass
```

## Testing

### Test Coverage
- Unit tests: `tests/tools/utils/` (if exists)
- Logging tests: Pytest fixtures for log capturing
- Import tests: Mock modules for dynamic import testing

### Key Test Scenarios
1. **kwargs_compatible_with**: Class hierarchy, functions, edge cases
2. **import_object**: Valid paths, fallback modules, missing modules
3. **Logging**: Level filtering, format types, third-party suppression
4. **Path utilities**: Cross-platform paths, URI schemes
5. **Tensor utilities**: Shape manipulation, batch format handling

### Fixtures Used
- `tmp_path` (pytest built-in): Temporary paths for file operations
- `caplog` (pytest built-in): Log message capturing

## Performance Considerations
- `kwargs_compatible_with` caches signature inspection per class
- `import_object` uses Python's import cache
- Logger configuration is one-time on module import
- Dictionary filtering uses generator expressions for efficiency

## Future Improvements / TODOs
- [ ] LRU cache for `get_mro_keys` to avoid repeated introspection
- [ ] Async logging support for high-throughput scenarios
- [ ] Additional tensor utilities (batching, padding, masking)
- [ ] Configuration validation helpers
- [ ] Performance profiling utilities

## Related Modules
- `dlkit.tools.config`: Uses import utilities for dynamic component loading
- `dlkit.tools.registry`: Uses import utilities for component resolution
- `dlkit.runtime.workflows`: Uses logging throughout
- All modules: Use get_logger for consistent logging

## Change Log
- **2025-10-03**: Comprehensive documentation created
- **2024-10-01**: Third-party log suppression added
- **2024-09-28**: Loguru migration from standard logging
- **2024-09-20**: System utilities added (mkdir_for_local, workers)
