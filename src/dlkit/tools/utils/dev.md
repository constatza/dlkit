# Utilities Module

## Overview
The utilities module provides essential helper functions and system-level utilities for DLKit, including dynamic import/reflection utilities, centralized logging configuration with loguru, and server worker management. It implements clean functional programming principles with pure helper functions and proper separation of concerns.

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
- Loguru `backtrace` and `diagnose` are **opt-in via env vars** (`DLKIT_LOG_BACKTRACE`, `DLKIT_LOG_DIAGNOSE`), not tied to debug mode

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
| `recommended_uvicorn_workers()` | Function | Calculate optimal worker count | `int` |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `get_mro_keys()` | Function | Extract parameter names from class MRO | `tuple[dict, dict]` |
| `split_module_path()` | Function | Split module:obj path notation | `tuple[str | None, str]` |
| `_is_debug_enabled()` | Function | Check if debug logging enabled | `bool` |
| `_backtrace_enabled()` | Function | Read `DLKIT_LOG_BACKTRACE` env var | `bool` |
| `_diagnose_enabled()` | Function | Read `DLKIT_LOG_DIAGNOSE` env var | `bool` |
| `_debug_filter()` | Function | Filter debug messages by module origin | `bool` |
| `_suppress_third_party_logging()` | Function | Suppress noisy third-party logs | `None` |
| `_get_default_log_file_path()` | Function | Get default log file path | `Path` |

### Protocols/Interfaces
None - pure utility functions

### Sub-modules
| Sub-module | Purpose | Key Functions |
|------------|---------|---------------|
| `general.py` | General utilities (import, reflection, filtering) | `import_object`, `kwargs_compatible_with`, `filter_dict` |
| `logging_config.py` | Centralized loguru logging configuration | `configure_logging`, `get_logger` |
| `system_utils.py` | Server worker management | `recommended_uvicorn_workers` |
| `tensordict_utils.py` | TensorDict/sequence helpers | `tensordict_to_numpy`, `sequence_to_tensordict` |
| `error_handling.py` | Error handling utilities | Cross-cutting error helpers |
| `subprocess.py` | Subprocess management | Process lifecycle utilities |

> **Moved modules**: Path utilities (`mkdir_for_local`, `normalize_user_path`,
> `coerce_root_dir_to_absolute`) now live in `dlkit.tools.io.paths`.
> Tensor utilities (`ensure2d`) now live in `dlkit.core.datasets.tensor_utils`.
> Checkpoint security now lives in `dlkit.core.models.wrappers.security`.
> Metric collection now lives in `dlkit.core.training.metrics.collect`.

## Dependencies

### Internal Dependencies
- `dlkit.tools.config.environment`: Environment settings (`DLKitEnvironment`)

### External Dependencies
- `loguru`: Structured logging (`logger`)
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

all_kwargs = {
    "hidden_size": 128,
    "num_layers": 3,
    "learning_rate": 0.001,
    "batch_size": 32
}

compatible = kwargs_compatible_with(MyModel, **all_kwargs)
# Result: {"hidden_size": 128, "num_layers": 3}

incompatible = kwargs_compatible_with(MyModel, which="incompatible", **all_kwargs)
# Result: {"learning_rate": 0.001, "batch_size": 32}

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

Linear = import_object("torch.nn:Linear")
model = Linear(10, 5)

MyModel = import_object("MyModel", fallback_module="dlkit.core.models.nn.ffnn")

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

logger = get_logger(__name__)
logger.info("Processing started")

logger = get_logger(__name__, component="training")
logger.debug("Epoch completed", epoch=10, loss=0.5)
```

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

configure_logging()
configure_logging(level="DEBUG")
configure_logging(format_type="simple")
```

---

### Component 5: Logging Debug Modes (env vars)

**Purpose**: Two opt-in environment variables gate loguru's advanced traceback features. Both default to `0` (off) and are independent of the log level.

| Variable | Effect |
|---|---|
| `DLKIT_LOG_BACKTRACE=1` | Enable loguru's extended call-chain display in captured exceptions |
| `DLKIT_LOG_DIAGNOSE=1` | Enable loguru's local-variable dump inside tracebacks |

**Why opt-in**: `diagnose=True` dumps every local variable (including full tensors/arrays) when loguru captures an exception. Under pytest, `debug_mode` was always `True`, causing output floods. Neither feature is load-bearing in the current codebase — DLKit does not use `logger.exception()`, `logger.catch()`, or `sys.excepthook`.

**Example**:
```bash
# See local variables in tracebacks (verbose — avoid in CI)
DLKIT_LOG_DIAGNOSE=1 dlkit train config.toml

# See extended call chains (useful for debugging deeply nested calls)
DLKIT_LOG_BACKTRACE=1 uv run python script.py
```

---

### Component 6: `recommended_uvicorn_workers()`

**Purpose**: Calculate optimal worker count for Uvicorn/Gunicorn servers using standard (2 * cores) + 1 heuristic.

**Returns**: `int` - Recommended worker count

**Example**:
```python
from dlkit.tools.utils.system_utils import recommended_uvicorn_workers

workers = recommended_uvicorn_workers()

import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000, workers=workers)
```

**Implementation Notes**:
- Uses Gunicorn's recommended formula: (2 * cores) + 1
- Falls back to 8 cores if `os.cpu_count()` returns None

---

### Component 7: Third-Party Log Suppression

**Purpose**: Suppress noisy third-party library logs using loguru interception and filtering.

**Affected Libraries**:
- `alembic`, `sqlalchemy`, `werkzeug`, `urllib3`, `mlflow`

**Example**:
```python
from dlkit.tools.utils.logging_config import get_logger

# Third-party logs are suppressed to WARNING automatically
logger = get_logger(__name__)
logger.debug("This debug message shows")
```

## Usage Patterns

### Common Use Case 1: Dynamic Component Construction
```python
from dlkit.tools.utils.general import import_object, kwargs_compatible_with

model_class = import_object("torch.nn:Linear")

all_config = {
    "in_features": 784,
    "out_features": 10,
    "bias": True,
    "learning_rate": 0.001
}

compatible_kwargs = kwargs_compatible_with(model_class, **all_config)
model = model_class(**compatible_kwargs)
```

### Common Use Case 2: Structured Logging
```python
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__, component="training")

logger.info("Training started", model="ResNet50", dataset="ImageNet")

try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", error=str(e), exc_info=True)
```

### Common Use Case 3: Dictionary Filtering
```python
from dlkit.tools.utils.general import filter_dict

config = {
    "model_path": "/path/to/model.ckpt",
    "learning_rate": 0.001,
    "batch_size": 32,
}

paths = filter_dict(config, lambda v: isinstance(v, str))
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: Invalid parameters (e.g., wrong `which` value in `filter_dict`)
- `ImportError`: Module or object not found in `import_object`

## Testing

### Test Coverage
- Unit tests: `tests/tools/utils/`

### Key Test Scenarios
1. **kwargs_compatible_with**: Class hierarchy, functions, edge cases
2. **import_object**: Valid paths, fallback modules, missing modules
3. **Logging**: Level filtering, format types, third-party suppression

### Fixtures Used
- `tmp_path` (pytest built-in): Temporary paths for file operations
- `caplog` (pytest built-in): Log message capturing

## Performance Considerations
- `kwargs_compatible_with` caches signature inspection per class
- `import_object` uses Python's import cache
- Logger configuration is one-time on module import
- Dictionary filtering uses generator expressions for efficiency

## Related Modules
- `dlkit.tools.io.paths`: Path utilities (`mkdir_for_local`, `normalize_user_path`, `coerce_root_dir_to_absolute`)
- `dlkit.core.datasets.tensor_utils`: Tensor/dataloader helpers (`ensure2d`)
- `dlkit.core.models.wrappers.security`: Checkpoint security configuration
- `dlkit.core.training.metrics.collect`: Metric value collection
- `dlkit.tools.config`: Uses import utilities for dynamic component loading
- `dlkit.tools.registry`: Uses import utilities for component resolution
- All modules: Use get_logger for consistent logging
