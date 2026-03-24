# I/O Module

## Overview
The I/O module provides comprehensive file input/output operations for DLKit, including TOML configuration loading, array/tensor loading with precision management, sparse matrix pack storage/loading, path resolution with context awareness, standardized location management, and data format handling (tables, indices). It implements clean separation between path resolution (pure functions), directory creation (explicit provisioning), and data loading with strong type safety throughout.

## Architecture & Design Patterns
- **Single Responsibility Principle (SRP)**: Separate modules for config loading, path resolution, provisioning, and data loading
- **Pure Functions**: Path resolution functions have no side effects (no directory creation)
- **Explicit Provisioning**: Directory creation isolated in `provisioning.py` module
- **Strategy Pattern**: Multiple resolvers (`GenericPathResolver`, URL resolvers) for different path types
- **Registry Pattern**: `ResolverRegistry` manages path resolvers with extensibility
- **Context Pattern**: `ResolverContext` provides environment-aware path resolution
- **Partial Parsing**: Efficient section-based TOML loading minimizes overhead
- **Environment Awareness**: Test detection automatically routes artifacts to `tests/` directory
- **Precision Service Integration**: Array loading uses precision service for consistent dtype resolution

Key architectural decisions:
- No paths section injection during config load - environment root is authoritative
- Path resolution follows precedence: CLI override > config > environment > CWD
- Test environment detection prevents pollution of working directory
- Configuration validation separated from loading (Pydantic models)
- Dynaconf compatibility for legacy migrations

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `load_config()` | Function | Load TOML config with validation | `BaseModel` or `dict` |
| `load_sections_config()` | Function | Load multiple sections efficiently | `dict[str, BaseModel]` |
| `load_section_config()` | Function | Load single config section | `BaseModel` |
| `load_raw_config()` | Function | Load TOML as dict without validation | `dict[str, Any]` |
| `write_config()` | Function | Write settings to TOML file | `Path` |
| `check_section_exists()` | Function | Check if section exists without full parsing | `bool` |
| `get_available_sections()` | Function | List available sections | `list[str]` |
| `register_section_mapping()` | Function | Register model-section mapping | `None` |
| `load_array()` | Function | Load array/tensor with precision management | `Tensor` |
| `read_table()` | Function | Load tabular data (CSV, Parquet) | `pl.DataFrame` |
| `load_split_indices()` | Function | Load train/val/test indices | `IndexSplit` |
| `save_sparse_pack()` | Function | Save sparse payload files (`indices`, `values`, `nnz_ptr`, `values_scale`) | `None` |
| `open_sparse_pack()` | Function | Open sparse pack directly from payload files | `AbstractSparsePackReader` |
| `validate_sparse_pack()` | Function | Validate sparse payload consistency (shape/nnz/ranges/dtype) | `None` |
| `is_sparse_pack_dir()` | Function | Check whether directory has required sparse payload files | `bool` |
| `save_split_indices()` | Function | Save index splits to JSON | `None` |
| `mkdir_for_local()` | Function | Ensure local directory exists for URI/path | `None` |
| `normalize_user_path()` | Function | Normalize user-supplied path (tilde, relative) | `Path \| None` |
| `coerce_root_dir_to_absolute()` | Function | Coerce root_dir value to absolute Path | `Path \| None` |
| `locations` | Module | Standardized location resolution | Various `Path` functions |
| `provisioning` | Module | Explicit directory creation | Various `ensure_*` functions |
| `sparse` | Module | Sparse matrix pack I/O (COO now, CSR-ready) with Pydantic contracts and value-scale support | `save_sparse_pack()`, `open_sparse_pack()`, `validate_sparse_pack()` |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `PartialTOMLParser` | Class | Efficient section-based TOML parsing | Section data |
| `_preprocess_sections()` | Function | Apply path resolution to section data | `dict[str, Any]` |
| `_to_toml_compatible()` | Function | Convert values to TOML-serializable | `Any` |
| `_resolve_section_models()` | Function | Resolve section name to model class mappings | `dict[str, type[BaseModel]]` |
| `_compute_root_dict()` | Function | Compute effective root with precedence | `Path` |
| `_process_path_field()` | Function | Process individual path field | `Any` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `PathResolver` | `resolve()`, `can_resolve()` | Abstract path resolution interface |

### Sub-modules
| Sub-module | Purpose | Key Components |
|------------|---------|----------------|
| `resolution/` | Path resolution with context | `GenericPathResolver`, `ResolverContext`, `ResolverRegistry` |
| `locations.py` | Standardized location management (pure) | `root()`, `output()`, `checkpoints_dir()`, etc. |
| `provisioning.py` | Explicit directory creation | `ensure_run_dirs()`, `ensure_mlflow_local_storage()` |
| `config.py` | TOML configuration loading | `load_config()`, `load_sections_config()`, `write_config()` |
| `arrays.py` | Array/tensor loading | `load_array()`, `load_array_with_session_precision()` |
| `tables.py` | Tabular data loading | `read_table()` |
| `index.py` | Index split persistence | `load_split_indices()`, `save_split_indices()` |
| `sparse/` | Sparse matrix pack persistence and reading — OCP registry, ISP protocols, LSP ABC | `SparseFormat`, `PackFiles`, `PackManifest`, `CooPackCodec`, `CooPackReader`, `register_format`, `value_scale` + `denormalize` flow |
| `parsers.py` | TOML parsing utilities | `PartialTOMLParser` |
| `paths.py` | Path normalization and local URI helpers | `mkdir_for_local()`, `normalize_user_path()`, `coerce_root_dir_to_absolute()` |

## Dependencies

### Internal Dependencies
- `dlkit.tools.config`: Settings models (`GeneralSettings`, section settings)
- `dlkit.tools.config.environment`: Environment management (`DLKitEnvironment`)
- `dlkit.core.datatypes.urls`: URL handling (`tilde_expand_strict`)
- `dlkit.core.datatypes.split`: Split datatype (`IndexSplit`)
- `dlkit.core.datatypes.secure_uris`: Secure path handling (`SecurePath`)
- `dlkit.interfaces.api.overrides.path_context`: Path override context
- `dlkit.interfaces.api.services.precision_service`: Precision management

### External Dependencies
- `dynaconf`: Configuration management (`Dynaconf`)
- `pydantic`: Validation and serialization (`BaseModel`, `FilePath`, `validate_call`)
- `tomlkit`: TOML writing (`document`, `table`, `dumps`)
- `polars`: Tabular data reading (`pl.read_csv`, `pl.read_parquet`)
- `torch`: Tensor operations and dtype management
- `numpy`: Array loading (`np.load`, `np.loadtxt`)
- `pathlib`: Path manipulation

## Key Components

### Component 1: `load_config()`

**Purpose**: Load TOML config file with optional Pydantic validation and automatic path resolution.

**Parameters**:
- `config_path: Path | str` - Path to the TOML config file
- `model_class: type[T] | None = None` - Pydantic model class for validation (None for raw dict)
- `raw: bool = False` - If True, return raw config dict without validation

**Returns**: `T | dict[str, Any]` - Validated model instance or raw dict

**Raises**:
- `FileNotFoundError`: If config file doesn't exist
- `ConfigValidationError`: If Pydantic validation fails
- `ConfigSectionError`: If required sections are missing

**Example**:
```python
from dlkit.tools.io import load_config
from dlkit.tools.config import GeneralSettings

# Load with automatic validation to GeneralSettings
settings = load_config("config.toml")

# Load with explicit model class
from dlkit.tools.config import SessionSettings
session = load_config("config.toml", model_class=SessionSettings)

# Load as raw dict
raw_config = load_config("config.toml", raw=True)
```

**Implementation Notes**:
- Uses Dynaconf for initial TOML parsing
- Applies automatic path resolution based on root precedence
- Resolves paths in MODEL.checkpoint, TRAINING.trainer.default_root_dir, DATASET paths
- Handles tilde expansion and relative path resolution
- Filters Dynaconf metadata before validation
- No directory creation during load (pure function)
- Defaults to GeneralSettings validation if no model_class specified

---

### Component 2: `load_sections_config()`

**Purpose**: Load and validate multiple sections from TOML config efficiently using partial parsing.

**Parameters**:
- `config_path: Path | str` - Path to the TOML config file
- `section_configs: Mapping[str, type[BaseModel] | None] | Sequence[str]` - Section names to model classes or list of section names

**Returns**: `dict[str, BaseModel]` - Dictionary mapping uppercased section names to validated instances

**Raises**:
- `FileNotFoundError`: If config file doesn't exist
- `ConfigSectionError`: If required sections are missing
- `ConfigValidationError`: If validation fails for any section

**Example**:
```python
from dlkit.tools.io import load_sections_config
from dlkit.tools.config import SessionSettings, ModelComponentSettings

# Load with explicit model classes
sections = load_sections_config("config.toml", {
    "SESSION": SessionSettings,
    "MODEL": ModelComponentSettings
})

# Load using registered defaults
sections = load_sections_config("config.toml", ["SESSION", "MODEL", "TRAINING"])

# Access loaded sections
session = sections["SESSION"]
model = sections["MODEL"]
```

**Implementation Notes**:
- Uses `PartialTOMLParser` for efficient parsing (only requested sections)
- Automatically resolves model classes from registry when not provided
- Applies path preprocessing before validation
- Validates each section independently
- Returns uppercased section names for consistency
- More efficient than `load_config()` when only subset of config needed

---

### Component 3: `write_config()`

**Purpose**: Write DLKit configuration to TOML file with proper serialization.

**Parameters**:
- `config: BaseModel | dict[str, Any]` - Settings model or raw dict to write
- `output_path: Path | str` - Destination TOML file path
- `by_alias: bool = True` - Use field aliases (e.g., PATHS.root instead of root_dir)
- `exclude_none: bool = True` - Exclude fields that are None
- `exclude_unset: bool = False` - Exclude fields not explicitly set (Pydantic only)
- `sort_sections: bool = True` - Write sections in sorted order for stable diffs

**Returns**: `Path` - Path to the written TOML file

**Example**:
```python
from dlkit.tools.io import write_config
from dlkit.tools.config import GeneralSettings

# Load settings
settings = GeneralSettings.from_toml_file("input.toml")

# Write to new file
output_path = write_config(settings, "output.toml")

# Write with custom options
write_config(
    settings,
    "output.toml",
    exclude_none=True,
    exclude_unset=True,
    sort_sections=True
)
```

**Implementation Notes**:
- Converts Pydantic models to dicts using `model_dump()`
- Handles special types: Path → str, Enum → value, torch.dtype → str
- Uses tomlkit for clean TOML output
- Sorts sections alphabetically for stable diffs
- Excludes None values by default to reduce config size
- Supports both Pydantic models and raw dicts

---

### Component 4: `load_array()`

**Purpose**: Load array or tensor from disk with precision-aware dtype resolution.

**Parameters**:
- `path: FilePath` - Path to .npy, .txt/.csv, or .pt/.pth file
- `dtype: torch.dtype | None = None` - Explicit dtype (None uses precision service)
- `precision_provider: object | None = None` - Optional precision provider
- `**kwargs` - Additional kwargs forwarded to loader (np.load, torch.load, etc.)

**Returns**: `Tensor` - Loaded tensor with appropriate precision

**Raises**:
- `ValueError`: If file format is unsupported
- `TypeError`: If loader returns unexpected type

**Example**:
```python
from dlkit.tools.io import load_array
import torch

# Use session precision (default)
tensor = load_array("data.npy")

# Override with explicit dtype
tensor = load_array("data.npy", dtype=torch.float16)

# Load with numpy kwargs
tensor = load_array("data.txt", delimiter=",", skiprows=1)

# Load PyTorch checkpoint
tensor = load_array("weights.pt")
```

**Implementation Notes**:
- Supports .npy, .txt, .csv, .pt, .pth formats
- Integrates with precision service for consistent dtype resolution
- Uses frozen loader map for security (MappingProxyType)
- Converts numpy arrays to torch tensors automatically
- Validates loaded data type
- Pydantic validation for path input

---

### Component 5: `locations` Module

**Purpose**: Centralized, environment-aware path policy (pure functions, no side effects).

**Key Functions**:
- `root() -> Path` - Resolved root directory (context > CWD)
- `output(*parts, env=None) -> Path` - Path under standard output directory
- `predictions_dir(env=None) -> Path` - Predictions output directory
- `checkpoints_dir(env=None) -> Path` - Model checkpoints directory
- `splits_dir(env=None) -> Path` - Data split indices directory
- `figures_dir(env=None) -> Path` - Figures/plots directory
- `lightning_work_dir(env=None) -> Path` - PyTorch Lightning working directory
- `mlruns_dir(env=None) -> Path` - MLflow runs directory
- `mlruns_backend_uri(env=None) -> str` - MLflow SQLite backend URI
- `mlartifacts_dir(env=None) -> Path` - MLflow artifacts directory
- `optuna_storage_uri(env=None) -> str` - Optuna SQLite storage URI

**Example**:
```python
from dlkit.tools.io import locations

# Get root directory
root = locations.root()

# Get standard output locations
checkpoints = locations.checkpoints_dir()
predictions = locations.predictions_dir()
figures = locations.figures_dir()

# Get MLflow locations
mlruns = locations.mlruns_dir()
mlflow_uri = locations.mlruns_backend_uri()

# Custom output subdirectory
custom = locations.output("experiments", "trial_001")

# Test environment automatically routes to tests/artifacts
# (when pytest is running or DLKIT_TEST_MODE=1)
```

**Implementation Notes**:
- All functions are pure (no side effects, no directory creation)
- Honors path override context (CLI/API can override)
- Detects test environment and routes to `tests/artifacts/`
- Formats database URIs with POSIX paths for cross-platform consistency
- Resolves paths relative to root with proper absolute resolution
- Environment parameter allows custom DLKitEnvironment instance

---

### Component 6: `provisioning` Module

**Purpose**: Explicit directory creation for run and server-related folders. All directory creation isolated here to avoid side effects in path resolution.

**Key Functions**:
- `ensure_internal_dirs(env=None) -> None` - Create internal DLKit directories (.dlkit)
- `ensure_run_dirs(env=None, needs=(...)) -> None` - Create standard output directories
- `ensure_mlflow_local_storage(env=None) -> None` - Create MLflow storage locations

**Example**:
```python
from dlkit.tools.io import provisioning

# Ensure internal directories exist
provisioning.ensure_internal_dirs()

# Ensure standard run directories
provisioning.ensure_run_dirs()

# Ensure specific directories
provisioning.ensure_run_dirs(needs=["predictions", "checkpoints"])

# Ensure MLflow storage
provisioning.ensure_mlflow_local_storage()
```

**Implementation Notes**:
- Uses `locations` module for path resolution
- Creates parent directories recursively
- Idempotent - safe to call multiple times
- Separate from path resolution for SRP
- Default `needs` includes: predictions, checkpoints, figures, lightning

---

### Component 7: Path Resolution System (`resolution/`)

**Purpose**: Generic path resolution with context awareness, supporting multiple path types and schemes.

**Key Classes**:
- `PathResolver` (Protocol) - Abstract resolver interface
- `GenericPathResolver` - Handles tilde expansion and relative paths
- `ResolverContext` - Provides resolution context (root, home paths)
- `ResolverRegistry` - Manages multiple resolvers with extensibility

**Factory Functions**:
- `create_resolver_registry() -> ResolverRegistry` - Create registry with defaults
- `create_resolver_context(root_path) -> ResolverContext` - Create resolution context
- `create_default_resolver_system(root_path) -> tuple[ResolverRegistry, ResolverContext]` - Complete system

**Example**:
```python
from dlkit.tools.io.resolution import (
    create_default_resolver_system,
    GenericPathResolver
)
from pathlib import Path

# Create resolver system
registry, context = create_default_resolver_system(Path("/project/root"))

# Resolve paths
resolver = GenericPathResolver()

# Tilde expansion
home_path = resolver.resolve("~/data/file.txt", context)

# Relative path resolution
rel_path = resolver.resolve("data/file.txt", context)
# Resolves to: /project/root/data/file.txt

# Absolute paths pass through
abs_path = resolver.resolve("/absolute/path", context)
```

**Implementation Notes**:
- `GenericPathResolver` uses pathlib for cross-platform compatibility
- Handles tilde expansion with context home path
- Resolves relative paths against context root
- Normalizes paths (removes .., ., symlinks)
- Extensible via ResolverRegistry for custom schemes
- URLs rejected by path resolver (use URL resolver instead)

---

### Component 8: `read_table()`

**Purpose**: Read tabular data from CSV or Parquet files into Polars DataFrame.

**Parameters**:
- `file_path: str` - Path to .csv, .parquet, or .pq file
- `**read_kwargs` - Additional kwargs passed to Polars reader

**Returns**: `pl.DataFrame` - Loaded DataFrame

**Raises**:
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file extension is unsupported

**Example**:
```python
from dlkit.tools.io import read_table

# Read CSV
df = read_table("data.csv")

# Read Parquet
df = read_table("data.parquet")

# With Polars reader options
df = read_table("data.csv", separator=";", skip_rows=1)
```

**Implementation Notes**:
- Supports .csv, .parquet, .pq extensions
- Delegates to Polars readers (`pl.read_csv`, `pl.read_parquet`)
- File existence validation before reading
- Case-insensitive extension matching

---

### Component 9: `paths.py` — Path Normalization Utilities

**Purpose**: User-path normalization and local-directory provisioning for URIs.
Moved here from `tools/utils/system_utils.py` to co-locate I/O concerns.

**Key Functions**:
- `mkdir_for_local(uri, *, root)` — Creates the local directory implied by a file/sqlite URI or plain path. Ignores remote schemes.
- `normalize_user_path(value, *, require_absolute)` — Expands `~`, resolves relative paths against CWD, optionally enforces absolute.
- `coerce_root_dir_to_absolute(value)` — Thin wrapper: `normalize_user_path(value, require_absolute=True)`.

**Example**:
```python
from dlkit.tools.io.paths import mkdir_for_local, normalize_user_path, coerce_root_dir_to_absolute

mkdir_for_local("sqlite:///./mlruns/mlflow.db")      # creates ./mlruns/
mkdir_for_local("file:///abs/path/data.db")           # creates /abs/path/

normalize_user_path("~/runs")                         # → Path(<home>/runs)
normalize_user_path("relative/run")                   # → Path(<cwd>/relative/run)

coerce_root_dir_to_absolute("/abs/root")              # → Path("/abs/root")
coerce_root_dir_to_absolute("relative")               # → None (not absolute)
```

**Implementation Notes**:
- Handles `file://`, `sqlite://` schemes and plain paths
- Ignores non-local schemes (`http`, `https`, `s3`, etc.)
- Cross-platform: uses `pathlib` throughout
- Lazy imports `url_resolver` to avoid circular imports at module init time

---

### Component 10: `load_split_indices()` and `save_split_indices()`

**Purpose**: Load and save train/val/test index splits for reproducible data partitioning.

**load_split_indices Parameters**:
- `path: FilePath` - Path to JSON file with splits

**Returns**: `IndexSplit` - Split indices datatype

**save_split_indices Parameters**:
- `idx_split: IndexSplit` - Split indices to save
- `path: Path` - Output JSON file path

**Returns**: `None`

**Example**:
```python
from dlkit.tools.io import load_split_indices, save_split_indices
from dlkit.core.datatypes.split import IndexSplit
from pathlib import Path

# Load existing splits
splits = load_split_indices(Path("splits.json"))
print(f"Train indices: {splits.train}")
print(f"Val indices: {splits.validation}")

# Create and save new splits
new_splits = IndexSplit(
    train=[0, 1, 2, 3, 4],
    validation=[5, 6],
    test=[7, 8, 9],
    predict=None
)
save_split_indices(new_splits, Path("new_splits.json"))
```

**Implementation Notes**:
- JSON format for human readability and version control
- Validates required keys (train, validation, test)
- predict field is optional
- Creates parent directories automatically when saving
- Excludes None values from saved JSON

## Usage Patterns

### Common Use Case 1: Loading Complete Configuration
```python
from dlkit.tools.io import load_config
from dlkit.tools.config import GeneralSettings

# Load and validate complete configuration
settings = load_config("config.toml")

# Access validated settings
print(f"Model: {settings.MODEL.name}")
print(f"Batch size: {settings.DATAMODULE.dataloader.batch_size}")
print(f"Max epochs: {settings.TRAINING.trainer.max_epochs}")

# Check optional features
if settings.mlflow_enabled:
    print(f"MLflow tracking: {settings.MLFLOW.tracking_uri}")
```

### Common Use Case 2: Efficient Partial Loading
```python
from dlkit.tools.io import load_sections_config

# Load only needed sections for evaluation
eval_sections = load_sections_config("config.toml", ["MODEL", "DATASET"])

model_config = eval_sections["MODEL"]
dataset_config = eval_sections["DATASET"]

# Much faster than loading entire config when only subset needed
```

### Common Use Case 3: Path Management with Test Detection
```python
from dlkit.tools.io import locations, provisioning

# Automatically routes to tests/artifacts during testing
checkpoints_dir = locations.checkpoints_dir()
predictions_dir = locations.predictions_dir()

# Explicit directory creation
provisioning.ensure_run_dirs(needs=["checkpoints", "predictions"])

# Save outputs
model.save(checkpoints_dir / "best_model.ckpt")
predictions.save(predictions_dir / "results.csv")
```

### Common Use Case 4: Loading Arrays with Precision Management
```python
from dlkit.tools.io import load_array
import torch

# Use session precision automatically
features = load_array("features.npy")

# Override precision for specific data
labels = load_array("labels.npy", dtype=torch.long)

# Load with numpy options
data = load_array("data.txt", delimiter=",", skiprows=1)
```

### Common Use Case 5: Configuration Round-Trip
```python
from dlkit.tools.io import load_config, write_config

# Load configuration
settings = load_config("original.toml")

# Modify settings (create new instance due to frozen=True)
updated_settings = settings.patch({
    "TRAINING.trainer.max_epochs": 200
})

# Save modified configuration
write_config(updated_settings, "modified.toml")
```

### Common Use Case 6: Custom Path Resolution
```python
from dlkit.tools.io.resolution import create_default_resolver_system
from pathlib import Path

# Create resolver for custom root
registry, context = create_default_resolver_system(Path("/custom/root"))

# Use resolver context for path operations
from dlkit.tools.config.environment import DLKitEnvironment
env = DLKitEnvironment(root_dir="/custom/root")

# All locations functions respect the custom root
from dlkit.tools.io import locations
custom_output = locations.output("results", env=env)
```

## Error Handling

**Exceptions Raised**:
- `FileNotFoundError`: Config file or data file doesn't exist
- `ConfigSectionError`: Required config section missing or invalid
- `ConfigValidationError`: Pydantic validation fails
- `ValueError`: Invalid file format, unsupported extension, or invalid path
- `TypeError`: Unexpected data type from loader
- `KeyError`: Missing required key in index split JSON

**Error Handling Pattern**:
```python
from dlkit.tools.io import load_config, ConfigSectionError, ConfigValidationError

try:
    settings = load_config("config.toml")
except FileNotFoundError:
    print("Config file not found")
except ConfigSectionError as e:
    print(f"Missing section: {e.section_name}")
    print(f"Available: {e.available_sections}")
except ConfigValidationError as e:
    print(f"Validation failed for {e.model_class}")
    print(f"Invalid data: {e.section_data}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

### Test Coverage
- Unit tests: `tests/tools/io/test_*.py`
- Partial config reading tests: `tests/tools/io/test_partial_config_reading.py`
- Split persistence tests: `tests/tools/io/test_split_persistence.py`
- Tilde expansion tests: `tests/core/datatypes/test_tilde_expansion.py`

### Key Test Scenarios
1. **Config loading**: TOML parsing, validation, section loading, path resolution
2. **Partial loading**: Efficiency, missing sections, registry defaults
3. **Path resolution**: Tilde expansion, relative paths, absolute paths, context precedence
4. **Test environment**: Artifact routing to tests/ directory
5. **Array loading**: Precision resolution, format support, dtype conversion
6. **Write round-trip**: Load → modify → save → load verification
7. **Split persistence**: Save → load → validate indices

### Fixtures Used
- `tmp_path` (pytest built-in): Temporary paths for test files
- Custom config fixtures in `tests/conftest.py`

### Cross-Platform Path Assertions
`str(Path(...))` on Windows returns backslashes; `.as_posix()` always returns forward slashes. Tests that mix the two in containment or equality checks will fail on Windows even when paths are correct.

Rules:
- Use `artifact_path.is_relative_to(base)` instead of `str(base) in str(artifact_path)`.
- When string containment is unavoidable: `Path(x).as_posix() in Path(y).as_posix()`.
- URI strings (`file://`, `sqlite://`) from `url_resolver` always use forward slashes — compare against `.as_posix()`, never against `str(Path(...))`.
- Never write `"foo/bar" in s or "foo\\\\bar" in s`; normalize before comparing.

## Performance Considerations
- Partial TOML parsing minimizes overhead for section-based loading
- Path resolution cached during config load (single computation per path)
- Frozen loader map prevents runtime modification overhead
- Test detection runs once and caches result
- Polars used for efficient tabular data loading
- Lazy imports in path resolution prevent circular dependencies

## Future Improvements / TODOs
- [ ] YAML configuration file support alongside TOML
- [ ] Async file loading for large datasets
- [ ] Configuration schema validation (JSON Schema)
- [ ] Incremental config updates (patch operations)
- [ ] Remote config loading (HTTP/S3)
- [ ] Config encryption for sensitive values
- [ ] Automatic backup before write operations
- [ ] Configuration diff utilities

## Related Modules
- `dlkit.tools.config`: Settings models and validation
- `dlkit.core.datatypes`: Data types (`IndexSplit`, `SecurePath`)
- `dlkit.interfaces.api.overrides`: Path override context
- `dlkit.interfaces.api.services.precision_service`: Precision management
- `dlkit.runtime.workflows`: Uses io for checkpoint and output management
- `dlkit.tools.utils`: Cross-cutting reflection, logging, and server-worker utilities

## Change Log
- **2026-03-03**: Sparse pack SOLID refactor — ISP protocol split (`SparseWriter`/`SparseLoader`/`SparseCodec`), `AbstractSparsePackReader` ABC for LSP, OCP format registry (`_registry.py`), `CooPackWriter` renamed to `CooPackCodec`, validate-before-write correctness fix, `_validate_coo_pack` made pure (no I/O), single load in `validate_sparse_pack`, `SparseFeature` removed from sparse package exports
- **2025-10-03**: Comprehensive documentation created
- **2024-10-02**: Test environment detection and artifact routing added
- **2024-09-30**: Precision service integration for array loading
- **2024-09-28**: Path provisioning separated from resolution (SRP)
- **2024-09-24**: Partial TOML parsing for efficient section loading
- **2024-09-20**: Section mapping registry with defaults
