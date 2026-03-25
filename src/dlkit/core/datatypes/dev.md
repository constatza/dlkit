# Core Datatypes Module

## Overview
The datatypes module provides foundational type definitions, dataset splitting utilities, and secure URL/path handling for DLKit. It implements Pydantic v2-based validation for hyperparameters, URLs, and paths with built-in tilde expansion and security checks. The module ensures type safety across configurations while providing flexible dataset splitting and indexing capabilities.

## Architecture & Design Patterns
- **Type Aliases**: Python 3.12+ type aliases for flexible hyperparameter definitions
- **Lazy Property Pattern**: SplitDataset computes splits on-demand from indices
- **Pydantic Validators**: BeforeValidator and AfterValidator for URL/path validation
- **Security-by-Design**: Strict tilde expansion and path traversal prevention
- **Immutable Splits**: IndexSplit as immutable Pydantic model for reproducibility
- **Subset View Pattern**: _SubsetDataset provides lightweight dataset views

Key architectural decisions:
- Tilde expansion happens before validation for consistent behavior
- Type aliases enable union types for simple values or optimization ranges
- IndexSplit separates index generation from dataset implementation
- URL validation uses Pydantic v2 primitives only (no urllib/httpx dependencies)
- Security checks prevent path traversal attacks in local paths

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `IntHyperparameter` | Type Alias | Integer hyperparameter (value or range) | N/A |
| `FloatHyperparameter` | Type Alias | Float hyperparameter (value or range) | N/A |
| `StrHyperparameter` | Type Alias | String hyperparameter (value or choices) | N/A |
| `Hyperparameter` | Type Alias | Union of all hyperparameter types | N/A |
| `SplitDataset` | Class | Dataset with train/val/test/predict splits | N/A |
| `IndexSplit` | Class | Immutable split indices | N/A |
| `Splitter` | Class | Generate random dataset splits | `IndexSplit` |
| `SimpleTildePath` | Annotated Type | Path string with tilde expansion | N/A |
| `SimpleMLflowURI` | Annotated Type | MLflow URI with tilde expansion | N/A |
| `MLflowServerUri` | Type Alias | HTTP(S) URL for MLflow server | N/A |
| `MLflowArtifactsUri` | Type Alias | Artifact destination URL/path | N/A |
| `CloudStorageUri` | Type Alias | Cloud storage URL (S3, GCS, etc.) | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_SubsetDataset` | Class | Subset view over base dataset | N/A |
| `expand_tilde_in_value` | Function | Expand tilde in strings before validation | `Any` |
| `tilde_expand_strict` | Function | Strict tilde expansion with security checks | `Any` |
| `local_path_security_check` | Function | Normalize and validate local paths | `Any` |
| `_validate_mlflow_backend` | Function | Validate MLflow backend store URLs | `Any` |
| `_validate_artifact_destination` | Function | Validate artifact storage destinations | `Any` |
| `_validate_mlflow_tracking` | Function | Validate MLflow tracking server URLs | `Any` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `NormalizerName` | N/A | Literal type for normalization layer names |

## Dependencies

### Internal Dependencies
- None (foundational module)

### External Dependencies
- `pydantic`: Type validation and model definition
- `pydantic_core`: URL parsing and validation primitives
- `torch`: Random permutation for split generation
- `pathlib`: Path manipulation and home directory resolution

## Key Components

### Component 1: `Hyperparameter` Type Aliases

**Purpose**: Define flexible hyperparameter types that support both single values and optimization ranges. Enables configuration to specify either fixed hyperparameters or search spaces for optimization.

**Type Definitions**:
- `IntHyperparameter = int | dict[str, int] | dict[str, tuple[int, ...]]`
- `FloatHyperparameter = float | dict[str, float | int] | dict[str, tuple[float, ...]]`
- `StrHyperparameter = str | dict[str, str] | dict[str, tuple[str, ...]]`
- `Hyperparameter = IntHyperparameter | FloatHyperparameter | StrHyperparameter`

**Example**:
```python
from dlkit.core.datatypes import IntHyperparameter, FloatHyperparameter

# Single value
batch_size: IntHyperparameter = 32

# Optimization range (min, max)
hidden_size: IntHyperparameter = {"suggest_int": (64, 512)}

# Float with search space
learning_rate: FloatHyperparameter = {"suggest_float": (1e-5, 1e-2)}

# Categorical choices
optimizer: StrHyperparameter = {"suggest_categorical": ("adam", "sgd", "adamw")}
```

**Implementation Notes**:
- Uses Python 3.12+ type alias syntax for cleaner definitions
- Dict keys indicate optimization method (e.g., "suggest_int", "suggest_float")
- Tuple values represent ranges or choices for hyperparameter search
- Type checker understands union semantics for static analysis

---

### Component 2: `SplitDataset[T]`

**Purpose**: Provides a unified interface for accessing train/validation/test/predict splits of a dataset. Supports both index-based splitting and custom dataset overrides.

**Constructor Parameters**:
- `dataset: T` - Base dataset to split
- `split: IndexSplit` - Index split defining train/val/test/predict indices

**Properties**:
- `raw: T` - Original unfiltered dataset
- `train: T` - Training split (computed from indices or custom)
- `validation: T` - Validation split (computed from indices or custom)
- `test: T` - Test split (computed from indices or custom)
- `predict: T` - Prediction split (defaults to raw if not specified)

**Example**:
```python
from dlkit.core.datatypes import SplitDataset, Splitter

# Create base dataset
dataset = MyDataset(data)

# Generate random split
splitter = Splitter(num_samples=len(dataset), test_ratio=0.2, val_ratio=0.1)
split = splitter.split()

# Create split dataset
split_dataset = SplitDataset(dataset, split)

# Access splits
train_data = split_dataset.train  # 70% of data
val_data = split_dataset.validation  # 10% of data
test_data = split_dataset.test  # 20% of data
predict_data = split_dataset.predict  # Full dataset if predict indices not set

# Override with custom split
split_dataset.train = CustomTrainDataset(data)
```

**Implementation Notes**:
- Generic type `[T]` ensures type safety across splits
- Lazy evaluation: splits computed on first access
- Custom splits take precedence over index-based computation
- `_normalize_indices` handles PyTorch tensor indices and tuples
- Predict split defaults to raw dataset when indices are None

---

### Component 3: `Splitter`

**Purpose**: Generate reproducible random train/validation/test splits using PyTorch's random permutation. Creates immutable `IndexSplit` objects with configurable split ratios.

**Constructor Parameters**:
- `num_samples: int` - Total number of samples in dataset
- `test_ratio: float` - Fraction of data for testing (0.0-1.0)
- `val_ratio: float` - Fraction of data for validation (0.0-1.0)

**Key Methods**:
- `split() -> IndexSplit` - Generate random split indices

**Returns**: `IndexSplit` - Immutable split with train/val/test/predict indices

**Raises**:
- `ValueError`: If `num_samples` is negative

**Example**:
```python
from dlkit.core.datatypes import Splitter

# Create splitter for 1000 samples (70% train, 10% val, 20% test)
splitter = Splitter(num_samples=1000, test_ratio=0.2, val_ratio=0.1)

# Generate split
split = splitter.split()

print(f"Train: {len(split.train)} samples")  # 700
print(f"Val: {len(split.validation)} samples")  # 100
print(f"Test: {len(split.test)} samples")  # 200
print(f"Predict: {len(split.predict) if split.predict else 0}")  # 0 (empty)

# Access indices
train_indices = split.train  # tuple[int, ...]
```

**Implementation Notes**:
- Uses `torch.randperm` for reproducible randomization
- Train count computed as remainder: `num_samples - test_count - val_count`
- Indices returned as tuples for immutability
- Predict indices include any remaining samples (usually empty)
- Split can be persisted as Pydantic model for reproducibility

---

### Component 4: `IndexSplit`

**Purpose**: Pydantic model representing immutable dataset split indices. Ensures split consistency across training runs and enables serialization/deserialization.

**Fields**:
- `train: tuple[int, ...]` - Training set indices
- `validation: tuple[int, ...]` - Validation set indices
- `test: tuple[int, ...]` - Test set indices
- `predict: tuple[int, ...] | None` - Prediction set indices (optional)

**Example**:
```python
from dlkit.core.datatypes import IndexSplit

# Create from explicit indices
split = IndexSplit(train=(0, 2, 4, 6, 8), validation=(1, 3), test=(5, 7, 9), predict=None)

# Serialize to dict
split_dict = split.model_dump()

# Deserialize from dict
loaded_split = IndexSplit(**split_dict)

# Save to JSON for reproducibility
with open("split.json", "w") as f:
    f.write(split.model_dump_json())
```

**Implementation Notes**:
- Pydantic BaseModel enables automatic validation
- Tuples enforce immutability (cannot modify indices)
- Supports JSON serialization for split persistence
- Optional predict field allows flexibility in dataset usage

---

### Component 5: `expand_tilde_in_value`

**Purpose**: Universal tilde (~) expansion for paths and URLs before Pydantic validation. Ensures consistent home directory resolution across all path/URL types.

**Parameters**:
- `value: Any` - Input value (processed only if string containing ~)

**Returns**: `Any` - Value with tilde expanded, or original value if no expansion needed

**Raises**:
- `ValueError`: If tilde appears in middle of path (security risk)

**Example**:
```python
from dlkit.core.datatypes.tilde_expansion import expand_tilde_in_value

# Plain path expansion
path = expand_tilde_in_value("~/project/data.csv")
# Result: "/home/user/project/data.csv"

# URL path expansion
url = expand_tilde_in_value("sqlite:///~/database/mlflow.db")
# Result: "sqlite:///home/user/database/mlflow.db"

# File URL expansion
file_url = expand_tilde_in_value("file:///~/docs/report.pdf")
# Result: "file:///home/user/docs/report.pdf"

# Invalid tilde position raises error
try:
    bad_path = expand_tilde_in_value("/some/~/path")  # ValueError
except ValueError as e:
    print(e)  # "Tilde must appear at the start of the path"

# Non-string values pass through unchanged
number = expand_tilde_in_value(42)  # Returns 42
```

**Implementation Notes**:
- Handles both plain paths and URLs with path components
- Uses `pathlib.Path.home()` for cross-platform home resolution
- Rejects tildes in middle of paths (potential security issue)
- URL paths exempt from middle-tilde restriction
- Normalizes path separators (backslash to forward slash)
- Uses `pydantic_core.Url` for URL parsing (no urllib dependency)

---

### Component 6: URL Type Hierarchy

**Purpose**: Comprehensive URL and path validation using Pydantic v2 with security checks and tilde expansion. Supports HTTP, file, SQLite, cloud storage, database, and Databricks URLs.

**Basic URL Types**:
- `HttpUrl` - HTTP/HTTPS URLs for web services
- `FileUrl` - File protocol URLs (file:///)
- `SQLiteUrl` - SQLite database URLs (sqlite:///)
- `CloudStorageUrl` - Cloud storage (s3://, gs://, wasbs://, hdfs://)
- `DbUrl` - SQL database URLs (postgresql, mysql, mssql, oracle)
- `DatabricksUrl` - Databricks URLs (databricks://profile:workspace)

**Composite MLflow Types**:
- `MLflowBackendUrl` - Any valid MLflow backend store URL
- `MLflowTrackingUrl` - MLflow tracking server URL
- `ArtifactDestination` - Artifact storage location (URL or local path)
- `LocalPath` - Secure local path with strict validation

**Example**:
```python
from pydantic import BaseModel
from dlkit.core.datatypes.urls import (
    HttpUrl,
    SQLiteUrl,
    MLflowBackendUrl,
    ArtifactDestination,
    LocalPath,
)


class MLflowConfig(BaseModel):
    tracking_uri: HttpUrl  # "http://localhost:5000"
    backend_store_uri: SQLiteUrl  # "sqlite:///~/mlflow/mlflow.db"
    artifact_location: ArtifactDestination  # "~/mlflow/artifacts" or "s3://bucket/path"


# Automatic tilde expansion
config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    backend_store_uri="sqlite:///~/mlflow.db",
    artifact_location="~/artifacts",
)
# backend_store_uri expanded to "sqlite:///home/user/mlflow.db"
# artifact_location expanded to "/home/user/artifacts"

# Validation catches invalid URLs
try:
    bad_config = MLflowConfig(
        tracking_uri="not-a-url",
        backend_store_uri="sqlite:///",  # Empty path
        artifact_location="~/path",
    )
except ValueError as e:
    print(e)  # Validation error details

# S3 bucket validation
from dlkit.core.datatypes.urls import CloudStorageUrl
from pydantic import ValidationError

try:
    # Invalid bucket name (uppercase)
    bad_s3 = CloudStorageUrl("s3://MyBucket/path")
except ValidationError:
    pass  # Raised due to uppercase in bucket name

# Valid cloud URLs
s3_url = CloudStorageUrl("s3://my-bucket/artifacts/")
gs_url = CloudStorageUrl("gs://my-project-bucket/data/")
hdfs_url = CloudStorageUrl("hdfs:///user/data/")
```

**Implementation Notes**:
- All types use `BeforeValidator` for tilde expansion
- `AfterValidator` enforces scheme-specific rules
- SQLite URLs require triple-slash and non-empty path
- S3 bucket names validated against AWS naming rules
- Databricks URLs use custom regex validation
- Composite types (MLflowBackendUrl) try multiple adapters in sequence
- No urllib/httpx dependencies - uses pydantic_core.Url only
- Type adapters enable reusable validation logic

---

### Component 7: `SimpleTildePath` and `SimpleMLflowURI`

**Purpose**: Lightweight annotated types for paths and URIs with tilde expansion but minimal security checks. Designed for simple use cases where strict validation is handled downstream.

**Type Definitions**:
```python
SimpleTildePath = Annotated[
    str,
    BeforeValidator(expand_tilde_in_value),
    Field(description="Path string with pre-validation ~ expansion (no extra checks)"),
]

SimpleMLflowURI = Annotated[
    str,
    BeforeValidator(expand_tilde_in_value),
    Field(description="Generic MLflow-style URI with pre-validation ~ expansion only"),
]
```

**Example**:
```python
from pydantic import BaseModel
from dlkit.core.datatypes import SimpleTildePath, SimpleMLflowURI


class SimpleConfig(BaseModel):
    data_path: SimpleTildePath
    mlflow_uri: SimpleMLflowURI


# Tilde expansion happens automatically
config = SimpleConfig(data_path="~/data/train.csv", mlflow_uri="sqlite:///~/mlflow.db")

print(config.data_path)  # "/home/user/data/train.csv"
print(config.mlflow_uri)  # "sqlite:///home/user/mlflow.db"
```

**Implementation Notes**:
- Minimal validation overhead - just tilde expansion
- Validation deferred to downstream consumers
- Useful when configuration keys are user-defined
- Not recommended for security-critical paths
- Use `LocalPath` or `SecurePath` for stricter validation

---

### Component 8: `_SubsetDataset`

**Purpose**: Lightweight view over a base dataset using precomputed indices. Provides PyTorch Dataset interface without copying data.

**Constructor Parameters**:
- `base_dataset` - Original dataset to create subset from
- `indices` - List/tuple of indices to include in subset

**Methods**:
- `__len__() -> int` - Number of samples in subset
- `__getitem__(i: int)` - Get sample at subset index i

**Example**:
```python
from dlkit.core.datatypes.dataset import _SubsetDataset


# Base dataset
class MyDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# Create subset
dataset = MyDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
subset = _SubsetDataset(dataset, [0, 2, 4, 6, 8])

print(len(subset))  # 5
print(subset[0])  # 0 (first element in subset)
print(subset[2])  # 4 (third element in subset)
```

**Implementation Notes**:
- Zero-copy design - stores reference to base dataset
- Indices converted to list for consistent behavior
- Compatible with PyTorch DataLoader
- Used internally by SplitDataset for split views
- Handles both list and tuple indices
- Supports advanced indexing with PyTorch tensors

## Usage Patterns

### Common Use Case 1: Configuration with Hyperparameters
```python
from pydantic import BaseModel
from dlkit.core.datatypes import IntHyperparameter, FloatHyperparameter


class ModelConfig(BaseModel):
    # Fixed hyperparameters
    batch_size: IntHyperparameter = 32

    # Hyperparameters for optimization
    hidden_size: IntHyperparameter = {"suggest_int": (64, 512)}
    learning_rate: FloatHyperparameter = {"suggest_float": (1e-5, 1e-2)}


# Training with fixed values
config = ModelConfig(batch_size=32, hidden_size=128, learning_rate=0.001)

# Optimization configuration
opt_config = ModelConfig(
    batch_size=32,
    hidden_size={"suggest_int": (64, 512)},
    learning_rate={"suggest_loguniform": (1e-5, 1e-2)},
)
```

### Common Use Case 2: Dataset Splitting
```python
from dlkit.core.datatypes import SplitDataset, Splitter
from torch.utils.data import DataLoader

# Load dataset
dataset = load_my_dataset()

# Create reproducible split
splitter = Splitter(num_samples=len(dataset), test_ratio=0.15, val_ratio=0.15)
split = splitter.split()

# Create split dataset
split_dataset = SplitDataset(dataset, split)

# Create dataloaders
train_loader = DataLoader(split_dataset.train, batch_size=32, shuffle=True)
val_loader = DataLoader(split_dataset.validation, batch_size=32)
test_loader = DataLoader(split_dataset.test, batch_size=32)

# Save split for reproducibility
import json

with open("split.json", "w") as f:
    json.dump(split.model_dump(), f)
```

### Common Use Case 3: MLflow Configuration with URLs
```python
from pydantic import BaseModel
from dlkit.core.datatypes import SimpleTildePath, SimpleMLflowURI


class MLflowSettings(BaseModel):
    tracking_uri: SimpleMLflowURI
    backend_store_uri: SimpleMLflowURI
    artifact_location: SimpleTildePath


# Configuration with tilde expansion
settings = MLflowSettings(
    tracking_uri="http://localhost:5000",
    backend_store_uri="sqlite:///~/mlflow/mlflow.db",
    artifact_location="~/mlflow/artifacts",
)

# Paths automatically expanded
print(settings.backend_store_uri)  # "sqlite:///home/user/mlflow/mlflow.db"
print(settings.artifact_location)  # "/home/user/mlflow/artifacts"
```

### Common Use Case 4: Custom Split Override
```python
from dlkit.core.datatypes import SplitDataset, Splitter

# Create base split
dataset = load_dataset()
splitter = Splitter(num_samples=len(dataset), test_ratio=0.2, val_ratio=0.1)
split = splitter.split()
split_dataset = SplitDataset(dataset, split)

# Override train split with augmented version
augmented_train = AugmentedDataset(split_dataset.train)
split_dataset.train = augmented_train

# Other splits remain index-based
val_loader = DataLoader(split_dataset.validation, batch_size=32)
test_loader = DataLoader(split_dataset.test, batch_size=32)
```

## Error Handling

**Exceptions Raised**:
- `ValueError`: Invalid hyperparameter ranges, negative num_samples, tilde in middle of path
- `pydantic.ValidationError`: Invalid URL formats, scheme mismatches, bucket naming violations
- `pydantic_core.ValidationError`: Core validation errors from URL parsing

**Error Handling Pattern**:
```python
from pydantic import ValidationError
from dlkit.core.datatypes import Splitter, MLflowBackendUrl

# Handle splitter errors
try:
    splitter = Splitter(num_samples=-10, test_ratio=0.2, val_ratio=0.1)
except ValueError as e:
    print(f"Invalid splitter config: {e}")

# Handle URL validation errors
from pydantic import BaseModel


class Config(BaseModel):
    backend_uri: MLflowBackendUrl


try:
    config = Config(backend_uri="invalid://url")
except ValidationError as e:
    print(f"URL validation failed: {e}")
    # Fall back to default
    config = Config(backend_uri="sqlite:///mlflow.db")
```

## Testing

### Test Coverage
- Unit tests:
  - `tests/core/datatypes/test_tilde_expansion.py`
  - `tests/core/test_shape_specs.py`
- Integration tests:
  - `tests/integration/test_transforms_persistence_and_inference.py`

### Key Test Scenarios
1. **Hyperparameter type validation**: Verify int/float/str unions work correctly
2. **Split generation**: Test reproducibility and ratio calculations
3. **Tilde expansion**: Path and URL expansion with security checks
4. **URL validation**: All supported URL schemes and error cases
5. **Subset indexing**: Correct index remapping and length calculation
6. **Custom split overrides**: Property setters preserve override behavior
7. **Pydantic serialization**: IndexSplit round-trip through JSON

### Fixtures Used
- `tmp_path` (pytest built-in): Temporary directories for path tests
- Dataset fixtures from `tests/conftest.py` for split testing

## Performance Considerations
- Lazy split computation - only create subset views when accessed
- Zero-copy subset design - no data duplication
- Tilde expansion cached by Pydantic after first validation
- URL validation uses TypeAdapter caching for efficiency
- IndexSplit tuples prevent accidental mutation overhead
- Path normalization uses string operations (no pathlib overhead)

## Future Improvements / TODOs
- [ ] Add stratified splitting for classification datasets
- [ ] Support custom random seeds for Splitter reproducibility
- [ ] Add k-fold cross-validation split generator
- [ ] Implement split visualization utilities
- [ ] Add group-based splitting (e.g., by patient ID)
- [ ] Support time-series aware splitting
- [ ] Add split statistics (class distribution, size validation)
- [ ] Implement split merging/combining utilities

## Related Modules
- `dlkit.core.datasets`: Uses SplitDataset for train/val/test access
- `dlkit.core.datamodules`: Integrates Splitter for data loading
- `dlkit.tools.config`: Uses hyperparameter types in settings
- `dlkit.tools.io.split_provider`: Persists and loads IndexSplit objects
- `dlkit.runtime.workflows.strategies.optuna`: Uses hyperparameter ranges for optimization

## Change Log
- **2025-10-03**: Comprehensive documentation with enriched docstrings
- **2024-10-02**: Migrated to Pydantic v2 URL validation
- **2024-09-30**: Added strict tilde expansion with security checks
- **2024-09-24**: Introduced hyperparameter type aliases for optimization
- **2024-09-20**: Refactored SplitDataset with lazy property evaluation
