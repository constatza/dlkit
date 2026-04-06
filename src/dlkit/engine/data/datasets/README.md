# Runtime Datasets

This directory contains the runtime-owned dataset implementations for loading
and managing training and inference data.

## Available Datasets

### FlexibleDataset

The primary dataset for array-based data with flexible feature/target configuration.

**Supported File Formats:**
- `.npy` - NumPy single array files
- `.npz` - NumPy multi-array archive files
- `.pt` / `.pth` - PyTorch tensor files
- `.txt` / `.csv` - Text-based array files
- sparse pack directories - COO payload packs (default files: `indices.npy`, `values.npy`, `nnz_ptr.npy`, `values_scale.npy`; customizable via `SparseFeature.files`)

**Key Features:**
- Default mode loads arrays into memory upfront for simple workflows
- Optional `memmap_cache_dir` builds/uses disk-backed cache to reduce peak RAM usage
- Configure arbitrary features and targets
- Automatic precision handling via `PrecisionService`
- Support for in-memory arrays (testing/programmatic usage)

### GraphDataset

Specialized dataset for graph-structured data (see `graph.py`).

### ForecastingDataset

Specialized dataset for temporal/sequential data with windowing support (see `timeseries.py`).

---

## FlexibleDataset Usage

### Basic Usage

```python
from dlkit.engine.data.datasets.flexible import FlexibleDataset
from dlkit.infrastructure.config.data_entries import Feature, Target

# Single .npy files
features = [Feature(name="x", path="features.npy")]
targets = [Target(name="y", path="labels.npy")]
dataset = FlexibleDataset(features=features, targets=targets)

# Access samples
sample = dataset[0]
# {'x': tensor([...]), 'y': tensor([...])}
```

### NPZ Multi-Array Files

**How It Works:**
The entry `name` is used as the array key to select specific arrays from `.npz` files.

#### Programmatic API

```python
from dlkit.engine.data.datasets.flexible import FlexibleDataset
from dlkit.infrastructure.config.data_entries import Feature, Target

# data.npz contains arrays: "features", "targets", "latent"

# Load features and targets from same NPZ file
features = [Feature(name="features", path="data.npz")]
targets = [Target(name="targets", path="data.npz")]
dataset = FlexibleDataset(features=features, targets=targets)

# Access samples
sample = dataset[0]
# {'features': tensor([...]), 'targets': tensor([...])}

# Load multiple features from same NPZ
features = [Feature(name="features", path="data.npz"), Feature(name="latent", path="data.npz")]
dataset = FlexibleDataset(features=features)

sample = dataset[0]
# {'features': tensor([...]), 'latent': tensor([...])}
```

#### TOML Configuration

```toml
[DATASET]
name = "dlkit.engine.data.datasets.flexible.FlexibleDataset"

# Load features and targets from same NPZ file
[[DATASET.features]]
name = "features"  # Used as array key
path = "data.npz"

[[DATASET.features]]
name = "latent"    # Loads different array from same file
path = "data.npz"

[[DATASET.targets]]
name = "targets"
path = "data.npz"
```

**Important:** The entry name must match an array key in the `.npz` file.

### Mixed File Formats

```python
features = [
    Feature(name="x", path="features.npy"),  # Load from .npy
    Feature(name="embeddings", path="data.npz"),  # Load "embeddings" from .npz
]
targets = [Target(name="y", path="labels.pt")]  # Load from PyTorch file
dataset = FlexibleDataset(features=features, targets=targets)
```

### Sparse Matrix Context Features

Use sparse pack directories for per-sample matrices consumed by custom losses.

#### Programmatic API (explicit `SparseFeature`)

```python
from dlkit.infrastructure.config.data_entries import Feature, SparseFeature, Target

features = [
    Feature(name="x", path="features.npy"),
    SparseFeature(name="matrix", path="matrix_pack", model_input=False, loss_input="matrix"),
]
targets = [Target(name="y", path="labels.npy")]
dataset = FlexibleDataset(features=features, targets=targets)
```

#### Programmatic API (custom sparse payload names)

```python
from dlkit.infrastructure.config.data_entries import Feature, SparseFeature, SparseFilesConfig, Target

features = [
    Feature(name="x", path="features.npy"),
    SparseFeature(
        name="matrix",
        path="matrix_pack",
        model_input=False,
        loss_input="matrix",
        denormalize=True,  # apply values_scale during read
        files=SparseFilesConfig(
            indices="row_index.npy",
            values="entries.npy",
            nnz_ptr="offsets.npy",
            values_scale="scale.npy",
        ),
    ),
]
targets = [Target(name="y", path="labels.npy")]
dataset = FlexibleDataset(features=features, targets=targets)
```

#### TOML / generic `Feature` path auto-detection

If `path` points to a directory containing sparse payload files, `FlexibleDataset`
auto-detects it as a sparse pack and loads sparse tensors.

```toml
[[DATASET.features]]
name = "matrix"
path = "matrix_pack"    # directory with sparse payload files
model_input = false
loss_input = "matrix"

# Optional: override sparse payload filenames
files = { indices = "row_index.npy", values = "entries.npy", nnz_ptr = "offsets.npy", values_scale = "scale.npy" }

# Optional: apply values_scale when materializing sparse tensors
denormalize = true
```

Sparse tensors are stored in the dataset feature TensorDict and collated via
`torch.stack` on same-shape COO tensors.

### In-Memory Data (Testing/Programmatic)

```python
import numpy as np

# Use value-based entries for in-memory arrays
features = [Feature(name="x", value=np.ones((100, 10)))]
targets = [Target(name="y", value=np.zeros((100, 1)))]
dataset = FlexibleDataset(features=features, targets=targets)
```

### Precision Control

```python
from dlkit.infrastructure.precision import precision_override, PrecisionStrategy

# Load data in specific precision
with precision_override(PrecisionStrategy.FULL_64):
    dataset = FlexibleDataset(features=features, targets=targets)
    # All tensors will be float64
```

### TOML Configuration

```toml
[DATASET]
name = "dlkit.engine.data.datasets.flexible.FlexibleDataset"

# Single-file features
[[DATASET.features]]
name = "x"
path = "features.npy"

# NPZ multi-array features
[[DATASET.features]]
name = "embeddings"
path = "data.npz"  # Loads array "embeddings" from data.npz

[[DATASET.features]]
name = "latent"
path = "data.npz"  # Loads array "latent" from same file

# Targets
[[DATASET.targets]]
name = "y"
path = "labels.npy"
```

### Full Workflow Integration (Programmatic)

Using NPZ files with the complete DLKit training workflow:

```python
from dlkit.interfaces.api import train
from dlkit.infrastructure.config import load_settings, DatasetSettings
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.core.updater import update_settings

# Load base configuration from TOML
config = load_settings("config.toml", inference=False)

# Inject dataset with NPZ files programmatically
update_settings(
    config,
    {
        "DATASET": DatasetSettings(
            name="dlkit.engine.data.datasets.flexible.FlexibleDataset",
            features=(
                Feature(name="features", path="data.npz"),
                Feature(name="latent", path="data.npz"),
            ),
            targets=(Target(name="targets", path="data.npz"),),
        )
    },
)

# Train with NPZ data
result = train(config, epochs=100)
print(f"Final metrics: {result.metrics}")
```

### Inference with NPZ Files

```python
from dlkit import load_model
import numpy as np

# Create NPZ file for inference
test_data = {
    "features": np.random.randn(100, 64).astype(np.float32),
    "latent": np.random.randn(100, 32).astype(np.float32),
}
np.savez("test_data.npz", **test_data)

# Load predictor (trained on NPZ data)
predictor = load_model("model.ckpt", device="cuda")

# Option 1: Load and pass arrays
npz = np.load("test_data.npz")
result = predictor.predict({"features": npz["features"], "latent": npz["latent"]})

# Option 2: Pass file paths (if supported by predictor)
# predictor handles loading internally
predictions = result.predictions

predictor.unload()
```

---

## NPZ File Format Details

### Creating NPZ Files

```python
import numpy as np

# Create multi-array NPZ file
features = np.random.randn(1000, 64).astype(np.float32)
targets = np.random.randint(0, 10, (1000, 1))
latent = np.random.randn(1000, 32).astype(np.float32)

# Save with named arrays
np.savez("data.npz", features=features, targets=targets, latent=latent)

# Or use savez_compressed for smaller files
np.savez_compressed("data.npz", features=features, targets=targets, latent=latent)
```

### Inspecting NPZ Files

```python
import numpy as np

# Load NPZ to see available arrays
npz = np.load("data.npz")
print("Available arrays:", list(npz.keys()))
# Available arrays: ['features', 'targets', 'latent']

# Check shapes
for key in npz.keys():
    print(f"{key}: {npz[key].shape}")
# features: (1000, 64)
# targets: (1000, 1)
# latent: (1000, 32)
```

### Best Practices

1. **Consistent Array Names:** Use consistent naming across NPZ files
   ```python
   # Good: All files use "features" and "targets"
   np.savez("train.npz", features=X_train, targets=y_train)
   np.savez("val.npz", features=X_val, targets=y_val)
   ```

2. **Match Entry Names to Array Keys:**
   ```python
   # Entry name must match NPZ array key
   features = [Feature(name="features", path="data.npz")]  # ✅ Correct
   features = [Feature(name="x", path="data.npz")]  # ❌ Fails if "x" not in npz
   ```

3. **Use Compressed NPZ for Large Files:**
   ```python
   # Regular: Fast but large
   np.savez("data.npz", features=features)

   # Compressed: Slower but smaller
   np.savez_compressed("data.npz", features=features)
   ```

4. **Consistent First Dimension:**
   ```python
   # All arrays must have same number of samples
   features = np.random.randn(1000, 64)
   targets = np.random.randn(1000, 1)
   np.savez("data.npz", features=features, targets=targets)  # ✅ Both have 1000 samples
   ```

---

## Architecture Notes

### Why Separate Dataset Classes?

DLKit follows the **specialized dataset pattern** (like PyTorch/torchvision):
- `FlexibleDataset`: Upfront array loading (NPY, NPZ, PT)
- `GraphDataset`: Graph-structured data
- `TimeSeriesDataset`: Temporal/sequential data

**Not** a universal dataset with pluggable loaders. Each dataset is optimized for its use case.

### Extending File Format Support

To add new array formats (HDF5, Zarr, Parquet):

1. Add loader function to `tools/io/arrays.py`:
   ```python
   def _load_hdf5(path: Path, dataset_path: str, **kwargs) -> np.ndarray:
       import h5py

       with h5py.File(path, "r") as f:
           return f[dataset_path][:]
   ```

2. Register in `_LOADER_MAP`:
   ```python
   _LOADER_MAP = MappingProxyType({
       ".npy": np.load,
       ".npz": _load_npz,
       ".h5": _load_hdf5,  # New format
       # ...
   })
   ```

3. FlexibleDataset automatically supports the new format!

### For Different Loading Patterns

If you need **index-based loading** (e.g., directory of images where each file is one sample):
- Create a specialized dataset (e.g., `ImageDataset`)
- `FlexibleDataset` is optimized for upfront array loading
- Don't try to make one dataset do everything

See `LOADER_INTERFACE_ANALYSIS.md` in project root for detailed architectural reasoning.

---

## See Also

- **Array I/O**: `src/dlkit/tools/io/arrays.py` - Low-level array loading functions
- **Data Entries**: `src/dlkit/tools/config/data_entries.py` - Feature/Target configuration
- **Precision System**: `src/dlkit/interfaces/api/services/precision_service.py` - Dtype management
- **Architecture Guide**: `CLAUDE.md` in project root - Full system architecture
