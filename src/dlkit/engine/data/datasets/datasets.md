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
- zarr pack directories - dense matrix packs read via `Feature()` / `PathFeature` by pointing at the zarr directory

**Key Features:**
- Default mode loads arrays into memory upfront for simple workflows
- Configure arbitrary features and targets
- Named `Feature` entries double as model-dispatch keys when `model_input=true`
- Automatic precision handling via `PrecisionService`
- Support for in-memory arrays (testing/programmatic usage)

### GraphDataset

Specialized dataset for graph-structured data (see `graph.py`).
Import graph datasets from `dlkit.engine.data.datasets.graph`; the broad
`dlkit.engine.data.datasets` package surface stays graph-free.

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

When a feature has `model_input=True` (the default), DLKit's standard Lightning
wrapper dispatches it to `model.forward()` by keyword, using `Feature.name` as
the parameter name. For example, `Feature(name="x", ...)` binds as
`model(x=tensor)`. Set `model_input=False` to keep a feature in the batch
without sending it to the model.

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
[data]
name = "dlkit.engine.data.datasets.flexible.FlexibleDataset"

# Load features and targets from same NPZ file
[[data.features]]
name = "features"  # Used as array key
path = "data.npz"

[[data.features]]
name = "latent"    # Loads different array from same file
path = "data.npz"

[[data.targets]]
name = "targets"
path = "data.npz"
```

**Important:** The entry name must match an array key in the `.npz` file.
If the same feature also uses `model_input=true`, that name must also match the
corresponding `model.forward()` parameter name.

### Mixed File Formats

```python
features = [
    Feature(name="x", path="features.npy"),  # Load from .npy
    Feature(name="embeddings", path="data.npz"),  # Load "embeddings" from .npz
]
targets = [Target(name="y", path="labels.pt")]  # Load from PyTorch file
dataset = FlexibleDataset(features=features, targets=targets)
```

### Matrix Context Features

Use zarr pack directories for per-sample matrices consumed by custom losses.

#### Programmatic API

```python
from pathlib import Path

from dlkit.infrastructure.config.data_entries import Feature, Target

features = [
    Feature(name="x", path="features.npy"),
    Feature(name="K", path=Path("/data/stiffness_matrices"), model_input=False, loss_input="K"),
]
targets = [Target(name="y", path="labels.npy")]
dataset = FlexibleDataset(features=features, targets=targets)
```

#### TOML Configuration

```toml
[[data.features]]
name = "K"
path = "/data/stiffness_matrices"   # zarr pack directory
model_input = false
loss_input = "K"
```

Dense matrices are read from the zarr pack on each sample access and stored in
the dataset feature TensorDict.

This keeps `"K"` available to losses and metrics via the batch while excluding
it from `model.forward()` because `model_input=false`.

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
[data]
name = "dlkit.engine.data.datasets.flexible.FlexibleDataset"

# Single-file features
[[data.features]]
name = "x"
path = "features.npy"

# NPZ multi-array features
[[data.features]]
name = "embeddings"
path = "data.npz"  # Loads array "embeddings" from data.npz

[[data.features]]
name = "latent"
path = "data.npz"  # Loads array "latent" from same file

# Targets
[[data.targets]]
name = "y"
path = "labels.npy"
```

If these features are consumed by the standard Lightning wrapper, the model
must declare matching `forward()` parameters, for example
`forward(self, x, embeddings, latent)` for the configuration above.

### Full Workflow Integration (Programmatic)

Using NPZ files with the complete DLKit training workflow:

```python
from dlkit.interfaces.api import train
from dlkit.infrastructure.config import DataSettings, load_job
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.core.updater import update_settings

# Load base configuration from TOML
config = load_job("config.toml")

# Inject dataset with NPZ files programmatically
update_settings(
    config,
    {
        "data": DataSettings(
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

**Not** a universal dataset with pluggable loaders. Each dataset is optimized for its use case.

### Extending File Format Support

To add new array formats (HDF5, Zarr, Parquet):

1. Add loader function to `infrastructure/io/arrays.py`:
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

- **Array I/O**: `src/dlkit/infrastructure/io/arrays.py` - Low-level array loading functions
- **Data Entries**: `src/dlkit/infrastructure/config/data_entries.py` - Feature/Target configuration
- **Precision System**: `src/dlkit/infrastructure/precision/precision.md` - Dtype management
- **Architecture Guide**: `AGENTS.md` in project root - Full system architecture
