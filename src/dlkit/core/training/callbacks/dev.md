# Training Callbacks Module

## Overview
The training callbacks module provides PyTorch Lightning callbacks for experiment tracking, prediction persistence, and metric logging. It implements specialized callbacks that integrate with MLflow for experiment tracking and enable efficient storage of model predictions as NumPy arrays.

## Architecture & Design Patterns
- **Callback Pattern**: Hooks into PyTorch Lightning training lifecycle events
- **Accumulator Pattern**: NumpyWriter accumulates batch predictions before writing
- **Adapter Pattern**: MLflowEpochLogger adapts Lightning metrics to MLflow format
- **Fail-Safe Design**: Callbacks log errors but don't crash training
- **Lazy Initialization**: Output directories created only when needed

Key architectural decisions:
- Callbacks are orthogonal to training logic (can be added/removed freely)
- MLflow integration through adapter pattern for loose coupling
- Predictions accumulated in memory before disk write for efficiency
- Epoch-based logging preferred over step-based for clearer metrics
- Error handling prevents callback failures from interrupting training

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `MLflowEpochLogger` | Class | Log metrics to MLflow with epoch numbers | N/A |
| `NumpyWriter` | Class | Save predictions to NumPy arrays | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_log_metrics` | Method | Filter and log metrics by prefix | `None` |
| `_store_predictions` | Method | Accumulate batch predictions | `None` |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.servers`: MLflow adapter for artifact logging (`create_mlflow_adapter`)

### External Dependencies
- `lightning.pytorch`: Callback base class and trainer integration
- `loguru`: Structured logging
- `pydantic`: Directory path validation
- `numpy`: Array serialization
- `torch`: Tensor handling

## Key Components

### Component 1: `MLflowEpochLogger`

**Purpose**: Callback that logs training, validation, and test metrics to MLflow using epoch numbers instead of global steps. Ensures MLflow plots display epochs on the x-axis for easier interpretation.

**Constructor Parameters**:
- `run_context: Any` - MLflow run context with `log_metrics(metrics, step)` method

**Key Methods**:
- `on_train_epoch_end(trainer, pl_module) -> None` - Log training metrics at epoch end
- `on_validation_epoch_end(trainer, pl_module) -> None` - Log validation metrics at epoch end
- `on_test_epoch_end(trainer, pl_module) -> None` - Log test metrics at epoch end
- `_log_metrics(trainer, prefix) -> None` - Internal method to filter and log metrics

**Example**:
```python
from lightning.pytorch import Trainer
from dlkit.core.training.callbacks import MLflowEpochLogger
from dlkit.runtime.workflows.strategies.tracking import MLflowTracker

# Setup MLflow tracking
tracker = MLflowTracker()
tracker.setup_mlflow_config(settings.MLFLOW)

with tracker:
    with tracker.create_run(experiment_name="training") as run_context:
        # Create callback with run context
        epoch_logger = MLflowEpochLogger(run_context)

        # Add to trainer callbacks
        trainer = Trainer(max_epochs=10, callbacks=[epoch_logger])

        # Metrics automatically logged with epoch numbers
        trainer.fit(model, datamodule)
        # MLflow UI shows: epoch 0, 1, 2, ... on x-axis
```

**Implementation Notes**:
- Hooks into `on_train_epoch_end`, `on_validation_epoch_end`, `on_test_epoch_end`
- Extracts metrics from `trainer.callback_metrics` (Lightning's accumulated metrics)
- Filters metrics by prefix ("train", "val", "test") to avoid duplicates
- Always includes learning rate ("lr") regardless of prefix
- Converts PyTorch tensors to float before logging
- Logs all filtered metrics with `epoch` as step parameter
- Fail-safe: catches exceptions and logs warnings without crashing
- Uses `logger.debug` for successful logging to reduce noise

---

### Component 2: `NumpyWriter`

**Purpose**: Callback to accumulate model predictions during inference and persist them as NumPy `.npy` files. Supports multiple outputs per batch and automatic MLflow artifact logging.

**Constructor Parameters**:
- `output_dir: DirectoryPath | None = None` - Output directory for predictions (defaults to MLflow artifact URI or `./predictions`)
- `filenames: Sequence[str] = ("predictions",)` - Filenames for output arrays (one per prediction key)

**Key Methods**:
- `on_predict_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None` - Accumulate batch predictions
- `on_predict_epoch_end(trainer, pl_module) -> None` - Concatenate and save all predictions
- `_store_predictions(key, value) -> None` - Store single prediction tensor

**Returns**: None (writes files as side effect)

**Raises**:
- `OSError`: When file writing fails (logged but not propagated)

**Example**:
```python
from pathlib import Path
from lightning.pytorch import Trainer
from dlkit.core.training.callbacks import NumpyWriter
import numpy as np

# Scenario 1: Basic usage with single output
writer = NumpyWriter(output_dir=Path("./predictions"))

trainer = Trainer(callbacks=[writer])
predictions = trainer.predict(model, datamodule)
# Saves: ./predictions/predictions.npy

# Scenario 2: Multiple outputs with custom filenames
writer = NumpyWriter(
    output_dir=Path("./results"), filenames=("reconstructions", "latent_codes", "attention_weights")
)


# Model returns dict in predict_step
class MyModel(pl.LightningModule):
    def predict_step(self, batch, batch_idx):
        return {
            "output1": reconstructions,  # Saved as reconstructions.npy
            "output2": latent_codes,  # Saved as latent_codes.npy
            "output3": attention,  # Saved as attention_weights.npy
        }


trainer = Trainer(callbacks=[writer])
trainer.predict(model, datamodule)

# Load predictions
reconstructions = np.load("./results/reconstructions.npy")
latent = np.load("./results/latent_codes.npy")

# Scenario 3: Automatic MLflow integration
# When output_dir=None and MLflow run is active
writer = NumpyWriter()  # Uses MLflow artifact URI
# Predictions saved to MLflow run artifacts and logged automatically
```

**Implementation Notes**:
- Accumulates predictions in `_predictions` dict mapping keys to tensor lists
- Supports multiple output formats:
  - `Mapping[str, Tensor]`: Uses dict keys as filenames (or custom filenames)
  - `list | tuple`: Uses filenames from constructor
  - `Tensor`: Single output with first filename
- Concatenates tensors along batch dimension (dim=0) at epoch end
- Converts to NumPy array and saves as `.npy` format
- MLflow integration:
  - If `output_dir=None`, queries MLflow adapter for artifact URI
  - Sets `_use_mlflow=True` when MLflow active
  - Logs saved files as MLflow artifacts in "predictions" directory
  - Falls back to `./predictions` if MLflow unavailable
- Parent directory created automatically with `parents=True, exist_ok=True`
- Validates directory paths with Pydantic `DirectoryPath` type
- Fail-safe error handling: logs errors but continues processing other keys
- Uses `logger.debug` for success, `logger.error` for failures

---

### Component 3: Callback Lifecycle Integration

**Purpose**: Understanding how callbacks integrate with PyTorch Lightning training lifecycle for proper usage and debugging.

**Callback Hooks Used**:
- `on_train_epoch_end` - After all training batches processed
- `on_validation_epoch_end` - After all validation batches processed
- `on_test_epoch_end` - After all test batches processed
- `on_predict_batch_end` - After each prediction batch
- `on_predict_epoch_end` - After all prediction batches

**Execution Order**:
```
Training Loop:
1. trainer.fit() called
2. For each epoch:
   a. Training batches processed
   b. on_train_epoch_end() → MLflowEpochLogger logs train metrics
   c. Validation batches processed
   d. on_validation_epoch_end() → MLflowEpochLogger logs val metrics
3. Training complete

Prediction Loop:
1. trainer.predict() called
2. For each batch:
   a. Model predict_step() returns outputs
   b. on_predict_batch_end() → NumpyWriter accumulates
3. on_predict_epoch_end() → NumpyWriter saves files
```

**Example**:
```python
from lightning.pytorch import Trainer
from dlkit.core.training.callbacks import MLflowEpochLogger, NumpyWriter

# Combine callbacks for training and prediction
epoch_logger = MLflowEpochLogger(run_context)
numpy_writer = NumpyWriter(output_dir=Path("./predictions"))

# Training with metric logging
trainer = Trainer(max_epochs=10, callbacks=[epoch_logger])
trainer.fit(model, datamodule)

# Prediction with output saving
predict_trainer = Trainer(callbacks=[numpy_writer])
predict_trainer.predict(model, datamodule)
```

**Implementation Notes**:
- Callbacks receive trainer and module references for context
- Can access `trainer.current_epoch`, `trainer.callback_metrics`, etc.
- Multiple callbacks execute in registration order
- Callback exceptions caught by Lightning (unless fatal)
- Return values ignored (callbacks work via side effects)

## Usage Patterns

### Common Use Case 1: MLflow Metric Logging
```python
from dlkit.runtime.workflows.strategies.tracking import MLflowTracker
from dlkit.core.training.callbacks import MLflowEpochLogger
from lightning.pytorch import Trainer

# Setup tracking
tracker = MLflowTracker()
tracker.setup_mlflow_config(settings.MLFLOW)

with tracker:
    with tracker.create_run(experiment_name="my_experiment") as run_context:
        # Create epoch logger
        logger_callback = MLflowEpochLogger(run_context)

        # Train with automatic metric logging
        trainer = Trainer(max_epochs=50, callbacks=[logger_callback], log_every_n_steps=10)

        trainer.fit(model, datamodule)
        # Metrics logged to MLflow with epoch numbers
```

### Common Use Case 2: Saving Inference Predictions
```python
from pathlib import Path
from dlkit.core.training.callbacks import NumpyWriter
from lightning.pytorch import Trainer
import numpy as np

# Configure prediction writer
output_dir = Path("./experiment_outputs/predictions")
writer = NumpyWriter(output_dir=output_dir, filenames=("reconstructions", "encodings"))

# Run inference
trainer = Trainer(callbacks=[writer])
trainer.predict(model, datamodule)

# Load and analyze predictions
reconstructions = np.load(output_dir / "reconstructions.npy")
encodings = np.load(output_dir / "encodings.npy")

print(f"Reconstructions shape: {reconstructions.shape}")
print(f"Encodings shape: {encodings.shape}")
```

### Common Use Case 3: Combined Training and Prediction
```python
from pathlib import Path
from dlkit.core.training.callbacks import MLflowEpochLogger, NumpyWriter
from lightning.pytorch import Trainer

# Training phase
with tracker.create_run("training") as run_context:
    epoch_logger = MLflowEpochLogger(run_context)

    trainer = Trainer(max_epochs=100, callbacks=[epoch_logger])
    trainer.fit(model, datamodule)

# Prediction phase
numpy_writer = NumpyWriter(
    output_dir=Path("./predictions"), filenames=("predictions", "uncertainties")
)

predict_trainer = Trainer(callbacks=[numpy_writer])
predict_trainer.predict(model, datamodule)
```

### Common Use Case 4: Custom Prediction Output Structure
```python
from dlkit.core.training.callbacks import NumpyWriter
import lightning.pytorch as pl
import torch


class MultiOutputModel(pl.LightningModule):
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        reconstructed = self.forward(x)
        latent = self.encode(x)
        attention = self.get_attention_weights(x)

        # Return as dict - keys match filenames
        return {"reconstructed": reconstructed, "latent": latent, "attention": attention}


# Configure writer with matching filenames
writer = NumpyWriter(
    output_dir=Path("./outputs"), filenames=("reconstructed", "latent", "attention")
)

trainer = Trainer(callbacks=[writer])
trainer.predict(model, datamodule)

# Files saved: reconstructed.npy, latent.npy, attention.npy
```

## Error Handling

**Exceptions Raised**:
- `OSError`: File writing failures in NumpyWriter (logged, not propagated)
- `ValueError`: Non-tensor outputs in NumpyWriter (logged as warning)
- `TypeError`: Metric conversion failures in MLflowEpochLogger (logged, skipped)

**Error Handling Pattern**:
```python
from dlkit.core.training.callbacks import NumpyWriter, MLflowEpochLogger
from pathlib import Path
import logging

# NumpyWriter error handling
try:
    writer = NumpyWriter(output_dir=Path("/readonly/path"))
    trainer = Trainer(callbacks=[writer])
    trainer.predict(model, datamodule)
except OSError:
    pass  # Error logged by callback, training continues

# MLflowEpochLogger error handling
try:
    logger = MLflowEpochLogger(None)  # Invalid run context
    trainer = Trainer(callbacks=[logger])
    trainer.fit(model, datamodule)
except Exception:
    pass  # Error logged, training continues


# Robust callback initialization
def create_callbacks(output_dir: Path | None, run_context: Any | None):
    callbacks = []

    # Only add epoch logger if run context available
    if run_context is not None:
        try:
            callbacks.append(MLflowEpochLogger(run_context))
        except Exception as e:
            logging.warning(f"Failed to create epoch logger: {e}")

    # Only add writer if output dir valid
    if output_dir is not None and output_dir.exists():
        try:
            callbacks.append(NumpyWriter(output_dir=output_dir))
        except Exception as e:
            logging.warning(f"Failed to create numpy writer: {e}")

    return callbacks
```

## Testing

### Test Coverage
- Unit tests: `tests/core/training/callbacks/` (to be created)
- Integration tests:
  - `tests/integration/test_mlflow_training_integration.py`
  - `tests/integration/test_transforms_persistence_and_inference.py`

### Key Test Scenarios
1. **Epoch logging**: Verify metrics logged with correct epoch numbers
2. **Metric filtering**: Test train/val/test prefix filtering
3. **Prediction accumulation**: Verify batch predictions concatenated correctly
4. **Multiple outputs**: Test dict, list, tuple, tensor output formats
5. **MLflow integration**: Verify artifact logging when MLflow active
6. **Fallback behavior**: Test directory fallback when MLflow unavailable
7. **Error resilience**: Callbacks don't crash on invalid inputs
8. **File persistence**: Verify .npy files saved correctly

### Fixtures Used
- `tmp_path` (pytest built-in): Temporary output directories
- `mock_trainer`: Mock PyTorch Lightning Trainer for callback testing
- `mock_run_context`: Mock MLflow run context
- `sample_predictions`: Sample tensor outputs for prediction testing

## Performance Considerations
- Predictions accumulated in memory before disk write (trade memory for I/O efficiency)
- Batch-wise accumulation avoids holding full dataset in single tensor
- NumPy `.npy` format used for efficient array serialization
- Metric filtering reduces MLflow logging overhead
- Directory creation deferred until first write
- MLflow adapter queried once during initialization
- Tensor-to-float conversion uses fast PyTorch methods
- Concatenation along dim=0 uses efficient torch.cat

## Future Improvements / TODOs
- [ ] Add support for compressed `.npz` format for large predictions
- [ ] Implement streaming writes for memory-constrained environments
- [ ] Add prediction checkpointing for resumable inference
- [ ] Support custom serialization formats (HDF5, Parquet)
- [ ] Add metric aggregation options (mean, median, percentiles)
- [ ] Implement automatic outlier detection in metrics
- [ ] Add progress bar for prediction saving
- [ ] Support distributed prediction aggregation across GPUs
- [ ] Add metadata files describing prediction schema

## Related Modules
- `dlkit.runtime.workflows.strategies.tracking`: MLflow tracking for run context creation
- `dlkit.interfaces.servers.mlflow_adapter`: MLflow adapter for artifact logging
- `lightning.pytorch.callbacks`: PyTorch Lightning callback base classes
- `dlkit.core.models.wrappers`: Model wrappers that produce prediction outputs

## Change Log
- **2025-10-03**: Comprehensive documentation with enriched docstrings
- **2024-10-02**: Added MLflowEpochLogger for epoch-based metric logging
- **2024-09-30**: Enhanced NumpyWriter with multiple output format support
- **2024-09-24**: Initial NumpyWriter implementation with MLflow integration
