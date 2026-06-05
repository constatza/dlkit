# Inference Module

`dlkit.interfaces.inference` is the public checkpoint-inference adapter.
Implementation lives in `dlkit.engine.inference`; this package re-exports that
runtime predictor surface for users.

## Overview

The interface layer is a thin public adapter over `dlkit.engine.inference`.
It does not own separate predictor, loading, shape, or transform
implementations.

## Public Surface
- `load_model()`
- `load_model_from_settings()`
- `validate_checkpoint()`
- `get_checkpoint_info()`
- `CheckpointPredictor`
- `IPredictor`
- `PredictionOutput`
- `PredictorConfig`

These symbols are re-exported from `dlkit.engine.inference`.

## Usage

```python
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

## Dependency Direction

`interfaces.inference -> runtime.predictor -> runtime/core/nn/tools/shared`

## Design Rules
- keep this package as a public adapter only
- do not duplicate runtime predictor implementation modules here
- use the runtime predictor as the single source of truth for checkpoint
  loading, transform reconstruction, shape inference, and precision-aware
  prediction

## Notes
- Unified workflow execution no longer handles inference.
- `execute()` rejects inference settings and points callers to `load_model()`.
- `load_model_from_settings()` resolves `MODEL.checkpoint` from an
  `InferenceWorkflowConfig` unless an explicit `checkpoint_path=` override is provided.
- `CheckpointPredictor` exposes `feature_names` and `predict_target_key` as public
  metadata properties.
- For DeepONet-style checkpoints, `feature_names` preserves both the branch
  feature entry and the query-coordinate `target_coordinates` entry in
  training-time order.
- The runtime predictor owns checkpoint validation, metadata extraction, and
  model lifecycle management.
- Checkpoint transform reconstruction accepts the serialized `entry_configs[*].transforms`
  metadata written by DLKit and normalizes those specs before module construction.
