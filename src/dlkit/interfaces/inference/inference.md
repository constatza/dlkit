# Inference Module

## Overview
`dlkit.interfaces.inference` is now a thin public adapter over `dlkit.engine.inference`.
The interface layer no longer owns separate predictor, loading, shape, or transform
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

## Dependency Direction

`interfaces.inference -> runtime.predictor -> runtime/core/nn/tools/shared`

## Design Rules
- keep this package as a public adapter only
- do not duplicate runtime predictor implementation modules here
- use the runtime predictor as the single source of truth for checkpoint loading,
  transform reconstruction, shape inference, and precision-aware prediction

## Usage

```python
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

## Notes
- Unified workflow execution no longer handles inference.
- `execute()` rejects inference settings and points callers to `load_model()`.
- `load_model_from_settings()` resolves `MODEL.checkpoint` from an
  `InferenceWorkflowConfig` unless an explicit `checkpoint_path=` override is provided.
- `CheckpointPredictor` exposes `feature_names` and `predict_target_key` as public
  metadata properties.
- The runtime predictor owns checkpoint validation, metadata extraction, and
  model lifecycle management.

## Related Modules
- [`README.md`](README.md)
- [`../../runtime/predictor/__init__.py`](../../runtime/predictor/__init__.py)
