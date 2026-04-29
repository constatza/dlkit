# Inference Module

`dlkit.interfaces.inference` is the public checkpoint-inference adapter.
Implementation lives in `dlkit.engine.inference`; this package re-exports that
runtime predictor surface for users.

## Public API
- `load_model()`
- `load_model_from_settings()`
- `validate_checkpoint()`
- `get_checkpoint_info()`
- `CheckpointPredictor`
- `IPredictor`
- `PredictionOutput`
- `PredictorConfig`

## Usage

```python
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

## Architecture
- no duplicated predictor implementation in `interfaces/inference`
- no interface-local loading or transform modules
- inference stays separate from `train()`, `optimize()`, and `execute()`
- direct predictor calls return `PredictionOutput`, not a bare tensor/tuple

See [`inference.md`](inference.md) for the developer-facing architecture notes.
