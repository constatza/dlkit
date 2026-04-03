# Precision

`dlkit.tools.precision` provides semantic precision configuration, thread-safe runtime overrides, and a central resolution service. It has **no dlkit dependencies** — only torch and stdlib.

## Modules

| Module | Responsibility |
|--------|---------------|
| `strategy.py` | `PrecisionStrategy` enum — semantic modes, conversions, aliases |
| `context.py` | `PrecisionContext` — thread-local runtime overrides |
| `service.py` | `PrecisionService` — central resolution with priority chain |

## Precision Modes

| Strategy | Lightning value | `torch.dtype` | Memory factor |
|----------|----------------|---------------|---------------|
| `FULL_64` | `"64"` | `float64` | 2.0× |
| `FULL_32` | `32` | `float32` | 1.0× (default) |
| `MIXED_16` | `"16-mixed"` | `float32` (weights) | 0.7× |
| `TRUE_16` | `"16"` | `float16` | 0.5× |
| `MIXED_BF16` | `"bf16-mixed"` | `float32` (weights) | 0.7× |
| `TRUE_BF16` | `"bf16"` | `bfloat16` | 0.5× |

Mixed modes keep weights in `float32`; forward pass runs under `torch.autocast` managed by Lightning.

## Resolution Priority

`PrecisionService.resolve_precision()` picks the first of:

1. Thread-local context override (`precision_override(...)` context manager)
2. `PrecisionProvider` passed by the caller (e.g. `SessionSettings`)
3. Explicit `default` argument
4. `PrecisionStrategy.FULL_32`

## Configuration

Set precision in TOML under `[SESSION]`:

```toml
[SESSION]
precision = "float32"      # or "float16", "bfloat16", "double", "16-mixed", etc.
```

Accepted aliases (case-insensitive): `float32 / f32 / single`, `float16 / f16 / half`, `bfloat16 / bf16`, `float64 / f64 / double`, `16-mixed / amp`, `bf16-mixed`. Integer strings are rejected — use semantic names.

## Lightning Integration

DLKit does **not** replace Lightning's precision handling — it augments it:

- **Lightning owns**: `torch.autocast` for the forward pass, loss scaling, gradient unscaling
- **DLKit owns**: model weight casting (`model.to(dtype=...)`) at init and after checkpoint restore, per-entry data dtype resolution, inference-time precision context

`TrainerSettings.build()` converts the resolved strategy to a Lightning-compatible value and passes it to `Trainer(precision=...)`. The Lightning wrapper (`runtime.adapters.lightning`) re-casts model weights via `configure_model()` after distributed setup or checkpoint restore.

## Inference

`CheckpointPredictor` infers precision from the loaded model's parameter dtype via `PrecisionService.infer_precision_from_model()` and activates a context override for the duration of each `predict()` call. No separate inference config is required.

## Thread-Safe Overrides

```python
from dlkit.tools.precision import precision_override, PrecisionStrategy

with precision_override(PrecisionStrategy.TRUE_16):
    result = predictor.predict(x)   # runs under float16 context
```

Each thread gets its own override stack via `threading.local()`.

## Per-Entry Data Dtype

`DataEntry.get_effective_dtype(precision_provider)` returns the entry's explicit `dtype` if set, otherwise delegates to `PrecisionService`. This allows individual features or targets to use a different dtype than the global session precision.
