# dlkit.domain.metrics

Metrics for evaluating regression and physics-based ML predictions.
All stateful metrics are compatible with `torchmetrics.MetricCollection`,
PyTorch Lightning, MLflow logging, and distributed training (DDP).

## Standard Metrics

Thin aliases to [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
— identical API, no additional configuration required.

| Name | Formula | Typical use |
|------|---------|-------------|
| `MeanSquaredError` | mean((p − t)²) | General regression baseline |
| `MeanAbsoluteError` | mean(\|p − t\|) | Robust to outliers |
| `MeanSquaredLogError` | mean((log(p+1) − log(t+1))²) | Targets spanning orders of magnitude |
| `MeanAbsolutePercentageError` | mean(\|p − t\| / \|t\|) × 100 | Relative % error |
| `R2Score` | 1 − SS_res / SS_tot | Explained variance (0–1) |

```python
from dlkit.domain.metrics import MeanSquaredError, R2Score
from torchmetrics import MetricCollection

metrics = MetricCollection({"mse": MeanSquaredError(), "r2": R2Score()})
metrics.update(preds, targets)
results = metrics.compute()  # {"mse": tensor(...), "r2": tensor(...)}
```

---

## Custom Stateful Metrics

Custom `torchmetrics.Metric` subclasses for vector and physics-based evaluation.
Use these in `MetricCollection` exactly like the standard metrics above.

### `RelativeVectorNormError`

Relative error: **mean(‖pred − target‖ₚ / ‖target‖ₚ)** per sample.

```python
from dlkit.domain.metrics import RelativeVectorNormError

metric = RelativeVectorNormError(norm_ord=2, vector_dim=-1, eps=1e-8)
metric.update(preds, targets)   # preds/targets: (B, ..., D)
value = metric.compute()        # scalar
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `norm_ord` | `2` | Norm order: `1`, `2`, or `float('inf')` |
| `vector_dim` | `-1` | Dimension along which the norm is computed |
| `eps` | `1e-8` | Added to denominator to avoid division by zero |

---

### `AbsoluteVectorNormError`

Absolute error in a given norm: **mean(‖pred − target‖ₚ)** per sample.

```python
from dlkit.domain.metrics import AbsoluteVectorNormError

metric = AbsoluteVectorNormError(norm_ord=2, vector_dim=-1)
metric.update(preds, targets)   # (B, ..., D)
value = metric.compute()        # scalar
```

---

### `TemporalDerivativeError`

MSE of **nth-order finite differences** — measures velocity/acceleration error in
sequential predictions.

```python
from dlkit.domain.metrics import TemporalDerivativeError

velocity_err = TemporalDerivativeError(n=1, derivative_dim=1)
accel_err    = TemporalDerivativeError(n=2, derivative_dim=1)

velocity_err.update(preds, targets)  # preds/targets: (B, T, D), T >= n+1
value = velocity_err.compute()       # scalar
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n` | `1` | Derivative order (1 = velocity, 2 = acceleration) |
| `derivative_dim` | `1` | Axis along which to differentiate (time axis) |

---

### `EnergyNormError`

**‖pred − target‖_A** = sqrt((pred − target)ᵀ A (pred − target)) — error in a
custom positive semi-definite matrix norm. Reduces to `AbsoluteVectorNormError`
when A = I.

```python
from dlkit.domain.metrics import EnergyNormError

metric = EnergyNormError()
metric.update(preds, targets, matrix)
# preds/targets: (B, D)
# matrix: (B, D, D) or (1, D, D) shared across batch
value = metric.compute()  # scalar
```

---

### `RelativeEnergyNormError`

**‖pred − target‖_A / ‖target‖_A** — dimensionless relative error in the energy
norm. Analogue of the Notay loss for evaluating preconditioner-based models.

```python
from dlkit.domain.metrics import RelativeEnergyNormError

metric = RelativeEnergyNormError(eps=1e-8)
metric.update(preds, targets, matrix)  # same shapes as EnergyNormError
value = metric.compute()               # scalar
```

---

## Functional API

Pure stateless functions for one-off computations, custom losses, or composing
new metrics. Import directly from `dlkit.domain.metrics`:

```python
from dlkit.domain.metrics import (
    # Composable building blocks
    compute_error_vectors,         # preds - target
    compute_vector_norm,           # ‖tensor‖ₚ along dim
    safe_divide,                   # a / (b + eps)
    apply_aggregation,             # reduce with any callable

    # Vector metrics (return scalar by default)
    relative_vector_norm_error,  # configurable ord, dim, aggregator
    relative_l1_error,           # ord=1 partial
    relative_l2_error,           # ord=2 partial
    relative_linf_error,         # ord=inf partial

    # Energy norm primitives
    compute_quadratic_form,        # vᵀ A v per sample → (B,)
    compute_energy_norm,           # sqrt(vᵀ A v) per sample → (B,)

    # Temporal metrics
    compute_temporal_derivative,   # nth finite difference (B,T,D) → (B,T-n,D)
    temporal_derivative_error,     # MSE of nth derivative
    first_derivative_error,        # n=1 partial
    second_derivative_error,       # n=2 partial
)
```

**When to use functional vs. stateful:**

| Situation | Use |
|-----------|-----|
| Training / Lightning `MetricCollection` | Stateful (accumulates across batches) |
| One-off evaluation on a single batch | Functional |
| Custom differentiable loss | Functional (`torch.no_grad` not forced) |
| Distributed training (DDP) | Stateful (handles `dist_reduce_fx` automatically) |

---

## Utilities

```python
from dlkit.domain.metrics import collect_metrics, MetricsPayload
# MetricsPayload = dict[str, float | Tensor]

raw: dict[str, Any] = trainer.callback_metrics
clean: MetricsPayload = collect_metrics(raw)  # converts Tensors → float
```
