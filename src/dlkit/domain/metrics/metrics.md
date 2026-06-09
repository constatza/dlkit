# Metrics Module — Developer Reference

Developer documentation for `dlkit.domain.metrics`.
For the user-facing API catalog see [README.md](README.md).

## File Layout

| File | Responsibility |
|------|----------------|
| `__init__.py` | Public re-exports and `__all__` |
| `compat.py` | Thin re-exports from `torchmetrics.regression` (no added logic) |
| `functional.py` | Pure, stateless metric functions and internal `_update`/`_compute` helpers |
| `torchmetrics_wrappers.py` | Stateful `torchmetrics.Metric` subclasses |
| `collect.py` | `collect_metrics()` — converts metric dicts to `float` for serialisation |

## Design Decisions

**Functional core + stateful wrappers split**
`functional.py` contains pure functions so they can be unit-tested without state,
used as differentiable losses, or composed freely. `torchmetrics_wrappers.py`
wraps these into the update/compute protocol for batch accumulation and DDP.

**`compat.py` delegation**
Standard regression metrics (`MSE`, `MAE`, `R2Score`, …) are not reimplemented.
`compat.py` re-exports them from `torchmetrics.regression` to keep dlkit's public
namespace unified while avoiding maintenance burden.

**Update/compute helpers in `functional.py`**
Private `_*_update` and `_*_compute` functions separate per-sample state
accumulation from final reduction. They are consumed exclusively by
`torchmetrics_wrappers.py` — do not call them directly from outside the module.

## Shape Contracts

| Metric family | Input | Output |
|---------------|-------|--------|
| Vector metrics | `(B, …, D)` | scalar |
| Temporal metrics | `(B, T, D)` with `T ≥ n+1` | scalar |
| Energy norm | `(B, D)` + matrix `(B, D, D)` or `(1, D, D)` | scalar |

`B` = batch, `T` = time steps, `D` = feature dimension.
Pass shape `(1, D, D)` for a matrix shared across the batch — broadcasting is
handled internally.

## Adding a New Metric

1. **Functional core** — add `_my_metric_update(preds, target, ...) → Tensor`
   (per-sample values, not aggregated) and `_my_metric_compute(state, total) → Tensor`
   to `functional.py`. Export via `__all__` only if users should call these directly.

2. **Stateful wrapper** — add a `torchmetrics.Metric` subclass to
   `torchmetrics_wrappers.py`:
   - `add_state("sum_x", default=tensor(0.0), dist_reduce_fx="sum")`
   - `add_state("total", default=tensor(0),   dist_reduce_fx="sum")`
   - `update()` calls the `_update` helper and accumulates into `sum_x` and `total`.
   - `compute()` calls the `_compute` helper.

3. **Export** — add the class name to `__init__.py` imports and `__all__`.

4. **Tests** — add tests under `tests/domain/metrics/` following existing patterns:
   - Unit-test `_update`/`_compute` with fixed tensors.
   - Test the `Metric` subclass end-to-end via `.update()` / `.compute()`.
   - Verify `MetricCollection` compatibility.

## Cross-References

- `dlkit.domain.losses` — differentiable loss functions (backprop-safe variants)
- `dlkit.engine.adapters.lightning.metrics_routing` — wiring metrics into Lightning steps
- [User-facing catalog](README.md)
