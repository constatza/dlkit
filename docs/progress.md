# Artifact Visualization System — Implementation Progress

Branch: `feature/artifact-visualization-system`
Plan: `.claude/plans/we-must-revamp-and-robust-adleman.md`

## Objective

Add a modular, opt-in visualization layer that logs PNG plot artifacts to MLflow
during training: loss curves, parity plots, residual plots, error histograms, and
residuals-vs-index plots. SOLID-compliant — new plot types extend via `IFigureGenerator`
without touching existing callbacks.

---

## Task Status

| # | Task | Status | Commit(s) |
|---|------|--------|-----------|
| 1 | Fix Python 2 syntax errors | ✅ Done | `94f3da8` |
| 2 | Add matplotlib dep, `"plot"` ArtifactKind, tach module | ✅ Done | `b3451bc` |
| 3 | `domain/analysis/figures/` sub-package | ✅ Done | `4b2eeed`, `13ce85d` |
| 4 | `IFigureGenerator` protocol + concrete generators | ✅ Done | `caf4059` |
| 5 | `PlotSettings` config + `JobConfig.plots` field | 🔄 Pending |  |
| 6 | `LossCurvePlotCallback` + `PredictionPlotCallback` | 🔄 Pending |  |
| 7 | `TrackingDecorator._inject_plot_callbacks` | 🔄 Pending |  |
| 8 | Tests for figures and callbacks | 🔄 Pending |  |

---

## What's Done

### Domain Layer (`src/dlkit/domain/analysis/`)
Pure numpy + matplotlib, zero framework coupling.

- **`figures/_backend.py`** — matplotlib Agg backend isolation (headless-safe)
- **`figures/training.py`** — `loss_curve_figure(train_losses, val_losses)` → Figure
- **`figures/regression.py`** — four regression analysis plots:
  - `parity_figure` — predicted vs actual with R² annotation
  - `residual_figure` — residuals vs predicted with y=0 reference
  - `error_histogram_figure` — error distribution with normal overlay
  - `residual_vs_index_figure` — residuals vs sample index (trend detection)
  - All scatter plots: flatten any-shape inputs to 1-D, cap at `max_points` via random subsample
- **`protocols.py`** — `IFigureGenerator` Protocol (OCP: add new plots without touching callbacks)
- **`generators.py`** — `ParityGenerator`, `ResidualGenerator`, `ErrorHistogramGenerator`, `ResidualVsIndexGenerator` (frozen dataclasses)

### Manifest changes
- `matplotlib>=3.9,<4.0` added to `pyproject.toml`
- `"plot"` added to `ArtifactKind` Literal
- `dlkit.domain.analysis` module added to `tach.toml`

---

## What's Left

### Task 5 — Config (`PlotSettings`)
New `infrastructure/config/plot_settings.py` with opt-in flags:
`enabled`, `loss_curve`, `parity`, `residual`, `error_histogram`, `residual_vs_index`, `dpi`, `artifact_dir`, `max_scatter_points`.
Add `plots: PlotSettings` field to `JobConfig`.

### Task 6 — Engine Callbacks (`plot_callbacks.py`)
- `LossCurvePlotCallback` — accumulates epoch losses, logs PNG at `on_fit_end`
- `PredictionPlotCallback` — accumulates TensorDict outputs, runs injected generators at `on_predict_epoch_end`
- Module-level `_plot_and_log` helper (temp file → `IRunContext.log_artifact` → close figure)

### Task 7 — Tracking Decorator Integration
Add `_inject_plot_callbacks` to `TrackingDecorator`. Builds generator list from `settings.plots` flags, appends callbacks to trainer. No-op when `plots.enabled=False` or no trainer present.

### Task 8 — Tests
Unit tests for all 5 figure functions and both callbacks (fixture-based, multi-dim flattening verified).

---

## User-Facing Config (when complete)

```toml
[plots]
enabled = true
loss_curve = true
parity = true
residual = true
error_histogram = true
residual_vs_index = false
dpi = 150
artifact_dir = "plots"
max_scatter_points = 5000
```

With `[tracking] backend = "mlflow"`, plots appear under `artifacts/plots/` in the MLflow run.
