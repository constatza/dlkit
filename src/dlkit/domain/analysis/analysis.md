# domain.analysis — Developer Reference

## Module layout

```
domain/analysis/
├── protocols.py      IFigureGenerator protocol
├── generators.py     Concrete generators (parity, residual, histogram, index)
└── figures/
    ├── training.py   loss_curve_figure()
    └── regression.py parity_figure(), residual_figure(), error_histogram_figure(),
                      residual_vs_index_figure()
```

Configuration lives in `infrastructure/config/plot_settings.py` (`PlotSettings`).
Lightning callbacks live in `engine/adapters/lightning/plot_callbacks.py`.

The domain layer owns only pure figure generation (numpy in → matplotlib Figure
out). Orchestration, accumulation, and upload are engine concerns.

Regression residual and error plots use raw signed values computed as
`targets - predictions`. The plotting layer does not scale or normalize them.
`error_histogram_figure()` uses NumPy adaptive bins by default, can limit the
display range to an explicit percentile window supplied by the caller or
`PlotSettings`, reports full/view ranges in the title, and overlays both a
normal fit and an in-module Gaussian KDE. When a percentile window is active,
it is further clamped by a Tukey IQR fence (`[Q1 - 3*IQR, Q3 + 3*IQR]`, tighter
bound wins) since percentile clipping alone is insufficient for heavy-tailed
error distributions. The density y-axis is log-scaled, since regression error
distributions are commonly leptokurtic (a tall near-zero spike plus a much
wider spread) and a linear scale flattens everything but the spike.

## IFigureGenerator protocol

`IFigureGenerator` is the extension point for custom plot types:

```python
from dlkit.domain.analysis.protocols import IFigureGenerator
from dataclasses import dataclass
import numpy as np
from matplotlib.figure import Figure

@dataclass(frozen=True)
class MyGenerator:
    name: str = "my_plot"

    def generate(self, predictions: np.ndarray, targets: np.ndarray) -> Figure:
        ...  # return a matplotlib Figure; caller closes it
```

The class does not need to inherit from anything — `IFigureGenerator` is a
`@runtime_checkable` Protocol. Using a frozen dataclass is the convention because
generators carry no mutable state.

**Contract:**
- `name` is the artifact filename stem (`"my_plot"` → `my_plot.png`).
- `generate` receives flat 1-D numpy arrays. Input flattening is done by
  `PredictionPlotCallback` before calling any generator.
- The returned `Figure` is closed by the caller after upload — do not close it
  inside `generate`.
- Plot failures are caught and logged as warnings; generators must not swallow
  their own exceptions silently.

## Wiring a custom generator

Pass custom generators to `PredictionPlotCallback` directly:

```python
from dlkit.engine.adapters.lightning.plot_callbacks import PredictionPlotCallback

callback = PredictionPlotCallback(
    run_context=run_ctx,
    generators=[MyGenerator()],
    settings=plot_settings,
)
trainer.callbacks.append(callback)
```

Or extend `build_plot_callbacks` in `plot_callbacks.py` if the generator is
built-in and should be toggled by a `PlotSettings` flag.

## Data flow

```
PlotSettings (TOML)
    └─▶ select_enabled_generators()      # engine/adapters/lightning/plot_callbacks.py
            │   (single source of truth: which generators are enabled)
            │
            ├─▶ build_plot_callbacks()               # training path (Lightning Trainer)
            │       ├─▶ LossCurvePlotCallback         # on_train_epoch_end → on_fit_end
            │       └─▶ PredictionPlotCallback         # on_predict_batch_end → on_predict_epoch_end
            │               └─▶ IFigureGenerator.generate(preds, targets)
            │                       └─▶ _plot_and_log() → IArtifactLogger.log_artifact_content()
            │
            └─▶ generate_regression_figures()        # eval-only path (no Trainer)
                    # engine/inference/evaluation.py
                    └─▶ IFigureGenerator.generate(preds, targets)
                            └─▶ log_evaluation_result() → IRunContext.log_artifact_content()
```

`build_plot_callbacks` and `generate_regression_figures` both call
`select_enabled_generators(plots)` rather than re-implementing the
`PlotSettings` flag-gating separately — this is the DRY seam that keeps
training-time and eval-only plots identical. `select_enabled_generators`
defers the import of domain generators until call-time. This is intentional:
`engine.tracking` cannot import `domain` directly (DAG rule), so it delegates
generator/callback construction to `engine.adapters.lightning` which is
allowed to cross that boundary; `engine.inference` reuses the same function
rather than duplicating the flag-gating logic.

## Layer boundaries

| Layer | Allowed to import | Purpose |
|-------|-------------------|---------|
| `domain.analysis` | `common` only | pure figure generation |
| `engine.adapters.lightning` | `domain`, `engine`, `infrastructure` | callbacks, accumulation, upload |
| `infrastructure.config` | `common` | `PlotSettings` definition |

`IFigureGenerator` and the concrete generators must never import anything above
`domain`. Upload logic (`IArtifactLogger`, `log_artifact_content`) lives
exclusively in the engine layer.

## Adding a new built-in plot type

1. Add a figure function to `figures/regression.py` (or `figures/training.py`
   for training-time plots).
2. Add a generator in `generators.py`.
3. Add a boolean flag to `PlotSettings` in
   `infrastructure/config/plot_settings.py`.
4. Wire it in `select_enabled_generators` in
   `engine/adapters/lightning/plot_callbacks.py` — both the training-callback
   path (`build_plot_callbacks`) and the eval-only path
   (`engine/inference/evaluation.py::generate_regression_figures`) pick it up
   automatically since both call `select_enabled_generators`.
5. Update `README.md` in this directory with the new flag and what it shows.
