# Runtime Inference Module

`dlkit.engine.inference` owns checkpoint loading, stateful prediction, and
eval-only orchestration — all without a Lightning `Trainer`. The public
adapter over this module is `dlkit.interfaces.inference`; that package should
be the only consumer outside `engine`.

## Modules

- `api.py` — `load_model()`, `load_model_from_settings()`: checkpoint →
  `CheckpointPredictor` factory functions.
- `predictor.py` — `CheckpointPredictor`: loads a checkpoint once (model,
  fitted transforms, precision), then serves `.predict()` calls without
  reloading.
- `loading.py`, `checkpoint_reader.py`, `model_builder.py`, `transforms.py` —
  checkpoint parsing, model reconstruction, and transform restoration.
- `batch_prediction.py` — dataset-level orchestration over a
  `LightningDataModule`:
  - `run_batched_prediction(predictor, datamodule)` — predictions only, over
    `predict_dataloader()`. Used by `dlkit predict`.
  - `run_batched_evaluation(predictor, datamodule, split="test")` —
    predictions **and** ground-truth targets, over `test_dataloader()` by
    default (`predict_dataloader()` only when `split="predict"` is
    explicit). Requires `predictor.predict_target_key` to be set.
- `evaluation.py` — eval-only orchestration, no Lightning `Trainer`:
  - `compute_regression_metrics(predictions, targets) -> dict[str, float]` —
    MAE/RMSE/R2 via `torchmetrics.regression`, single-target only.
  - `generate_regression_figures(predictions, targets, plots) -> dict[str, Figure]`
    — calls `select_enabled_generators(plots)`
    (`engine.adapters.lightning.plot_callbacks`) directly on flattened numpy
    arrays. Same generators, same plots as training — no callback machinery.
  - `evaluate_checkpoint(predictor, datamodule, plots, split="test") -> EvaluationResult`
    — composes the two functions above plus `run_batched_evaluation`.
  - `log_evaluation_result(result, run_context, plots)` — logs metrics +
    figures to any `IRunContext`. Unlike training's `_plot_and_log`, this does
    **not** close the figures afterward (`EvaluationResult.figures` still
    holds references for the caller).

## Why `evaluate()` itself lives in `interfaces.inference`, not here

The full `evaluate()` entrypoint (settings → checkpoint → datamodule →
`evaluate_checkpoint` → optional MLflow logging) needs
`engine.workflows.factories` (datamodule construction) and `engine.tracking`
(MLflow) — dependencies this module is not permitted under `tach.toml`. Those
are available to `dlkit.interfaces.inference`, so the top-level `evaluate()`
function is defined there (`interfaces/inference/evaluate.py`) and simply
composes the building blocks in this module.

## Dependency Direction

`engine.inference -> domain, engine.data, engine.adapters.lightning,
infrastructure, common`

No upward imports; `engine.inference` never imports `interfaces.*` or
`engine.tracking`/`engine.workflows.*`.

## Design Rules

- `run_batched_evaluation` always resolves a *labeled* split. Never assume
  `predict_dataloader()` aliases `test_dataloader()` — `GraphDataModule`
  gives predict a genuinely separate partition from test.
- `evaluate_checkpoint`/`generate_regression_figures` never construct a
  Lightning `Trainer` and never update model weights — that is the entire
  point of the eval-only path, as distinct from `train()`/`optimize()`.
- Metric/figure computation always runs with transformed predictions on the
  same scale as raw dataset targets (`apply_transforms=True` is not a
  caller-facing knob anywhere in this module).
