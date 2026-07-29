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
- `evaluate()` — eval-only checkpoint stats/plots (see below)
- `evaluate_checkpoint()`, `log_evaluation_result()` — lower-level building
  blocks `evaluate()` composes

`load_model()`, `load_model_from_settings()`, `validate_checkpoint()`,
`get_checkpoint_info()`, `CheckpointPredictor`, `IPredictor`,
`PredictionOutput`, `PredictorConfig`, `evaluate_checkpoint()`, and
`log_evaluation_result()` are re-exported from `dlkit.engine.inference`.
`evaluate()` is re-exported from `dlkit.engine.workflows.entrypoints.evaluate`
(`evaluate.py` here is a thin re-export, not the implementation — see below
for why) to keep this package's public import path stable.

Batch-evaluating every child run of a multirun/sweep parent is now
`dlkit.interfaces.api.functions.core.evaluate_multirun()` — it composes an
`ExistingRunsSource` (`engine.workflows.multi_run`) with the same
`MultiRunOrchestrator` pipeline every other sweep uses, rather than living
here as a bespoke mechanism. See
`engine/workflows/entrypoints/entrypoints.md`.

## Usage

```python
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

### Eval-only stats/plots (`evaluate()`)

`evaluate()` answers a different question than `load_model()`/`predict()`:
given an *already-trained* checkpoint and a *labeled* dataset split, produce
the same MAE/RMSE/R2 and parity/residual/error-histogram/residual-vs-index
plots that training produces — without constructing a Lightning `Trainer` or
updating weights. It is a fourth workflow entrypoint, on equal footing with
`train()`/`optimize()`/`converge()`: settings-driven, dispatchable through
`execute()`, and usable as a multirun child.

```python
from dlkit.interfaces.inference import evaluate

result = evaluate(inference_settings)
result.metrics      # {"mae": ..., "rmse": ..., "r2": ...}
result.figures       # {"parity_plot": Figure, "residual_plot": Figure, ...}
```

Requires `settings.data.targets` to be configured (there is no plot without
ground truth). `settings.split` (`"test"` default, or `"predict"`) selects
which labeled partition to evaluate against — `predict_dataloader()` is a
genuinely different partition from `test_dataloader()` for
`GraphDataModule`-backed configs, so this is a real choice, not cosmetic. Set
`settings.tracking.backend = "mlflow"` (e.g. via `apply_mlflow_flag()`) to
also open an MLflow run and log the metrics/figures as artifacts — the same
convention `train()`/`optimize()` use, not a separate boolean kwarg.

`settings.model.checkpoint` accepts a literal path, or a `CheckpointSource`
resolved from a previously trained MLflow run instead of a local path
(downloading the run's checkpoint artifact to a temp directory first):

```python
from dlkit.interfaces.inference import evaluate
from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint

# Exact, caller-named run.
settings = inference_settings.patch({"model": {"checkpoint": RunCheckpoint(run_id="abc123")}})
result = evaluate(settings)

# Most recently started run in an experiment; experiment_name defaults to
# settings.experiment.name (or "dlkit-evaluate" if that is also unset).
settings = inference_settings.patch({"model": {"checkpoint": LatestRunCheckpoint()}})
result = evaluate(settings)
```

An `overrides: EvaluationOverrides | None` parameter (`checkpoint_path`,
`experiment_name`, `run_name`, `tags`, `batch_size`, `split`, `device`) is
also accepted for request-scoped overrides without hand-patching settings —
see `dlkit.interfaces.api.domain.override_types.EvaluationOverrides`.

### Batch evaluation over a sweep (`evaluate_multirun()`)

Moved to `dlkit.interfaces.api.functions.core.evaluate_multirun()`, next to
its siblings `run_multirun_config()`/`run_multirun_spec()` — batch-evaluating
a sweep's children is itself just another multirun sweep, not a bespoke
mechanism:

```python
from dlkit.interfaces.api.functions.core import evaluate_multirun

batch = evaluate_multirun(inference_settings, parent_run_id="parent-run-id")
batch.parent_run_id   # the new evaluate-sweep's own parent run id
for outcome in batch.children:
    outcome.run_id       # the checkpoint-source child run, once tagged
    outcome.result        # EvaluationResult for that child (on ChildSuccess)
```

Returns `MultiRunResult[ChildOutcome[WorkflowResult]]` — the same shape every
other sweep returns, not a bespoke `ChildEvaluation` record. Internally
builds an `ExistingRunsSource` (`engine.workflows.multi_run`): each active
child of `parent_run_id` (found via the standard `mlflow.parentRunId` tag,
covering both dlkit-native sweeps and externally-linked runs) becomes one
evaluate `RunSpec`. Opens its own new "evaluate sweep" parent run, tagged
`multirun.source_parent_run_id` back to the run being evaluated, so its
children are discoverable the same way any sweep's are. `failure_policy`
defaults to `"fail_fast"` (matching the old all-or-nothing behavior exactly);
`"continue"`/`"continue_mark_parent_failed"` are also available.

## Dependency Direction

`interfaces.inference -> runtime.predictor -> runtime/core/nn/tools/shared`

## Design Rules
- keep this package as a public adapter only
- do not duplicate runtime predictor implementation modules here
- use the runtime predictor as the single source of truth for checkpoint
  loading, transform reconstruction, shape inference, and precision-aware
  prediction

## Notes
- `execute()` dispatches `InferenceJobConfig` to `evaluate()`, the same as
  every other workflow settings type — evaluate is a full peer of
  train/optimize/converge, not a special case. `load_model()` remains the
  right choice for raw predictions with no ground truth/metrics/plots.
- `load_model_from_settings()` resolves `model.checkpoint` from an
  `InferenceJobConfig` unless an explicit `checkpoint_path=` override is provided.
- `CheckpointPredictor` exposes `feature_names` and `predict_target_key` as public
  metadata properties, plus `describe_inputs() -> dict[str, str]` to inspect the
  required `predict()` kwargs before calling it.
- Checkpoints produced by the standard Lightning wrapper also persist
  `forward_arg_map`, allowing inference to reconstruct named feature dispatch
  and apply the correct feature transform before calling `model.forward(**kwargs)`.
  `predict()` validates caller kwargs against this contract before the model is
  called, raising `dlkit.common.errors.ForwardContractError` (naming the real
  expected kwargs) on a mismatch instead of a raw `TypeError` from inside the
  model — see `engine/inference/inference.md` for the full mechanism.
- For DeepONet-style checkpoints, `feature_names` preserves both the branch
  feature entry and the query-coordinate `target_coordinates` entry in
  training-time order.
- The runtime predictor owns checkpoint validation, metadata extraction, and
  model lifecycle management.
- Checkpoint transform reconstruction accepts the serialized `entry_configs[*].transforms`
  metadata written by DLKit and normalizes those specs before module construction.
- `evaluate()` always runs with `apply_transforms=True` internally and does not
  expose it as a caller-facing flag: predictions come back inverse-transformed
  to raw scale, and dataset targets are already raw, so anything else would
  silently compare predictions/targets on mismatched scales.
- `evaluate()`'s default `PlotSettings` (all four regression plots enabled) is
  intentionally different from training's `PlotSettings` default (opt-in,
  every flag `False`) — plots are the entire point of calling `evaluate()`.
  An explicit `plots=` argument, or `[plots] enabled = true` already set on
  the settings object, overrides this default.
