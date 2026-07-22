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
- `evaluate_multirun()`, `ChildEvaluation` — batch `evaluate()` over every
  child run of a multirun/sweep parent (see below)

`load_model()`, `load_model_from_settings()`, `validate_checkpoint()`,
`get_checkpoint_info()`, `CheckpointPredictor`, `IPredictor`,
`PredictionOutput`, `PredictorConfig`, `evaluate_checkpoint()`, and
`log_evaluation_result()` are re-exported from `dlkit.engine.inference`.
`evaluate()` and `evaluate_multirun()`/`ChildEvaluation` are defined in this
package (`evaluate.py`, `evaluate_multirun.py`) — see below for why.

## Usage

```python
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

### Eval-only stats/plots (`evaluate()`)

`evaluate()` (defined in this package's `evaluate.py`, not `engine.inference`,
because it also needs `engine.workflows.factories` for datamodule construction
and `engine.tracking` for optional MLflow logging — a wider dependency set than
`engine.inference` itself is allowed under the DAG) answers a different
question than `load_model()`/`predict()`: given an *already-trained*
checkpoint and a *labeled* dataset split, produce the same MAE/RMSE/R2 and
parity/residual/error-histogram/residual-vs-index plots that training
produces — without constructing a Lightning `Trainer` or updating weights.

```python
from dlkit.interfaces.inference import evaluate

result = evaluate(inference_settings, checkpoint_path="model.ckpt")
result.metrics      # {"mae": ..., "rmse": ..., "r2": ...}
result.figures       # {"parity_plot": Figure, "residual_plot": Figure, ...}
```

Requires `settings.data.targets` to be configured (there is no plot without
ground truth). `split="test"` (default) or `split="predict"` selects which
labeled partition to evaluate against — `predict_dataloader()` is a genuinely
different partition from `test_dataloader()` for `GraphDataModule`-backed
configs, so this is a real choice, not cosmetic. Pass `log_to_mlflow=True` to
also open an MLflow run and log the metrics/figures as artifacts.

`checkpoint_path` and `run_checkpoint` are mutually exclusive; passing both
raises `ConfigurationError`. `run_checkpoint` resolves the checkpoint from a
previously trained MLflow run instead of a local path, downloading the run's
checkpoint artifact to a temp directory first:

```python
from dlkit.interfaces.inference import evaluate
from dlkit.common.checkpoint_source import LatestRunCheckpoint, RunCheckpoint

# Exact, caller-named run.
result = evaluate(inference_settings, run_checkpoint=RunCheckpoint(run_id="abc123"))

# Most recently started run in an experiment; experiment_name defaults to
# settings.experiment.name (or "dlkit-evaluate" if that is also unset).
result = evaluate(inference_settings, run_checkpoint=LatestRunCheckpoint())
result = evaluate(inference_settings, run_checkpoint=LatestRunCheckpoint(experiment_name="exp"))
```

### Batch evaluation over a sweep (`evaluate_multirun()`)

`evaluate_multirun()` and `ChildEvaluation` live in this package's
`evaluate_multirun.py` and are importable as
`from dlkit.interfaces.inference import evaluate_multirun, ChildEvaluation`.
They are reachable at that path only — unlike `evaluate()`, they are not
re-exported as `dlkit.evaluate_multirun` or
`dlkit.interfaces.api.evaluate_multirun`.

`evaluate_multirun()` fans a single `evaluate()` call out over every active
child run of a multirun/sweep parent run, matching on the
`mlflow.parentRunId` tag convention — this covers both dlkit-native nested
sweeps (`MultiRunOrchestrator`) and externally-linked runs sharing the same
convention:

```python
from dlkit.interfaces.inference import evaluate_multirun

batch = evaluate_multirun(inference_settings, parent_run_id="parent-run-id")
batch.parent_run_id   # "parent-run-id"
for child in batch.children:
    child.run_id       # the child run the checkpoint was pulled from
    child.result        # EvaluationResult for that child
```

Returns a `MultiRunResult[ChildEvaluation]` (`dlkit.common.MultiRunResult`):
`parent_run_id` plus one `ChildEvaluation` per active child run, in ascending
`start_time` order. `ChildEvaluation.run_id` names the run the checkpoint was
pulled from; this is distinct from `ChildEvaluation.result.mlflow_run_id`,
which (only when `log_to_mlflow=True`) names the run created to log that eval
result. `split`, `plots`, `log_to_mlflow`, `hooks`, `device`, and
`batch_size` are forwarded unchanged to every child `evaluate()` call.
Raises `WorkflowError` if `parent_run_id` does not exist or has no active
child runs.

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
- `load_model_from_settings()` resolves `model.checkpoint` from an
  `InferenceJobConfig` unless an explicit `checkpoint_path=` override is provided.
- `CheckpointPredictor` exposes `feature_names` and `predict_target_key` as public
  metadata properties.
- Checkpoints produced by the standard Lightning wrapper also persist
  `forward_arg_map`, allowing inference to reconstruct named feature dispatch
  and apply the correct feature transform before calling `model.forward(**kwargs)`.
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
