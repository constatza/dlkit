# Optimization Subsystem

`dlkit.engine.training.optimization` owns live optimizer-policy assembly,
manual/automatic optimization control, scheduler wiring, transition triggers,
parameter partitioning, and optimization-state checkpoint round-tripping.

## Parameter Role Classification

Default Muon role classification is graph-based and model-agnostic.

- `GraphParameterRoleClassifier` inspects the `nn.Module` tree and uses
  `torch.fx` to classify executed parameter-owning sites as `INPUT`,
  `HIDDEN`, or `OUTPUT`.
- Structural roles such as `BIAS`, `NORMALIZATION`, and `EMBEDDING` are
  assigned from exact module ownership before graph boundary reduction.
- Composite wrappers are traced through to the fundamental parameter-owning
  sublayers that actually sit on the executed input/output boundary.
- No default naming heuristics are used.
- Ambiguous, unsupported, shared, tied, or untraceable parameters remain
  `UNKNOWN` and therefore stay on the general-purpose optimizer.

## Mode Selection

The Lightning wrapper inspects the assembled optimizer program and chooses the
controller automatically.

| Program shape | Controller | `automatic_optimization` |
|---|---|---|
| Single stage | `AutomaticOptimizationController` | `True` |
| Single concurrent stage | `AutomaticOptimizationController` | `True` |
| 2+ sequential stages | `ManualOptimizationController` | `False` |
| Any stage containing LBFGS | `ManualOptimizationController` | `False` |

Sequential stages always use manual optimization. Only the current stage
optimizer is stepped per batch; inactive stages remain idle until the program
advances.

When a staged/manual program runs inside a Lightning trainer, DLKit uses
Lightning's manual-optimization APIs for the active stage:
- `manual_backward()` drives backward so precision/strategy hooks remain active
- `optimizers(use_pl_optimizer=True)` resolves the active Lightning optimizer
- the active optimizer runs inside Lightning's `toggle_model()` scope

When no trainer is attached, the same step policies fall back to raw PyTorch
`zero_grad()`, `backward()`, and `step()`. This keeps direct unit tests and
isolated controller tests working without a Trainer.

## Scheduler Semantics

Schedulers are attached to optimizer-policy stages.

- Use `default_scheduler` only when `TRAINING.optimizer.stages` is empty.
- Use `TRAINING.optimizer.stages[*].scheduler` for staged programs.
- A concurrent stage still has exactly one stage scheduler, attached to the
  `ConcurrentOptimizer` wrapper that owns all sub-optimizers.
- LBFGS stages can use schedulers; closure-based stepping affects optimizer
  execution semantics, not scheduler compatibility.
- In manual mode, the current stage scheduler is stepped at `on_epoch_end` for
  the epoch that just finished, before any trigger evaluation or stage advance.
- `scheduler_frequency` is treated as an epoch cadence in manual mode.
- `ReduceLROnPlateau` consumes the configured monitor metric from callback
  metrics; if that metric is missing while the scheduler is due, training raises
  a `WorkflowError` instead of silently leaving the scheduler stale.

The manual epoch-end order is:
1. step the current stage scheduler if due
2. evaluate the current stage trigger
3. reset the trigger if it fired
4. advance to the next stage

Automatic mode is unchanged: Lightning owns scheduler stepping through
`configure_optimizers()`.

## LR Tuning Semantics

Lightning LR finder works only with a single optimizer view. DLKit therefore
keeps LR tuning separate from the live optimization program:

- single-stage policies are tuned directly
- sequential staged policies are tuned through a temporary projection that
  contains only stage 0
- the suggested learning rate is written back only to stage 0 of the real
  policy before training starts
- later stages keep their configured learning rates unchanged
- concurrent stage-0 policies and closure-based stage-0 optimizers are rejected
  explicitly for LR finder in v1

## Concurrent Closure Semantics

`ConcurrentOptimizer` forwards closures selectively:
- with no closure, all sub-optimizers step normally
- with a closure and no LBFGS sub-optimizer, the closure is executed exactly
  once before all sub-optimizers step
- with exactly one LBFGS sub-optimizer, only that LBFGS optimizer receives the
  closure; after its final closure evaluation, all other sub-optimizers step on
  the resulting gradients
- with multiple LBFGS sub-optimizers, training raises immediately instead of
  duplicating closure execution across sub-optimizers

## Config Patterns

### Default single optimizer

No explicit stages are required. `OptimizerPolicySettings()` defaults to AdamW
with `lr=1e-3`.

```toml
# nothing required — defaults apply
```

### Default optimizer with scheduler

```toml
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
```

### Default Muon-family with automatic companion split

When `default_optimizer` is a lone `MuonSettings` or `BatchedMuonSettings`,
the builder partitions the model automatically. The Muon-family optimizer
receives only 2D hidden-layer weights selected by `MuonEligibleSelector`;
every remaining parameter is routed to an AdamW companion at the same learning
rate. Muon-family settings default `adjust_lr_fn` to `"match_rms_adamw"`, so
this auto-split path follows the PyTorch-recommended mode for reusing
AdamW-tuned learning rate and weight decay values on the Muon side. If every
parameter is Muon-eligible, the builder returns a plain
Muon-family optimizer with no companion wrapper.

This is the convenience path: it optimizes for user-friendliness, not maximum
companion-optimizer configurability.

```toml
[TRAINING.optimizer.default_optimizer]
name = "Muon"
lr = 0.02
```

```toml
[TRAINING.optimizer.default_optimizer]
name = "BatchedMuon"
lr = 0.02
```

### Sequential stages

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 0.01}
trigger = {at_epoch = 10}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
```

A lone `Muon` or `BatchedMuon` in a stage receives the same automatic
`MuonEligibleSelector` / `NonMuonSelector` split as in the default path.
Use `ConcurrentOptimizerSettings` when explicit control over the companion
optimizer is needed.

### Sequential with plateau trigger and per-stage scheduler

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-3}
scheduler = {name = "StepLR", step_size = 10, gamma = 0.5, frequency = 1}
trigger = {patience = 5, monitor = "val_loss", min_delta = 1e-4, mode = "min"}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
scheduler = {name = "ReduceLROnPlateau", factor = 0.5, patience = 3, monitor = "val_loss"}
```

### Concurrent optimizers

`ConcurrentOptimizerSettings` fits anywhere an optimizer fits. When `selectors`
are omitted and **exactly one** sub-optimizer is `MuonSettings` or
`BatchedMuonSettings`, the builder auto-assigns `MuonEligibleSelector` to that
optimizer and `NonMuonSelector` to the rest. Two or more Muon-family optimizers
with empty selectors raise `ValidationError` at config construction time (both
would receive identical parameter sets). For any other concurrent split, explicit
selectors are required.

This is the explicit path: use it when you want independent control over the
Muon-family optimizer and the companion AdamW settings such as raw `lr`,
`weight_decay`, or future companion choices.

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]
```

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [
  {name = "Muon", lr = 0.02, adjust_lr_fn = "match_rms_adamw"},
  {name = "AdamW", lr = 3e-4, weight_decay = 0.01},
]
```

### Concurrent as one stage of a sequential program

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "Concurrent", optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]}
scheduler = {name = "StepLR", step_size = 20, gamma = 0.1}
trigger = {at_epoch = 50}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
```

## Python API

```python
from dlkit.infrastructure.config import (
    OptimizerPolicySettings,
    OptimizationStageSettings,
    ParameterSelectorSettings,
    TriggerSettings,
)
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    BatchedMuonSettings,
    ConcurrentOptimizerSettings,
    MuonSettings,
    ReduceLROnPlateauSettings,
    StepLRSettings,
)

program = OptimizerPolicySettings(
    default_optimizer=AdamWSettings(lr=5e-4, weight_decay=0.01),
    default_scheduler=ReduceLROnPlateauSettings(patience=10, factor=0.5),
)

program = OptimizerPolicySettings(
    stages=(
        OptimizationStageSettings(
            optimizer=AdamWSettings(lr=1e-3),
            scheduler=StepLRSettings(step_size=10, gamma=0.5),
            trigger=TriggerSettings(patience=5, monitor="val_loss"),
        ),
        OptimizationStageSettings(
            optimizer=AdamSettings(lr=1e-4),
            scheduler=ReduceLROnPlateauSettings(patience=3, factor=0.5),
        ),
    )
)

program = OptimizerPolicySettings(
    default_optimizer=ConcurrentOptimizerSettings(
        optimizers=(AdamWSettings(lr=1e-3), AdamWSettings(lr=5e-4)),
        selectors=(
            ParameterSelectorSettings(prefix="encoder"),
            ParameterSelectorSettings(prefix="decoder"),
        ),
    )
)

program = OptimizerPolicySettings(
    default_optimizer=MuonSettings(lr=0.02)
)

program = OptimizerPolicySettings(
    default_optimizer=BatchedMuonSettings(lr=0.02)
)

program = OptimizerPolicySettings(
    stages=(
        OptimizationStageSettings(
            optimizer=ConcurrentOptimizerSettings(
                optimizers=(MuonSettings(lr=0.02), AdamWSettings(lr=3e-4))
            ),
            scheduler=StepLRSettings(step_size=20, gamma=0.1),
            trigger=TriggerSettings(at_epoch=50),
        ),
        OptimizationStageSettings(optimizer=AdamWSettings(lr=1e-4)),
    )
)
```
