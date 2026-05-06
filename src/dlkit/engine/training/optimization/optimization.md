# Optimization Subsystem

`dlkit.engine.training.optimization` owns live optimizer-policy assembly,
manual/automatic optimization control, scheduler wiring, transition triggers,
parameter partitioning, and optimization-state checkpoint round-tripping.

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

## Scheduler Semantics

Schedulers are attached to optimizer-policy stages.

- Use `default_scheduler` only when `TRAINING.optimizer.stages` is empty.
- Use `TRAINING.optimizer.stages[*].scheduler` for staged programs.
- A concurrent stage still has exactly one stage scheduler, attached to the
  `ConcurrentOptimizer` wrapper that owns all sub-optimizers.
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

### Default Muon with automatic companion split

When `default_optimizer` is a lone `MuonSettings`, the builder partitions the
model automatically. Muon receives only 2D hidden-layer weights selected by
`MuonEligibleSelector`; every remaining parameter is routed to an AdamW
companion at the same learning rate. If every parameter is Muon-eligible, the
builder returns a plain Muon optimizer with no companion wrapper.

```toml
[TRAINING.optimizer.default_optimizer]
name = "Muon"
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
are omitted and at least one sub-optimizer is `MuonSettings`, the builder
auto-assigns `MuonEligibleSelector` to Muon and `NonMuonSelector` to the rest.
For any other concurrent split, explicit selectors are required.

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]
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
