# Staged Optimization Subsystem

Supports multi-stage optimizer programs: sequential stage switching, concurrent
multi-optimizer groups, parameter partitioning by role or module path, and
plateau/epoch-based stage transitions.

---

## Mode selection

The Lightning wrapper inspects the assembled program and chooses a controller
automatically.

| Program shape | Controller | `automatic_optimization` |
|---|---|---|
| Single stage (default) | `AutomaticOptimizationController` | `True` |
| `ConcurrentOptimizerSettings` (N parallel sub-optimizers) | `AutomaticOptimizationController` | `True` |
| 2+ sequential `OptimizationStageSettings` | `ManualOptimizationController` | `False` |
| Any stage containing LBFGS | `ManualOptimizationController` | `False` |

**Sequential stages always use manual optimization.** Only the active stage's
optimizer is stepped per batch. Stage 1's optimizer is **not** updated while
stage 0 is active.

---

## Config patterns

### Default (single AdamW)

No explicit stages needed. `OptimizerPolicySettings()` produces AdamW with
`lr=1e-3` on all parameters.

```toml
# nothing required — defaults apply (AdamW lr=1e-3)
```

### Sequential stages (AdamW warm-up then fine-tune)

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 0.01}
trigger = {at_epoch = 10}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
# no trigger → runs for the remainder of training
```

### Sequential with plateau trigger

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-3}
trigger = {patience = 5, monitor = "val_loss", min_delta = 1e-4, mode = "min"}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
```

### Concurrent optimizers — Muon + AdamW (most common case, no stages needed)

`ConcurrentOptimizerSettings` fits anywhere an optimizer fits.
When `selectors` is omitted and at least one sub-optimizer is `MuonSettings`, the
builder auto-assigns `MuonEligibleSelector` to each Muon and `NonMuonSelector` to
the rest. **Omitting `selectors` is only valid when Muon is present** — for any
other concurrent split, explicit `selectors` are required.

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]
```

### Concurrent with explicit selectors (encoder/decoder split)

Use inline arrays so `optimizers` and `selectors` stay aligned:

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Adam", lr = 1e-3}, {name = "Adam", lr = 5e-4}]
selectors  = [{prefix = "encoder"},        {prefix = "decoder"}]
```

### Concurrent as one phase of a sequential program

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "Concurrent", optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]}
trigger = {at_epoch = 50}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
```

### With a learning rate scheduler

Schedulers attach at the stage level (including concurrent stages).

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 0.01}
scheduler = {name = "StepLR", step_size = 10, gamma = 0.1}
trigger = {at_epoch = 30}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-3}
```

---

## Python API

```python
from dlkit.infrastructure.config import (
    OptimizerPolicySettings,
    OptimizationStageSettings,
    TriggerSettings,
    ParameterSelectorSettings,
)
from dlkit.infrastructure.config.optimizer_component import (
    AdamWSettings,
    AdamSettings,
    MuonSettings,
    ConcurrentOptimizerSettings,
    StepLRSettings,
    ReduceLROnPlateauSettings,
)

# Single stage — simplest case (AdamW lr=1e-3, no scheduler)
program = OptimizerPolicySettings()

# Custom default optimizer + scheduler (no explicit stages needed)
program = OptimizerPolicySettings(
    default_optimizer=AdamWSettings(lr=5e-4, weight_decay=0.01),
    default_scheduler=ReduceLROnPlateauSettings(patience=10, factor=0.5),
)

# Muon + AdamW concurrent (auto-inferred selectors, no stages needed)
program = OptimizerPolicySettings(
    default_optimizer=ConcurrentOptimizerSettings(
        optimizers=(MuonSettings(lr=0.02), AdamWSettings(lr=3e-4))
    )
)

# Concurrent with explicit selectors (encoder/decoder split)
program = OptimizerPolicySettings(
    default_optimizer=ConcurrentOptimizerSettings(
        optimizers=(AdamWSettings(lr=1e-3), AdamWSettings(lr=5e-4)),
        selectors=(
            ParameterSelectorSettings(prefix="encoder"),
            ParameterSelectorSettings(prefix="decoder"),
        ),
    )
)

# Two sequential stages with an epoch trigger
program = OptimizerPolicySettings(
    stages=(
        OptimizationStageSettings(
            optimizer=AdamWSettings(lr=1e-2),
            trigger=TriggerSettings(at_epoch=10),
        ),
        OptimizationStageSettings(
            optimizer=AdamWSettings(lr=1e-4),
        ),
    )
)

# Sequential with plateau trigger + per-stage scheduler
program = OptimizerPolicySettings(
    stages=(
        OptimizationStageSettings(
            optimizer=AdamWSettings(lr=1e-3),
            scheduler=StepLRSettings(step_size=10, gamma=0.5),
            trigger=TriggerSettings(patience=5),
        ),
        OptimizationStageSettings(
            optimizer=AdamWSettings(lr=1e-4),
        ),
    )
)

# Concurrent as one phase of a sequential program
program = OptimizerPolicySettings(
    stages=(
        OptimizationStageSettings(
            optimizer=ConcurrentOptimizerSettings(
                optimizers=(MuonSettings(lr=0.02), AdamWSettings(lr=3e-4))
            ),
            trigger=TriggerSettings(at_epoch=50),
        ),
        OptimizationStageSettings(optimizer=AdamWSettings(lr=1e-4)),
    )
)
```
