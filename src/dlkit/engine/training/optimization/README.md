# Staged Optimization Subsystem

Supports multi-stage optimizer programs: sequential stage switching, concurrent
multi-optimizer groups, parameter partitioning by role or module path, and
plateau/epoch-based stage transitions.

---

## Mode selection

The Lightning wrapper inspects the assembled program and chooses a controller
automatically. You never set this manually.

| Program shape | Controller | `automatic_optimization` |
|---|---|---|
| Single stage (default) | `AutomaticOptimizationController` | `True` |
| `ConcurrentOptimizationSettings` (N parallel optimizers) | `AutomaticOptimizationController` | `True` |
| 2+ sequential `OptimizationStageSettings` | `ManualOptimizationController` | `False` |
| Any stage containing LBFGS | `ManualOptimizationController` | `False` |

**Sequential stages always use manual optimization.** Registering all optimizers
with Lightning's automatic mode would cause every stage's optimizer to step on
every batch, defeating staged training. The manual controller reads
`program.current` each step and only advances when the trigger fires.

---

## Discriminator fields in TOML

`stages` and `trigger` are **discriminated unions**. When loading from TOML or
any dict-based source, Pydantic requires the discriminator field to be present
in the data to know which variant to instantiate.

| Field | Discriminator key | Values |
|---|---|---|
| `[[OPTIMIZATION.stages]]` | `kind` | `"stage"`, `"concurrent"` |
| `trigger.*` | `kind` | `"epoch"`, `"plateau"` |
| `selector.*` | `kind` | `"role"`, `"module_path"`, `"muon_eligible"`, `"non_muon"`, `"intersection"`, `"union"`, `"difference"` |

Python construction is unaffected — all discriminator fields have defaults and
can be omitted when building objects in code.

---

## Config patterns

### Default (single AdamW)
No explicit stages needed. `OptimizerPolicySettings()` produces AdamW with
`lr=1e-3` on all parameters.

```toml
[OPTIMIZATION]
# nothing required — defaults apply
```

### Sequential stages (SGD then Adam)

Provide a list of `[[OPTIMIZATION.stages]]` entries. The first stage runs until
its trigger fires; the program then advances to the next stage permanently.
Each entry must include `kind = "stage"` and each trigger must include
`kind` so Pydantic can identify the variant.

```toml
[[OPTIMIZATION.stages]]
kind = "stage"
optimizer.name = "SGD"
optimizer.lr = 0.01

  [OPTIMIZATION.stages.trigger]
  kind = "epoch"
  at_epoch = 10              # switch to Adam at epoch 10

[[OPTIMIZATION.stages]]
kind = "stage"
optimizer.name = "Adam"
optimizer.lr = 1e-3
# no trigger → runs for the remainder of training
```

Because sequential stages force `ManualOptimizationController`, only the
currently active stage's optimizer is stepped per batch. Stage 1's Adam
parameters are **not** updated while stage 0 is active.

### Sequential with plateau trigger

```toml
[[OPTIMIZATION.stages]]
kind = "stage"
optimizer.name = "SGD"

  [OPTIMIZATION.stages.trigger]
  kind = "plateau"
  monitor = "val_loss"
  patience = 5
  min_delta = 1e-4
  mode = "min"

[[OPTIMIZATION.stages]]
kind = "stage"
optimizer.name = "Adam"
optimizer.lr = 1e-4
```

### Concurrent optimizers (GAN-style, disjoint parameter sets)

Use `kind = "concurrent"` — a single stage entry with nested `optimizers`.
All optimizers in the group step **every** batch on their respective parameter
subsets. The nested `optimizers` entries are `OptimizationStageSettings`
directly, not a discriminated union, so `kind` is not required inside them.

```toml
[[OPTIMIZATION.stages]]
kind = "concurrent"
# group-level trigger (advances beyond this whole group when it fires)
# omit or set trigger = null for no group-level transition

[[OPTIMIZATION.stages.optimizers]]
optimizer.name = "Adam"

  [OPTIMIZATION.stages.optimizers.selector]
  kind = "module_path"
  prefix = "encoder"

[[OPTIMIZATION.stages.optimizers]]
optimizer.name = "Adam"

  [OPTIMIZATION.stages.optimizers.selector]
  kind = "module_path"
  prefix = "decoder"
```

Concurrent groups use `AutomaticOptimizationController`; Lightning receives a
list of per-optimizer config dicts (including schedulers when present).

### With a learning rate scheduler

Schedulers attach per-stage, not per-program. Extra kwargs are forwarded to the
PyTorch scheduler constructor via `extra="allow"` on `SchedulerComponentSettings`.

```toml
[[OPTIMIZATION.stages]]
kind = "stage"
optimizer.name = "SGD"
optimizer.lr = 0.1
scheduler.name = "StepLR"
scheduler.step_size = 10
scheduler.gamma = 0.1

  [OPTIMIZATION.stages.trigger]
  kind = "epoch"
  at_epoch = 30

[[OPTIMIZATION.stages]]
kind = "stage"
optimizer.name = "Adam"
optimizer.lr = 1e-3
```

---

## Key constraint: sequential stages vs. concurrent groups

| Want | Use |
|---|---|
| Optimizer A for first N epochs, then optimizer B | Sequential `OptimizationStageSettings` list with `kind = "stage"` |
| Two optimizers updating **simultaneously** on different parameters | `kind = "concurrent"` with `selector` per sub-stage |
| Single optimizer on all params (most cases) | Default / empty stages |

Mixing sequential and concurrent entries in the same program is supported.
A program can contain `[concurrent_group, sequential_stage, sequential_stage]`;
the sequential entries will still force manual optimization.

---

## Python API (no discriminator fields needed)

When constructing settings in Python, defaults apply and discriminator fields
can be omitted entirely:

```python
from dlkit.infrastructure.config import (
    OptimizerPolicySettings,
    OptimizationStageSettings,
    ConcurrentOptimizationSettings,
    EpochTriggerSettings,
    PlateauTriggerSettings,
    RoleSelectorSettings,
    ModulePathSelectorSettings,
)

# Single stage — simplest case
program = OptimizerPolicySettings()

# Two sequential stages with an epoch trigger
program = OptimizerPolicySettings(
    stages=(
        OptimizationStageSettings(
            trigger=EpochTriggerSettings(at_epoch=10),
        ),
        OptimizationStageSettings(
            trigger=PlateauTriggerSettings(patience=5),
        ),
    )
)

# Concurrent group with parameter selectors
program = OptimizerPolicySettings(
    stages=(
        ConcurrentOptimizationSettings(
            optimizers=(
                OptimizationStageSettings(
                    selector=ModulePathSelectorSettings(prefix="encoder"),
                ),
                OptimizationStageSettings(
                    selector=ModulePathSelectorSettings(prefix="decoder"),
                ),
            )
        ),
    )
)
```
