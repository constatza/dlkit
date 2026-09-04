# Compute Topology

`dlkit.infrastructure.compute` resolves node/device topology (`devices`,
`num_nodes`) for the Lightning `Trainer`, bridging to Lightning's own
`ClusterEnvironment` plugins (`lightning.fabric.plugins.environments`)
instead of reimplementing scheduler detection or rank/world-size wiring —
Lightning already owns that, unconditionally, on every `Trainer`
construction. This package only answers one question per environment: how
many nodes, how many devices per node.

## Modules

| Module | Responsibility |
|--------|---------------|
| `topology.py` | `ComputeTopology` (resolved node/device counts), `ComputeEnvironment` (ABC bridging to one Lightning `ClusterEnvironment` class) |
| `environments.py` | Concrete environments: Local, SLURM, TorchElastic, LSF, MPI, Kubeflow. The only file that reads scheduler env vars for this concern. |
| `resolver.py` | `resolve_compute_environment()` — detection-priority selection, or explicit override |

## What Works Automatically vs. What You Must Set Explicitly

| Environment | Launch mechanism | Rank/world-size wiring | `devices`/`num_nodes` |
|---|---|---|---|
| Local | plain `python`/`dlkit train ...` | N/A (single process) | Automatic — `num_nodes=1, devices="auto"` |
| SLURM | `srun ...` | Automatic (Lightning's `SLURMEnvironment`) | Automatic — derived from `SLURM_NNODES`/`SLURM_NTASKS_PER_NODE` |
| TorchElastic | `torchrun ...` | Automatic (Lightning's `TorchElasticEnvironment`) | Automatic — derived from `WORLD_SIZE`/`LOCAL_WORLD_SIZE` |
| LSF | `jsrun ...` | Automatic (Lightning's `LSFEnvironment`) | **Manual** — `run.compute.devices`/`.num_nodes` are required fields when `environment = "lsf"` |
| MPI | `mpirun`/`mpiexec ...` + `mpi4py` installed | Automatic (Lightning's `MPIEnvironment`) | **Manual** — required fields when `environment = "mpi"` |
| Kubeflow | PyTorchJob operator | **Manual** — set `run.compute.environment = "kubeflow"` explicitly | **Manual** — required fields when `environment = "kubeflow"` |

Why LSF/MPI/Kubeflow can't auto-derive topology:

- **LSF** exposes rank/world-size via `jsrun`-set env vars, but node count
  is only recoverable by parsing the `LSB_DJOB_RANKFILE` host list (which
  Lightning does internally via a private method) — not depended on here.
- **MPI** exposes nothing via env vars at all; Lightning's own
  `MPIEnvironment` computes local rank/node rank via live `mpi4py`
  collective calls (`comm.gather`/`comm.bcast`) *after* the process group
  exists. Running those calls just to size a `Trainer` before training even
  starts would be invasive and easy to get wrong.
- **Kubeflow**'s `PyTorchJob` CRD sets `MASTER_ADDR`/`MASTER_PORT`/
  `WORLD_SIZE`/`RANK` but no per-node device count, and Lightning's own
  `KubeflowEnvironment.detect()` literally `raise`s `NotImplementedError` —
  Kubeflow support must always be opted into explicitly, never auto-selected.

For SLURM/TorchElastic, once launched correctly (`srun`/`torchrun`), no
compute-related config is needed at all — `[training.trainer]` doesn't have
to mention `compute` or `strategy`. A common SLURM mistake Lightning itself
warns about: submitting via plain `sbatch` without `srun` in front of the
python command — that only requests the allocation, it doesn't launch one
process per task, so `SLURM_NTASKS` etc. won't reflect what you expect.

## Configuration

Two settings models divide this cleanly, each matching its own runtime
counterpart:

- **`TrainerSettings.devices`/`.num_nodes`/`.strategy`** — plain fields
  mirroring `Trainer.__init__` directly, the same way `accelerator`/
  `precision` already do. This is where you force a specific value for
  *this* trainer. When left `None`, `TrainerSettings.build()` derives them.
- **`RunSettings.compute`** (`[run.compute]`) — says *which environment* to
  resolve against. Job-wide config, like `run.precision`/`run.seed`, not a
  `Trainer` constructor argument, so it doesn't belong on the settings model
  whose whole purpose is reflection into `Trainer(**kwargs)`.
  `TrainerSettings.build(session=...)` reads `session.compute` the same way
  it already reads `session.precision` — see
  [`../config/config.md`](../config/config.md#compute-topology).

`ComputeEnvironmentSettings` (`infrastructure/config/compute_settings.py`,
named after the `ComputeEnvironment` runtime class it selects between) is a
discriminated union keyed on `environment`. `AutoComputeSettings`,
`LocalComputeSettings`, `SlurmComputeSettings`, `TorchElasticComputeSettings`
carry no fields beyond `environment` — forcing a value for these is
`TrainerSettings`'s job, not `ComputeEnvironmentSettings`'s. `LSFComputeSettings`,
`MPIComputeSettings`, `KubeflowComputeSettings` are the exception: they
declare `devices`/`num_nodes` as **required** fields, because those three
environments structurally cannot auto-derive them — omitting them fails at
config-load time (`ValidationError`), not silently at runtime.

```toml
[run]
type = "train"

[run.compute]
environment = "auto"   # "local" | "slurm" | "torchelastic" | "lsf" | "mpi" | "kubeflow"
# devices = 4           # only declared on (and required by) lsf/mpi/kubeflow
# num_nodes = 2         # only declared on (and required by) lsf/mpi/kubeflow

[training.trainer]
accelerator = "gpu"
strategy = "ddp"       # optional; Lightning defaults to "ddp" when
                       # devices/num_nodes resolve to more than one process
# devices = 4           # optional; overrides whatever run.compute resolves to
# num_nodes = 2          # optional; overrides whatever run.compute resolves to
```

Precedence in `TrainerSettings.build()`, three tiers: this trainer's own
explicit `devices`/`num_nodes` win first; then the compute environment's
required fields (LSF/MPI/Kubeflow only); then auto-detected topology
(local/SLURM/torchrun).

## Two Axes, Kept Separate

- **Where the job runs** (node/device counts): `TrainerSettings.devices`/
  `.num_nodes`, resolved via `run.compute` when unset.
- **How work is parallelized once running** (`TrainerSettings.strategy`): a
  plain Lightning passthrough. DLKit does not select or validate
  strategies — Lightning's own `_choose_strategy()` already defaults to
  `"ddp"` when multiple devices/nodes are resolved, and to single-device
  otherwise.

## Extending to a New Scheduler

Two files, not one: add a `ComputeEnvironment` subclass in `environments.py`
naming the Lightning `ClusterEnvironment` it bridges to, plus one entry in
`resolver.py`'s `_ENVIRONMENTS` (detection order) or `_EXPLICIT_ONLY` (if it
can never be auto-detected, like Kubeflow) — `TrainerSettings.build()` itself
needs no changes, it only calls `resolve_compute_environment()`. You also
need a matching settings class in `infrastructure/config/compute_settings.py`'s
`ComputeEnvironmentSettings` union, using the same environment name, or the
new environment is resolvable but never selectable from config. A parity
test (`tests/infrastructure/config/test_compute_settings.py`) asserts the two
enumerations agree.

## I/O Notes

- **Inputs** (dataset loading): no special handling needed. Each DDP rank
  independently builds its own dataset/DataLoader shard via Lightning's
  `DistributedSampler` — this is correct by construction. The only
  requirement is that `DLKIT_ROOT_DIR`/`run.root_dir` point at storage
  reachable from every node; `sbatch`/`srun` propagate the submitting
  shell's environment to every node by default.
- **Outputs** (MLflow tracking): guarded separately in
  `engine.tracking.tracking_decorator.TrackingDecorator` via
  `trainer.is_global_zero` — only rank 0 creates/owns the MLflow run. See
  `engine/tracking/tracking.md`.
