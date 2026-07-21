# I/O Module

`dlkit.infrastructure.io` owns filesystem-facing support code.

## Responsibilities
- TOML file loading and config-loading errors
- path preprocessing and legacy path-context state
- array, table, index, and sparse-pack I/O
- collision-safe filesystem imports for user modules
- safe NumPy-to-tensor conversion, including copying read-only buffers before
  exposing them as writable PyTorch tensors

## Current Layout
- `config_loader.py`: TOML readers and eager typed config loaders
- `config_errors.py`: config loading exceptions
- `path_resolver.py`: path normalization helpers used by config preprocessing
- `path_context.py` / `path_context_state.py`: compatibility shims for older path-override flows
- `explicit_path_context.py`: explicit path-context structures used during migration
- `paths.py`, `locations.py`: user-path normalization and DLKit-internal locations
- `arrays.py`, `tables.py`, `index.py`, `tensor_entries.py`, `packs/`: data-loading helpers
  (`index.py` reads external split files as JSON or TOML, dispatched by suffix)
- `system.py`: module/class loading from modules or filesystem paths

## Path Resolution Architecture

Workflow config path preprocessing resolves relative paths from the config file
location. Dataset-owned paths may additionally anchor to `data.root`.
DLKit no longer uses a global project root setting.

### Internal Locations
- `locations.output(...)` resolves under `DLKIT_INTERNAL_DIR` (default
  `.dlkit/`) for DLKit-owned internal files. Currently only the local MLflow
  SQLite tracking DB (`.dlkit/mlflow/mlflow.db`) uses it, and only when MLflow
  tracking is explicitly enabled (`--mlflow` / `tracking.backend = "mlflow"`)
  with no server or explicit URI reachable — `.dlkit/` is never created on a
  plain `dlkit train` run.
- File logging is opt-in and lives under the same directory
  (`.dlkit/logs/dlkit_<timestamp>.log`) via `--log-file` or `DLKIT_LOG_FILE`;
  see `dlkit.infrastructure.utils.utils.md`. Without either, no log file or
  directory is created — only stderr is used.
- `locations.py` should be treated as DLKit-internal infrastructure only, not
  as the owner of user-facing predictions/checkpoints/splits directories.
- Generated index splits are persisted locally under
  `training.trainer.default_root_dir/splits/` whenever that root is
  configured (needed so evaluation can later reload the exact split used at
  training time rather than regenerating one); with no local root
  configured, a generated split stays in-memory only.
- Durable run artifacts belong to the active tracking backend and should be
  logged through `IRunContext`.
- Non-MLflow training outputs should be contained by Lightning under
  `training.trainer.default_root_dir`.

## Ownership Boundary
- `tools.io` owns raw config loading and config-relative path preprocessing.
- `tools.config` owns typed settings, validation, patching, and workflow models.
- `tools.io` owns DLKit-internal location helpers.

## Notes
- Dynamic filesystem imports use unique module names internally and only register the original module stem in `sys.modules` when that alias is safe.
- `arrays.py` owns NumPy buffer mutability handling. Callers should not need to
  special-case read-only `.npy`, `.npz`, or in-memory `ndarray` inputs before
  converting them to tensors.
- `split_provider.py` exposes `resolve_training_split()` (producer — may
  generate a fresh, seeded split via `RatioSplitStrategy`, or reload via
  `ExternalFileSplitStrategy` when an explicit filepath is given; always
  persists the resolved split when a local path is provided) and
  `resolve_evaluation_split()` (consumer — has no ratio/seed parameters at
  all, can only reload a persisted split file). `ExternalFileSplitStrategy`
  lives here rather than alongside `RatioSplitStrategy` in
  `infrastructure.types.split` specifically to avoid a `types -> io`
  dependency cycle (`io` already depends on `types` for
  `IndexSplit`/`SplitStrategy`). `split_provider.py` applies configured
  training-split caps after split generation or loading so convergence
  sweeps can evaluate bounded sample sizes without changing durable split
  definitions.
- `index.py`'s `save_split_indices` writes atomically (temp file in the same
  directory + `os.replace`) and treats a destination that already exists
  with byte-identical content as a no-op — this matters because parallel
  Optuna trials sharing one training root can otherwise race writing the
  same split file.
