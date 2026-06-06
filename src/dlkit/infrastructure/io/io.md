# I/O Module

`dlkit.infrastructure.io` owns filesystem-facing support code.

## Responsibilities
- TOML file loading and partial section reads
- config-section registry and config-loading errors
- path preprocessing and legacy path-context state
- array, table, index, and sparse-pack I/O
- collision-safe filesystem imports for user modules

## Current Layout
- `config_loader.py`: TOML readers and writers
- `config_section_registry.py`: section-to-model mapping
- `config_errors.py`: config loading exceptions
- `path_resolver.py`: path normalization helpers used by config preprocessing
- `path_context.py` / `path_context_state.py`: compatibility shims for older path-override flows
- `explicit_path_context.py`: explicit path-context structures used during migration
- `paths.py`, `locations.py`: user-path normalization and DLKit-internal locations
- `arrays.py`, `tables.py`, `index.py`, `tensor_entries.py`, `packs/`: data-loading helpers
- `system.py`: module/class loading from modules or filesystem paths

## Path Resolution Architecture

Workflow config path preprocessing resolves relative paths from the config file
location. Dataset-owned paths may additionally anchor to `DATASET.root_dir`.
DLKit no longer uses a global project root setting.

### Internal Locations
- `locations.output(...)` resolves under `DLKIT_INTERNAL_DIR` (default
  `.dlkit/`) for DLKit-owned internal files such as local MLflow/Optuna
  databases.
- `locations.py` should be treated as DLKit-internal infrastructure only, not
  as the owner of user-facing predictions/checkpoints/splits directories.
- Generated index splits do not create local files by default.
- Durable run artifacts belong to the active tracking backend and should be
  logged through `IRunContext`.
- Non-MLflow training outputs should be contained by Lightning under
  `TRAINING.trainer.default_root_dir`.

## Ownership Boundary
- `tools.io` owns raw config loading and section resolution.
- `tools.config` owns typed settings, validation, patching, and workflow models.
- `tools.io` owns config-relative path preprocessing and DLKit-internal
  location helpers.

## Notes
- Dynamic filesystem imports use unique module names internally and only register the original module stem in `sys.modules` when that alias is safe.
