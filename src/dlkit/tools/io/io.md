# I/O Module

`dlkit.tools.io` owns filesystem-facing support code.

## Responsibilities
- TOML file loading and partial section reads
- config-section registry and config-loading errors
- path preprocessing and path-context state
- array, table, index, and sparse-pack I/O
- collision-safe filesystem imports for user modules

## Current Layout
- `config_loader.py`: TOML readers and writers
- `config_section_registry.py`: section-to-model mapping
- `config_errors.py`: config loading exceptions
- `path_context.py` / `path_context_state.py`: request-scoped path overrides
- `paths.py`, `locations.py`, `provisioning.py`: path resolution and directory provisioning
- `arrays.py`, `tables.py`, `index.py`, `tensor_entries.py`, `sparse/`: data-loading helpers
- `system.py`: module/class loading from modules or filesystem paths

## Ownership Boundary
- `tools.io` owns raw config loading and section resolution.
- `tools.config` owns typed settings, validation, patching, and workflow models.

## Notes
- Path resolution precedence is: explicit path context, `DLKIT_ROOT_DIR`, `SESSION.root_dir`, current working directory.
- Dynamic filesystem imports use unique module names internally and only register the original module stem in `sys.modules` when that alias is safe.
