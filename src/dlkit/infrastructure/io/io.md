# I/O Module

`dlkit.infrastructure.io` owns filesystem-facing support code.

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
- `path_resolver.py`: unified path resolution service (single source of truth)
- `path_context.py` / `path_context_state.py`: request-scoped path overrides
- `explicit_path_context.py`: explicit (non-thread-local) path context structures
- `paths.py`, `locations.py`: path resolution and directory provisioning
- `arrays.py`, `tables.py`, `index.py`, `tensor_entries.py`, `sparse/`: data-loading helpers
- `system.py`: module/class loading from modules or filesystem paths

## Path Resolution Architecture

**PathResolver** (`path_resolver.py`) is the unified, centralized service for all path resolution.

### Design
- **Single source of truth**: All path resolution logic consolidated in `PathResolver` class
- **Dependency injection**: Accepts thread-local context and environment settings as inputs (no implicit global reads)
- **Clear precedence**: context > DLKIT_ROOT_DIR > SESSION.root_dir > cwd
- **Pure function semantics**: Given the same inputs, always produces the same output

### Precedence Rules
When resolving paths, PathResolver applies this priority order:
1. **Thread-local context** (API/CLI overrides via `get_current_path_context()`)
2. **DLKIT_ROOT_DIR env var** (via EnvironmentSettings)
3. **SESSION.root_dir** (config-based, propagated at load time)
4. **Current working directory** (fallback)

### Public API
- `resolve(path: Path | str | None) -> Path`: Resolve a path or root directory
- `resolve_component_path(component_path: str) -> Path`: Resolve special DLKit paths (output, data, checkpoints)
- `from_defaults()`: Factory method using current thread-local context and global environment
- `get_root() -> Path`: Shorthand for root directory resolution
- `has_context_override() -> bool`: Check if thread-local context has explicit override
- `has_env_override() -> bool`: Check if DLKIT_ROOT_DIR env var is set

### Migration Notes
- Old `resolve_with_context()` now delegates to `PathResolver`
- Old `_sync_session_root_to_environment()` simplified to wrapper (mutation no longer required as PathResolver handles it)
- `locations.py` functions use PathResolver for all path resolution

## Ownership Boundary
- `tools.io` owns raw config loading and section resolution.
- `tools.config` owns typed settings, validation, patching, and workflow models.
- `tools.io` owns path resolution logic via `PathResolver` (centralized, no cross-layer mutation)

## Notes
- Path resolution precedence is explicit and enforced in `PathResolver._resolve_root()`
- Dynamic filesystem imports use unique module names internally and only register the original module stem in `sys.modules` when that alias is safe.
