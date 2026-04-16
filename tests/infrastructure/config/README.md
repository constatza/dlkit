# DLKit Config Test Suite

This directory covers the configuration layer, workflow loaders, patching utilities, and precision handling.

## Layout

```text
tests/tools/config/
├── README.md
├── conftest.py
├── test_mlflow_settings.py
├── test_partial_config_loading.py
├── test_eager_validation.py
├── test_environment.py
├── test_updater.py
├── test_dataset_settings.py
├── test_data_entries.py
├── test_sparse_feature.py
├── test_integration.py
├── precision/
├── core/
└── components/
```

## Main Coverage Areas

- `test_mlflow_settings.py`
  - Validates the flat `MLflowSettings` model
  - Confirms legacy `server` / `client` blocks are rejected
  - Confirms `tracking_uri` / `artifacts_destination` are env-only and invalid in TOML
  - Covers `max_retries`, aliases, tags, and defaults

- `test_partial_config_loading.py`
  - Covers `load_settings()` returning `TrainingWorkflowSettings`
  - Covers `load_sections()` partial workflow loading
  - Verifies strict mode for missing sections

- `test_eager_validation.py` and `test_config_missing_path.py`
  - Verify fail-fast validation for malformed configs and missing paths

- `test_environment.py`
  - Verifies MLflow retry-related environment defaults

- `test_updater.py`
  - Covers strict in-place config mutation via `update_settings()`

- `precision/`
  - Covers `PrecisionStrategy`, context handling, services, and end-to-end precision behavior

- `core/`
  - Covers base settings classes, factories, and patch compilation utilities

- `components/`
  - Covers component settings models such as `ModelComponentSettings`

## Configuration Behaviors Under Test

### Flat MLflow configuration

The test suite assumes the current MLflow contract:

```toml
[MLFLOW]
experiment_name = "baseline"
run_name = "trial-01"
tags = { team = "platform" }
register_model = true
registered_model_name = "FFNN"
registered_model_aliases = ["candidate"]
registered_model_version_tags = { team = "platform" }
max_retries = 3
```

The `[MLFLOW]` section itself enables tracking. The old `enabled` field is
intentionally invalid and covered by tests.

Infrastructure is env-driven:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_ARTIFACT_URI`

These TOML shapes are intentionally invalid and covered by tests:

- `[MLFLOW.server]`
- `[MLFLOW.client]`
- `tracking_uri = "..."`
- `artifacts_destination = "..."`

### Protocol-oriented loading

The low-level config I/O surface is based on the protocol contracts in `src/dlkit/tools/io/protocols.py`:

- `ConfigParser`
- `SectionExtractor`
- `ConfigValidator[T]`
- `PartialConfigReader`

The tests exercise the concrete behavior through:

- `load_settings()`
- `load_sections()`
- `load_sections_config()`
- `load_section_config()`
- section mapping registration / reset helpers

## Running the Suite

```bash
uv run pytest tests/tools/config -v
```

Focused runs:

```bash
uv run pytest tests/tools/config/test_mlflow_settings.py -v
uv run pytest tests/tools/config/test_partial_config_loading.py -v
uv run pytest tests/tools/config/precision -v
```
