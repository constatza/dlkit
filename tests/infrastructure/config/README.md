# DLKit Config Test Suite

This directory covers the configuration layer, workflow loaders, patching utilities, and precision handling.

## Layout

```text
tests/infrastructure/config/
├── README.md
├── conftest.py
├── test_job_config.py
├── test_load_job.py
├── test_eager_validation.py
├── test_environment.py
├── test_updater.py
├── test_integration.py
├── precision/
├── core/
└── components/
```

## Main Coverage Areas

- `test_job_config.py`
  - Validates the `JobConfig` / `TrainingJobConfig` / `InferenceJobConfig` / `SearchJobConfig` models
  - Confirms lowercase section names are accepted and legacy uppercase sections are rejected
  - Covers required sections, aliases, and defaults

- `test_load_job.py`
  - Covers `load_job()` returning the appropriate job config subtype
  - Covers profile merging and run-type detection
  - Verifies validation errors for malformed or incomplete job payloads

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

### Lowercase job configuration

The test suite assumes the current lowercase `JobConfig` contract:

```toml
[experiment]
name = "baseline"
run_name = "trial-01"
register_model = true

[tracking]
uri = "file:///tmp/mlruns"

[tracking.registry]
name = "FFNN"
aliases = ["candidate"]
tags = { team = "platform" }
max_retries = 3
```

Tracking is modeled through lowercase `experiment` and `tracking` sections. The
old `[MLFLOW]` section and nested legacy shapes are intentionally invalid and
covered by tests.

Infrastructure is env-driven:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_ARTIFACT_URI`

These TOML shapes are intentionally invalid and covered by tests:

- `[MLFLOW]`
- `[MLFLOW.server]`
- `[MLFLOW.client]`
- `tracking_uri = "..."`
- `artifacts_destination = "..."`

### Protocol-oriented loading

The low-level config I/O surface is based on the protocol contracts in
`src/dlkit/infrastructure/config/core/sources.py` and related loaders:

- `ConfigParser`
- `SectionExtractor`
- `ConfigValidator[T]`
- `PartialConfigReader`

The tests exercise the concrete behavior through:

- `load_job()`
- run-type detection and profile merging
- path preprocessing and lowercase section normalization
- patch/update helpers

## Running the Suite

```bash
uv run pytest tests/infrastructure/config -v
```

Focused runs:

```bash
uv run pytest tests/infrastructure/config/test_job_config.py -v
uv run pytest tests/infrastructure/config/test_load_job.py -v
uv run pytest tests/infrastructure/config/precision -v
```
