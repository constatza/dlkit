# DLKit Architecture Summary

## Current Package Model

DLKit now follows this package DAG:

```text
interfaces -> runtime, domain, tools, shared
runtime -> domain, tools, shared
domain -> shared
tools -> shared
shared -> (none)
```

There is no `dlkit.core` package.
There is no top-level `dlkit.nn` package.

Current ownership:

- `dlkit.common`
  - errors
  - results
  - model state
  - minimal shape contracts
- `dlkit.infrastructure`
  - configuration
  - path and I/O infrastructure
  - URL/path/split datatypes
  - registries and utilities
- `dlkit.domain`
  - `domain.nn`
  - `domain.shapes`
  - `domain.transforms`
  - `domain.metrics`
  - `domain.losses`
- `dlkit.engine`
  - Lightning adapters
  - runtime datasets and splits
  - execution
  - tracking
  - predictor services
  - workflow orchestration
- `dlkit.interfaces`
  - API
  - CLI
  - inference-facing adapters

## Completed Refactor Outcomes

- Deleted `dlkit.core`.
- Moved model families to `dlkit.domain.nn`.
- Moved rich shape-spec logic to `dlkit.domain.shapes`.
- Moved training primitives to domain:
  - `dlkit.domain.transforms`
  - `dlkit.domain.metrics`
  - `dlkit.domain.losses`
- Moved runtime-owned dataflow code to runtime:
  - datamodules -> `runtime.adapters.lightning.datamodules`
  - datasets -> `runtime.data.datasets`
  - split helpers -> `runtime.data.splits`
  - graph runtime types -> `runtime.data.graph.types`
- Moved CLI-facing postprocessing to `interfaces.cli.presenters`.
- Moved Lightning callbacks to `runtime.adapters.lightning.callbacks`.
- Moved URL/path/split infrastructure to `tools.datatypes`.
- Removed package-level cycles from the curated architecture graph views.

## Enforcement

- `tach.toml` defines the package dependency constraints.
- `uv run tach check` is expected to stay green.
- `tests/architecture/test_curated_dependency_graphs.py` fails if the curated graph views contain SCCs.
- Tracked dependency graphs under `docs/architecture/diagrams/` are generated from Tach:
  - `overview.dot`
  - `shared.dot`
  - `tools.dot`
  - `domain.dot`
  - `runtime.dot`
  - `interfaces.dot`

## Notes

- This document is a current-state summary, not a migration log.
- Historical refactor steps that referenced deleted `core/*` paths are intentionally removed.
