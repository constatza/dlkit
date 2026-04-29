# Architecture Docs

This directory contains the tracked architecture graphs for the current
`dlkit` package layout.

## Current Graph Set

- `diagrams/overview.dot`: top-level `dlkit.*` package overview
- `diagrams/common.dot`: `dlkit.common`
- `diagrams/infrastructure.dot`: `dlkit.infrastructure`
- `diagrams/domain.dot`: `dlkit.domain`
- `diagrams/engine.dot`: `dlkit.engine`
- `diagrams/interfaces.dot`: `dlkit.interfaces`

## Package DAG

The maintained architectural dependency direction is:

```text
interfaces -> engine, domain, infrastructure, common
engine -> domain, infrastructure, common
domain -> common
infrastructure -> common, infrastructure.precision
common -> (none)
infrastructure.precision -> (none)
```

These docs describe the current code layout:

- `common`: shared result types, errors, state, hooks, and protocols
- `infrastructure`: config, IO, registry, utilities, and precision support
- `domain`: model-local logic, transforms, metrics, losses, and NN families
- `engine`: runtime orchestration for data, training, inference, tracking, and workflows
- `interfaces`: public API, CLI, and inference-facing adapters

## Source Of Truth

`tach.toml` is the source of truth for package-level dependency rules.

The tracked DOT files are generated from actual imports rather than from an
aspirational architecture sketch. If an import edge exists in the codebase, it
should appear in the generated graph.

## Regeneration

Generate the raw Tach map:

```bash
uv run tach map -o /tmp/dlkit-map.json
```

Render the curated graph set:

```bash
uv run python scripts/render_tach_dependency_graph.py /tmp/dlkit-map.json docs/architecture/diagrams/overview.dot dlkit
uv run python scripts/render_tach_dependency_graph.py /tmp/dlkit-map.json docs/architecture/diagrams/common.dot dlkit.common
uv run python scripts/render_tach_dependency_graph.py /tmp/dlkit-map.json docs/architecture/diagrams/infrastructure.dot dlkit.infrastructure dlkit.common,dlkit.infrastructure.precision
uv run python scripts/render_tach_dependency_graph.py /tmp/dlkit-map.json docs/architecture/diagrams/domain.dot dlkit.domain dlkit.common
uv run python scripts/render_tach_dependency_graph.py /tmp/dlkit-map.json docs/architecture/diagrams/engine.dot dlkit.engine dlkit.common,dlkit.infrastructure,dlkit.domain
uv run python scripts/render_tach_dependency_graph.py /tmp/dlkit-map.json docs/architecture/diagrams/interfaces.dot dlkit.interfaces dlkit.common,dlkit.infrastructure,dlkit.domain,dlkit.engine
```

Run the architecture checks:

```bash
uv run tach check
```
