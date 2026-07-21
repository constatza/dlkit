# Seeding

`dlkit.infrastructure.seeding` provides centralized global-RNG seeding. It has
**no dlkit dependencies** — only `lightning.pytorch` (imported lazily).

## Modules

| Module | Responsibility |
|--------|---------------|
| `service.py` | `apply_global_seed(seed, *, workers=True)` — the single call site for `lightning.pytorch.seed_everything` |

## Usage

`apply_global_seed` is normally not called directly. Use
`infrastructure.config.run_settings.apply_run_context(run: RunSettings)`
instead, which resolves the seed via `RunSettings.resolve_seed()`, calls
`apply_global_seed`, and applies the run's precision override for the
duration of a `with` block:

```python
from dlkit.infrastructure.config.run_settings import apply_run_context

with apply_run_context(settings.run) as seed:
    ...  # global RNG seeded, precision override active
```

`apply_run_context` is a standalone function, not a `RunSettings` method:
`RunSettings` stays pure data plus pure resolution methods
(`resolve_seed`, `get_precision_strategy`); the side-effecting global-state
mutation lives in this explicit action function instead.

Not thread/async-safe — mutates process-global RNG state. Safe for
today's single-threaded CLI/script entrypoints.
