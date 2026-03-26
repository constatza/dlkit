# DLKit — Dependency Coupling Assessment

> Companion to `deps.dot`. Render the diagram with:
> ```
> dot -Tsvg deps.dot -o deps.svg
> ```

---

## Layer model (intended)

```
interfaces  (api · cli · inference)   ← top, consumer
    ↓
runtime     (factories · orchestrator · strategies · optimization)
    ↓
core        (models · shape_specs · datasets · training · datatypes)
    ↓
tools       (config · io · registry · utils)   ← foundation
```

A dependency is **normal** when it points downward (or stays within a layer).
A dependency is a **violation** when it points upward (lower layer imports from higher layer).

---

## Completed quick wins

| # | Change | Violations fixed | Status |
|---|---|---|---|
| QW-1 | Moved exception hierarchy → `tools/utils/errors.py` | 3 (tools→api/domain, core→api/domain) | ✓ Done |
| QW-2 | Moved `PathOverrideContext` → `tools/io/path_context.py` | 4 (tools→api/overrides, runtime→api/overrides) | ✓ Done |
| QW-3 | Moved `PrecisionContext`/`PrecisionService` → `tools/config/precision/` | 8 (tools/core/runtime→api/services + factories) | ✓ Done |
| QW-4 | Moved `TrackingHooks` → `runtime/workflows/tracking_hooks.py` | 2 (runtime→api/tracking_hooks) | ✓ Done |

**17 of 25 violations eliminated.** Remaining 8 are Theme B (runtime → api/domain for result value objects) — addressed by FR-1.

---

## Current violation themes

### Theme A — PrecisionService stranded too high *(highest impact)*

`get_precision_service()` and `precision_override()` live in `interfaces/api/services/`
but are needed by every layer below:

| Violating file | Import |
|---|---|
| `tools/io/arrays.py` | `get_precision_service` |
| `tools/config/data_entries.py` | `get_precision_service` |
| `tools/config/trainer_settings.py` | `get_precision_service` |
| `core/datasets/flexible.py` | `get_precision_service` |
| `core/datasets/graph.py` | `get_precision_service` |
| `core/models/wrappers/base.py` | `get_precision_service` (lazy) |
| `runtime/strategies/vanilla_executor.py` | `get_precision_service` |
| `runtime/factories/build_factory.py` | `precision_override` (lazy) |

**Root cause:** precision resolution is a foundational I/O concern, not an interface concern.

---

### Theme B — Domain value objects have no home below interfaces *(~10 violations)*

`TrainingResult`, `OptimizationResult`, `ModelState`, `WorkflowError`, `TrackingHooks`,
`DLKitError` live in `interfaces/api/domain/` but carry no interface-layer logic.

| Violating module | Types imported |
|---|---|
| `runtime/workflows/orchestrator` | `TrainingResult`, `OptimizationResult`, `TrackingHooks` |
| `runtime/strategies/vanilla_executor` | `TrainingResult`, `ModelState`, `WorkflowError` |
| `runtime/strategies/tracking_decorator` | `TrainingResult` |
| `runtime/strategies/optuna_optimizer` | `WorkflowError` |
| `runtime/optimization/*` | `TrainingResult`, `WorkflowError` |
| `core/training/transforms/errors.py` | `DLKitError` (used as base class) |
| `tools/utils/error_handling.py` | `ConfigurationError`, `WorkflowError` |
| `tools/io/url_resolver.py` | `ConfigurationError` |

---

### Theme C — PathOverrideContext belongs in tools *(4 violations)*

`interfaces/api/overrides/path_context.py` is used by:

- `tools/config/core/sources.py` — settings source resolution
- `tools/io/locations.py` — path normalisation
- `runtime/strategies/tracking/mlflow_tracker.py`
- `runtime/factories/build_factory.py`

Path context is infrastructure; it has no business sitting above tools.

---

### Theme D — Factory infrastructure types leak into core *(architectural concern)*

`BuildContext` and `FactoryProvider` from `tools/config/core/` appear as constructor
parameters in `core/models/wrappers/base.py` and `core/training/transforms/chain.py`.
`IntHyperparameter` from `core/datatypes/base.py` is imported by
`tools/config/components/model_components.py` (small bidirectional dependency).

These do not invert the layer order but express overly tight coupling between adjacent layers.

---

## Quick wins

Each item touches 1–3 files and can be done independently.

### QW-1 — Move base exception hierarchy to `tools/utils/errors.py`

`DLKitError`, `WorkflowError`, `ConfigurationError` are pure stdlib dataclasses.
Move them out of `interfaces/api/domain/errors.py` into `tools/utils/errors.py`.
Have `interfaces/api/domain/errors.py` re-export for backward compat.

**Fixes:** Theme B (base exceptions), Theme D (error_handling.py, url_resolver.py)
**Files:** `tools/utils/errors.py` (new), `interfaces/api/domain/errors.py` (re-export),
`core/training/transforms/errors.py`, `tools/utils/error_handling.py`,
`tools/io/url_resolver.py` (import-site updates)

---

### QW-2 — Move `PathOverrideContext` down to `tools/config/environment.py`

`interfaces/api/overrides/path_context.py` only imports stdlib + tools.
Move the implementation to `tools/config/environment.py` (or `tools/io/paths.py`).
Have `api/overrides/path_context.py` thin-re-export.

**Fixes:** Theme C entirely (4 violations)
**Files:** `tools/config/environment.py` (extend), `interfaces/api/overrides/path_context.py`
(re-export), + 4 import-site updates

---

### QW-3 — Move `PrecisionContext` / `precision_override` to `tools/config/precision/`

`PrecisionContext`, `PrecisionProvider` protocol, and `precision_override()` are pure
threading + stdlib constructs. Move to `tools/config/precision/context.py`.
Move service logic to `tools/config/precision/service.py`.
Have `interfaces/api/domain/precision.py` and `interfaces/api/services/precision_service.py`
re-export.

**Fixes:** Theme A entirely (7–8 violations) — highest-leverage single change
**Files:** `tools/config/precision/context.py` (new), `tools/config/precision/service.py`
(new/moved), re-export updates in interfaces, ~8 import-site updates

---

### QW-4 — Move `TrackingHooks` to `runtime/workflows/tracking_hooks.py`

`interfaces/api/tracking_hooks.py` is a frozen dataclass with no interface logic.
Move to `runtime/workflows/tracking_hooks.py`; re-export from `interfaces/api/`.

**Fixes:** 2 `runtime → api/tracking_hooks` violations
**Files:** `runtime/workflows/tracking_hooks.py` (new), `interfaces/api/tracking_hooks.py`
(re-export), 3 import-site updates

---

## Future refactors

These are larger changes that require separate planning.

### FR-1 — Introduce `dlkit.domain` shared kernel

Create `src/dlkit/domain/` as a new package layer between core and runtime.
Migrate `TrainingResult`, `OptimizationResult`, `ModelState` there.
Runtime builds them; interfaces re-exports them.

**Fixes:** Remaining Theme B violations (~10 sites)
**Scope:** New package, ~15 import sites, full public API re-export audit.
**Prerequisite:** None, but do QW-1 first to avoid double-migration of error types.

---

### FR-2 — Inject built objects into core wrappers; remove `BuildContext` from core

`core/models/wrappers/base.py` and `core/training/transforms/chain.py` carry
`BuildContext` / `FactoryProvider` because construction logic was never fully pushed to
`runtime/factories`. The fix is to have `runtime/factories/build_factory.py` pass
fully-constructed objects, so wrapper/chain classes are pure domain objects.

**Fixes:** Theme D (factory infrastructure leak into core)
**Scope:** `core/models/wrappers/` (3–4 files), `core/training/transforms/chain.py`,
`runtime/factories/build_factory.py`.

---

### FR-3 — Resolve `tools/config ↔ core/datatypes` micro-cycle

`IntHyperparameter` / `FloatHyperparameter` are config parameter type aliases, not domain
types. Move them into `tools/config/core/types.py`. Have `core/datatypes/base.py`
re-export for backward compat.

**Fixes:** Theme D (bidirectional tools/config ↔ core/datatypes coupling)
**Scope:** 2–3 files.

---

### FR-4 — Evaluate collapsing `api/overrides` after QW-2

After QW-2, `api/overrides` becomes a thin pass-through. Consider merging residual
override manager + normaliser into `api/services` to remove the package boundary.

---

## Violation count summary

| Theme | Original | Remaining | Fix |
|---|---|---|---|
| A — PrecisionService | 8 | 0 | QW-3 ✓ |
| B — Domain value objects | 10 | 8 | QW-1 partial ✓, FR-1 pending |
| C — PathOverrideContext | 4 | 0 | QW-2 ✓ |
| D — Factory infra + cycle | 3 (concern) | 3 (concern) | QW-4 partial ✓, FR-2+FR-3 pending |
| **Total** | **25** | **8** | |
