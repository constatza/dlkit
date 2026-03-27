# DLKit Domain — Shared Kernel

## Overview

`dlkit.domain` is the shared kernel layer introduced in FR-1. It sits between `tools` and `core`,
providing immutable result and state types that are consumed by both the runtime and interfaces layers
without creating upward dependencies.

**Layer position**:
```
interfaces  ← imports from dlkit.domain
runtime     ← imports from dlkit.domain
domain      ← imports from tools only (GeneralSettings, tensordict_utils)
core        ← does NOT import from domain
tools       ← foundation, imported by domain
```

## Module Structure

| File | Contents |
|------|----------|
| `results.py` | `TrainingResult`, `InferenceResult`, `OptimizationResult` — frozen dataclasses |
| `state.py` | `ModelState` — frozen dataclass holding loaded model + inference state |
| `__init__.py` | Re-exports all four types |

## Public API

| Name | Type | Purpose |
|------|------|---------|
| `TrainingResult` | Frozen dataclass | Result of a training workflow execution |
| `InferenceResult` | Frozen dataclass | Result of an inference workflow execution |
| `OptimizationResult` | Frozen dataclass | Result of a hyperparameter optimization run |
| `ModelState` | Frozen dataclass | Complete model and component state after training |

### `TrainingResult`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingResult:
    model_state: ModelState | None
    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    duration_seconds: float
    predictions: list[Any] | None = None
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None

    @property
    def checkpoint_path(self) -> Path | None: ...
    @property
    def stacked(self) -> TensorDict | None: ...
    def to_numpy(self, *keys) -> dict[str, Any] | None: ...
```

### `InferenceResult`

```python
@dataclass(frozen=True)
class InferenceResult:
    model_state: ModelState
    predictions: Any
    metrics: dict[str, Any] | None
    duration_seconds: float
```

### `OptimizationResult`

```python
@dataclass(frozen=True)
class OptimizationResult:
    best_trial: Any
    training_result: TrainingResult
    study_summary: dict[str, Any]
    duration_seconds: float
```

### `ModelState`

```python
@dataclass(frozen=True)
class ModelState:
    model: LightningModule
    datamodule: LightningDataModule
    trainer: Any | None
    settings: GeneralSettings
    feature_names: tuple[str, ...]
    predict_target_key: str
```

## Backward Compatibility

`interfaces/api/domain/models.py` is a thin re-export shim:
```python
from dlkit.domain import InferenceResult, ModelState, OptimizationResult, TrainingResult
```
Existing code importing from `dlkit.interfaces.api.domain.models` continues to work unchanged.

## Design Constraints

- MUST NOT import from `core`, `runtime`, or `interfaces`.
- MAY import from `tools/config` (for `GeneralSettings`, workflow configs) and `tools/utils`.
- All types are frozen dataclasses — value objects, not entities.
