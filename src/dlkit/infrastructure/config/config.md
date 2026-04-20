# Configuration Module

`dlkit.infrastructure.config` owns typed settings, validation, patching, precision
services, and component-construction support.

## Responsibilities
- immutable Pydantic settings models
- workflow settings and workflow-specific config views
- validation and patch application
- component settings and factory support
- precision strategies and services
- security-oriented config types

## Current Structure
- `core/`: base settings, patching, factories, context
- `model_components.py`: model/loss/metric/wrapper settings
- `workflow_settings_base.py`, `training_workflow_settings.py`, `inference_workflow_settings.py`: workflow-specific settings
- `workflow_settings.py`: re-export shim for workflow settings
- `dataset_settings.py`: dataset config, including explicit `family`
- `security/uri_types.py`: secure URI/path config types
- `precision/`: precision strategy and runtime precision services
- `optimization_trigger.py`: `EpochTriggerSettings`, `PlateauTriggerSettings`, `TriggerSpec` (discriminated union)
- `optimization_selector.py`: seven focused selector classes + `ParameterSelectorSettings` (discriminated union)
- `optimization_stage.py`: `OptimizationStageSettings`, `ConcurrentOptimizationSettings`, `StageSpec` (discriminated union)
- `policy.py`: `OptimizerPolicySettings` — top-level staged optimizer config

## Optimization Discriminated Unions

Trigger, selector, and stage fields use Pydantic discriminated unions.
The discriminator field must be present when loading from TOML or any dict source.

| Type alias | Discriminator | Variants |
|---|---|---|
| `TriggerSpec` | `kind` | `"epoch"`, `"plateau"` |
| `ParameterSelectorSettings` | `kind` | `"role"`, `"module_path"`, `"muon_eligible"`, `"non_muon"`, `"intersection"`, `"union"`, `"difference"` |
| `StageSpec` | `kind` | `"stage"`, `"concurrent"` |

Python construction omits the discriminator — all fields have defaults.

## Ownership Boundary
- `tools.io` reads TOML files and resolves sections.
- `tools.config` validates those payloads into settings models and applies runtime overrides.

## Notes
- `DATASET.family` is the explicit dataset-family override. Runtime heuristics only apply when it is unset.
- Component `module_path` values remain optional; runtime builders apply default module namespaces.
