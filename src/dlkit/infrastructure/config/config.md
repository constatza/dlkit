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

## Ownership Boundary
- `tools.io` reads TOML files and resolves sections.
- `tools.config` validates those payloads into settings models and applies runtime overrides.

## Notes
- `DATASET.family` is the explicit dataset-family override. Runtime heuristics only apply when it is unset.
- Component `module_path` values remain optional; runtime builders apply default module namespaces.
