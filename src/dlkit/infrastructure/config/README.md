# DLKit Configuration System

DLKit uses typed Pydantic settings plus runtime-owned module defaults.

## Key Points
- `infrastructure.io` owns TOML loading and section discovery.
- `infrastructure.config` owns typed settings, validation, and patching.
- Workflow settings are split across `workflow_settings_base.py`, `training_workflow_settings.py`, and `inference_workflow_settings.py`.
- `workflow_settings.py` remains the re-export shim.
- The public config surface uses the canonical workflow classes directly; legacy alias exports were removed.
- Secure URI/path config types live under `infrastructure.config.security.uri_types`.
- `DATASET.family` can explicitly select the runtime dataset family.
- `infrastructure.precision` owns the precision service — see [`../precision/README.md`](../precision/README.md).

## Recommended Entry Points
```python
from dlkit.infrastructure.config.factories import load_settings
from dlkit.infrastructure.config.workflow_configs import (
    TrainingWorkflowConfig,
    OptimizationWorkflowConfig,
    InferenceWorkflowConfig,
)

# Load typed workflow config — type depends on SESSION.workflow in the TOML
settings = load_settings("config.toml")  # -> TrainingWorkflowConfig | OptimizationWorkflowConfig | InferenceWorkflowConfig

# Load partial sections only
from dlkit.infrastructure.config.factories import load_sections_config
partial = load_sections_config("config.toml", ["MODEL", "DATASET"])
```
