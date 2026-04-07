# DLKit Configuration System

DLKit uses typed Pydantic settings plus runtime-owned module defaults.

## Key Points
- `tools.io` owns TOML loading and section discovery.
- `tools.config` owns typed settings, validation, and patching.
- Workflow settings are split across `workflow_settings_base.py`, `training_workflow_settings.py`, and `inference_workflow_settings.py`.
- `workflow_settings.py` remains the re-export shim.
- The public config surface uses the canonical workflow classes directly; legacy alias exports were removed.
- Secure URI/path config types live under `tools.config.security.uri_types`.
- `DATASET.family` can explicitly select the runtime dataset family.
- `tools.precision` owns the precision service — see [`../precision/README.md`](../precision/README.md).

## Recommended Entry Points
```python
from dlkit.infrastructure.config import GeneralSettings, load_sections, load_settings

settings = load_settings("config.toml")
partial = load_sections("config.toml", ["MODEL", "DATASET"])
full = GeneralSettings.from_toml_file("config.toml")
```
