"""DEPRECATED: Import from dlkit.core.shape_specs package instead.

This module has been completely rewritten. The new shape handling system
is available in the dlkit.core.shape_specs package.

Breaking Changes:
- IShapeConsumer interface removed (use registry-based detection)
- ShapeSpec god object decomposed into focused components
- All shape handling now uses value objects and strategy pattern
"""

import importlib

# Re-export from new package
_shape_specs_package = importlib.import_module("dlkit.core.shape_specs")
globals().update(
    {name: value for name, value in vars(_shape_specs_package).items() if not name.startswith("_")}
)
