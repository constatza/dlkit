"""DEPRECATED: Import from dlkit.core.shape_specs package instead.

This module has been completely rewritten. The new shape handling system
is available in the dlkit.core.shape_specs package.

Breaking Changes:
- IShapeConsumer interface removed (use registry-based detection)
- ShapeSpec god object decomposed into focused components
- All shape handling now uses value objects and strategy pattern
"""

# Re-export from new package
from .shape_specs import *
