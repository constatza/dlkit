"""Shape inference utilities for transform classes.

Individual transforms implement ``infer_output_shape(in_shape)`` as an instance
method.  This module is intentionally minimal; the registry-based approach has
been removed in favour of the instance-method pattern.
"""
