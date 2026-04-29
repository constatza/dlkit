"""Canonical public type alias for workflow config union.

This module provides the type alias for all supported workflow config types.
Use this when type-hinting functions that accept any workflow config.

Example:
    ```python
    from dlkit.infrastructure.config.workflow_types import WorkflowConfig


    def my_function(config: WorkflowConfig) -> None:
        if isinstance(config, TrainingWorkflowConfig):
            # handle training
            pass
    ```
"""

from __future__ import annotations

from dlkit.infrastructure.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

type WorkflowConfig = TrainingWorkflowConfig | OptimizationWorkflowConfig | InferenceWorkflowConfig
