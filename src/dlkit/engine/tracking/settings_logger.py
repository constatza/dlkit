"""Settings and model-parameter logging service.

Single Responsibility: Serialize workflow configuration and model hyperparameters
into an active run context. Extracted from IExperimentTracker so tracker implementations
do not carry application-level serialization concerns.
"""

from __future__ import annotations

from torch import nn

from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.infrastructure.utils.logging_config import get_logger

from .interfaces import IRunContext

type _WorkflowSettings = GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig

logger = get_logger(__name__)

_COMPONENT_FIELDS = frozenset({"name", "module_path", "checkpoint", "shape"})


class SettingsLogger:
    """Serializes workflow settings and model hyperparameters into a run context.

    Args:
        None — stateless service; all context passed per-call.
    """

    def log_settings(self, settings: _WorkflowSettings, run_context: IRunContext) -> None:
        """Save complete configuration as a TOML artifact on the active run.

        Args:
            settings: Workflow settings to serialize.
            run_context: Active run context to log the artifact to.

        Raises:
            RuntimeError: If serialization or artifact logging fails.
        """
        try:
            from dlkit.infrastructure.io import serialize_config_to_string

            toml_content = serialize_config_to_string(
                settings,
                exclude_unset=True,
                exclude_value_entries=True,
            )
            run_context.log_artifact_content(toml_content, "GeneralSettings.toml")
        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_model_parameters(
        self, model: nn.Module, run_context: IRunContext, settings: _WorkflowSettings
    ) -> None:
        """Log model hyperparameters extracted from settings.MODEL.

        Excludes structural fields (name, module_path, checkpoint, shape).

        Args:
            model: Model instance (currently unused; reserved for future introspection).
            run_context: Active run context to log parameters to.
            settings: Settings object containing MODEL configuration.

        Raises:
            RuntimeError: If parameter extraction or logging fails.
        """
        try:
            if settings.MODEL is None:
                return
            params = settings.MODEL.model_dump(exclude_none=True)
            hparams = {k: v for k, v in params.items() if k not in _COMPONENT_FIELDS}
            if hparams:
                run_context.log_params(hparams)
        except Exception as e:
            raise RuntimeError("Couldn't log model parameters") from e
