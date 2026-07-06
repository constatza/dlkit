"""Settings and model-parameter logging service.

Single Responsibility: Serialize workflow configuration and model hyperparameters
into an active run context. Extracted from IExperimentTracker so tracker implementations
do not carry application-level serialization concerns.
"""

from __future__ import annotations

from lightning.pytorch import LightningModule

from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from .interfaces import IRunContext

type _WorkflowSettings = JobConfig

logger = get_logger(__name__)


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
            run_context.log_artifact_content(toml_content, "job_config.toml")
        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_model_parameters(
        self, model: LightningModule, run_context: IRunContext, settings: _WorkflowSettings
    ) -> None:
        """Log the model's effective (post-default-resolution) hyperparameters.

        Reads ``model.hparams``, populated at construction time via
        ``save_hyperparameters()`` in ``CoreLightningWrapper`` (see
        ``_build_model_from_settings`` in ``engine.adapters.lightning.base``),
        so hyperparameters left unset in settings — and resolved to a
        network-internal default — are logged accurately rather than dropped.

        Args:
            model: Constructed Lightning wrapper exposing ``.hparams``.
            run_context: Active run context to log parameters to.
            settings: JobConfig object containing model configuration.

        Raises:
            RuntimeError: If parameter extraction or logging fails.
        """
        try:
            if settings.model is None:
                return
            hparams = dict(model.hparams)
            if hparams:
                run_context.log_params(hparams)
        except Exception as e:
            raise RuntimeError("Couldn't log model parameters") from e
