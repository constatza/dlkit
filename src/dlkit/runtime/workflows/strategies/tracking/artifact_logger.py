"""Artifact logging service for MLflow tracking.

Single Responsibility: Log checkpoints, models, and user-defined artifacts to MLflow.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
import re
from dataclasses import dataclass

from dlkit.core.models.wrappers.base import ProcessingLightningWrapper
from dlkit.interfaces.api.domain import TrainingResult
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker, IRunContext

logger = get_logger(__name__)

DEFAULT_MODEL_ARTIFACT_PATH = "model"
TAG_LOGGED_MODEL_URI = "mlflow_logged_model_uri"
TAG_LOGGED_MODEL_ARTIFACT_PATH = "mlflow_logged_model_artifact_path"
TAG_MODEL_CLASS = "mlflow_model_class"
TAG_MODEL_REGISTRATION_ENABLED = "mlflow_model_registration_enabled"


def _resolve_model_class_name(model: Any) -> str:
    """Return the effective class name for MLflow, unwrapping DLKit Lightning wrappers.

    Args:
        model: Any model object, possibly a ProcessingLightningWrapper.

    Returns:
        Class name of the underlying nn.Module when model is a wrapper,
        otherwise the class name of model itself.
    """
    if isinstance(model, ProcessingLightningWrapper):
        return type(model.model).__name__
    return type(model).__name__


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingModelRegistration:
    """Run-scoped model registration payload finalized after run closure."""

    model_name: str
    model_uri: str
    aliases: tuple[str, ...]
    version_tags: dict[str, str]


class ArtifactLogger:
    """Handles artifact logging to MLflow.

    Single Responsibility: Log checkpoints, models, and user-defined artifacts.
    Delegates configuration access to ConfigAccessor.

    Args:
        tracker: Experiment tracker implementation
    """

    def __init__(self, tracker: IExperimentTracker):
        """Initialize with experiment tracker.

        Args:
            tracker: Experiment tracker implementation
        """
        self._tracker = tracker

    def log_training_artifacts(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext,
    ) -> PendingModelRegistration | None:
        """Log all training artifacts (checkpoints and optional model registration).

        Args:
            components: Build components containing trainer
            settings: Global settings
            run_context: Run context for logging

        Returns:
            Pending model registration payload for post-run finalization, if any.
        """
        self.log_checkpoints(components, run_context)
        return self.maybe_register_model(settings, components, run_context)

    def log_checkpoints(
        self,
        components: BuildComponents,
        run_context: IRunContext,
    ) -> None:
        """Log model checkpoints as artifacts.

        Logs best and last checkpoints from PyTorch Lightning trainer.

        Args:
            components: Build components containing trainer
            run_context: Run context for logging
        """
        try:
            trainer = getattr(components, "trainer", None)
            if not trainer:
                logger.debug("No trainer found in components")
                return

            # Find ModelCheckpoint callback
            ckpt_cb = self._find_checkpoint_callback(trainer)
            if not ckpt_cb:
                logger.debug("No ModelCheckpoint callback found")
                return

            # Log best and last checkpoints
            best = getattr(ckpt_cb, "best_model_path", None)
            last = getattr(ckpt_cb, "last_model_path", None)

            if best:
                run_context.log_artifact(Path(best), "checkpoints")
                logger.debug(f"Logged best checkpoint: {best}")
            if last and last != best:
                run_context.log_artifact(Path(last), "checkpoints")
                logger.debug(f"Logged last checkpoint: {last}")

        except Exception as e:
            logger.warning(f"Failed to log checkpoints: {e}")

    def maybe_register_model(
        self,
        settings: GeneralSettings,
        components: BuildComponents,
        run_context: IRunContext,
    ) -> PendingModelRegistration | None:
        """Log model artifact and optionally register it in MLflow model registry."""
        try:
            accessor = ConfigAccessor(settings)
            configured_name = accessor.get_registered_model_name()
            model_name = configured_name or self._derive_registered_model_name(components.model)
            registration_enabled = accessor.should_register_model()
            aliases = self._resolve_aliases(settings, accessor)
            version_tags = self._resolve_version_tags(settings, accessor)
            model_uri = self._log_model_artifact(
                run_context=run_context,
                model=components.model,
                model_name=model_name,
                registration_enabled=registration_enabled,
            )

            if not model_uri:
                logger.warning(f"Model registration skipped for '{model_name}': no model URI")
                return

            self._set_logged_model_tags(
                run_context=run_context,
                model=components.model,
                model_uri=model_uri,
                registration_enabled=registration_enabled,
            )

            match registration_enabled:
                case False:
                    logger.info(
                        "Logged model artifact '{}' for run '{}' without registry registration",
                        model_name,
                        run_context.run_id,
                    )
                    return None
                case True:
                    return PendingModelRegistration(
                        model_name=model_name,
                        model_uri=model_uri,
                        aliases=aliases,
                        version_tags=version_tags,
                    )

        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
            return None

    def finalize_model_registration(
        self,
        pending: PendingModelRegistration | None,
        run_context: IRunContext,
    ) -> None:
        """Finalize registry alias/tag attachment after run closure."""
        if pending is None:
            return
        self._register_model_aliases_and_tags(
            run_context=run_context,
            model_name=pending.model_name,
            model_uri=pending.model_uri,
            aliases=pending.aliases,
            version_tags=pending.version_tags,
        )

    def _log_model_artifact(
        self,
        *,
        run_context: IRunContext,
        model: Any,
        model_name: str,
        registration_enabled: bool,
    ) -> str | None:
        match registration_enabled:
            case True:
                return run_context.log_model(
                    model=model,
                    artifact_path=DEFAULT_MODEL_ARTIFACT_PATH,
                    registered_model_name=model_name,
                )
            case False:
                return run_context.log_model(
                    model=model,
                    artifact_path=DEFAULT_MODEL_ARTIFACT_PATH,
                )

    def _set_logged_model_tags(
        self,
        *,
        run_context: IRunContext,
        model: Any,
        model_uri: str,
        registration_enabled: bool,
    ) -> None:
        run_context.set_tag(TAG_MODEL_CLASS, _resolve_model_class_name(model))
        run_context.set_tag(TAG_LOGGED_MODEL_URI, model_uri)
        run_context.set_tag(TAG_LOGGED_MODEL_ARTIFACT_PATH, DEFAULT_MODEL_ARTIFACT_PATH)
        run_context.set_tag(TAG_MODEL_REGISTRATION_ENABLED, str(registration_enabled).lower())

    def _register_model_aliases_and_tags(
        self,
        run_context: IRunContext,
        model_name: str,
        model_uri: str,
        *,
        aliases: tuple[str, ...],
        version_tags: dict[str, str],
    ) -> None:
        version = run_context.get_latest_model_version(
            model_name,
            run_id=run_context.run_id,
        )
        if version is None:
            version = run_context.get_latest_model_version(model_name)
        if version is None:
            logger.warning(f"Model registration incomplete for '{model_name}': no version found")
            return

        self._apply_aliases(run_context, model_name, version, aliases=aliases)
        self._apply_version_tags(run_context, model_name, version, tags=version_tags)
        run_context.set_tag("mlflow_registered_model_name", model_name)
        run_context.set_tag("mlflow_registered_model_version", str(version))
        run_context.set_tag("mlflow_registered_model_uri", model_uri)
        logger.info(
            "Registered model '{}' version {} with aliases {} and tags {}",
            model_name,
            version,
            ",".join(aliases),
            ",".join(sorted(version_tags.keys())),
        )

    def log_user_artifacts(
        self,
        settings: GeneralSettings,
        run_context: IRunContext,
        result: TrainingResult,
    ) -> None:
        """Orchestrate logging of user-defined artifacts and params from settings.EXTRAS.

        Args:
            settings: Global settings
            run_context: Run context for logging
            result: Training result (not currently used but kept for extensibility)
        """
        accessor = ConfigAccessor(settings)
        extras = accessor.get_extras()
        if not extras:
            logger.debug("No EXTRAS configuration found")
            return

        try:
            self._log_user_params(accessor, run_context)
            self._log_user_file_artifacts(accessor, run_context)
            self._log_user_toml_artifacts(accessor, run_context)
        except Exception as e:
            logger.warning(f"Failed to log user-defined artifacts/params: {e}")

    def _find_checkpoint_callback(self, trainer: Any) -> Any | None:
        """Find ModelCheckpoint callback in trainer.

        Args:
            trainer: PyTorch Lightning trainer

        Returns:
            ModelCheckpoint callback or None
        """
        # Check direct attribute first
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb:
            return ckpt_cb

        # Search in callbacks list
        if hasattr(trainer, "callbacks"):
            try:
                from pytorch_lightning.callbacks import ModelCheckpoint

                candidates = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
                return candidates[0] if candidates else None
            except ImportError as e:
                logger.warning(f"Failed to import ModelCheckpoint: {e}")
                return None

        return None

    def _derive_registered_model_name(self, model: Any) -> str:
        """Build registered model name from model class name."""
        raw_name = _resolve_model_class_name(model) or "DLKitModel"
        normalized = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)
        return normalized or "DLKitModel"

    def _apply_aliases(
        self,
        run_context: IRunContext,
        model_name: str,
        version: int,
        *,
        aliases: tuple[str, ...],
    ) -> None:
        """Attach aliases to the latest registered version."""
        for alias in aliases:
            match alias:
                case "latest":
                    # MLflow manages `latest` as a reserved built-in alias.
                    continue
                case _:
                    run_context.set_model_alias(model_name=model_name, alias=alias, version=version)

    def _apply_version_tags(
        self,
        run_context: IRunContext,
        model_name: str,
        version: int,
        *,
        tags: dict[str, str],
    ) -> None:
        for key, value in tags.items():
            run_context.set_model_version_tag(
                model_name=model_name,
                version=version,
                key=key,
                value=value,
            )

    def _resolve_aliases(
        self,
        settings: GeneralSettings,
        accessor: ConfigAccessor,
    ) -> tuple[str, ...]:
        _ = settings
        configured_aliases = accessor.get_registered_model_aliases() or ()
        return self._dedupe_aliases(configured_aliases)

    def _resolve_version_tags(
        self,
        settings: GeneralSettings,
        accessor: ConfigAccessor,
    ) -> dict[str, str]:
        _ = settings
        return accessor.get_registered_model_version_tags()

    def _dedupe_aliases(self, aliases: tuple[str, ...]) -> tuple[str, ...]:
        normalized_aliases: list[str] = []
        for alias in aliases:
            normalized = self._normalize_alias(alias)
            if normalized is None:
                continue
            if normalized in normalized_aliases:
                continue
            normalized_aliases.append(normalized)
        return tuple(normalized_aliases)

    def _normalize_alias(self, alias: str) -> str | None:
        normalized = str(alias).strip()
        return normalized or None

    def _log_user_params(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined parameters from EXTRAS.mlflow_params.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        params_dict = accessor.get_mlflow_params()
        if not params_dict:
            return

        # Filter out non-serializable values
        safe_params = {}
        for key, value in params_dict.items():
            try:
                safe_params[key] = str(value) if value is not None else ""
            except Exception as e:
                logger.warning(f"Skipping non-serializable param '{key}': {e}")

        if safe_params:
            run_context.log_params(safe_params)
            logger.debug(f"Logged {len(safe_params)} custom params from EXTRAS.mlflow_params")

    def _log_user_file_artifacts(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined file artifacts from EXTRAS.mlflow_artifacts.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        artifacts = accessor.get_mlflow_artifacts()
        if not artifacts:
            return

        for artifact_path in artifacts:
            try:
                path = Path(artifact_path)
                if path.exists() and path.is_file():
                    artifact_dir = str(path.parent) if path.parent != Path(".") else ""
                    run_context.log_artifact(path, artifact_dir=artifact_dir)
                    logger.debug(f"Logged artifact: {artifact_path}")
                else:
                    logger.warning(f"Artifact not found or not a file: {artifact_path}")
            except Exception as e:
                logger.warning(f"Failed to log artifact '{artifact_path}': {e}")

    def _log_user_toml_artifacts(
        self,
        accessor: ConfigAccessor,
        run_context: IRunContext,
    ) -> None:
        """Log user-defined dicts as TOML artifacts from EXTRAS.mlflow_artifacts_toml.

        Converts dict values to TOML files and logs them to MLflow.

        Args:
            accessor: Configuration accessor
            run_context: Run context for logging
        """
        artifacts_toml = accessor.get_mlflow_artifacts_toml()
        if not artifacts_toml:
            return

        from dlkit.tools.io.config import write_config

        for name, data_dict in artifacts_toml.items():
            try:
                if not isinstance(data_dict, dict):
                    logger.warning(f"Skipping non-dict TOML artifact: {name}")
                    continue

                # Create temporary TOML file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".toml", delete=False, prefix=f"{name}_"
                ) as temp_file:
                    temp_path = Path(temp_file.name)

                # Write dict to TOML
                write_config(data_dict, temp_path, exclude_none=True)

                # Log the TOML file
                run_context.log_artifact(temp_path, artifact_dir="config")

                # Clean up temp file
                os.unlink(temp_path)

                logger.debug(f"Logged TOML artifact: {name}.toml → config/")

            except Exception as e:
                logger.warning(f"Failed to log TOML artifact '{name}': {e}")
