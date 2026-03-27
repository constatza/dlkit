"""Override manager service for applying runtime parameter overrides to settings objects."""

from __future__ import annotations

from typing import Any

from dlkit.tools.config.core.base_settings import BasicSettings
from dlkit.tools.io import locations


def _build_patch(settings: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Build a flat patch dict from runtime overrides.

    Pure function — no side effects, no ``self``.  The returned dict is passed
    to ``settings.patch()`` which validates and applies every entry.

    Args:
        settings: Current settings object (read-only — used only for presence checks).
        overrides: Raw keyword overrides supplied by the caller.

    Returns:
        Flat patch dict suitable for ``BasicSettings.patch()``.
    """
    patch: dict[str, Any] = {}

    # MODEL
    if (cp := overrides.get("checkpoint_path")) and settings.MODEL:
        patch["MODEL.checkpoint"] = cp

    # TRAINING
    if settings.TRAINING:
        if (epochs := overrides.get("epochs")) is not None:
            patch["TRAINING.epochs"] = epochs
            patch["TRAINING.trainer.max_epochs"] = epochs
        if (lr := overrides.get("learning_rate")) is not None:
            patch["TRAINING.optimizer.lr"] = float(lr)
        if loss := overrides.get("loss_function"):
            patch["TRAINING.loss_function"] = {
                "name": loss,
                "module_path": overrides.get("loss_module", "dlkit.core.training.functional"),
            }

    # DATAMODULE
    if (bs := overrides.get("batch_size")) is not None and settings.DATAMODULE:
        patch["DATAMODULE.dataloader.batch_size"] = bs

    # MLFLOW — patch() constructs MLflowSettings from a dict even when MLFLOW is None.
    # The caller is responsible for enabling MLflow (e.g. via settings or CLI adapter);
    # no bool toggle lives here.
    mlflow_fields = {
        k: v
        for k, v in overrides.items()
        if k in ("experiment_name", "run_name", "register_model", "tags") and v is not None
    }
    if mlflow_fields:
        patch["MLFLOW"] = mlflow_fields

    # OPTUNA
    optuna_fields = {
        k: v
        for k, v in {
            "enabled": overrides.get("enable_optuna"),
            "n_trials": overrides.get("trials"),
            "study_name": overrides.get("study_name"),
        }.items()
        if v is not None
    }

    match (bool(optuna_fields), bool(settings.OPTUNA)):
        case (False, _):
            pass
        case (True, True):
            patch["OPTUNA"] = optuna_fields
        case (True, False) if optuna_fields.get("enabled"):
            patch["OPTUNA"] = {
                "enabled": True,
                "n_trials": optuna_fields.get("n_trials", 3),
                "study_name": optuna_fields.get("study_name", "default_study"),
                "storage": locations.optuna_storage_uri(),
            }

    return patch


class BasicOverrideManager[T: BasicSettings]:
    """Manager for applying basic runtime overrides to settings objects.

    Uses ``BasicSettings.patch()`` for all mutations — no manual ``model_copy()``
    chains.  Path context side effects are kept strictly separate from settings
    mutation.
    """

    def apply_overrides(
        self,
        base_settings: T,
        **overrides: Any,
    ) -> T:
        """Apply runtime overrides to base settings.

        Args:
            base_settings: Base settings object to override.
            **overrides: Runtime parameter overrides.

        Returns:
            New settings instance with overrides applied.

        Example:
            >>> manager = BasicOverrideManager()
            >>> new_settings = manager.apply_overrides(
            ...     settings,
            ...     checkpoint_path=Path("./model.ckpt"),
            ...     epochs=100,
            ...     experiment_name="my-exp",
            ... )
        """
        self._apply_path_context_side_effects(overrides)
        patch = _build_patch(base_settings, overrides)
        return base_settings.patch(patch) if patch else base_settings

    def _apply_path_context_side_effects(self, overrides: dict[str, Any]) -> None:
        """Set thread-local root_dir context.

        Pure side effect — does not mutate settings.  Only ``root_dir`` is used;
        it is the single output root from which all standard locations derive.

        Args:
            overrides: Raw override dict supplied by the caller.
        """
        root_dir = overrides.get("root_dir")
        if not root_dir:
            return

        from dlkit.tools.io.path_context import PathOverrideContext, set_path_context
        from dlkit.tools.io.paths import normalize_user_path

        set_path_context(
            PathOverrideContext(root_dir=normalize_user_path(root_dir, require_absolute=True))
        )

    def validate_overrides(
        self,
        settings: Any,
        **overrides: Any,
    ) -> list[str]:
        """Validate runtime overrides that Pydantic cannot check at the model level.

        Only filesystem checks belong here — numeric bounds, type coercions, and
        section-presence checks are all handled by ``settings.patch()`` via Pydantic.

        Args:
            settings: Base settings to validate against.
            **overrides: Override parameters to validate.

        Returns:
            List of validation error messages (empty if all valid).
        """
        from pathlib import Path

        errors: list[str] = []

        if "checkpoint_path" in overrides and overrides["checkpoint_path"] is not None:
            checkpoint_path = overrides["checkpoint_path"]
            if not isinstance(checkpoint_path, Path):
                try:
                    checkpoint_path = Path(checkpoint_path)
                except TypeError, ValueError:
                    errors.append("checkpoint_path must be a valid path")
                    return errors

            if not checkpoint_path.exists():
                errors.append(f"Checkpoint file does not exist: {checkpoint_path}")

        return errors


# Global override manager instance
basic_override_manager = BasicOverrideManager()
