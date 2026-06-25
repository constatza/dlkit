"""Base workflow settings shared by training and inference flows."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

from pydantic import Field, field_validator, model_validator

from dlkit.infrastructure.precision.strategy import PrecisionStrategy

from .core.base_settings import BasicSettings
from .core.patching import patch_model
from .core.sources import DLKitTomlSource, _read_env_patches
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .extras_settings import ExtrasSettings
from .model_components import ModelComponentSettings
from .training_settings import TrainingSettings


# ponytail: minimal stub; engine files that use BaseWorkflowSettings still depend on this.
# Removed by Task 3 (engine wiring) when BaseWorkflowSettings is fully retired.
class SessionSettings(BasicSettings):
    """Backward-compatible session settings stub — session_settings.py removed in config redesign.

    Retained with legacy fields (seed, precision, name, root_dir) to avoid breaking engine
    tests that still instantiate this class directly. Will be removed in Task 5.
    """

    workflow: str = "train"
    seed: int = 42
    name: str = "dlkit-session"
    root_dir: Path | None = None

    # Precision type: PrecisionStrategy is imported at TYPE_CHECKING time only;
    # at runtime the field_validator below coerces string/int values.
    precision: PrecisionStrategy | None = None

    @field_validator("precision", mode="before")
    @classmethod
    def _coerce_precision(cls, v: object) -> PrecisionStrategy | None:
        """Coerce string/int precision values to PrecisionStrategy at validation time.

        Args:
            v: Raw precision value from init or TOML.

        Returns:
            PrecisionStrategy enum member, or None if not provided.
        """
        if v is None:
            return None
        from dlkit.infrastructure.precision.strategy import PrecisionStrategy

        if isinstance(v, PrecisionStrategy):
            return v
        try:
            return PrecisionStrategy(str(v).lower())
        except ValueError:
            raise ValueError(
                f"Invalid precision value '{v}'. Valid values: "
                + ", ".join(s.value for s in PrecisionStrategy)
            ) from None

    @property
    def is_inference_mode(self) -> bool:
        """Return True when the workflow is inference mode."""
        return self.workflow == "inference"

    @property
    def is_training_mode(self) -> bool:
        """Return True when the workflow is training mode."""
        return self.workflow in ("train", "training")

    @property
    def is_optimization_mode(self) -> bool:
        """Return True when the workflow is optimization mode."""
        return self.workflow in ("optimize", "optimization", "search")

    def get_precision_strategy(self) -> PrecisionStrategy:
        """Return the configured precision strategy (satisfies PrecisionProvider protocol).

        Coerces string/int values to PrecisionStrategy if needed.

        Returns:
            PrecisionStrategy resolved from the precision field, or the default.

        Raises:
            NotImplementedError: If precision is None (signals the service to use its default).
        """
        if self.precision is None:
            raise NotImplementedError("No precision configured — use service default")
        from dlkit.infrastructure.precision.strategy import PrecisionStrategy

        if isinstance(self.precision, PrecisionStrategy):
            return self.precision
        # Coerce string or int to PrecisionStrategy
        return PrecisionStrategy(str(self.precision))


class BaseWorkflowSettings(BasicSettings):
    """Base settings class implementing common functionality for all workflows."""

    SESSION: SessionSettings | None = Field(
        default=None, description="Session mode control and execution settings"
    )
    MODEL: ModelComponentSettings | None = Field(default=None, description="Model configuration")
    DATAMODULE: DataModuleSettings | None = Field(
        default=None,
        description="Data loading and processing configuration",
    )
    DATASET: DatasetSettings | None = Field(
        default=None, description="Dataset-specific configuration"
    )
    EXTRAS: ExtrasSettings | None = Field(
        default=None,
        description="Optional free-form helper options for user scripts; ignored by core",
    )
    # Backward-compatible sections — accepted but ignored when using the new architecture.
    # These fields let tests still construct settings with UPPERCASE keys.
    TRAINING: TrainingSettings | None = Field(
        default=None,
        description="Training configuration (backward compat — use training_settings instead)",
    )
    MLFLOW: object | None = Field(
        default=None,
        description="MLflow configuration (backward compat — use tracking_settings instead)",
    )
    OPTUNA: object | None = Field(
        default=None,
        description="Optuna configuration (backward compat — use search_settings instead)",
    )

    _workflow_type: ClassVar[str] = "base"

    @model_validator(mode="after")
    def validate_nested_paths(self) -> BaseWorkflowSettings:
        """Validate nested DATASET feature/target paths eagerly."""
        if self.DATASET is not None:
            from dlkit.infrastructure.config.validators import _validate_entry_paths

            _validate_entry_paths(self.DATASET.features, "Feature")
            _validate_entry_paths(self.DATASET.targets, "Target")
        return self

    @classmethod
    def from_toml(
        cls,
        config_path: Path | str,
        *,
        sections: list[str] | None = None,
        **overrides: str | int | float | bool | None,
    ) -> Self:
        """Load workflow settings from a TOML file."""
        source = DLKitTomlSource(Path(config_path), sections=sections)
        settings: Self = cls.model_validate(source())
        if env := _read_env_patches("DLKIT_", "__", uppercase_section=True):
            settings = patch_model(settings, env)
        if overrides:
            settings = patch_model(settings, overrides)
        return settings

    @property
    def is_training(self) -> bool:
        return not (self.SESSION and self.SESSION.is_inference_mode)

    @property
    def is_inference(self) -> bool:
        return bool(self.SESSION and self.SESSION.is_inference_mode)

    @property
    def is_testing(self) -> bool:
        return False

    @property
    def has_data_config(self) -> bool:
        return self.DATAMODULE is not None and self.DATASET is not None

    @property
    def has_dataset_config(self) -> bool:
        """Alias for has_data_config — backward compatible predicate for batch inference."""
        return self.has_data_config

    def get_datamodule_config(self) -> DataModuleSettings:
        if not self.DATAMODULE:
            raise ValueError("No datamodule configuration available")
        return self.DATAMODULE

    def get_dataset_config(self) -> DatasetSettings:
        if not self.DATASET:
            raise ValueError("No dataset configuration available")
        return self.DATASET
