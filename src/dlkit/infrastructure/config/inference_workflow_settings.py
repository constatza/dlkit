"""Inference workflow settings."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from .workflow_settings_base import BaseWorkflowSettings


class InferenceWorkflowSettings(BaseWorkflowSettings):
    """Settings class specialized for inference workflows."""

    model_config = SettingsConfigDict(env_prefix="DLKIT_", env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type,
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (init_settings, env_settings)

    _workflow_type: ClassVar[str] = "inference"

    @model_validator(mode="after")
    def validate_inference_checkpoint(self):
        if self.SESSION and self.SESSION.inference:
            if not (self.MODEL and self.MODEL.checkpoint):
                raise ValueError(
                    "Checkpoint path must be provided when running in inference mode. "
                    "Add 'checkpoint = \"/path/to/model.ckpt\"' under [MODEL] section."
                )
        return self
