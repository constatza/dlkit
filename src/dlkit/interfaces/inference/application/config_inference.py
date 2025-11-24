"""Config-based inference support.

This module provides utilities for building dataloaders from configuration
files, enabling batch inference on datasets defined in configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dlkit.tools.config import GeneralSettings
from dlkit.interfaces.api.domain.errors import WorkflowError
from ..domain.models import ModelState


def build_prediction_dataloader_from_config(
    config: GeneralSettings,
    model_state: ModelState,
    device: str = "auto"
) -> DataLoader:
    """Build prediction dataloader from configuration settings.

    This function extracts dataset/split/datamodule settings from the config
    and builds a dataloader suitable for batch inference.

    Args:
        config: General settings with dataset/datamodule configuration
        model_state: Loaded model state (for validation)
        device: Target device for tensors

    Returns:
        DataLoader configured from config settings

    Raises:
        WorkflowError: If config is invalid or dataloader cannot be built

    Example:
        >>> from dlkit.tools.config import load_settings
        >>> config = load_settings("config.toml")
        >>> dataloader = build_prediction_dataloader_from_config(config, model_state)
        >>> for batch in dataloader:
        ...     result = predictor.predict(batch)
    """
    try:
        # Validate config has required sections
        if not hasattr(config, 'DATASET') or config.DATASET is None:
            raise WorkflowError(
                "Config must have [DATASET] section for prediction",
                {"function": "build_prediction_dataloader_from_config"}
            )

        dataset_config = config.DATASET

        # Build dataset from config
        from dlkit.tools.config.core.factories import FactoryProvider
        from dlkit.tools.config.core.context import BuildContext

        build_context = BuildContext(mode="inference")

        # Create dataset
        dataset = FactoryProvider.create_component(
            dataset_config,
            build_context
        )

        # Get dataloader settings from config or use defaults
        batch_size = 32
        num_workers = 0
        drop_last = False

        if hasattr(config, 'DATALOADER') and config.DATALOADER is not None:
            batch_size = getattr(config.DATALOADER, 'batch_size', batch_size)
            num_workers = getattr(config.DATALOADER, 'num_workers', num_workers)
            drop_last = getattr(config.DATALOADER, 'drop_last', drop_last)

        # Build dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Never shuffle for inference
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=device not in ("cpu", "auto")
        )

        return dataloader

    except WorkflowError:
        raise
    except Exception as e:
        raise WorkflowError(
            f"Failed to build dataloader from config: {str(e)}",
            {"function": "build_prediction_dataloader_from_config", "error": str(e)}
        ) from e


def validate_config_for_prediction(config: GeneralSettings) -> tuple[bool, str | None]:
    """Validate configuration for prediction use.

    Args:
        config: Configuration to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> valid, error = validate_config_for_prediction(config)
        >>> if not valid:
        ...     print(f"Invalid config: {error}")
    """
    if not hasattr(config, 'DATASET') or config.DATASET is None:
        return False, "Config missing [DATASET] section"

    return True, None
