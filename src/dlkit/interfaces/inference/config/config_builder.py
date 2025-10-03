"""Configuration builders for inference.

This module provides utilities to build InferenceConfig from
various sources like checkpoints and training configurations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .inference_config import InferenceConfig


class InferenceConfigBuilder:
    """Builder for creating InferenceConfig instances.

    This class follows the Builder pattern to construct inference configurations
    from different sources while maintaining flexibility and type safety.
    """

    def __init__(self) -> None:
        """Initialize the inference config builder."""
        self._checkpoint_path: Path | None = None
        self._feature_names: list[str] = []
        self._target_names: list[str] = []
        self._transform_names: list[str] = []
        self._model_shape: Shape | None = None
        self._device: str = "auto"
        self._dtype: torch.dtype = torch.float32
        self._batch_size: int = 32
        self._apply_transforms: bool = True
        self._wrapper_settings: dict[str, Any] = {}
        self._entry_configs: dict[str, Any] = {}

    def with_checkpoint(self, checkpoint_path: Path | str) -> InferenceConfigBuilder:
        """Set the model checkpoint path.

        Args:
            checkpoint_path: Path to the trained model checkpoint

        Returns:
            Self for method chaining
        """
        self._checkpoint_path = Path(checkpoint_path)
        return self

    def with_feature_names(self, feature_names: list[str]) -> InferenceConfigBuilder:
        """Set the expected feature names.

        Args:
            feature_names: List of input feature names

        Returns:
            Self for method chaining
        """
        self._feature_names = feature_names.copy()
        return self

    def with_target_names(self, target_names: list[str]) -> InferenceConfigBuilder:
        """Set the expected target names.

        Args:
            target_names: List of output target names

        Returns:
            Self for method chaining
        """
        self._target_names = target_names.copy()
        return self

    def with_device(self, device: str) -> InferenceConfigBuilder:
        """Set the inference device.

        Args:
            device: Device specification ("auto", "cpu", "cuda", "mps")

        Returns:
            Self for method chaining
        """
        self._device = device
        return self

    def with_batch_size(self, batch_size: int) -> InferenceConfigBuilder:
        """Set the default batch size.

        Args:
            batch_size: Default batch size for processing

        Returns:
            Self for method chaining
        """
        self._batch_size = batch_size
        return self

    def with_transforms(self, apply: bool) -> InferenceConfigBuilder:
        """Enable or disable transform application.

        Args:
            apply: Whether to apply fitted transforms

        Returns:
            Self for method chaining
        """
        self._apply_transforms = apply
        return self

    def from_checkpoint_metadata(self, metadata: dict[str, Any]) -> InferenceConfigBuilder:
        """Load configuration from checkpoint inference metadata.

        Args:
            metadata: Inference metadata from checkpoint

        Returns:
            Self for method chaining
        """
        self._feature_names = metadata.get("feature_names", [])
        self._target_names = metadata.get("target_names", [])
        self._transform_names = metadata.get("transform_names", [])
        self._model_shape = metadata.get("model_shape")
        self._wrapper_settings = metadata.get("wrapper_settings", {})
        self._entry_configs = metadata.get("entry_configs", {})
        return self

    def build(self) -> InferenceConfig:
        """Build the InferenceConfig instance.

        Returns:
            InferenceConfig: Complete inference configuration

        Raises:
            ValueError: If required configuration is missing
        """
        if self._checkpoint_path is None:
            raise ValueError("Checkpoint path is required")

        return InferenceConfig(
            model_checkpoint_path=self._checkpoint_path,
            feature_names=self._feature_names,
            target_names=self._target_names,
            transform_names=self._transform_names,
            model_shape=self._model_shape,
            device=self._device,
            dtype=self._dtype,
            batch_size=self._batch_size,
            apply_transforms=self._apply_transforms,
            wrapper_settings=self._wrapper_settings,
            entry_configs=self._entry_configs,
        )


def build_inference_config_from_checkpoint(
    checkpoint_path: Path | str,
    device: str = "auto",
    batch_size: int = 32,
    apply_transforms: bool = True
) -> InferenceConfig:
    """Build inference configuration directly from a checkpoint file.

    This is the primary way to create inference configurations.
    All necessary information is extracted from the checkpoint's inference metadata.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        device: Device specification ("auto", "cpu", "cuda", "mps")
        batch_size: Default batch size for processing
        apply_transforms: Whether to apply fitted transforms

    Returns:
        InferenceConfig: Complete inference configuration

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        KeyError: If checkpoint doesn't contain inference metadata
        ValueError: If inference metadata is invalid

    Example:
        >>> config = build_inference_config_from_checkpoint("model.ckpt")
        >>> print(config.feature_names)
        ['x', 'features']
        >>> print(config.target_names)
        ['y', 'output']
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load checkpoint to extract inference metadata
    try:
        # Use weights_only=False for DLKit checkpoints which may contain custom classes
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}") from e

    # Extract inference metadata
    if "inference_metadata" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain inference metadata. "
            f"This checkpoint may be from an older version of DLKit or "
            f"was not saved with inference support."
        )

    metadata = checkpoint["inference_metadata"]

    # Validate metadata structure
    required_fields = ["feature_names", "target_names", "transform_names"]
    missing_fields = [field for field in required_fields if field not in metadata]
    if missing_fields:
        raise ValueError(f"Inference metadata missing required fields: {missing_fields}")

    # Build configuration using the builder
    builder = InferenceConfigBuilder()
    config = (builder
              .with_checkpoint(checkpoint_path)
              .with_device(device)
              .with_batch_size(batch_size)
              .with_transforms(apply_transforms)
              .from_checkpoint_metadata(metadata)
              .build())

    return config


def build_inference_config_from_training_config(
    training_config_path: Path | str,
    checkpoint_path: Path | str,
    device: str = "auto",
    batch_size: int = 32
) -> InferenceConfig:
    """Build inference configuration from training configuration.

    This method extracts inference requirements from a training configuration
    and combines them with a checkpoint path. Useful for initial inference
    setup before the checkpoint contains inference metadata.

    Args:
        training_config_path: Path to training TOML configuration
        checkpoint_path: Path to the trained model checkpoint
        device: Device specification ("auto", "cpu", "cuda", "mps")
        batch_size: Default batch size for processing

    Returns:
        InferenceConfig: Inference configuration

    Raises:
        FileNotFoundError: If configuration or checkpoint files don't exist
        ImportError: If training configuration loading fails

    Example:
        >>> config = build_inference_config_from_training_config(
        ...     "training_config.toml",
        ...     "model.ckpt"
        ... )
    """
    from dlkit.tools.config import load_training_settings
    from dlkit.tools.config.data_entries import Target

    # Load training configuration
    training_settings = load_training_settings(training_config_path)

    # Extract feature and target names from dataset configuration
    feature_names = []
    target_names = []

    if hasattr(training_settings, 'DATASET') and training_settings.DATASET:
        # Extract from features and targets
        for feature in training_settings.DATASET.features:
            feature_names.append(feature.name)
        for target in training_settings.DATASET.targets:
            target_names.append(target.name)

    # Build configuration
    builder = InferenceConfigBuilder()
    config = (builder
              .with_checkpoint(checkpoint_path)
              .with_feature_names(feature_names)
              .with_target_names(target_names)
              .with_device(device)
              .with_batch_size(batch_size)
              .build())

    return config