"""Factory for creating predictor instances with dependency injection.

This module provides factory classes for creating predictor instances
following the Factory pattern with proper dependency injection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.tools.config.precision.strategy import PrecisionStrategy
from .predictor import CheckpointPredictor, PredictorConfig, IPredictor


class PredictorFactory:
    """Factory for creating predictor instances.

    This factory encapsulates predictor creation logic and handles
    dependency injection of use cases and other dependencies.

    The factory follows the Factory pattern with dependency injection:
    - Dependencies injected via constructor
    - Multiple creation methods for different scenarios
    - Consistent interface across predictor types

    Usage:
        >>> factory = PredictorFactory(model_loading_use_case, inference_execution_use_case)
        >>> predictor = factory.create_from_checkpoint("model.ckpt", device="cuda")
        >>> result = predictor.predict(inputs)
    """

    def __init__(
        self,
        model_loading_use_case: Any,  # ModelLoadingUseCase
        inference_execution_use_case: Any,  # InferenceExecutionUseCase
    ):
        """Initialize factory with required use cases.

        Args:
            model_loading_use_case: Use case for loading models from checkpoints
            inference_execution_use_case: Use case for executing inference
        """
        self._model_loading_use_case = model_loading_use_case
        self._inference_execution_use_case = inference_execution_use_case

    def create_from_checkpoint(
        self,
        checkpoint_path: Path | str,
        device: str = "auto",
        batch_size: int = 32,
        apply_transforms: bool = True,
        auto_load: bool = True,
        precision: PrecisionStrategy | None = None
    ) -> CheckpointPredictor:
        """Create predictor from checkpoint path.

        This is the primary method for creating predictors. It creates
        a predictor configured to load from the specified checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Target device ("auto", "cpu", "cuda", "mps", "cuda:0", etc.)
            batch_size: Default batch size for inference
            apply_transforms: Whether to apply fitted transforms from checkpoint
            auto_load: If True, load model immediately (default)
            precision: Optional precision override. If None, infers from model dtype.

        Returns:
            CheckpointPredictor: Configured predictor instance

        Example:
            >>> factory = get_predictor_factory()
            >>> predictor = factory.create_from_checkpoint("model.ckpt", device="cuda")
            >>> result = predictor.predict(inputs)
        """
        config = PredictorConfig(
            checkpoint_path=Path(checkpoint_path),
            device=device,
            batch_size=batch_size,
            apply_transforms=apply_transforms,
            auto_load=auto_load,
            precision=precision
        )

        return CheckpointPredictor(
            config=config,
            model_loading_use_case=self._model_loading_use_case,
            inference_execution_use_case=self._inference_execution_use_case
        )

    def create_from_config(
        self,
        config: PredictorConfig
    ) -> CheckpointPredictor:
        """Create predictor from predictor configuration.

        Useful when you want to pre-configure predictor settings
        and create the predictor later.

        Args:
            config: Predictor configuration

        Returns:
            CheckpointPredictor: Configured predictor instance

        Example:
            >>> config = PredictorConfig(checkpoint_path=Path("model.ckpt"), device="cuda")
            >>> predictor = factory.create_from_config(config)
        """
        return CheckpointPredictor(
            config=config,
            model_loading_use_case=self._model_loading_use_case,
            inference_execution_use_case=self._inference_execution_use_case
        )

    def create_lazy(
        self,
        checkpoint_path: Path | str,
        device: str = "auto",
        batch_size: int = 32,
        apply_transforms: bool = True,
        precision: PrecisionStrategy | None = None
    ) -> CheckpointPredictor:
        """Create predictor without loading model (lazy initialization).

        The predictor will load the model on first predict() call or
        when load() is explicitly called.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Target device
            batch_size: Default batch size
            apply_transforms: Whether to apply transforms
            precision: Optional precision override. If None, infers from model dtype.

        Returns:
            CheckpointPredictor: Unloaded predictor instance

        Example:
            >>> predictor = factory.create_lazy("model.ckpt")
            >>> # Model not loaded yet
            >>> predictor.load()  # Explicit load
            >>> result = predictor.predict(inputs)
        """
        return self.create_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=batch_size,
            apply_transforms=apply_transforms,
            auto_load=False,  # Don't auto-load
            precision=precision
        )
