"""Model reconstruction from checkpoint-only information.

This module provides the Builder pattern implementation for reconstructing
models from checkpoints without requiring training configuration files,
eliminating the need for manual shape parameters during inference.
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

from lightning import LightningModule

from dlkit.core.shape_specs import ShapeSpec
from dlkit.core.shape_specs import ShapeInferenceEngine, ShapeSystemFactory
from dlkit.tools.config.components.model_components import ModelComponentSettings
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.interfaces.api.domain.errors import WorkflowError


class ModelReconstructionBuilder:
    """Builder for reconstructing models from checkpoint-only information.

    This class implements the Builder pattern to incrementally reconstruct
    PyTorch Lightning models from checkpoint metadata without requiring
    the original training configuration files.

    The builder follows these steps:
    1. Load checkpoint and validate structure
    2. Infer shape using strategy chain
    3. Reconstruct model settings from metadata
    4. Build model with inferred components
    5. Load state dict into model
    """

    def __init__(self, checkpoint_path: Path | str):
        """Initialize builder with checkpoint path.

        Args:
            checkpoint_path: Path to the model checkpoint file

        Raises:
            FileNotFoundError: If checkpoint path does not exist
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Builder state
        self.checkpoint: Optional[Dict[str, Any]] = None
        self.shape_spec: Optional[ShapeSpec] = None
        self.model_settings: Optional[ModelComponentSettings] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self._is_legacy_checkpoint = False

    def load_checkpoint(self) -> ModelReconstructionBuilder:
        """Load and validate checkpoint structure.

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If checkpoint loading fails
        """
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False  # DLKit metadata contains custom objects
            )

            if not isinstance(self.checkpoint, dict):
                raise WorkflowError(f"Invalid checkpoint format: expected dict, got {type(self.checkpoint)}")

            # Extract metadata if available
            self.metadata = self.checkpoint.get('dlkit_metadata', {})
            self._is_legacy_checkpoint = not bool(self.metadata)

            if self._is_legacy_checkpoint:
                logger.info("Detected legacy checkpoint format")
            else:
                logger.info(f"Detected enhanced checkpoint format (version: {self.metadata.get('version', 'unknown')})")

            return self

        except Exception as e:
            raise WorkflowError(
                f"Failed to load checkpoint {self.checkpoint_path}: {str(e)}",
                {"component": "ModelReconstructionBuilder", "step": "load_checkpoint"}
            ) from e

    def infer_shape(self, dataset: Optional[Any] = None) -> ModelReconstructionBuilder:
        """Infer shape using strategy chain.

        Args:
            dataset: Optional dataset for shape inference fallback

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If shape inference fails completely
        """
        if self.checkpoint is None:
            raise RuntimeError("Must call load_checkpoint() first")

        try:
            logger.info("Starting shape inference chain")
            # Create shape inference engine
            shape_factory = ShapeSystemFactory.create_production_system()
            inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)

            # Extract model settings for shape inference if available
            model_settings = None
            if self.metadata and 'model_settings' in self.metadata:
                try:
                    model_settings = self._deserialize_model_settings(self.metadata['model_settings'])
                except Exception as e:
                    logger.warning(f"Failed to deserialize model settings for shape inference: {e}")

            # Run shape inference
            self.shape_spec = inference_engine.infer_from_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model_settings=model_settings
            )

            if self.shape_spec is None:
                raise WorkflowError(
                    "Shape inference failed with all strategies. "
                    "Please ensure the checkpoint contains valid shape information or provide a dataset.",
                    {"component": "ModelReconstructionBuilder", "step": "infer_shape"}
                )

            logger.info(f"Shape inference successful: {self.shape_spec}")
            return self

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Shape inference failed: {str(e)}",
                {"component": "ModelReconstructionBuilder", "step": "infer_shape"}
            ) from e

    def reconstruct_model_settings(self) -> ModelReconstructionBuilder:
        """Reconstruct model settings from checkpoint metadata.

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If model settings reconstruction fails
        """
        if self.checkpoint is None:
            raise RuntimeError("Must call load_checkpoint() first")

        try:
            logger.info("Reconstructing model settings")

            if self.metadata and 'model_settings' in self.metadata:
                # Enhanced metadata path
                self.model_settings = self._deserialize_model_settings(self.metadata['model_settings'])
                logger.info("Model settings reconstructed from enhanced metadata")
            else:
                # Legacy inference - attempt to reconstruct from available information
                self.model_settings = self._infer_model_settings_legacy()
                if self.model_settings:
                    logger.info("Model settings reconstructed from legacy inference")

            if self.model_settings is None:
                raise WorkflowError(
                    "Could not reconstruct model settings from checkpoint metadata. "
                    "This checkpoint may be incompatible with automatic inference.",
                    {"component": "ModelReconstructionBuilder", "step": "reconstruct_model_settings"}
                )

            return self

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Model settings reconstruction failed: {str(e)}",
                {"component": "ModelReconstructionBuilder", "step": "reconstruct_model_settings"}
            ) from e

    def build_model(self) -> LightningModule:
        """Build the final model with all inferred components.

        Returns:
            Reconstructed PyTorch Lightning model with loaded weights

        Raises:
            WorkflowError: If model building fails
            RuntimeError: If builder is not fully configured
        """
        self._validate_builder_state()

        try:
            logger.info("Building model from reconstructed components")

            # Create build context for inference
            model_context = BuildContext(mode="inference")

            # Apply shape overrides if available
            if self.shape_spec and not self.shape_spec.is_empty():
                # Ensure canonical aliases for wrapper compatibility
                shape_with_aliases = self.shape_spec.with_canonical_aliases()
                model_context = model_context.with_overrides(unified_shape=shape_with_aliases)
                logger.info(f"Applied shape overrides via ShapeSpec: {shape_with_aliases}")

            # Build base model using factory
            model = FactoryProvider.create_component(self.model_settings, model_context)
            logger.info(f"Base model created: {type(model).__name__}")

            # Load state dict with flexible key matching
            state_dict = self._extract_state_dict()
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            # Log key mismatches for debugging
            if missing_keys:
                logger.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

            # Set model to evaluation mode for inference
            model.eval()
            logger.info("Model successfully reconstructed and loaded")

            return model

        except Exception as e:
            raise WorkflowError(
                f"Model building failed: {str(e)}",
                {"component": "ModelReconstructionBuilder", "step": "build_model"}
            ) from e

    def _validate_builder_state(self) -> None:
        """Ensure builder is fully configured before building.

        Raises:
            RuntimeError: If required components are missing
        """
        missing = []
        if self.checkpoint is None:
            missing.append("checkpoint")
        if self.shape_spec is None:
            missing.append("shape_spec")
        if self.model_settings is None:
            missing.append("model_settings")

        if missing:
            raise RuntimeError(
                f"Builder not fully configured. Missing: {', '.join(missing)}. "
                "Call load_checkpoint(), infer_shape(), and reconstruct_model_settings()."
            )

    def _extract_state_dict(self) -> Dict[str, Any]:
        """Extract state dict from checkpoint with fallback handling.

        Returns:
            State dictionary for model loading
        """
        if "state_dict" in self.checkpoint:
            state_dict = self.checkpoint["state_dict"]
        else:
            # Fallback: assume entire checkpoint is state dict
            state_dict = self.checkpoint

        # Handle common key prefix issues
        if isinstance(state_dict, dict) and all(k.startswith("model.") for k in state_dict.keys()):
            logger.info("Stripping 'model.' prefix from state dict keys")
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        return state_dict

    def _deserialize_model_settings(self, settings_data: Dict[str, Any]) -> Optional[ModelComponentSettings]:
        """Deserialize model settings from checkpoint metadata.

        Args:
            settings_data: Serialized model settings

        Returns:
            ModelComponentSettings instance or None if deserialization fails
        """
        try:
            # Reconstruct ModelComponentSettings from serialized data
            name = settings_data.get('name')
            module_path = settings_data.get('module_path')
            params = settings_data.get('params', {})

            if name is None:
                return None

            # Create minimal model settings
            return ModelComponentSettings(
                name=name,
                module_path=module_path,
                params=params
            )

        except Exception as e:
            logger.warning(f"Failed to deserialize model settings: {e}")
            return None

    def _infer_model_settings_legacy(self) -> Optional[ModelComponentSettings]:
        """Attempt to infer model settings from legacy checkpoint format.

        Returns:
            ModelComponentSettings instance or None if inference fails
        """
        try:
            # This is a complex inference task for legacy checkpoints
            # For now, return None to indicate that legacy checkpoints
            # without enhanced metadata cannot be automatically reconstructed
            logger.warning(
                "Legacy checkpoint format detected without model settings. "
                "Automatic reconstruction not supported for this checkpoint format."
            )
            return None

        except Exception as e:
            logger.warning(f"Legacy model settings inference failed: {e}")
            return None


# Convenience function for simple reconstruction
def reconstruct_model_from_checkpoint(
    checkpoint_path: Path | str,
    dataset: Optional[Any] = None
) -> LightningModule:
    """Reconstruct model from checkpoint using the builder pattern.

    This is a convenience function that handles the full reconstruction process.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset: Optional dataset for shape inference fallback

    Returns:
        Reconstructed PyTorch Lightning model

    Raises:
        WorkflowError: If reconstruction fails at any step
        FileNotFoundError: If checkpoint doesn't exist
    """
    return (ModelReconstructionBuilder(checkpoint_path)
            .load_checkpoint()
            .infer_shape(dataset)
            .reconstruct_model_settings()
            .build_model())
