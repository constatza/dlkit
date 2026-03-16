"""Checkpoint serialization concern for ProcessingLightningWrapper.

Encapsulates DLKit metadata save/load logic for Lightning checkpoints.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class ICheckpointSerializer(Protocol):
    """Protocol for checkpoint serialization and deserialization.

    Implementations handle saving/restoring DLKit metadata to/from Lightning checkpoints.
    """

    def serialize(self, checkpoint: dict[str, Any], wrapper_class_name: str) -> None:
        """Save DLKit metadata to checkpoint.

        Args:
            checkpoint: Checkpoint dict to augment in-place.
            wrapper_class_name: Name of the wrapper class (for metadata).
        """
        ...

    def deserialize(self, checkpoint: dict[str, Any]) -> None:
        """Restore and validate checkpoint metadata.

        Args:
            checkpoint: Checkpoint dict to restore from and mutate in-place.

        Raises:
            ValueError: If checkpoint metadata is missing or invalid.
        """
        ...


class DLKitCheckpointSerializer:
    """Handles dlkit_metadata save/load for Lightning checkpoints.

    Integrates with WrapperCheckpointMetadata to persist model and entry configs.
    """

    def __init__(self, checkpoint_metadata: Any, model: nn.Module) -> None:
        """Initialize with checkpoint metadata and model reference.

        Args:
            checkpoint_metadata: WrapperCheckpointMetadata instance or None.
            model: The PyTorch model (used for model family detection).
        """
        self._checkpoint_metadata = checkpoint_metadata
        self._model = model

    def serialize(self, checkpoint: dict[str, Any], wrapper_class_name: str) -> None:
        """Save DLKit metadata to checkpoint.

        Args:
            checkpoint: Checkpoint dict to augment in-place.
            wrapper_class_name: Name of the wrapper class.
        """
        dlkit_metadata: dict[str, Any] = {"wrapper_type": wrapper_class_name}

        if self._checkpoint_metadata is not None:
            meta = self._checkpoint_metadata
            dlkit_metadata["model_settings"] = self._serialize_model_settings(meta.model_settings)
            dlkit_metadata["entry_configs"] = self._serialize_entry_configs(meta.entry_configs)
            dlkit_metadata["shape_summary"] = self._compute_shape_summary(meta.shape_summary)
            dlkit_metadata["feature_names"] = list(meta.feature_names)
            dlkit_metadata["predict_target_key"] = meta.predict_target_key
            dlkit_metadata["model_family"] = self._detect_model_family()
        else:
            dlkit_metadata["model_settings"] = {}
            dlkit_metadata["entry_configs"] = []
            dlkit_metadata["shape_summary"] = {}
            dlkit_metadata["model_family"] = "external"

        checkpoint["dlkit_metadata"] = dlkit_metadata

    def deserialize(self, checkpoint: dict[str, Any]) -> None:
        """Restore and normalize checkpoint metadata.

        Args:
            checkpoint: Checkpoint dict to restore from (mutated in-place).

        Raises:
            ValueError: If checkpoint metadata is missing.
        """
        if "dlkit_metadata" not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'dlkit_metadata'. This checkpoint uses a legacy format "
                "that is no longer supported. Please re-train your model to generate "
                "a compatible checkpoint."
            )

        from dlkit.core.models.wrappers.checkpoint_dto import normalize_checkpoint_metadata

        checkpoint["dlkit_metadata"] = normalize_checkpoint_metadata(checkpoint["dlkit_metadata"])

    def _detect_model_family(self) -> str:
        """Detect model family identifier.

        Returns:
            Model family string (e.g., 'dlkit_nn', 'graph', 'external').
        """
        try:
            from dlkit.runtime.workflows.factories.model_detection import detect_model_type

            if self._checkpoint_metadata is not None:
                model_type = detect_model_type(self._checkpoint_metadata.model_settings, None)  # type: ignore[arg-type]
                return model_type.value
        except Exception:
            pass
        return "external"

    def _serialize_model_settings(self, model_settings: Any) -> dict[str, Any]:
        """Serialize model settings to a flat checkpoint DTO.

        Args:
            model_settings: Model configuration settings.

        Returns:
            Serialized model configuration dict.
        """
        try:
            from dlkit.core.models.wrappers.checkpoint_dto import ModelCheckpointDTO
            from dlkit.tools.config.components.model_components import (
                ModelComponentSettings,
                extract_init_kwargs,
            )

            name = getattr(model_settings, "name", None)
            module_path = getattr(model_settings, "module_path", None) or ""
            if name is None:
                return {}

            if isinstance(model_settings, ModelComponentSettings):
                init_kwargs = extract_init_kwargs(model_settings)
                all_hyperparams = model_settings.model_dump()
            elif hasattr(model_settings, "model_dump"):
                all_fields = model_settings.model_dump()
                excluded = {"name", "module_path", "checkpoint"}
                init_kwargs = {k: v for k, v in all_fields.items() if k not in excluded and v is not None}
                all_hyperparams = all_fields
            else:
                init_kwargs = {}
                all_hyperparams = {}

            # When name is a type, extract the proper string name and module path
            if isinstance(name, type):
                serialized_name = name.__qualname__
                serialized_module = name.__module__
            else:
                serialized_name = str(name)
                serialized_module = module_path

            dto = ModelCheckpointDTO(
                name=serialized_name,
                module_path=serialized_module,
                resolved_init_kwargs=init_kwargs,
                all_hyperparams=all_hyperparams,
            )
            return dto.model_dump()
        except Exception:
            return {}

    def _serialize_entry_configs(self, entry_configs: tuple) -> list[dict[str, Any]]:
        """Serialize entry configurations.

        Args:
            entry_configs: Tuple of DataEntry objects.

        Returns:
            List of serialized entry config dicts.
        """
        try:
            return [
                {
                    "name": e.name,
                    "class_name": e.__class__.__name__,
                    "transforms": [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in getattr(e, "transforms", [])
                    ],
                }
                for e in entry_configs
            ]
        except Exception:
            return []

    def _compute_shape_summary(self, shape_summary: Any) -> dict[str, Any]:
        """Serialize ShapeSummary for checkpoint persistence.

        Args:
            shape_summary: ShapeSummary instance or None.

        Returns:
            Dict with in_shapes/out_shapes or empty dict.
        """
        if shape_summary is None:
            return {}
        try:
            return {
                "in_shapes": [list(s) for s in shape_summary.in_shapes],
                "out_shapes": [list(s) for s in shape_summary.out_shapes],
            }
        except Exception:
            return {}
