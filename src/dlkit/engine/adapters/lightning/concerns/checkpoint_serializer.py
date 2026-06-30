"""Checkpoint serialization concern for ProcessingLightningWrapper.

Encapsulates DLKit metadata save/load logic for Lightning checkpoints.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import nn

from dlkit.infrastructure.utils.logging_config import get_logger

from ._checkpoint_serializer_helpers import serialize_shapes as _serialize_shapes

logger = get_logger(__name__)


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
            has_context = getattr(meta, "context", None) is not None
            dlkit_metadata["model_settings"] = self._serialize_model_settings(
                meta.model_settings, has_context=has_context
            )
            dlkit_metadata["entry_configs"] = self._serialize_entry_configs(meta.entry_configs)
            dlkit_metadata["feature_names"] = list(meta.feature_names)
            dlkit_metadata["forward_arg_map"] = dict(meta.forward_arg_map)
            dlkit_metadata["predict_target_key"] = meta.predict_target_key
            dlkit_metadata["model_family"] = self._detect_model_family()
            dlkit_metadata["input_shapes"] = _serialize_shapes(getattr(meta, "input_shapes", None))
            dlkit_metadata["output_shapes"] = _serialize_shapes(
                getattr(meta, "output_shapes", None)
            )
        else:
            dlkit_metadata["model_settings"] = {}
            dlkit_metadata["entry_configs"] = []
            dlkit_metadata["model_family"] = "external"
            dlkit_metadata["input_shapes"] = None
            dlkit_metadata["output_shapes"] = None

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

    def _detect_model_family(self) -> str:
        """Detect model family identifier.

        Returns:
            Model family string (e.g., 'dlkit_nn', 'graph', 'external').
        """
        try:
            from dlkit.domain.nn.detection import detect_model_type

            if self._checkpoint_metadata is not None:
                model_type = detect_model_type(self._checkpoint_metadata.model_settings)
                return model_type.value
        except (ImportError, AttributeError) as exc:
            logger.warning("Could not detect model family, falling back to 'external': {}", exc)
        return "external"

    def _serialize_model_settings(
        self, model_settings: Any, *, has_context: bool = True
    ) -> dict[str, Any]:
        """Serialize model settings to a flat checkpoint DTO.

        Shape kwargs are excluded from ``hyper_kwargs`` at save time so that
        inference can reconstruct them via ``from_context`` without stripping.

        Args:
            model_settings: Model configuration settings.

        Returns:
            Serialized model configuration dict.
        """
        import importlib

        from dlkit.engine.adapters.lightning.checkpoint_dto import ModelCheckpointDTO
        from dlkit.infrastructure.config.model_components import (
            ModelComponentSettings,
            extract_init_kwargs,
        )

        name = getattr(model_settings, "name", None)
        module_path = getattr(model_settings, "module_path", None) or ""
        if name is None:
            return {}

        if not isinstance(model_settings, ModelComponentSettings):
            raise TypeError(
                "Checkpoint serialization requires ModelComponentSettings; "
                f"got {type(model_settings)!r}"
            )
        init_kwargs = extract_init_kwargs(model_settings)

        # When name is a type, extract the proper string name and module path
        if isinstance(name, type):
            serialized_name = name.__qualname__
            serialized_module = name.__module__
            model_cls: type | None = name
        else:
            serialized_name = str(name)
            serialized_module = module_path
            try:
                mod = importlib.import_module(module_path or "dlkit.domain.nn")
                model_cls = getattr(mod, serialized_name, None)
            except (ImportError, AttributeError) as exc:
                logger.warning(
                    "Could not import model class {}.{}, shape_kwarg_names will not be applied: {}",
                    module_path,
                    serialized_name,
                    exc,
                )
                model_cls = None

        # Strip shape kwargs only when a ShapeContext was provided — if the user
        # passed in_features/out_features explicitly in config, keep them.
        shape_keys: frozenset[str] = (
            model_cls.shape_kwarg_names()
            if has_context and model_cls is not None and hasattr(model_cls, "shape_kwarg_names")
            else frozenset()
        )
        hyper_kwargs = {k: v for k, v in init_kwargs.items() if k not in shape_keys}

        dto = ModelCheckpointDTO(
            name=serialized_name,
            module_path=serialized_module,
            hyper_kwargs=hyper_kwargs,
        )
        return dto.model_dump()

    def _serialize_entry_configs(self, entry_configs: tuple) -> list[dict[str, Any]]:
        """Serialize entry configurations.

        Args:
            entry_configs: Tuple of DataEntry objects.

        Returns:
            List of serialized entry config dicts.
        """
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
