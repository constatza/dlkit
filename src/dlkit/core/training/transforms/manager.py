"""Transform management service for Lightning wrappers.

This module provides a service class that handles transform lifecycle management,
including application, caching, checkpoint persistence, and device management.

Design Pattern: Service Layer
- Encapsulates transform management logic
- Provides clean API for Lightning wrapper integration
- Handles side effects (caching, persistence) explicitly
"""

from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleDict

from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.training.transforms.base import InvertibleTransform


class TransformManager:
    """Manages transform chains for Lightning wrappers.

    This service handles:
    - Transform application (forward and inverse)
    - Transform caching across training phases
    - Checkpoint persistence (via ModuleDict)
    - Device management for transforms

    Architecture:
    - Service Pattern: Encapsulates transform management concerns
    - Dependency Injection: Injected into Lightning wrapper
    - SOLID Compliance: Single responsibility (transform management only)

    Example:
        >>> manager = TransformManager()
        >>> # Apply forward transforms
        >>> transformed = manager.apply_forward({'features': data}, transform_chains)
        >>> # Apply inverse transforms
        >>> denormalized = manager.apply_inverse({'predictions': preds}, transform_chains)
    """

    def __init__(self):
        """Initialize the transform manager."""
        self.fitted_transforms: ModuleDict = ModuleDict()

    def apply_forward(
        self,
        data: dict[str, Tensor],
        transform_chains: dict[str, TransformChain],
    ) -> dict[str, Tensor]:
        """Apply forward transforms to data tensors.

        Args:
            data: Dictionary mapping entry names to tensors.
            transform_chains: Dictionary mapping entry names to transform chains.

        Returns:
            Dictionary with transformed tensors. Entries without transforms
            are returned unchanged.

        Example:
            >>> data = {'features': torch.randn(32, 64)}
            >>> chains = {'features': feature_transform_chain}
            >>> transformed = manager.apply_forward(data, chains)
        """
        result = {}
        for name, tensor in data.items():
            chain = transform_chains.get(name)
            if chain is not None and callable(chain):
                result[name] = chain(tensor)
            else:
                result[name] = tensor
        return result

    def apply_inverse(
        self,
        data: dict[str, Tensor],
        transform_chains: dict[str, TransformChain],
    ) -> dict[str, Tensor]:
        """Apply inverse transforms to data tensors using Protocol check.

        Only applies inverse if the chain implements InvertibleTransform Protocol.
        This provides type safety and clear interface contracts.

        Args:
            data: Dictionary mapping entry names to tensors.
            transform_chains: Dictionary mapping entry names to transform chains.

        Returns:
            Dictionary with inverse-transformed tensors. Entries without
            invertible transforms are returned unchanged.

        Example:
            >>> predictions = {'target': torch.randn(32, 10)}
            >>> chains = {'target': target_transform_chain}
            >>> denormalized = manager.apply_inverse(predictions, chains)
        """
        result = {}
        for name, tensor in data.items():
            chain = transform_chains.get(name)
            if chain is not None and isinstance(chain, InvertibleTransform):
                try:
                    result[name] = chain.inverse_transform(tensor)
                except Exception:
                    # Gracefully handle inverse failures (e.g., SampleNormL2 without forward)
                    result[name] = tensor
            else:
                result[name] = tensor
        return result

    def get_transform_chain(
        self,
        name: str,
        runtime_cache: dict[str, TransformChain] | None = None,
    ) -> TransformChain | None:
        """Get transform chain for an entry, checking cache first then persisted transforms.

        Args:
            name: Entry name to look up.
            runtime_cache: Optional runtime cache to check first.

        Returns:
            Transform chain if found, None otherwise.
        """
        # Try runtime cache first
        if runtime_cache is not None:
            chain = runtime_cache.get(name)
            if chain is not None:
                return chain

        # Fall back to persisted transforms
        return self.fitted_transforms.get(name)  # type: ignore[return-value]

    def persist_transforms(
        self,
        transform_caches: list[dict[str, TransformChain]],
    ) -> None:
        """Persist transform chains from runtime caches to ModuleDict.

        This ensures transforms are saved in Lightning checkpoints.

        Args:
            transform_caches: List of cache dictionaries from different phases
                (train, val, test, predict).

        Side Effects:
            Updates self.fitted_transforms ModuleDict with missing chains.
        """
        for cache in transform_caches:
            for name, chain in cache.items():
                try:
                    if name not in self.fitted_transforms:
                        self.fitted_transforms[name] = chain
                except Exception:
                    # Be resilient; this is best-effort registration
                    pass

    def hydrate_caches(
        self,
        target_caches: list[dict[str, TransformChain]],
    ) -> None:
        """Populate runtime caches from persisted ModuleDict.

        This enables inference-only runs by reusing transforms loaded
        from checkpoints.

        Args:
            target_caches: List of cache dictionaries to populate.

        Side Effects:
            Populates target_caches with transforms from fitted_transforms.
        """
        if not isinstance(self.fitted_transforms, ModuleDict) or len(self.fitted_transforms) == 0:
            return

        for name, chain in self.fitted_transforms.items():
            for cache in target_caches:
                cache[name] = chain

    def to_checkpoint_dict(self) -> dict[str, Any]:
        """Serialize manager state for checkpoint.

        Returns:
            Dictionary with serializable manager state.
        """
        return {
            "transform_names": list(self.fitted_transforms.keys()) if self.fitted_transforms else [],
            "num_transforms": len(self.fitted_transforms) if self.fitted_transforms else 0,
        }

    def from_checkpoint_dict(self, state: dict[str, Any]) -> None:
        """Restore manager state from checkpoint.

        Note: fitted_transforms ModuleDict is restored by Lightning automatically,
        so this method is mainly for validation and future extensions.

        Args:
            state: Dictionary with serialized manager state.
        """
        # Lightning automatically loads the fitted_transforms ModuleDict
        # This method is for future extensions and validation
        expected_names = state.get("transform_names", [])
        if expected_names:
            actual_names = list(self.fitted_transforms.keys())
            if set(expected_names) != set(actual_names):
                import warnings
                warnings.warn(
                    f"Checkpoint transform names mismatch: "
                    f"expected {expected_names}, got {actual_names}",
                    UserWarning,
                    stacklevel=2,
                )

    def get_device(self) -> torch.device:
        """Get the device of the first transform in fitted_transforms.

        Returns:
            Device of transforms, or CPU if no transforms.
        """
        if self.fitted_transforms and len(self.fitted_transforms) > 0:
            first_transform = next(iter(self.fitted_transforms.values()))
            # Get device from first parameter
            try:
                return next(first_transform.parameters()).device
            except StopIteration:
                # No parameters, check buffers
                try:
                    return next(first_transform.buffers()).device
                except StopIteration:
                    pass
        return torch.device('cpu')
