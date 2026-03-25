"""Fixtures for precision testing following SOLID principles.

This module provides a clean factory-based approach to creating test models
that supports both the old BaseModel interface and the new ShapeAware/ShapeAgnostic
architecture, following the Factory and Adapter patterns.
"""

from typing import Any, Protocol

import pytest
import torch

from dlkit.core.models.nn.base import DLKitModel
from dlkit.core.shape_specs import ModelFamily, ShapeSource, create_shape_spec
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.tools.config.precision import PrecisionStrategy


class PrecisionTestModelProtocol(Protocol):
    """Protocol for models used in precision testing.

    This protocol defines the interface that test models must implement,
    providing clean separation between test logic and model implementation
    following the Interface Segregation Principle.
    """

    def get_precision_strategy(self) -> PrecisionStrategy | None:
        """Get the precision strategy used by this model."""
        ...

    def get_model_dtype(self) -> torch.dtype:
        """Get the current dtype of model parameters."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        ...

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step for compatibility with old BaseModel interface."""
        ...


class TestModelFactory:
    """Factory for creating test models with different interfaces.

    This factory abstracts the creation of test models, supporting both
    the old BaseModel-style interface and the new shape-aware architecture.
    Follows the Factory Pattern to decouple test code from concrete model classes.
    """

    @staticmethod
    def create_precision_test_model(
        shape_dict: dict[str, tuple], model_type: str = "shape_aware"
    ) -> PrecisionTestModelProtocol:
        """Create a test model for precision testing.

        Args:
            shape_dict: Shape specification in old format {"x": (dim,), "y": (dim,)}
            model_type: "shape_aware" or "shape_agnostic" (for future extension)

        Returns:
            Model implementing PrecisionTestModelProtocol

        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "shape_aware":
            return ShapeAwareTestModel(shape_dict=shape_dict)
        if model_type == "shape_agnostic":
            # Future extension point for ShapeAgnostic models
            raise NotImplementedError("ShapeAgnostic test models not yet implemented")
        raise ValueError(f"Unsupported model_type: {model_type}")


class ShapeAwareTestModel(DLKitModel):
    """Test model for shape-based model construction.

    This model uses DLKitModel with shape specification for model configuration,
    following the Adapter Pattern to maintain compatibility with existing tests.
    """

    def __init__(self, shape_dict: dict[str, tuple]):
        """Initialize test model with shape specification.

        Args:
            shape_dict: Shape specification in old format
        """
        super().__init__()

        # Convert old shape format to new IShapeSpec
        shape_spec = create_shape_spec(
            shapes=shape_dict,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )
        self._unified_shape = shape_spec

        # Build simple linear model architecture
        input_dim = shape_dict["x"][0]
        output_dim = shape_dict["y"][0]
        self.linear = torch.nn.Linear(input_dim, output_dim)

        # Apply precision from context (simulating Lightning behavior)
        service = get_precision_service()
        precision_strategy = service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.linear(x)

    def get_precision_strategy(self) -> PrecisionStrategy | None:
        """Get the effective precision strategy for this model.

        Returns the resolved precision strategy considering context overrides
        and session configuration. Infers precision from the model's actual dtype.
        """
        # Infer precision from model's actual dtype
        model_dtype = self.get_model_dtype()
        if model_dtype == torch.float64:
            return PrecisionStrategy.FULL_64
        if model_dtype == torch.float32:
            return PrecisionStrategy.FULL_32
        if model_dtype == torch.float16:
            return PrecisionStrategy.TRUE_16
        if model_dtype == torch.bfloat16:
            return PrecisionStrategy.TRUE_BF16
        # Fallback to service resolution
        precision_service = get_precision_service()
        return precision_service.resolve_precision()

    def get_model_dtype(self) -> torch.dtype:
        """Get the current dtype of model parameters."""
        return next(self.parameters()).dtype

    def predict_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Prediction step for compatibility with old BaseModel interface.

        This method provides compatibility with the old BaseModel interface,
        supporting both tuple batches (x, y) and single tensor batches.
        """
        # Handle tuple batch (x, y) - use only x for prediction
        if isinstance(batch, tuple) and len(batch) >= 1:
            x = batch[0]
        else:
            # Handle single tensor batch
            x = batch

        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected tensor input, got {type(x)}")

        # Run forward pass (Lightning handles precision casting)
        return self.forward(x)


@pytest.fixture
def test_model_factory() -> TestModelFactory:
    """Provide test model factory.

    Returns:
        TestModelFactory instance for creating test models
    """
    return TestModelFactory()


@pytest.fixture
def sample_shape() -> dict[str, tuple]:
    """Standard shape specification for testing.

    Returns:
        Shape dictionary with input and output dimensions
    """
    return {"x": (10,), "y": (5,)}
