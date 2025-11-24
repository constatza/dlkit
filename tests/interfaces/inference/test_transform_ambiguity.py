"""Tests for transform ambiguity resolution with separated Feature/Target transforms.

This test suite validates that:
1. Single target transforms are automatically selected (no ambiguity)
2. Multiple target transforms raise clear errors
3. Dict-based predictions work correctly
4. Empty target transforms return raw predictions
5. Feature and Target transforms are separated at the source (not at usage)

Note: Transforms are now separated during checkpoint loading and model state creation,
so tests receive only target_transforms (Feature transforms are handled separately).
"""

from __future__ import annotations

import torch
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from dlkit.interfaces.inference.infrastructure.adapters import DirectInferenceExecutor
from dlkit.core.training.transforms.errors import TransformAmbiguityError
from dlkit.tools.config.data_entries import Feature, Target


@pytest.fixture
def executor() -> DirectInferenceExecutor:
    """Create DirectInferenceExecutor instance."""
    return DirectInferenceExecutor()


@pytest.fixture
def mock_transform_chain() -> Mock:
    """Create mock TransformChain that returns input unchanged."""
    chain = Mock()
    chain.to.return_value = chain
    chain.inverse_transform.side_effect = lambda x: x  # Return input unchanged
    return chain


@pytest.fixture
def single_feature_single_target_entry_configs() -> Dict[str, Any]:
    """Entry configs with one feature and one target (unambiguous case)."""
    return {
        "x": Feature(name="x", value=torch.randn(10, 5)),
        "y": Target(name="y", value=torch.randn(10, 3))
    }


@pytest.fixture
def multi_target_entry_configs() -> Dict[str, Any]:
    """Entry configs with multiple targets (ambiguous case)."""
    return {
        "x": Feature(name="x", value=torch.randn(10, 5)),
        "rhs": Target(name="rhs", value=torch.randn(10, 3)),
        "sol": Target(name="sol", value=torch.randn(10, 8))
    }


@pytest.fixture
def single_target_entry_configs() -> Dict[str, Any]:
    """Entry configs with only one target among multiple entries."""
    return {
        "features": Feature(name="features", value=torch.randn(10, 5)),
        "metadata": Feature(name="metadata", value=torch.randn(10, 2)),
        "target": Target(name="target", value=torch.randn(10, 3))
    }


class TestSingleTargetSmartSelection:
    """Test automatic selection when only one target transform exists."""

    def test_single_target_selected_automatically(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock,
        single_feature_single_target_entry_configs: Dict[str, Any]
    ):
        """Single target transform should be selected automatically."""
        predictions = torch.randn(10, 5)
        # Transforms are separated at source - pass only target transforms
        target_transforms = {"y": mock_transform_chain}
        device = "cpu"

        result = executor._apply_inverse_tensor_transform(
            predictions,
            target_transforms,
            device
        )

        # Should use the 'y' transform
        mock_transform_chain.inverse_transform.assert_called_once()
        assert torch.allclose(result, predictions)

    def test_single_target_among_multiple_features(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock,
        single_target_entry_configs: Dict[str, Any]
    ):
        """Single target should be selected (features already separated at source)."""
        predictions = torch.randn(10, 5)
        # Features separated at source - only target transforms passed here
        target_transforms = {"target": mock_transform_chain}
        device = "cpu"

        result = executor._apply_inverse_tensor_transform(
            predictions,
            target_transforms,
            device
        )

        # Should use the 'target' transform
        assert mock_transform_chain.inverse_transform.call_count >= 1


class TestMultiTargetAmbiguity:
    """Test error handling when multiple target transforms exist."""

    def test_multiple_targets_raises_error(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock,
        multi_target_entry_configs: Dict[str, Any]
    ):
        """Multiple target transforms should raise TransformAmbiguityError."""
        predictions = torch.randn(10, 5)
        # Both are targets - ambiguous!
        target_transforms = {
            "rhs": mock_transform_chain,
            "sol": mock_transform_chain
        }
        device = "cpu"

        with pytest.raises(TransformAmbiguityError) as exc_info:
            executor._apply_inverse_tensor_transform(
                predictions,
                target_transforms,
                device
            )

        # Error should mention both targets
        error_msg = str(exc_info.value)
        assert "rhs" in error_msg
        assert "sol" in error_msg
        assert "multiple target transforms" in error_msg.lower()

    def test_multiple_targets_with_feature_transform(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock,
        multi_target_entry_configs: Dict[str, Any]
    ):
        """Multiple targets should raise error (features already separated)."""
        predictions = torch.randn(10, 5)
        # Feature 'x' is in feature_transforms (not passed here)
        # Only target transforms passed - still ambiguous!
        target_transforms = {
            "rhs": mock_transform_chain,  # Target transform
            "sol": mock_transform_chain   # Target transform
        }
        device = "cpu"

        with pytest.raises(TransformAmbiguityError) as exc_info:
            executor._apply_inverse_tensor_transform(
                predictions,
                target_transforms,
                device
            )

        # Should only mention target transforms in error
        error_msg = str(exc_info.value)
        assert "rhs" in error_msg
        assert "sol" in error_msg


class TestDictPredictions:
    """Test dict-based predictions work correctly (no ambiguity)."""

    def test_dict_predictions_use_key_matching(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock,
        multi_target_entry_configs: Dict[str, Any]
    ):
        """Dict predictions should match by key, no ambiguity."""
        predictions = {
            "rhs": torch.randn(10, 5),
            "sol": torch.randn(10, 8)
        }
        fitted_transforms = {
            "rhs": mock_transform_chain,
            "sol": mock_transform_chain
        }
        device = "cpu"

        # Use the dict transform method
        result = executor._apply_inverse_dict_transforms(
            predictions,
            fitted_transforms,
            device
        )

        # Should apply transform to both outputs by key
        assert "rhs" in result
        assert "sol" in result
        assert mock_transform_chain.inverse_transform.call_count == 2


class TestSimplestCases:
    """Test the simplest unambiguous cases (separated transforms from source)."""

    def test_single_target_transform_applies(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock
    ):
        """Single target transform should apply (trivial case)."""
        predictions = torch.randn(10, 5)
        target_transforms = {"y": mock_transform_chain}
        device = "cpu"

        result = executor._apply_inverse_tensor_transform(
            predictions,
            target_transforms,
            device
        )

        # Should use the single target transform
        mock_transform_chain.inverse_transform.assert_called_once()

    def test_no_target_transforms_returns_raw(
        self,
        executor: DirectInferenceExecutor
    ):
        """Empty target_transforms should return predictions unchanged."""
        predictions = torch.randn(10, 5)
        target_transforms = {}  # No target transforms
        device = "cpu"

        result = executor._apply_inverse_tensor_transform(
            predictions,
            target_transforms,
            device
        )

        # Should return raw predictions
        assert torch.allclose(result, predictions)

    def test_multiple_targets_fail_fast(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock
    ):
        """Multiple target transforms should fail immediately."""
        predictions = torch.randn(10, 5)
        target_transforms = {
            "y1": mock_transform_chain,
            "y2": mock_transform_chain
        }
        device = "cpu"

        with pytest.raises(TransformAmbiguityError):
            executor._apply_inverse_tensor_transform(
                predictions,
                target_transforms,
                device
            )


class TestDictTransformApplication:
    """Test that dict predictions work correctly with separated transforms."""

    def test_dict_with_single_target(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock
    ):
        """Dict predictions should work when targets are separated."""
        predictions_dict = {"y": torch.randn(10, 5)}
        target_transforms = {"y": mock_transform_chain}
        device = "cpu"

        # Use the dict transform method
        result = executor._apply_inverse_dict_transforms(
            predictions_dict,
            target_transforms,
            device
        )

        # Should apply transform to the 'y' key
        assert "y" in result
        mock_transform_chain.inverse_transform.assert_called_once()

    def test_dict_with_multiple_targets(
        self,
        executor: DirectInferenceExecutor,
        mock_transform_chain: Mock
    ):
        """Dict predictions with multiple targets should apply each."""
        predictions_dict = {
            "y1": torch.randn(10, 5),
            "y2": torch.randn(10, 3)
        }
        target_transforms = {
            "y1": mock_transform_chain,
            "y2": mock_transform_chain
        }
        device = "cpu"

        result = executor._apply_inverse_dict_transforms(
            predictions_dict,
            target_transforms,
            device
        )

        # Should apply transforms to both keys
        assert "y1" in result
        assert "y2" in result
        assert mock_transform_chain.inverse_transform.call_count == 2
