"""Tests for parameter inventory."""

from __future__ import annotations

import torch.nn as nn

from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.engine.training.optimization.inventory import (
    ParameterDescriptor,
    TorchParameterInventory,
)


class TestTorchParameterInventory:
    """Tests for TorchParameterInventory."""

    def test_inventory_returns_all_parameters(self, tiny_model: nn.Sequential) -> None:
        """Inventory lists all trainable parameters from the model.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)
        descriptors = inventory.list_parameters()

        # tiny_model has 2 Linear layers, each with weight + bias = 4 parameters
        expected_count = len(list(tiny_model.parameters()))
        assert len(descriptors) == expected_count
        assert len(descriptors) == 4

    def test_descriptor_names_are_correct(self, tiny_model: nn.Sequential) -> None:
        """Descriptor names match named_parameters() keys.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)
        descriptors = inventory.list_parameters()

        # Extract names from descriptors and from model
        desc_names = [d.name for d in descriptors]
        model_names = [name for name, _ in tiny_model.named_parameters()]

        assert desc_names == model_names

    def test_descriptor_shapes_are_correct(self, tiny_model: nn.Sequential) -> None:
        """Descriptor shapes match parameter shapes.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)
        descriptors = inventory.list_parameters()

        # Verify shapes match
        for desc in descriptors:
            assert desc.shape == desc.parameter.shape

    def test_descriptor_ndim_is_correct(self, tiny_model: nn.Sequential) -> None:
        """Descriptor ndim matches parameter rank.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)
        descriptors = inventory.list_parameters()

        # Verify ndim matches len(shape)
        for desc in descriptors:
            assert desc.ndim == len(desc.parameter.shape)

    def test_descriptor_module_path_is_correct(self, tiny_model: nn.Sequential) -> None:
        """Descriptor module_path is extracted correctly from parameter name.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)
        descriptors = inventory.list_parameters()

        # Extract module paths
        # Example: "0.weight" should have module_path="0"
        for desc in descriptors:
            if "." in desc.name:
                expected_path = desc.name.rsplit(".", 1)[0]
                assert desc.module_path == expected_path
            else:
                assert desc.module_path == ""

    def test_default_role_is_unknown(self, tiny_model: nn.Sequential) -> None:
        """Without a role_resolver, all parameters have UNKNOWN role.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)
        descriptors = inventory.list_parameters()

        # All should have UNKNOWN role by default
        for desc in descriptors:
            assert desc.role == ParameterRole.UNKNOWN

    def test_role_resolver_overrides_roles(self, tiny_model: nn.Sequential) -> None:
        """Role resolver is called and overrides default UNKNOWN role.

        Args:
            tiny_model: Two-layer Linear model.
        """

        def resolver(desc: ParameterDescriptor) -> ParameterRole:
            """Assign HIDDEN role to 2-D parameters, BIAS to others.

            Args:
                desc: Parameter descriptor.

            Returns:
                HIDDEN if ndim==2, BIAS otherwise.
            """
            if desc.ndim == 2:
                return ParameterRole.HIDDEN
            return ParameterRole.BIAS

        inventory = TorchParameterInventory(tiny_model, role_resolver=resolver)
        descriptors = inventory.list_parameters()

        # Verify roles were resolved
        for desc in descriptors:
            if desc.ndim == 2:
                assert desc.role == ParameterRole.HIDDEN
            else:
                assert desc.role == ParameterRole.BIAS

    def test_results_are_cached(self, tiny_model: nn.Sequential) -> None:
        """Calling list_parameters() twice returns the same cached object.

        Args:
            tiny_model: Two-layer Linear model.
        """
        inventory = TorchParameterInventory(tiny_model)

        # First call
        result1 = inventory.list_parameters()

        # Second call
        result2 = inventory.list_parameters()

        # Same object (not just equal, but identical)
        assert result1 is result2
