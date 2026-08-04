"""Unit tests for ``dlkit.domain.nn.base.DLKitModule``.

Covers the ``_DLKitModuleMeta`` contract-enforcement matrix: abstract bases
are skipped, concrete classes require ``InputSpec``, empty ``InputSpec`` is
accepted (single-flat-input convention), non-empty ``InputSpec`` must agree
with ``forward()``'s parameter names, and multi-level ABC chains resolve
correctly (mirroring the real ``DeepONet`` -> ``_FlatBranchDeepONet`` ->
``VarWidthDeepONet`` shape).
"""

from __future__ import annotations

import pytest
import torch

from dlkit.common.errors import ForwardContractError
from dlkit.domain.nn.base import DLKitModule
from dlkit.domain.nn.contracts import InputSpec

# ---------------------------------------------------------------------------
# Helper models — reusable, correctly-defined DLKitModule subclasses
# ---------------------------------------------------------------------------


class _SingleInputModel(DLKitModule):
    """Concrete model with an empty InputSpec (single-flat-input convention)."""

    class InputSpec(InputSpec):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _MultiInputModel(DLKitModule):
    """Concrete model with a non-empty InputSpec matching forward()'s params."""

    class InputSpec(InputSpec):
        branch: int = 0
        trunk: int = 0

    def forward(self, branch: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        return branch + trunk


class _AbstractIntermediateModel(DLKitModule):
    """Concrete InputSpec declared, but forward() left abstract — still abstract overall."""

    class InputSpec(InputSpec):
        branch: int = 0
        trunk: int = 0


class _ConcreteLeafModel(_AbstractIntermediateModel):
    """Leaf of a 3-level chain: DLKitModule -> _AbstractIntermediateModel -> this."""

    def forward(self, branch: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        return branch + trunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def single_input_model() -> _SingleInputModel:
    """An instantiated model with an empty InputSpec."""
    return _SingleInputModel()


@pytest.fixture
def sample_input() -> torch.Tensor:
    """A small tensor sized for _SingleInputModel's Linear(4, 4)."""
    return torch.randn(2, 4)


# ---------------------------------------------------------------------------
# Abstract-skip behaviour
# ---------------------------------------------------------------------------


class TestAbstractSkip:
    def test_dlkit_module_itself_is_not_instantiable(self) -> None:
        """DLKitModule declares forward() abstract; it cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            DLKitModule()

    def test_intermediate_abstract_subclass_does_not_raise_on_definition(self) -> None:
        """A subclass that still leaves forward() abstract is not contract-checked yet."""
        assert issubclass(_AbstractIntermediateModel, DLKitModule)

    def test_intermediate_abstract_subclass_is_not_instantiable(self) -> None:
        """Still-abstract intermediate bases remain non-instantiable, same as nn.Module/ABC."""
        with pytest.raises(TypeError, match="abstract"):
            _AbstractIntermediateModel()


# ---------------------------------------------------------------------------
# Concrete-class contract enforcement
# ---------------------------------------------------------------------------


class TestConcreteContractEnforcement:
    def test_missing_input_spec_raises_at_class_definition(self) -> None:
        """A concrete subclass with no InputSpec at all raises immediately."""
        with pytest.raises(ForwardContractError, match="declares no InputSpec"):

            class _MissingSpec(DLKitModule):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x

    def test_empty_input_spec_is_accepted(
        self, single_input_model: _SingleInputModel, sample_input: torch.Tensor
    ) -> None:
        """An empty InputSpec passes — nothing named to check — and the model works."""
        output = single_input_model(sample_input)
        assert output.shape == sample_input.shape

    def test_mismatched_input_spec_field_raises_at_class_definition(self) -> None:
        """An InputSpec field with no matching forward() parameter raises immediately."""
        with pytest.raises(ForwardContractError, match=r"declares \['branch'\]"):

            class _Mismatched(DLKitModule):
                class InputSpec(InputSpec):
                    branch: int = 0

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x

    def test_matching_multi_field_input_spec_is_accepted(self) -> None:
        """InputSpec fields that all match forward() parameters pass cleanly."""
        branch, trunk = torch.randn(2, 3), torch.randn(2, 3)
        model = _MultiInputModel()
        result = model(branch=branch, trunk=trunk)
        assert torch.equal(result, branch + trunk)


# ---------------------------------------------------------------------------
# Multi-level ABC chain
# ---------------------------------------------------------------------------


class TestMultiLevelChain:
    def test_leaf_of_three_level_chain_is_instantiable(self) -> None:
        """DLKitModule -> abstract intermediate (with InputSpec) -> concrete leaf resolves cleanly."""
        model = _ConcreteLeafModel()
        branch, trunk = torch.randn(2, 3), torch.randn(2, 3)
        result = model(branch=branch, trunk=trunk)
        assert torch.equal(result, branch + trunk)

    def test_leaf_inherits_parent_input_spec(self) -> None:
        """The leaf reuses the intermediate base's InputSpec without redeclaring it."""
        assert _ConcreteLeafModel.InputSpec is _AbstractIntermediateModel.InputSpec
