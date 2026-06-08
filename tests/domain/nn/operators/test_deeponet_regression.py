import pytest

from dlkit.common.errors import WorkflowError
from dlkit.domain.nn.contracts import BranchTrunkSpec, ContractConsumer, TabulaRSpec
from dlkit.domain.nn.factory import build_model
from dlkit.domain.nn.operators.deeponet import VarWidthDeepONet


def test_deeponet_protocol_match():
    assert issubclass(VarWidthDeepONet, ContractConsumer), "Must match protocol structurally"


def test_deeponet_missing_contract_args():
    # If contract is None, it should fail with a WorkflowError explicitly stating that a contract is expected
    with pytest.raises(WorkflowError, match="This model expects a contract"):
        build_model(
            VarWidthDeepONet, kwargs={"branch_layers": [64], "trunk_layers": [64]}, contract=None
        )


def test_deeponet_wrong_contract():
    # If contract is wrong, it should fail with a different error
    with pytest.raises(TypeError, match="require BranchTrunkSpec"):
        build_model(
            VarWidthDeepONet,
            kwargs={"branch_layers": [64], "trunk_layers": [64]},
            contract=TabulaRSpec(in_shape=(10,), out_shape=(1,)),
        )


def test_deeponet_correct_contract():
    # If contract is correct, it should succeed
    contract = BranchTrunkSpec(branch_shape=(10,), query_shape=(2,), out_features=1)
    model = build_model(
        VarWidthDeepONet, kwargs={"branch_layers": [64], "trunk_layers": [64]}, contract=contract
    )
    assert isinstance(model, VarWidthDeepONet)
