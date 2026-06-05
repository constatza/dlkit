from .base import GridOperatorBase, IGridOperator, IOperatorNetwork, IQueryOperator
from .deeponet import DeepONet, EmbeddedDeepONet, FFNNDeepONet, VarWidthDeepONet
from .fno import FourierNeuralOperator1d

__all__ = [
    # Protocols
    "IOperatorNetwork",
    "IGridOperator",
    "IQueryOperator",
    # Composable base classes
    "GridOperatorBase",
    # Concrete operators
    "FourierNeuralOperator1d",
    "DeepONet",
    "VarWidthDeepONet",
    "FFNNDeepONet",
    "EmbeddedDeepONet",
]
