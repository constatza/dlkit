from .base import GridOperatorBase, IGridOperator, IOperatorNetwork, IQueryOperator
from .deeponet import DeepONet, MLPDeepONet
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
    "MLPDeepONet",
]
