"""Lightning-based model wrappers with protocol-based composition.

Provides enhanced Lightning wrappers that integrate with the TensorDict
data pipeline for named, routed loss/metric computation and named transform chains.
"""

from .base import ProcessingLightningWrapper
from .components import (
    MetricRoute,
    ModelOutputSpec,
    NamedBatchTransformer,
    RoutedLossComputer,
    RoutedMetricsUpdater,
    TensorDictModelInvoker,
    WrapperCheckpointMetadata,
    _build_invoker_from_entries,
    _NullLossComputer,
    _NullMetricsUpdater,
    _NullModelInvoker,
)
from .factories import WrapperFactory
from .graph import GraphLightningWrapper
from .protocols import (
    IBatchTransformer,
    IFittableBatchTransformer,
    ILossComputer,
    IMetricsUpdater,
    IModelInvoker,
)
from .standard import StandardLightningWrapper
from .timeseries import TimeSeriesLightningWrapper

__all__ = [
    "GraphLightningWrapper",
    "IBatchTransformer",
    "IFittableBatchTransformer",
    "ILossComputer",
    "IMetricsUpdater",
    "IModelInvoker",
    "MetricRoute",
    "ModelOutputSpec",
    "NamedBatchTransformer",
    "ProcessingLightningWrapper",
    "RoutedLossComputer",
    "RoutedMetricsUpdater",
    "StandardLightningWrapper",
    "TensorDictModelInvoker",
    "TimeSeriesLightningWrapper",
    "WrapperCheckpointMetadata",
    "WrapperFactory",
    "_NullLossComputer",
    "_NullMetricsUpdater",
    "_NullModelInvoker",
    "_build_invoker_from_entries",
]
