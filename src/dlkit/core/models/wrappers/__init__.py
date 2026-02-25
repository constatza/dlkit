"""Lightning-based model wrappers with protocol-based composition.

Provides enhanced Lightning wrappers that integrate with the TensorDict
data pipeline for named, routed loss/metric computation and named transform chains.
"""

from .base import ProcessingLightningWrapper
from .standard import StandardLightningWrapper
from .graph import GraphLightningWrapper
from .timeseries import TimeSeriesLightningWrapper
from .factories import WrapperFactory
from .protocols import (
    ILossComputer,
    IMetricsUpdater,
    IModelInvoker,
    IBatchTransformer,
    IFittableBatchTransformer,
)
from .components import (
    StandardModelInvoker,
    RoutedLossComputer,
    MetricRoute,
    RoutedMetricsUpdater,
    NamedBatchTransformer,
    WrapperCheckpointMetadata,
    _NullModelInvoker,
    _NullLossComputer,
    _NullMetricsUpdater,
)

__all__ = [
    "ProcessingLightningWrapper",
    "StandardLightningWrapper",
    "GraphLightningWrapper",
    "TimeSeriesLightningWrapper",
    "WrapperFactory",
    "ILossComputer",
    "IMetricsUpdater",
    "IModelInvoker",
    "IBatchTransformer",
    "IFittableBatchTransformer",
    "StandardModelInvoker",
    "RoutedLossComputer",
    "MetricRoute",
    "RoutedMetricsUpdater",
    "NamedBatchTransformer",
    "WrapperCheckpointMetadata",
    "_NullModelInvoker",
    "_NullLossComputer",
    "_NullMetricsUpdater",
]
