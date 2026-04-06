from enum import StrEnum


class DatasetFamily(StrEnum):
    FLEXIBLE = "flexible"
    GRAPH = "graph"
    TIMESERIES = "timeseries"


class DataModuleName(StrEnum):
    IN_MEMORY = "InMemoryModule"
    GRAPH = "GraphDataModule"
    TIMESERIES = "TimeSeriesDataModule"
