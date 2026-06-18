from enum import StrEnum


class DatasetFamily(StrEnum):
    FLEXIBLE = "flexible"
    GRAPH = "graph"


class DataModuleName(StrEnum):
    IN_MEMORY = "ArrayDataModule"
    GRAPH = "GraphDataModule"
