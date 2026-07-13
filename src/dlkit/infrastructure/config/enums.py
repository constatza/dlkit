from enum import StrEnum


class DatasetFamily(StrEnum):
    FLEXIBLE = "flexible"
    GRAPH = "graph"


class DataModuleName(StrEnum):
    IN_MEMORY = "ArrayDataModule"
    GRAPH = "GraphDataModule"


class AdjustLrFn(StrEnum):
    """Muon-family learning-rate shape adjustment strategy.

    ``ORIGINAL`` applies ``lr * sqrt(max(1, A/B))``; ``MATCH_RMS_ADAMW`` applies
    ``0.2 * lr * sqrt(max(A, B))`` so Muon can reuse AdamW-tuned hyperparameters.
    """

    ORIGINAL = "original"
    MATCH_RMS_ADAMW = "match_rms_adamw"
