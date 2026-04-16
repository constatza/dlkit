"""Semantic parameter role definitions for neural network modules."""

from __future__ import annotations

from enum import Enum, auto


class ParameterRole(Enum):
    """Semantic role of a trainable parameter within a neural network.

    Used to determine optimizer eligibility (e.g. Muon requires hidden-layer
    weights with ndim == 2 and excludes input/output layers).
    """

    INPUT = auto()  # First-layer weights (embeddings, input projections)
    HIDDEN = auto()  # Interior weight matrices
    OUTPUT = auto()  # Final-layer weights / heads
    BIAS = auto()  # Any bias vector
    NORMALIZATION = auto()  # Layer-norm / batch-norm scale and shift
    EMBEDDING = auto()  # Token or positional embedding tables
    ENCODER = auto()  # Encoder-specific weights in autoencoder / seq2seq
    DECODER = auto()  # Decoder-specific weights
    UNKNOWN = auto()  # Could not be classified; default to conservative optimizer
