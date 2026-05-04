from __future__ import annotations

import dlkit.nn as public_nn
from dlkit.domain import nn as domain_nn


def test_domain_nn_exports_residual_dense_ffnns() -> None:
    assert domain_nn.FeedForwardNN is not None
    assert domain_nn.ConstantWidthFFNN is not None


def test_public_nn_exports_residual_dense_ffnns() -> None:
    assert public_nn.FeedForwardNN is not None
    assert public_nn.ConstantWidthFFNN is not None


def test_public_namespaces_export_symmetric_constrained_pairs() -> None:
    for namespace in (domain_nn, public_nn):
        assert namespace.ConstantWidthFactorizedFFNN is not None
        assert namespace.ConstantWidthSimpleFactorizedFFNN is not None
        assert namespace.EmbeddedSPDFFNN is not None
        assert namespace.EmbeddedSimpleSPDFFNN is not None
        assert namespace.ScaleEquivariantEmbeddedSPDFactorizedFFNN is not None
        assert namespace.ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN is not None
