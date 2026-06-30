"""Transform.state_dict() must not silently drop nested buffers.

nn.Module.state_dict()'s recursive traversal calls child.state_dict(destination=...)
and relies entirely on in-place mutation of the shared `destination` dict — it
discards each call's return value. Transform.state_dict() must reuse that exact
dict object. A naive `destination or {}` breaks this: an empty (but non-None)
dict is falsy, so `{} or {}` silently swaps in a *different* new dict, and
whatever that child writes is orphaned — never merged back into the real
accumulator, with no error raised. This surfaces whenever a Transform is the
first thing visited during traversal (the accumulator is still empty at that
point) — e.g. a bare TransformChain, or any composed Transform-in-Transform
nesting, independent of whatever Lightning wrapper happens to surround it.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import ModuleList

from dlkit.domain.transforms.base import Transform
from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.ica import ICA
from dlkit.domain.transforms.incremental_pca import IncrementalPCA
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.domain.transforms.pca import PCA
from dlkit.domain.transforms.standard import StandardScaler
from dlkit.domain.transforms.truncated_svd import TruncatedSVD


def test_chain_state_dict_includes_nested_transform_buffers() -> None:
    """A fitted chain's state_dict() must contain every nested transform's buffers,
    not just the chain's own top-level _fitted flag."""
    chain = TransformChain(ModuleList([MinMaxScaler(dim=0), StandardScaler(dim=0)]), entry_name="x")
    chain.fit(torch.randn(20, 4))

    state = chain.state_dict()

    assert "transforms.0.min" in state
    assert "transforms.0.max" in state
    assert "transforms.1.mean" in state
    assert "transforms.1.std" in state


def test_chain_state_dict_round_trips_into_a_fresh_unfitted_chain() -> None:
    """A fresh, unfitted chain of the same structure must load a fitted chain's
    full state_dict() with strict=True — no manual buffer registration needed."""
    data = torch.randn(20, 4)
    fitted = TransformChain(
        ModuleList([MinMaxScaler(dim=0), StandardScaler(dim=0)]), entry_name="x"
    )
    fitted.fit(data)

    fresh = TransformChain(ModuleList([MinMaxScaler(dim=0), StandardScaler(dim=0)]), entry_name="x")
    fresh.load_state_dict(fitted.state_dict(), strict=True)

    assert fresh.fitted
    assert fresh.transforms[0].fitted
    assert fresh.transforms[1].fitted
    assert torch.allclose(fresh(data), fitted(data))


@pytest.mark.parametrize(
    ("factory", "data"),
    [
        (lambda: StandardScaler(dim=0), torch.randn(20, 4)),
        (lambda: MinMaxScaler(dim=0), torch.randn(20, 4)),
        (lambda: PCA(n_components=2), torch.randn(20, 4)),
        (lambda: IncrementalPCA(n_components=2), torch.randn(20, 4)),
        (lambda: TruncatedSVD(n_components=2), torch.randn(20, 4)),
        (lambda: ICA(n_components=2), torch.randn(20, 4)),
    ],
    ids=["StandardScaler", "MinMaxScaler", "PCA", "IncrementalPCA", "TruncatedSVD", "ICA"],
)
def test_fittable_transform_round_trips_through_state_dict(
    factory: type[Transform], data: torch.Tensor
) -> None:
    """Every FittableTransform: fit -> state_dict -> load into a fresh unfitted
    instance with strict=True, with no manual buffer registration anywhere."""
    fitted = factory()
    fitted.fit(data)

    fresh = factory()
    fresh.load_state_dict(fitted.state_dict(), strict=True)

    assert fresh.fitted
    assert torch.allclose(fresh(data), fitted(data))
