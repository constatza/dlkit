"""requires_materialized_fit marks transforms that cannot stream-fit.

Most fittable transforms implement IncrementalFittableTransform and stream-fit
in bounded memory (see TransformChain.fit_from_dataloader). PCA, TruncatedSVD,
and ICA are batch-only algorithms (true of sklearn too — only IncrementalPCA
has a streaming variant), so they materialize the full dataset during fit.
This marker makes that tradeoff an explicit, queryable class attribute instead
of an implicit `isinstance(transform, IncrementalFittableTransform)` branch.
"""

from __future__ import annotations

from dlkit.domain.transforms.base import Transform
from dlkit.domain.transforms.ica import ICA
from dlkit.domain.transforms.incremental_pca import IncrementalPCA
from dlkit.domain.transforms.minmax import MinMaxScaler
from dlkit.domain.transforms.pca import PCA
from dlkit.domain.transforms.standard import StandardScaler
from dlkit.domain.transforms.truncated_svd import TruncatedSVD


def test_requires_materialized_fit_defaults_to_false_on_base_class() -> None:
    assert Transform.requires_materialized_fit is False


def test_pca_requires_materialized_fit() -> None:
    assert PCA.requires_materialized_fit is True


def test_truncated_svd_requires_materialized_fit() -> None:
    assert TruncatedSVD.requires_materialized_fit is True


def test_ica_requires_materialized_fit() -> None:
    assert ICA.requires_materialized_fit is True


def test_standard_scaler_does_not_require_materialized_fit() -> None:
    assert StandardScaler.requires_materialized_fit is False


def test_minmax_scaler_does_not_require_materialized_fit() -> None:
    assert MinMaxScaler.requires_materialized_fit is False


def test_incremental_pca_does_not_require_materialized_fit() -> None:
    assert IncrementalPCA.requires_materialized_fit is False
