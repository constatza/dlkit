from typing import cast

import pytest
import torch

from dlkit.domain.transforms.errors import TransformNotFittedError
from dlkit.domain.transforms.truncated_svd import TruncatedSVD


@pytest.fixture
def data() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(80, 10)


@pytest.fixture
def fitted_tsvd(data: torch.Tensor) -> TruncatedSVD:
    t = TruncatedSVD(n_components=4)
    t.fit(data)
    return t


class TestTruncatedSVD:
    def test_fit_sets_fitted_flag(self, data: torch.Tensor) -> None:
        t = TruncatedSVD(n_components=4)
        assert not t.fitted
        t.fit(data)
        assert t.fitted

    def test_forward_reduces_last_dim(self, fitted_tsvd: TruncatedSVD, data: torch.Tensor) -> None:
        out = fitted_tsvd(data)
        assert out.shape == (80, 4)

    def test_forward_raises_when_not_fitted(self, data: torch.Tensor) -> None:
        t = TruncatedSVD(n_components=4)
        with pytest.raises(TransformNotFittedError):
            t(data)

    def test_does_not_subtract_mean(self, data: torch.Tensor) -> None:
        # Shifting data changes the output — confirms no centering
        t1, t2 = TruncatedSVD(n_components=4), TruncatedSVD(n_components=4)
        t1.fit(data)
        t2.fit(data + 100.0)
        assert not torch.allclose(t1(data), t2(data))

    def test_inverse_transform_shape(self, fitted_tsvd: TruncatedSVD, data: torch.Tensor) -> None:
        reconstructed = fitted_tsvd.inverse_transform(fitted_tsvd(data))
        assert reconstructed.shape == data.shape

    def test_state_dict_round_trip(self, fitted_tsvd: TruncatedSVD, data: torch.Tensor) -> None:
        out1 = fitted_tsvd(data)
        t2 = TruncatedSVD(n_components=4)
        t2.load_state_dict(fitted_tsvd.state_dict())
        torch.testing.assert_close(t2(data), out1)

    def test_multidim_input(self, fitted_tsvd: TruncatedSVD) -> None:
        x = torch.randn(5, 3, 10)
        out = fitted_tsvd(x)
        assert out.shape == (5, 3, 4)

    def test_infer_output_shape(self) -> None:
        t = TruncatedSVD(n_components=4)
        assert t.infer_output_shape((80, 10)) == (80, 4)
        assert t.infer_output_shape((5, 3, 10)) == (5, 3, 4)

    def test_fit_stores_singular_values(self, fitted_tsvd: TruncatedSVD) -> None:
        assert hasattr(fitted_tsvd, "singular_values")
        sv = cast("torch.Tensor", fitted_tsvd.singular_values)
        assert sv.shape == (4,)
        assert (sv >= 0).all()

    def test_fit_stores_explained_energy_ratio(self, fitted_tsvd: TruncatedSVD) -> None:
        ratio = cast("torch.Tensor", fitted_tsvd.explained_energy_ratio)
        assert ratio.ndim == 0
        assert 0.0 < ratio.item() <= 1.0

    def test_explained_energy_ratio_in_state_dict_round_trip(
        self, fitted_tsvd: TruncatedSVD
    ) -> None:
        t2 = TruncatedSVD(n_components=4)
        t2.load_state_dict(fitted_tsvd.state_dict())
        torch.testing.assert_close(
            cast("torch.Tensor", t2.explained_energy_ratio),
            cast("torch.Tensor", fitted_tsvd.explained_energy_ratio),
        )
