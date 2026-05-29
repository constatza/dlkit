from pathlib import Path

import pytest
import torch

from dlkit.domain.transforms.errors import TransformNotFittedError
from dlkit.domain.transforms.ica import ICA


@pytest.fixture
def data() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(100, 8)


@pytest.fixture
def fitted_ica(data: torch.Tensor) -> ICA:
    t = ICA(n_components=4)
    t.fit(data)
    return t


class TestICA:
    def test_fit_sets_fitted_flag(self, data: torch.Tensor) -> None:
        t = ICA(n_components=4)
        assert not t.fitted
        t.fit(data)
        assert t.fitted

    def test_forward_reduces_last_dim(self, fitted_ica: ICA, data: torch.Tensor) -> None:
        out = fitted_ica(data)
        assert out.shape == (100, 4)

    def test_forward_raises_when_not_fitted(self, data: torch.Tensor) -> None:
        t = ICA(n_components=4)
        with pytest.raises(TransformNotFittedError):
            t(data)

    def test_inverse_transform_shape(self, fitted_ica: ICA, data: torch.Tensor) -> None:
        projected = fitted_ica(data)
        reconstructed = fitted_ica.inverse_transform(projected)
        assert reconstructed.shape == data.shape

    def test_state_dict_round_trip(self, fitted_ica: ICA, data: torch.Tensor) -> None:
        out1 = fitted_ica(data)
        t2 = ICA(n_components=4)
        t2.load_state_dict(fitted_ica.state_dict())
        torch.testing.assert_close(t2(data), out1)

    def test_multidim_input(self, fitted_ica: ICA) -> None:
        x = torch.randn(5, 3, 8)
        out = fitted_ica(x)
        assert out.shape == (5, 3, 4)

    def test_infer_output_shape(self) -> None:
        t = ICA(n_components=4)
        assert t.infer_output_shape((100, 8)) == (100, 4)
        assert t.infer_output_shape((5, 3, 8)) == (5, 3, 4)

    def test_file_checkpoint_round_trip(
        self, fitted_ica: ICA, data: torch.Tensor, tmp_path: Path
    ) -> None:
        # Exercises _load_from_state_dict: empty placeholder buffers must be
        # replaced with correctly-shaped tensors before PyTorch copies weights in.
        path = tmp_path / "ica.pt"
        torch.save(fitted_ica.state_dict(), path)

        ica2 = ICA(n_components=4)
        ica2.load_state_dict(torch.load(path, weights_only=True))

        assert ica2.fitted
        torch.testing.assert_close(ica2.mean, fitted_ica.mean)
        torch.testing.assert_close(ica2.components, fitted_ica.components)
        torch.testing.assert_close(ica2.mixing, fitted_ica.mixing)
        torch.testing.assert_close(ica2(data), fitted_ica(data))
