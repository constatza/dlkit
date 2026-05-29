from pathlib import Path

import pytest
import torch
from torch.nn import ModuleList

from dlkit.domain.transforms.base import IncrementalFittableTransform
from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.errors import TransformNotFittedError
from dlkit.domain.transforms.incremental_pca import IncrementalPCA


@pytest.fixture
def data() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(120, 12)


@pytest.fixture
def fitted_ipca(data: torch.Tensor) -> IncrementalPCA:
    t = IncrementalPCA(n_components=5, batch_size=30)
    t.reset_fit_state()
    for i in range(0, 120, 30):
        t.update_fit(data[i : i + 30])
    t.finalize_fit()
    return t


class TestIncrementalPCA:
    def test_implements_incremental_protocol(self) -> None:
        assert isinstance(IncrementalPCA(n_components=5), IncrementalFittableTransform)

    def test_streaming_fit_sets_fitted_flag(self, fitted_ipca: IncrementalPCA) -> None:
        assert fitted_ipca.fitted

    def test_forward_reduces_last_dim(
        self, fitted_ipca: IncrementalPCA, data: torch.Tensor
    ) -> None:
        out = fitted_ipca(data)
        assert out.shape == (120, 5)

    def test_forward_raises_when_not_fitted(self, data: torch.Tensor) -> None:
        t = IncrementalPCA(n_components=5)
        with pytest.raises(TransformNotFittedError):
            t(data)

    def test_inverse_transform_shape(self, fitted_ipca: IncrementalPCA, data: torch.Tensor) -> None:
        projected = fitted_ipca(data)
        reconstructed = fitted_ipca.inverse_transform(projected)
        assert reconstructed.shape == data.shape

    def test_state_dict_round_trip(self, fitted_ipca: IncrementalPCA, data: torch.Tensor) -> None:
        out1 = fitted_ipca(data)
        t2 = IncrementalPCA(n_components=5)
        t2.load_state_dict(fitted_ipca.state_dict())
        torch.testing.assert_close(t2(data), out1)

    def test_chain_fits_via_dataloader(self, data: torch.Tensor) -> None:
        t = IncrementalPCA(n_components=5, batch_size=30)
        chain = TransformChain(ModuleList([t]))
        batches = [{"x": data[i : i + 30]} for i in range(0, 120, 30)]
        chain.fit_from_dataloader(batches, tensor_selector=lambda b: b["x"])
        assert chain.fitted
        assert chain(data).shape == (120, 5)

    def test_infer_output_shape(self) -> None:
        t = IncrementalPCA(n_components=5)
        assert t.infer_output_shape((120, 12)) == (120, 5)

    def test_finalize_fit_stores_explained_variance_ratio(
        self, fitted_ipca: IncrementalPCA
    ) -> None:
        ratio = fitted_ipca.explained_variance_ratio
        assert ratio.shape == (5,)
        assert 0.0 < ratio.sum().item() <= 1.0

    def test_explained_variance_ratio_in_state_dict_round_trip(
        self, fitted_ipca: IncrementalPCA
    ) -> None:
        t2 = IncrementalPCA(n_components=5)
        t2.load_state_dict(fitted_ipca.state_dict())
        torch.testing.assert_close(
            t2.explained_variance_ratio, fitted_ipca.explained_variance_ratio
        )

    def test_file_checkpoint_round_trip(
        self, fitted_ipca: IncrementalPCA, data: torch.Tensor, tmp_path: Path
    ) -> None:
        # Exercises _load_from_state_dict: empty placeholder buffers must be
        # replaced with correctly-shaped tensors before PyTorch copies weights in.
        path = tmp_path / "incremental_pca.pt"
        torch.save(fitted_ipca.state_dict(), path)

        ipca2 = IncrementalPCA(n_components=5)
        ipca2.load_state_dict(torch.load(path, weights_only=True))

        assert ipca2.fitted
        torch.testing.assert_close(ipca2.mean, fitted_ipca.mean)
        torch.testing.assert_close(ipca2.components, fitted_ipca.components)
        torch.testing.assert_close(
            ipca2.explained_variance_ratio, fitted_ipca.explained_variance_ratio
        )
        torch.testing.assert_close(ipca2(data), fitted_ipca(data))
