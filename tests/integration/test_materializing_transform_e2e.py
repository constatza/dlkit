"""End-to-end coverage for batch-only (materializing) transforms.

PCA, TruncatedSVD, and ICA (Transform.requires_materialized_fit = True) take a
different code path through TransformChain.fit_from_dataloader than the
incremental transforms (StandardScaler, MinMaxScaler, IncrementalPCA) — see
domain/transforms/chain.py. Every other transform test in the suite exercises
that materializing path at the unit level only (mocked dataloaders, hand-built
TransformChain instances). This test drives it through the real production
path instead: BuildFactory -> IBuildStrategy.build() -> trainer.fit() ->
checkpoint save -> dlkit.load_model() -> predict(), with a chain that mixes an
incremental transform ahead of a materializing one on the same entry — the
exact "PCA vs a batch-fitted transform" interaction.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch

import dlkit
from dlkit.domain.transforms.chain import TransformChain
from dlkit.domain.transforms.pca import PCA
from dlkit.domain.transforms.standard import StandardScaler
from dlkit.engine.adapters.lightning.datamodules.array import ArrayDataModule
from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.transform_pipeline import NamedBatchTransformer
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.infrastructure.config.transform_settings import TransformSettings


def _with_mixed_incremental_and_materializing_transform(
    training_settings: TrainingJobConfig,
) -> TrainingJobConfig:
    """Attach [StandardScaler (incremental), PCA (materializing)] to the feature entry."""
    standard_scaler = TransformSettings(
        name="StandardScaler", module_path="dlkit.domain.transforms.standard", dim=0
    )
    pca = TransformSettings.model_validate(
        {"name": "PCA", "module_path": "dlkit.domain.transforms.pca", "n_components": 2}
    )
    data = training_settings.data
    feature = data.features[0].model_copy(update={"transforms": [standard_scaler, pca]})
    updated_data = data.model_copy(update={"features": (feature, *data.features[1:])})
    return training_settings.model_copy(update={"data": updated_data})


def test_materializing_transform_fits_through_build_factory(
    training_settings: TrainingJobConfig,
) -> None:
    """PCA fits correctly via the real build path, after an incremental
    transform already fitted ahead of it in the same chain."""
    settings = _with_mixed_incremental_and_materializing_transform(training_settings)

    components = BuildFactory().build_components(settings)
    model = components.model
    assert isinstance(model, StandardLightningWrapper)

    batch_transformer = model.batch_transformer
    assert isinstance(batch_transformer, NamedBatchTransformer)
    assert batch_transformer.is_fitted()
    chain = cast(TransformChain, batch_transformer._feature_chains["x"])
    standard_scaler = cast(StandardScaler, chain.transforms[0])
    pca = cast(PCA, chain.transforms[1])
    assert standard_scaler.fitted
    assert pca.fitted
    assert cast(torch.Tensor, pca.components).shape[0] == 2


def test_materializing_transform_survives_checkpoint_round_trip(
    training_settings: TrainingJobConfig, tmp_path: Path
) -> None:
    """Train with a mixed incremental+materializing chain, save a real Lightning
    checkpoint, then load_model() and predict — exercising the checkpoint fix
    (Transform.state_dict() dict-identity) and the simplified reconstruction
    path (engine/inference/transforms.py) together against a materializing
    transform specifically, not just the incremental ones every other test uses.
    """
    settings = _with_mixed_incremental_and_materializing_transform(training_settings)
    components = BuildFactory().build_components(settings)
    model, trainer, datamodule = components.model, components.trainer, components.datamodule
    assert trainer is not None
    assert isinstance(datamodule, ArrayDataModule)

    trainer.fit(model, datamodule=datamodule)
    ckpt_path = tmp_path / "materializing_transform.ckpt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    raw_batch = next(iter(datamodule.predict_dataloader()))
    x = raw_batch["features", "x"]

    with dlkit.load_model(
        checkpoint_path=ckpt_path, device="cpu", apply_transforms=True
    ) as predictor:
        result = predictor.predict(x=x)

    predictions = result.predictions
    assert isinstance(predictions, torch.Tensor)
    assert not torch.isnan(predictions).any()
