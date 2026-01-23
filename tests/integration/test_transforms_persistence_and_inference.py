from __future__ import annotations

from pathlib import Path
from typing import Any

import dlkit

import numpy as np
import torch
import pytest
from lightning.pytorch import Trainer
from torch import nn, Tensor

from dlkit.core.datasets.flexible import FlexibleDataset
from dlkit.core.datamodules.array import InMemoryModule
from dlkit.core.datatypes.split import IndexSplit
from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.transform_settings import TransformSettings
from dlkit.core.models.wrappers.standard import StandardLightningWrapper
from dlkit.core.models.nn.base import ShapeAwareModel
from dlkit.core.shape_specs import IShapeSpec


class IdentityHead(ShapeAwareModel):
    """Simple identity model that maps x -> y shape.

    Now follows the ShapeAware pattern with unified_shape parameter.
    """

    def __init__(self, *, unified_shape: IShapeSpec):
        super().__init__(unified_shape=unified_shape)
        self.last_x: Tensor | None = None

        # Extract shapes from the unified shape spec
        in_shape = unified_shape.get_shape("x") or (1,)
        out_shape = unified_shape.get_shape("y") or (1,)

        in_dim = int(in_shape[0])
        out_dim = int(out_shape[0])

        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        with torch.no_grad():
            if in_dim == out_dim:
                self.proj.weight.copy_(torch.eye(in_dim))

    def accepts_shape(self, shape_spec: IShapeSpec) -> bool:
        """Validate if this model can accept the given shape specification.

        Args:
            shape_spec: Shape specification to validate

        Returns:
            True if shape is acceptable (has both x and y shapes)
        """
        return shape_spec.has_shape("x") and shape_spec.has_shape("y")

    def forward(self, x: Tensor) -> Tensor:
        # Record input to verify direct transform application
        self.last_x = x.detach()
        return self.proj(x)


def _make_data(
    tmp_path: Path, n: int = 32, d: int = 4
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(loc=10.0, scale=5.0, size=(n, d)).astype(np.float32)
    # Targets equal to features for identity mapping
    Y = X.copy().astype(np.float32)
    fx = tmp_path / "features.npy"
    fy = tmp_path / "targets.npy"
    np.save(fx, X)
    np.save(fy, Y)
    return fx, fy, X, Y


def _build_datamodule(fx: Path, fy: Path, batch_size: int = 8) -> InMemoryModule:
    dataset = FlexibleDataset(
        features=[Feature(name="x", path=fx)], targets=[Target(name="y", path=fy)]
    )
    n = len(dataset)
    # Edge case: zero validation/test to ensure transforms fit on the full dataset
    train = tuple(range(0, n))
    val = tuple()
    test = train
    # Predict over the training indices for exact inverse-transform comparisons
    split = IndexSplit(train=train, validation=val, test=test, predict=train)
    from dlkit.tools.config.dataloader_settings import DataloaderSettings

    dm = InMemoryModule(
        dataset=dataset,
        split=split,
        dataloader=DataloaderSettings(
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
            persistent_workers=False,
        ),
    )
    dm.setup("fit")
    dm.setup("validate")
    dm.setup("test")
    dm.setup("predict")
    return dm


def _entry_configs(fx: Path, fy: Path) -> dict[str, Feature | Target]:
    # Apply MinMaxScaler to both x and y; direct for features, inverse for targets at predict
    ts = TransformSettings(
        name="MinMaxScaler", module_path="dlkit.core.training.transforms.minmax", dim=0
    )
    x = Feature(name="x", path=fx, transforms=[ts])
    y = Target(name="y", path=fy, transforms=[ts])
    return {"x": x, "y": y}


def _build_wrapper(entry_cfgs: dict[str, Feature | Target]) -> StandardLightningWrapper:
    from dlkit.core.shape_specs import create_shape_spec, ModelFamily, ShapeSource

    model_settings = ModelComponentSettings(name=IdentityHead, module_path="tests.helpers")
    wrapper_settings = WrapperComponentSettings()

    # x/y shapes provided by FlexibleDataset are 1D (feature dimension)
    # Create proper ShapeSpec using modern system
    shape_spec = create_shape_spec(
        shapes={"x": (4,), "y": (4,)},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.CONFIGURATION,
    )

    return StandardLightningWrapper(
        settings=wrapper_settings,
        model_settings=model_settings,
        shape_spec=shape_spec,
        entry_configs=entry_cfgs,
    )


def _basic_trainer() -> Trainer:
    return Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        limit_train_batches=0,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        accelerator="cpu",
        devices=1,
    )


def _extract_prediction_tensor(result: Any) -> torch.Tensor:
    """Retrieve the first tensor prediction from nested inference outputs."""
    queue: list[Any] = [result.predictions]
    while queue:
        current = queue.pop(0)
        if torch.is_tensor(current):
            return current
        if isinstance(current, dict):
            queue.extend(current.values())
    raise AssertionError("Inference result did not contain tensor predictions")


@pytest.fixture(scope="module")
def predictor_transform_setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Train a lightweight checkpoint with fitted transforms for reuse across tests."""
    tmp_dir = Path(tmp_path_factory.mktemp("predictor_transform_space"))
    fx, fy, X, Y = _make_data(tmp_dir)
    dm = _build_datamodule(fx, fy)
    entries = _entry_configs(fx, fy)
    wrapper = _build_wrapper(entries)
    trainer = _basic_trainer()

    trainer.fit(wrapper, datamodule=dm)
    ckpt_path = tmp_dir / "predictor_space.ckpt"
    trainer.save_checkpoint(ckpt_path)

    raw_features = torch.from_numpy(X).float()
    raw_targets = torch.from_numpy(Y).float()
    normalized_features = wrapper.feature_transforms({"x": raw_features})["x"]

    return {
        "checkpoint": ckpt_path,
        "raw_features": raw_features,
        "raw_targets": raw_targets,
        "normalized_features": normalized_features,
        "wrapper": wrapper,
    }


def test_transforms_persist_and_apply_with_load_from_checkpoint(tmp_path: Path) -> None:
    # Arrange
    fx, fy, X, Y = _make_data(tmp_path)
    dm = _build_datamodule(fx, fy)
    entries = _entry_configs(fx, fy)
    wrapper = _build_wrapper(entries)
    trainer = _basic_trainer()

    # Act: fit just to trigger on_fit_start transform fitting and persistence
    trainer.fit(wrapper, datamodule=dm)
    # Save and reload via Lightning checkpoint
    ckpt_path = tmp_path / "model_with_transforms.ckpt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    # Create shape_spec for loading
    from dlkit.core.shape_specs import create_shape_spec, ModelFamily, ShapeSource

    load_shape_spec = create_shape_spec(
        shapes={"x": (4,), "y": (4,)},
        model_family=ModelFamily.DLKIT_NN,
        source=ShapeSource.CONFIGURATION,
    )

    loaded = StandardLightningWrapper.load_from_checkpoint(
        str(ckpt_path),
        settings=WrapperComponentSettings(),
        model_settings=ModelComponentSettings(name=IdentityHead, module_path="tests.helpers"),
        shape_spec=load_shape_spec,
        entry_configs=entries,
        strict=False,
    )

    # Predict
    preds = trainer.predict(loaded, datamodule=dm)
    assert isinstance(preds, list) and len(preds) > 0
    batch_out = preds[0]
    assert isinstance(batch_out, dict)
    # Predictions are renamed to target keys by the naming step
    inv_pred = batch_out.get("predictions", {}).get("y")
    inv_targ = batch_out.get("targets", {}).get("y")
    assert inv_pred is not None and inv_targ is not None

    # Assert: inverse-transformed predictions and targets are in original space (strict)
    # Compare first batch predictions to raw target batch
    raw_batch = next(iter(dm.predict_dataloader()))
    raw_y = raw_batch["y"]
    assert torch.allclose(inv_targ, raw_y, atol=1e-6)
    assert torch.allclose(inv_pred, raw_y, atol=1e-6)

    # Assert: direct transforms applied exactly via wrapper’s chain
    assert loaded.model.last_x is not None
    x_in = loaded.model.last_x
    raw_x = next(iter(dm.predict_dataloader()))["x"]
    expected_x_in = loaded.feature_transforms({"x": raw_x})["x"]
    assert torch.allclose(x_in, expected_x_in, atol=1e-6, rtol=0)


def test_direct_inference_api_with_real_checkpoint(tmp_path: Path) -> None:
    """Test the high-level dlkit.infer() API using a real trained checkpoint."""
    import dlkit

    # Arrange: Set up and train a model (same as previous test)
    fx, fy, X, Y = _make_data(tmp_path)
    dm = _build_datamodule(fx, fy)
    entries = _entry_configs(fx, fy)
    wrapper = _build_wrapper(entries)
    trainer = _basic_trainer()

    # Train and save checkpoint
    trainer.fit(wrapper, datamodule=dm)
    ckpt_path = tmp_path / "model_for_direct_inference.ckpt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    # Prepare input data for direct inference
    # Use raw data (before transforms) as the new API should handle transforms internally
    test_inputs = {"x": torch.from_numpy(X[:8]).float()}  # First 8 samples

    # Act: Test the high-level dlkit.load_model() API
    with dlkit.load_model(
        checkpoint_path=ckpt_path, batch_size=4, device="cpu", apply_transforms=True
    ) as predictor:
        result = predictor.predict(test_inputs)

    # Assert: Verify the result structure and content
    assert hasattr(result, "predictions")
    assert isinstance(result.predictions, dict)

    # Check the actual structure - it might be nested
    if "predictions" in result.predictions:
        predictions = result.predictions["predictions"]
    elif "y" in result.predictions:
        predictions = result.predictions["y"]
    else:
        # Get the first available prediction
        predictions = next(iter(result.predictions.values()))

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[0] == 8  # Same number of samples as input
    assert predictions.shape[1] == 4  # Output dimension

    # Verify predictions are reasonable (not NaN, not all zeros)
    assert not torch.isnan(predictions).any()
    assert not torch.allclose(predictions, torch.zeros_like(predictions))

    # Test with different batch size
    with dlkit.load_model(
        checkpoint_path=ckpt_path,
        batch_size=16,  # Larger than input size
        device="cpu",
    ) as predictor:
        result_batch16 = predictor.predict(test_inputs)

    # Should get same predictions regardless of batch size
    if "predictions" in result.predictions:
        pred1 = result.predictions["predictions"]
    elif "y" in result.predictions:
        pred1 = result.predictions["y"]
    else:
        pred1 = next(iter(result.predictions.values()))

    if "predictions" in result_batch16.predictions:
        pred2 = result_batch16.predictions["predictions"]
    elif "y" in result_batch16.predictions:
        pred2 = result_batch16.predictions["y"]
    else:
        pred2 = next(iter(result_batch16.predictions.values()))

    assert torch.allclose(pred1, pred2, atol=1e-6)

    # Test with transforms disabled
    with dlkit.load_model(checkpoint_path=ckpt_path, apply_transforms=False) as predictor:
        result_no_transforms = predictor.predict(test_inputs)

    # Test with transforms disabled should succeed
    # Note: The actual difference may be minimal with identity transforms in this test
    assert result_no_transforms is not None
    assert result_no_transforms.predictions is not None


def test_predictor_returns_original_space_by_default(
    predictor_transform_setup: dict[str, Any],
) -> None:
    """load_model must apply transforms so outputs match raw targets."""
    raw_batch = predictor_transform_setup["raw_features"][:8].clone()
    expected = predictor_transform_setup["raw_targets"][:8]

    with dlkit.load_model(
        checkpoint_path=predictor_transform_setup["checkpoint"],
        batch_size=4,
        device="cpu",
        apply_transforms=True,
    ) as predictor:
        result = predictor.predict({"x": raw_batch})

    predictions = _extract_prediction_tensor(result)
    assert torch.allclose(predictions, expected, atol=1e-6)


def test_predictor_accepts_pretransformed_inputs_when_disabled(
    predictor_transform_setup: dict[str, Any],
) -> None:
    """Users can supply normalized tensors when apply_transforms=False."""
    normalized_batch = predictor_transform_setup["normalized_features"][:8].clone()

    with dlkit.load_model(
        checkpoint_path=predictor_transform_setup["checkpoint"],
        batch_size=4,
        device="cpu",
        apply_transforms=False,
    ) as predictor:
        result = predictor.predict({"x": normalized_batch})

    predictions = _extract_prediction_tensor(result)
    assert torch.allclose(predictions, normalized_batch, atol=1e-6)


def test_manual_inverse_matches_default_path(predictor_transform_setup: dict[str, Any]) -> None:
    """Manual transform handling should match builtin behavior."""
    raw_batch = predictor_transform_setup["raw_features"][:8].clone()
    normalized_batch = predictor_transform_setup["normalized_features"][:8].clone()
    wrapper: StandardLightningWrapper = predictor_transform_setup["wrapper"]

    with dlkit.load_model(
        checkpoint_path=predictor_transform_setup["checkpoint"],
        batch_size=4,
        device="cpu",
        apply_transforms=True,
    ) as predictor_default:
        default_result = predictor_default.predict({"x": raw_batch})

    default_predictions = _extract_prediction_tensor(default_result)

    with dlkit.load_model(
        checkpoint_path=predictor_transform_setup["checkpoint"],
        batch_size=4,
        device="cpu",
        apply_transforms=False,
    ) as predictor_manual:
        manual_result = predictor_manual.predict({"x": normalized_batch})

    manual_predictions = _extract_prediction_tensor(manual_result)
    manual_predictions_raw = wrapper.target_transforms_inverse({"y": manual_predictions})["y"]

    assert torch.allclose(manual_predictions_raw, default_predictions, atol=1e-6)


def test_transforms_persist_and_apply_with_torch_save(tmp_path: Path) -> None:
    # Arrange
    fx, fy, X, Y = _make_data(tmp_path)
    dm = _build_datamodule(fx, fy)
    entries = _entry_configs(fx, fy)
    wrapper = _build_wrapper(entries)
    trainer = _basic_trainer()
    # Fit once so transforms are fitted and stored in ModuleDict
    trainer.fit(wrapper, datamodule=dm)

    # Save state_dict via torch.save and reload into a fresh wrapper instance
    sd_path = tmp_path / "wrapper_state.pth"
    torch.save(wrapper.state_dict(), sd_path)
    assert sd_path.exists()

    rewrapped = _build_wrapper(entries)
    state = torch.load(sd_path, map_location="cpu")
    rewrapped.load_state_dict(state, strict=False)

    # Predict
    preds = trainer.predict(rewrapped, datamodule=dm)
    assert isinstance(preds, list) and len(preds) > 0
    batch_out = preds[0]
    # Predictions are renamed to target keys by the naming step
    inv_pred = batch_out.get("predictions", {}).get("y")
    inv_targ = batch_out.get("targets", {}).get("y")
    assert inv_pred is not None and inv_targ is not None

    raw_batch = next(iter(dm.predict_dataloader()))
    raw_y = raw_batch["y"]
    assert torch.allclose(inv_targ, raw_y, atol=1e-6)
    assert torch.allclose(inv_pred, raw_y, atol=1e-6)

    # Verify direct transform applied exactly via wrapper’s chain
    assert rewrapped.model.last_x is not None
    x_in = rewrapped.model.last_x
    raw_x = next(iter(dm.predict_dataloader()))["x"]
    expected_x_in = rewrapped.feature_transforms({"x": raw_x})["x"]
    assert torch.allclose(x_in, expected_x_in, atol=1e-6, rtol=0)
