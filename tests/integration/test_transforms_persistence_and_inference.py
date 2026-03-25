from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from torch import Tensor

import dlkit
from dlkit.core.datamodules.array import InMemoryModule
from dlkit.core.datasets.flexible import FlexibleDataset
from dlkit.core.datatypes.split import IndexSplit
from dlkit.core.models.nn.ffnn.simple import ConstantWidthFFNN
from dlkit.core.models.wrappers.functions import apply_inverse_chain
from dlkit.core.models.wrappers.standard import StandardLightningWrapper
from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.transform_settings import TransformSettings

MODEL_MODULE_PATH = "dlkit.core.models.nn.ffnn.simple"
MODEL_NAME = "ConstantWidthFFNN"


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
    train = tuple(range(n))
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


def _entry_configs(fx: Path, fy: Path) -> tuple[Feature | Target, ...]:
    # Apply MinMaxScaler to both x and y; direct for features, inverse for targets at predict
    ts = TransformSettings(
        name="MinMaxScaler", module_path="dlkit.core.training.transforms.minmax", dim=0
    )
    x = Feature(name="x", path=fx, transforms=[ts])
    y = Target(name="y", path=fy, transforms=[ts])
    return (x, y)


def _build_wrapper(entry_cfgs: tuple[Feature | Target, ...]) -> StandardLightningWrapper:
    from dlkit.core.shape_specs.simple_inference import ShapeSummary

    model_settings = ModelComponentSettings(
        name=MODEL_NAME,
        module_path=MODEL_MODULE_PATH,
        hidden_size=4,
        num_layers=1,
    )
    wrapper_settings = WrapperComponentSettings()

    # x/y shapes provided by FlexibleDataset are 1D (feature dimension = 4)
    shape_summary = ShapeSummary(in_shapes=((4,),), out_shapes=((4,),))

    wrapper = StandardLightningWrapper(
        settings=wrapper_settings,
        model_settings=model_settings,
        shape_summary=shape_summary,
        entry_configs=entry_cfgs,
    )
    _configure_identity_ffnn(wrapper.model)
    return wrapper


def _configure_identity_ffnn(model: ConstantWidthFFNN) -> None:
    with torch.no_grad():
        model.embedding_layer.weight.copy_(torch.eye(4))
        model.embedding_layer.bias.zero_()
        model.regression_layer.weight.copy_(torch.eye(4))
        model.regression_layer.bias.zero_()


def _capture_forward_input(model: torch.nn.Module) -> tuple[dict[str, Tensor], Any]:
    state: dict[str, Tensor] = {}

    def _hook(
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if args:
            tensor = args[0]
        else:
            tensor = kwargs["x"]
        state["last_x"] = tensor.detach().clone()

    handle = model.register_forward_pre_hook(_hook, with_kwargs=True)
    return state, handle


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
    """Retrieve the first tensor prediction from inference output.

    Handles the new predict() API: returns Tensor (single output) or
    tuple[Tensor, ...] (multi-output, first element is the prediction).
    """
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor):
        return result[0]
    raise AssertionError(f"Expected Tensor or tuple[Tensor,...] but got {type(result)}: {result}")


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
    normalized_features = wrapper._batch_transformer._feature_chains["x"](raw_features)

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

    from dlkit.core.shape_specs.simple_inference import ShapeSummary

    loaded = StandardLightningWrapper.load_from_checkpoint(
        str(ckpt_path),
        settings=WrapperComponentSettings(),
        model_settings=ModelComponentSettings(
            name=MODEL_NAME,
            module_path=MODEL_MODULE_PATH,
            hidden_size=4,
            num_layers=1,
        ),
        shape_summary=ShapeSummary(in_shapes=((4,),), out_shapes=((4,),)),
        entry_configs=entries,
        strict=False,
    )
    captured, hook = _capture_forward_input(loaded.model)

    # Predict
    try:
        preds = trainer.predict(loaded, datamodule=dm)
    finally:
        hook.remove()
    assert isinstance(preds, list) and len(preds) > 0
    batch_out = preds[0]
    # Standard format: TensorDict with 'predictions' and 'targets' (and optional 'latents')
    from tensordict import TensorDict as _TensorDict

    assert isinstance(batch_out, _TensorDict)
    inv_pred = batch_out["predictions"]
    inv_targ = batch_out["targets"]["y"]
    assert inv_pred is not None and inv_targ is not None

    # Assert: inverse-transformed predictions and targets are in original space (strict)
    raw_batch = next(iter(dm.predict_dataloader()))
    raw_y = raw_batch["targets", "y"]
    assert torch.allclose(inv_targ, raw_y, atol=1e-6)
    assert torch.allclose(inv_pred, raw_y, atol=1e-6)

    # Assert: direct transforms applied exactly via wrapper's named chain
    assert "last_x" in captured
    x_in = captured["last_x"]
    raw_x = next(iter(dm.predict_dataloader()))["features", "x"]
    expected_x_in = loaded._batch_transformer._feature_chains["x"](raw_x)
    assert torch.allclose(x_in, expected_x_in, atol=1e-6, rtol=0)


def test_direct_inference_api_with_real_checkpoint(tmp_path: Path) -> None:
    """Test the high-level dlkit.load_model() API using a real trained checkpoint."""
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

    # predict() mirrors model.forward() — pass tensors as keyword args
    x_tensor = torch.from_numpy(X[:8]).float()

    # Act: Test the high-level dlkit.load_model() API
    with dlkit.load_model(
        checkpoint_path=ckpt_path, batch_size=4, device="cpu", apply_transforms=True
    ) as predictor:
        predictions = predictor.predict(x=x_tensor)

    # predict() returns Tensor for single-output models
    predictions = _extract_prediction_tensor(predictions)
    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[0] == 8  # Same number of samples as input
    assert predictions.shape[1] == 4  # Output dimension (identity model: in == out)

    # Verify predictions are reasonable (not NaN, not all zeros)
    assert not torch.isnan(predictions).any()
    assert not torch.allclose(predictions, torch.zeros_like(predictions))

    # Test with different batch size — should get same predictions
    with dlkit.load_model(
        checkpoint_path=ckpt_path, batch_size=16, device="cpu", apply_transforms=True
    ) as predictor:
        predictions_b16 = predictor.predict(x=x_tensor)

    predictions_b16 = _extract_prediction_tensor(predictions_b16)
    assert torch.allclose(predictions, predictions_b16, atol=1e-6)

    # Test with transforms disabled
    with dlkit.load_model(checkpoint_path=ckpt_path, apply_transforms=False) as predictor:
        result_no_transforms = predictor.predict(x=x_tensor)

    assert result_no_transforms is not None


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
        result = predictor.predict(x=raw_batch)

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
        result = predictor.predict(x=normalized_batch)

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
        default_result = predictor_default.predict(x=raw_batch)

    default_predictions = _extract_prediction_tensor(default_result)

    with dlkit.load_model(
        checkpoint_path=predictor_transform_setup["checkpoint"],
        batch_size=4,
        device="cpu",
        apply_transforms=False,
    ) as predictor_manual:
        manual_result = predictor_manual.predict(x=normalized_batch)

    manual_predictions = _extract_prediction_tensor(manual_result)
    manual_predictions_raw = apply_inverse_chain(
        manual_predictions, wrapper._batch_transformer._target_chains["y"]
    )

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
    captured, hook = _capture_forward_input(rewrapped.model)
    try:
        preds = trainer.predict(rewrapped, datamodule=dm)
    finally:
        hook.remove()
    assert isinstance(preds, list) and len(preds) > 0
    batch_out = preds[0]
    # Standard format: TensorDict with 'predictions' and 'targets' (and optional 'latents')
    from tensordict import TensorDict as _TensorDict

    assert isinstance(batch_out, _TensorDict)
    inv_pred = batch_out["predictions"]
    inv_targ = batch_out["targets"]["y"]
    assert inv_pred is not None and inv_targ is not None

    raw_batch = next(iter(dm.predict_dataloader()))
    raw_y = raw_batch["targets", "y"]
    assert torch.allclose(inv_targ, raw_y, atol=1e-6)
    assert torch.allclose(inv_pred, raw_y, atol=1e-6)

    # Verify direct transform applied exactly via wrapper’s named chain
    assert "last_x" in captured
    x_in = captured["last_x"]
    raw_x = next(iter(dm.predict_dataloader()))["features", "x"]
    expected_x_in = rewrapped._batch_transformer._feature_chains["x"](raw_x)
    assert torch.allclose(x_in, expected_x_in, atol=1e-6, rtol=0)
