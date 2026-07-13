"""Batched prediction orchestration over a predict dataloader.

Runtime orchestration only: no console output, no progress reporting. Callers
(e.g. the CLI) own presentation concerns and simply call
``run_batched_prediction`` for the batch-iterate/predict/concatenate work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lightning.pytorch import LightningDataModule

    from .predictor import CheckpointPredictor


def run_batched_prediction(
    predictor: CheckpointPredictor,
    datamodule: LightningDataModule | None,
) -> torch.Tensor | None:
    """Run inference over every batch of a predict dataloader and concatenate results.

    For each batch, dispatches named feature tensors to ``predictor.predict()``
    as keyword arguments using ``predictor.feature_names`` (restored from
    checkpoint metadata) to select and order them. When ``feature_names`` is
    empty (e.g. legacy checkpoints), falls back to passing every feature key
    present in the batch.

    Args:
        predictor: Loaded predictor used to run inference on each batch.
        datamodule: Datamodule providing a ``predict_dataloader()`` once set up
            for the ``"predict"`` stage. When ``None`` (no data section
            configured), no batches are processed.

    Returns:
        Prediction tensors concatenated along the batch dimension, or ``None``
        when no batches produced a tensor prediction.
    """
    if datamodule is None:
        return None

    datamodule.setup("predict")
    loader = datamodule.predict_dataloader()

    feature_names = predictor.feature_names
    all_predictions: list[torch.Tensor] = []
    for batch in loader:
        features_td = batch["features"]
        if feature_names:
            feature_kwargs = {
                name: features_td[name] for name in feature_names if name in features_td.keys()
            }
        else:
            feature_kwargs = {k: features_td[k] for k in features_td.keys()}

        output = predictor.predict(**feature_kwargs)
        prediction = output.predictions
        if isinstance(prediction, torch.Tensor):
            all_predictions.append(prediction)

    return torch.cat(all_predictions, dim=0) if all_predictions else None
