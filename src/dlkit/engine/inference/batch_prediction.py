"""Batched prediction/evaluation orchestration over a datamodule.

Runtime orchestration only: no console output, no progress reporting. Callers
(e.g. the CLI) own presentation concerns and simply call
``run_batched_prediction``/``run_batched_evaluation`` for the
batch-iterate/predict/concatenate work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .predictor import _move_tensor_to_device

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lightning.pytorch import LightningDataModule

    from .predictor import CheckpointPredictor


def _dispatch_feature_kwargs(
    predictor: CheckpointPredictor,
    features_td: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Select and order a batch's feature tensors for ``predictor.predict()``.

    Uses ``predictor.feature_names`` (restored from checkpoint metadata) to
    select and order feature kwargs.

    Args:
        predictor: Loaded predictor whose ``feature_names`` drives dispatch.
        features_td: Batch's ``"features"`` entry, keyed by feature name.

    Returns:
        Keyword arguments ready to pass to ``predictor.predict(**kwargs)``.

    Raises:
        ValueError: If ``predictor.feature_names`` is empty — a legacy/
            positional-mode checkpoint has no named contract to dispatch
            batch keys against, and guessing which batch keys to pass would
            silently risk passing the wrong tensor to the wrong kwarg.
    """
    feature_names = predictor.feature_names
    if not feature_names:
        raise ValueError(
            "run_batched_prediction()/run_batched_evaluation() require "
            "predictor.feature_names to be populated from checkpoint metadata "
            "(legacy/positional-mode checkpoints are not supported by batched "
            "prediction). Re-export the checkpoint with named feature entries."
        )
    return {name: features_td[name] for name in feature_names if name in features_td.keys()}


def run_batched_prediction(
    predictor: CheckpointPredictor,
    datamodule: LightningDataModule | None,
) -> torch.Tensor | None:
    """Run inference over every batch of a predict dataloader and concatenate results.

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

    all_predictions: list[torch.Tensor] = []
    for batch in loader:
        feature_kwargs = _dispatch_feature_kwargs(predictor, batch["features"])
        output = predictor.predict(**feature_kwargs)
        prediction = output.predictions
        if isinstance(prediction, torch.Tensor):
            all_predictions.append(prediction)

    return torch.cat(all_predictions, dim=0) if all_predictions else None


def run_batched_evaluation(
    predictor: CheckpointPredictor,
    datamodule: LightningDataModule,
    split: Literal["test", "predict"] = "test",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference over a labeled split, collecting predictions and targets.

    Unlike ``run_batched_prediction``, this always resolves a labeled split —
    ``test_dataloader()`` by default. ``predict_dataloader()`` is only used
    when ``split="predict"`` is explicitly requested (e.g. a datamodule whose
    predict partition also carries ground-truth targets); it must never be
    assumed to alias ``test_dataloader()``, since some datamodules
    (``GraphDataModule``) give it a genuinely separate partition.

    Args:
        predictor: Loaded predictor used to run inference on each batch.
        datamodule: Datamodule providing the requested split's dataloader.
        split: Which labeled split to evaluate against.

    Returns:
        ``(predictions, targets)`` tensors concatenated along the batch
        dimension.

    Raises:
        ValueError: ``predictor.predict_target_key`` is unset, the target key
            is absent from a batch's targets, or the split produced no
            batches.
    """
    target_key = predictor.predict_target_key
    if not target_key:
        raise ValueError(
            "run_batched_evaluation() requires predictor.predict_target_key to be set "
            "(restored from checkpoint metadata) to select the matching target entry."
        )

    datamodule.setup(split)
    loader = datamodule.test_dataloader() if split == "test" else datamodule.predict_dataloader()

    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for batch in loader:
        feature_kwargs = _dispatch_feature_kwargs(predictor, batch["features"])
        output = predictor.predict(**feature_kwargs)
        prediction = output.predictions
        if not isinstance(prediction, torch.Tensor):
            continue

        targets_td = batch["targets"]
        if target_key not in targets_td.keys():
            raise ValueError(
                f"Target entry '{target_key}' not found in batch targets "
                f"(available: {tuple(targets_td.keys())}). Ensure settings.data.targets "
                "includes the entry matching predict_target_key."
            )
        all_predictions.append(prediction)
        target = targets_td[target_key]
        all_targets.append(_move_tensor_to_device(target, prediction.device))

    if not all_predictions:
        raise ValueError(
            f"run_batched_evaluation(): the '{split}' split produced no batches. "
            "Check that settings.data.targets is configured and the split is non-empty."
        )

    return torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0)
