"""Prediction key naming strategies.

This module provides single-responsibility strategies focused on naming of
prediction keys after outputs have been classified. These strategies do not
decide whether an output is a prediction or a latent — they only determine
what keys predictions should use, typically aiming to align with target names.
"""

from __future__ import annotations

from abc import ABC

import torch

from .interfaces import OutputNamer


class IdentityNamer(OutputNamer, ABC):
    """Return predictions unchanged.

    Useful as a safe default or when explicit naming is handled elsewhere.
    """

    def rename_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        model_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return dict(predictions)


class TargetNameByShapeNamer(OutputNamer):
    """Rename prediction keys to target names using shape matching.

    Rules (KISS):
    - If there is exactly one target and one prediction and their shapes match,
      rename the prediction key to the single target name.
    - For multiple targets/predictions, map any prediction whose shape exactly
      matches a target's shape to that target's name.
    - If no clear mapping exists, keep the original prediction name.

    Notes:
    - This strategy only handles naming; it assumes predictions were already
      classified. It does not reclassify or drop entries.
    - Exact shape equality is used for determinism and simplicity. If looser
      matching is desired, introduce a separate strategy rather than adding
      tolerance here to keep responsibilities clear.
    """

    def rename_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        *,
        model_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if not predictions or not targets:
            return dict(predictions)

        # Single-target, single-prediction fast path
        if len(predictions) == 1 and len(targets) == 1:
            p_name, p_tensor = next(iter(predictions.items()))
            t_name, t_tensor = next(iter(targets.items()))
            if tuple(p_tensor.shape) == tuple(t_tensor.shape):
                return {t_name: p_tensor}
            return dict(predictions)

        # Build map of shapes to target names (allow duplicates by keeping first)
        target_by_shape: dict[tuple[int, ...], str] = {}
        for name, t in targets.items():
            shp = tuple(t.shape)
            # Don't overwrite an existing mapping to keep behavior stable
            target_by_shape.setdefault(shp, name)

        renamed: dict[str, torch.Tensor] = {}
        used_target_names: set[str] = set()

        for p_name, p_tensor in predictions.items():
            shp = tuple(p_tensor.shape)
            t_name = target_by_shape.get(shp)
            if t_name is not None and t_name not in used_target_names:
                renamed[t_name] = p_tensor
                used_target_names.add(t_name)
            else:
                # Fallback: keep original name
                renamed[p_name] = p_tensor

        return renamed
