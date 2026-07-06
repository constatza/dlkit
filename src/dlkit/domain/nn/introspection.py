"""Recover a model's effective (post-default-resolution) hyperparameters."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from dlkit.domain.nn.contracts import HyperParam, HyperparameterAware


def effective_hyperparameters(
    model: nn.Module, overrides: Mapping[str, HyperParam]
) -> dict[str, HyperParam]:
    """Merge settings-supplied overrides with the model's own declared hyperparameters.

    Model-declared values win: they reflect what the model actually built with,
    whereas ``overrides`` may be silently absent for any field the user left
    unset in settings.

    Args:
        model: The constructed domain nn.Module.
        overrides: Non-structural settings fields supplied/forwarded for this build.

    Returns:
        Flat dict of hyperparameter name -> value, suitable for ``run_context.log_params``.
    """
    declared = model.hyperparameters if isinstance(model, HyperparameterAware) else {}
    return {**overrides, **declared}


__all__ = ["effective_hyperparameters"]
