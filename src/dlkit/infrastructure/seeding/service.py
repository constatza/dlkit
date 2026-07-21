"""Centralized global-RNG seeding for DLKit.

This module is the single place in DLKit that calls Lightning's
``seed_everything`` directly — every other call site seeds indirectly through
this function (or through ``infrastructure.config.run_settings.apply_run_context``,
which wraps it).
"""

from __future__ import annotations


def apply_global_seed(seed: int, *, workers: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch global RNG state.

    Thin wrapper around ``lightning.pytorch.seed_everything``, kept as the
    sole call site for that function so seeding behavior stays centralized
    and swappable in one place.

    Args:
        seed: Seed value to apply to all global RNGs.
        workers: Whether to also seed dataloader worker processes via
            Lightning's ``WORKER_SPECIFIC_INFO`` environment mechanism.
    """
    from lightning.pytorch import seed_everything

    seed_everything(seed, workers=workers)
