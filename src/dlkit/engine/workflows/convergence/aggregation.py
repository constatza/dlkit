"""Pure aggregation functions for convergence study results."""

from __future__ import annotations

import statistics
from collections import defaultdict

from dlkit.common.metric_stages import MetricStage, swap_stage
from dlkit.common.results import ConvergencePoint, TrainingResult
from dlkit.infrastructure.config.convergence_settings import ConvergenceSettings


def aggregate_results(
    results: tuple[TrainingResult, ...],
    sizes: tuple[int, ...],
    cfg: ConvergenceSettings,
) -> tuple[ConvergencePoint, ...]:
    """Aggregate TrainingResults into ConvergencePoints, one per sample size.

    Results must be ordered as: n0r0, n0r1, ..., n1r0, n1r1, ...

    Args:
        results: TrainingResults in sweep order (n outer, repeat inner).
        sizes: Ordered sample sizes used to generate the sweep.
        cfg: Convergence settings (repeats, target_metric, target, c).

    Returns:
        One ConvergencePoint per size, in size order.
    """
    repeats = cfg.repeats
    grouped: dict[int, list[TrainingResult]] = defaultdict(list)
    for idx, result in enumerate(results):
        size_idx = idx // repeats
        grouped[size_idx].append(result)

    points: list[ConvergencePoint] = []
    prev_val_mean: float | None = None
    train_metric = swap_stage(cfg.target_metric, MetricStage.VAL, MetricStage.TRAIN)

    for i, n in enumerate(sizes):
        group = grouped[i]
        train_vals = [float(r.metrics.get(train_metric, 0.0)) for r in group]
        val_vals = [float(r.metrics.get(cfg.target_metric, 0.0)) for r in group]

        train_mean = statistics.mean(train_vals)
        train_std = statistics.stdev(train_vals) if len(train_vals) > 1 else 0.0
        val_mean = statistics.mean(val_vals)
        val_std = statistics.stdev(val_vals) if len(val_vals) > 1 else 0.0
        gap = val_mean - train_mean
        marginal_gain = (prev_val_mean - val_mean) if prev_val_mean is not None else None

        converged: bool | None = None
        if cfg.target is not None:
            converged = (val_mean + cfg.c * val_std) <= cfg.target

        points.append(
            ConvergencePoint(
                n=n,
                train_mean=train_mean,
                train_std=train_std,
                val_mean=val_mean,
                val_std=val_std,
                gap=gap,
                marginal_gain=marginal_gain,
                converged=converged,
            )
        )
        prev_val_mean = val_mean

    return tuple(points)


def find_n_star(points: tuple[ConvergencePoint, ...]) -> int | None:
    """Return smallest n where converged=True, or None.

    Args:
        points: Sequence of convergence points.

    Returns:
        Smallest n meeting the convergence criterion, or None.
    """
    for point in points:
        if point.converged is True:
            return point.n
    return None


def build_summary_dict(
    points: tuple[ConvergencePoint, ...],
    n_star: int | None,
) -> dict[str, object]:
    """Serialise convergence results to a dict suitable for TOML export.

    Args:
        points: Convergence points to serialise.
        n_star: Identified n* value, or None.

    Returns:
        Dict with 'n_star' and 'points' keys.
    """
    return {
        "n_star": n_star,
        "points": [
            {
                "n": p.n,
                "train_mean": p.train_mean,
                "train_std": p.train_std,
                "val_mean": p.val_mean,
                "val_std": p.val_std,
                "gap": p.gap,
                "marginal_gain": p.marginal_gain,
                "converged": p.converged,
            }
            for p in points
        ],
    }
