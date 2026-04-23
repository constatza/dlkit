"""Metrics routing: maps batch keys to per-metric update calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from tensordict import TensorDict
from torch import Tensor
from torchmetrics import Metric

from dlkit.infrastructure.config.model_components import MetricInputRef

from .batch_namespace import _parse_key


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricRoute:
    """Value object carrying per-metric routing configuration.

    Attributes:
        metric: The torchmetrics Metric to update.
        target_ns: Target namespace ('targets').
        target_name: Target entry name.
        extra_inputs: Extra input refs for additional kwargs.
    """

    metric: Metric
    target_ns: str
    target_name: str
    extra_inputs: tuple[MetricInputRef, ...]


class RoutedMetricsUpdater:
    """Routes each metric to its configured target key and extra inputs.

    Does NOT use MetricCollection.update() because that broadcasts the same
    target to all metrics — per-metric target routing requires individual calls.

    Attributes:
        _routes: Dict mapping stage ('val', 'test') to list of MetricRoute.
        _parsed_extra: Pre-parsed extra input routes for fast lookup during update.
    """

    def __init__(
        self,
        val_routes: list[MetricRoute],
        test_routes: list[MetricRoute],
    ) -> None:
        """Initialize with per-stage metric routes.

        Args:
            val_routes: Metric routes for validation stage.
            test_routes: Metric routes for test stage.
        """
        self._routes: dict[str, list[MetricRoute]] = {
            "val": val_routes,
            "test": test_routes,
        }
        self._parsed_extra: dict[
            str, list[tuple[Metric, str, str, list[tuple[str, tuple[str, str]]]]]
        ] = {
            stage: [
                (
                    r.metric,
                    r.target_ns,
                    r.target_name,
                    [(ref.arg, _parse_key(ref.key)) for ref in r.extra_inputs],
                )
                for r in routes
            ]
            for stage, routes in self._routes.items()
        }

    def update(self, predictions: Tensor, batch: TensorDict, stage: str) -> None:
        """Update metrics for the given stage.

        Args:
            predictions: Model output tensor.
            batch: TensorDict containing features and targets.
            stage: Stage identifier ('val' or 'test').
        """
        for metric, target_ns, target_name, extra_routes in self._parsed_extra.get(stage, []):
            target = batch[target_ns, target_name].to(dtype=predictions.dtype)
            extra = {kwarg: batch[route] for kwarg, route in extra_routes}
            cast(Any, metric).update(predictions, target, **extra)

    def compute(self, stage: str) -> dict[str, float | Tensor]:
        """Compute accumulated metric values for the given stage.

        Args:
            stage: Stage identifier ('val' or 'test').

        Returns:
            Dictionary mapping metric class names to computed values.
        """
        return {
            type(route.metric).__name__: cast(Any, route.metric).compute()
            for route in self._routes.get(stage, [])
        }

    def reset(self, stage: str) -> None:
        """Reset metric state for the given stage.

        Args:
            stage: Stage identifier ('val' or 'test').
        """
        for route in self._routes.get(stage, []):
            route.metric.reset()
