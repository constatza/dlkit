"""Step logging concern for ProcessingLightningWrapper.

Encapsulates metric logging to Lightning with proper trainer availability checks.
"""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor

from dlkit.common.metric_stages import STAGE_ALIASES, MetricStage, metric_key


def _format_metric_name(stage: str, name: str) -> str:
    """Normalize metric names per stage conventions (pure function).

    Args:
        stage: Stage identifier ('train', 'val', 'test', etc.).
        name: Raw metric name.

    Returns:
        Normalized metric name string.
    """
    stage_lower = stage.lower()
    name_lower = name.lower()

    # `MetricStage` is a `StrEnum`, so a plain lowercase string key (e.g.
    # "val") looks up correctly at runtime even for stages not in the enum
    # (falling back to the default); the cast only tells the type checker
    # this is a valid dict key, it has no runtime effect.
    for alias in STAGE_ALIASES.get(cast(MetricStage, stage_lower), (stage_lower,)):
        if name_lower.startswith(alias):
            return name

    return metric_key(stage_lower, name)


class LightningStepLogger:
    """Delegates to LightningModule.log() and log_dict() methods.

    Checks trainer availability cleanly using _can_log() without try/except blocks.
    """

    def __init__(self, module: Any) -> None:
        """Initialize with a reference to the LightningModule.

        Args:
            module: The LightningModule instance.
        """
        self._module = module

    def _can_log(self) -> bool:
        """Check if trainer is available for logging.

        Returns:
            True if trainer is attached and not None.
        """
        try:
            return getattr(self._module, "trainer", None) is not None
        except RuntimeError:
            # LightningModule raises RuntimeError if trainer not attached
            return False

    def _log(self, name: str, value: Any, **kwargs: Any) -> None:
        """Log a single value if trainer is available.

        Args:
            name: Metric name.
            value: Metric value.
            **kwargs: Additional arguments to pass to self.log().
        """
        if self._can_log():
            self._module.log(name, value, **kwargs)

    def _log_dict(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        """Log a dict of metrics if trainer is available.

        Args:
            metrics: Dict mapping metric names to values.
            **kwargs: Additional arguments to pass to self.log_dict().
        """
        if self._can_log():
            self._module.log_dict(metrics, **kwargs)

    def log_stage_outputs(
        self,
        stage: str,
        loss: Tensor | None,
        metrics: dict[str, Any] | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Log loss and metrics for a stage.

        Args:
            stage: Stage identifier.
            loss: Scalar loss tensor (optional).
            metrics: Additional metrics dict (optional).
            batch_size: Batch size for epoch-level weighted averaging.
        """
        if loss is not None:
            self._log(
                _format_metric_name(stage, "loss"),
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        if metrics:
            formatted = {_format_metric_name(stage, k): v for k, v in metrics.items()}
            self._log_dict(
                formatted,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

    def log_lr(self, lr: float) -> None:
        """Log current learning rate.

        Args:
            lr: Learning rate value.
        """
        self._log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)
