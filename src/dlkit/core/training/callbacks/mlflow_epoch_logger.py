"""Callback for logging metrics to MLflow using epoch numbers instead of steps."""

from __future__ import annotations

from typing import Any, Callable

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger


class MLflowEpochLogger(Callback):
    """Callback that logs metrics to MLflow using epoch numbers as the x-axis."""

    def __init__(self, run_context: Any):
        super().__init__()
        self.run_context = run_context

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_metrics(trainer, stage="train")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_metrics(trainer, stage="val")

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_metrics(trainer, stage="test")

    def _log_metrics(self, trainer: Trainer, stage: str) -> None:
        if not self.run_context or getattr(trainer, "sanity_checking", False):
            return

        try:
            metrics = getattr(trainer, "callback_metrics", None) or {}
            prepared = self._collect_stage_metrics(metrics, stage)

            if not prepared:
                return

            step_resolver: Callable[[Trainer, str], int] = getattr(
                self, "_resolve_step", self._default_step_resolver
            )
            step = step_resolver(trainer, stage)

            self.run_context.log_metrics(prepared, step=step)
            logger.debug(
                "Logged {} metrics for stage '{}' at step {}",
                len(prepared),
                stage,
                step,
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to log metrics with epoch logger: {exc}")

    @staticmethod
    def _default_step_resolver(trainer: Trainer, stage: str) -> int:  # pragma: no cover - simple
        return getattr(trainer, "current_epoch", 0)

    def _collect_stage_metrics(self, metrics: dict[str, Any], stage: str) -> dict[str, float]:
        stage_aliases = {
            "train": ("train", "training"),
            "val": ("val", "valid", "validation"),
            "test": ("test", "testing"),
        }

        prefixes = stage_aliases.get(stage, (stage,))
        collected: dict[str, float] = {}

        for key, raw_value in metrics.items():
            normalized_key = key.lower()

            # Learning rate is logged only for training stage
            if normalized_key == "lr" and stage != "train":
                continue

            metric_key = None
            if self._matches_stage_prefix(key, prefixes):
                metric_key = key
            else:
                # Skip metrics clearly tied to a different stage
                if normalized_key.startswith("train") and stage != "train":
                    continue
                if normalized_key.startswith("val") and stage != "val":
                    continue
                if normalized_key.startswith("test") and stage != "test":
                    continue

                metric_key = self._format_metric_key(stage, key)

            numeric = self._to_numeric(raw_value)
            if numeric is None:
                logger.debug(f"Skipping non-numeric metric: {key}")
                continue

            collected[metric_key] = numeric

        return collected

    @staticmethod
    def _format_metric_key(stage: str, key: str) -> str:
        stage_lower = stage.lower()
        if stage_lower == "test":
            suffix = " test"
            if key.lower().endswith(suffix):
                return key
            return f"{key}{suffix}"
        return key

    @staticmethod
    def _matches_stage_prefix(metric_key: str, prefixes: tuple[str, ...]) -> bool:
        normalized = metric_key.lower()
        for prefix in prefixes:
            prefix_lower = prefix.lower()
            if normalized.startswith(prefix_lower):
                return True
            if normalized.startswith(f"{prefix_lower}/"):
                return True
            if normalized.startswith(f"{prefix_lower}_"):
                return True
            if normalized.startswith(f"{prefix_lower}."):
                return True
        return False

    @staticmethod
    def _to_numeric(value: Any) -> float | None:
        candidate = value

        for attr in ("detach",):
            if hasattr(candidate, attr):
                try:
                    candidate = getattr(candidate, attr)()
                except Exception:  # pragma: no cover - defensive branch
                    return None

        if hasattr(candidate, "cpu"):
            try:
                candidate = candidate.cpu()
            except Exception:  # pragma: no cover - defensive branch
                return None

        if hasattr(candidate, "item") and callable(candidate.item):
            try:
                candidate = candidate.item()
            except Exception:  # pragma: no cover - defensive branch
                return None

        if isinstance(candidate, (int, float)):
            return float(candidate)

        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None
