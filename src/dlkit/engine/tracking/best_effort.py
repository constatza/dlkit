"""Shared decorator for best-effort tracking operations that must never
raise into a training/optimization run."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from dlkit.infrastructure.config.environment import best_effort_retry_budget
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


def best_effort(action: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Wrap a tracking function so failures are logged, not raised.

    Runs under a scoped, fail-fast MLflow HTTP retry budget (see
    ``best_effort_retry_budget``) since a call that's allowed to fail
    shouldn't burn the retry budget sized for calls that must not.

    Args:
        action: Short present-tense description used in the warning on
            failure (e.g. "log study metadata").
    """

    def decorator(func: Callable[..., None]) -> Callable[..., None]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            try:
                with best_effort_retry_budget():
                    func(*args, **kwargs)
            except Exception as e:
                logger.opt(exception=True).warning("Failed to {}: {}", action, e)

        return wrapper

    return decorator
