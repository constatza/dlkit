"""Public re-export of the evaluation workflow entrypoint.

The implementation lives in ``dlkit.engine.workflows.entrypoints.evaluate``,
as a sibling of ``train``/``optimize``/``converge`` — see that module for
details. This package keeps the public import path
(``dlkit.interfaces.inference.evaluate``) stable.
"""

from __future__ import annotations

from dlkit.engine.workflows.entrypoints.evaluate import evaluate

__all__ = ["evaluate"]
