"""FlowMatchingWrapper — Level 3 generative wrapper.

Specialises ``ContinuousFlowWrapper`` with flow matching stochastic supervision
via ``FlowMatchingSupervisionBuilder`` injected as ``batch_transforms``.

Loss computation is delegated to the ``RoutedLossComputer`` configured in the
build factory (default: MSE against ``batch["targets"]["ut"]``), allowing the
loss function to be configured via ``TRAINING.loss_function`` in TOML just like
any other workflow.

Inference (ODE integration from Gaussian noise) is inherited from
``ContinuousFlowWrapper``.
"""

from __future__ import annotations

from typing import Any

from dlkit.runtime.adapters.lightning.continuous_flow import ContinuousFlowWrapper


class FlowMatchingWrapper(ContinuousFlowWrapper):
    """Flow matching algorithm on top of the continuous-flow base.

    Adds stochastic supervision ``(xt, t, ut)`` via ``FlowMatchingSupervisionBuilder``
    injected as ``batch_transforms``.  Loss is routed through the injected
    ``RoutedLossComputer`` — configurable via ``TRAINING.loss_function``.

    Inference (ODE integration from x0 ~ N(0, I)) is inherited from
    ``ContinuousFlowWrapper`` via ``ODEPredictionStrategy``.

    Args:
        supervision_builder: ``IBatchTransform`` that computes ``(xt, t, ut)``
            from ``x1`` data.  Passed as ``batch_transforms=[supervision_builder]``
            to the base class.
        **kwargs: Forwarded to ``ContinuousFlowWrapper.__init__``.
    """

    def __init__(
        self,
        *,
        supervision_builder: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
            batch_transforms=[supervision_builder],
        )
