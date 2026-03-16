"""GenerativeLightningWrapper — marker base class for all generative models.

This class exists purely as a named checkpoint in the wrapper hierarchy.
It signals "this is a generative model" for:

- ``isinstance`` checks in factory routing
- ``issubclass`` guards in ``WrapperFactory`` / ``BuildFactory``
- Documentation and type annotations

No method overrides — all behaviour is injected via protocols in the base class.
Subclasses specialise supervision (Level 3) or inference strategy (Level 2).
"""

from dlkit.core.models.wrappers.base import ProcessingLightningWrapper


class GenerativeLightningWrapper(ProcessingLightningWrapper):
    """Marker base for any generative Lightning wrapper.

    Guarantees:
    - ``batch_transforms`` is non-empty (coupled supervision injected).
    - ``prediction_strategy`` is a generative strategy (e.g. ODE-based).

    Subclasses specialise supervision signal and loss (Level 3) or inference
    strategy (Level 2) without changing the training loop in the base class.

    Use ``isinstance(wrapper, GenerativeLightningWrapper)`` to branch on
    generative vs. discriminative behaviour in factory code.
    """

    # No overrides — marker class only.
    pass
