"""Flow matching supervision builder — IBatchTransform implementation.

Transforms a raw data batch by computing stochastic interpolations
(xt, t) and velocity targets (ut) for flow matching training.

The builder reads the target data ``x1`` from the batch features, samples
noise ``x0 ~ N(0, I)`` and time ``t ~ Uniform(0, 1)``, then writes:

- ``batch["features"]["xt"]`` — interpolated sample at time ``t``
- ``batch["features"]["t"]``  — time tensor (per-sample)
- ``batch["targets"]["ut"]``  — velocity target

The original ``x1`` feature is removed from features and moved to targets
under ``"x1"`` so downstream loss computers or metrics can access it.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import Tensor

from dlkit.domain.nn.generative.functions.paths import linear_path
from dlkit.domain.nn.generative.functions.targets import displacement_target
from dlkit.domain.nn.generative.interfaces import INoiseSampler, ITimeSampler
from dlkit.domain.nn.generative.samplers.noise import GaussianNoiseSampler
from dlkit.domain.nn.generative.samplers.time import UniformTimeSampler


class FlowMatchingSupervisionBuilder:
    """Construct flow matching supervision tensors from raw data batch.

    Reads ``x1`` from ``batch["features"][x1_key]``, samples noise and time,
    and injects ``(xt, t)`` into features and ``ut`` into targets.

    Implements the ``IBatchTransform`` protocol:
    ``__call__(batch, generator) -> TensorDict``.

    Args:
        x1_key: Feature key holding the target samples ``x1``
            (default ``"x1"``).
        time_sampler: Samples per-batch time values (default: uniform ``[0, 1]``).
        noise_sampler: Samples initial noise ``x0`` (default: standard Gaussian).
    """

    def __init__(
        self,
        x1_key: str = "x1",
        time_sampler: ITimeSampler | None = None,
        noise_sampler: INoiseSampler | None = None,
    ) -> None:
        self._x1_key = x1_key
        self._time_sampler: ITimeSampler = time_sampler or UniformTimeSampler()
        self._noise_sampler: INoiseSampler = noise_sampler or GaussianNoiseSampler()

    def __call__(
        self,
        batch: TensorDict,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Build supervision tensors and inject them into the batch.

        Args:
            batch: Input TensorDict with at least ``batch["features"][x1_key]``.
            generator: Optional RNG for reproducible sampling.

        Returns:
            Modified TensorDict with::

                batch["features"]["xt"] = linear_path(x0, x1, t)
                batch["features"]["t"] = t
                batch["targets"]["ut"] = x1 - x0

        Raises:
            KeyError: If ``x1_key`` is not found in ``batch["features"]``.
        """
        features: TensorDict = cast(TensorDict, batch["features"])
        x1: Tensor = cast(Tensor, features[self._x1_key])

        batch_size = x1.shape[0]
        device = x1.device
        dtype = x1.dtype

        assert device is not None, "x1 device must not be None"
        assert dtype is not None, "x1 dtype must not be None"

        # Sample x0 ~ N(0, I) and t ~ Uniform(0, 1)
        x0: Tensor = self._noise_sampler(x1, generator)
        t: Tensor = self._time_sampler(batch_size, device=device, dtype=dtype, generator=generator)

        # Compute interpolation and velocity target
        xt: Tensor = linear_path(x0, x1, t)
        ut: Tensor = displacement_target(x0, x1)

        # Build new features: replace x1 with (xt, t)
        new_feature_keys = [k for k in features.keys() if k != self._x1_key]
        new_features_dict: dict[str | tuple[str, ...], Tensor] = {
            k: cast(Tensor, features[k]) for k in new_feature_keys
        }
        new_features_dict["xt"] = xt
        new_features_dict["t"] = t
        new_features = TensorDict(cast(Any, new_features_dict), batch_size=[batch_size])

        # Build new targets: add ut (keep existing targets)
        existing_targets: TensorDict = batch.get("targets", TensorDict({}, batch_size=[batch_size]))
        new_targets_dict: dict[str, Tensor] = dict(existing_targets.items())
        new_targets_dict["ut"] = ut
        new_targets = TensorDict(cast(Any, new_targets_dict), batch_size=[batch_size])

        # Rebuild batch with updated features and targets
        return batch.update({"features": new_features, "targets": new_targets})
