"""IPredictionStrategy implementations for predict_step delegation.

Strategies encapsulate the full predict_step logic so wrapper subclasses can
swap inference behaviour (discriminative vs. ODE-based generative) without
modifying the training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from tensordict import TensorDict
from torch import Tensor, nn

from dlkit.core.models.wrappers.base import _batch_size_of, _leaf_device, _leaf_dtype

if TYPE_CHECKING:
    from dlkit.core.models.wrappers.protocols import IBatchTransformer, IModelInvoker


class DiscriminativePredictionStrategy:
    """predict_step for standard discriminative models.

    Replicates the legacy ``ProcessingLightningWrapper.predict_step`` behaviour
    exactly: clone targets → transform → invoke → inverse-transform predictions.

    Implements ``IPredictionStrategy``.

    Args:
        model_invoker: Invokes the model on a batch.
        batch_transformer: Per-slot transform chains.
        predict_target_key: Target entry name whose chain is inverted at predict time.
    """

    def __init__(
        self,
        model_invoker: IModelInvoker,
        batch_transformer: IBatchTransformer,
        predict_target_key: str,
    ) -> None:
        self._model_invoker = model_invoker
        self._batch_transformer = batch_transformer
        self._predict_target_key = predict_target_key

    def predict(
        self,
        model: nn.Module,
        batch: Any,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Run discriminative prediction on a batch.

        Clones targets before transforming so the output always carries the
        original (untransformed) ground truth.

        Args:
            model: PyTorch model to invoke.
            batch: TensorDict batch from the dataloader.
            generator: Ignored (discriminative models are deterministic at predict time).

        Returns:
            TensorDict with keys ``"predictions"``, ``"targets"``, and ``"latents"``.
        """
        original_targets = batch["targets"].clone()
        transformed_batch = self._batch_transformer.transform(batch)
        enriched_batch = self._model_invoker.invoke(model, transformed_batch)

        predictions: Tensor | TensorDict = cast(Tensor | TensorDict, enriched_batch["predictions"])
        predictions = self._batch_transformer.inverse_transform_predictions(
            predictions, self._predict_target_key
        )

        batch_size = _batch_size_of(predictions)

        raw_latents = enriched_batch.get("latents", None)
        if raw_latents is not None:
            latents: Tensor | TensorDict = raw_latents
        else:
            latents = torch.zeros(
                batch_size,
                0,
                dtype=_leaf_dtype(predictions),
                device=_leaf_device(predictions),
            )

        return TensorDict(
            {"predictions": predictions, "targets": original_targets, "latents": latents},
            batch_size=[batch_size],
        )


class ODEPredictionStrategy:
    """predict_step for continuous-time flow models via ODE integration.

    Generates samples by integrating from ``x0 ~ p_0`` (typically Gaussian) to
    ``x1`` using a fixed-step solver.  Generic: works for any model that accepts
    ``model(x, t)`` and returns a velocity.

    Implements ``IPredictionStrategy``.

    Shape must be configured via ``configure_shape()`` before the first call
    (done by ``ContinuousFlowWrapper.on_fit_start``).

    Args:
        x0_sampler: Samples initial noise matching a reference tensor.
        solver: Fixed-step ODE solver (e.g. ``euler_step``, ``heun_step``).
        n_steps: Number of uniform integration steps.
        t_span: ``(t_start, t_end)`` integration interval.
    """

    def __init__(
        self,
        x0_sampler: Any,
        solver: Any,
        n_steps: int = 100,
        t_span: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._x0_sampler = x0_sampler
        self._solver = solver
        self._n_steps = n_steps
        self._t_span = t_span
        self._data_shape: tuple[int, ...] | None = None

    def configure_shape(self, data_shape: tuple[int, ...]) -> None:
        """Set the per-sample data shape for noise generation.

        Args:
            data_shape: Spatial dimensions (excluding batch), e.g. ``(3, 32, 32)``.
        """
        self._data_shape = data_shape

    @property
    def data_shape(self) -> tuple[int, ...] | None:
        """Per-sample data shape, or ``None`` if not yet configured.

        Returns:
            Shape tuple or None.
        """
        return self._data_shape

    @property
    def n_steps(self) -> int:
        """Number of ODE integration steps.

        Returns:
            Integer step count.
        """
        return self._n_steps

    def predict(
        self,
        model: nn.Module,
        batch: Any,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Generate samples by ODE integration from Gaussian noise.

        Args:
            model: Velocity-field model accepting ``(x, t)`` inputs.
            batch: TensorDict batch — used only for batch size and device/dtype inference.
            generator: Optional RNG for reproducible noise sampling.

        Returns:
            TensorDict with ``"predictions"`` (generated samples), ``"targets"``
            (cloned from batch), and empty ``"latents"``.

        Raises:
            RuntimeError: If ``configure_shape()`` has not been called yet.
        """
        if self._data_shape is None:
            raise RuntimeError(
                "ODEPredictionStrategy.configure_shape() must be called before predict(). "
                "This is done automatically by ContinuousFlowWrapper.on_fit_start()."
            )
        from dlkit.core.models.nn.generative.functions.solvers import integrate

        # Infer batch size, device, dtype from batch
        features = batch.get("features", batch)
        first_tensor: Tensor | None = None
        try:
            first_tensor = next(iter(features.values()))
        except StopIteration, AttributeError:
            pass

        if first_tensor is not None:
            n = first_tensor.shape[0]
            device = first_tensor.device
            dtype = first_tensor.dtype
        else:
            n = int(batch.batch_size[0])
            device = torch.device("cpu")
            dtype = torch.float32

        # Sample initial noise x0 ~ p_0
        ref = torch.empty((n, *self._data_shape), device=device, dtype=dtype)
        x0 = self._x0_sampler(ref, generator)

        # Build model_fn adapter: model accepts (x, t)
        def model_fn(x: Tensor, t: Tensor) -> Tensor:
            return model(x, t)

        # Integrate ODE
        x1_hat = integrate(model_fn, x0, self._t_span, self._solver, self._n_steps)

        # Clone targets for output alignment
        try:
            targets = batch["targets"].clone()
        except Exception:
            targets = TensorDict({}, batch_size=[n])

        latents = torch.zeros(n, 0, dtype=dtype, device=device)

        return TensorDict(
            {"predictions": x1_hat, "targets": targets, "latents": latents},
            batch_size=[n],
        )
