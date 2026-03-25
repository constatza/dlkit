# `dlkit.core.models.nn.generative` — Flow Matching

## What is flow matching?

Flow matching (Lipman et al., 2022, *"Flow Matching for Generative Modeling"*) is a
simulation-free method for training continuous normalizing flows. Rather than solving a
reverse SDE or maximising a likelihood bound, it regresses a neural network directly onto
a target vector field that transports a simple source distribution (e.g. standard Gaussian)
to a data distribution along deterministic straight-line paths. Training reduces to a plain
mean-squared-error objective over stochastically sampled times `t ~ Uniform(0, 1)`, making
it computationally cheap and stable relative to score-based or diffusion approaches.

---

## Module layout

```
generative/
├── interfaces.py          # Single-method protocols: ITimeSampler, INoiseSampler,
│                          #   IInterpolationPath, IVelocityTarget, IModelAdapter,
│                          #   IFixedStepSolver, IModelFn — all @runtime_checkable
├── supervision.py         # FlowMatchingSupervisionBuilder — transforms a raw
│                          #   TensorDict batch into (xt, t) features + ut targets
├── samplers/
│   ├── time.py            # UniformTimeSampler  — samples t ~ Uniform(t_min, t_max)
│   └── noise.py           # GaussianNoiseSampler — samples x0 ~ N(0, I)
└── functions/             # Pure, stateless functions — no side effects
    ├── broadcast.py       # broadcast_time(t, ref) — shape alignment helper
    ├── paths.py           # linear_path, noise_schedule_path — interpolation paths
    ├── targets.py         # displacement_target — computes ut = x1 - x0
    └── solvers.py         # euler_step, heun_step, integrate — ODE integration
```

---

## Python API

```python
import torch
import torch.nn as nn
from tensordict import TensorDict

from dlkit.core.models.nn.generative.supervision import FlowMatchingSupervisionBuilder
from dlkit.core.models.nn.generative.functions.solvers import euler_step, integrate


# 1. Build a tiny velocity model: input is (xt, t) concatenated -> velocity
class VelocityNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 1, 64), nn.SiLU(), nn.Linear(64, dim))

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_col = t.unsqueeze(-1)  # (B, 1)
        return self.net(torch.cat([xt, t_col], dim=-1))


dim = 8
model = VelocityNet(dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 2. Build the supervision builder (defaults: UniformTimeSampler, GaussianNoiseSampler)
builder = FlowMatchingSupervisionBuilder(x1_key="x1")

# 3. Simulate one training step on a TensorDict batch
x1 = torch.randn(32, dim)
batch = TensorDict({"features": TensorDict({"x1": x1}, batch_size=[32])}, batch_size=[32])

batch = builder(batch)  # injects xt, t into features; ut into targets
xt = batch["features"]["xt"]  # (32, dim)
t = batch["features"]["t"]  # (32,)
ut = batch["targets"]["ut"]  # (32, dim)  — supervision signal

v_pred = model(xt, t)
loss = torch.nn.functional.mse_loss(v_pred, ut)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 4. Inference — integrate from x0 ~ N(0, I) to x1 using Euler solver
x0 = torch.randn(4, dim)
model_fn = lambda x, t: model(x, t)
x_generated = integrate(model_fn, x0, t_span=(0.0, 1.0), solver=euler_step, n_steps=100)
```

---

## TOML configuration

The `[GENERATIVE]` block controls the flow matching wrapper; the loss function is set
independently under `[TRAINING]`.

```toml
[GENERATIVE]
algorithm      = "flow_matching"
x1_key         = "x1"          # feature key holding target samples
path_type      = "linear"      # interpolation path: "linear" | "noise_schedule"
solver         = "euler"       # ODE solver used at inference: "euler" | "heun"
n_inference_steps = 100        # number of fixed steps during generation
val_seed       = 42            # RNG seed for reproducible validation samples

[TRAINING.loss_function]
name = "mse"                   # any registered loss name; MSE is standard for flow matching
```

---

## Extending: custom `ITimeSampler`

`ITimeSampler` is a `@runtime_checkable` single-method protocol. Any callable class
with the right signature is accepted automatically — no registration required.

```python
import torch
from torch import Tensor


class LogitNormalTimeSampler:
    """Sample times from a logit-normal distribution (used in SD3 / Flux)."""

    def __call__(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        u = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
        return torch.sigmoid(u)  # logit-normal -> (0, 1)


from dlkit.core.models.nn.generative.supervision import FlowMatchingSupervisionBuilder

builder = FlowMatchingSupervisionBuilder(
    x1_key="x1",
    time_sampler=LogitNormalTimeSampler(),
)
```

Pass any `INoiseSampler` implementation to `noise_sampler=` in the same way to swap
the source distribution (e.g. use mini-batch optimal-transport couplings).
