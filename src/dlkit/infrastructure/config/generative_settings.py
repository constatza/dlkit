"""Discriminated union configuration for generative algorithms.

Each algorithm class is a separate Pydantic model with a ``Literal`` ``algorithm``
field used as the discriminator.  Pydantic resolves the union at parse time so
typos are caught before training starts.

Usage in TOML::

    [GENERATIVE]
    algorithm = "flow_matching"
    # path_type = "linear"
    # solver    = "euler"
    # x1_key    = "x1"

New algorithms: add a new settings class + add to the ``GenerativeSettings`` union.
No edits to existing classes needed (OCP).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class FlowMatchingSettings(BasicSettings):
    """Settings for the flow matching generative algorithm.

    Flow matching (Lipman et al., 2022) trains a velocity field on straight-line
    interpolations between noise and data, then generates via ODE integration.

    Args:
        algorithm: Discriminator literal — must be ``"flow_matching"``.
        path_type: Interpolation path.  ``"linear"`` = straight lines (standard FM);
            ``"noise_schedule"`` = EDM-style stochastic interpolant.
        target_type: Velocity supervision target.  ``"displacement"`` = ``x1 - x0``
            (constant velocity field for linear paths).
        solver: Fixed-step ODE solver for inference.
        n_inference_steps: Number of ODE steps at generation time.
        val_seed: Base seed for deterministic validation supervision.
        x1_key: Dataset entry name that contains the training targets ``x1``.
    """

    algorithm: Literal["flow_matching"] = "flow_matching"
    path_type: Literal["linear", "noise_schedule"] = "linear"
    target_type: Literal["displacement"] = "displacement"
    solver: Literal["euler", "heun"] = "euler"
    n_inference_steps: int = 100
    val_seed: int = 42
    x1_key: str = "x1"


class CNFSettings(BasicSettings):
    """Settings for Continuous Normalising Flow (CNF / FFJORD) training.

    CNF training integrates the model as an ODE during the forward pass and
    computes NLL via the instantaneous change-of-variables formula.

    Args:
        algorithm: Discriminator literal — must be ``"cnf"``.
        solver: Fixed-step ODE solver.
        n_inference_steps: Number of ODE steps at generation time.
        divergence: Divergence estimator for log-det computation.
            ``"exact"`` = full Jacobian trace; ``"hutchinson"`` = stochastic estimate.
        val_seed: Base seed for deterministic validation.
    """

    algorithm: Literal["cnf"] = "cnf"
    solver: Literal["euler", "heun"] = "euler"
    n_inference_steps: int = 100
    divergence: Literal["exact", "hutchinson"] = "hutchinson"
    val_seed: int = 42


# Discriminated union — add new algorithms by adding a class above and unioning here.
GenerativeSettings = Annotated[
    FlowMatchingSettings | CNFSettings,
    Field(discriminator="algorithm"),
]
