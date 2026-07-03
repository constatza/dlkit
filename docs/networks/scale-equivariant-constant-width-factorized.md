# ScaleEquivariantConstantWidthFactorizedFFNN

## Rename History

| Date | Commit | Class name | Config key (dl-experiments) |
|---|---|---|---|
| Pre-Apr 2026 | — | `NormScaledConstantWidthFactorizedFFNN` | — |
| 2026-04-15 | `77443ff` | **`ScaleEquivariantConstantWidthFactorizedFFNN`** | `constant-width-factorized` |
| 2026-05-27 | `cd90372` | *(dropped in consolidation)* | — |
| 2026-06-26 | `7d1d0be` | **`ScaleEquivariantConstantWidthFactorizedFFNN`** | `scale-equivariant-constant-width-factorized` |
| HEAD | — | same | same |

The softplus exact replica added 2026-07-01:

| Date | Class name | Config key |
|---|---|---|
| 2026-07-01 | **`ScaleEquivariantConstantWidthSoftplusFactorizedFFNN`** | `scale-equivariant-constant-width-softplus-factorized` (suggested) |

---

## Architectural Invariant — No Projection Layers

**Every layer in the body, including the last, is a `FactorizedLinear`.** There is
no embedding projection at the input and no plain `nn.Linear` regression at the
output. This is the defining property that distinguishes `ConstantWidth*` from the
`Embedded*` and `FactorizedFFNN` families, which all end with a plain `nn.Linear`
regression layer and optionally start with one.

Contrast:
- `ConstantWidthFactorizedFFNN(n, n, L)`: `L × FactorizedLinear(n→n)` residual — **L factorized layers**
- `EmbeddedFactorizedFFNN(n, n, n, L)`: `Linear(n→n)` + `L × FactorizedLinear(n→n)` residual + `Linear(n→n)` — **L+2 layers, 2 plain**
- `FactorizedFFNN(n, n, n, L)`: `FactorizedLinear(n→n)` (no skip) + `(L-1) × FactorizedLinear(n→n)` residual + `Linear(n→n)` — **L+1 layers, last plain**

This also means `in_features == out_features` is not a stylistic restriction but a
structural requirement: residual `x̂ = h + x̂` requires the same dimension throughout.

---

## Mathematical Definition

Let `x ∈ ℝⁿ` with `n = in_features = out_features`, `L = num_layers`.

### Step 1 — L2 normalisation

```
‖x‖   = √(∑ᵢ xᵢ²)            (per sample, keepdim)
ε     = eps_gain × finfo.eps   (default eps_gain = 10.0)
x̂    = x / max(‖x‖, ε)
```

### Step 2 — L residual factorized blocks

For ℓ = 1 … L (all layers identical, including the last):

```
h = Norm(x̂)              (Identity when normalize=None — default)
h = σ(h)                  (σ = GELU by default)
h = W_ℓ h + b_ℓ          (FactorizedLinear n→n; see table below)
h = Dropout(h)            (Identity when dropout=0.0 — default)
x̂ = h + x̂               (residual add; identity skip since n==n throughout)
```

### Step 3 — Rescale

```
output = x̂ × ‖x‖
```

**Key invariant:** `f(αx) = α f(x)` for any scalar α > 0.

---

## Factorized Linear Layer

The effective weight matrix is:

```
W = diag(φ(s)) @ B
```

where `B ∈ ℝⁿˣⁿ` is the base weight and `s ∈ ℝⁿ` is the learnable log-scale.

### What "scale" means

Each output neuron `i` has its own learnable scalar `sᵢ` (the log-scale).
The actual per-neuron scale factor is `φ(sᵢ)`, which multiplies the entire
`i`-th row of `B`:

```
W[i, :] = φ(sᵢ) × B[i, :]
```

This means:
- At init: `φ(sᵢ) ≈ 1` for all `i`, so `W ≈ B` — the network starts as a plain
  Kaiming-initialised linear layer.
- During training: each neuron can grow or shrink its effective scale independently,
  giving the network a learnable per-output magnitude on top of its direction.
- Combined with scale equivariance: the SE wrapper normalises `x` to unit L2 norm
  before the body, then rescales the output by `‖x‖`. The body's `φ(s)` factors
  therefore act on unit-norm inputs only, controlling relative output directions
  without affecting the overall output magnitude (which is locked to `‖x‖`).

### Softplus vs exp: scale gradient at large s

```
φ = softplus:  φ′(s) = sigmoid(s) → 1 as s → ∞   (bounded gradient)
φ = exp:       φ′(s) = exp(s)      → ∞ as s → ∞   (unbounded gradient)
```

With softplus, a neuron whose log-scale drifts large still has a bounded
gradient, making training more numerically stable. With exp, large positive `s`
can cause scale explosions during optimisation.

### Historical vs current scale function

| Property | Historical (pre-`cd90372`, softplus) | Current exp (`7d1d0be`) | Softplus replica (`7d1d0be`+) |
|---|---|---|---|
| Class | `ScaleEquivariantConstantWidthFactorizedFFNN` | `ScaleEquivariantConstantWidthFactorizedFFNN` | `ScaleEquivariantConstantWidthSoftplusFactorizedFFNN` |
| φ(s) | `softplus(s) = log(1 + eˢ)` | `exp(s) = eˢ` | `softplus(s) = log(1 + eˢ)` |
| s init mean | `log(e-1) ≈ 0.5413` | `0.0` | `log(e-1) ≈ 0.5413` |
| E[φ(s)] at init | `softplus(0.5413) ≈ 1.0` ✓ | `exp(0.0) = 1.0` ✓ | `softplus(0.5413) ≈ 1.0` ✓ |
| φ′(0) | `sigmoid(0) = 0.5` | `1.0` | `0.5` |
| Large-s growth | linear | exponential | linear |
| `pos_fn` exposed | Yes (param) | No (hardcoded) | No (hardcoded via `_softplus_unit_layer_factory`) |
| B init | Kaiming uniform (a=0.0) | same | same |
| b init | zeros | same | same |

### Block order (pre-activation residual)

```
norm → activation → FactorizedLinear → dropout → (+x)
```

No double-activation. No missing layer. Identical between historical and current.

---

## Default Parameters

| Parameter | Default | Notes |
|---|---|---|
| `in_features` | — | required; must equal `out_features` |
| `out_features` | — | required |
| `num_layers` | — | required |
| `bias` | `True` | bias in each factorized linear |
| `mean` | `0.0` | offset from unit-scale point |
| `std` | `0.1` | log-scale init spread |
| `activation` | `None → GELU` | per-block nonlinearity |
| `normalize` | `None` | no batch/layer norm |
| `dropout` | `0.0` | no dropout |
| `norm` | `"l2"` | SE wrapper norm type |
| `eps_gain` | `10.0` | safe-division floor multiplier |
| `keep_stats` | `False` | if True, forward returns (output, {"norm": …}) |

---

## Source Files

- Body: [src/dlkit/domain/nn/ffnn/constrained.py](../src/dlkit/domain/nn/ffnn/constrained.py) — `ConstantWidthFactorizedFFNN`, `ConstantWidthSoftplusFactorizedFFNN`
- Wrapper: [src/dlkit/domain/nn/ffnn/scale_equivariant.py](../src/dlkit/domain/nn/ffnn/scale_equivariant.py) — `ScaleEquivariantConstantWidthFactorizedFFNN`, `ScaleEquivariantConstantWidthSoftplusFactorizedFFNN`
- Primitives: [src/dlkit/domain/nn/primitives/parametrized_layers.py](../src/dlkit/domain/nn/primitives/parametrized_layers.py) — `FactorizedLinear`, `SoftplusFactorizedLinear`
- SE wrapper: [src/dlkit/domain/nn/primitives/scale_equivariant.py](../src/dlkit/domain/nn/primitives/scale_equivariant.py) — `ScaleEquivariantWrapper`
