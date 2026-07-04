# FactorizedLinearNetwork

## Rename History

| Date | Commit | Class name | Config key (dl-experiments) |
|---|---|---|---|
| 2026-05-19 | `e572293` | **`FactorizedLinearNetwork`** | `factorized-linear` |
| 2026-06-23 | `58d337c` | same | same — **regression: mean changed to 1.0** |
| 2026-07-01 | HEAD | same | same — **fixed: mean restored to 0.0** |

---

## Mathematical Definition

Single-layer linear network with a learnable per-row scale factor:

```
output = W x + b

W = diag(φ(s)) @ B
```

where:
- `B ∈ ℝ^{out × in}` — base weight, Kaiming uniform (a=√5)
- `s ∈ ℝ^{out}` — learnable log-scale, sampled at init from N(mean, std²)
- `b ∈ ℝ^{out}` — bias, zeros at init (when bias=True)
- `φ = exp` — scale nonlinearity

No activation, no normalization, no hidden layers.

---

## What "scale" means

Each output neuron `i` has a learnable `sᵢ` (log-scale).
The actual scale factor `φ(sᵢ) = exp(sᵢ)` multiplies row `i` of `B`:

```
W[i, :] = exp(sᵢ) × B[i, :]
```

At init with `mean=0.0`: `exp(0) = 1`, so `W ≈ B` (plain Kaiming-init linear).
During training: each output neuron independently rescales its magnitude.

`FactorizedLinearNetwork` is the linear special case (no hidden layers, no
activation, no skip). It is useful as a baseline to check whether the
per-neuron scale factor alone — without depth or residuals — provides any
benefit over a plain `nn.Linear`.

## Default Parameters

| Parameter | Historical (introduced `e572293`) | Bug (`58d337c`) | Fixed (HEAD) |
|---|---|---|---|
| `in_features` | required | required | required |
| `out_features` | required | required | required |
| `bias` | `True` | `True` | `True` |
| `mean` | **`0.0` → exp(0)=1 (unit scale)** | **`1.0` → exp(1)≈2.72 (regression)** | **`0.0` → unit scale (fixed)** |
| `std` | `0.1` | `0.1` | `0.1` |

The `mean` parameter is the literal Gaussian mean for `log_scale` initialisation.
With `φ = exp`:
- `mean=0.0` → `exp(0.0) = 1.0` — unit scale at init ✓
- `mean=1.0` → `exp(1.0) ≈ 2.72` — inflated scale at init ✗

---

## Scale Init Correction

The `58d337c` commit changed the public `FactorizedLinear` from softplus to exp
and removed the `_unit_scale_log_mean` correction that had been applied automatically.
All multi-layer factorized classes received `mean=0.0` in the follow-up `7d1d0be`
commit, but `FactorizedLinearNetwork` was missed. Fixed 2026-07-01.

---

## Source File

[src/dlkit/domain/nn/ffnn/linear.py](../src/dlkit/domain/nn/ffnn/linear.py)
