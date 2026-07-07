# Constant-Width & Scale-Equivariant Architectures: May vs Current

Comparison baseline: commit `d11a3b1` (2026-05-19, last commit in the early/mid-May
window before the 2026-05-27 constrained-hierarchy consolidation) vs current `develop`
HEAD. Scope: mathematical and hyperparameter-default differences only.

## 1. Scale-equivariant wrapper — math unchanged

`ScaleEquivariantWrapper` computes:

```
f(x) = base_model(x / ‖x‖) * ‖x‖
```

L2 norm, `eps_gain=10.0` (`eps = finfo(x.dtype).eps * eps_gain`), giving `f(αx) = α·f(x)`
for `α > 0`. This formula is byte-identical between `d11a3b1` and HEAD, and predates
even that baseline — the 2026-04-15 `NormScaled → ScaleEquivariant` rename (`77443ff`)
was naming-only, no math change.

The only structural addition since `d11a3b1` is `ConditionedScaleEquivariantWrapper`
(2026-06-12, `a48614a`), used by FiLM-conditioned variants:
`base_model(x/‖x‖, condition) * ‖x‖` — same formula, condition passed through unscaled.

## 2. `FactorizedLinear` primitive — where the real drift happened

Shared by both the constant-width and scale-equivariant families. Weight:

```
W = diag(φ(log_scale)) @ base_weight
log_scale ~ N(mean, std=0.1)
```

| Stage | Commit | `pos_fn` (`φ`) | Kaiming `a` on `base_weight` | `mean` |
|---|---|---|---|---|
| Baseline | `d11a3b1` (05-19) | free kwarg, default `torch.exp` | `0.0` | `0.0` |
| Same-day fix | `37e8cc9` (05-19) | default → `F.softplus`, bounded gradient; auto `_unit_scale_log_mean` correction added | `√5` (fixes 2^depth variance blow-up from `a=0` under residual stacking) | corrected so `φ(log_scale)≈1` regardless of `pos_fn` |
| "Paper-style RWF" | `58d337c` (06-23) | split into `FactorizedLinear` (exp, hardcoded, no longer a kwarg) + `SoftplusFactorizedLinear` (separate class); auto-correction removed | `√5` | **regression: default → `1.0`**, i.e. `exp(1.0)≈2.72` inflated scale at init (compounds to ~3700x over an 8-layer body) |
| Fix (multi-layer) | `7d1d0be` (06-26) | unchanged | reverted to `0.0` | reverted to `0.0` |
| Fix (single-layer, `FactorizedLinearNetwork`) | `e558cd0` (07-02) | unchanged | — | reverted to `0.0` (this class was missed by the 06-26 fix) |

Net at HEAD: numerically equivalent to the `d11a3b1` baseline for the exp path
(`a=0.0`, `mean=0.0`), but the codebase carried a live scale-inflation bug for
~34 days (06-23 to 07-02 for the single-layer variant). `pos_fn` is no longer a
runtime-swappable constructor kwarg — callers now pick `FactorizedLinear` vs
`SoftplusFactorizedLinear` as distinct classes instead of passing `pos_fn=`.

## 3. Activation defaults

- `d11a3b1`: every class hardcoded `nn.functional.gelu` as the literal Python default.
- HEAD: low-level generic/composable base classes (`ParametricDenseBlock`,
  `_ConstantWidthParametricBody`, `_EmbeddedParametricBody`) default to **ReLU**
  via `resolve_activation()`'s internal default. The concrete public classes
  (`EmbeddedFactorizedFFNN`, `ConstantWidthFactorizedFFNN`, `ScaleEquivariant*`, etc.)
  explicitly request `default="gelu"`, so their effective default is unchanged —
  but anyone building on the generic base classes directly now silently gets ReLU
  instead of GELU. Mechanism added in `76176f4` (06-30).

## 4. `hidden_size` defaulting — diverges by family

- Factorized `Embedded*` family (`constrained.py`): optional, auto-resolves to
  `in_features` when `in_features == out_features` (`_resolve_hidden_size`,
  propagated repo-wide in `a7ebef9`, 05-27); raises otherwise.
- Plain dense `FFNN`/`EmbeddedFFNN` (`residual.py`): **required, no default** —
  a silent `max(in_features, out_features)` default was deliberately removed in
  `5f89376` (06-30) to close a path that caused CUDA OOM from oversized layers.

These two families now have opposite `hidden_size` defaulting philosophies.

## 5. SPD/Symmetric family — deleted entirely

At `d11a3b1`, both hierarchies had ~8 SPD-constrained classes each (`SPDLinear`,
`SPDFactorizedLinear`, Gershgorin-based positive-definite parametrization). All
deleted in `4ea3f6f` (06-24, "remove symmetric and SPD network variants"). Only
Factorized/SoftplusFactorized variants survive at HEAD.

## 6. `num_layers` semantics

`_ConstantWidthParametricBody`: `num_layers <= 0` raised at `d11a3b1`. HEAD allows
`num_layers = 0` (identity body), changed in `cd90372` (05-27).

## 7. Naming/API churn

`ConstantWidth` prefix dropped from the SPD hierarchy (`cd90372`, 05-27) but kept
on surviving Factorized pure-body classes. Dense `ConstantWidthFFNN` → `FFNN`,
`FeedForwardNN` → `VarWidthFFNN` (`ba00d0d`). New `StandardEntryConsumer` contract
mixin replaced ad-hoc `from_shape` classmethods (data-layer redesign, `504181c`).

## 8. Test-encoded intent shift

At `d11a3b1`, tests directly asserted `f(αx) = α·f(x)`. Current tests deliberately
removed that assertion — equivariance holds by construction of the wrapper
regardless of body internals — and instead assert (a) unit-scale-at-init and
(b) body-signal-does-not-diverge across depth, specifically to catch the
`mean=1.0` regression class of bug described in §2.

## Bottom line

The scale-equivariance property itself never changed — it's a wrapper-level
guarantee, invariant to everything that happened to the bodies it wraps. All the
real mathematical drift lives in the shared `FactorizedLinear` initialization
scheme, which round-tripped through a real ~34-day regression (`mean` default
`0.0 → 1.0 → 0.0`) between late June and early July. Everything else (SPD removal,
prefix drops, hidden_size/num_layers validation tightening, activation resolution
mechanism) is API surface and validation strictness, not architecture math.
