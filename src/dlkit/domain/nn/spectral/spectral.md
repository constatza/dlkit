# Spectral / Frequency-Domain Networks

Layers and models that operate (partly or fully) in the Fourier frequency domain.

## Why spectral augmentation?

Standard MLPs trained with gradient descent exhibit *spectral bias*: they learn
low-frequency components of the target function first and struggle to represent
high-frequency components without very deep networks or careful regularisation.
Providing explicit Fourier features at the input breaks this hierarchy and lets
the network learn high-frequency structure without relying on depth alone.

## Module layout

| Symbol | File | Description |
|--------|------|-------------|
| `ISpectralLayer` | `base.py` | Protocol: layer with a learnable spectral path |
| `SpectralConv1d` | `layers.py` | Learnable complex-weight multiplication in Fourier domain |
| `FourierLayer` | `layers.py` | FNO building block: spectral conv + local Conv1d skip |
| `FourierEnhancedFFNN` | `ffnn.py` | Single-branch MLP with spectral feature concatenation |
| `DualPathFFNN` | `ffnn.py` | Parallel spatial + spectral MLP branches |
| `FourierFeatureNetwork` | `coordinate.py` | Coordinate Fourier feature encoder followed by an MLP |
| `HashEncodingNetwork` | `coordinate.py` | Multiresolution hashed grid encoder plus MLP head |
| `Siren` | `coordinate.py` | Sinusoidal coordinate network with SIREN initialisation |
| `ModifiedMLP` | `coordinate.py` | Coordinate network with U/V gating |
| `ScaleEquivariantFourierFeatureNetwork` | `coordinate.py` | Norm-scaled wrapper over `FourierFeatureNetwork` |
| `ScaleEquivariantSiren` | `coordinate.py` | Norm-scaled wrapper over `Siren` |
| `ScaleEquivariantModifiedMLP` | `coordinate.py` | Norm-scaled wrapper over `ModifiedMLP` |

---

## Primitives

### SpectralConv1d

Learnable spectral convolution.  The complex spectrum is truncated to
`n_modes` coefficients and multiplied by a learned `(in_channels,
out_channels, n_modes)` complex weight matrix, then transformed back.

```
x (B, C_in, L)
  → rfft → truncate to n_modes
  → einsum with W ∈ ℂ^(C_in × C_out × n_modes)
  → pad zeros for higher modes
  → irfft
→ y (B, C_out, L)
```

```python
from dlkit.domain.nn.spectral import SpectralConv1d

layer = SpectralConv1d(in_channels=8, out_channels=8, n_modes=16)
y = layer(x)  # x: (B, 8, L)
```

### FourierLayer

Single FNO-style residual block combining a spectral path and a local
pointwise convolution.

```
y = activation(SpectralConv1d(x) + Conv1d(x, kernel=1))
```

```python
from dlkit.domain.nn.spectral import FourierLayer

block = FourierLayer(channels=32, n_modes=16)
y = block(x)  # x: (B, 32, L)
```

---

## Frequency-Enhanced FFNNs

These networks accept flat feature vectors `(batch, in_features)`, compute
a truncated Fourier representation, and merge it with the spatial features.

### FourierEnhancedFFNN

Concatenates `[x, rfft(x).real[:n], rfft(x).imag[:n]]` → single MLP.

```
augmented = cat([x, rfft(x)_real_trunc, rfft(x)_imag_trunc])
output    = ConstantWidthFFNN(augmented)
```

The augmented input dimension is `in_features + n_modes * 2`.

```python
from dlkit.domain.nn.spectral import FourierEnhancedFFNN

model = FourierEnhancedFFNN(
    in_features=64,
    out_features=16,
    hidden_size=128,
    num_layers=4,
    n_modes=16,
)
y = model(x)  # x: (B, 64)
```

### DualPathFFNN

Two independent branches learn spatial and spectral representations separately:

```
h_spatial  = SpatialBranch(x)              # ConstantWidthFFNN
h_spectral = SpectralBranch(rfft_feats(x)) # ConstantWidthFFNN
merged     = h_spatial + h_spectral        # or cat + linear (merge="concat")
output     = Linear(merged)
```

```python
from dlkit.domain.nn.spectral import DualPathFFNN

model = DualPathFFNN(
    in_features=64,
    out_features=16,
    hidden_size=128,
    num_layers=3,
    n_modes=16,
    merge="add",   # or "concat"
)
```

---

## Coordinate Spectral-Bias Networks

These architectures address spectral bias at the coordinate level rather than
through Fourier convolutions. All three
implement `from_contract(contract, **kwargs)` for factory-compatible construction.

### FourierFeatureNetwork

Projects input coordinates through a random (or learned) frequency matrix
before an MLP, directly countering spectral bias (Tancik et al. 2020).

```
γ(x) = [sin(2π B x), cos(2π B x)]  ∈ ℝ^{2m}
output = ConstantWidthFFNN(γ(x))
```

where `B ∈ ℝ^{m×d}` is sampled from `N(0, σ²)` at construction time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Coordinate input dimension |
| `out_features` | `int` | required | Network output dimension |
| `hidden_size` | `int` | required | Width of the internal MLP |
| `num_layers` | `int` | required | Number of hidden MLP layers |
| `n_frequencies` | `int` | required | Frequency vectors `m`; encoding output is `2m` |
| `sigma` | `float` | `1.0` | Std-dev for sampling `B` |
| `learnable_B` | `bool` | `False` | If `True`, `B` is an `nn.Parameter` |
| `activation` | `Callable` | `F.gelu` | Activation for the internal MLP |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalisation for the internal MLP |
| `dropout` | `float` | `0.0` | Dropout for the internal MLP |

`from_contract` extracts `in_features` and `out_features` from a `TabulaRSpec`.

### HashEncodingNetwork

Multiresolution hashed grid encoder in the style of Instant-NGP. The input is
first normalized into a bounded coordinate box, encoded across multiple hash
tables at increasing resolutions, concatenated with the raw input by default,
and then passed through a residual MLP head.

```
encoded = [x, hash_level_0(x), ..., hash_level_{L-1}(x)]
output  = ConstantWidthFFNN(encoded)
```

This representation is strongest when the input space is genuinely
coordinate-like and local neighborhoods in that space are semantically
meaningful.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Coordinate input dimension |
| `out_features` | `int` | required | Output dimension |
| `hidden_size` | `int` | required | Width of the MLP head |
| `num_layers` | `int` | required | Number of hidden MLP layers |
| `num_levels` | `int` | `16` | Number of hash-grid resolutions |
| `features_per_level` | `int` | `2` | Feature channels stored per level |
| `log2_hashmap_size` | `int` | `19` | Hash table size per level as `2^k` |
| `base_resolution` | `int` | `16` | Lowest grid resolution |
| `finest_resolution` | `int` | `512` | Highest grid resolution |
| `bounds` | `tuple[(float, float), ...] \| None` | `None` | Per-dimension input bounds; defaults to `(-1, 1)` |
| `include_input` | `bool` | `True` | Concatenate raw input with hashed features |
| `activation` | `Callable` | `F.gelu` | Activation in the MLP head |
| `normalize` | `"batch" \| "layer" \| None` | `None` | Normalisation for the MLP head |
| `dropout` | `float` | `0.0` | Dropout for the MLP head |

`from_contract` extracts `in_features` and `out_features` from a `TabulaRSpec`.

### Siren

Sinusoidal representation network (Sitzmann et al. 2020).  Uses `sin`
activations throughout with layer-specific weight initialisation that promotes
well-conditioned gradients.

```
x₀ = sin(ω₀ · W₀ x)
xₖ = sin(Wₖ xₖ₋₁)   for k = 1, …, L−1
output = W_out x_{L-1}
```

Initialisation:
- First layer: `W ~ U(-1/d, 1/d)` where `d = in_features`
- Hidden layers: `W ~ U(-sqrt(6/d)/ω₀, sqrt(6/d)/ω₀)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `hidden_size` | `int` | required | Width of all hidden layers |
| `num_layers` | `int` | required | Number of hidden layers (>= 1) |
| `omega0` | `float` | `30.0` | First-layer frequency multiplier |

Raises `ValueError` if `num_layers < 1`.  `from_contract` is supported.

### ModifiedMLP

U/V encoder gating (Wang et al. 2022).  Two encoder branches modulate each
hidden state to provide richer coordinate-conditioned gating than a plain MLP.

```
U = σ(W_u x + b_u)
V = σ(W_v x + b_v)
h₀ = σ(W₀ x + b₀)
zₖ = σ(Wₖ hₖ + bₖ)
hₖ₊₁ = zₖ ⊙ U + (1 − zₖ) ⊙ V
output = W_out h_L + b_out
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_features` | `int` | required | Input dimension |
| `out_features` | `int` | required | Output dimension |
| `hidden_size` | `int` | required | Width of all hidden layers |
| `num_layers` | `int` | required | Hidden linear layers (>= 2) |
| `activation` | `Callable` | `torch.sigmoid` | Gating activation σ |

Raises `ValueError` if `num_layers < 2`.  `from_contract` is supported.

```python
from dlkit.domain.nn.spectral import ModifiedMLP

model = ModifiedMLP(
    in_features=2,
    out_features=1,
    hidden_size=128,
    num_layers=4,
)
```

### Scale-Equivariant Wrappers

`ScaleEquivariantFourierFeatureNetwork`, `ScaleEquivariantSiren`, and
`ScaleEquivariantModifiedMLP` compose the corresponding base architecture with
shared norm-based input/output scaling:

```
norm = ||x||
output = norm * base_model(x / max(norm, eps))
```

This keeps scale-equivariant behavior in one reusable primitive rather than
reimplementing it in each spectral-bias model.

---

## TOML configuration

```toml
[model]
name = "FourierEnhancedFFNN"
hidden_size = 128
num_layers = 4
n_modes = 16
```

```toml
[model]
name = "DualPathFFNN"
hidden_size = 128
num_layers = 3
n_modes = 16
merge = "concat"
```

`FourierEnhancedFFNN` and `DualPathFFNN` expose `from_contract(contract, **kwargs)`,
so the shared model factory can build them from a `TabulaRSpec`.

---

## Design notes

- `SpectralConv1d` uses `torch.fft.rfft` with `norm="ortho"` for energy
  conservation across resolutions.
- Complex weights are stored as two real `nn.Parameter` tensors to avoid
  PyTorch version-specific complex gradient edge cases.
- `FourierLayer` is the reused building block for `FourierNeuralOperator1d`
  (see `operators/`).
- The spectral feature extraction in `FourierEnhancedFFNN` / `DualPathFFNN`
  is deliberately not an `nn.Module` — it is a pure function that requires no
  learnable parameters, keeping the graph clean.
