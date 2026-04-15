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

`in_features` and `out_features` are injected automatically from the dataset
shape summary (factory "ffnn" strategy).

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
