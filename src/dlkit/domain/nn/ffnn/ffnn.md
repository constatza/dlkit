# Feed-Forward Neural Networks

`dlkit.domain.nn.ffnn` groups flat-input neural networks by architecture first.
The package distinguishes:
- residual vs plain
- dense vs constrained linear bodies
- standard vs scale-equivariant wrappers
- constant-width bodies vs embedded input/output projections

## Module layout

| File | Purpose |
|---|---|
| `linear.py` | Linear baseline |
| `simple.py` | Plain dense FFNNs without skip connections |
| `residual.py` | Residual dense FFNNs with skip connections |
| `constrained.py` | Constrained linear FFNN builders and explicit plain/residual variants |
| `scale_equivariant.py` | Class-based scale-equivariant wrappers for dense and constrained FFNNs |

## Variant matrix

### Dense

| Architecture | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Variable-width | `SimpleFeedForwardNN` | `FeedForwardNN` | Use `ScaleEquivariantFFNN(base_model=...)` | Use `ScaleEquivariantFFNN(base_model=...)` |
| Constant-width | `ConstantWidthSimpleFFNN` | `ConstantWidthFFNN` | `ScaleEquivariantConstantWidthSimpleFFNN` | `ScaleEquivariantConstantWidthFFNN` |

### Constrained constant-width

| Layer family | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Factorized | `ConstantWidthSimpleFactorizedFFNN` | `ConstantWidthFactorizedFFNN` | `ScaleEquivariantConstantWidthSimpleFactorizedFFNN` | `ScaleEquivariantConstantWidthFactorizedFFNN` |
| SPD | `ConstantWidthSimpleSPDFFNN` | `ConstantWidthSPDFFNN` | `ScaleEquivariantConstantWidthSimpleSPDFFNN` | `ScaleEquivariantConstantWidthSPDFFNN` |
| SPD-factorized | `ConstantWidthSimpleSPDFactorizedFFNN` | `ConstantWidthSPDFactorizedFFNN` | `ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN` | `ScaleEquivariantConstantWidthSPDFactorizedFFNN` |

### Constrained embedded

| Layer family | Plain | Residual | Scale-equivariant plain | Scale-equivariant residual |
|---|---|---|---|---|
| Factorized | `EmbeddedSimpleFactorizedFFNN` | `EmbeddedFactorizedFFNN` | `ScaleEquivariantEmbeddedSimpleFactorizedFFNN` | `ScaleEquivariantEmbeddedFactorizedFFNN` |
| SPD | `EmbeddedSimpleSPDFFNN` | `EmbeddedSPDFFNN` | `ScaleEquivariantEmbeddedSimpleSPDFFNN` | `ScaleEquivariantEmbeddedSPDFFNN` |
| SPD-factorized | `EmbeddedSimpleSPDFactorizedFFNN` | `EmbeddedSPDFactorizedFFNN` | `ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN` | `ScaleEquivariantEmbeddedSPDFactorizedFFNN` |

## Low-level constrained builders

`constrained.py` also keeps the reusable builder-oriented classes:
- `ParametricDenseBlock`
- `ConstantWidthParametricFFNN`
- `EmbeddedParametricFFNN`

These remain available for custom compositions, but the preferred public model
surface is the explicit plain/residual class matrix above.

## Naming rules

- No `Simple` prefix means the model uses residual/skip connections.
- `Simple...` means the model is plain and does not use skip connections.
- `ConstantWidth...` means the network body is square and width-preserving.
- `Embedded...` means the network has input embedding and output projection layers.
- `ScaleEquivariant...` means the model wraps a base FFNN with norm-based
  input/output scaling.

## Shape-aware construction

The classes that naturally map dataset feature counts to `in_features` and
`out_features` implement `from_shape(shape, **kwargs)`. This includes:
- dense FFNNs
- embedded constrained FFNNs
- scale-equivariant embedded FFNNs
- scale-equivariant constant-width dense FFNNs

Square constant-width constrained bodies use `size`, so they are configured
explicitly rather than through `from_shape()`.

## Configuration guidance

For config-driven construction, prefer the package export surface:

```toml
[MODEL]
name = "EmbeddedSimpleFactorizedFFNN"
module_path = "dlkit.domain.nn.ffnn"
hidden_size = 64
num_layers = 3
```

If you want the concrete dense residual implementation module directly:

```toml
[MODEL]
name = "ConstantWidthFFNN"
module_path = "dlkit.domain.nn.ffnn.residual"
hidden_size = 128
num_layers = 4
```
