# Graph NN Module

## Residual/Plain naming convention

Same as the FFNN family:
- Unprefixed class name means residual connections active
- `Simple...` prefix means no residual connections

## Class matrix

| Class | Base | Residual |
|---|---|---|
| `GATv2Message` | `_GATv2MessageBase` | Yes |
| `SimpleGATv2Message` | `_GATv2MessageBase` | No |
| `GATv2Projection` | `GProjection` | Yes (via `GATv2Message`) |
| `SimpleGATv2Projection` | `GProjection` | No |
| `ScaledGATv2Projection` | `ScaledGProjection` | Yes |
| `ScaledSimpleGATv2Projection` | `ScaledGProjection` | No |

## Construction protocol

All built-ins are concrete classes. No public `residual: bool` on any constructor.

Graph models implement `from_entries(input_shapes, output_shapes, **kwargs)` for dataset-driven construction. `in_channels` and `out_channels` are read from the last dimension of the first input and output shapes; `edge_dim` may be passed via `kwargs`.

## Config example

```toml
[model]
name = "GATv2Projection"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
heads = 4
```
