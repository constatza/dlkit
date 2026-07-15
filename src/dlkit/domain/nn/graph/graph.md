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

Graph models implement `from_context(context, **kwargs)` for dataset-driven construction. `in_channels` and `out_channels` are read from the last dimension of the first input and output shapes; `edge_dim` may be passed via `kwargs`.

## MoE and Hyper-Connection primitives

`dlkit.gnn` re-exports graph-compatible primitives:
- `GraphHyperConnection` wraps a graph module with multi-lane residual mixing and forwards `edge_index`/`edge_attr` per lane.
- `SparseMoE` and `TopKRouter` can be used on graph node features before or after message passing.

V1 does not include concrete graph model classes such as `MoEGATv2Projection`
or `HyperGATv2Projection`. Compose these primitives with existing graph
projection/message modules when that behavior is needed.

## Config example

```toml
[model]
name = "GATv2Projection"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
heads = 4
```
