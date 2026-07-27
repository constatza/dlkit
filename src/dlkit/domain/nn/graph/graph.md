# Graph NN Module

## Pluggable graph convolution (`conv_kind`)

Message-passing aggregation is a Strategy selected via `conv_kind:
Literal["gatv2", "gcn", "sage"]`, resolved through
`resolve_graph_conv_factory` in `graph/conv.py`. Every conv kind is
normalized to the same call contract `forward(x, edge_index, edge_attr=None)
-> Tensor` by a thin per-kind adapter, so message stacks, projection
networks, and `GraphHyperConnection`/`GraphHyperSequential` never need to
know which aggregation algorithm is behind the module they're calling.
Adding a new conv kind means one new adapter class plus one new `match` arm
in `resolve_graph_conv_factory` — nothing else in the graph package changes.

### Module layout

| File | Purpose |
|---|---|
| `conv.py` | PyG-normalization layer only: `GraphConvKind`, `resolve_graph_conv_factory`, the per-kind adapter classes |
| `message.py` | Message-stack composition layer built on `conv.py`: `GraphMessage`/`SimpleGraphMessage`/`GATv2Message`/`SimpleGATv2Message` |
| `embedded.py` | `EmbeddedGraphNetwork`/`ScaledEmbeddedGraphNetwork` — the conv-kind-generic composites |
| `gatv2_presets.py` | GATv2-only convenience presets (`GATv2Projection` family) over the generic embedded classes |
| `projection_networks.py` | `ProjectionNetwork`/`GProjection`/`ScaledGProjection` — the injected-`message_module` base classes |

This mirrors the split used by the FFNN family: `primitives/dense.py` (block-strategy
factory) is a separate file from `ffnn/hyper_moe.py` (the composition layer built on
top of it).

### Supported kinds

| `conv_kind` | Underlying PyG conv | Edge features | Residual mechanism |
|---|---|---|---|
| `"gatv2"` | `GATv2Conv` | `edge_dim` any positive int; full edge feature support via `GATv2Conv`'s own attention mechanism | Native: `GATv2Conv(residual=...)` builds its own learned residual projection |
| `"gcn"` | `GCNConv` | `edge_dim` must be `None` or `1` (a single scalar edge weight); anything else raises `ValueError` at construction | Applied by the adapter: `x + conv(x, edge_index, edge_weight)` |
| `"sage"` | `SAGEConv` | Not supported at all — `edge_dim` must be `None`; a non-`None` `edge_dim` raises `ValueError` at construction, and a non-`None` `edge_attr` passed at forward time (despite `edge_dim=None`) raises `ValueError` as a defensive backstop | Applied by the adapter: `x + conv(x, edge_index)` |

Residual handling is intentionally kind-specific rather than uniform:
`GATv2Conv` already implements a correct learned residual internally, while
`GCNConv`/`SAGEConv` have no such constructor kwarg, so their adapters add
`x + conv(...)` themselves.

Dropout semantics are also kind-specific: for `"gatv2"` it is `GATv2Conv`'s
native attention-coefficient dropout; for `"gcn"`/`"sage"` it is feature
dropout applied to the conv's output inside the adapter (these convs have
no native dropout kwarg).

## Residual/Plain naming convention

Same as the FFNN family:
- Unprefixed class name means residual connections active
- `Simple...` prefix means no residual connections

## Class matrix

| Class | Base | Residual | `conv_kind` |
|---|---|---|---|
| `GraphMessage` | `_GraphMessageBase` | Yes | any (constructor param) |
| `SimpleGraphMessage` | `_GraphMessageBase` | No | any (constructor param) |
| `GATv2Message` | `GraphMessage` | Yes | fixed `"gatv2"` |
| `SimpleGATv2Message` | `SimpleGraphMessage` | No | fixed `"gatv2"` |
| `EmbeddedGraphNetwork` | `GProjection` | Constructor param (default `True`) | any (constructor param) |
| `ScaledEmbeddedGraphNetwork` | `ScaledGProjection` | Constructor param (default `True`) | any (constructor param) |
| `GATv2Projection` | `EmbeddedGraphNetwork` | Yes | fixed `"gatv2"` |
| `SimpleGATv2Projection` | `EmbeddedGraphNetwork` | No | fixed `"gatv2"` |
| `ScaledGATv2Projection` | `ScaledEmbeddedGraphNetwork` | Yes | fixed `"gatv2"` |
| `ScaledSimpleGATv2Projection` | `ScaledEmbeddedGraphNetwork` | No | fixed `"gatv2"` |

`GATv2Message`/`SimpleGATv2Message` and the `GATv2Projection` family are
convenience presets kept for backward compatibility: one-line subclasses
fixing `conv_kind="gatv2"` over the generic `GraphMessage`/
`SimpleGraphMessage`/`EmbeddedGraphNetwork`/`ScaledEmbeddedGraphNetwork`
classes. None of them expose a public `residual` constructor parameter.

## Construction protocol

All built-ins are concrete classes. No public `residual: bool` on any of
the `GATv2*`-preset constructors (`EmbeddedGraphNetwork`/
`ScaledEmbeddedGraphNetwork` do expose `residual` directly, since they are
the generic, conv-kind-agnostic composites).

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
name = "EmbeddedGraphNetwork"
module_path = "dlkit.domain.nn.graph"
hidden_size = 64
num_layers = 3
conv_kind = "gcn"
```

GATv2-specific knobs (`heads`, `concat`) or SAGE-specific knobs (`aggr`)
travel as extra top-level keys alongside `conv_kind`, forwarded to the
underlying conv as `**conv_kwargs`:

```toml
[model]
name = "EmbeddedGraphNetwork"
module_path = "dlkit.domain.nn.graph"
hidden_size = 64
num_layers = 3
conv_kind = "gatv2"
heads = 4
```

The legacy GATv2-only presets remain available unchanged:

```toml
[model]
name = "GATv2Projection"
module_path = "dlkit.domain.nn"
hidden_size = 64
num_layers = 3
heads = 4
```
