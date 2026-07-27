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
| `composites.py` | `EmbeddedHyperGraphNetwork`/`EmbeddedMoEGraphNetwork` — Hyper-Connection and Sparse-MoE composites, generic over `conv_kind` |
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
| `EmbeddedHyperGraphNetwork` | `GProjection` | Conv sublayer fixed `False` (outer Hyper-Connection provides the residual path) | any (constructor param) |
| `EmbeddedMoEGraphNetwork` | `GProjection` | Conv sublayer fixed `True` (no outer wrapper providing residual) | any (constructor param) |

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
- `GraphHyperConnection`/`GraphHyperSequential` wrap graph modules with multi-lane residual mixing and forward `edge_index`/`edge_attr` per lane.
- `SparseMoE` and `TopKRouter` can be used on graph node features before or after message passing.

`EmbeddedHyperGraphNetwork` and `EmbeddedMoEGraphNetwork` (`graph/composites.py`)
are the concrete composites built from these primitives, the GNN analog of the
FFNN family's `EmbeddedHyperFFNN`/`EmbeddedMoEFFNN`. Both are `GProjection`
subclasses using the same injected-`message_module` pattern as
`EmbeddedGraphNetwork`; they get `from_context` for free from
`BaseGraphNetwork`, and neither has a public `residual` bool for the conv
sublayer since its residual-ness is an implementation detail of the wrapping
mechanism, not a caller-facing choice:

- **`EmbeddedHyperGraphNetwork`**: message module is a `GraphHyperSequential`
  wrapping single-layer conv adapters. The wrapped conv adapters are always
  built with `residual=False`. `GraphHyperConnection`'s lane-mixing matrices
  are identity-biased at init specifically so a *plain, non-residual* wrapped
  module starts as an exact multi-lane residual identity — if the conv
  adapter also had its own residual (GATv2's native one, or GCN/SAGE's
  adapter-level `x + conv(...)`), the stack would get residual-on-residual,
  breaking that identity-at-init property. Raw conv adapters apply no
  activation on their own (the caller loop applies it, same as
  `_GraphMessageBase.forward`), so each layer is wrapped in a private
  `_ActivatedGraphConv` to restore the post-conv nonlinearity that
  `GraphHyperSequential` itself does not apply.
- **`EmbeddedMoEGraphNetwork`**: message module is a private `_GraphMoEBody`
  that interleaves a graph-conv sublayer and a `SparseMoE` FFN sublayer per
  layer, Transformer-block style (two *independently* residual sublayers).
  Here the conv sublayer is always built with `residual=True`, since there is
  no outer Hyper-Connection wrapper supplying a residual path for it — it
  needs its own. Expert blocks are built via `make_dense_block`
  (`block_kind`/`linear_kind` configurable, e.g. `"swiglu"`), reusing the same
  block-strategy factory as the FFNN family rather than reinventing expert
  types.

  `EmbeddedMoEGraphNetwork` intentionally does **not** support a
  `return_stats=True` tuple-returning `forward()`, unlike `EmbeddedMoEFFNN`.
  `GProjection`/`ProjectionNetwork.forward()` has a fixed contract that always
  returns a plain `Tensor` (`_apply_message_module` calls
  `self._out_proj(x)` on the message module's output, which would break if
  that output were ever a tuple). Widening that shared contract is out of
  scope and risky — it is relied on by `GraphHyperConnection`/
  `GraphHyperSequential` and every other graph composite. Every `SparseMoE`
  layer inside `EmbeddedMoEGraphNetwork` is always called with
  `return_stats=False`, and no `return_stats` parameter is exposed on its
  constructor or `forward()`. If routing diagnostics are ever needed, reach
  into the private `_message_module` structure directly (the same convention
  already used by this package's own tests) rather than widening the public
  `forward()` contract.

There is no `ScaledEmbeddedHyperGraphNetwork`/`ScaledEmbeddedMoEGraphNetwork`
counterpart for either class — out of scope for this pair; either would
follow the same `ScaledGProjection`-basing pattern as
`ScaledEmbeddedGraphNetwork` if ever needed.

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

`EmbeddedHyperGraphNetwork` adds `num_lanes` (Hyper-Connection residual lanes):

```toml
[model]
name = "EmbeddedHyperGraphNetwork"
module_path = "dlkit.domain.nn.graph"
hidden_size = 64
num_layers = 3
conv_kind = "gatv2"
num_lanes = 2
heads = 4
```

`EmbeddedMoEGraphNetwork` adds `num_experts`/`top_k` and the usual
`block_kind`/`linear_kind` expert knobs:

```toml
[model]
name = "EmbeddedMoEGraphNetwork"
module_path = "dlkit.domain.nn.graph"
hidden_size = 64
num_layers = 3
conv_kind = "gcn"
num_experts = 4
top_k = 2
block_kind = "swiglu"
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
