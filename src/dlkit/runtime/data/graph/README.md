# Runtime Graph Data

`dlkit.runtime.data.graph` owns graph-specific runtime transport types.

Current contents:

- `GraphDict`: flattened dict representation for graph pipelines
- `GraphInput`: accepted graph input union for graph runtime flows
- re-exported PyG transport types from `types.py`

This package is runtime data infrastructure, not domain logic.
