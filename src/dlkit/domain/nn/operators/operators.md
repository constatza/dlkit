# Neural Operator Architectures

## Naming conventions

- `branch_shape`: branch sample shape excluding batch
- `query_shape`: query sample shape excluding batch
- `out_features`: output feature count per query location
- `n_queries`: number of query locations in one batch item
- `query_dim`: width of one query-coordinate vector
- `trunk_width`: shared latent width on the DeepONet branch/trunk side

Use `query` for DeepONet data tensors and tensor shapes. Use `trunk` only for
the network side. Use `spatial_shape` for generic grid operators and `length`
only for 1-D operators.

## Shared interfaces

### `IOperatorNetwork`

Input/output dimensions:
- exposes `out_features`

### `IGridOperator`

Input/output dimensions:
- input: `(B, in_channels, *spatial_shape)`
- output: `(B, out_channels, *spatial_shape)`

### `IQueryOperator`

Input/output dimensions:
- branch input: `(B, *branch_shape)`
- query input: `(B, n_queries, query_dim)`
- output: `(B, n_queries, out_features)`

## `FourierNeuralOperator1d`

Input/output dimensions:
- input: `(B, in_channels, length)`
- output: `(B, out_channels, length)`

Architecture dimensions:
- lifting: `(B, in_channels, length) -> (B, width, length)`
- body: `(B, width, length) -> (B, width, length)`
- projection: `(B, width, length) -> (B, out_channels, length)`

Constructor dimensions:
- `in_channels`
- `out_channels`
- `width`
- `n_modes`
- `n_layers`

## `DeepONet`

Input/output dimensions:
- branch input: `(B, *branch_shape)`
- query input: `(B, n_queries, query_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- `branch_net` output: `(B, trunk_width * out_features)`
- `trunk_net` output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `trunk_width`
- `out_features`

## `VarWidthDeepONet`

Input/output dimensions:
- branch input after flattening: `(B, flattened_branch_width)`
- query input: `(B, n_queries, query_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- branch FFNN output: `(B, trunk_width * out_features)`
- trunk FFNN output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `branch_in_features`: flattened branch width
- `branch_in_features = prod(branch_shape)` when built from `BranchTrunkSpec`
- common sensor-vector case: `branch_shape = (n_sensors,) -> branch_in_features = n_sensors`
- `query_dim = query_shape[-1]` when built from `BranchTrunkSpec`
- `trunk_width`
- `out_features`
- `branch_layers`
- `trunk_layers`

## `FFNNDeepONet`

Input/output dimensions:
- branch input after flattening: `(B, flattened_branch_width)`
- query input: `(B, n_queries, query_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- branch FFNN output: `(B, trunk_width * out_features)`
- trunk FFNN output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `branch_in_features`: flattened branch width
- `branch_in_features = prod(branch_shape)` when built from `BranchTrunkSpec`
- common sensor-vector case: `branch_shape = (n_sensors,) -> branch_in_features = n_sensors`
- `query_dim = query_shape[-1]` when built from `BranchTrunkSpec`
- `trunk_width`
- `out_features`
- `branch_hidden_size`
- `branch_num_layers`
- `trunk_hidden_size`
- `trunk_num_layers`

## `EmbeddedDeepONet`

Input/output dimensions:
- branch input after flattening: `(B, flattened_branch_width)`
- query input: `(B, n_queries, query_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- branch FFNN output: `(B, trunk_width * out_features)`
- trunk FFNN output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `branch_in_features`: flattened branch width
- `branch_in_features = prod(branch_shape)` when built from `BranchTrunkSpec`
- common sensor-vector case: `branch_shape = (n_sensors,) -> branch_in_features = n_sensors`
- `query_dim = query_shape[-1]` when built from `BranchTrunkSpec`
- `trunk_width`
- `out_features`
- `branch_hidden_size`
- `branch_num_layers`
- `trunk_hidden_size`
- `trunk_num_layers`
