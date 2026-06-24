# Neural Operator Architectures

## Dataset Configuration & Field Roles

Unlike standard regression models (e.g., FFNNs) that consume a single `FEATURE` field, continuous operator models (like DeepONet) map a branch signal to an output operator evaluated at explicit query coordinates.

To successfully construct a DeepONet via the engine's automated contract resolution, the dataset configuration must explicitly distinguish these inputs:
- **Branch Inputs** (the condition or sensor data) must be assigned `FieldRole.FEATURE`.
- **Trunk Inputs** (the continuous coordinates) MUST be assigned `FieldRole.TARGET_COORDINATES`.

If the query coordinates are not explicitly marked as `TARGET_COORDINATES`, the engine will assume a standard multi-input regression problem and fail to initialize the DeepONet.

## Naming conventions

- `branch_shape`: branch sample shape excluding batch
- `trunk_shape`: trunk sample shape excluding batch
- `out_features`: output feature count per query location
- `n_queries`: number of query locations in one batch item
- `trunk_dim`: width of one trunk-coordinate vector
- `trunk_width`: shared latent width on the DeepONet branch/trunk side

Use `branch` and `trunk` consistently for DeepONet data tensors, forward
parameters, and constructor kwargs. Use `spatial_shape` for generic grid
operators and `length` only for 1-D operators.

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
- trunk input: `(B, n_queries, trunk_dim)`
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
- trunk input: `(B, n_queries, trunk_dim)`
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
- trunk input: `(B, n_queries, trunk_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- branch FFNN output: `(B, trunk_width * out_features)`
- trunk FFNN output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `branch_in_features`: flattened branch width
- `branch_in_features = prod(branch_shape)` derived from the first input shape
- common sensor-vector case: `branch_shape = (n_sensors,) -> branch_in_features = n_sensors`
- `trunk_dim = trunk_shape[-1]` derived from the trunk input shape
- `trunk_width`
- `out_features`
- `branch_layers`
- `trunk_layers`

## `FFNNDeepONet`

Input/output dimensions:
- branch input after flattening: `(B, flattened_branch_width)`
- trunk input: `(B, n_queries, trunk_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- branch FFNN output: `(B, trunk_width * out_features)`
- trunk FFNN output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `branch_in_features`: flattened branch width
- `branch_in_features = prod(branch_shape)` derived from the first input shape
- common sensor-vector case: `branch_shape = (n_sensors,) -> branch_in_features = n_sensors`
- `trunk_dim = trunk_shape[-1]` derived from the trunk input shape
- `trunk_width`
- `out_features`
- `branch_hidden_size`
- `branch_num_layers`
- `trunk_hidden_size`
- `trunk_num_layers`

## `EmbeddedDeepONet`

Input/output dimensions:
- branch input after flattening: `(B, flattened_branch_width)`
- trunk input: `(B, n_queries, trunk_dim)`
- output: `(B, n_queries, out_features)`

Architecture dimensions:
- branch FFNN output: `(B, trunk_width * out_features)`
- trunk FFNN output: `(B * n_queries, trunk_width * out_features)`

Constructor dimensions:
- `branch_in_features`: flattened branch width
- `branch_in_features = prod(branch_shape)` derived from the first input shape
- common sensor-vector case: `branch_shape = (n_sensors,) -> branch_in_features = n_sensors`
- `trunk_dim = trunk_shape[-1]` derived from the trunk input shape
- `trunk_width`
- `out_features`
- `branch_hidden_size`
- `branch_num_layers`
- `trunk_hidden_size`
- `trunk_num_layers`
