# Lightning DataModules

`dlkit.runtime.adapters.lightning.datamodules` owns the Lightning-specific
data-module implementations for DLKit.

Current classes:

- `BaseDataModule`: shared Lightning `DataModule` base with split handling
- `InMemoryModule`: array/TensorDict datamodule
- `GraphDataModule`: PyG datamodule
- `TimeSeriesDataModule`: pytorch-forecasting datamodule

Ownership rule:

- datasets live in `dlkit.runtime.data.datasets`
- split helpers live in `dlkit.runtime.data.splits`
- Lightning-specific batching and loader behavior lives here

This package is runtime adapter code, not domain code.
