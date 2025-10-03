Minimal End-to-End Training Example

This example demonstrates a tiny, ready-to-run training setup using:
- A flexible dataset with x/y arrays (CSV files)
- A simple custom PyTorch model
- A minimal DLKit configuration

Directory layout
- data/x.csv, data/y.csv: toy arrays (100 samples, 4 → 1)
- model.py: SimpleNet (one hidden layer MLP) that accepts a shape kwarg
- config.toml: flattened DLKit config using the FlexibleDataset and InMemoryModule

Run with the CLI
- uv run dlkit train run examples/minimal_e2e/config.toml --epochs 3

Run with the Python API
- from dlkit.interfaces.api import train
- from dlkit.tooling.config import GeneralSettings
- cfg = GeneralSettings.from_file("examples/minimal_e2e/config.toml")
- result = train(cfg, epochs=3)
- print(result.metrics)

Outputs
- Artifacts and logs are written under examples/minimal_e2e/outputs.

Notes on targets and predictions
- Loss pairing is strict: the pipeline pairs predictions to targets by name.
- With one target and one prediction, a single tensor is auto-paired.
- For autoencoders, set `is_autoencoder = true` on the wrapper (programmatic) to use features as targets when no targets are listed.
