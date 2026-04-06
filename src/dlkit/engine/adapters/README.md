# Runtime Adapters

`dlkit.engine.adapters` contains framework-facing runtime integrations.

`adapters/lightning/` remains nested because the Lightning integration is a
subsystem of its own with wrappers, datamodules, callbacks, checkpoint helpers,
and protocol/composition utilities.

Keeping that code under `lightning/` preserves a clean boundary between
framework-agnostic runtime orchestration and the concrete Lightning adapter
implementation.
