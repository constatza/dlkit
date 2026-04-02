# CLI Presenters

This package owns CLI-facing presentation helpers.

## Current Modules
- `protocol.py`: `IResultPresenter`
- `presenter_utils.py`: shared summary and graph-detection helpers
- `array.py`: dense prediction postprocessing
- `graph.py`: graph prediction postprocessing
- `__init__.py`: presenter exports

These helpers stay in `interfaces.cli` because they shape human-facing output,
not domain logic or runtime execution.
