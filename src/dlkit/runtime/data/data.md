# Runtime Data

`dlkit.runtime.data` owns runtime-facing dataset implementations and data
services. It is the canonical home for data-loading concerns that used to sit
under `dlkit.core`.

Current responsibilities:

- dataset-family resolution
- dataset implementations in `datasets/`
- dataset split views in `splits.py`
- shape inference helpers shared by runtime services
