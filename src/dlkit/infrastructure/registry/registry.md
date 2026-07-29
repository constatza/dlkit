# Registry

`dlkit.infrastructure.registry` provides name-based registration and
resolution for models, datasets, losses, metrics, and datamodules. It has
**no dlkit dependencies beyond `dlkit.common`** — in particular it must not
import `dlkit.domain`, since `infrastructure` and `domain` are sibling
layers per `tach.toml` (both depend on `common` only, neither on the
other).

## Modules

| Module | Responsibility |
|--------|---------------|
| `base.py` | `LockedRegistry[T]` — thread-safe canonical-key → object store with alias resolution and a single optional "forced" selection per registry |
| `public.py` | `register_model`/`register_dataset`/`register_loss`/`register_metric`/`register_datamodule` decorators, `resolve_from_registry`, `get_forced`, `describe_model`, `list_registered_models`/`list_registered_datasets` |
| `resolve.py` | `resolve_component(kind, name, module_path)` — the actual lookup order used by factories |

One `LockedRegistry` instance per `kind` (`_MODELS`, `_DATASETS`, `_LOSSES`,
`_METRICS`, `_DATAMODULES` in `public.py`); `register_model` and friends are
each `_make_register(kind)` bound to one of them. There is exactly one model
registry — `dlkit.nn.DLKitModule` subclasses and plain third-party
`nn.Module`s that opt in are registered into the same `_MODELS` store; see
"Model contract validation" below for why this isn't two separate systems.

## Resolution order

`resolve_component(kind, name, module_path)` (`resolve.py`) tries, in order:

1. A forced selection (`register_*(..., use=True)`) for that `kind`, ignoring `name` entirely.
2. The registry, by canonical name or alias.
3. A dotted-path import (`name`, falling back to `module_path` as the module if `name` is bare) via `infrastructure.utils.general.import_object`.

`resolve_from_registry` raises `KeyError` with a `difflib.get_close_matches`
suggestion when `name` is registered under neither its canonical key nor an
alias — the same suggestion mechanism `describe_model`/`_describe_entry`
use for their own not-found errors.

## Model contract validation

`register_model()` additionally validates the target's forward-kwarg
contract before registering it — `_validate_model_contract` in `public.py`,
gated on `kind == "model"` inside `_make_register` (other kinds are
untouched):

```python
input_spec = getattr(target, "InputSpec", None)      # must exist
field_names = frozenset(input_spec.model_fields)      # may be empty
# if non-empty, every field name must match a forward() parameter
```

This is **structural** (plain `getattr`/duck-typing on `InputSpec` and
`forward`), not nominal — it deliberately does **not** check
`issubclass(target, DLKitModule)`, and does not import `dlkit.domain` at
all. A plain `nn.Module` with an `InputSpec` inner class attached registers
exactly as cleanly as a `dlkit.nn.DLKitModule` subclass; the latter simply
already passed the equivalent check once, earlier, at class-definition time
(`domain.nn.base._DLKitModuleMeta`). See `domain/nn/nn.md` for the
cost/benefit reasoning behind keeping this structural rather than requiring
`DLKitModule` inheritance — the short version is that dlkit's own models
never go through `register_model` at all (they resolve by dotted-path
import), so this is purely the third-party opt-in surface, and forcing an
inheritance dependency there for no added safety (the structural check
gives the identical guarantee) would only hurt interop.

Both this check and `domain.nn.base.validate_model_contract` share the same
underlying reflection primitive, `dlkit.common.forward_contract.
check_forward_kwargs` — see `common/common.md` for why that primitive lives
in `common` and not here or in `domain`.

Raises `dlkit.common.errors.ForwardContractError` — not `ValueError` — on
either a missing `InputSpec` or a field with no matching `forward()`
parameter, naming the available parameters in the message.

## Test isolation

`_reset_for_tests()` (module-level in `public.py`, not exported via
`__all__`) clears every registry's mapping/aliases/forced-key. Tests call it
from a `setup_function()` hook so registrations from one test never leak
into the next.
