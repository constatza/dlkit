"""Strict, functional patching for immutable Pydantic models.

This module provides pure-function utilities for producing new Pydantic model
instances from an overlay of overrides. It supports:

- Dotted-key overrides: ``"a.b.c"`` is expanded to ``{"a": {"b": {"c": ...}}}``
- Mixed overrides: dotted and nested dicts can be combined in one call
- Strict merge semantics: any key collision raises ``ValueError`` (no silent overwrites)
- Validated updates: patch values are validated against the field's annotation
  before being passed to ``model_copy()``, preventing Pydantic v2 "dict pollution"
- Extra-field support: ``extra="allow"`` models can patch their extra fields

Public API::

    patch_model(model, overrides)  # compile + apply in one call
    apply_patch(model, patch)  # apply a pre-compiled nested patch
    compile_mixed_overrides(overrides)  # compile dotted/nested mix → nested dict

Internal helpers are also exported for independent unit testing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, TypeVar, cast

from pydantic import BaseModel, TypeAdapter

M = TypeVar("M", bound=BaseModel)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def is_mapping(x: Any) -> bool:
    """Return ``True`` for any dict-like object.

    Args:
        x: Value to inspect.

    Returns:
        bool: ``True`` iff ``x`` is a ``Mapping`` instance.
    """
    return isinstance(x, Mapping)


def split_overrides(
    overrides: Mapping[str, Any],
    *,
    sep: str = ".",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split overrides into *(dotted, nested)* buckets.

    Args:
        overrides: Mixed overrides mapping.
        sep: Dotted-path separator (default ``"."``).

    Returns:
        tuple[dict, dict]: ``(dotted, nested)`` as plain dicts.
            *dotted* contains keys that include ``sep``;
            *nested* contains all other keys.
    """
    if not overrides:
        return {}, {}

    is_dotted_key = lambda k: isinstance(k, str) and (sep in k)  # noqa: E731

    dotted = {k: v for k, v in overrides.items() if is_dotted_key(k)}
    nested = {k: v for k, v in overrides.items() if not is_dotted_key(k)}
    return dotted, nested


# ---------------------------------------------------------------------------
# Dotted compilation (pure from the caller's perspective)
# ---------------------------------------------------------------------------


def insert_path(
    root: dict[str, Any],
    *,
    key: str,
    value: Any,
    sep: str = ".",
) -> dict[str, Any]:
    """Insert a dotted path into *root*, returning *root* for explicit dataflow.

    Mutates *root* in place but returns it so callers can compose / pipeline.

    Strict semantics:

    - Cannot traverse into a leaf: an existing value at a parent segment must
      be a ``dict`` (or absent).
    - Cannot overwrite a subtree with a leaf: an existing value at the leaf
      segment must not be a ``dict``.

    Args:
        root: Accumulator dict (tree built so far).
        key: Dotted key such as ``"a.b.c"``.
        value: Value to place at the leaf.
        sep: Path separator (default ``"."``).

    Returns:
        dict: The same *root* dict, mutated.

    Raises:
        ValueError: On empty / invalid keys or structural conflicts.
    """
    if not key:
        raise ValueError("Override key must be non-empty.")
    if not sep:
        raise ValueError("Separator `sep` must be non-empty.")

    parts = key.split(sep)
    if any(p == "" for p in parts):
        raise ValueError(f"Invalid dotted key (empty segment): {key!r}")

    cursor: dict[str, Any] = root
    *parents, leaf = parts

    for p in parents:
        existing = cursor.get(p)

        match existing:
            case None:
                nxt: dict[str, Any] = {}
                cursor[p] = nxt
                cursor = nxt
            case dict() as subtree:
                cursor = subtree
            case _:
                raise ValueError(
                    f"Conflict: {key!r} traverses into {p!r}, but {p!r} is already a leaf."
                )

    match cursor.get(leaf):
        case dict():
            raise ValueError(
                f"Conflict: {key!r} sets leaf {leaf!r}, but {leaf!r} is already a parent."
            )
        case _:
            cursor[leaf] = value

    return root


def compile_dotted_overrides(
    dotted: Mapping[str, Any],
    *,
    sep: str = ".",
) -> dict[str, Any]:
    """Compile dotted overrides into a nested dict patch.

    Example::

        {"a.b": 1, "a.c": 2}  →  {"a": {"b": 1, "c": 2}}

    Strict semantics — any of these is illegal:

    - ``{"a": 1, "a.b": 2}``
    - ``{"a.b": 1, "a": 2}``
    - no silent overwrites (collisions become errors)

    Args:
        dotted: Mapping of dotted keys to values.
        sep: Path separator (default ``"."``).

    Returns:
        dict: Nested patch dict.

    Raises:
        ValueError: On invalid keys or structural conflicts.
    """
    if not dotted:
        return {}

    bad_keys = [k for k in dotted.keys() if not isinstance(k, str)]
    if bad_keys:
        raise ValueError(f"Dotted override keys must be strings. Bad keys: {bad_keys!r}")

    root: dict[str, Any] = {}
    for k, v in dotted.items():
        insert_path(root, key=cast(str, k), value=v, sep=sep)
    return root


# ---------------------------------------------------------------------------
# Strict patch merge (pure)
# ---------------------------------------------------------------------------


def strict_merge_patches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Deep-merge two patch dicts, raising on **any** overlap or collision.

    Rules:

    - Both sides have the same key and both values are ``dict`` → recurse.
    - Both sides have the same key and either value is not a ``dict`` → ``ValueError``.
    - Key present in only one side → carry through unchanged.

    This is a strict merge: there is no "right wins" or "left wins" priority.
    Any collision is an error.

    Args:
        left: Patch dict A.
        right: Patch dict B.

    Returns:
        dict: New merged patch dict.

    Raises:
        ValueError: On collisions or structural conflicts.
    """
    if not left:
        return dict(right)
    if not right:
        return dict(left)

    keys = set(left.keys()) | set(right.keys())

    def merge_one(k: str) -> tuple[str, Any]:
        in_left = k in left
        in_right = k in right

        if in_left and not in_right:
            return k, left[k]
        if in_right and not in_left:
            return k, right[k]

        lval = left[k]
        rval = right[k]

        match (lval, rval):
            case (dict() as ld, dict() as rd):
                return k, strict_merge_patches(
                    cast(Mapping[str, Any], ld),
                    cast(Mapping[str, Any], rd),
                )
            case _:
                raise ValueError(
                    f"Override collision at {k!r}: cannot merge "
                    f"{type(lval).__name__} with {type(rval).__name__}."
                )

    return dict(map(merge_one, keys))


def compile_mixed_overrides(
    overrides: Mapping[str, Any],
    *,
    sep: str = ".",
) -> dict[str, Any]:
    """Compile mixed overrides (dotted + nested) into a single nested patch dict.

    Strict policy: if the dotted-compiled patch overlaps with the nested patch
    at any key, a ``ValueError`` is raised.

    Args:
        overrides: Mixed overrides mapping (plain Python dicts accepted).
        sep: Dotted separator (default ``"."``).

    Returns:
        dict: Nested patch dict.

    Raises:
        ValueError: On dotted-compilation conflicts or merge collisions.
    """
    if not overrides:
        return {}

    dotted, nested = split_overrides(overrides, sep=sep)
    compiled = compile_dotted_overrides(dotted, sep=sep) if dotted else {}

    match (bool(compiled), bool(nested)):
        case (False, False):
            return {}
        case (True, False):
            return compiled
        case (False, True):
            return dict(nested)
        case (True, True):
            return strict_merge_patches(compiled, nested)


# ---------------------------------------------------------------------------
# Validated model patching (no dict pollution)
# ---------------------------------------------------------------------------


def iter_validated_updates(
    model: BaseModel,
    patch: Mapping[str, Any],
    *,
    revalidate: bool,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(field_name, validated_value)`` for safe ``model_copy(update=...)``.

    Core behaviour:

    - If the current field value is a ``BaseModel`` **and** the patch value is a
      mapping → recurse (nested models stay models).
    - Otherwise → validate / coerce the patch value against the field annotation
      via ``TypeAdapter`` (validates containers recursively).
    - For models with ``extra="allow"``: extra fields (those not in
      ``model_fields``) are passed through as-is without annotation validation,
      because their types are not tracked by Pydantic.

    Args:
        model: Target model instance.
        patch: Nested patch dict.
        revalidate: Forwarded to recursive ``apply_patch`` calls.

    Yields:
        tuple[str, Any]: ``(field_name, validated_value)`` pairs.

    Raises:
        KeyError: Unknown field name on a model that does not allow extras.
        pydantic.ValidationError: On validation failures.
    """
    if not patch:
        return

    allows_extras = model.model_config.get("extra") == "allow"
    model_fields = type(model).model_fields

    for field_name, patch_value in patch.items():
        if field_name in model_fields:
            field_info = model_fields[field_name]
            current_value = getattr(model, field_name)

            match (current_value, patch_value):
                case (BaseModel() as nested_model, dict() | Mapping() as nested_patch):
                    # Recurse to keep nested model types intact.
                    yield (
                        field_name,
                        apply_patch(
                            cast(BaseModel, nested_model),
                            cast(Mapping[str, Any], nested_patch),
                            revalidate=revalidate,
                        ),
                    )
                case _:
                    # Rebuild annotated type to include field-level constraints
                    # (e.g. PositiveInt = Annotated[int, Gt(0)]).
                    # field_info.annotation is the raw type; field_info.metadata
                    # carries constraints like Gt/Ge that TypeAdapter needs.
                    if field_info.metadata:
                        from typing import Annotated

                        annotation = Annotated[tuple([field_info.annotation, *field_info.metadata])]
                    else:
                        annotation = field_info.annotation
                    adapter = TypeAdapter(annotation)
                    yield field_name, adapter.validate_python(patch_value)

        elif allows_extras:
            # Extra fields have no annotation — pass through as-is.
            yield field_name, patch_value

        else:
            raise KeyError(f"Unknown field {field_name!r} for {model.__class__.__name__}")


def apply_patch(model: M, patch: Mapping[str, Any], *, revalidate: bool = True) -> M:
    """Apply a nested patch to a Pydantic model, returning a **new** instance.

    Prevents Pydantic v2 "dict pollution" by validating patch values *before*
    calling ``model_copy(update=...)``.  ``model_copy`` copies the model's
    ``__dict__`` directly (no ``model_dump`` / ``model_validate`` round-trip),
    so fields marked ``exclude=True`` are preserved in the copy.

    Args:
        model: Base model instance.
        patch: Nested patch dict (plain Python dicts accepted).
        revalidate: If ``True``, performs a full-model validation pass on the
            result without serialising to ``dict`` (uses ``from_attributes=True``
            so excluded fields survive).

    Returns:
        M: New model instance of the same concrete type.

    Raises:
        KeyError: Unknown field on a non-extra model.
        pydantic.ValidationError: On invalid patched values or revalidation failure.
    """
    if not patch:
        return cast(M, model.model_copy(deep=True))

    updates = dict(iter_validated_updates(model, patch, revalidate=revalidate))
    new_model = cast(M, model.model_copy(update=updates, deep=True))

    if not revalidate:
        return new_model

    # Full-model validation without serialising to dict — excluded fields survive.
    validated = TypeAdapter(model.__class__).validate_python(new_model, from_attributes=True)
    return cast(M, validated)


def patch_model(
    model: M,
    overrides: Mapping[str, Any],
    *,
    sep: str = ".",
    revalidate: bool = True,
) -> M:
    """Public entry point: compile mixed overrides strictly, then apply to *model*.

    Args:
        model: Base model instance.
        overrides: Mixed overrides (supports dotted keys and plain nested dicts).
        sep: Dotted-path separator (default ``"."``).
        revalidate: Passed through to :func:`apply_patch`.

    Returns:
        M: New model instance with overrides applied.

    Raises:
        ValueError: On override key conflicts.
        KeyError: Unknown field name.
        pydantic.ValidationError: On type mismatches.
    """
    if not overrides:
        return cast(M, model.model_copy(deep=True))

    patch = compile_mixed_overrides(overrides, sep=sep)
    return apply_patch(model, patch, revalidate=revalidate)

