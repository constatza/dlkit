"""Dense prediction presentation utilities for the CLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from rich.pretty import Pretty

from .presenter_utils import NotStackableError


def to_numpy(obj: Any) -> Any:
    """Recursively convert tensors into numpy arrays."""

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        return {key: to_numpy(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_numpy(value) for value in obj]
    return obj


def stack_batches(
    batches: Sequence[Any], *, keys: Sequence[str] | None = None, mode: str = "list"
) -> Any:
    """Stack a sequence of dense batches into arrays."""

    if not batches:
        return []

    first = batches[0]

    try:
        import torch

        if isinstance(first, (torch.Tensor, np.ndarray)):
            return _stack_list_tensors(batches, mode=mode)
    except Exception:
        if isinstance(first, np.ndarray):
            return _stack_list_tensors(batches, mode=mode)

    if isinstance(first, Mapping):
        all_keys = set().union(*(batch.keys() for batch in batches))
        use_keys = (
            [key for key in keys if key in all_keys] if keys is not None else sorted(all_keys)
        )

        stacked: dict[str, Any] = {}
        for key in use_keys:
            values = [batch[key] for batch in batches if key in batch]
            try:
                stacked[key] = _stack_list_tensors(values, mode=mode)
            except NotStackableError:
                stacked[key] = [to_numpy(value) for value in values]
        return stacked

    return np.asarray(batches)


def to_plot_data(
    preds: Sequence[Any] | Mapping[str, Any] | Any,
    targets: Sequence[Any] | Mapping[str, Any] | None = None,
) -> Any:
    """Convert dense predictions and optional targets into plot-ready arrays."""

    if isinstance(preds, list):
        try:
            return stack_batches(preds, mode="stack")
        except NotStackableError:
            return stack_batches(preds, mode="list")

    if isinstance(preds, Mapping):
        result: dict[str, Any] = {}
        for key, value in preds.items():
            if isinstance(value, list):
                try:
                    result[str(key)] = stack_batches(value, mode="stack")
                except NotStackableError:
                    result[str(key)] = stack_batches(value, mode="list")
            else:
                result[str(key)] = to_numpy(value)
        if targets is not None:
            if isinstance(targets, Mapping):
                result["targets"] = {key: to_numpy(value) for key, value in targets.items()}
            else:
                result["tgt/y"] = to_numpy(targets)
        return result

    return to_numpy(preds)


def _stack_list_tensors(items: Sequence[Any], *, mode: str) -> np.ndarray | list[np.ndarray]:
    """Stack a list of tensors or arrays according to the requested mode."""

    try:
        import torch

        tensors: list[torch.Tensor] = []
        for item in items:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, np.ndarray):
                tensors.append(torch.from_numpy(item))
            else:
                raise NotStackableError(
                    f"Unsupported type {type(item).__name__} for tensor stacking"
                )

        if mode == "list":
            return [tensor.detach().cpu().numpy() for tensor in tensors]

        if mode == "stack":
            base_shape = tensors[0].shape[1:]
            for tensor in tensors[1:]:
                if tensor.shape[1:] != base_shape:
                    raise NotStackableError(
                        f"Cannot stack: shape mismatch {tensor.shape} vs (*, {base_shape})"
                    )
            return torch.cat(tensors, dim=0).detach().cpu().numpy()

        if mode == "pad":
            max_len = max(int(tensor.shape[0]) for tensor in tensors)
            padded: list[torch.Tensor] = []
            for tensor in tensors:
                if int(tensor.shape[0]) == max_len:
                    padded.append(tensor)
                    continue
                pad_len = max_len - int(tensor.shape[0])
                pad = torch.zeros(
                    (pad_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
                )
                padded.append(torch.cat([tensor, pad], dim=0))
            return torch.stack(padded, dim=0).detach().cpu().numpy()

        raise ValueError(f"Unknown mode: {mode}")
    except NotStackableError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise NotStackableError(str(exc)) from exc


class ArrayResultPresenter:
    """Simple dense-result presenter implementing the CLI presenter protocol."""

    def present(self, result: Any, console) -> None:
        console.print(Pretty(to_plot_data(result)))
