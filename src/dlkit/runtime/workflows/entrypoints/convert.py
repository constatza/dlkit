"""Runtime-owned checkpoint export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from dlkit.runtime.adapters.lightning.factories import WrapperFactory
from dlkit.runtime.workflows.factories.build_factory import FlexibleBuildStrategy
from dlkit.shared.errors import WorkflowError

from ._settings import WorkflowSettings


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvertResult:
    """Result of a checkpoint export action."""

    output_path: Path
    opset: int
    inputs: list[tuple[int, ...]]


def convert_checkpoint_to_onnx(
    checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    settings: WorkflowSettings | None = None,
    shape: str | None = None,
    opset: int = 17,
    batch_size: int | None = None,
) -> ConvertResult:
    """Export a checkpointed wrapper to ONNX."""
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    _validate_convert_inputs(checkpoint, output, shape=shape, opset=opset, batch_size=batch_size)

    try:
        wrapper = WrapperFactory.create_wrapper_from_checkpoint(str(checkpoint))
        wrapper.eval()

        default_batch = batch_size if batch_size is not None else 1
        input_shapes, inferred_from_config = _parse_or_infer_shapes(shape, settings, default_batch)
        if not input_shapes:
            raise WorkflowError(
                "Could not determine input shape. Provide --shape or a config with dataset.",
                {"workflow": "convert"},
            )

        if inferred_from_config and batch_size is not None:
            batch_dims = {dimensions[0] for dimensions in input_shapes}
            if len(batch_dims) != 1 or next(iter(batch_dims)) != batch_size:
                raise WorkflowError(
                    "Batch size mismatch between provided --batch-size and dataloader batch",
                    {
                        "workflow": "convert",
                        "expected_batch": batch_size,
                        "found_batches": sorted(list(batch_dims)),
                    },
                )

        if len(input_shapes) == 1:
            example_inputs: Any = torch.ones(input_shapes[0], dtype=torch.float32)
            input_names = ["input"]
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
        else:
            example_inputs = tuple(torch.ones(shape, dtype=torch.float32) for shape in input_shapes)
            input_names = [f"input{i}" for i in range(len(input_shapes))]
            dynamic_axes = {name: {0: "batch"} for name in input_names}
            dynamic_axes["output"] = {0: "batch"}

        torch.onnx.export(
            wrapper,
            cast(Any, example_inputs),
            str(output),
            opset_version=opset,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        return ConvertResult(output_path=output, opset=opset, inputs=input_shapes)
    except WorkflowError:
        raise
    except Exception as exc:
        raise WorkflowError(
            f"Conversion failed: {exc}",
            {"workflow": "convert", "error_type": type(exc).__name__},
        ) from exc


def _validate_convert_inputs(
    checkpoint_path: Path,
    output_path: Path,
    *,
    shape: str | None,
    opset: int,
    batch_size: int | None,
) -> None:
    if not checkpoint_path.exists():
        raise WorkflowError(
            f"Checkpoint file not found: {checkpoint_path}",
            {"workflow": "convert", "checkpoint": str(checkpoint_path)},
        )
    if output_path.is_dir():
        raise WorkflowError(
            f"Output path points to a directory, expected file: {output_path}",
            {"workflow": "convert", "output": str(output_path)},
        )
    if opset < 9:
        raise WorkflowError(
            "Unsupported opset version (min 9)",
            {"workflow": "convert", "opset": opset},
        )
    if shape and (batch_size is None or batch_size < 1):
        raise WorkflowError(
            "For --shape, provide --batch-size >= 1 (CLI shapes omit batch)",
            {"workflow": "convert", "batch_size": batch_size},
        )


def _parse_or_infer_shapes(
    shape_spec: str | None,
    settings: WorkflowSettings | None,
    default_batch: int,
) -> tuple[list[tuple[int, ...]], bool]:
    if shape_spec:
        parts = [part.strip() for part in shape_spec.split(";") if part.strip()]
        shapes: list[tuple[int, ...]] = []
        for part in parts:
            dims = [
                dimension.strip()
                for dimension in part.replace("x", ",").split(",")
                if dimension.strip()
            ]
            try:
                parsed_dims = [int(dimension) for dimension in dims]
            except ValueError as exc:
                raise WorkflowError(
                    f"Invalid shape spec: '{part}'",
                    {"shape": shape_spec, "part": part},
                ) from exc
            if len(parsed_dims) < 1:
                raise WorkflowError("Shape must include feature dimensions", {"shape": shape_spec})
            if any(dimension <= 0 for dimension in parsed_dims):
                raise WorkflowError(
                    "All shape dimensions must be positive",
                    {"shape": shape_spec},
                )
            shapes.append(tuple([default_batch, *parsed_dims]))
        return shapes, False

    if settings is None:
        raise WorkflowError(
            "No shape provided and no config available to infer from dataloader",
            {"workflow": "convert"},
        )

    strategy = FlexibleBuildStrategy()
    components = strategy.build(settings)
    datamodule = components.datamodule

    loader = None
    for name in ("predict_dataloader", "val_dataloader", "test_dataloader", "train_dataloader"):
        if hasattr(datamodule, name):
            try:
                candidate = getattr(datamodule, name)()
                loader = candidate
                if loader is not None:
                    break
            except Exception:
                continue
    if loader is None:
        raise WorkflowError(
            "Could not construct a dataloader for shape inference",
            {"workflow": "convert"},
        )

    try:
        batch = next(iter(loader))
    except Exception as exc:
        raise WorkflowError(
            "Failed to get a batch from dataloader",
            {"workflow": "convert", "error": str(exc)},
        ) from exc

    shapes: list[tuple[int, ...]] = []
    if isinstance(batch, dict):
        tensor = batch.get("x")
        if tensor is None:
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break
        if tensor is None:
            raise WorkflowError("Could not find input tensor 'x' in batch", {"workflow": "convert"})
        shapes.append(tuple(int(dimension) for dimension in tensor.shape))
    elif isinstance(batch, (list, tuple)):
        tensor = batch[0]
        if not hasattr(tensor, "shape"):
            raise WorkflowError("First element of batch has no shape", {"workflow": "convert"})
        shapes.append(tuple(int(dimension) for dimension in tensor.shape))
    else:
        if not hasattr(batch, "shape"):
            raise WorkflowError(
                "Batch object is not a Tensor and not a supported container",
                {"workflow": "convert"},
            )
        shapes.append(tuple(int(dimension) for dimension in batch.shape))

    return shapes, True
