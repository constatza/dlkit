"""Model conversion command (e.g., checkpoint -> ONNX)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.runtime.workflows.factories.build_factory import FlexibleBuildStrategy
from dlkit.core.models.wrappers.factories import WrapperFactory
from .base import BaseCommand


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvertCommandInput:
    """Input dataflow for model conversion/export."""

    checkpoint_path: Path | str
    output_path: Path | str
    # Optional: either provide a shape (comma-separated dims; multiple inputs via ';')
    # or pass a config so we can infer from dataset.
    shape: str | None = None
    opset: int = 17
    batch_size: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvertResult:
    """Result of a conversion/export action."""

    output_path: Path
    opset: int
    inputs: list[tuple[int, ...]]


class ConvertCommand(BaseCommand[ConvertCommandInput, ConvertResult]):
    """Command to convert a checkpointed model to ONNX."""

    def __init__(self, command_name: str = "convert") -> None:
        super().__init__(command_name)

    def validate_input(
        self, input_data: ConvertCommandInput, settings: BaseSettingsProtocol | None
    ) -> None:
        # Basic path checks
        cp = Path(input_data.checkpoint_path)
        if not cp.exists():
            raise WorkflowError(
                f"Checkpoint file not found: {cp}",
                {"command": self.command_name, "checkpoint": str(cp)},
            )
        op = Path(input_data.output_path)
        if op.is_dir():
            raise WorkflowError(
                f"Output path points to a directory, expected file: {op}",
                {"command": self.command_name, "output": str(op)},
            )
        if input_data.opset < 9:
            raise WorkflowError(
                "Unsupported opset version (min 9)",
                {"command": self.command_name, "opset": input_data.opset},
            )
        # Batch size only required/validated when using --shape (CLI path)
        if input_data.shape and (input_data.batch_size is None or input_data.batch_size < 1):
            raise WorkflowError(
                "For --shape, provide --batch-size >= 1 (CLI shapes omit batch)",
                {"command": self.command_name, "batch_size": input_data.batch_size},
            )

    def execute(
        self,
        input_data: ConvertCommandInput,
        settings: BaseSettingsProtocol | None,
        **kwargs: Any,
    ) -> ConvertResult:
        try:
            self.validate_input(input_data, settings)

            checkpoint_path = Path(input_data.checkpoint_path)
            output_path = Path(input_data.output_path)

            # Load wrapper from checkpoint
            wrapper = WrapperFactory.create_wrapper_from_checkpoint(str(checkpoint_path))
            wrapper.eval()

            # Determine input shapes
            default_batch = input_data.batch_size if input_data.batch_size is not None else 1
            input_shapes, inferred_from_cfg = self._parse_or_infer_shapes(
                input_data.shape, settings, default_batch
            )
            if not input_shapes:
                raise WorkflowError(
                    "Could not determine input shape. Provide --shape or a config with dataset.",
                    {"command": self.command_name},
                )

            # Batch-size consistency checks
            if inferred_from_cfg and input_data.batch_size is not None:
                # Ensure dataloader batch matches provided batch-size
                batch_dims = {s[0] for s in input_shapes}
                if len(batch_dims) != 1 or next(iter(batch_dims)) != input_data.batch_size:
                    raise WorkflowError(
                        "Batch size mismatch between provided --batch-size and dataloader batch",
                        {
                            "command": self.command_name,
                            "expected_batch": input_data.batch_size,
                            "found_batches": sorted(list(batch_dims)),
                        },
                    )

            # Build example inputs (Tensor or tuple of Tensors)
            example_inputs: Any
            if len(input_shapes) == 1:
                example_inputs = torch.ones(input_shapes[0], dtype=torch.float32)
                input_names = ["input"]
                dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
            else:
                # Multiple inputs supported by ONNX; wrapper must accept them.
                example_seq = [torch.ones(s, dtype=torch.float32) for s in input_shapes]
                example_inputs = tuple(example_seq)
                input_names = [f"input{i}" for i in range(len(input_shapes))]
                dynamic_axes = {name: {0: "batch"} for name in input_names}
                dynamic_axes["output"] = {0: "batch"}

            # Export via torch.onnx.export
            torch.onnx.export(
                wrapper,
                example_inputs,
                str(output_path),
                opset_version=input_data.opset,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

            return ConvertResult(
                output_path=output_path, opset=input_data.opset, inputs=input_shapes
            )

        except WorkflowError:
            raise
        except Exception as e:
            raise WorkflowError(
                f"Conversion failed: {e}",
                {"command": self.command_name, "error_type": type(e).__name__},
            ) from e

    def _parse_or_infer_shapes(
        self, shape_spec: str | None, settings: BaseSettingsProtocol | None, default_batch: int
    ) -> tuple[list[tuple[int, ...]], bool]:
        """Parse a user shape spec or infer from config dataloader.

        Returns (shapes, inferred_from_config).
        Shapes must include batch dimension.
        """
        if shape_spec:
            parts = [p.strip() for p in shape_spec.split(";") if p.strip()]
            shapes: list[tuple[int, ...]] = []
            for p in parts:
                dims = [d.strip() for d in p.replace("x", ",").split(",") if d.strip()]
                try:
                    idims = [int(d) for d in dims]
                except ValueError as e:
                    raise WorkflowError(
                        f"Invalid shape spec: '{p}'",
                        {"shape": shape_spec, "part": p},
                    ) from e
                if len(idims) < 1:
                    raise WorkflowError(
                        "Shape must include feature dimensions", {"shape": shape_spec}
                    )
                if any(d <= 0 for d in idims):
                    raise WorkflowError(
                        "All shape dimensions must be positive", {"shape": shape_spec}
                    )
                # Prefix batch dimension for CLI-provided feature dims
                shapes.append(tuple([default_batch, *idims]))
            return shapes, False

        # No explicit shape: require valid settings to infer from dataloader
        if settings is None:
            raise WorkflowError(
                "No shape provided and no config available to infer from dataloader",
                {"command": self.command_name},
            )

        # Build via strategy and get a batch from a dataloader
        strategy = FlexibleBuildStrategy()
        comps = strategy.build(settings)
        dm = comps.datamodule

        loader = None
        for name in ("predict_dataloader", "val_dataloader", "test_dataloader", "train_dataloader"):
            if hasattr(dm, name):
                try:
                    candidate = getattr(dm, name)()
                    loader = candidate
                    if loader is not None:
                        break
                except Exception:
                    continue
        if loader is None:
            raise WorkflowError(
                "Could not construct a dataloader for shape inference",
                {"command": self.command_name},
            )

        try:
            batch = next(iter(loader))
        except Exception as e:
            raise WorkflowError("Failed to get a batch from dataloader", {"error": str(e)}) from e

        # Extract input tensor shape(s) including batch
        import torch

        shapes: list[tuple[int, ...]] = []
        if isinstance(batch, dict):
            x = batch.get("x")
            if x is None:
                # Fallback to first tensor-like
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        x = v
                        break
            if x is None:
                raise WorkflowError("Could not find input tensor 'x' in batch", {})
            shapes.append(tuple(int(d) for d in x.shape))
        elif isinstance(batch, (list, tuple)):
            x = batch[0]
            if not hasattr(x, "shape"):
                raise WorkflowError("First element of batch has no shape", {})
            shapes.append(tuple(int(d) for d in x.shape))
        else:
            # Single tensor
            if not hasattr(batch, "shape"):
                raise WorkflowError(
                    "Batch object is not a Tensor and not a supported container", {}
                )
            shapes.append(tuple(int(d) for d in batch.shape))

        return shapes, True
