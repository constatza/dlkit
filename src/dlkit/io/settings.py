import copy
from typing import List, Optional, TypeVar, Generic, NoReturn
from pathlib import Path
from pydantic import BaseModel, Field, DirectoryPath, FilePath
import pydantic


Model = TypeVar("Model", bound=BaseModel)

import copy
import typing
from typing import NoReturn, Optional, Iterable, Type, Tuple, Any, Generic, TypeVar

import pydantic
from pydantic import BaseModel

Model = TypeVar("Model", bound=BaseModel)


class IntRange(BaseModel):
    low: int = Field(..., description="Minimum value")
    high: int = Field(..., description="Maximum value")
    step: Optional[int] = Field(default=1, description="Step size (optional)")


class FloatRange(BaseModel):
    low: float = Field(..., description="Minimum value")
    high: float = Field(..., description="Maximum value")
    step: Optional[float] = Field(default=None, description="Step size (optional)")
    log: Optional[bool] = Field(
        default=False, description="If true, sample on a log scale"
    )


class CategoricalStr(BaseModel):
    values: List[str] = Field(..., description="List of possible values")


class CategoricalInt(BaseModel):
    values: List[int] = Field(..., description="List of possible values")


class CategoricalFloat(BaseModel):
    values: List[float] = Field(..., description="List of possible values")


# For a hyperparameter that can either be a fixed value or a range, you can use a union:
IntHyper = int | IntRange | CategoricalInt
FloatHyper = float | FloatRange | CategoricalFloat


class Partial(Generic[Model]):
    """Generate a new class with all attributes optional.

    Notes:
        This will wrap a class inheriting from BaseModel and will recursively
        convert all its attributes and its children's attributes to optionals.

    Example:
        Partial[SomeModel]
        Partial[(SomeModel, ['field_to_exclude'])]
    """

    def __new__(cls, *args: object, **kwargs: object) -> "Partial[Model]":
        raise TypeError("Cannot instantiate abstract Partial class.")

    def __init_subclass__(cls, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError(f"Cannot subclass {cls.__module__}.Partial")

    def __class_getitem__(cls, item: Any) -> type[Model]:
        """
        Convert a model to a partial model with all fields made optional.
        Optionally, you can exclude fields by passing a tuple:

            Partial[(SomeModel, ['exclude_field1', 'exclude_field2'])]
        """
        # Determine wrapped_class and exclude set
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(
                    "When passing a tuple, it must be (Model, exclude_iterable)"
                )
            wrapped_class, exclude_iterable = item
            if not isinstance(exclude_iterable, Iterable):
                raise TypeError("exclude must be an iterable of field names")
            exclude = set(exclude_iterable)
        else:
            wrapped_class = item
            exclude = set()

        def _make_field_optional(
            field: pydantic.fields.FieldInfo,
        ) -> Tuple[Any, pydantic.fields.FieldInfo]:
            tmp_field = copy.deepcopy(field)
            annotation = field.annotation
            # If the field is a BaseModel, then recursively convert its attributes to optionals.
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                tmp_field.annotation = typing.Optional[Partial[annotation]]  # type: ignore[assignment, valid-type]
                tmp_field.default = {}
            else:
                tmp_field.annotation = typing.Optional[field.annotation]  # type: ignore[assignment]
                tmp_field.default = None
            return tmp_field.annotation, tmp_field

        # Create a new model with all fields converted to optionals, excluding specified fields.
        new_fields = {
            field_name: _make_field_optional(field_info)
            for field_name, field_info in wrapped_class.model_fields.items()
            if field_name not in exclude
        }
        return pydantic.create_model(
            f"Partial{wrapped_class.__name__}",
            __base__=wrapped_class,
            __module__=wrapped_class.__module__,
            **new_fields,  # type: ignore
        )


class ConfigModel(BaseModel):

    class Config:
        extra = "allow"


class MLflowServer(ConfigModel):
    host: str = Field(default="127.0.0.1", description="MLflow server host address.")
    port: int = Field(default=5000, description="MLflow server port number.")
    backend_store_uri: str = Field(..., description="URI for the backend store.")
    default_artifact_root: str = Field(..., description="Default artifact root path.")
    tracking_uri: str = Field(..., description="Tracking URI for MLflow.")
    terminate_apps_on_port: bool = Field(
        default=False, description="Whether to terminate apps on this port."
    )


class MLflow(ConfigModel):
    experiment_name: str = Field(
        default="experiment", description="MLflow experiment name."
    )
    server: MLflowServer = Field(..., description="Nested MLflow server settings.")
    enable_checkpointing: bool = Field(
        default=False, description="Whether to enable checkpointing."
    )
    ckpt_path: Optional[str] = Field(
        default=None, description="Path to the checkpoint file."
    )


class Trainer(ConfigModel):
    max_epochs: IntHyper = Field(
        default=100, description="Maximum number of epochs to train for.", gt=0
    )
    gradient_clip_val: Optional[float] = Field(
        default=None, description="Value for gradient clipping (if any)."
    )
    fast_dev_run: bool | int = Field(
        default=False,
        description="Flag for fast development run or number of batches to run in fast dev mode.",
    )


class ModelConfig(Partial[ConfigModel]):
    name: str = Field(..., description="Model name.")
    num_layers: IntHyper = Field(..., description="Number of layers.", gt=0)
    latent_size: IntHyper = Field(..., description="Latent dimension size.", gt=0)
    kernel_size: IntHyper = Field(..., description="Convolution kernel size.", gt=0)
    final_channels: IntHyper = Field(..., description="Final number of channels.", gt=0)
    final_timesteps: IntHyper = Field(
        ..., description="Final number of timesteps.", gt=0
    )


class Optimizer(ConfigModel):
    name: str = Field(default="Adam", description="Optimizer name.")
    lr: Optional[IntHyper] = Field(default=None, description="Learning rate.")
    weight_decay: Optional[float] = Field(
        default=None, description="Optional weight decay."
    )


class Scheduler(ConfigModel):
    name: str = Field(default="ReduceLROnPlateau", description="Scheduler name.")
    factor: float = Field(default=0.8, description="Reduction factor.")
    patience: int = Field(
        default=10,
        description="Number of epochs with no improvement before reducing the LR.",
    )
    min_lr: float = Field(default=1e-5, description="Minimum learning rate.")


class Transforms(ConfigModel):
    name: str = Field(..., description="Name of the transform.")
    dim: List[int] = Field(
        ..., description="List of dimensions to apply the transform on."
    )


class Pruner(ConfigModel):
    name: str = Field(
        default="MedianPruner",
        description="Pruner algorithm name for hyperparameter optimization.",
    )
    n_warmup_steps: int = Field(
        default=20, description="Number of warmup steps before pruning starts."
    )
    interval_steps: int = Field(
        default=1, description="Interval between pruning steps."
    )


class Sampler(ConfigModel):
    name: str = Field(
        default="TPESampler",
        description="Sampler algorithm name for hyperparameter optimization.",
    )
    seed: Optional[int] = Field(
        default=None, description="Optional random seed for reproducibility."
    )


class Datamodule(ConfigModel):
    name: str = Field(..., description="Datamodule name.")
    test_size: float = Field(..., description="Fraction of data used for testing.")


class Dataloader(ConfigModel):
    num_workers: int = Field(default=5, description="Number of worker processes.")
    batch_size: int = Field(default=128, description="Batch size.")
    persistent_workers: bool = Field(
        default=True, description="Whether to use persistent workers."
    )
    pin_memory: bool = Field(default=True, description="Whether to pin memory.")


class Paths(ConfigModel):
    features: FilePath = Field(..., description="Path to the features file.")
    targets: Optional[FilePath] = Field(
        default=None, description="Path to the targets file (if any)."
    )
    predictions: Optional[str] = Field(
        default=None, description="Path to the predictions file."
    )
    input: Optional[DirectoryPath] = Field(default=None, description="Input directory.")
    output: Optional[DirectoryPath] = Field(
        default=None, description="Output directory for generated files."
    )
    idx_split: Optional[FilePath] = Field(
        default=None, description="Path to the index split file."
    )


# Top-level configuration model.
# Use BaseModel here because dynaconf will load the TOML and do interpolation.
# If you later need env variable overrides, change BaseModel to BaseSettings.
class Settings(BaseModel):
    MLFLOW: MLflow
    OPTUNA: Optional[dict] = {}
    TRAINER: Trainer
    MODEL: ModelConfig
    OPTIMIZER: Optimizer
    SCHEDULER: Scheduler
    TRANSFORMS: list[Transforms]
    DATAMODULE: Datamodule
    DATALOADER: Dataloader
    PATHS: Paths
    PRUNER: Optional[Pruner] = None
    SAMPLER: Optional[Sampler] = None
