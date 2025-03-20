from pydantic import (
    BaseModel,
    Field,
    DirectoryPath,
    FilePath,
    RootModel,
    model_validator,
)
from collections.abc import Sequence
from pathlib import Path


class IntRange(BaseModel):
    low: int = Field(..., description="Minimum value")
    high: int = Field(..., description="Maximum value")
    step: int = Field(default=1, description="Step size (optional)")


class FloatRange(BaseModel):
    low: float = Field(..., description="Minimum value")
    high: float = Field(..., description="Maximum value")
    step: float = Field(default=1, description="Step size (optional)")
    log: bool | None = Field(
        default=False, description="If true, sample on a log scale"
    )


class StrSequence(RootModel):
    root: Sequence[str] = Field(..., description="List of possible values")


class IntSequence(RootModel):
    root: Sequence[int] = Field(..., description="List of possible values")


class FloatSequence(RootModel):
    root: Sequence[float] = Field(..., description="List of possible values")


# For a hyperparameter that can either be a fixed value or a range, you can use a union:
IntHyper = int | IntRange | IntSequence
FloatHyper = float | FloatRange | FloatSequence
StrHyper = str | StrSequence


class BaseSettings(BaseModel):

    class Config:
        extra = "allow"


class MLflowServer(BaseSettings):
    host: str = Field(default="127.0.0.1", description="MLflow server host address.")
    port: int = Field(default=5000, description="MLflow server port number.")
    backend_store_uri: str = Field(..., description="URI for the backend store.")
    default_artifact_root: str = Field(..., description="Default artifact root path.")
    terminate_apps_on_port: bool = Field(
        default=False, description="Whether to terminate apps on this port."
    )


class MLflowClient(BaseSettings):
    experiment_name: str = Field(
        default="experiment", description="MLflow experiment name."
    )
    tracking_uri: str = Field(..., description="Tracking URI for MLflow.")
    enable_checkpointing: bool = Field(
        default=False, description="Whether to enable checkpointing."
    )
    ckpt_path: str | None = Field(
        default=None, description="Path to the checkpoint file."
    )


class MLflow(BaseSettings):
    server: MLflowServer = Field(..., description="MLflow server settings.")
    client: MLflowClient = Field(..., description="MLflow client settings.")


class Trainer(BaseSettings):
    max_epochs: IntHyper = Field(
        default=100,
        description="Maximum number of epochs to train for.",
    )
    gradient_clip_val: float | None = Field(
        default=None, description="Value for gradient clipping (if any)."
    )
    fast_dev_run: bool | int = Field(
        default=False,
        description="Flag for fast development run or number of batches to run in fast dev mode.",
    )
    default_root_dir: DirectoryPath | None = Field(
        default=None, description="Default root directory for the model."
    )
    logger: bool = Field(default=False, description="Whether to log the model.")


class ModelSettings(BaseSettings):
    name: str = Field(..., description="Model namespace path.")
    input_shape: IntSequence | None = Field(
        None, description="Input shape of the model."
    )
    output_shape: IntSequence | None = Field(
        None, description="Output shape of the model."
    )
    num_layers: IntHyper | None = Field(default=None, description="Number of layers.")
    latent_size: IntHyper | None = Field(
        default=None, description="Latent dimension size."
    )
    kernel_size: IntHyper | None = Field(
        default=None, description="Convolution kernel size."
    )
    latent_channels: IntHyper | None = Field(
        default=None, description="Number of latent channels before reduce to vector."
    )
    latent_width: IntHyper | None = Field(
        default=None, description="Latent width before reduce to vector."
    )
    latent_height: IntHyper | None = Field(
        default=None, description="Latent height before reduce to vector."
    )


class Optimizer(BaseSettings):
    name: str = Field(default="Adam", description="Optimizer name.")
    lr: FloatHyper | None = Field(default=None, description="Learning rate.")
    weight_decay: float | None = Field(
        default=None, description="Optional weight decay."
    )


class Scheduler(BaseSettings):
    name: str = Field(default="ReduceLROnPlateau", description="Scheduler name.")
    factor: float = Field(default=0.8, description="Reduction factor.")
    patience: int = Field(
        default=10,
        description="Number of epochs with no improvement before reducing the LR.",
    )
    min_lr: float = Field(default=1e-5, description="Minimum learning rate.")


class Transform(BaseSettings):
    name: str = Field(..., description="Name of the transform.")
    dim: list[int] = Field(
        ..., description="List of dimensions to apply the transform on."
    )


class Pruner(BaseSettings):
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


class Sampler(BaseSettings):
    name: str = Field(
        default="TPESampler",
        description="Sampler algorithm name for hyperparameter optimization.",
    )
    seed: int | None = Field(
        default=None, description="Optional random seed for reproducibility."
    )


class Dataloader(BaseSettings):
    num_workers: int = Field(default=5, description="Number of worker processes.")
    batch_size: int = Field(default=64, description="Batch size.")
    persistent_workers: bool = Field(
        default=True, description="Whether to use persistent workers."
    )
    pin_memory: bool = Field(default=True, description="Whether to pin memory.")


class Datamodule(BaseSettings):
    name: str = Field(..., description="Datamodule name.")
    test_size: float = Field(..., description="Fraction of data used for testing.")
    dataloader: Dataloader = Field(..., description="Dataloader settings.")
    transforms: Sequence[Transform] = Field(
        ..., description="List of transforms to apply."
    )


class Paths(BaseSettings):
    features: FilePath = Field(..., description="Path to the features file.")
    targets: FilePath | None = Field(
        default=None, description="Path to the targets file (if any)."
    )
    input: DirectoryPath | None = Field(default=None, description="Input directory.")
    output: DirectoryPath | None = Field(
        default=None, description="Output directory for generated files."
    )
    predictions: Path | None = Field(
        default=None, description="Path to the (future) predictions file."
    )

    # !! idx split default value must be None !!
    idx_split: FilePath | None = Field(
        default=None, description="Path to the index split file."
    )

    @model_validator(mode="before")
    def populate_predictions(cls, values):
        if values["predictions"] is None and values["output"] is not None:
            values["predictions"] = values["output"] / "predictions.npy"
        return values


class Optuna(BaseSettings):
    sampler: Sampler | None = Field(default=None, description="Optuna sampler.")
    pruner: Pruner | None = Field(default=None, description="Optuna pruner.")


# Top-level configuration model.
# Use BaseModel here because dynaconf will load the TOML and do interpolation.
# If you later need env variable overrides, change BaseModel to BaseSettings.
class Settings(BaseModel):
    MLFLOW: MLflow
    OPTUNA: Optuna
    TRAINER: Trainer
    MODEL: ModelSettings
    OPTIMIZER: Optimizer
    SCHEDULER: Scheduler
    DATAMODULE: Datamodule
    PATHS: Paths
    PRUNER: Pruner | None = None
    SAMPLER: Sampler | None = None
