from pathlib import Path
from pydantic import Field, DirectoryPath, FilePath, model_validator
from .base_settings import BaseSettings


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
    ckpt_path: str | None = Field(
        default=None, description="Path to the checkpoint file."
    )

    @model_validator(mode="before")
    def populate_predictions(cls, data):
        if "output" not in data:
            return data
        if "predictions" not in data:
            data["predictions"] = f"{data['output']}/predictions.npy"
        return data

    @model_validator(mode="before")
    def populate_targets(cls, data):
        if "targets" not in data:
            data["targets"] = data["features"]
        return data
