from pathlib import Path
from pydantic import Field, DirectoryPath, FilePath, model_validator
from .base_settings import BaseSettings
from pydantic import field_validator, ValidationInfo


class PathSettings(BaseSettings):

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
    figures: DirectoryPath | None = Field(
        default=None, description="Directory path for figures.")

    @field_validator("output", mode="after")
    @classmethod
    def populate_output(cls, value, info: ValidationInfo):
        if value is None:
            directory = info.data["features"].parent.parent / "output"
            directory.mkdir(exist_ok=True, parents=True)
            return directory
        return value

    @field_validator("predictions", mode="after")
    @classmethod
    def populate_predictions(cls, value, info: ValidationInfo):
        print("Validate predictions")
        if value is None:
            return info.data["output"] / "predictions.csv"
        return value

    @field_validator("targets", mode="after")
    @classmethod
    def populate_targets(cls, value: FilePath | None, info: ValidationInfo):
        if value is None:
            return info.data["features"]
        return value

    @field_validator("figures", mode="after")
    @classmethod
    def populate_figures(cls, value: DirectoryPath | None, info: ValidationInfo):
        if value is None:
            return info.data["output"] / "figures"
        return value


