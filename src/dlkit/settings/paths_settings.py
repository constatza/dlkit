from pathlib import Path

from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    ValidationInfo,
    field_validator,
    model_validator,
)

from dlkit.settings.base_settings import BaseSettings


class PathSettings(BaseSettings):
    """PathSettings is a configuration class for managing directory and file paths
    used in the application. It extends BaseSettings and utilizes Pydantic's
    validation features to ensure paths are correctly set and directories exist.

    Attributes:
        root (DirectoryPath): The root directory.
        input_dir (DirectoryPath | None): The input directory, defaults to a subdirectory
            of root if not provided.
        output_dir (DirectoryPath | None): The output directory for generated files,
            defaults to a subdirectory of root if not provided.
        checkpoints_dir (DirectoryPath | None): Directory path for checkpoints.
        figures_dir (DirectoryPath | None): Directory path for figures.
        predictions_dir (DirectoryPath | None): Directory path for predictions.
        features (FilePath): Path to the features file.
        targets (FilePath | None): Path to the targets file, defaults to the features
            path if not provided.
        idx_split (FilePath | None): Path to the index split file.
        checkpoint (FilePath | None): Path to the checkpoint file.

    Methods:
        populate_targets(cls, value, info): Ensures the targets path defaults to the
            features path if not provided.
        populate_basics(cls, value, info): Ensures input and output directories are
            created if they do not exist.
        ensure_directories(self): Ensures all directory paths exist, creating them
            if necessary.
    """

    root: DirectoryPath = Field(..., description="Root directory.")
    settings: FilePath = Field(..., description="Path to the settings directory.")

    input_dir: DirectoryPath | None = Field(None, description="Input directory.")
    output_dir: DirectoryPath | None = Field(
        None, description="Output directory for generated files."
    )
    checkpoints_dir: DirectoryPath | None = Field(
        None, description="Directory path for checkpoints."
    )
    figures_dir: DirectoryPath | None = Field(
        default=None, description="Directory path for figures."
    )
    predictions_dir: DirectoryPath | None = Field(
        None, description="Directory path for predictions."
    )

    features: FilePath = Field(..., description="Path to the features file.")
    targets: FilePath | None = Field(default=None, description="Path to the targets file (if any).")

    # !! idx split default value must be None !!
    idx_split: FilePath | None = Field(default=None, description="Path to the index split file.")
    checkpoint: FilePath | None = Field(default=None, description="Path to the checkpoint file.")

    @field_validator("targets")
    @classmethod
    def populate_targets(cls, value: FilePath | None, info: ValidationInfo):
        if value is None:
            return info.data["features"]
        return value

    @field_validator("output_dir", "input_dir")
    @classmethod
    def populate_basics(cls, value: DirectoryPath | None, info: ValidationInfo):
        if value is None:
            new_path = info.data["root"] / info.field_name
            new_path.mkdir(exist_ok=True, parents=True)
            return new_path
        return value

    @model_validator(mode="after")
    def ensure_directories(self):
        update = {}
        for key, value in self.model_extra.items():
            value = Path(value)
            if key.endswith("_dir") and not value.is_dir():
                value.mkdir(exist_ok=True, parents=True)
            update[key] = value
        return self.model_copy(update=update)
