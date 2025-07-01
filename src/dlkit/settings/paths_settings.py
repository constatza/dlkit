from pathlib import Path

from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    model_validator,
)

from dlkit.settings.base_settings import BasicSettings
from loguru import logger


class PathSettings(BasicSettings):
    """PathSettings is a configuration class for managing directory and file paths
    used in the application. It extends BaseSettings and utilizes Pydantic's
    validation x to ensure paths are correctly set and directories exist.

    Attributes:
        root_dir (DirectoryPath): The root directory.
        input_dir (DirectoryPath | None): The input directory, defaults to a subdirectory
            of root if not provided.
        output_dir (DirectoryPath | None): The output directory for generated files,
            defaults to a subdirectory of root if not provided.
        checkpoints_dir (DirectoryPath | None): Directory path for checkpoints.
        figures_dir (DirectoryPath | None): Directory path for figures.
        predictions_dir (DirectoryPath | None): Directory path for predictions.
        x (FilePath): Path to the x file.
        targets (FilePath | None): Path to the targets file, defaults to the x
            path if not provided.
        idx_split (FilePath | None): Path to the index split file.
        checkpoint (FilePath | None): Path to the checkpoint file.

    Methods:
        populate_targets(cls, value, info): Ensures the targets path defaults to the
            x path if not provided.
        populate_basics(cls, value, info): Ensures input and output directories are
            created if they do not exist.
        ensure_directories(self): Ensures all directory paths exist, creating them
            if necessary.
    """

    root_dir: DirectoryPath = Field(default=Path("."), description="Root directory.", alias="root")
    settings: FilePath | None = Field(
        None, description="Path to the settings directory.", alias="self"
    )

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

    mlruns_dir: DirectoryPath | None = Field(
        None, description="Directory path for mlflow runs output."
    )

    @model_validator(mode="after")
    def ensure_directories(self):
        update = {}
        for key, value in self.model_extra.items():
            value = Path(value)
            if key.endswith("_dir") and not value.is_dir():
                value = self.root_dir / key.split("_dir")[0]
                value.mkdir(exist_ok=True, parents=True)
            update[key] = value
            logger.info(f"{key} set to {str(value)}")
        return self.model_copy(update=update)
