from pathlib import Path
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.setup.transforms import initialize_transforms


def initialize_datamodule(config):
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :param config: dict: The configuration object for the datamodule.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    datamodel_config = config.get("datamodule")
    transforms_config = config.get("transforms")
    paths_config = config.get("paths")

    # Include hyperparameter suggestions
    datamodule_class = import_dynamically(
        datamodel_config.get("name"), prepend="dlkit.datamodules"
    )

    features_pipeline, targets_pipeline = initialize_transforms(transforms_config)

    save_dir = (
        paths_config.get("datamodule", None)
        or Path(paths_config.get("output")) / "datamodule"
    )

    features_path = paths_config.get("features")
    targets_path = paths_config.get("targets", None)

    save_dir.mkdir(parents=True, exist_ok=True)

    filtered_config = filter_kwargs(datamodel_config)

    datamodule_instance = datamodule_class(
        **filtered_config,
        save_dir=save_dir,
        features_path=features_path,
        targets_path=targets_path,
        features_pipeline=features_pipeline,
        targets_pipeline=targets_pipeline,
        dataloader_config=config.get("dataloader"),
    )

    return datamodule_instance
