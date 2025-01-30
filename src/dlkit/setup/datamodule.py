from pathlib import Path
from dlkit.datasets.lazy_dataset import LazyDataset
from dlkit.transforms.chaining import TransformationChain
from dlkit.utils.system_utils import import_dynamically, filter_kwargs
from dlkit.setup.transforms import initialize_transforms


def initialize_datamodule(config):
    """
    Dynamically imports and sets up the datamodule based on the provided configuration.
    :param config: dict: The configuration object for the datamodule.
    :return: LightningDataModule: The instantiated datamodule object.
    """

    datamodel_config = config.get("datamodule")
    paths_config = config.get("paths")

    # Include hyperparameter suggestions
    datamodule_class = import_dynamically(
        datamodel_config.get("name"), prepend="dlkit.datamodules"
    )

    save_dir = (
        paths_config.get("datamodule", None)
        or Path(paths_config.get("output")) / "datamodule"
    )

    features_path = paths_config.get("features")
    targets_path = paths_config.get("targets", None)

    save_dir.mkdir(parents=True, exist_ok=True)

    transforms: TransformationChain = initialize_transforms(config)
    dataset = LazyDataset(
        features_path,
        targets_path=targets_path,
        test_size=datamodel_config.get("test_size", 0.3),
        val_size=datamodel_config.get("val_size", 0.5),
        indices_path=paths_config.get("indices", None),
        transforms=transforms,
    )

    datamodule_instance = datamodule_class(
        dataset,
        save_dir=save_dir,
        dataloader_config=config.get("dataloader"),
        batch_size=datamodel_config.get("batch_size", 64),
    )

    return datamodule_instance
