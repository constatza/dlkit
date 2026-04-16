"""DatasetSettings specific behaviour tests."""

from pathlib import Path

from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.config.dataset_settings import DatasetSettings


def test_get_init_kwargs_preserves_data_entries(tmp_path: Path) -> None:
    """DatasetSettings.get_init_kwargs keeps nested DataEntry objects (not dicts)."""
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    x_path.write_text("dummy")
    y_path.write_text("dummy")

    features = (Feature(name="x", path=x_path),)
    targets = (Target(name="y", path=y_path),)
    settings = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=features,
        targets=targets,
    )

    init_kwargs = settings.get_init_kwargs()

    assert init_kwargs["features"][0] is features[0]
    assert init_kwargs["targets"][0] is targets[0]
