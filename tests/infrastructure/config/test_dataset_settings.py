"""DatasetSettings specific behaviour tests."""

from pathlib import Path

from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.entry_types import NpyEntry


def test_get_init_kwargs_preserves_data_entries(tmp_path: Path) -> None:
    """DatasetSettings.get_init_kwargs keeps nested DataEntry objects (not dicts)."""
    x_path = tmp_path / "x.npy"
    y_path = tmp_path / "y.npy"
    x_path.write_text("dummy")
    y_path.write_text("dummy")

    features = (NpyEntry(name="x", path=x_path, data_role=DataRole.FEATURE),)
    targets = (NpyEntry(name="y", path=y_path, data_role=DataRole.TARGET),)
    settings = DatasetSettings(
        name="FlexibleDataset",
        module_path="dlkit.engine.data.datasets",
        features=features,
        targets=targets,
    )

    init_kwargs = settings.get_init_kwargs()

    assert init_kwargs["entries"][0] is features[0]
    assert init_kwargs["entries"][1] is targets[0]
