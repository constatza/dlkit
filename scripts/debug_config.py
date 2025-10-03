#!/usr/bin/env python3
"""Debug script to understand dynaconf dataflow structure."""

from pathlib import Path
import tempfile
from dynaconf import Dynaconf
from pprint import pprint

# Create a temporary config file like the test
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    cfg_dir = tmpdir / "cfg"
    cfg_dir.mkdir()

    # Create dummy files
    (cfg_dir / "X.npy").write_text("dummy")
    (cfg_dir / "Y.npy").write_text("dummy")
    (cfg_dir / "indices.txt").write_text("dummy")
    (cfg_dir / "model.ckpt").write_text("dummy")

    # Write config exactly like test
    config_text = """[PATHS]
output_dir = "outputs"
checkpoints_dir = "checkpoints"
figures_dir = "figures"
predictions_dir = "preds"
mlruns_dir = "mlruns"

[DATASET]
name = "FlexibleDataset"
root_dir = "."

[[DATASET.features]]
name = "X"
path = "X.npy"

[[DATASET.targets]]
name = "Y"
path = "Y.npy"

[DATASET.split]
filepath = "indices.txt"

[MODEL]
name = "ConstantWidthFFNN"
module_path = "dlkit.core.models.nn"
checkpoint = "model.ckpt"

[TRAINING.trainer]
default_root_dir = "work"
"""
    config_file = cfg_dir / "config.toml"
    config_file.write_text(config_text)

    print(f"Config file: {config_file}")
    print(f"Files exist: X.npy={cfg_dir / 'X.npy' in cfg_dir.glob('*')}")
    print()

    # Load with Dynaconf like GeneralSettings.from_file does
    dyn_conf = Dynaconf(
        root_path=str(config_file.parent),
        settings_files=[config_file.name],
        envvar_prefix="DLKIT",
    )

    print("Dynaconf raw dataflow:")
    as_dict = dict(dyn_conf)
    pprint(as_dict)
    print()

    print("DATASET structure:")
    dataset = as_dict.get("DATASET", {})
    pprint(dataset)
    print()

    print("Type of features:", type(dataset.get("features")))
    print("Features content:", dataset.get("features"))
    print("Type of targets:", type(dataset.get("targets")))
    print("Targets content:", dataset.get("targets"))
