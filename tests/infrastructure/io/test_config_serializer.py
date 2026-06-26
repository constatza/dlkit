from __future__ import annotations

import torch

from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.io import serialize_config_to_string


def test_serialize_config_strips_in_memory_entry_values_from_job_config() -> None:
    job = JobConfig.model_validate(
        {
            "run": {"type": "train"},
            "data": {
                "features": [
                    {
                        "name": "x",
                        "value": torch.tensor([1.0, 2.0, 3.0]),
                        "data_role": "feature",
                    }
                ],
                "targets": [
                    {
                        "name": "y",
                        "value": torch.tensor([0.0]),
                        "data_role": "target",
                    }
                ],
            },
        }
    )

    toml_str = serialize_config_to_string(job, exclude_value_entries=True)

    assert "[data]" in toml_str
    assert "[[data.features]]" in toml_str
    assert "[[data.targets]]" in toml_str
    assert "value =" not in toml_str
