import os
import torch

from dynaconf import Dynaconf
import polars as pl
import numpy as np

from dlkit.transforms.


settings_path = "./config.toml"
settings = Dynaconf(settings_files=[settings_path], envvar_prefix="DLKIT")


def main():
    variables = ("u", "p", "cox", "tcell")
    data = [np.load(f"{settings.paths.input}{os.sep}{var}.npy") for var in variables]

    # concat along axis 2
    data = np.concatenate(data, axis=1)
    data = np.transpose(data, (0, 2, 1)).astype(np.float32)
    N, T, D = data.shape
    time = np.linspace(0, 1, T).reshape(-1, 1)
    time = np.tile(time, (N, 1))

    flattened_data = data.reshape(-1, D)

    df = pl.DataFrame(
        flattened_data, schema=[f"feat_{i}" for i in range(D)]
    ).with_columns(
        [
            pl.arange(0, N * T).alias("sample") // T,
            pl.arange(0, N * T).alias("step") % T,
        ]
    )

    df = pl.concat([df, pl.DataFrame({"time": time.flatten()})], how="horizontal")

    df.write_parquet(
        settings.paths.input + os.path.sep + "dataset.parquet",
        compression="snappy",
    )

    # convert dataframe features to torch tensor
    # get feature columns only and convert to torch tensor
    features = df[[f"feat_{i}" for i in range(D)]].to_numpy().astype(np.float32)
    features = torch.from_numpy(features)





if __name__ == "__main__":
    main()
