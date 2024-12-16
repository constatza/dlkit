from os import times

import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def rm_columns_with_any_null(df: pl.DataFrame) -> pl.DataFrame:
    has_nulls = df.select(pl.all().is_null().any()).to_dict()
    selection = []
    for name, has_nulls in has_nulls.items():
        if not has_nulls.item():
            selection.append(name)

    return df.select(selection)


def workflow(path):

    df = pl.read_csv(
        path,
        has_header=False,
    )

    df = df.pipe(rm_columns_with_any_null)
    return df


def filter_no_zeros(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_row_index().filter(~pl.any_horizontal(pl.all().eq(0)))


def count_zero_rows(df):
    # count zeros of each column
    import matplotlib.ticker as mtick

    zeros = df.select(pl.all().eq(0).sum()).to_numpy().flatten()
    num_rows = df.shape[0]
    num_steps = df.shape[1]
    fig, ax = plt.subplots()
    ax.plot(num_rows - zeros)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Samples with complete history")
    secay = ax.secondary_yaxis(
        "right", functions=(lambda x: x / num_rows, lambda x: x * num_rows)
    )
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: x / num_steps, lambda x: x * num_steps)
    )
    secay.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    secax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid()
    plt.show()


def plot_all(df_time, df_vol):
    times = df_time.select(pl.exclude("index")).to_numpy().squeeze()
    volumes = df_vol.select(pl.exclude("index")).to_numpy()
    plt.semilogy(times, volumes)
    plt.show()


def interpolate(df_time, df_vol):
    from numpy import interp

    # select first row as common times for all volumes
    seconds_per_day = 24 * 3600
    times = df_time.select(pl.exclude("index")).to_numpy() / seconds_per_day
    volumes = df_vol.select(pl.exclude("index")).to_numpy()
    position_to_interpolate = times[0, :]

    # for each pair of times, volumes interpolate at position_to_interpolate
    new_volumes = []
    for time, volume in zip(times, volumes):
        new_volumes.append(interp(position_to_interpolate, time, volume))

    # create a new dataframe with the interpolated volumes and the common times at the first columns
    new_df = pl.DataFrame(
        {
            "time": position_to_interpolate,
            **{f"volume_{i}": new_volume for i, new_volume in enumerate(new_volumes)},
        }
    )

    return new_df


def main():
    from dlkit.io.readers import read_study

    write = True

    study = read_study("./config.toml")
    volumes_path = study.paths.volumes_collected
    times_path = study.paths.times_collected
    parameters_path = study.paths.parameters_collected

    df_vol = workflow(volumes_path)
    df_time = workflow(times_path)
    df_params = workflow(parameters_path)

    count_zero_rows(df_time)

    # drop the last columns
    # drop first column of volumes that is useless
    df_vol = df_vol.drop("column_1", df_vol.columns[-1])
    df_time = df_time.drop(df_time.columns[-1])

    total_timeseries = df_vol.shape[0]
    df_time = df_time.pipe(filter_no_zeros)
    complete_timeseries = df_time.shape[0]

    # filter same rows in the volumes
    df_vol = df_vol.with_row_index().join(
        df_time.select("index"), on="index", how="inner"
    )
    df_params = df_params.with_row_index().join(
        df_vol.select("index"), on="index", how="inner"
    )
    df = interpolate(df_time, df_vol)
    plot_all(df.select("time"), df.select(pl.exclude("time")))

    print(f"Total timeseries: {total_timeseries}")
    print(f"Complete timeseries: {complete_timeseries}")
    print(
        f"Percentage of complete timeseries: {complete_timeseries/total_timeseries*100:.2f}%"
    )

    if write:
        df_time.write_csv(
            volumes_path.with_stem("times_processed"),
            float_scientific=True,
            float_precision=10,
            include_header=False,
        )

        df_vol.write_csv(
            volumes_path.with_stem("volumes_processed"),
            float_scientific=True,
            float_precision=10,
            include_header=False,
        )

        df_params.select(pl.exclude("index")).write_csv(
            study.paths.features,
            float_scientific=True,
            include_header=False,
        )

        df.select(pl.exclude("index", "time")).write_csv(
            study.paths.targets,
            float_scientific=True,
            include_header=False,
        )


if __name__ == "__main__":
    main()
