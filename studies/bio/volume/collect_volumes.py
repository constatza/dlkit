def read_files(samples, path):
    # data are the first lines of the files
    lines = []
    for sample in samples:
        # TODO: DOES NOT WORK
        number = f"{sample:d}"
        current = path.with_name(path.name.replace("*", number))
        with open(current, "r") as f:
            first_line = f.readline()
        lines.append(first_line)
    return lines


def clean_line(line):
    line = " ".join(line.strip().split())
    line = line.replace(" ", ",")
    return line


def write_data(data, path):
    with open(path, "w") as f:
        for line in data:
            f.write(line + "\n")


def main():

    from dlkit.io.settings import read_study

    study = read_study("./config.toml")
    volumes = study.paths.volumes
    times = study.paths.times

    # scan the data and append each parameter as a row to the dataframe
    samples = [i for i in range(1023)]
    volumes_data = read_files(samples, volumes)
    times_data = read_files(samples, times)

    volumes_data = [clean_line(line) for line in volumes_data]
    times_data = [clean_line(line) for line in times_data]
    write_data(volumes_data, study.paths.processed / "volumes_collected.csv")
    write_data(times_data, study.paths.processed / "times_collected.csv")


if __name__ == "__main__":
    main()
