from pathlib import Path
import polars as pl


def read_table(file_path: str, **read_kwargs) -> pl.DataFrame:
    """
    Read tabular data from CSV or Parquet into a Polars DataFrame.

    Supported formats:
      - .csv      → pl.read_csv()
      - .parquet  → pl.read_parquet()
      - .pq       → pl.read_parquet()

    Args:
        file_path (str): Path to the input file.
        **read_kwargs: Additional keyword args passed to the Polars reader.

    Returns:
        pl.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")
    ext = path.suffix.lower()
    if ext == ".csv":
        # Read CSV via Polars
        return pl.read_csv(
            file_path, **read_kwargs
        )  # :contentReference[oaicite:0]{index=0}
    elif ext in {".parquet", ".pq"}:
        # Read Parquet via Polars
        return pl.read_parquet(
            file_path, **read_kwargs
        )  # :contentReference[oaicite:1]{index=1}
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported extensions: .csv, .parquet, .pq"
        )
