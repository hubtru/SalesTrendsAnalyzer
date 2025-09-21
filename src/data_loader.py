"""Utilities for loading and normalising the raw sales data files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


RAW_OUTPUT_PATH = Path("Data/merged_raw.csv")
CLEAN_OUTPUT_PATH = Path("Data/merged_cleaned.csv")


def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def convert_to_csv(
    folder_path: str | Path,
    destination_path: str | Path,
    column_names: Sequence[str],
    merged_output_path: str | Path | None = RAW_OUTPUT_PATH,
) -> pd.DataFrame:
    """Read semi-structured raw files, export them as CSV, and merge them.

    Parameters
    ----------
    folder_path:
        Source directory containing the raw text files.
    destination_path:
        Directory where the normalised CSV files should be written.
    column_names:
        Sequence of column names expected in the raw files.
    merged_output_path:
        Optional path where the merged dataset should be stored.
    """

    source_dir = Path(folder_path)
    output_dir = Path(destination_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not column_names:
        raise ValueError("column_names must contain at least one entry")

    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for path in sorted(source_dir.iterdir()):
        if path.is_dir():
            continue
        if path.suffix.lower() not in {".txt", ".csv"}:
            continue

        df = pd.read_csv(path, sep=";", names=column_names, header=None)
        frames.append(df)
        df.to_csv(output_dir / f"{path.stem}.csv", index=False)

    if not frames:
        raise ValueError(f"No readable raw files were found in {source_dir}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna()
    merged = merged.drop_duplicates()
    merged = merged.reset_index(drop=True)

    total_rows = sum(len(frame.index) for frame in frames)
    if total_rows != len(merged.index):
        print(
            "Warning: merged row count does not match the sum of individual files."
        )
        print(f"Merged rows: {len(merged.index)}")
        print(f"Rows across files: {total_rows}")

    if merged_output_path is not None:
        merged_output = Path(merged_output_path)
        merged_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merged_output, index=False)

    return merged


def clean_data(
    df: pd.DataFrame,
    output_path: str | Path | None = CLEAN_OUTPUT_PATH,
) -> pd.DataFrame:
    """Normalise dtypes, remove duplicates, and persist the cleaned dataset."""

    required_columns = (
        "Date",
        "StoreID",
        "ProductID",
        "Quantity",
        "Price",
        "Quantity_perWeek",
        "Price_Total_perOrder",
    )
    _ensure_columns(df, required_columns)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")

    numeric_columns = [
        "Quantity",
        "Quantity_perWeek",
        "Price",
        "Price_Total_perOrder",
    ]

    for column in numeric_columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            series = pd.to_numeric(
                series.astype(str).str.replace(",", "."), errors="coerce"
            )
        df[column] = series

    df["Quantity"] = df["Quantity"].round().astype("Int64")
    df["Quantity_perWeek"] = df["Quantity_perWeek"].round().astype("Int64")

    df = df.sort_values(by=["Date", "StoreID", "ProductID"])
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination, index=False)

    return df


if __name__ == "__main__":
    folder_path = Path("Data/Raw")
    destination_path = Path("Data/CSV")
    column_names = [
        "Date",
        "StoreID",
        "ProductID",
        "Quantity",
        "Price",
        "Quantity_perWeek",
        "Price_Total_perOrder",
    ]

    raw_df = convert_to_csv(folder_path, destination_path, column_names)
    clean_data(raw_df)
