"""Feature engineering utilities for the cleaned sales dataset."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterable

import holidays
import pandas as pd


FEATURE_OUTPUT_PATH = Path("Data/merged_cleaned_FE_imputed(v).csv")


def _ensure_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Create a complete date-store-product grid and impute missing prices."""

    required_columns = ("Date", "StoreID", "ProductID", "Quantity", "Price")
    _ensure_columns(df, required_columns)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "StoreID", "ProductID"])

    if df.empty:
        raise ValueError("No records available for feature engineering.")

    full_index = product(
        pd.date_range(start=df["Date"].min(), end=df["Date"].max()),
        sorted(df["StoreID"].unique()),
        sorted(df["ProductID"].unique()),
    )

    df_full = pd.DataFrame(full_index, columns=["Date", "StoreID", "ProductID"])
    df_merged = pd.merge(
        df_full,
        df[["Date", "StoreID", "ProductID", "Quantity", "Price"]],
        how="left",
        on=["Date", "StoreID", "ProductID"],
    )

    df_merged["Quantity"] = pd.to_numeric(
        df_merged["Quantity"], errors="coerce"
    ).fillna(0)
    df_merged["Quantity"] = df_merged["Quantity"].round().astype("Int64")
    general_avg = df_merged["Price"].dropna().mean()
    if pd.isna(general_avg):
        general_avg = 0.0

    df_merged["Price_store_avg"] = df_merged.groupby(["StoreID", "ProductID"])[
        "Price"
    ].transform(lambda x: x.fillna(x.mean()))
    df_merged["Price_product_avg"] = df_merged["Price_store_avg"].fillna(
        df_merged.groupby(["ProductID"])["Price"].transform(lambda x: x.fillna(x.mean()))
    )
    df_merged["Price_imputed"] = df_merged["Price_product_avg"].fillna(general_avg)
    df_merged = df_merged.drop(["Price", "Price_store_avg", "Price_product_avg"], axis=1)

    return date_features(df_merged)


def date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar-based features for downstream models."""

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    if df.empty:
        return df

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayoftheMonth"] = df["Date"].dt.day
    df["WeekoftheMonth"] = ((df["DayoftheMonth"] - 1) // 7 + 1).astype(int)
    df["DayoftheWeek"] = df["Date"].dt.isoweekday()
    df["WeekoftheYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayoftheYear"] = df["Date"].dt.dayofyear

    df["isWeekend"] = df["DayoftheWeek"].isin({6, 7})
    df["isWeekStart"] = df["DayoftheWeek"] == 1
    df["isWeekEnd"] = df["DayoftheWeek"] == 7
    df["isMonthStart"] = df["Date"].dt.is_month_start
    df["isMonthEnd"] = df["Date"].dt.is_month_end

    def get_season(value: pd.Timestamp) -> str:
        month = value.month
        if 3 <= month <= 5:
            return "Spring"
        if 6 <= month <= 8:
            return "Summer"
        if 9 <= month <= 11:
            return "Autumn"
        return "Winter"

    df["Season"] = df["Date"].apply(get_season)
    df = pd.get_dummies(df, columns=["Season"], prefix="Season")

    germany_holidays = holidays.country_holidays("DE", years=range(2020, 2024))
    df["isHoliday"] = df["Date"].isin(germany_holidays)

    return df


if __name__ == "__main__":
    df = pd.read_csv("Data/merged_cleaned.csv")
    df = impute(df)
    df.to_csv(FEATURE_OUTPUT_PATH, index=False)
