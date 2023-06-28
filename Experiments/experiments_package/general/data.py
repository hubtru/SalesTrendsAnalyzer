import math

import numpy as np
import pandas as pd

from .config import USED_STORE_ID, DatasetOptions, ProductIds
from .window_generator import WindowGenerator


def _load_data(data_origin):
    """data_origin => location of imputed features CSV"""
    original_data = pd.read_csv(data_origin, parse_dates=["Date"])
    not_indexed = ["ProductID", "Quantity", "Price_imputed", "Year"]
    df_pivot = original_data.pivot_table(
        index=[col for col in list(original_data.columns) if col not in not_indexed],
        columns=["ProductID"],
        values="Quantity",
    )
    df_pivot["total_quantity_day"] = df_pivot.sum(axis=1)

    df_pivot = df_pivot.loc[df_pivot.index.get_level_values("StoreID") == USED_STORE_ID]
    df_pivot = df_pivot.droplevel("StoreID")
    df_pivot = df_pivot.reset_index()
    df_pivot.columns = df_pivot.columns.astype(str)

    return df_pivot


def _add_fourier_features(
    data,
    features_to_period=None,
):
    if features_to_period is None:
        features_to_period = {
            "Month": 12.0,
            "DayoftheMonth": 31.0,
            "WeekoftheMonth": 4.0,
            "DayoftheWeek": 7.0,
            "WeekoftheYear": 52.0,
            "DayoftheYear": 365.0,
        }
    data_copy = data.copy()

    for column, period in features_to_period.items():
        data_copy[f"{column}_cos"] = np.cos(2 * np.pi * data_copy["Month"] / period)
        data_copy[f"{column}_sin"] = np.sin(2 * np.pi * data_copy["Month"] / period)

    return data_copy.drop(
        features_to_period.keys(),
        axis=1,
    )


def _get_dataset(data_origin):
    df_pivot = _load_data(data_origin)
    date_time = pd.to_datetime(df_pivot.pop("Date"), format="%Y-%m-%d")
    data = _add_fourier_features(df_pivot)
    return data, date_time


def _split_data(data, train_frac=0.7, valid_frac=0.2, test_frac=0.1):
    if not math.isclose(train_frac + valid_frac + test_frac, 1):
        raise ValueError(
            f"Invalid split sizes: {train_frac} + {valid_frac} + {test_frac}"
            + f" has to be 1 but is: {train_frac + valid_frac + test_frac}"
        )

    size = len(data)
    train_df = data[0 : int(size * train_frac)]
    val_df = data[int(size * train_frac) : int(size * (train_frac + valid_frac))]
    test_df = data[int(size * (train_frac + valid_frac)) :]

    return train_df, val_df, test_df


def _normalize(data, norm_values_to_use=None):
    norm_values_used = norm_values_to_use
    if norm_values_to_use is None:
        std = data.std()
        std = std.where(lambda x: np.invert(np.isclose(x, 0)), other=0.3)
        norm_values_used = {
            "std": std,
            "mean": data.mean(),
        }

    return (data - norm_values_used["mean"]) / (
        norm_values_used["std"]
    ), norm_values_used


def get_window_dataset(dataset_options: DatasetOptions) -> WindowGenerator:
    data, _ = _get_dataset(dataset_options.data_origin)

    data = data.drop(
        labels=dataset_options.drop_columns,
        axis="columns",
    )

    train_df, val_df, test_df = _split_data(data)

    train_df, normalization_params = _normalize(train_df)
    val_df, _ = _normalize(val_df, normalization_params)
    test_df, _ = _normalize(test_df, normalization_params)

    return WindowGenerator(
        input_width=dataset_options.window_width,
        label_width=dataset_options.label_width,
        shift=dataset_options.shift,
        label_columns=[ProductIds.BENS_LUNCHTIME.value],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
