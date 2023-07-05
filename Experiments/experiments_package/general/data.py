"""
Methods that are used for handling and loading the dataset initially.
"""

import math

import numpy as np
import pandas as pd

from .config import USED_STORE_ID, DatasetOptions, normalize
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
    train_df = data[0: int(size * train_frac)]
    val_df = data[int(size * train_frac): int(size * (train_frac + valid_frac))]
    test_df = data[int(size * (train_frac + valid_frac)):]

    return train_df, val_df, test_df


def get_window_dataset(dataset_options: DatasetOptions) -> WindowGenerator:
    data, time_stamps = _get_dataset(dataset_options.data_origin)

    data = data.drop(
        labels=dataset_options.drop_columns,
        axis="columns",
    )

    train_df, val_df, test_df = _split_data(data)

    train_df, normalization_params = normalize(train_df)
    val_df, _ = normalize(val_df, normalization_params)
    test_df, _ = normalize(test_df, normalization_params)

    return WindowGenerator(
        input_width=dataset_options.window_width,
        label_width=dataset_options.label_width,
        shift=dataset_options.shift,
        label_columns=dataset_options.label_columns,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        normalization_params=normalization_params,
        time_stamps=time_stamps
    )


# import globally to save computation
menu = pd.read_csv("./../Data/Sushi Menu.csv")


def get_all_product_ids():
    global menu
    return list(set(list(menu["ID"].astype(str))))


def get_no_sushi_in_product(p_id: str):
    """
    Gets the productId of a sushi menu item (e.g. 4260705920294) and returns the Number of individual
    sushi pieces are in this menu item.
    """
    global menu

    sushi_count = menu[menu['ID'].astype(str) == p_id]["Count"].astype(int).reset_index(drop=True)

    if len(sushi_count) == 0:
        raise AttributeError(f"Product number {p_id} not found in menu")
    elif len(sushi_count) > 1:
        raise AttributeError(f"Product number {p_id} found several times in menu (but has to be unique)")
    return sushi_count[0]
