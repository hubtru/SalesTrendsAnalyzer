import pandas as pd
from datetime import date
import holidays
from itertools import product
import calendar
import numpy as np


def impute(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df1 = pd.DataFrame(columns=["Date", "StoreID", "ProductID"])
    df1["Date"] = pd.date_range(start=df["Date"].min(), end=df["Date"].max())

    dates = df1["Date"].unique()
    stores = df["StoreID"].unique()
    products = df["ProductID"].unique()
    all = list(product(dates, stores, products))

    df2 = pd.DataFrame(all, columns=["Date", "StoreID", "ProductID"])

    df3 = pd.merge(
        df2,
        df[["Date", "StoreID", "ProductID", "Quantity", "Price"]],
        how="left",
        on=["Date", "StoreID", "ProductID"],
    )
    df3["Quantity"] = df3["Quantity"].fillna(0)
    general_avg = df3["Price"].mean()
    df3["Price_store_avg"] = df3.groupby(["StoreID", "ProductID"])["Price"].transform(
        lambda x: x.fillna(x.mean())
    )
    df3["Price_product_avg"] = df3["Price_store_avg"].fillna(
        df3.groupby(["ProductID"])["Price"].transform(lambda x: x.fillna(x.mean()))
    )
    df3["Price_imputed"] = df3["Price_product_avg"].fillna(general_avg)
    df3 = df3.drop(["Price", "Price_store_avg", "Price_product_avg"], axis=1)

    df4 = weather_features(df3)
    df5 = date_features(df4)
    return df5


def weather_features(df):
    weather = pd.read_csv("Data/weather.csv")

    def recalculate_features(weather):
        copied_columns = [
            "temperature_2m_max (°C)",
            "temperature_2m_min (°C)",
            "temperature_2m_mean (°C)",
            "apparent_temperature_max (°C)",
            "apparent_temperature_min (°C)",
            "apparent_temperature_mean (°C)",
            "precipitation_sum (mm)",
            "windgusts_10m_max (km/h)",
            "windspeed_10m_max (km/h)",
            "shortwave_radiation_sum (MJ/m²)",
            "precipitation_hours (h)",
        ]
        weather_new = pd.DataFrame()
        weather_new["Date"] = pd.to_datetime(weather["time"])
        for col in copied_columns:
            weather_new[col.split(" ")[0]] = weather[col]

        weather_new["evapotranspiration_res"] = weather[
            "et0_fao_evapotranspiration (mm)"
        ] - weather.index.to_series().apply(
            lambda x: (1.87 * np.sin(x * 2 * np.pi / 365 + 0.26 * np.pi))
            + np.mean(weather["et0_fao_evapotranspiration (mm)"])
        )

        # 1 if snowfall > 0, else 0
        weather_new["IsSnowfall"] = np.heaviside(weather["snowfall_sum (cm)"], 0)

        wcode_map = {
            0: "no clouds developing",
            1: "clouds dissolving",
            2: "unchanged sky state",
            3: "clouds developing",
        }

        weather_new["weather_condition"] = weather["weathercode (wmo code)"].apply(
            lambda x: wcode_map[x] if x in [0, 1, 2, 3] else "precipitation"
        )
        return weather_new

    weather_features = recalculate_features(weather)
    return pd.merge(
        df,
        weather_features,
        how="left",
        on=["Date"],
    )

    return df


def date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.strftime("%Y").astype(int)
    df["Month"] = df["Date"].dt.strftime("%m").astype(int)
    df["DayoftheMonth"] = df["Date"].dt.strftime("%d").astype(int)

    df["WeekoftheMonth"] = (
        df["DayoftheMonth"]
        .apply(lambda x: 4 if (int(x) - 1) // 7 + 1 == 5 else (int(x) - 1) // 7 + 1)
        .astype(int)
    )
    df["DayoftheWeek"] = (
        df["DayoftheMonth"]
        .apply(
            lambda x: 7
            if int(x) % 7 == 0
            else 7 + (int(x) - 28)
            if int(x) > 28
            else int(x) % 7
        )
        .astype(int)
    )

    df["WeekoftheYear"] = df["Date"].dt.strftime("%U").astype(int)
    df["DayoftheYear"] = df["Date"].dt.strftime("%j").astype(int)

    df["DayName"] = df["Date"].dt.day_name()
    df["isWeekend"] = df["DayName"].isin(["Saturday", "Sunday"])

    df["isWeekStart"] = df["DayoftheWeek"].apply(
        lambda x: True if int(x) == 1 else False
    )
    df["isWeekEnd"] = df["DayoftheWeek"].apply(lambda x: True if int(x) == 7 else False)
    df["isMonthStart"] = df["DayoftheMonth"].apply(
        lambda x: True if int(x) == 1 else False
    )
    df["isMonthEnd"] = df["Date"].apply(
        lambda x: True if x.day == calendar.monthrange(x.year, x.month)[1] else False
    )

    def get_season(date):
        year = date.year
        if date.month >= 3 and date.month <= 5:
            return f"Spring"
        elif date.month >= 6 and date.month <= 8:
            return f"Summer"
        elif date.month >= 9 and date.month <= 11:
            return f"Autumn"
        else:
            return f"Winter"

    df["Season"] = df["Date"].apply(get_season)

    df = pd.get_dummies(df, columns=["Season"], prefix=["Season"])

    de_holidays = holidays.Germany(years=[2020, 2021, 2022, 2023])
    df["isHoliday"] = [True if x in de_holidays else False for x in df["Date"]]

    df = df.drop(["DayName"], axis=1)

    return df


if __name__ == "__main__":
    df = pd.read_csv("Data/merged_cleaned.csv")
    df = impute(df)
    df.to_csv("Data/merged_cleaned_FE_imputed(v)_w.csv", index=False)
