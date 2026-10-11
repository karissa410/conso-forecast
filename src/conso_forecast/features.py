"""Feature engineering for day-ahead (J+1) hourly forecasting.

Forecasting setup
-----------------
The forecast for the 24 hours of day D+1 is issued at the end of day D (midnight, local
time). Consumption features may therefore only use values from day D or earlier. On the
regular local grid this means: lags of at least 24 hours, and daily statistics of day D
shifted by one day. `tests/test_features.py` checks this property directly.

Weather features use the *observed* temperature of the target hour (reanalysis data).
In production one would use a weather forecast instead; this "oracle weather" assumption
is stated as a limitation in the README.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

TARGET = "consumption_mw"
HEATING_THRESHOLD_C = 15.0
COOLING_THRESHOLD_C = 22.0

CATEGORICAL_FEATURES = ["hour", "day_of_week", "month"]
CALENDAR_FLAGS = ["is_weekend", "is_holiday", "is_bridge", "is_christmas_period"]
SEASONAL_FEATURES = ["day_of_year_sin", "day_of_year_cos"]
LAG_FEATURES = [
    "lag_1d",
    "lag_2d",
    "lag_7d",
    "prev_day_mean",
    "prev_day_min",
    "prev_day_max",
    "prev_day_last",
    "mean_7d",
]
WEATHER_FEATURES = [
    "temperature_c",
    "temperature_smoothed",
    "temperature_day_mean",
    "temperature_lag_1d",
    "heating_degrees",
    "cooling_degrees",
]
NUMERIC_FEATURES = SEASONAL_FEATURES + LAG_FEATURES + WEATHER_FEATURES
FEATURES = CATEGORICAL_FEATURES + CALENDAR_FLAGS + NUMERIC_FEATURES


def french_holidays(years: range) -> set[pd.Timestamp]:
    """Metropolitan French public holidays (Alsace-Moselle specific days excluded)."""
    return {pd.Timestamp(day) for day in holidays.France(years=years)}


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    dates = index.normalize()
    years = range(index.year.min() - 1, index.year.max() + 2)
    holiday_dates = french_holidays(years)
    is_holiday = dates.isin(list(holiday_dates))

    # A "bridge" (pont) is a Monday before a Tuesday holiday or a Friday after a Thursday
    # holiday: many people take it off, so consumption looks like a weekend day.
    next_day_holiday = (dates + pd.Timedelta(days=1)).isin(list(holiday_dates))
    prev_day_holiday = (dates - pd.Timedelta(days=1)).isin(list(holiday_dates))
    is_bridge = ~is_holiday & (
        ((index.dayofweek == 0) & next_day_holiday) | ((index.dayofweek == 4) & prev_day_holiday)
    )
    is_christmas = ((index.month == 12) & (index.day >= 24)) | (
        (index.month == 1) & (index.day == 1)
    )

    angle = 2 * np.pi * (index.dayofyear - 1) / 365.25
    return pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_holiday": is_holiday.astype(int),
            "is_bridge": is_bridge.astype(int),
            "is_christmas_period": is_christmas.astype(int),
            "day_of_year_sin": np.sin(angle),
            "day_of_year_cos": np.cos(angle),
        },
        index=index,
    )


def _previous_day_stat(series: pd.Series, stat: str) -> pd.Series:
    """Daily statistic of day D, aligned on every hour of day D+1."""
    dates = series.index.normalize()
    daily = series.groupby(dates).agg(stat)
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    previous = daily.shift(1)
    return pd.Series(previous.reindex(dates).to_numpy(), index=series.index)


def lag_features(consumption: pd.Series) -> pd.DataFrame:
    """Past-consumption features available at the end of day D for every hour of D+1."""
    previous_day_mean = _previous_day_stat(consumption, "mean")
    dates = consumption.index.normalize()
    daily_mean = consumption.groupby(dates).mean()
    daily_mean = daily_mean.reindex(pd.date_range(daily_mean.index.min(), dates.max(), freq="D"))
    mean_7d = daily_mean.rolling(7, min_periods=5).mean().shift(1)
    return pd.DataFrame(
        {
            "lag_1d": consumption.shift(24),
            "lag_2d": consumption.shift(48),
            "lag_7d": consumption.shift(168),
            "prev_day_mean": previous_day_mean,
            "prev_day_min": _previous_day_stat(consumption, "min"),
            "prev_day_max": _previous_day_stat(consumption, "max"),
            "prev_day_last": _previous_day_stat(consumption, "last"),
            "mean_7d": pd.Series(mean_7d.reindex(dates).to_numpy(), index=consumption.index),
        },
        index=consumption.index,
    )


def weather_features(temperature: pd.Series) -> pd.DataFrame:
    # Buildings react to temperature with inertia: a smoothed temperature (half-life one
    # day) usually explains heating demand better than the instantaneous value.
    smoothed = temperature.ewm(halflife=24, ignore_na=True).mean()
    dates = temperature.index.normalize()
    day_mean = temperature.groupby(dates).transform("mean")
    return pd.DataFrame(
        {
            "temperature_c": temperature,
            "temperature_smoothed": smoothed,
            "temperature_day_mean": day_mean,
            "temperature_lag_1d": temperature.shift(24),
            "heating_degrees": (HEATING_THRESHOLD_C - smoothed).clip(lower=0),
            "cooling_degrees": (temperature - COOLING_THRESHOLD_C).clip(lower=0),
        },
        index=temperature.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature table (one row per local hour) plus the target column."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("build_features expects a DatetimeIndex")
    steps = df.index.to_series().diff().dropna()
    if (steps != pd.Timedelta(hours=1)).any():
        raise ValueError("build_features expects a regular hourly index (the local grid)")
    table = pd.concat(
        [
            calendar_features(df.index),
            lag_features(df[TARGET]),
            weather_features(df["temperature_c"]),
            df[[TARGET]],
        ],
        axis=1,
    )
    return table[FEATURES + [TARGET]]
