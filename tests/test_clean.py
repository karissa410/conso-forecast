import numpy as np
import pandas as pd
import pytest

from conso_forecast.config import City
from conso_forecast.data import clean


def _utc_hours(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def test_parse_eco2mix_renames_and_indexes_in_utc():
    raw = pd.DataFrame(
        {
            "date_heure": ["2024-01-01T00:15:00+01:00", "2024-01-01T00:00:00+01:00"],
            "consommation": ["70000", "71000"],
            "prevision_j1": [69000, 70500],
            "eolien": [1, 2],
        }
    )
    df = clean.parse_eco2mix(raw)
    assert list(df.columns) == ["consumption_mw", "rte_forecast_d1_mw"]
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")  # sorted
    assert df["consumption_mw"].iloc[0] == 71000


def test_parse_eco2mix_requires_consumption():
    with pytest.raises(ValueError, match="consommation"):
        clean.parse_eco2mix(pd.DataFrame({"date_heure": ["2024-01-01T00:00:00+00:00"]}))


def test_parse_open_meteo():
    payload = {
        "hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"], "temperature_2m": [1.5, None]}
    }
    series = clean.parse_open_meteo(payload)
    assert series.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert series.iloc[0] == 1.5 and np.isnan(series.iloc[1])


def test_duplicates_are_dropped_and_counted():
    index = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"], tz="UTC")
    df, n = clean.drop_duplicate_timestamps(pd.DataFrame({"x": [1, 2, 3]}, index=index))
    assert n == 1
    assert df["x"].tolist() == [1, 3]


def test_implausible_values_become_missing():
    series, n = clean.mask_implausible(
        pd.Series([0.0, 50_000.0, 500_000.0, np.nan]), 20_000, 110_000
    )
    assert n == 2
    assert series.isna().tolist() == [True, False, True, True]


def test_quarter_hours_are_averaged_into_hours():
    index = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 10, np.nan, 30, np.nan]}, index=index)
    hourly = clean.to_hourly_utc(df)
    assert hourly["x"].tolist() == [2.5, 20.0]


def test_local_grid_handles_spring_daylight_saving():
    # 2024-03-31: clocks jump from 02:00 to 03:00 in Paris, so the day has 23 real hours.
    index = _utc_hours("2024-03-30 23:00", 23)
    grid = clean.to_local_grid(pd.DataFrame({"x": np.arange(23.0)}, index=index))
    assert len(grid) == 24
    assert np.isnan(grid.loc["2024-03-31 02:00", "x"])
    assert (grid.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_local_grid_averages_the_repeated_autumn_hour():
    # 2024-10-27: 02:00-03:00 happens twice in Paris (00:00 and 01:00 UTC).
    index = _utc_hours("2024-10-26 22:00", 25)
    values = np.arange(25.0)
    grid = clean.to_local_grid(pd.DataFrame({"x": values}, index=index))
    assert len(grid) == 24
    assert grid.loc["2024-10-27 02:00", "x"] == pytest.approx((values[2] + values[3]) / 2)


def test_only_short_gaps_are_interpolated():
    series = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0, np.nan])
    filled = clean.interpolate_short_gaps(series, max_gap=3)
    assert filled.iloc[1] == 2.0  # 1-hour gap filled
    assert filled.iloc[3:7].isna().all()  # 4-hour gap left missing
    assert np.isnan(filled.iloc[8])  # trailing gap: nothing to interpolate towards


def test_national_temperature_is_weighted_and_renormalised():
    cities = (City("A", 0, 0, 3.0), City("B", 0, 0, 1.0))
    temperatures = pd.DataFrame({"A": [10.0, np.nan], "B": [20.0, 20.0]})
    national = clean.national_temperature(temperatures, cities)
    assert national.tolist() == [12.5, 20.0]


def test_full_cleaning_on_synthetic_data(synthetic_settings):
    df, report = clean.clean(
        clean.load_raw_eco2mix(synthetic_settings),
        clean.load_raw_weather(synthetic_settings),
        synthetic_settings,
    )
    assert list(df.columns) == ["consumption_mw", "rte_forecast_d1_mw", "temperature_c"]
    assert df.index.min() == pd.Timestamp(synthetic_settings.start_date)
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert report.duplicated_timestamps == 1
    assert report.implausible_values >= 1
    assert report.missing_hours_after_cleaning == 0
    assert df["consumption_mw"].between(20_000, 110_000).all()
