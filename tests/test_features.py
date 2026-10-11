import numpy as np
import pandas as pd
import pytest

from conso_forecast import features


def _hourly_frame(start: str, days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=24 * days, freq="h")
    return pd.DataFrame(
        {
            "consumption_mw": 50_000 + rng.normal(0, 1000, len(index)),
            "temperature_c": 10 + rng.normal(0, 3, len(index)),
        },
        index=index,
    )


def test_calendar_flags_for_known_dates():
    index = pd.DatetimeIndex(
        ["2024-07-14 12:00", "2024-05-10 12:00", "2024-12-25 08:00", "2024-07-15 12:00"]
    )
    cal = features.calendar_features(index)
    assert cal["is_holiday"].tolist() == [1, 0, 1, 0]  # Bastille Day, -, Christmas, -
    assert cal["is_bridge"].tolist() == [0, 1, 0, 0]  # Friday after Ascension (9 May 2024)
    assert cal["is_christmas_period"].tolist() == [0, 0, 1, 0]
    assert cal["is_weekend"].tolist() == [1, 0, 0, 0]  # 14 July 2024 was a Sunday


def test_lags_are_exact_shifts_on_the_hourly_grid():
    df = _hourly_frame("2024-01-01", 10)
    lags = features.lag_features(df["consumption_mw"])
    t = pd.Timestamp("2024-01-09 15:00")
    assert lags.loc[t, "lag_1d"] == df.loc[t - pd.Timedelta(days=1), "consumption_mw"]
    assert lags.loc[t, "lag_7d"] == df.loc[t - pd.Timedelta(days=7), "consumption_mw"]
    previous_day = df.loc["2024-01-08", "consumption_mw"]
    assert lags.loc[t, "prev_day_mean"] == pytest.approx(previous_day.mean())
    assert lags.loc[t, "prev_day_last"] == previous_day.iloc[-1]


def test_consumption_features_do_not_use_the_forecast_day():
    """No leakage: changing consumption on day D+1 and later must not change D+1 features."""
    df = _hourly_frame("2024-01-01", 20)
    target_day = pd.Timestamp("2024-01-12")
    original = features.build_features(df)

    tampered = df.copy()
    tampered.loc[tampered.index >= target_day, "consumption_mw"] *= 3
    modified = features.build_features(tampered)

    day = original.index.normalize() == target_day
    pd.testing.assert_frame_equal(
        original.loc[day, features.FEATURES], modified.loc[day, features.FEATURES]
    )


def test_build_features_has_expected_columns_and_rejects_irregular_index():
    df = _hourly_frame("2024-01-01", 10)
    table = features.build_features(df)
    assert list(table.columns) == features.FEATURES + [features.TARGET]
    with pytest.raises(ValueError, match="regular hourly"):
        features.build_features(df.drop(df.index[5]))


def test_heating_degrees_are_zero_when_warm():
    index = pd.date_range("2024-07-01", periods=72, freq="h")
    weather = features.weather_features(pd.Series(25.0, index=index))
    assert (weather["heating_degrees"] == 0).all()
    assert (weather["cooling_degrees"] == 3.0).all()
