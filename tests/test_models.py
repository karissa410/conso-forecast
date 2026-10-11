import numpy as np
import pandas as pd

from conso_forecast import baselines, models
from conso_forecast.features import FEATURES, build_features


def _features(days: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    index = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    df = pd.DataFrame(
        {
            "consumption_mw": 50_000
            + 5000 * np.sin(2 * np.pi * index.hour / 24)
            + rng.normal(0, 300, len(index)),
            "temperature_c": 8 + rng.normal(0, 2, len(index)),
        },
        index=index,
    )
    return build_features(df)


def test_baselines_read_the_lag_features():
    table = _features(10)
    predictions = baselines.predict_baselines(table)
    assert set(predictions.columns) == set(baselines.BASELINES)
    pd.testing.assert_series_equal(
        predictions["baseline_previous_day"], table["lag_1d"], check_names=False
    )


def test_ridge_handles_missing_lags():
    table = _features()
    model = models.make_ridge(1.0).fit(table[FEATURES], table["consumption_mw"])
    assert np.isfinite(model.predict(table[FEATURES])).all()  # first week has NaN lags


def test_lgbm_search_space_is_reproducible():
    first = models.sample_lgbm_params(np.random.default_rng(3))
    second = models.sample_lgbm_params(np.random.default_rng(3))
    assert first == second
    assert 0.5 <= first["feature_fraction"] <= 1.0
