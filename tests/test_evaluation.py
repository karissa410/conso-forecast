import numpy as np
import pandas as pd
import pytest

from conso_forecast import evaluation
from conso_forecast.config import Settings


def test_metrics_on_known_values():
    y_true, y_pred = np.array([100.0, 200.0]), np.array([110.0, 180.0])
    assert evaluation.mae(y_true, y_pred) == 15.0
    assert evaluation.rmse(y_true, y_pred) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert evaluation.mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_rejects_zero_targets():
    with pytest.raises(ValueError):
        evaluation.mape(np.array([0.0, 1.0]), np.array([1.0, 1.0]))


def _table(start: str, end: str) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"consumption_mw": 1.0}, index=index)


def test_split_is_chronological_and_disjoint():
    settings = Settings(
        val_start="2021-01-01",
        test_start="2022-01-01",
        excluded_train_periods=(("2020-03-17", "2020-05-10"),),
    )
    split = evaluation.chronological_split(_table("2019-01-01", "2022-12-31 23:00"), settings)
    assert split.train.index.max() < split.val.index.min()
    assert split.val.index.max() < split.test.index.min()
    assert split.test.index.min() == pd.Timestamp("2022-01-01")
    assert not split.train.index.normalize().isin([pd.Timestamp("2020-04-01")]).any()
    assert len(split.train_val) == len(split.train) + len(split.val)


def test_split_refuses_a_too_short_test_period():
    settings = Settings(val_start="2021-01-01", test_start="2022-01-01")
    with pytest.raises(ValueError, match="test period"):
        evaluation.chronological_split(_table("2019-01-01", "2022-03-31"), settings)


def test_compare_models_uses_common_hours():
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    y = pd.Series([100.0, 100.0, 100.0], index=index)
    predictions = pd.DataFrame({"a": [90.0, 100.0, np.nan], "b": [100.0, 120.0, 0.0]}, index=index)
    result = evaluation.compare_models(y, predictions)
    assert result.loc["a", "mae_mw"] == 5.0
    assert result.loc["b", "mae_mw"] == 10.0
    assert (result["n_hours"] == 2).all()
    assert result.index[0] == "a"  # sorted best first
