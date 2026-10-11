"""Evaluation protocol: chronological split without leakage, and error metrics.

    train  : start  -> val_start   (model fitting)
    val    : val_start -> test_start (model selection and hyperparameter search)
    test   : test_start -> end     (touched once, at the very end, by `evaluate`)

Shuffled K-fold would leak the future into training (consumption is autocorrelated), so
the split is strictly chronological. The final models are refit on train + validation
before the single test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from conso_forecast.config import Settings
from conso_forecast.features import TARGET

MIN_TEST_DAYS = 300


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def train_val(self) -> pd.DataFrame:
        return pd.concat([self.train, self.val])


def remove_periods(table: pd.DataFrame, periods: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """Drop rows whose local date falls in any of the inclusive [start, end] periods."""
    keep = np.ones(len(table), dtype=bool)
    dates = table.index.normalize()
    for start, end in periods:
        keep &= ~((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    return table.loc[keep]


def chronological_split(table: pd.DataFrame, settings: Settings) -> Split:
    val_start, test_start = pd.Timestamp(settings.val_start), pd.Timestamp(settings.test_start)
    if not val_start < test_start:
        raise ValueError("val_start must be before test_start")
    labelled = table.loc[table[TARGET].notna()]
    train = remove_periods(
        labelled.loc[labelled.index < val_start], settings.excluded_train_periods
    )
    val = labelled.loc[(labelled.index >= val_start) & (labelled.index < test_start)]
    test = labelled.loc[labelled.index >= test_start]
    test_days = test.index.normalize().nunique()
    if test_days < MIN_TEST_DAYS:
        raise ValueError(
            f"The test period only has {test_days} days of data (< {MIN_TEST_DAYS}). "
            "Move `test_start` / `val_start` earlier in config.py."
        )
    return Split(train=train, val=val, test=test)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Safe here: consumption never nears zero."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE is undefined when the true value is zero")
    return float(100 * np.mean(np.abs((y_true - y_pred) / y_true)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_mw": mae(y_true, y_pred),
        "rmse_mw": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }


def compare_models(y_true: pd.Series, predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics of every prediction column, computed on the rows where *all* are available.

    Using a common set of hours keeps the comparison fair (lag baselines have no value
    right after a data gap, for example).
    """
    mask = y_true.notna() & predictions.notna().all(axis=1)
    rows = {
        name: compute_metrics(y_true[mask].to_numpy(), predictions.loc[mask, name].to_numpy())
        for name in predictions.columns
    }
    result = pd.DataFrame(rows).T.sort_values("mae_mw")
    result["n_hours"] = int(mask.sum())
    return result
