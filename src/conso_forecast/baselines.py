"""Naive baselines. Any model worth deploying must beat these.

Both are leak-free under the forecasting setup (issued at the end of day D):
- persistence 1 day: consumption at the same hour on day D ("yesterday's value");
- persistence 7 days: consumption at the same hour one week earlier (same weekday).
"""

from __future__ import annotations

import pandas as pd

BASELINES = {
    "baseline_previous_day": "lag_1d",
    "baseline_previous_week": "lag_7d",
}


def predict_baselines(features: pd.DataFrame) -> pd.DataFrame:
    """One column per baseline, read from the matching lag feature."""
    return pd.DataFrame({name: features[lag] for name, lag in BASELINES.items()})
