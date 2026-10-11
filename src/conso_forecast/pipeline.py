"""Training (model selection on validation) and final evaluation (test set, used once)."""

from __future__ import annotations

import json
import logging

import pandas as pd

from conso_forecast import tracking
from conso_forecast.baselines import predict_baselines
from conso_forecast.config import Settings
from conso_forecast.data.clean import load_clean
from conso_forecast.evaluation import Split, chronological_split, compare_models
from conso_forecast.features import FEATURES, TARGET, build_features
from conso_forecast.models import (
    fit_final_models,
    load_models,
    save_models,
    tune_lgbm,
    tune_ridge,
)

logger = logging.getLogger(__name__)

RTE_COLUMN = "rte_forecast_d1_mw"
RTE_NAME = "rte_official_forecast"


def prepare(settings: Settings) -> tuple[pd.DataFrame, Split]:
    clean_df = load_clean(settings)
    return clean_df, chronological_split(build_features(clean_df), settings)


def rte_benchmark(clean_df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series | None:
    """RTE's own day-ahead forecast, when the source provides it (reference, not a feature)."""
    if RTE_COLUMN not in clean_df.columns:
        return None
    series = clean_df[RTE_COLUMN].reindex(index)
    return series.rename(RTE_NAME) if series.notna().mean() > 0.9 else None


def _with_benchmark(predictions: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    rte = rte_benchmark(clean_df, predictions.index)
    return predictions if rte is None else predictions.assign(**{RTE_NAME: rte})


def run_training(settings: Settings) -> pd.DataFrame:
    settings.paths.ensure()
    clean_df, split = prepare(settings)
    results = settings.paths.results
    tracking.setup(settings)
    with tracking.run("train", stage="selection"):
        tracking.log_params(
            {
                "train_rows": len(split.train),
                "val_rows": len(split.val),
                "val_start": settings.val_start,
                "test_start": settings.test_start,
                "n_features": len(FEATURES),
            }
        )
        ridge = tune_ridge(split.train, split.val)
        lgbm = tune_lgbm(split.train, split.val, settings.n_search_trials, settings.random_seed)

        predictions = predict_baselines(split.val).assign(
            ridge=ridge.val_predictions, lightgbm=lgbm.val_predictions
        )
        predictions = _with_benchmark(predictions, clean_df)
        val_metrics = compare_models(split.val[TARGET], predictions)
        for name, row in val_metrics.iterrows():
            tracking.log_metrics(row.drop("n_hours").to_dict(), prefix=f"val_{name}_")

    val_metrics.to_csv(results / "val_metrics.csv")
    ridge.trials.to_csv(results / "ridge_trials.csv", index=False)
    lgbm.trials.to_csv(results / "lgbm_trials.csv", index=False)
    best_params = {"ridge": ridge.params, "lightgbm": lgbm.params}
    (results / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    models = fit_final_models(split.train_val, ridge.params, lgbm.params, settings.random_seed)
    save_models(models, settings)
    logger.info("Validation metrics:\n%s", val_metrics.round(2))
    return val_metrics


def run_evaluation(settings: Settings) -> pd.DataFrame:
    """Score every model on the held-out test year. Run this once, after model selection."""
    clean_df, split = prepare(settings)
    models = load_models(settings)
    test = split.test
    predictions = predict_baselines(test).assign(
        ridge=models["ridge"].predict(test[FEATURES]),
        lightgbm=models["lightgbm"].predict(test[FEATURES]),
    )
    predictions = _with_benchmark(predictions, clean_df)
    test_metrics = compare_models(test[TARGET], predictions)

    results = settings.paths.results
    test_metrics.to_csv(results / "test_metrics.csv")
    predictions.assign(actual=test[TARGET]).to_parquet(results / "test_predictions.parquet")

    tracking.setup(settings)
    with tracking.run("evaluate-test", stage="test"):
        for name, row in test_metrics.iterrows():
            tracking.log_metrics(row.drop("n_hours").to_dict(), prefix=f"test_{name}_")
        tracking.log_artifact(str(results / "test_metrics.csv"))
    logger.info("Test metrics:\n%s", test_metrics.round(2))
    return test_metrics
