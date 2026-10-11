"""Models: regularised linear regression (Ridge) and gradient boosting (LightGBM).

Model selection happens on the validation year only; every trial is logged to MLflow.
The selected configuration is then refit on train + validation and saved to `models/`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from conso_forecast import tracking
from conso_forecast.config import Settings
from conso_forecast.evaluation import compute_metrics
from conso_forecast.features import (
    CALENDAR_FLAGS,
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)

RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
EARLY_STOPPING_ROUNDS = 100
MAX_BOOSTING_ROUNDS = 5000


@dataclass
class TuningResult:
    params: dict[str, Any]
    val_metrics: dict[str, float]
    val_predictions: pd.Series
    trials: pd.DataFrame


# --------------------------------------------------------------------------- Ridge


def make_ridge(alpha: float) -> Pipeline:
    """One-hot calendar categories, impute and scale numeric features, then Ridge."""
    preprocess = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC_FEATURES + CALENDAR_FLAGS,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=alpha))])


def tune_ridge(train: pd.DataFrame, val: pd.DataFrame) -> TuningResult:
    trials, best = [], None
    for alpha in RIDGE_ALPHAS:
        model = make_ridge(alpha).fit(train[FEATURES], train[TARGET])
        predictions = pd.Series(model.predict(val[FEATURES]), index=val.index)
        metrics = compute_metrics(val[TARGET].to_numpy(), predictions.to_numpy())
        with tracking.run(f"ridge-alpha-{alpha}", nested=True, model="ridge"):
            tracking.log_params({"alpha": alpha})
            tracking.log_metrics(metrics, prefix="val_")
        trials.append({"alpha": alpha, **metrics})
        if best is None or metrics["mae_mw"] < best.val_metrics["mae_mw"]:
            best = TuningResult({"alpha": alpha}, metrics, predictions, pd.DataFrame())
    best.trials = pd.DataFrame(trials)
    return best


# --------------------------------------------------------------------------- LightGBM


def sample_lgbm_params(rng: np.random.Generator) -> dict[str, Any]:
    """One random configuration from the search space."""
    return {
        "learning_rate": float(10 ** rng.uniform(-2, -0.7)),
        "num_leaves": int(rng.choice([15, 31, 63, 127, 255])),
        "min_child_samples": int(rng.choice([10, 20, 50, 100, 200])),
        "feature_fraction": float(rng.uniform(0.5, 1.0)),
        "bagging_fraction": float(rng.uniform(0.5, 1.0)),
        "bagging_freq": 1,
        "lambda_l2": float(10 ** rng.uniform(-3, 1)),
        "objective": str(rng.choice(["l2", "l1"])),
    }


def make_lgbm(params: dict[str, Any], n_estimators: int, seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        **params,
        n_estimators=n_estimators,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )


def tune_lgbm(train: pd.DataFrame, val: pd.DataFrame, n_trials: int, seed: int) -> TuningResult:
    """Random search; each trial uses early stopping on the validation year."""
    rng = np.random.default_rng(seed)
    trials, best = [], None
    for trial in range(n_trials):
        params = sample_lgbm_params(rng)
        model = make_lgbm(params, MAX_BOOSTING_ROUNDS, seed)
        model.fit(
            train[FEATURES],
            train[TARGET],
            eval_X=val[FEATURES],
            eval_y=val[TARGET],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        params = {**params, "n_estimators": int(model.best_iteration_ or MAX_BOOSTING_ROUNDS)}
        predictions = pd.Series(model.predict(val[FEATURES]), index=val.index)
        metrics = compute_metrics(val[TARGET].to_numpy(), predictions.to_numpy())
        with tracking.run(f"lgbm-trial-{trial:02d}", nested=True, model="lightgbm"):
            tracking.log_params(params)
            tracking.log_metrics(metrics, prefix="val_")
        logger.info("LightGBM trial %d: val MAE %.0f MW", trial, metrics["mae_mw"])
        trials.append({**params, **metrics})
        if best is None or metrics["mae_mw"] < best.val_metrics["mae_mw"]:
            best = TuningResult(params, metrics, predictions, pd.DataFrame())
    best.trials = pd.DataFrame(trials)
    return best


# --------------------------------------------------------------------------- final fit


def fit_final_models(
    train_val: pd.DataFrame, ridge_params: dict[str, Any], lgbm_params: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Refit the selected configurations on train + validation."""
    ridge = make_ridge(ridge_params["alpha"]).fit(train_val[FEATURES], train_val[TARGET])
    params = dict(lgbm_params)
    n_estimators = params.pop("n_estimators")
    lgbm = make_lgbm(params, n_estimators, seed).fit(train_val[FEATURES], train_val[TARGET])
    return {"ridge": ridge, "lightgbm": lgbm}


def save_models(models: dict[str, Any], settings: Settings) -> None:
    settings.paths.models.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, settings.paths.models / f"{name}.joblib")


def load_models(settings: Settings) -> dict[str, Any]:
    models = {}
    for name in ("ridge", "lightgbm"):
        path = settings.paths.models / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `conso-forecast train` first")
        models[name] = joblib.load(path)
    return models
