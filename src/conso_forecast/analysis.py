"""Error analysis on the test year: when and where the best model fails, and why.

Runs after `evaluate`. Looking at the test set here is fine because no modelling decision
is taken from it any more; the conclusions feed the "Limitations" section of the report.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from conso_forecast import plotting
from conso_forecast.baselines import BASELINES
from conso_forecast.config import Settings
from conso_forecast.data.clean import load_clean
from conso_forecast.eda import daily_table
from conso_forecast.features import FEATURES, TARGET, build_features
from conso_forecast.models import load_models
from conso_forecast.pipeline import RTE_NAME

OUR_MODELS = ("ridge", "lightgbm")
TEMPERATURE_BINS = [-np.inf, 0, 5, 10, 15, 20, 25, np.inf]
TEMPERATURE_LABELS = ["< 0", "0-5", "5-10", "10-15", "15-20", "20-25", "> 25"]
PERMUTATION_SAMPLE = 5000


def load_test_results(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = settings.paths.results
    metrics_path = results / "test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found: run `conso-forecast evaluate` first")
    metrics = pd.read_csv(metrics_path, index_col=0)
    predictions = pd.read_parquet(results / "test_predictions.parquet")
    return metrics, predictions


def pick_models(metrics: pd.DataFrame) -> tuple[str, str]:
    """Best of our models and best naive baseline, by test MAE."""
    best_model = metrics.loc[list(OUR_MODELS), "mae_mw"].idxmin()
    best_baseline = metrics.loc[list(BASELINES), "mae_mw"].idxmin()
    return best_model, best_baseline


def absolute_errors(predictions: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    mask = predictions.notna().all(axis=1)
    kept = predictions.loc[mask]
    return pd.DataFrame({n: (kept[n] - kept["actual"]).abs() for n in names}, index=kept.index)


def mae_by(errors: pd.DataFrame, key: pd.Series | pd.Index) -> pd.DataFrame:
    return errors.groupby(key, observed=True).mean()


def worst_days(
    predictions: pd.DataFrame, model: str, daily: pd.DataFrame, n: int = 10
) -> list[dict]:
    errors = (predictions[model] - predictions["actual"]).abs()
    by_day = errors.groupby(errors.index.normalize()).mean().nlargest(n)
    return [
        {
            "date": day.strftime("%Y-%m-%d (%a)"),
            "mae_mw": round(float(value), 0),
            "day_type": daily.loc[day, "day_type"],
            "temperature_c": round(float(daily.loc[day, "temperature_c"]), 1),
        }
        for day, value in by_day.items()
    ]


def feature_importances(model, test: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Gain importance (from training) and permutation importance (MAE increase on test)."""
    gain = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=FEATURES, dtype=float
    )
    sample = test.dropna(subset=[TARGET]).sample(
        min(PERMUTATION_SAMPLE, len(test)), random_state=seed
    )
    permutation = permutation_importance(
        model,
        sample[FEATURES],
        sample[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=seed,
    )
    return pd.DataFrame(
        {
            "gain_pct": 100 * gain / gain.sum(),
            "permutation_mae_increase_mw": permutation.importances_mean,
        },
        index=FEATURES,
    ).sort_values("permutation_mae_increase_mw", ascending=False)


# --------------------------------------------------------------------------- figures


def _plot_profile(errors: pd.DataFrame, key, xlabel: str, title: str, path, xticks=None) -> None:
    table = mae_by(errors, key)
    fig, ax = plotting.new_figure(7.5, 3.8)
    for name in table.columns:
        ax.plot(
            table.index,
            table[name],
            marker="o",
            markersize=4,
            color=plotting.MODEL_COLORS[name],
            label=plotting.MODEL_LABELS[name],
        )
    if xticks is not None:
        ax.set_xticks(*xticks)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean absolute error (MW)")
    ax.set_title(title)
    fig.legend(loc="outside lower center", ncols=len(table.columns))
    plotting.save(fig, path)


def plot_importance(importance: pd.DataFrame, path, top: int = 15) -> None:
    top_features = importance.head(top).iloc[::-1]
    fig, ax = plotting.new_figure(7, 4.6)
    ax.barh(
        top_features.index,
        top_features["permutation_mae_increase_mw"],
        color=plotting.SERIES[0],
        height=0.6,
    )
    ax.set_xlabel("Increase in test MAE when the feature is shuffled (MW)")
    ax.set_title("Which inputs the LightGBM forecast relies on")
    ax.grid(axis="y", visible=False)
    plotting.save(fig, path)


def plot_example_weeks(predictions: pd.DataFrame, model: str, baseline: str, path) -> None:
    """Small multiples: the worst week and a typical (median-error) week of the test year."""
    errors = (predictions[model] - predictions["actual"]).abs()
    weekly = errors.groupby(errors.index.to_period("W")).mean()
    weekly = weekly[errors.groupby(errors.index.to_period("W")).size() == 168]
    worst = weekly.idxmax()
    typical = (weekly - weekly.median()).abs().idxmin()
    fig, axes = plotting.plt.subplots(2, 1, figsize=(9, 6), layout="constrained", sharey=True)
    for ax, week, label in ((axes[0], worst, "Worst week"), (axes[1], typical, "Typical week")):
        window = predictions[predictions.index.to_period("W") == week]
        for name in ("actual", baseline, model):
            ax.plot(
                window.index,
                window[name],
                color=plotting.MODEL_COLORS[name],
                label=plotting.MODEL_LABELS[name],
                linewidth=2.0 if name == "actual" else plotting.LINE_WIDTH,
            )
        ax.set_title(f"{label}: {week.start_time:%d %b %Y}, MAE {weekly[week]:,.0f} MW")
        plotting.thousands_gw(ax.yaxis)
        ax.set_ylabel("GW")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    plotting.save(fig, path)


def run_analysis(settings: Settings) -> dict:
    settings.paths.ensure()
    metrics, predictions = load_test_results(settings)
    best_model, best_baseline = pick_models(metrics)
    compared = [best_model, best_baseline] + ([RTE_NAME] if RTE_NAME in metrics.index else [])
    errors = absolute_errors(predictions, compared)

    clean_df = load_clean(settings)
    daily = daily_table(clean_df)
    day_type = daily["day_type"].reindex(errors.index.normalize()).to_numpy()
    temperature = clean_df["temperature_c"].reindex(errors.index)
    temperature_band = pd.cut(temperature, TEMPERATURE_BINS, labels=TEMPERATURE_LABELS)

    figures = settings.paths.figures
    _plot_profile(
        errors,
        errors.index.hour,
        "Hour of day (local time)",
        "Forecast error by hour of day",
        figures / "errors_by_hour.png",
        xticks=(range(0, 24, 3),),
    )
    _plot_profile(
        errors,
        errors.index.month,
        "Month",
        "Forecast error by month (test year)",
        figures / "errors_by_month.png",
        xticks=(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]),
    )
    plot_example_weeks(predictions, best_model, best_baseline, figures / "example_weeks.png")

    models = load_models(settings)
    test = build_features(clean_df).loc[predictions.index]
    importance = feature_importances(models["lightgbm"], test, settings.random_seed)
    importance.to_csv(settings.paths.results / "feature_importance.csv")
    plot_importance(importance, figures / "feature_importance.png")

    signed_error = predictions[best_model] - predictions["actual"]
    summary = {
        "best_model": best_model,
        "best_baseline": best_baseline,
        "mae_by_day_type": mae_by(errors, day_type).round(0).to_dict(orient="index"),
        "mae_by_temperature_band": mae_by(errors, temperature_band)
        .round(0)
        .to_dict(orient="index"),
        "mae_by_month": mae_by(errors, errors.index.month).round(0).to_dict(orient="index"),
        "mae_by_hour": mae_by(errors, errors.index.hour).round(0).to_dict(orient="index"),
        "mean_bias_mw": round(float(signed_error.mean()), 0),
        "worst_days": worst_days(predictions, best_model, daily),
        "top_features": importance.head(5).index.tolist(),
    }
    (settings.paths.results / "error_analysis.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
