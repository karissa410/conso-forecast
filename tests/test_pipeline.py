"""End-to-end run of every pipeline step on synthetic data (no network)."""

import json

import pandas as pd

from conso_forecast.analysis import run_analysis
from conso_forecast.data.clean import run_cleaning
from conso_forecast.eda import run_eda
from conso_forecast.pipeline import run_evaluation, run_training
from conso_forecast.report import README_START, run_report


def test_full_pipeline_on_synthetic_data(synthetic_settings):
    settings = synthetic_settings
    run_cleaning(settings)
    run_eda(settings)
    val_metrics = run_training(settings)
    test_metrics = run_evaluation(settings)
    run_analysis(settings)
    report_path = run_report(settings)

    results, figures = settings.paths.results, settings.paths.figures
    for name in (
        "data_quality.json",
        "eda_summary.json",
        "val_metrics.csv",
        "test_metrics.csv",
        "best_params.json",
        "error_analysis.json",
        "feature_importance.csv",
    ):
        assert (results / name).exists(), name
    for name in (
        "eda_daily_consumption.png",
        "errors_by_month.png",
        "feature_importance.png",
        "example_weeks.png",
    ):
        assert (figures / name).exists(), name
    assert (settings.paths.models / "lightgbm.joblib").exists()
    assert (settings.paths.root / "mlflow.db").exists()

    # On this easy synthetic signal the trained models must beat the naive baselines.
    for metrics in (val_metrics, test_metrics):
        assert metrics.loc["lightgbm", "mae_mw"] < metrics.loc["baseline_previous_day", "mae_mw"]
    assert "rte_official_forecast" in test_metrics.index

    # The test predictions only cover the test year.
    predictions = pd.read_parquet(results / "test_predictions.parquet")
    assert predictions.index.min() >= pd.Timestamp(settings.test_start)

    report_text = report_path.read_text(encoding="utf-8")
    assert "lower** than the best naive baseline" in report_text
    readme = (settings.paths.root / "README.md").read_text(encoding="utf-8")
    assert README_START in readme and "pending" not in readme
    assert json.loads((results / "error_analysis.json").read_text())["best_model"] in (
        "ridge",
        "lightgbm",
    )
