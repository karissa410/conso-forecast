"""Thin wrapper around MLflow so the rest of the code does not depend on it directly.

Runs are stored locally in `mlflow.db` (SQLite). Browse them with `make mlflow-ui`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from conso_forecast.config import Settings

EXPERIMENT_NAME = "conso-forecast"


def setup(settings: Settings) -> None:
    mlflow.set_tracking_uri(settings.paths.mlflow_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        artifacts = (settings.paths.root / "mlartifacts").resolve().as_uri()
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifacts)
    mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def run(name: str, nested: bool = False, **tags: str) -> Iterator[None]:
    with mlflow.start_run(run_name=name, nested=nested, tags=tags or None):
        yield


def log_params(params: dict[str, Any]) -> None:
    mlflow.log_params({key: str(value) for key, value in params.items()})


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    mlflow.log_metrics({f"{prefix}{key}": float(value) for key, value in metrics.items()})


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)
