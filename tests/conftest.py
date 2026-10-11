"""Shared fixtures: a small synthetic copy of the raw data, written in the real raw formats.

Tests never touch the network. The synthetic data mimics the real sources closely enough
(15-minute eCO2mix CSV in UTC, Open-Meteo JSON per city and year) to run every step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conso_forecast.config import CITIES, Paths, Settings
from conso_forecast.data.download import eco2mix_raw_path, weather_raw_path

README_TEMPLATE = "# Test README\n\n<!-- RESULTS:START -->\npending\n<!-- RESULTS:END -->\n"


def synthetic_temperature(index_utc: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    day_of_year = index_utc.dayofyear.to_numpy()
    hour = index_utc.hour.to_numpy()
    seasonal = 12 - 8 * np.cos(2 * np.pi * (day_of_year - 20) / 365.25)
    daily = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather_noise = np.repeat(rng.normal(0, 2.5, len(index_utc) // 24 + 1), 24)[: len(index_utc)]
    return seasonal + daily + weather_noise


def synthetic_consumption(index_utc: pd.DatetimeIndex, temperature: np.ndarray, rng) -> np.ndarray:
    local = index_utc.tz_convert("Europe/Paris")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    profile = 6000 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16) + 3000 * (hour >= 18) * (
        hour <= 21
    )
    weekend = np.where(local.dayofweek.to_numpy() >= 5, -0.08, 0.0)
    heating = 2400 * np.clip(15 - temperature, 0, None)
    base = 45_000 + profile + heating
    return base * (1 + weekend) + rng.normal(0, 600, len(index_utc))


def write_synthetic_raw(settings: Settings, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(settings.start_date, tz="UTC") - pd.Timedelta(hours=1)
    end = pd.Timestamp(settings.end_date, tz="UTC") + pd.Timedelta(hours=23)
    hours = pd.date_range(start, end, freq="h")
    city_temperatures = {c.name: synthetic_temperature(hours, rng) for c in settings.cities}
    mean_temperature = np.mean(list(city_temperatures.values()), axis=0)

    quarter = pd.date_range(start, end + pd.Timedelta(minutes=45), freq="15min")
    temperature_q = np.repeat(mean_temperature, 4)[: len(quarter)]
    consumption = synthetic_consumption(quarter, temperature_q, rng)
    eco2mix = pd.DataFrame(
        {
            "perimetre": "France",
            "nature": "Données consolidées",
            "date_heure": quarter.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "consommation": consumption.round(0),
            "prevision_j1": (consumption + rng.normal(0, 1200, len(quarter))).round(0),
        }
    )
    # Realistic defects: a duplicated record and an impossible value.
    eco2mix = pd.concat([eco2mix, eco2mix.iloc[[100]]], ignore_index=True)
    eco2mix.loc[200, "consommation"] = 0

    # Same chunking as the downloader: the first file also holds the last UTC day before.
    years = pd.to_datetime(eco2mix["date_heure"]).dt.year.clip(lower=settings.start_year)
    for year in range(settings.start_year, settings.end_year + 1):
        path = eco2mix_raw_path(settings, year)
        path.parent.mkdir(parents=True, exist_ok=True)
        eco2mix[years == year].to_csv(path, sep=";", index=False)

    for city in settings.cities:
        series = pd.Series(city_temperatures[city.name], index=hours)
        for year in range(settings.start_year, settings.end_year + 1):
            part = series[np.maximum(series.index.year, settings.start_year) == year]
            payload = {
                "latitude": city.latitude,
                "longitude": city.longitude,
                "hourly_units": {"time": "iso8601", "temperature_2m": "°C"},
                "hourly": {
                    "time": part.index.strftime("%Y-%m-%dT%H:%M").tolist(),
                    "temperature_2m": part.round(1).tolist(),
                },
            }
            path = weather_raw_path(settings, city, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")


def make_settings(root: Path) -> Settings:
    return Settings(
        paths=Paths(root=root),
        start_date="2020-01-01",
        end_date="2022-12-31",
        val_start="2021-07-01",
        test_start="2022-01-01",
        n_search_trials=2,
        cities=CITIES[:3],
    )


@pytest.fixture(scope="session")
def synthetic_settings(tmp_path_factory) -> Settings:
    """Settings pointing to a temporary project root that holds synthetic raw data."""
    root = tmp_path_factory.mktemp("project")
    settings = make_settings(root)
    settings.paths.ensure()
    write_synthetic_raw(settings)
    (root / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    return settings
