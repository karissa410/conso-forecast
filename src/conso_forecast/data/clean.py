"""Turn raw downloads into one clean hourly table on a French local-time grid.

Design choice: the model works on a *local wall-clock grid* with exactly 24 values per
day. French consumption follows local time (people wake up at 7:00 local, not UTC), and a
regular grid makes "same hour yesterday" an exact 24-step shift. On daylight-saving days
the duplicated autumn hour is averaged and the missing spring hour is interpolated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from conso_forecast.config import LOCAL_TZ, City, Settings
from conso_forecast.data.download import eco2mix_raw_path, weather_raw_path

logger = logging.getLogger(__name__)

# French national consumption has stayed between ~28 GW (summer nights) and ~102 GW
# (February 2012 record). Values outside this band are recording errors.
PLAUSIBLE_CONSUMPTION_MW = (20_000.0, 110_000.0)
PLAUSIBLE_TEMPERATURE_C = (-30.0, 50.0)

ECO2MIX_COLUMNS = {
    "date_heure": "timestamp",
    "consommation": "consumption_mw",
    "prevision_j1": "rte_forecast_d1_mw",
}


@dataclass
class QualityReport:
    expected_hours: int = 0
    duplicated_timestamps: int = 0
    implausible_values: int = 0
    missing_hours_before_interpolation: int = 0
    interpolated_hours: int = 0
    missing_hours_after_cleaning: int = 0
    missing_temperature_hours: int = 0
    first_timestamp: str = ""
    last_timestamp: str = ""


# --------------------------------------------------------------------------- parsing


def parse_eco2mix(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the useful eCO2mix columns, renamed, indexed by UTC timestamp.

    The RTE day-ahead forecast (`prevision_j1`) is kept only as an external benchmark:
    it must never be used as a model feature.
    """
    missing = {"date_heure", "consommation"} - set(raw.columns)
    if missing:
        raise ValueError(f"eCO2mix file is missing required columns: {sorted(missing)}")
    df = raw[[c for c in ECO2MIX_COLUMNS if c in raw.columns]].rename(columns=ECO2MIX_COLUMNS)
    if "rte_forecast_d1_mw" not in df.columns:
        df["rte_forecast_d1_mw"] = np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for column in ("consumption_mw", "rte_forecast_d1_mw"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.set_index("timestamp").sort_index()


def parse_open_meteo(payload: dict) -> pd.Series:
    """Hourly temperature series (UTC index) from one Open-Meteo archive JSON response."""
    hourly = payload["hourly"]
    index = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    values = pd.to_numeric(pd.Series(hourly["temperature_2m"], index=index), errors="coerce")
    return values.rename("temperature_c")


# --------------------------------------------------------------------------- cleaning steps


def drop_duplicate_timestamps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicated = df.index.duplicated(keep="first")
    return df.loc[~duplicated], int(duplicated.sum())


def mask_implausible(series: pd.Series, low: float, high: float) -> tuple[pd.Series, int]:
    """Replace values outside [low, high] (and zeros reported as data) by NaN."""
    bad = series.notna() & ((series < low) | (series > high))
    return series.mask(bad), int(bad.sum())


def to_hourly_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Average sub-hourly (15 or 30 min) records into hours labelled by their start."""
    return df.resample("h", label="left", closed="left").mean()


def to_local_grid(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Re-index an hourly UTC frame on a regular, timezone-naive local wall-clock grid.

    The duplicated autumn hour is averaged; the missing spring hour becomes NaN (it is
    filled later by short-gap interpolation).
    """
    local = df.copy()
    local.index = local.index.tz_convert(tz).tz_localize(None)
    local = local.groupby(level=0).mean()
    full_index = pd.date_range(local.index.min(), local.index.max(), freq="h")
    local = local.reindex(full_index)
    local.index.name = "datetime"
    return local


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate runs of at most `max_gap` consecutive NaN; leave longer runs.

    `Series.interpolate(limit=...)` is not enough: it would partially fill long gaps.
    """
    is_nan = series.isna()
    run_id = (is_nan != is_nan.shift()).cumsum()
    run_length = is_nan.groupby(run_id).transform("sum")
    fillable = is_nan & (run_length <= max_gap)
    interpolated = series.interpolate(method="linear", limit_area="inside")
    return series.where(~fillable, interpolated)


def national_temperature(city_temperatures: pd.DataFrame, cities: tuple[City, ...]) -> pd.Series:
    """Population-weighted mean temperature; weights are renormalised when a city is missing."""
    weights = pd.Series({c.name: c.population_weight for c in cities})
    weights = weights[city_temperatures.columns]
    available = city_temperatures.notna()
    weighted_sum = city_temperatures.fillna(0.0).mul(weights, axis=1).sum(axis=1)
    weight_total = available.mul(weights, axis=1).sum(axis=1)
    return (weighted_sum / weight_total.replace(0.0, np.nan)).rename("temperature_c")


# --------------------------------------------------------------------------- full pipeline


def load_raw_eco2mix(settings: Settings) -> pd.DataFrame:
    frames = []
    for year in range(settings.start_year, settings.end_year + 1):
        path = eco2mix_raw_path(settings, year)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `conso-forecast download` first")
        frames.append(pd.read_csv(path, sep=";", low_memory=False))
    return pd.concat(frames, ignore_index=True)


def load_raw_weather(settings: Settings) -> pd.DataFrame:
    columns = {}
    for city in settings.cities:
        parts = []
        for year in range(settings.start_year, settings.end_year + 1):
            path = weather_raw_path(settings, city, year)
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run `conso-forecast download` first")
            parts.append(parse_open_meteo(json.loads(path.read_text(encoding="utf-8"))))
        columns[city.name] = pd.concat(parts)
    return pd.DataFrame(columns)


def clean(
    raw_eco2mix: pd.DataFrame, raw_weather: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, QualityReport]:
    """Full cleaning: returns the hourly local-time table and a data-quality summary."""
    report = QualityReport()

    conso = parse_eco2mix(raw_eco2mix)
    conso, report.duplicated_timestamps = drop_duplicate_timestamps(conso)
    implausible_total = 0
    for column in ("consumption_mw", "rte_forecast_d1_mw"):
        conso[column], n_bad = mask_implausible(conso[column], *PLAUSIBLE_CONSUMPTION_MW)
        implausible_total += n_bad
    conso = to_local_grid(to_hourly_utc(conso))

    weather = raw_weather.copy()
    weather = weather[~weather.index.duplicated(keep="first")].sort_index()
    for column in weather.columns:
        weather[column], n_bad = mask_implausible(weather[column], *PLAUSIBLE_TEMPERATURE_C)
        implausible_total += n_bad
    temperature = to_local_grid(national_temperature(weather, settings.cities).to_frame())
    report.implausible_values = implausible_total

    start = pd.Timestamp(settings.start_date)
    end = pd.Timestamp(settings.end_date) + pd.Timedelta(hours=23)
    grid = pd.date_range(start, end, freq="h", name="datetime")
    df = conso.join(temperature, how="outer").reindex(grid)

    # Trailing hours not yet published are not "missing data": cut the table at the last
    # observed consumption so they are not counted as gaps.
    last_observed = df["consumption_mw"].last_valid_index()
    if last_observed is None:
        raise ValueError("No consumption data in the requested period")
    df = df.loc[:last_observed]

    report.expected_hours = len(df)
    report.missing_hours_before_interpolation = int(df["consumption_mw"].isna().sum())
    gap = settings.max_interpolation_gap_hours
    for column in df.columns:
        df[column] = interpolate_short_gaps(df[column], gap)
    report.missing_hours_after_cleaning = int(df["consumption_mw"].isna().sum())
    report.interpolated_hours = (
        report.missing_hours_before_interpolation - report.missing_hours_after_cleaning
    )
    report.missing_temperature_hours = int(df["temperature_c"].isna().sum())
    report.first_timestamp = str(df.index.min())
    report.last_timestamp = str(df.index.max())
    return df, report


def run_cleaning(settings: Settings) -> pd.DataFrame:
    settings.paths.ensure()
    df, report = clean(load_raw_eco2mix(settings), load_raw_weather(settings), settings)
    df.to_parquet(settings.paths.processed / "hourly.parquet")
    (settings.paths.results / "data_quality.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    logger.info("Clean data: %s hours from %s to %s", len(df), df.index.min(), df.index.max())
    return df


def load_clean(settings: Settings) -> pd.DataFrame:
    path = settings.paths.processed / "hourly.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `conso-forecast clean` first")
    return pd.read_parquet(path)
