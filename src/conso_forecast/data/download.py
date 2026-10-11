"""Download raw data: RTE eCO2mix consumption (ODRE) and Open-Meteo hourly temperature.

Raw files are written once and never modified: re-running skips files that already
exist, unless `force=True`. Everything downstream is rebuilt from these files.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from conso_forecast.config import City, Settings

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 120
# Pause between Open-Meteo requests: the free API limits the request rate per minute.
OPEN_METEO_PAUSE_SECONDS = 2.0


def make_session() -> requests.Session:
    """HTTP session that retries transient errors and rate limiting (HTTP 429)."""
    retry = Retry(
        total=8,
        backoff_factor=5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "conso-forecast (open-source portfolio project)"
    return session


def first_day(settings: Settings, year: int) -> str:
    """First UTC day to download for `year`.

    The first local hour of the study period (00:00 in Paris) is 23:00 UTC the day before,
    so the first chunk starts one day early. Later chunks do not overlap.
    """
    return f"{year - 1}-12-31" if year == settings.start_year else f"{year}-01-01"


def eco2mix_export_url(settings: Settings, year: int) -> str:
    """URL of the CSV export of one calendar year (UTC) of the national eCO2mix dataset."""
    start = first_day(settings, year)
    where = f"date_heure >= date'{start}' AND date_heure < date'{year + 1}-01-01'"
    return (
        f"{settings.odre_api}/{settings.eco2mix_dataset}/exports/csv"
        f"?where={requests.utils.quote(where)}&order_by=date_heure&timezone=UTC&delimiter=%3B"
    )


def open_meteo_params(settings: Settings, city: City, year: int) -> dict[str, str | float]:
    """Query parameters for one year of hourly 2 m temperature at one city (UTC timestamps)."""
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": first_day(settings, year),
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m",
        "models": settings.open_meteo_model,
        "timezone": "GMT",
    }


def _write_atomically(path: Path, content: bytes) -> None:
    """Write to a temporary file then rename, so an interrupted download leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def eco2mix_raw_path(settings: Settings, year: int) -> Path:
    return settings.paths.raw / "eco2mix" / f"eco2mix_{year}.csv"


def weather_raw_path(settings: Settings, city: City, year: int) -> Path:
    return settings.paths.raw / "weather" / f"{city.name.lower()}_{year}.json"


def download_eco2mix(settings: Settings, session: requests.Session, force: bool = False) -> None:
    for year in range(settings.start_year, settings.end_year + 1):
        path = eco2mix_raw_path(settings, year)
        if path.exists() and not force:
            logger.info("eCO2mix %s already downloaded", year)
            continue
        logger.info("Downloading eCO2mix %s", year)
        response = session.get(eco2mix_export_url(settings, year), timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        _write_atomically(path, response.content)


def download_weather(settings: Settings, session: requests.Session, force: bool = False) -> None:
    for city in settings.cities:
        for year in range(settings.start_year, settings.end_year + 1):
            path = weather_raw_path(settings, city, year)
            if path.exists() and not force:
                continue
            logger.info("Downloading temperature for %s %s", city.name, year)
            response = session.get(
                settings.open_meteo_api,
                params=open_meteo_params(settings, city, year),
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            _write_atomically(path, response.content)
            time.sleep(OPEN_METEO_PAUSE_SECONDS)


def download_all(settings: Settings, force: bool = False) -> None:
    settings.paths.ensure()
    session = make_session()
    download_eco2mix(settings, session, force=force)
    download_weather(settings, session, force=force)
