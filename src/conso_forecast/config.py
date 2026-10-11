"""Project-wide configuration: paths, study period, data sources and evaluation split.

Every other module reads its settings from here, so changing the study period or the
test year is a one-line edit in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Timestamps are stored in UTC; calendar features are computed in French local time.
LOCAL_TZ = "Europe/Paris"


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    population_weight: float


# Ten largest French metropolitan areas. Weights are approximate populations of the
# INSEE "aires d'attraction des villes" (millions, 2020). The national temperature is
# their population-weighted mean, a simplified version of what RTE does with 32 stations.
CITIES: tuple[City, ...] = (
    City("Paris", 48.857, 2.352, 13.0),
    City("Lyon", 45.764, 4.836, 2.3),
    City("Marseille", 43.297, 5.381, 1.9),
    City("Lille", 50.629, 3.057, 1.5),
    City("Toulouse", 43.605, 1.444, 1.5),
    City("Bordeaux", 44.838, -0.579, 1.4),
    City("Nantes", 47.218, -1.554, 1.0),
    City("Strasbourg", 48.573, 7.752, 0.8),
    City("Montpellier", 43.611, 3.877, 0.8),
    City("Rennes", 48.117, -1.678, 0.8),
)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. Tests build one pointing to a temporary directory."""

    root: Path = PROJECT_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def results(self) -> Path:
        return self.reports / "results"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def mlflow_uri(self) -> str:
        return f"sqlite:///{(self.root / 'mlflow.db').as_posix()}"

    def ensure(self) -> None:
        for directory in (self.raw, self.processed, self.figures, self.results, self.models):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)

    # Study period (inclusive, local calendar dates).
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"

    # Evaluation protocol: train < val_start <= validation < test_start <= test.
    # The test year is only used once, by `conso-forecast evaluate`, after model selection.
    val_start: str = "2024-01-01"
    test_start: str = "2025-01-01"

    # Periods removed from *training* only (they stay in the cleaned data and in the EDA).
    # First COVID-19 lockdown: consumption patterns unrelated to anything the model can know.
    excluded_train_periods: tuple[tuple[str, str], ...] = (("2020-03-17", "2020-05-10"),)

    # Gaps up to this many hours are interpolated during cleaning; longer gaps stay missing.
    max_interpolation_gap_hours: int = 3

    # Hyperparameter search budget for LightGBM.
    n_search_trials: int = 20
    random_seed: int = 42

    # Sources (both open data, see README "Data").
    eco2mix_dataset: str = "eco2mix-national-cons-def"
    odre_api: str = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
    open_meteo_api: str = "https://archive-api.open-meteo.com/v1/archive"
    open_meteo_model: str = "era5"
    cities: tuple[City, ...] = CITIES

    @property
    def start_year(self) -> int:
        return int(self.start_date[:4])

    @property
    def end_year(self) -> int:
        return int(self.end_date[:4])
