from urllib.parse import unquote

from conso_forecast.config import CITIES, Paths, Settings
from conso_forecast.data import download


def test_eco2mix_url_targets_one_utc_year(tmp_path):
    settings = Settings(paths=Paths(root=tmp_path))
    url = unquote(download.eco2mix_export_url(settings, 2019))
    assert settings.eco2mix_dataset in url
    assert "/exports/csv" in url
    assert "date'2019-01-01'" in url and "date'2020-01-01'" in url


def test_first_chunk_starts_one_day_early_to_cover_the_first_local_hour(tmp_path):
    settings = Settings(paths=Paths(root=tmp_path), start_date="2015-01-01")
    url = unquote(download.eco2mix_export_url(settings, 2015))
    assert "date'2014-12-31'" in url
    assert download.open_meteo_params(settings, CITIES[0], 2015)["start_date"] == "2014-12-31"
    assert "timezone=UTC" in url


def test_open_meteo_params_request_utc_hourly_temperature(tmp_path):
    settings = Settings(paths=Paths(root=tmp_path))
    params = download.open_meteo_params(settings, CITIES[0], 2018)
    assert params["start_date"] == "2018-01-01"
    assert params["end_date"] == "2018-12-31"
    assert params["hourly"] == "temperature_2m"
    assert params["timezone"] == "GMT"


def test_write_atomically_leaves_no_partial_file(tmp_path):
    target = tmp_path / "sub" / "file.csv"
    download._write_atomically(target, b"a;b\n1;2\n")
    assert target.read_bytes() == b"a;b\n1;2\n"
    assert list(target.parent.iterdir()) == [target]


class _FailingSession:
    def get(self, *args, **kwargs):
        raise AssertionError("existing files must not be downloaded again")


def test_existing_files_are_not_downloaded_again(tmp_path):
    settings = Settings(
        paths=Paths(root=tmp_path),
        start_date="2020-01-01",
        end_date="2020-12-31",
        cities=CITIES[:1],
    )
    download.eco2mix_raw_path(settings, 2020).parent.mkdir(parents=True)
    download.eco2mix_raw_path(settings, 2020).write_text("x")
    download.weather_raw_path(settings, CITIES[0], 2020).parent.mkdir(parents=True)
    download.weather_raw_path(settings, CITIES[0], 2020).write_text("{}")
    download.download_eco2mix(settings, _FailingSession())
    download.download_weather(settings, _FailingSession())
