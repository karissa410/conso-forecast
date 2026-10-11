"""Command-line entry point: `conso-forecast <step>` (or `python -m conso_forecast <step>`).

Steps, in pipeline order: download, clean, eda, train, evaluate, analyze, report.
`conso-forecast all` runs every step; it is what `make all` calls.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from conso_forecast.config import Settings


def _download(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.data.download import download_all

    download_all(settings, force=args.force)


def _clean(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.data.clean import run_cleaning

    run_cleaning(settings)


def _eda(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.eda import run_eda

    run_eda(settings)


def _train(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.pipeline import run_training

    run_training(settings)


def _evaluate(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.pipeline import run_evaluation

    run_evaluation(settings)


def _analyze(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.analysis import run_analysis

    run_analysis(settings)


def _report(settings: Settings, args: argparse.Namespace) -> None:
    from conso_forecast.report import run_report

    path = run_report(settings)
    logging.getLogger(__name__).info("Report written to %s", path)


STEPS: dict[str, Callable[[Settings, argparse.Namespace], None]] = {
    "download": _download,
    "clean": _clean,
    "eda": _eda,
    "train": _train,
    "evaluate": _evaluate,
    "analyze": _analyze,
    "report": _report,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="conso-forecast", description=__doc__)
    parser.add_argument("step", choices=[*STEPS, "all"], help="pipeline step to run")
    parser.add_argument("--force", action="store_true", help="re-download existing raw files")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    settings = Settings()
    steps = list(STEPS) if args.step == "all" else [args.step]
    for step in steps:
        logging.getLogger(__name__).info("=== %s ===", step)
        STEPS[step](settings, args)


if __name__ == "__main__":
    main()
