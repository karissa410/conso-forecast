"""Shared figure style: thin marks, recessive axes, one fixed colour per role."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: figures are written to files
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"  # de-emphasised series (baselines, "everything else")

# Categorical slots in fixed order: a model keeps its colour in every figure.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MODEL_COLORS = {
    "lightgbm": SERIES[0],
    "ridge": SERIES[1],
    "rte_official_forecast": SERIES[2],
    "baseline_previous_day": NEUTRAL,
    "baseline_previous_week": SERIES[3],
    "actual": INK,
}
MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "ridge": "Ridge regression",
    "rte_official_forecast": "RTE official forecast",
    "baseline_previous_day": "Baseline: previous day",
    "baseline_previous_week": "Baseline: previous week",
    "actual": "Actual",
}
LINE_WIDTH = 1.5

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "font.size": 10,
        "lines.linewidth": LINE_WIDTH,
        "figure.dpi": 110,
    }
)


def new_figure(width: float = 8.0, height: float = 4.0) -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots(figsize=(width, height), layout="constrained")


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def thousands_gw(axis) -> None:
    """Show MW values as GW on the given axis (e.g. ax.yaxis)."""
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
