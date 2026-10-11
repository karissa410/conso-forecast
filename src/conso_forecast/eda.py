"""Exploratory analysis: seasonality, calendar effects, temperature effect, missing values.

Produces the figures in `reports/figures/eda_*.png` and the numbers in
`reports/results/eda_summary.json` that the report quotes. The notebook
`notebooks/01_exploration.ipynb` calls these functions for interactive exploration.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from conso_forecast import plotting
from conso_forecast.config import Settings
from conso_forecast.data.clean import load_clean
from conso_forecast.features import HEATING_THRESHOLD_C, TARGET, calendar_features

DAY_TYPES = ("Weekday", "Saturday", "Sunday", "Bridge day", "Public holiday")


def daily_table(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean consumption and temperature with calendar flags."""
    daily = df[[TARGET, "temperature_c"]].groupby(df.index.normalize()).mean()
    calendar = calendar_features(daily.index)
    day_type = np.select(
        [
            calendar["is_holiday"] == 1,
            calendar["is_bridge"] == 1,
            calendar["day_of_week"] == 5,
            calendar["day_of_week"] == 6,
        ],
        ["Public holiday", "Bridge day", "Saturday", "Sunday"],
        default="Weekday",
    )
    return daily.assign(day_type=day_type, month=daily.index.month, year=daily.index.year)


def calendar_effects(daily: pd.DataFrame) -> pd.Series:
    """Mean deviation (%) of each day type from ordinary weekdays of the same month.

    Comparing within the same month removes the (much larger) seasonal effect.
    """
    period = daily.index.to_period("M")
    reference = (
        daily[TARGET].where(daily["day_type"] == "Weekday").groupby(period).transform("mean")
    )
    deviation = 100 * (daily[TARGET] / reference - 1)
    return deviation.groupby(daily["day_type"]).mean().reindex(DAY_TYPES)


def heating_gradient(daily: pd.DataFrame) -> float:
    """MW of extra daily-mean consumption per degree below the heating threshold (weekdays)."""
    cold = daily[(daily["day_type"] == "Weekday") & (daily["temperature_c"] < HEATING_THRESHOLD_C)]
    cold = cold.dropna(subset=[TARGET, "temperature_c"])
    slope, _ = np.polyfit(cold["temperature_c"], cold[TARGET], 1)
    return float(-slope)


def summarize(df: pd.DataFrame) -> dict:
    daily = daily_table(df)
    winter = df[TARGET][df.index.month.isin([12, 1, 2])].mean()
    summer = df[TARGET][df.index.month.isin([6, 7, 8])].mean()
    missing_by_year = df[TARGET].isna().groupby(df.index.year).sum()
    return {
        "first_hour": str(df.index.min()),
        "last_hour": str(df.index.max()),
        "n_hours": len(df),
        "mean_consumption_mw": float(df[TARGET].mean()),
        "min_consumption_mw": float(df[TARGET].min()),
        "max_consumption_mw": float(df[TARGET].max()),
        "max_consumption_at": str(df[TARGET].idxmax()),
        "winter_over_summer_ratio": float(winter / summer),
        "heating_gradient_mw_per_degree": heating_gradient(daily),
        "calendar_effects_pct": calendar_effects(daily).round(2).to_dict(),
        "temperature_correlation_daily": float(daily[TARGET].corr(daily["temperature_c"])),
        "missing_hours_by_year": {int(k): int(v) for k, v in missing_by_year.items()},
    }


# --------------------------------------------------------------------------- figures


def plot_daily_consumption(df: pd.DataFrame, settings: Settings) -> None:
    daily = df[TARGET].groupby(df.index.normalize()).mean()
    fig, ax = plotting.new_figure(9, 3.6)
    ax.plot(daily.index, daily, color=plotting.NEUTRAL, linewidth=0.6, label="Daily mean")
    ax.plot(
        daily.index,
        daily.rolling(30, center=True, min_periods=20).mean(),
        color=plotting.SERIES[0],
        label="30-day average",
    )
    plotting.thousands_gw(ax.yaxis)
    ax.set_ylabel("Consumption (GW)")
    ax.set_title("French national electricity consumption")
    fig.legend(loc="outside lower center", ncols=2)
    plotting.save(fig, settings.paths.figures / "eda_daily_consumption.png")


def plot_hourly_profiles(df: pd.DataFrame, settings: Settings) -> None:
    season = np.where(df.index.month.isin([12, 1, 2]), "Winter", "")
    season = np.where(df.index.month.isin([6, 7, 8]), "Summer", season)
    day = np.where(df.index.dayofweek >= 5, "weekend", "weekday")
    groups = pd.Series(season, index=df.index) + " " + pd.Series(day, index=df.index)
    profiles = pd.DataFrame(
        {"group": groups, "hour": df.index.hour, "value": df[TARGET]}
    ).pivot_table(index="hour", columns="group", values="value", aggfunc="mean")
    fig, ax = plotting.new_figure(8, 4)
    order = ["Winter weekday", "Winter weekend", "Summer weekday", "Summer weekend"]
    for color, name in zip(plotting.SERIES, order, strict=True):
        if name in profiles:
            ax.plot(profiles.index, profiles[name], color=color, label=name)
    plotting.thousands_gw(ax.yaxis)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Mean consumption (GW)")
    ax.set_title("Mean daily profile by season and day type")
    fig.legend(loc="outside lower center", ncols=4)
    plotting.save(fig, settings.paths.figures / "eda_hourly_profiles.png")


def plot_temperature_effect(df: pd.DataFrame, settings: Settings) -> None:
    daily = daily_table(df)
    weekdays = daily[daily["day_type"] == "Weekday"]
    fig, ax = plotting.new_figure(6.5, 4.2)
    ax.scatter(
        weekdays["temperature_c"],
        weekdays[TARGET],
        s=9,
        color=plotting.SERIES[0],
        alpha=0.35,
        linewidths=0,
    )
    ax.axvline(HEATING_THRESHOLD_C, color=plotting.INK_SECONDARY, linewidth=0.8)
    ax.annotate(
        f"Heating threshold ({HEATING_THRESHOLD_C:.0f} °C)",
        xy=(HEATING_THRESHOLD_C, ax.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        color=plotting.INK_SECONDARY,
        fontsize=9,
    )
    plotting.thousands_gw(ax.yaxis)
    ax.set_xlabel("Daily mean temperature, population-weighted (°C)")
    ax.set_ylabel("Daily mean consumption (GW)")
    ax.set_title("Daily consumption versus temperature (working days)")
    plotting.save(fig, settings.paths.figures / "eda_temperature_effect.png")


def plot_calendar_effects(df: pd.DataFrame, settings: Settings) -> None:
    effects = calendar_effects(daily_table(df)).iloc[::-1]
    fig, ax = plotting.new_figure(7, 3.4)
    bars = ax.barh(effects.index, effects.to_numpy(), color=plotting.SERIES[0], height=0.6)
    labels = [f"{0.0 if abs(v) < 0.05 else v:+.1f} %" for v in effects.to_numpy()]
    ax.bar_label(bars, labels=labels, padding=4, color=plotting.INK_SECONDARY, fontsize=9)
    span = max(abs(effects.min()), abs(effects.max()), 1.0)
    ax.set_xlim(min(effects.min(), 0) - 0.3 * span, max(effects.max(), 0) + 0.3 * span)
    ax.axvline(0, color=plotting.INK_SECONDARY, linewidth=0.8)
    ax.set_xlabel("Deviation from an ordinary weekday of the same month (%)")
    ax.set_title("Calendar effects: weekends, public holidays and bridge days")
    ax.grid(axis="y", visible=False)
    plotting.save(fig, settings.paths.figures / "eda_calendar_effects.png")


def run_eda(settings: Settings) -> dict:
    settings.paths.ensure()
    df = load_clean(settings)
    plot_daily_consumption(df, settings)
    plot_hourly_profiles(df, settings)
    plot_temperature_effect(df, settings)
    plot_calendar_effects(df, settings)
    summary = summarize(df)
    (settings.paths.results / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
