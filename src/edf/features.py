"""Feature engineering: calendar + weather features (no target leakage).

All features are computable for the 2022 horizon, since they depend only on
the calendar and on the weather feed (both available for the future), never on
past values of the targets.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .data import REGIONS, WEATHER_VARS

LOCAL_TZ = "Europe/Paris"
HDD_BASE = 16.0  # heating-degree base temperature (°C)
CDD_BASE = 22.0  # cooling-degree base temperature (°C)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical calendar features in French local time, plus holidays/trend."""
    local = index.tz_convert(LOCAL_TZ)
    hour = local.hour + local.minute / 60.0
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()

    years = range(int(local.year.min()), int(local.year.max()) + 1)
    fr = holidays.France(years=years)
    is_holiday = np.fromiter((ts.date() in fr for ts in local), dtype=float)

    df = pd.DataFrame(index=index)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_weekend"] = (dow >= 5).astype(float)
    df["is_holiday"] = is_holiday
    df["year_trend"] = (local.year - int(local.year.min())).to_numpy(dtype=float)
    return df


def weather_features(weather_aligned: pd.DataFrame) -> pd.DataFrame:
    """Derive weather features from the per-region weather frame.

    `weather_aligned` is the output of ``data.align_weather`` (columns
    ``<var>_<region>`` already on the target timestamp grid).
    """
    df = weather_aligned.copy()
    areas = REGIONS + ["France"]

    # Heating / cooling degrees per area (the main drivers of demand).
    for area in areas:
        t = df[f"t_{area}"]
        df[f"hdd_{area}"] = (HDD_BASE - t).clip(lower=0)
        df[f"cdd_{area}"] = (t - CDD_BASE).clip(lower=0)

    # Thermal inertia: rolling means of the national temperature (48 = 24 h).
    #
    # NOTE on the train/val boundary: this rolling window is computed over the
    # full timeline, so the first val-year rows average in the last train-year
    # temperatures. This is NOT target leakage. The rolling means are built
    # purely from *exogenous* weather (an external feed that is fully available
    # for the future 2022 horizon at prediction time), never from past values of
    # the targets. The challenge provides weather for the test period precisely
    # so such weather-derived features can be used; nothing about the targets
    # crosses the boundary.
    t_fr = df["t_France"]
    df["t_France_roll24h"] = t_fr.rolling(48, min_periods=1).mean()
    df["t_France_roll48h"] = t_fr.rolling(96, min_periods=1).mean()
    return df


def build_feature_matrix(
    index: pd.DatetimeIndex, weather_aligned: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate calendar and weather features for the given timestamps."""
    cal = calendar_features(index)
    wx = weather_features(weather_aligned)
    X = pd.concat([cal, wx], axis=1)
    return X.astype("float32")
