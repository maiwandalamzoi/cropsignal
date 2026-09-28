"""
Established agricultural drought indices, computed at parcel scale.

These are not invented scores. Each is a published, operationally used
index with citable thresholds, which matters because the output is meant to
sit inside a national early-warning bulletin rather than a demo:

  VCI  Vegetation Condition Index          Kogan 1990
       100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)
       Where this period's greenness sits within the site's own historical
       range for this time of year. 0 = worst on record, 100 = best.

  TCI  Temperature Condition Index         Kogan 1995
       100 * (LST_max - LST) / (LST_max - LST_min)
       The same idea inverted for temperature: high LST means stress, so
       the index is flipped to keep "low value = bad" consistent with VCI.

  VHI  Vegetation Health Index             Kogan 1995
       alpha * VCI + (1 - alpha) * TCI,  alpha = 0.5
       Combines the two. Used operationally by NOAA/STAR and by FAO's
       Agricultural Stress Index System (ASIS).
       VHI <= 40 -> vegetation stress;  <= 26 -> severe stress.

  ESI  Evaporative Stress Index            Anderson et al. 2007
       ET / PET, standardised the same way.
       Falls when a canopy closes its stomata under water stress, which
       happens *before* greenness drops - the earliest of these signals.

Two design points matter more than the formulas.

*Scale.* VHI is normally produced from 1 km AVHRR or MODIS. A 1 km pixel
over smallholder farmland is a mixture of fields, tracks, homesteads and
bush, so the index describes the landscape rather than any farm in it.
Here VCI comes from 10 m Sentinel-2 restricted to a verified cropland mask,
and only TCI is left at MODIS thermal resolution. The result is a
crop-specific index, which is what makes it usable for targeting - the
difference between "this district is in drought" and "these parcels are".

*Baseline length.* min/max over a short record is unstable: one exceptional
year sets the extreme and every later year is scored against it. The record
length actually used is reported alongside every index, and
`MIN_BASELINE_YEARS` refuses to compute at all below a floor, rather than
returning a confident-looking number from three years of data.

READ THE THRESHOLDS WITH THE BASELINE IN MIND
---------------------------------------------
The NOAA/STAR cut-offs (40 stress, 26 severe) were established against
AVHRR records spanning thirty years and more. Over such a record, VHI
below 40 genuinely is unusual.

Over a six-year record it is not. Min-max scaling guarantees that each
site's worst observation scores 0 and its best 100, so every site displays
its own worst season as an extreme drought *by construction*, whether or
not that season was remarkable in a longer context. On this project's
Sentinel-2 data that produces around half of all observations below the
"stress" threshold - a figure which says more about the six-year window
than about the crops.

So on a short baseline these indices rank seasons *within the sample*.
They do not classify drought in absolute terms, and the NOAA thresholds
should not be quoted as if they did. A published bulletin needs the long
climatology - typically the MODIS or AVHRR record - even when the
operational product is produced at Sentinel-2 resolution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VHI_ALPHA = 0.5

#: NOAA/STAR operational thresholds.
STRESS_THRESHOLD = 40
SEVERE_THRESHOLD = 26
EXTREME_THRESHOLD = 10

#: Below this many years of record the min/max range is too unstable to use.
MIN_BASELINE_YEARS = 5


def _condition_index(values: pd.Series, groups: list, invert: bool = False,
                     min_years: int | None = None) -> pd.Series:
    """
    Scale a variable to 0-100 against its own historical range within each group.

    `invert=True` flips the scale, for variables where high means stressed
    (temperature). Groups that do not meet the record-length floor return
    NaN rather than a number computed from too little history.
    """
    df = pd.DataFrame({"v": values.values, "_g": list(zip(*[g.values for g in groups]))})
    vmin = df.groupby("_g")["v"].transform("min")
    vmax = df.groupby("_g")["v"].transform("max")
    rng = (vmax - vmin).replace(0, np.nan)

    if invert:
        idx = 100 * (vmax - df["v"]) / rng
    else:
        idx = 100 * (df["v"] - vmin) / rng

    if min_years is not None:
        n = df.groupby("_g")["v"].transform("count")
        idx = idx.where(n >= min_years)

    return pd.Series(idx.values, index=values.index).clip(0, 100)


def vegetation_condition_index(df: pd.DataFrame, ndvi_col: str = "NDVI",
                               group_cols=("site", "period_of_year"),
                               min_years: int = MIN_BASELINE_YEARS) -> pd.Series:
    """VCI (Kogan 1990) - NDVI scaled against the site's own historical range."""
    groups = [df[c] for c in group_cols]
    return _condition_index(df[ndvi_col], groups, invert=False, min_years=min_years)


def temperature_condition_index(df: pd.DataFrame, lst_col: str = "LST_day",
                                group_cols=("site", "period_of_year"),
                                min_years: int = MIN_BASELINE_YEARS) -> pd.Series:
    """TCI (Kogan 1995) - LST scaled and inverted, so low = hot = stressed."""
    groups = [df[c] for c in group_cols]
    return _condition_index(df[lst_col], groups, invert=True, min_years=min_years)


def vegetation_health_index(vci: pd.Series, tci: pd.Series,
                            alpha: float = VHI_ALPHA) -> pd.Series:
    """VHI (Kogan 1995) - the weighted combination of VCI and TCI."""
    return (alpha * vci + (1 - alpha) * tci).clip(0, 100)


def evaporative_stress_index(df: pd.DataFrame, esi_col: str = "ESI",
                             group_cols=("site", "period_of_year"),
                             min_years: int = MIN_BASELINE_YEARS) -> pd.Series:
    """ESI scaled to 0-100 against its own history, for comparability with VHI."""
    groups = [df[c] for c in group_cols]
    return _condition_index(df[esi_col], groups, invert=False, min_years=min_years)


def classify(vhi: pd.Series) -> pd.Series:
    """NOAA/STAR drought severity classes from VHI."""
    return pd.cut(
        vhi,
        bins=[-0.01, EXTREME_THRESHOLD, SEVERE_THRESHOLD, STRESS_THRESHOLD, 100.01],
        labels=["extreme", "severe", "moderate", "none"],
    )


def add_drought_indices(df: pd.DataFrame, ndvi_col: str = "NDVI",
                        lst_col: str = "LST_day", esi_col: str = "ESI",
                        group_cols=("site", "period_of_year"),
                        min_years: int = MIN_BASELINE_YEARS) -> pd.DataFrame:
    """
    Add VCI, TCI, VHI, ESI_index and a severity class to a time series.

    The frame needs one row per (site, period) with the vegetation,
    thermal and water variables already extracted, plus a `period_of_year`
    column so that each observation is compared against the same part of
    the season in other years.
    """
    out = df.copy()
    out["VCI"] = vegetation_condition_index(out, ndvi_col, group_cols, min_years)

    if lst_col in out.columns and out[lst_col].notna().any():
        out["TCI"] = temperature_condition_index(out, lst_col, group_cols, min_years)
        out["VHI"] = vegetation_health_index(out["VCI"], out["TCI"])
    else:
        # Without thermal data VHI is undefined. Do not silently substitute
        # VCI for it - they are different indices with different thresholds.
        out["TCI"] = np.nan
        out["VHI"] = np.nan

    if esi_col in out.columns and out[esi_col].notna().any():
        out["ESI_index"] = evaporative_stress_index(out, esi_col, group_cols, min_years)
    else:
        out["ESI_index"] = np.nan

    out["drought_class"] = classify(out["VHI"])
    out["baseline_years"] = out.groupby(list(group_cols))[ndvi_col].transform("count")
    return out


def season_summary(df: pd.DataFrame, by=("site", "country", "season_year")) -> pd.DataFrame:
    """
    Per-site, per-season drought summary of the kind an early-warning
    bulletin carries: how low the index went, and for how long.

    Duration matters as much as depth. A single 8-day composite at VHI 35
    is noise; six consecutive ones is a failed season.
    """
    by = list(by)
    g = df.groupby(by)
    out = g.agg(
        vhi_min=("VHI", "min"),
        vhi_mean=("VHI", "mean"),
        vci_mean=("VCI", "mean"),
        periods=("VHI", "count"),
    ).reset_index()

    stressed = df[df["VHI"] <= STRESS_THRESHOLD].groupby(by).size().rename("periods_stressed")
    severe = df[df["VHI"] <= SEVERE_THRESHOLD].groupby(by).size().rename("periods_severe")
    out = out.merge(stressed, on=by, how="left").merge(severe, on=by, how="left")
    out[["periods_stressed", "periods_severe"]] = (
        out[["periods_stressed", "periods_severe"]].fillna(0).astype(int)
    )
    out["stressed_share"] = (out["periods_stressed"] / out["periods"]).round(3)
    return out
