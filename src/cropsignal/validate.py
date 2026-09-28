"""
Validation that does not use the satellite signal the labels were built from.

There is no public ground-truth "this field was stressed on this date"
dataset at scale, so an anomaly label extracted from NDVI cannot be scored
against field records. It can, however, be tested against evidence that is
*independent of the vegetation index itself*:

- `weather_association` - does the flag co-occur with drought/heat that was
  measured separately (reanalysis weather)? A real stress flag should. A
  flag that is mostly mis-timed phenology should not.

- `phenology_contamination` - does the flag fire more often in seasons whose
  peak simply arrived early or late? A flag that is really detecting
  "the season shifted" rather than "the crop suffered" will.

Together these two are the discriminating test: the useful detector is the
one with the higher weather association *and* the lower phenology
contamination. Either number alone can be gamed by flagging more or fewer
observations.

- `yield_association` - the weakest rung, kept honest: national statistics
  give one number per country-year, so with 5-6 usable years this is a
  directional sanity check and nothing more. It is reported with its n.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), NaN when either class is empty."""
    ok = np.isfinite(scores) & np.isfinite(labels)
    s, y = scores[ok], labels[ok]
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = pd.Series(s).rank().values
    r_pos = ranks[y == 1].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def weather_association(df: pd.DataFrame, label_col: str,
                        stress_index_col: str = "weather_stress_index") -> dict:
    """
    How well an independently measured weather-stress index separates
    flagged from unflagged observations.

    Returns AUC (0.5 = the flag carries no weather information) plus the
    mean index in each class, so an AUC can be read alongside its effect size.
    """
    y = df[label_col].values.astype(float)
    s = df[stress_index_col].values.astype(float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    return {
        "auc": _auc(s, y),
        "mean_index_flagged": float(s[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_index_unflagged": float(s[y == 0].mean()) if (y == 0).any() else float("nan"),
        "n": int(y.size),
        "flag_rate": float(y.mean()) if y.size else float("nan"),
    }


def phenology_contamination(df: pd.DataFrame, label_col: str,
                            shift_col: str = "pos_shift_days") -> dict:
    """
    How much of the flag is explained by the season having simply moved.

    `shift_col` is the season's peak offset from that site's own median peak
    (days, signed). We test the *absolute* shift: both an early and a late
    season mis-align a calendar comparison.

    `auc_abs_shift` near 0.5 means the flag is indifferent to how far the
    season moved - what we want. Well above 0.5 means the detector is partly
    reporting phenological timing as if it were crop stress.
    """
    y = df[label_col].values.astype(float)
    shift = np.abs(df[shift_col].values.astype(float))
    ok = np.isfinite(y) & np.isfinite(shift)
    y, shift = y[ok], shift[ok]
    return {
        "auc_abs_shift": _auc(shift, y),
        "mean_abs_shift_flagged": float(shift[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_abs_shift_unflagged": float(shift[y == 0].mean()) if (y == 0).any() else float("nan"),
        "n": int(y.size),
    }


def yield_association(season_flags: pd.DataFrame, yields: pd.DataFrame,
                      country_col: str = "country", year_col: str = "year",
                      rate_col: str = "flag_rate",
                      yield_col: str = "yield_kg_ha") -> pd.DataFrame:
    """
    Correlate a country-year stress rate with that country-year's reported
    yield anomaly (deviation from the country's own mean over the window).

    National statistics aggregate thousands of fields, most of them nowhere
    near our sample points, so a weak correlation here is not evidence
    against the detector. The column `n_years` is reported because with
    n < 10 the correlation is descriptive, not inferential.
    """
    merged = season_flags.merge(yields, on=[country_col, year_col], how="inner")
    rows = []
    for country, g in merged.groupby(country_col):
        g = g.dropna(subset=[rate_col, yield_col])
        if len(g) < 3:
            rows.append({country_col: country, "n_years": len(g), "pearson_r": float("nan")})
            continue
        anom = g[yield_col] - g[yield_col].mean()
        r = np.corrcoef(g[rate_col].values, anom.values)[0, 1]
        rows.append({country_col: country, "n_years": int(len(g)), "pearson_r": float(r)})
    return pd.DataFrame(rows)


def build_weather_stress_index(weather: pd.DataFrame,
                               precip_anom_col: str = "precip_anom",
                               tmax_anom_col: str = "tmax_anom") -> np.ndarray:
    """
    A simple, transparent drought/heat index from reanalysis weather only:
    hot anomaly plus dry anomaly, each already normalised against that
    location's own climatology.

        index = tmax_anom - precip_anom

    High = hotter and drier than normal for that place at that time of year.
    Deliberately not tuned - a tuned index would start absorbing the very
    signal it is meant to independently corroborate.
    """
    return (weather[tmax_anom_col].values.astype(float)
            - weather[precip_anom_col].values.astype(float))
