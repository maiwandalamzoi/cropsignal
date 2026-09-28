"""
Anomaly detectors for satellite vegetation time series.

Two estimators with the same API, so they can be swapped in a benchmark:

- `CalendarAnomaly`  - the standard approach. Compare an observation to the
  same *calendar* window in other years. This is what operational
  vegetation-condition monitoring largely does, and what the baseline in
  most crop-stress pipelines does.

- `PhenoShift`       - compare an observation to the same *phenological
  phase* in other years, after warping each year onto a common phase axis.

Both score with a leave-one-year-out z-score, so neither can see the year
it is judging. The only difference is the axis the comparison happens on -
which is precisely the hypothesis under test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .phenology import (DAYS_PER_YEAR, detect_season_start, phenometrics,
                        to_phenological_year)
from .warp import reference_curve, to_phase


#: Smallest baseline spread that is physically interpretable, as a fraction
#: of the variable's own observed range.
#:
#: A z-score divides a deviation by the spread of the baseline years, so as
#: that spread approaches zero the score approaches infinity. Left unbounded,
#: a site whose seasons repeat closely is reported as permanently anomalous
#: on differences far smaller than the sensor can resolve: for NDVI over a
#: 0.6 range, 2% is about 0.012, which is the order of residual atmospheric
#: correction, BRDF and geolocation noise in a Sentinel-2 composite. Below
#: that, a difference is not evidence of anything, and the observation is
#: left unscored rather than flagged.
SIGMA_FLOOR_FRAC = 0.02


def _sigma_floor(values: np.ndarray, frac: float = SIGMA_FLOOR_FRAC) -> float:
    """Smallest baseline standard deviation worth dividing by."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return np.inf
    value_range = float(finite.max() - finite.min())
    return max(value_range * frac, 1e-12)


class _BaseAnomaly:
    """Shared plumbing: column handling, fit/transform contract, z -> flag."""

    def __init__(self, site_col: str = "site", time_col: str = "date",
                 value_col: str = "NDVI", z_threshold: float = -1.0,
                 min_baseline_years: int = 3):
        self.site_col = site_col
        self.time_col = time_col
        self.value_col = value_col
        self.z_threshold = z_threshold
        self.min_baseline_years = min_baseline_years
        self.sites_: dict = {}

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().reset_index(drop=True)
        out[self.time_col] = pd.to_datetime(out[self.time_col])
        out["_doy"] = out[self.time_col].dt.dayofyear.astype(float)
        out["_year"] = out[self.time_col].dt.year.astype(int)
        return out

    def _finalize(self, out: pd.DataFrame) -> pd.DataFrame:
        out["anomaly"] = np.where(
            np.isfinite(out["z"]), (out["z"] <= self.z_threshold).astype(float), np.nan
        )
        return out.drop(columns=["_doy", "_year"], errors="ignore")


class CalendarAnomaly(_BaseAnomaly):
    """
    Leave-one-year-out z-score against the same calendar window in other years.

    `period_days` sets the width of the calendar bucket (16 to match a
    Sentinel-2 16-day composite, 8 for MODIS, and so on).
    """

    def __init__(self, period_days: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.period_days = period_days

    def fit(self, df: pd.DataFrame):
        return self  # nothing to learn; the calendar is fixed by definition

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._prepare(df)
        out["bucket"] = ((out["_doy"] - 1) // self.period_days).astype(int)
        out["z"] = _loyo_zscore(
            out, group_cols=[self.site_col, "bucket"], year_col="_year",
            value_col=self.value_col, min_years=self.min_baseline_years,
        )
        # Reported for interface parity with PhenoShift; undefined here.
        out["phase"] = np.nan
        out["pos_shift_days"] = np.nan
        return self._finalize(out.drop(columns=["bucket"]))


class PhenoShift(_BaseAnomaly):
    """
    Phenology-aligned anomaly detection.

    Fitting learns, per site: the climatological season start (the trough
    that defines where one phenological year ends and the next begins), and
    the Start/Peak/End of Season anchors for every individual season.

    Transforming warps each observation onto the canonical phase axis and
    z-scores it against the *other* years resampled at that same phase.

    Sites whose seasons are too flat or too sparse for reliable anchors
    (`min_seasons_with_metrics` not met) fall back to calendar scoring and
    are listed in `fallback_sites_` - the method should not pretend to warp
    a signal that has no season in it.
    """

    def __init__(self, period_days: int = 16, amp_frac: float = 0.2,
                 n_phase_bins: int = 48, min_seasons_with_metrics: int = 3,
                 min_season_span: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.period_days = period_days
        self.amp_frac = amp_frac
        self.n_phase_bins = n_phase_bins
        self.min_seasons_with_metrics = min_seasons_with_metrics
        self.min_season_span = min_season_span
        self.fallback_sites_: list = []

    def fit(self, df: pd.DataFrame):
        data = self._prepare(df)
        self.sites_ = {}
        self.fallback_sites_ = []

        for site, g in data.groupby(self.site_col):
            start_doy = detect_season_start(g["_doy"].values, g[self.value_col].values)
            pheno_year, days = to_phenological_year(
                g["_doy"].values, g["_year"].values, start_doy
            )
            metrics = {}
            for py in np.unique(pheno_year):
                sel = pheno_year == py
                season_days = days[sel]
                # The first and last phenological year of any record are cut
                # off by the record's own boundaries. Warping such a stub
                # onto the canonical axis stretches a fragment of a season
                # across the full phase range, and it then disagrees with
                # every complete year at almost every phase - manufacturing
                # anomalies out of nothing but the start date of the data.
                span = (season_days.max() - season_days.min()) / DAYS_PER_YEAR
                if span < self.min_season_span:
                    continue
                m = phenometrics(season_days, g[self.value_col].values[sel],
                                 amp_frac=self.amp_frac)
                if m is not None:
                    metrics[int(py)] = m

            if len(metrics) < self.min_seasons_with_metrics:
                self.fallback_sites_.append(site)

            self.sites_[site] = {
                "season_start_doy": float(start_doy),
                "metrics": metrics,
                "median_pos": float(np.median([m["pos"] for m in metrics.values()]))
                if metrics else float("nan"),
            }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.sites_:
            raise RuntimeError("PhenoShift.transform called before fit")

        data = self._prepare(df)
        grid = np.linspace(0.0, 1.0, self.n_phase_bins)

        n = len(data)
        z_all = np.full(n, np.nan)
        phase_all = np.full(n, np.nan)
        shift_all = np.full(n, np.nan)
        pyear_all = np.full(n, np.nan)

        for site, g in data.groupby(self.site_col):
            state = self.sites_.get(site)
            if state is None:
                continue
            pos = g.index.values  # positional, since _prepare reset the index
            pheno_year, days = to_phenological_year(
                g["_doy"].values, g["_year"].values, state["season_start_doy"]
            )
            pyear_all[pos] = pheno_year

            metrics = state["metrics"]
            values = g[self.value_col].values.astype(float)

            # A z-score divides by the spread of the baseline years. Where
            # those years agree closely the spread approaches zero, and the
            # division turns a difference of no physical consequence into an
            # enormous score - a site with a very stable history would be
            # reported as permanently anomalous. Requiring the spread to be
            # a non-trivial fraction of the variable's own range leaves such
            # observations unscored instead, which is the honest answer:
            # there is no baseline variation to judge them against.
            sigma_floor = _sigma_floor(values)

            # Warp every season that has anchors onto the phase axis.
            phase = np.full(len(g), np.nan)
            for py, m in metrics.items():
                sel = pheno_year == py
                if sel.any():
                    phase[sel] = to_phase(days[sel], m)
                    shift_all[pos[sel]] = m["pos"] - state["median_pos"]
            phase_all[pos] = phase

            # Leave-one-year-out reference in phase space.
            curves = {
                py: reference_curve(phase[pheno_year == py], values[pheno_year == py], grid)
                for py in metrics
            }
            for py in metrics:
                others = [c for k, c in curves.items() if k != py]
                if len(others) < self.min_baseline_years:
                    continue
                stack = np.vstack(others)
                with np.errstate(invalid="ignore"):
                    counts = np.sum(np.isfinite(stack), axis=0)
                    mu = np.nanmean(stack, axis=0)
                    sigma = np.nanstd(stack, axis=0, ddof=1)
                valid = (counts >= self.min_baseline_years) & (sigma >= sigma_floor)
                mu = np.where(valid, mu, np.nan)
                sigma = np.where(valid, sigma, np.nan)

                sel = pheno_year == py
                p = phase[sel]
                mu_i = np.interp(p, grid, mu, left=np.nan, right=np.nan)
                sd_i = np.interp(p, grid, sigma, left=np.nan, right=np.nan)
                z_all[pos[sel]] = (values[sel] - mu_i) / sd_i

        out = data.copy()
        out["z"] = z_all
        out["phase"] = phase_all
        out["pheno_year"] = pyear_all
        out["pos_shift_days"] = shift_all

        # Sites without usable phenology are scored on the calendar instead,
        # so a benchmark never silently drops them.
        if self.fallback_sites_:
            fb = out[self.site_col].isin(self.fallback_sites_).values
            if fb.any():
                cal = CalendarAnomaly(
                    period_days=self.period_days, site_col=self.site_col,
                    time_col=self.time_col, value_col=self.value_col,
                    z_threshold=self.z_threshold,
                    min_baseline_years=self.min_baseline_years,
                ).fit_transform(data.loc[fb])
                out.loc[fb, "z"] = cal["z"].values

        return self._finalize(out)


def _loyo_zscore(df: pd.DataFrame, group_cols, year_col: str, value_col: str,
                 min_years: int) -> np.ndarray:
    """Leave-one-year-out z-score within each group (shared by both detectors)."""
    z = np.full(len(df), np.nan)
    positions = {ix: i for i, ix in enumerate(df.index)}
    for _, grp in df.groupby(group_cols):
        years = grp[year_col].values
        vals = grp[value_col].values.astype(float)
        for i, y in enumerate(years):
            others = vals[years != y]
            others = others[np.isfinite(others)]
            if others.size < min_years:
                continue
            sigma = others.std(ddof=1)
            if not np.isfinite(sigma) or sigma < _sigma_floor(vals):
                continue
            z[positions[grp.index[i]]] = (vals[i] - others.mean()) / sigma
    return z
