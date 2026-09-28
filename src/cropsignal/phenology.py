"""
Land-surface phenology extraction from a vegetation-index time series.

The functions here answer one question per site: *when does this place's
growing season actually happen?* — as opposed to assuming it happens on
fixed calendar dates. Everything downstream (warping, anomaly scoring)
is built on the anchors produced here.

Method follows the standard amplitude-threshold approach used in land
surface phenology work (White et al. 1997; Jonsson & Eklundh 2004,
TIMESAT): smooth the annual curve, then take Start/Peak/End of Season
from fixed fractions of the season's own amplitude.

KNOWN LIMITATION - ONE SEASON PER YEAR
--------------------------------------
`phenometrics` extracts a single Start/Peak/End triple per phenological
year. That is wrong wherever a year holds more than one cropping cycle,
and measurement on this project's own data shows that is the common case,
not the exception: all twelve Kenyan sites are bimodal, peaking around May
(long rains) and November-December (short rains), and the Afghan sites show
a second August peak alongside the April one.

On such a site the extractor picks whichever cycle happens to be larger
that year, so the anchors can jump between cycles between years, and the
warp then aligns the long rains of one year against the short rains of
another. The measured consequence is in the repository README: phase
alignment scores *worse* than the calendar baseline in Kenya.

Handling this properly means detecting the number of cycles per year and
warping each separately. Until that exists, treat the phase-aligned
detector as applicable to single-cycle systems only, and prefer the
calendar baseline elsewhere.
"""
from __future__ import annotations

import numpy as np

DAYS_PER_YEAR = 365.25


def _circular_bin_mean(doy: np.ndarray, values: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean value per day-of-year bin, NaN where a bin has no observations."""
    edges = np.linspace(0, DAYS_PER_YEAR, n_bins + 1)
    idx = np.clip(np.digitize(doy, edges) - 1, 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = values[idx == b]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[b] = sel.mean()
    return out


def _circular_smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving average that wraps around the year, NaN-tolerant."""
    n = len(y)
    half = window // 2
    out = np.full(n, np.nan)
    for i in range(n):
        idx = [(i + k) % n for k in range(-half, half + 1)]
        sel = y[idx]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            out[i] = sel.mean()
    return out


def detect_season_start(doy: np.ndarray, values: np.ndarray, n_bins: int = 24) -> float:
    """
    Day-of-year at which this site's *phenological year* should begin.

    Taken as the trough of the site's multi-year climatological curve: the
    time of year when vegetation is reliably at its lowest, i.e. between
    seasons. Cutting the year there keeps each growing season intact inside
    one phenological year instead of splitting it across a January boundary.

    This is what makes the method hemisphere-agnostic without configuration:
    a Dutch site lands on a winter trough (~Dec-Jan), a New Zealand site on
    its own winter trough (~Jun-Jul), and neither needs to be told which
    hemisphere it is in.

    Returns the day-of-year (float, 1..365) of the climatological minimum.
    """
    doy = np.asarray(doy, dtype=float)
    values = np.asarray(values, dtype=float)
    clim = _circular_smooth(_circular_bin_mean(doy, values, n_bins))
    if not np.any(np.isfinite(clim)):
        return 1.0
    b = int(np.nanargmin(clim))
    bin_width = DAYS_PER_YEAR / n_bins
    return float(b * bin_width + bin_width / 2.0)


def to_phenological_year(doy: np.ndarray, year: np.ndarray, season_start_doy: float):
    """
    Re-index calendar observations onto phenological years.

    Returns (pheno_year, days_since_start) where `pheno_year` labels the
    season an observation belongs to and `days_since_start` is its position
    within that season in days (0 .. ~365).
    """
    doy = np.asarray(doy, dtype=float)
    year = np.asarray(year, dtype=float)
    before = doy < season_start_doy
    pheno_year = np.where(before, year - 1, year)
    days = np.where(before, doy + DAYS_PER_YEAR - season_start_doy, doy - season_start_doy)
    return pheno_year.astype(int), days


def phenometrics(days: np.ndarray, values: np.ndarray, amp_frac: float = 0.2,
                 min_points: int = 8, min_amplitude: float = 0.05) -> dict | None:
    """
    Extract Start / Peak / End of Season from one season's curve.

    `days` is position within the phenological year, `values` the vegetation
    index. SOS and EOS are the crossings of `base + amp_frac * amplitude` on
    the rising and falling limbs either side of the peak.

    Returns None when the season is too sparse or too flat for the anchors
    to mean anything (bare soil, permanent water, evergreen cover) — callers
    must fall back to the calendar for those sites rather than warp on noise.
    """
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(days) & np.isfinite(values)
    days, values = days[ok], values[ok]
    if days.size < min_points:
        return None

    order = np.argsort(days)
    days, values = days[order], values[order]

    # Light smoothing before taking the peak: a single cloud-contaminated
    # composite should not be allowed to define Peak of Season.
    smoothed = _moving_average(values, window=3)

    peak_i = int(np.argmax(smoothed))
    peak_value = float(smoothed[peak_i])
    base_value = float(np.min(smoothed))
    amplitude = peak_value - base_value
    if amplitude < min_amplitude:
        return None

    threshold = base_value + amp_frac * amplitude
    sos = _crossing(days[: peak_i + 1], smoothed[: peak_i + 1], threshold, rising=True)
    eos = _crossing(days[peak_i:], smoothed[peak_i:], threshold, rising=False)
    if sos is None or eos is None or not (sos < days[peak_i] < eos):
        return None

    return {
        "sos": float(sos),
        "pos": float(days[peak_i]),
        "eos": float(eos),
        "peak_value": peak_value,
        "base_value": base_value,
        "amplitude": float(amplitude),
        "n_points": int(days.size),
    }


def _moving_average(y: np.ndarray, window: int = 3) -> np.ndarray:
    if len(y) < window:
        return y.copy()
    pad = window // 2
    padded = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _crossing(days: np.ndarray, values: np.ndarray, threshold: float, rising: bool):
    """Linearly interpolated day at which `values` crosses `threshold`."""
    if days.size < 2:
        return None
    if rising:
        idx = np.where(values >= threshold)[0]
        if idx.size == 0 or idx[0] == 0:
            return float(days[0]) if idx.size else None
        i = idx[0]
    else:
        idx = np.where(values <= threshold)[0]
        if idx.size == 0:
            return float(days[-1])
        if idx[0] == 0:
            return float(days[0])
        i = idx[0]
    y0, y1 = values[i - 1], values[i]
    if y1 == y0:
        return float(days[i])
    frac = (threshold - y0) / (y1 - y0)
    return float(days[i - 1] + frac * (days[i] - days[i - 1]))
