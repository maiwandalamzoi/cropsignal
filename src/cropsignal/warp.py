"""
Monotone piecewise-linear time warping from calendar time onto a canonical
phenological phase axis.

The warp is the heart of the method. Calendar time says "this observation
is from mid-April". Phase says "this observation is 40% of the way through
this site's growing season, just past green-up". Two years whose seasons
started three weeks apart are not comparable in calendar time; on the phase
axis they are.

Phase is in [0, 1) over one phenological year, with the four extracted
anchors pinned to fixed positions:

    0.00  season start (climatological trough)
    0.25  SOS  - start of season (green-up)
    0.50  POS  - peak of season
    0.75  EOS  - end of season (senescence)
    1.00  next season start

Between anchors the map is linear, so it is monotone by construction and
invertible.
"""
from __future__ import annotations

import numpy as np

from .phenology import DAYS_PER_YEAR

#: Canonical phase position of each anchor.
ANCHOR_PHASES = (0.0, 0.25, 0.50, 0.75, 1.0)


def anchor_days(metrics: dict, season_length: float = DAYS_PER_YEAR) -> np.ndarray:
    """The four anchor positions in days-since-season-start, plus the year end."""
    return np.array([0.0, metrics["sos"], metrics["pos"], metrics["eos"], season_length])


def to_phase(days: np.ndarray, metrics: dict, season_length: float = DAYS_PER_YEAR) -> np.ndarray:
    """
    Map days-since-season-start onto canonical phase in [0, 1].

    Anchors that are out of order or coincident (possible on a noisy or
    near-flat season) would break monotonicity, so they are nudged apart by
    a day before interpolating rather than silently producing a
    non-invertible warp.
    """
    xp = anchor_days(metrics, season_length)
    xp = _make_strictly_increasing(xp)
    return np.interp(np.asarray(days, dtype=float), xp, np.array(ANCHOR_PHASES))


def from_phase(phase: np.ndarray, metrics: dict, season_length: float = DAYS_PER_YEAR) -> np.ndarray:
    """Inverse warp: canonical phase back to days-since-season-start."""
    xp = _make_strictly_increasing(anchor_days(metrics, season_length))
    return np.interp(np.asarray(phase, dtype=float), np.array(ANCHOR_PHASES), xp)


def _make_strictly_increasing(x: np.ndarray, min_gap: float = 1.0) -> np.ndarray:
    out = np.array(x, dtype=float)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + min_gap
    return out


def reference_curve(phases: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Resample one season's (phase, value) samples onto a shared phase grid.

    Returns NaN outside the observed phase range instead of extrapolating —
    a season that was only observed from green-up onward should contribute
    nothing to the reference before that point.
    """
    ok = np.isfinite(phases) & np.isfinite(values)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    p, v = phases[ok], values[ok]
    order = np.argsort(p)
    p, v = p[order], v[order]
    out = np.interp(grid, p, v)
    out[(grid < p[0]) | (grid > p[-1])] = np.nan
    return out
