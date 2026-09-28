"""
STEP 2 - Look at single sites with your own eyes before trusting any metric.

Run:  python analysis/step2_look_at_one_site.py
      python analysis/step2_look_at_one_site.py Lelystad
      python analysis/step2_look_at_one_site.py --csv data/legacy/sentinel2_timeseries.csv

Aggregate statistics hide the thing this whole project is about. Print one
site's NDVI as a text sparkline, one row per year, and the question answers
itself: do the seasons line up vertically, or do they slide?

  - A site whose seasons line up vertically is one where comparing
    "week 14 this year vs week 14 last year" is a fair comparison.
  - A site whose peak slides left and right between years is one where that
    comparison comes apart - and where a calendar-based anomaly detector
    will report the slide as damage.

Read the printed peak-position row at the bottom of each site. That spread,
in days, is the entire motivation for phase alignment.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cropsignal import detect_season_start  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CSV = _REPO_ROOT / "data" / "legacy" / "sentinel2_timeseries.csv"
_OWN = _REPO_ROOT / "data" / "raw" / "timeseries.csv"
DEFAULT_CSV = _OWN if _OWN.exists() else LEGACY_CSV


def _parse_args(argv):
    """--csv PATH, else default resolution; anything else is a site name."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="time series CSV to read (default: this repo's "
                                  "own extract, or the bundled legacy dataset)")
    ap.add_argument("sites", nargs="*", help="site name(s) to show")
    args = ap.parse_args(argv)
    return Path(args.csv) if args.csv else DEFAULT_CSV, (args.sites or None)

BLOCKS = " .:-=+*#%@"


def sparkline(values, vmin, vmax):
    out = []
    for v in values:
        if not np.isfinite(v):
            out.append(" ")
            continue
        frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        out.append(BLOCKS[int(np.clip(frac, 0, 0.999) * len(BLOCKS))])
    return "".join(out)


def show_site(df, site):
    g = df[df["name"] == site].sort_values("period_start")
    if g.empty:
        print(f"  (no rows for {site})")
        return
    country = g["country"].iloc[0]
    vmin, vmax = g["NDVI"].min(), g["NDVI"].max()
    start_doy = detect_season_start(g["doy"].values, g["NDVI"].values)

    print(f"\n{'=' * 78}")
    print(f"{site}  ({country})   NDVI {vmin:.2f}..{vmax:.2f}   "
          f"detected season start: day {start_doy:.0f}")
    print(f"{'=' * 78}")
    print("        J   F   M   A   M   J   J   A   S   O   N   D")

    peaks = {}
    for year, gy in g.groupby("year"):
        # place each observation in its 16-day slot so rows align vertically
        row = np.full(23, np.nan)
        for _, r in gy.iterrows():
            row[min(int((r["doy"] - 1) // 16), 22)] = r["NDVI"]
        print(f"{year}   {sparkline(row, vmin, vmax)}")
        if np.isfinite(row).sum() >= 10:
            peaks[year] = int(np.nanargmax(row) * 16 + 8)

    if len(peaks) >= 2:
        vals = list(peaks.values())
        print(f"\n  peak NDVI day-of-year by year: "
              + ", ".join(f"{y}:{d}" for y, d in peaks.items()))
        print(f"  spread: {max(vals) - min(vals)} days between earliest and latest peak")
        print(f"  {'-> seasons roughly calendar-locked' if max(vals) - min(vals) < 60 else '-> seasons move a lot; calendar comparison is unfair here'}")


def main(sites=None, csv_path: Path = DEFAULT_CSV):
    df = pd.read_csv(csv_path, parse_dates=["period_start"])
    df["year"] = df["period_start"].dt.year
    df["doy"] = df["period_start"].dt.dayofyear

    if not sites:
        # one representative site per country
        sites = [df[df["country"] == c]["name"].iloc[0] for c in sorted(df["country"].unique())]

    print(f"Reading {csv_path}")
    print("Darker/denser characters = higher NDVI. Each row is one year.")
    print("If the dense band sits in the same columns every year, the season is")
    print("calendar-locked. If it slides, it is not.")
    for s in sites:
        show_site(df, s)

    print(f"\n{'=' * 78}")
    print("Try other sites:  python analysis/step2_look_at_one_site.py <SiteName>")
    print("All sites:", ", ".join(sorted(df["name"].unique())))


if __name__ == "__main__":
    _csv, _sites = _parse_args(sys.argv[1:])
    main(_sites, _csv)
