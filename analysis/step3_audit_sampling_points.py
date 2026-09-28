"""
STEP 3 - Audit whether each sampling point is actually looking at farmland.

Run:  python analysis/step3_audit_sampling_points.py

This is the step that should run before any model is trained, and the one
that is usually skipped.

A satellite pipeline will happily return a clean, complete, well-formatted
time series for a point that sits on a town centre, a river, or a mountain
ridge. Nothing downstream errors. The model trains, the metrics print, the
dashboard renders - and the whole thing describes rooftops.

Two cheap checks catch it, using only the vegetation index itself:

  peak NDVI          - what this point reaches at its greenest. Cropland at
                       canopy closure reaches 0.7-0.9. A point that never
                       exceeds ~0.3 is not growing a crop: it is built-up,
                       bare rock, desert, or water.

  seasonal amplitude - the range of the point's monthly NDVI climatology.
                       An annual crop *must* swing: bare soil before sowing,
                       closed canopy at peak, senescence at harvest. A point
                       that sits flat all year is either not cropland, or is
                       so mixed with non-crop cover that the crop signal is
                       swamped.

A point failing either check cannot support a crop-stress label, because
there is no crop signal in it to be anomalous.

Note on permanent pasture: grazed pasture is genuinely greener year-round
than arable land, so it legitimately shows lower amplitude at high NDVI.
The audit separates that case (high peak, low amplitude) from the fatal one
(low peak), rather than failing pasture by mistake.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LEGACY_CSV = (Path(__file__).resolve().parents[2]
              / "crop-stress-prediction" / "data" / "raw" / "sentinel2_timeseries.csv")
DEFAULT_CSV = LEGACY_CSV

def _resolve_csv(argv):
    """--csv PATH, else this repo's extract, else the legacy one."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="time series CSV to audit")
    args, _ = ap.parse_known_args(argv)
    if args.csv:
        return Path(args.csv)
    own = Path(__file__).resolve().parents[1] / "data" / "raw" / "timeseries.csv"
    if own.exists():
        return own
    return DEFAULT_CSV


PEAK_MIN = 0.55        # below this a point is not reaching crop canopy closure
AMPLITUDE_MIN = 0.15   # below this there is no growing-season cycle to speak of
WATER_PEAK_MAX = 0.15  # peak this low means water or fully built-up


def classify(peak: float, amp: float) -> str:
    if peak < WATER_PEAK_MAX:
        return "FAIL - water/built-up"
    if peak < PEAK_MIN and amp < AMPLITUDE_MIN:
        return "FAIL - no crop signal"
    if peak < PEAK_MIN:
        return "WEAK - low canopy peak"
    if amp < AMPLITUDE_MIN:
        return "OK?  - pasture-like (green year-round)"
    return "OK   - cropland-like"


def main(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["period_start"])
    df["month"] = df["period_start"].dt.month

    rows = []
    for (country, site), g in df.groupby(["country", "name"]):
        clim = g.groupby("month")["NDVI"].mean()
        peak = float(g["NDVI"].quantile(0.95))
        amp = float(clim.max() - clim.min())
        rows.append({
            "country": country, "site": site,
            "peak_ndvi": round(peak, 2),
            "seasonal_amp": round(amp, 3),
            "verdict": classify(peak, amp),
        })

    audit = pd.DataFrame(rows).sort_values(["country", "peak_ndvi"])

    print("=" * 78)
    print("SAMPLING POINT AUDIT")
    print("=" * 78)
    print(audit.to_string(index=False))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    counts = audit["verdict"].value_counts()
    for verdict, n in counts.items():
        print(f"  {n:3d} / {len(audit)}   {verdict}")

    failed = audit[audit["verdict"].str.startswith("FAIL")]
    print()
    if len(failed):
        print(f"{len(failed)} of {len(audit)} points carry no usable crop signal.")
        print("Any label, model, or metric computed over these points is describing")
        print("something other than a crop. They must be relocated or dropped before")
        print("the numbers downstream mean anything.")
        print()
        print("Failing points:")
        for _, r in failed.iterrows():
            print(f"  {r['site']:15s} ({r['country']:12s})  "
                  f"peak {r['peak_ndvi']:.2f}  amp {r['seasonal_amp']:.3f}")
    else:
        print("All points show a plausible crop signal.")

    print()
    print("Per-country usable share:")
    audit["usable"] = ~audit["verdict"].str.startswith("FAIL")
    print(audit.groupby("country")["usable"]
          .agg(usable="sum", total="size")
          .assign(share=lambda d: (d["usable"] / d["total"] * 100).round(0).astype(int).astype(str) + "%")
          .to_string())

    return audit


if __name__ == "__main__":
    import sys
    main(_resolve_csv(sys.argv[1:]))
