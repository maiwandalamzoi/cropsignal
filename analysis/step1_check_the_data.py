"""
STEP 1 - Check what data we actually have, before believing anything built on it.

Run:  python analysis/step1_check_the_data.py

This step answers four questions, in order of how badly a bad answer would
invalidate everything downstream:

  1. Coverage   - how many sites, years, observations? Enough to build a
                  per-site multi-year baseline at all?
  2. Gaps       - are observations regular, or does cloud cover leave holes
                  big enough to hide a stress event?
  3. Range      - are the index values physically plausible (NDVI in -1..1,
                  cropland peaking well above bare soil)?
  4. Balance    - is any country/region over-represented enough that a
                  global metric would mostly describe that one place?

What would make the data unusable: fewer than ~4 years per site (no
leave-one-year-out baseline possible), or median gaps much larger than the
compositing period (the series is mostly interpolation, not observation).
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



def main(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"Satellite CSV not found at {csv_path}\n"
                         f"Point this script at your own extract with --csv.")

    df = pd.read_csv(csv_path, parse_dates=["period_start"])
    df["year"] = df["period_start"].dt.year
    df["doy"] = df["period_start"].dt.dayofyear

    print("=" * 70)
    print("1. COVERAGE")
    print("=" * 70)
    print(f"rows                {len(df):,}")
    print(f"sites               {df['name'].nunique()}")
    print(f"countries           {df['country'].nunique()}  ({', '.join(sorted(df['country'].unique()))})")
    print(f"date range          {df['period_start'].min().date()} -> {df['period_start'].max().date()}")
    per_site_years = df.groupby("name")["year"].nunique()
    print(f"years per site      min {per_site_years.min()}, median {per_site_years.median():.0f}, max {per_site_years.max()}")
    thin = per_site_years[per_site_years < 4]
    print(f"sites with <4 years {len(thin)}"
          + (f"  -> {list(thin.index)}" if len(thin) else "  (good: all sites can support a leave-one-year-out baseline)"))

    print()
    print("=" * 70)
    print("2. GAPS BETWEEN OBSERVATIONS (cloud cover shows up here)")
    print("=" * 70)
    gaps = df.sort_values(["name", "period_start"]).groupby("name")["period_start"].diff().dt.days
    print(f"median gap          {gaps.median():.0f} days   (compositing period is 16 days)")
    print(f"90th percentile     {gaps.quantile(0.9):.0f} days")
    print(f"gaps > 32 days      {(gaps > 32).sum():,}  ({(gaps > 32).mean() * 100:.1f}% of intervals)")
    print("\nWorst-covered sites (mean gap, days):")
    print(gaps.groupby(df["name"]).mean().sort_values(ascending=False).head(5).round(1).to_string())

    print()
    print("=" * 70)
    print("3. VALUE RANGES (are these physically plausible?)")
    print("=" * 70)
    for col in ["NDVI", "EVI", "SAVI", "NDMI"]:
        if col in df.columns:
            s = df[col]
            flag = "  <-- OUT OF RANGE" if (s.min() < -1.01 or s.max() > 1.01) else ""
            print(f"{col:6s} min {s.min():7.3f}  median {s.median():7.3f}  max {s.max():7.3f}  "
                  f"missing {s.isna().mean() * 100:4.1f}%{flag}")
    if "pixel_count" in df.columns:
        print(f"\npixel_count: median {df['pixel_count'].median():.0f}, "
              f"{(df['pixel_count'] < 10).mean() * 100:.1f}% of rows built from <10 pixels "
              f"(those are the noisy ones)")

    print()
    print("=" * 70)
    print("4. BALANCE ACROSS COUNTRIES")
    print("=" * 70)
    bal = df.groupby("country").agg(rows=("NDVI", "size"), sites=("name", "nunique"),
                                    ndvi_median=("NDVI", "median"),
                                    ndvi_p95=("NDVI", "quantile"))
    bal["share_%"] = (bal["rows"] / len(df) * 100).round(1)
    print(bal[["rows", "sites", "share_%", "ndvi_median"]].to_string())

    print()
    print("READ THIS BEFORE MOVING ON:")
    print("  - Every metric later is an average over these rows. If one country")
    print("    dominates the row count, always read per-country numbers too.")
    print("  - Rows built from few pixels are noisy; a single such composite can")
    print("    fake a 'stress' dip. That is why the method smooths before taking a peak.")
    return df


if __name__ == "__main__":
    import sys
    main(_resolve_csv(sys.argv[1:]))
