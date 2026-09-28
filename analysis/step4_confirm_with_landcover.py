"""
STEP 4 - Confirm the sampling-point diagnosis against an independent dataset.

Run:  python analysis/step4_confirm_with_landcover.py

Step 3 inferred "this point is not farmland" from the NDVI signal itself.
That reasoning is circular if left there: the same numbers that raised the
suspicion cannot also be the proof.

So ask a dataset that knows nothing about our NDVI series. ESA WorldCover
(10 m, 2021) classifies every pixel on Earth into land cover types. Sample
it inside each point's buffer and read off what is actually on the ground.

If step 3's failing points come back as Built-up / Bare / Water here, the
diagnosis holds on independent evidence.

Requires Earth Engine credentials (see .env.example).

Reads the 36 legacy coordinates straight from the bundled CSV
(data/legacy/sentinel2_timeseries.csv) rather than importing the original
project's locations.py module. An earlier version imported that module
directly from a sibling repo on disk, which meant this script only ran for
someone who happened to have both repos checked out side by side - not a
standalone clone of this one.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ee  # noqa: E402

from cropsignal.gee import init  # noqa: E402

LEGACY_CSV = REPO_ROOT / "data" / "legacy" / "sentinel2_timeseries.csv"
FIELD_BUFFER_M = 60  # matches the buffer the legacy extraction used

# ESA WorldCover v200 class codes
CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse", 70: "Snow/ice", 80: "Water",
    90: "Herb. wetland", 95: "Mangroves", 100: "Moss/lichen",
}
#: Classes that can legitimately carry an agricultural signal.
AGRI = {30, 40, 10}  # grassland (pasture), cropland, tree cover (orchards)


def load_locations(csv_path: Path = LEGACY_CSV) -> list:
    """(name, country, region, lat, lon, climate_zone) for each unique site."""
    cols = ["name", "country", "region", "lat", "lon", "climate_zone"]
    df = pd.read_csv(csv_path, usecols=cols).drop_duplicates("name")
    return list(df[cols].itertuples(index=False, name=None))


def main(buffer_m: int = FIELD_BUFFER_M, csv_path: Path = LEGACY_CSV) -> pd.DataFrame:
    init()
    locations = load_locations(csv_path)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    feats = [
        ee.Feature(ee.Geometry.Point([lon, lat]).buffer(buffer_m), {"name": name})
        for name, _c, _r, lat, lon, _z in locations
    ]
    fc = ee.FeatureCollection(feats)

    reduced = wc.reduceRegions(
        collection=fc, reducer=ee.Reducer.frequencyHistogram(), scale=10
    ).getInfo()["features"]

    meta = {n: (c, z) for n, c, _r, _la, _lo, z in locations}
    rows = []
    for f in reduced:
        name = f["properties"]["name"]
        hist = f["properties"].get("histogram") or {}
        total = sum(hist.values()) or 1
        shares = {int(float(k)): v / total for k, v in hist.items()}
        dominant = max(shares, key=shares.get) if shares else None
        country, zone = meta[name]
        rows.append({
            "site": name,
            "country": country,
            "declared_zone": zone,
            "dominant_cover": CLASSES.get(dominant, "?"),
            "dominant_%": round(shares.get(dominant, 0) * 100),
            "agri_%": round(sum(v for k, v in shares.items() if k in AGRI) * 100),
            "builtup_%": round(shares.get(50, 0) * 100),
            "water_%": round(shares.get(80, 0) * 100),
        })

    df = pd.DataFrame(rows).sort_values(["country", "agri_%"])

    print("=" * 88)
    print(f"ESA WorldCover composition inside each {buffer_m} m sampling buffer")
    print("=" * 88)
    print(df.to_string(index=False))

    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    bad = df[df["agri_%"] < 50]
    print(f"{len(bad)} of {len(df)} points are less than half agricultural land cover.")
    if len(bad):
        print()
        for _, r in bad.iterrows():
            print(f"  {r['site']:15s} {r['country']:12s} declared '{r['declared_zone']}' "
                  f"but is {r['dominant_%']}% {r['dominant_cover']} "
                  f"(agri {r['agri_%']}%, built-up {r['builtup_%']}%, water {r['water_%']}%)")
    print()
    print("Compare this list against step 3. Points flagged by both were")
    print("identified independently: once from the NDVI signal, once from a")
    print("land-cover product that never saw our time series.")

    out = REPO_ROOT / "analysis" / "landcover_audit.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    return df


if __name__ == "__main__":
    main()
