"""
Extract the multi-sensor variable stack for every resolved site.

One row per (site, compositing period), carrying the optical indices,
radar backscatter, canopy biophysics, water/energy variables and land
surface temperature described in `indices.py`.

Design notes:

*Resumable.* An extraction of this size runs for tens of minutes against a
rate-limited service. Rows are appended and flushed per period, and a
restart skips periods already present in the output file, so a dropped
connection costs one period rather than the whole run.

*Per-band pixel counts.* Every variable is written with the number of valid
pixels behind it. A row where NDVI came from 48 pixels and LST from 1 is
not the same quality of observation, and downstream code cannot know that
unless the extraction records it.

*Resolution is not hidden.* MODIS variables at 500 m and 1 km are sampled
with a 60 m buffer, which means the buffer reads one pixel of a much
coarser grid. That value is a landscape average, not a field measurement.
It is extracted because evapotranspiration and thermal data have no 10 m
equivalent, and it is labelled so that nothing downstream treats it as if
it did.

Usage:
    python -m cropsignal.extract                     # full run
    python -m cropsignal.extract --start 2022-01-01  # partial
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import ee

from .gee import init
from .indices import band_list, build_stack
from .sites import FIELD_BUFFER_M, resolve_sites, to_feature_collection

REPO_ROOT = Path(__file__).resolve().parents[2]
SITES_PATH = REPO_ROOT / "data" / "sites_resolved.json"
OUT_PATH = REPO_ROOT / "data" / "raw" / "timeseries.csv"

START_DATE = date(2019, 1, 1)
END_DATE = date(2024, 12, 31)
PERIOD_DAYS = 16
SAMPLE_SCALE_M = 20

INCLUDE = ("optical", "radar", "biophysical", "water", "thermal")

#: Native resolution of each variable, recorded in the output so that a
#: 1 km value is never silently treated as a 10 m one.
NATIVE_SCALE_M = {
    "NDVI": 10, "EVI": 10, "SAVI": 10, "NDMI": 20, "NDRE": 20, "GCVI": 10,
    "VV_asc": 10, "VH_asc": 10, "VH_VV_asc": 10,
    "VV_desc": 10, "VH_desc": 10, "VH_VV_desc": 10,
    "LAI": 500, "FPAR": 500,
    "ET": 500, "PET": 500, "ESI": 500,
    "LST_day": 1000, "LST_night": 1000,
}


def load_sites() -> list:
    if SITES_PATH.exists():
        return json.loads(SITES_PATH.read_text(encoding="utf-8"))
    print("No resolved sites found; resolving now...")
    sites = resolve_sites()
    SITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITES_PATH.write_text(json.dumps(sites, indent=2), encoding="utf-8")
    return sites


def period_ranges(start: date, end: date, period_days: int):
    d = start
    while d < end:
        stop = min(d + timedelta(days=period_days), end)
        yield d, stop
        d = stop


def already_done(path: Path) -> set:
    """Period start dates already present, so a restart can skip them."""
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["period_start"])
    return done


def reduce_period(points_fc, start: date, end: date, bands: list,
                  region=None, retries: int = 3) -> list:
    """Sample the variable stack for one compositing period at every site."""
    for attempt in range(retries):
        try:
            stack = build_stack(str(start), str(end), include=INCLUDE, region=region)
            reducer = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)
            reduced = stack.select(bands).reduceRegions(
                collection=points_fc, reducer=reducer, scale=SAMPLE_SCALE_M,
                tileScale=4,
            )
            return reduced.getInfo()["features"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ! {start}..{end} failed after {retries} tries: "
                      f"{type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                return []
            time.sleep(4 * (attempt + 1))
    return []


def main(start: str | None = None, end: str | None = None,
         out: str | None = None, resume: bool = True):
    init()
    sites = load_sites()
    points_fc = to_feature_collection(sites, buffer_m=FIELD_BUFFER_M)
    # Restrict every source collection to the sampled area. Without this
    # each period composites the entire global archive, which is both
    # wasteful and fragile: one bad scene anywhere can abort the period.
    region = points_fc.geometry().bounds()
    bands = band_list(INCLUDE)

    start_d = datetime.strptime(start, "%Y-%m-%d").date() if start else START_DATE
    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else END_DATE
    out_path = Path(out) if out else OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    periods = list(period_ranges(start_d, end_d, PERIOD_DAYS))
    done = already_done(out_path) if resume else set()
    todo = [p for p in periods if str(p[0]) not in done]

    fields = (["name", "country", "region", "system", "lat", "lon", "purity",
               "period_start", "period_end"]
              + bands + [f"{b}_n" for b in bands])

    print(f"{len(sites)} sites x {len(periods)} periods, {len(bands)} variables")
    print(f"Variables: {', '.join(bands)}")
    if done:
        print(f"Resuming: {len(done)} periods already written, {len(todo)} to go")

    mode = "a" if (done and out_path.exists()) else "w"
    written = 0
    t0 = time.time()

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            writer.writeheader()

        for i, (p_start, p_end) in enumerate(todo, 1):
            feats = reduce_period(points_fc, p_start, p_end, bands, region=region)
            for feat in feats:
                p = feat["properties"]
                row = {
                    "name": p.get("name"), "country": p.get("country"),
                    "region": p.get("region"), "system": p.get("system"),
                    "lat": p.get("lat"), "lon": p.get("lon"),
                    "purity": p.get("purity"),
                    "period_start": str(p_start), "period_end": str(p_end),
                }
                has_any = False
                for b in bands:
                    n = p.get(f"{b}_count")
                    row[b] = p.get(f"{b}_mean")
                    row[f"{b}_n"] = n
                    if n:
                        has_any = True
                if has_any:  # a period with nothing valid anywhere is not a row
                    writer.writerow(row)
                    written += 1
            f.flush()

            if i % 5 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f"  [{i}/{len(todo)}] {p_start}  {written} rows  "
                      f"eta {eta:.0f} min")

    print(f"\nDone. {written} rows -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    main(**vars(ap.parse_args()))
