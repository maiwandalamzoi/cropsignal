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

WHEN A SITE HAS BEEN VERIFIED ANOTHER WAY
------------------------------------------
NDVI peak/amplitude is a proxy for "is this cropland?" - a cheap one, useful
when nothing else is available. `sites.py` / `masks.py` in this package
produce something stronger: each site's cropland share as measured by two
independent land-cover products (ESA WorldCereal + WorldCover), which has
nothing to do with the vegetation index at all.

Where that purity score exists for a site, it is trusted over the NDVI
heuristic. A verified-cropland site with a low NDVI peak is not a false
positive to re-litigate - it is a real, low-vigor field (arid irrigation, a
sparse crop, a partial-season extract), and re-flagging it as "FAIL" on
NDVI alone would just be re-introducing a cruder version of the same
mistake this audit exists to catch: trusting one signal that happens to
agree with your priors.

Purity is applied only when it can be tied to the *exact coordinate* being
audited, not merely to a site name. Two datasets in this project reuse the
same names ("Zaranj", "Shakardara", "GhazniCity", ...) for what are, at the
old hand-picked coordinates, different and often non-cropland points - see
`sites.py`. Matching by name alone would let a verified site's purity leak
onto an unrelated, unverified point of the same name and silently defeat
the audit on exactly the dataset it was built to catch. Coordinates are
matched to within ~200 m (comfortably inside one sampling buffer) before a
purity score is trusted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_LEGACY_CSV = _REPO_ROOT / "data" / "legacy" / "sentinel2_timeseries.csv"
_SIBLING_LEGACY_CSV = (_REPO_ROOT.parent / "crop-stress-prediction"
                       / "data" / "raw" / "sentinel2_timeseries.csv")
#: A bundled copy ships inside this repo (data/legacy/) so the original audit
#: is reproducible from a standalone clone - a fresh `git clone` of cropsignal
#: alone has no sibling crop-stress-prediction checkout to read. That sibling
#: path is tried second, only for local development against a live copy.
LEGACY_CSV = _BUNDLED_LEGACY_CSV if _BUNDLED_LEGACY_CSV.exists() else _SIBLING_LEGACY_CSV
DEFAULT_CSV = LEGACY_CSV
OWN_CSV = _REPO_ROOT / "data" / "raw" / "timeseries.csv"
OWN_SITES_JSON = _REPO_ROOT / "data" / "sites_resolved.json"

#: Within this many degrees (~200 m at these latitudes) a CSV row's
#: coordinate is considered the same point as a resolved site's coordinate.
COORD_MATCH_TOL_DEG = 0.002


def _resolve_csv(argv):
    """--csv PATH, else this repo's extract, else the legacy one."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="time series CSV to audit")
    args, _ = ap.parse_known_args(argv)
    if args.csv:
        return Path(args.csv)
    if OWN_CSV.exists():
        return OWN_CSV
    return DEFAULT_CSV


def _load_purity_by_coord(sites_json: Path = OWN_SITES_JSON) -> list:
    """[(name, lat, lon, purity), ...] from a resolve_sites() output file."""
    if not sites_json.exists():
        return []
    import json
    sites = json.loads(sites_json.read_text(encoding="utf-8"))
    return [(s["name"], s["lat"], s["lon"], s["purity"]) for s in sites]


def _purity_for(name: str, lat: float, lon: float, purity_table: list,
                tol: float = COORD_MATCH_TOL_DEG) -> float | None:
    """
    Purity for a (name, lat, lon), only if a resolved site of that name
    sits within `tol` degrees of this exact coordinate.

    The name has to match too, not just the coordinate - belt and braces
    against two unrelated sites ever landing within tolerance of each other.
    """
    for rname, rlat, rlon, rpurity in purity_table:
        if rname == name and abs(rlat - lat) <= tol and abs(rlon - lon) <= tol:
            return rpurity
    return None


PEAK_MIN = 0.55        # below this a point is not reaching crop canopy closure
AMPLITUDE_MIN = 0.15   # below this there is no growing-season cycle to speak of
WATER_PEAK_MAX = 0.15  # peak this low means water or fully built-up
PURITY_MIN = 0.90      # cropland share (independent land-cover products) to trust outright


def classify(peak: float, amp: float, purity: float | None = None) -> str:
    """
    Verdict for one site.

    `purity` (0-1, or None) is the cropland share from an independent
    land-cover check (WorldCereal + WorldCover), tied to this exact
    coordinate - see module docstring. When it clears `PURITY_MIN`, it
    settles the "is this cropland?" question and NDVI is read only as a
    vigor signal, not as a verdict. When `purity` is None - no independent
    check exists for this point - NDVI is all there is, and the original
    thresholds decide alone.
    """
    if purity is not None and purity >= PURITY_MIN:
        if peak < WATER_PEAK_MAX:
            # A real contradiction between two independent sources, not a
            # case where purity should just win by default.
            return "CHECK - verified cropland but NDVI reads as water/bare"
        if peak < PEAK_MIN:
            return "OK   - verified cropland, low-vigor signal"
        return "OK   - verified cropland"

    if peak < WATER_PEAK_MAX:
        return "FAIL - water/built-up"
    if peak < PEAK_MIN and amp < AMPLITUDE_MIN:
        return "FAIL - no crop signal"
    if peak < PEAK_MIN:
        return "WEAK - low canopy peak"
    if amp < AMPLITUDE_MIN:
        return "OK?  - pasture-like (green year-round)"
    return "OK   - cropland-like"


def main(csv_path: Path = DEFAULT_CSV, sites_json: Path = OWN_SITES_JSON) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["period_start"])
    df["month"] = df["period_start"].dt.month

    purity_table = _load_purity_by_coord(sites_json)
    if purity_table:
        print(f"Cross-checking against {len(purity_table)} independently verified "
              f"sites in {sites_json.name}\n(matched by name + coordinate, not name alone)\n")

    rows = []
    for (country, site), g in df.groupby(["country", "name"]):
        clim = g.groupby("month")["NDVI"].mean()
        peak = float(g["NDVI"].quantile(0.95))
        amp = float(clim.max() - clim.min())
        lat, lon = float(g["lat"].iloc[0]), float(g["lon"].iloc[0])
        purity = _purity_for(site, lat, lon, purity_table)
        rows.append({
            "country": country, "site": site,
            "peak_ndvi": round(peak, 2),
            "seasonal_amp": round(amp, 3),
            "purity": purity,
            "verdict": classify(peak, amp, purity),
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
    checked = audit[audit["verdict"].str.startswith("CHECK")]
    verified_low_vigor = audit[audit["verdict"].str.contains("low-vigor")]

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
        print("No point fails outright.")

    if len(checked):
        print()
        print(f"{len(checked)} point(s) need a manual look: verified as cropland by "
              f"WorldCereal/WorldCover,\nbut NDVI reads as water or bare ground at "
              f"this coordinate. That is a real\ndisagreement between two independent "
              f"sources, not something this script can\nresolve on its own.")
        for _, r in checked.iterrows():
            print(f"  {r['site']:15s} ({r['country']:12s})  "
                  f"peak {r['peak_ndvi']:.2f}  purity {r['purity']:.0%}")

    if len(verified_low_vigor):
        print()
        print(f"{len(verified_low_vigor)} point(s) are confirmed cropland (WorldCereal + "
              f"WorldCover) but score\nbelow the NDVI heuristic alone. Real, low-vigor "
              f"fields, not false positives -\nsee the module docstring for why purity "
              f"overrides NDVI here:")
        for _, r in verified_low_vigor.iterrows():
            print(f"  {r['site']:15s} ({r['country']:12s})  "
                  f"peak {r['peak_ndvi']:.2f}  purity {r['purity']:.0%}")

    print()
    print("Per-country usable share:")
    audit["usable"] = ~audit["verdict"].str.startswith(("FAIL", "CHECK"))
    print(audit.groupby("country")["usable"]
          .agg(usable="sum", total="size")
          .assign(share=lambda d: (d["usable"] / d["total"] * 100).round(0).astype(int).astype(str) + "%")
          .to_string())

    return audit


if __name__ == "__main__":
    import sys
    main(_resolve_csv(sys.argv[1:]))
