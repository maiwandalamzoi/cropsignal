"""
STEP 5 - Drought indices, phenology, and an honest test of the anomaly labels.

Run:  python analysis/step5_drought_and_phenology.py

This step does three things, in increasing order of how much they can be
argued with.

1. DROUGHT INDICES. Compute VCI, TCI and VHI (Kogan 1990, 1995) per site
   per period, at 10 m cropland resolution rather than the 1 km at which
   VHI is usually produced, and summarise each season the way an
   early-warning bulletin would: how low the index went, and for how long.

2. PHENOLOGY. Extract Start / Peak / End of Season per site per year and
   measure how much the season actually moves between years. This is the
   quantity that decides whether comparing an observation to the same
   calendar date last year is a fair comparison or a category error.

3. VALIDATION. Compare the calendar-based and phenology-aligned anomaly
   detectors - but not against each other's labels, which would prove
   nothing, and not against ground truth, which does not exist for these
   sites.

   Instead, against evidence the vegetation indices cannot contain.
   Evapotranspiration (MODIS MOD16A2GF) and land surface temperature
   (MOD11A2) are measured by different instruments through different
   physics than the Sentinel-2 reflectance the labels are built from. A
   real crop-stress flag should coincide with a crop that is transpiring
   below its potential and running hot. A flag that is really just
   detecting a season that started three weeks late has no reason to.

   Two numbers decide it, and both are needed - either alone can be gamed
   by flagging more or fewer observations:

     water-stress AUC     higher is better: the flag tracks independently
                          measured evaporative and thermal stress
     phenology AUC        closer to 0.5 is better: the flag is indifferent
                          to how far the season shifted
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cropsignal.anomaly import CalendarAnomaly, PhenoShift  # noqa: E402
from cropsignal.drought import (STRESS_THRESHOLD, add_drought_indices,  # noqa: E402
                                season_summary)
from cropsignal.indices import select_dominant_orbit  # noqa: E402
from cropsignal.phenology import (detect_season_start, phenometrics,  # noqa: E402
                                  to_phenological_year)
from cropsignal.validate import (_auc, phenology_contamination,  # noqa: E402
                                 weather_association)

DATA = REPO / "data" / "raw" / "timeseries.csv"
PERIOD_DAYS = 16


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit(f"No extraction found at {DATA}\n"
                         f"Run:  python -m cropsignal.extract")
    df = pd.read_csv(DATA, parse_dates=["period_start"])
    df = select_dominant_orbit(df, site_col="name")
    df["site"] = df["name"]
    df["year"] = df["period_start"].dt.year
    df["doy"] = df["period_start"].dt.dayofyear
    df["period_of_year"] = ((df["doy"] - 1) // PERIOD_DAYS).astype(int)
    df["season_year"] = df["year"]
    return df.sort_values(["site", "period_start"]).reset_index(drop=True)


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_coverage(df):
    section("1. VARIABLE COVERAGE")
    groups = {
        "Optical (S2, 10-20 m)": ["NDVI", "EVI", "SAVI", "NDMI", "NDRE", "GCVI"],
        "Radar (S1, 10 m)": ["VV", "VH", "VH_VV"],
        "Biophysics (MODIS, 500 m)": ["LAI", "FPAR"],
        "Water (MODIS, 500 m)": ["ET", "PET", "ESI"],
        "Thermal (MODIS, 1 km)": ["LST_day", "LST_night"],
    }
    print(f"{'variable group':28s} {'Afghanistan':>12s} {'Kenya':>8s}")
    print("-" * 52)
    for label, cols in groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        cov = df.groupby("country")[cols].apply(lambda g: g.notna().mean().mean())
        af = f"{cov.get('Afghanistan', float('nan')):.0%}"
        ke = f"{cov.get('Kenya', float('nan')):.0%}"
        print(f"{label:28s} {af:>12s} {ke:>8s}")

    print()
    print("Radar matters most where optical coverage is worst: Kenya's growing")
    print("seasons are cloudy, and Sentinel-1 sees through cloud.")
    print(f"\nOrbit chosen per site: "
          f"{df.groupby('country')['orbit'].agg(lambda s: s.mode()[0]).to_dict()}")


def report_drought(df):
    section("2. DROUGHT INDICES (VCI / TCI / VHI, Kogan 1990, 1995)")
    out = add_drought_indices(df, ndvi_col="NDVI", lst_col="LST_day",
                              esi_col="ESI", group_cols=("site", "period_of_year"),
                              min_years=5)
    scored = out["VHI"].notna().sum()
    print(f"VHI computed for {scored:,} of {len(out):,} observations "
          f"({scored/len(out):.0%}); the rest lack 5 years of record at that "
          f"point in the season.")

    print(f"\nWorst seasons by share of the season under VHI {STRESS_THRESHOLD} "
          f"(NOAA/STAR stress threshold):")
    seasons = season_summary(out, by=("site", "country", "season_year"))
    seasons = seasons[seasons["periods"] >= 10]
    worst = seasons.sort_values("stressed_share", ascending=False).head(12)
    print(worst[["site", "country", "season_year", "vhi_min", "vhi_mean",
                 "periods_stressed", "periods", "stressed_share"]]
          .to_string(index=False))

    print("\nMean share of season under stress, by country and year:")
    piv = seasons.pivot_table(index="season_year", columns="country",
                              values="stressed_share", aggfunc="mean")
    print((piv * 100).round(0).astype("Int64").to_string())
    return out


def report_phenology(df):
    section("3. PHENOLOGY: HOW FAR DOES THE SEASON ACTUALLY MOVE?")
    rows = []
    for site, g in df.groupby("site"):
        g = g.dropna(subset=["NDVI"])
        if len(g) < 40:
            continue
        start = detect_season_start(g["doy"].values, g["NDVI"].values)
        py, days = to_phenological_year(g["doy"].values, g["year"].values, start)
        peaks, sos = [], []
        for p in np.unique(py):
            sel = py == p
            if sel.sum() < 12:
                continue
            span = (days[sel].max() - days[sel].min()) / 365.25
            if span < 0.75:
                continue
            m = phenometrics(days[sel], g["NDVI"].values[sel])
            if m:
                peaks.append(m["pos"])
                sos.append(m["sos"])
        if len(peaks) >= 4:
            rows.append({
                "site": site, "country": g["country"].iloc[0],
                "seasons": len(peaks),
                "pos_sd_days": round(float(np.std(peaks, ddof=1)), 1),
                "sos_sd_days": round(float(np.std(sos, ddof=1)), 1),
                "pos_range_days": int(max(peaks) - min(peaks)),
            })
    ph = pd.DataFrame(rows)
    if ph.empty:
        print("Not enough complete seasons yet to measure phenological drift.")
        return ph

    print("Inter-annual variability of season timing, per site:")
    print(ph.sort_values(["country", "pos_sd_days"]).to_string(index=False))
    print("\nBy country (median across sites):")
    print(ph.groupby("country")[["pos_sd_days", "sos_sd_days", "pos_range_days"]]
          .median().round(1).to_string())
    print("\nThe larger these are, the less defensible it is to compare an")
    print("observation to the same calendar date in another year.")
    return ph


def build_independent_stress_index(df):
    """
    A crop-stress signal built only from water and energy variables.

    ESI (ET/PET) falls when a canopy closes its stomata; LST rises when it
    stops transpiring. Both are standardised against each site's own
    distribution at that point in the season, then combined so that high
    means stressed. Nothing here touches the reflectance the anomaly labels
    are derived from, which is the entire point.
    """
    out = df.copy()
    idx = np.full(len(out), np.nan)
    have = 0
    for _, g in out.groupby(["site", "period_of_year"]):
        esi, lst = g.get("ESI"), g.get("LST_day")
        parts = []
        if esi is not None and esi.notna().sum() >= 4 and esi.std(ddof=1) > 0:
            parts.append(-(esi - esi.mean()) / esi.std(ddof=1))   # low ESI = stressed
        if lst is not None and lst.notna().sum() >= 4 and lst.std(ddof=1) > 0:
            parts.append((lst - lst.mean()) / lst.std(ddof=1))    # high LST = stressed
        if parts:
            idx[out.index.get_indexer(g.index)] = np.nanmean(
                np.vstack([p.values for p in parts]), axis=0)
            have += 1
    out["water_stress_index"] = idx
    return out


def report_validation(df):
    section("4. VALIDATION AGAINST EVIDENCE THE LABELS CANNOT CONTAIN")
    base = build_independent_stress_index(df)
    usable = base["water_stress_index"].notna().sum()
    print(f"Independent water/thermal stress index available for "
          f"{usable:,} of {len(base):,} observations ({usable/len(base):.0%}).")
    print("Built from MODIS evapotranspiration and land surface temperature")
    print("only - no Sentinel-2 reflectance, so it cannot restate the labels.\n")

    results = []
    for label, Detector in [("calendar", CalendarAnomaly), ("phenology-aligned", PhenoShift)]:
        det = Detector(site_col="site", time_col="period_start", value_col="NDVI")
        scored = det.fit_transform(base)
        scored["water_stress_index"] = base["water_stress_index"].values
        if "pos_shift_days" not in scored or scored["pos_shift_days"].isna().all():
            scored["pos_shift_days"] = np.nan

        for country in sorted(scored["country"].dropna().unique()):
            sub = scored[scored["country"] == country]
            w = weather_association(sub, "anomaly", "water_stress_index")
            row = {"detector": label, "country": country,
                   "n": w["n"], "flag_rate": w["flag_rate"],
                   "water_auc": w["auc"]}
            if sub["pos_shift_days"].notna().any():
                p = phenology_contamination(sub, "anomaly", "pos_shift_days")
                row["phenology_auc"] = p["auc_abs_shift"]
            else:
                row["phenology_auc"] = np.nan
            results.append(row)

    res = pd.DataFrame(results)
    res["flag_rate"] = (res["flag_rate"] * 100).round(1)
    res["water_auc"] = res["water_auc"].round(3)
    res["phenology_auc"] = res["phenology_auc"].round(3)

    print("water_auc   : higher is better (0.5 = the flag carries no independent signal)")
    print("phenology_auc: closer to 0.5 is better (higher = flagging shifted seasons)")
    print()
    print(res.to_string(index=False))

    print()
    for country in sorted(res["country"].unique()):
        sub = res[res["country"] == country].set_index("detector")
        if len(sub) < 2:
            continue
        delta = sub.loc["phenology-aligned", "water_auc"] - sub.loc["calendar", "water_auc"]
        verdict = ("phenology alignment helps" if delta > 0.01 else
                   "no material difference" if delta > -0.01 else
                   "calendar baseline is better here")
        print(f"  {country:12s} water_auc {sub.loc['calendar','water_auc']:.3f} -> "
              f"{sub.loc['phenology-aligned','water_auc']:.3f}  ({delta:+.3f})  {verdict}")

    print()
    print("Read these as effect sizes, not proof. An AUC near 0.5 on both")
    print("detectors would mean neither label tracks independent stress, and")
    print("no amount of comparing them to each other would reveal that.")
    return res


def main():
    df = load()
    print(f"Loaded {len(df):,} rows, {df['site'].nunique()} sites, "
          f"{df['period_start'].min().date()} to {df['period_start'].max().date()}")
    report_coverage(df)
    drought = report_drought(df)
    report_phenology(df)
    res = report_validation(drought)

    out_dir = REPO / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    drought.to_csv(out_dir / "indices_with_drought.csv", index=False)
    res.to_csv(out_dir / "validation_results.csv", index=False)
    print(f"\nWrote {out_dir / 'indices_with_drought.csv'}")
    print(f"Wrote {out_dir / 'validation_results.csv'}")


if __name__ == "__main__":
    main()
