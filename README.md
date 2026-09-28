# CropSignal

Multi-sensor Earth observation for agricultural monitoring: **verified cropland sampling**,
a **multi-index satellite time series**, **published drought indices at field scale**, and a
**phenology-aligned anomaly detector that does not beat its baseline** — built for smallholder
systems in Afghanistan and Kenya.

24 sites · 2019–2024 · 16 variables · 5 sensors · 3,287 observations · 48 tests

Author: **Maiwand Jan Alamzoi** — [m.alamzoi123@gmail.com](mailto:m.alamzoi123@gmail.com) ·
[github.com/maiwandalamzoi](https://github.com/maiwandalamzoi)

---

## Why this exists

I built a crop-stress prediction pipeline. It trained cleanly, produced sensible-looking
metrics, and worked noticeably better in Afghanistan (F1 0.63) than in the Netherlands (0.19)
or New Zealand (0.26). I assumed that was a modelling problem and started designing a better
algorithm.

Then I audited the input data.

**Fifteen of the thirty-six sampling points were not on farmland.** One sat on a river. Several
sat inside towns. The coordinates had been chosen as *"representative farmland near each named
locality"* — but the localities were town names, at two decimal places, roughly ±550 m. The
pipeline sampled town centres and called them fields.

Nothing errored. The time series were complete, the gaps were clean, six full years per site.
The models trained. The dashboards rendered. The whole thing described rooftops.

| Site | Declared as | ESA WorldCover says |
|---|---|---|
| Tiel (NL) | horticulture | **100% water** — the river Waal |
| Venlo (NL) | horticulture | **100% built-up** |
| Hastings (NZ) | horticulture | **100% built-up** |
| Ghazni City (AF) | highland rainfed | 87% built-up |
| Gore (NZ) | dairy pasture | 80% water |

This repository is what I built after that. Its organising principle is that **a sampling point
should never be an unverified input**, and more generally that a pipeline which cannot fail
loudly will fail silently instead.

---

## What it does

### 1. Sampling points are derived, not typed in

Sites are declared as *localities to search near* plus a farming system. The coordinate is
resolved against ESA **WorldCereal** and **WorldCover**, and comes back with the distance it
had to move and the cropland purity it achieved. A locality with no qualifying cropland is
dropped, not silently kept.

```python
from cropsignal.sites import resolve_sites

sites = resolve_sites()
# TransNzoia  (Kenya)  ->  1.02004, 34.95210  moved  233 m  purity 100%
# GhazniCity  (Afgh.)  -> 33.55396, 68.41578  moved  589 m  purity 100%
```

All 24 localities resolved at 100% cropland purity, moving between 3 m and 1,413 m.

### 2. Sixteen variables, five sensors — not just NDVI

NDVI saturates at canopy closure, says nothing about water status, and cannot tell a small
crop from a hot one that has stopped transpiring.

| Group | Variables | Source | Native |
|---|---|---|---|
| Optical | NDVI, EVI, SAVI, NDMI, NDRE, GCVI | Sentinel-2 SR + Cloud Score+ | 10–20 m |
| Radar | VV, VH, VH/VV | Sentinel-1 GRD | 10 m |
| Biophysics | LAI, FPAR | MODIS MCD15A3H | 500 m |
| Water | ET, PET, ESI | MODIS MOD16A2GF | 500 m |
| Thermal | LST day/night | MODIS MOD11A2 | 1 km |

Resolution is recorded per variable, so a 1 km thermal value is never silently treated as a
10 m measurement.

### 3. Drought indices at cropland scale

VCI, TCI and VHI as published (Kogan 1990, 1995), with NOAA/STAR operational thresholds
(VHI ≤ 40 stress, ≤ 26 severe) — and the ESI evaporative stress ratio.

VHI is normally produced at 1 km. Over smallholder farmland a 1 km pixel is a mixture of
fields, tracks, homesteads and bush, so the index describes the landscape rather than any farm
in it. Here VCI comes from 10 m Sentinel-2 restricted to a verified cropland mask. That is the
difference between *"this district is in drought"* and *"these parcels are"* — which is what
targeting actually requires.

```python
from cropsignal.drought import add_drought_indices
out = add_drought_indices(df)   # VCI, TCI, VHI, ESI_index, drought_class
```

Below a five-year record the indices return `NaN` rather than a confident-looking number
computed from too little history.

> **Read the thresholds with the baseline in mind.** NOAA/STAR's 40 / 26 cut-offs were set
> against AVHRR records of thirty years and more. Over six years, min-max scaling guarantees
> each site's worst observation scores 0 — so every site shows its worst season as an extreme
> drought *by construction*. On this data that puts roughly half of all observations below the
> "stress" line, which says more about the six-year window than about the crops. On a short
> baseline these indices **rank seasons within the sample; they do not classify drought in
> absolute terms.** An operational bulletin needs the long MODIS/AVHRR climatology even when
> the product is delivered at Sentinel-2 resolution.

### 4. Phenology-aligned anomaly detection

Most operational vegetation-anomaly methods compare an observation to the same *calendar date*
in previous years. That assumes the growing season happens at the same time each year.

`PhenoShift` extracts Start / Peak / End of Season per site per year, warps each season onto a
common phase axis, and scores anomalies **at matching phase rather than matching date**.
`CalendarAnomaly` implements the conventional approach with an identical API, so the two can be
compared directly on the same data.

The season boundary is detected from each site's own climatological trough, which makes the
method hemisphere-agnostic without configuration.

---

## Validation

There is no public ground-truth *"this field was stressed on this date"* dataset for these
sites. Nobody walked the field. Any label built from NDVI is a proxy, and comparing two
NDVI-derived labels to each other proves nothing.

So the labels are tested against **evidence the vegetation indices cannot contain**: MODIS
evapotranspiration and land surface temperature, measured by different instruments through
different physics than Sentinel-2 reflectance. A real crop-stress flag should coincide with a
canopy transpiring below its potential and running hot. A flag that is really detecting *"the
season started three weeks late"* has no reason to.

Two numbers decide it, and both are needed — either alone can be gamed by flagging more or
fewer observations:

- **water-stress AUC** — higher is better; the flag tracks independently measured stress
- **phenology AUC** — closer to 0.5 is better; the flag is indifferent to how far the season shifted

### Results — the phase-aligned detector does not win

24 sites, 2019–2024, 3,287 observations, independent index available for 77% of them.

| Detector | Country | n | Flagged | **Water-stress AUC** | **Phenology AUC** |
|---|---|---:|---:|---:|---:|
| Calendar | Afghanistan | 1,310 | 19.3% | 0.547 | 0.489 |
| Phase-aligned | Afghanistan | 1,065 | 23.0% | **0.573** | 0.533 |
| Calendar | Kenya | 984 | 21.4% | **0.708** | 0.510 |
| Phase-aligned | Kenya | 761 | 23.8% | 0.630 | 0.544 |

Phase alignment gains a marginal +0.026 AUC in Afghanistan and **loses 0.078 in Kenya**. On the
contamination test it is *worse in both countries* — 0.533 and 0.544 against the baseline's
0.489 and 0.510 — which is the precise opposite of what it was built to do.

**Why it fails, measured rather than guessed.** The method extracts one Start/Peak/End of
Season per year. That assumption is violated at every site in the sample:

| Country | Seasonal peaks per year | Months |
|---|---|---|
| Kenya | **2** (all 12 sites) | ~May (long rains), ~Nov–Dec (short rains) |
| Afghanistan | **2** | ~April, ~August |

With two cropping cycles a year, the extractor anchors on whichever cycle is larger that year —
so the anchors jump between cycles, and the warp ends up aligning one year's long rains against
another year's short rains. Bimodal cropping is the norm across East Africa, which is exactly
where this kind of tool is most needed.

**So: use the calendar baseline.** The phase-aligned detector ships because it is a clean
negative result with a diagnosed cause, not because it works. Fixing it means detecting the
number of cycles per year and warping each separately. That is not done here.

The parts that *do* hold up are the verified sampling, the multi-sensor extraction, and the
drought indices — plus the finding that the independent water/thermal signal is detectable at
all (Kenya calendar AUC 0.708 is a real effect, not noise).

### What this is not

Not a lab-confirmed disease classifier. Not validated against field records, because none
exist for these sites. The honest validation ladder is:

1. **Independent EO variables** (in hand, n ≈ 3,300) — ET and LST versus the NDVI-derived flag
2. **National yield statistics** (in hand, weak) — FAOSTAT gives one number per country-year;
   with 5–6 usable years this is directional only and is reported with its n
3. **Field observations** — missing, and the rung that would actually settle it

---

## Reproduce it

```bash
git clone https://github.com/maiwandalamzoi/cropsignal
cd cropsignal
pip install -e ".[analysis,dev]"

# Earth Engine credentials (service account or `earthengine authenticate`)
cp .env.example .env    # then fill in GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY

python -m cropsignal.extract                        # ~100 min, 24 sites x 138 periods
python analysis/step5_drought_and_phenology.py      # indices, phenology, validation
pytest -q                                            # 48 tests, no credentials needed
```

The audit that started all of this runs on its own:

```bash
python analysis/step1_check_the_data.py          # coverage, gaps, value ranges
python analysis/step2_look_at_one_site.py        # NDVI per year as ASCII curves
python analysis/step3_audit_sampling_points.py   # is each point actually farmland?
python analysis/step4_confirm_with_landcover.py  # confirm it independently (needs GEE)
```

Run them in that order. Step 3 infers the verdict from the NDVI signal; step 4 checks the same
question against a land-cover product that never saw the time series. Step 2 is the one that
cannot be skipped — the aggregate statistics in step 1 looked completely fine.

---

## Bugs this pipeline shipped, and what they teach

Every one produced clean output and no error. They are in `tests/test_regressions.py` so they
cannot return quietly.

| Bug | Symptom | Why it was invisible |
|---|---|---|
| Sampling points on towns and a river | Models fitted to rooftops | CSV complete, gaps clean, metrics plausible |
| EVI unclamped | Values from −2.55 to 3.84 fed to XGBoost | Valid range is −1…1; nothing checked |
| Sentinel-1 hardcoded to descending orbit | **Zero** radar for all of Kenya | Kenya is acquired on ascending passes only |
| `"ASCENDING"[:4]` → `"asce"` | Every ascending band missing | Worked for `"DESCENDING"` → `"desc"` |
| `median()` on an empty collection | Whole periods dropped | MOD16A2 begins in 2021; 2019–20 silently failed |
| Cloud Score+ joined with `saveFirst` | One unmatched scene aborted the period | Unmatched primaries carry a null score |
| No region filter on collections | Composited the entire global archive | A bad scene anywhere killed every site |
| `bestEffort=True` on a 15 km disc | 9 of 12 Kenyan sites "had no cropland" | EE silently coarsened the scale until the request fit |
| σ floor of 0 in the z-score | Stable sites permanently "anomalous" | Dividing by ~1e-16 on near-identical years |
| Validation verdict from NaN | Printed "calendar baseline is better" on zero data | Two NaNs compared, falling through to the last branch |

The pattern: **the dangerous failures are the ones that return a value.** An exception gets
fixed in ten minutes. A plausible number gets published.

---

## Design notes

- **Every sensor fails independently.** An archive gap or an orbit that does not cover a country
  yields missing values for that variable alone, never a dropped period.
- **One orbit per site, chosen from the data.** Backscatter depends on viewing geometry, so
  ascending and descending passes are not interchangeable — but hardcoding either excludes whole
  countries. Both are carried; the dominant one is selected per site.
- **Indices refuse to compute on too little history**, rather than returning a number that looks
  authoritative.
- **Resumable extraction.** Rows are flushed per period and a restart skips completed ones.

## Data sources

| Product | Use | Access |
|---|---|---|
| [Sentinel-2 SR Harmonized](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) | optical indices | GEE |
| [Cloud Score+](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) | cloud masking | GEE |
| [Sentinel-1 GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) | radar backscatter | GEE |
| [MODIS MCD15A3H](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD15A3H) | LAI, FPAR | GEE |
| [MODIS MOD16A2GF](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF) | ET, PET | GEE |
| [MODIS MOD11A2](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2) | LST | GEE |
| [ESA WorldCereal](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCereal_2021_MODELS_v100) | cropland mask | GEE |
| [ESA WorldCover v200](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200) | land cover | GEE |

## References

- Kogan, F.N. (1990). Remote sensing of weather impacts on vegetation in non-homogeneous areas.
  *International Journal of Remote Sensing*, 11(8).
- Kogan, F.N. (1995). Application of vegetation index and brightness temperature for drought
  detection. *Advances in Space Research*, 15(11).
- Anderson, M.C. et al. (2007). A climatological study of evapotranspiration and moisture stress
  across the continental United States. *Journal of Geophysical Research*, 112.
- White, M.A. et al. (1997). A continental phenology model for monitoring vegetation responses
  to interannual climatic variability. *Global Biogeochemical Cycles*, 11(2).
- Jönsson, P. & Eklundh, L. (2004). TIMESAT — a program for analyzing time-series of satellite
  sensor data. *Computers & Geosciences*, 30(8).

## License

MIT — see [LICENSE](LICENSE).
