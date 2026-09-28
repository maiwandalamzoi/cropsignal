"""
Multi-sensor, multi-index extraction of agricultural EO variables.

NDVI alone is a thin description of a crop. It saturates once the canopy
closes, says nothing about water status, and cannot distinguish a crop that
is merely small from one that is hot and transpiring poorly. Operational
agricultural monitoring therefore reads several variables together, and
this module extracts the ones that matter for crop condition, biomass and
drought:

  Optical structure and greenness (Sentinel-2, 10-20 m)
    NDVI   normalised difference vegetation index - greenness, saturates high
    EVI    enhanced vegetation index - resists saturation, corrects aerosols
    SAVI   soil-adjusted VI - for partial canopies over bright soil
    NDMI   normalised difference moisture index (NIR/SWIR) - canopy water
    NDRE   red-edge index - nitrogen status, less prone to saturation
    GCVI   green chlorophyll vegetation index - strongly tied to LAI in cereals

  Canopy biophysics (MODIS MCD15A3H, 500 m, 4-daily)
    LAI    leaf area index - the variable crop growth models actually want
    FPAR   fraction of absorbed PAR - drives light-use-efficiency biomass models

  Water and energy (MODIS MOD16A2 500 m / MOD11A2 1 km, 8-daily)
    ET     actual evapotranspiration
    PET    potential evapotranspiration
    ET/PET evaporative stress ratio - a direct water-stress indicator
    LST    land surface temperature, day and night

  Radar structure (Sentinel-1 GRD, 10 m)
    VV, VH backscatter, and the VH/VV ratio. Radar sees through cloud, which
           is what makes it indispensable in a tropical growing season where
           optical data is missing for weeks at a time.

Scaling factors are applied here, once, so that downstream code never has
to remember that MODIS LAI is stored at 0.1 and LST at 0.02 Kelvin.
"""
from __future__ import annotations

import ee

S2_ASSET = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
S1_ASSET = "COPERNICUS/S1_GRD"
LAI_ASSET = "MODIS/061/MCD15A3H"
# Gap-filled evapotranspiration. The near-real-time MOD16A2 collection in
# Earth Engine begins only in 2021, so a historical run against it returns no
# ET at all before that date - silently, as missing values. MOD16A2GF covers
# 2000 onward. The trade-off is that the gap-filled product is reprocessed at
# the end of each year, so it suits historical analysis; an operational
# near-real-time system should read MOD16A2 and accept the shorter record.
ET_ASSET = "MODIS/061/MOD16A2GF"
LST_ASSET = "MODIS/061/MOD11A2"

OPTICAL_BANDS = ["NDVI", "EVI", "SAVI", "NDMI", "NDRE", "GCVI"]
RADAR_BANDS = ["VV_asc", "VH_asc", "VH_VV_asc", "VV_desc", "VH_desc", "VH_VV_desc"]
BIOPHYS_BANDS = ["LAI", "FPAR"]
WATER_BANDS = ["ET", "PET", "ESI"]
THERMAL_BANDS = ["LST_day", "LST_night"]
ALL_BANDS = OPTICAL_BANDS + BIOPHYS_BANDS + WATER_BANDS + THERMAL_BANDS + RADAR_BANDS

#: Band-name suffix per Sentinel-1 orbit direction.
ORBIT_SUFFIX = {"ASCENDING": "asc", "DESCENDING": "desc"}

#: Cloud Score+ quality below which a Sentinel-2 pixel is discarded.
#: 0.60 is the value ESA/Google suggest for general land applications.
CS_THRESHOLD = 0.60


def _safe_median(collection: ee.ImageCollection, band_names: list) -> ee.Image:
    """
    Median of a collection that is guaranteed to carry `band_names`.

    The median of an *empty* Earth Engine collection is an image with no
    bands, and selecting from it raises. Inside a multi-sensor extraction
    that single failure aborts the whole compositing period, discarding the
    sensors that did return data - so a product with a gap in its archive,
    or an orbit that does not cover a country, takes everything else down
    with it.

    Merging a fully masked placeholder keeps the band structure intact, so
    an unavailable sensor yields missing values for itself alone.
    """
    placeholder = (ee.Image.constant([0] * len(band_names))
                   .rename(band_names)
                   .updateMask(ee.Image.constant(0))
                   .float())
    safe = collection.map(lambda i: i.float()).merge(ee.ImageCollection([placeholder]))
    return safe.median().select(band_names)


# --------------------------------------------------------------------------
# Sentinel-2 optical
# --------------------------------------------------------------------------

def _mask_s2(img: ee.Image) -> ee.Image:
    """
    Cloud-mask a Sentinel-2 scene using Cloud Score+, falling back to SCL.

    Cloud Score+ is a learned per-pixel usability score trained across the
    whole archive. It is markedly better than SCL at the cases that quietly
    corrupt a time series - thin cirrus, haze, and cloud shadow edges - which
    are exactly the pixels that survive an SCL filter and then depress NDVI
    just enough to look like crop stress.
    """
    scl = img.select("SCL")
    scl_ok = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(scl_ok).updateMask(img.select("cs_cdf").gte(CS_THRESHOLD))


def _s2_indices(img: ee.Image) -> ee.Image:
    b2 = img.select("B2").divide(10000)    # blue
    b3 = img.select("B3").divide(10000)    # green
    b4 = img.select("B4").divide(10000)    # red
    b5 = img.select("B5").divide(10000)    # red edge 1
    b8 = img.select("B8").divide(10000)    # NIR
    b11 = img.select("B11").divide(10000)  # SWIR 1

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")

    # EVI's denominator can approach zero over water, wet bare soil and
    # cloud edges, which sends the index to implausible magnitudes. An
    # earlier version of this pipeline shipped EVI values from -2.5 to 3.8
    # straight into a gradient-boosting model. Clamp to the physical range.
    evi = b8.subtract(b4).multiply(2.5).divide(
        b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1)
    ).clamp(-1, 1).rename("EVI")

    savi = b8.subtract(b4).divide(b8.add(b4).add(0.5)).multiply(1.5).rename("SAVI")
    ndmi = b8.subtract(b11).divide(b8.add(b11)).rename("NDMI")
    ndre = b8.subtract(b5).divide(b8.add(b5)).rename("NDRE")
    gcvi = b8.divide(b3).subtract(1).clamp(-1, 20).rename("GCVI")

    return img.addBands([ndvi, evi, savi, ndmi, ndre, gcvi])


def s2_composite(start: str, end: str, region: ee.Geometry | None = None) -> ee.Image:
    """
    Cloud-masked median Sentinel-2 index composite for a date window.

    `region` matters for correctness as well as cost. Without it the
    collection is the whole global archive for those dates, so a single
    malformed or unmatched scene on the other side of the world can abort
    the composite for every site being sampled.

    Cloud Score+ is attached with an *inner* join rather than `saveFirst`.
    saveFirst emits every primary scene, matched or not, leaving unmatched
    ones with a null score that raises when read; an inner join emits only
    matched pairs, so the failure cannot arise.
    """
    s2 = ee.ImageCollection(S2_ASSET).filterDate(start, end)
    cs = ee.ImageCollection(S2_CLOUD_SCORE).filterDate(start, end)
    if region is not None:
        s2 = s2.filterBounds(region)
        cs = cs.filterBounds(region)

    joined = ee.Join.inner("primary", "secondary").apply(
        primary=s2, secondary=cs.select("cs_cdf"),
        condition=ee.Filter.equals(leftField="system:index",
                                   rightField="system:index"),
    )
    paired = ee.ImageCollection(joined.map(
        lambda f: ee.Image(f.get("primary")).addBands(ee.Image(f.get("secondary")))
    ))
    return _safe_median(paired.map(_mask_s2).map(_s2_indices).select(OPTICAL_BANDS),
                        OPTICAL_BANDS)


# --------------------------------------------------------------------------
# Sentinel-1 radar
# --------------------------------------------------------------------------

def _s1_orbit_composite(start: str, end: str, orbit: str,
                        region: ee.Geometry | None = None) -> ee.Image:
    """
    Median backscatter for one orbit, guaranteed to carry its bands.

    An orbit with no acquisitions over a period yields an empty collection,
    whose median is an image with *no bands at all* - and selecting a band
    from it throws, taking down the whole period's extraction including the
    sensors that did have data. Merging a fully masked dummy image keeps the
    band structure intact, so an orbit with no data comes back as missing
    values for that orbit rather than as a failure.
    """
    s1 = (ee.ImageCollection(S1_ASSET)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", orbit))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"]))
    if region is not None:
        s1 = s1.filterBounds(region)

    med = _safe_median(s1, ["VV", "VH"])

    suffix = ORBIT_SUFFIX[orbit]
    ratio = med.select("VH").subtract(med.select("VV"))
    return (med.select(["VV", "VH"], [f"VV_{suffix}", f"VH_{suffix}"])
               .addBands(ratio.rename(f"VH_VV_{suffix}")))


def s1_composite(start: str, end: str, region: ee.Geometry | None = None) -> ee.Image:
    """
    Median VV/VH backscatter (dB) and the VH/VV ratio, for both orbits.

    Backscatter depends on viewing geometry, so ascending and descending
    passes are not interchangeable: averaging them together introduces
    steps in a time series that have nothing to do with the crop. The
    conventional fix is to pick one orbit and filter to it.

    That fix fails silently across regions. Sentinel-1's acquisition plan
    is not globally uniform - Kenya, for instance, is acquired on
    ascending passes only, so a pipeline hardcoded to descending returns
    no radar there at all while continuing to work in other countries.

    Both orbits are therefore carried as separate bands, and the dominant
    orbit is chosen per site afterwards (`select_dominant_orbit`). Each
    site's series stays on one consistent geometry, and no region is
    excluded by a choice made for a different part of the world.
    """
    asc = _s1_orbit_composite(start, end, "ASCENDING", region)
    desc = _s1_orbit_composite(start, end, "DESCENDING", region)
    return asc.addBands(desc)


# --------------------------------------------------------------------------
# MODIS biophysical, water and thermal
# --------------------------------------------------------------------------

def modis_biophysical(start: str, end: str) -> ee.Image:
    """LAI and FPAR, scaled to physical units and QC-filtered."""
    coll = ee.ImageCollection(LAI_ASSET).filterDate(start, end)

    def clean(img):
        # FparLai_QC bit 0 == 0 means the main algorithm succeeded; anything
        # else is a backup retrieval whose LAI is not comparable.
        qc = img.select("FparLai_QC")
        good = qc.bitwiseAnd(1).eq(0)
        lai = img.select("Lai").multiply(0.1).rename("LAI")
        fpar = img.select("Fpar").multiply(0.01).rename("FPAR")
        return lai.addBands(fpar).updateMask(good)

    return _safe_median(coll.map(clean).select(BIOPHYS_BANDS), BIOPHYS_BANDS)


def modis_water(start: str, end: str) -> ee.Image:
    """
    Actual and potential evapotranspiration, plus the evaporative stress index.

    ESI = ET / PET is the fraction of atmospheric demand the canopy is
    actually meeting. It falls when a crop closes its stomata under water
    stress, and it does so *before* greenness declines - which is what makes
    it valuable for early warning rather than after-the-fact damage mapping.
    """
    coll = ee.ImageCollection(ET_ASSET).filterDate(start, end)

    def clean(img):
        # Fill values (>32760) mark water, barren and unclassified pixels.
        et = img.select("ET").multiply(0.1).rename("ET")
        pet = img.select("PET").multiply(0.1).rename("PET")
        valid = img.select("ET").lt(32760).And(img.select("PET").lt(32760))
        return et.addBands(pet).updateMask(valid)

    med = _safe_median(coll.map(clean).select(["ET", "PET"]), ["ET", "PET"])
    esi = med.select("ET").divide(med.select("PET").max(0.01)).clamp(0, 1.5).rename("ESI")
    return med.addBands(esi).select(WATER_BANDS)


def modis_thermal(start: str, end: str) -> ee.Image:
    """Day and night land surface temperature in degrees Celsius."""
    coll = ee.ImageCollection(LST_ASSET).filterDate(start, end)

    def clean(img):
        day = img.select("LST_Day_1km").multiply(0.02).subtract(273.15).rename("LST_day")
        night = img.select("LST_Night_1km").multiply(0.02).subtract(273.15).rename("LST_night")
        # QC bits 0-1 == 0 -> good quality retrieval
        qc_day = img.select("QC_Day").bitwiseAnd(3).eq(0)
        return day.updateMask(qc_day).addBands(night)

    return _safe_median(coll.map(clean).select(THERMAL_BANDS), THERMAL_BANDS)


# --------------------------------------------------------------------------
# Combined stack
# --------------------------------------------------------------------------

def build_stack(start: str, end: str, include: tuple = ("optical", "radar",
                                                        "biophysical", "water",
                                                        "thermal"),
                region: ee.Geometry | None = None) -> ee.Image:
    """
    One image carrying every requested variable for a compositing window.

    Bands are at their native resolutions; `reduceRegions` resamples each to
    the requested scale on read. A 60 m buffer sampling a 1 km MODIS LST
    pixel is reading one pixel's value, not a 60 m measurement - that
    resolution mismatch is real and is documented rather than hidden.
    """
    parts = []
    if "optical" in include:
        parts.append(s2_composite(start, end, region))
    if "radar" in include:
        parts.append(s1_composite(start, end, region))
    if "biophysical" in include:
        parts.append(modis_biophysical(start, end))
    if "water" in include:
        parts.append(modis_water(start, end))
    if "thermal" in include:
        parts.append(modis_thermal(start, end))

    stack = parts[0]
    for p in parts[1:]:
        stack = stack.addBands(p)
    return stack


def band_list(include: tuple) -> list:
    """Band names produced by `build_stack` for a given `include` selection."""
    out = []
    if "optical" in include:
        out += OPTICAL_BANDS
    if "radar" in include:
        out += RADAR_BANDS
    if "biophysical" in include:
        out += BIOPHYS_BANDS
    if "water" in include:
        out += WATER_BANDS
    if "thermal" in include:
        out += THERMAL_BANDS
    return out


def select_dominant_orbit(df, site_col: str = "name"):
    """
    Collapse the per-orbit radar bands into single VV/VH/VH_VV columns.

    For each site the orbit with more valid observations across the whole
    record wins, and that orbit's values are used throughout. Choosing once
    per site - rather than per row - is what keeps the series on a single
    viewing geometry; taking whichever orbit happened to be available each
    period would reintroduce exactly the steps the split avoids.

    Adds an `orbit` column recording the choice.
    """
    import numpy as np

    out = df.copy()
    out["orbit"] = None
    for site, g in out.groupby(site_col):
        n_asc = g["VV_asc"].notna().sum()
        n_desc = g["VV_desc"].notna().sum()
        orbit = "asc" if n_asc >= n_desc else "desc"
        idx = g.index
        out.loc[idx, "orbit"] = orbit
        for base in ("VV", "VH", "VH_VV"):
            out.loc[idx, base] = g[f"{base}_{orbit}"].values
    return out
