"""
Cropland masking and automated site selection.

This module exists because of a specific failure. An earlier version of this
work chose 36 sampling points by hand - "representative farmland near each
named locality", taken from public geography at two decimal places. An audit
later found that 15 of the 36 were not farmland at all: one sat on a river,
several sat squarely inside towns. Nothing in the pipeline objected. The
time series were complete and clean, the models trained, the metrics
printed, and the whole thing described rooftops.

The lesson is not "be more careful with coordinates". It is that a sampling
point should never be an unverified input in the first place. Here, points
are *derived* from cropland masks and carry a purity score, so a point that
is not on cropland cannot enter the pipeline silently.

Masks used, in the order they are trusted:

  ESA WorldCereal (2021, 10 m) - the only global product built specifically
      to delineate *cropland for cereal monitoring*, with a per-pixel
      confidence band. This is the primary mask. FAO's own EO work uses
      WorldCereal as a national crop-mapping baseline.

  ESA WorldCover (2021, 10 m) - general land cover. Used to separate
      pasture (grassland) from arable, and to exclude built-up and water.

  Dynamic World (near real time, 10 m) - optional cross-check; its `crops`
      probability band reflects the actual year rather than a 2021 epoch,
      which matters where cropland has been abandoned or newly cleared.

Agreement between independent products is treated as the confidence signal:
a pixel both WorldCereal and WorldCover call cropland is a far safer
sampling target than one where they disagree.
"""
from __future__ import annotations

import ee

# --- ESA WorldCover v200 class codes -------------------------------------
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP = 10, 20, 30, 40
WC_BUILTUP, WC_BARE, WC_SNOW, WC_WATER = 50, 60, 70, 80
WC_WETLAND, WC_MANGROVE, WC_MOSS = 90, 95, 100

WC_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse", 70: "Snow/ice", 80: "Water",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss/lichen",
}

#: WorldCover classes that can carry an agricultural signal, per farming system.
#: Orchards and vineyards are routinely classified as tree cover, so
#: horticulture has to admit class 10 or it would reject every orchard.
SYSTEM_CLASSES = {
    "arable": [WC_CROP],
    "pasture": [WC_GRASS, WC_CROP],
    "horticulture": [WC_CROP, WC_TREE],
    "any": [WC_CROP, WC_GRASS, WC_TREE],
}

WORLDCEREAL_ASSET = "ESA/WorldCereal/2021/MODELS/v100"
WORLDCOVER_ASSET = "ESA/WorldCover/v200"
DYNAMICWORLD_ASSET = "GOOGLE/DYNAMICWORLD/V1"


def worldcover() -> ee.Image:
    """ESA WorldCover v200 land-cover map (single global image)."""
    return ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map")


def worldcereal(min_confidence: int = 50) -> ee.Image:
    """
    ESA WorldCereal temporary-crops mask, thresholded on its own confidence.

    The collection is tiled by agro-ecological zone, so it is mosaicked
    rather than taken as `.first()`. `classification` is 100 where the model
    says temporary crops; `confidence` is 0-100.

    Returns a 0/1 image, unmasked (0 rather than masked-out) so that it can
    be combined with other masks without holes turning into missing data.
    """
    coll = ee.ImageCollection(WORLDCEREAL_ASSET)
    classification = coll.select("classification").mosaic()
    confidence = coll.select("confidence").mosaic()
    crops = classification.gt(0).And(confidence.gte(min_confidence))
    return crops.unmask(0).rename("worldcereal_crop")


def dynamic_world_crops(year: int, min_prob: float = 0.5) -> ee.Image:
    """
    Median `crops` probability from Dynamic World over one calendar year.

    Unlike the two 2021-epoch products this reflects the year in question,
    which is how abandoned or newly cleared land is caught.
    """
    coll = (ee.ImageCollection(DYNAMICWORLD_ASSET)
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("crops"))
    return coll.median().gte(min_prob).unmask(0).rename("dw_crop")


def agri_mask(system: str = "arable") -> ee.Image:
    """WorldCover-based agricultural mask for one farming system."""
    classes = SYSTEM_CLASSES[system]
    wc = worldcover()
    mask = wc.eq(classes[0])
    for c in classes[1:]:
        mask = mask.Or(wc.eq(c))
    return mask.rename("agri")


def combined_crop_mask(system: str = "arable", min_confidence: int = 50,
                       require_both: bool = True) -> ee.Image:
    """
    Cropland mask combining WorldCereal and WorldCover.

    `require_both=True` keeps only pixels where the two independent products
    agree. That is deliberately conservative: it loses genuine cropland at
    the margins, and in exchange the points that survive are ones two
    separately produced global datasets both call agriculture.

    For pasture systems WorldCereal (temporary *crops*) will correctly say
    "no", so agreement is not required there - WorldCover grassland governs.
    """
    wc_agri = agri_mask(system)
    if system == "pasture" or not require_both:
        return wc_agri.rename("crop_mask")
    return wc_agri.And(worldcereal(min_confidence)).rename("crop_mask")


def purity_image(mask: ee.Image, buffer_m: float) -> ee.Image:
    """
    For every pixel, the share of `mask` within `buffer_m` of it.

    This is the purity a sampling buffer would have if centred there, so
    thresholding it answers "where could I put a buffer and have it be
    genuinely all cropland?" directly.
    """
    return mask.focal_mean(radius=buffer_m, units="meters").rename("purity")


def purity_at(mask: ee.Image, lat: float, lon: float, buffer_m: float,
              scale_m: float = 20) -> float:
    """Agricultural purity of a buffer centred on one coordinate."""
    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    val = mask.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                            scale=scale_m, maxPixels=1e9).getInfo()
    return float(list(val.values())[0]) if val else float("nan")


#: Radii, in metres, tried in turn by `nearest_crop_point`.
SEARCH_RINGS = (1000, 2500, 5000, 10000, 15000)


def nearest_crop_point(mask: ee.Image, lat: float, lon: float,
                       buffer_m: float = 60, search_radius_m: float = 15000,
                       purity_min: float = 0.90, scale_m: float | None = None,
                       rings=SEARCH_RINGS) -> dict | None:
    """
    Nearest location to (lat, lon) whose sampling buffer is >= `purity_min`
    cropland.

    Nearest, rather than best or largest, so the resulting point still
    represents the locality it was meant to represent instead of drifting
    to whatever field happens to be biggest in the region.

    Searched as a sequence of expanding discs rather than one large one.
    That is not only cheaper - it is the difference between a correct and
    an incorrect answer. A single 15 km disc at 20 m is ~1.8 billion
    pixels; asked to reduce that, Earth Engine's `bestEffort` silently
    coarsens the scale until the request fits, and at the coarser scale the
    high-purity pixels this function exists to find are averaged away. The
    call then returns "no cropland here" for a locality surrounded by it.
    Small discs stay under the pixel limit, so no silent coarsening occurs.

    `scale_m` defaults to `buffer_m`: locating a 60 m buffer does not
    require a 20 m search grid, and the coarser grid keeps every ring
    exact rather than approximate.

    Returns {'lat', 'lon', 'moved_m', 'purity'} or None if nothing within
    `search_radius_m` qualifies - in which case the site should be dropped,
    not silently kept.
    """
    import math

    scale_m = scale_m or buffer_m
    purity = purity_image(mask, buffer_m)
    lonlat = ee.Image.pixelLonLat()
    dlon = lonlat.select("longitude").subtract(lon).multiply(math.cos(math.radians(lat)))
    dlat = lonlat.select("latitude").subtract(lat)
    dist = dlon.hypot(dlat).multiply(111320).rename("dist")

    stacked = (dist
               .addBands(lonlat.select("longitude"))
               .addBands(lonlat.select("latitude"))
               .addBands(purity)
               .updateMask(purity.gte(purity_min)))

    origin = ee.Geometry.Point([lon, lat])
    for radius in [r for r in rings if r <= search_radius_m] or [search_radius_m]:
        # Reducer.min(4): minimum distance, plus the other bands at that pixel.
        res = stacked.reduceRegion(
            reducer=ee.Reducer.min(4), geometry=origin.buffer(radius),
            scale=scale_m, maxPixels=1e9, bestEffort=False,
        ).getInfo()
        if res and res.get("min") is not None:
            return {
                "lat": round(res["min2"], 5),
                "lon": round(res["min1"], 5),
                "moved_m": int(res["min"]),
                "purity": round(res["min3"], 3),
            }
    return None


def sample_crop_points(region: ee.Geometry, system: str = "arable",
                       n_points: int = 12, buffer_m: float = 60,
                       purity_min: float = 0.95, seed: int = 42,
                       scale_m: float = 20) -> list:
    """
    Draw sampling points at random from high-purity cropland inside `region`.

    The alternative to hand-picking. Because the candidate pool is defined
    by the mask, every point is on cropland by construction, and because
    selection is random within it, the sample is not steered towards places
    that happen to look good.

    Returns a list of {'lat', 'lon'} dicts (fewer than `n_points` if the
    region does not contain enough qualifying cropland).
    """
    mask = combined_crop_mask(system)
    candidates = purity_image(mask, buffer_m).gte(purity_min).selfMask()
    pts = candidates.stratifiedSample(
        numPoints=n_points, classBand="purity", region=region, scale=scale_m,
        seed=seed, geometries=True, dropNulls=True, tileScale=4,
    ).getInfo()["features"]
    return [{"lon": f["geometry"]["coordinates"][0],
             "lat": f["geometry"]["coordinates"][1]} for f in pts]


def describe_cover(lat: float, lon: float, buffer_m: float = 60,
                   scale_m: float = 10) -> dict:
    """
    WorldCover composition inside a buffer, as {class name: share}.

    Used by the audit to say what a point actually is, in words, rather
    than only whether it passed a threshold.
    """
    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
    hist = worldcover().reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=geom,
        scale=scale_m, maxPixels=1e9).getInfo().get("Map", {})
    total = sum(hist.values()) or 1
    return {WC_NAMES.get(int(float(k)), k): round(v / total, 3)
            for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}
