"""
Sampling sites, defined as localities to search near rather than coordinates
to trust.

The distinction is the whole point. A site here carries an *approximate*
anchor - a county or district, good to a kilometre or so - and a farming
system. The actual sampling coordinate is resolved by `resolve_sites()`
against the cropland masks, and comes back with the distance it had to move
and the cropland purity it achieved.

So a coordinate can no longer be wrong in the way the earlier version was:
either the resolver finds verified cropland near the locality and reports
how far away it was, or the site is dropped. There is no path by which an
unverified point reaches the extraction stage.

Two countries, chosen to be genuinely different monitoring problems:

  Afghanistan - Central Asian smallholder systems, largely irrigated from
      snowmelt in a semi-arid climate, single annual cropping cycle, sharp
      seasonal contrast. Spring wheat and winter wheat dominate.

  Kenya - sub-Saharan smallholder systems with *two* growing seasons a
      year: the long rains (March-May) and the short rains
      (October-December). Bimodal cropping breaks the single-season
      assumption built into most phenology extraction, including the first
      version of the one in this package, and it is the more common case
      across the tropics. Sites span the high-potential maize belt of the
      Rift Valley through to the semi-arid, drought-prone southeast.
"""
from __future__ import annotations

#: (name, country, region, approx_lat, approx_lon, system, notes)
#: `system` selects the mask: arable | pasture | horticulture.
LOCALITIES = [
    # ---------------- Kenya: high-potential maize/wheat belt ----------------
    ("TransNzoia",  "Kenya", "Trans Nzoia",  1.020, 34.950, "arable", "maize breadbasket, long rains dominant"),
    ("UasinGishu",  "Kenya", "Uasin Gishu",  0.520, 35.270, "arable", "maize and wheat, high potential"),
    ("Bungoma",     "Kenya", "Bungoma",      0.570, 34.560, "arable", "maize, high rainfall"),
    ("Kakamega",    "Kenya", "Kakamega",     0.280, 34.750, "arable", "maize, western highlands"),
    ("Narok",       "Kenya", "Narok",       -1.090, 35.870, "arable", "large-scale wheat and maize"),
    ("Nakuru",      "Kenya", "Nakuru",      -0.300, 36.070, "arable", "mixed cereals, Rift Valley"),
    ("Bomet",       "Kenya", "Bomet",       -0.780, 35.340, "arable", "maize, tea-growing highlands"),

    # ---------------- Kenya: semi-arid, drought-prone southeast -------------
    ("Machakos",    "Kenya", "Machakos",    -1.520, 37.260, "arable", "semi-arid, bimodal, drought-prone"),
    ("Kitui",       "Kenya", "Kitui",       -1.370, 38.010, "arable", "semi-arid, frequent short-rain failure"),
    ("Makueni",     "Kenya", "Makueni",     -1.800, 37.620, "arable", "semi-arid, high food-insecurity risk"),
    ("Embu",        "Kenya", "Embu",        -0.530, 37.450, "arable", "mid-altitude, mixed cropping"),
    ("Meru",        "Kenya", "Meru",         0.050, 37.650, "arable", "mid-altitude, bimodal"),

    # ---------------- Afghanistan: irrigated river valleys ------------------
    ("Kama",        "Afghanistan", "Nangarhar",  34.420, 70.550, "arable", "lowland irrigated"),
    ("Injil",       "Afghanistan", "Herat",      34.280, 62.150, "arable", "plains irrigated"),
    ("Dawlatabad",  "Afghanistan", "Balkh",      36.850, 66.850, "arable", "plains irrigated"),
    ("Khanabad",    "Afghanistan", "Kunduz",     36.650, 68.950, "arable", "plains irrigated"),
    ("NadAli",      "Afghanistan", "Helmand",    31.550, 64.250, "arable", "arid irrigated"),
    ("Arghandab",   "Afghanistan", "Kandahar",   31.720, 65.650, "arable", "arid irrigated"),
    ("Shakardara",  "Afghanistan", "Kabul",      34.650, 69.030, "arable", "highland irrigated"),
    ("Zaranj",      "Afghanistan", "Nimroz",     31.000, 61.870, "arable", "arid, Sistan basin"),

    # ---------------- Afghanistan: rainfed highlands ------------------------
    ("Maimana",     "Afghanistan", "Faryab",     35.920, 64.780, "arable", "plains rainfed"),
    ("GhazniCity",  "Afghanistan", "Ghazni",     33.550, 68.420, "arable", "highland rainfed"),
    ("Faizabad",    "Afghanistan", "Badakhshan", 37.100, 70.550, "arable", "highland rainfed"),
    ("Chaghcharan", "Afghanistan", "Ghor",       34.520, 65.250, "arable", "highland rainfed"),
]

FIELD_BUFFER_M = 60        # radius of the sampled area, metres
SEARCH_RADIUS_M = 15000    # how far from a locality anchor we will look
PURITY_MIN = 0.90          # required cropland share inside the buffer


def resolve_sites(localities=None, buffer_m: float = FIELD_BUFFER_M,
                  search_radius_m: float = SEARCH_RADIUS_M,
                  purity_min: float = PURITY_MIN, verbose: bool = True) -> list:
    """
    Resolve each locality anchor to a verified cropland coordinate.

    Returns a list of site dicts carrying the resolved coordinate together
    with `moved_m` and `purity`, so every downstream artefact can state on
    what basis its sampling point was chosen. Localities with no qualifying
    cropland within `search_radius_m` are omitted and reported.
    """
    from .masks import combined_crop_mask, nearest_crop_point

    localities = localities or LOCALITIES
    masks = {s: combined_crop_mask(s) for s in {loc[5] for loc in localities}}

    resolved, dropped = [], []
    for name, country, region, lat, lon, system, notes in localities:
        hit = nearest_crop_point(masks[system], lat, lon, buffer_m=buffer_m,
                                 search_radius_m=search_radius_m,
                                 purity_min=purity_min)
        if hit is None:
            dropped.append((name, country))
            if verbose:
                print(f"  DROPPED {name:13s} ({country}) - no cropland "
                      f">={purity_min:.0%} pure within {search_radius_m/1000:.0f} km")
            continue
        resolved.append({
            "name": name, "country": country, "region": region,
            "system": system, "notes": notes,
            "lat": hit["lat"], "lon": hit["lon"],
            "anchor_lat": lat, "anchor_lon": lon,
            "moved_m": hit["moved_m"], "purity": hit["purity"],
        })
        if verbose:
            print(f"  {name:13s} ({country:11s}) -> {hit['lat']:9.5f}, {hit['lon']:9.5f}  "
                  f"moved {hit['moved_m']:5d} m  purity {hit['purity']:.0%}")

    if verbose:
        print(f"\nResolved {len(resolved)}/{len(localities)} localities "
              f"({len(dropped)} dropped).")
    return resolved


def to_feature_collection(sites: list, buffer_m: float = FIELD_BUFFER_M):
    """Earth Engine FeatureCollection of the resolved sampling buffers."""
    import ee
    feats = [
        ee.Feature(
            ee.Geometry.Point([s["lon"], s["lat"]]).buffer(buffer_m),
            {"name": s["name"], "country": s["country"], "region": s["region"],
             "system": s["system"], "lat": s["lat"], "lon": s["lon"],
             "purity": s["purity"]},
        )
        for s in sites
    ]
    return ee.FeatureCollection(feats)
