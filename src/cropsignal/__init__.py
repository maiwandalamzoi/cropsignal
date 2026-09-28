"""
CropSignal - multi-sensor Earth observation for agricultural monitoring.

Four things, in the order you would use them:

1. `masks` / `sites` - decide *where* to sample. Sampling points are derived
   from ESA WorldCereal and WorldCover rather than typed in, and each carries
   the distance it moved from its locality anchor and the cropland purity it
   achieved. This exists because an earlier version of this work hand-picked
   36 coordinates and 15 of them turned out to be towns, a river, and bare
   ground - with no error raised anywhere in the pipeline.

2. `indices` / `extract` - pull the variables. Sixteen of them across five
   sensors: Sentinel-2 optical indices, Sentinel-1 radar backscatter, MODIS
   LAI/FPAR, evapotranspiration and land surface temperature. Each sensor
   fails independently, so an archive gap or an orbit that does not cover a
   country yields missing values for that variable alone.

3. `drought` - VCI, TCI and VHI as published (Kogan 1990, 1995) with the
   NOAA/STAR operational thresholds, computed at 10 m cropland resolution
   rather than the 1 km at which VHI is usually produced.

4. `phenology` / `warp` / `anomaly` - detect anomalies at matching
   phenological *phase* rather than matching calendar date, and compare that
   against the conventional calendar approach on identical data.

`validate` scores the resulting labels against evidence the vegetation
indices cannot contain - evapotranspiration and thermal data - because two
NDVI-derived labels compared to each other establish nothing.

    from cropsignal import resolve_sites, add_drought_indices, PhenoShift

    sites = resolve_sites()                       # verified cropland coordinates
    out = add_drought_indices(df)                 # VCI / TCI / VHI / ESI
    flags = PhenoShift(site_col="name", time_col="period_start",
                       value_col="NDVI").fit_transform(df)
"""
from .anomaly import CalendarAnomaly, PhenoShift
from .drought import (add_drought_indices, classify, season_summary,
                      temperature_condition_index, vegetation_condition_index,
                      vegetation_health_index)
from .indices import ALL_BANDS, band_list, build_stack, select_dominant_orbit
from .masks import (combined_crop_mask, describe_cover, nearest_crop_point,
                    purity_at, sample_crop_points, worldcereal, worldcover)
from .phenology import detect_season_start, phenometrics, to_phenological_year
from .sites import LOCALITIES, resolve_sites, to_feature_collection
from .validate import (phenology_contamination, weather_association,
                       yield_association)
from .warp import from_phase, to_phase

__version__ = "0.1.0"

__all__ = [
    # where to sample
    "LOCALITIES", "resolve_sites", "to_feature_collection",
    "combined_crop_mask", "worldcereal", "worldcover", "nearest_crop_point",
    "sample_crop_points", "purity_at", "describe_cover",
    # what to pull
    "build_stack", "band_list", "ALL_BANDS", "select_dominant_orbit",
    # drought
    "add_drought_indices", "vegetation_condition_index",
    "temperature_condition_index", "vegetation_health_index",
    "season_summary", "classify",
    # phenology and anomalies
    "detect_season_start", "phenometrics", "to_phenological_year",
    "to_phase", "from_phase", "PhenoShift", "CalendarAnomaly",
    # validation
    "weather_association", "phenology_contamination", "yield_association",
]
