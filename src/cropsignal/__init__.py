"""
PhenoShift - phenology-aligned anomaly detection for satellite vegetation
time series.

The problem it addresses: almost every operational vegetation-anomaly
method compares an observation to the same *calendar date* in previous
years. That assumes the growing season happens at the same time every
year. In strongly seasonal, irrigated systems it roughly does. In
temperate and maritime agriculture it does not - sowing dates, green-up
and harvest move by weeks between years - and a calendar comparison then
reports a shifted season as if it were a damaged crop.

PhenoShift warps each season onto a common phenological phase axis first
(season start / green-up / peak / senescence), then scores anomalies at
matching phase rather than matching date.

    from phenoshift import PhenoShift

    ps = PhenoShift(site_col="name", time_col="period_start", value_col="NDVI")
    out = ps.fit_transform(df)
    out[["name", "period_start", "phase", "z", "anomaly"]]

`CalendarAnomaly` implements the conventional approach with an identical
API so the two can be compared directly on the same data.
"""
from .anomaly import CalendarAnomaly, PhenoShift
from .phenology import detect_season_start, phenometrics, to_phenological_year
from .warp import from_phase, to_phase
from .validate import (build_weather_stress_index, phenology_contamination,
                       weather_association, yield_association)

__version__ = "0.1.0"

__all__ = [
    "PhenoShift",
    "CalendarAnomaly",
    "detect_season_start",
    "phenometrics",
    "to_phenological_year",
    "to_phase",
    "from_phase",
    "weather_association",
    "phenology_contamination",
    "yield_association",
    "build_weather_stress_index",
]
