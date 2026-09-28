"""
Tests for the drought indices.

These check the published definitions (Kogan 1990, 1995) and the guards
that keep an index from returning a confident number it has not earned.
"""
import numpy as np
import pandas as pd
import pytest

from cropsignal.drought import (SEVERE_THRESHOLD, STRESS_THRESHOLD,
                                add_drought_indices, classify,
                                season_summary,
                                temperature_condition_index,
                                vegetation_condition_index,
                                vegetation_health_index)


def frame(ndvi, lst=None, site="A", poy=0):
    n = len(ndvi)
    return pd.DataFrame({
        "site": [site] * n,
        "country": ["X"] * n,
        "period_of_year": [poy] * n,
        "season_year": list(range(2019, 2019 + n)),
        "NDVI": ndvi,
        "LST_day": lst if lst is not None else [np.nan] * n,
        "ESI": [np.nan] * n,
    })


class TestVCI:
    def test_endpoints(self):
        """The historical minimum scores 0 and the maximum 100, by definition."""
        df = frame([0.2, 0.4, 0.6, 0.8, 1.0])
        vci = vegetation_condition_index(df, min_years=5)
        assert vci.iloc[0] == pytest.approx(0)
        assert vci.iloc[-1] == pytest.approx(100)

    def test_midpoint(self):
        df = frame([0.0, 0.25, 0.5, 0.75, 1.0])
        vci = vegetation_condition_index(df, min_years=5)
        assert vci.iloc[2] == pytest.approx(50)

    def test_short_record_returns_nan(self):
        """
        Three years cannot define a min/max range worth scoring against.
        Returning NaN is the point: a number here would look authoritative
        and be built on nothing.
        """
        df = frame([0.2, 0.5, 0.9])
        assert vegetation_condition_index(df, min_years=5).isna().all()

    def test_constant_series_returns_nan(self):
        df = frame([0.5] * 6)
        assert vegetation_condition_index(df, min_years=5).isna().all()

    def test_sites_scored_independently(self):
        """A site is compared to its own history, never to another site's."""
        a = frame([0.1, 0.2, 0.3, 0.4, 0.5], site="A")
        b = frame([0.6, 0.7, 0.8, 0.9, 1.0], site="B")
        vci = vegetation_condition_index(pd.concat([a, b], ignore_index=True),
                                         min_years=5)
        assert vci.iloc[0] == pytest.approx(0)    # worst year at A
        assert vci.iloc[5] == pytest.approx(0)    # worst year at B, not global


class TestTCI:
    def test_is_inverted(self):
        """Hottest on record must score 0, coldest 100."""
        df = frame([0.5] * 5, lst=[20.0, 25.0, 30.0, 35.0, 40.0])
        tci = temperature_condition_index(df, min_years=5)
        assert tci.iloc[-1] == pytest.approx(0)   # hottest
        assert tci.iloc[0] == pytest.approx(100)  # coolest


class TestVHI:
    def test_equal_weighting(self):
        vci = pd.Series([0.0, 50.0, 100.0])
        tci = pd.Series([100.0, 50.0, 0.0])
        assert vegetation_health_index(vci, tci).tolist() == [50.0, 50.0, 50.0]

    def test_alpha_shifts_weight(self):
        vci, tci = pd.Series([0.0]), pd.Series([100.0])
        assert vegetation_health_index(vci, tci, alpha=1.0).iloc[0] == 0.0
        assert vegetation_health_index(vci, tci, alpha=0.0).iloc[0] == 100.0


class TestClassify:
    def test_noaa_thresholds(self):
        out = classify(pd.Series([5.0, 20.0, 35.0, 80.0]))
        assert list(out) == ["extreme", "severe", "moderate", "none"]

    def test_boundaries_are_inclusive(self):
        """VHI == 40 is stressed, not 'none' - the threshold is <= 40."""
        out = classify(pd.Series([float(STRESS_THRESHOLD), float(SEVERE_THRESHOLD)]))
        assert list(out) == ["moderate", "severe"]


class TestAddIndices:
    def test_vhi_is_nan_without_thermal_data(self):
        """
        Without LST there is no TCI, so there is no VHI. Substituting VCI
        would silently change which index the NOAA thresholds are applied
        to, and those thresholds are not interchangeable.
        """
        out = add_drought_indices(frame([0.2, 0.4, 0.6, 0.8, 1.0]), min_years=5)
        assert out["VCI"].notna().any()
        assert out["VHI"].isna().all()

    def test_vhi_present_with_thermal_data(self):
        out = add_drought_indices(
            frame([0.2, 0.4, 0.6, 0.8, 1.0], lst=[35.0, 32.0, 30.0, 28.0, 25.0]),
            min_years=5)
        assert out["VHI"].notna().all()
        assert (out["VHI"].between(0, 100)).all()

    def test_baseline_years_recorded(self):
        out = add_drought_indices(frame([0.2, 0.4, 0.6, 0.8, 1.0]), min_years=5)
        assert (out["baseline_years"] == 5).all()


class TestSeasonSummary:
    def test_counts_duration_not_only_depth(self):
        """
        One bad composite is noise; a run of them is a failed season. The
        summary has to distinguish the two.
        """
        df = pd.DataFrame({
            "site": ["A"] * 6,
            "country": ["X"] * 6,
            "season_year": [2021] * 6,
            "VHI": [80.0, 78.0, 35.0, 82.0, 85.0, 79.0],
            "VCI": [70.0] * 6,
        })
        brief = season_summary(df).iloc[0]
        df2 = df.copy()
        df2["VHI"] = [35.0, 30.0, 25.0, 20.0, 33.0, 38.0]
        sustained = season_summary(df2).iloc[0]

        assert brief["periods_stressed"] == 1
        assert sustained["periods_stressed"] == 6
        assert sustained["periods_severe"] == 2
        assert sustained["stressed_share"] > brief["stressed_share"]
