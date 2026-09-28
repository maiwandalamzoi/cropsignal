"""
Regression tests for bugs that were actually found in this pipeline.

Each test here corresponds to a defect that shipped, produced clean-looking
output, and was caught only by an audit. They are kept as tests so that the
same class of failure cannot return quietly.
"""
import numpy as np
import pandas as pd
import pytest

from cropsignal.anomaly import CalendarAnomaly, PhenoShift
from cropsignal.indices import (ORBIT_SUFFIX, RADAR_BANDS, band_list,
                                select_dominant_orbit)


class TestOrbitSuffix:
    """
    Bug: band names were derived with `orbit[:4].lower()`, which turns
    'DESCENDING' into 'desc' correctly but 'ASCENDING' into 'asce'. Every
    ascending band was then missing under the name the extractor asked for,
    and the whole period failed - including the sensors that had data.
    """

    def test_suffixes_are_exact(self):
        assert ORBIT_SUFFIX["ASCENDING"] == "asc"
        assert ORBIT_SUFFIX["DESCENDING"] == "desc"

    def test_declared_radar_bands_match_suffixes(self):
        for suffix in ORBIT_SUFFIX.values():
            for base in ("VV", "VH", "VH_VV"):
                assert f"{base}_{suffix}" in RADAR_BANDS

    def test_band_list_matches_declared_bands(self):
        bands = band_list(("optical", "radar", "biophysical", "water", "thermal"))
        assert len(bands) == len(set(bands)), "duplicate band names"
        for b in RADAR_BANDS:
            assert b in bands


class TestOrbitSelection:
    """
    Bug: the extractor hardcoded the descending orbit. Kenya is acquired on
    ascending passes only, so every Kenyan site returned no radar at all
    while the pipeline reported success.
    """

    def test_picks_the_orbit_with_more_observations(self):
        df = pd.DataFrame({
            "name": ["Kenya1"] * 4,
            "VV_asc": [-10.0, -11.0, -10.5, -9.0],
            "VH_asc": [-17.0, -18.0, -17.5, -16.0],
            "VH_VV_asc": [-7.0, -7.0, -7.0, -7.0],
            "VV_desc": [np.nan] * 4,
            "VH_desc": [np.nan] * 4,
            "VH_VV_desc": [np.nan] * 4,
        })
        out = select_dominant_orbit(df)
        assert (out["orbit"] == "asc").all()
        assert out["VV"].notna().all()

    def test_one_orbit_per_site_not_per_row(self):
        """
        Taking whichever orbit was available each period would reintroduce
        the geometry steps that splitting the orbits exists to prevent.
        """
        df = pd.DataFrame({
            "name": ["S"] * 4,
            "VV_asc": [-10.0, np.nan, -10.5, -9.0],
            "VH_asc": [-17.0, np.nan, -17.5, -16.0],
            "VH_VV_asc": [-7.0, np.nan, -7.0, -7.0],
            "VV_desc": [np.nan, -14.0, np.nan, np.nan],
            "VH_desc": [np.nan, -21.0, np.nan, np.nan],
            "VH_VV_desc": [np.nan, -7.0, np.nan, np.nan],
        })
        out = select_dominant_orbit(df)
        assert out["orbit"].nunique() == 1
        # the single descending observation must NOT be spliced in
        assert out["VV"].isna().sum() == 1

    def test_sites_choose_independently(self):
        df = pd.DataFrame({
            "name": ["A", "A", "B", "B"],
            "VV_asc": [-10.0, -11.0, np.nan, np.nan],
            "VH_asc": [-17.0, -18.0, np.nan, np.nan],
            "VH_VV_asc": [-7.0, -7.0, np.nan, np.nan],
            "VV_desc": [np.nan, np.nan, -13.0, -12.0],
            "VH_desc": [np.nan, np.nan, -20.0, -19.0],
            "VH_VV_desc": [np.nan, np.nan, -7.0, -7.0],
        })
        out = select_dominant_orbit(df)
        assert out.loc[out.name == "A", "orbit"].iloc[0] == "asc"
        assert out.loc[out.name == "B", "orbit"].iloc[0] == "desc"
        assert out["VV"].notna().all()


class TestAnomalyIndexAlignment:
    """
    Bug class: detectors wrote scores back by positional index while
    grouping by site, so values could be attached to the wrong rows when
    the input index was not a clean range. Both detectors reset the index
    on entry; these tests pin that down.
    """

    @staticmethod
    def _series(n_years=8, n_per_year=23, peak=180, sites=("A", "B"),
                noise=0.02, seed=0, failed_years=(), failed_amplitude=0.25):
        """
        A realistic multi-year NDVI record.

        Noise is not decoration here. Without it every baseline year is
        identical, the baseline spread is zero, and a z-score is undefined
        by construction - so a test on noiseless data passes whatever the
        detector does. `noise` at 0.02 NDVI is the order of residual
        atmospheric and BRDF variation in a real Sentinel-2 composite.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for site in sites:
            for y in range(2019, 2019 + n_years):
                amplitude = failed_amplitude if y in failed_years else 0.6
                # each season also starts a few days early or late, as real ones do
                shift = rng.normal(0, 8)
                for p in range(n_per_year):
                    doy = p * 16 + 8
                    delta = abs(doy - (peak + shift))
                    delta = min(delta, 365.25 - delta)
                    ndvi = (0.15 + amplitude * np.exp(-0.5 * (delta / 60) ** 2)
                            + rng.normal(0, noise))
                    rows.append({"site": site,
                                 "date": pd.Timestamp(year=y, month=1, day=1)
                                         + pd.Timedelta(days=doy - 1),
                                 "NDVI": ndvi})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("detector", [CalendarAnomaly, PhenoShift])
    def test_non_range_index_is_handled(self, detector):
        df = self._series()
        shuffled = df.sample(frac=1.0, random_state=0)  # non-monotonic index
        out = detector(site_col="site", time_col="date",
                       value_col="NDVI").fit_transform(shuffled)
        assert len(out) == len(df)
        assert "z" in out.columns

    @pytest.mark.parametrize("detector", [CalendarAnomaly, PhenoShift])
    def test_false_positive_rate_is_bounded(self, detector):
        """
        On a record of ordinary seasons - noisy, and starting a few days
        early or late, as real ones do - a detector should flag few
        observations. The z <= -1 threshold implies roughly 16% under a
        normal baseline, so materially more than that means the detector is
        reporting normal variation as stress.
        """
        out = detector(site_col="site", time_col="date",
                       value_col="NDVI").fit_transform(self._series(seed=3))
        scored = out["anomaly"].notna().sum()
        assert scored > 0, "detector scored nothing at all"
        rate = out["anomaly"].fillna(0).sum() / scored
        assert rate < 0.25, f"false-positive rate {rate:.0%} on ordinary seasons"

    @pytest.mark.parametrize("detector", [CalendarAnomaly, PhenoShift])
    def test_detects_a_real_failure(self, detector):
        """
        The complement of the test above, and the one that stops a detector
        from passing by simply never flagging anything: a season at 40% of
        normal amplitude is a crop failure and must be flagged.
        """
        df = self._series(seed=5, failed_years=(2023,))
        out = detector(site_col="site", time_col="date",
                       value_col="NDVI").fit_transform(df)
        out["yr"] = pd.to_datetime(out["date"]).dt.year

        failed = out[out["yr"] == 2023]["anomaly"].fillna(0).mean()
        normal = out[out["yr"] != 2023]["anomaly"].fillna(0).mean()
        assert failed > 0.25, f"only {failed:.0%} of the failed season flagged"
        assert failed > normal * 2, (
            f"failed season ({failed:.0%}) not distinguished from "
            f"normal seasons ({normal:.0%})")


class TestEVIRange:
    """
    Bug: EVI was computed without clamping. Its denominator approaches zero
    over water and wet bare soil, and the pipeline shipped EVI values from
    -2.55 to 3.84 straight into a gradient-boosting model. Valid EVI is
    within [-1, 1].
    """

    def test_clamp_bounds_are_declared(self):
        import inspect

        from cropsignal import indices
        src = inspect.getsource(indices._s2_indices)
        assert ".clamp(-1, 1)" in src, "EVI must be clamped to its physical range"


class TestEmptyCollectionSafety:
    """
    Bug: several composites called `.median()` on a collection that could be
    empty, and then selected bands from the result. The median of an empty
    Earth Engine collection is an image with *no bands*, so the select threw
    and the whole compositing period was abandoned - including the sensors
    that had data.

    This cost the project three separate outages, all silent:
      - Sentinel-1 descending orbit, which does not cover Kenya at all;
      - MOD16A2 evapotranspiration, whose Earth Engine archive begins in
        2021, so every period in 2019-2020 failed;
      - Sentinel-2 scenes with no matching Cloud Score+ image.

    None raised an error the pipeline reported. Each produced a CSV that was
    simply missing rows.
    """

    def test_safe_median_is_used_by_every_composite(self):
        import inspect

        from cropsignal import indices

        for fn in (indices.s2_composite, indices._s1_orbit_composite,
                   indices.modis_biophysical, indices.modis_water,
                   indices.modis_thermal):
            src = inspect.getsource(fn)
            assert "_safe_median" in src, (
                f"{fn.__name__} takes a median without the empty-collection "
                f"guard; an archive gap there silently drops whole periods")

    def test_cloud_score_join_is_inner(self):
        """
        `saveFirst` emits unmatched primaries with a null score, which raises
        when read. Only an inner join can guarantee every image has a score.
        """
        import inspect

        from cropsignal import indices
        src = inspect.getsource(indices.s2_composite)
        assert "ee.Join.inner" in src
        # the call, not the docstring that explains why it is not used
        assert "ee.Join.saveFirst(" not in src

    def test_composites_accept_a_region(self):
        """
        Without a region filter the collection is the entire global archive
        for the period, so one malformed scene anywhere aborts the composite
        for every site being sampled.
        """
        import inspect

        for fn_name in ("s2_composite", "s1_composite", "build_stack"):
            from cropsignal import indices
            sig = inspect.signature(getattr(indices, fn_name))
            assert "region" in sig.parameters, f"{fn_name} cannot be region-filtered"


class TestSearchScaling:
    """
    Bug: the cropland search reduced a 15 km disc at 20 m - about 1.8 billion
    pixels - with `bestEffort=True`. Earth Engine responded by silently
    coarsening the scale until the request fit, and at the coarser scale the
    high-purity pixels the search exists to find were averaged away. Nine of
    twelve Kenyan localities were reported as having no cropland within
    15 km, while in fact every one of them had cropland at 100% purity.
    """

    def test_search_does_not_use_best_effort(self):
        import inspect

        from cropsignal import masks
        src = inspect.getsource(masks.nearest_crop_point)
        assert "bestEffort=False" in src, (
            "bestEffort silently coarsens the scale and loses the candidates")

    def test_search_expands_in_rings(self):
        from cropsignal.masks import SEARCH_RINGS
        assert list(SEARCH_RINGS) == sorted(SEARCH_RINGS), "rings must expand"
        assert SEARCH_RINGS[0] <= 1000, "first ring should be cheap and local"
