"""
Tests for season detection, phenometric extraction and phase warping.

Built on synthetic curves, because only there is the right answer known.
A synthetic season with a peak placed at day 200 must produce a Peak of
Season near day 200; no amount of agreement with another satellite product
establishes that the way a constructed case does.
"""
import numpy as np
import pytest

from cropsignal.phenology import (DAYS_PER_YEAR, detect_season_start,
                                  phenometrics, to_phenological_year)
from cropsignal.warp import from_phase, to_phase


def synthetic_season(peak_doy, amplitude=0.6, base=0.15, width=60, n=23,
                     noise=0.0, seed=0):
    """One year of 16-day NDVI samples with a Gaussian growing season."""
    rng = np.random.default_rng(seed)
    doy = np.arange(n) * 16 + 8
    # Circular distance to the peak: a growing season that peaks in January
    # is high again the following December, so the curve has to wrap.
    delta = np.abs(doy - peak_doy)
    delta = np.minimum(delta, DAYS_PER_YEAR - delta)
    ndvi = base + amplitude * np.exp(-0.5 * (delta / width) ** 2)
    if noise:
        ndvi = ndvi + rng.normal(0, noise, size=n)
    return doy, ndvi


class TestSeasonStart:
    def test_northern_season_starts_in_winter(self):
        """A mid-summer peak implies a season boundary in winter."""
        doy, ndvi = synthetic_season(peak_doy=196)
        start = detect_season_start(doy, ndvi)
        # trough is opposite the peak: around day 196 +/- 182 -> near year end/start
        assert start < 60 or start > 300

    def test_southern_season_starts_in_their_winter(self):
        """
        A January peak (Southern Hemisphere) implies a mid-year boundary.

        This is the case that makes the method hemisphere-agnostic: nothing
        is told which hemisphere the site is in.
        """
        doy, ndvi = synthetic_season(peak_doy=15)
        start = detect_season_start(doy, ndvi)
        assert 120 < start < 250

    def test_flat_series_does_not_crash(self):
        doy = np.arange(23) * 16 + 8
        start = detect_season_start(doy, np.full(23, 0.4))
        assert 0 <= start <= DAYS_PER_YEAR


class TestPhenologicalYear:
    def test_observations_before_start_belong_to_previous_season(self):
        doy = np.array([10.0, 200.0, 350.0])
        year = np.array([2021, 2021, 2021])
        py, days = to_phenological_year(doy, year, season_start_doy=100.0)
        assert py.tolist() == [2020, 2021, 2021]
        assert days[0] == pytest.approx(10 + DAYS_PER_YEAR - 100)
        assert days[1] == pytest.approx(100)

    def test_days_since_start_is_non_negative(self):
        doy = np.arange(1, 366, 10).astype(float)
        year = np.full(len(doy), 2022)
        for start in (1.0, 90.0, 180.0, 300.0):
            _, days = to_phenological_year(doy, year, start)
            assert (days >= 0).all()
            assert (days < DAYS_PER_YEAR + 1).all()


class TestPhenometrics:
    def test_recovers_a_known_peak(self):
        doy, ndvi = synthetic_season(peak_doy=200)
        m = phenometrics(doy, ndvi)
        assert m is not None
        assert abs(m["pos"] - 200) < 24  # within one and a half composites

    def test_anchors_are_ordered(self):
        doy, ndvi = synthetic_season(peak_doy=180)
        m = phenometrics(doy, ndvi)
        assert m["sos"] < m["pos"] < m["eos"]

    def test_flat_series_returns_none(self):
        """A site with no season must not be given invented anchors."""
        doy = np.arange(23) * 16 + 8
        assert phenometrics(doy, np.full(23, 0.4)) is None

    def test_low_amplitude_returns_none(self):
        doy, ndvi = synthetic_season(peak_doy=200, amplitude=0.02)
        assert phenometrics(doy, ndvi) is None

    def test_too_few_points_returns_none(self):
        assert phenometrics(np.array([10.0, 50.0]), np.array([0.2, 0.8])) is None

    def test_survives_noise(self):
        doy, ndvi = synthetic_season(peak_doy=150, noise=0.03, seed=7)
        m = phenometrics(doy, ndvi)
        assert m is not None
        assert abs(m["pos"] - 150) < 40


class TestWarp:
    @pytest.fixture
    def metrics(self):
        return {"sos": 90.0, "pos": 180.0, "eos": 270.0,
                "peak_value": 0.8, "base_value": 0.2, "amplitude": 0.6,
                "n_points": 23}

    def test_anchors_map_to_canonical_phases(self, metrics):
        assert to_phase(np.array([0.0]), metrics)[0] == pytest.approx(0.0)
        assert to_phase(np.array([90.0]), metrics)[0] == pytest.approx(0.25)
        assert to_phase(np.array([180.0]), metrics)[0] == pytest.approx(0.50)
        assert to_phase(np.array([270.0]), metrics)[0] == pytest.approx(0.75)

    def test_warp_is_monotone(self, metrics):
        days = np.linspace(0, DAYS_PER_YEAR, 500)
        phase = to_phase(days, metrics)
        assert np.all(np.diff(phase) >= -1e-12)

    def test_round_trip(self, metrics):
        days = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 360.0])
        assert from_phase(to_phase(days, metrics), metrics) == pytest.approx(days, abs=1e-6)

    def test_two_offset_seasons_align_in_phase_space(self):
        """
        The property the whole method rests on: two seasons that are
        identical in shape but offset by six weeks in calendar time must
        map onto the same phase axis.
        """
        early = {"sos": 70.0, "pos": 160.0, "eos": 250.0}
        late = {"sos": 112.0, "pos": 202.0, "eos": 292.0}
        assert to_phase(np.array([160.0]), early)[0] == pytest.approx(
            to_phase(np.array([202.0]), late)[0])

    def test_degenerate_anchors_stay_monotone(self):
        """Coincident anchors must not produce a non-invertible warp."""
        bad = {"sos": 100.0, "pos": 100.0, "eos": 100.0}
        phase = to_phase(np.linspace(0, 365, 200), bad)
        assert np.all(np.diff(phase) >= -1e-12)
