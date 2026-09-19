"""Unit tests for pipeline/utils.py core functions."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow import without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.utils import (
    anomaly_threshold, chondrite_normalize, load_nure, CHONDRITE_SUN89,
    sample_raster_at_xy, load_radiometric_at_points, load_raster_window,
    classify_rad_concordance, linear_mean_2sd, radiometric_tif_path,
    RAD_AGREE_BOTH, RAD_AGREE_STREAM, RAD_AGREE_AIR, RAD_AGREE_NONE,
)


# ── anomaly_threshold ─────────────────────────────────────────────────────────

class TestAnomalyThreshold:
    def test_returns_none_for_fewer_than_five_positives(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert anomaly_threshold(s) is None

    def test_returns_none_for_all_nan(self):
        s = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
        assert anomaly_threshold(s) is None

    def test_returns_none_for_all_nonpositive(self):
        s = pd.Series([0.0, -1.0, -2.0, -3.0, -4.0, -5.0])
        assert anomaly_threshold(s) is None

    def test_correct_mean_plus_2sd_on_log_normal(self):
        # Construct a series of exactly log10-values that give a known mean+2SD
        # log10([10, 100, 1000, 10, 100]) → [1, 2, 3, 1, 2] → mean=1.8, std≈0.748
        values = pd.Series([10.0, 100.0, 1000.0, 10.0, 100.0])
        result = anomaly_threshold(values)
        log_vals = np.log10(values)
        expected = 10 ** (log_vals.mean() + 2 * log_vals.std())
        assert result == pytest.approx(expected, rel=1e-6)

    def test_ignores_nan_values(self):
        s_with_nan = pd.Series([10.0, 100.0, np.nan, 1000.0, 10.0, 100.0])
        s_clean    = pd.Series([10.0, 100.0, 1000.0, 10.0, 100.0])
        assert anomaly_threshold(s_with_nan) == pytest.approx(anomaly_threshold(s_clean), rel=1e-6)

    def test_ignores_nonpositive_values(self):
        s_with_neg = pd.Series([10.0, 100.0, 0.0, -5.0, 1000.0, 10.0, 100.0])
        s_clean    = pd.Series([10.0, 100.0, 1000.0, 10.0, 100.0])
        assert anomaly_threshold(s_with_neg) == pytest.approx(anomaly_threshold(s_clean), rel=1e-6)

    def test_is_deterministic(self):
        s = pd.Series([1.5, 3.2, 7.8, 15.1, 22.4, 44.7, 88.3, 120.0])
        assert anomaly_threshold(s) == anomaly_threshold(s)

    def test_returns_value_greater_than_most_inputs(self):
        # Threshold should exceed ~97.5% of background values
        rng = np.random.default_rng(0)
        values = pd.Series(10 ** rng.normal(1.0, 0.3, 200))
        thresh = anomaly_threshold(values)
        assert thresh is not None
        frac_below = (values < thresh).mean()
        assert frac_below > 0.90  # conservative check


# ── chondrite_normalize ───────────────────────────────────────────────────────

class TestChondriteNormalize:
    def test_correct_normalized_values(self):
        df = pd.DataFrame({'La': [0.237], 'Ce': [0.612], 'Nd': [0.934]})
        result = chondrite_normalize(df, ['La', 'Ce', 'Nd'])
        # La/0.237 = 1.0 exactly; Ce/0.612 = 1.0 exactly
        assert result['La'].iloc[0] == pytest.approx(1.0, rel=1e-6)
        assert result['Ce'].iloc[0] == pytest.approx(1.0, rel=1e-6)
        assert result['Nd'].iloc[0] == pytest.approx(0.934 / CHONDRITE_SUN89['Nd'], rel=1e-6)

    def test_missing_column_returns_nan_not_error(self):
        df = pd.DataFrame({'La': [10.0]})
        # Ce is not in df; should get NaN column, not KeyError
        result = chondrite_normalize(df, ['La', 'Ce'])
        assert result['La'].iloc[0] == pytest.approx(10.0 / CHONDRITE_SUN89['La'], rel=1e-6)
        assert pd.isna(result['Ce'].iloc[0])

    def test_does_not_modify_original_dataframe(self):
        df = pd.DataFrame({'La': [5.0], 'Ce': [12.0]})
        la_before = df['La'].iloc[0]
        chondrite_normalize(df, ['La', 'Ce'])
        assert df['La'].iloc[0] == la_before  # original unchanged

    def test_unknown_element_column_passes_through_unchanged(self):
        # Element not in CHONDRITE_SUN89 (e.g. 'Au') → value unchanged by normalization
        df = pd.DataFrame({'Au': [0.5]})
        result = chondrite_normalize(df, ['Au'])
        assert result['Au'].iloc[0] == pytest.approx(0.5, rel=1e-6)


# ── load_nure ─────────────────────────────────────────────────────────────────

class TestLoadNure:
    def _make_csv(self, rows: dict, path: str):
        pd.DataFrame(rows).to_csv(path, index=False)

    def _cfg(self, path):
        return {'data': {'nure_csv': path}}

    def test_half_mdl_substitution_for_small_negatives(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            fname = f.name
        try:
            self._make_csv({'lat': [48.0], 'lon': [-119.0], 'Th': [-5.0]}, fname)
            df = load_nure(self._cfg(fname))
            assert df['Th'].iloc[0] == pytest.approx(2.5, rel=1e-6)
        finally:
            os.unlink(fname)

    def test_large_negatives_become_nan(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            fname = f.name
        try:
            self._make_csv({'lat': [48.0], 'lon': [-119.0], 'Th': [-99.0]}, fname)
            df = load_nure(self._cfg(fname))
            assert pd.isna(df['Th'].iloc[0])
        finally:
            os.unlink(fname)

    def test_boundary_exactly_minus_ten_is_half_mdl(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            fname = f.name
        try:
            self._make_csv({'lat': [48.0], 'lon': [-119.0], 'Ce': [-10.0]}, fname)
            df = load_nure(self._cfg(fname))
            assert df['Ce'].iloc[0] == pytest.approx(5.0, rel=1e-6)
        finally:
            os.unlink(fname)

    def test_p_pct_to_ppm_conversion(self):
        # If median P < 1, should be multiplied by 10000
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            fname = f.name
        try:
            self._make_csv({
                'lat': [48.0, 48.1], 'lon': [-119.0, -119.1],
                'P': [0.04, 0.06],  # median 0.05 < 1 → pct units
            }, fname)
            df = load_nure(self._cfg(fname))
            assert df['P'].iloc[0] == pytest.approx(400.0, rel=1e-4)
        finally:
            os.unlink(fname)

    def test_positive_values_unchanged(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            fname = f.name
        try:
            self._make_csv({'lat': [48.0], 'lon': [-119.0], 'Th': [25.3]}, fname)
            df = load_nure(self._cfg(fname))
            assert df['Th'].iloc[0] == pytest.approx(25.3, rel=1e-6)
        finally:
            os.unlink(fname)

    def test_coord_columns_not_modified(self):
        # lat/lon should never be half-MDL substituted even if negative
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            fname = f.name
        try:
            self._make_csv({'lat': [-5.0], 'lon': [-119.0], 'Th': [10.0]}, fname)
            df = load_nure(self._cfg(fname))
            assert df['lat'].iloc[0] == pytest.approx(-5.0, rel=1e-6)
        finally:
            os.unlink(fname)


def _write_tiny_tif(path, origin_x, origin_y, px, values, nodata=None):
    """Write a north-up single-band float32 GeoTIFF. origin is the NW corner."""
    import rasterio
    from rasterio.transform import from_origin

    arr = np.asarray(values, dtype=np.float32)
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': from_origin(origin_x, origin_y, px, px),
    }
    if nodata is not None:
        profile['nodata'] = nodata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


class TestRadiometricSampling:
    def test_missing_raster_returns_nan(self):
        vals = sample_raster_at_xy('/no/such/file.tif', [-118.2], [48.1])
        assert len(vals) == 1
        assert np.isnan(vals[0])

    def test_empty_config_returns_nan_columns(self):
        df = load_radiometric_at_points({'data': {}}, [-118.2, -117.9], [48.1, 48.5])
        assert list(df.columns) == ['rad_K', 'rad_eTh', 'rad_eU', 'rad_eU_eTh', 'rad_K_eTh']
        assert df.isna().all().all()

    def test_samples_known_pixel(self, tmp_path):
        # 2×2 grid, 1° pixels, origin NW (-120, 49). Pixel centers:
        # (-119.5, 48.5)=10, (-118.5, 48.5)=20
        # (-119.5, 47.5)=30, (-118.5, 47.5)=40
        tif = tmp_path / 'eth.tif'
        _write_tiny_tif(tif, -120.0, 49.0, 1.0, [[10, 20], [30, 40]])
        vals = sample_raster_at_xy(str(tif), [-119.5, -118.5], [48.5, 47.5])
        assert vals[0] == pytest.approx(10.0)
        assert vals[1] == pytest.approx(40.0)

    def test_nodata_becomes_nan(self, tmp_path):
        tif = tmp_path / 'eth.tif'
        _write_tiny_tif(tif, -120.0, 49.0, 1.0, [[-9999, 12]], nodata=-9999)
        vals = sample_raster_at_xy(str(tif), [-119.5, -118.5], [48.5, 48.5])
        assert np.isnan(vals[0])
        assert vals[1] == pytest.approx(12.0)

    def test_ratio_columns_when_tiffs_present(self, tmp_path):
        k = tmp_path / 'k.tif'
        th = tmp_path / 'th.tif'
        u = tmp_path / 'u.tif'
        _write_tiny_tif(k, -120.0, 49.0, 1.0, [[2.0, 2.0], [2.0, 2.0]])
        _write_tiny_tif(th, -120.0, 49.0, 1.0, [[10.0, 10.0], [10.0, 10.0]])
        _write_tiny_tif(u, -120.0, 49.0, 1.0, [[4.0, 4.0], [4.0, 4.0]])
        cfg = {'data': {
            'radiometric_k_tif': str(k),
            'radiometric_eth_tif': str(th),
            'radiometric_eu_tif': str(u),
        }}
        df = load_radiometric_at_points(cfg, [-119.5], [48.5])
        assert df['rad_K'].iloc[0] == pytest.approx(2.0)
        assert df['rad_eTh'].iloc[0] == pytest.approx(10.0)
        assert df['rad_eU'].iloc[0] == pytest.approx(4.0)
        assert df['rad_eU_eTh'].iloc[0] == pytest.approx(0.4)
        assert df['rad_K_eTh'].iloc[0] == pytest.approx(0.2)

    def test_missing_path_key_is_none(self):
        assert radiometric_tif_path({'data': {}}, 'eth') is None
        assert radiometric_tif_path({'data': {'radiometric_eth_tif': '/no/file.tif'}}, 'eth') is None


class TestRasterWindow:
    def test_missing_returns_none(self):
        data, extent = load_raster_window('/no/such.tif', (-120, -117, 47.5, 49.1))
        assert data is None and extent is None

    def test_window_stays_inside_requested_bounds(self, tmp_path):
        tif = tmp_path / 'eth.tif'
        # 10×10, 0.2° pixels covering -120→-118, 47.2→49.2
        grid = np.arange(100, dtype=float).reshape(10, 10)
        _write_tiny_tif(tif, -120.0, 49.2, 0.2, grid)
        data, extent = load_raster_window(str(tif), (-119.6, -118.6, 47.8, 48.8))
        assert data is not None
        assert extent[0] >= -119.6 - 0.2  # at most one pixel slack
        assert extent[1] <= -118.6 + 0.2
        assert extent[2] >= 47.8 - 0.2
        assert extent[3] <= 48.8 + 0.2
        assert data.size < 100  # actually clipped


class TestLinearMean2sd:
    def test_none_if_too_few(self):
        assert linear_mean_2sd([1.0, 2.0, np.nan]) is None

    def test_matches_formula(self):
        v = np.array([10.0, 12.0, 11.0, 9.0, 13.0, 10.5])
        assert linear_mean_2sd(v) == pytest.approx(v.mean() + 2 * v.std())

    def test_drops_nonpositive(self):
        v = np.array([10.0, 12.0, 0.0, -5.0, 11.0, 9.0, 13.0])
        pos = np.array([10.0, 12.0, 11.0, 9.0, 13.0])
        assert linear_mean_2sd(v) == pytest.approx(pos.mean() + 2 * pos.std())


class TestRadConcordance:
    def test_four_classes(self):
        stream = [True, True, False, False]
        air = [True, False, True, False]
        out = classify_rad_concordance(stream, air)
        assert list(out) == [RAD_AGREE_BOTH, RAD_AGREE_STREAM, RAD_AGREE_AIR, RAD_AGREE_NONE]

    def test_missing_airborne_is_na(self):
        stream = [True, False]
        air = [np.nan, np.nan]
        out = classify_rad_concordance(stream, air)
        assert out.isna().all()

    def test_does_not_invent_stream_only_when_unsampled(self):
        # A stream-Th site with no airborne sample must not be called stream_only
        out = classify_rad_concordance([True], [np.nan])
        assert pd.isna(out.iloc[0])
