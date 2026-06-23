"""Unit tests for pipeline/utils.py core functions."""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Allow import without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.utils import anomaly_threshold, chondrite_normalize, load_nure, CHONDRITE_SUN89


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
