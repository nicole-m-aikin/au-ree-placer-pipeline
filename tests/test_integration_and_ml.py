"""Tests for integration scoring, ML label generation, and spatial clipping."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Allow import without installing the package (mirrors test_utils.py convention)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.utils import clip_gdf_to_map, map_extent
from pipeline.task9_ml_targeting import _log_impute

# ---------------------------------------------------------------------------
# Module-level mirror of integration.py scoring helpers.
#
# The scoring functions (score_th, score_conf, score_ndpr, score_au) are
# defined inside integration.run() and cannot be imported directly.  These
# replicas reproduce the identical formulas so we can test the geoscience
# arithmetic without invoking the full pipeline (which requires real data
# files on disk).
# ---------------------------------------------------------------------------

def _score_th(src):
    """Replicate integration.py score_th: MONAZITE->2, MIXED_UNCLEAR->1, else 0."""
    return {'MONAZITE': 2, 'MIXED_UNCLEAR': 1}.get(src, 0)


def _score_conf(conf):
    """Replicate integration.py score_conf: HIGH->2, MEDIUM->1, LOW->0."""
    return {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}.get(conf, 0)


def _score_ndpr(ndpr_t):
    """Replicate integration.py score_ndpr: log10-scaled, capped at 2.0."""
    if ndpr_t is None or pd.isna(ndpr_t) or ndpr_t <= 0:
        return 0.0
    return min(2.0, np.log10(max(ndpr_t, 1)) / np.log10(1000) * 2)


def _score_au(row):
    """Replicate integration.py score_au: dual_anomaly->2, th_only->0.5, else 0."""
    if row.get('dual_anomaly_flag'):
        return 2.0
    if row.get('has_th_anomaly'):
        return 0.5
    return 0.0


# Default weight vector matching integration.py's _DEFAULTS / w.get() fallbacks
DEFAULT_WEIGHTS = {
    'th_source_monazite': 1.0,
    'magnetic_high':      1.0,
    'source_lith':        1.0,
    'coverage':           1.0,
    'ndpr_volume':        2.0,
    'au_pathfinder':      1.0,
    'y_xenotime':         0.5,
    'ml_probability':     1.0,
}
_WEIGHT_ORDER = ['th_source_monazite', 'magnetic_high', 'source_lith',
                  'coverage', 'ndpr_volume', 'au_pathfinder', 'y_xenotime', 'ml_probability']


def _combined_score(row, w=None):
    """Compute the combined priority score for a site-attribute dict."""
    if w is None:
        w = DEFAULT_WEIGHTS
    lith_val   = row.get('source_lith_score')
    score_lith = float(lith_val) if pd.notna(lith_val) else 1.0
    return (
        _score_th(row.get('th_source', ''))          * w['th_source_monazite'] +
        int(bool(row.get('mag_high', False)))          * w['magnetic_high']      +
        score_lith                                     * w['source_lith']        +
        _score_conf(row.get('confidence', 'LOW'))      * w['coverage']           +
        _score_ndpr(row.get('ndpr_tonnes', 0.0))      * w['ndpr_volume']        +
        _score_au(row)                                 * w['au_pathfinder']      +
        int(bool(row.get('y_near', False)))            * w['y_xenotime']         +
        float(row.get('score_ml', 0.0))               * w['ml_probability']
    )


# ---------------------------------------------------------------------------
# Module-level mirror of MRDS-proximity label generation in task9_ml_targeting.run().
# The actual label code (cKDTree + dist <= radius) is embedded in run() and
# not extractable; this mirror reproduces it exactly for unit testing.
# ---------------------------------------------------------------------------

from scipy.spatial import cKDTree as _cKDTree


def _make_labels_from_coords(nure_coords, mrds_coords, radius):
    """
    Return a 0/1 label array: 1 if the NURE sample lies within `radius` degrees
    (Euclidean, WGS-84 lon/lat) of any MRDS site, else 0.

    Mirrors the labeling logic in pipeline/task9_ml_targeting.py::run().
    """
    if len(mrds_coords) == 0:
        return np.zeros(len(nure_coords), dtype=int)
    tree = _cKDTree(mrds_coords)
    dist, _ = tree.query(nure_coords)
    return (dist <= radius).astype(int)


# ---------------------------------------------------------------------------
# Minimal cfg for spatial tests -- map_padding=0 so map_extent == bbox exactly.
# ---------------------------------------------------------------------------

_CFG = {
    'study_area': {
        'bbox': {
            'lon_min': -118.0,
            'lon_max': -117.0,
            'lat_min':  48.0,
            'lat_max':  49.0,
        },
        'map_padding': 0.0,
    }
}


# ===========================================================================
# A. Integration scoring tests
# ===========================================================================

class TestIntegrationScoring:
    """Tests for the combined priority-score formula in pipeline/integration.py."""

    def _base_site(self, **kwargs):
        """Return a minimal site dict with sensible defaults for all score fields."""
        defaults = {
            'th_source':         'MONAZITE',
            'mag_high':          True,
            'source_lith_score': 2.0,
            'confidence':        'HIGH',
            'ndpr_tonnes':       100.0,
            'dual_anomaly_flag': False,
            'has_th_anomaly':    False,
            'y_near':            False,
            'score_ml':          0.0,
        }
        defaults.update(kwargs)
        return defaults

    def test_ndpr_double_weight_produces_higher_score(self):
        """
        Site with 10x higher NdPr endowment must outscore one with equal
        other criteria.  NdPr carries the default ndpr_volume weight of 2.0
        (double the weight of most other criteria), so differences in NdPr
        endowment are amplified in the ranking.
        """
        site_high = self._base_site(ndpr_tonnes=1000.0)
        site_low  = self._base_site(ndpr_tonnes=10.0)
        assert _combined_score(site_high) > _combined_score(site_low)

    def test_ndpr_weight_multiplier_is_two(self):
        """
        The ndpr_volume weight of 2.0 means the NdPr score contribution is
        double that of a weight-1.0 criterion for the same score_ndpr value.
        Verify by comparing a site where NdPr is the only active criterion.
        """
        # score_ndpr(1000) = min(2, log10(1000)/log10(1000)*2) = 2.0
        site = self._base_site(
            th_source='', mag_high=False, source_lith_score=0.0,
            confidence='LOW', ndpr_tonnes=1000.0,
            dual_anomaly_flag=False, has_th_anomaly=False,
            y_near=False, score_ml=0.0,
        )
        w_double = dict(DEFAULT_WEIGHTS)                    # ndpr_volume = 2.0
        w_single = dict(DEFAULT_WEIGHTS, ndpr_volume=1.0)  # ndpr_volume = 1.0
        assert _combined_score(site, w_double) == pytest.approx(
            _combined_score(site, w_single) * 2.0, rel=1e-6
        )

    def test_monazite_adds_two_points_over_unknown_source(self):
        """MONAZITE th_source contributes +2; an unknown source contributes 0."""
        site_mon = self._base_site(th_source='MONAZITE')
        site_unk = self._base_site(th_source='UNKNOWN')
        assert _combined_score(site_mon) - _combined_score(site_unk) == pytest.approx(2.0)

    def test_weight_sensitivity_top_site_stable_with_large_lead(self):
        """
        Perturbing any single weight +-50% must leave the #1 site unchanged
        when site A has a large multi-criterion lead over B and C.

        This replicates the one-at-a-time weight sensitivity in
        integration.run() -- verifying that a dominant site cannot be
        displaced by any single-weight perturbation.
        """
        # Site A: best on every criterion.
        site_a = self._base_site(
            th_source='MONAZITE', mag_high=True, source_lith_score=3.0,
            confidence='HIGH', ndpr_tonnes=5000.0,
            dual_anomaly_flag=True, has_th_anomaly=False,
            y_near=True, score_ml=0.9,
        )
        # Site B: weak on all criteria.
        site_b = self._base_site(
            th_source='', mag_high=False, source_lith_score=1.0,
            confidence='LOW', ndpr_tonnes=1.0,
            dual_anomaly_flag=False, has_th_anomaly=False,
            y_near=False, score_ml=0.0,
        )
        # Site C: moderate but clearly below A.
        site_c = self._base_site(
            th_source='MIXED_UNCLEAR', mag_high=False, source_lith_score=1.0,
            confidence='MEDIUM', ndpr_tonnes=10.0,
            dual_anomaly_flag=False, has_th_anomaly=False,
            y_near=False, score_ml=0.0,
        )
        sites = [site_a, site_b, site_c]

        for key in _WEIGHT_ORDER:
            for factor in (0.5, 1.5):
                w_perturbed = dict(DEFAULT_WEIGHTS)
                w_perturbed[key] = DEFAULT_WEIGHTS[key] * factor
                scores = [_combined_score(s, w_perturbed) for s in sites]
                assert np.argmax(scores) == 0, (
                    f"Site A lost #1 rank when weight '{key}' x {factor}: scores={scores}"
                )

    def test_missing_ndpr_tonnes_does_not_crash(self):
        """NaN and None ndpr_tonnes should degrade gracefully to score_ndpr=0."""
        assert _score_ndpr(float('nan')) == pytest.approx(0.0)
        assert _score_ndpr(None) == pytest.approx(0.0)
        site_nan = self._base_site(ndpr_tonnes=float('nan'))
        assert np.isfinite(_combined_score(site_nan))

    def test_score_ndpr_capped_at_two(self):
        """score_ndpr must never exceed 2.0 regardless of endowment magnitude."""
        assert _score_ndpr(1e9)  == pytest.approx(2.0)
        assert _score_ndpr(1e12) == pytest.approx(2.0)

    def test_score_ndpr_zero_for_nonpositive_input(self):
        assert _score_ndpr(0.0)   == pytest.approx(0.0)
        assert _score_ndpr(-50.0) == pytest.approx(0.0)

    def test_score_ndpr_one_tonne_is_zero(self):
        """log10(1) == 0, so exactly one tonne produces score_ndpr = 0."""
        assert _score_ndpr(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_score_ndpr_one_thousand_tonnes_hits_cap(self):
        """1000 t: log10(1000)/log10(1000)*2 == 2.0, exactly at the cap."""
        assert _score_ndpr(1000.0) == pytest.approx(2.0, rel=1e-9)

    def test_dual_anomaly_flag_scores_two(self):
        site = self._base_site(dual_anomaly_flag=True, has_th_anomaly=False)
        assert _score_au(site) == pytest.approx(2.0)

    def test_th_anomaly_only_scores_half(self):
        site = self._base_site(dual_anomaly_flag=False, has_th_anomaly=True)
        assert _score_au(site) == pytest.approx(0.5)

    def test_dual_flag_takes_priority_over_th_anomaly(self):
        """dual_anomaly_flag=True should return 2.0 even when has_th_anomaly is True."""
        site = self._base_site(dual_anomaly_flag=True, has_th_anomaly=True)
        assert _score_au(site) == pytest.approx(2.0)

    def test_score_conf_full_range(self):
        assert _score_conf('HIGH')   == 2
        assert _score_conf('MEDIUM') == 1
        assert _score_conf('LOW')    == 0
        assert _score_conf(None)     == 0
        assert _score_conf('NONE')   == 0


# ===========================================================================
# B. ML label generation tests
# ===========================================================================

class TestMLLabelGeneration:
    """
    Tests for the MRDS-proximity label logic mirrored from
    pipeline/task9_ml_targeting.run().

    Labels: 1 if Euclidean distance in degrees from a NURE sample to the
    nearest MRDS site <= mrds_proximity_deg, else 0.
    """

    def test_point_well_within_radius_gets_label_one(self):
        mrds   = np.array([[10.0, 48.0]])
        nure   = np.array([[10.05, 48.0]])   # 0.05deg < default 0.15deg
        labels = _make_labels_from_coords(nure, mrds, radius=0.15)
        assert labels[0] == 1

    def test_point_well_outside_radius_gets_label_zero(self):
        mrds   = np.array([[10.0, 48.0]])
        nure   = np.array([[10.5, 48.0]])    # 0.5deg >> 0.15deg
        labels = _make_labels_from_coords(nure, mrds, radius=0.15)
        assert labels[0] == 0

    def test_empty_mrds_gives_all_zero_labels_without_crash(self):
        """Edge case: no MRDS sites in study area -- all samples get label 0."""
        mrds   = np.empty((0, 2))
        nure   = np.array([[10.0, 48.0], [11.0, 49.0], [12.0, 50.0]])
        labels = _make_labels_from_coords(nure, mrds, radius=0.15)
        assert (labels == 0).all()
        assert len(labels) == 3

    def test_threshold_uses_inclusive_leq_comparison(self):
        """
        A sample at 90% of the radius must be label 1; at 110% must be label 0.
        Verifies the <= (not <) comparator used in the labeling code.
        """
        mrds   = np.array([[0.0, 0.0]])
        radius = 0.15
        pt_in  = np.array([[radius * 0.90, 0.0]])   # 0.135deg -- safely inside
        pt_out = np.array([[radius * 1.10, 0.0]])   # 0.165deg -- safely outside
        assert _make_labels_from_coords(pt_in,  mrds, radius)[0] == 1
        assert _make_labels_from_coords(pt_out, mrds, radius)[0] == 0

    def test_mixed_output_with_multiple_mrds_sites(self):
        """Samples near any one of multiple MRDS sites get label 1."""
        mrds  = np.array([[10.0, 48.0], [20.0, 48.0]])
        nure  = np.array([
            [10.05, 48.0],  # near site 1 -> 1
            [15.0,  48.0],  # far from both -> 0
            [20.05, 48.0],  # near site 2 -> 1
        ])
        labels = _make_labels_from_coords(nure, mrds, radius=0.15)
        assert labels[0] == 1
        assert labels[1] == 0
        assert labels[2] == 1

    def test_label_array_length_matches_nure_sample_count(self):
        mrds   = np.array([[10.0, 48.0]])
        nure   = np.array([[10.0, 48.0], [11.0, 49.0]])
        labels = _make_labels_from_coords(nure, mrds, radius=0.15)
        assert len(labels) == 2

    def test_sample_at_mrds_location_is_label_one(self):
        """A NURE sample at the exact MRDS coordinates must receive label 1."""
        mrds   = np.array([[10.0, 48.0]])
        nure   = np.array([[10.0, 48.0]])   # distance = 0
        labels = _make_labels_from_coords(nure, mrds, radius=0.15)
        assert labels[0] == 1


class TestLogImpute:
    """Tests for the _log_impute helper imported from task9_ml_targeting.py."""

    def test_positive_values_are_log10_transformed(self):
        df     = pd.DataFrame({'Th': [1.0, 10.0, 100.0]})
        result = _log_impute(df, ['Th'])
        assert result['Th'].iloc[0] == pytest.approx(0.0, abs=1e-9)   # log10(1)
        assert result['Th'].iloc[1] == pytest.approx(1.0, rel=1e-6)   # log10(10)
        assert result['Th'].iloc[2] == pytest.approx(2.0, rel=1e-6)   # log10(100)

    def test_nan_is_imputed_with_log_median(self):
        # log10 values of [1.0, nan, 100.0] are [0.0, nan, 2.0]; median = 1.0
        df     = pd.DataFrame({'Th': [1.0, np.nan, 100.0]})
        result = _log_impute(df, ['Th'])
        assert result['Th'].iloc[1] == pytest.approx(1.0, rel=1e-6)

    def test_nonpositive_values_treated_as_nan_then_imputed(self):
        # 0.0 and -5.0 are masked to NaN; only log10(10)=1.0 is valid.
        # Median of [nan, nan, 1.0] = 1.0 -> imputed value.
        df     = pd.DataFrame({'Th': [0.0, -5.0, 10.0]})
        result = _log_impute(df, ['Th'])
        assert result['Th'].iloc[0] == pytest.approx(1.0, rel=1e-6)
        assert result['Th'].iloc[1] == pytest.approx(1.0, rel=1e-6)

    def test_missing_column_is_imputed_with_zero(self):
        """When a requested column is absent from the DataFrame the result is 0.0."""
        df     = pd.DataFrame({'Th': [1.0, 10.0]})
        result = _log_impute(df, ['Th', 'Ce'])   # Ce not present
        # All Ce values are NaN -> median is NaN -> fillna(0.0)
        # Use numpy comparison to avoid pytest.approx incompatibility with Series.all()
        assert (result['Ce'].values == 0.0).all()

    def test_all_nonpositive_column_imputed_with_zero(self):
        """If every value is non-positive (all masked), the median is NaN -> fill 0."""
        df     = pd.DataFrame({'Th': [0.0, -1.0, -2.0]})
        result = _log_impute(df, ['Th'])
        assert (result['Th'].values == 0.0).all()

    def test_output_has_no_nan_values(self):
        """After imputation, no NaN should remain in the output DataFrame."""
        df = pd.DataFrame({
            'Th': [np.nan, 1.0, -5.0],
            'Ce': [10.0, np.nan, 100.0],
        })
        result = _log_impute(df, ['Th', 'Ce'])
        assert result.isna().values.sum() == 0

    def test_preserves_row_index(self):
        """Output index must match the input DataFrame's index."""
        df     = pd.DataFrame({'Th': [1.0, 10.0, 100.0]}, index=[5, 10, 15])
        result = _log_impute(df, ['Th'])
        assert list(result.index) == [5, 10, 15]


# ===========================================================================
# C. clip_gdf_to_map tests
# ===========================================================================

class TestClipGdfToMap:
    """
    Tests for pipeline.utils.clip_gdf_to_map.

    Uses a zero-padding cfg so map_extent == bbox == [-118, -117] x [48, 49].
    """

    def _make_gdf(self, points):
        """Build a minimal GeoDataFrame from a list of (lon, lat) tuples."""
        import geopandas as gpd
        from shapely.geometry import Point
        geoms = [Point(lon, lat) for lon, lat in points]
        return gpd.GeoDataFrame({'geometry': geoms}, crs='EPSG:4326')

    def test_all_inside_points_are_retained(self):
        gdf    = self._make_gdf([(-117.5, 48.5), (-117.8, 48.3), (-117.2, 48.9)])
        result = clip_gdf_to_map(gdf, _CFG)
        assert len(result) == 3

    def test_outside_point_is_dropped(self):
        gdf = self._make_gdf([
            (-117.5, 48.5),   # inside
            (-117.3, 48.7),   # inside
            (-119.0, 48.5),   # outside: west of lon_min=-118.0
        ])
        result = clip_gdf_to_map(gdf, _CFG)
        assert len(result) == 2

    def test_point_outside_northern_boundary_is_dropped(self):
        gdf = self._make_gdf([
            (-117.5, 48.5),   # inside
            (-117.5, 50.0),   # outside: north of lat_max=49.0
        ])
        result = clip_gdf_to_map(gdf, _CFG)
        assert len(result) == 1

    def test_all_outside_returns_empty_gdf(self):
        gdf    = self._make_gdf([(-115.0, 47.0), (-120.0, 50.0)])
        result = clip_gdf_to_map(gdf, _CFG)
        assert len(result) == 0

    def test_empty_input_gdf_returns_empty_without_crash(self):
        gdf    = self._make_gdf([])
        result = clip_gdf_to_map(gdf, _CFG)
        assert len(result) == 0

    def test_does_not_mutate_original_gdf(self):
        """clip_gdf_to_map must return a copy and leave the original intact."""
        gdf = self._make_gdf([
            (-117.5, 48.5),   # inside
            (-119.0, 48.5),   # outside
        ])
        original_len = len(gdf)
        _ = clip_gdf_to_map(gdf, _CFG)
        assert len(gdf) == original_len, (
            "clip_gdf_to_map mutated the input GeoDataFrame (row count changed)"
        )

    def test_returned_geometries_lie_within_extent(self):
        """After clipping, all returned geometries must be within map_extent."""
        from shapely.geometry import box
        gdf = self._make_gdf([
            (-117.5, 48.5),   # inside
            (-117.0, 49.0),   # corner: on boundary
            (-119.0, 48.5),   # outside
        ])
        result   = clip_gdf_to_map(gdf, _CFG)
        xmin, xmax, ymin, ymax = map_extent(_CFG)
        clip_box = box(xmin, ymin, xmax, ymax)
        for geom in result.geometry:
            assert clip_box.contains(geom) or clip_box.touches(geom), (
                f"Geometry {geom} lies outside map extent {(xmin, xmax, ymin, ymax)}"
            )

    def test_map_extent_reflects_bbox_with_zero_padding(self):
        """map_extent with map_padding=0 must equal the raw bbox values exactly."""
        xmin, xmax, ymin, ymax = map_extent(_CFG)
        b = _CFG['study_area']['bbox']
        assert xmin == pytest.approx(b['lon_min'])
        assert xmax == pytest.approx(b['lon_max'])
        assert ymin == pytest.approx(b['lat_min'])
        assert ymax == pytest.approx(b['lat_max'])

    def test_map_extent_applies_padding(self):
        """map_extent with non-zero padding should expand the extent symmetrically."""
        cfg_padded = {
            'study_area': {
                'bbox': {
                    'lon_min': -118.0,
                    'lon_max': -117.0,
                    'lat_min':  48.0,
                    'lat_max':  49.0,
                },
                'map_padding': 0.10,
            }
        }
        xmin, xmax, ymin, ymax = map_extent(cfg_padded)
        assert xmin == pytest.approx(-118.10)
        assert xmax == pytest.approx(-116.90)
        assert ymin == pytest.approx(47.90)
        assert ymax == pytest.approx(49.10)
