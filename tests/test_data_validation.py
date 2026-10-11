"""Unit tests for pipeline/data_validation.py (CIM-style NURE / MRDS validation)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.data_validation import (
    censor_half_mdl, coord_prec_arcsec, dedup_mrds, feature_gate, precision_mask,
    submitter_prediction, validate_nure,
)

CFG = {'study_area': {'bbox': {'lon_min': -120.0, 'lon_max': -117.0,
                               'lat_min': 47.5, 'lat_max': 49.1}}}


def _frame(n, **cols):
    rng = np.random.default_rng(0)
    base = {
        'lab_id': [f'C{i:04d}' for i in range(n)],
        'lat': rng.uniform(47.6, 49.0, n),
        'lon': rng.uniform(-119.9, -117.1, n),
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestCoordPrecision:
    def test_parse_labels(self):
        assert coord_prec_arcsec('1 second') == 1.0
        assert coord_prec_arcsec('7.5 minutes') == 450.0
        assert coord_prec_arcsec('0.01 degree') == pytest.approx(36.0)
        assert coord_prec_arcsec(np.nan) is None
        assert coord_prec_arcsec('somewhere') is None

    def test_default_keeps_fine_drops_coarse_and_unknown(self):
        labels = ['1 second', '5 seconds', '10 seconds', '15 seconds', '0.01 degree',
                  '1 minute', '7.5 minutes', '10 minutes', np.nan, 'unknown']
        df = _frame(len(labels), coord_prec_est=labels,
                    sample_source=['stream/river'] * len(labels))
        clean, ledger = validate_nure(df, CFG)
        assert sorted(clean['coord_prec_est']) == sorted(labels[:5])
        row = ledger[ledger['check'] == 'coord_precision'].iloc[0]
        assert (row['n_in'], row['n_flagged'], row['n_out']) == (10, 5, 5)
        assert '7.5 minutes: 1' in row['note']

    def test_list_config_keeps_exact_labels(self):
        df = _frame(3, coord_prec_est=['1 second', '5 seconds', '0.01 degree'])
        keep, _ = precision_mask(df, ['1 second'])
        assert keep.tolist() == [True, False, False]

    def test_bad_config_raises(self):
        df = _frame(1, coord_prec_est=['1 second'])
        with pytest.raises(ValueError):
            precision_mask(df, 'about here')


class TestSampleMedium:
    def test_default_keeps_stream_only(self):
        srcs = ['stream/river', 'lake', 'surface', 'wetland/swamp/marsh/bog/fen', 'auger', None]
        df = _frame(len(srcs), coord_prec_est=['1 second'] * len(srcs), sample_source=srcs)
        clean, ledger = validate_nure(df, CFG)
        assert clean['sample_source'].tolist() == ['stream/river']
        row = ledger[ledger['check'] == 'sample_medium'].iloc[0]
        assert row['n_flagged'] == 5 and row['n_out'] == 1

    def test_config_override(self):
        df = _frame(3, sample_source=['stream/river', 'lake', 'surface'])
        cfg = dict(CFG, nure_validation={'sample_sources': ['stream/river', 'lake']})
        clean, _ = validate_nure(df, cfg)
        assert len(clean) == 2


class TestCensoring:
    def test_half_mdl_any_magnitude_with_mdl_and_flag(self):
        df = _frame(4, Th=[-100.0, 12.0, -6.0, np.nan])
        out, _ = censor_half_mdl(df, ['Th'])
        assert out['Th'].tolist()[:3] == [50.0, 12.0, 3.0]
        assert pd.isna(out['Th'].iloc[3])
        assert out['Th_mdl'].iloc[0] == 100.0 and out['Th_mdl'].iloc[2] == 6.0
        assert pd.isna(out['Th_mdl'].iloc[1])
        assert bool(out['Th_censored'].iloc[0]) and bool(out['Th_censored'].iloc[2])
        assert not bool(out['Th_censored'].iloc[1])
        assert pd.isna(out['Th_censored'].iloc[3])

    def test_wt_pct_conversion_scales_mdl(self):
        df = _frame(3, P=[0.09, 0.05, -0.01], Fe=[3.1, 2.0, 4.0])
        out, converted = censor_half_mdl(df, ['P', 'Fe'])
        assert set(converted) == {'P', 'Fe'}
        assert out['P'].tolist() == pytest.approx([900.0, 500.0, 50.0])
        assert out['P_mdl'].iloc[2] == pytest.approx(100.0)
        assert out['Fe'].tolist() == pytest.approx([31000.0, 20000.0, 40000.0])

    def test_ppm_fe_not_converted(self):
        df = _frame(3, Fe=[31000.0, 20000.0, 40000.0])
        out, converted = censor_half_mdl(df, ['Fe'])
        assert converted == []
        assert out['Fe'].tolist() == [31000.0, 20000.0, 40000.0]

    def test_ledger_reports_legacy_nan_count(self):
        df = _frame(3, Th=[-100.0, -6.0, 12.0])
        _, ledger = validate_nure(df, CFG)
        row = ledger[(ledger['check'] == 'censored') & (ledger['field'] == 'Th')].iloc[0]
        assert row['n_flagged'] == 2
        assert 'legacy load_nure sets 1 to NaN' in row['note']


class TestFeatureGate:
    def test_drops_on_censored_fraction(self):
        th = [-100.0] * 6 + [10.0, 12.0, 14.0, 11.0, 13.0, 9.0]
        df, _ = censor_half_mdl(_frame(len(th), Th=th), ['Th'])
        g = feature_gate(df, ['Th'], max_censored_frac=0.4).iloc[0]
        assert not g['keep']
        assert 'censored fraction 0.50 > 0.4' in g['reason']

    def test_drops_when_mdl_above_anomaly_threshold(self):
        As = [3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.5, 4.5, 5.5, 6.5, -200.0, -200.0]
        df, _ = censor_half_mdl(_frame(len(As), As=As), ['As'])
        g = feature_gate(df, ['As']).iloc[0]
        assert g['censored_frac'] < 0.5
        assert not g['keep']
        assert 'median MDL 200 > anomaly threshold' in g['reason']

    def test_keeps_clean_feature_and_flags_absent(self):
        df, _ = censor_half_mdl(_frame(6, La=[20.0, 30.0, 40.0, 50.0, 35.0, -2.0]), ['La'])
        g = feature_gate(df, ['La', 'Zr']).set_index('feature')
        assert g.loc['La', 'keep'] and g.loc['La', 'reason'] == 'kept'
        assert not g.loc['Zr', 'keep'] and g.loc['Zr', 'reason'] == 'column absent'


class TestMissingMetadata:
    def test_other_belt_schema_degrades_gracefully(self):
        df = _frame(30, Th=np.r_[[-2.0] * 5, np.linspace(5, 20, 25)])
        clean, ledger, details = validate_nure(df, CFG, return_details=True)
        assert len(clean) == 30
        notes = ledger.set_index('check')
        assert 'not assessable' in notes.loc['coord_precision', 'rule']
        assert 'not assessable' in notes.loc['sample_medium', 'rule']
        preds = ledger[ledger['check'] == 'submitter_prediction']
        assert preds['note'].str.contains('not assessable').all()
        assert len(details['batches']) == 0

    def test_bbox_drops_outside_and_keeps_raw_index(self):
        df = _frame(3, Th=[1.0, 2.0, 3.0])
        df.loc[1, 'lon'] = -125.0
        clean, ledger = validate_nure(df, CFG)
        assert clean.index.tolist() == [0, 2]
        assert ledger[ledger['check'] == 'study_area'].iloc[0]['n_flagged'] == 1


class TestLabBatch:
    def test_predicts_submitter_when_units_differ(self):
        rng = np.random.default_rng(1)
        u = np.r_[rng.lognormal(1, 0.3, 30), rng.lognormal(1, 0.3, 30) * 1000]
        la = rng.lognormal(3, 0.3, 60)
        df = _frame(60, U=u, La=la, submitter=['A'] * 30 + ['B'] * 30)
        res = submitter_prediction(df, ['U', 'La'])
        assert res['status'] == 'ok'
        assert res['chance'] == 0.5
        assert res['balanced_accuracy'] > 0.9

    def test_skips_with_one_large_submitter(self):
        df = _frame(30, U=np.linspace(1, 5, 30), submitter=['A'] * 25 + ['B'] * 5)
        res = submitter_prediction(df, ['U'])
        assert res['status'].startswith('skipped')


class TestDedupMrds:
    def _gdf(self, dep_ids, xs, ys):
        import geopandas as gpd
        return gpd.GeoDataFrame({'dep_id': dep_ids},
                                geometry=gpd.points_from_xy(xs, ys), crs='EPSG:4326')

    def test_drops_dep_id_then_coords(self):
        gdf = self._gdf([1, 1, 2, 3, 4],
                        [-118.0, -118.5, -118.12341, -118.12344, -117.9],
                        [48.0, 48.5, 48.2, 48.2, 48.9])
        ded, report = dedup_mrds(gdf)
        assert len(ded) == 3
        assert [r['n_dropped'] for r in report] == [1, 1]
        assert ded['dep_id'].tolist() == [1, 2, 4]

    def test_no_dep_id_column(self):
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-118.0, -118.0], [48.0, 48.0]),
                               crs='EPSG:4326')
        ded, report = dedup_mrds(gdf)
        assert len(ded) == 1
        assert 'not assessable' in report[0]['rule']


class TestProximityLabelsHelper:
    def test_no_dem_is_radius_only(self):
        import geopandas as gpd
        from pipeline.task9_ml_targeting import proximity_labels
        mrds = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-118.0], [48.0]), crs='EPSG:4326')
        df = pd.DataFrame({'lon': [-118.05, -118.5], 'lat': [48.0, 48.0]})
        y, desc = proximity_labels(df, mrds, None, 0.15, 200, 'placer_gold')
        assert y.tolist() == [1, 0]
        assert 'no DEM elevation filter' in desc


class TestSubmitterConfound:
    def _data(self):
        import geopandas as gpd
        rng = np.random.default_rng(1)
        mrds = gpd.GeoDataFrame(geometry=gpd.points_from_xy([-118.0], [48.0]), crs='EPSG:4326')
        # Submitter A mostly near the deposit (positive), B mostly far; La is noise.
        near_a, far_a, near_b, far_b = 40, 20, 10, 50
        lon = np.concatenate([
            rng.uniform(-118.05, -117.95, near_a), rng.uniform(-119.5, -119.0, far_a),
            rng.uniform(-118.05, -117.95, near_b), rng.uniform(-119.5, -119.0, far_b)])
        lat = np.concatenate([
            rng.uniform(47.95, 48.05, near_a), rng.uniform(48.5, 49.0, far_a),
            rng.uniform(47.95, 48.05, near_b), rng.uniform(48.5, 49.0, far_b)])
        n = len(lon)
        df = pd.DataFrame({'lon': lon, 'lat': lat,
                           'submitter': ['A'] * (near_a + far_a) + ['B'] * (near_b + far_b),
                           'La': rng.lognormal(3, 0.5, n)})
        return df, mrds

    def test_submitter_alone_predicts_label(self):
        from pipeline.validation_impact import submitter_confound
        df, mrds = self._data()
        t = submitter_confound(df, ['La'], mrds, {'ml': {}}).set_index('group')
        only = t.loc['submitter_only_predictor']
        assert only['n_samples'] == 120 and only['n_positive'] == 50
        assert only['shuffled_auc_mean'] > 0.7
        assert t.loc['within:A', 'n_positive'] == 40
        assert t.loc['within:B', 'n_positive'] == 10
        assert t.loc['within:A', 'shuffled_n_folds'] == 5

    def test_small_minority_class_not_scored(self):
        from pipeline.validation_impact import submitter_confound
        df, mrds = self._data()
        extra = pd.DataFrame({'lon': [-118.0, -119.2], 'lat': [48.0, 48.7],
                              'submitter': ['C', 'C'], 'La': [10.0, 20.0]})
        t = submitter_confound(pd.concat([df, extra], ignore_index=True), ['La'],
                               mrds, {'ml': {}}).set_index('group')
        assert 'not scored' in t.loc['within:C', 'note']

    def test_no_submitter_column(self):
        from pipeline.validation_impact import submitter_confound
        df, mrds = self._data()
        t = submitter_confound(df.drop(columns='submitter'), ['La'], mrds, {'ml': {}})
        assert t.iloc[0]['note'] == 'no submitter column'
