"""Tests for pipeline.verification: HARD, pair resolution, concordance, WGS overlap, register."""

import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline import verification as V


# ── HARD and pair statistics ─────────────────────────────────────────────────

class TestHard:
    def test_identical_values_give_zero(self):
        assert V.hard(10.0, 10.0) == pytest.approx(0.0)

    def test_known_value(self):
        # |12 - 8| / (12 + 8) = 0.2
        assert V.hard(12.0, 8.0) == pytest.approx(0.2)

    def test_symmetric(self):
        assert V.hard(3.0, 7.0) == pytest.approx(V.hard(7.0, 3.0))

    def test_zero_sum_is_nan(self):
        assert np.isnan(V.hard(0.0, 0.0))

    def test_vectorised(self):
        np.testing.assert_allclose(V.hard([1, 2, 5], [1, 3, 5]), [0.0, 0.2, 0.0])


class TestPairPrecisionStats:
    def test_below_min_pairs_has_no_stats(self):
        s = V.pair_precision_stats([1, 2, 3], [1, 2, 3], min_pairs=10)
        assert s['n_both_detected'] == 3
        assert not s['meets_min_pairs']
        assert np.isnan(s['median_hard'])

    def test_perfect_duplicates(self):
        a = np.arange(1, 21, dtype=float)
        s = V.pair_precision_stats(a, a)
        assert s['median_hard'] == pytest.approx(0.0)
        assert s['pct_hard_le_10'] == pytest.approx(100.0)
        assert s['median_log10_ratio'] == pytest.approx(0.0)
        assert s['spearman_rho'] == pytest.approx(1.0)

    def test_constant_bias_shows_in_log_ratio(self):
        a = np.arange(1, 21, dtype=float)
        s = V.pair_precision_stats(a, a * 2)
        # HARD for b = 2a is (2a - a)/(3a) = 1/3 everywhere
        assert s['median_hard'] == pytest.approx(1 / 3, abs=1e-4)
        assert s['pct_hard_le_20'] == pytest.approx(0.0)
        assert s['median_log10_ratio'] == pytest.approx(np.log10(2), abs=1e-4)

    def test_censored_and_nan_are_ignored(self):
        a = np.r_[np.arange(1, 11, dtype=float), -5.0, np.nan]
        b = np.r_[np.arange(1, 11, dtype=float), 4.0, 3.0]
        assert V.pair_precision_stats(a, b)['n_both_detected'] == 10


def test_six_step_grid():
    flags = V.on_six_step_grid([50, 70, 100, 150, 0.3, 12.8, 33, np.nan, -5])
    assert flags.tolist() == [True, True, True, True, True, False, False, False, False]


# ── Pair resolution on a synthetic frame ─────────────────────────────────────

def _frame():
    return pd.DataFrame({
        'lab_id':          ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
        'field_id':        ['023290136', 'X', 'FC-1', 'FC-1', 'FC-1', 'FAR', 'FAR'],
        'job_id':          ['J1', 'J1', 'J2', 'J3', 'J3', 'J4', 'J5'],
        'date_submitted':  ['1/1/2000', '1/2/2000', '1/1/1980', '1/1/1985', '1/1/1985',
                            '1/1/1990', '1/1/1991'],
        'previous_lab_id': [None, '23290136.0', None, None, None, None, '099999999'],
        'lat':             [48.0, 48.0, 48.5, 48.5, 48.5, 48.1, 48.9],
        'lon':             [-118.0, -118.0, -117.5, -117.5, -117.5, -118.1, -118.9],
        'Th':              [12.0, 12.4, -100.0, -100.0, 30.0, 5.0, 5.0],
        'La':              [60.0, 66.0, 50.0, 70.0, 50.0, 10.0, 10.0],
    })


class TestResolvePreviousLabIds:
    def test_resolves_via_field_id_with_leading_zeros_and_float(self):
        df = _frame()
        pairs, summ = V.resolve_previous_lab_ids(df, {'pool': df})
        assert summ['n_candidates'] == 2
        assert summ['n_resolved'] == 1
        assert pairs.iloc[0][['orig_lab_id', 're_lab_id']].tolist() == ['C1', 'C2']

    def test_hssr_matches_counted(self):
        df = _frame()
        _, summ = V.resolve_previous_lab_ids(df, {'pool': df}, hssr_found={'99999999'})
        assert summ['hssr_scanned'] and summ['hssr_matches'] == 1

    def test_missing_column(self):
        df = _frame().drop(columns='previous_lab_id')
        pairs, summ = V.resolve_previous_lab_ids(df, {'pool': df})
        assert len(pairs) == 0 and summ['n_candidates'] == 0

    def test_scan_hssr_zip(self, tmp_path):
        z = tmp_path / 'nuresed-csv.zip'
        with zipfile.ZipFile(z, 'w') as zf:
            zf.writestr('nuresed.csv', 'rec_no,prime_id\n5000001,023290136\n5000002,123290136X\n')
        found = V.scan_hssr_for_ids(str(z), {'23290136', '11111111', '5000002'})
        assert found == {'23290136', '5000002'}


class TestFieldIdPairs:
    def test_same_field_id_and_coords_pairs_with_earliest(self):
        fp = V.field_id_pairs(_frame())
        fc = fp[fp['orig_lab_id'] == 'C3']
        assert sorted(fc['re_lab_id']) == ['C4', 'C5']
        assert not fc['same_job'].any()

    def test_different_coordinates_are_not_paired(self):
        fp = V.field_id_pairs(_frame())
        assert not {'C6', 'C7'} & (set(fp['orig_lab_id']) | set(fp['re_lab_id']))

    def test_detail_and_summary_censoring(self):
        df = _frame()
        fp = V.field_id_pairs(df)
        fp['pair_source'] = 'field_id_cross_job'
        det = V.pair_detail(fp, df, ['Th', 'La'], 'test', pair_source_col='pair_source')
        th = det[det['element'] == 'Th']
        assert len(th) == 2
        assert th['orig_censored'].all()
        s = V.summarize_pairs(det)
        row = s[(s['element'] == 'Th')].iloc[0]
        assert row['n_both_censored'] == 1
        assert row['n_one_censored'] == 1
        # C5 Th 30 is below C3's MDL of 100 → consistent
        assert row['n_one_censored_detected_above_partner_mdl'] == 0
        assert (s['element'] == 'ALL_ELEMENTS_POOLED').any()

    def test_reanalysis_verification_records_resolution(self):
        df = _frame()
        summary, detail, res = V.reanalysis_verification({'test': df}, hssr_zip=None)
        assert res['test']['n_resolved'] == 1
        allrow = summary[(summary['pair_source'] == 'previous_lab_id') & (summary['element'] == 'ALL')]
        assert allrow['status'].iloc[0] == 'resolution: 1 of 2 resolved'
        assert set(detail['pair_source']) >= {'previous_lab_id', 'field_id_cross_job'}


# ── Concordance ──────────────────────────────────────────────────────────────

class TestConcordance:
    def test_monotonic_gives_rho_one(self):
        x = np.arange(20, dtype=float)
        c = V.concordance(x, x ** 2)
        assert c['n'] == 20 and c['rho'] == pytest.approx(1.0)

    def test_nan_pairs_dropped(self):
        x = np.array([1, 2, 3, np.nan, 5, 6], dtype=float)
        y = np.array([6, 5, 4, 3, np.nan, 1], dtype=float)
        c = V.concordance(x, y)
        assert c['n'] == 4 and c['rho'] == pytest.approx(-1.0)

    def test_too_few_or_constant_is_nan(self):
        assert np.isnan(V.concordance([1, 2], [1, 2])['rho'])
        assert np.isnan(V.concordance([1, 1, 1, 1], [1, 2, 3, 4])['rho'])

    def test_uncorrelated_is_small(self):
        rng = np.random.default_rng(0)
        c = V.concordance(rng.normal(size=500), rng.normal(size=500))
        assert abs(c['rho']) < 0.15


# ── WGS vs NURE ──────────────────────────────────────────────────────────────

def test_parse_censored():
    assert V._parse_censored('<4') == (2.0, True)
    assert V._parse_censored(5.1) == (5.1, False)
    assert np.isnan(V._parse_censored(np.nan)[0])


def test_wgs_nure_comparison_within_and_beyond_radius():
    wgs = pd.DataFrame({'wgs_site': ['Near', 'Far'], 'wgs_n_samples': [3, 2],
                        'wgs_lat': [48.0, 49.0], 'wgs_lon': [-118.0, -120.0],
                        'wgs_Th_ppm': [10.0, 5.0]})
    nure = pd.DataFrame({'lab_id': ['N1'], 'lat': [48.01], 'lon': [-118.0],
                         'Th': [5.0], 'Th_censored': [False], '_extract': ['t']})
    res = V.wgs_nure_comparison(wgs, nure, elements=['Th'])
    near = res[res['wgs_site'] == 'Near'].iloc[0]
    far = res[res['wgs_site'] == 'Far'].iloc[0]
    assert near['nure_nearest_km'] == pytest.approx(1.11, abs=0.01)
    assert near['log10_wgs_over_nure_Th'] == pytest.approx(np.log10(2), abs=1e-3)
    assert np.isnan(far['nure_Th_ppm'])
    summ = res[res['wgs_site'] == '__summary__'].iloc[0]
    assert summ['n_nure_within_radius'] == 1
    assert summ['note'].startswith('1 of 2')


def test_wgs_no_overlap_is_recorded():
    wgs = pd.DataFrame({'wgs_site': ['Far'], 'wgs_n_samples': [1], 'wgs_lat': [49.0],
                        'wgs_lon': [-120.0], 'wgs_Th_ppm': [5.0]})
    nure = pd.DataFrame({'lab_id': ['N1'], 'lat': [47.0], 'lon': [-117.0], 'Th': [5.0]})
    res = V.wgs_nure_comparison(wgs, nure, elements=['Th'])
    assert 'no usable overlap' in res.iloc[-1]['note']


# ── Register ─────────────────────────────────────────────────────────────────

def test_file_facts_hash_and_skip(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_bytes(b'abc')
    f = V.file_facts(str(p))
    assert f['sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    assert f['file_size_bytes'] == 3 and f['file_mtime_utc'].endswith('Z')
    big = V.file_facts(str(p), max_bytes=1)
    assert big['sha256'] == '' and 'skipped' in big['hash_note']
    missing = V.file_facts(str(tmp_path / 'nope.csv'))
    assert missing['n_files'] == 0 and missing['hash_note'] == 'file missing locally'


def test_build_register_rows(tmp_path):
    belt = tmp_path / 'test_belt'
    belt.mkdir()
    nure = tmp_path / 'nure.csv'
    nure.write_text('lab_id,lat,lon,coord_prec_est,sample_source,Th\nA,48,-118,1 second,stream/river,5\n')
    cfg = {
        'study_area': {'name': 'Test', 'short': 'tb',
                       'bbox': {'lon_min': -119, 'lon_max': -117, 'lat_min': 47, 'lat_max': 49}},
        'outputs_dir': str(tmp_path / 'outputs'),
        'data': {'nure_csv': str(nure), 'mrds_geojson': str(tmp_path / 'missing.geojson'),
                 'aeromagnetic_tif': None, 'sgmc_state': 'ZZ'},
        'aeromagnetic_anomaly_centers': [{'lon': -118, 'lat': 48, 'amplitude': 1, 'width': 1}],
    }
    cp = belt / 'config.yaml'
    cp.write_text(yaml.safe_dump(cfg))
    ver = {'test_belt': {'nure_pairs:tb': ('re-analysis pairs', 'ok')}}
    reg = V.build_register([str(cp)], verification=ver)
    assert set(reg['belt']) == {'test_belt'}
    nrow = reg[reg['dataset'] == 'NURE/NGDB stream sediment extract'].iloc[0]
    assert nrow['status'] == 'accepted-with-caveat'
    assert len(nrow['sha256']) == 64
    assert 'coord_prec_est' in nrow['validation_checks']
    mrow = reg[reg['dataset'] == 'MRDS mine sites'].iloc[0]
    assert mrow['status'] == 'not-verified' and 'missing' in mrow['notes']
    mag = reg[reg['dataset'].str.startswith('Aeromagnetic')].iloc[0]
    assert mag['status'] == 'rejected: synthetic placeholder'
    assert set(reg.columns) >= {'dataset', 'belt', 'local_path', 'source_url', 'file_size_bytes',
                                'sha256', 'file_mtime_utc', 'validation_checks',
                                'verification_method', 'verification_result', 'status', 'notes'}
