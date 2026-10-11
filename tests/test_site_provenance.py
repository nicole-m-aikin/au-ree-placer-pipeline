"""Tests for pipeline/site_provenance.py (config site ↔ MRDS verification)."""

import os
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.site_provenance import (
    CLASS_FAR, CLASS_NEAR, CLASS_NEAREST, haversine_km, match_kind, name_tokens,
    names_match, run, verify_sites,
)

ROOT = os.path.join(os.path.dirname(__file__), '..')


def _mrds(rows):
    df = pd.DataFrame(rows, columns=['dep_id', 'site_name', 'lon', 'lat'])
    df['dep_id'] = df['dep_id'].astype(str)
    df['dev_stat'] = 'Prospect'
    df['commodity'] = 'Gold'
    df['url'] = 'https://example/' + df['dep_id']
    return df


class TestNameMatching:
    def test_generic_words_stripped(self):
        assert name_tokens('Northport Placer', strip_generic=True) == ['northport']
        assert name_tokens('Republic Gold District', strip_generic=True) == ['republic']
        assert name_tokens('Sanpoil River Placer', strip_generic=True) == ['sanpoil']
        assert name_tokens('Old Dominion Mine', strip_generic=True) == ['old', 'dominion']

    def test_meyers_myers_alias(self):
        assert match_kind('Meyers Creek Placer', 'Myers Creek-Chewsaw-Mary Ann Creek') == 'exact'

    def test_partial_token_set_is_not_a_match(self):
        assert not names_match('Kettle Falls Placer', 'Meyers Falls Placer')
        assert not names_match('Old Dominion Mine', 'Dominion Creek')

    def test_joined_spelling(self):
        assert match_kind('San Poil Placer', 'Sanpoil River, West Fork Placers') == 'exact'

    def test_fuzzy_spelling(self):
        assert match_kind('Sanpoil River Placer', 'Sandpoil River, West Fla.') == 'fuzzy'

    def test_all_generic_name_never_matches(self):
        assert not names_match('Gold Placer', 'Gold Placer Mine')


def test_haversine_one_degree_latitude():
    assert haversine_km(-118.0, 48.0, -118.0, 49.0) == pytest.approx(111.2, abs=0.2)


class TestVerifySites:
    MRDS = _mrds([
        ('1', 'Northport Bar Placer', -117.786, 48.922),
        ('2', 'Bossburg Bar', -118.049, 48.756),
        ('3', 'Galena Farm', -117.81, 48.72),
        ('4', 'Sandpoil River, West Fla.', -118.773, 48.458),
        ('5', 'Sanpoil River, West Fork Placers', -118.746, 48.457),
    ])

    def _row(self, name, lon, lat, **kw):
        t = verify_sites([dict(name=name, lon=lon, lat=lat, **kw)], self.MRDS)
        return t.iloc[0]

    def test_near_name_match(self):
        r = self._row('Northport Placer', -117.78, 48.92)
        assert r['match_class'] == CLASS_NEAR
        assert r['match_dep_id'] == '1'
        assert r['match_dist_km'] < 1.0
        assert r['nearest_dep_id'] == '1'

    def test_far_name_match_reports_nearest_separately(self):
        r = self._row('Bossburg Placer', -117.80, 48.72)
        assert r['match_class'] == CLASS_FAR
        assert r['match_dep_id'] == '2'
        assert r['match_dist_km'] > 5
        assert r['nearest_site_name'] == 'Galena Farm'

    def test_exact_beats_closer_fuzzy_when_both_far(self):
        r = self._row('Sanpoil River Placer', -118.85, 48.20)
        assert r['match_class'] == CLASS_FAR
        assert r['match_site_name'] == 'Sanpoil River, West Fork Placers'
        assert r['match_kind'] == 'exact'
        assert 'Sandpoil' in r['other_name_matches']

    def test_nearest_only(self):
        r = self._row('Colville Placer', -117.90, 48.55)
        assert r['match_class'] == CLASS_NEAREST
        assert pd.isna(r['match_dep_id'])
        assert r['nearest_dep_id'] in set(self.MRDS['dep_id'])

    def test_config_mrds_id_checked_against_extract(self):
        assert not self._row('Northport Placer', -117.78, 48.92, mrds_id='M004331')[
            'config_mrds_id_in_extract']
        assert self._row('Northport Placer', -117.78, 48.92, mrds_id='1')[
            'config_mrds_id_in_extract']
        assert not self._row('Northport Placer', -117.78, 48.92, mrds_id=None)[
            'config_mrds_id_in_extract']

    def test_near_km_threshold(self):
        t = verify_sites([dict(name='Bossburg Placer', lon=-117.80, lat=48.72)],
                         self.MRDS, near_km=25)
        assert t.iloc[0]['match_class'] == CLASS_NEAR

    def test_no_sites_gives_empty_table_with_columns(self):
        t = verify_sites([], self.MRDS)
        assert t.empty and 'match_class' in t.columns


def test_run_on_ne_washington_config(tmp_path):
    cfg_path = os.path.join(ROOT, 'configs', 'ne_washington', 'config.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mrds = os.path.join(ROOT, cfg['data']['mrds_geojson'])
    if not os.path.exists(mrds):
        pytest.skip('NE WA MRDS extract not present')
    table = run(cfg, out_path=str(tmp_path / 'sv.csv'))
    assert len(table) == len(cfg['sites'])
    assert (tmp_path / 'sv.csv').exists()
    by_name = table.set_index('site_name')
    for s in cfg['sites']:
        row = by_name.loc[s['name']]
        review = s.get('mrds_review') or {}
        # Config mrds_id must be a real dep_id or null — never a fabricated id.
        if s.get('mrds_id'):
            assert row['config_mrds_id_in_extract']
            if review.get('decision') == 'accepted':
                assert str(review['dep_id']) == str(s['mrds_id'])
                assert row['review_decision'] == 'accepted'
            else:
                assert row['match_class'] == CLASS_NEAR
                assert row['match_dep_id'] == str(s['mrds_id'])
        if review.get('decision') == 'rejected':
            assert not s.get('mrds_id')
            assert row['review_dep_id'] == str(review['dep_id'])
        assert s['mrds_match'] in (CLASS_NEAR, CLASS_FAR, CLASS_NEAREST,
                                   'accepted_on_review', 'no_match')
        assert 'dims_source' in s


def test_review_columns_carried_from_config():
    mrds = _mrds([('10131500', 'Bender Custom Mill', -119.60, 48.95)])
    site = dict(name='Brender Placer', lon=-119.42, lat=48.51,
                location_status='unverified',
                mrds_review={'decision': 'rejected', 'dep_id': '10131500',
                             'note': 'mill, not a placer'})
    r = verify_sites([site], mrds).iloc[0]
    assert r['review_decision'] == 'rejected'
    assert r['review_dep_id'] == '10131500'
    assert r['review_note'] == 'mill, not a placer'
    assert r['config_location_status'] == 'unverified'
    plain = verify_sites([dict(name='Brender Placer', lon=-119.42, lat=48.51)], mrds).iloc[0]
    assert pd.isna(plain['review_decision'])


@pytest.mark.parametrize('belt', [
    'california_sierra', 'idaho_batholith', 'montana_placer',
    'colorado_wet_mtns', 'fall_zone_nc',
])
def test_run_on_other_belts(belt, tmp_path):
    with open(os.path.join(ROOT, 'configs', belt, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    mrds = os.path.join(ROOT, cfg['data']['mrds_geojson'])
    if not os.path.exists(mrds):
        pytest.skip(f'{belt} MRDS extract not present')
    table = run(cfg, out_path=str(tmp_path / 'sv.csv'))
    assert len(table) == len(cfg.get('sites') or [])
    assert table['match_class'].isin([CLASS_NEAR, CLASS_FAR, CLASS_NEAREST]).all()
