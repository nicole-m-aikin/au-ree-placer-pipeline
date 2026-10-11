"""Land-access label priority, elev gate, and within-type ranks."""

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.task11_land_access import (
    ACCESS_COLS,
    ACCESS_OK_TYPES,
    TRIBAL_NOT_LOADED,
    TRIBAL_REASON,
    _cell_access_from_pans,
    apply_access_ranks,
    annotate_points,
    classify_point,
    config_state,
    padus_manager_type,
    rules_note,
    sync_walk_rank,
)

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'land_access')


def _cfg(tmp_path=None, aiannh='aiannh_fixture.geojson'):
    padus = os.path.join(FIXTURE, 'padus_fixture.geojson')
    mlrs = os.path.join(FIXTURE, 'mlrs_fixture.geojson')
    aiannh_p = os.path.join(FIXTURE, aiannh)
    return {
        'study_area': {
            'short': 'ca_sierra_placer',
            'bbox': {
                'lon_min': -121.2, 'lon_max': -120.8,
                'lat_min': 39.15, 'lat_max': 39.50,
            },
        },
        'task11': {
            'land_access': True,
            'min_pour_elev_m': 50,
            'claim_buffer_m': 50,
        },
        'data': {
            'padus_geojson': padus,
            'mlrs_geojson': mlrs,
            'aiannh_geojson': aiannh_p,
            'sgmc_state': 'CA',
        },
    }


def test_padus_manager_maps_blm_usfs_state_park():
    blm = padus_manager_type({
        'Mang_Name': 'Bureau of Land Management',
        'Pub_Access': 'OA',
        'Des_Tp': 'Recreation Management Area',
        'Unit_Nm': 'South Yuba',
    })
    assert blm[0] == 'blm'
    usfs = padus_manager_type({
        'Mang_Name': 'USDA Forest Service',
        'Pub_Access': 'OA',
        'Des_Tp': 'National Forest',
        'Unit_Nm': 'Tahoe NF',
    })
    assert usfs[0] == 'usfs'
    sp = padus_manager_type({
        'Mang_Name': 'California Department of Parks and Recreation',
        'Pub_Access': 'OA',
        'Des_Tp': 'State Park',
        'Unit_Nm': 'South Yuba River SP',
    })
    assert sp[0] == 'state_park'


def test_claimed_beats_blm():
    got = classify_point(
        {'CSE_NAME': 'TEST CLAIM'},
        {'Mang_Name': 'Bureau of Land Management', 'Pub_Access': 'OA'},
        None,
        'free',
    )
    assert got['access_type'] == 'claimed'
    assert got['access_ok'] == 'no'
    assert 'TEST CLAIM' in got['access_reason']


def test_restricted_pub_access():
    got = classify_point(
        False,
        {
            'Mang_Name': 'California Department of Parks and Recreation',
            'Des_Tp': 'State Park',
            'Pub_Access': 'XA',
            'Unit_Nm': 'Closed Unit',
        },
        None,
        'free',
    )
    assert got['access_type'] == 'restricted'
    assert got['access_ok'] == 'no'


def test_private_when_outside_padus():
    got = classify_point(False, None, None, 'free')
    assert got['access_type'] == 'private'
    assert got['access_ok'] == 'no'


def test_blm_open_is_access_ok():
    got = classify_point(
        False,
        {
            'Mang_Name': 'Bureau of Land Management',
            'Pub_Access': 'OA',
            'Unit_Nm': 'South Yuba',
        },
        None,
        'free',
    )
    assert got['access_type'] == 'blm'
    assert got['access_ok'] == 'yes'
    assert got['access_type'] in ACCESS_OK_TYPES


def test_annotate_points_fixture_priority():
    cfg = _cfg()
    pts = gpd.GeoDataFrame(
        {
            'block_id': [1, 2, 3, 4],
            'name': ['blm_free', 'blm_claimed', 'restricted', 'private'],
        },
        geometry=[
            Point(-121.00, 39.33),   # BLM, outside claim
            Point(-121.00, 39.33),   # same BLM poly — will also hit claim if inside
            Point(-121.09, 39.21),   # restricted state park
            Point(-121.15, 39.30),   # outside all
        ],
        crs='EPSG:4326',
    )
    # Put claimed point inside the claim polygon
    pts.loc[1, 'geometry'] = Point(-121.000, 39.330)
    # Claim covers -121.005..-120.995, 39.325..39.335 — center is claimed
    pts.loc[0, 'geometry'] = Point(-121.015, 39.335)  # BLM poly but west of claim
    out = annotate_points(pts, cfg)
    by = dict(zip(out['name'], out['access_type']))
    assert by['blm_free'] == 'blm'
    assert by['blm_claimed'] == 'claimed'
    assert by['restricted'] == 'restricted'
    assert by['private'] == 'private'
    assert list(out.loc[out['name'] == 'blm_free', 'access_ok'])[0] == 'yes'
    assert list(out.loc[out['name'] == 'blm_claimed', 'access_ok'])[0] == 'no'


def test_tribal_beats_restricted_and_blm():
    got = classify_point(
        False,
        {'Mang_Name': 'Bureau of Land Management', 'Pub_Access': 'OA'},
        None,
        'free',
        tribal_hit='Fixture Rancheria',
    )
    assert got['access_type'] == 'tribal'
    assert got['access_ok'] == 'no'
    assert got['access_reason'] == TRIBAL_REASON
    assert got['tribal_land'] is True
    assert got['tribal_name'] == 'Fixture Rancheria'
    restricted = classify_point(
        False, {'Mang_Name': 'X', 'Pub_Access': 'XA'}, None, 'free',
        tribal_hit='Fixture Rancheria',
    )
    assert restricted['access_type'] == 'tribal'


def test_claimed_beats_tribal_but_keeps_tribal_flag():
    got = classify_point(
        {'CSE_NAME': 'TEST CLAIM'}, None, None, 'free',
        tribal_hit='Fixture Rancheria',
    )
    assert got['access_type'] == 'claimed'
    assert got['access_ok'] == 'no'
    assert got['tribal_land'] is True
    assert 'tribal' in got['access_reason']


def test_non_tribal_point_has_tribal_false():
    got = classify_point(False, None, None, 'free')
    assert got['tribal_land'] is False
    assert got['tribal_name'] == ''


def test_annotate_points_tribal_fixture():
    pts = gpd.GeoDataFrame(
        {'name': ['blm_tribal', 'usfs_tdsa', 'private', 'blm_free']},
        geometry=[
            Point(-121.015, 39.345),  # BLM poly ∩ Fixture Rancheria
            Point(-120.925, 39.42),   # USFS poly ∩ statistical TDSA
            Point(-121.15, 39.30),    # outside everything
            Point(-121.015, 39.335),  # BLM, outside rancheria + claim
        ],
        crs='EPSG:4326',
    )
    out = annotate_points(pts, _cfg())
    for col in ACCESS_COLS:
        assert col in out.columns
    row = out.set_index('name')
    assert row.loc['blm_tribal', 'access_type'] == 'tribal'
    assert row.loc['blm_tribal', 'access_ok'] == 'no'
    assert row.loc['blm_tribal', 'access_reason'] == TRIBAL_REASON
    assert bool(row.loc['blm_tribal', 'tribal_land']) is True
    assert row.loc['blm_tribal', 'tribal_name'] == 'Fixture Rancheria'
    assert 'tribal permission required' in row.loc['blm_tribal', 'rules_note']
    # Statistical areas are not tribal land — note only.
    assert row.loc['usfs_tdsa', 'access_type'] == 'usfs'
    assert row.loc['usfs_tdsa', 'access_ok'] == 'yes'
    assert bool(row.loc['usfs_tdsa', 'tribal_land']) is False
    note = row.loc['usfs_tdsa', 'rules_note']
    assert 'Notice of Intent' in note
    assert 'suction-dredge' in note
    assert 'Fixture Band TDSA' in note
    assert 'landowner permission required' in row.loc['private', 'rules_note']
    assert row.loc['blm_free', 'access_ok'] == 'yes'
    assert bool(row.loc['blm_free', 'tribal_land']) is False
    assert TRIBAL_NOT_LOADED not in row.loc['blm_free', 'rules_note']


def test_missing_tribal_layer_is_noop():
    pts = gpd.GeoDataFrame(
        {'name': ['blm_tribal', 'private']},
        geometry=[Point(-121.015, 39.345), Point(-121.15, 39.30)],
        crs='EPSG:4326',
    )
    out = annotate_points(pts, _cfg(aiannh='does_not_exist.geojson'))
    assert list(out['tribal_land']) == [False, False]
    assert list(out['tribal_name']) == ['', '']
    assert list(out['access_type']) == ['blm', 'private']
    assert all(TRIBAL_NOT_LOADED in n for n in out['rules_note'])


def test_tribal_screen_runs_with_land_access_off():
    cfg = _cfg()
    cfg['task11']['land_access'] = False
    pts = gpd.GeoDataFrame(
        {'name': ['blm_tribal']}, geometry=[Point(-121.015, 39.345)], crs='EPSG:4326',
    )
    out = annotate_points(pts, cfg)
    assert out['access_ok'].iloc[0] == 'no'
    assert bool(out['tribal_land'].iloc[0]) is True
    assert out['tribal_name'].iloc[0] == 'Fixture Rancheria'


def test_rules_note_state_and_manager():
    wa = rules_note('usfs', 'WA')
    assert 'district ranger' in wa and 'WDFW Gold and Fish' in wa
    assert 'suction-dredge' in rules_note('blm', 'CA')
    assert rules_note('private', '') == 'landowner permission required'
    assert TRIBAL_NOT_LOADED in rules_note('usfs', 'CA', tribal_loaded=False)
    assert config_state({'study_area': {'short': 'ne_wa'}}) == 'WA'
    assert config_state({'study_area': {'short': 'ca_sierra_placer'}}) == 'CA'


def test_cell_rollup_carries_tribal_columns():
    pans = pd.DataFrame({
        'block_id': [1, 1, 2],
        'access_type': ['tribal', 'usfs', 'tribal'],
        'access_ok': ['no', 'yes', 'no'],
        'access_reason': [TRIBAL_REASON, 'usfs ok', TRIBAL_REASON],
        'manager_name': ['R', 'USFS', 'R'],
        'claim_status': ['free'] * 3,
        'pub_access': ['', 'OA', ''],
        'tribal_land': [True, False, True],
        'tribal_name': ['R', '', 'R'],
        'rules_note': ['tribal permission required', 'USFS: …', 'tribal permission required'],
    })
    cell = _cell_access_from_pans(pans).set_index('block_id')
    assert cell.loc[1, 'access_type'] == 'usfs'
    assert bool(cell.loc[1, 'tribal_land']) is False
    assert cell.loc[2, 'access_type'] == 'tribal'
    assert bool(cell.loc[2, 'tribal_land']) is True


def test_elev_gate_clears_walk_rank():
    walk = pd.DataFrame({
        'block_id': [1, 2, 3],
        'campaign_class': ['expedition', 'expedition', 'confirm'],
        'catchment_ok': [True, True, True],
        'pour_elev_m': [8.0, 400.0, 200.0],
        'max_p': [0.9, 0.85, 0.8],
        'nearest_gold_deg': [0.2, 0.15, 0.01],
        'n_hobby_hit': [0, 0, 0],
        'access_type': ['private', 'blm', 'usfs'],
        'access_ok': ['no', 'yes', 'yes'],
        'access_reason': ['valley', 'blm ok', 'usfs ok'],
        'manager_name': ['', 'BLM', 'USFS'],
        'claim_status': ['free', 'free', 'free'],
        'pub_access': ['', 'OA', 'OA'],
    })
    cfg = _cfg()
    ranked = apply_access_ranks(walk, cfg)
    # Valley-floor expedition loses walk_rank
    assert pd.isna(ranked.loc[0, 'walk_rank'])
    assert pd.notna(ranked.loc[1, 'walk_rank'])
    assert pd.notna(ranked.loc[2, 'walk_rank'])
    # access_rank only for access_ok=yes that pass screen
    assert pd.isna(ranked.loc[0, 'access_rank'])
    assert set(ranked.loc[ranked['access_rank'].notna(), 'block_id']) == {2, 3}


def test_rank_in_access_type_within_blm():
    walk = pd.DataFrame({
        'block_id': [1, 2, 3, 4],
        'campaign_class': ['expedition', 'confirm', 'expedition', 'watch'],
        'catchment_ok': [True, True, True, True],
        'pour_elev_m': [500, 400, 450, 300],
        'max_p': [0.95, 0.9, 0.7, 0.5],
        'nearest_gold_deg': [0.2, 0.01, 0.1, 0.3],
        'n_hobby_hit': [0, 0, 1, 0],
        'access_type': ['blm', 'blm', 'private', 'blm'],
        'access_ok': ['yes', 'yes', 'no', 'yes'],
        'access_reason': ['a', 'b', 'c', 'd'],
        'manager_name': ['', '', '', ''],
        'claim_status': ['free'] * 4,
        'pub_access': ['OA'] * 4,
    })
    ranked = apply_access_ranks(walk, _cfg())
    blm = ranked[ranked['access_type'] == 'blm'].sort_values('rank_in_access_type')
    assert list(blm['rank_in_access_type'].astype(int)) == [1, 2, 3]
    # Expedition + more isolated first among BLM
    assert int(blm.iloc[0]['block_id']) == 1


def test_geometry_access_sort_raises_ok_cell_over_hotter_private():
    walk = pd.DataFrame({
        'block_id': [1, 2, 3],
        'campaign_class': ['confirm', 'confirm', 'confirm'],
        'catchment_ok': [True, True, True],
        'pour_elev_m': [400.0, 200.0, 180.0],
        'max_p': [0.70, 1.0, 0.99],
        'nearest_gold_deg': [0.02, 0.01, 0.01],
        'n_hobby_hit': [0, 0, 0],
        'trap_score': [4.0, 7.5, 7.5],
        'access_type': ['usfs', 'private', 'claimed'],
        'access_ok': ['yes', 'no', 'no'],
        'access_reason': ['usfs ok', 'private', 'claimed'],
        'manager_name': ['USFS', '', ''],
        'claim_status': ['free', 'free', 'claimed'],
        'pub_access': ['OA', '', ''],
    })
    cfg = _cfg()
    cfg['task11']['rank_by'] = 'geometry_access'
    ranked = apply_access_ranks(walk, cfg)
    order = list(
        ranked.loc[ranked['walk_rank'].notna()]
        .sort_values('walk_rank')['block_id']
        .astype(int)
    )
    assert order[0] == 1
    assert order == [1, 2, 3]
    assert int(ranked.loc[ranked['block_id'] == 1, 'access_rank'].iloc[0]) == 1
    assert pd.isna(ranked.loc[ranked['block_id'] == 2, 'access_rank'].iloc[0])


def test_pans_carry_the_final_walk_rank_not_the_pre_access_one():
    # Pans are cut before the access re-rank; the CSV must show the final order.
    pans = pd.DataFrame({'block_id': [7, 7, 9, 5], 'walk_rank': [5.0, 5.0, 1.0, 6.0]})
    walk = pd.DataFrame({
        'block_id': [7, 9, 5],
        'walk_rank': pd.array([1, 2, pd.NA], dtype='Int64'),
    })
    out = sync_walk_rank(pans, walk)
    assert out['walk_rank'].tolist()[:3] == [1, 1, 2]
    assert pd.isna(out['walk_rank'].iloc[3])
    assert pans['walk_rank'].tolist() == [5.0, 5.0, 1.0, 6.0]
