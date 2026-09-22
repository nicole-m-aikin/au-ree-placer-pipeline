"""Land-access label priority, elev gate, and within-type ranks."""

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.task11_land_access import (
    ACCESS_OK_TYPES,
    apply_access_ranks,
    annotate_points,
    classify_point,
    padus_manager_type,
)

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'land_access')


def _cfg(tmp_path=None):
    padus = os.path.join(FIXTURE, 'padus_fixture.geojson')
    mlrs = os.path.join(FIXTURE, 'mlrs_fixture.geojson')
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
