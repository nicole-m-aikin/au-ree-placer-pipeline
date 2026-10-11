"""Statewide GEE bundle — cross-region dedupe, re-rank, JSON-safe features."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.gee_export import (
    county_coverage,
    dedupe_across_regions,
    feature_dicts,
    rerank_statewide,
)


def _pans():
    return pd.DataFrame({
        'pin_id': ['a:1', 'b:1', 'b:2', 'a:2'],
        'region': ['a', 'b', 'b', 'a'],
        'county': ['Kern', 'Kern', 'Kern', 'Tulare'],
        'lon': [-118.5000, -118.5005, -118.4000, -118.6],
        'lat': [35.5000, 35.5003, 35.4000, 35.9],
        'evidence_score': [70.0, 80.0, 60.0, 50.0],
        'trap_score': [6.0, 6.0, 5.0, 5.0],
        'rec_score': [70.0, 80.0, np.nan, 50.0],
        'stake_score': [np.nan, 80.0, 60.0, np.nan],
        'chem_c': [0.5, np.nan, 0.2, np.nan],
    })


def test_dedupe_keeps_higher_evidence_across_regions_only():
    out = dedupe_across_regions(_pans(), radius_m=150)
    assert set(out['pin_id']) == {'b:1', 'b:2', 'a:2'}


def test_dedupe_leaves_same_region_neighbours():
    p = _pans()
    p['region'] = 'a'
    assert len(dedupe_across_regions(p, radius_m=150)) == 4


def test_rerank_statewide_ranks_per_county():
    out = rerank_statewide(dedupe_across_regions(_pans()))
    kern = out[out['county'] == 'Kern'].set_index('pin_id')
    assert int(kern.loc['b:1', 'rec_rank_in_county']) == 1
    assert pd.isna(kern.loc['b:2', 'rec_rank_in_county'])
    assert int(kern.loc['b:2', 'stake_rank_in_county']) == 2


def test_feature_dicts_are_json_safe():
    feats = feature_dicts(_pans())
    text = json.dumps(feats, allow_nan=False)
    assert '"rec_score": null' in text
    assert feats[0]['geometry']['coordinates'] == [-118.5, 35.5]


def test_county_coverage_lists_unbuilt_counties():
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import box

    counties = gpd.GeoDataFrame({'GEOID': ['06029', '06107', '06001'],
                                 'NAME': ['Kern', 'Tulare', 'Alameda']},
                                geometry=[box(0, 0, 1, 1)] * 3, crs='EPSG:4326')
    cov = county_coverage(_pans(), counties).set_index('county')
    assert cov.loc['Kern', 'status'] == 'built'
    assert cov.loc['Kern', 'n_stakeable'] == 2
    assert cov.loc['Kern', 'n_with_nure'] == 2
    assert cov.loc['Alameda', 'status'] == 'not_built'
