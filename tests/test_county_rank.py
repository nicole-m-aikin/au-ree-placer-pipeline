"""County ranking — seeds, evidence, stakeable gates. Synthetic data, no network."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.county_rank import (
    CHANNEL_BONUS,
    CLOSED_CLAIM_BONUS,
    MINED_FACTOR,
    assign_counties,
    bearing_upstream_deg,
    channel_bonus,
    channel_component,
    chem_component,
    closed_claim_hits,
    corridor_alongs,
    evidence_score,
    gold_site_weight,
    grid_seeds,
    hobby_seeds,
    padus_overlap_flags,
    rank_within_county,
    reach_hits,
    score_pans,
    seed_specs,
    trap_on_new_scale,
    source_component,
    stake_status,
    upstream_source_component,
)

gpd = pytest.importorskip('geopandas')
from shapely.geometry import LineString, Point, box  # noqa: E402


def _pts(xy):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in xy], crs='EPSG:4326')


# ── Seeds ─────────────────────────────────────────────────────────────────────

def test_grid_seeds_one_per_cell_and_prefers_producer():
    mrds = pd.DataFrame({
        'site_name': ['prospect', 'big mine', 'far away'],
        'dev_stat': ['Prospect', 'Past Producer', 'Occurrence'],
        'lon': [-118.500, -118.501, -118.300],
        'lat': [35.500, 35.501, 35.700],
    })
    seeds = grid_seeds(mrds, cell_km=3.0, max_seeds=10, lat0=35.5)
    assert len(seeds) == 2
    top = seeds.iloc[0]
    assert top['name'] == 'big mine'
    assert top['n_sites'] == 2 and top['n_producers'] == 1


def test_grid_seeds_caps_and_handles_empty():
    mrds = pd.DataFrame({
        'site_name': [f's{i}' for i in range(20)],
        'dev_stat': 'Occurrence',
        'lon': -118.0 + 0.1 * np.arange(20),
        'lat': 35.5,
    })
    assert len(grid_seeds(mrds, 3.0, max_seeds=5, lat0=35.5)) == 5
    assert grid_seeds(pd.DataFrame(), 3.0, 5).empty


def test_seed_specs_put_hobby_pins_first():
    hobby = pd.DataFrame({'name': ['Park'], 'creek_name': ['Kern River'],
                          'lon': [-118.4], 'lat': [35.7]})
    seeds = pd.concat([
        grid_seeds(pd.DataFrame({'site_name': ['m'], 'dev_stat': ['Producer'],
                                 'lon': [-118.5], 'lat': [35.5]}), 3.0, 5),
        hobby_seeds(hobby),
    ], ignore_index=True)
    rc = {'up_km': 0.5, 'down_km': 4, 'snap_m': 600, 'min_snap_km2': 1.0}
    specs = seed_specs(seeds, rc, 'ca_test')
    assert specs[0]['seed_kind'] == 'hobby'
    assert specs[0]['creek_name'] == 'Kern River'
    assert specs[1]['walk_id'] == 'ca_test:w0002'


# ── Evidence ─────────────────────────────────────────────────────────────────

def test_source_component_decays_and_zeroes_missing():
    s = source_component([0.0, 3.0, np.nan])
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(np.exp(-1.0))
    assert s[2] == 0.0


def test_old_triple_flag_collapses_to_one_break():
    got = trap_on_new_scale(
        [6.0, 4.0, 8.0],
        ['slope_break;knickpoint_foot;power_drop',
         'knickpoint_foot;power_drop;tributary_junction',
         'gradient_break;tributary_junction;valley_opens'],
    )
    assert list(got) == [2.0, 5.0, 8.0]


def test_evidence_score_ignores_chemistry():
    full = evidence_score([8.0], [1.0], [1.0])
    blank = evidence_score([8.0], [1.0], [np.nan])
    weak = evidence_score([8.0], [1.0], [0.2])
    assert full[0] == pytest.approx(100.0)
    assert blank[0] == full[0]
    assert weak[0] == full[0]


def test_chem_component_uses_best_sample_in_radius():
    nure = pd.DataFrame({
        'lon': [-118.500, -118.501, -118.900],
        'lat': [35.500, 35.500, 35.900],
        'As': [1.0, 50.0, 100.0],
    })
    nure = pd.concat([nure] * 4, ignore_index=True)
    got = chem_component(np.array([-118.5, -118.2]), np.array([35.5, 35.2]),
                         nure, ['Au', 'As'], radius_km=1.0, lat0=35.5)
    assert np.isfinite(got[0]) and got[0] > 0.5
    assert np.isnan(got[1])


def test_upstream_corridor_ignores_the_next_ridge():
    lat0 = 35.5
    # Creek runs south; upstream is north.
    line = LineString([( -118.5, 35.55), (-118.5, 35.45)])
    bearing = bearing_upstream_deg(line, -118.5, 35.50, lat0)
    assert bearing == pytest.approx(0.0, abs=5.0)
    north = 35.50 + 5.0 / 111.32
    east = -118.5 + 5.0 / (111.32 * np.cos(np.radians(lat0)))
    along_up = corridor_alongs(-118.5, 35.50, bearing, [ -118.5], [north], lat0)
    along_side = corridor_alongs(-118.5, 35.50, bearing, [east], [35.50], lat0)
    assert len(along_up) == 1 and along_up[0] == pytest.approx(5.0, abs=0.2)
    assert len(along_side) == 0
    assert upstream_source_component(along_up) == pytest.approx(np.exp(-5.0 / 6.0), abs=0.02)
    assert upstream_source_component([]) == 0.0
    assert upstream_source_component([0.0, 0.0]) == 1.0


def test_channel_bonus_only_on_the_unit():
    assert channel_component([0.0, 2.0, np.nan])[0] == pytest.approx(1.0)
    assert channel_bonus([0.0])[0] == pytest.approx(CHANNEL_BONUS)
    assert channel_bonus([0.8])[0] == 0.0
    assert channel_bonus([2.0])[0] == 0.0
    assert channel_bonus([180.0])[0] == 0.0
    assert channel_bonus([np.nan])[0] == 0.0
    pans = pd.DataFrame({
        'lon': [-118.5, -118.5, -118.5],
        'lat': [35.5, 35.5, 35.5],
        'trap_score': [6.0, 6.0, 6.0],
    })
    rc = {'chem_elements': ['As'], 'chem_radius_km': 3.0}
    scored = score_pans(
        pans, None, np.array([[-118.5, 35.5]]), rc, 35.5,
        upstream_alongs=[np.array([0.0]), np.array([0.0]), np.array([0.0])],
        channel_km=[0.0, 180.0, 0.0],
        on_mined=[False, False, True],
    )
    on_gravel, far, mined = scored['evidence_score']
    assert far == pytest.approx(85.9)
    assert on_gravel == pytest.approx(min(100.0, far + CHANNEL_BONUS))
    assert mined == pytest.approx(round(min(100.0, far + CHANNEL_BONUS) * MINED_FACTOR, 1))


def test_reach_hits_follow_the_bend_and_drop_the_seed():
    lat0 = 35.5
    line = LineString([(-118.50, 35.55), (-118.50, 35.50), (-118.45, 35.50)])
    pin_lon, pin_lat = -118.47, 35.50
    idx, alongs = reach_hits(
        line, pin_lon, pin_lat,
        [-118.50, -118.47, -118.50],
        [35.53, pin_lat + 3.0 / 111.32, 35.55],
        lat0,
        seed_lon=-118.50, seed_lat=35.55,
    )
    assert list(idx) == [0]
    assert alongs[0] > 1.0


def test_prospect_weighs_less_than_a_producer():
    assert gold_site_weight('Big Mine', 'Past Producer') == 1.0
    assert gold_site_weight('Unnamed Prospect', 'Prospect') == 0.25
    assert upstream_source_component([0.0], [0.25]) == pytest.approx(0.25)
    assert upstream_source_component([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_chem_component_nan_without_nure():
    assert np.isnan(chem_component(np.array([-118.5]), np.array([35.5]),
                                   None, ['As'], 3.0, 35.5)).all()


# ── Stakeable gates ──────────────────────────────────────────────────────────

def _padus():
    return gpd.GeoDataFrame({
        'Mang_Name': ['USFS', 'USFS', 'BLM', 'USFS'],
        'Category': ['Fee', 'Designation', 'Fee', 'Proclamation'],
        'Des_Tp': ['NF', 'WSR', 'ACEC', 'PROC'],
        'GAP_Sts': ['3', '2', '3', '4'],
        'Unit_Nm': ['Sequoia NF', 'Kern WSR', 'Erskine Creek ACEC', 'Sequoia NF'],
    }, geometry=[box(0, 0, 10, 10), box(4, 4, 5, 5), box(20, 0, 30, 10),
                 box(40, 0, 50, 10)], crs='EPSG:4326')


def test_padus_overlap_flags_sees_stacked_polygons():
    flags = padus_overlap_flags(_pts([(1, 1), (4.5, 4.5), (25, 5), (45, 5), (99, 99)]),
                                _padus())
    assert list(flags['federal_manager']) == ['USFS', 'USFS', 'BLM', '', '']
    assert flags.loc[1, 'withdrawal_hint'].startswith('wild & scenic river corridor')
    assert flags.loc[0, 'withdrawal_hint'] == ''
    assert 'critical environmental concern' in flags.loc[2, 'caution_note']
    assert list(flags['on_fee_land']) == [True, True, True, False, False]


def test_stake_status_gate_order():
    assert stake_status(True, 'claimed', 'wilderness', 'BLM') == 'tribal'
    assert stake_status(False, 'claimed', 'wilderness', 'BLM') == 'claimed'
    assert stake_status(False, 'free', 'wilderness', 'BLM') == 'withdrawn_hint'
    assert stake_status(False, 'free', '', '') == 'not_federal'
    assert stake_status(False, 'free', '', 'BLM', 'state_park') == 'mixed_manager'
    assert stake_status(False, 'unchecked', '', 'BLM', 'blm') == 'unchecked'
    assert stake_status(False, 'free', '', 'USFS', 'usfs') == 'open_screen'


def test_closed_claim_hits_within_radius():
    lat = 35.5
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat))
    claim = box(-118.5, lat - 0.001, -118.5 + 100 / m_per_deg_lon, lat + 0.001)
    closed = gpd.GeoDataFrame({'CSE_NAME': ['CURLY']}, geometry=[claim], crs='EPSG:4326')
    near = -118.5 + 300 / m_per_deg_lon
    far = -118.5 + 2000 / m_per_deg_lon
    names = closed_claim_hits(_pts([(near, lat), (far, lat)]), closed, radius_m=300)
    assert names == ['CURLY', '']
    assert CLOSED_CLAIM_BONUS == 0.0


# ── Counties + ranks ─────────────────────────────────────────────────────────

def test_assign_counties_and_rank_within_county():
    counties = gpd.GeoDataFrame({'GEOID': ['06029', '06107'], 'NAME': ['Kern', 'Tulare']},
                                geometry=[box(0, 0, 10, 10), box(0, 10, 10, 20)],
                                crs='EPSG:4326')
    pts = _pts([(1, 1), (2, 2), (3, 15), (50, 50)])
    name, geoid = assign_counties(pts, counties)
    assert name == ['Kern', 'Kern', 'Tulare', '']
    assert geoid[2] == '06107'

    df = pd.DataFrame({'county': name, 'score': [50.0, 90.0, 70.0, 99.0],
                       'trap_score': [6.0, 6.0, 6.0, 6.0]})
    ok = pd.Series([True, True, False, True])
    ranked = rank_within_county(df, ok, 'score', 'rank')
    assert ranked['rank'].iloc[0] == 2
    assert ranked['rank'].iloc[1] == 1
    assert ranked['rank'].iloc[2:].isna().all()
