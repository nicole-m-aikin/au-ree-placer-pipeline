"""Trap-geometry votes — no DEM required."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.task11_pan_traps import pick_separated, score_reach, why_flags


def test_slope_break_flags_the_foot_of_a_steep_reach():
    # 150 m steps. Steep for three cells, then almost flat — gold drops at the foot.
    s = np.arange(8) * 150.0
    z = np.array([200.0, 185.0, 170.0, 155.0, 153.0, 152.0, 151.0, 150.0])
    scored = score_reach(z, s)
    feet = scored.index[scored['slope_break']].tolist()
    assert feet, 'expected a slope-break flag on the flat below the drop'
    # Trap is at the foot (first vertex on the flat), not up on the steep lip.
    assert min(feet) >= 3
    assert scored.loc[min(feet), 'trap_score'] > 0
    assert not bool(scored.loc[1, 'slope_break'])


def test_knickpoint_foot_is_downstream_of_the_sl_peak():
    s = np.arange(7) * 150.0
    z = np.array([100.0, 98.0, 96.0, 80.0, 79.0, 78.0, 77.0])
    scored = score_reach(z, s)
    assert bool(scored['sl_foot'].any())
    # Foot is after the big drop (index 3 is the lip).
    feet = scored.index[scored['sl_foot']].tolist()
    assert min(feet) >= 3


def test_pick_separated_keeps_distance():
    df = pd.DataFrame({
        'x': [0.0, 50.0, 800.0],
        'y': [0.0, 0.0, 0.0],
        'trap_score': [5.0, 4.0, 3.0],
    })
    picked = pick_separated(df, n=2, min_sep_m=400)
    assert len(picked) == 2
    assert picked.iloc[0]['x'] == 0.0
    assert picked.iloc[1]['x'] == 800.0


def test_pick_separated_ignores_zero_scores():
    df = pd.DataFrame({
        'x': [0.0, 800.0],
        'y': [0.0, 0.0],
        'trap_score': [0.0, 0.0],
    })
    picked = pick_separated(df, n=2)
    assert picked.empty


def test_why_flags_names_the_votes():
    row = {
        'slope_break': True, 'sl_foot': False, 'power_drop': True,
        'junction': True, 'valley_open': False, 'near_nure': True,
    }
    why = why_flags(row)
    assert 'slope_break' in why
    assert 'power_drop' in why
    assert 'tributary_junction' in why
    assert 'near_hot_nure' in why
    assert 'knickpoint_foot' not in why


def test_geopackage_writes_nure_and_pan_layers(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point, box

    from pipeline.task11_field_campaign import write_field_campaign_gpkg

    cells = gpd.GeoDataFrame(
        {'block_id': [1], 'campaign_class': ['expedition'], 'walk_rank': [1]},
        geometry=[box(-118.4, 48.0, -118.0, 48.4)],
        crs='EPSG:4326',
    )
    nure = gpd.GeoDataFrame(
        {'walk_rank': [1], 'role': ['high_p_sample'], 'p_anomalous': [0.9]},
        geometry=[Point(-118.20, 48.20)],
        crs='EPSG:4326',
    )
    pans = gpd.GeoDataFrame(
        {'walk_rank': [1], 'pan_order': [1], 'why': ['slope_break'], 'trap_score': [4.0]},
        geometry=[Point(-118.21, 48.18)],
        crs='EPSG:4326',
    )
    path = tmp_path / 'task11_field_campaign.gpkg'
    written = write_field_campaign_gpkg(
        str(path), cells=cells, nure_spots=nure, pan_locations=pans,
    )
    names = {n for n, _ in written}
    assert 'nure_spots' in names
    assert 'pan_locations' in names
    assert 'field_spots' not in names
    got = gpd.read_file(path, layer='pan_locations')
    assert got.iloc[0]['why'] == 'slope_break'
    assert got.iloc[0]['rank_bin'] == '#1-3'
