"""Walk-list ranking and pour-point rules — no DEM required."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.ml_spatial import block_ids, leave_one_block_aucs
from pipeline.task11_basemaps import hillshade, lith_type_from_generalize
from pipeline.geo_crs import sgmc_state, utm_epsg
from pipeline.task11_field_campaign import (
    CELL_DEG,
    campaign_class,
    campaign_probability_path,
    clean_gpkg_copy,
    named_rivers_gdf,
    pick_pour_candidate,
    pour_points_at_snap,
    spatial_block_polygons,
    walk_sort_key,
    write_field_campaign_gpkg,
)


def test_utm_zone_is_11n_for_idaho_and_10n_for_sierra():
    assert utm_epsg(-115.1, 45.5) == 'EPSG:32611'
    assert utm_epsg(-121.05, 39.3) == 'EPSG:32610'


def test_sgmc_state_follows_config():
    assert sgmc_state({'data': {'sgmc_state': 'ID'}}) == 'ID'
    assert sgmc_state({'study_area': {'short': 'ca_sierra_placer'}}) == 'CA'
    assert sgmc_state({'study_area': {'short': 'ne_wa'}}) == 'WA'


def test_clean_gpkg_copy_is_belt_specific():
    assert clean_gpkg_copy({'study_area': {'short': 'ne_wa'}}).endswith(
        'task11_field_campaign.gpkg'
    )
    assert 'ca_sierra_placer' in clean_gpkg_copy(
        {'study_area': {'short': 'ca_sierra_placer'}}
    )


def test_campaign_probability_prefers_transfer_scores(tmp_path):
    tables = tmp_path / 'tables'
    tables.mkdir()
    scores = tables / 'task12_idaho_transfer_scores.csv'
    scores.write_text('lon,lat,p_anomalous\n-115.5,45.4,0.4\n')
    cfg = {
        'outputs_dir': str(tmp_path),
        'study_area': {'short': 'id_batholith'},
    }
    path, is_transfer = campaign_probability_path(cfg)
    assert path == str(scores)
    assert is_transfer is True


def test_campaign_class_is_honest_about_known_mines():
    assert campaign_class(0.85, 0.20) == 'expedition'
    assert campaign_class(0.85, 0.01) == 'confirm'
    assert campaign_class(0.85, 0.05) == 'confirm'
    assert campaign_class(0.50, 0.20) == 'watch'
    assert campaign_class(0.20, 0.20) == 'skip'
    assert campaign_class(None, 0.20) == 'skip'


def test_pour_point_is_lowest_elev_among_high_p():
    df = pd.DataFrame({
        'lon': [-118.1, -118.2, -118.15],
        'lat': [48.4, 48.5, 48.45],
        'p_anomalous': [0.9, 0.4, 0.7],
        'elev_m': [800.0, 400.0, 500.0],
    })
    pour = pick_pour_candidate(df)
    # median P is 0.7, so the 0.4 / 400 m grab is not in the high-P set
    assert abs(pour['lon'] - (-118.15)) < 1e-9
    assert pour['elev_m'] == 500.0


def test_pour_point_falls_back_to_highest_p_without_elev():
    df = pd.DataFrame({
        'lon': [-118.1, -118.2],
        'lat': [48.4, 48.5],
        'p_anomalous': [0.3, 0.8],
    })
    pour = pick_pour_candidate(df)
    assert pour['p_anomalous'] == 0.8


def test_walk_sort_puts_expedition_ahead_of_confirm():
    a = {'campaign_class': 'confirm', 'max_p': 0.99}
    b = {'campaign_class': 'expedition', 'max_p': 0.70}
    assert walk_sort_key(b) < walk_sort_key(a)


def test_block_grid_covers_map_and_matches_block_ids():
    xmin, xmax, ymin, ymax = -120.08, -116.92, 47.42, 49.18
    gdf = spatial_block_polygons(xmin, xmax, ymin, ymax, cell_deg=CELL_DEG)
    assert len(gdf) >= 40
    # A point inside the first cell gets that cell's block_id
    row = gdf.iloc[0]
    mid_lon = (row['lon_min'] + row['lon_max']) / 2
    mid_lat = (row['lat_min'] + row['lat_max']) / 2
    assert int(block_ids([mid_lon], [mid_lat], cell_deg=CELL_DEG)[0]) == int(row['block_id'])


def test_geopackage_has_qgis_layers(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point, box

    cells = gpd.GeoDataFrame(
        {'block_id': [1], 'campaign_class': ['expedition'], 'walk_rank': [1]},
        geometry=[box(-118.4, 48.0, -118.0, 48.4)],
        crs='EPSG:4326',
    )
    spots = gpd.GeoDataFrame(
        {'walk_rank': [1], 'role': ['pour_point'], 'p_anomalous': [0.9]},
        geometry=[Point(-118.2, 48.2)],
        crs='EPSG:4326',
    )
    path = tmp_path / 'task11_field_campaign.gpkg'
    written = write_field_campaign_gpkg(
        str(path), cells=cells, catchments=None, pour_points=spots, nure_spots=spots,
    )
    assert path.exists()
    names = {n for n, _ in written}
    assert names == {'cells', 'pour_points', 'nure_spots'}
    got = gpd.read_file(path, layer='cells')
    assert len(got) == 1
    assert got.crs.to_epsg() == 4326
    assert got.iloc[0]['campaign_class'] == 'expedition'
    assert got.iloc[0]['rank_label'] == '#1'
    assert got.iloc[0]['rank_bin'] == '#1-3'
    import sqlite3
    n_styles = sqlite3.connect(path).execute(
        'SELECT count(*) FROM layer_styles WHERE useAsDefault=1'
    ).fetchone()[0]
    assert n_styles >= 1


def test_named_rivers_from_config():
    cfg = {'rivers': [
        {'name': 'Test R.', 'coords': [[-118.0, 48.5], [-118.0, 48.0]]},
        {'name': 'too short', 'coords': [[-117.0, 48.0]]},
    ]}
    gdf = named_rivers_gdf(cfg)
    assert len(gdf) == 1
    assert gdf.iloc[0]['name'] == 'Test R.'
    assert gdf.crs.to_epsg() == 4326


def test_pour_points_move_to_d8_snap():
    import geopandas as gpd
    from shapely.geometry import Point, box

    pours = gpd.GeoDataFrame(
        {'block_id': [1], 'lon': [-117.484], 'lat': [48.071], 'walk_rank': [1]},
        geometry=[Point(-117.484, 48.071)],
        crs='EPSG:4326',
    )
    walk = pd.DataFrame({
        'block_id': [1],
        'pour_snap_lon': [-117.488],
        'pour_snap_lat': [48.055],
    })
    catchments = gpd.GeoDataFrame(
        {'block_id': [1], 'pour_lon': [-117.488472], 'pour_lat': [48.054861]},
        geometry=[box(-117.50, 48.05, -117.45, 48.10)],
        crs='EPSG:4326',
    )
    moved = pour_points_at_snap(pours, walk, catchments)
    assert abs(moved.geometry.iloc[0].x - (-117.488472)) < 1e-8
    assert abs(moved.geometry.iloc[0].y - 48.054861) < 1e-8
    assert abs(moved.iloc[0]['sample_lon'] - (-117.484)) < 1e-8


def test_geopackage_writes_river_layers(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString, box

    cells = gpd.GeoDataFrame(
        {'block_id': [1], 'campaign_class': ['expedition'], 'walk_rank': [1]},
        geometry=[box(-118.4, 48.0, -118.0, 48.4)],
        crs='EPSG:4326',
    )
    rivers = named_rivers_gdf({
        'rivers': [{'name': 'Kettle R.', 'coords': [[-118.2, 48.3], [-118.2, 48.1]]}],
    })
    streams = gpd.GeoDataFrame(
        {'acc_pct': [99.0]},
        geometry=[LineString([(-118.25, 48.35), (-118.22, 48.15)])],
        crs='EPSG:4326',
    )
    path = tmp_path / 'task11_field_campaign.gpkg'
    written = write_field_campaign_gpkg(
        str(path), cells=cells, named_rivers=rivers, streams=streams,
    )
    names = {n for n, _ in written}
    assert 'named_rivers' in names
    assert 'streams' in names
    got = gpd.read_file(path, layer='named_rivers')
    assert got.iloc[0]['name'] == 'Kettle R.'


def test_lith_type_maps_sgmc_generalize():
    assert lith_type_from_generalize('Metamorphic, gneiss') == 'MCC_metapelite'
    assert lith_type_from_generalize('Igneous, intrusive') == 'felsic_intrusive'
    assert lith_type_from_generalize('unknown rock') == 'sedimentary_cover'


def test_hillshade_is_8bit_and_finite():
    z = np.linspace(100, 200, 25).reshape(5, 5)
    hs = hillshade(z, resolution=1.0)
    assert hs.shape == (5, 5)
    assert hs.dtype == np.uint8
    assert hs.min() >= 0 and hs.max() <= 255


def test_leave_one_block_aucs_scores_separated_cells():
    rng = np.random.default_rng(2)
    # Two cells: west is yes, east is no, plus a mixed middle so both classes exist
    lon = np.concatenate([
        rng.uniform(-119.9, -119.6, 40),
        rng.uniform(-118.3, -118.0, 40),
        rng.uniform(-117.5, -117.2, 40),
    ])
    lat = rng.uniform(48.0, 48.3, 120)
    y = np.concatenate([np.ones(40, dtype=int), np.zeros(40, dtype=int),
                        rng.integers(0, 2, 40)])
    X = np.column_stack([lon, y.astype(float) + rng.normal(0, 0.05, 120)])
    out = leave_one_block_aucs(X, y, lon, lat, cell_deg=0.4, n_estimators=20)
    assert len(out) >= 2
    scored = [v for v in out.values() if v is not None]
    assert scored
    assert all(0.0 <= v <= 1.0 for v in scored)
