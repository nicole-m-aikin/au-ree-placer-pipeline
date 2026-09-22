"""Hobby / pamphlet pan overlay — join rules, not AUC."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.task11_hobby_reports import (
    COLUMNS,
    apply_hobby_counts,
    attach_walk_to_catchments,
    clip_to_bbox,
    format_hit_rate_text,
    hit_rate_table,
    hobby_reports_path,
    join_catchments,
    join_pan_pins,
    load_reports,
    normalize_reports,
    reports_gdf,
)


BBOX_CFG = {
    'study_area': {
        'name': 'Test box',
        'short': 'ne_wa',
        'bbox': {
            'lon_min': -120.0, 'lon_max': -117.0,
            'lat_min': 47.5, 'lat_max': 49.1,
        },
    }
}


def _catchment(block_id=7, campaign_class='expedition', walk_rank=1):
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {
            'block_id': [block_id],
            'campaign_class': [campaign_class],
            'walk_rank': [walk_rank],
        },
        geometry=[box(-118.4, 48.0, -118.0, 48.4)],
        crs='EPSG:4326',
    )


def test_hobby_path_follows_config_then_short_name():
    cfg = {
        'study_area': {'short': 'ne_wa'},
        'data': {'hobby_reports_csv': 'data/hobby_reports/hobby_reports_ne_wa.csv'},
    }
    path = hobby_reports_path(cfg)
    assert path.endswith('hobby_reports_ne_wa.csv')
    assert os.path.exists(path)


def test_normalize_aliases_and_confidence_clamp():
    df = pd.DataFrame({
        'name': ['a', 'b'],
        'lon': [-118.2, -118.1],
        'lat': [48.2, 48.1],
        'gold_class': ['colours', 'nothing'],
        'location_precision': ['GPS', 'stream'],
        'geom_confidence': [9, 'x'],
        'source_type': ['opt-in', 'DNR'],
        'public_land': ['Yes', 'maybe'],
    })
    got = normalize_reports(df)
    assert list(got['gold_class']) == ['color', 'blank']
    assert list(got['location_precision']) == ['point', 'creek']
    assert list(got['geom_confidence']) == [5, 2]
    assert list(got['source_type']) == ['form', 'pamphlet']
    assert list(got['public_land']) == ['yes', 'unknown']
    assert got['report_id'].notna().all()


def test_clip_drops_outside_bbox():
    df = pd.DataFrame({
        'report_id': ['in', 'out'],
        'lon': [-118.2, -122.0],
        'lat': [48.2, 48.2],
        'gold_class': ['unknown', 'unknown'],
        'location_precision': ['creek', 'creek'],
        'geom_confidence': [3, 3],
        'source_type': ['pamphlet', 'pamphlet'],
        'public_land': ['unknown', 'unknown'],
    })
    got = clip_to_bbox(normalize_reports(df), BBOX_CFG)
    assert list(got['report_id']) == ['in']


def test_district_inside_polygon_is_not_a_catchment_hit():
    df = pd.DataFrame({
        'report_id': ['D1'],
        'lon': [-118.2],
        'lat': [48.2],
        'gold_class': ['unknown'],
        'location_precision': ['district'],
        'geom_confidence': [3],
        'source_type': ['pamphlet'],
        'public_land': ['unknown'],
    })
    gdf = join_catchments(reports_gdf(normalize_reports(df)), _catchment())
    assert int(gdf.iloc[0]['block_id']) == 7
    assert bool(gdf.iloc[0]['catchment_hit']) is False


def test_walk_list_fills_campaign_class_on_bare_catchments():
    bare = _catchment()
    bare = bare.drop(columns=['campaign_class', 'walk_rank'])
    walk = pd.DataFrame({
        'block_id': [7],
        'campaign_class': ['confirm'],
        'walk_rank': [4],
    })
    got = attach_walk_to_catchments(bare, walk)
    assert got.iloc[0]['campaign_class'] == 'confirm'
    df = pd.DataFrame({
        'report_id': ['C1'],
        'lon': [-118.2],
        'lat': [48.2],
        'gold_class': ['unknown'],
        'location_precision': ['creek'],
        'geom_confidence': [3],
        'source_type': ['pamphlet'],
        'public_land': ['unknown'],
    })
    gdf = join_catchments(reports_gdf(normalize_reports(df)), got)
    assert gdf.iloc[0]['campaign_class'] == 'confirm'
    rates = hit_rate_table(join_pan_pins(gdf, None), walk=walk)
    assert int(rates.iloc[0]['n_confirm_hit']) == 1


def test_creek_inside_expedition_catchment_is_a_hit():
    df = pd.DataFrame({
        'report_id': ['C1'],
        'lon': [-118.2],
        'lat': [48.2],
        'gold_class': ['unknown'],
        'location_precision': ['creek'],
        'geom_confidence': [3],
        'source_type': ['pamphlet'],
        'public_land': ['unknown'],
    })
    gdf = join_catchments(reports_gdf(normalize_reports(df)), _catchment())
    assert bool(gdf.iloc[0]['catchment_hit']) is True
    assert gdf.iloc[0]['campaign_class'] == 'expedition'


def test_creek_in_skip_catchment_is_not_a_hit():
    df = pd.DataFrame({
        'report_id': ['C2'],
        'lon': [-118.2],
        'lat': [48.2],
        'gold_class': ['flake'],
        'location_precision': ['creek'],
        'geom_confidence': [3],
        'source_type': ['form'],
        'public_land': ['yes'],
    })
    gdf = join_catchments(
        reports_gdf(normalize_reports(df)),
        _catchment(campaign_class='skip'),
    )
    assert bool(gdf.iloc[0]['catchment_hit']) is False


def test_only_point_precision_gets_pin_distance():
    import geopandas as gpd
    from shapely.geometry import Point

    df = pd.DataFrame({
        'report_id': ['P1', 'C1'],
        'lon': [-118.21, -118.21],
        'lat': [48.18, 48.18],
        'gold_class': ['flake', 'flake'],
        'location_precision': ['point', 'creek'],
        'geom_confidence': [5, 3],
        'source_type': ['form', 'form'],
        'public_land': ['yes', 'yes'],
    })
    pans = gpd.GeoDataFrame(
        {'pan_order': [1]},
        geometry=[Point(-118.21, 48.1805)],
        crs='EPSG:4326',
    )
    gdf = join_pan_pins(reports_gdf(normalize_reports(df)), pans)
    point = gdf[gdf['report_id'] == 'P1'].iloc[0]
    creek = gdf[gdf['report_id'] == 'C1'].iloc[0]
    assert bool(point['pin_scored']) is True
    assert point['nearest_pan_pin_m'] < 100
    assert bool(creek['pin_scored']) is False
    assert pd.isna(creek['nearest_pan_pin_m'])


def test_hit_rate_counts_expedition_and_refuses_auc_language():
    df = pd.DataFrame({
        'report_id': ['C1', 'D1'],
        'lon': [-118.2, -118.2],
        'lat': [48.2, 48.2],
        'gold_class': ['unknown', 'unknown'],
        'location_precision': ['creek', 'district'],
        'geom_confidence': [3, 3],
        'source_type': ['pamphlet', 'pamphlet'],
        'public_land': ['unknown', 'unknown'],
    })
    gdf = join_catchments(reports_gdf(normalize_reports(df)), _catchment())
    gdf = join_pan_pins(gdf, pans=None)
    walk = pd.DataFrame({
        'block_id': [7, 8],
        'campaign_class': ['expedition', 'confirm'],
    })
    rates = hit_rate_table(gdf, walk=walk)
    row = rates.iloc[0]
    assert int(row['n_reports_in_bbox']) == 2
    assert int(row['n_catchment_hit']) == 1
    assert int(row['n_expedition_hit']) == 1
    assert int(row['n_district']) == 1
    assert int(row['n_unknown']) == 2
    assert int(row['n_walkable_with_hit']) == 1
    text = format_hit_rate_text(BBOX_CFG, rates)
    assert 'not AUC' in text
    assert 'ROC-AUC' not in text or 'Do not quote this as ROC-AUC' in text


def test_apply_hobby_counts_only_uses_hits():
    walk = pd.DataFrame({
        'block_id': [7, 8],
        'campaign_class': ['expedition', 'confirm'],
    })
    hobby = pd.DataFrame({
        'block_id': [7, 7, 8],
        'catchment_hit': [True, False, True],
    })
    got = apply_hobby_counts(walk, hobby)
    assert list(got['n_hobby_hit']) == [1, 1]


def test_seed_csv_loads_inside_ne_wa_bbox():
    path = hobby_reports_path(BBOX_CFG)
    df = load_reports(path)
    assert set(COLUMNS).issubset(df.columns)
    assert len(df) >= 20
    clipped = clip_to_bbox(df, BBOX_CFG)
    assert len(clipped) == len(df)
    assert (clipped['gold_class'] == 'unknown').all()
    assert 'form' not in set(clipped['source_type'])


def test_geopackage_writes_hobby_layer(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point, box

    from pipeline.task11_field_campaign import write_field_campaign_gpkg

    cells = gpd.GeoDataFrame(
        {'block_id': [1], 'campaign_class': ['expedition'], 'walk_rank': [1]},
        geometry=[box(-118.4, 48.0, -118.0, 48.4)],
        crs='EPSG:4326',
    )
    hobby = gpd.GeoDataFrame(
        {
            'report_id': ['C1'],
            'name': ['test bar'],
            'gold_class': ['unknown'],
            'location_precision': ['creek'],
            'catchment_hit': [True],
            'walk_rank': [1],
        },
        geometry=[Point(-118.2, 48.2)],
        crs='EPSG:4326',
    )
    path = tmp_path / 'task11_field_campaign.gpkg'
    written = write_field_campaign_gpkg(
        str(path), cells=cells, hobby_reports=hobby,
    )
    names = {n for n, _ in written}
    assert 'hobby_reports' in names
    got = gpd.read_file(path, layer='hobby_reports')
    assert got.iloc[0]['gold_class'] == 'unknown'
    assert got.iloc[0]['name'] == 'test bar'


def test_missing_csv_is_empty_not_an_error(tmp_path):
    cfg = {
        'study_area': {'short': 'no_such_belt', 'bbox': BBOX_CFG['study_area']['bbox']},
        'data': {'hobby_reports_csv': str(tmp_path / 'does_not_exist.csv')},
    }
    df = load_reports(hobby_reports_path(cfg))
    assert df.empty
    assert list(df.columns) == list(COLUMNS)
