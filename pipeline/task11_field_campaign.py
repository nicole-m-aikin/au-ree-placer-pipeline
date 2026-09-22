"""
Task 11: Catchment walk list — cells, pours, NURE grabs, and pan pins.

This is the expedition product. It does not retrain the forest. It takes the
published Task 9 probabilities, groups them into the same 0.4° cells used for
spatial CV, snaps one pour point per occupied cell, lists the highest-P NURE
grabs in that drainage, then votes for pan locations on the D8 creek
(slope break, power drop, knickpoint foot, tributary mouth).

Campaign classes (plain language):
  expedition — high P (≥0.6) and far from a mapped gold mine
  confirm    — high P next to a known mine (calibration walk)
  watch      — moderate P (0.4–0.6)
  skip       — low P or no NURE in the cell

Outputs:
  {outputs_dir}/figures/fig11_catchment_walk_list_map.png
  {outputs_dir}/tables/task11_catchment_walk_list.csv
  {outputs_dir}/tables/task11_nure_spots.csv
  {outputs_dir}/tables/task11_pan_locations.csv
  {outputs_dir}/gis/task11_field_campaign.gpkg
  {outputs_dir}/gis/task11_field_campaign_shp.zip    # ArcGIS Online
  ~/projects/task11_field_campaign.gpkg              # QGIS copy (path has no +)
  {outputs_dir}/geojson/task11_spatial_cv_blocks_0p4deg.geojson
  {outputs_dir}/geojson/task11_catchment_polygons.geojson
  {outputs_dir}/geojson/task11_pour_points.geojson
  {outputs_dir}/geojson/task11_field_campaign_targets.geojson
  {outputs_dir}/text/task11_field_campaign_summary.txt

GeoPackage layers (EPSG:4326): cells, catchments, named_rivers, streams,
pour_points, nure_spots, pan_locations. Pour points sit on the D8 outlet.
NURE spots are chemistry grabs. Pan locations are the trap-geometry pins.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from pipeline.ml_artifacts import load_gold_mrds
from pipeline.ml_preprocess import FEATURES, _log_impute
from pipeline.ml_spatial import (
    FAR_FROM_MINE_DEG,
    block_ids,
    leave_one_block_aucs,
    nearest_gold_mrds_deg,
    spatial_flag,
)
from pipeline.task9_ml_targeting import delineate_catchments
from pipeline.utils import (
    WONG, setup_mpl, load_nure, watermark, save_fig, ensure_outputs, out,
    bbox, map_extent, hillshade, north_arrow, scale_bar,
    canada_border, locator_inset, topo_contours, rivers_with_arrows,
    clip_gdf_to_map, resolve_data_path,
    MAP_W, MAP_H, _FIG_LM, _FIG_RM, _FIG_TM, _FIG_BM,
    _FIG_VGAP, _FIG_CW, _FIG_CG, _ax_rect,
)

CELL_DEG = 0.4
N_NURE_SPOTS = 3
N_FIELD_SPOTS = N_NURE_SPOTS  # old name
HIGH_P = 0.6
MOD_P = 0.4
# 0.15° is the *training* label radius and paints most of this gold belt
# "near a mine." A crew uses a tighter walk radius: ~5 km from a pin.
FIELD_FAR_DEG = 0.05
# ~0.08 deg² is a local tributary. Bigger than that is a main-stem snap
# (Columbia / Kettle) and is not a walkable field drainage.
CATCHMENT_AREA_CAP_DEG2 = 0.08
# Region-wide channels use a high accumulation cut (same idea as the pour
# snap). Tributaries *inside* a walkable catchment use a lower cut so you
# can see the creek through a small basin like #1.
STREAM_MAIN_PCT = 99.0
STREAM_TRIB_PCT = 96.5
CLASS_ORDER = {'expedition': 0, 'confirm': 1, 'watch': 2, 'skip': 3}
CLASS_COLOR = {
    'expedition': WONG['vermillion'],
    'confirm': WONG['blue'],
    'watch': WONG['yellow'],
    'skip': '#CCCCCC',
}


def campaign_class(max_p, distance_deg, field_far_deg=FIELD_FAR_DEG):
    """One-word field priority from chemistry + km-scale isolation.

    distance_deg is to the nearest gold MRDS pin. The 0.15° training
    radius is *not* used here — it would mark almost every NE WA cell
    as already known. None / NaN distance → treat as confirm, not new.
    """
    if max_p is None or (isinstance(max_p, float) and np.isnan(max_p)):
        return 'skip'
    far = (
        distance_deg is not None
        and not (isinstance(distance_deg, float) and np.isnan(distance_deg))
        and distance_deg > field_far_deg
    )
    if max_p >= HIGH_P:
        return 'expedition' if far else 'confirm'
    if max_p >= MOD_P:
        return 'watch'
    return 'skip'


def unique_sample_xy(df):
    """Collapse duplicate NURE coordinates. Mean P, mean elev if present."""
    if df is None or df.empty:
        return df
    agg = {'p_anomalous': 'mean'}
    if 'elev_m' in df.columns:
        agg['elev_m'] = 'mean'
    if 'label' in df.columns:
        agg['label'] = 'max'
    return df.groupby(['lat', 'lon'], as_index=False).agg(agg)


def pick_pour_candidate(df):
    """Most-downstream high-P grab in a cell.

    Among samples at or above the cell median P, pick the lowest DEM
    elevation (the trap). No elev → highest P. Returns a Series or None.
    """
    if df is None or len(df) == 0:
        return None
    work = unique_sample_xy(df.dropna(subset=['lon', 'lat']).copy())
    if work is None or work.empty:
        return None
    p = work['p_anomalous']
    high = work[p >= p.median()] if p.notna().any() else work
    if high.empty:
        high = work
    if 'elev_m' in high.columns and high['elev_m'].notna().any():
        high = high.sort_values(['elev_m', 'p_anomalous'],
                                ascending=[True, False], na_position='last')
    else:
        high = high.sort_values('p_anomalous', ascending=False, na_position='last')
    return high.iloc[0]


def walk_sort_key(row):
    """Expedition first, then high P. Used for walk_rank."""
    return (CLASS_ORDER.get(row['campaign_class'], 9),
            -float(row['max_p']) if pd.notna(row.get('max_p')) else 0.0)


def spatial_block_polygons(xmin, xmax, ymin, ymax, cell_deg=CELL_DEG):
    """All 0.4° cells that touch the map box. Origin matches block_ids()."""
    import geopandas as gpd
    from shapely.geometry import box

    x0 = np.floor(xmin / cell_deg) * cell_deg
    y0 = np.floor(ymin / cell_deg) * cell_deg
    rows = []
    x = x0
    while x < xmax - 1e-12:
        y = y0
        while y < ymax - 1e-12:
            bid = int(np.floor((x + 1e-9) / cell_deg)) * 10_000 + int(
                np.floor((y + 1e-9) / cell_deg)
            )
            rows.append({
                'block_id': bid,
                'lon_min': float(x),
                'lon_max': float(x + cell_deg),
                'lat_min': float(y),
                'lat_max': float(y + cell_deg),
                'geometry': box(x, y, x + cell_deg, y + cell_deg),
            })
            y += cell_deg
        x += cell_deg
    return gpd.GeoDataFrame(rows, crs='EPSG:4326')


def _sample_elev(dem_path, lons, lats):
    if not dem_path or not os.path.exists(dem_path):
        return np.full(len(lons), np.nan)
    import rasterio
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        vals = []
        for v in src.sample(zip(lons, lats)):
            z = v[0]
            if nodata is not None and z == nodata:
                vals.append(np.nan)
            else:
                vals.append(float(z) if np.isfinite(z) else np.nan)
    return np.asarray(vals, dtype=float)


def _proximity_labels(df, gold_xy, dem_path, radius_deg, max_elev_diff):
    """Published-style yes/no: near a gold MRDS pin, same valley floor."""
    coords = np.column_stack([df['lon'].values, df['lat'].values])
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(gold_xy, dtype=float))
    dist, idx = tree.query(coords)
    in_radius = dist <= radius_deg
    if dem_path and os.path.exists(dem_path):
        nure_z = _sample_elev(dem_path, df['lon'].values, df['lat'].values)
        gold_z = _sample_elev(dem_path, gold_xy[:, 0], gold_xy[:, 1])
        near_z = gold_z[idx]
        on_floor = (np.abs(nure_z - near_z) <= max_elev_diff) | np.isnan(nure_z - near_z)
        return (in_radius & on_floor).astype(int)
    return in_radius.astype(int)


def _pick_nure_spots(pool, pour, n=N_NURE_SPOTS):
    """Top-P unique NURE grabs, listed upstream → downstream."""
    if pool is None or pool.empty:
        spots = []
    else:
        u = unique_sample_xy(pool)
        u = u[u['p_anomalous'] >= MOD_P] if (u['p_anomalous'] >= MOD_P).any() else u
        u = u.sort_values('p_anomalous', ascending=False).head(n)
        spots = [r for _, r in u.iterrows()]
    if pour is not None:
        already = any(
            abs(s['lon'] - pour['lon']) < 1e-5 and abs(s['lat'] - pour['lat']) < 1e-5
            for s in spots
        )
        if not already:
            spots.append(pour)
    if not spots:
        return []
    spot_df = pd.DataFrame(spots)
    if 'elev_m' in spot_df.columns and spot_df['elev_m'].notna().any():
        spot_df = spot_df.sort_values('elev_m', ascending=False, na_position='last')
    else:
        spot_df = spot_df.sort_values('p_anomalous', ascending=False)
    return [r for _, r in spot_df.iterrows()]


def named_rivers_gdf(cfg):
    """Config rivers as LineStrings. These are schematic named trunks."""
    import geopandas as gpd
    from shapely.geometry import LineString

    rows = []
    for river in cfg.get('rivers') or []:
        coords = river.get('coords') or []
        if len(coords) < 2:
            continue
        rows.append({
            'name': str(river.get('name') or ''),
            'geometry': LineString([(float(c[0]), float(c[1])) for c in coords]),
        })
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return gpd.GeoDataFrame(rows, crs='EPSG:4326')


def pour_points_at_snap(pour_points, walk, catchments=None):
    """Move pour markers to the D8 outlet. Keep the NURE grab as sample_*."""
    import geopandas as gpd

    if pour_points is None or len(pour_points) == 0:
        return pour_points
    snap = None
    if (catchments is not None and len(catchments)
            and {'pour_lon', 'pour_lat', 'block_id'} <= set(catchments.columns)):
        snap = (
            catchments[['block_id', 'pour_lon', 'pour_lat']]
            .drop_duplicates('block_id')
            .rename(columns={'pour_lon': 'snap_lon', 'pour_lat': 'snap_lat'})
        )
    if ((snap is None or snap.empty) and walk is not None
            and {'pour_snap_lon', 'pour_snap_lat'} <= set(walk.columns)):
        snap = (
            walk[['block_id', 'pour_snap_lon', 'pour_snap_lat']]
            .dropna(subset=['pour_snap_lon', 'pour_snap_lat'])
            .drop_duplicates('block_id')
            .rename(columns={'pour_snap_lon': 'snap_lon', 'pour_snap_lat': 'snap_lat'})
        )
    if snap is None or snap.empty:
        return pour_points
    out_gdf = pour_points.drop(columns=['snap_lon', 'snap_lat'], errors='ignore')
    out_gdf = out_gdf.merge(snap, on='block_id', how='left')
    if 'lon' in out_gdf.columns:
        out_gdf['sample_lon'] = out_gdf['lon']
        out_gdf['sample_lat'] = out_gdf['lat']
    use = out_gdf['snap_lon'].notna()
    if not bool(use.any()):
        return pour_points
    out_gdf.loc[use, 'lon'] = out_gdf.loc[use, 'snap_lon']
    out_gdf.loc[use, 'lat'] = out_gdf.loc[use, 'snap_lat']
    return out_gdf.set_geometry(
        gpd.points_from_xy(out_gdf['lon'], out_gdf['lat']),
        crs=out_gdf.crs or 'EPSG:4326',
    )


def extract_dem_streams(dem_path, percentile=STREAM_MAIN_PCT, cache_path=None,
                        catchments=None, trib_percentile=STREAM_TRIB_PCT):
    """D8 channels from the same DEM used to snap pours.

    Main stems use a high accumulation cut so the layer matches the pour
    snap. If catchments are given, finer tributaries are added *inside*
    those polygons so a small basin still shows its creek.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    if cache_path and os.path.exists(cache_path):
        return gpd.read_file(cache_path)
    if not dem_path or not os.path.exists(dem_path):
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    from pysheds.grid import Grid

    print(f"  Extracting DEM streams from {os.path.basename(dem_path)}...")
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)
    acc_arr = np.asarray(acc)
    main_cut = int(np.percentile(acc_arr.ravel(), percentile))
    mask = acc > main_cut
    print(f"  Stream mask: accumulation > p{percentile} ({main_cut} cells)")
    if catchments is not None and len(catchments) and trib_percentile is not None:
        try:
            from rasterio.features import rasterize
            trib_cut = int(np.percentile(acc_arr.ravel(), trib_percentile))
            shapes = [
                (geom, 1) for geom in catchments.geometry
                if geom is not None and not geom.is_empty
            ]
            if shapes:
                catch_mask = rasterize(
                    shapes, out_shape=acc_arr.shape, transform=grid.affine,
                    fill=0, dtype='uint8',
                )
                mask = mask | ((acc > trib_cut) & (catch_mask == 1))
                print(f"  + tributaries inside catchments > p{trib_percentile} "
                      f"({trib_cut} cells)")
        except Exception as e:
            warnings.warn(f"  catchment tributary mask failed: {e}")
    net = grid.extract_river_network(fdir, mask)
    rows = []
    for feat in net.get('features', []):
        geom = shape(feat['geometry'])
        if geom.is_empty or geom.length < 1e-4:
            continue
        rows.append({'geometry': geom})
    gdf = gpd.GeoDataFrame(rows, crs='EPSG:4326') if rows else gpd.GeoDataFrame(
        geometry=[], crs='EPSG:4326',
    )
    if cache_path and len(gdf):
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
        gdf.to_file(cache_path, driver='GeoJSON')
    print(f"  DEM streams: {len(gdf)} segments")
    return gdf


GPKG_NAME = 'task11_field_campaign.gpkg'
GPKG_LAYERS = ('cells', 'catchments', 'named_rivers', 'streams',
               'pour_points', 'nure_spots', 'pan_locations', 'geology',
               'geology_structure', 'lidar_index')
# Folder name "Au + REE pipeline" breaks some QGIS/GDAL path parsers.
# A second copy lives next to the projects folder (no spaces, no +).
CLEAN_GPKG_COPY = os.path.expanduser(
    '~/projects/task11_field_campaign.gpkg'
)


def clean_gpkg_copy(cfg):
    """QGIS-safe copy. WA keeps the historic filename; other belts get a slug."""
    short = str((cfg.get('study_area') or {}).get('short') or '').strip()
    if short in ('', 'ne_wa', 'ne_washington'):
        return CLEAN_GPKG_COPY
    return os.path.expanduser(f'~/projects/task11_{short}_field_campaign.gpkg')


def campaign_probability_path(cfg):
    """Task 9 CSV, or Task 12 transfer scores if this belt was never retrained."""
    p9 = out(cfg, 'tables', 'task9_ml_nure_probability.csv')
    if os.path.exists(p9):
        return p9, False
    from pipeline.task12_second_belt import belt_slug
    p12 = out(cfg, 'tables', f'task12_{belt_slug(cfg)}_transfer_scores.csv')
    if os.path.exists(p12):
        return p12, True
    raise FileNotFoundError(
        f"Need {p9} (Task 9) or {p12} (Task 12 transfer). "
        "Score the frozen forest first; do not retrain on this box."
    )


def campaign_gold_xy(cfg):
    """Local belt MRDS gold pins when present; else the NE WA training sidecar."""
    path = resolve_data_path(cfg, 'mrds_geojson')
    if path and os.path.exists(path):
        import geopandas as gpd
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        elif '4326' not in str(gdf.crs):
            gdf = gdf.to_crs('EPSG:4326')
        if 'target_commodity' in gdf.columns:
            gold = gdf[
                gdf['target_commodity'].fillna('').str.contains('Gold')
                & ~gdf['target_commodity'].fillna('').str.contains('Rare Earth')
            ]
            if len(gold):
                gdf = gold
        if len(gdf) == 0:
            return None
        return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])
    return load_gold_mrds()
SHP_NAME = {
    'campaign_class': 'camp_class',
    'nearest_gold_deg': 'gold_deg',
    'nearest_gold_km': 'gold_km',
    'spatial_flag': 'spat_flag',
    'n_positive_labels': 'n_pos',
    'n_field_spots': 'n_spots',
    'n_nure_spots': 'n_nspots',
    'n_pan_locations': 'n_pans',
    'catchment_ok': 'catch_ok',
    'p_anomalous': 'p_anom',
    'spot_order': 'spot_order',
    'rank_label': 'rank_lbl',
}


def _agol_ready(gdf):
    """WGS84 + types QGIS and ArcGIS Online will read without choking."""
    import geopandas as gpd

    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    out_gdf = gdf.copy()
    if 'geometry' not in out_gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    if out_gdf.crs is None:
        out_gdf = out_gdf.set_crs('EPSG:4326')
    else:
        out_gdf = out_gdf.to_crs('EPSG:4326')
    out_gdf = out_gdf[out_gdf.geometry.notna() & ~out_gdf.geometry.is_empty].copy()
    try:
        bad = ~out_gdf.geometry.is_valid
        if bad.any():
            out_gdf.loc[bad, 'geometry'] = out_gdf.loc[bad, 'geometry'].buffer(0)
    except Exception:
        pass
    # Catchments can be GeometryCollection after clip/buffer — force MultiPolygon.
    try:
        mixed = out_gdf.geom_type.isin(['GeometryCollection', 'Polygon', 'MultiPolygon'])
        if mixed.any() and out_gdf.geom_type.nunique() > 1:
            out_gdf['geometry'] = out_gdf.geometry.apply(
                lambda g: g if g.geom_type in ('Polygon', 'MultiPolygon')
                else getattr(g, 'convex_hull', g)
            )
    except Exception:
        pass
    for col in list(out_gdf.columns):
        if col == 'geometry':
            continue
        series = out_gdf[col]
        if pd.api.types.is_bool_dtype(series):
            out_gdf[col] = series.astype('float64')
        elif str(series.dtype) in ('Int64', 'boolean'):
            out_gdf[col] = series.astype('float64')
        elif series.dtype == object:
            out_gdf[col] = series.where(series.notna(), None).map(
                lambda v: None if v is None else str(v)
            )
    return out_gdf


def _strip_gpkg_tile_tables(path):
    """Drop *empty* tile-matrix tables so QGIS does not treat the file as a raster.

    Keep the tables when real LiDAR rasters were appended.
    """
    import sqlite3
    con = sqlite3.connect(path)
    has_raster = False
    try:
        n = con.execute(
            "SELECT count(*) FROM gpkg_contents WHERE data_type='tiles'"
        ).fetchone()[0]
        has_raster = n > 0
    except Exception:
        has_raster = False
    if has_raster:
        con.close()
        return
    for table in ('gpkg_tile_matrix', 'gpkg_tile_matrix_set',
                  'gpkg_2d_gridded_coverage_ancillary',
                  'gpkg_2d_gridded_tile_ancillary'):
        con.execute(f'DROP TABLE IF EXISTS {table}')
    con.commit()
    con.close()


def write_field_campaign_gpkg(path, cells=None, catchments=None,
                              pour_points=None, nure_spots=None,
                              field_spots=None, pan_locations=None,
                              streams=None, named_rivers=None,
                              geology=None, geology_structure=None,
                              lidar_index=None):
    """Write vector layers to a GeoPackage. Skips empty layers. Overwrites path."""
    import pyogrio

    if nure_spots is None:
        nure_spots = field_spots
    layers = {
        'cells': ('Polygon', cells),
        'geology': ('Unknown', geology),
        'geology_structure': ('Unknown', geology_structure),
        'catchments': ('MultiPolygon', catchments),
        'named_rivers': ('LineString', named_rivers),
        'streams': ('LineString', streams),
        'lidar_index': ('Polygon', lidar_index),
        'pour_points': ('Point', pour_points),
        'nure_spots': ('Point', nure_spots),
        'pan_locations': ('Point', pan_locations),
    }
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    written = []
    first = True
    for name, (gtype, gdf) in layers.items():
        from pipeline.task11_qgis_styles import add_rank_label
        ready = add_rank_label(_agol_ready(gdf))
        if ready.empty:
            continue
        if gtype == 'MultiPolygon':
            from shapely.geometry import MultiPolygon
            def _as_multi(g):
                if g.geom_type == 'MultiPolygon':
                    return g
                if g.geom_type == 'Polygon':
                    return MultiPolygon([g])
                if hasattr(g, 'geoms'):
                    polys = [p for p in g.geoms if p.geom_type in ('Polygon', 'MultiPolygon')]
                    return MultiPolygon(polys) if polys else g
                return g
            ready['geometry'] = ready.geometry.apply(_as_multi)
        if gtype == 'LineString':
            from shapely.geometry import LineString, MultiLineString
            exploded = []
            for _, row in ready.iterrows():
                g = row.geometry
                parts = list(g.geoms) if g.geom_type == 'MultiLineString' else [g]
                for part in parts:
                    if part.geom_type != 'LineString' or part.is_empty:
                        continue
                    rec = row.drop(labels='geometry').to_dict()
                    rec['geometry'] = part
                    exploded.append(rec)
            if exploded:
                import geopandas as gpd
                ready = gpd.GeoDataFrame(exploded, crs=ready.crs)
            else:
                continue
        pyogrio.write_dataframe(
            ready, path, layer=name, driver='GPKG',
            geometry_type=gtype, append=not first,
            layer_options={'SPATIAL_INDEX': 'NO'},
        )
        first = False
        written.append((name, int(len(ready))))
    if os.path.exists(path):
        _strip_gpkg_tile_tables(path)
        from pipeline.task11_qgis_styles import embed_qgis_styles
        embed_qgis_styles(path)
    return written


def _write_shapefile_zip(zip_path, layers):
    """Sidecar zip for ArcGIS Online (it is picky about GeoPackages)."""
    import shutil
    import tempfile
    import zipfile
    import geopandas as gpd

    tmp = tempfile.mkdtemp(prefix='task11_shp_')
    try:
        written = []
        for name, gdf in layers.items():
            from pipeline.task11_qgis_styles import add_rank_label
            ready = add_rank_label(_agol_ready(gdf))
            if ready.empty:
                continue
            ready = ready.rename(columns={k: v for k, v in SHP_NAME.items()
                                          if k in ready.columns})
            shp = os.path.join(tmp, f'{name}.shp')
            ready.to_file(shp, driver='ESRI Shapefile')
            written.append(name)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fn in os.listdir(tmp):
                zf.write(os.path.join(tmp, fn), arcname=fn)
        return written
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _walkable_ranks(walk):
    """Skip cells keep their class but lose a walk number. #29 is not a target."""
    out = walk.copy()
    skip = out['campaign_class'] == 'skip'
    if 'walk_rank' in out.columns:
        out.loc[skip, 'walk_rank'] = pd.NA
    return out


def _nure_spots_csv(cfg):
    """Prefer the honest name; fall back to the old field_spots file."""
    new = out(cfg, 'tables', 'task11_nure_spots.csv')
    old = out(cfg, 'tables', 'task11_field_spots.csv')
    if os.path.exists(new):
        return pd.read_csv(new)
    if os.path.exists(old):
        return pd.read_csv(old)
    return pd.DataFrame()


def write_gpkg_from_outputs(cfg):
    """Build the GeoPackage from the Task 11 CSV/GeoJSON already on disk."""
    import geopandas as gpd
    from shapely.geometry import Point

    walk = _walkable_ranks(pd.read_csv(out(cfg, 'tables', 'task11_catchment_walk_list.csv')))
    walk.to_csv(out(cfg, 'tables', 'task11_catchment_walk_list.csv'), index=False)
    spots_df = _nure_spots_csv(cfg)
    cells = gpd.read_file(out(cfg, 'geojson', 'task11_spatial_cv_blocks_0p4deg.geojson'))
    rank_cols = [c for c in (
        'walk_rank', 'campaign_class', 'n_positive_labels', 'nearest_gold_deg',
        'nearest_gold_km', 'pour_lon', 'pour_lat', 'n_field_spots',
        'n_nure_spots', 'n_pan_locations',
        'catchment_ok', 'notes',
    ) if c in walk.columns]
    cells = cells.drop(columns=[c for c in rank_cols if c in cells.columns], errors='ignore')
    cells = cells.merge(walk[['block_id'] + rank_cols], on='block_id', how='left')

    catch_path = out(cfg, 'geojson', 'task11_catchment_polygons.geojson')
    catchments = gpd.read_file(catch_path) if os.path.exists(catch_path) else None

    if catchments is not None and len(catchments) and 'block_id' in catchments.columns:
        meta = walk[['block_id', 'campaign_class', 'walk_rank', 'max_p',
                     'nearest_gold_km']].drop_duplicates('block_id')
        catchments = catchments.drop(
            columns=['campaign_class', 'walk_rank', 'max_p', 'nearest_gold_km'],
            errors='ignore',
        ).merge(meta, on='block_id', how='left')
        if 'area_deg2' in catchments.columns:
            catchments = catchments[
                catchments['area_deg2'].fillna(0) <= CATCHMENT_AREA_CAP_DEG2
            ]
        catchments = catchments[
            catchments['campaign_class'].isin(['expedition', 'confirm', 'watch'])
        ]

    pour_path = out(cfg, 'geojson', 'task11_pour_points.geojson')
    if os.path.exists(pour_path):
        pour_points = gpd.read_file(pour_path)
        pour_points = pour_points.drop(columns=['walk_rank'], errors='ignore').merge(
            walk[['block_id', 'walk_rank', 'campaign_class']].drop_duplicates('block_id'),
            on='block_id', how='left', suffixes=('', '_w'),
        )
        if 'campaign_class_w' in pour_points.columns:
            pour_points['campaign_class'] = pour_points['campaign_class_w']
            pour_points = pour_points.drop(columns='campaign_class_w')
        pour_points = pour_points[pour_points['campaign_class'].isin(
            ['expedition', 'confirm', 'watch']
        )]
        pour_points = pour_points_at_snap(pour_points, walk, catchments)
    elif not walk.empty:
        src = walk.dropna(subset=['pour_lon', 'pour_lat'])
        pour_points = gpd.GeoDataFrame(
            src.copy(),
            geometry=[Point(xy) for xy in zip(src['pour_lon'], src['pour_lat'])],
            crs='EPSG:4326',
        )
        pour_points = pour_points_at_snap(pour_points, walk, catchments)
    else:
        pour_points = None

    if not spots_df.empty:
        spots_df = spots_df.merge(
            walk[['block_id', 'walk_rank']].drop_duplicates('block_id'),
            on='block_id', how='left', suffixes=('', '_w'),
        )
        if 'walk_rank_w' in spots_df.columns:
            spots_df['walk_rank'] = spots_df['walk_rank_w']
            spots_df = spots_df.drop(columns='walk_rank_w')
        spots_df = spots_df[spots_df['campaign_class'].isin(
            ['expedition', 'confirm', 'watch']
        )].copy()
        # The NURE grab is a sample. The hydrologic pour is the snap outlet.
        if 'role' in spots_df.columns:
            spots_df.loc[spots_df['role'] == 'pour_point', 'role'] = 'high_p_sample'
        spots_df.to_csv(out(cfg, 'tables', 'task11_nure_spots.csv'), index=False)
        nure_spots = gpd.GeoDataFrame(
            spots_df.copy(),
            geometry=gpd.points_from_xy(spots_df['lon'], spots_df['lat']),
            crs='EPSG:4326',
        )
    else:
        nure_spots = None

    named_rivers = named_rivers_gdf(cfg)
    stream_cache = out(cfg, 'geojson', 'task11_dem_streams.geojson')
    streams = extract_dem_streams(
        resolve_data_path(cfg, 'dem_tif'),
        cache_path=stream_cache,
        catchments=catchments,
    )
    if streams is not None and len(streams):
        from shapely.geometry import box as _box
        xmin, xmax, ymin, ymax = map_extent(cfg)
        streams = streams.clip(_box(xmin, ymin, xmax, ymax))

    acc_cache = out(cfg, 'geojson', 'task11_d8_accumulation.tif')
    from pipeline.task11_pan_traps import ensure_accumulation_tif, pan_locations_gdf
    acc_path = ensure_accumulation_tif(resolve_data_path(cfg, 'dem_tif'), acc_cache)
    pans = pan_locations_gdf(
        catchments, streams, resolve_data_path(cfg, 'dem_tif'),
        nure_spots=nure_spots, acc_path=acc_path,
    )
    if pans is not None and len(pans):
        pans.drop(columns='geometry', errors='ignore').to_csv(
            out(cfg, 'tables', 'task11_pan_locations.csv'), index=False,
        )
        walk = walk.drop(columns=['n_pan_locations'], errors='ignore')
        counts = pans.groupby('block_id').size().rename('n_pan_locations')
        walk = walk.merge(counts, on='block_id', how='left')
        walk['n_pan_locations'] = walk['n_pan_locations'].fillna(0).astype(int)
        walk.to_csv(out(cfg, 'tables', 'task11_catchment_walk_list.csv'), index=False)
    else:
        pans = None

    layers = dict(
        cells=cells, catchments=catchments,
        named_rivers=named_rivers, streams=streams,
        pour_points=pour_points, nure_spots=nure_spots,
        pan_locations=pans,
    )
    gis_dir = os.path.join(cfg['outputs_dir'], 'gis')
    os.makedirs(gis_dir, exist_ok=True)
    path = os.path.join(gis_dir, GPKG_NAME)
    written = write_field_campaign_gpkg(path, **layers)
    from pipeline.task11_basemaps import add_basemaps_to_gpkg
    fetch_lidar = bool((cfg.get('task11') or {}).get('fetch_lidar', True))
    extra = add_basemaps_to_gpkg(cfg, path, catchments=catchments, fetch_lidar=fetch_lidar)
    written.extend(extra)
    # Keep a copy on the old geojson path so existing notes still resolve.
    legacy = out(cfg, 'geojson', GPKG_NAME)
    import shutil
    shutil.copy2(path, legacy)
    try:
        dest_copy = clean_gpkg_copy(cfg)
        shutil.copy2(path, dest_copy)
        print(f"  Copied to {dest_copy}  (open this one in QGIS — no + in the path)")
    except OSError as e:
        print(f"  Could not copy to {clean_gpkg_copy(cfg)}: {e}")
    zip_path = os.path.join(gis_dir, 'task11_field_campaign_shp.zip')
    shp_layers = _write_shapefile_zip(zip_path, layers)
    print(f"  Wrote {path}")
    for name, n in written:
        print(f"    layer {name}: {n} features")
    qml_dir = gis_dir
    from pipeline.task11_qgis_styles import LAYER_QML
    for lyr, qml in LAYER_QML.items():
        qml_path = os.path.join(qml_dir, 'task11_{0}.qml'.format(lyr))
        with open(qml_path, 'w') as f:
            f.write(qml)
        try:
            with open(os.path.expanduser('~/projects/task11_{0}.qml'.format(lyr)), 'w') as f:
                f.write(qml)
        except OSError:
            pass
    print(f"  Wrote shapefile zip {zip_path} ({', '.join(shp_layers)})")
    _publish_lidar_hillshades(gis_dir)
    _write_qgis_project(os.path.join(gis_dir, 'task11_field_campaign.qgs'), path, cfg=cfg)
    try:
        dest_copy = clean_gpkg_copy(cfg)
        qgs_copy = dest_copy.replace('.gpkg', '.qgs')
        _write_qgis_project(qgs_copy, dest_copy, cfg=cfg)
    except OSError:
        pass
    return path


def _publish_lidar_hillshades(gis_dir):
    """Copy 1 m hillshades next to the GPKG with names QGIS will drag in."""
    import shutil

    dests = [gis_dir, os.path.expanduser('~/projects'),
             os.path.expanduser('~/projects/task11_lidar')]
    for rank, src in _lidar_hillshade_tifs():
        name = f'LIDAR_{rank:02d}_hillshade.tif'
        for dest_dir in dests:
            try:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dest_dir, name))
            except OSError:
                pass


def _lidar_hillshade_tifs():
    """Walk-rank hillshade GeoTIFFs on disk, if the LiDAR fetch has run."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lidar_dir = os.path.join(root, 'ne_wa_ree', 'data', 'lidar')
    found = []
    if not os.path.isdir(lidar_dir):
        return found
    for name in sorted(os.listdir(lidar_dir)):
        if name.startswith('lidar_r') and name.endswith('_hs.tif'):
            try:
                rank = int(name[7:9])
            except ValueError:
                continue
            found.append((rank, os.path.join(lidar_dir, name)))
    return found


def _web_mercator_extent(cfg):
    """Map canvas in EPSG:3857 from the study bbox."""
    xmin, xmax, ymin, ymax = map_extent(cfg)
    def _xy(lon, lat):
        import math
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * 20037508.34 / 180.0
        return x, y
    x0, y0 = _xy(xmin, ymin)
    x1, y1 = _xy(xmax, ymax)
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def _write_qgis_project(qgs_path, gpkg_path, cfg=None):
    """QGIS project: ranked vectors on top of 1 m LiDAR hillshades."""
    gpkg_path = os.path.abspath(gpkg_path)
    layers = [
        ('geology', 'Polygon', 'geology — SGMC lithology'),
        ('geology_structure', 'Line', 'geology — faults / contacts'),
        ('cells', 'Polygon', 'cells — walk rank'),
        ('catchments', 'MultiPolygon', 'catchments — walk rank'),
        ('named_rivers', 'Line', 'named rivers'),
        ('streams', 'Line', 'DEM streams'),
        ('lidar_index', 'Polygon', 'LiDAR 1 m clip footprints'),
        ('nure_spots', 'Point', 'NURE spots — chemistry grabs'),
        ('pour_points', 'Point', 'pour points — D8 outlet'),
        ('pan_locations', 'Point', 'pan locations — trap geometry'),
    ]
    rasters = _lidar_hillshade_tifs()
    maplayers = []
    tree = []
    # Vectors first in the tree (drawn on top). Rasters last (drawn under).
    for name, gtype, title in layers:
        lid = f't11_{name}'
        tree.append(f'<layer-tree-layer name="{name}" id="{lid}"/>')
        maplayers.append(f'''  <maplayer type="vector" geometry="{gtype}" hasScaleBasedVisibilityFlag="0">
    <id>{lid}</id>
    <layername>{title}</layername>
    <srs><spatialrefsys><authid>EPSG:4326</authid></spatialrefsys></srs>
    <datasource>{gpkg_path}|layername={name}</datasource>
    <provider>ogr</provider>
  </maplayer>''')
    for rank, tif in rasters:
        lid = f't11_lidar_{rank:02d}'
        title = f'LiDAR #{rank} hillshade (1 m)'
        tree.append(f'<layer-tree-layer name="lidar_{rank:02d}" id="{lid}"/>')
        maplayers.append(f'''  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">
    <id>{lid}</id>
    <layername>{title}</layername>
    <datasource>{os.path.abspath(tif)}</datasource>
    <provider>gdal</provider>
  </maplayer>''')
    if cfg is not None:
        cxmin, cxmax, cymin, cymax = _web_mercator_extent(cfg)
    else:
        cxmin, cxmax, cymin, cymax = -13380000, -13030000, 5980000, 6300000
    xml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" projectname="Task 11 field campaign">
  <title>Task 11 — ranked walk list</title>
  <projectCrs><spatialrefsys><authid>EPSG:3857</authid></spatialrefsys></projectCrs>
  <layer-tree-group>
    {''.join(tree)}
  </layer-tree-group>
  <mapcanvas>
    <units>meters</units>
    <extent>
      <xmin>{cxmin}</xmin><xmax>{cxmax}</xmax>
      <ymin>{cymin}</ymin><ymax>{cymax}</ymax>
    </extent>
  </mapcanvas>
  <projectlayers>
{os.linesep.join(maplayers)}
  </projectlayers>
</qgis>
'''
    with open(qgs_path, 'w') as f:
        f.write(xml)
    print(f"  Wrote QGIS project {qgs_path}")


def _stagger_oy(lons, lats, base=6, collide_deg=0.15):
    oys = []
    for i, (x, y) in enumerate(zip(lons, lats)):
        bump = 0
        for j in range(i):
            if abs(x - lons[j]) < collide_deg and abs(y - lats[j]) < collide_deg:
                bump += 10
        oys.append(base + bump)
    return oys


def run(cfg):
    setup_mpl()
    ensure_outputs(cfg['outputs_dir'])
    import geopandas as gpd
    from shapely.geometry import Point

    prob_path, p_is_transfer = campaign_probability_path(cfg)
    print(f"  Using probabilities from {prob_path}"
          f"{' (frozen WA transfer — not a local model)' if p_is_transfer else ''}")

    df = load_nure(cfg)
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    df = df.dropna(subset=['lat', 'lon'])
    df = df[
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    ].copy().reset_index(drop=True)

    prob = pd.read_csv(prob_path)
    # Duplicate NURE coordinates (same grab site, extra rows) would explode a
    # naive merge. One P per unique lat/lon is what a field crew can walk.
    prob_u = prob.groupby(['lat', 'lon'], as_index=False)['p_anomalous'].mean()
    df = df.merge(prob_u, on=['lat', 'lon'], how='left')
    if df['p_anomalous'].isna().all():
        raise ValueError("Task 9 probability CSV did not match any NURE coordinates.")

    dem_path = resolve_data_path(cfg, 'dem_tif')
    df['elev_m'] = _sample_elev(dem_path, df['lon'].values, df['lat'].values)
    df['block_id'] = block_ids(df['lon'].values, df['lat'].values, cell_deg=CELL_DEG)

    gold_xy = campaign_gold_xy(cfg)
    if gold_xy is None:
        warnings.warn("no gold MRDS for this belt; distance flags will be unknown")
    ml_cfg = cfg.get('ml') or {}
    radius = ml_cfg.get('mrds_proximity_deg', FAR_FROM_MINE_DEG)
    elev_diff = ml_cfg.get('mrds_elev_diff_m', 200)

    y = None
    if gold_xy is not None:
        y = _proximity_labels(df, gold_xy, dem_path, radius, elev_diff)
        df['label'] = y
        df['nearest_gold_deg'] = [
            nearest_gold_mrds_deg(lon, lat, gold_xy)
            for lon, lat in zip(df['lon'], df['lat'])
        ]
    else:
        df['label'] = np.nan
        df['nearest_gold_deg'] = np.nan

    loco = {}
    avail = [f for f in FEATURES if f in df.columns]
    # Every NURE row is missing at least one element. Impute; do not dropna.
    if y is not None and len(avail) >= 3 and df['p_anomalous'].notna().any():
        if df['label'].nunique() == 2 and len(df) >= 20:
            X, _ = _log_impute(df, avail)
            print("  Leave-one-cell-out AUCs (same 0.4° cells as spatial CV)...")
            loco = leave_one_block_aucs(
                X.values, df['label'].astype(int).values,
                df['lon'].values, df['lat'].values,
                cell_deg=CELL_DEG, n_estimators=200,
            )
            n_scored = sum(v is not None for v in loco.values())
            print(f"  Leave-one-cell-out: {n_scored}/{len(loco)} cells scored")

    xmin, xmax, ymin, ymax = map_extent(cfg)
    blocks = spatial_block_polygons(xmin, xmax, ymin, ymax, cell_deg=CELL_DEG)

    pour_rows = []
    for bid, sub in df.groupby('block_id'):
        pour = pick_pour_candidate(sub)
        if pour is None:
            continue
        flag = spatial_flag(pour.get('nearest_gold_deg')) if gold_xy is not None else 'unknown_no_location'
        if gold_xy is not None:
            d = nearest_gold_mrds_deg(float(pour['lon']), float(pour['lat']), gold_xy)
            flag = spatial_flag(d)
        else:
            d = None
        pour_rows.append({
            'block_id': int(bid),
            'name': f'block_{int(bid)}',
            'lon': float(pour['lon']),
            'lat': float(pour['lat']),
            'p_anomalous': float(pour['p_anomalous']) if pd.notna(pour['p_anomalous']) else np.nan,
            'elev_m': float(pour['elev_m']) if 'elev_m' in pour.index and pd.notna(pour.get('elev_m')) else np.nan,
            'nearest_gold_deg': d,
            'spatial_flag': flag,
        })
    pour_df = pd.DataFrame(pour_rows)

    catch_gdf = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    if dem_path and os.path.exists(dem_path) and not pour_df.empty:
        pour_gdf = gpd.GeoDataFrame(
            pour_df.copy(),
            geometry=[Point(xy) for xy in zip(pour_df['lon'], pour_df['lat'])],
            crs='EPSG:4326',
        )
        snap_max = ml_cfg.get('catchment_snap_max_m', 5000)
        snap_pct = ml_cfg.get('catchment_snap_percentile', 99.5)
        print(f"  Delineating catchments for {len(pour_gdf)} cell pour points...")
        catch_gdf = delineate_catchments(
            pour_gdf, dem_path, snap_max_m=snap_max, snap_percentile=snap_pct,
        )
        if len(catch_gdf) and 'orig_lon' in catch_gdf.columns:
            catch_gdf['block_id'] = block_ids(
                catch_gdf['orig_lon'].values, catch_gdf['orig_lat'].values,
                cell_deg=CELL_DEG,
            )
            catch_gdf['area_deg2'] = catch_gdf.geometry.area

    nure_pts = gpd.GeoDataFrame(
        df[['lat', 'lon', 'p_anomalous', 'elev_m', 'block_id']].copy(),
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326',
    )

    walk_rows = []
    spot_rows = []
    for _, blk in blocks.iterrows():
        bid = int(blk['block_id'])
        in_cell = df[df['block_id'] == bid]
        n_nure = int(len(in_cell))
        mean_p = float(in_cell['p_anomalous'].mean()) if n_nure else np.nan
        max_p = float(in_cell['p_anomalous'].max()) if n_nure else np.nan
        n_pos = int(in_cell['label'].fillna(0).sum()) if n_nure else 0
        pour = pour_df[pour_df['block_id'] == bid]
        pour_lon = float(pour['lon'].iloc[0]) if len(pour) else np.nan
        pour_lat = float(pour['lat'].iloc[0]) if len(pour) else np.nan
        pour_elev = float(pour['elev_m'].iloc[0]) if len(pour) else np.nan
        snap_lon = np.nan
        snap_lat = np.nan
        flag = 'unknown_no_location'
        gold_d = np.nan
        if len(pour):
            gold_d = pour['nearest_gold_deg'].iloc[0]
            flag = pour['spatial_flag'].iloc[0]
        elif n_nure and gold_xy is not None:
            gold_d = float(in_cell['nearest_gold_deg'].min())
            flag = spatial_flag(gold_d)

        cls = campaign_class(max_p, gold_d) if n_nure else 'skip'
        catch_ok = False
        catch_area = np.nan
        notes = []
        pool = in_cell
        if len(catch_gdf) and 'block_id' in catch_gdf.columns:
            this_c = catch_gdf[catch_gdf['block_id'] == bid]
            if len(this_c):
                snap_lon = float(this_c['pour_lon'].iloc[0])
                snap_lat = float(this_c['pour_lat'].iloc[0])
                catch_area = float(this_c['area_deg2'].iloc[0])
                if catch_area > CATCHMENT_AREA_CAP_DEG2:
                    notes.append('catchment_too_large_used_cell_spots')
                    catch_ok = False
                else:
                    catch_ok = True
                    joined = gpd.sjoin(
                        nure_pts, this_c[['geometry']], how='inner', predicate='within',
                    )
                    if len(joined):
                        pool = joined.drop(columns='index_right', errors='ignore')
        elif n_nure and (not dem_path or not os.path.exists(dem_path)):
            notes.append('no_dem_cell_spots_only')
        elif n_nure and len(pour) and not catch_ok:
            notes.append('catchment_failed_used_cell_spots')

        pour_series = None
        if len(pour):
            pour_series = pd.Series({
                'lon': pour_lon, 'lat': pour_lat,
                'p_anomalous': float(pour['p_anomalous'].iloc[0]),
                'elev_m': pour_elev,
            })
        spots = _pick_nure_spots(pool, pour_series) if n_nure else []

        walk_rows.append({
            'block_id': bid,
            'lon_min': blk['lon_min'], 'lon_max': blk['lon_max'],
            'lat_min': blk['lat_min'], 'lat_max': blk['lat_max'],
            'n_nure': n_nure,
            'n_positive_labels': n_pos,
            'mean_p': None if np.isnan(mean_p) else round(mean_p, 4),
            'max_p': None if np.isnan(max_p) else round(max_p, 4),
            'loco_auc': (
                None if (loco.get(bid) is None or n_nure < 8)
                else round(float(loco[bid]), 4)
            ),
            'nearest_gold_deg': None if (gold_d is None or (isinstance(gold_d, float) and np.isnan(gold_d)))
            else round(float(gold_d), 4),
            'nearest_gold_km': None if (gold_d is None or (isinstance(gold_d, float) and np.isnan(gold_d)))
            else round(float(gold_d) * 111.0, 1),
            'spatial_flag': flag if n_nure else 'no_nure',
            'campaign_class': cls,
            'pour_lon': None if np.isnan(pour_lon) else round(pour_lon, 6),
            'pour_lat': None if np.isnan(pour_lat) else round(pour_lat, 6),
            'pour_snap_lon': None if np.isnan(snap_lon) else round(snap_lon, 6),
            'pour_snap_lat': None if np.isnan(snap_lat) else round(snap_lat, 6),
            'pour_elev_m': None if np.isnan(pour_elev) else round(pour_elev, 1),
            'n_nure_spots': len(spots),
            'n_field_spots': len(spots),
            'catchment_ok': catch_ok,
            'catchment_area_deg2': None if np.isnan(catch_area) else round(catch_area, 4),
            'notes': ';'.join(notes),
        })

    walk = pd.DataFrame(walk_rows)
    walk['_ord'] = walk['campaign_class'].map(CLASS_ORDER)
    # Most isolated high-P drainage first inside each class.
    walk['_iso'] = -walk['nearest_gold_deg'].fillna(-1)
    walk = walk.sort_values(
        ['_ord', '_iso', 'max_p'], ascending=[True, True, False],
        na_position='last',
    ).reset_index(drop=True)
    walkable = walk['campaign_class'].isin(['expedition', 'confirm', 'watch'])
    walk['walk_rank'] = pd.Series(pd.NA, index=walk.index, dtype='Int64')
    walk.loc[walkable, 'walk_rank'] = np.arange(1, int(walkable.sum()) + 1)
    walk = walk.drop(columns=['_ord', '_iso'])

    # Rebuild spots with walk_rank (second pass, cheap)
    rank_of = dict(zip(walk['block_id'], walk['walk_rank']))
    cls_of = dict(zip(walk['block_id'], walk['campaign_class']))
    for _, blk in walk.iterrows():
        bid = int(blk['block_id'])
        in_cell = df[df['block_id'] == bid]
        if in_cell.empty:
            continue
        pool = in_cell
        if blk['catchment_ok'] and len(catch_gdf) and 'block_id' in catch_gdf.columns:
            this_c = catch_gdf[catch_gdf['block_id'] == bid]
            if len(this_c):
                joined = gpd.sjoin(
                    nure_pts, this_c[['geometry']], how='inner', predicate='within',
                )
                if len(joined):
                    pool = joined.drop(columns='index_right', errors='ignore')
        pour_series = None
        if pd.notna(blk['pour_lon']):
            pour_series = pd.Series({
                'lon': blk['pour_lon'], 'lat': blk['pour_lat'],
                'p_anomalous': blk['max_p'] if pd.isna(blk.get('mean_p')) else (
                    float(pour_df.loc[pour_df['block_id'] == bid, 'p_anomalous'].iloc[0])
                    if bid in set(pour_df['block_id']) else blk['max_p']
                ),
                'elev_m': blk['pour_elev_m'],
            })
        spots = _pick_nure_spots(pool, pour_series)
        for order, s in enumerate(spots, 1):
            is_pour = (
                pd.notna(blk['pour_lon'])
                and abs(float(s['lon']) - float(blk['pour_lon'])) < 1e-5
                and abs(float(s['lat']) - float(blk['pour_lat'])) < 1e-5
            )
            spot_rows.append({
                'walk_rank': blk['walk_rank'],
                'block_id': bid,
                'campaign_class': cls_of[bid],
                'spot_order': order,
                'role': 'pour_point' if is_pour else 'high_p_sample',
                'lon': round(float(s['lon']), 6),
                'lat': round(float(s['lat']), 6),
                'p_anomalous': round(float(s['p_anomalous']), 4) if pd.notna(s['p_anomalous']) else None,
                'elev_m': None if ('elev_m' not in s.index or pd.isna(s.get('elev_m')))
                else round(float(s['elev_m']), 1),
            })

    spots_df = pd.DataFrame(spot_rows)

    # Attach loco / campaign onto block polygons
    blocks = blocks.merge(
        walk[['block_id', 'n_nure', 'mean_p', 'max_p', 'loco_auc',
              'campaign_class', 'walk_rank', 'spatial_flag']],
        on='block_id', how='left',
    )

    walk.to_csv(out(cfg, 'tables', 'task11_catchment_walk_list.csv'), index=False)
    spots_df.to_csv(out(cfg, 'tables', 'task11_nure_spots.csv'), index=False)
    blocks.to_file(out(cfg, 'geojson', 'task11_spatial_cv_blocks_0p4deg.geojson'),
                   driver='GeoJSON')
    if len(catch_gdf):
        catch_gdf.to_file(out(cfg, 'geojson', 'task11_catchment_polygons.geojson'),
                          driver='GeoJSON')
    if not pour_df.empty:
        pg = pour_df.merge(walk[['block_id', 'campaign_class', 'walk_rank']], on='block_id')
        gpd.GeoDataFrame(
            pg,
            geometry=gpd.points_from_xy(pg['lon'], pg['lat']),
            crs='EPSG:4326',
        ).to_file(out(cfg, 'geojson', 'task11_pour_points.geojson'), driver='GeoJSON')
    if not spots_df.empty:
        gpd.GeoDataFrame(
            spots_df.copy(),
            geometry=gpd.points_from_xy(spots_df['lon'], spots_df['lat']),
            crs='EPSG:4326',
        ).to_file(out(cfg, 'geojson', 'task11_field_campaign_targets.geojson'),
                  driver='GeoJSON')

    write_gpkg_from_outputs(cfg)

    _write_summary(cfg, walk, spots_df, loco, p_is_transfer=p_is_transfer)
    pan_path = out(cfg, 'tables', 'task11_pan_locations.csv')
    pans_df = pd.read_csv(pan_path) if os.path.exists(pan_path) else pd.DataFrame()
    _draw_figure(cfg, blocks, catch_gdf, pour_df, spots_df, walk, pans_df=pans_df)
    print("Task 11 complete — fig11 walk list saved.")


def _write_summary(cfg, walk, spots_df, loco, p_is_transfer=False):
    n_cells = len(walk)
    n_occ = int((walk['n_nure'] > 0).sum())
    counts = walk['campaign_class'].value_counts().to_dict()
    claim = "This is a screen, not a gold claim. Stars are NURE; circles are pans."
    if p_is_transfer or (cfg.get('task11') or {}).get('p_is_transfer'):
        claim = (
            "P is the frozen NE Washington forest (lookalike), not a local gold model. "
            "Gold distance uses this belt's MRDS pins. Stars are NURE; circles are pans."
        )
    lines = [
        "FIELD CAMPAIGN WALK LIST",
        f"{cfg.get('study_area', {}).get('name', 'Study area')} — 0.4° cells + pour points",
        claim,
        "=" * 68,
        "",
        f"Cells on the map:     {n_cells}  (occupied by NURE: {n_occ})",
        f"Expedition (high P, >5 km from a gold pin): {counts.get('expedition', 0)}",
        f"Confirm    (high P, ≤5 km from a gold pin): {counts.get('confirm', 0)}",
        f"Watch      (moderate P):                     {counts.get('watch', 0)}",
        f"Skip       (low P or empty):                 {counts.get('skip', 0)}",
        "",
    ]
    scored_walk = walk.loc[walk['loco_auc'].notna(), 'loco_auc']
    if len(scored_walk):
        lines += [
            f"Leave-one-cell-out AUC (cells with ≥8 grabs): "
            f"{float(scored_walk.mean()):.3f}  "
            f"(min {float(scored_walk.min()):.3f}, max {float(scored_walk.max()):.3f}, "
            f"n={int(scored_walk.notna().sum())} cells)",
            "Same story as the 0.94 vs 0.62 folds — some cells transfer, some do not.",
            "",
        ]
    lines += ["WALK ORDER (expedition first, most isolated first)", ""]
    show = walk[walk['campaign_class'].isin(['expedition', 'confirm', 'watch'])]
    for _, r in show.iterrows():
        km = r.get('nearest_gold_km')
        km_bit = f"{km} km from gold  " if pd.notna(km) else ""
        lines.append(
            f"  #{int(r['walk_rank']):>2}  {r['campaign_class']:<11}  "
            f"max P={r['max_p']}  {km_bit}"
            f"pour {r['pour_lat']}, {r['pour_lon']}  "
            f"nure={int(r.get('n_nure_spots', r.get('n_field_spots', 0)))}"
        )
    if show.empty:
        lines.append("  (none — every occupied cell was skip)")
    lines += [
        "",
        "nure_spots = NURE chemistry grabs (the forest). They say this creek.",
        "pan_locations = trap-geometry pins on the D8 line. They say stand here.",
        "Pour point = where the water leaves. It is not automatically a pan pin.",
        "See task11_nure_spots.csv and task11_pan_locations.csv.",
    ]
    path = out(cfg, 'text', 'task11_field_campaign_summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  Wrote {path}")
    print('\n'.join(lines[:16]))


def _figure_from_outputs(cfg):
    """Redraw Fig 11 from written tables/geojson — no DEM loop."""
    import geopandas as gpd
    walk = pd.read_csv(out(cfg, 'tables', 'task11_catchment_walk_list.csv'))
    spots_df = _nure_spots_csv(cfg)
    pan_path = out(cfg, 'tables', 'task11_pan_locations.csv')
    pans_df = pd.read_csv(pan_path) if os.path.exists(pan_path) else pd.DataFrame()
    blocks = gpd.read_file(out(cfg, 'geojson', 'task11_spatial_cv_blocks_0p4deg.geojson'))
    catch_path = out(cfg, 'geojson', 'task11_catchment_polygons.geojson')
    pour_path = out(cfg, 'geojson', 'task11_pour_points.geojson')
    catch_gdf = gpd.read_file(catch_path) if os.path.exists(catch_path) else gpd.GeoDataFrame()
    pour_df = (
        gpd.read_file(pour_path).drop(columns='geometry', errors='ignore')
        if os.path.exists(pour_path) else pd.DataFrame()
    )
    _draw_figure(cfg, blocks, catch_gdf, pour_df, spots_df, walk, pans_df=pans_df)
    print("Task 11 figure redrawn from existing outputs.")


def _draw_figure(cfg, blocks, catch_gdf, pour_df, spots_df, walk, pans_df=None):
    _BAR_H = 3.6
    figW = _FIG_LM + MAP_W + _FIG_CG + _FIG_CW + _FIG_RM
    figH = _FIG_BM + _BAR_H + _FIG_VGAP + MAP_H + _FIG_TM
    fig = plt.figure(figsize=(figW, figH))

    map_bottom = _FIG_BM + _BAR_H + _FIG_VGAP
    ax = fig.add_axes(_ax_rect(_FIG_LM, map_bottom, MAP_W, MAP_H, figW, figH))
    cax = fig.add_axes(_ax_rect(_FIG_LM + MAP_W + _FIG_CG, map_bottom, _FIG_CW, MAP_H, figW, figH))
    axb = fig.add_axes(_ax_rect(_FIG_LM, _FIG_BM, MAP_W + _FIG_CG + _FIG_CW, _BAR_H, figW, figH))

    xmin, xmax, ymin, ymax = map_extent(cfg)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('auto')
    hillshade(cfg, ax, alpha=0.30, zorder=0)
    topo_contours(ax, cfg, color='gray', alpha=0.35)

    blk = clip_gdf_to_map(blocks, cfg)
    if 'n_nure' in blk.columns:
        blk = blk.copy()
        blk.loc[blk['n_nure'].fillna(0) < 8, 'loco_auc'] = np.nan
    scored = blk.dropna(subset=['loco_auc'])
    if len(scored):
        scored.plot(
            ax=ax, column='loco_auc', cmap='cividis', vmin=0.50, vmax=1.00,
            edgecolor='#333333', linewidth=0.45, alpha=0.55, zorder=2,
        )
        sm = plt.cm.ScalarMappable(cmap='cividis',
                                   norm=plt.Normalize(vmin=0.50, vmax=1.00))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label('Leave-one-cell-out AUC', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        cax.axis('off')
    unscored = blk[blk['loco_auc'].isna()]
    if len(unscored):
        unscored.plot(
            ax=ax, facecolor='#e8e8e8', edgecolor='#666666',
            linewidth=0.45, alpha=0.35, zorder=1,
        )

    if catch_gdf is not None and len(catch_gdf):
        cg = catch_gdf.copy()
        if 'block_id' in cg.columns:
            keep = walk[walk['campaign_class'].isin(['expedition', 'confirm', 'watch'])]
            cg = cg.merge(keep[['block_id', 'campaign_class']], on='block_id', how='inner')
        cg = clip_gdf_to_map(cg, cfg)
        if len(cg) and 'area_deg2' in cg.columns:
            cg = cg[cg['area_deg2'].fillna(0) <= CATCHMENT_AREA_CAP_DEG2]
        if len(cg):
            for cls, color in CLASS_COLOR.items():
                sub = cg[cg['campaign_class'] == cls] if 'campaign_class' in cg.columns else cg
                if len(sub):
                    sub.plot(
                        ax=ax, facecolor=color, edgecolor='#222222',
                        linewidth=0.9, alpha=0.28, zorder=3,
                    )

    rivers_with_arrows(ax, cfg, color='#7ecef4', lw=1.2, alpha=0.85, zorder=4)
    stream_path = out(cfg, 'geojson', 'task11_dem_streams.geojson')
    if os.path.exists(stream_path):
        import geopandas as gpd
        streams = clip_gdf_to_map(gpd.read_file(stream_path), cfg)
        if len(streams):
            streams.plot(ax=ax, color='#4a90b8', linewidth=0.55, alpha=0.85, zorder=4)

    if not spots_df.empty:
        field = spots_df
        if 'role' in spots_df.columns:
            field = spots_df[spots_df['role'] != 'pan']
        if len(field):
            sizes = 35 + 90 * field['p_anomalous'].fillna(0).clip(0, 1) if 'p_anomalous' in field.columns else 40
            ax.scatter(
                field['lon'], field['lat'], s=sizes, marker='*',
                c=[CLASS_COLOR.get(c, '#888') for c in field['campaign_class']],
                edgecolors='black', linewidths=0.4, zorder=6, clip_on=True,
            )
    if pans_df is not None and not pans_df.empty:
        ax.scatter(
            pans_df['lon'], pans_df['lat'], s=42, marker='o',
            facecolors='none',
            edgecolors=[CLASS_COLOR.get(c, '#111') for c in pans_df['campaign_class']],
            linewidths=1.4, zorder=8, clip_on=True,
        )

    if not pour_df.empty:
        meta = walk[['block_id', 'campaign_class', 'walk_rank']].drop_duplicates('block_id')
        pplot = pour_df.drop(
            columns=['campaign_class', 'walk_rank'], errors='ignore',
        ).merge(meta, on='block_id', how='left')
        if 'pour_snap_lon' in walk.columns:
            snap = walk[['block_id', 'pour_snap_lon', 'pour_snap_lat']].drop_duplicates(
                'block_id'
            )
            pplot = pplot.merge(snap, on='block_id', how='left')
            use = pplot['pour_snap_lon'].notna()
            pplot.loc[use, 'lon'] = pplot.loc[use, 'pour_snap_lon']
            pplot.loc[use, 'lat'] = pplot.loc[use, 'pour_snap_lat']
        ax.scatter(
            pplot['lon'], pplot['lat'], s=55, marker='v',
            c=[CLASS_COLOR.get(c, '#888') for c in pplot['campaign_class']],
            edgecolors='black', linewidths=0.6, zorder=7, clip_on=True,
        )
        label_me = pd.concat([
            pplot[pplot['campaign_class'] == 'expedition'],
            pplot[pplot['campaign_class'] == 'confirm'].nsmallest(6, 'walk_rank'),
        ]).drop_duplicates('block_id')
        if len(label_me):
            oys = _stagger_oy(label_me['lon'].tolist(), label_me['lat'].tolist())
            for (_, row), oy in zip(label_me.iterrows(), oys):
                ax.annotate(
                    f"#{int(row['walk_rank'])}",
                    (row['lon'], row['lat']),
                    xytext=(4, oy), textcoords='offset points',
                    fontsize=7, fontweight='bold', color='black',
                    clip_on=True,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')],
                )

    for s in cfg.get('sites') or []:
        ax.scatter(
            s['lon'], s['lat'], s=28, marker='D',
            facecolors='none', edgecolors='#333333', linewidths=0.7,
            zorder=5, clip_on=True,
        )

    canada_border(ax, cfg)
    north_arrow(ax, x=0.96, y=0.88, size=9)
    scale_bar(ax, cfg, length_km=50, x=0.04, y=0.08)
    locator_inset(fig, ax, cfg)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_title(
        'A.  Every 0.4-degree cell, NURE grabs (stars), and pan pins (circles)\n'
        '(cell colour = leave-one-cell-out AUC;  grey = cannot score)',
        fontsize=9,
    )

    ranked = walk[walk['campaign_class'].isin(['expedition', 'confirm', 'watch'])].copy()
    ranked = ranked.sort_values('walk_rank', ascending=True).head(15)
    ranked = ranked.sort_values('walk_rank', ascending=False)
    if ranked.empty:
        axb.text(0.5, 0.5, 'No occupied cells', ha='center', va='center')
    else:
        y_pos = np.arange(len(ranked))
        colors = [CLASS_COLOR.get(c, '#ccc') for c in ranked['campaign_class']]
        vals = ranked['max_p'].fillna(0).values
        axb.barh(y_pos, vals, color=colors, edgecolor='white', lw=0.3, height=0.75)
        labels = []
        for _, r in ranked.iterrows():
            km = r['nearest_gold_km'] if 'nearest_gold_km' in r.index else (
                r['nearest_gold_deg'] * 111 if pd.notna(r.get('nearest_gold_deg')) else None
            )
            km_s = f'{km:.0f} km' if pd.notna(km) else ''
            labels.append(f"#{int(r['walk_rank'])}  {km_s}")
        axb.set_yticks(y_pos)
        axb.set_yticklabels(labels, fontsize=8)
        axb.set_xlim(0, 1.12)
        axb.set_ylim(-0.6, len(ranked) - 0.4)
        for y, (_, r) in zip(y_pos, ranked.iterrows()):
            axb.text(float(r['max_p']) + 0.015, y, f"{float(r['max_p']):.2f}",
                     va='center', fontsize=7, color='#333333')
        axb.axvline(HIGH_P, color='gray', ls='--', lw=0.8)
        axb.set_xlabel('Max P(placer-like) in the cell', fontsize=9)
        axb.tick_params(labelsize=8)
        axb.grid(True, axis='x', alpha=0.3)
    axb.set_title(
        'B.  Top 15 drainages to walk  (red = expedition / >5 km from a gold pin; '
        'blue = confirm)',
        fontsize=9,
    )

    handles = [
        mpatches.Patch(facecolor=WONG['vermillion'], edgecolor='black',
                       label='Expedition (high P, >5 km from a gold pin)'),
        mpatches.Patch(facecolor=WONG['blue'], edgecolor='black',
                       label='Confirm (high P, standing on known gold)'),
        mpatches.Patch(facecolor=WONG['yellow'], edgecolor='black', label='Watch (moderate P)'),
        mpatches.Patch(facecolor='#e8e8e8', edgecolor='#666666', label='Skip / empty / no AUC'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='black',
               markeredgecolor='black', markersize=7, label='Pour point (outlet)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black',
               markeredgecolor='black', markersize=9, label='NURE grab (chemistry)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='black', markersize=7, label='Pan location (trap geometry)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
               markeredgecolor='#333333', markersize=6, label='12 ranked config sites'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=6.0, framealpha=0.92,
              bbox_to_anchor=(0.01, 0.99))

    watermark(fig, cfg)
    save_fig(fig, out(cfg, 'figures', 'fig11_catchment_walk_list_map.png'))


if __name__ == '__main__':
    import sys
    import yaml
    flags = {'--figure-only', '--gpkg-only'}
    args = [a for a in sys.argv[1:] if a not in flags]
    cfg_path = args[0] if args else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if '--gpkg-only' in sys.argv[1:]:
        write_gpkg_from_outputs(cfg)
    elif '--figure-only' in sys.argv[1:]:
        setup_mpl()
        _figure_from_outputs(cfg)
    else:
        run(cfg)
