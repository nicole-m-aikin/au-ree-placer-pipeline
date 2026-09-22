"""Recreational-panning overlay for the Task 11 GeoPackage.

Not a forest feature. Not AUC. Not a scrape.

People (and state pamphlets) name creeks where a pan has been, or may be,
legal and historically gold-bearing. That is a positive-only occurrence
layer. Join it to catchments. Only a GPS-grade point gets a distance to a
Task 11 pan pin. Districts stay on the map and do not count as a drainage
hit.

The seed CSVs are a cited gazetteer (gold_class=unknown). Append opt-in
form rows — especially blanks — to the same file. Do not retrain Task 9.
"""

import os
import sys

import numpy as np
import pandas as pd

from pipeline.geo_crs import utm_epsg
from pipeline.utils import bbox, out, resolve_data_path

COLUMNS = (
    'report_id',
    'name',
    'creek_name',
    'lon',
    'lat',
    'gold_class',
    'location_precision',
    'geom_confidence',
    'source_type',
    'source_cite',
    'source_url',
    'date_approx',
    'hours_or_pans',
    'public_land',
    'notes',
)

GOLD_CLASS = ('blank', 'color', 'flake', 'picker', 'nugget', 'unknown')
PRECISION = ('point', 'bar', 'reach', 'creek', 'district')
SOURCE_TYPE = ('form', 'pamphlet', 'guidebook', 'personal', 'unknown')
PUBLIC_LAND = ('yes', 'no', 'unknown')

# Creek-or-better can vote on a catchment. District is too coarse.
CATCHMENT_PRECISION = ('point', 'bar', 'reach', 'creek')
# Only a real waypoint is allowed to score a 400 m pin.
PIN_PRECISION = ('point',)
RECOVERY_CLASS = ('color', 'flake', 'picker', 'nugget')

_GOLD_ALIAS = {
    'colours': 'color', 'colors': 'color', 'colour': 'color',
    'flakes': 'flake', 'fine': 'color', 'fines': 'color',
    'pickers': 'picker', 'nuggets': 'nugget',
    'none': 'blank', 'nothing': 'blank', 'zero': 'blank',
    'na': 'unknown', 'n/a': 'unknown', '': 'unknown',
}
_PREC_ALIAS = {
    'gps': 'point', 'waypoint': 'point', 'hole': 'point',
    'gravel_bar': 'bar', 'point_bar': 'bar',
    'stretch': 'reach', 'stream': 'creek', 'river': 'creek',
    'camp': 'district', 'area': 'district',
}
_SRC_ALIAS = {
    'survey': 'form', 'opt-in': 'form', 'opt_in': 'form',
    'dnr': 'pamphlet', 'usfs': 'pamphlet', 'blm': 'pamphlet',
    'state_park': 'pamphlet', 'book': 'guidebook',
    'self': 'personal', 'field': 'personal',
}


def hobby_reports_path(cfg):
    """Config path, or data/hobby_reports/hobby_reports_{short}.csv."""
    explicit = resolve_data_path(cfg, 'hobby_reports_csv')
    if explicit:
        return explicit
    short = str((cfg.get('study_area') or {}).get('short') or '').strip()
    if not short:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, 'data', 'hobby_reports', f'hobby_reports_{short}.csv')


def _norm_token(value, allowed, aliases, default):
    raw = '' if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    key = raw.strip().lower().replace(' ', '_')
    key = aliases.get(key, key)
    if key in allowed:
        return key
    return default


def _norm_conf(value):
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return 2
    return max(1, min(5, n))


def normalize_reports(df):
    """Coerce a user/gazetteer table onto the schema. Extra columns kept."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(COLUMNS))
    out_df = df.copy()
    for col in COLUMNS:
        if col not in out_df.columns:
            out_df[col] = pd.NA
    out_df['lon'] = pd.to_numeric(out_df['lon'], errors='coerce')
    out_df['lat'] = pd.to_numeric(out_df['lat'], errors='coerce')
    out_df['gold_class'] = out_df['gold_class'].map(
        lambda v: _norm_token(v, GOLD_CLASS, _GOLD_ALIAS, 'unknown')
    )
    out_df['location_precision'] = out_df['location_precision'].map(
        lambda v: _norm_token(v, PRECISION, _PREC_ALIAS, 'district')
    )
    out_df['source_type'] = out_df['source_type'].map(
        lambda v: _norm_token(v, SOURCE_TYPE, _SRC_ALIAS, 'unknown')
    )
    out_df['public_land'] = out_df['public_land'].map(
        lambda v: _norm_token(v, PUBLIC_LAND, {}, 'unknown')
    )
    out_df['geom_confidence'] = out_df['geom_confidence'].map(_norm_conf)
    if 'report_id' not in df.columns or out_df['report_id'].isna().all():
        out_df['report_id'] = [
            'HR{0:04d}'.format(i + 1) for i in range(len(out_df))
        ]
    else:
        blank = out_df['report_id'].isna() | (out_df['report_id'].astype(str).str.strip() == '')
        if blank.any():
            out_df.loc[blank, 'report_id'] = [
                'HR{0:04d}'.format(i + 1) for i in range(int(blank.sum()))
            ]
    return out_df


def load_reports(path):
    """Read a CSV. Comment lines starting with # are ignored."""
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=list(COLUMNS))
    df = pd.read_csv(path, comment='#', skipinitialspace=True)
    if df.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    return normalize_reports(df)


def clip_to_bbox(df, cfg):
    """Drop rows with no coords or outside the study bbox (not map padding)."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(COLUMNS))
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    keep = (
        df['lon'].notna() & df['lat'].notna()
        & (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
    )
    return df.loc[keep].copy()


def reports_gdf(df):
    import geopandas as gpd

    if df is None or len(df) == 0:
        return gpd.GeoDataFrame(columns=list(COLUMNS) + ['geometry'], crs='EPSG:4326')
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326',
    )
    return gdf


def join_catchments(gdf, catchments):
    """Point-in-polygon onto walkable catchments. Districts do not get a hit."""
    import geopandas as gpd

    if gdf is None or len(gdf) == 0:
        return gdf
    out_gdf = gdf.copy()
    for col in ('block_id', 'campaign_class', 'walk_rank'):
        if col not in out_gdf.columns:
            out_gdf[col] = pd.NA
    out_gdf['catchment_hit'] = False
    if catchments is None or len(catchments) == 0:
        return out_gdf
    poly = catchments.copy()
    if poly.crs is None:
        poly = poly.set_crs('EPSG:4326')
    elif '4326' not in str(poly.crs):
        poly = poly.to_crs('EPSG:4326')
    keep_cols = [c for c in ('block_id', 'campaign_class', 'walk_rank', 'geometry')
                 if c in poly.columns]
    poly = poly[keep_cols].drop_duplicates()
    left = out_gdf.drop(
        columns=['index_right', 'block_id', 'campaign_class', 'walk_rank'],
        errors='ignore',
    )
    joined = gpd.sjoin(left, poly, how='left', predicate='intersects')
    # A point on a shared edge can match two polygons; keep the first.
    if 'index_right' in joined.columns:
        joined = joined.drop(columns='index_right')
    joined = joined[~joined.index.duplicated(keep='first')]
    for col in ('block_id', 'campaign_class', 'walk_rank'):
        if col in joined.columns:
            out_gdf[col] = joined.reindex(out_gdf.index)[col].values
    in_poly = out_gdf['block_id'].notna()
    eligible = out_gdf['location_precision'].isin(CATCHMENT_PRECISION)
    # Skip cells are not walk targets. Missing class still counts — the
    # caller decides which polygons to pass in.
    skip = out_gdf['campaign_class'].isin(('skip',))
    out_gdf['catchment_hit'] = in_poly & eligible & ~skip
    return out_gdf


def join_pan_pins(gdf, pans):
    """Nearest Task 11 pin, in metres, only for precision=point."""
    if gdf is None or len(gdf) == 0:
        return gdf
    out_gdf = gdf.copy()
    out_gdf['nearest_pan_pin_m'] = np.nan
    out_gdf['pin_scored'] = False
    pin_ok = out_gdf['location_precision'].isin(PIN_PRECISION)
    if pans is None or len(pans) == 0 or not pin_ok.any():
        return out_gdf
    pts = out_gdf.loc[pin_ok]
    lon0 = float(np.nanmean(np.concatenate([
        pts.geometry.x.values, pans.geometry.x.values,
    ])))
    lat0 = float(np.nanmean(np.concatenate([
        pts.geometry.y.values, pans.geometry.y.values,
    ])))
    crs = utm_epsg(lon0, lat0)
    a = pts.to_crs(crs)
    b = pans.to_crs(crs)
    bxy = np.column_stack([b.geometry.x.values, b.geometry.y.values])
    dist = []
    for geom in a.geometry:
        d = np.hypot(bxy[:, 0] - geom.x, bxy[:, 1] - geom.y)
        dist.append(float(np.min(d)) if len(d) else np.nan)
    out_gdf.loc[a.index, 'nearest_pan_pin_m'] = dist
    out_gdf.loc[a.index, 'pin_scored'] = True
    return out_gdf


def hit_rate_table(gdf, walk=None):
    """One-row catchment hit-rate. This is not ROC-AUC."""
    empty = {
        'n_reports_in_bbox': 0,
        'n_catchment_eligible': 0,
        'n_catchment_hit': 0,
        'n_expedition_hit': 0,
        'n_confirm_hit': 0,
        'n_watch_hit': 0,
        'n_no_catchment': 0,
        'n_district': 0,
        'n_recovery': 0,
        'n_blank': 0,
        'n_unknown': 0,
        'n_pin_scored': 0,
        'median_pin_m': np.nan,
        'n_walkable_catchments': 0,
        'n_walkable_with_hit': 0,
    }
    row = dict(empty)
    if walk is not None and len(walk):
        walkable = walk['campaign_class'].isin(('expedition', 'confirm', 'watch'))
        row['n_walkable_catchments'] = int(walkable.sum())
    if gdf is None or len(gdf) == 0:
        return pd.DataFrame([row])
    row['n_reports_in_bbox'] = int(len(gdf))
    row['n_catchment_eligible'] = int(gdf['location_precision'].isin(CATCHMENT_PRECISION).sum())
    hit = gdf['catchment_hit'] == True  # noqa: E712
    row['n_catchment_hit'] = int(hit.sum())
    cls = gdf.loc[hit, 'campaign_class']
    row['n_expedition_hit'] = int((cls == 'expedition').sum())
    row['n_confirm_hit'] = int((cls == 'confirm').sum())
    row['n_watch_hit'] = int((cls == 'watch').sum())
    row['n_no_catchment'] = int(gdf['block_id'].isna().sum())
    row['n_district'] = int((gdf['location_precision'] == 'district').sum())
    row['n_recovery'] = int(gdf['gold_class'].isin(RECOVERY_CLASS).sum())
    row['n_blank'] = int((gdf['gold_class'] == 'blank').sum())
    row['n_unknown'] = int((gdf['gold_class'] == 'unknown').sum())
    scored = gdf['pin_scored'] == True  # noqa: E712
    row['n_pin_scored'] = int(scored.sum())
    if scored.any():
        row['median_pin_m'] = float(np.nanmedian(gdf.loc[scored, 'nearest_pan_pin_m']))
    if row['n_walkable_catchments'] and 'block_id' in gdf.columns:
        hit_ids = set(gdf.loc[hit, 'block_id'].dropna().tolist())
        row['n_walkable_with_hit'] = len(hit_ids)
    return pd.DataFrame([row])


def format_hit_rate_text(cfg, rates):
    r = rates.iloc[0].to_dict() if rates is not None and len(rates) else {}
    name = (cfg.get('study_area') or {}).get('name', 'Study area')
    lines = [
        'HOBBY / PAMPHLET PAN OVERLAY',
        f'{name} — catchment hit-rate, not AUC',
        '=' * 68,
        '',
        'This layer does not retrain the forest. Gazetteer rows are unknown,',
        'not recoveries. Blanks from the opt-in form are the 0-class we lack.',
        'Creek/reach/bar/point can hit a catchment. District cannot.',
        'Only precision=point gets nearest_pan_pin_m.',
        '',
        f"Reports in bbox:              {int(r.get('n_reports_in_bbox') or 0)}",
        f"Catchment-eligible:           {int(r.get('n_catchment_eligible') or 0)}",
        f"Joined a walkable catchment:  {int(r.get('n_catchment_hit') or 0)}",
        f"  expedition hits:            {int(r.get('n_expedition_hit') or 0)}",
        f"  confirm hits:               {int(r.get('n_confirm_hit') or 0)}",
        f"  watch hits:                 {int(r.get('n_watch_hit') or 0)}",
        f"No catchment:                 {int(r.get('n_no_catchment') or 0)}",
        f"District (mapped, not a hit): {int(r.get('n_district') or 0)}",
        f"Recoveries (color or better): {int(r.get('n_recovery') or 0)}",
        f"Blanks:                       {int(r.get('n_blank') or 0)}",
        f"Unknown / gazetteer:          {int(r.get('n_unknown') or 0)}",
        f"Pin-scored (precision=point): {int(r.get('n_pin_scored') or 0)}",
    ]
    med = r.get('median_pin_m')
    if med is not None and pd.notna(med):
        lines.append(f'  median distance to pin:     {med:.0f} m')
    lines += [
        f"Walkable catchments:          {int(r.get('n_walkable_catchments') or 0)}",
        f"Walkable with ≥1 eligible hit:{int(r.get('n_walkable_with_hit') or 0)}",
        '',
        'Do not quote this as ROC-AUC. People post flakes, not blanks.',
        'Append rows via data/hobby_reports/OPT_IN_FORM.txt — do not scrape.',
    ]
    return '\n'.join(lines) + '\n'


def attach_walk_to_catchments(catchments, walk):
    """Catchment GeoJSON is hydrology only. Class / rank live on the walk list."""
    if catchments is None or walk is None or len(catchments) == 0 or len(walk) == 0:
        return catchments
    if 'block_id' not in catchments.columns or 'block_id' not in walk.columns:
        return catchments
    meta_cols = [c for c in ('block_id', 'campaign_class', 'walk_rank', 'max_p',
                             'nearest_gold_km') if c in walk.columns]
    meta = walk[meta_cols].drop_duplicates('block_id')
    out_c = catchments.drop(
        columns=[c for c in meta_cols if c != 'block_id' and c in catchments.columns],
        errors='ignore',
    )
    return out_c.merge(meta, on='block_id', how='left')


def _load_walk_layers(cfg):
    """Best-effort load of Task 11 polygons / pins already on disk."""
    import geopandas as gpd

    catch = None
    pans = None
    walk = None
    catch_path = out(cfg, 'geojson', 'task11_catchment_polygons.geojson')
    if os.path.exists(catch_path):
        catch = gpd.read_file(catch_path)
    pan_path = out(cfg, 'tables', 'task11_pan_locations.csv')
    if os.path.exists(pan_path):
        pdf = pd.read_csv(pan_path)
        if {'lon', 'lat'}.issubset(pdf.columns) and len(pdf):
            pans = gpd.GeoDataFrame(
                pdf, geometry=gpd.points_from_xy(pdf['lon'], pdf['lat']),
                crs='EPSG:4326',
            )
        elif 'geometry' in pdf.columns:
            pans = gpd.GeoDataFrame(pdf, crs='EPSG:4326')
    walk_path = out(cfg, 'tables', 'task11_catchment_walk_list.csv')
    if os.path.exists(walk_path):
        walk = pd.read_csv(walk_path)
    return catch, pans, walk


def build_hobby_reports(cfg, catchments=None, pans=None, walk=None):
    """Load → clip → join → write CSV / hit-rate / GeoJSON. Returns a GeoDataFrame."""
    path = hobby_reports_path(cfg)
    raw = load_reports(path)
    clipped = clip_to_bbox(raw, cfg)
    gdf = reports_gdf(clipped)
    catchments = attach_walk_to_catchments(catchments, walk)
    gdf = join_catchments(gdf, catchments)
    gdf = join_pan_pins(gdf, pans)
    rates = hit_rate_table(gdf, walk=walk)

    table_path = out(cfg, 'tables', 'task11_hobby_reports.csv')
    rate_path = out(cfg, 'tables', 'task11_hobby_hit_rate.csv')
    text_path = out(cfg, 'text', 'task11_hobby_reports_summary.txt')
    gj_path = out(cfg, 'geojson', 'task11_hobby_reports.geojson')
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    os.makedirs(os.path.dirname(gj_path), exist_ok=True)

    table = gdf.drop(columns='geometry', errors='ignore')
    table.to_csv(table_path, index=False)
    rates.to_csv(rate_path, index=False)
    text = format_hit_rate_text(cfg, rates)
    with open(text_path, 'w') as f:
        f.write(text)
    if gdf is not None and len(gdf):
        gdf.to_file(gj_path, driver='GeoJSON')
    elif os.path.exists(gj_path):
        os.remove(gj_path)
    print(f'  Wrote {table_path} ({len(gdf)} reports)')
    print(text.splitlines()[0] if text else '')
    return gdf


def apply_hobby_counts(walk, hobby):
    """Add n_hobby_hit to the walk list (eligible catchment hits only)."""
    if walk is None or len(walk) == 0:
        return walk
    out_walk = walk.copy()
    out_walk['n_hobby_hit'] = 0
    if hobby is None or len(hobby) == 0 or 'catchment_hit' not in hobby.columns:
        return out_walk
    hits = hobby.loc[hobby['catchment_hit'] == True, 'block_id']  # noqa: E712
    if hits.empty:
        return out_walk
    counts = hits.value_counts()
    out_walk['n_hobby_hit'] = out_walk['block_id'].map(counts).fillna(0).astype(int)
    return out_walk


def _gpkg_paths(cfg):
    from pipeline.task11_field_campaign import GPKG_NAME, clean_gpkg_copy

    paths = [
        os.path.join(cfg['outputs_dir'], 'gis', GPKG_NAME),
        out(cfg, 'geojson', GPKG_NAME),
    ]
    try:
        paths.append(clean_gpkg_copy(cfg))
    except Exception:
        pass
    return [p for p in paths if p and os.path.exists(p)]


def append_hobby_layer(cfg, hobby):
    """Replace hobby_reports on existing GeoPackages. Does not rebuild streams."""
    if hobby is None or len(hobby) == 0:
        return []
    import pyogrio
    from pipeline.task11_field_campaign import _agol_ready, _strip_gpkg_tile_tables
    from pipeline.task11_qgis_styles import add_rank_label, embed_qgis_styles

    ready = add_rank_label(_agol_ready(hobby))
    if ready.empty:
        return []
    written = []
    for path in _gpkg_paths(cfg):
        try:
            pyogrio.write_dataframe(
                ready, path, layer='hobby_reports', driver='GPKG',
                geometry_type='Point', append=True,
                layer_options={'SPATIAL_INDEX': 'NO', 'OVERWRITE': 'YES'},
            )
            _strip_gpkg_tile_tables(path)
            embed_qgis_styles(path)
            written.append(path)
            print(f'  Updated hobby_reports in {path} ({len(ready)} features)')
        except Exception as e:
            print(f'  Could not update {path}: {e}')
    return written


def run(cfg):
    catch, pans, walk = _load_walk_layers(cfg)
    hobby = build_hobby_reports(cfg, catchments=catch, pans=pans, walk=walk)
    if walk is not None and len(walk) and hobby is not None:
        updated = apply_hobby_counts(walk, hobby)
        walk_path = out(cfg, 'tables', 'task11_catchment_walk_list.csv')
        if os.path.exists(walk_path):
            updated.to_csv(walk_path, index=False)
    append_hobby_layer(cfg, hobby)


if __name__ == '__main__':
    import yaml

    argv = [a for a in sys.argv[1:] if a != '--config']
    cfg_path = argv[0] if argv else 'configs/ne_washington/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    run(cfg)
