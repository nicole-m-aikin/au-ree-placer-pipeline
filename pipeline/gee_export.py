"""Merge every region's county ranking into statewide layers for Earth Engine.

Usage:
  python -m pipeline.gee_export                                  # local bundle
  python -m pipeline.gee_export --upload --project placer-picker-511303

Local bundle (statewide/outputs/):
  tables/gee_pans.csv, geojson/gee_pans.geojson
  tables/county_coverage.csv
  geojson/gee_walks.geojson, gee_mrds_gold.geojson, gee_nure.geojson,
  gee_claims_active.geojson, gee_claims_closed.geojson, gee_padus41.geojson

--upload writes under projects/<project>/assets/<asset_root>/:
  pans, walks, mrds_gold, nure, claims_active, claims_closed, padus41, coverage
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.county_rank import rank_within_county, to_km_xy
from pipeline.ml_artifacts import REPO_ROOT

STATEWIDE_DIR = REPO_ROOT / 'statewide' / 'outputs'
DEDUPE_M = 150.0
CHUNK = 2500
PADUS_MANAGERS = frozenset({'BLM', 'USFS', 'SPR'})
GEE_COLS = [
    'pin_id', 'region', 'county', 'county_geoid', 'lon', 'lat', 'elev_m',
    'creek_name', 'seed_name', 'trap_score', 'why', 'nearest_gold_km',
    'n_gold_mrds_2km', 'upstream_gold_km', 'n_upstream_gold',
    'channel_km', 'on_mined', 'mined_name', 'chem_c', 'evidence_score',
    'access_type', 'manager_name', 'claim_status', 'on_fee_land', 'rules_note',
    'rec_score', 'rec_rank_in_county',
    'federal_manager', 'withdrawal_hint', 'caution_note', 'recent_closed_claim',
    'stake_status', 'stake_score', 'stake_rank_in_county',
    # Phase 1 EE sample flags (present after county_rank with gee_sample on)
    'water_regime', 'water_pct', 'lith_name', 'lith_favor', 'nlcd_name',
    'land_filter', 'road_m', 'fire_year', 'has_lidar_1m', 'slope10_deg',
    'best_months',
]


def region_configs(state='CA'):
    found = []
    for path in sorted(glob.glob(str(REPO_ROOT / 'configs' / '*' / 'config.yaml'))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        st = str((cfg.get('study_area') or {}).get('state')
                 or (cfg.get('data') or {}).get('sgmc_state') or '').upper()
        if 'county_rank' in cfg and st == state:
            found.append(path)
    return found


def _load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_region_pans(config_paths):
    frames = []
    for path in config_paths:
        cfg = _load_cfg(path)
        csv = REPO_ROOT / cfg['outputs_dir'] / 'tables' / 'county_rank_pans.csv'
        if not csv.exists():
            print(f'  skip {path}: no {csv.name} yet (run pipeline.county_rank)')
            continue
        frames.append(pd.read_csv(csv, dtype={'county_geoid': str}))
    if not frames:
        return pd.DataFrame(columns=GEE_COLS)
    return pd.concat(frames, ignore_index=True)


def dedupe_across_regions(pans, radius_m=DEDUPE_M):
    if len(pans) < 2 or pans['region'].nunique() < 2:
        return pans.reset_index(drop=True)
    from scipy.spatial import cKDTree

    work = pans.sort_values('evidence_score', ascending=False, kind='mergesort')
    work = work.reset_index(drop=True)
    xy = to_km_xy(work['lon'], work['lat'], float(work['lat'].mean()))
    tree = cKDTree(xy)
    region = work['region'].to_numpy()
    drop = np.zeros(len(work), dtype=bool)
    for i in range(len(work)):
        if drop[i]:
            continue
        for j in tree.query_ball_point(xy[i], radius_m / 1000.0):
            if j > i and region[j] != region[i]:
                drop[j] = True
    return work.loc[~drop].reset_index(drop=True)


def rerank_statewide(pans):
    pans = pans.copy()
    pans['county'] = pans['county'].fillna('').astype(str)
    rec_ok = pans['rec_score'].notna()
    stake_ok = pans['stake_score'].notna()
    pans = rank_within_county(pans, rec_ok, 'rec_score', 'rec_rank_in_county')
    pans = rank_within_county(pans, stake_ok, 'stake_score', 'stake_rank_in_county')
    return pans


def county_coverage(pans, counties):
    rows = []
    for _, c in counties.sort_values('NAME').iterrows():
        sub = pans[pans['county'] == c['NAME']]
        rows.append({
            'county': c['NAME'],
            'county_geoid': c['GEOID'],
            'status': 'built' if len(sub) else 'not_built',
            'regions': ';'.join(sorted(sub['region'].unique())) if len(sub) else '',
            'n_pins': int(len(sub)),
            'n_recreational': int(sub['rec_score'].notna().sum()) if len(sub) else 0,
            'n_stakeable': int(sub['stake_score'].notna().sum()) if len(sub) else 0,
            'n_with_nure': int(sub['chem_c'].notna().sum()) if len(sub) else 0,
            'n_land_drop': (
                int((sub['land_filter'] == 'drop').sum())
                if len(sub) and 'land_filter' in sub.columns else 0
            ),
        })
    return pd.DataFrame(rows)


def _read_gdf(path):
    import geopandas as gpd

    if not path or not Path(path).exists() or Path(path).stat().st_size < 40:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs('EPSG:4326')
    elif '4326' not in str(g.crs):
        g = g.to_crs('EPSG:4326')
    return g


def _rel(cfg, key):
    p = (cfg.get('data') or {}).get(key)
    if not p:
        return None
    return str(REPO_ROOT / p) if not os.path.isabs(p) else p


def collect_walks(config_paths):
    import geopandas as gpd

    parts = []
    for path in config_paths:
        cfg = _load_cfg(path)
        short = (cfg.get('study_area') or {}).get('short') or 'study'
        gpath = REPO_ROOT / cfg['outputs_dir'] / 'geojson' / 'county_rank_walks.geojson'
        g = _read_gdf(gpath)
        if len(g) == 0:
            continue
        g = g.copy()
        g['region'] = short
        parts.append(g)
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')


def collect_mrds(config_paths):
    import geopandas as gpd

    parts = []
    for path in config_paths:
        cfg = _load_cfg(path)
        short = (cfg.get('study_area') or {}).get('short') or 'study'
        g = _read_gdf(_rel(cfg, 'mrds_geojson'))
        if len(g) == 0:
            continue
        if 'target_commodity' in g.columns:
            tc = g['target_commodity'].fillna('')
            g = g[tc.str.contains('Gold') & ~tc.str.contains('Rare Earth')]
        g = g.copy()
        g['region'] = short
        keep = [c for c in (
            'dep_id', 'site_name', 'dev_stat', 'code_list', 'target_commodity',
            'lon', 'lat', 'region', 'geometry',
        ) if c in g.columns]
        parts.append(g[keep])
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')
    if 'dep_id' in out.columns:
        out = out.drop_duplicates('dep_id', keep='first')
    return out


def collect_nure(config_paths):
    import geopandas as gpd

    from pipeline.utils import load_nure

    parts = []
    for path in config_paths:
        cfg = _load_cfg(path)
        short = (cfg.get('study_area') or {}).get('short') or 'study'
        npath = _rel(cfg, 'nure_csv')
        if not npath or not os.path.exists(npath):
            continue
        df = load_nure({'data': {'nure_csv': npath}}).dropna(subset=['lat', 'lon'])
        df = df.copy()
        df['region'] = short
        cols = ['lon', 'lat', 'region'] + [
            c for c in ('Au', 'As', 'Th', 'Ce', 'La', 'U', 'lab_id') if c in df.columns
        ]
        parts.append(df[cols])
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=['lon', 'lat'], keep='first')
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326',
    )


def collect_claims(config_paths, closed=False):
    import geopandas as gpd

    from pipeline.fetch_region import mlrs_closed_path
    from pipeline.task11_land_access import mlrs_path

    parts = []
    for path in config_paths:
        cfg = _load_cfg(path)
        short = (cfg.get('study_area') or {}).get('short') or 'study'
        cpath = mlrs_closed_path(cfg) if closed else mlrs_path(cfg)
        g = _read_gdf(cpath)
        if len(g) == 0:
            continue
        g = g.copy()
        g['region'] = short
        g['claim_layer'] = 'closed' if closed else 'active'
        keep = [c for c in (
            'OBJECTID', 'ID', 'CSE_NAME', 'CSE_DISP', 'BLM_PROD', 'CSE_NR',
            'LEG_CSE_NR', 'region', 'claim_layer', 'geometry',
        ) if c in g.columns]
        parts.append(g[keep])
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')
    if 'ID' in out.columns:
        out = out.drop_duplicates('ID', keep='first')
    elif 'CSE_NR' in out.columns:
        out = out.drop_duplicates('CSE_NR', keep='first')
    return out


def _polygonal(geom):
    """Polygon or MultiPolygon, or None. Earth Engine rejects empty geometry."""
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == 'GeometryCollection':
        bits = [
            p for p in geom.geoms
            if p.geom_type in ('Polygon', 'MultiPolygon') and not p.is_empty
        ]
        geom = unary_union(bits) if bits else None
    if geom is None or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
        return None
    return geom


def _simplify_for_ee(gdf, tol_deg=0.002):
    """Drop vertices so large fee polygons fit the EE table-export payload.

    0.002° ≈ 200 m — coarse but fine for a map overlay. Claim / access gates
    still use the local full-resolution PAD-US cache. Self-intersections from
    that simplify become empty geometry on an Earth Engine merge, so only
    valid polygons are kept.
    """
    if gdf is None or len(gdf) == 0:
        return gdf
    out = gdf.copy()
    out['geometry'] = out.geometry.simplify(tol_deg, preserve_topology=True)
    out['geometry'] = out.geometry.apply(_polygonal)
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    return out


def collect_padus41(config_paths):
    """Fee BLM / USFS / state-park polygons from the pipeline's PAD-US 4.1 caches."""
    import geopandas as gpd

    from pipeline.task11_land_access import padus_path

    parts = []
    for path in config_paths:
        cfg = _load_cfg(path)
        short = (cfg.get('study_area') or {}).get('short') or 'study'
        g = _read_gdf(padus_path(cfg))
        if len(g) == 0:
            continue
        mang = g.get('Mang_Name', pd.Series('', index=g.index)).fillna('').astype(str)
        cat = g.get('Category', pd.Series('', index=g.index)).fillna('').astype(str)
        keep = g[(cat == 'Fee') & mang.isin(PADUS_MANAGERS)].copy()
        if keep.empty:
            continue
        keep['region'] = short
        cols = [c for c in (
            'Unit_Nm', 'Mang_Name', 'Own_Name', 'Des_Tp', 'Pub_Access', 'GAP_Sts',
            'State_Nm', 'Category', 'region', 'geometry',
        ) if c in keep.columns]
        parts.append(keep[cols])
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')
    out['_wkb'] = out.geometry.apply(lambda g: g.wkb if g is not None else b'')
    out = out.drop_duplicates('_wkb').drop(columns='_wkb')
    return _simplify_for_ee(out)


def build_bundle(config_paths, state_fips='06', out_dir=STATEWIDE_DIR):
    import geopandas as gpd

    from pipeline.fetch_region import counties_path

    pans = load_region_pans(config_paths)
    if pans.empty:
        raise RuntimeError('No region has county_rank output yet.')
    n_in = len(pans)
    pans = rerank_statewide(dedupe_across_regions(pans))
    cols = [c for c in GEE_COLS if c in pans.columns]
    pans = pans[cols].sort_values(
        ['county', 'rec_rank_in_county', 'stake_rank_in_county'],
        na_position='last',
    )
    for sub in ('tables', 'geojson'):
        os.makedirs(out_dir / sub, exist_ok=True)
    pans.to_csv(out_dir / 'tables' / 'gee_pans.csv', index=False)
    gpd.GeoDataFrame(
        pans, geometry=gpd.points_from_xy(pans['lon'], pans['lat']), crs='EPSG:4326',
    ).to_file(out_dir / 'geojson' / 'gee_pans.geojson', driver='GeoJSON')

    layers = {
        'walks': collect_walks(config_paths),
        'mrds_gold': collect_mrds(config_paths),
        'nure': collect_nure(config_paths),
        'claims_active': collect_claims(config_paths, closed=False),
        'claims_closed': collect_claims(config_paths, closed=True),
        'padus41': collect_padus41(config_paths),
    }
    for name, gdf in layers.items():
        dest = out_dir / 'geojson' / f'gee_{name}.geojson'
        if len(gdf):
            gdf.to_file(dest, driver='GeoJSON')
            print(f'  Wrote {len(gdf)} → {dest.name}')
        else:
            print(f'  (no features for {name})')

    cov = None
    cpath = counties_path(state_fips)
    if os.path.exists(cpath):
        cov = county_coverage(pans, gpd.read_file(cpath))
        cov.to_csv(out_dir / 'tables' / 'county_coverage.csv', index=False)
        # Point-per-county for the EE coverage asset (status table)
        built = cov[cov['status'] == 'built']
        print(f'  Counties built: {len(built)}/{len(cov)} — ' + ', '.join(built['county']))
        cg = gpd.read_file(cpath)[['GEOID', 'NAME', 'geometry']].rename(
            columns={'GEOID': 'county_geoid', 'NAME': 'county'})
        cov_g = cg.merge(cov.drop(columns=[], errors='ignore'), on=['county', 'county_geoid'],
                         how='left')
        cov_g.to_file(out_dir / 'geojson' / 'gee_coverage.geojson', driver='GeoJSON')
        layers['coverage'] = cov_g

    print(f'  Statewide pins: {len(pans)} (from {n_in} before cross-region dedupe)')
    print(f'  Wrote {out_dir / "tables" / "gee_pans.csv"}')
    return pans, layers


# ── Earth Engine upload ──────────────────────────────────────────────────────

def _clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if v is pd.NA:
        return None
    if isinstance(v, str):
        return v
    return v


def feature_dicts_from_gdf(gdf, prop_cols=None):
    """GeoJSON features with JSON-safe properties (geometry from the gdf)."""
    from shapely.geometry import mapping

    if gdf is None or len(gdf) == 0:
        return []
    work = gdf.copy()
    if prop_cols is None:
        prop_cols = [c for c in work.columns if c != 'geometry']
    keep = [c for c in prop_cols if c in work.columns]
    feats = []
    for _, row in work.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        props = {c: _clean(row[c]) for c in keep}
        feats.append({
            'type': 'Feature',
            'geometry': mapping(geom),
            'properties': props,
        })
    return feats


def feature_dicts(pans):
    feats = []
    for rec in pans.to_dict(orient='records'):
        props = {k: _clean(v) for k, v in rec.items()}
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [props['lon'], props['lat']]},
            'properties': props,
        })
    return feats


def _ensure_folder(ee, folder):
    try:
        ee.data.getAsset(folder)
    except Exception:
        ee.data.createAsset({'type': 'FOLDER'}, folder)


def _delete_if_exists(ee, asset_id):
    try:
        ee.data.getAsset(asset_id)
    except Exception:
        return
    ee.data.deleteAsset(asset_id)


def _wait(tasks, poll_s=10):
    while True:
        states = [t.status()['state'] for t in tasks]
        if all(s in ('COMPLETED', 'FAILED', 'CANCELLED') for s in states):
            bad = [t.status() for t in tasks if t.status()['state'] != 'COMPLETED']
            if bad:
                raise RuntimeError(f'Earth Engine export failed: {bad}')
            return
        print(f'    tasks: {states}', flush=True)
        time.sleep(poll_s)


def upload_features(feats, project, asset_id, description, chunk=CHUNK):
    """Export a list of GeoJSON features to an Earth Engine table asset."""
    import ee

    if not feats:
        print(f'  skip upload {asset_id}: empty')
        return None
    ee.Initialize(project=project)
    folder = '/'.join(asset_id.split('/')[:-1])
    _ensure_folder(ee, folder)
    chunks = [feats[i:i + chunk] for i in range(0, len(feats), chunk)]
    if len(chunks) == 1:
        _delete_if_exists(ee, asset_id)
        task = ee.batch.Export.table.toAsset(
            ee.FeatureCollection(chunks[0]), description=description, assetId=asset_id)
        task.start()
        _wait([task])
        return asset_id
    parts, tasks = [], []
    for k, ch in enumerate(chunks):
        pid = f'{asset_id}_part_{k:03d}'
        _delete_if_exists(ee, pid)
        t = ee.batch.Export.table.toAsset(
            ee.FeatureCollection(ch), description=f'{description}_{k:03d}', assetId=pid)
        t.start()
        parts.append(pid)
        tasks.append(t)
    _wait(tasks)
    _delete_if_exists(ee, asset_id)
    merged = ee.FeatureCollection([ee.FeatureCollection(p) for p in parts]).flatten()
    t = ee.batch.Export.table.toAsset(merged, description=f'{description}_merge',
                                      assetId=asset_id)
    t.start()
    _wait([t])
    for p in parts:
        _delete_if_exists(ee, p)
    return asset_id


def _publish(ee, asset_id):
    acl = ee.data.getAssetAcl(asset_id)
    acl['all_users_can_read'] = True
    ee.data.setAssetAcl(asset_id, acl)
    print(f'  public read {asset_id}', flush=True)


def upload(pans, layers, project, asset_root='placer'):
    """Upload pans + side layers. Returns {name: asset_id}."""
    import ee

    ee.Initialize(project=project)
    folder = f'projects/{project}/assets/{asset_root}'
    _ensure_folder(ee, folder)
    written = {}
    pans_id = f'{folder}/pans'
    upload_features(feature_dicts(pans), project, pans_id, 'placer_pans')
    written['pans'] = pans_id
    print(f'  Earth Engine asset ready: {pans_id}')
    _publish(ee, pans_id)

    for name, gdf in layers.items():
        if gdf is None or len(gdf) == 0:
            continue
        aid = f'{folder}/{name}'
        # Smaller chunks for polygon-heavy layers (PAD-US fee polygons are huge).
        chunk = 5 if name == 'padus41' else (
            200 if name in ('claims_active', 'claims_closed', 'walks') else CHUNK
        )
        feats = feature_dicts_from_gdf(gdf)
        print(f'  Uploading {name}: {len(feats)} features …', flush=True)
        try:
            upload_features(feats, project, aid, f'placer_{name}', chunk=chunk)
            written[name] = aid
            print(f'  Earth Engine asset ready: {aid}')
            _publish(ee, aid)
        except Exception as exc:
            print(f'  FAILED {name}: {exc}', flush=True)
    return written


def main():
    parser = argparse.ArgumentParser(description='Statewide county-ranking bundle for GEE')
    parser.add_argument('--configs', nargs='*', default=None)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--project', default=os.environ.get('EE_PROJECT'),
                        help='Google Cloud project registered for Earth Engine')
    parser.add_argument('--asset-root', default='placer')
    args = parser.parse_args()
    configs = args.configs or region_configs('CA')
    print(f'  Regions: {", ".join(os.path.relpath(p, REPO_ROOT) for p in configs)}')
    pans, layers = build_bundle(configs)
    if args.upload:
        if not args.project:
            parser.error('--upload needs --project (or EE_PROJECT)')
        written = upload(pans, layers, args.project, args.asset_root)
        print(f'  Uploaded {len(written)} assets under '
              f'projects/{args.project}/assets/{args.asset_root}/')


if __name__ == '__main__':
    main()
