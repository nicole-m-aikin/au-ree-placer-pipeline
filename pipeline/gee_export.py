"""Merge every region's county ranking into one statewide layer for Earth Engine.

Usage:
  python -m pipeline.gee_export                                  # bundle only
  python -m pipeline.gee_export --upload --project my-ee-project # bundle + asset

Bundle (statewide/outputs/):
  tables/gee_pans.csv         one row per pin with longitude/latitude columns —
                              Code Editor → Assets → NEW → CSV auto-detects the
                              points, so this works without the Python API
  geojson/gee_pans.geojson
  tables/county_coverage.csv  every county in the state: built or not, regions,
                              pin counts, how many pins have NURE chemistry

Regions overlap at their edges. Pins from different regions within
DEDUPE_M of each other keep the higher evidence_score, then both ranks are
recomputed county-wide so a county split across two regions has one list.

--upload writes projects/<project>/assets/<asset_root>/pans with
Export.table.toAsset from in-memory features (no Cloud Storage bucket).
Large bundles go up in parts and are merged server-side.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import yaml

from pipeline.county_rank import rank_within_county, to_km_xy
from pipeline.ml_artifacts import REPO_ROOT

STATEWIDE_DIR = REPO_ROOT / 'statewide' / 'outputs'
DEDUPE_M = 150.0
CHUNK = 2500
GEE_COLS = [
    'pin_id', 'region', 'county', 'county_geoid', 'lon', 'lat', 'elev_m',
    'creek_name', 'seed_name', 'trap_score', 'why', 'nearest_gold_km',
    'n_gold_mrds_2km', 'chem_c', 'evidence_score',
    'access_type', 'manager_name', 'claim_status', 'on_fee_land', 'rules_note',
    'rec_score', 'rec_rank_in_county',
    'federal_manager', 'withdrawal_hint', 'caution_note', 'recent_closed_claim',
    'stake_status', 'stake_score', 'stake_rank_in_county',
]


def region_configs(state='CA'):
    """Configs with a county_rank section for this state, sorted by path."""
    found = []
    for path in sorted(glob.glob(str(REPO_ROOT / 'configs' / '*' / 'config.yaml'))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        st = str((cfg.get('study_area') or {}).get('state')
                 or (cfg.get('data') or {}).get('sgmc_state') or '').upper()
        if 'county_rank' in cfg and st == state:
            found.append(path)
    return found


def load_region_pans(config_paths):
    frames = []
    for path in config_paths:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        csv = REPO_ROOT / cfg['outputs_dir'] / 'tables' / 'county_rank_pans.csv'
        if not csv.exists():
            print(f'  skip {path}: no {csv.name} yet (run pipeline.county_rank)')
            continue
        frames.append(pd.read_csv(csv, dtype={'county_geoid': str}))
    if not frames:
        return pd.DataFrame(columns=GEE_COLS)
    return pd.concat(frames, ignore_index=True)


def dedupe_across_regions(pans, radius_m=DEDUPE_M):
    """Drop a pin when a higher-evidence pin from another region sits within radius_m."""
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
    """One row per county in the state, built or not."""
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
        })
    return pd.DataFrame(rows)


def build_bundle(config_paths, state_fips='06', out_dir=STATEWIDE_DIR):
    import geopandas as gpd

    from pipeline.fetch_region import counties_path

    pans = load_region_pans(config_paths)
    if pans.empty:
        raise RuntimeError('No region has county_rank output yet.')
    n_in = len(pans)
    pans = rerank_statewide(dedupe_across_regions(pans))
    cols = [c for c in GEE_COLS if c in pans.columns]
    pans = pans[cols].sort_values(['county', 'rec_rank_in_county', 'stake_rank_in_county'])
    for sub in ('tables', 'geojson'):
        os.makedirs(out_dir / sub, exist_ok=True)
    pans.to_csv(out_dir / 'tables' / 'gee_pans.csv', index=False)
    gdf = gpd.GeoDataFrame(pans, geometry=gpd.points_from_xy(pans['lon'], pans['lat']),
                           crs='EPSG:4326')
    gdf.to_file(out_dir / 'geojson' / 'gee_pans.geojson', driver='GeoJSON')
    cpath = counties_path(state_fips)
    if os.path.exists(cpath):
        cov = county_coverage(pans, gpd.read_file(cpath))
        cov.to_csv(out_dir / 'tables' / 'county_coverage.csv', index=False)
        built = cov[cov['status'] == 'built']
        print(f'  Counties built: {len(built)}/{len(cov)} — ' + ', '.join(built['county']))
    print(f'  Statewide pins: {len(pans)} (from {n_in} before cross-region dedupe)')
    print(f'  Wrote {out_dir / "tables" / "gee_pans.csv"}')
    return pans


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
    return v


def feature_dicts(pans):
    """GeoJSON-style features with JSON-safe properties (NaN → null)."""
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


def upload(pans, project, asset_root='placer'):
    """Write projects/<project>/assets/<asset_root>/pans. Returns the asset id."""
    import ee

    ee.Initialize(project=project)
    folder = f'projects/{project}/assets/{asset_root}'
    _ensure_folder(ee, folder)
    target = f'{folder}/pans'
    feats = feature_dicts(pans)
    chunks = [feats[i:i + CHUNK] for i in range(0, len(feats), CHUNK)]
    if len(chunks) == 1:
        _delete_if_exists(ee, target)
        task = ee.batch.Export.table.toAsset(
            ee.FeatureCollection(chunks[0]), description='placer_pans', assetId=target)
        task.start()
        _wait([task])
        return target
    parts, tasks = [], []
    for k, chunk in enumerate(chunks):
        pid = f'{folder}/pans_part_{k:03d}'
        _delete_if_exists(ee, pid)
        t = ee.batch.Export.table.toAsset(
            ee.FeatureCollection(chunk), description=f'placer_pans_{k:03d}', assetId=pid)
        t.start()
        parts.append(pid)
        tasks.append(t)
    _wait(tasks)
    _delete_if_exists(ee, target)
    merged = ee.FeatureCollection([ee.FeatureCollection(p) for p in parts]).flatten()
    t = ee.batch.Export.table.toAsset(merged, description='placer_pans_merge', assetId=target)
    t.start()
    _wait([t])
    for p in parts:
        _delete_if_exists(ee, p)
    return target


def main():
    parser = argparse.ArgumentParser(description='Statewide county-ranking bundle for GEE')
    parser.add_argument('--configs', nargs='*', default=None,
                        help='Region configs (default: every CA config with county_rank)')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--project', default=os.environ.get('EE_PROJECT'),
                        help='Google Cloud project registered for Earth Engine')
    parser.add_argument('--asset-root', default='placer')
    args = parser.parse_args()
    configs = args.configs or region_configs('CA')
    print(f'  Regions: {", ".join(os.path.relpath(p, REPO_ROOT) for p in configs)}')
    pans = build_bundle(configs)
    if args.upload:
        if not args.project:
            parser.error('--upload needs --project (or EE_PROJECT)')
        asset = upload(pans, args.project, args.asset_root)
        print(f'  Earth Engine asset ready: {asset}')


if __name__ == '__main__':
    main()
