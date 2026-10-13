"""Fetch every input the county ranking needs for one region config.

Usage:
  python -m pipeline.fetch_region --config configs/kern_river/config.yaml
  python -m pipeline.fetch_region --config configs/kern_river/config.yaml --force

DEM       Copernicus GLO-30 (pipeline.fetch_dem)
NURE      national HSSR clip (pipeline.task12_second_belt)
MRDS      gold sites from the bbox search (pipeline.task12_second_belt)
PAD-US    + active MLRS claims + AIANNH (pipeline.task11_land_access)
MLRS      claims closed / modified in the last year (BLM keeps one year only)
Counties  Census cartographic boundary file, cached once per state

Cached files follow the data/<layer>/ convention and stay gitignored.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
import yaml

from pipeline.ml_artifacts import REPO_ROOT
from pipeline.task11_land_access import (
    _dedupe_features,
    _query_arcgis,
    _short,
    _tile_bbox,
    _write_geojson,
    config_state,
    ensure_land_access_data,
)
from pipeline.utils import bbox, resolve_data_path

COUNTY_YEAR = 2024
COUNTY_URL = (
    f'https://www2.census.gov/geo/tiger/GENZ{COUNTY_YEAR}/shp/'
    f'cb_{COUNTY_YEAR}_us_county_500k.zip'
)
COUNTY_RAW = REPO_ROOT / 'data' / 'counties' / 'raw' / f'cb_{COUNTY_YEAR}_us_county_500k.zip'
STATE_FIPS = {
    'AK': '02', 'AZ': '04', 'CA': '06', 'CO': '08', 'ID': '16', 'MT': '30',
    'NC': '37', 'NM': '35', 'NV': '32', 'OR': '41', 'SC': '45', 'UT': '49',
    'VA': '51', 'WA': '53', 'WY': '56',
}
MLRS_CLOSED_URL = (
    'https://gis.blm.gov/nlsdb/rest/services/HUB/'
    'BLM_Natl_MLRS_Mining_Claims_Closed/FeatureServer/0/query'
)
MLRS_CLOSED_FIELDS = 'OBJECTID,ID,CSE_NAME,CSE_DISP,BLM_PROD,CSE_NR,LEG_CSE_NR,Modified'
TIMEOUT = 300


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return str(dest)


def state_fips(cfg):
    st = config_state(cfg)
    if st not in STATE_FIPS:
        raise ValueError(f'No FIPS code for state {st!r}; set study_area.state')
    return STATE_FIPS[st]


def counties_path(fips):
    return str(REPO_ROOT / 'data' / 'counties' / f'counties_{fips}.geojson')


def fetch_counties(fips, force=False):
    """One state's county polygons (GEOID, NAME) from the Census 500k file."""
    import geopandas as gpd

    dest = counties_path(fips)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 1000:
        print(f'  Counties already at {dest}', flush=True)
        return dest
    if force or not COUNTY_RAW.exists() or COUNTY_RAW.stat().st_size < 1000:
        print(f'  Downloading {COUNTY_URL}', flush=True)
        _download(COUNTY_URL, COUNTY_RAW)
    gdf = gpd.read_file(COUNTY_RAW)
    gdf = gdf[gdf['STATEFP'].astype(str) == fips].to_crs('EPSG:4326')
    gdf = gdf[['GEOID', 'NAME', 'STATEFP', 'geometry']]
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest, driver='GeoJSON')
    print(f'  Wrote {len(gdf)} counties → {dest}', flush=True)
    return dest


def mlrs_closed_path(cfg):
    return str(REPO_ROOT / 'data' / 'mlrs' / f'mlrs_closed_{_short(cfg)}.geojson')


def fetch_mlrs_closed(cfg, force=False):
    """Claims BLM closed (or touched) in the last year — re-staking candidates."""
    dest = mlrs_closed_path(cfg)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 40:
        print(f'  Closed MLRS claims already at {dest}', flush=True)
        return dest
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    feats = []
    for t0, t1, t2, t3 in _tile_bbox(lon_min, lat_min, lon_max, lat_max, step=0.5):
        try:
            fc = _query_arcgis(MLRS_CLOSED_URL, t0, t1, t2, t3, MLRS_CLOSED_FIELDS,
                               geometry_json=True)
            feats.extend(fc.get('features') or [])
        except Exception as exc:
            print(f'    closed-claim tile {t0:.2f},{t1:.2f} failed ({exc}); continuing',
                  flush=True)
    feats = _dedupe_features(feats)
    _write_geojson(dest, {'type': 'FeatureCollection', 'features': feats})
    print(f'  Wrote {len(feats)} closed MLRS claims → {dest}', flush=True)
    return dest


def run(cfg, force=False, skip_dem=False):
    from pipeline.task12_second_belt import fetch_mrds_gold, fetch_nure_hssr

    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    if skip_dem:
        print('  Skipping DEM (ee_trace / local D8 not needed for this fetch)', flush=True)
    else:
        from pipeline import fetch_dem
        fetch_dem.run(cfg, force=force)

    nure = resolve_data_path(cfg, 'nure_csv')
    if nure and (force or not os.path.exists(nure)):
        try:
            fetch_nure_hssr(lon_min, lat_min, lon_max, lat_max, nure)
        except RuntimeError as exc:
            print(f'  NURE: {exc} — ranking runs without chemistry', flush=True)
    mrds = resolve_data_path(cfg, 'mrds_geojson')
    if mrds and (force or not os.path.exists(mrds)):
        fetch_mrds_gold(lon_min, lat_min, lon_max, lat_max, mrds)

    cfg.setdefault('task11', {})['land_access'] = True
    ensure_land_access_data(cfg, force=force)
    try:
        fetch_mlrs_closed(cfg, force=force)
    except Exception as exc:
        print(f'  Closed MLRS fetch failed ({exc}); no re-stake hints', flush=True)
    fetch_counties(state_fips(cfg), force=force)
    try:
        from pipeline.placer_layers import ensure_placer_layers
        ensure_placer_layers(force=force)
    except Exception as exc:
        print(f'  Placer layers failed ({exc}); ranking continues without them',
              flush=True)


def main():
    parser = argparse.ArgumentParser(description='Fetch all county-ranking inputs for a region')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--skip-dem', action='store_true',
        help='Skip Copernicus DEM (ok when county_rank.ee_trace uses MERIT Hydro)',
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, force=args.force, skip_dem=args.skip_dem)
    print(json.dumps({'region': _short(cfg), 'status': 'fetched'}))


if __name__ == '__main__':
    main()
