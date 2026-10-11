"""Fetch PAD-US + BLM MLRS claims + Census AIANNH tribal areas for a bbox.

Usage:
  python -m pipeline.fetch_land_access --config configs/california_sierra/config.yaml
  python -m pipeline.fetch_land_access --config configs/california_sierra/config.yaml --force
  python -m pipeline.fetch_land_access --config configs/ne_washington/config.yaml --tribal-only

AIANNH = US Census TIGER/Line American Indian / Alaska Native / Native
Hawaiian Areas. The national zip is cached once under data/aiannh/raw/ and
clipped per study box to data/aiannh/aiannh_<short>.geojson (gitignored,
same convention as data/padus/ and data/mlrs/).
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
    aiannh_path,
    ensure_land_access_data,
    land_access_enabled,
)
from pipeline.utils import bbox

AIANNH_YEAR = 2025
AIANNH_URL = (
    f'https://www2.census.gov/geo/tiger/TIGER{AIANNH_YEAR}/AIANNH/'
    f'tl_{AIANNH_YEAR}_us_aiannh.zip'
)
AIANNH_RAW = REPO_ROOT / 'data' / 'aiannh' / 'raw' / f'tl_{AIANNH_YEAR}_us_aiannh.zip'
AIANNH_FIELDS = (
    'GEOID', 'NAME', 'NAMELSAD', 'LSAD', 'CLASSFP', 'COMPTYP', 'MTFCC', 'FUNCSTAT',
)
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


def fetch_aiannh(cfg, force=False, pad_deg=0.05):
    """Clip national AIANNH polygons to the config bbox (+pad). Returns path."""
    import geopandas as gpd

    dest = aiannh_path(cfg)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 40:
        print(f'  AIANNH already at {dest}', flush=True)
        return dest
    if force or not AIANNH_RAW.exists() or AIANNH_RAW.stat().st_size < 1000:
        print(f'  Downloading {AIANNH_URL}', flush=True)
        _download(AIANNH_URL, AIANNH_RAW)
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    box = (lon_min - pad_deg, lat_min - pad_deg, lon_max + pad_deg, lat_max + pad_deg)
    gdf = gpd.read_file(AIANNH_RAW, bbox=box)
    if gdf.crs is not None and '4326' not in str(gdf.crs):
        gdf = gdf.to_crs('EPSG:4326')
    gdf = gdf[[c for c in AIANNH_FIELDS if c in gdf.columns] + ['geometry']]
    gdf = gdf.assign(source=f'Census TIGER/Line {AIANNH_YEAR} AIANNH')
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if len(gdf):
        gdf.to_file(dest, driver='GeoJSON')
    else:
        with open(dest, 'w') as f:
            json.dump({'type': 'FeatureCollection', 'features': []}, f)
    print(f'  Wrote {len(gdf)} AIANNH features → {dest}', flush=True)
    return dest


def main():
    parser = argparse.ArgumentParser(
        description='Fetch PAD-US + MLRS + AIANNH land-access layers for a config bbox',
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--tribal-only', action='store_true',
        help='Only fetch the Census AIANNH tribal-area layer',
    )
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.tribal_only:
        print(f'AIANNH: {fetch_aiannh(cfg, force=args.force)}')
        return
    if not land_access_enabled(cfg):
        # Allow one-shot fetch even if the flag is off — set it for this run.
        cfg.setdefault('task11', {})['land_access'] = True
        print('  (land_access was false in config; enabling for this fetch)', flush=True)
    padus, mlrs = ensure_land_access_data(cfg, force=args.force)
    print(f'PAD-US: {padus}')
    print(f'MLRS:   {mlrs}')
    print(f'AIANNH: {aiannh_path(cfg)}')


if __name__ == '__main__':
    main()
