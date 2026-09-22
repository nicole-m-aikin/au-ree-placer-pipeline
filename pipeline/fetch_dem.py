"""Download a 30 m Copernicus GLO-30 mosaic for a study-area bbox.

Usage:
  python -m pipeline.fetch_dem --config configs/idaho_batholith/config.yaml
  python -m pipeline.fetch_dem --config configs/california_sierra/config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import requests

from pipeline.ml_artifacts import REPO_ROOT
from pipeline.utils import bbox, resolve_data_path

COPERNICUS = 'https://copernicus-dem-30m.s3.amazonaws.com'
TIMEOUT = 180


def _tile_name(lat, lon):
    lat_str = f'N{lat:02d}' if lat >= 0 else f'S{abs(lat):02d}'
    lon_str = f'E{abs(lon):03d}' if lon >= 0 else f'W{abs(lon):03d}'
    folder = f'Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM'
    return folder, f'{folder}.tif'


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=TIMEOUT, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return dest


def mosaic_tiles(tile_paths, output_path):
    import rasterio
    from rasterio.merge import merge

    datasets = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': transform,
            'compress': 'deflate',
        })
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()
    print(f'  Mosaicked {len(tile_paths)} tiles → {output_path}')


def fetch_copernicus_dem(west, south, east, north, dest):
    """Write a GLO-30 mosaic covering the bbox. Returns dest path."""
    dest = Path(dest)
    lats = range(int(math.floor(south)), int(math.ceil(north)))
    lons = range(int(math.floor(west)), int(math.ceil(east)))
    downloaded = []
    tmp_dir = dest.parent / f'.{dest.stem}_tiles'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for lat in lats:
        for lon in lons:
            folder, fname = _tile_name(lat, lon)
            url = f'{COPERNICUS}/{folder}/{fname}'
            tile_path = tmp_dir / f'{lat}_{lon}.tif'
            if tile_path.exists() and tile_path.stat().st_size > 1_000_000:
                try:
                    import rasterio
                    with rasterio.open(tile_path) as src:
                        src.read(1, window=rasterio.windows.Window(0, 0, 8, 8))
                    print(f'  Reusing {tile_path.name}', flush=True)
                    downloaded.append(tile_path)
                    continue
                except Exception:
                    print(f'  Corrupt tile {tile_path.name}; re-fetching', flush=True)
                    tile_path.unlink(missing_ok=True)
            print(f'  Fetching {url}', flush=True)
            try:
                _download(url, tile_path)
                downloaded.append(tile_path)
            except Exception as exc:
                print(f'  Tile {lat},{lon} failed: {exc}')
    if not downloaded:
        raise RuntimeError(
            f'No Copernicus tiles downloaded for ({west},{south})–({east},{north})'
        )
    if len(downloaded) == 1:
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(downloaded[0], dest)
    else:
        mosaic_tiles(downloaded, dest)
        for p in downloaded:
            try:
                p.unlink()
            except OSError:
                pass
    _clip_dem(dest, west, south, east, north)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
    return str(dest)


def _clip_dem(path, west, south, east, north, pad=0.02):
    """Keep only the study box so Task 11 D8 does not chew unused tiles."""
    import rasterio
    from rasterio.windows import from_bounds

    west, south, east, north = west - pad, south - pad, east + pad, north + pad
    with rasterio.open(path) as src:
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[0], width=data.shape[1],
            transform=transform, compress='deflate',
        )
        tmp = Path(str(path) + '.clip')
        with rasterio.open(tmp, 'w', **profile) as dst:
            dst.write(data, 1)
    os.replace(tmp, path)
    print(f'  Clipped DEM to bbox → {data.shape[1]}x{data.shape[0]}', flush=True)


def dest_from_cfg(cfg):
    path = resolve_data_path(cfg, 'dem_tif')
    if path:
        return path
    short = (cfg.get('study_area') or {}).get('short') or 'study'
    return str(REPO_ROOT / 'data' / 'dem' / f'{short}_30m.tif')


def run(cfg, force=False):
    dest = dest_from_cfg(cfg)
    if os.path.exists(dest) and not force and os.path.getsize(dest) > 1_000_000:
        print(f'  DEM already at {dest}', flush=True)
        return dest
    lon_min, lon_max, lat_min, lat_max = bbox(cfg)
    print(f'  Copernicus GLO-30 for {lon_min},{lat_min}–{lon_max},{lat_max}', flush=True)
    return fetch_copernicus_dem(lon_min, lat_min, lon_max, lat_max, dest)


if __name__ == '__main__':
    import yaml

    parser = argparse.ArgumentParser(description='Fetch a 30 m DEM for a config bbox')
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    path = run(cfg, force=args.force)
    print(f'DEM ready: {path}')
