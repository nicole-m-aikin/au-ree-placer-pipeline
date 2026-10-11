"""Pull a 10 m 3DEP DEM and JRC surface-water occurrence from Earth Engine.

Scouting-grade comparison inputs only (see gee/placer_explorer.js). Nothing
here writes into county_rank evidence_score; pipeline.compare_dem_sources
runs county_rank twice (30 m Copernicus vs this) and reports the delta
before anyone decides whether 10 m earns a place in production scoring.

3DEP 1 m would be truer still, but its coverage is patchy (flown project by
project) and a region can simply have none — 10 m (USGS/3DEP/10m_collection)
is national and a fair, like-for-like upgrade over the 30 m Copernicus
mosaic fetch_dem.py uses today.

Usage:
  python -m pipeline.fetch_gee_dem --config configs/kern_river_subset/config.yaml \
      --project placer-picker-511303 --layer 3dep_10m
  python -m pipeline.fetch_gee_dem --config configs/kern_river_subset/config.yaml \
      --project placer-picker-511303 --layer surface_water \
      --out data/water/ca_kern_river_subset_gsw_occurrence.tif

A bbox download is a single ee.Image.getDownloadURL call — fine for a
subset test box. A region-sized pull (Kern River's full ~0.75° x 0.75°
box, ~65M 10 m pixels) will hit Earth Engine's per-request pixel cap; tile
it instead of raising this script's scope.
"""

from __future__ import annotations

import argparse
import io
import zipfile

import requests
import yaml

from pipeline.utils import resolve_data_path

TIMEOUT = 180
MAX_PIXELS = 50_000_000  # getDownloadURL's per-request raw-pixel ceiling, with headroom


def _ee(project):
    import ee

    ee.Initialize(project=project)
    return ee


def _bbox_region(ee, west, south, east, north, pad):
    return ee.Geometry.Rectangle(
        [west - pad, south - pad, east + pad, north + pad], None, False)


def _check_pixel_count(west, south, east, north, pad, scale_m):
    m_per_deg = 111_320.0
    w = (east - west + 2 * pad) * m_per_deg
    h = (north - south + 2 * pad) * m_per_deg * 1.0
    px = (w / scale_m) * (h / scale_m)
    if px > MAX_PIXELS:
        raise ValueError(
            f'~{px / 1e6:.1f}M pixels at {scale_m} m over this bbox — over the '
            f'{MAX_PIXELS / 1e6:.0f}M single-request ceiling. Shrink the bbox or '
            'tile the download; this script does not do that.')


def _download_image(ee, image, region, scale_m, dest):
    url = image.getDownloadURL({
        'region': region, 'scale': scale_m, 'crs': 'EPSG:4326', 'format': 'GEO_TIFF',
    })
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    content_type = resp.headers.get('content-type', '')
    if 'zip' in content_type or resp.content[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith('.tif')]
            with open(dest, 'wb') as f:
                f.write(zf.read(tif_names[0]))
    else:
        with open(dest, 'wb') as f:
            f.write(resp.content)
    return dest


def fetch_3dep_10m(west, south, east, north, dest, project, pad=0.02):
    """Write a USGS/3DEP/10m_collection elevation mosaic over the bbox. Returns dest."""
    from pathlib import Path

    dest = Path(dest)
    _check_pixel_count(west, south, east, north, pad, scale_m=10)
    ee = _ee(project)
    region = _bbox_region(ee, west, south, east, north, pad)
    dem = ee.ImageCollection('USGS/3DEP/10m_collection').mosaic().select('elevation').clip(region)
    _download_image(ee, dem, region, scale_m=10, dest=dest)
    print(f'  Wrote 3DEP 10 m DEM → {dest}')
    return dest


def fetch_surface_water(west, south, east, north, dest, project, pad=0.02):
    """Write the JRC GSW 'occurrence' band (0-100 %) over the bbox. Returns dest."""
    from pathlib import Path

    dest = Path(dest)
    _check_pixel_count(west, south, east, north, pad, scale_m=30)
    ee = _ee(project)
    region = _bbox_region(ee, west, south, east, north, pad)
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(region)
    _download_image(ee, gsw, region, scale_m=30, dest=dest)
    print(f'  Wrote GSW occurrence → {dest}')
    return dest


def bbox_from_cfg(cfg):
    b = cfg['study_area']['bbox']
    return b['lon_min'], b['lat_min'], b['lon_max'], b['lat_max']


def main():
    parser = argparse.ArgumentParser(
        description='Fetch a 3DEP 10 m DEM or GSW surface-water occurrence raster from Earth Engine')
    parser.add_argument('--config', required=True)
    parser.add_argument('--layer', choices=['3dep_10m', 'surface_water'], required=True)
    parser.add_argument('--project', default=None,
                        help='Google Cloud project registered for Earth Engine (or EE_PROJECT)')
    parser.add_argument('--out', default=None, help='Override the destination path')
    args = parser.parse_args()
    import os
    project = args.project or os.environ.get('EE_PROJECT')
    if not project:
        parser.error('--project (or EE_PROJECT) is required')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    west, south, east, north = bbox_from_cfg(cfg)

    if args.layer == '3dep_10m':
        dest = args.out or resolve_data_path(cfg, 'dem_3dep10m_tif') or 'data/dem/_3dep10m.tif'
        fetch_3dep_10m(west, south, east, north, dest, project)
    else:
        dest = args.out or resolve_data_path(cfg, 'gsw_occurrence_tif') or 'data/water/_gsw_occurrence.tif'
        fetch_surface_water(west, south, east, north, dest, project)


if __name__ == '__main__':
    main()
