"""Download USGS NURE aerial gamma-ray concentration grids and clip to a bbox.

The GeoTIFFs linked as NArad_*_geog83.tif on mrdata.usgs.gov/radiometric are
3-band RGB *images* (no geotransform, no concentration values). The
quantitative grids on the same page are the Esri float grids (NAMrad_*.zip)
from Duval et al. 2005 / OFR 2005-1413, in DNAG spherical Transverse Mercator
at 2 km. This fetcher downloads those, reprojects to EPSG:4326, and writes
single-band GeoTIFFs that load_radiometric_at_points can sample.

Do not invent a synthetic K / eTh / eU grid.

Usage:
  python -m pipeline.fetch_radiometric --config configs/ne_washington/config.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import resolve_data_path

# Quantitative concentration grids (Esri FLT) — same page as the RGB previews.
RAD_ZIPS = {
    'k':   'https://mrdata.usgs.gov/radiometric/data/NAMrad_K.zip',
    'eth': 'https://mrdata.usgs.gov/radiometric/data/NAMrad_Th.zip',
    'eu':  'https://mrdata.usgs.gov/radiometric/data/NAMrad_U.zip',
}

# DNAG sphere Transverse Mercator (Duval / OFR 2005-1413 metadata)
DNAG_TM = (
    "+proj=tmerc +lat_0=0 +lon_0=-100 +k=0.926 +x_0=0 +y_0=0 "
    "+a=6371204 +b=6371204 +units=m +no_defs"
)

CONFIG_KEYS = {
    'k':   'radiometric_k_tif',
    'eth': 'radiometric_eth_tif',
    'eu':  'radiometric_eu_tif',
}

# Extra degrees beyond the study bbox. Matches DATA_SOURCES.md for NE WA
# (lon -120.2→-116.8, lat 47.3→49.3 when bbox is -120→-117 / 47.5→49.1).
DEFAULT_CLIP_PAD = 0.2
SRC_CELL_M = 2000.0

USER_AGENT = "AuREE-pipeline/1.0 (research; NURE radiometric clip)"
CHUNK = 1 << 20


def clip_bounds_from_cfg(cfg, pad=None):
    """Return (xmin, xmax, ymin, ymax) for the clipped GeoTIFFs."""
    b = cfg['study_area']['bbox']
    if pad is None:
        pad = cfg.get('radiometric_clip_pad', DEFAULT_CLIP_PAD)
    return (
        b['lon_min'] - pad,
        b['lon_max'] + pad,
        b['lat_min'] - pad,
        b['lat_max'] + pad,
    )


def dest_path(cfg, which, repo_root=None):
    """Local destination for a clipped band, from config or a study-area default."""
    key = CONFIG_KEYS[which]
    configured = (cfg.get('data') or {}).get(key)
    if configured:
        p = resolve_data_path(cfg, key)
        return Path(p)
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    short = (cfg.get('study_area') or {}).get('short', 'study')
    names = {'k': f'{short}_K.tif', 'eth': f'{short}_eTh.tif', 'eu': f'{short}_eU.tif'}
    return root / 'data' / 'radiometric' / names[which]


def clip_geotiff(src_path, dst_path, bounds):
    """Window-read an already-geographic src to bounds and write a GeoTIFF.

    bounds: (xmin, xmax, ymin, ymax). Returns (height, width) of the clip.
    """
    import rasterio
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds

    xmin, xmax, ymin, ymax = bounds
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        try:
            win = from_bounds(xmin, ymin, xmax, ymax, src.transform)
            win = win.intersection(Window(0, 0, src.width, src.height))
        except WindowError as exc:
            raise ValueError(f"Clip window empty for {src_path} at {bounds}") from exc
        if win.width <= 0 or win.height <= 0:
            raise ValueError(f"Clip window empty for {src_path} at {bounds}")
        win = win.round_offsets().round_lengths()
        data = src.read(1, window=win)
        profile = src.profile.copy()
        profile.update(
            height=int(win.height),
            width=int(win.width),
            transform=src.window_transform(win),
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        profile.pop('photometric', None)
        _atomic_write_tif(dst_path, data, profile, tags=src.tags())
    return int(win.height), int(win.width)


def reproject_clip_to_geo(src_path, dst_path, bounds_ll, src_crs=DNAG_TM, cell_m=SRC_CELL_M):
    """Reproject a projected concentration grid to EPSG:4326 and clip.

    bounds_ll: (lon_min, lon_max, lat_min, lat_max).
    Returns (height, width) of the written GeoTIFF.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, Resampling

    xmin, xmax, ymin, ymax = bounds_ll
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_epsg(4326)
    mid_lat = 0.5 * (ymin + ymax)
    deg = cell_m / (111320.0 * math.cos(math.radians(mid_lat)))
    width = max(1, int(math.ceil((xmax - xmin) / deg)))
    height = max(1, int(math.ceil((ymax - ymin) / deg)))
    transform = from_origin(xmin, ymax, deg, deg)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        source = src.read(1)
        data = np.full((height, width), nodata, dtype=np.float32)
        # Read the array (do not pass rasterio.band): Esri FLT has no
        # dataset CRS, and band() then ignores the explicit src_crs.
        reproject(
            source=source,
            destination=data,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': dst_crs,
            'transform': transform,
            'nodata': float(nodata),
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        }
        _atomic_write_tif(dst_path, data, profile)
    return height, width


def _atomic_write_tif(dst_path, data, profile, tags=None):
    fd, tmp_name = tempfile.mkstemp(suffix='.tif', dir=Path(dst_path).parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        import rasterio
        with rasterio.open(tmp, 'w', **profile) as dst:
            dst.write(np.asarray(data), 1)
            if tags:
                dst.update_tags(**tags)
        tmp.replace(dst_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp:
        total = resp.headers.get('Content-Length')
        total_n = int(total) if total and total.isdigit() else None
        downloaded = 0
        fd, tmp_name = tempfile.mkstemp(prefix=dest.name, dir=dest.parent)
        tmp = Path(tmp_name)
        try:
            with open(fd, 'wb') as out:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total_n:
                        pct = 100 * downloaded / total_n
                        print(f"\r  {dest.name}: {downloaded/1e6:.1f}/{total_n/1e6:.1f} MB ({pct:.0f}%)",
                              end='', flush=True)
            print()
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def _extract_flt(zip_path, dest_dir):
    """Unzip an Esri FLT archive and return the path to the .flt file."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    flts = list(dest_dir.rglob('*.flt'))
    if not flts:
        raise FileNotFoundError(f"No .flt in {zip_path}")
    return flts[0]


def fetch_band(which, cfg, bounds, *, force=False, cache_dir=None):
    """Download one national FLT grid, reproject, clip, write the config dest."""
    dst = dest_path(cfg, which)
    if dst.exists() and not force:
        print(f"  [ok] {which}: already present at {dst}")
        return dst
    url = RAD_ZIPS[which]
    print(f"  [get] {which} ← {url}")
    t0 = time.time()
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / 'data' / 'radiometric' / 'raw'
    cache_dir = Path(cache_dir)
    zpath = cache_dir / Path(url).name
    if not zpath.exists():
        _download(url, zpath)
    flt = _extract_flt(zpath, cache_dir / f'_{which}_esri')
    h, w = reproject_clip_to_geo(flt, dst, bounds)
    print(f"  [ok] {which}: {w}×{h} px EPSG:4326 in {time.time() - t0:.1f}s → {dst}")
    return dst


def run(cfg, *, force=False):
    bounds = clip_bounds_from_cfg(cfg)
    print(f"Radiometric clip bounds: lon {bounds[0]:.2f}→{bounds[1]:.2f}, "
          f"lat {bounds[2]:.2f}→{bounds[3]:.2f}")
    print("Source: Duval 2005 Esri FLT (quantitative). RGB NArad_*_geog83.tif are previews only.")
    written = {}
    for which in ('k', 'eth', 'eu'):
        written[which] = fetch_band(which, cfg, bounds, force=force)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description='Download + clip NURE aerial gamma-ray grids')
    parser.add_argument('--config', default='configs/ne_washington/config.yaml')
    parser.add_argument('--force', action='store_true', help='Re-download / re-clip even if present')
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg, force=args.force)
    print('\nPoint data.radiometric_*_tif at the clipped files (already set for NE WA).')
    print('Task 1 will draw the eTh overlay on the next run. No synthetic K/Th grid.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
