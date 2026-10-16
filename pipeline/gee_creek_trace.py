"""Creek walks on Earth Engine MERIT Hydro — replaces local pysheds D8 tracing.

Usage:
  python -m pipeline.gee_creek_trace --config configs/rand_district/config.yaml
  # or from county_rank when county_rank.ee_trace is true (default)

For each seed:
  1. Pull a small MERIT window around the seed (sampleRectangle)
  2. Snap to the highest upa (km²) cell within snap_m
  3. Walk upstream / downstream following MERIT dir (D8 powers of 2)
  4. Score traps with score_reach; pick pans_per_walk vertices

Returns the same shapes as task11_geometry_walk.walk_seeds: (reaches, pan_rows).
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import yaml

from pipeline.task11_pan_traps import (
    SPACING_M, confinement_along, junction_from_area, score_reach, why_flags,
)

import os

DEFAULT_PROJECT = os.environ.get('EE_PROJECT', 'placer-picker-511303')
# MERIT Hydro dir: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
DIR_DELTA = {
    1: (1, 0), 2: (1, -1), 4: (0, -1), 8: (-1, -1),
    16: (-1, 0), 32: (-1, 1), 64: (0, 1), 128: (1, 1),
}
M_PER_DEG = 111_320.0
PIXEL_M = 90.0


def ee_trace_enabled(cfg):
    cr = cfg.get('county_rank') or {}
    if 'ee_trace' in cr:
        return bool(cr['ee_trace'])
    return bool((cfg.get('gee') or {}).get('trace', True))


def gee_project(cfg):
    import os
    return str(
        (cfg.get('gee') or {}).get('project')
        or os.environ.get('EE_PROJECT')
        or DEFAULT_PROJECT
    )


def _init_ee(project):
    import ee
    ee.Initialize(project=project)
    return ee


def _window_arrays(ee, lon, lat, pad_deg):
    """Download MERIT dir/upa/elv for a small rectangle around a seed.

    Use MERIT elv (not 3DEP 10 m) for the long-profile: sampleRectangle returns
    each band at its native grid, so mixing 10 m into the MERIT window broadcast
    a single elev10 pixel across the whole array and zeroed trap scores.
    """
    region = ee.Geometry.Rectangle([
        lon - pad_deg, lat - pad_deg, lon + pad_deg, lat + pad_deg,
    ])
    # Cast to float so sampleRectangle defaultValue=-9999 is type-compatible
    # (MERIT dir is uint8; -9999 is rejected on integer bands).
    merit = (ee.Image('MERIT/Hydro/v1_0_1')
             .select(['dir', 'upa', 'elv']).toFloat())
    props = merit.sampleRectangle(region, defaultValue=-9999).getInfo()['properties']
    dir_a = np.array(props['dir'], dtype=float)
    upa_a = np.array(props['upa'], dtype=float)
    elv_a = np.array(props['elv'], dtype=float)
    elv_a = np.where(elv_a == -9999, np.nan, elv_a)
    upa_a = np.where(upa_a == -9999, np.nan, upa_a)
    west, south = lon - pad_deg, lat - pad_deg
    east, north = lon + pad_deg, lat + pad_deg
    nrows, ncols = dir_a.shape
    return {
        'dir': dir_a, 'upa': upa_a, 'elv': elv_a,
        'west': west, 'south': south, 'east': east, 'north': north,
        'res_x': (east - west) / max(1, ncols),
        'res_y': (north - south) / max(1, nrows),
        'nrows': nrows, 'ncols': ncols,
    }


def _lonlat_to_rc(w, lon, lat):
    c = int((lon - w['west']) / w['res_x'])
    r = int((w['north'] - lat) / w['res_y'])
    return r, c


def _rc_to_lonlat(w, r, c):
    return w['west'] + (c + 0.5) * w['res_x'], w['north'] - (r + 0.5) * w['res_y']


def _snap(w, lon, lat, snap_m, min_snap_km2):
    r0, c0 = _lonlat_to_rc(w, lon, lat)
    rad = max(1, int(float(snap_m) / PIXEL_M) + 1)
    best = None
    for r in range(max(0, r0 - rad), min(w['nrows'], r0 + rad + 1)):
        for c in range(max(0, c0 - rad), min(w['ncols'], c0 + rad + 1)):
            u = w['upa'][r, c]
            if not np.isfinite(u) or u < float(min_snap_km2) or u == -9999:
                continue
            lon2, lat2 = _rc_to_lonlat(w, r, c)
            d = math.hypot(
                (lon2 - lon) * M_PER_DEG * math.cos(math.radians(lat)),
                (lat2 - lat) * M_PER_DEG,
            )
            if d > float(snap_m):
                continue
            key = (u, -d)
            if best is None or key > best[0]:
                best = (key, r, c, d, u)
    return None if best is None else best[1:]


def _step(w, r, c, reverse=False):
    code = int(w['dir'][r, c]) if np.isfinite(w['dir'][r, c]) else 0
    if code not in DIR_DELTA:
        return None
    dlon, dlat = DIR_DELTA[code]
    if reverse:
        dlon, dlat = -dlon, -dlat
    nr, nc = r - dlat, c + dlon
    if not (0 <= nr < w['nrows'] and 0 <= nc < w['ncols']):
        return None
    return int(nr), int(nc)


def _path_to_vertices(path, elev, upa, spacing_m=SPACING_M):
    if not path or len(path) < 3:
        return None
    lons = np.array([p[0] for p in path], dtype=float)
    lats = np.array([p[1] for p in path], dtype=float)
    lat0 = float(np.nanmean(lats))
    cos = max(0.2, math.cos(math.radians(lat0)))
    step = np.hypot(np.diff(lons) * M_PER_DEG * cos, np.diff(lats) * M_PER_DEG)
    s = np.concatenate([[0.0], np.cumsum(step)])
    z = np.asarray(elev, dtype=float)
    a = np.asarray(upa, dtype=float)
    if s[-1] < spacing_m * 2:
        idx = np.arange(len(s))
    else:
        targets = np.arange(0.0, s[-1] + 1e-6, spacing_m)
        idx = np.unique(np.clip(np.searchsorted(s, targets), 0, len(s) - 1))
        idx = np.unique(np.concatenate([idx, [0, len(s) - 1]]))
    return {
        'lon': lons[idx], 'lat': lats[idx], 'z': z[idx], 's': s[idx],
        'acc': a[idx], 'full_lon': lons, 'full_lat': lats,
        's_full': s, 'reach_km': float(s[-1] / 1000.0),
    }


def _flags_on_vertices(vert, cells, junction, confinement):
    """Nearest cell flag for each resampled vertex."""
    n = len(vert['lon'])
    j = np.zeros(n, dtype=bool)
    conf = np.full(n, np.nan)
    if not cells:
        return j, conf
    clon = np.array([p[0] for p in cells], dtype=float)
    clat = np.array([p[1] for p in cells], dtype=float)
    for i, (lon, lat) in enumerate(zip(vert['lon'], vert['lat'])):
        d = (clon - lon) ** 2 + (clat - lat) ** 2
        k = int(np.argmin(d))
        j[i] = bool(junction[k])
        conf[i] = confinement[k]
    return j, conf


def _score_and_pick(vert, walk_id, seed, pans_per_walk, taken_xy, snap_i,
                    junction=None, confinement=None):
    import geopandas as gpd
    from shapely.geometry import LineString

    from pipeline.geo_crs import utm_epsg
    from pipeline.task11_geometry_walk import pick_pans

    if vert is None or len(vert['lon']) < 3:
        return None, []
    z = vert['z']
    if not np.isfinite(z).any():
        z = 1000.0 - (vert['s'] / 100.0)
    # Convert upa km² to a pseudo cell-count for score_reach omega
    acc = np.maximum(vert['acc'], 0.01) * 1e6 / (PIXEL_M ** 2)
    scored = score_reach(
        z, vert['s'], acc=acc, junction=junction, confinement=confinement,
    )
    utm = utm_epsg(float(vert['lon'][0]), float(vert['lat'][0]))
    pts = gpd.GeoSeries(
        gpd.points_from_xy(vert['lon'], vert['lat']), crs='EPSG:4326',
    ).to_crs(utm)
    s_snap = float(vert['s_full'][min(snap_i, len(vert['s_full']) - 1)])
    scored = scored.assign(
        lon=vert['lon'], lat=vert['lat'],
        x=pts.x.to_numpy(), y=pts.y.to_numpy(),
        along_km=np.round((vert['s'] - s_snap) / 1000.0, 2),
        upstream_km2=np.round(vert['acc'], 1),
        walk_id=walk_id,
    )
    reach = {
        'walk_id': walk_id,
        'seed_kind': seed.get('seed_kind', 'mrds'),
        'area': seed.get('seed_kind', 'mrds'),
        'report_id': '',
        'anchor_name': seed.get('name', ''),
        'creek_name': seed.get('creek_name', ''),
        'anchor_lon': float(seed['lon']),
        'anchor_lat': float(seed['lat']),
        'snap_lon': round(float(vert['full_lon'][snap_i]), 6),
        'snap_lat': round(float(vert['full_lat'][snap_i]), 6),
        'snap_m': seed.get('snap_d', 0),
        'snap_upstream_km2': seed.get('snap_u'),
        'reach_km': round(vert['reach_km'], 2),
        'traced_up_km': round(s_snap / 1000.0, 2),
        'traced_down_km': round((vert['s_full'][-1] - s_snap) / 1000.0, 2),
        'up_stop': 'length',
        'down_stop': 'length',
        'geometry': LineString(list(zip(vert['full_lon'], vert['full_lat']))),
    }
    picked = pick_pans(scored, int(pans_per_walk), taken_xy)
    pan_rows = []
    for _, p in picked.iterrows():
        taken_xy.append((float(p['x']), float(p['y'])))
        pan_rows.append({
            'walk_id': walk_id,
            'seed_kind': seed.get('seed_kind', 'mrds'),
            'seed_name': seed.get('name', ''),
            'creek_name': seed.get('creek_name', ''),
            'lon': round(float(p['lon']), 6),
            'lat': round(float(p['lat']), 6),
            'elev_m': None if not np.isfinite(p['z']) else round(float(p['z']), 1),
            'along_km': float(p['along_km']),
            'upstream_km2': (
                float(p['upstream_km2'])
                if np.isfinite(p.get('upstream_km2', np.nan)) else None
            ),
            'trap_score': round(float(p['trap_score']), 2),
            'why': why_flags(p),
        })
    return reach, pan_rows


def walk_seeds_ee(specs, project=DEFAULT_PROJECT, pans_per_walk=2,
                  up_km=0.5, down_km=4.0, snap_m=600.0, min_snap_km2=1.0):
    """Trace specs on MERIT Hydro. specs are seed_specs()-style dicts."""
    ee = _init_ee(project)
    reaches, pan_rows, taken_xy = [], [], []
    if not specs:
        return reaches, pan_rows
    # Window must cover up+down walk; pad in degrees
    pad = (float(up_km) + float(down_km)) * 1000.0 / M_PER_DEG + 0.03
    print(f'  EE MERIT creek trace: {len(specs)} seeds (pad={pad:.3f}°) …', flush=True)
    for k, spec in enumerate(specs, 1):
        lon, lat = float(spec['anchor_lon']), float(spec['anchor_lat'])
        try:
            w = _window_arrays(ee, lon, lat, pad)
        except Exception as exc:
            if k <= 3 or k % 50 == 0:
                print(f'    seed {k} window failed: {exc}', flush=True)
            continue
        snapped = _snap(w, lon, lat, spec.get('snap_m', snap_m),
                        spec.get('min_snap_km2', min_snap_km2))
        if snapped is None:
            continue
        r0, c0, snap_d, snap_u = snapped
        up_cells = int(max(2, math.ceil(float(spec.get('up_km', up_km)) * 1000 / PIXEL_M)))
        down_cells = int(max(3, math.ceil(float(spec.get('down_km', down_km)) * 1000 / PIXEL_M)))
        up_path, r, c = [], r0, c0
        for _ in range(up_cells):
            nxt = _step(w, r, c, reverse=True)
            if nxt is None:
                break
            r, c = nxt
            up_path.append((r, c))
        down_path, r, c = [], r0, c0
        for _ in range(down_cells):
            nxt = _step(w, r, c, reverse=False)
            if nxt is None:
                break
            r, c = nxt
            down_path.append((r, c))
        cells = list(reversed(up_path)) + [(r0, c0)] + down_path
        if len(cells) < 3:
            continue
        path = [_rc_to_lonlat(w, r, c) for r, c in cells]
        elev = [float(w['elv'][r, c]) if w['elv'][r, c] != -9999 else np.nan for r, c in cells]
        upa = [float(w['upa'][r, c]) if w['upa'][r, c] != -9999 else np.nan for r, c in cells]
        vert = _path_to_vertices(path, elev, upa)
        if vert is None:
            continue
        cell_ll = path
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        junc_cells = junction_from_area(np.asarray(upa, dtype=float))
        conf_cells = confinement_along(w['elv'], rows, cols)
        junc_v, conf_v = _flags_on_vertices(vert, cell_ll, junc_cells, conf_cells)
        seed = {
            'seed_kind': spec.get('seed_kind', 'mrds'),
            'name': spec.get('anchor_name', ''),
            'creek_name': spec.get('creek_name', ''),
            'lon': lon, 'lat': lat,
            'snap_d': round(float(snap_d), 0),
            'snap_u': round(float(snap_u), 2),
        }
        reach, pans = _score_and_pick(
            vert, spec['walk_id'], seed, pans_per_walk, taken_xy, len(up_path),
            junction=junc_v, confinement=conf_v,
        )
        if reach is None:
            continue
        reaches.append(reach)
        pan_rows.extend(pans)
        if k % 25 == 0:
            print(f'    EE-walked {k}/{len(specs)} seeds, {len(reaches)} reaches', flush=True)
    print(f'  EE creek trace: {len(reaches)} reaches, {len(pan_rows)} pans', flush=True)
    return reaches, pan_rows


def main():
    parser = argparse.ArgumentParser(description='EE MERIT Hydro creek walks')
    parser.add_argument('--config', required=True)
    parser.add_argument('--project', default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from pipeline.county_rank import (
        _load_hobby, _load_mrds, _short, grid_seeds, hobby_seeds, rank_cfg, seed_specs,
    )
    rc = rank_cfg(cfg)
    mrds = _load_mrds(cfg)
    hobby = _load_hobby(cfg)
    lat0 = float(np.mean([cfg['study_area']['bbox']['lat_min'],
                          cfg['study_area']['bbox']['lat_max']]))
    seeds = pd.concat([
        hobby_seeds(hobby),
        grid_seeds(mrds, float(rc['seed_cell_km']), int(rc['max_seeds']), lat0),
    ], ignore_index=True)
    specs = seed_specs(seeds, rc, _short(cfg))
    reaches, pans = walk_seeds_ee(
        specs, project=args.project or gee_project(cfg),
        pans_per_walk=int(rc['pans_per_walk']),
        up_km=float(rc['up_km']), down_km=float(rc['down_km']),
        snap_m=float(rc['snap_m']), min_snap_km2=float(rc['min_snap_km2']),
    )
    print(f'  {len(reaches)} reaches, {len(pans)} pans')


if __name__ == '__main__':
    main()
