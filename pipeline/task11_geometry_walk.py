"""Geometry-only walk list for chemistry-dark pins — no NURE, no forest, no P.

National NURE HSSR has no sediment north of ~39.0° in the Sierra box, so
South Yuba / Malakoff / Humbug cannot be ML-ranked. This module snaps each
named pin to its D8 creek, traces a walk reach up and down the channel,
votes trap vertices with the same score_reach weights Task 11 uses (no
near-NURE bonus, because there is no NURE), and labels every pin and reach
vertex with PAD-US / MLRS / tribal access.

Order is access first, trap score second. Nothing here says gold is in the
creek. It says: if you pan this reach, stand here.

Outputs:
  {outputs_dir}/tables/task11_geometry_walks.csv
  {outputs_dir}/tables/task11_geometry_pans.csv
  {outputs_dir}/geojson/task11_geometry_walks.geojson     # reach lines
  {outputs_dir}/geojson/task11_geometry_pans.geojson
  {outputs_dir}/gis/task11_geometry_walks.gpkg            # anchors, reaches, pans
  ~/projects/task11_{short}_geometry_walks.gpkg           # QGIS copy (no + in path)
  {outputs_dir}/text/task11_geometry_walks_summary.txt
"""

import math
import os
import shutil
import warnings

import numpy as np
import pandas as pd

from pipeline.geo_crs import utm_epsg
from pipeline.task11_pan_traps import (
    JUNCTION_M,
    MIN_SEP_M,
    SPACING_M,
    _confinement,
    ensure_accumulation_tif,
    score_reach,
    why_flags,
)
from pipeline.utils import out, resolve_data_path

# pysheds default dirmap: N, NE, E, SE, S, SW, W, NW → (drow, dcol).
D8_OFFSETS = {
    64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
    4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1),
}
M_PER_DEG = 111_320.0
WINDOW_PAD_DEG = 0.15
SNAP_M = 300.0
PANS_PER_KM = 0.5
MIN_PANS = 2
MAX_PANS = 5
# A tributary mouth is a real inflow, not D8 noise: ≥10% of the trunk and ≥0.5 km².
JUNCTION_FRAC = 0.10
JUNCTION_MIN_KM2 = 0.5
MIN_SNAP_KM2 = 0.5
FLAT_WATER_TOL_M = 0.5


def ensure_flowdir_tif(dem_path, cache_path):
    """D8 flow direction, cached. Same fill/flow as ensure_accumulation_tif."""
    if cache_path and os.path.exists(cache_path):
        return cache_path
    if not dem_path or not os.path.exists(dem_path):
        return None
    import rasterio
    from pysheds.grid import Grid

    print(f"  Computing D8 flow direction from {os.path.basename(dem_path)}...")
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(dem)))
    fdir = np.asarray(grid.flowdir(inflated), dtype=np.int32)
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
    profile.update(dtype='int32', count=1, compress='lzw', nodata=0)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    with rasterio.open(cache_path, 'w', **profile) as dst:
        dst.write(fdir, 1)
    print(f"  Wrote flow direction {cache_path}")
    return cache_path


def _read_window(path, lon_min, lat_min, lon_max, lat_max):
    import rasterio
    from rasterio.windows import Window, from_bounds

    with rasterio.open(path) as src:
        win = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
        win = win.round_offsets().round_lengths()
        # A requested pad past the raster edge (e.g. a seed within WINDOW_PAD_DEG
        # of the study-area bbox) gives a negative offset or oversized window.
        # read() silently clips it to the raster's real extent but
        # window_transform() would not, shifting the header by the overhang —
        # the same class of bug fetch_dem._clip_dem fixed for the DEM mosaic.
        # Clip the window first so pixels and transform agree.
        win = win.intersection(Window(0, 0, src.width, src.height))
        arr = src.read(1, window=win)
        transform = src.window_transform(win)
        res = abs(src.res[0])
    return arr, transform, res


def cell_km2(res_deg, lat):
    dy = res_deg * M_PER_DEG
    dx = dy * math.cos(math.radians(lat))
    return dx * dy / 1e6


def snap_to_channel(acc, transform, lon, lat, min_cells, snap_m=SNAP_M):
    """Nearest cell draining ≥ min_cells within snap_m. Returns (row, col, dist_m) or None.

    Gazetteer pins are reach/creek precision (often 0.5–2 km off the water),
    so "biggest cell nearby" grabs a side gully. Size picks the named creek;
    distance picks the spot on it.
    """
    rows, cols = np.indices(acc.shape)
    clon = transform.c + (cols + 0.5) * transform.a
    clat = transform.f + (rows + 0.5) * transform.e
    d = np.hypot(
        (clon - lon) * M_PER_DEG * math.cos(math.radians(lat)),
        (clat - lat) * M_PER_DEG,
    )
    ok = np.isfinite(acc) & (acc >= min_cells) & (d <= snap_m)
    if not ok.any():
        return None
    i = int(np.argmin(np.where(ok, d, np.inf)))
    r, c = np.unravel_index(i, acc.shape)
    return int(r), int(c), float(d[r, c])


def _step_m(dr, dc, res, lat):
    return math.hypot(dr * res * M_PER_DEG, dc * res * M_PER_DEG * math.cos(math.radians(lat)))


def trace_downstream(fdir, acc, r, c, res, lat, max_m, stop_cells=None):
    """Follow D8 downhill. Returns ([(row, col), ...], stop_reason); start excluded.

    stop_cells ends the walk on the first cell draining at least that much
    (the mouth, where a side creek enters the trunk river).
    """
    path = []
    s = 0.0
    while s < max_m:
        code = int(fdir[r, c])
        if code not in D8_OFFSETS:
            return path, 'sink_or_flat'
        dr, dc = D8_OFFSETS[code]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < fdir.shape[0] and 0 <= nc < fdir.shape[1]):
            return path, 'window_edge'
        s += _step_m(dr, dc, res, lat)
        path.append((nr, nc))
        if stop_cells is not None and float(acc[nr, nc]) >= stop_cells:
            return path, 'reached_trunk'
        r, c = nr, nc
    return path, 'length'


def trace_upstream(fdir, acc, r, c, res, lat, max_m):
    """Follow the largest inflow uphill. Returns ([(row, col), ...], stop_reason).

    The start cell is excluded.
    """
    path = []
    s = 0.0
    while s < max_m:
        best = None
        for code, (dr, dc) in D8_OFFSETS.items():
            nr, nc = r - dr, c - dc
            if not (0 <= nr < fdir.shape[0] and 0 <= nc < fdir.shape[1]):
                continue
            if int(fdir[nr, nc]) != code:
                continue
            a = float(acc[nr, nc])
            if best is None or a > best[0]:
                best = (a, nr, nc, dr, dc)
        if best is None:
            return path, 'channel_head'
        _, r, c, dr, dc = best
        s += _step_m(dr, dc, res, lat)
        path.append((r, c))
    return path, 'length'


def junction_mask(acc_path, s_path, s_vertices, cells_per_km2,
                  frac=JUNCTION_FRAC, min_km2=JUNCTION_MIN_KM2, radius_m=JUNCTION_M):
    """Vertices within radius_m of a real tributary inflow on the full-res path."""
    acc_path = np.asarray(acc_path, dtype=float)
    jumps = np.diff(acc_path) - 1.0
    floor = min_km2 * cells_per_km2
    hit = (jumps >= np.maximum(frac * acc_path[:-1], floor))
    s_hits = np.asarray(s_path, dtype=float)[1:][hit]
    if s_hits.size == 0:
        return np.zeros(len(s_vertices), dtype=bool)
    sv = np.asarray(s_vertices, dtype=float)
    return np.min(np.abs(sv[:, None] - s_hits[None, :]), axis=1) <= radius_m


def flat_water_mask(z, tol_m=FLAT_WATER_TOL_M):
    """Vertices on a DEM-flattened water surface (reservoir pool).

    GLO-30 flattens lakes to one elevation. A slope break at the head of a
    pool is real sedimentation, but it is under water — not a pan site.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    flat = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        win = z[lo:hi]
        win = win[np.isfinite(win)]
        if win.size >= 3 and float(win.max() - win.min()) < tol_m:
            flat[i] = True
    return flat


def resample_path(s_path, spacing_m=SPACING_M):
    """Indices into the cell path at ~spacing_m intervals (always keeps both ends)."""
    s_path = np.asarray(s_path, dtype=float)
    if s_path.size == 0:
        return np.array([], dtype=int)
    targets = np.arange(0.0, s_path[-1] + 1e-6, spacing_m)
    idx = np.searchsorted(s_path, targets)
    idx = np.clip(idx, 0, s_path.size - 1)
    idx = np.unique(np.concatenate([idx, [s_path.size - 1]]))
    return idx


def pick_pans(vertices, n, taken_xy, min_sep_m=MIN_SEP_M):
    """Greedy highest trap score; keep min_sep_m from this walk and earlier walks."""
    chosen = []
    for _, row in vertices.sort_values('trap_score', ascending=False).iterrows():
        if len(chosen) >= n or float(row['trap_score']) <= 0:
            break
        xy = (float(row['x']), float(row['y']))
        others = taken_xy + [(float(c['x']), float(c['y'])) for c in chosen]
        if all(math.hypot(xy[0] - ox, xy[1] - oy) >= min_sep_m for ox, oy in others):
            chosen.append(row)
    return pd.DataFrame(chosen)


def walk_specs(cfg):
    """Expand task11.geometry_walks into one spec per anchor pin."""
    gw = (cfg.get('task11') or {}).get('geometry_walks') or {}
    hobby_path = resolve_data_path(cfg, 'hobby_reports_csv')
    hobby = pd.read_csv(hobby_path) if hobby_path and os.path.exists(hobby_path) else pd.DataFrame()
    by_id = {str(r['report_id']): r for _, r in hobby.iterrows()} if len(hobby) else {}
    specs = []
    for area in gw.get('walks') or []:
        for rid in area.get('report_ids') or []:
            pin = by_id.get(str(rid))
            if pin is None:
                warnings.warn(f"  geometry walk: report_id {rid} not in hobby CSV",
                              stacklevel=2)
                continue
            specs.append({
                'walk_id': f"{area['id']}:{rid}",
                'area': area.get('name') or area['id'],
                'report_id': str(rid),
                'anchor_name': str(pin.get('name') or rid),
                'creek_name': str(pin.get('creek_name') or ''),
                'anchor_lon': float(pin['lon']),
                'anchor_lat': float(pin['lat']),
                'up_km': float(area.get('up_km', 2.0)),
                'down_km': float(area.get('down_km', 2.0)),
                'stop_above_km2': (
                    None if area.get('stop_above_km2') is None
                    else float(area['stop_above_km2'])
                ),
                'snap_m': float(area.get('snap_m', gw.get('snap_m', SNAP_M))),
                'min_snap_km2': float(area.get('min_snap_km2', MIN_SNAP_KM2)),
            })
    return specs


def build_walk(spec, fdir_path, acc_path, dem_path):
    """Trace one reach and score its vertices. Returns (reach_row, vertices_df) or None."""
    lon, lat = spec['anchor_lon'], spec['anchor_lat']
    pad = WINDOW_PAD_DEG
    fdir, transform, res = _read_window(fdir_path, lon - pad, lat - pad, lon + pad, lat + pad)
    acc, _, _ = _read_window(acc_path, lon - pad, lat - pad, lon + pad, lat + pad)
    cpk = 1.0 / cell_km2(res, lat)
    snap = snap_to_channel(acc, transform, lon, lat, spec['min_snap_km2'] * cpk,
                           snap_m=spec['snap_m'])
    if snap is None:
        return None
    r0, c0, snap_d = snap

    up, up_stop = trace_upstream(fdir, acc, r0, c0, res, lat, spec['up_km'] * 1000.0)
    stop_cells = None if spec['stop_above_km2'] is None else spec['stop_above_km2'] * cpk
    down, down_stop = trace_downstream(
        fdir, acc, r0, c0, res, lat, spec['down_km'] * 1000.0, stop_cells=stop_cells,
    )
    cells = list(reversed(up)) + [(r0, c0)] + down
    rows = np.array([rc[0] for rc in cells])
    cols = np.array([rc[1] for rc in cells])
    lons = transform.c + (cols + 0.5) * transform.a
    lats = transform.f + (rows + 0.5) * transform.e
    steps = [0.0] + [
        _step_m(rows[i] - rows[i - 1], cols[i] - cols[i - 1], res, lat)
        for i in range(1, len(cells))
    ]
    s_path = np.cumsum(steps)
    s_snap = float(s_path[len(up)])
    acc_path = acc[rows, cols].astype(float)

    idx = resample_path(s_path)
    if idx.size < 3:
        return None
    import geopandas as gpd

    from pipeline.task11_pan_traps import _sample_raster

    v_lon, v_lat, v_s = lons[idx], lats[idx], s_path[idx]
    utm = utm_epsg(lon, lat)
    v_utm = gpd.GeoSeries(gpd.points_from_xy(v_lon, v_lat), crs='EPSG:4326').to_crs(utm)
    vx, vy = v_utm.x.to_numpy(), v_utm.y.to_numpy()
    z = _sample_raster(dem_path, v_lon, v_lat)
    jn = junction_mask(acc_path, s_path, v_s, cpk)
    try:
        conf = _confinement(dem_path, list(v_lon), list(v_lat), list(vx), list(vy),
                            [0] * len(idx), utm=utm)
    except Exception:
        conf = np.full(len(idx), np.nan)
    scored = score_reach(z, v_s, acc=acc_path[idx], junction=jn, confinement=conf)
    flat = flat_water_mask(z)
    scored.loc[flat, 'trap_score'] = 0.0
    vertices = scored.assign(
        flat_water=flat,
        walk_id=spec['walk_id'], lon=v_lon, lat=v_lat, x=vx, y=vy,
        along_km=np.round((v_s - s_snap) / 1000.0, 2),
        upstream_km2=np.round(acc_path[idx] / cpk, 1),
    )

    from shapely.geometry import LineString
    reach = {
        **spec,
        'snap_lon': round(float(lons[len(up)]), 6),
        'snap_lat': round(float(lats[len(up)]), 6),
        'snap_m': round(float(snap_d), 0),
        'snap_upstream_km2': round(float(acc[r0, c0]) / cpk, 1),
        'reach_km': round(float(s_path[-1]) / 1000.0, 2),
        'traced_up_km': round(s_snap / 1000.0, 2),
        'traced_down_km': round((float(s_path[-1]) - s_snap) / 1000.0, 2),
        'up_stop': up_stop,
        'down_stop': down_stop,
        'pct_reach_flat_water': round(100.0 * float(flat.mean()), 0),
        'geometry': LineString(list(zip(lons, lats))),
    }
    return reach, vertices


def rank_walks(walks, pans):
    """Walk order: access-ok pan count, then best access-ok trap, then best trap."""
    ok = pans[pans['access_ok'].astype(str).str.lower() == 'yes']
    n_ok = ok.groupby('walk_id').size()
    best_ok = ok.groupby('walk_id')['trap_score'].max()
    best = pans.groupby('walk_id')['trap_score'].max()
    w = walks.copy()
    w['n_pans'] = w['walk_id'].map(pans.groupby('walk_id').size()).fillna(0).astype(int)
    w['n_pans_access_ok'] = w['walk_id'].map(n_ok).fillna(0).astype(int)
    w['best_trap_access_ok'] = w['walk_id'].map(best_ok)
    w['best_trap'] = w['walk_id'].map(best)
    w = w.sort_values(
        ['n_pans_access_ok', 'best_trap_access_ok', 'best_trap'],
        ascending=[False, False, False], na_position='last',
    ).reset_index(drop=True)
    w['walk_rank'] = np.arange(1, len(w) + 1)
    return w


def _reach_access(vertices_gdf):
    """Share of reach vertices by access class, and the managers seen along it."""
    rows = []
    for wid, grp in vertices_gdf.groupby('walk_id'):
        n = max(1, len(grp))
        at = grp['access_type'].astype(str)
        mgr = sorted({str(m) for m in grp.get('manager_name', pd.Series(dtype=str)).dropna()
                      if str(m).strip() and str(m) != 'nan'})
        rows.append({
            'walk_id': wid,
            'pct_reach_access_ok': round(
                100.0 * (grp['access_ok'].astype(str).str.lower() == 'yes').sum() / n, 0),
            'pct_reach_claimed': round(100.0 * (at == 'claimed').sum() / n, 0),
            'pct_reach_private': round(100.0 * (at == 'private').sum() / n, 0),
            'reach_managers': ';'.join(mgr),
        })
    return pd.DataFrame(rows)


def run(cfg):
    import geopandas as gpd

    from pipeline.ml_spatial import nearest_gold_mrds_deg
    from pipeline.task11_field_campaign import campaign_gold_xy
    from pipeline.task11_land_access import annotate_points

    specs = walk_specs(cfg)
    if not specs:
        print("  No task11.geometry_walks in config — nothing to do.")
        return None
    dem_path = resolve_data_path(cfg, 'dem_tif')
    acc_path = ensure_accumulation_tif(dem_path, out(cfg, 'geojson', 'task11_d8_accumulation.tif'))
    fdir_path = ensure_flowdir_tif(dem_path, out(cfg, 'geojson', 'task11_d8_flowdir.tif'))
    if not acc_path or not fdir_path:
        raise FileNotFoundError("Geometry walks need the belt DEM (data.dem_tif).")

    hobby_path = resolve_data_path(cfg, 'hobby_reports_csv')
    hobby_gdf = None
    if hobby_path and os.path.exists(hobby_path):
        h = pd.read_csv(hobby_path)
        hobby_gdf = gpd.GeoDataFrame(h, geometry=gpd.points_from_xy(h['lon'], h['lat']),
                                     crs='EPSG:4326')

    reaches = []
    all_vertices = []
    pan_rows = []
    taken_xy = []
    for spec in specs:
        built = build_walk(spec, fdir_path, acc_path, dem_path)
        if built is None:
            warnings.warn(f"  geometry walk {spec['walk_id']}: no channel near the pin",
                          stacklevel=2)
            continue
        reach, vertices = built
        reaches.append(reach)
        all_vertices.append(vertices)
        n = int(min(MAX_PANS, max(MIN_PANS, round(reach['reach_km'] * PANS_PER_KM))))
        picked = pick_pans(vertices, n, taken_xy)
        if picked.empty:
            continue
        picked = picked.sort_values('along_km').reset_index(drop=True)
        for order, (_, p) in enumerate(picked.iterrows(), 1):
            taken_xy.append((float(p['x']), float(p['y'])))
            pan_rows.append({
                'walk_id': spec['walk_id'],
                'area': spec['area'],
                'anchor_name': spec['anchor_name'],
                'pan_order': order,
                'lon': round(float(p['lon']), 6),
                'lat': round(float(p['lat']), 6),
                'elev_m': None if not np.isfinite(p['z']) else round(float(p['z']), 1),
                'along_km': float(p['along_km']),
                'upstream_km2': float(p['upstream_km2']),
                'trap_score': round(float(p['trap_score']), 2),
                'why': why_flags(p),
                'slope': None if not np.isfinite(p['slope']) else round(float(p['slope']), 5),
                'omega': None if not np.isfinite(p['omega']) else round(float(p['omega']), 1),
            })

    if not reaches:
        print("  No geometry walks could be traced.")
        return None

    walks = gpd.GeoDataFrame(reaches, crs='EPSG:4326')
    verts = pd.concat(all_vertices, ignore_index=True)
    verts_gdf = gpd.GeoDataFrame(
        verts, geometry=gpd.points_from_xy(verts['lon'], verts['lat']), crs='EPSG:4326',
    )
    verts_gdf = annotate_points(verts_gdf, cfg, hobby_gdf=hobby_gdf)
    walks = walks.merge(_reach_access(verts_gdf), on='walk_id', how='left')

    pans = pd.DataFrame(pan_rows)
    pans_gdf = gpd.GeoDataFrame(
        pans, geometry=gpd.points_from_xy(pans['lon'], pans['lat']), crs='EPSG:4326',
    )
    pans_gdf = annotate_points(pans_gdf, cfg, hobby_gdf=hobby_gdf)

    gold_xy = campaign_gold_xy(cfg)
    if gold_xy is not None:
        pans_gdf['nearest_gold_km'] = [
            round(float(nearest_gold_mrds_deg(x, y, gold_xy)) * 111.0, 1)
            for x, y in zip(pans_gdf['lon'], pans_gdf['lat'])
        ]

    walks = rank_walks(walks, pans_gdf)
    pans_gdf = pans_gdf.merge(walks[['walk_id', 'walk_rank']], on='walk_id', how='left')
    pans_gdf['_ok'] = np.where(pans_gdf['access_ok'].astype(str).str.lower() == 'yes', 0, 1)
    pans_gdf = pans_gdf.sort_values(
        ['_ok', 'walk_rank', 'trap_score'], ascending=[True, True, False],
    ).drop(columns='_ok').reset_index(drop=True)
    pans_gdf['pan_rank'] = np.arange(1, len(pans_gdf) + 1)

    walk_cols = [
        'walk_rank', 'walk_id', 'area', 'report_id', 'anchor_name', 'creek_name',
        'anchor_lon', 'anchor_lat', 'snap_lon', 'snap_lat', 'snap_m', 'snap_upstream_km2',
        'reach_km', 'traced_up_km', 'traced_down_km', 'up_stop', 'down_stop',
        'pct_reach_flat_water', 'n_pans', 'n_pans_access_ok', 'best_trap_access_ok', 'best_trap',
        'pct_reach_access_ok', 'pct_reach_claimed', 'pct_reach_private', 'reach_managers',
    ]
    pan_cols = [
        'pan_rank', 'walk_rank', 'walk_id', 'area', 'anchor_name', 'pan_order',
        'lon', 'lat', 'elev_m', 'along_km', 'upstream_km2', 'trap_score', 'why',
        'slope', 'omega', 'nearest_gold_km', 'access_type', 'access_ok', 'access_reason',
        'manager_name', 'claim_status', 'pub_access', 'tribal_land', 'tribal_name',
        'rules_note',
    ]
    pan_cols = [c for c in pan_cols if c in pans_gdf.columns]
    walks[walk_cols].to_csv(out(cfg, 'tables', 'task11_geometry_walks.csv'), index=False)
    pans_gdf[pan_cols].to_csv(out(cfg, 'tables', 'task11_geometry_pans.csv'), index=False)
    walks[walk_cols + ['geometry']].to_file(
        out(cfg, 'geojson', 'task11_geometry_walks.geojson'), driver='GeoJSON')
    pans_gdf[pan_cols + ['geometry']].to_file(
        out(cfg, 'geojson', 'task11_geometry_pans.geojson'), driver='GeoJSON')

    anchors = gpd.GeoDataFrame(
        walks[['walk_rank', 'walk_id', 'anchor_name', 'creek_name']].copy(),
        geometry=gpd.points_from_xy(walks['anchor_lon'], walks['anchor_lat']),
        crs='EPSG:4326',
    )
    gpkg = os.path.join(cfg['outputs_dir'], 'gis', 'task11_geometry_walks.gpkg')
    os.makedirs(os.path.dirname(gpkg), exist_ok=True)
    if os.path.exists(gpkg):
        os.remove(gpkg)
    anchors.to_file(gpkg, layer='anchors', driver='GPKG')
    walks[walk_cols + ['geometry']].to_file(gpkg, layer='walk_reaches', driver='GPKG')
    pans_gdf[pan_cols + ['geometry']].to_file(gpkg, layer='pan_locations', driver='GPKG')
    short = str((cfg.get('study_area') or {}).get('short') or 'study')
    try:
        shutil.copy2(gpkg, os.path.expanduser(f'~/projects/task11_{short}_geometry_walks.gpkg'))
    except OSError as e:
        print(f"  Could not copy GPKG to ~/projects: {e}")

    _write_summary(cfg, walks, pans_gdf)
    print("Task 11 geometry walks complete.")
    return walks, pans_gdf


def _write_summary(cfg, walks, pans):
    lines = [
        'GEOMETRY-ONLY WALK LIST (no chemistry)',
        'Chemistry-dark pins: national NURE HSSR has no sediment here.',
        'No P. No forest. Trap geometry on the 30 m D8 creek + land access.',
        '=' * 68,
        '',
        'WALKS (access-ok pans first, then best trap score)',
        '',
    ]
    for _, w in walks.iterrows():
        lines.append(
            f"  #{int(w['walk_rank'])}  {w['anchor_name']} ({w['creek_name']})  "
            f"reach {w['reach_km']:.1f} km (up {w['traced_up_km']:.1f} / "
            f"down {w['traced_down_km']:.1f}, stop: {w['down_stop']})  "
            f"pans {int(w['n_pans_access_ok'])}/{int(w['n_pans'])} access-ok  "
            f"reach {w['pct_reach_access_ok']:.0f}% access-ok, "
            f"{w['pct_reach_claimed']:.0f}% claimed, {w['pct_reach_private']:.0f}% private"
        )
    lines += ['', 'PAN PINS (access-ok first)', '']
    for _, p in pans.iterrows():
        side = 'down' if p['along_km'] >= 0 else 'up'
        lines.append(
            f"  pan#{int(p['pan_rank'])}  walk#{int(p['walk_rank'])}  "
            f"{p['lat']:.5f}, {p['lon']:.5f}  {p['elev_m']:.0f} m  "
            f"{abs(p['along_km']):.1f} km {side}stream of {p['anchor_name']}  "
            f"trap={p['trap_score']:.1f} ({p['why']})  "
            f"access={p['access_type']}/{p['access_ok']} ({p.get('manager_name') or ''})"
        )
    lines += [
        '',
        'Trap score = slope break 2 + knickpoint foot 2 + power drop 2 +',
        'tributary mouth 1.5 + valley opening 1 (max 8.5; no NURE bonus).',
        'Scores are relative to each reach. 30 m DEM sees system-scale traps,',
        'not a bar head or a bedrock crack. Confirm claims and rules on site.',
    ]
    path = out(cfg, 'text', 'task11_geometry_walks_summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    import sys

    import yaml

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/california_sierra/config.yaml'
    with open(cfg_path) as f:
        run(yaml.safe_load(f))
