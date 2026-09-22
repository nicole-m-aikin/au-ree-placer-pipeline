"""Where to put a pan — trap geometry on the D8 creek, after chemistry.

Not a forest feature. Not a NURE grab. Task 9 already said the drainage
looks like known gold placers. This module walks the extracted stream and
votes for vertices where heavies are likely to drop: slope break, stream-
power drop, knickpoint foot, tributary mouth, valley opening.

30 m DEM can see system-scale traps. It cannot see a bar head or a bedrock
crack. Pins are a walk list, not a reserve.
"""

import os
import warnings

import numpy as np
import pandas as pd

SPACING_M = 150.0
MIN_SEP_M = 400.0
N_PANS = 2
JUNCTION_M = 180.0
CONFINE_M = (150.0, 300.0)
UTM = 'EPSG:32611'
# Weights follow the cheap GIS list in LIT_REVIEW §10.3.
W_SLOPE_BREAK = 2.0
W_SL_FOOT = 2.0
W_POWER_DROP = 2.0
W_JUNCTION = 1.5
W_VALLEY_OPEN = 1.0
W_NEAR_NURE = 1.5


def score_reach(z, s, acc=None, junction=None, confinement=None):
    """Flag trap-like vertices on one downstream-ordered reach.

    z, s : elevation (m) and along-channel distance from the local source (m).
    acc  : D8 accumulation (cells). Missing → Ω uses Hack A ~ L².
    Returns a DataFrame aligned to the input length.
    """
    z = np.asarray(z, dtype=float)
    s = np.asarray(s, dtype=float)
    n = int(z.size)
    out = pd.DataFrame({
        'z': z,
        's_m': s,
        'slope': np.nan,
        'omega': np.nan,
        'hack_sl': np.nan,
        'slope_break': False,
        'sl_foot': False,
        'power_drop': False,
        'junction': False,
        'valley_open': False,
        'trap_score': 0.0,
    })
    if n < 3:
        return out

    ds = np.diff(s)
    ds = np.where(ds <= 1.0, np.nan, ds)
    seg_s = np.abs(np.diff(z)) / ds
    slope = np.concatenate([seg_s, [seg_s[-1]]])
    out['slope'] = slope

    if acc is None:
        a = np.maximum(s, SPACING_M) ** 2
    else:
        a = np.asarray(acc, dtype=float)
        a = np.where(np.isfinite(a) & (a > 0), a, np.maximum(s, SPACING_M) ** 2)
    omega = a * np.where(np.isfinite(slope), slope, 0.0)
    out['omega'] = omega
    L = np.maximum(s, SPACING_M)
    out['hack_sl'] = slope * L

    med_s = float(np.nanmedian(slope))
    med_w = float(np.nanmedian(omega))
    if not np.isfinite(med_s):
        med_s = 0.0
    if not np.isfinite(med_w):
        med_w = 0.0

    sb = np.zeros(n, dtype=bool)
    slf = np.zeros(n, dtype=bool)
    pdrop = np.zeros(n, dtype=bool)
    sl = out['hack_sl'].to_numpy()
    for i in range(1, n):
        prev, here = slope[i - 1], slope[i]
        if np.isfinite(prev) and np.isfinite(here):
            if prev > 1.4 * max(med_s, 1e-6) and here < med_s:
                sb[i] = True
            if med_s > 0 and here < 0.5 * prev and prev > med_s:
                sb[i] = True
        if i >= 2 and np.isfinite(sl[i - 1]) and np.isfinite(sl[i]) and np.isfinite(sl[i - 2]):
            if sl[i - 1] > sl[i - 2] and sl[i - 1] > sl[i] and sl[i - 1] > np.nanmedian(sl):
                slf[i] = True
        wo, wh = omega[i - 1], omega[i]
        if np.isfinite(wo) and np.isfinite(wh) and wo > med_w and wh < 0.65 * wo:
            pdrop[i] = True

    jn = np.zeros(n, dtype=bool)
    if junction is not None:
        jn = np.asarray(junction, dtype=bool)
        if jn.size != n:
            jn = np.zeros(n, dtype=bool)
    vo = np.zeros(n, dtype=bool)
    if confinement is not None:
        c = np.asarray(confinement, dtype=float)
        med_c = float(np.nanmedian(c))
        if np.isfinite(med_c) and med_c > 0:
            vo = np.isfinite(c) & (c < 0.55 * med_c)

    out['slope_break'] = sb
    out['sl_foot'] = slf
    out['power_drop'] = pdrop
    out['junction'] = jn
    out['valley_open'] = vo
    out['trap_score'] = (
        W_SLOPE_BREAK * sb.astype(float)
        + W_SL_FOOT * slf.astype(float)
        + W_POWER_DROP * pdrop.astype(float)
        + W_JUNCTION * jn.astype(float)
        + W_VALLEY_OPEN * vo.astype(float)
    )
    return out


def pick_separated(df, n=N_PANS, min_sep_m=MIN_SEP_M,
                   xcol='x', ycol='y', scorecol='trap_score'):
    """Greedy highest score, keep pins at least min_sep_m apart."""
    if df is None or df.empty:
        return df.iloc[0:0] if df is not None else pd.DataFrame()
    work = df.sort_values(scorecol, ascending=False)
    chosen = []
    for _, row in work.iterrows():
        if len(chosen) >= n:
            break
        if float(row[scorecol]) <= 0:
            break
        if all(np.hypot(float(row[xcol]) - float(c[xcol]),
                        float(row[ycol]) - float(c[ycol])) >= min_sep_m
               for c in chosen):
            chosen.append(row)
    if not chosen:
        return work.iloc[0:0]
    return pd.DataFrame(chosen).reset_index(drop=True)


def why_flags(row):
    bits = []
    for name, label in (
        ('slope_break', 'slope_break'),
        ('sl_foot', 'knickpoint_foot'),
        ('power_drop', 'power_drop'),
        ('junction', 'tributary_junction'),
        ('valley_open', 'valley_opens'),
        ('near_nure', 'near_hot_nure'),
    ):
        if bool(row.get(name)):
            bits.append(label)
    return ';'.join(bits) if bits else 'outlet_fallback'


def ensure_accumulation_tif(dem_path, cache_path):
    """D8 accumulation, cached. Same fill/flow as stream extract."""
    if cache_path and os.path.exists(cache_path):
        return cache_path
    if not dem_path or not os.path.exists(dem_path):
        return None
    from pysheds.grid import Grid
    import rasterio

    print(f"  Computing D8 accumulation from {os.path.basename(dem_path)}...")
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    pit_filled = grid.fill_pits(dem)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = np.asarray(grid.accumulation(fdir), dtype=np.float32)
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
    profile.update(dtype='float32', count=1, compress='lzw', nodata=-9999.0)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    with rasterio.open(cache_path, 'w', **profile) as dst:
        dst.write(acc, 1)
    print(f"  Wrote accumulation {cache_path}")
    return cache_path


def _sample_raster(path, xs, ys):
    if not path or not os.path.exists(path):
        return np.full(len(xs), np.nan)
    import rasterio
    with rasterio.open(path) as src:
        nodata = src.nodata
        vals = []
        for v in src.sample(zip(xs, ys)):
            z = v[0]
            if nodata is not None and z == nodata:
                vals.append(np.nan)
            else:
                vals.append(float(z) if np.isfinite(z) else np.nan)
    return np.asarray(vals, dtype=float)


def _line_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'LineString':
        return [geom]
    if geom.geom_type == 'MultiLineString':
        return [p for p in geom.geoms if p.geom_type == 'LineString' and not p.is_empty]
    if hasattr(geom, 'geoms'):
        out = []
        for g in geom.geoms:
            out.extend(_line_parts(g))
        return out
    return []


def _densify_utm_lines(streams_utm, spacing_m=SPACING_M):
    """Vertices every spacing_m. Columns: x, y, line_id, s_m (along that line)."""
    rows = []
    lid = 0
    for geom in streams_utm.geometry:
        for line in _line_parts(geom):
            length = float(line.length)
            if length < spacing_m * 0.5:
                continue
            n = max(2, int(round(length / spacing_m)))
            for i in range(n + 1):
                p = line.interpolate(i / float(n), normalized=True)
                rows.append({
                    'x': float(p.x), 'y': float(p.y),
                    'line_id': lid,
                    's_m': float(i / float(n) * length),
                })
            lid += 1
    return pd.DataFrame(rows)


def _mark_junctions(pts, radius_m=JUNCTION_M):
    """True where a vertex sits near a place two line_ids meet."""
    if pts.empty or pts['line_id'].nunique() < 2:
        return np.zeros(len(pts), dtype=bool)
    ends = []
    for lid, sub in pts.groupby('line_id'):
        ends.append(sub.iloc[0][['x', 'y', 'line_id']])
        if len(sub) > 1:
            ends.append(sub.iloc[-1][['x', 'y', 'line_id']])
    ends = pd.DataFrame(ends)
    nodes = []
    used = np.zeros(len(ends), dtype=bool)
    xy = ends[['x', 'y']].to_numpy(dtype=float)
    lids = ends['line_id'].to_numpy()
    for i in range(len(ends)):
        if used[i]:
            continue
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        near = d <= 40.0
        if np.unique(lids[near]).size >= 2:
            nodes.append((float(xy[i, 0]), float(xy[i, 1])))
            used[near] = True
    if not nodes:
        return np.zeros(len(pts), dtype=bool)
    pxy = pts[['x', 'y']].to_numpy(dtype=float)
    hit = np.zeros(len(pts), dtype=bool)
    for nx, ny in nodes:
        hit |= np.hypot(pxy[:, 0] - nx, pxy[:, 1] - ny) <= radius_m
    return hit


def _confinement(dem_path, lons, lats, xs, ys, line_groups):
    """Rough valley walls: mean z at ±150/300 m perpendicular minus channel z."""
    if not lons:
        return np.array([])
    import geopandas as gpd

    z0 = _sample_raster(dem_path, lons, lats)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    lids = np.asarray(line_groups)
    extra_x = []
    extra_y = []
    for lid in pd.unique(lids):
        idx = np.where(lids == lid)[0]
        xx, yy = xs[idx], ys[idx]
        dx = np.gradient(xx)
        dy = np.gradient(yy)
        nrm = np.hypot(dx, dy)
        nrm = np.where(nrm < 1.0, 1.0, nrm)
        px, py = -dy / nrm, dx / nrm
        for dist in CONFINE_M:
            extra_x.extend((xx + px * dist).tolist())
            extra_x.extend((xx - px * dist).tolist())
            extra_y.extend((yy + py * dist).tolist())
            extra_y.extend((yy - py * dist).tolist())
    if not extra_x:
        return np.full(len(z0), np.nan)
    extra = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(extra_x, extra_y), crs=UTM,
    ).to_crs('EPSG:4326')
    z_side = _sample_raster(
        dem_path,
        extra.geometry.x.to_numpy(),
        extra.geometry.y.to_numpy(),
    )
    # extras were appended per line, same row order as xs[idx]; rebuild to pts order
    wall_by_idx = np.full(len(z0), np.nan)
    cursor = 0
    n_off = len(CONFINE_M) * 2
    for lid in pd.unique(lids):
        idx = np.where(lids == lid)[0]
        n = len(idx)
        block = z_side[cursor:cursor + n_off * n].reshape(n_off, n)
        wall_by_idx[idx] = np.nanmean(block, axis=0)
        cursor += n_off * n
    return wall_by_idx - z0


def pan_locations_gdf(catchments, streams, dem_path, nure_spots=None,
                      acc_path=None, n_pans=N_PANS):
    """One or two pan pins per walkable catchment, on the D8 line."""
    import geopandas as gpd

    empty = gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
    if (catchments is None or len(catchments) == 0
            or streams is None or len(streams) == 0
            or not dem_path or not os.path.exists(dem_path)):
        return empty

    acc_vals_ok = acc_path and os.path.exists(acc_path)
    nure_utm = None
    if nure_spots is not None and len(nure_spots):
        hot = nure_spots
        if 'p_anomalous' in nure_spots.columns:
            hot = nure_spots[nure_spots['p_anomalous'].fillna(0) >= 0.4]
        if len(hot):
            nure_utm = hot.to_crs(UTM)

    rows = []
    for _, cat in catchments.iterrows():
        geom = cat.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            clipped = streams.clip(geom)
        except Exception as e:
            warnings.warn(f"  stream clip failed for block {cat.get('block_id')}: {e}")
            continue
        if clipped is None or clipped.empty:
            continue
        utm = clipped.to_crs(UTM)
        pts = _densify_utm_lines(utm)
        if pts.empty:
            continue
        # Channel should run downhill: flip a line if the last vertex is higher.
        flipped = []
        for lid, sub in pts.groupby('line_id'):
            sub = sub.sort_values('s_m')
            flipped.append(sub)
        pts = pd.concat(flipped, ignore_index=True)

        g4326 = gpd.GeoDataFrame(
            pts, geometry=gpd.points_from_xy(pts['x'], pts['y']), crs=UTM,
        ).to_crs('EPSG:4326')
        lons = g4326.geometry.x.to_numpy()
        lats = g4326.geometry.y.to_numpy()
        z = _sample_raster(dem_path, lons, lats)
        acc = _sample_raster(acc_path, lons, lats) if acc_vals_ok else None
        jn = _mark_junctions(pts)
        try:
            conf = _confinement(dem_path, lons, lats, pts['x'].tolist(),
                                pts['y'].tolist(), pts['line_id'].tolist())
        except Exception:
            conf = np.full(len(pts), np.nan)

        # Orient each line downstream (z falling) before scoring.
        pieces = []
        for lid, sub in pts.assign(
            z=z,
            lon=lons, lat=lats,
            acc=acc if acc is not None else np.nan,
            junction=jn,
            confinement=conf,
        ).groupby('line_id'):
            sub = sub.sort_values('s_m')
            zz = sub['z'].to_numpy()
            if np.isfinite(zz[0]) and np.isfinite(zz[-1]) and zz[-1] > zz[0] + 2.0:
                sub = sub.iloc[::-1].reset_index(drop=True)
                sub['s_m'] = sub['s_m'].max() - sub['s_m']
            scored = score_reach(
                sub['z'].to_numpy(),
                sub['s_m'].to_numpy(),
                acc=sub['acc'].to_numpy() if acc is not None else None,
                junction=sub['junction'].to_numpy(),
                confinement=sub['confinement'].to_numpy(),
            )
            for col in scored.columns:
                sub[col] = scored[col].to_numpy()
            pieces.append(sub)
        if not pieces:
            continue
        allv = pd.concat(pieces, ignore_index=True)
        allv['near_nure'] = False
        if nure_utm is not None and len(nure_utm):
            nxy = np.column_stack([
                nure_utm.geometry.x.to_numpy(),
                nure_utm.geometry.y.to_numpy(),
            ])
            pxy = allv[['x', 'y']].to_numpy(dtype=float)
            dmin = np.min(
                np.hypot(pxy[:, None, 0] - nxy[None, :, 0],
                         pxy[:, None, 1] - nxy[None, :, 1]),
                axis=1,
            )
            allv['near_nure'] = dmin <= 800.0
            allv['trap_score'] = allv['trap_score'] + W_NEAR_NURE * allv['near_nure'].astype(float)

        picked = pick_separated(allv, n=n_pans)
        if picked.empty:
            # No geometry vote — stand at the lowest on-channel vertex (outlet).
            fallback = allv.sort_values(['z', 's_m'], ascending=[True, False]).head(1)
            fallback = fallback.copy()
            fallback['trap_score'] = 0.0
            fallback['why'] = 'outlet_fallback'
            picked = fallback
        else:
            picked = picked.copy()
            picked['why'] = picked.apply(why_flags, axis=1)

        picked = picked.reset_index(drop=True)
        for order, (_, p) in enumerate(picked.iterrows(), 1):
            rows.append({
                'walk_rank': cat.get('walk_rank'),
                'block_id': cat.get('block_id'),
                'campaign_class': cat.get('campaign_class'),
                'pan_order': order,
                'role': 'pan',
                'lon': round(float(p['lon']), 6),
                'lat': round(float(p['lat']), 6),
                'elev_m': None if not np.isfinite(p.get('z', np.nan))
                else round(float(p['z']), 1),
                'trap_score': round(float(p['trap_score']), 2),
                'why': p.get('why') or why_flags(p),
                'slope': None if not np.isfinite(p.get('slope', np.nan))
                else round(float(p['slope']), 5),
                'omega': None if not np.isfinite(p.get('omega', np.nan))
                else round(float(p['omega']), 1),
            })

    if not rows:
        return empty
    df = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['lon'], df['lat']),
        crs='EPSG:4326',
    )
